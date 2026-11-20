from __future__ import annotations

from dataclasses import dataclass, field, replace
import gc
import heapq
import json
import math
import pickle
from functools import lru_cache
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

@dataclass(frozen=True)
class GpuSpec:
    clock_hz: float
    num_sms: int
    tensor_flops_per_s: float
    l2_size_bytes: int
    l2_bandwidth_bytes_per_s: float
    hbm_size_bytes: int
    hbm_bandwidth_bytes_per_s: float

def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b

@dataclass(frozen=True)
class FlashRates:
    gpu: GpuSpec
    vector_flops_per_s: float
    shared_memory_bytes_per_sm: int
    launch_overhead_s: float
    smem_bandwidth_bytes_per_s: float
    smem_latency_cycles: float
    l2_latency_cycles: float
    hbm_latency_cycles: float
    mma_latency_cycles: float

    sfu_ops_per_cycle: float = 16.0

    registers_per_sm: int = 65536
    register_allocation_per_thread: int = 8
    shared_memory_bytes_per_block: int = 163 * 1024
    max_ctas_per_sm: int = 32
    max_warps_per_sm: int = 64

    reduction_time_s: Callable[[int, int], float] | None = field(default=None, repr=False, compare=False)

    l2_bypass: bool = False

    @classmethod
    def from_hw(cls, hardware: Any, launch_overhead_s: float,
                sfu_ops_per_cycle: float = 16.0,
                registers_per_sm: int = 65536,
                register_allocation_per_thread: int = 8) -> "FlashRates":
        from opmodel import DType
        core = hardware.compute
        sram, l2, hbm = (hardware.memory_levels[k] for k in ("sram", "l2", "hbm"))
        clock = float(core.clock_hz)
        def latency(level):
            if level.latency_cycles is not None:
                return float(level.latency_cycles)
            if level.latency_s > 0:
                return float(level.latency_s) * clock
            raise ValueError(f"{level.name} latency is required for FlashAttention")
        if core.tensor_latency_cycles is None or core.shared_memory_bytes_per_sm is None:
            raise ValueError("tensor latency and per-SM shared memory are required for FlashAttention")
        gpu = GpuSpec(clock, int(core.num_sms), float(core.tensor_flops_per_s[DType.BF16]),
                      int(l2.size_bytes), float(l2.bandwidth_bytes_per_s),
                      int(hbm.size_bytes), float(hbm.bandwidth_bytes_per_s))
        return cls(gpu, float(core.vector_flops_per_s[DType.FP32]), int(core.shared_memory_bytes_per_sm),
                   float(launch_overhead_s), float(sram.bandwidth_bytes_per_s), latency(sram),
                   latency(l2), latency(hbm), float(core.tensor_latency_cycles),
                   float(sfu_ops_per_cycle), int(core.registers_per_sm),
                   int(getattr(core, "register_allocation_per_thread", None) or register_allocation_per_thread),
                   int(getattr(core, "shared_memory_bytes_per_block", None) or core.shared_memory_bytes_per_sm),
                   int(core.max_ctas_per_sm), int(core.max_warps_per_sm),
                   **({"l2_bypass": True} if getattr(hardware, "l2_bypass", False) else {}))

def l2_timeline_terms(rates: FlashRates, clock: float) -> tuple[float, FlashRates]:
    if rates.l2_bypass:
        return math.inf, replace(rates, l2_latency_cycles=0.0)
    return rates.gpu.l2_bandwidth_bytes_per_s / clock, rates

@dataclass(frozen=True)
class FlashSpec:
    q_heads: int
    kv_heads: int
    head_dim: int
    batch: int
    q_len: int
    kv_len: int
    causal: bool = True
    op: str = "fwd"
    source: str = "vllm"
    kv_layout: str = "contiguous"
    block_size: int = 0
    block_table_width: int = 0
    dtype: str = "bf16"
    sm_count: int = field(kw_only=True)
    rates: FlashRates | None = None
    spills: bool = False

    fast_forward: tuple | None = None

    def __post_init__(self):
        if not isinstance(self.sm_count, int) or isinstance(self.sm_count, bool) or self.sm_count <= 0:
            raise ValueError("sm_count must be a positive integer")

@dataclass(frozen=True)
class FlashKernel:
    name: str
    grid: tuple[int, int, int]
    ctas: int
    splits: int
    issued_tensor_flops: int
    useful_tensor_flops: int
    vector_ops: dict[str, int]
    sfu_ops: dict[str, int]
    smem_read_bytes: int
    smem_write_bytes: int
    l2_read_bytes: int
    l2_write_bytes: int
    reads: tuple[tuple[str, int], ...]
    writes: tuple[tuple[str, int], ...]
    aliases: dict[str, tuple[str, ...]]
    time_s: float
    diagnostics: dict[str, Any]

    smem_staging_bytes: int = 0

    @property
    def vector_fp32_flops(self) -> int:
        return self.vector_ops["fadd"] + self.vector_ops["fmul"] + 2 * self.vector_ops["ffma"]

    @property
    def vector_fp16_flops(self) -> int:
        return self.vector_ops["hadd"] + self.vector_ops["hmul"] + 2 * self.vector_ops["hfma"]

def num_splits_heuristic(batch_nheads_mblocks: int, num_sms: int,
                         num_n_blocks: int, max_splits: int = 128) -> int:
    if min(batch_nheads_mblocks, num_sms, num_n_blocks, max_splits) < 1:
        raise ValueError("split heuristic needs positive counts")
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1
    cap = min(max_splits, num_sms, num_n_blocks)
    efficiencies = []
    for splits in range(1, cap + 1):
        if splits > 1 and _ceildiv(num_n_blocks, splits) == _ceildiv(num_n_blocks, splits - 1):
            efficiencies.append(0.0)
        else:
            waves = batch_nheads_mblocks * splits / num_sms
            efficiencies.append(waves / math.ceil(waves))
    best = max(efficiencies)
    return next(s for s, eff in enumerate(efficiencies, 1) if eff >= 0.85 * best)

def _validate(spec: FlashSpec, *, describe_only: bool = False) -> FlashRates | None:
    for name in ("q_heads", "kv_heads", "head_dim", "batch", "q_len", "kv_len", "sm_count"):
        value = getattr(spec, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"FlashSpec.{name} must be a positive integer")
    if spec.q_heads % spec.kv_heads:
        raise ValueError("FlashSpec local q_heads must be divisible by kv_heads")
    if spec.source not in ("vllm", "dao", "dao-pip-unknown"):
        raise ValueError("FlashSpec.source must be vllm, dao or dao-pip-unknown")
    if spec.dtype != "bf16":
        raise NotImplementedError("FlashAttention supports BF16 only: resource entries are the BF16 instantiations")
    if spec.op not in ("fwd", "bwd"):
        raise ValueError("FlashSpec.op must be fwd or bwd")
    if spec.op == "bwd" and (spec.source == "vllm" or spec.kv_layout != "contiguous"):
        raise NotImplementedError("FlashAttention backward describes the contiguous Dao training kernels")
    if spec.head_dim not in (64, 128, 192):
        raise NotImplementedError("S1 source traits cover head_dim 64 and 128 (192: vLLM ordinary forward only)")
    if spec.head_dim == 192 and (spec.op != "fwd" or spec.source != "vllm" or spec.kv_layout != "contiguous"
                                 or spec.spills):
        raise NotImplementedError("head_dim 192 is described for the vLLM non-split forward without spills only")
    if spec.kv_layout not in ("contiguous", "paged"):
        raise ValueError("FlashSpec.kv_layout must be contiguous or paged")
    if spec.kv_layout == "paged":
        if spec.source != "vllm":
            raise NotImplementedError("S2 paged KV uses the vLLM kvcache host; Dao uses contiguous mha_fwd")
        for name in ("block_size", "block_table_width"):
            value = getattr(spec, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"paged FlashSpec.{name} must be a positive integer")

        if spec.block_size % 16:
            raise ValueError("paged FlashSpec.block_size must be divisible by 16")
        if spec.kv_len > spec.block_size * spec.block_table_width:
            raise ValueError("actual KV length exceeds block-table capacity")
    if describe_only:
        return None
    rates = spec.rates
    if rates is None:
        raise ValueError("FlashSpec.rates must come from the hardware configuration")
    if rates.gpu.num_sms != spec.sm_count:
        raise ValueError("FlashSpec.sm_count disagrees with hardware rates")
    for value in (rates.gpu.clock_hz, rates.gpu.tensor_flops_per_s,
                  rates.gpu.l2_bandwidth_bytes_per_s, rates.gpu.hbm_bandwidth_bytes_per_s,
                  rates.vector_flops_per_s, rates.sfu_ops_per_cycle, rates.smem_bandwidth_bytes_per_s,
                  rates.shared_memory_bytes_per_sm, rates.registers_per_sm,
                  rates.register_allocation_per_thread, rates.shared_memory_bytes_per_block,
                  rates.max_ctas_per_sm, rates.max_warps_per_sm):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Flash rates/capacities must be finite and positive")
    if not math.isfinite(rates.launch_overhead_s) or rates.launch_overhead_s < 0:
        raise ValueError("Flash launch overhead must be finite and nonnegative")
    return rates

def _dispatch(spec: FlashSpec) -> tuple[FlashSpec, int, bool]:
    forced = spec.kv_layout == "paged"

    swap = spec.q_len == 1 and spec.q_heads > spec.kv_heads
    if swap:
        spec = replace(spec, q_heads=spec.kv_heads,
                       q_len=spec.q_heads // spec.kv_heads, causal=False)
    elif spec.q_len == 1 and spec.causal:
        spec = replace(spec, causal=False)
    splits = 1
    if spec.source in ("dao", "dao-pip-unknown") or swap or forced:
        block_n = 256 if spec.head_dim == 64 else 128
        heuristic_len = spec.block_table_width * spec.block_size if forced else spec.kv_len
        splits = num_splits_heuristic(spec.batch * spec.q_heads * _ceildiv(spec.q_len, 64),
                                      2 * spec.sm_count, _ceildiv(heuristic_len, block_n))
    return spec, splits, forced or splits > 1

@lru_cache(maxsize=1)
def _compiled_resources() -> dict:
    return json.loads(Path(__file__).with_name("flash_kernel_resources.json").read_text())

def _occupancy(spec: FlashSpec, rates: FlashRates, kind: str, smem: int) -> tuple[int, dict]:
    build = "dao" if spec.source == "dao-pip-unknown" else spec.source
    row = _compiled_resources()["builds"][build][kind][str(spec.head_dim)]
    if smem > rates.shared_memory_bytes_per_block:
        raise ValueError("FlashAttention CTA exceeds per-block shared memory limit")
    granularity = rates.register_allocation_per_thread
    warps = _ceildiv(row["threads_per_cta"], 32)
    hardware_bound = min(rates.max_ctas_per_sm, rates.max_warps_per_sm // warps)
    bounds = []
    for reg in row["register_range"]:
        rounded = _ceildiv(reg, granularity) * granularity
        register_bound = rates.registers_per_sm // (row["threads_per_cta"] * rounded)
        bounds.append(min(register_bound, hardware_bound, rates.shared_memory_bytes_per_sm // smem)
                      if smem else min(register_bound, hardware_bound))
    if min(bounds) < 1:
        raise ValueError("hardware resources cannot hold a FlashAttention CTA")
    if bounds[0] != bounds[-1] and kind != "combine":
        raise NotImplementedError(f"{kind} occupancy ambiguity across compiled register variants")

    resident = min(bounds)
    return resident, {"registers_per_thread": row["registers_per_thread"],
                      "register_range": row["register_range"],
                      "threads_per_cta": row["threads_per_cta"],
                      "register_occupancy_bounds": bounds,
                      "resource_data": "flash_kernel_resources.json",
                      "resident_ctas_per_sm": resident,
                      **({"source_assumption": "dao-pip-unknown: Dao 979702c8 kernels/traits and fused vector lowering; NGC build unverified",
                          "compiled_resource_proxy": "torch 2.10 BF16 sm80; provisional stand-in",
                          "backward_assumption": "non-deterministic default also used for Megatron"}
                         if spec.source == "dao-pip-unknown" else {})}

_VECTOR_KEYS = ("fadd", "fmul", "ffma", "fmnmx", "hadd", "hmul", "hfma")

def _tile_vectors(m: int, n: int, d: int, first: bool, source: str) -> tuple[dict, dict]:
    row_lanes = 4 * m
    vec = dict.fromkeys(_VECTOR_KEYS, 0)
    vec["fadd"] = m * n - row_lanes if first else m * n + row_lanes
    vec["fmnmx"] = m * n + row_lanes if first else m * n + 2 * row_lanes
    vec["fmul"] = row_lanes if first else 3 * row_lanes + m * d

    if source == "dao":
        vec["fmul"] += m * n
        vec["fadd"] += m * n
    else:
        vec["ffma"] = m * n

    return vec, {"ex2": m * n + (0 if first else row_lanes), "rcp": 0, "lg2": 0}

def _epilogue_vectors(m: int, d: int) -> tuple[dict, dict]:
    vec = dict.fromkeys(_VECTOR_KEYS, 0)
    vec.update(fadd=8 * m, fmul=m * d + m, ffma=m)
    return vec, {"ex2": 0, "rcp": m, "lg2": m}

class _Node:
    __slots__ = ("resource", "service", "latency", "owner", "callback", "pending", "ready", "done", "children")

    def __init__(self, resource, service, latency, owner, callback=None, pending=0, ready=0.0, done=None,
                 children=None):
        self.resource = resource
        self.service = service
        self.latency = latency
        self.owner = owner
        self.callback = callback
        self.pending = pending
        self.ready = ready
        self.done = done
        self.children = [] if children is None else children

class _Waiters:
    __slots__ = ("order", "acquired", "arrivals")

    def __init__(self):
        self.order: list = []
        self.acquired: list = []
        self.arrivals = 0

class _Waiter:
    __slots__ = ("node", "waiters", "number", "arrival_time", "arrival_serial", "first", "end", "queued")

    def __init__(self, node, waiters, number, arrival_time, arrival_serial, first):
        self.node = node
        self.waiters = waiters
        self.number = number
        self.arrival_time = arrival_time
        self.arrival_serial = arrival_serial
        self.first = first
        self.end = first
        self.queued = False

    def __lt__(self, other):
        if other.__class__ is not _Waiter:
            return True
        if self.waiters is other.waiters:
            return self.number < other.number
        mine, theirs = self.waiters.acquired, other.waiters.acquired
        i, j = self.end - 1, other.end - 1
        while True:
            my_arrival, their_arrival = i < self.first, j < other.first
            a = self.arrival_time if my_arrival else mine[i]
            b = other.arrival_time if their_arrival else theirs[j]
            if a != b:
                return a < b
            if my_arrival or their_arrival:
                if my_arrival and their_arrival:
                    return self.arrival_serial < other.arrival_serial
                return their_arrival
            i -= 1
            j -= 1

    def __gt__(self, other):
        if other.__class__ is not _Waiter:
            return False
        return other.__lt__(self)

class _Timeline:
    def __init__(self):
        self.queue: list = []
        self.busy: dict[str, float] = {}
        self.service: dict[str, float] = {}
        self.serial = 0
        self.now = 0.0
        self.waiting: dict[str, _Waiters] = {}

    def push(self, when: float, completion: bool, node: _Node) -> None:
        self.serial += 1
        heapq.heappush(self.queue, (when, 0 if completion else 1, node.owner, self.serial, node))

    def add(self, deps=(), resource=None, service=0.0, latency=0.0,
            owner=0, callback=None) -> _Node:
        ready = self.now
        node = _Node(resource, service, latency, owner, callback)
        pending = 0
        for dep in deps:
            done = dep.done
            if done is not None:
                if done > ready:
                    ready = done
            else:
                dep.children.append(node)
                pending += 1
        node.ready = ready
        node.pending = pending
        if pending == 0:
            self.serial += 1
            heapq.heappush(self.queue, (ready, 1, owner, self.serial, node))
        return node

    def run(self) -> float:
        thresholds = gc.get_threshold()
        gc.set_threshold(max(thresholds[0], 100_000), *thresholds[1:])
        try:
            return self._run()
        finally:
            gc.set_threshold(*thresholds)

    def _run(self) -> float:
        queue = self.queue
        busy = self.busy
        service = self.service
        waiting = self.waiting
        pop = heapq.heappop
        push = heapq.heappush
        while queue:
            now, event, owner, token, node = pop(queue)
            self.now = now
            if event == 1:
                resource = node.resource
                if not resource:
                    self.serial += 1
                    duration = node.latency if node.latency > node.service else node.service
                    push(queue, (now + duration, 0, owner, self.serial, node))
                    continue
                waiters = waiting.get(resource)
                if token.__class__ is _Waiter:
                    order = waiters.order
                    if busy[resource] > now:
                        if order[0][2] is token:
                            token.end = len(waiters.acquired)
                            push(queue, (busy[resource], 1, owner, token, node))
                        else:
                            token.queued = False
                        continue
                    if order[0][2] is not token:
                        raise RuntimeError("flash timeline: a waiting node was reached before the head of its queue")
                    pop(order)
                elif busy.get(resource, 0.0) > now:
                    self.serial += 1
                    if waiters is None:
                        waiters = waiting[resource] = _Waiters()
                    waiters.arrivals += 1
                    waiter = _Waiter(node, waiters, waiters.arrivals, now, token, len(waiters.acquired))
                    order = waiters.order
                    push(order, (owner, waiters.arrivals, waiter))
                    if order[0][2] is waiter:
                        waiter.queued = True
                        push(queue, (busy[resource], 1, owner, waiter, node))
                    continue

                free = now + node.service
                busy[resource] = free
                service[resource] = service.get(resource, 0.0) + node.service
                self.serial += 1

                duration = node.latency if node.latency > node.service else node.service
                push(queue, (now + duration, 0, owner, self.serial, node))
                if waiters is not None:
                    order = waiters.order
                    if order:
                        if free > now:
                            waiters.acquired.append(now)
                            self.serial += len(order)
                        head = order[0][2]
                        if not head.queued:
                            head.queued = True
                            head.end = len(waiters.acquired)
                            push(queue, (free, 1, head.node.owner, head, head.node))
                    elif waiters.acquired:
                        waiters.acquired.clear()
            else:
                node.done = now
                for child in node.children:
                    pending = child.pending - 1
                    child.pending = pending
                    if now > child.ready:
                        child.ready = now
                    if pending == 0:
                        self.serial += 1
                        push(queue, (child.ready, 1, child.owner, self.serial, child))
                node.children.clear()
                if node.callback:
                    node.callback(node)
        return self.now

def _structural_time(spec: FlashSpec, rates: FlashRates, m: int, n: int,
                     tiles: list[int], resident: int, ldgsts: int,
                     unique_reads: int, ranges: list[tuple[int, int]] | None = None,
                     output_element_bytes: int = 2,
                     spill_program=None, warps: int = 4) -> tuple[float, dict]:
    gpu = rates.gpu
    clock, sms, d = gpu.clock_hz, gpu.num_sms, spec.head_dim

    tensor_rate = gpu.tensor_flops_per_s / clock / sms / 4
    smem_rate = rates.smem_bandwidth_bytes_per_s / clock / sms
    l2_rate, rates = l2_timeline_terms(rates, clock)
    hbm_rate = gpu.hbm_bandwidth_bytes_per_s / clock

    vector_rate = rates.vector_flops_per_s / clock / sms / 2 / 4
    sfu_rate = rates.sfu_ops_per_cycle / 4
    mma_latency = rates.mma_latency_cycles

    hbm_fraction = 1.0 if rates.l2_bypass else unique_reads / ldgsts if ldgsts else 0.0
    native = None

    ff = spec.fast_forward if ranges is None else None
    if spill_program is None:
        import flash_native
        native = flash_native.forward(
            sms=sms, resident=resident, warps=warps, m=m, n=n, d=d, q_len=spec.q_len, kv_len=spec.kv_len,
            dao=spec.source == "dao", output_element_bytes=output_element_bytes, tiles=tiles, ranges=ranges,
            smem_rate=smem_rate, l2_rate=l2_rate, hbm_rate=hbm_rate, tensor_rate=tensor_rate,
            vector_rate=vector_rate, sfu_rate=sfu_rate, mma_latency=mma_latency,
            smem_latency=rates.smem_latency_cycles, l2_latency=rates.l2_latency_cycles,
            hbm_latency=rates.hbm_latency_cycles, hbm_fraction=hbm_fraction, ff=ff)
    if native is None and ff is not None:
        raise ValueError("flash_fast_forward needs the native flash timeline (and no spill program)")
    if native is not None:
        cycles, starts, finishes, services = native
    else:
        cycles, starts, finishes, services = _structural_schedule(
            spec, rates, m, n, d, tiles, resident, ranges, output_element_bytes, spill_program, warps, sms,
            smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate, mma_latency, hbm_fraction)
    return cycles / clock + rates.launch_overhead_s, {
        "kernel_cycles": cycles, "cta_start_cycles": starts, "cta_end_cycles": finishes,
        "resource_service_cycles": services,
        "hbm_timing_unique_read_bytes": unique_reads,
        "hbm_timing_policy": "extended_timing first-touch unique demand; activity remains L2-sim owned",
        "rate_contract": "opmodel HardwareSpec rates and latencies; util=1",
        "cta_order": "blockIdx.x fastest; next free SM slot; deterministic SM-fast initial slots",
    }

def _structural_schedule(spec, rates, m, n, d, tiles, resident, ranges, output_element_bytes, spill_program, warps,
                         sms, smem_rate, l2_rate, hbm_rate, tensor_rate, vector_rate, sfu_rate, mma_latency,
                         hbm_fraction):
    timeline = _Timeline()
    starts = [0.0] * len(tiles)
    finishes = [0.0] * len(tiles)
    next_cta = 0

    def launch(sm: int, slot: int) -> None:
        nonlocal next_cta
        if next_cta == len(tiles):
            return
        cid = next_cta
        next_cta += 1
        starts[cid] = timeline.now
        mb = cid % _ceildiv(spec.q_len, m)
        valid_q = min(m, spec.q_len - mb * m)
        nt = tiles[cid]
        n_start, n_end = ranges[cid] if ranges is not None else (0, nt)
        smem_resource = f"smem:{sm}"
        spill_writes = []

        def node(deps=(), resource=None, service=0.0, latency=0.0, callback=None):
            return timeline.add(deps, resource, service, latency, cid, callback)

        def spill(phase, deps=(), index=0):
            if spill_program is None:
                return tuple(deps)
            from flash_attention_spills import service
            return service(spill_program, phase, index, nt, deps, node,
                           smem_resource, smem_rate, l2_rate, 4, spill_writes,
                           smem_latency_cycles=rates.smem_latency_cycles,
                           l2_latency_cycles=rates.l2_latency_cycles)

        def smem(count, deps=()):
            return node(deps, smem_resource, count / smem_rate,
                        rates.smem_latency_cycles if count else 0)

        def load(count, deps=()):
            l2_count = count
            l2 = node(deps, "l2", l2_count / l2_rate, rates.l2_latency_cycles if l2_count else 0)
            hbm = node(deps, "hbm", count * hbm_fraction / hbm_rate,
                       rates.hbm_latency_cycles if count and hbm_fraction else 0)
            return smem(count, (l2, hbm))

        def vector(count, warp, deps):
            return node(deps, f"vector:{sm}:{warp % 4}", count / vector_rate)

        def sfu(count, warp, deps):
            return node(deps, f"sfu:{sm}:{warp % 4}", count / sfu_rate)

        def gemm(warp, deps, qk):
            inner, cols = (d, n) if qk else (n, d)

            operands = ((m // warps + cols) if qk else cols) * 16 * 2
            load_service = operands / smem_rate
            load_latency = rates.smem_latency_cycles if operands else 0
            math_service = 2 * (m // warps) * cols * 16 / tensor_rate
            tensor_resource = f"tensor:{sm}:{warp % 4}"
            add = timeline.add
            loads, maths = [], []
            for group in range(inner // 16):
                prereq = list(deps)
                if loads:
                    prereq.append(loads[-1])
                if group > 1:
                    prereq.append(maths[group - 2])
                ld = add(prereq, smem_resource, load_service, load_latency, cid)
                loads.append(ld)
                md = [ld] + ([maths[-1]] if maths else [])
                mm = add(md, tensor_resource, math_service, mma_latency, cid)
                maths.append(mm)
            return maths[-1]

        def complete(_):
            finishes[cid] = timeline.now
            launch(sm, slot)

        def epilogue(deps):
            deps = spill("epilogue", deps)
            if nt:
                normed = []
                for w in range(warps):
                    reduction = vector(8 * m / warps, w, deps)
                    reciprocal = sfu(m / warps, w, (reduction,))
                    logarithm = sfu(m / warps, w, (reduction,))
                    norm = vector((m * d + 2 * m) / warps, w, (reciprocal, logarithm))
                    normed.append(norm)
                staged = smem(m * d * output_element_bytes, normed)
                loaded = smem(m * d * output_element_bytes, (staged,))
                deps = (loaded,)
            stored = valid_q * (d * output_element_bytes + 4)
            l2_count = stored
            hbm_count = stored
            l2 = node(deps, "l2", l2_count / l2_rate, rates.l2_latency_cycles if l2_count else 0)
            hbm = node(deps, "hbm", hbm_count / hbm_rate, rates.hbm_latency_cycles if hbm_count else 0)
            node((l2, hbm, *spill_writes), callback=complete)

        if nt == 0:
            epilogue(())
            return
        prologue = spill("prologue")
        qload = load(valid_q * d * 2, prologue)
        last_n = n_end - 1
        kload = load(min(n, spec.kv_len - last_n * n) * d * 2)

        def tile(index, prev, ready_k):
            block = n_end - 1 - index
            kv_rows = min(n, spec.kv_len - block * n)
            begin = node(spill("tile", tuple(prev) + (qload, ready_k), index))

            vload = load(kv_rows * d * 2, (begin,))

            vzero = smem((n - kv_rows) * d * 2, (begin,)) if kv_rows < n else vload
            qk = [gemm(w, (begin,), True) for w in range(warps)]
            barrier = node(tuple(qk) + (vload, vzero))
            next_k = load(n * d * 2, (barrier,)) if block > n_start else None
            pv = []
            for w in range(warps):
                row_lanes = 4 * m / warps
                max_ops = (m * n + (4 * m if index == 0 else 8 * m)) / warps
                reduction = vector(max_ops, w, (barrier,))
                if index:
                    alpha_args = vector(2 * row_lanes, w, (reduction,))
                    alpha = sfu(row_lanes, w, (alpha_args,))
                    reduction = vector(row_lanes + m * d / warps, w, (alpha,))
                scaled_max = vector(row_lanes, w, (reduction,))
                score = vector(m * n / warps * (2 if spec.source == "dao" else 1), w, (scaled_max,))
                exp = sfu(m * n / warps, w, (score,))
                summed = vector((m * n - (4 * m if index == 0 else 0)) / warps, w, (exp,))
                pv.append(gemm(w, (summed,), False))
            if block > n_start:
                node(pv, callback=lambda done: tile(index + 1, (done,), next_k))
            else:
                node(pv, callback=lambda done: epilogue((done,)))

        tile(0, (), kload)

    for slot in range(resident):
        for sm in range(sms):
            launch(sm, slot)
    cycles = timeline.run()
    services = {}
    for resource, value in timeline.service.items():
        kind = resource.split(":")[0]
        services[kind] = services.get(kind, 0.0) + value
    return cycles, starts, finishes, services

def _flash_cache_key(*inputs):
    return pickle.dumps(inputs, protocol=4)

def _flash_resource_key(spec):
    build = "dao" if spec.source == "dao-pip-unknown" else spec.source
    rows = _compiled_resources()["builds"][build]
    kinds = (("bwd_dot_do_o", "bwd_convert_dq", "bwd_dq_dk_dv")
             if spec.op == "bwd" else ("fwd",) if spec.head_dim == 192 else ("fwd", "splitkv", "combine"))
    result = []
    for kind in kinds:
        row = rows[kind][str(spec.head_dim)]
        result.append(row if spec.spills else
                      (row["registers_per_thread"], row["register_range"],
                       row["threads_per_cta"]))
    return pickle.dumps(result, protocol=4)

@lru_cache(maxsize=128)
def _cached_flash_kernels(spec, describe_only, exact_key):
    if describe_only:
        return _flash_kernels_uncached(spec, describe_only=True)

    import flash_disk_cache
    return flash_disk_cache.memo("kernels", exact_key, lambda: _flash_kernels_uncached(spec, describe_only=False))

def flash_kernels(spec: FlashSpec, *, describe_only: bool = False) -> list[FlashKernel]:
    _validate(spec, describe_only=describe_only)

    if spec.op == "bwd" and spec.q_heads != spec.kv_heads:
        return _flash_kernels_uncached(spec, describe_only=describe_only)

    if spec.rates is not None:
        spec = replace(spec, rates=replace(spec.rates, reduction_time_s=None))
    return deepcopy(_cached_flash_kernels(spec, describe_only,
                                        _flash_cache_key(spec, describe_only, None if describe_only else _flash_resource_key(spec))))

def _flash_kernels_uncached(spec: FlashSpec, *, describe_only: bool = False) -> list[FlashKernel]:
    rates = _validate(spec, describe_only=describe_only)
    if spec.op == "bwd":
        if describe_only:
            from flash_attention_backward import backward_kernels
            return backward_kernels(spec, None, describe_only=True)
        from flash_attention_backward import backward_kernels
        return backward_kernels(spec, rates)
    original_spec = spec
    spec, splits, split_kernel = _dispatch(spec)
    if split_kernel:
        result = [_splitkv_kernel(spec, rates, splits, original_spec, describe_only=describe_only)]
        if splits > 1:
            result.append(_combine_kernel(spec, rates, splits, describe_only=describe_only))
        if spec.spills and not describe_only:
            from flash_attention_spills import apply_counts, select_program
            result[0] = apply_counts(result[0], select_program(spec, "splitkv", splits),
                                     result[0].diagnostics["tiles_per_cta"], 128)
        return result

    m, n, warps = 128, (128 if spec.head_dim == 64 else 64), (8 if spec.head_dim == 192 else 4)
    d = spec.head_dim
    grid = (_ceildiv(spec.q_len, m), spec.batch, spec.q_heads)
    per_query = []
    for mb in range(grid[0]):
        end = _ceildiv(spec.kv_len, n)
        if spec.causal:
            end = min(end, _ceildiv((mb + 1) * m + spec.kv_len - spec.q_len, n))
        per_query.append(max(0, end))
    tiles = per_query * (spec.batch * spec.q_heads)
    ctas = math.prod(grid)
    total_tiles = sum(tiles)
    active_ctas = sum(t > 0 for t in tiles)
    issued = 4 * m * n * d * total_tiles
    useful_pairs = sum(min(spec.kv_len, max(0, q + spec.kv_len - spec.q_len + 1))
                       for q in range(spec.q_len)) if spec.causal else spec.q_len * spec.kv_len
    useful = 4 * d * spec.batch * spec.q_heads * useful_pairs
    vectors = dict.fromkeys(_VECTOR_KEYS, 0)
    sfus = dict(ex2=0, rcp=0, lg2=0)
    for count, component in ((active_ctas, _tile_vectors(m, n, d, True, spec.source)),
                             (total_tiles - active_ctas, _tile_vectors(m, n, d, False, spec.source)),
                             (active_ctas, _epilogue_vectors(m, d))):
        vec, sfu = component
        for k in vectors:
            vectors[k] += count * vec[k]
        for k in sfus:
            sfus[k] += count * sfu[k]

    q_load = sum(min(m, spec.q_len - mb * m) * d * 2
                 for mb, t in enumerate(per_query) if t) * spec.batch * spec.q_heads
    kv_load = sum(2 * min(spec.kv_len, t * n) * d * 2 for t in per_query) * spec.batch * spec.q_heads
    ldgsts = q_load + kv_load

    v_clear = sum((n - min(n, spec.kv_len - (t - 1) * n)) * d * 2
                  for t in per_query if t) * spec.batch * spec.q_heads

    ldsm = warps * (m // warps * d + 2 * n * d) * 2 * total_tiles

    smem_o = active_ctas * m * d * 2
    smem_alloc = 2 * d * (m + 2 * n)
    resident, resources = (None, {}) if describe_only else _occupancy(spec, rates, "fwd", smem_alloc)
    q_bytes = spec.batch * spec.q_heads * spec.q_len * d * 2
    kv_bytes = spec.batch * spec.kv_heads * spec.kv_len * d * 2
    lse_bytes = spec.batch * spec.q_heads * spec.q_len * 4
    reads = (("q", q_load), ("k", kv_bytes), ("v", kv_bytes))
    if spec.source == "vllm":
        reads += (("cu_seqlens", 2 * (spec.batch + 1) * 4),)
    writes = (("o", q_bytes), ("lse", lse_bytes))

    aliases = {"qkv": ("q", "k", "v")} if spec.source == "vllm" else {}
    spill_program = None
    if spec.spills and not describe_only:
        from flash_attention_spills import select_program
        spill_program = select_program(spec, "fwd")
    if describe_only:
        time_s, timing = 0.0, {"time_mode": "counts_only", "timing_status": "UNPRICED"}
    else:
        time_s, timing = _structural_time(spec, rates, m, n, tiles, resident, ldgsts,
                                        q_load + 2 * kv_bytes, spill_program=spill_program, warps=warps)

    counter_map = {op: f"sm__sass_thread_inst_executed_op_{op}_pred_on.sum"
                   for op in _VECTOR_KEYS if op != "fmnmx"}
    counter_map.update(fmnmx=None, ex2=None, rcp=None, lg2=None,
                       issued_tensor_flops="tensor_flops_old",
                       useful_tensor_flops=None, smem_ldsm="smem_ldsm",
                       smem_ldgsts="smem_ldgsts", smem_ld="smem_ld", smem_st="smem_st",
                       l2_read_bytes="xbar2l1_read_bytes", l2_write_bytes="l12xbar_write_bytes")
    if not describe_only:
        tensor_sm = rates.gpu.tensor_flops_per_s / rates.gpu.clock_hz / spec.sm_count
        vector_sm = rates.vector_flops_per_s / rates.gpu.clock_hz / spec.sm_count / 2
        smem_sm = rates.smem_bandwidth_bytes_per_s / rates.gpu.clock_hz / spec.sm_count
        phase_bounds = {
            "qk": {"tensor": 2 * m * n * d / tensor_sm,
                   "smem": (m * d + warps * n * d) * 2 / smem_sm,
                   "mma_dependency": d // 16 * rates.mma_latency_cycles},
            "pv": {"tensor": 2 * m * n * d / tensor_sm,
                   "smem": warps * n * d * 2 / smem_sm,
                   "mma_dependency": n // 16 * rates.mma_latency_cycles},
        }
        for label, first in (("softmax_first", True), ("softmax_later", False)):
            v, s = _tile_vectors(m, n, d, first, spec.source)
            phase_bounds[label] = {"vector": sum(v.values()) / vector_sm,
                                   "sfu": sum(s.values()) / rates.sfu_ops_per_cycle}
    diagnostics = {
        "block_m": m, "block_n": n, "warps": warps,
        "smem_bytes_per_cta": smem_alloc,
        **({} if describe_only else {"resident_ctas_per_sm": resident}),
        "tiles_per_cta": tiles, "total_tiles": total_tiles,
        **({} if describe_only else {"waves": _ceildiv(ctas, spec.sm_count * resident)}),
        "smem_ldsm": ldsm, "smem_ldgsts": ldgsts, "smem_ld": smem_o, "smem_st": smem_o + v_clear,
        "smem_v_tail_clear": v_clear,
        **({} if describe_only else {"phase_service_lower_bounds_cycles": phase_bounds,
            "bottleneck_per_phase": {p: max(bounds, key=bounds.get) for p, bounds in phase_bounds.items()}}),
        "counter_map": counter_map, "vector_count_scope": "source expressions; compiler predication not fitted",
        "epilogue_sfu_scope": "one reciprocal/log2 per row per DESIGN section 9",
        "time_mode": "structural", **resources,
        **({} if describe_only else {"clock_hz": rates.gpu.clock_hz}),
        "host_gqa_swap": spec.q_heads != original_spec.q_heads,
        "l2_byte_scope": "delivered logical operand/output bytes, not lookup bytes or a fitted sector multiplier",
        **timing,
    }
    kernel = FlashKernel("flash_fwd_kernel", grid, ctas, 1, issued, useful, vectors, sfus,
                        ldsm + smem_o, smem_o + v_clear, ldgsts, q_bytes + lse_bytes,
                        reads, writes, aliases, time_s, diagnostics, smem_staging_bytes=ldgsts)
    if spec.spills and not describe_only:
        from flash_attention_spills import apply_counts
        kernel = apply_counts(kernel, spill_program, tiles, 128)
    return [kernel]

def _splitkv_kernel(spec: FlashSpec, rates: FlashRates, splits: int,
                    original_spec: FlashSpec, *, describe_only=False) -> FlashKernel:
    m, n, d, warps = 64, (256 if spec.head_dim == 64 else 128), spec.head_dim, 4
    qm = _ceildiv(spec.q_len, m)
    grid = (qm, splits, spec.batch * spec.q_heads) if splits > 1 else (qm, spec.batch, spec.q_heads)

    nblocks = _ceildiv(spec.kv_len, n)
    per_split = _ceildiv(nblocks, splits)
    local_ranges = []
    for split in range(splits):
        for mb in range(qm):
            start, end = split * per_split, min(nblocks, (split + 1) * per_split)
            if spec.causal:
                end = min(end, _ceildiv((mb + 1) * m + spec.kv_len - spec.q_len, n))
            local_ranges.append((start, max(start, end)))
    ranges = local_ranges * (spec.batch * spec.q_heads)
    tiles = [end - start for start, end in ranges]
    active = sum(t > 0 for t in tiles)
    total_tiles = sum(tiles)
    issued = 4 * m * n * d * total_tiles
    useful_pairs = sum(min(spec.kv_len, max(0, q + spec.kv_len - spec.q_len + 1))
                       for q in range(spec.q_len)) if spec.causal else spec.q_len * spec.kv_len
    useful = 4 * d * spec.batch * spec.q_heads * useful_pairs
    vectors, sfus = dict.fromkeys(_VECTOR_KEYS, 0), dict(ex2=0, rcp=0, lg2=0)

    for count, (vec, sfu) in (
        (active, _tile_vectors(m, n, d, True, spec.source)),
        (total_tiles - active, _tile_vectors(m, n, d, False, spec.source)),
        (active, _epilogue_vectors(m, d)),
    ):
        for key in vectors:
            vectors[key] += count * vec[key]
        for key in sfus:
            sfus[key] += count * sfu[key]
    q_load = kv_load = v_clear = 0
    for cid, (start, end) in enumerate(ranges):
        if start == end:
            continue
        valid_q = min(m, spec.q_len - (cid % qm) * m)
        q_load += valid_q * d * 2

        kv_load += 2 * (min(spec.kv_len, end * n) - start * n) * d * 2
        v_clear += (n - min(n, spec.kv_len - (end - 1) * n)) * d * 2
    ldgsts = q_load + kv_load
    ldsm = warps * (m // warps * d + 2 * n * d) * 2 * total_tiles

    output_element = 4 if splits > 1 else 2
    staged = active * m * d * output_element
    smem_alloc = 2 * d * (m + 2 * n)
    resident, resources = (None, {}) if describe_only else _occupancy(spec, rates, "splitkv", smem_alloc)
    rows = spec.batch * spec.q_heads * spec.q_len
    q_bytes = rows * d * 2
    kv_bytes = spec.batch * spec.kv_heads * spec.kv_len * d * 2
    if spec.kv_layout == "paged":
        pages = _ceildiv(spec.kv_len, spec.block_size)
        block_ranges = [(b, h, 0, pages) for b in range(spec.batch) for h in range(spec.kv_heads)]
        reads = (("q", q_bytes),
                 ("kv_cache", 2 * spec.batch * spec.kv_heads * pages * spec.block_size * d * 2),
                 ("block_table", spec.batch * pages * 4))
        aliases = {"qkv": ("q",)}
    else:
        block_ranges = []
        reads = (("q", q_bytes), ("k", kv_bytes), ("v", kv_bytes))
        aliases = {"qkv": ("q", "k", "v")} if spec.source == "vllm" else {}
    writes = (("out_accum", splits * rows * d * 4), ("lse_accum", splits * rows * 4)) if splits > 1 else (
        ("o", rows * d * 2), ("lse", rows * 4))
    spill_program = None
    if spec.spills and not describe_only:
        from flash_attention_spills import select_program
        spill_program = select_program(spec, "splitkv", splits)
    if describe_only:
        time_s, timing = 0.0, {"time_mode": "counts_only", "timing_status": "UNPRICED"}
    else:
        time_s, timing = _structural_time(spec, rates, m, n, tiles, resident, ldgsts,
                                        q_bytes + 2 * kv_bytes, ranges, output_element, spill_program=spill_program)
    diagnostics = {
        "block_m": m, "block_n": n, "warps": warps, "smem_bytes_per_cta": smem_alloc,
        "tiles_per_cta": tiles, "n_ranges_per_cta": ranges, "total_tiles": total_tiles,
        "empty_ctas": len(tiles) - active,
        **({} if describe_only else {"waves": _ceildiv(len(tiles), spec.sm_count * resident)}),
        "smem_ldsm": ldsm, "smem_ldgsts": ldgsts, "smem_ld": staged, "smem_st": staged + v_clear,
        "smem_v_tail_clear": v_clear, "host_gqa_swap": spec.q_heads != original_spec.q_heads,
        "kernel_q_len": spec.q_len, "kernel_heads": spec.q_heads,
        "partition_kv_len": spec.kv_len,
        "heuristic_kv_len": spec.block_table_width * spec.block_size if spec.kv_layout == "paged" else spec.kv_len,
        "kv_cache_block_ranges": block_ranges,
        "block_range_schema": "(sequence, kv_head, first_page, end_page), end exclusive",
        "l2_byte_scope": "logical Q/K/V copies and output stores; scalar metadata transaction multiplicity unmodeled",
        "vector_count_scope": "source expressions using S1 lowering; compiler predication not fitted",
        "epilogue_sfu_scope": "one reciprocal/log2 per row per DESIGN section 9",
        **({} if describe_only else {"clock_hz": rates.gpu.clock_hz}),
        "time_mode": "structural", **resources, **timing,
    }
    return FlashKernel("flash_fwd_splitkv_kernel", grid, len(tiles), splits, issued, useful,
                       vectors, sfus, ldsm + staged, staged + v_clear,
                       ldgsts, sum(size for _, size in writes), reads, writes, aliases, time_s, diagnostics,
                       smem_staging_bytes=ldgsts)

def _combine_vectors(mc: int, splits: int, max_splits: int, d: int) -> tuple[dict, dict]:
    nlse = _ceildiv(max_splits * mc, 128)
    reduction = int(math.log2(min(128 // mc, max_splits)))
    initial_exp = 128 * nlse
    scales = splits * mc
    vec = dict.fromkeys(_VECTOR_KEYS, 0)
    vec["fmnmx"] = 128 * (nlse - 1 + reduction)
    vec["fadd"] = initial_exp + 128 * (nlse - 1 + reduction) + 128 + scales

    vec["fmul"] = initial_exp + 128 + scales

    vec["ffma"] = mc * d * splits
    return vec, {"ex2": initial_exp + scales, "lg2": 128, "rcp": 0}

def _combine_kernel(spec: FlashSpec, rates: FlashRates, splits: int, *, describe_only=False) -> FlashKernel:
    mc = 8 if spec.head_dim == 64 else 4
    max_splits = 1 << (splits - 1).bit_length()
    rows = spec.batch * spec.q_heads * spec.q_len
    ctas = _ceildiv(rows, mc)

    smem_alloc = 4 * max_splits * (mc + 1)

    resident, resources = (None, {}) if describe_only else _occupancy(spec, rates, "combine", smem_alloc)
    vec, sfu = _combine_vectors(mc, splits, max_splits, spec.head_dim)

    transpose = 4 * max_splits * mc
    scales = 4 * splits * mc
    smem_read = smem_write = ctas * (transpose + scales)
    reads = (("out_accum", splits * rows * spec.head_dim * 4), ("lse_accum", splits * rows * 4))
    writes = (("o", rows * spec.head_dim * 2), ("lse", rows * 4))
    if describe_only:
        time_s, timing = 0.0, {"time_mode": "counts_only", "timing_status": "UNPRICED"}
    else:
        time_s, timing = _combine_time(spec, rates, splits, mc, max_splits, resident, vec, sfu)
    return FlashKernel(
        "flash_fwd_splitkv_combine_kernel", (ctas, 1, 1), ctas, splits, 0, 0,
        {k: v * ctas for k, v in vec.items()}, {k: v * ctas for k, v in sfu.items()},
        smem_read, smem_write, sum(n for _, n in reads), sum(n for _, n in writes),
        reads, writes, {}, time_s, {
            "block_m": mc, "max_splits": max_splits, "warps": 4,
            "smem_bytes_per_cta": smem_alloc,
            **({} if describe_only else {"waves": _ceildiv(ctas, spec.sm_count * resident)}),
            "smem_ld": smem_read, "smem_st": smem_write, "smem_ldsm": 0, "smem_ldgsts": 0,
            "smem_scale_read_scope": "warp-broadcast distinct words",
            **({} if describe_only else {"combine_occupancy": "minimum over compiled Log_max_splits variants"}),
            "vector_count_scope": "source expressions; expf/logf change-of-base SFU representation, binary lowering unverified",
            "time_mode": "structural", **({} if describe_only else {"clock_hz": rates.gpu.clock_hz}),
            **resources, **timing,
        })

def _combine_time(spec: FlashSpec, rates: FlashRates, splits: int, mc: int,
                  max_splits: int, resident: int, vectors: dict, sfus: dict,
                  ) -> tuple[float, dict]:
    gpu = rates.gpu
    clock, sms, d = gpu.clock_hz, gpu.num_sms, spec.head_dim
    l2_rate, rates = l2_timeline_terms(rates, clock)
    hbm_rate = gpu.hbm_bandwidth_bytes_per_s / clock
    smem_rate = rates.smem_bandwidth_bytes_per_s / clock / sms
    vector_rate = rates.vector_flops_per_s / clock / sms / 2 / 4
    sfu_rate = rates.sfu_ops_per_cycle / 4
    timeline = _Timeline()
    rows = spec.batch * spec.q_heads * spec.q_len
    ctas = _ceildiv(rows, mc)
    starts = [0.0] * ctas
    finishes = [0.0] * ctas
    next_cta = 0

    def launch(sm, slot):
        nonlocal next_cta
        if next_cta == ctas:
            return
        cid = next_cta
        next_cta += 1
        starts[cid] = timeline.now
        valid = min(mc, rows - cid * mc)

        def node(deps=(), resource=None, service=0.0, latency=0.0, callback=None):
            return timeline.add(deps, resource, service, latency, cid, callback)

        def memory(count, deps=()):
            l2_count = count
            hbm_count = count
            l2 = node(deps, "l2", l2_count / l2_rate, rates.l2_latency_cycles if l2_count else 0)
            hbm = node(deps, "hbm", hbm_count / hbm_rate, rates.hbm_latency_cycles if hbm_count else 0)
            return node((l2, hbm))

        def shared(count, deps):
            return node(deps, f"smem:{sm}", count / smem_rate, rates.smem_latency_cycles)

        def finish(_):
            finishes[cid] = timeline.now
            launch(sm, slot)

        lse = memory(splits * valid * 4)
        staged = shared(4 * max_splits * mc, (lse,))
        transposed = shared(4 * max_splits * mc, (staged,))
        norms = []

        nlse = _ceildiv(max_splits * mc, 128)
        reduction = int(math.log2(min(128 // mc, max_splits)))
        for w in range(4):
            vresource, sresource = f"vector:{sm}:{w}", f"sfu:{sm}:{w}"
            maxima = node((transposed,), vresource, vectors["fmnmx"] / 4 / vector_rate)
            args = node((maxima,), vresource, 2 * 128 * nlse / 4 / vector_rate)
            exp = node((args,), sresource, 128 * nlse / 4 / sfu_rate)
            summed = node((exp,), vresource, 128 * (nlse - 1 + reduction) / 4 / vector_rate)
            log = node((summed,), sresource, 128 / 4 / sfu_rate)
            logsum = node((log,), vresource, 2 * 128 / 4 / vector_rate)
            scale_args = node((logsum,), vresource, 2 * splits * mc / 4 / vector_rate)
            norms.append(node((scale_args,), sresource, splits * mc / 4 / sfu_rate))
        scale_store = shared(4 * splits * mc, norms)
        lse_store = memory(valid * 4, norms)

        def output_step(split, deps):
            partial = memory(valid * d * 4, deps)
            scale = shared(mc * 4, (partial,))
            weighted = [node((scale,), f"vector:{sm}:{w}", mc * d / 4 / vector_rate)
                        for w in range(4)]
            if split + 1 < splits:
                node(weighted, callback=lambda done: output_step(split + 1, (done,)))
            else:
                output = memory(valid * d * 2, weighted)
                node((output, lse_store), callback=finish)

        output_step(0, (scale_store,))

    for slot in range(resident):
        for sm in range(sms):
            launch(sm, slot)
    cycles = timeline.run()
    services = {}
    for resource, value in timeline.service.items():
        kind = resource.split(":")[0]
        services[kind] = services.get(kind, 0.0) + value
    return cycles / clock + rates.launch_overhead_s, {
        "kernel_cycles": cycles, "cta_start_cycles": starts,
        "cta_end_cycles": finishes, "resource_service_cycles": services,
        "rate_contract": "opmodel HardwareSpec rates and S1 resources; vector/SFU program order",
        "hbm_timing_unique_read_bytes": splits * rows * (d + 1) * 4,
        "hbm_timing_policy": "extended_timing first-touch unique demand; activity remains L2-sim owned",
        "cta_order": "blockIdx.x fastest; next free SM slot; deterministic SM-fast initial slots",
    }
