from __future__ import annotations

import bisect
import heapq
import math
import re
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from kernel_trace import Access, Kernel, LayerDims, step_trace, SILU_FWD_FLOPS
from kernel_trace_train import (F32, BF16, I64, TrainOpts, flash_num_splits, param_sizes, train_dims,
                                train_step_trace)
from opmodel.split_k import normalize_split_k, split_k_counts, split_k_lengths

MIB = 1 << 20

@dataclass(frozen=True)
class RAccess:
    tensor: str
    nbytes: float
    weight: bool = False
    offset: float = 0.0

SPLIT_K_WORKSPACE = "@split_k:workspace"
SPLIT_K_LT_WORKSPACE = "@split_k:lt_workspace"

@dataclass(frozen=True)
class SplitKKernel(Kernel):
    split_k: int = 1
    reduction_scheme: str = "none"
    slice_lengths: Tuple[int, ...] = ()
    original_reads: Tuple[Access, ...] = ()
    original_writes: Tuple[Access, ...] = ()
    epilogue_reads: Tuple[Access, ...] = ()

def expand_split_k(kernel: Kernel, split_k: int = 1, reduction_scheme="none", *,
                   batch: int = 1, output_bytes: int = 2, cta_k: int = 1,
                   workspace_tensor: str = SPLIT_K_WORKSPACE) -> List[Kernel]:
    s, scheme = normalize_split_k(split_k, reduction_scheme)
    if s == 1:
        return [kernel]
    if kernel.kind != "gemm" or kernel.gemm is None:
        raise ValueError("split-K expansion requires a GEMM kernel")
    if len(kernel.writes) != 1:
        raise ValueError("split-K expansion requires one GEMM output")
    m, k, n = kernel.gemm
    counts = split_k_counts(batch, m, n, k, s, scheme, output_bytes=output_bytes)
    out = kernel.writes[0]
    expected = batch * m * n * output_bytes
    if out.nbytes != expected:
        raise ValueError(f"GEMM output extent {out.nbytes} does not match {expected} bytes")
    lengths = split_k_lengths(k, s, cta_k, scheme)
    operands, sources = kernel.reads[:2], kernel.reads[2:]
    if scheme == "inplace":
        gemm = SplitKKernel(**{**kernel.__dict__,
            "reads": kernel.reads + (out,) * (s - 1),
            "writes": (out,) * s, "flops": kernel.flops + counts.reduction_flops},
            split_k=s, reduction_scheme=scheme, slice_lengths=lengths,
            original_reads=kernel.reads, original_writes=kernel.writes,
            epilogue_reads=sources)
        return [gemm]
    workspace = Access(workspace_tensor, float(counts.workspace_bytes))
    gemm = SplitKKernel(**{**kernel.__dict__, "reads": operands, "writes": (workspace,)}, split_k=s,
                        reduction_scheme=scheme, slice_lengths=lengths,
                        original_reads=kernel.reads, original_writes=kernel.writes,
                        epilogue_reads=sources)
    reduction = Kernel(kernel.op, kernel.name + ".split_k_reduce", "pointwise",
                       (workspace,) + sources, kernel.writes, counts.reduction_flops,
                       activity_name=kernel.activity_name)
    return [gemm, reduction]

def gemm_trace_role(kernel: Kernel) -> str:
    name = kernel.name.lower()
    return ("dgrad" if name.endswith((".dx", ".gemm_dx")) else
            "wgrad" if name.endswith((".dw", ".gemm_dw")) else "forward")

def trace_op_name(kernel: Kernel) -> str:
    if kernel.op != "MLP":
        return kernel.op
    name = kernel.name.removesuffix(".split_k_reduce")
    return {"gate_proj": "ffn1", "up_proj": "ffn1", "gate_up_proj": "ffn1",
            "silu": "gelu", "mul": "gelu", "silu_and_mul": "gelu",
            "down_proj": "ffn2"}.get(name, "ffn1")

def expand_forced_split_k(kernels: Sequence[Kernel], resolver):
    expanded = []
    counts = {"inplace": 0, "compute_type": 0, "output_type": 0}
    for kernel in kernels:
        if kernel.kind != "gemm" or kernel.gemm is None:
            expanded.append(kernel)
            continue
        role = gemm_trace_role(kernel)
        template, rows = resolver((*kernel.gemm, 1), role)
        if template is None or template.split_k == 1:
            expanded.append(kernel)
            continue
        _, scheme = normalize_split_k(template.split_k, template.reduction_scheme)
        counts[scheme] += 1
        workspace = (SPLIT_K_LT_WORKSPACE if rows and rows[0].get("which_path") == "lt_1mib"
                     else SPLIT_K_WORKSPACE)
        expanded.extend(expand_split_k(kernel, template.split_k, scheme, cta_k=template.cta_k,
                                       workspace_tensor=workspace))
    return expanded, counts

def expand_selected_split_k(kernels: Sequence[Kernel], selector):
    expanded = []
    counts = {"inplace": 0, "compute_type": 0, "output_type": 0}
    for kernel in kernels:
        template = selector(kernel) if kernel.kind == "gemm" and kernel.gemm else None
        if template is None or template.split_k == 1:
            expanded.append(kernel)
            continue
        _, scheme = normalize_split_k(template.split_k, template.reduction_scheme)
        counts[scheme] += 1
        expanded.extend(expand_split_k(kernel, template.split_k, scheme, cta_k=template.cta_k))
    return expanded, counts

class CachingAllocator:
    K_MIN_BLOCK, K_SMALL_SIZE, K_SMALL_BUFFER = 512, 1 << 20, 2 << 20
    K_LARGE_BUFFER, K_MIN_LARGE_ALLOC, K_ROUND_LARGE = 20 << 20, 10 << 20, 2 << 20

    def __init__(self, base: int = 1 << 40):
        self.blocks: Dict[int, list] = {}
        self.by_end: Dict[int, int] = {}
        self.free_lists = {"small": [], "large": []}
        self.next_va = base
        self.n_segments = 0

    @classmethod
    def round_size(cls, n: int) -> int:
        return cls.K_MIN_BLOCK if n < cls.K_MIN_BLOCK else cls.K_MIN_BLOCK * -(-n // cls.K_MIN_BLOCK)

    @classmethod
    def allocation_size(cls, size: int) -> int:
        if size <= cls.K_SMALL_SIZE:
            return cls.K_SMALL_BUFFER
        if size < cls.K_MIN_LARGE_ALLOC:
            return cls.K_LARGE_BUFFER
        return cls.K_ROUND_LARGE * -(-size // cls.K_ROUND_LARGE)

    def _set(self, addr, size, seg, free, pool):
        self.blocks[addr] = [size, seg, free, pool]
        self.by_end[addr + size] = addr

    def _drop(self, addr):
        size = self.blocks.pop(addr)[0]
        del self.by_end[addr + size]

    def malloc(self, nbytes: int) -> int:
        size = self.round_size(max(1, int(nbytes)))
        pool = "small" if size <= self.K_SMALL_SIZE else "large"
        fl = self.free_lists[pool]
        i = bisect.bisect_left(fl, (size, -1))
        if i < len(fl):
            bsize, addr = fl.pop(i)
            seg = self.blocks[addr][1]
        else:
            bsize = self.allocation_size(size)
            addr = self.next_va
            self.next_va += bsize + self.K_ROUND_LARGE
            seg = self.n_segments
            self.n_segments += 1
            self._set(addr, bsize, seg, True, pool)
        rem = bsize - size
        split = rem >= self.K_MIN_BLOCK if pool == "small" else rem > self.K_SMALL_SIZE
        self._drop(addr)
        if split:
            self._set(addr, size, seg, False, pool)
            self._set(addr + size, rem, seg, True, pool)
            bisect.insort(fl, (rem, addr + size))
        else:
            self._set(addr, bsize, seg, False, pool)
        return addr

    def free(self, addr: int) -> None:
        size, seg, _, pool = self.blocks[addr]
        fl = self.free_lists[pool]
        nxt = addr + size
        if nxt in self.blocks and self.blocks[nxt][2] and self.blocks[nxt][1] == seg:
            nsize = self.blocks[nxt][0]
            fl.remove((nsize, nxt))
            self._drop(nxt)
            size += nsize
        prv = self.by_end.get(addr)
        self._drop(addr)
        if prv is not None and self.blocks[prv][2] and self.blocks[prv][1] == seg:
            psize = self.blocks[prv][0]
            fl.remove((psize, prv))
            self._drop(prv)
            addr, size = prv, size + psize
        self._set(addr, size, seg, True, pool)
        bisect.insort(fl, (size, addr))

class L2Cache:
    def __init__(self, capacity: int):
        self.cap = int(capacity)
        self.ext: Dict[int, list] = {}
        self._s: List[int] = []
        self._ids: List[int] = []
        self.heap: List[Tuple[int, int]] = []
        self.total = 0
        self.clock = 0
        self._eid = 0
        self.written = 0.0
        self.written_by: Dict[str, float] = {}
        self.evict_log: Optional[dict] = None

    @property
    def starts(self) -> List[Tuple[int, int]]:
        return list(zip(self._s, self._ids))

    def _add(self, s, e, stamp, dirty, tag):
        if e <= s:
            return
        self._eid += 1
        self.ext[self._eid] = [s, e, stamp, dirty, tag]
        i = bisect.bisect_right(self._s, s)
        self._s.insert(i, s)
        self._ids.insert(i, self._eid)
        heapq.heappush(self.heap, (stamp, self._eid))
        self.total += e - s

    def _index(self, s, eid):
        i = bisect.bisect_left(self._s, s)
        while self._ids[i] != eid:
            i += 1
        return i

    def _remove(self, eid):
        s, e = self.ext.pop(eid)[:2]
        i = self._index(s, eid)
        del self._s[i]
        del self._ids[i]
        self.total -= e - s

    def _overlapping(self, a, b):
        i = max(0, bisect.bisect_right(self._s, a) - 1)
        out = []
        while i < len(self._s) and self._s[i] < b:
            eid = self._ids[i]
            if self.ext[eid][1] > a:
                out.append(eid)
            i += 1
        return out

    def _overlap_range(self, a, b):
        starts = self._s
        i = bisect.bisect_right(starts, a) - 1
        if i < 0:
            i = 0
        elif self.ext[self._ids[i]][1] <= a:
            i += 1
        return i, bisect.bisect_left(starts, b, i)

    def _evict(self):
        total, cap = self.total, self.cap
        if total <= cap:
            return
        ext, heap, starts, ids = self.ext, self.heap, self._s, self._ids
        heappop, heappush = heapq.heappop, heapq.heappush
        while total > cap:
            stamp, eid = heappop(heap)
            x = ext.pop(eid, None)
            if x is None:
                continue
            s, e, _, dirty, tag = x
            size = e - s
            excess = total - cap
            gone = size if size < excess else excess
            i = bisect.bisect_left(starts, s)
            total -= size
            if gone < size:
                self._eid += 1
                rest = self._eid
                ext[rest] = [s + gone, e, stamp, dirty, tag]
                starts[i] = s + gone
                ids[i] = rest
                heappush(heap, (stamp, rest))
                total += size - gone
            else:
                del starts[i]
                del ids[i]
            if dirty:
                self.written += gone
                previous = self.written_by.get(tag)
                if self.evict_log is not None and tag not in self.evict_log:
                    self.evict_log[tag] = previous
                self.written_by[tag] = (0.0 if previous is None else previous) + gone
        self.total = total

    def touch(self, a: int, b: int, write: bool, tag: str, frac: float = 1.0) -> Tuple[float, float]:
        if b <= a:
            return 0.0, 0.0
        self.clock += 1
        now = self.clock
        ext, heap, ids = self.ext, self.heap, self._ids
        heappush = heapq.heappush
        i, j = self._overlap_range(a, b)
        eid = self._eid
        total = self.total
        hit = 0
        new_s, new_ids = [], []
        right = None
        parts = []
        for index in range(i, j):
            s, e, st, d, tg = ext.pop(ids[index])
            total -= e - s
            if s < a:
                eid += 1
                ext[eid] = [s, a, st, d, tg]
                heappush(heap, (st, eid))
                total += a - s
                new_s.append(s)
                new_ids.append(eid)
                lo = a
            else:
                lo = s
            if b < e:
                eid += 1
                ext[eid] = [b, e, st, d, tg]
                heappush(heap, (st, eid))
                total += e - b
                right = eid
                hi = b
            else:
                hi = e
            hit += hi - lo
            parts.append((lo, hi, d, tg))
        if write:
            eid += 1
            ext[eid] = [a, b, now, True, tag]
            heappush(heap, (now, eid))
            new_s.append(a)
            new_ids.append(eid)
        else:
            cur = a
            for lo, hi, d, tg in parts:
                if cur < lo:
                    eid += 1
                    ext[eid] = [cur, lo, now, False, tag]
                    heappush(heap, (now, eid))
                    new_s.append(cur)
                    new_ids.append(eid)
                eid += 1
                ext[eid] = [lo, hi, now, d, tg]
                heappush(heap, (now, eid))
                new_s.append(lo)
                new_ids.append(eid)
                cur = hi
            if cur < b:
                eid += 1
                ext[eid] = [cur, b, now, False, tag]
                heappush(heap, (now, eid))
                new_s.append(cur)
                new_ids.append(eid)
        total += b - a
        if right is not None:
            new_s.append(b)
            new_ids.append(right)
        self._s[i:j] = new_s
        ids[i:j] = new_ids
        self._eid = eid
        self.total = total
        if total > self.cap:
            self._evict()
        return hit * frac, (b - a - hit) * frac

SHARED, PRIVATE = "shared", "private"

class PartitionedL2Cache(L2Cache):
    def __init__(self, shared_capacity: int, physical_capacity: int, private_copies):
        cs, ph = int(shared_capacity), int(physical_capacity)
        pc = Fraction(str(private_copies)) if not isinstance(private_copies, Fraction) else private_copies
        if not (1 <= pc and pc * cs <= ph):
            raise ValueError(f"private copies {pc} must lie between 1 and the shared copies {Fraction(ph, cs)}")
        w_s, w_p, cap = ph * pc.denominator, pc.numerator * cs, ph * cs * pc.denominator
        g = math.gcd(math.gcd(w_s, w_p), cap)
        self.w = {SHARED: w_s // g, PRIVATE: w_p // g}
        self.w_ghost = self.w[SHARED] - self.w[PRIVATE]
        self.private_copies = pc
        super().__init__(cap // g)
        self.ghost: Dict[int, list] = {}
        self.gstarts: List[Tuple[int, int]] = []

    def _add(self, s, e, stamp, dirty, tag, cls=SHARED):
        if e <= s:
            return
        self._eid += 1
        self.ext[self._eid] = [s, e, stamp, dirty, tag, cls]
        i = bisect.bisect_right(self._s, s)
        self._s.insert(i, s)
        self._ids.insert(i, self._eid)
        heapq.heappush(self.heap, (stamp, self._eid))
        self.total += (e - s) * self.w[cls]

    def _remove(self, eid):
        s, e, _, _, _, cls = self.ext.pop(eid)
        i = self._index(s, eid)
        del self._s[i]
        del self._ids[i]
        self.total -= (e - s) * self.w[cls]

    def _add_ghost(self, s, e, stamp):
        if e <= s or self.w_ghost == 0:
            return
        self._eid += 1
        self.ghost[self._eid] = [s, e, stamp]
        bisect.insort(self.gstarts, (s, self._eid))
        heapq.heappush(self.heap, (stamp, self._eid))
        self.total += (e - s) * self.w_ghost

    def _remove_ghost(self, gid):
        s, e = self.ghost.pop(gid)[:2]
        del self.gstarts[bisect.bisect_left(self.gstarts, (s, gid))]
        self.total -= (e - s) * self.w_ghost

    def _drop_ghosts(self, a, b):
        if not self.gstarts:
            return
        i = max(0, bisect.bisect_right(self.gstarts, (a, math.inf)) - 1)
        hits = []
        while i < len(self.gstarts) and self.gstarts[i][0] < b:
            gid = self.gstarts[i][1]
            if self.ghost[gid][1] > a:
                hits.append(gid)
            i += 1
        for gid in hits:
            s, e, st = self.ghost[gid]
            self._remove_ghost(gid)
            self._add_ghost(s, a, st)
            self._add_ghost(b, e, st)

    def _evict(self):
        total, cap = self.total, self.cap
        if total <= cap:
            return
        ext, ghost, heap, starts, ids, w = self.ext, self.ghost, self.heap, self._s, self._ids, self.w
        heappop, heappush = heapq.heappop, heapq.heappush
        while total > cap:
            stamp, eid = heappop(heap)
            excess = total - cap
            x = ext.pop(eid, None)
            if x is not None:
                s, e, _, dirty, tag, cls = x
                weight = w[cls]
                size = e - s
                gone = min(size, -(-excess // weight))
                i = bisect.bisect_left(starts, s)
                total -= size * weight
                if gone < size:
                    self._eid += 1
                    rest = self._eid
                    ext[rest] = [s + gone, e, stamp, dirty, tag, cls]
                    starts[i] = s + gone
                    ids[i] = rest
                    heappush(heap, (stamp, rest))
                    total += (size - gone) * weight
                else:
                    del starts[i]
                    del ids[i]
                if dirty:
                    self.written += gone
                    previous = self.written_by.get(tag)
                    if self.evict_log is not None and tag not in self.evict_log:
                        self.evict_log[tag] = previous
                    self.written_by[tag] = (0.0 if previous is None else previous) + gone
                continue
            g = ghost.get(eid)
            if g is not None:
                s, e, _ = g
                gone = min(e - s, -(-excess // self.w_ghost))
                self.total = total
                self._remove_ghost(eid)
                if gone < e - s:
                    self._add_ghost(s + gone, e, stamp)
                total = self.total
        self.total = total

    def touch(self, a: int, b: int, write: bool, tag: str, frac: float = 1.0, cls: str = SHARED) -> Tuple[float, float]:
        if b <= a:
            return 0.0, 0.0
        self.clock += 1
        now = self.clock
        ext, heap, ids, w = self.ext, self.heap, self._ids, self.w
        heappush = heapq.heappush
        i, j = self._overlap_range(a, b)
        eid = self._eid
        total = self.total
        hit = 0
        new_s, new_ids = [], []
        right = None
        parts = []
        for index in range(i, j):
            s, e, st, d, tg, c = ext.pop(ids[index])
            weight = w[c]
            total -= (e - s) * weight
            if s < a:
                eid += 1
                ext[eid] = [s, a, st, d, tg, c]
                heappush(heap, (st, eid))
                total += (a - s) * weight
                new_s.append(s)
                new_ids.append(eid)
                lo = a
            else:
                lo = s
            if b < e:
                eid += 1
                ext[eid] = [b, e, st, d, tg, c]
                heappush(heap, (st, eid))
                total += (e - b) * weight
                right = eid
                hi = b
            else:
                hi = e
            hit += hi - lo
            parts.append((lo, hi, d, tg, c, st))
        self._eid = eid
        self.total = total
        weight = w[cls]
        if write:
            if self.gstarts:
                self._drop_ghosts(a, b)
                eid, total = self._eid, self.total
            eid += 1
            ext[eid] = [a, b, now, True, tag, cls]
            heappush(heap, (now, eid))
            new_s.append(a)
            new_ids.append(eid)
        else:
            if cls == SHARED and self.gstarts:
                self._drop_ghosts(a, b)
                eid, total = self._eid, self.total
            ghosts = cls == PRIVATE and self.w_ghost != 0
            cur = a
            for lo, hi, d, tg, c, st in parts:
                if cur < lo:
                    eid += 1
                    ext[eid] = [cur, lo, now, False, tag, cls]
                    heappush(heap, (now, eid))
                    new_s.append(cur)
                    new_ids.append(eid)
                if ghosts and c == SHARED:
                    self._eid, self.total = eid, total
                    self._add_ghost(lo, hi, st)
                    eid, total = self._eid, self.total
                eid += 1
                ext[eid] = [lo, hi, now, d, tg, cls]
                heappush(heap, (now, eid))
                new_s.append(lo)
                new_ids.append(eid)
                cur = hi
            if cur < b:
                eid += 1
                ext[eid] = [cur, b, now, False, tag, cls]
                heappush(heap, (now, eid))
                new_s.append(cur)
                new_ids.append(eid)
        total += (b - a) * weight
        if right is not None:
            new_s.append(b)
            new_ids.append(right)
        self._s[i:j] = new_s
        ids[i:j] = new_ids
        self._eid = eid
        self.total = total
        if total > self.cap:
            self._evict()
        return hit * frac, (b - a - hit) * frac

_DIRECTIONS = ("forward.", "backward.", "recompute.", "optimizer.")

def _role(k: Kernel) -> str:
    return k.name.split(".", 1)[1] if k.name.startswith(_DIRECTIONS) else k.name

def _broadcast_read(role: str, a) -> bool:
    if role.endswith(("rmsnorm.mul_weight", "rmsnorm.mul_grad_w")):
        return bool(a.weight)
    if role.endswith(("qk_norm.mul_weight", "qk_norm.mul_grad_normalized")):
        return bool(a.weight)
    if role.startswith("bias_copy") or role == "bias_expand_copy":
        return True
    if role.endswith(("rmsnorm.mul", "rmsnorm.mul_rs", "rmsnorm.mul_grad_rs")):
        return a.tensor.endswith(".rs")
    if role.endswith("rmsnorm.mean_div"):
        return True
    if role in ("rms_norm", "fused_add_rms_norm", "q_norm", "k_norm", "q_a_layernorm", "kv_a_layernorm"):
        return bool(a.weight)
    return role.startswith("rope.") and a.tensor in ("cos", "sin")

GEMM_MIN_TILE_M = 64

def sharing_class(k: Kernel, a, write: bool, phase: str, gemm_min_tile_m: int = GEMM_MIN_TILE_M) -> str:
    role = _role(k)
    kind = k.kind
    if kind == "grouped_gemm":
        if write:
            return PRIVATE
        grouped = k.grouped
        if a.weight:
            expert_size = 2 * grouped.input_width * grouped.output_width
            local_index = int(a.offset) // expert_size
            expert = grouped.local_experts[local_index]
            return SHARED if grouped.padded_counts[expert] > grouped.tile.block_m else PRIVATE
        metadata = ('.sorted_ids', '.block_experts', '.local_block_experts', '.padded_count', '.weights')
        if not grouped.local_blocks or a.tensor.endswith(metadata):
            return PRIVATE
        if k.name == 'moe.w2' and not a.tensor.endswith('.cache2'):
            return PRIVATE
        if grouped.output_width > grouped.tile.block_n:
            return SHARED
        if k.name == 'moe.w13':
            token = int(a.offset) // (2 * grouped.input_width) - grouped.chunk_start
            slot = grouped.routing_offset + token * grouped.top_k
            local = set(grouped.local_experts)
            return SHARED if sum((slot + j) % len(grouped.expert_counts) in local
                                 for j in range(grouped.top_k)) > 1 else PRIVATE
        return PRIVATE
    if kind in ("gemm", "batched_gemm"):
        if write:
            return PRIVATE
        if k.gemm is None:
            return SHARED
        if a.tensor == k.reads[0].tensor:
            return SHARED

        return PRIVATE if int(k.gemm[0]) <= gemm_min_tile_m else SHARED
    if kind == "attention" and role.endswith("mla_decode_stage1"):
        if write:
            return PRIVATE
        if a.tensor == k.reads[0].tensor:
            return SHARED
        return SHARED if getattr(k, "head_groups", 1) > 1 else PRIVATE
    if kind == "attention":
        if write:
            return SHARED if role.endswith("dq_dk_dv") and a.tensor.endswith(".dq_accum") else PRIVATE
        if role.endswith(("combine", "dot_do_o", "convert_dq")):
            return PRIVATE
        if role.endswith(("dq_dk_dv", "flash_bwd")):
            sizes = {r.tensor.rsplit(".", 1)[-1]: r.nbytes for r in k.reads}
            gqa = sizes.get("kb", 0.0) < sizes.get("qb", 0.0)
            if a.tensor.endswith((".kb", ".v")):
                return SHARED if gqa else PRIVATE
            return PRIVATE if a.tensor.endswith(".o") else SHARED
        if a.tensor == k.reads[0].tensor:
            return SHARED if role.endswith("splitkv") else PRIVATE
        return PRIVATE if phase == "decode" else SHARED
    if kind == "pointwise" and not write and _broadcast_read(role, a):
        return SHARED
    return PRIVATE

def _cfg_dims(cfg: dict) -> Tuple[int, int, int, int, int, int, int]:
    h, H = cfg["hidden_size"], cfg["num_attention_heads"]
    return (h, H, cfg["num_key_value_heads"], cfg.get("head_dim") or h // H, cfg["intermediate_size"],
            cfg["vocab_size"], cfg["num_hidden_layers"])

def flash_splits(batch: int, heads: int, seqlen_q: int, seqlen_k: int, head_dim: int, sm_count: int) -> int:
    if not isinstance(sm_count, int) or isinstance(sm_count, bool) or sm_count <= 0:
        raise ValueError("sm_count must be a positive integer from the hardware description")
    block_n = 256 if head_dim <= 64 else (128 if head_dim <= 128 else 64)
    n_blocks = -(-seqlen_k // block_n)
    work = batch * heads * (-(-seqlen_q // 64))
    sms = 2 * sm_count
    if work >= 0.8 * sms:
        return 1
    max_splits = min(128, sms, n_blocks)
    cdiv = lambda x, y: -(-x // y)  # noqa: E731
    elig = lambda s: s == 1 or cdiv(n_blocks, s) != cdiv(n_blocks, s - 1)  # noqa: E731
    eff = [(work * s / sms) / math.ceil(work * s / sms) if elig(s) else 0.0 for s in range(1, max_splits + 1)]
    best = max(eff) if eff else 1.0
    for s in range(1, max_splits + 1):
        if elig(s) and eff[s - 1] >= 0.85 * best:
            return s
    return 1

def inference_dims(cfg: dict, phase: str, batch: int = 1, seq: int = 1024) -> LayerDims:
    family = cfg.get("model_type", "llama")
    if family not in ("llama", "qwen2", "qwen3"):
        raise ValueError(f"inference trace does not describe model_type {family!r}")
    if family in ("qwen2", "qwen3") and cfg.get("use_sliding_window", False):
        raise ValueError("Qwen inference trace does not describe sliding-window attention")
    if family == "qwen3" and cfg.get("attention_bias", False):
        raise ValueError("Qwen3 inference trace does not describe projection bias")
    h, H, Hk, hd, I, V, _ = _cfg_dims(cfg)
    T = batch * seq if phase == "prefill" else batch
    return LayerDims(tokens=T, batch=batch, hidden=h, heads=H, kv_heads=Hk, head_dim=hd, intermediate=I, vocab=V,
                     act_bytes=BF16, weight_bytes=BF16, kv_bytes=BF16, past=0 if phase == "prefill" else seq,
                     head_tokens=batch, model_type=cfg.get("model_type", "llama"),
                     tied_embeddings=bool(cfg.get("tie_word_embeddings", False)),
                     qwen2_bias_mode=cfg.get("qwen2_bias_mode"))

def inference_launch_trace(cfg: dict, phase: str, batch: int = 1, seq: int = 1024, *, sm_count: int) -> List[Kernel]:
    d = inference_dims(cfg, phase, batch, seq)
    h, H, Hk, hd, I, V, L = _cfg_dims(cfg)
    S, T = (seq, d.tokens) if phase == "prefill" else (1, d.tokens)
    ks = step_trace(d, "eager", L, SILU_FWD_FLOPS)
    A = Access
    emb = ks[0]
    out = [replace(emb, reads=(A("ids", I64 * T),) + emb.reads)]
    pro = []
    mk = lambda n, kind, r, w: Kernel("rope" if n.startswith("rotary") else "embedding", n, kind, tuple(r), tuple(w))  # noqa: E731
    if phase == "prefill":
        pro += [mk("prologue.arange", "pointwise", [], [A("cache_pos0", I64 * S)]),
                mk("prologue.add_past", "pointwise", [A("cache_pos0", I64 * S)], [A("cache_pos", I64 * S)])]
        pos = "cache_pos"
    else:
        pos = "pos"
    pro += [mk("rotary.pos_float", "pointwise", [A(pos, I64 * S)], [A("pos_f", F32 * S)]),
            mk("rotary.gemv", "gemm", [A("inv_freq", F32 * hd / 2), A("pos_f", F32 * S)], [A("freqs", F32 * S * hd / 2)]),
            mk("rotary.cat", "pointwise", [A("freqs", F32 * S * hd)], [A("emb", F32 * S * hd)]),
            mk("rotary.cos", "pointwise", [A("emb", F32 * S * hd)], [A("cos0", F32 * S * hd)]),
            mk("rotary.scale_cos", "pointwise", [A("cos0", F32 * S * hd)], [A("cos32", F32 * S * hd)]),
            mk("rotary.sin", "pointwise", [A("emb", F32 * S * hd)], [A("sin0", F32 * S * hd)]),
            mk("rotary.scale_sin", "pointwise", [A("sin0", F32 * S * hd)], [A("sin32", F32 * S * hd)]),
            mk("rotary.cast_cos", "pointwise", [A("cos32", F32 * S * hd)], [A("cos", BF16 * S * hd)]),
            mk("rotary.cast_sin", "pointwise", [A("sin32", F32 * S * hd)], [A("sin", BF16 * S * hd)])]
    out += pro
    if phase == "prefill":
        ns = flash_splits(batch, H, seq, seq, hd, sm_count)
        assert ns == flash_num_splits(d, sm_count)
        hq, sq = H, seq
    else:
        swapped = H > Hk
        hq, sq = (Hk, H // Hk) if swapped else (H, 1)
        ns = flash_splits(batch, hq, sq, seq + 1, hd, sm_count)
    for k in ks[1:]:
        if k.kind != "attention":
            out.append(k)
            continue
        p = k.writes[0].tensor.rsplit(".", 1)[0]
        lse = A(f"{p}.lse", F32 * batch * hq * sq)
        if ns > 1:
            acc = [A(f"{p}.lse_accum", F32 * ns * batch * hq * sq), A(f"{p}.out_accum", F32 * ns * batch * hq * sq * hd)]
            out.append(replace(k, name="attention.splitkv", writes=tuple(acc)))
            out.append(replace(k, name="attention.combine", reads=tuple(acc), writes=k.writes + (lse,)))
        else:
            out.append(replace(k, name="attention.fwd", writes=k.writes + (lse,)))
    return out

def _mta_segments(sizes: Sequence[int], max_tensors: int, max_blocks: int = 320, chunk: int = 65536):
    out, cur, blocks, tensors = [], [], 0, 0
    for idx, n in enumerate(sizes):
        if n == 0:
            continue
        tensors += 1
        chunks = -(-n // chunk)
        for c in range(chunks):
            blocks += 1
            lo, m = c * chunk, min(chunk, n - c * chunk)
            if cur and cur[-1][0] == idx and cur[-1][1] + cur[-1][2] == lo:
                cur[-1] = (idx, cur[-1][1], cur[-1][2] + m)
            else:
                cur.append((idx, lo, m))
            if (tensors == max_tensors and c == chunks - 1) or blocks == max_blocks:
                out.append(cur)
                cur, blocks = [], 0
                tensors = 0 if c == chunks - 1 else 1
    if cur:
        out.append(cur)
    return out

def train_launch_trace(cfg: dict, batch: int = 1, seq: int = 1024, *, sm_count: int,
                       checkpoint: bool = False, loss_bytes: float = 4.0) -> List[Kernel]:
    d = train_dims(cfg, batch, seq)
    L = cfg["num_hidden_layers"]
    if cfg.get("model_type", "llama") == "llama" and not bool(cfg.get("tie_word_embeddings", True)):
        raise ValueError("the training trace describes tied embeddings only (head reads embed_tokens.weight)")

    ks = train_step_trace(d, L, TrainOpts(flash_launches=True, softmax_reads=3, sm_count=int(sm_count), loss_bytes=loss_bytes),
                          checkpoint=checkpoint)
    if checkpoint:
        ks = _version_recomputed_buffers(ks)
    ps = param_sizes(d, L)
    names = [n for n, _ in ps]
    steps = _mta_segments([1] * len(ps), 110)
    fused = _mta_segments([n for _, n in ps], 36)
    out, i_s, i_f = [], 0, 0
    for k in ks:
        if k.name == "optimizer.adamw.step_count":
            r = tuple(RAccess(f"{names[j]}.step", F32) for j, _, _ in steps[i_s])
            out.append(replace(k, reads=r, writes=r))
            i_s += 1
        elif k.name == "optimizer.adamw.fused":
            rd, wr = [], []
            for j, lo, m in fused[i_f]:
                p = names[j]
                acc = {s: RAccess(f"{p}{s}", F32 * m, s == "", F32 * lo) for s in ("", ".grad", ".exp_avg", ".exp_avg_sq")}
                rd += [acc[""], acc[".grad"], acc[".exp_avg"], acc[".exp_avg_sq"]]
                wr += [acc[""], acc[".exp_avg"], acc[".exp_avg_sq"]]
            out.append(replace(k, reads=tuple(rd), writes=tuple(wr)))
            i_f += 1
        else:
            out.append(k)
    assert i_s == len(steps) and i_f == len(fused), (i_s, len(steps), i_f, len(fused))
    return out

def _version_recomputed_buffers(kernels: Sequence[Kernel]) -> List[Kernel]:
    active = set()
    result = []
    for k in kernels:
        if k.name.startswith("recompute."):
            active.update(a.tensor for a in k.writes)
        def version(a):
            return replace(a, tensor="recomputed." + a.tensor) if a.tensor in active else a
        if k.name.startswith(("recompute.", "backward.")):
            k = replace(k, reads=tuple(version(a) for a in k.reads),
                        writes=tuple(version(a) for a in k.writes))
        result.append(k)
    return result

@dataclass
class StepRules:
    alias: Callable[[str], str]
    persistent: Callable[[str], bool]
    hold: Callable[[str], Optional[str]]
    step_end: Callable[[str], bool]
    carry: Dict[str, str] = field(default_factory=dict)
    alloc_before: Optional[Callable[[str], Optional[str]]] = None
    pool: Optional["PoolHistory"] = None
    phase: str = ""
    collective_buffers: Optional[str] = None

@dataclass
class PoolHistory:
    construction: List[Tuple[str, int, bool]]
    workspace: int = 8519680
    lazy: List[str] = field(default_factory=list)
    lazy_anchor: str = ""

def pool_history(cfg: dict, weight_bytes: float, train: bool) -> PoolHistory:
    d = train_dims(cfg, 1, 1)
    L = cfg["num_hidden_layers"]
    ps = param_sizes(d, L)
    order = []
    for name, n in ps:
        if name not in ("w_lnf", "w_head"):
            order.append((name, int(weight_bytes * n), True))
    order.append(("w_lnf", int(weight_bytes * d.hidden), True))
    order.append(("inv_freq", int(F32 * d.head_dim / 2), True))
    tied = bool(cfg.get("tie_word_embeddings", False))
    order.append(("lm_head.untied" if tied else "w_head", int(weight_bytes * d.hidden * d.vocab), not tied))
    lazy = []
    if train:
        for name, _ in ps:
            lazy += [f"{name}.step", f"{name}.exp_avg", f"{name}.exp_avg_sq"]
    return PoolHistory(order, lazy=lazy, lazy_anchor="optimizer.adamw.step_count" if train else "")

def flash_alloc_order(t: str) -> Optional[str]:
    m = re.match(r"((?:recomputed\.)?L\d+)\.(o|lse)$", t)
    return f"{m.group(1)}.lse_accum" if m else None

_STRIDED = ("rope.neg", "rope.cat", "rope.neg_bwd", "rope.slice2_copy", "rope.slice1_copy")

def access_mode(k: Kernel, a) -> str:
    t = a.tensor
    if t.endswith(".rows") or t.endswith(".at_target") or (k.name == "backward.nll_loss" and t == "glogp"):
        return "spread"
    if k.name.endswith(_STRIDED) or _VLLM_QKV_SLICE.search(t) or _VLLM_MLA_SLICE.search(t):
        return "strided"
    return "range"

_VLLM_QKV_SLICE = re.compile(r"\.qkv\.(q|k|v|qk)$")

_VLLM_MLA_SLICE = re.compile(r"\.mla_(qc|kvc|kpe|qpe|qnope|knope|v)$")

def _spread_parent(t: str) -> str:
    for suf in (".rows", ".at_target"):
        if t.endswith(suf):
            return t[: -len(suf)]
    return t

def inference_rules(cfg: dict, phase: str) -> StepRules:
    tied = bool(cfg.get("tie_word_embeddings", False))
    L = cfg["num_hidden_layers"]

    def alias(t):
        t = _spread_parent(t)
        return "w_embed" if (tied and t == "w_head") else t

    def persistent(t):
        return t.startswith("w_") or re.match(r"L\d+\.(w|b[qkv])", t) is not None or t in ("ids", "inv_freq", "pos")

    def hold(t):
        m = re.match(r"(L\d+)\.(q|k|kr|v|qr|y1|y2)$", t)
        if m:
            p, n = m.groups()
            return {"q": f"{p}.kr", "k": f"{p}.kr", "kr": f"{p}.vcache", "v": f"{p}.vcache", "qr": f"{p}.ao",
                    "y1": f"{p}.ao", "y2": f"{p}.dn"}[n]
        m = re.match(r"(.+)\.tmp\.(var|n32)$", t)
        if m:
            return m.group(1)
        return {"cos": "yf", "sin": "yf", "cos32": "sin", "sin32": "sin"}.get(t)

    if phase == "prefill":
        step_end = lambda t: t == "logits" or re.match(r"L\d+\.[kv]cache$", t) is not None  # noqa: E731
        carry = {}
    else:
        step_end = lambda t: t == "logits"  # noqa: E731
        carry = {f"L{i}.{c}cache": f"L{i}.{c}cache_old" for i in range(L) for c in "kv"}
    return StepRules(alias, persistent, hold, step_end, carry, flash_alloc_order, pool_history(cfg, BF16, False),
                     phase=phase)

def vllm_launch_trace(cfg: dict, phase: str, batch: int = 1, seq: int = 1024, *, sm_count: int,
                      engine=None, tp: int = 1, rank: int = 0, collective_buffers=None,
                      fully_connected=True, tp_allreduce="vllm_size_rule", moe_routing_step=0, **moe_options) -> List[Kernel]:
    import kernel_trace_vllm as kv
    eng = engine or kv.VllmEngine(sm_count=int(sm_count))
    if eng.sm_count != sm_count:
        raise ValueError("engine.sm_count disagrees with hardware sm_count")
    moe_world = (moe_options.get("ep", 1) > 1 or
                 (cfg.get("model_type") == "qwen3_moe" and cfg.get("attention_dp", cfg.get("moe_dp", 1)) > 1))
    if (tp > 1 or moe_world) and collective_buffers not in ("occupy_l2", "leave_l2"):
        raise ValueError("collective_buffers must be occupy_l2 or leave_l2 for tp > 1")
    kernels = kv.vllm_step_trace(cfg, phase, int(batch), int(seq), eng, tp=tp, rank=rank, moe_routing_step=moe_routing_step, **moe_options)
    if tp == 1:
        return kernels
    from program.vllm_allreduce import choose_allreduce
    result = []
    graph_record = 1
    for k in kernels:
        if k.kind != "collective" or k.name == "logits.gather" or k.name.startswith("moe."):
            result.append(k)
            continue
        size = int(k.reads[0].nbytes)
        choice = choose_allreduce(tp_allreduce, tp, fully_connected, size, phase == "decode" and eng.cuda_graph)
        if not choice.algorithm.startswith("custom_"):
            result.append(k)
            continue
        inp = k.reads[0]
        if choice.copy_input:
            ipc = kv.VAccess("collective.ipc", size)
            result.append(replace(k, name=k.name + ".ipc_copy", reads=(inp,), writes=(ipc,)))
            inp = ipc

        blocks = min(36, -(-size // (16 * 512)))
        record_offset = graph_record * 64 if not choice.copy_input else 0
        if not choice.copy_input:
            graph_record += 1
        reads = [inp, kv.VAccess("collective.rank_data", 64, offset=record_offset)]
        writes = []
        for block in range(blocks):
            for offset in (block * 32, 1152 + block * 32):
                reads.append(kv.VAccess("collective.meta", tp * 4, offset=offset))
            reads.append(kv.VAccess("collective.meta", 4, offset=2304 + block * 4))
            writes.append(kv.VAccess("collective.meta", 4, offset=2304 + block * 4))
        if choice.algorithm == "custom_two_shot":
            part = (size // 16 // tp + (size // 16 % tp if rank == tp - 1 else 0)) * 16
            reads.append(kv.VAccess("collective.meta", part, offset=2560))
            writes.append(kv.VAccess("collective.meta", part, offset=2560))
        result.append(replace(k, reads=tuple(reads), writes=tuple(writes) + k.writes))
    return result

def vllm_pool_history(cfg: dict, tp: int = 1, rank: int = 0, ep: int = 1) -> PoolHistory:
    moe = cfg.get("model_type") == "qwen3_moe"
    h, H, Hk, hd, I, V, L = _cfg_dims(dict(cfg, intermediate_size=cfg["moe_intermediate_size"]) if moe else cfg)
    mla = str(cfg.get("attention_type", "")).lower() == "mla"
    if mla and tp > 1:
        Hk = tp
    from kernel_trace_vllm_moe import deepseek_moe_layers, is_deepseek_moe
    ds_moe_layers = frozenset(deepseek_moe_layers(cfg)) if mla and is_deepseek_moe(cfg) else frozenset()

    moe_tp = (1 if ep > 1 else tp * int(cfg.get("attention_dp", cfg.get("moe_dp", 1)))) if moe else tp
    if H % tp or I % moe_tp or (Hk % tp and not (moe and tp % Hk == 0)):
        raise ValueError("tp must divide attention heads, KV heads and intermediate size")
    from kernel_trace_vllm import local_kv_heads
    H, Hk, I = H // tp, local_kv_heads(Hk, tp), I // moe_tp
    qn, kn = H * hd, Hk * hd
    vp = 64 * -(-V // 64)
    if vp % tp:
        raise ValueError("tp must divide padded vocabulary size")
    vp //= tp
    w = int(BF16)

    q = w
    if cfg.get("linear_weight_bytes") is not None:
        if not mla:
            raise NotImplementedError("precision.linear_weights is described for the DeepSeek MLA trace only")
        if not float(cfg["linear_weight_bytes"]).is_integer():
            raise NotImplementedError("the L2 weight layout needs whole bytes per linear-weight element")
        q = int(cfg["linear_weight_bytes"])
    order = [("w_embed", w * vp * h, True)]
    for i in range(L):
        if mla:
            Q, R, N, P, Vd = (int(cfg.get("q_lora_rank") or 0), int(cfg["kv_lora_rank"]), int(cfg["qk_nope_head_dim"]),
                              int(cfg["qk_rope_head_dim"]), int(cfg["v_head_dim"]))
            if Q:
                order += [(f"L{i}.wqkva", q * h * (Q + R + P), True), (f"L{i}.w_qanorm", w * Q, True),
                          (f"L{i}.wqb", q * Q * H * (N + P), True)]
            else:
                order += [(f"L{i}.wkva", q * h * (R + P), True), (f"L{i}.wq", q * h * H * (N + P), True)]
            order += [(f"L{i}.w_kvanorm", w * R, True), (f"L{i}.wkvb", w * R * H * (N + Vd), True)]
            if q != w:
                order.append((f"L{i}.wkvb_q", q * R * H * (N + Vd), True))
            order += [(f"L{i}.wo", q * H * Vd * h, True)]
            if i == 0:
                order.append(("cos_sin_cache", w * P * int(cfg.get("max_position_embeddings") or 131072), True))
            if i in ds_moe_layers:
                e, ds_i = int(cfg["num_experts"]), int(cfg["moe_intermediate_size"])
                ds_tp = 1 if ep > 1 else tp * int(cfg.get("attention_dp", cfg.get("moe_dp", 1)))
                si = int(cfg.get("n_shared_experts") or 0) * ds_i // tp
                if e % ep or ds_i % ds_tp:
                    raise ValueError("ep must divide num_experts and the expert TP world moe_intermediate_size")
                order.append((f"L{i}.moe.router_weight", w * e * h, True))
                if str(cfg.get("topk_method")) == "noaux_tc":
                    order.append((f"L{i}.moe.router_bias", 4 * e, True))
                if si:
                    order += [(f"L{i}.moe.shared_wgu", q * h * 2 * si, True), (f"L{i}.moe.shared_wd", q * si * h, True)]
                if ep > 1:
                    order.append((f"L{i}.moe.expert_map", 4 * e, True))
                order += [(f"L{i}.moe.w13", q * (e // ep) * h * 2 * (ds_i // ds_tp), True),
                          (f"L{i}.moe.w2", q * (e // ep) * (ds_i // ds_tp) * h, True)]
            else:
                order += [(f"L{i}.wgu", q * h * 2 * I, True), (f"L{i}.wd", q * I * h, True)]
            order += [(f"L{i}.w_ln1", w * h, True), (f"L{i}.w_ln2", w * h, True)]
            continue
        order.append((f"L{i}.wqkv", w * h * (qn + 2 * kn), True))
        if cfg.get("model_type") == "qwen2":
            order.append((f"L{i}.bqkv", w * (qn + 2 * kn), True))
        order.append((f"L{i}.wo", w * qn * h, True))
        if i == 0:
            order.append(("cos_sin_cache", w * hd * int(cfg.get("max_position_embeddings") or 131072), True))
        if cfg.get("model_type") in ("qwen3", "qwen3_moe"):
            order += [(f"L{i}.w_qnorm", w * hd, True), (f"L{i}.w_knorm", w * hd, True)]
        if moe:
            e = cfg["num_experts"]
            if e % ep:
                raise ValueError("ep must divide num_experts")
            order += [(f"L{i}.moe.router_weight", w * e * h, True)]
            if ep > 1:
                order.append((f"L{i}.moe.expert_map", 4 * e, True))
            order += [
                      (f"L{i}.moe.w13", w * (e // ep) * h * 2 * I, True),
                      (f"L{i}.moe.w2", w * (e // ep) * I * h, True)]
        else:
            order += [(f"L{i}.wgu", w * h * 2 * I, True), (f"L{i}.wd", w * I * h, True)]
        order += [(f"L{i}.w_ln1", w * h, True),
                  (f"L{i}.w_ln2", w * h, True)]
    order.append(("w_lnf", w * h, True))
    tied = bool(cfg.get("tie_word_embeddings", False))
    order.append(("lm_head.untied" if tied else "w_head", w * vp * h, not tied))
    if tp > 1:
        order += [("collective.ipc", 8 * MIB, True), ("collective.meta", 8 * MIB + 2560, True),
                  ("collective.rank_data", 8 * MIB, True)]
    return PoolHistory(order)

_VLLM_INPUTS = ("ids", "positions", "slot_mapping", "seq_lens", "block_table", "query_start_loc",
                "selected_token_indices", "temperatures", "sample_indices", "penalty_rows", "penalty_cols",
                "cos_sin_cache", "context_lens")

def vllm_rules(cfg: dict, phase: str, tp: int = 1, rank: int = 0, collective_buffers=None, ep: int = 1) -> StepRules:
    from kernel_trace_vllm_moe import deepseek_moe_layers, is_deepseek_moe
    deepseek = str(cfg.get("attention_type", "")).lower() == "mla" and is_deepseek_moe(cfg)
    moe = cfg.get("model_type") == "qwen3_moe" or deepseek
    moe_dp = cfg.get("attention_dp", cfg.get("moe_dp", 1)) if moe else 1
    L = cfg["num_hidden_layers"]

    moe_layer = (frozenset(deepseek_moe_layers(cfg)).__contains__ if deepseek else (lambda i: True))
    if (tp > 1 or ep > 1) and collective_buffers not in ("occupy_l2", "leave_l2"):
        raise ValueError("collective_buffers must be occupy_l2 or leave_l2 for tp > 1")
    last_dn = f"L{L - 1}.dn"

    def alias(t):
        t = _spread_parent(t)
        if moe:
            t = re.sub(r"(L\d+\.moe)\.chunk\d+\.(cache1|cache3)$", r"\1.cache13", t)
            t = re.sub(r"(L\d+\.moe)\.chunk\d+\.cache2$", r"\1.cache2", t)
            m = re.match(r"^L(\d+)\.dn" + (r"(?:\.local)?$" if tp == 1 else r"\.local$"), t)
            if m and moe_layer(int(m.group(1))):
                t = re.sub(r"^(L\d+)\.dn" + (r"(?:\.local)?$" if tp == 1 else r"\.local$"),
                           r"\1.ao" if moe_dp == 1 else r"\1.moe.all_reduce_output", t)
        return _VLLM_MLA_SLICE.sub("", _VLLM_QKV_SLICE.sub(".qkv", t))

    alias = _memoized(alias)

    def persistent(t):
        return (t.startswith("w_") or re.match(r"L\d+\.(w|bqkv$|[kv]cache$)", t) is not None or (moe and re.fullmatch(r"L\d+\.moe\.(router_weight|w13|w2|expert_map|router_bias|shared_wgu|shared_wd)", t) is not None) or t in _VLLM_INPUTS
                or t.startswith("g.") or t.startswith("collective."))

    def hold(t):
        if t == "L0.y1":
            return "L0.ao"
        m = re.match(r"(L\d+)\.(?:qnorm|knorm)$", t)
        if m:
            return f"{m.group(1)}.ao"

        m = re.match(r"(L\d+)\.(?:mla_a|mla_q|kvcn|rope\.kr)$", t)
        if m:
            return f"{m.group(1)}.ao"
        m = re.match(r"L(\d+)\.(dn|qkv|ao)$", t)
        if not m:
            return None
        i, n = int(m.group(1)), m.group(2)
        if n == "dn":
            return f"L{i + 1}.ao" if i + 1 < L else None
        if n == "ao" and moe and moe_dp > 1 and tp == 1 and moe_layer(i):
            return f"L{i}.moe.all_reduce_output"
        return f"L{i}.ao" if n == "qkv" else f"L{i}.dn"

    end = {last_dn, "logits", "logits32", "probs", "logprobs", "tok", "query_start_loc_dec"}
    if moe:
        end.add(alias(last_dn))
    return StepRules(alias, persistent, hold, lambda t: t in end, {}, vllm_alloc_order, vllm_pool_history(cfg, tp, rank, ep),
                     phase=phase, collective_buffers=collective_buffers if tp > 1 or ep > 1 or moe_dp > 1 else None)

def _memoized(function):
    memo = {}

    def call(t):
        try:
            return memo[t]
        except KeyError:
            value = memo[t] = function(t)
            return value
    return call

def vllm_alloc_order(t: str) -> Optional[str]:
    m = re.match(r"(L\d+)\.lse$", t)
    if m:
        return f"{m.group(1)}.lse_accum"
    m = re.match(r"(L\d+\.moe)\.(?:cache13|cache2)$", t)
    if m:
        return f"{m.group(1)}.chunk0.sorted_ids"
    m = re.match(r"(L\d+\.moe)\.(?:weights_raw|expert_ids|token_expert_indices)$", t)
    return f"{m.group(1)}.logits32" if m else None

def train_rules(cfg: dict, checkpoint: bool = False) -> StepRules:
    L = cfg["num_hidden_layers"]
    resid = f"gx{L}.tmp.gx1"

    def alias(t):
        t = _spread_parent(t)
        if re.match(r"(L\d+\.gr1|gx\d+)$", t):
            return resid
        m = re.match(r"(L\d+)\.(gm|go)$", t)
        if m:
            p, n = m.groups()
            return f"{p}.{n}.bf16@{p}.{'wd' if n == 'gm' else 'wo'}"
        return t

    def persistent(t):
        if t in ("ids", "inv_freq"):
            return True
        base = re.sub(r"\.(exp_avg|exp_avg_sq|step)$", "", t)

        is_param = (re.match(r"w_[a-z_0-9]*$", base) is not None
                    or re.match(r"L\d+\.(?:w[a-z_0-9]*|b[qkv])$", base) is not None)
        return is_param and (base == t or t.endswith((".exp_avg", ".exp_avg_sq", ".step")))

    def hold(t):
        if checkpoint and re.fullmatch(r"(?:L\d+\.)?w[a-z_0-9]*\.bf16", t):
            return "loss"

        if re.fullmatch(r"L\d+\.b[qkv]\.bf16", t):
            return "loss"
        m = re.match(r"((?:recomputed\.)?L\d+)\.(q|k|kr|qr|y1|y2)$", t)
        if m:
            p, n = m.groups()
            return {"q": f"{p}.kr", "k": f"{p}.kr", "kr": f"{p}.ao", "qr": f"{p}.ao", "y1": f"{p}.ao",
                    "y2": f"{p}.dn"}[n]
        m = re.match(r"(.+)\.tmp\.var$", t)
        if m and not t.startswith("g"):
            return m.group(1)
        return {"cos": "yf", "sin": "yf", "yf": "loss", "logits": "loss"}.get(t)

    def step_end(t):
        return t.endswith(".grad") or t == "loss"

    return StepRules(alias, persistent, hold, step_end, {}, flash_alloc_order, pool_history(cfg, F32, True),
                     phase="train")

@dataclass
class Buffer:
    name: str
    size: int
    first: int
    free_after: int
    persistent: bool = False
    carried_in: bool = False
    order: Tuple[float, float] = (0.0, 0.0)

def tensor_lifetimes(kernels: Sequence[Kernel], rules: StepRules) -> Dict[str, Buffer]:
    n = len(kernels)
    wsize: Dict[str, float] = {}
    rsize: Dict[str, float] = {}
    first: Dict[str, int] = {}
    last: Dict[str, int] = {}
    first_is_read: Dict[str, bool] = {}
    writers: Dict[str, int] = {}
    pos: Dict[str, Tuple[float, float]] = {}
    carried_ids = set(rules.carry.values())
    spread: Dict[str, float] = {}
    for i, k in enumerate(kernels):
        serial = isinstance(k, SplitKKernel) and k.reduction_scheme == "inplace"
        reads = k.original_reads if serial else k.reads
        writes = k.original_writes if serial else k.writes
        for a, is_w in [(a, False) for a in reads] + [(a, True) for a in writes]:
            if a.nbytes <= 0:
                continue
            t = rules.alias(a.tensor)
            ext = getattr(a, "offset", 0.0) + a.nbytes
            if k.name == "moe.align_sort" and re.search(r"\.moe\.chunk\d+\.scratch$", t):
                ext = min(ext, wsize.get(t, ext))
            if access_mode(k, a) == "spread":
                ext = 0.0
                spread[t] = max(spread.get(t, 0.0), a.nbytes)
            if is_w:
                wsize[t] = max(wsize.get(t, 0.0), ext)
                writers[a.tensor] = i
            else:
                rsize[t] = max(rsize.get(t, 0.0), ext)
            if t not in first:
                first[t] = i
                first_is_read[t] = not is_w
                pos[t] = (float(i), float(len(pos)))
            last[t] = i
    for i, k in enumerate(kernels):
        if isinstance(k, SplitKKernel) and k.reduction_scheme != "inplace":
            for a in k.original_writes:
                t = rules.alias(a.tensor)
                if t in first and first[t] > i:
                    first[t] = i
                    pos[t] = (float(i), pos[t][1])
    if rules.alloc_before is not None:
        for t in list(first):
            anc = rules.alloc_before(t)
            if anc in first and first[anc] < first[t]:
                first[t] = first[anc]

                tail = t.rsplit('.', 1)[-1]
                adjustment = {'cache13': .001, 'cache2': .002, 'weights_raw': .001,
                              'expert_ids': .002, 'token_expert_indices': .003}.get(tail, 1e-3 * len(t))
                pos[t] = (pos[anc][0], pos[anc][1] - 0.5 + adjustment)

    moe_extra = {}
    layers = {m.group(1) for t in first if (m := re.match(r"(L\d+\.moe)\.chunk\d+\.sorted_ids$", t))}

    layer_chunks: Dict[str, List[int]] = {}
    for t in first:
        if (m := re.fullmatch(r"(L\d+\.moe)\.chunk(\d+)\.sorted_ids", t)):
            layer_chunks.setdefault(m.group(1), []).append(int(m.group(2)))
    for p in layers:
        chunks = sorted(layer_chunks.get(p, ()))
        final_sum = last.get(p + '.cache13', 0)
        layer = p.split('.')[0]
        final_layer = max(final_sum, writers.get(p + '.all_reduce_output', 0),
                          writers.get(layer + '.dn', 0))
        normalized = writers.get(p + '.weights', 0)
        for suffix in ('logits32', 'token_expert_indices', 'weights_raw'):
            moe_extra[p + '.' + suffix] = normalized
        for suffix in ('expert_ids', 'weights'):
            moe_extra[p + '.' + suffix] = final_sum
        moe_extra[p + '.cache2'] = final_sum
        for suffix in ('logits', 'gathered_logits', 'gathered_hidden'):
            moe_extra[p + '.' + suffix] = final_layer
        for index, chunk in enumerate(chunks):
            anchor = final_sum
            if index + 1 < len(chunks):
                nxt = p + '.chunk' + str(chunks[index + 1])
                anchor = writers.get(nxt + '.local_block_experts', writers.get(nxt + '.sorted_ids', final_sum))
            cp = p + '.chunk' + str(chunk)
            for suffix in ('sorted_ids', 'padded_count',
                           'local_block_experts' if cp + '.local_block_experts' in first else 'block_experts'):
                moe_extra[cp + '.' + suffix] = anchor
    bufs: Dict[str, Buffer] = {}
    for t in first:
        size = int(wsize.get(t) or rsize.get(t) or 0)
        if size <= 0 and rules.persistent(t) and spread.get(t):
            size = int(spread[t])
        if size <= 0:
            continue
        split_workspace = t in {rules.alias(SPLIT_K_WORKSPACE), rules.alias(SPLIT_K_LT_WORKSPACE)}
        pers = rules.persistent(t) or split_workspace
        car = t in carried_ids
        if first_is_read[t] and not (pers or car):
            raise ValueError(f"buffer {t!r} is read before any write and is neither persistent nor carried")
        if pers:
            bufs[t] = (Buffer(t, size, first[t], None, persistent=True, order=pos[t])
                       if split_workspace else Buffer(t, size, -1, None, persistent=True))
            continue
        if t in rules.carry:
            free = None
        elif rules.step_end(t):
            free = n
        else:
            free = last[t]
            free = max(free, moe_extra.get(t, free))
            anchor = rules.hold(t)
            if anchor is not None and anchor in writers:
                free = max(free, writers[anchor])
        bufs[t] = Buffer(t, size, -1 if car else first[t], free, carried_in=car, order=pos[t])
    return bufs

def plan_addresses(kernels: Sequence[Kernel], bufs: Dict[str, Buffer], rules: StepRules, steps: int):
    alloc = CachingAllocator()
    n = len(kernels)
    ph = rules.pool
    pers: Dict[str, int] = {}
    lazy = set(ph.lazy) if ph else set()
    first_gemm = next((i for i, k in enumerate(kernels) if k.kind == "gemm"), 0)
    lazy_at = next((i for i, k in enumerate(kernels) if ph and k.name == ph.lazy_anchor), None)
    if ph:
        dropped = []
        for t, size, kept in ph.construction:
            a = alloc.malloc(max(1, size))
            if kept and t in bufs:
                pers[t] = a
            elif not kept:
                dropped.append(a)
        for a in dropped:
            alloc.free(a)
    split_workspace = rules.alias(SPLIT_K_WORKSPACE)
    lt_workspace = rules.alias(SPLIT_K_LT_WORKSPACE)
    for t, b in bufs.items():
        if b.persistent and t not in pers and t not in lazy and t not in (split_workspace, lt_workspace):
            pers[t] = alloc.malloc(b.size)
    born: Dict[int, List[str]] = {}
    dies: Dict[int, List[str]] = {}
    for t, b in bufs.items():
        if b.persistent or b.carried_in:
            pass
        else:
            born.setdefault(b.first, []).append(t)
        if not b.persistent and b.free_after is not None:
            dies.setdefault(b.free_after, []).append(t)
    for i in born:
        born[i].sort(key=lambda t: bufs[t].order)
    for i in dies:
        dies[i].sort(key=lambda t: bufs[t].order)
    carried: Dict[str, int] = {}
    maps = []
    for _ in range(steps):
        addr = dict(pers)
        alloc.events = []
        for t, b in bufs.items():
            if b.carried_in:
                addr[t] = carried[t] if t in carried else alloc.malloc(b.size)
        for i in range(n + 1):
            if ph and not maps and i == first_gemm and ph.workspace:
                a = alloc.malloc(ph.workspace)
                if split_workspace in bufs:
                    b = bufs[split_workspace]
                    if b.size > ph.workspace:
                        raise ValueError(f"split-K workspace requires {b.size} bytes; pool reserves {ph.workspace}")
                    pers[split_workspace] = addr[split_workspace] = a
            if ph and not maps and i == lazy_at:
                for t in ph.lazy:
                    if t in bufs:
                        pers[t] = addr[t] = alloc.malloc(bufs[t].size)
            for t in born.get(i, ()):
                addr[t] = alloc.malloc(bufs[t].size)
                alloc.events.append(("alloc", t, bufs[t].size, addr[t]))
            if not maps and split_workspace in bufs and split_workspace not in pers and i == bufs[split_workspace].first:
                b = bufs[split_workspace]
                pers[split_workspace] = addr[split_workspace] = alloc.malloc(b.size)
                alloc.events.append(("alloc", split_workspace, b.size, addr[split_workspace]))
            if not maps and lt_workspace in bufs and i == bufs[lt_workspace].first:
                b = bufs[lt_workspace]
                if ph and b.size > MIB:
                    raise ValueError(f"split-K Lt workspace requires {b.size} bytes; pool reserves {MIB}")
                size = MIB if ph else b.size
                pers[lt_workspace] = addr[lt_workspace] = alloc.malloc(size)
                alloc.events.append(("alloc", lt_workspace, size, addr[lt_workspace]))
            for t in dies.get(i, ()):
                alloc.free(addr[t])
                alloc.events.append(("free", t, bufs[t].size, addr[t]))
        carried = {rules.carry[t]: addr[t] for t in rules.carry if t in addr}
        maps.append(addr)
        pers = {t: a for t, a in addr.items() if bufs[t].persistent}
    return maps, alloc

def plan_addresses_onepass(kernels, bufs, rules):
    alloc = CachingAllocator()
    ph = rules.pool
    addr = {}
    lazy = set(ph.lazy) if ph else set()

    if ph:
        dropped = []
        for tensor, size, kept in ph.construction:
            pointer = alloc.malloc(max(1, size))
            if kept and tensor in bufs:
                addr[tensor] = pointer
            elif not kept:
                dropped.append(pointer)
        for pointer in dropped:
            alloc.free(pointer)

    for tensor, buffer in bufs.items():
        if buffer.persistent and tensor not in addr and tensor not in lazy:
            addr[tensor] = alloc.malloc(buffer.size)

    if ph and ph.workspace:
        alloc.malloc(ph.workspace)
    if ph:
        for tensor in ph.lazy:
            if tensor in bufs and tensor not in addr:
                addr[tensor] = alloc.malloc(bufs[tensor].size)

    born, dies = {}, {}
    for tensor, buffer in bufs.items():
        if not buffer.persistent and not buffer.carried_in:
            born.setdefault(buffer.first, []).append(tensor)
        if not buffer.persistent and buffer.free_after is not None:
            dies.setdefault(buffer.free_after, []).append(tensor)
    for events in (born, dies):
        for tensors in events.values():
            tensors.sort(key=lambda tensor: bufs[tensor].order)

    for tensor, buffer in bufs.items():
        if buffer.carried_in:
            addr[tensor] = alloc.malloc(buffer.size)

    alloc.events = []
    for kernel_index in range(len(kernels) + 1):
        for tensor in born.get(kernel_index, ()):
            addr[tensor] = alloc.malloc(bufs[tensor].size)
            alloc.events.append(('alloc', tensor, bufs[tensor].size, addr[tensor]))
        for tensor in dies.get(kernel_index, ()):
            alloc.free(addr[tensor])
            alloc.events.append(('free', tensor, bufs[tensor].size, addr[tensor]))

    return addr, alloc

@dataclass
class SimResult:
    capacity: int
    hbm_read: List[float]
    hbm_write: List[float]
    written_by: List[Dict[str, float]]
    end_dirty: float
    segments: int

    @property
    def read_total(self) -> float:
        return sum(self.hbm_read)

    @property
    def write_total(self) -> float:
        return sum(self.hbm_write)

class _PassOp(tuple):
    passes: float = 1.0

def _with_passes(op: tuple, passes: float) -> tuple:
    if passes <= 1.0:
        return op
    out = _PassOp(op)
    out.passes = float(passes)
    return out

def gemm_passes_from_traffic(transaction_bytes) -> Tuple[float, float]:
    return tuple(max(1.0, float(transaction_bytes[f"{x}_dram_read"]) / float(transaction_bytes[f"{x}_dram_unique"]))
                 if transaction_bytes.get(f"{x}_dram_unique") else 1.0 for x in "ab")

def _kernel_ops(k: Kernel, addr: Dict[str, int], bufs: Dict[str, Buffer], rules: StepRules,
                classify: Optional[Callable] = None, passes: Sequence[float] = ()):
    ops = []
    accesses = [(a, False) for a in k.reads] + [(a, True) for a in k.writes]
    swept = [float(p) for p in passes[:len(k.reads)]]
    if any(p > 1.0 for p in swept) and k.kind == 'grouped_gemm':
        raise ValueError("operand passes are described for gemm and batched_gemm launches only")
    swept += [1.0] * (len(accesses) - len(swept))
    if classify is not None and k.kind == 'grouped_gemm':
        split = []
        g = k.grouped
        for a, w in accesses:
            unit = (2 * g.input_width * g.output_width if a.weight else
                    2 * g.input_width if k.name == 'moe.w13' and not w and g.local_blocks
                    and a.tensor == k.reads[0].tensor and g.output_width <= g.tile.block_n else 0)
            if not unit:
                split.append((a, w))
                continue
            off, end = int(a.offset), int(a.offset + a.nbytes)
            while off < end:
                hi = min(end, (off // unit + 1) * unit)
                split.append((replace(a, offset=float(off), nbytes=float(hi-off)), w))
                off = hi
        accesses = split
        swept = [1.0] * len(accesses)
    for (a, w), p in zip(accesses, swept):
        nb = int(round(a.nbytes))
        if nb <= 0:
            continue
        x = (classify(k, a, w),) if classify is not None else ()
        t = rules.alias(a.tensor)
        base, size = addr[t], bufs[t].size
        mode = access_mode(k, a)
        if mode == "spread":
            pieces = max(1, min(256, nb // 1024))
            ln = nb // pieces
            for j in range(pieces):
                s = base + (size * j) // pieces
                ops.append(_with_passes((s, s + ln, w, t, 1.0) + x, p))
        elif mode == "strided" and nb < size:
            ops.append(_with_passes((base, base + size, w, t, nb / size) + x, p))
        else:
            off = int(getattr(a, "offset", 0.0))
            reps, rem = divmod(off + nb, size) if off == 0 and nb > size else (0, nb)
            if reps:
                ops.extend([_with_passes((base, base + size, w, t, 1.0) + x, p)] * reps)
            if rem:
                ops.append(_with_passes((base + off, base + off + rem, w, t, 1.0) + x, p))
    if k.kind == 'grouped_gemm':
        merged = []
        for op in ops:
            if merged and merged[-1][1] == op[0] and merged[-1][2:] == op[2:]:
                merged[-1] = (merged[-1][0], op[1], *op[2:])
            else:
                merged.append(op)
        ops = merged
    return ops

def _stream_length(op) -> int:
    n = op[1] - op[0]
    return int(round(n * op.passes)) if isinstance(op, _PassOp) else n

def _stream_rounds(ops, slice_bytes: int):
    total = sum(_stream_length(op) for op in ops)
    R = max(1, -(-total // slice_bytes))
    rounds: List[list] = [[] for _ in range(R)]
    for op in ops:
        a, b = op[0], op[1]
        rest = tuple(op[2:])
        if isinstance(op, _PassOp):
            n, length = b - a, _stream_length(op)
            npc = min(R, max(1, -(-length // 4096)))
            lo = 0
            for j in range(npc):
                hi = length * (j + 1) // npc
                v = lo
                while v < hi:
                    s = v % n
                    e = min(n, s + hi - v)
                    rounds[j * R // npc].append((a + s, a + e) + rest)
                    v += e - s
                lo = hi
            continue
        n = b - a
        npc = min(R, max(1, -(-n // 4096)))
        lo = a
        for j in range(npc):
            hi = a + n * (j + 1) // npc
            rounds[j * R // npc].append((lo, hi) + rest)
            lo = hi
    for r in rounds:
        r.sort(key=lambda op: op[2])
    return rounds

def _run_kernel_reference(cache: L2Cache, ops, slice_bytes: int, collective: bool = False) -> Tuple[float, float, Dict[str, float]]:
    rounds = _stream_rounds(ops, slice_bytes)
    w0 = cache.written
    read = 0.0

    evicted = cache.evict_log = {}
    try:
        for r in rounds:
            for op in r:
                if collective and op[2]:
                    op = op[:3] + ("@collective:" + op[3],) + op[4:]
                _, miss = cache.touch(*op)
                if not op[2]:
                    read += miss
    finally:
        cache.evict_log = None

    by = {}
    if evicted:
        written_by = cache.written_by
        for t in (written_by if len(evicted) > 1 else evicted):
            if t in evicted:
                previous = evicted[t]
                delta = written_by[t] - (0.0 if previous is None else previous)
                if delta > 0:
                    by[t] = delta
    excluded = sum(v for t, v in by.items() if t.startswith("@collective:"))
    by = {t: v for t, v in by.items() if not t.startswith("@collective:")}
    return (0.0 if collective else read), cache.written - w0 - excluded, by

_EVENT_LIMIT = 4096
_REFERENCE = False

def _pass_touches(L: int, n: int, npc: int, j0: int, j1: int) -> int:
    if j1 <= j0:
        return 0
    v0, v1 = L * j0 // npc, L * j1 // npc
    count = (j1 - j0) + v1 // n - v0 // n
    for q in range(v0 // n + 1, v1 // n + 1):
        m = q * n
        i = -(-m * npc // L)
        if j0 < i <= j1 and L * i // npc == m:
            count -= 1
    return count

class _KernelStream:
    def __init__(self, cache, ops, slice_bytes, collective):
        unique, mult, lengths = [], [], []
        previous = None
        for op in ops:
            if op is previous or (previous is not None and op == previous and type(op) is type(previous)
                    and getattr(op, "passes", None) == getattr(previous, "passes", None)
                    and all(type(x) is type(y) for x, y in zip(op, previous))):
                mult[-1] += 1
                continue
            unique.append(op)
            mult.append(1)
            lengths.append(_stream_length(op))
            previous = op
        R = max(1, -(-sum(m * L for m, L in zip(mult, lengths)) // slice_bytes))
        geo = []
        for op, L in zip(unique, lengths):
            rest = tuple(op[2:])
            if collective and rest[0]:
                rest = (rest[0], "@collective:" + rest[1]) + rest[2:]
            geo.append((op[0], op[1] - op[0], L, min(R, max(1, -(-L // 4096))), isinstance(op, _PassOp), rest))
        self.cache, self.R, self.geo, self.mult, self.collective = cache, R, geo, mult, collective

    def rounds(self, k0, k1):
        return [[op for op, m in rnd for _ in range(m)] for rnd in self.rounds_rle(k0, k1)]

    def rounds_rle(self, k0, k1):
        R = self.R
        reads = [[] for _ in range(k1 - k0)]
        writes = [[] for _ in range(k1 - k0)]
        for (a, n, L, npc, is_pass, rest), m in zip(self.geo, self.mult):
            j0 = (k0 * npc + R - 1) // R
            j1 = (k1 * npc + R - 1) // R
            if j0 >= j1:
                continue
            out = writes if rest[0] else reads
            if is_pass:
                lo = L * j0 // npc
                for j in range(j0, j1):
                    hi = L * (j + 1) // npc
                    lst = out[j * R // npc - k0]
                    for _ in range(m):
                        v = lo
                        while v < hi:
                            s = v % n
                            e = min(n, s + hi - v)
                            lst.append(((a + s, a + e) + rest, 1))
                            v += e - s
                    lo = hi
            else:
                lo = a + n * j0 // npc
                for j in range(j0, j1):
                    hi = a + n * (j + 1) // npc
                    out[j * R // npc - k0].append(((lo, hi) + rest, m))
                    lo = hi
        for r, w in zip(reads, writes):
            r += w
        return reads

    def entries(self, k0, k1):
        R = self.R
        reads = [[] for _ in range(k1 - k0)]
        writes = [[] for _ in range(k1 - k0)]
        for o, (a, n, L, npc, is_pass, rest) in enumerate(self.geo):
            j0 = (k0 * npc + R - 1) // R
            j1 = (k1 * npc + R - 1) // R
            out = writes if rest[0] else reads
            lo = L * j0 // npc
            for j in range(j0, j1):
                hi = L * (j + 1) // npc
                lst = out[j * R // npc - k0]
                if is_pass:
                    v = lo
                    while v < hi:
                        s = v % n
                        e = min(n, s + hi - v)
                        lst.append((o, a + s, a + e))
                        v += e - s
                else:
                    lst.append((o, a + lo, a + hi))
                lo = hi
        for r, w in zip(reads, writes):
            r += w
        return reads

    def run(self):
        R = self.R
        read = 0.0
        touch = self.cache.touch
        plan = self._plan()
        r = next_try = 0
        while r < R:
            if plan and r >= next_try:
                kind, value = self._try_skip(r)
                if kind == "skip":
                    h, events = value
                    out = self._skip(r, h, read, events)
                    if out is not None:
                        read, r = out, h
                        continue
                    next_try = r + 1
                elif kind == "never":
                    plan = False
                else:
                    next_try = value
            k1 = R if not plan else min(R, max(r + 1, next_try))
            while r < k1:
                k = min(k1, r + 256)
                for rnd in self.rounds_rle(r, k):
                    previous, repeats = None, 0
                    for op, m in rnd:
                        if op == previous and all(type(x) is type(y) for x, y in zip(op, previous)):
                            repeats += m
                            continue
                        if repeats:
                            read = self._repeat(previous, repeats, read)
                        _, miss = touch(*op)
                        if not op[2]:
                            read += miss
                        previous, repeats = op, m - 1
                    if repeats:
                        read = self._repeat(previous, repeats, read)
                r = k
        return read

    def _repeat(self, op, repeats, read):
        cache = self.cache
        a, b = op[0], op[1]
        part = type(cache) is PartitionedL2Cache
        if b <= a or type(cache) not in (L2Cache, PartitionedL2Cache) or (b - a) * (
                cache.w[op[5] if len(op) > 5 else SHARED] if part else 1) > cache.cap:
            for _ in range(repeats):
                _, miss = cache.touch(*op)
                if not op[2]:
                    read += miss
            return read
        cache.clock += repeats
        stamp = cache.clock
        ext, ids, heap = cache.ext, cache._ids, cache.heap
        i, j = cache._overlap_range(a, b)
        eid = cache._eid
        for index in range(i, j):
            x = ext.pop(ids[index])
            x[2] = stamp
            eid += 1
            ext[eid] = x
            ids[index] = eid
            heapq.heappush(heap, (stamp, eid))
        cache._eid = eid
        return read

    def _plan(self):
        cache, geo, R = self.cache, self.geo, self.R
        if (type(cache) not in (L2Cache, PartitionedL2Cache) or cache.cap <= 0 or len(geo) == 0
                or any(m > 1 for m in self.mult)):
            return False
        part = type(cache) is PartitionedL2Cache
        if part:
            w = [cache.w[g[5][3] if len(g[5]) > 3 else SHARED] for g in geo]
        else:
            if any(len(g[5]) != 3 for g in geo):
                return False
            w = [1] * len(geo)
        if any(g[1] <= 0 for g in geo):
            return False

        groups: Dict[Tuple[int, int], List[int]] = {}
        for o, g in enumerate(geo):
            groups.setdefault((g[0], g[1]), []).append(o)
        placeholder = [False] * len(geo)
        inplace = set()
        for members in groups.values():
            if len(members) == 1:
                continue
            if len(members) != 2:
                return False
            o1, o2 = members
            if geo[o1][4] or geo[o2][4] or geo[o1][5][0] == geo[o2][5][0] or w[o1] != w[o2]:
                return False
            placeholder[o2 if geo[o1][5][0] else o1] = True
            inplace.update(members)
        end = None
        for a, n in sorted(groups):
            if end is not None and a < end:
                return False
            end = a + n
        items = [o for o in range(len(geo)) if not placeholder[o]]
        if sum(w[o] * geo[o][2] for o in items) < cache.cap:
            return False

        if sum(w[o] * -(-g[2] // g[3]) for o, g in enumerate(geo)) >= cache.cap:
            return False

        w_lo = sum(w[o] * (geo[o][2] // R) for o in items if geo[o][3] == R)
        unsafe = []
        for o, (a, n, L, npc, is_pass, rest) in enumerate(geo):
            if not is_pass:
                continue
            pmax = -(-L // npc)
            d = (n - pmax) // pmax

            if d <= 0 or ((d * R) // npc - 1) * w_lo < cache.cap:
                unsafe.append(o)
        self.w, self.placeholder, self.items, self.unsafe = w, placeholder, items, unsafe
        self.part, self.inplace = part, inplace
        reads = [g for g in geo if not g[5][0]]
        self.frac_one = all(g[5][2] == 1.0 for g in reads)
        return True

    def _J(self, o, k):
        npc = self.geo[o][3]
        return (k * npc + self.R - 1) // self.R

    def _wp(self, r, k):
        R, geo, w = self.R, self.geo, self.w
        total = 0
        for o in self.items:
            L, npc = geo[o][2], geo[o][3]
            total += w[o] * (L * ((k * npc + R - 1) // R) // npc - L * ((r * npc + R - 1) // R) // npc)
        return total

    def _first_resident(self, lo, hi, ghosts):
        cache = self.cache
        x = None
        i, j = cache._overlap_range(lo, hi)
        if i < j:
            x = max(lo, cache._s[i])
        if ghosts:
            starts, spans = ghosts
            k = bisect.bisect_right(starts, lo) - 1
            if k >= 0 and spans[k][1] > lo:
                y = lo
            else:
                k += 1
                y = spans[k][0] if k < len(spans) and spans[k][0] < hi else None
            if y is not None and (x is None or y < x):
                x = y
        return x

    def _first_conflict_round(self, o, r, ghosts):
        a, n, L, npc, is_pass, rest = self.geo[o]
        j0 = (r * npc + self.R - 1) // self.R
        if j0 >= npc:
            return None
        v0 = L * j0 // npc
        if not is_pass:
            x = self._first_resident(a + v0, a + n, ghosts)
            if x is None:
                return None
            v = x - a
        else:
            off = v0 % n
            end = min(L, v0 + n)
            len1 = min(end, v0 + n - off) - v0
            x = self._first_resident(a + off, a + off + len1, ghosts)
            if x is not None:
                v = v0 + (x - a - off)
            else:
                len2 = end - v0 - len1
                x = self._first_resident(a, a + len2, ghosts) if len2 > 0 else None
                if x is None:
                    return None
                v = v0 + len1 + (x - a)
        j = -(-(v + 1) * npc // L) - 1
        return j * self.R // npc

    def _try_skip(self, r):
        R, cap = self.R, self.cache.cap
        if self._wp(r, R) < cap:
            return "never", None
        lo, hi = r, R
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._wp(r, mid) >= cap:
                hi = mid
            else:
                lo = mid
        k_cap = hi
        h = R
        for o in self.unsafe:
            a, n, L, npc, _, _ = self.geo[o]
            v = L * self._J(o, r) // npc + n
            if v < L:
                h = min(h, (-(-(v + 1) * npc // L) - 1) * R // npc)
        if h < k_cap:
            return "retry", max(r + 1, h + 1)
        ghosts = None
        gs = getattr(self.cache, "ghost", None)
        if gs:
            spans = []
            for s, e in sorted((g[0], g[1]) for g in gs.values()):
                if spans and s <= spans[-1][1]:
                    if e > spans[-1][1]:
                        spans[-1] = (spans[-1][0], e)
                else:
                    spans.append((s, e))
            ghosts = ([s for s, _ in spans], spans)
        bad, events = None, []
        for o in self.items:
            q = self._first_conflict_round(o, r, ghosts)
            if q is None or q >= k_cap:
                continue
            found = None if self.geo[o][4] or o in self.inplace else self._old_line_touches(o, r, k_cap, ghosts)
            if found is None:
                bad = q if bad is None else min(bad, q)
            else:
                events += found
        if bad is not None:
            return "retry", max(r + 1, bad + 1)
        if not events:
            return "skip", (h, None)
        if len(events) > _EVENT_LIMIT:
            return "retry", max(r + 1, min(e[0] for e in events) + 1)
        plan = self._resolve_events(r, sorted(events))
        if isinstance(plan, int):
            return "retry", max(r + 1, plan + 1)
        return "skip", (h, plan)

    def _old_line_touches(self, o, r, k_cap, ghosts):
        cache, R = self.cache, self.R
        a, n, L, npc, _, rest = self.geo[o]
        j0, j1 = (r * npc + R - 1) // R, (k_cap * npc + R - 1) // R
        lo, hi = a + n * j0 // npc, a + n * j1 // npc
        spans = []
        i, j = cache._overlap_range(lo, hi)
        for index in range(i, j):
            x = cache.ext[cache._ids[index]]
            spans.append((max(lo, x[0]), min(hi, x[1])))
        if ghosts:
            starts, gspans = ghosts
            k = max(0, bisect.bisect_right(starts, lo) - 1)
            while k < len(gspans) and gspans[k][0] < hi:
                if gspans[k][1] > lo:
                    spans.append((max(lo, gspans[k][0]), min(hi, gspans[k][1])))
                k += 1
        pieces = set()
        for s, e in spans:
            first = -(-(s - a + 1) * npc // n) - 1
            last = -(-(e - a) * npc // n) - 1
            pieces.update(range(first, last + 1))
        return [(jj * R // npc, rest[0], o, jj) for jj in sorted(pieces)]

    def _resolve_events(self, r, events):
        cache, geo, R, w = self.cache, self.geo, self.R, self.w
        cap = cache.cap
        part = self.part
        wl = cache.w if part else None
        w_ghost = cache.w_ghost if part else 0
        w_max = max(cache.w.values()) if part else 1
        ext = dict(cache.ext)
        starts, ids = list(cache._s), list(cache._ids)
        ghost = dict(getattr(cache, "ghost", {}))
        gstarts = list(getattr(cache, "gstarts", ()))
        stamps = sorted({x[2] for x in ext.values()} | {g[2] for g in ghost.values()})
        rank = {st: i for i, st in enumerate(stamps)}
        tree = [0] * (len(stamps) + 1)

        def add(st, value):
            i = rank[st] + 1
            while i < len(tree):
                tree[i] += value
                i += i & -i

        def below(i):
            total = 0
            while i > 0:
                total += tree[i]
                i -= i & -i
            return total

        def weight(x):
            return wl[x[5]] if part else 1

        for x in ext.values():
            tree[rank[x[2]] + 1] += weight(x) * (x[1] - x[0])
        for g in ghost.values():
            tree[rank[g[2]] + 1] += w_ghost * (g[1] - g[0])
        for i in range(1, len(tree)):
            parent = i + (i & -i)
            if parent < len(tree):
                tree[parent] += tree[i]
        eid = cache._eid
        removed = 0
        M = 0
        W_Q = cache.total
        round_cache = {}
        plan = {}
        for q, write, o, j in events:
            if q not in round_cache:
                before, acc = {}, 0
                for writes in (False, True):
                    for oo, (_, _, L2, npc2, _, rest2) in enumerate(geo):
                        if bool(rest2[0]) != writes:
                            continue
                        jq = (q * npc2 + R - 1) // R
                        if jq >= npc2 or jq * R // npc2 != q:
                            continue
                        before[oo] = acc
                        if not (oo in self.inplace and not self.placeholder[oo]):
                            acc += w[oo] * (L2 * (jq + 1) // npc2 - L2 * jq // npc2)
                round_cache[q] = (self._wp(r, q), before)
            base, before = round_cache[q]
            M = max(M, W_Q + base + before[o] - removed - cap)
            ev_lo, ev_hi = (M, M + w_max - 1) if M > 0 else (0, 0)

            def state(st):
                i = rank[st]
                if ev_hi <= below(i):
                    return "resident"
                if ev_lo >= below(i + 1):
                    return "gone"
                return None

            a, n, L, npc, _, rest = geo[o]
            lo, hi = a + n * j // npc, a + n * (j + 1) // npc
            cls = (rest[3] if len(rest) > 3 else SHARED) if part else None
            i0 = max(0, bisect.bisect_right(starts, lo) - 1)
            hits = []
            k = i0
            while k < len(starts) and starts[k] < hi:
                x = ext[ids[k]]
                if x[1] > lo:
                    st = state(x[2])
                    if st is None:
                        return q
                    if st == "resident":
                        if part and not write and cls == PRIVATE and x[5] == SHARED and w_ghost:
                            return q
                        hits.append(k)
                k += 1
            drop = []
            if part and ghost and (write or cls == SHARED):
                k = 0
                while k < len(gstarts) and gstarts[k][0] < hi:
                    gid = gstarts[k][1]
                    g = ghost[gid]
                    if g[1] > lo:
                        st = state(g[2])
                        if st is None:
                            return q
                        if st == "resident":
                            drop.append(gid)
                    k += 1

            parts = []
            new_starts = []
            for k in hits:
                x = ext.pop(ids[k])
                s0, e0, st0 = x[0], x[1], x[2]
                lo2, hi2 = max(lo, s0), min(hi, e0)
                wx = weight(x)
                add(st0, -wx * (hi2 - lo2))
                removed += wx * (hi2 - lo2)
                if s0 < lo:
                    eid += 1
                    ext[eid] = [s0, lo] + x[2:]
                    new_starts.append((s0, eid))
                if hi < e0:
                    eid += 1
                    ext[eid] = [hi, e0] + x[2:]
                    new_starts.append((hi, eid))
                parts.append((lo2, hi2, x[3], x[4]))
            if hits:
                for k in reversed(hits):
                    del starts[k]
                    del ids[k]
                for s0, e1 in new_starts:
                    pos = bisect.bisect_left(starts, s0)
                    starts.insert(pos, s0)
                    ids.insert(pos, e1)
            for gid in drop:
                gs0, ge0, gst = ghost.pop(gid)
                del gstarts[bisect.bisect_left(gstarts, (gs0, gid))]
                lo2, hi2 = max(lo, gs0), min(hi, ge0)
                add(gst, -w_ghost * (hi2 - lo2))
                removed += w_ghost * (hi2 - lo2)
                for s1, e1 in ((gs0, lo), (hi, ge0)):
                    if s1 < e1:
                        eid += 1
                        ghost[eid] = [s1, e1, gst]
                        bisect.insort(gstarts, (s1, eid))
            hit = sum(h2 - l2 for l2, h2, _, _ in parts)
            if write:
                items = [(lo, hi, True, rest[1])]
            else:
                items, cur = [], lo
                for l2, h2, d2, t2 in parts:
                    if cur < l2:
                        items.append((cur, l2, False, rest[1]))
                    items.append((l2, h2, d2, t2))
                    cur = h2
                if cur < hi:
                    items.append((cur, hi, False, rest[1]))
            plan[(o, j)] = (items, hit)
        q0_dirty = [(ext[e][4], ext[e][1] - ext[e][0]) for _, e in sorted((x[2], e) for e, x in ext.items() if x[3])]
        return {"events": plan, "q0_dirty": q0_dirty}

    def _sequential_reads(self, r, h, read, hits=None):
        import numpy as np
        R = self.R
        rounds, ops, subs, values = [], [], [], []
        for o, (a, n, L, npc, is_pass, rest) in enumerate(self.geo):
            if rest[0]:
                continue
            j0, j1 = (r * npc + R - 1) // R, (h * npc + R - 1) // R
            if j1 <= j0:
                continue
            if not is_pass and L * (j1 + 1) < (1 << 62) and j1 * R < (1 << 62):
                j = np.arange(j0, j1, dtype=np.int64)
                rounds.append(j * R // npc)
                ops.append(np.full(j1 - j0, o, dtype=np.int64))
                subs.append(np.zeros(j1 - j0, dtype=np.int64))
                sizes = L * (j + 1) // npc - L * j // npc
                for (oo, jj), hit in (hits or {}).items():
                    if oo == o and j0 <= jj < j1:
                        sizes[jj - j0] -= hit
                values.append(sizes.astype(np.float64) * rest[2])
                continue
            rr, vv, ss = [], [], []
            lo = L * j0 // npc
            for jj in range(j0, j1):
                hi = L * (jj + 1) // npc
                v, sub = lo, 0
                hit = hits.get((o, jj), 0) if hits and not is_pass else 0
                while v < hi:
                    s = v % n
                    e = min(n, s + hi - v)
                    rr.append(jj * R // npc)
                    ss.append(sub)
                    vv.append((e - s - hit) * rest[2])
                    v += e - s
                    sub += 1
                lo = hi
            rounds.append(np.array(rr, dtype=np.int64))
            ops.append(np.full(len(rr), o, dtype=np.int64))
            subs.append(np.array(ss, dtype=np.int64))
            values.append(np.array(vv, dtype=np.float64))
        if not values:
            return read
        rounds, ops, subs, values = (np.concatenate(x) for x in (rounds, ops, subs, values))
        order = np.lexsort((subs, ops, rounds))
        return float(np.add.accumulate(np.concatenate(([read], values[order])))[-1])

    def _skip(self, r, h, read, plan=None):
        cache, geo, R, w = self.cache, self.geo, self.R, self.w
        cap = cache.cap
        events = plan["events"] if plan else {}

        if plan:
            q0_dirty = plan["q0_dirty"]
        else:
            ext = cache.ext
            q0_dirty = [(ext[e][4], ext[e][1] - ext[e][0])
                        for _, e in sorted((x[2], e) for e, x in ext.items() if x[3])]

        excess = self._wp(r, h) - cap
        lo, hi = r, h
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self._wp(r, mid) <= excess:
                lo = mid
            else:
                hi = mid
        k_f = lo
        left = excess - self._wp(r, k_f)
        gone_bytes = {}
        for o in self.items:
            L, npc = geo[o][2], geo[o][3]
            gone_bytes[o] = L * ((k_f * npc + R - 1) // R) // npc - L * ((r * npc + R - 1) // R) // npc
        extra = []
        for (o, j), (items, _) in events.items():
            q = j * R // geo[o][3]
            if q < k_f and not geo[o][5][0]:
                extra += [((q, 0, o, i), t2, h2 - l2) for i, (l2, h2, d2, t2) in enumerate(items) if d2]
        rounds = self.entries(k_f, h)
        front, front_gone = 0, 0
        for index, (o, s, e) in enumerate(rounds[0]):
            if left <= 0:
                front = index
                break
            if self.placeholder[o]:
                continue
            ev = events.get((o, self._J(o, k_f))) if events else None
            weight = w[o] * (e - s)
            if weight - w[o] < left:
                gone_bytes[o] += e - s
                left -= weight
                front = index + 1
                if ev and not geo[o][5][0]:
                    extra += [((k_f, 0, o, i), t2, h2 - l2) for i, (l2, h2, d2, t2) in enumerate(ev[0]) if d2]
                continue
            gone = -(-left // w[o])
            if ev and len(ev[0]) > 1 and gone < e - s:
                return None
            gone_bytes[o] += gone
            left = 0
            if ev and not geo[o][5][0] and ev[0][0][2]:
                extra.append(((k_f, 0, o, 0), ev[0][0][3], gone))
            if gone < e - s:
                front, front_gone = index, gone
            else:
                front = index + 1
            break
        else:
            front = len(rounds[0])
        p_dirty = [((self._J(o, r) * R // geo[o][3], 1, o, 0), geo[o][5][1], gone_bytes[o]) for o in self.items
                   if geo[o][5][0] and gone_bytes[o] > 0]
        p_dirty = [(tag, nbytes) for _, tag, nbytes in sorted(p_dirty + extra)]

        hits = {key: hit for key, (_, hit) in events.items() if not geo[key[0]][5][0] and hit}
        if not self.collective:
            if self.frac_one:
                read += sum(geo[o][2] * ((h * geo[o][3] + R - 1) // R) // geo[o][3]
                            - geo[o][2] * ((r * geo[o][3] + R - 1) // R) // geo[o][3]
                            for o in range(len(geo)) if not geo[o][5][0]) - sum(hits.values())
            else:
                read = self._sequential_reads(r, h, read, hits)

        def touches(k0, k1):
            total = 0
            for o, (a, n, L, npc, is_pass, rest) in enumerate(geo):
                j0, j1 = (k0 * npc + R - 1) // R, (k1 * npc + R - 1) // R
                total += _pass_touches(L, n, npc, j0, j1) if is_pass else j1 - j0
            return total
        clock = cache.clock + touches(r, k_f)
        eid = cache._eid
        part = self.part
        new_ext, order, heap, total = {}, [], [], 0
        by_round = {}
        for (o, j), value in events.items():
            by_round[(j * R // geo[o][3], o)] = value
        event_rounds = {k for k, _ in by_round}
        for offset, rnd in enumerate(rounds):
            first = offset == 0
            k = k_f + offset
            in_events = k in event_rounds
            for index, (o, s, e) in enumerate(rnd):
                clock += 1
                if first and index < front or self.placeholder[o]:
                    continue
                rest = geo[o][5]
                ev = by_round.get((k, o)) if in_events else None
                pieces = list(ev[0]) if ev else [(s, e, rest[0], rest[1])]
                if first and index == front and front_gone:
                    l2, h2, d2, t2 = pieces[0]
                    pieces[0] = (l2 + front_gone, h2, d2, t2)
                for l2, h2, d2, t2 in pieces:
                    eid += 1
                    if part:
                        new_ext[eid] = [l2, h2, clock, d2, t2, rest[3] if len(rest) > 3 else SHARED]
                    else:
                        new_ext[eid] = [l2, h2, clock, d2, t2]
                    order.append((l2, eid))
                    heap.append((clock, eid))
                    total += w[o] * (h2 - l2)
        if clock != cache.clock + touches(r, h) or total > cap:
            raise AssertionError("l2_sim streaming skip: inconsistent stream geometry")

        log, written_by = cache.evict_log, cache.written_by
        for tag, nbytes in q0_dirty + p_dirty:
            previous = written_by.get(tag)
            if log is not None and tag not in log:
                log[tag] = previous
            written_by[tag] = (0.0 if previous is None else previous) + nbytes
            cache.written += nbytes
        order.sort()
        ext = cache.ext
        ext.clear()
        ext.update(new_ext)
        cache._s[:] = [s for s, _ in order]
        cache._ids[:] = [i for _, i in order]
        cache.heap[:] = heap
        cache.total = total
        cache.clock = clock
        cache._eid = eid
        if part:
            cache.ghost.clear()
            cache.gstarts.clear()
        return read

def _run_kernel(cache: L2Cache, ops, slice_bytes: int, collective: bool = False) -> Tuple[float, float, Dict[str, float]]:
    if _REFERENCE:
        return _run_kernel_reference(cache, ops, slice_bytes, collective)
    if len(ops) <= 16 and sum(_stream_length(op) for op in ops) <= slice_bytes:
        return _run_kernel_reference(cache, ops, slice_bytes, collective)
    w0 = cache.written

    evicted = cache.evict_log = {}
    try:
        read = _KernelStream(cache, ops, slice_bytes, collective).run()
    finally:
        cache.evict_log = None

    by = {}
    if evicted:
        written_by = cache.written_by
        for t in (written_by if len(evicted) > 1 else evicted):
            if t in evicted:
                previous = evicted[t]
                delta = written_by[t] - (0.0 if previous is None else previous)
                if delta > 0:
                    by[t] = delta
    excluded = sum(v for t, v in by.items() if t.startswith("@collective:"))
    by = {t: v for t, v in by.items() if not t.startswith("@collective:")}
    return (0.0 if collective else read), cache.written - w0 - excluded, by

def _periodic_state(cache):
    entries = [(x[2], eid, ('live', *x[:2], *x[3:])) for eid, x in cache.ext.items()]
    entries += [(x[2], eid, ('ghost', *x[:2])) for eid, x in getattr(cache, 'ghost', {}).items()]
    ranks = {stamp: i for i, stamp in enumerate(sorted({x[0] for x in entries}))}
    return tuple((ranks[stamp], value) for stamp, _, value in sorted(entries))

def _write_reset_suffix(reverse_ops, bound, weights=None):
    def add(spans, a, b):
        i = bisect.bisect_left(spans, (a,))
        if i and spans[i - 1][1] >= a:
            i -= 1
        j = i
        removed = 0
        while j < len(spans) and spans[j][0] <= b:
            lo, hi = spans[j]
            a, b = min(a, lo), max(b, hi)
            removed += hi - lo
            j += 1
        spans[i:j] = [(a, b)]
        return b - a - removed

    def subtract(spans, a, b):
        i = bisect.bisect_left(spans, (a,))
        if i and spans[i - 1][1] > a:
            i -= 1
        j = i
        replacement = []
        while j < len(spans) and spans[j][0] < b:
            lo, hi = spans[j]
            if lo < a:
                replacement.append((lo, a))
            if hi > b:
                replacement.append((b, hi))
            j += 1
        spans[i:j] = replacement

    def intersects(spans, a, b):
        i = bisect.bisect_left(spans, (a,))
        return ((i > 0 and spans[i - 1][1] > a) or
                (i < len(spans) and spans[i][0] < b))

    def search(promote):
        block, exposed, all_exposed = [], [], []
        covered = 0
        have_x = False
        for n, op in enumerate(reverse_ops, 1):
            a, b, write = op[:3]
            if write:
                subtract(all_exposed, a, b)
            else:
                add(all_exposed, a, b)
            if have_x and intersects(exposed, a, b):
                block, covered = [], 0
                if promote:
                    exposed = all_exposed.copy()
                    continue
                have_x = False
                exposed, all_exposed = [], []
                if not write:
                    add(all_exposed, a, b)
            covered += add(block, a, b) * (weights[op[5]] if weights is not None else 1)
            if not have_x:
                if write:
                    subtract(exposed, a, b)
                else:
                    add(exposed, a, b)
            if covered >= bound:
                if have_x:
                    return n
                have_x = True
                block, covered = [], 0
        return None

    if not isinstance(reverse_ops, (list, tuple)):
        reverse_ops = tuple(reverse_ops)
    return search(True) or search(False)

def _prime_periodic(cache, reverse_ops, *, allow_period_replay=True):
    if cache.cap <= 0:
        return
    weight = min(cache.w.values()) if isinstance(cache, PartitionedL2Cache) else 1
    bound = -(-cache.cap // weight)
    tail, intervals = [], []
    covered = blocks = 0
    reset = False
    for op in reverse_ops:
        tail.append(op)
        a, b = op[:2]
        i = bisect.bisect_left(intervals, (a, b))
        if (i and intervals[i - 1][1] > a) or (i < len(intervals) and intervals[i][0] < b):
            intervals, covered, blocks = [], 0, 0
            i = 0
        intervals.insert(i, (a, b))
        covered += b - a
        if covered >= bound:
            blocks += 1
            covered = 0
            if blocks == 2:
                reset = True
                break
    ops = list(reversed(tail))
    if not reset and not allow_period_replay:
        spans = sorted((op[0], op[1]) for op in ops)
        covered_union = 0
        end = None
        for start, stop in spans:
            covered_union += max(0, stop - max(start, end if end is not None else start))
            end = max(end if end is not None else stop, stop)
        max_weight = max(cache.w.values()) if isinstance(cache, PartitionedL2Cache) else 1
        if cache.ext or getattr(cache, "ghost", None) or covered_union * max_weight > cache.cap:
            weights = cache.w if isinstance(cache, PartitionedL2Cache) else None
            n = _write_reset_suffix(tail, cache.cap, weights=weights)
            if n is None:
                raise ValueError('onepass periodic_prime: no reset witness or fitting working set; period replay disabled')
            ops = list(reversed(tail[:n]))
        reset = True
    previous = _periodic_state(cache)
    for _ in range(1 if reset else 10):
        for op in ops:
            cache.touch(*op)
        state = _periodic_state(cache)
        if reset or state == previous:
            cache.written = 0.0
            cache.written_by.clear()
            return
        previous = state
    raise ValueError('periodic_prime cache state did not converge within ten periods')

def _split_k_inplace_parts(kernel):
    k_total = sum(kernel.slice_lengths)
    consumed = 0
    for index, length in enumerate(kernel.slice_lengths):
        inputs = []
        for a in kernel.original_reads[:2]:
            lo = round(a.nbytes * consumed / k_total)
            hi = round(a.nbytes * (consumed + length) / k_total)
            inputs.append(RAccess(a.tensor, hi - lo, a.weight, getattr(a, "offset", 0.0) + lo))
        consumed += length
        if index:
            inputs.extend(kernel.original_writes)
        else:
            inputs.extend(kernel.epilogue_reads)
        yield Kernel(kernel.op, kernel.name, kernel.kind, tuple(inputs), kernel.original_writes,
                     gemm=kernel.gemm, activity_name=kernel.activity_name)

def _run_split_k_inplace(cache, kernel, addr, bufs, rules, classify, slice_bytes, passes=()):
    read, write, by = 0.0, 0.0, {}
    for part in _split_k_inplace_parts(kernel):
        r, w, b = _run_kernel(cache, _kernel_ops(part, addr, bufs, rules, classify, passes), slice_bytes)
        read += r
        write += w
        for name, value in b.items():
            by[name] = by.get(name, 0.0) + value
    return read, write, by

def simulate(kernels: Sequence[Kernel], capacity_bytes: int, rules: StepRules, steps: int = 3,
             slice_fraction: float = 1 / 32, sharing_classes: bool = False,
             physical_capacity_bytes: Optional[int] = None, private_copies=None,
             gemm_min_tile_m: int = GEMM_MIN_TILE_M, steady_state: str = 'replay3',
             address_plan: str = 'warmup_replay', operand_passes=None) -> SimResult:
    run = _SimRun(kernels, capacity_bytes, rules, steps, slice_fraction, sharing_classes, physical_capacity_bytes,
                  private_copies, gemm_min_tile_m, steady_state, address_plan, operand_passes)
    cache = run.cache
    ops_per_step = []
    for s in range(run.steps):
        rd, wr, by = [], [], []
        for i, k in enumerate(kernels):
            r, w, b = run.kernel(cache, s, i)
            run.record(i, r, w, b, rd, wr, by)
        ops_per_step.append((rd, wr, by))
    rd, wr, by = ops_per_step[-1]
    return run.result(cache, rd, wr, by)

class _SimRun:
    def __init__(self, kernels, capacity_bytes, rules, steps, slice_fraction, sharing_classes,
                 physical_capacity_bytes, private_copies, gemm_min_tile_m, steady_state, address_plan,
                 operand_passes):
        if steady_state not in ('replay3', 'periodic_prime'):
            raise ValueError(f'unknown L2 steady_state: {steady_state!r}')
        if sharing_classes and (physical_capacity_bytes is None or physical_capacity_bytes <= 0):
            raise ValueError("sharing_classes requires physical_capacity_bytes from the hardware description")
        if sharing_classes and private_copies is None:
            raise ValueError("sharing_classes requires private_copies from the hardware L2 policy")
        if address_plan not in ('warmup_replay', 'onepass'):
            raise ValueError(f'unknown L2 address_plan: {address_plan!r}')
        operand_passes = dict(operand_passes or {})
        for i in operand_passes:
            if kernels[i].kind not in ('gemm', 'batched_gemm'):
                raise ValueError(f"operand passes given for a {kernels[i].kind} launch ({kernels[i].name})")
        bufs = tensor_lifetimes(kernels, rules)
        if address_plan == 'onepass':
            addr, alloc = plan_addresses_onepass(kernels, bufs, rules)
            maps = [addr] * steps
        else:
            maps, alloc = plan_addresses(kernels, bufs, rules, steps)
        if sharing_classes:
            cache = PartitionedL2Cache(capacity_bytes, physical_capacity_bytes, private_copies)
            phase = rules.phase
            classify = lambda k, a, w: sharing_class(k, a, w, phase, gemm_min_tile_m)  # noqa: E731
        else:
            cache = L2Cache(capacity_bytes)
            classify = None
        slice_bytes = max(4096, int(capacity_bytes * slice_fraction))
        if steady_state == 'periodic_prime':
            addr = maps[-1]

            def reverse_ops():
                for i in range(len(kernels) - 1, -1, -1):
                    k = kernels[i]
                    collective = k.kind == 'collective'
                    if collective and rules.collective_buffers == 'leave_l2':
                        continue
                    parts = (list(_split_k_inplace_parts(k)) if isinstance(k, SplitKKernel)
                             and k.reduction_scheme == 'inplace' else [k])
                    for part in reversed(parts):
                        rounds = _stream_rounds(_kernel_ops(part, addr, bufs, rules, classify,
                                                            operand_passes.get(i, ())), slice_bytes)
                        for r in reversed(rounds):
                            for op in reversed(r):
                                if collective and op[2]:
                                    op = op[:3] + ('@collective:' + op[3],) + op[4:]
                                yield op

            _prime_periodic(cache, reverse_ops(), allow_period_replay=address_plan != 'onepass')
            maps, steps = [addr], 1
        self.kernels, self.capacity, self.rules, self.steps = kernels, capacity_bytes, rules, steps
        self.operand_passes, self.bufs, self.maps, self.alloc = operand_passes, bufs, maps, alloc
        self.cache, self.classify, self.slice_bytes = cache, classify, slice_bytes

    def skips(self, i):
        return self.kernels[i].kind == "collective" and self.rules.collective_buffers == "leave_l2"

    def ops(self, s, i):
        k = self.kernels[i]
        passes = self.operand_passes.get(i, ())
        if isinstance(k, SplitKKernel) and k.reduction_scheme == "inplace":
            return [_kernel_ops(part, self.maps[s], self.bufs, self.rules, self.classify, passes)
                    for part in _split_k_inplace_parts(k)]
        return _kernel_ops(k, self.maps[s], self.bufs, self.rules, self.classify, passes)

    def kernel(self, cache, s, i, ops=None):
        k = self.kernels[i]
        if self.skips(i):
            return 0.0, 0.0, {}
        if isinstance(k, SplitKKernel) and k.reduction_scheme == "inplace":
            if ops is None:
                return _run_split_k_inplace(cache, k, self.maps[s], self.bufs, self.rules, self.classify,
                                            self.slice_bytes, self.operand_passes.get(i, ()))
            read, write, by = 0.0, 0.0, {}
            for part_ops in ops:
                r, w, b = _run_kernel(cache, part_ops, self.slice_bytes)
                read += r
                write += w
                for name, value in b.items():
                    by[name] = by.get(name, 0.0) + value
            return read, write, by
        if ops is None:
            ops = self.ops(s, i)
        return _run_kernel(cache, ops, self.slice_bytes, k.kind == "collective")

    def record(self, i, r, w, b, rd, wr, by):
        kernels = self.kernels
        if self.skips(i):
            rd.append(0.0); wr.append(0.0); by.append({})
            return
        if kernels[i].kind == "collective":
            j = next((i for i in range(len(wr) - 1, -1, -1) if kernels[i].kind != "collective"), None)
            if j is not None:
                wr[j] += w
                for t, value in b.items():
                    by[j][t] = by[j].get(t, 0.0) + value
            w, b = 0.0, {}
        rd.append(r)
        wr.append(w)
        by.append(b)

    def result(self, cache, rd, wr, by):
        end_dirty = sum(x[1] - x[0] for x in cache.ext.values() if x[3])
        return SimResult(int(self.capacity), rd, wr, by, end_dirty, self.alloc.n_segments)

_LOCKSTEP = [True]
_CACHE_FIELDS = {L2Cache: {"cap", "ext", "_s", "_ids", "heap", "total", "clock", "_eid", "written", "written_by",
                           "evict_log"}}
_CACHE_FIELDS[PartitionedL2Cache] = _CACHE_FIELDS[L2Cache] | {"w", "w_ghost", "private_copies", "ghost", "gstarts"}

def _clone_cache(cache):
    copy = object.__new__(type(cache))
    state = dict(cache.__dict__)
    state["ext"] = {eid: list(x) for eid, x in cache.ext.items()}
    state["_s"], state["_ids"] = list(cache._s), list(cache._ids)
    heap = [(x[2], eid) for eid, x in cache.ext.items()]
    if type(cache) is PartitionedL2Cache:
        state["ghost"] = {gid: list(g) for gid, g in cache.ghost.items()}
        state["gstarts"] = list(cache.gstarts)
        heap += [(g[2], gid) for gid, g in cache.ghost.items()]
    heapq.heapify(heap)
    state["heap"] = heap
    state["written_by"] = dict(cache.written_by)
    state["evict_log"] = None
    copy.__dict__.update(state)
    return copy

def _ops_equal(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if type(x) is not type(y):
            return False
        if type(x) is list:
            if not _ops_equal(x, y):
                return False
            continue
        if x != y or (type(x) is _PassOp and x.passes != y.passes):
            return False
        for u, v in zip(x, y):
            if type(u) is not type(v):
                return False
    return True

def simulate_ranks(kernel_lists, capacity_bytes, rules_list, steps=3, slice_fraction=1 / 32, sharing_classes=False,
                   physical_capacity_bytes=None, private_copies=None, gemm_min_tile_m=GEMM_MIN_TILE_M,
                   steady_state='replay3', address_plan='warmup_replay', operand_passes_list=None):
    count = len(kernel_lists)
    passes = list(operand_passes_list) if operand_passes_list is not None else [None] * count
    settings = (steps, slice_fraction, sharing_classes, physical_capacity_bytes, private_copies, gemm_min_tile_m,
                steady_state, address_plan)
    if not _LOCKSTEP[0] or count < 2:
        return [simulate(kernel_lists[r], capacity_bytes, rules_list[r], steps, slice_fraction, sharing_classes,
                         physical_capacity_bytes, private_copies, gemm_min_tile_m, steady_state, address_plan,
                         passes[r]) for r in range(count)]
    runs = [_SimRun(kernel_lists[r], capacity_bytes, rules_list[r], *settings[:6], steady_state, address_plan,
                    passes[r]) for r in range(count)]
    if any(set(run.cache.__dict__) != _CACHE_FIELDS.get(type(run.cache)) for run in runs):
        raise AssertionError("simulate_ranks: unknown L2 cache representation; _clone_cache would not copy it")
    lanes = {}
    for r, run in enumerate(runs):
        lanes.setdefault((len(run.kernels), run.steps), []).append(r)
    results = [None] * count
    for members in lanes.values():
        _lockstep(runs, members, results)
    return results

def _lockstep(runs, members, results):
    n, steps = len(runs[members[0]].kernels), runs[members[0]].steps

    reps, context = [], {}
    for m in members:
        run = runs[m]
        for c, rep in enumerate(reps):
            other = runs[rep]
            if other.rules is run.rules and other.bufs == run.bufs and other.maps == run.maps:
                context[m] = c
                break
        else:
            context[m] = len(reps)
            reps.append(m)

    live_until = {}
    for m in members:
        for t, buf in runs[m].bufs.items():
            until = math.inf if buf.free_after is None else buf.free_after
            if until > live_until.get(t, -1):
                live_until[t] = until
    interned, by_id = {}, {}

    def fingerprint(k):
        key = id(k)
        hit = by_id.get(key)
        if hit is not None and hit[0] is k:
            return hit[1]
        try:
            value = interned.setdefault(k, len(interned))
        except TypeError:
            value = ("id", key)
        by_id[key] = (k, value)
        return value

    def canonical(cache, s, i):
        state = _periodic_state(cache)
        dirty = {value[4] for _, value in state if value[0] == 'live' and value[3]}

        def relevant(t):
            if s < steps - 1 or t in dirty:
                return True
            base = t[len("@collective:"):] if t.startswith("@collective:") else t
            return live_until.get(base, -1) >= i
        return state, tuple(t for t in cache.written_by if relevant(t))

    groups = []
    for m in members:
        groups.append([runs[m].cache, [m]])
    groups = _merge_groups(groups, canonical, 0, 0)
    lists = {}
    for s in range(steps):
        lists = {m: ([], [], []) for m in members}
        for i in range(n):
            next_groups = []
            built = {}

            def ranges(sig, run):
                hit = built.get(sig)
                if hit is None:
                    skip = run.skips(i)
                    hit = built[sig] = (skip, None if skip else run.ops(s, i))
                return hit
            for cache, group in groups:
                sigs = {}
                for m in group:
                    run = runs[m]
                    sigs.setdefault((context[m], fingerprint(run.kernels[i]),
                                     tuple(run.operand_passes.get(i, ()))), []).append(m)
                if len(sigs) == 1:
                    (sig, _), = sigs.items()
                    run = runs[group[0]]
                    r, w, b = run.kernel(cache, s, i, ranges(sig, run)[1])
                    for m in group:
                        runs[m].record(i, r, w, dict(b), *lists[m])
                    next_groups.append([cache, group])
                    continue
                classes = []
                for sig, sig_members in sigs.items():
                    skip, ops = ranges(sig, runs[sig_members[0]])

                    collective = runs[sig_members[0]].kernels[i].kind == "collective"
                    for c in classes:
                        if c[0] == skip and c[3] == collective and (skip or _ops_equal(c[1], ops)):
                            c[2].extend(sig_members)
                            break
                    else:
                        classes.append([skip, ops, list(sig_members), collective])
                for index, (skip, ops, group_members, _) in enumerate(classes):
                    c = cache if index == len(classes) - 1 else _clone_cache(cache)
                    r, w, b = runs[group_members[0]].kernel(c, s, i, ops)
                    for m in group_members:
                        runs[m].record(i, r, w, dict(b), *lists[m])
                    next_groups.append([c, sorted(group_members)])
            groups = _merge_groups(next_groups, canonical, s, i + 1) if len(next_groups) > 1 else next_groups
    for cache, group in groups:
        for m in group:
            results[m] = runs[m].result(cache, *lists[m])

def _merge_groups(groups, canonical, s, i):
    if len(groups) < 2:
        return groups
    buckets = {}
    for g in groups:
        cache = g[0]

        buckets.setdefault((cache.total, tuple(cache._s), tuple(start for start, _ in getattr(cache, "gstarts", ()))),
                           []).append(g)
    merged = []
    for bucket in buckets.values():
        if len(bucket) == 1:
            merged.append(bucket[0])
            continue
        states = {}
        for g in bucket:
            key = canonical(g[0], s, i)
            hit = states.get(key)
            if hit is None:
                states[key] = g
                merged.append(g)
            else:
                hit[1] = sorted(hit[1] + g[1])
    return merged
