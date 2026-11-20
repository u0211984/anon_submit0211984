from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

def _vectors(**values):
    from flash_attention import _VECTOR_KEYS
    result = dict.fromkeys(_VECTOR_KEYS, 0)
    result.update(values)
    return result

def _gemm_bytes(x, y, k, atom_m):
    return 2 * (x * k * (8 // atom_m) + y * k * atom_m)

def _scalar_row_read_bytes(rows, warp_copies, row_offset=0):
    groups, tail = divmod(rows, 8)
    offset = row_offset % 8
    sectors = groups * (1 + bool(offset))
    if tail:
        sectors += (offset + tail + 7) // 8
    return sectors * 32 * warp_copies

def _time(spec, rates, ctas, resident, program, unique_reads, delivered_reads,
          atomic_unique_reads=0, atomic_payload_bytes=0, delivered_writes=0,
          spill_program=None, tiles=None, native=None):
    gpu = rates.gpu
    clock, sms = gpu.clock_hz, gpu.num_sms
    smem_rate = rates.smem_bandwidth_bytes_per_s / clock / sms
    from flash_attention import l2_timeline_terms
    l2_rate, rates = l2_timeline_terms(rates, clock)
    hbm_rate = gpu.hbm_bandwidth_bytes_per_s / clock
    tensor_rate = gpu.tensor_flops_per_s / clock / sms / 4
    vector_rate = rates.vector_flops_per_s / clock / sms / 2 / 4
    sfu_rate = rates.sfu_ops_per_cycle / 4

    fraction = 1.0 if rates.l2_bypass else unique_reads / delivered_reads if delivered_reads else 0.0
    if rates.l2_bypass:
        atomic_unique_reads = atomic_payload_bytes
    atomic_fraction = atomic_unique_reads / atomic_payload_bytes if atomic_payload_bytes else 0.0
    result = None
    if native is not None and spill_program is None:
        import flash_native
        result = flash_native.backward(
            sms=sms, resident=resident, ctas=ctas, smem_rate=smem_rate, l2_rate=l2_rate, hbm_rate=hbm_rate,
            tensor_rate=tensor_rate, vector_rate=vector_rate, sfu_rate=sfu_rate,
            mma_latency=rates.mma_latency_cycles, smem_latency=rates.smem_latency_cycles,
            l2_latency=rates.l2_latency_cycles, hbm_latency=rates.hbm_latency_cycles, fraction=fraction,
            atomic_fraction=atomic_fraction, **native)
    if result is None and native is not None and native.get("ff") is not None:
        raise ValueError("flash_fast_forward needs the native flash timeline (and no spill program)")
    if result is None:
        result = _schedule(rates, ctas, resident, program, spill_program, tiles, sms, smem_rate, l2_rate, hbm_rate,
                           tensor_rate, vector_rate, sfu_rate, fraction, atomic_fraction)
    cycles, starts, finishes, services = result
    return cycles / clock + rates.launch_overhead_s, {
        "kernel_cycles": cycles, "cta_start_cycles": starts, "cta_end_cycles": finishes,
        "resource_service_cycles": services,
        "hbm_timing_unique_read_bytes": unique_reads + atomic_unique_reads,
        "hbm_timing_external_unique_read_bytes": unique_reads,
        "hbm_timing_atomic_internal_unique_read_bytes": atomic_unique_reads,
        "hbm_timing_atomic_internal_read_fraction": atomic_fraction,
        "hbm_timing_policy": "extended_timing first-touch unique demand; activity remains L2-sim owned",
        "rate_contract": "opmodel HardwareSpec rates and latencies; util=1",
        "cta_order": "blockIdx.x fastest; next free SM slot; deterministic SM-fast initial slots",
        "warp_to_smsp": "warp_id modulo 4; 8 warps share 4 SMSPs",
    }

def atomic_rmw_read_bytes(kernel) -> int:
    return kernel.diagnostics.get("dq_atomic_write_bytes", 0)

def _schedule(rates, ctas, resident, program, spill_program, tiles, sms, smem_rate, l2_rate, hbm_rate, tensor_rate,
              vector_rate, sfu_rate, fraction, atomic_fraction):
    from flash_attention import _Timeline
    timeline = _Timeline()
    starts, finishes = [0.0] * ctas, [0.0] * ctas
    next_cta = 0

    def launch(sm, slot):
        nonlocal next_cta
        if next_cta == ctas:
            return
        cid = next_cta
        next_cta += 1
        starts[cid] = timeline.now
        spill_writes = []
        smem_resource = f"smem:{sm}"
        vector_resources = tuple(f"vector:{sm}:{w}" for w in range(4))
        sfu_resources = tuple(f"sfu:{sm}:{w}" for w in range(4))
        tensor_resources = tuple(f"tensor:{sm}:{w}" for w in range(4))

        def node(deps=(), resource=None, service=0.0, latency=0.0, callback=None):
            return timeline.add(deps, resource, service, latency, cid, callback)

        def spill(phase, deps=(), index=0):
            if spill_program is None:
                return tuple(deps)
            from flash_attention_spills import service
            return service(spill_program, phase, index, tiles[cid], deps, node,
                           smem_resource, smem_rate, l2_rate, 8, spill_writes,
                           smem_latency_cycles=rates.smem_latency_cycles,
                           l2_latency_cycles=rates.l2_latency_cycles)

        def shared(count, deps=()):
            return node(deps, smem_resource, count / smem_rate,
                        rates.smem_latency_cycles if count else 0)

        def read(count, deps=(), staging=False):
            l2_count = count
            hbm_count = count * fraction
            l2 = node(deps, "l2", l2_count / l2_rate, rates.l2_latency_cycles if l2_count else 0)
            hbm = node(deps, "hbm", hbm_count / hbm_rate,
                       rates.hbm_latency_cycles if hbm_count else 0)
            return shared(count, (l2, hbm)) if staging else node((l2, hbm))

        def write(count, deps=(), atomic=False):
            l2_count = count
            l2 = node(deps, "l2", l2_count / l2_rate,
                      rates.l2_latency_cycles if l2_count else 0)

            hbm_count = count * (1 + atomic_fraction if atomic else 1)
            hbm = node(deps, "hbm", hbm_count / hbm_rate,
                       rates.hbm_latency_cycles if hbm_count else 0)
            return node((l2, hbm))

        def vector(count, warp, deps=()):
            return node(deps, vector_resources[warp % 4], count / vector_rate)

        def sfu(count, warp, deps=()):
            return node(deps, sfu_resources[warp % 4], count / sfu_rate)

        def gemm(x, y, k, a, warp, deps=()):
            b = 8 // a
            loads, maths = [], []
            load_count = 2 * (x // a + y // b) * 16
            math_service = 2 * (x // a) * (y // b) * 16 / tensor_rate
            tensor_resource = tensor_resources[warp % 4]

            load_service = load_count / smem_rate
            load_latency = rates.smem_latency_cycles if load_count else 0
            math_latency = rates.mma_latency_cycles
            add = timeline.add

            for group in range(k // 16):
                prereq = list(deps)
                if loads:
                    prereq.append(loads[-1])
                if group > 1:
                    prereq.append(maths[group - 2])
                ld = add(prereq, smem_resource, load_service, load_latency, cid)
                loads.append(ld)
                mm = add([ld] + ([maths[-1]] if maths else []),
                         tensor_resource, math_service, math_latency, cid)
                maths.append(mm)
            return maths[-1]

        def finish(_):
            finishes[cid] = timeline.now
            launch(sm, slot)

        def complete(done):
            if spill_writes:
                node(spill_writes, callback=finish)
            else:
                finish(done)

        program(cid, node, shared, read, write, vector, sfu, gemm, complete, spill)

    for slot in range(resident):
        for sm in range(sms):
            launch(sm, slot)
    cycles = timeline.run()
    services = {}
    for key, value in timeline.service.items():
        kind = key.split(":")[0]
        services[kind] = services.get(kind, 0.0) + value
    return cycles, starts, finishes, services

def _simple(spec, rates, convert, describe_only=False):
    from flash_attention import _flash_cache_key, _flash_resource_key
    if describe_only:
        return deepcopy(_cached_simple(spec, None, convert, None, True))
    rates = replace(rates, reduction_time_s=None)
    spec = replace(spec, rates=rates)
    return deepcopy(_cached_simple(spec, rates, convert, _flash_cache_key(spec, rates, convert, _flash_resource_key(spec))))

@lru_cache(maxsize=128)
def _cached_simple(spec, rates, convert, exact_key, describe_only=False):
    from flash_attention import FlashKernel, _ceildiv, _occupancy
    m, d = (128 if spec.head_dim == 64 else 64), spec.head_dim
    grid = (_ceildiv(spec.q_len, m), spec.batch, spec.q_heads)
    ctas = math.prod(grid)
    rows = ctas * m
    valid = spec.batch * spec.q_heads * spec.q_len
    qbytes, accum = valid * d * 2, rows * d * 4
    if convert:
        kind, name = "bwd_convert_dq", "flash_bwd_convert_dq_kernel"

        vec = _vectors(fadd=rows * d, fmul=rows * d)
        reads, writes = (("dq_accum", accum),), (("dq", qbytes),)
        ld = st = rows * d * 2
        smem = m * d * 2
    else:
        kind, name = "bwd_dot_do_o", "flash_bwd_dot_do_o_kernel"

        vec = _vectors(fmul=16 * rows, ffma=rows * (d - 8), fadd=24 * rows)
        reads = (("do", qbytes), ("o", qbytes))
        writes = (("dsum", rows * 4), ("dq_accum", accum))
        ld = st = smem = 0
    l2read, l2write = sum(v for _, v in reads), sum(v for _, v in writes)
    resident, resources = (None, {}) if describe_only else _occupancy(spec, rates, kind, smem)

    def program(cid, node, shared, read, write, vector, sfu, gemm, complete, spill):
        valid_m = min(m, spec.q_len - (cid % grid[0]) * m)
        incoming = read(m * d * 4 if convert else valid_m * d * 4)
        if convert:
            ops = 2 * m * d
        else:
            ops = 16 * m + m * (d - 8) + 24 * m
        work = [vector(ops / 8, w, (incoming,)) for w in range(8)]
        if convert:
            staged = shared(m * d * 2, work)
            work = [shared(m * d * 2, (staged,))]
        outgoing = valid_m * d * 2 if convert else m * (d * 4 + 4)
        node((write(outgoing, work),), callback=complete)

    time_s, timing = ((0.0, {}) if describe_only else
                      _time(spec, rates, ctas, resident, program, l2read, l2read,
                            delivered_writes=l2write,
                            native=dict(kind="convert_dq" if convert else "dot_do_o", m=m, n=128, d=d, a=1, b=1,
                                        q_len=spec.q_len, kv_len=spec.kv_len, grid0=grid[0], nq=grid[0],
                                        batch=spec.batch, q_heads=spec.q_heads, dao=spec.source == "dao",
                                        minima=())))
    return FlashKernel(name, grid, ctas, 1, 0, 0, vec, dict(ex2=0, rcp=0, lg2=0),
                       ld, st, l2read, l2write, reads, writes, {}, time_s, {
                           "block_m": m, "block_n": 128, "warps": 8,
                           "smem_bytes_per_cta": smem,
                           "smem_ldsm": 0, "smem_ldgsts": 0, "smem_ld": ld, "smem_st": st,
                           "rounded_query_rows": _ceildiv(spec.q_len, m) * m,
                           "clear_dq_accum": not convert,
                           "nsplits": 1,
                           "source": "D/src/flash_bwd_launch_template.h:52-138; flash_bwd_preprocess_kernel.h:25-136,183-262",
                           "vector_count_scope": "source arithmetic; runtime convert loop includes first split add; dot products contract to FMA",
                           "time_mode": "counts_only" if describe_only else "structural", **resources, **timing,
                       })

def _main_args(spec, rates):
    from flash_attention import _flash_cache_key, _flash_resource_key

    rates = replace(rates, reduction_time_s=None)
    spec = replace(spec, rates=rates)
    return spec, rates, _flash_cache_key(spec, rates, _flash_resource_key(spec))

_PREFETCHED: dict = {}
_EXECUTOR = None

def prefetch_main(spec, rates) -> None:
    global _EXECUTOR
    args = _main_args(spec, rates)
    if args[2] in _PREFETCHED:
        return
    if _EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor
        _EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-bwd")
    _PREFETCHED[args[2]] = _EXECUTOR.submit(_cached_main, *args)

def _main(spec, rates, describe_only=False):
    if describe_only:
        return deepcopy(_cached_main(spec, None, None, True))
    args = _main_args(spec, rates)
    pending = _PREFETCHED.pop(args[2], None)
    if pending is not None:
        return deepcopy(pending.result())
    return deepcopy(_cached_main(*args))

@lru_cache(maxsize=128)
def _cached_main(spec, rates, exact_key, describe_only=False):
    if describe_only:
        return _main_uncached(spec, rates, True)

    import flash_disk_cache
    return flash_disk_cache.memo("bwd_main", exact_key, lambda: _main_uncached(spec, rates, False))

def _main_uncached(spec, rates, describe_only):
    from flash_attention import FlashKernel, _ceildiv, _occupancy
    m, n, d = (128 if spec.head_dim == 64 else 64), 128, spec.head_dim
    a = 4 if d == 64 else 2
    b = 8 // a
    grid = (_ceildiv(spec.kv_len, n), spec.batch, spec.q_heads)
    ctas = math.prod(grid)
    nq = _ceildiv(spec.q_len, m)

    minima = [max(0, (nb * n + spec.q_len - spec.kv_len) // m)
              if spec.causal else 0 for nb in range(grid[0])]
    tiles = [(nq - lo) for lo in minima] * (spec.batch * spec.q_heads)
    pairs = sum(tiles)
    replicas = spec.batch * spec.q_heads
    valid_q_visits = sum(spec.q_len - lo * m for lo in minima) * replicas
    issued = 10 * m * n * d * pairs
    valid_pairs = sum(min(spec.kv_len, max(0, q + spec.kv_len - spec.q_len + 1))
                      for q in range(spec.q_len)) if spec.causal else spec.q_len * spec.kv_len
    useful = 10 * d * replicas * valid_pairs

    lse_load = sum(_scalar_row_read_bytes(spec.q_len - lo * m, b, h * spec.q_len)
                   for h in range(replicas) for lo in minima)
    dsum_load = pairs * _scalar_row_read_bytes(m, b)
    qdo_load = valid_q_visits * d * 2 * 2
    kv_load = replicas * spec.kv_len * d * 2 * 2
    ldgsts = qdo_load + kv_load

    tail_clear = ((pairs * m - valid_q_visits) * d * 2 * 2
                  + replicas * (grid[0] * n - spec.kv_len) * d * 2 * 2)
    gemm_shapes = ((m, n, d, a), (m, n, d, a),
                   (n, d, m, 4), (m, d, n, a), (n, d, m, 4))
    per_gemm = [_gemm_bytes(*shape) for shape in gemm_shapes]
    ldsm = sum(per_gemm) * pairs

    epilogue = ctas * 2 * n * d * 2
    st = pairs * 2 * m * n * 2 + epilogue + tail_clear
    ld = epilogue
    atomic = pairs * m * d * 4
    expanded = replicas * spec.kv_len * d * 2
    kv_bytes = spec.batch * spec.kv_heads * spec.kv_len * d * 2
    touched_rows = spec.q_len - min(minima) * m
    rounded_touched = nq * m - min(minima) * m
    q_bytes = replicas * touched_rows * d * 2
    accum_bytes = replicas * rounded_touched * d * 4
    reads = (("q", q_bytes), ("k", kv_bytes), ("v", kv_bytes), ("do", q_bytes),
             ("lse", replicas * touched_rows * 4), ("dsum", replicas * rounded_touched * 4),
             ("dq_accum", accum_bytes))
    gqa = spec.q_heads != spec.kv_heads
    writes = (("dq_accum", accum_bytes),
              ("dk_expanded" if gqa else "dk", expanded),
              ("dv_expanded" if gqa else "dv", expanded))

    l2read, l2write = ldgsts + lse_load + dsum_load, 2 * expanded + atomic
    vec = _vectors(fadd=2 * m * n * pairs,
                   fmul=(2 * m * n + 4 * b * m) * pairs + ctas * n * d)
    if spec.source != "dao":
        vec["fadd"] -= m * n * pairs
        vec["fmul"] -= m * n * pairs
        vec["ffma"] = m * n * pairs
    smem = 2 * (3 * m * d + 2 * n * d + 2 * m * n)
    resident, resources = (None, {}) if describe_only else _occupancy(spec, rates, "bwd_dq_dk_dv", smem)
    per_tile_vec = (4 * m * n + 4 * b * m if spec.source == "dao"
                    else 3 * m * n + 4 * b * m)
    spill_program = None
    if spec.spills and not describe_only:
        from flash_attention_spills import select_program
        spill_program = select_program(spec, "bwd_dq_dk_dv")

    def program(cid, node, shared, read, write, vector, sfu, gemm, complete, spill):
        nb = cid % grid[0]
        lo, valid_n = minima[nb], min(n, spec.kv_len - nb * n)
        prologue = spill("prologue")
        k = read(valid_n * d * 2, prologue, staging=True)
        v = read(valid_n * d * 2, prologue, staging=True)
        kvzero = shared(2 * (n - valid_n) * d * 2)
        first_rows = min(m, spec.q_len - (nq - 1) * m)
        q = read(first_rows * d * 2, staging=True)
        do = read(first_rows * d * 2, staging=True)
        qzero = shared(2 * (m - first_rows) * d * 2)
        bidb = (cid // grid[0]) % spec.batch
        bidh = cid // (grid[0] * spec.batch)
        lse_offset = (bidb * spec.q_heads + bidh) * spec.q_len
        first_lse = read(_scalar_row_read_bytes(first_rows, b, lse_offset))
        atoms = []

        def epilogue(deps):
            deps = spill("epilogue", deps)
            scale = [vector(n * d / 8, w, deps) for w in range(8)]
            stage = shared(2 * n * d * 2, scale)
            loaded = shared(2 * n * d * 2, (stage,))
            out = write(2 * valid_n * d * 2, (loaded,))
            node(tuple(atoms) + (out,), callback=complete)

        def tile(mb, previous, ready_q, ready_do, ready_lse):
            begin = node(spill("tile", tuple(previous) + (ready_q, ready_do, k, v, kvzero, qzero),
                               nq - 1 - mb))
            ds = read(_scalar_row_read_bytes(m, b), (begin,))

            s_ready, ds_done = [], []
            for w in range(8):
                qk = gemm(m, n, d, a, w, (begin,))
                score = vector((2 if spec.source == "dao" else 1) * m * n / 8
                               + (4 * b * m) / 8, w, (qk, ready_lse))
                exp = sfu(m * n / 8, w, (score,))
                pstore = shared(m * n * 2 / 8, (exp,))
                dp = gemm(m, n, d, a, w, (pstore,))
                dswork = vector(2 * m * n / 8, w, (dp, ds))
                ds_done.append(dswork)
                s_ready.append(shared(m * n * 2 / 8, (dswork,)))
            barrier = node(s_ready)

            next_q = read(m * d * 2, ds_done, staging=True) if mb > lo else None
            dv = [gemm(n, d, m, 4, w, (barrier,)) for w in range(8)]
            dvbar = node(dv)
            next_do = read(m * d * 2, (dvbar,), staging=True) if mb > lo else None
            dk, dqs = [], []
            for w in range(8):
                dq = gemm(m, d, n, a, w, (dvbar,))
                dqs.append(dq)
                atoms.append(write(m * d * 4 / 8, (dq,), atomic=True))
                dk.append(gemm(n, d, m, 4, w, (dq,)))
            if mb > lo:
                next_lse = read(_scalar_row_read_bytes(m, b, lse_offset), dqs)
                node(dk, callback=lambda done: tile(mb - 1, (done,), next_q, next_do, next_lse))
            else:
                node(dk, callback=lambda done: epilogue((done,)))

        tile(nq - 1, (), q, do, first_lse)

    time_s, timing = ((0.0, {}) if describe_only else _time(spec, rates, ctas, resident, program,
                           sum(value for name, value in reads if name != "dq_accum"), l2read,
                           atomic_unique_reads=accum_bytes, atomic_payload_bytes=atomic,
                           delivered_writes=l2write,
                           spill_program=spill_program, tiles=tiles,
                           native=dict(kind="main", m=m, n=n, d=d, a=a, b=b, q_len=spec.q_len,
                                       kv_len=spec.kv_len, grid0=grid[0], nq=nq, batch=spec.batch,
                                       q_heads=spec.q_heads, dao=spec.source == "dao", minima=minima,
                                       ff=spec.fast_forward)))
    kernel = FlashKernel("flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel", grid, ctas, 1,
                       issued, useful, vec, dict(ex2=pairs * m * n, rcp=0, lg2=0),
                       ldsm + ld, st, l2read, l2write, reads, writes, {}, time_s, {
                           "block_m": m, "block_n": n, "warps": 8,
                           "atom_layouts": {"sdp": [a, b], "dkv": [4, 2], "dq": [a, b]},
                           "smem_bytes_per_cta": smem,
                           "tiles_per_cta": tiles, "total_tiles": pairs,
                           "m_block_min_per_n": minima, "m_iteration_order": "descending",
                           "smem_ldsm": ldsm, "smem_ldgsts": ldgsts, "smem_ld": ld, "smem_st": st,
                           "smem_tail_clear": tail_clear,
                           "smem_gemm_operand_bytes_per_pair": dict(zip(("qk", "dp", "dv", "dq", "dk"), per_gemm)),
                           "scalar_row_replication": 4 * b,
                           "scalar_row_warp_copies": b,
                           "scalar_load_sector_bytes": 32,
                           "l2_lse_scalar_read_bytes": lse_load, "l2_dsum_scalar_read_bytes": dsum_load,
                           "dq_atomic_read_bytes": 0, "dq_atomic_write_bytes": atomic,
                           "dq_atomic_internal_read_bytes": atomic,
                           "dq_atomic_instruction": "RED.E.ADD.F32.FTZ.RN.STRONG.GPU; no returned old value",
                           "dq_atomic_sass_provenance": "sibling flash_attention_rebuild/checks/S3/atomic_audit/flash_s3_atomic_audit.json; flash_s3_atomic_d64_sm80.sass:4017-4289; flash_s3_atomic_d128_sm80.sass:2991-3245; selected causal/even-MN/even-K BF16 SM80 symbols",
                           "l2_byte_scope": "SM-L2 operands including coalesced scalar warp-sector requests and RED update payload; internal atomic old-value read excluded; no fitted transaction multiplier",
                           "tensor_io_scope": "unique touched tensor spans; scalar repetition and visited-pair repetition appear only in L2 demand",
                           "source": "D/src/flash_bwd_kernel.h:100-790; kernel_traits.h:175-345; flash_bwd_launch_template.h:160-256",
                           "phase_order": ["qk", "exp_p_store", "dp", "ds_store", "dv", "dq_atomic", "dk"],
                           "vector_instructions_per_pair": per_tile_vec,
                           "vector_count_scope": ("source scalar arithmetic; torch UNFUSE_FMA; BF16 conversions are not FP16 arithmetic"
                                                  if spec.source == "dao" else
                                                  "source scalar arithmetic; pip fused score lowering assumed per DESIGN; BF16 conversions are not FP16 arithmetic"),
                           "time_mode": "counts_only" if describe_only else "structural", **resources, **timing,
                       }, smem_staging_bytes=ldgsts)
    if spec.spills and not describe_only:
        from flash_attention_spills import apply_counts
        kernel = apply_counts(kernel, spill_program, tiles, 256)
    return kernel

def _gqa(spec, rates, tensor, describe_only=False):
    from flash_attention import FlashKernel
    callback = None if describe_only else rates.reduction_time_s
    if callback is None and not describe_only:
        raise ValueError("GQA backward requires FlashRates.reduction_time_s from the existing reduction roofline")
    groups = spec.q_heads // spec.kv_heads
    outputs = spec.batch * spec.kv_len * spec.kv_heads * spec.head_dim
    incoming, outgoing = outputs * groups * 2, outputs * 2
    adds = outputs * (groups - 1)

    time_s = 0.0 if describe_only else float(callback(adds, incoming + outgoing))
    if not math.isfinite(time_s) or time_s < 0:
        raise ValueError("FlashRates.reduction_time_s must return finite nonnegative seconds")
    return FlashKernel(f"gqa_sum_{tensor}", (1, 1, 1), 1, 1, 0, 0,
                       _vectors(fadd=adds), dict(ex2=0, rcp=0, lg2=0),
                       0, 0, incoming, outgoing,
                       ((f"{tensor}_expanded", incoming),), ((tensor, outgoing),), {}, time_s, {
                           "groups": groups, "reduction_outputs": outputs,
                           "grid_scope": "one logical at::sum_out record; CUDA reduction launch geometry is not modeled",
                           "source": "D/flash_api.cpp:897-967; P/flash_api.cpp:976-1036",
                           "time_mode": "counts_only" if describe_only else "existing vector/reduction roofline callback including launch overhead",
                           **({} if describe_only else {"time_source": "train_timing._price_kernels / vector_roofline(flops,moved,name='',mem_level=num_levels-1)+O"}),
                           "smem_ldsm": 0, "smem_ldgsts": 0, "smem_ld": 0, "smem_st": 0,
                       })

def backward_kernels(spec, rates, describe_only=False):
    if spec.source not in ("dao", "dao-pip-unknown"):
        raise NotImplementedError("FlashAttention backward is implemented for the Dao source only")
    if spec.kv_layout != "contiguous":
        raise NotImplementedError("FlashAttention backward requires contiguous Q/K/V")
    if spec.dtype != "bf16":
        raise NotImplementedError("FlashAttention backward is BF16 only")

    if not describe_only and rates.shared_memory_bytes_per_sm < 144 * 1024:
        raise NotImplementedError("FlashAttention backward implements the A100 >=144 KiB traits only")
    if not describe_only and spec.q_heads != spec.kv_heads and rates.reduction_time_s is None:
        raise ValueError("GQA backward requires FlashRates.reduction_time_s from the existing reduction roofline")
    result = [_simple(spec, rates, False, describe_only), _main(spec, rates, describe_only),
              _simple(spec, rates, True, describe_only)]
    if spec.q_heads != spec.kv_heads:
        result.extend((_gqa(spec, rates, "dk", describe_only), _gqa(spec, rates, "dv", describe_only)))
    return result
