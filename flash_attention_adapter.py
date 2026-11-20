from dataclasses import replace
from activity import MemoryTraffic, OperationActivity
from flash_attention_backward import atomic_rmw_read_bytes
from opmodel import settings

def source_notes(source):
    if source != "dao-pip-unknown":
        return {}
    return {"source_assumption": "Dao 979702c8 kernels/traits; fused pip vector lowering; NGC build unverified",
            "compiled_resource_proxy": "torch 2.10 BF16 sm80, provisional",
            "backward_assumption": "non-deterministic default, including Megatron"}

def flash_rates(timing, overhead):
    from flash_attention import FlashRates
    rate_kwargs = dict(
        sfu_ops_per_cycle=settings.get("RAPID_FLASH_SFU_OPS_PER_CYCLE"),
        registers_per_sm=settings.get("RAPID_FLASH_REGISTERS_PER_SM"),
        register_allocation_per_thread=settings.get("RAPID_FLASH_REGISTER_ALLOCATION_PER_THREAD"),
    )
    if getattr(timing.hw_config.sw_config, "gemm_timing_backend", "extended_roofline") == "opmodel":
        backend = timing._opmodel_backend or timing._gemm_activity_backend
        if backend is None:
            raise ValueError("FlashAttention requires the op-model hardware description")
        return FlashRates.from_hw(backend.hardware, overhead,
                                  sfu_ops_per_cycle=rate_kwargs["sfu_ops_per_cycle"],
                                  register_allocation_per_thread=rate_kwargs["register_allocation_per_thread"])
    from extended_timing import flash_rates_from_hw
    return flash_rates_from_hw(timing.hw_config, overhead, **rate_kwargs)

def _run(timing, batch, hidden, q_len, kv_len, q_heads, kv_heads, sm_count, *,
         decode=False, op="fwd", explicit_all_gather=False, return_kernels=False, describe_only=False,
         head_dim=None):
    from flash_attention import FlashSpec, flash_kernels

    if not settings.get("RAPID_FLASH_REBUILD"):
        raise ValueError("flash_rebuild: false is obsolete; use the FlashAttention rebuild")
    if int(timing.cp) != 1 and not explicit_all_gather:
        raise NotImplementedError("FlashAttention rebuild CP>1 requires explicit all-gather q_len/kv_len; ring/P2P is unsupported")
    for name in ("RAPID_FA_TILE_UTIL_FWD", "RAPID_FA_TILE_UTIL_BWD", "RAPID_FA_PREFILL_SCALE"):
        if settings.origin(name) != "default":
            raise ValueError(f"{name} is obsolete with RAPID_FLASH_REBUILD")
    tp = max(1, int(timing.tp))

    replicated_kv = (int(kv_heads) % tp != 0 and tp % int(kv_heads) == 0
                     and str(getattr(timing, "model_type", "")).lower() == "qwen3_moe"
                     and str(timing.run_type).lower() == "inference"
                     and timing.implementation_options["flash_source"] == "vllm")
    if int(q_heads) % tp or (int(kv_heads) % tp and not replicated_kv):
        raise NotImplementedError("FlashAttention rebuild requires whole local query and KV heads")
    if float(timing.precision.activations) != 2.0:
        raise NotImplementedError("FlashAttention rebuild supports 16-bit activations only")
    d = int(head_dim) if head_dim is not None else int(getattr(timing, "head_dim", None) or hidden // q_heads)
    local_q, local_kv = int(q_heads) // tp, (1 if replicated_kv else int(kv_heads) // tp)
    source = timing.implementation_options["flash_source"]
    if source == "vllm" and str(timing.run_type).lower() == "training":
        raise NotImplementedError("FlashAttention S1 training source is torch/Dao")
    counts_only = bool(getattr(timing.hw_config, "counts_only", False))
    if counts_only and op == "bwd":
        core = timing.hw_config.tech_config.core
        if (getattr(core, "shared_memory_bytes_per_block", 0) or 0) < 144 * 1024:
            raise NotImplementedError("FlashAttention backward requires the >=144 KiB shared-memory launch branch")
    if counts_only:
        rates = None
    else:
        overhead = (timing.decode_launch_overhead(kv_len) if decode and source == "vllm"
                    and hasattr(timing, "decode_launch_overhead") else timing.O)
        rates = flash_rates(timing, overhead)
    if op == "bwd" and not counts_only:
        rates = replace(rates, reduction_time_s=lambda flops, moved:
                        timing.vector_roofline(flops, moved, name="", mem_level=timing.num_levels - 1) + timing.O)
    block_size = block_table_width = 0
    kv_layout = "contiguous"
    if decode and source == "vllm":
        from kernel_trace_vllm import VllmEngine, decode_uses_cuda_graph
        eng = VllmEngine(sm_count=int(sm_count), **dict(getattr(timing, "engine_settings", None) or {}))
        block_size = eng.block_size

        seq_to_capture = min(8192, int(eng.max_model_len))
        graph = decode_uses_cuda_graph(kv_len, eng)
        width_len = seq_to_capture if graph else int(kv_len)
        block_table_width = (width_len + block_size - 1) // block_size
        kv_layout = "paged"

    fast_forward = None if counts_only or not settings.get("RAPID_FLASH_FAST_FORWARD") else (
        float(settings.get("RAPID_FLASH_FAST_FORWARD_RTOL")), int(settings.get("RAPID_FLASH_FAST_FORWARD_PERIODS")))
    forward_fast_forward = fast_forward if fast_forward and settings.get("RAPID_FLASH_FAST_FORWARD_FWD") else None
    spec = FlashSpec(q_heads=local_q, kv_heads=local_kv, head_dim=d, batch=int(batch),
                     q_len=int(q_len), kv_len=int(kv_len), causal=str(timing.mode).upper() == "LLM",
                     op=op, source=source, kv_layout=kv_layout, block_size=block_size,
                     block_table_width=block_table_width, dtype="bf16", sm_count=int(sm_count), rates=rates,
                     spills=False if counts_only else bool(settings.get("RAPID_FLASH_SPILLS")),
                     fast_forward=fast_forward if op == "bwd" else forward_fast_forward)
    if describe_only and not return_kernels:
        raise ValueError("describe_only requires return_kernels; it is not a timing estimate")
    if (op == "fwd" and not (describe_only or counts_only) and str(timing.run_type).lower() == "training"
            and source in ("dao", "dao-pip-unknown") and not decode):
        from flash_attention_backward import prefetch_main
        prefetch_main(replace(spec, op="bwd", fast_forward=fast_forward), spec.rates)
    kernels = flash_kernels(spec, describe_only=True) if describe_only or counts_only else flash_kernels(spec)
    if decode:
        timing._decode_flash_launch_count = len(kernels)
    if return_kernels:
        return spec, kernels

    qo = float(batch) * int(q_len) * local_q * d * timing.precision.activations
    kv = float(batch) * int(kv_len) * 2 * local_kv * d * timing.precision.activations
    activity = OperationActivity(hbm=MemoryTraffic(read_bytes=qo + kv, write_bytes=qo)) if op == "fwd" else OperationActivity()

    staging_energy = bool(settings.get("OPMODEL_SMEM_STAGING"))
    l2_bypass = bool(getattr(timing, "l2_bypass", False))
    for kernel in kernels:
        v = kernel.vector_ops
        vector = sum(v.get(x, 0) for x in ("fadd", "fmul", "hadd", "hmul")) + 2 * sum(
            v.get(x, 0) for x in ("ffma", "hfma"))
        activity += OperationActivity(useful_tensor_flops=kernel.useful_tensor_flops,
                                      issued_tensor_flops=kernel.issued_tensor_flops,
                                      vector_flops=vector,

                                      smem=MemoryTraffic(kernel.smem_read_bytes + kernel.diagnostics.get("l1_spill_read_bytes", 0),
                                                         kernel.smem_write_bytes + kernel.diagnostics.get("l1_spill_write_bytes", 0)),
                                      l2=MemoryTraffic(kernel.l2_read_bytes, kernel.l2_write_bytes),
                                      smem_staging_bytes=kernel.smem_staging_bytes,
                                      smem_staging_energy_bytes=kernel.smem_staging_bytes if staging_energy else 0,
                                      **({"atomic_rmw_read_bytes": atomic_rmw_read_bytes(kernel)} if l2_bypass else {}))
    floor = timing.kernel_min_seconds() if hasattr(timing, "kernel_min_seconds") else None
    return sum(k.time_s if floor is None else max(k.time_s, floor) for k in kernels), activity, source

def _lengths(seq, q_len, kv_len):
    if (q_len is None) != (kv_len is None):
        raise ValueError("all-gather attention requires both explicit q_len and kv_len")
    return (seq, seq, False) if q_len is None else (q_len, kv_len, True)

def forward(timing, batch, hidden, seq, q_heads, kv_heads, sm_count, *, q_len=None, kv_len=None):
    q, k, explicit = _lengths(seq, q_len, kv_len)
    return _run(timing, batch, hidden, q, k, q_heads, kv_heads, sm_count, explicit_all_gather=explicit)

def backward(timing, batch, hidden, seq, q_heads, kv_heads, sm_count, *, q_len=None, kv_len=None):
    q, k, explicit = _lengths(seq, q_len, kv_len)
    return _run(timing, batch, hidden, q, k, q_heads, kv_heads, sm_count,
                op="bwd", explicit_all_gather=explicit)

def training_all_gather(timing, batch, hidden, seq, q_heads, kv_heads, sm_count, *, op="fwd"):
    from kernel_trace_megatron import cp_query_chunks

    call = {"fwd": forward, "bwd": backward}[op]
    ranks = []
    for rank in range(int(timing.cp)):
        seconds, activity, source = 0.0, OperationActivity(), None
        for q_len, kv_len in cp_query_chunks(seq, int(timing.cp), rank):
            elapsed, chunk_activity, source = call(
                timing, batch, hidden, seq, q_heads, kv_heads, sm_count,
                q_len=q_len, kv_len=kv_len)
            seconds += elapsed
            activity += chunk_activity
        ranks.append((seconds, activity, source))

    return max(ranks, key=lambda result: result[0])

def mla_prefill(timing, batch, seq, heads, head_dim, sm_count, *, return_kernels=False):
    return _run(timing, batch, 0, seq, seq, heads, heads, sm_count, head_dim=head_dim, return_kernels=return_kernels)

def decode(timing, batch, hidden, kv_len, q_heads, kv_heads, sm_count):
    return _run(timing, batch, hidden, 1, kv_len, q_heads, kv_heads, sm_count, decode=True)
