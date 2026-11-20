import re
from dataclasses import fields, is_dataclass, replace

from activity import MemoryTraffic, OperationActivity, ZERO_ACTIVITY
from flash_attention_backward import atomic_rmw_read_bytes
from opmodel import settings

MOE_ATOMIC_SCATTERS = (".moe.unpermute.scatter_add", ".moe.permute.index_select_bwd.index_add")

def moe_atomic_bypass(kernel):
    if not kernel.name.endswith(MOE_ATOMIC_SCATTERS):
        return kernel, 0.0
    payload, (dest,) = kernel.reads[0].nbytes, kernel.writes
    if not payload >= dest.nbytes:
        raise ValueError(f"{kernel.name}: atomic source {payload} B is smaller than its destination {dest.nbytes} B")
    extra = float(payload - dest.nbytes)
    return replace(kernel, writes=(replace(dest, nbytes=payload),), l2_reread=kernel.l2_reread + extra), extra

def selected(model):
    return (str(getattr(model, "run_type", "")).lower() == "training"
            and getattr(model, "implementation", None) == "fused")

def enabled(timing):
    return selected(timing) and bool(settings.get("RAPID_L2_RESIDENCY"))

def kernel_timing(timing):
    return enabled(timing) and timing.implementation_options.get("megatron_timing", "operator_path") == "kernel_trace"

def multi_gpu(timing):
    from program.axes import CANONICAL_AXES
    return any(int(getattr(timing, degree, 1) or 1) != 1 for degree in CANONICAL_AXES)

def flattened(timing):
    mode = getattr(timing, "execution_mode", None)
    return getattr(mode, "value", mode) == "full_astrasim_flattened"

def dp_grad_sync(timing):
    if not enabled(timing) or (int(getattr(timing, "dp", 1) or 1) <= 1 and not moe_expert_parallel(timing)):
        return "legacy"
    return "megatron_" + timing.implementation_options.get("megatron_dp_grad_sync", "end_of_backward")

def moe_expert_parallel(timing):
    return (enabled(timing) and bool(getattr(timing, "use_moe", False)) and int(getattr(timing, "ep", 1) or 1) > 1
            and timing.implementation_options.get("megatron_timing", "operator_path") == "kernel_trace")

def data_parallel_groups(timing):
    dp = int(getattr(timing, "dp", 1) or 1)
    ep = int(getattr(timing, "ep", 1) or 1) if bool(getattr(timing, "use_moe", False)) else 1
    return dp * ep, dp

def distributed_optimizer(timing):
    return dp_grad_sync(timing) != "legacy" and int(getattr(timing, "zero_stage", 0) or 0) == 1

def static_bytes_per_param(timing):
    if not enabled(timing):
        return None
    dp = int(getattr(timing, "dp", 1) or 1)
    zero = int(getattr(timing, "zero_stage", 0) or 0)
    if dp > 1 and zero not in (0, 1):
        raise ValueError("Megatron dp>1 describes dp_zero_stage 0 (DDP) and 1 (distributed optimizer) only")
    p = timing.precision
    shards = dp if distributed_optimizer(timing) else 1
    return (float(p.parameters), float(p.gradients),
            (float(p.master_parameters) + 2.0 * float(p.optimizer_states)) / shards)

def static_bytes_per_param_groups(timing):
    if not enabled(timing) or not bool(getattr(timing, "use_moe", False)):
        return None
    p = timing.precision
    state = float(p.master_parameters) + 2.0 * float(p.optimizer_states)
    dense, expert = data_parallel_groups(timing) if distributed_optimizer(timing) else (1, 1)
    return float(p.parameters), float(p.gradients), state / dense, state / expert

def serial_time(program):
    from program.ir import ComputeOp
    return sum(float(op.duration[0]) for op in program.ops if isinstance(op, ComputeOp))

def mla_trace(timing):
    return (str(getattr(timing, "attention_type", "mha")).lower() == "mla"
            and timing.implementation_options.get("mla_training", "megatron_trace") == "megatron_trace")

SELECTIVE_DESCRIBED = ("layernorm", "mla_up_proj")

def recompute_modules(timing):
    opts = getattr(timing, "implementation_options", None) or {}
    if opts.get("megatron_recompute_granularity", "none") != "selective":
        return ()
    return tuple(str(opts.get("megatron_recompute_modules", "core_attn")).split(","))

def recompute_layernorm(timing, layer):
    if "layernorm" not in recompute_modules(timing) or layer is None:
        return False, False
    mla = str(getattr(timing, "attention_type", "mha")).lower() == "mla"
    moe = bool(getattr(timing, "use_moe", False)) and timing.is_moe_layer(int(layer))
    return mla, moe

def selective_memory(timing, layernorm_bytes, mla_up_bytes):
    from memory_estimation import MemKind
    mods = recompute_modules(timing)
    if not mods:
        return None
    if not enabled(timing):
        raise ValueError("megatron_recompute_granularity=selective requires the Megatron training description "
                         "(implementation fused, training, hardware.l2_model step simulation)")
    mla = str(getattr(timing, "attention_type", "mha")).lower() == "mla"
    out = {"dense": {}, "moe": {}}
    if "layernorm" in mods:
        if mla:
            out["dense"][MemKind.LAYERNORM1] = out["moe"][MemKind.LAYERNORM1] = float(layernorm_bytes)
        if bool(getattr(timing, "use_moe", False)):
            out["moe"][MemKind.LAYERNORM2] = float(layernorm_bytes)
    if "mla_up_proj" in mods:
        out["dense"][MemKind.QKV_PROJ] = out["moe"][MemKind.QKV_PROJ] = float(mla_up_bytes)
    return out

def selective_remat(timing, freed):
    from memory_estimation import MemKind
    hosts = {MemKind.LAYERNORM1: (MemKind.OUTPUT_PROJ,),
             MemKind.LAYERNORM2: (MemKind.MLP, MemKind.MOE_DISPATCH, MemKind.MOE_COMBINE, MemKind.ROUTER),
             MemKind.QKV_PROJ: (MemKind.ATTENTION,)}
    return [(layer_type, hosts[kind], kind, 2.0 * nbytes)
            for layer_type, kinds in freed.items() for kind, nbytes in kinds.items() if nbytes]

def placement(timing):
    from program.placement import LEGACY_PLACEMENT, PlacementPolicy, WorkSpread
    from program.work import WorkKind
    if not enabled(timing):
        return LEGACY_PLACEMENT

    return PlacementPolicy("megatron_per_rank", {kind: WorkSpread.PER_CLUSTER_RANK for kind in WorkKind})

def validate(timing):
    if not enabled(timing):
        return
    if bool(getattr(timing, "tp_sp", False)):
        raise ValueError("Megatron kernel trace does not describe sequence parallelism")
    if int(getattr(timing, "pipeline_interleave", 1)) != 1:
        raise ValueError("Megatron kernel trace does not describe interleaved/virtual pipeline stages")
    if int(getattr(timing, "ep", 1)) != 1 and not bool(getattr(timing, "use_moe", False)):
        raise ValueError("Megatron kernel trace MoE/ep: ep > 1 requires MoE layers")
    opts_early = timing.implementation_options
    if bool(getattr(timing, "use_moe", False)):
        if int(timing.tp) != 1 or int(timing.cp) != 1:
            raise ValueError("Megatron MoE/ep trace describes tp=1, cp=1 (TP needs sequence parallelism: moe_layer.py:129-138)")
        if int(timing.dp) != 1 and opts_early.get("megatron_timing", "operator_path") != "kernel_trace":
            raise ValueError("Megatron MoE with dp > 1 is described with megatron_timing=kernel_trace only")
        if int(getattr(timing, "n_shared_experts", 0) or 0) < 0:
            raise ValueError("n_shared_experts must be >= 0")
        if int(timing.moe_num_experts) % int(timing.ep):
            raise ValueError("Megatron MoE trace requires num_experts divisible by ep")
    opts = timing.implementation_options
    if int(opts.get("megatron_grad_bucket_params", 0) or 0) > 0:
        raise ValueError("megatron_grad_bucket_params > 0 is not described: MegatronDP per_work_item reduces one "
                         "bucket per work item")
    if dp_grad_sync(timing) != "legacy":
        if opts.get("megatron_timing", "operator_path") != "kernel_trace":
            raise ValueError("Megatron dp>1 is described with megatron_timing=kernel_trace only")
        if (opts.get("megatron_dp_buckets", "per_reducer") == "one"
                and opts.get("megatron_dp_grad_sync", "end_of_backward") != "end_of_backward"):
            raise ValueError("megatron_dp_buckets=one (disable_bucketing) requires megatron_dp_grad_sync=end_of_backward")
        if int(getattr(timing, "zero_stage", 0) or 0) not in (0, 1):
            raise ValueError("Megatron dp>1 describes dp_zero_stage 0 (DDP all-reduce) and 1 (distributed "
                             "optimizer); ZeRO-2/3 and Megatron custom FSDP are not described")
        if str(getattr(timing, "dp_microbatch", "every_mb")) != "last_mb":
            raise ValueError("Megatron dp>1 requires sw_param.dp_microbatch=last_mb")
        if bool(getattr(timing.model, "gradient_clipping", False)):
            raise ValueError("Megatron dp>1 does not describe gradient clipping (the grad-norm all-reduce)")
        if (opts.get("megatron_zero_grad") != "grad_buffer" or opts.get("megatron_optimizer") != "te_fused_adam"
                or opts.get("megatron_tied_grad") != "autograd_sum"):
            raise ValueError("Megatron dp>1 describes megatron_zero_grad=grad_buffer, megatron_optimizer=te_fused_adam "
                             "and megatron_tied_grad=autograd_sum (the Megatron-Core 0.12.1 preset)")
    mla = mla_trace(timing)
    if opts.get("megatron_recompute_granularity", "none") == "selective":
        mods = recompute_modules(timing)
        missing = [m for m in mods if m not in SELECTIVE_DESCRIBED]
        if missing:
            raise NotImplementedError(f"Megatron selective recompute of {missing} is not described "
                                      f"(described: {list(SELECTIVE_DESCRIBED)})")
        if bool(getattr(timing, "full_recomputation", False)):
            raise ValueError("megatron_recompute_granularity=selective excludes sw_param.full_recomputation "
                             "(transformer_config.py:684-688: one granularity)")
        if "mla_up_proj" in mods and not mla:
            raise ValueError("mla_up_proj in megatron_recompute_modules requires the Megatron MLA description")
    if str(getattr(timing, "attention_type", "mha")).lower() == "mla" and not mla:
        raise ValueError("Megatron kernel trace with MLA requires implementation.options.mla_training=megatron_trace")
    if str(timing.model_type) not in ("llama", "qwen2", "qwen3") and not (mla and str(timing.model_type) == "deepseek_v3"):
        raise ValueError("Megatron kernel trace describes only llama/qwen2/qwen3 (and deepseek_v3 MLA layers)")
    if mla:
        if int(getattr(timing, "tp", 1)) != 1 or int(getattr(timing, "cp", 1)) != 1:
            raise ValueError("the Megatron MLA description is tp = cp = 1; its TP/CP shapes are not described "
                             "(paper_audit/mla/MEGATRON_PPDP_REPORT.md)")
    p = timing.precision
    if any(float(x) == 1 for x in (p.tensor_format, p.parameters)):
        raise ValueError("Megatron kernel trace does not describe fp8")
    if (float(p.tensor_format) != 2 or float(p.parameters) != 2
            or any(float(x) != 4 for x in (p.master_parameters, p.gradients, p.optimizer_states, p.stats))):
        raise ValueError("Megatron kernel trace requires 16-bit tensor/parameters and fp32 master_parameters, gradients, optimizer_states and stats")
    opts = timing.implementation_options
    if opts.get("gradient_accumulation_fusion", False):
        raise ValueError("Megatron kernel trace does not describe gradient_accumulation_fusion=true fused wgrad accumulation")
    if opts.get("megatron_param_copy", "separate_copy") != "separate_copy":
        raise ValueError("Megatron kernel trace only describes megatron_param_copy=separate_copy")
    if opts["tp_allreduce"] != "nccl" or opts["vocab_parallel_head_gather"]:
        raise ValueError("Megatron kernel trace requires tp_allreduce=nccl and vocab_parallel_head_gather=false")
    if opts["flash_source"] not in ("dao", "dao-pip-unknown"):
        raise ValueError("Megatron training requires flash_source=dao | dao-pip-unknown")
    if not bool(getattr(timing, "flash_attention", False)) and not mla:
        raise ValueError("Megatron kernel trace requires FlashAttention")
    mode = getattr(getattr(timing, "execution_mode", None), "value", getattr(timing, "execution_mode", None))
    if (opts.get("megatron_timing", "operator_path") == "kernel_trace" and not multi_gpu(timing)
            and mode == "full_astrasim_hierarchical"):
        raise ValueError("megatron_timing=kernel_trace is timed by the analytical (serial sum) or "
                         "full_astrasim_flattened mode, not full_astrasim_hierarchical")
    if opts.get("megatron_timing", "operator_path") == "kernel_trace" and multi_gpu(timing):
        if not flattened(timing):
            raise ValueError("megatron_timing=kernel_trace on more than one GPU is timed on the FLAT program: "
                             "set execution_backend.astrasim.mode=full_astrasim_flattened")
        if int(timing.cp) != 1:
            raise ValueError("megatron_timing=kernel_trace on more than one GPU describes cp=1 only")
        from sim_settings import nccl_latency_rule
        if int(timing.tp) > 1 and not (
                float(getattr(timing.hw_config.sw_config, "collective_launch_overhead", 0.0) or 0.0) > 0
                or nccl_latency_rule(timing.hw_config) in ("tuning_formula", "tuning_protocol")):
            raise ValueError("megatron_timing=kernel_trace with tp>1 needs recorded endpoint collectives: "
                             "sw_param.collective_launch_overhead > 0 or a tuning nccl_latency_rule")
    if int(timing.tp) > 1 and settings.get("RAPID_L2_COLLECTIVE_BUFFERS") not in ("occupy_l2", "leave_l2"):
        raise ValueError("Megatron tp>1 requires hardware.l2_model.collective_buffers=occupy_l2 | leave_l2")

def _flash_activity(record, l2_bypass=False):
    v = record.vector_ops
    vector = sum(v.get(x, 0) for x in ("fadd", "fmul", "hadd", "hmul")) + 2 * sum(
        v.get(x, 0) for x in ("ffma", "hfma"))
    return OperationActivity(
        useful_tensor_flops=record.useful_tensor_flops,
        issued_tensor_flops=record.issued_tensor_flops, vector_flops=vector,
        smem=MemoryTraffic(record.smem_read_bytes, record.smem_write_bytes),
        l2=MemoryTraffic(record.l2_read_bytes, record.l2_write_bytes),
        smem_staging_bytes=record.smem_staging_bytes,
        smem_staging_energy_bytes=record.smem_staging_bytes if settings.get("OPMODEL_SMEM_STAGING") else 0,

        **({"atomic_rmw_read_bytes": atomic_rmw_read_bytes(record)} if l2_bypass else {}))

def _exact_key(value):
    if is_dataclass(value):
        return (type(value), tuple((f.name, _exact_key(getattr(value, f.name)))
                                   for f in fields(value)))
    if isinstance(value, dict):
        return tuple((_exact_key(k), _exact_key(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(map(_exact_key, value))
    return value

def _replay_key(trace, capacity, kwargs):
    kernels = _expert_canonical(trace.kernels) if trace.rules.alloc_before is None else trace.kernels
    return kernels, _exact_key(trace.rules), capacity, _exact_key(kwargs)

_EXPERT_NUMBER = re.compile(r"(expert|\.e)(\d+)(?=\.)")
_STRING_ACCESSES = ("reads", "writes", "original_reads", "original_writes", "epilogue_reads")

def _expert_canonical(kernels):
    ids = {}

    def number(match):
        return match.group(1) + str(ids.setdefault(int(match.group(2)), len(ids)))

    def rename(text):
        return _EXPERT_NUMBER.sub(number, text) if ("expert" in text or ".e" in text) else text
    out = []
    for kernel in kernels:
        if not is_dataclass(kernel):
            out.append(kernel)
            continue
        changes = {}
        for f in fields(kernel):
            value = getattr(kernel, f.name)
            if isinstance(value, str):
                renamed = rename(value)
                if renamed != value:
                    changes[f.name] = renamed
            elif f.name in _STRING_ACCESSES and value:
                renamed = tuple(replace(a, tensor=rename(a.tensor)) if ("expert" in a.tensor or ".e" in a.tensor)
                                else a for a in value)
                if any(x is not y for x, y in zip(renamed, value)):
                    changes[f.name] = renamed
        out.append(replace(kernel, **changes) if changes else kernel)
    return tuple(out)

def _streaming_price(timing, kernel):
    context = (timing.O, getattr(timing, "_decode_graph_pricing", False),
               getattr(timing, "stream_launch_overhead_s", None),
               timing.core.num_bundle, timing.core.get_throughput("vector"),
               tuple((m.get_throughput(), m.get_latency()) for m in timing.mem_layer),
               timing.num_levels, timing.precision.activations,
               getattr(timing, "head_dim", None))
    key = (kernel.kind, kernel.name, kernel.flops,
           tuple(a.nbytes for a in kernel.reads), tuple(a.nbytes for a in kernel.writes),
           getattr(kernel, "l2_reread", 0.0), context)
    cache = timing.__dict__.setdefault("_megatron_streaming_times", {})
    if key not in cache:
        cache[key] = timing._price_kernels([kernel], record=False)
    return cache[key]

_BGEMM_NAME = "__megatron_attention_bgemm__"

def _batched_gemm(timing, shape, batch):
    cache = timing.__dict__.setdefault("_megatron_bgemm_cache", {})
    key = (tuple(shape), int(batch), timing.O)
    if key not in cache:
        name = _BGEMM_NAME
        old = timing._gemm_activity_by_name.pop(name, None)
        seconds = timing.get_gemm_time(*shape, name, batch=int(batch))[0]
        activity = timing._gemm_activity_by_name.pop(name, ZERO_ACTIVITY)
        if old is not None:
            timing._gemm_activity_by_name[name] = old
        cache[key] = seconds, activity
    return cache[key]

def _cp_copy(kernel):
    return kernel.op == 'attention' and kernel.name in (
        'cp_reorder', 'cp_grad_reorder', 'backward.cp_zero',
        'backward.cp_dq_copy', 'backward.cp_chunk_accum', 'forward.cp_output_copy')

def apply(program, timing, *, _group_replays=True):
    import kernel_trace_megatron
    import l2_sim
    import sim_settings
    from program.ir import ComputeOp
    from program.vtp import _gemm, _GEMM_NAME as _VTP_GEMM_NAME

    validate(timing)
    traces = kernel_trace_megatron.build(program, timing)
    spec = sim_settings.CacheSpec.from_hw(timing.hw_config)
    policy = sim_settings.SimPolicy.current(spec)
    capacity = spec.shared_capacity_bytes if policy.sharing_classes else int(timing._l2_capacity_bytes())
    kwargs = {"gemm_min_tile_m": policy.gemm_min_tile_m, "steady_state": policy.steady_state, "address_plan": policy.address_plan}
    if policy.sharing_classes:
        kwargs.update(sharing_classes=True, physical_capacity_bytes=spec.physical_bytes,
                      private_copies=policy.private_copies)
    kernel_trace_timing = kernel_timing(timing)
    l2_bypass = bool(getattr(timing, "l2_bypass", False))
    summaries = {}
    replays = {}
    for op in program.ops:
        if isinstance(op, ComputeOp):
            op.activity = ZERO_ACTIVITY
            op.invocation_activity = ZERO_ACTIVITY
    for trace in traces.values():
        for uid, size in trace.dp_payload.items():
            program.ops[uid].size_bytes = size
    if timing.implementation_options.get("megatron_dp_buckets", "per_reducer") == "one" and dp_grad_sync(timing) != "legacy":
        program.meta.misc["megatron_dp_buckets"] = "one"
    for rank, trace in traces.items():
        key = _replay_key(trace, capacity, kwargs) if _group_replays else rank
        if key not in replays and l2_bypass:
            replays[key] = None, []
        if key not in replays:
            passes = timing._trace_gemm_passes(trace.kernels, name=_VTP_GEMM_NAME, kinds=("gemm",))
            passes.update(timing._trace_gemm_passes(trace.kernels, name=_BGEMM_NAME, kinds=("batched_gemm",)))
            result = l2_sim.simulate(trace.kernels, capacity, trace.rules, operand_passes=passes, **kwargs)
            replays[key] = result, []
        result, price_variants = replays[key]

        priced = next((p for flash, specs, p in price_variants
                       if flash == trace.flash and specs == trace.flash_specs), None)
        if priced is None:
            priced = []
            for index, kernel in enumerate(trace.kernels):
                if kernel.kind == "collective":
                    continue
                if index in trace.flash:
                    activity = _flash_activity(trace.flash[index], l2_bypass)
                    seconds = float(trace.flash[index].time_s)
                elif kernel.kind == "gemm":
                    seconds, activity = _gemm(timing, kernel.gemm)
                elif kernel.kind == "batched_gemm":
                    seconds, activity = _batched_gemm(timing, kernel.gemm, kernel.batch)
                elif l2_bypass and kernel.name.endswith(MOE_ATOMIC_SCATTERS):
                    atomic, extra = moe_atomic_bypass(kernel)
                    seconds, activity = _streaming_price(timing, atomic)
                    activity = replace(activity, l2=MemoryTraffic(activity.l2.read_bytes - extra, activity.l2.write_bytes),
                                       atomic_rmw_read_bytes=activity.atomic_rmw_read_bytes + extra)
                else:
                    seconds, activity = _streaming_price(timing, kernel)

                activity = replace(activity, hbm=activity.l2_bypassed().hbm if l2_bypass else MemoryTraffic(
                    result.hbm_read[index], result.hbm_write[index]))
                copy_time = 0.0
                if _cp_copy(kernel):
                    copy_time = _streaming_price(timing, kernel)[0]
                priced.append((index, activity, copy_time, seconds))
            price_variants.append((trace.flash, trace.flash_specs, priced))
        activities = {}
        copy_times = {}
        kernel_times = {}
        for index, activity, copy_time, seconds in priced:
            source = trace.sources[index]
            activities[source.uid] = activities.get(source.uid, ZERO_ACTIVITY) + activity
            copy_times[source.uid] = copy_times.get(source.uid, 0.) + copy_time
            kernel_times[source.uid] = kernel_times.get(source.uid, 0.) + seconds
        for uid, activity in activities.items():
            op = program.ops[uid]
            if not isinstance(op, ComputeOp):
                raise ValueError(f"Megatron kernel owner {uid} is not a compute operation")
            op.activity = activity
            op.invocation_activity = activity
            if kernel_trace_timing:
                seconds = kernel_times[uid] + sum(float(call.duration_seconds) for call in op.embedded_collectives
                                                  if int(call.participants) > 1)
                op.duration = tuple(seconds for _ in op.duration)
                if op.invocation_duration is not None:
                    op.invocation_duration = seconds
                continue
            delta = copy_times.get(uid, 0.)
            if delta:
                op.duration = tuple(value + delta for value in op.duration)
                if op.invocation_duration is not None:
                    op.invocation_duration += delta
        if kernel_trace_timing:
            for op in program.ops:
                if isinstance(op, ComputeOp) and op.device == rank and op.uid not in activities:
                    op.duration = tuple(0.0 for _ in op.duration)
                    if op.invocation_duration is not None:
                        op.invocation_duration = 0.0
        if l2_bypass:
            hbm = sum((activity.hbm for _, activity, _, _ in priced), MemoryTraffic())
            summaries[str(rank)] = {"kernels": len(trace.kernels), "hbm_read_bytes": hbm.read_bytes,
                                    "hbm_write_bytes": hbm.write_bytes, "l2_bypass": True}
        else:
            summaries[str(rank)] = {"kernels": len(trace.kernels), "hbm_read_bytes": result.read_total,
                                    "hbm_write_bytes": result.write_total}
    program.meta.misc["megatron_per_rank"] = summaries
    timing.__dict__["megatron_traces"] = traces
    timing.resolved_settings["megatron_trace"] = {
        "per_rank": summaries,
        "gradient_accumulation_fusion": timing.implementation_options.get("gradient_accumulation_fusion", False),
        "megatron_param_copy": timing.implementation_options.get("megatron_param_copy", "separate_copy"),
        "parameter_update": "master_update_then_model_copy",
        "timing": "kernel_trace" if kernel_trace_timing else "operator_path",
        "flash_backward": "default_non_deterministic", "frequency_schedule": "fixed_hardware_clock",
        "reference_point": "minimum_time_Megatron_Perseus"}
    if int(timing.cp) > 1:
        timing.resolved_settings["megatron_trace"]["cp_copy_timing"] = "streaming_cold_kernel"
    if recompute_modules(timing):
        timing.resolved_settings["megatron_trace"]["selective_recompute"] = {
            "granularity": "selective", "modules": list(recompute_modules(timing)),
            "restore_copy": "cudaMemcpyAsync_d2d", "source": "megatron-core 0.12.1 tensor_parallel/random.py:494-574"}
    if multi_gpu(timing) and kernel_trace_timing:
        timing.resolved_settings["megatron_trace"]["step_timing"] = "flat_program_network_simulator"
    if dp_grad_sync(timing) != "legacy":
        timing.resolved_settings["megatron_trace"].update(
            dp_grad_sync=dp_grad_sync(timing), distributed_optimizer=distributed_optimizer(timing),
            grad_scaling=timing.implementation_options.get("megatron_grad_scaling", "pre_scale"),
            overlap_param_gather=False, dp_shard_described="data_parallel_rank_0",
            dp_payload="padded_grad_buffer")
        if timing.implementation_options.get("megatron_dp_buckets", "per_reducer") == "one":
            timing.resolved_settings["megatron_trace"]["dp_buckets"] = "one_collective_per_kind"
    return program
