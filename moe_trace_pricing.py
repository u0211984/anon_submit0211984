from dataclasses import dataclass, replace

def _dense_gemm(model, shape, weight_bytes=None):
    from opmodel import settings
    from program.vtp import _gemm
    _, k, n = map(int, shape)
    def price():
        policy = tuple(settings.get(key) for key in (
            "OPMODEL_GEMV_RULE", "OPMODEL_GEMV_MAX_M", "OPMODEL_GEMV_MAX_N", "OPMODEL_GEMV_STREAM_BW"))
        policy += (bool(getattr(model, "_gemv_disabled", False)),)
        cache = model.__dict__.setdefault("_moe_dense_gemm_caches", {}).setdefault(policy, {})
        absent = object()
        previous = model.__dict__.get("_vtp_gemm_cache", absent)
        model._vtp_gemm_cache = cache
        try:
            return _gemm(model, shape) if weight_bytes is None else _gemm(model, shape, weight_bytes)
        finally:
            if previous is absent:
                model.__dict__.pop("_vtp_gemm_cache", None)
            else:
                model._vtp_gemm_cache = previous
    if settings.get("OPMODEL_GEMV_RULE") == "trtllm" and _trtllm_ineligible(shape):
        with settings.override({"OPMODEL_GEMV_RULE": "cublas"}):
            return price()
    return price()

def _trtllm_ineligible(shape):
    _, k, n = map(int, shape)
    return n > 128000 or n % 2 or k % 8

def _gemm_operand_passes(model, kernels):
    from opmodel import settings
    from program.vtp import _GEMM_NAME
    def masked(keep):
        return [k if k.kind != "gemm" or not k.gemm or keep(k) else replace(k, kind="gemm_not_this_rule")
                for k in kernels]
    if settings.get("OPMODEL_GEMV_RULE") != "trtllm":
        return model._trace_gemm_passes(kernels, name=_GEMM_NAME)
    passes = model._trace_gemm_passes(masked(lambda k: not _trtllm_ineligible(k.gemm)), name=_GEMM_NAME)
    with settings.override({"OPMODEL_GEMV_RULE": "cublas"}):
        passes.update(model._trace_gemm_passes(masked(lambda k: _trtllm_ineligible(k.gemm)), name=_GEMM_NAME))
    return passes

@dataclass(frozen=True)
class PricedMoeKernel:
    name: str
    latency_s: float
    read_bytes: float
    write_bytes: float
    useful_flops: float
    issued_flops: float
    activity: object

def _sector_bytes(nbytes, sector):
    return sector * -(-float(nbytes) // sector) if nbytes > 0 else 0.0

TRITON_DEFAULT_NUM_WARPS, TRITON_DEFAULT_NUM_STAGES = 4, 3

def triton_mma_v2_warps(block_m, block_n, num_warps):
    reps, warps = [-(-block_m // 16), -(-block_n // 8)], [1, 1]
    while warps[0] * warps[1] < num_warps:
        if reps[0] >= reps[1]:
            warps[0] *= 2
            if reps[0] != 1:
                reps[0] //= 2
        else:
            warps[1] *= 2
            reps[1] //= 2
    return tuple(warps)

def vllm_default_config_template(model, tile):
    from opmodel.models.extended_roofline import GemmKernelTemplate
    backend = model._opmodel_backend or model._gemm_activity_backend
    if backend is None:
        raise NotImplementedError("opmodel_tile counts require the operator-model backend")
    wm, wn = triton_mma_v2_warps(tile.block_m, tile.block_n, TRITON_DEFAULT_NUM_WARPS)
    template = GemmKernelTemplate(
        name=f"vllm_fused_moe_triton_{tile.block_m}x{tile.block_n}x{tile.block_k}_w{TRITON_DEFAULT_NUM_WARPS}"
             f"_s{TRITON_DEFAULT_NUM_STAGES}",
        cta_m=tile.block_m, cta_n=tile.block_n, cta_k=tile.block_k,
        warp_m=tile.block_m // wm, warp_n=tile.block_n // wn, warp_k=tile.block_k, num_warp_tile_k=1,
        mma_m=16, mma_n=8, mma_k=16, pipeline_stages=TRITON_DEFAULT_NUM_STAGES,
        warps_per_cta=TRITON_DEFAULT_NUM_WARPS, registers_per_thread=64,
        source="vLLM 0.10.2 fused_moe.py:834-848 get_default_config; Triton 3.4.0 AccelerateMatmul.cpp:69-130",
        dtypes=(backend.dtype,))
    return template, backend

def grouped_opmodel_tile_counts(model, kernel):
    g = kernel.grouped
    if g.tile_rule == "vllm_default_config":
        template, backend = vllm_default_config_template(model, g.tile)
    else:
        template, backend = model.rapid_moe_selected_template(
            counts=g.expert_counts, hidden_size=kernel.spec.hidden, intermediate_size=kernel.spec.intermediate)
    bm, bn, bk = g.tile.block_m, g.tile.block_n, g.tile.block_k
    if (template.cta_m, template.cta_n, template.cta_k) != (bm, bn, bk):
        raise ValueError("opmodel_tile: the grouped GEMM's tile is not the op model's selected template")
    k, n = int(g.input_width), int(g.output_width)
    unit = backend.predict_gemm_with_template(dim1=bm, dim2=k, dim3=n, template=template, name=kernel.name)
    tx = dict(unit.diagnostics.get("transaction_bytes") or {})
    a_req, b_req, d_store = (float(tx[key]) for key in ("a_l2_requested", "b_l2_requested", "d_store"))
    smem_unit = float(unit.memory_accesses[1])
    if abs(float(unit.memory_accesses[2]) - (a_req + b_req + d_store)) > 1e-6 * max(1.0, unit.memory_accesses[2]):
        raise ValueError("opmodel_tile: unit GEMM L2 bytes are not A + B requests + D stores")
    blocks = float(g.local_blocks)
    rows = float(sum(g.expert_counts[e] for e in g.local_experts))
    n_tiles = -(-n // bn)
    live = blocks * n_tiles
    launched = float(g.launched_grid)
    level = backend.hardware.memory_levels.get("l2")
    sector = int(getattr(level, "sector_size_bytes", None) or 32)
    meta = live * (_sector_bytes(4 * bm, sector) + 2 * sector) + max(0.0, launched - live) * sector
    if kernel.name.endswith("w2"):
        meta += rows * n_tiles * sector
    l2_read = blocks * b_req + rows / bm * a_req + meta
    l2_write = rows / bm * d_store
    issued = blocks * float(unit.flops)
    useful = float(g.useful_flops)
    if issued + 1e-6 < useful:
        raise ValueError("opmodel_tile: issued tensor FLOPs below useful")
    return dict(l2_read=l2_read, l2_write=l2_write, smem=blocks * smem_unit, issued=issued, useful=useful), template.name

MOE_COUNTS_ROLES = {"moe.router": "moe_router", "moe.logits_cast": "moe_routing", "moe.topk_softmax": "moe_routing",
                    "moe.renorm_sum": "moe_routing", "moe.renorm_divide": "moe_routing",
                    "moe.align_small": "moe_align", "moe.align_count_prefix_init": "moe_align",
                    "moe.align_sort": "moe_align", "moe.w13": "moe_w13", "moe.silu_and_mul": "moe_swiglu",
                    "moe.w2": "moe_w2", "moe.sum": "moe_sum"}
LINEAR_SOFTMAX_COUNTS_ROLES = {"fused_add_rms_norm": "final_norm", "prune_hidden_states": "lm_head",
                               "lm_head": "lm_head"}

def counts_role(kernel):
    name = kernel.name.removesuffix(".split_k_reduce")
    if kernel.op == "MLP":
        if name not in MOE_COUNTS_ROLES:
            raise ValueError(f"no counts role for MoE kernel {kernel.name!r}")
        return ("MLP", MOE_COUNTS_ROLES[name])
    if kernel.op == "linear_softmax":
        role = LINEAR_SOFTMAX_COUNTS_ROLES.get(name, "head_softmax" if name.startswith("sampler.") else None)
        if role is None:
            raise ValueError(f"no counts role for head kernel {kernel.name!r}")
        return ("linear_softmax", role)
    return (kernel.op, "")

def price_moe_layer_trace(model, kernels, grouped_traffic="logical"):
    if grouped_traffic not in ("logical", "cta_operand_proxy", "opmodel_tile"):
        raise ValueError("moe_grouped_traffic must be logical | cta_operand_proxy | opmodel_tile")
    """Price one trace with the timing model's existing GEMM/roofline APIs.

    Every input kernel is one launch. Logical traffic is a cold-streaming
    approximation, not CTA-level traffic or the step-4 allocator simulation.
    Returned activity uses the normal OperationActivity contract; its caller
    owns aggregation into energy reports.
    """

    l2_bypass = bool(getattr(model, "l2_bypass", False))
    priced = []
    for index, kernel in enumerate(kernels):
        name = f"moe_trace_{index}_{kernel.name}"
        traffic = float(kernel.read_bytes + kernel.write_bytes)
        if kernel.kind == "gemm" and kernel.gemm is not None:
            m, k, n = kernel.gemm
            from program.vtp import gemm_weight_bytes
            latency, activity = _dense_gemm(model, (m, k, n), gemm_weight_bytes(model, kernel))
            useful = issued = float(2 * m * k * n)
        else:
            grouped = getattr(kernel, "grouped", None)
            if grouped is not None:
                useful = float(grouped.useful_flops)
                issued = float(grouped.padded_flops)
                duration = model.roofline(issued, traffic, name="", activity_kind=None,
                                          compute_unit="matrix", mem_level=model.num_levels - 1)
                from activity import MemoryTraffic, OperationActivity
                moved = MemoryTraffic(read_bytes=kernel.read_bytes, write_bytes=kernel.write_bytes)
                l2 = moved
                smem = MemoryTraffic()
                staging = 0.
                if grouped_traffic == "opmodel_tile":
                    tile_counts, _ = grouped_opmodel_tile_counts(model, kernel)
                    useful, issued = tile_counts["useful"], tile_counts["issued"]
                    l2 = MemoryTraffic(read_bytes=tile_counts["l2_read"], write_bytes=tile_counts["l2_write"])
                    smem = MemoryTraffic(read_bytes=tile_counts["smem"])
                    duration = model.roofline(issued, traffic, name="", activity_kind=None,
                                              compute_unit="matrix", mem_level=model.num_levels - 1)
                if grouped_traffic == "cta_operand_proxy":
                    from math import ceil
                    local_rows = sum(grouped.expert_counts[e] for e in grouped.local_experts)
                    a_bytes = 2 * local_rows * grouped.input_width * ceil(grouped.output_width / grouped.tile.block_n)
                    wq = getattr(getattr(model, "precision", None), "linear_weights", None)
                    b_bytes = (2 if wq is None else wq) * grouped.local_blocks * grouped.input_width * grouped.output_width
                    operand = a_bytes + b_bytes
                    metadata = sum(a.nbytes for a in kernel.reads if not a.weight and
                                   (a.tensor.endswith("sorted_ids") or a.tensor.endswith("block_experts") or a.tensor.endswith("padded_count") or a.tensor.endswith("weights")))
                    l2 = MemoryTraffic(read_bytes=operand + metadata, write_bytes=kernel.write_bytes)

                    smem = MemoryTraffic(read_bytes=operand)
                    staging = operand

                    l2_bw = float(model.mem_layer[model.num_levels - 2].get_throughput())
                    smem_bw = float(model.mem_layer[model.num_levels - 3].get_throughput())
                    if l2_bw <= 0 or smem_bw <= 0:
                        raise ValueError("grouped CTA proxy requires positive L2 and SMEM bandwidth")
                    if l2_bypass:
                        duration = (smem.read_bytes + staging) / smem_bw
                    else:
                        duration = max(duration, (l2.read_bytes + l2.write_bytes) / l2_bw,
                                       (smem.read_bytes + staging) / smem_bw)
                if l2_bypass:
                    dram = model.roofline(issued, l2.read_bytes + l2.write_bytes, name="", activity_kind=None,
                                          compute_unit="matrix", mem_level=model.num_levels - 1)
                    duration = max(duration, dram) if grouped_traffic == "cta_operand_proxy" else dram
                from opmodel import settings
                activity = OperationActivity(useful_tensor_flops=useful, issued_tensor_flops=issued,
                                             vector_flops=float(kernel.flops), hbm=moved, l2=l2,
                                             smem=smem, smem_staging_bytes=staging,
                                             smem_staging_energy_bytes=staging if settings.get("OPMODEL_SMEM_STAGING") else 0.)
                if kernel.flops:
                    vector = float(kernel.flops)
                    duration = max(duration, vector / model.core.get_throughput("vector"))
                latency = float(model.O or 0.0) + duration

                floor = getattr(model, "kernel_floor", None)
                if floor is not None:
                    latency = floor(latency)
            else:
                useful = issued = float(kernel.flops)
                latency, activity = model._price_kernels([kernel], record=False)
        priced.append(PricedMoeKernel(kernel.name, float(latency), float(kernel.read_bytes),
                                      float(kernel.write_bytes), useful, issued, activity))
    return priced

@dataclass(frozen=True)
class PricedMoeRank:
    rank: int
    compute_s: float
    activity: object
    kernels: tuple
    moe_kernels: tuple
    stages: tuple
    dynamic_j: float

@dataclass(frozen=True)
class PricedMoeStep:
    latency_s: float | None
    ranks: tuple
    activity: object
    collectives: tuple
    network_j: float
    dynamic_gpu_j: float
    static_gpu_j: float | None
    assumptions: dict

    @property
    def total_j(self):
        if self.static_gpu_j is None:
            return None
        return self.network_j + self.dynamic_gpu_j + self.static_gpu_j

def _network_parameters(timing, axis):
    dim = timing.hw_config.network_layout.parallelism_map[axis]
    if not isinstance(dim.bandwidth, (int, float)):
        raise NotImplementedError("MoE ring pricing requires scalar per-link bandwidth")
    topology = str(dim.topology_type).lower().replace("_", "").replace("-", "")
    if topology not in ("fc", "fullyconnected", "switch", "ring"):
        raise NotImplementedError("MoE ring pricing supports FC, Switch and Ring placement")
    return dict(bandwidth_bytes_s=float(dim.bandwidth)*float(dim.util),
                launch_s=float(timing.O or 0.), latency_s=float(dim.latency),
                joules_per_byte=8*float(dim.energy_per_bit))

def _collective_price(timing, kernel, tp, dp, collective_rule, phase, a2a=None):
    from program.moe_rank import Collective, price_collectives
    from program.vllm_allreduce import choose_allreduce, custom_seconds
    if collective_rule != "network_model":
        raise ValueError("moe_collective_rule must explicitly be network_model")
    if kernel.name in ("moe.a2a_dispatch", "moe.a2a_combine"):
        from program.moe_rank import deepep_rank_bytes, price_all_to_all
        backend, tokens, hidden, experts, top_k, ep = a2a
        part = 0 if kernel.name == "moe.a2a_dispatch" else 1
        sizes = [deepep_rank_bytes(backend, tokens, hidden, experts, top_k, ep, r)[part] for r in range(ep)]
        return price_all_to_all(kernel.name, tuple(range(ep)), sizes, **_network_parameters(timing, "ep"))
    is_dp = kernel.name.startswith(("moe.broadcast_", "moe.dp_all_reduce"))
    world = dp if is_dp else tp
    params = _network_parameters(timing, "ep" if is_dp else "tp")
    size = int(kernel.reads[0].nbytes)
    kind = "broadcast" if kernel.name.startswith("moe.broadcast_") else "all_reduce"
    if kernel.name == "logits.gather":
        from program.moe_rank import CollectivePrice
        from program.vtp_gather import gather_energy, gather_seconds, logits_gather_plan
        from sim_settings import nccl_latency_rule
        event = Collective(kernel.name, "gather", tuple(range(world)), size, root=0)
        plan = logits_gather_plan(world, size)
        link = timing.links["tp"]
        latency = 0.0 if nccl_latency_rule(timing.hw_config) in ("tuning_formula", "tuning_protocol") else link.latency
        dim = timing.hw_config.network_layout.parallelism_map["tp"]
        topology = str(dim.topology_type).lower().replace("_", "").replace("-", "")
        seconds = gather_seconds(world, size, link.bandwidth, latency,
                                 topology in ("fc", "fullyconnected"))
        dimension_index = timing.hw_config.network_layout.dimensions.index(dim)
        from copy import copy
        from opmodel import settings

        implicit = (getattr(dim, "link_energy_rule", None) is None and
                    getattr(timing.hw_config.network_layout, "link_energy_rule", None) is None and
                    settings.get("RAPID_LINK_ENERGY_RULE") is None)
        energy_hw = timing.hw_config
        if implicit:
            energy_hw = copy(energy_hw)
            energy_hw.network_layout = copy(energy_hw.network_layout)
            object.__setattr__(energy_hw.network_layout, "link_energy_rule", "per_payload_byte")
        energy = gather_energy([dict(label=kernel.name, participants=world, size_bytes=size,
                                     active_seconds=seconds)], energy_hw, dimension_index)
        seconds += float(getattr(timing.hw_config.sw_config, "collective_launch_overhead", 0.) or 0.)
        return CollectivePrice(event, seconds, plan.network_bytes, energy.link_dynamic_j)
    event = Collective(kernel.name, kind, tuple(range(world)), size,
                       int(kernel.name.rsplit("_", 1)[1]) if kind == "broadcast" else None)
    if not is_dp:
        dim = timing.hw_config.network_layout.parallelism_map["tp"]
        topology = str(dim.topology_type).lower().replace("_", "").replace("-", "")
        full = topology in ("fc", "fullyconnected", "switch")
        rule = timing.implementation_options["tp_allreduce"]
        graph = phase == "decode" and bool((timing.engine_settings or {}).get("cuda_graph", True))
        choice = choose_allreduce(rule, world, full, size, graph)
        if choice.algorithm.startswith("custom_"):
            from program.moe_rank import CollectivePrice

            policy = dict(bandwidth=params["bandwidth_bytes_s"],
                          per_peer_links=topology in ("fc", "fullyconnected"),
                          latency=params["latency_s"], kernel_overhead=params["launch_s"],
                          barrier_timing=timing.implementation_options["vllm_custom_barrier_timing"],
                          hbm_bandwidth=float(timing.mem_layer[timing.num_levels-1].get_throughput()))
            seconds = custom_seconds(choice, size, world, policy)
            per_rank = (world-1)*size
            if choice.hops == 2:
                per_rank = 2*per_rank/world
            wire = world*per_rank
            priced = CollectivePrice(event, seconds, wire, wire*params["joules_per_byte"])
            return priced
    priced = price_collectives((event,), **params)[0]
    if not is_dp and kind == "all_reduce":
        from sim_settings import nccl_latency_rule
        from timing_model import CollectiveType
        latency_rule = nccl_latency_rule(timing.hw_config)
        if latency_rule == "tuning_protocol":
            raise NotImplementedError("moe_source_collective_rule=network_model does not describe tuning_protocol")
        link = timing.links["tp"]
        seconds = timing.network_model.collective(kind=CollectiveType.ALL_REDUCE, size_bytes=size, participants=world,
                                                  ib=link.bandwidth, ll=link.latency, axis="tp")
        base = float(getattr(timing.hw_config.sw_config, "collective_launch_overhead", 0.) or 0.)
        if latency_rule == "tuning_formula":
            from program.nccl_latency import dimension_params, formula_seconds
            dim_params = dimension_params(timing.hw_config, "tp", world)
            seconds += formula_seconds(base, CollectiveType.ALL_REDUCE, world,
                                       dim_params["latency"], dim_params["algorithms"]["ALL_REDUCE"])
        else:
            seconds += base
        return replace(priced, seconds=seconds)

    if kind == "all_reduce":
        from sim_settings import nccl_latency_rule
        latency_rule = nccl_latency_rule(timing.hw_config)
        base = float(getattr(timing.hw_config.sw_config, "collective_launch_overhead", 0.) or 0.)
        if latency_rule == "tuning_protocol":
            from program.nccl_latency import protocol_model
            from timing_model import CollectiveType
            pm = protocol_model(timing.hw_config, "ep" if is_dp else "tp", CollectiveType.ALL_REDUCE, world, size)
            seconds = params["launch_s"] + pm["transfer_seconds"] + pm["latency_seconds"]
        elif latency_rule == "tuning_formula":
            from program.nccl_latency import dimension_params, formula_seconds
            from timing_model import CollectiveType
            dim_params = dimension_params(timing.hw_config, "ep" if is_dp else "tp", world)
            seconds = (priced.seconds - 2*(world-1)*params["latency_s"] +
                       formula_seconds(base, CollectiveType.ALL_REDUCE, world,
                                       dim_params["latency"], dim_params["algorithms"]["ALL_REDUCE"]))
        else:
            seconds = priced.seconds + base
        priced = replace(priced, seconds=seconds)
    return priced

_RANK_BATCH = 16

def _rank_key(kernels):
    from program import trace_identity
    if not trace_identity.enabled():
        return None
    try:
        return trace_identity.TraceKey(kernels)
    except Exception:
        return None

def price_moe_rank_step(timing, cfg, phase, batch, seq, tile_rule, *,
                        ep=None, moe_dp=None, collective_buffers=None,
                        moe_collective_rule=None, moe_token_counts=None,
                        replay_steps=3, attention_time_rule="native", moe_routing_step=0):
    from activity import MemoryTraffic, OperationActivity, ZERO_ACTIVITY
    from kernel_trace_vllm import VllmEngine
    from flash_attention_adapter import _run
    from opmodel import settings
    from program.energy import l2_bypassed, price_activity
    from program.vtp import _gemm
    import l2_sim
    import sim_settings

    tp = int(timing.tp)
    if attention_time_rule not in ("native", "counts_only"):
        raise ValueError("attention_time_rule must be native | counts_only")
    if attention_time_rule == "counts_only" and phase != "prefill":
        raise ValueError("counts_only attention is an ordinary-forward prefill probe")
    dp = int(moe_dp if moe_dp is not None else cfg.get("moe_dp", cfg.get("attention_dp", getattr(timing, "moe_dp", 1))))
    ep = int(ep if ep is not None else (tp*dp if tp*dp > 1 else 1))

    if dp < 1 or tp < 1 or (ep != 1 and ep != tp*dp):
        raise ValueError("MoE EP must equal attention TP times moe_dp")
    backend = (getattr(timing, "implementation_options", None) or {}).get("vllm_all2all_backend", "naive")
    from kernel_trace_vllm_moe import _all2all_backend

    backend = _all2all_backend(dict(vllm_all2all_backend=backend), tp=tp, ep=ep, dp=dp)
    if int(getattr(timing, "cp", 1)) != 1 or int(getattr(timing, "pp", 1)) != 1:
        raise NotImplementedError("source MoE rank pricing requires CP1/PP1")
    if int(getattr(timing, "replica_count", 1)) != 1:
        raise ValueError("source MoE pricing describes one TP x moe_dp world; price independent replicas explicitly")
    if (float(timing.precision.activations) != 2 or float(timing.precision.parameters) != 2
            or float(timing.precision.kv_cache) != 2):
        raise ValueError("source MoE rank pricing requires bf16 activations, weights and KV")
    world = tp*dp
    mode = collective_buffers if collective_buffers is not None else settings.get("RAPID_L2_COLLECTIVE_BUFFERS")
    if world > 1 and mode not in ("occupy_l2", "leave_l2"):
        raise ValueError("MoE ranks require collective_buffers=occupy_l2 | leave_l2")
    if world > 1 and moe_collective_rule != "network_model":
        raise ValueError("parallel MoE requires explicit moe_collective_rule=network_model")
    cfg = dict(cfg, attention_dp=dp)
    grouped_traffic = cfg.get("moe_grouped_traffic", "logical")
    cache_spec = sim_settings.CacheSpec.from_hw(timing.hw_config)
    policy = sim_settings.SimPolicy.current(cache_spec)

    l2_bypass = bool(getattr(timing, "l2_bypass", False))
    simulate = policy.step_simulation and not l2_bypass
    capacity = 0 if l2_bypass else int(timing._l2_capacity_bytes())
    sim_kwargs = dict(gemm_min_tile_m=policy.gemm_min_tile_m, steps=replay_steps,
                      steady_state=policy.steady_state, address_plan=policy.address_plan)
    if policy.sharing_classes and not l2_bypass:
        capacity = cache_spec.shared_capacity_bytes
        sim_kwargs.update(sharing_classes=True, physical_capacity_bytes=cache_spec.physical_bytes,
                          private_copies=policy.private_copies)
    eng = VllmEngine(sm_count=int(timing.core.num_bundle), **dict(timing.engine_settings or {}))
    q_len = int(seq) if phase == "prefill" else 1
    kv_len = int(seq) if phase == "prefill" else int(seq)+1
    flash_key = (phase, int(batch), int(seq), tp, timing.O,
                 settings.get("RAPID_FLASH_SPILLS"), attention_time_rule)
    flash_cache = timing.__dict__.setdefault("_moe_flash_prices", {})

    mla = str(cfg.get("attention_type", "")).lower() == "mla"
    if mla:
        flash_cache[flash_key] = (None, ())
    if flash_key not in flash_cache:
        flash_kwargs = {"describe_only": True} if attention_time_rule == "counts_only" else {}
        flash_cache[flash_key] = _run(timing, batch, int(timing.hidden_dim), q_len, kv_len,
            int(timing.num_heads), int(timing.kv_heads), int(timing.core.num_bundle),
            decode=phase == "decode", return_kernels=True, **flash_kwargs)
    flash_spec, flash_records = flash_cache[flash_key]
    if moe_token_counts is not None:
        expected = int(batch)*(int(seq) if phase == "prefill" else 1)
        if tuple(moe_token_counts) != (expected,)*dp:
            raise ValueError("full-step pricing requires equal per-DP batches; use moe_layer_trace for uneven counts")
    rank_prices = []
    priced_classes = {}

    dense_trace = world == 1 and grouped_traffic == "opmodel_tile"
    selected_split_k = dense_trace and settings.get("OPMODEL_GEMM_CATALOG") == "cublaslt_sm80_bf16_splitk"
    split_backend = timing._opmodel_backend or timing._gemm_activity_backend

    from program import trace_identity
    layer_cache = {} if trace_identity.enabled() else None
    network_records = []
    stage_keys = None
    tp_fully_connected = True
    if tp > 1:
        tp_dimension = timing.hw_config.network_layout.parallelism_map["tp"]
        tp_topology = str(tp_dimension.topology_type).lower().replace("_", "").replace("-", "")
        tp_fully_connected = tp_topology in ("fc", "fullyconnected", "switch")

    rules_by_tp_rank = {}
    for first in range(0, world, _RANK_BATCH):
        batch_ranks = range(first, min(world, first + _RANK_BATCH))
        traces, rank_keys = {}, {}
        for rank in batch_ranks:
            ks = l2_sim.vllm_launch_trace(cfg, phase, batch, seq, sm_count=eng.sm_count, engine=eng, tp=tp,
                rank=rank%tp, ep=ep, moe_rank=rank if ep > 1 or dp > 1 else 0, moe_token_counts=moe_token_counts,
                collective_buffers=mode, moe_tile_rule=tile_rule,
                tp_allreduce=timing.implementation_options["tp_allreduce"], fully_connected=tp_fully_connected,
                rapid_tile_selector=timing.rapid_moe_tile_selector, moe_routing_step=moe_routing_step,
                moe_layer_cache=layer_cache)
            if selected_split_k:
                def select_trace_kernel(kernel):
                    return split_backend.selected_gemm_template(
                        dim1=kernel.gemm[0], dim2=kernel.gemm[1], dim3=kernel.gemm[2],
                        name=kernel.name, role=l2_sim.gemm_trace_role(kernel),
                        allow_gemv=not getattr(timing, "_gemv_disabled", False))

                from dataclasses import fields as _fields
                from kernel_trace import Kernel as _Kernel
                from kernel_trace_vllm_moe import MoeKernel as _MoeKernel
                ks = [_Kernel(**{f.name: getattr(k, f.name) for f in _fields(_Kernel)})
                      if isinstance(k, _MoeKernel) and k.kind == "gemm" else k for k in ks]
                ks, _ = l2_sim.expand_selected_split_k(ks, select_trace_kernel)
            traces[rank] = ks
            rank_keys[rank] = _rank_key(ks)

        first_rank, priced_ranks = {}, []
        for rank in batch_ranks:
            key = rank_keys[rank]
            if key is None or (key not in priced_classes and first_rank.setdefault(key, rank) == rank):
                priced_ranks.append(rank)
        sims = {}
        if simulate:
            for rank in priced_ranks:
                if rank % tp not in rules_by_tp_rank:
                    rules_by_tp_rank[rank % tp] = l2_sim.vllm_rules(cfg, phase, tp=tp, rank=rank%tp, ep=ep,
                                                                    collective_buffers=mode)
            passes = [timing._trace_gemm_passes(traces[rank], named=selected_split_k, roles=True) if dense_trace
                      else _gemm_operand_passes(timing, traces[rank]) for rank in priced_ranks]
            rules = [rules_by_tp_rank[rank % tp] for rank in priced_ranks]
            if trace_identity.enabled():
                results = l2_sim.simulate_ranks([traces[rank] for rank in priced_ranks], capacity, rules,
                                                operand_passes_list=passes, **sim_kwargs)
            else:
                results = [l2_sim.simulate(traces[rank], capacity, rule, operand_passes=p, **sim_kwargs)
                           for rank, rule, p in zip(priced_ranks, rules, passes)]
            sims = dict(zip(priced_ranks, results))
        for rank in batch_ranks:
            ks, key = traces[rank], rank_keys[rank]
            if key is not None and key in priced_classes:
                rank_prices.append(replace(priced_classes[key], rank=rank))
                continue
            if simulate:
                res = sims[rank]
            activities, rows, all_rows, stages, keys = ZERO_ACTIVITY, [], [], [], []
            compute = segment = 0.
            flash_indices = {}
            counts_roles = {} if dense_trace else None
            for index, kernel in enumerate(ks):
                if kernel.kind == "collective":
                    if kernel.name.endswith(".ipc_copy"):
                        continue
                    keys.append(kernel.name)
                    stages.append(segment)
                    segment = 0.
                    if rank == 0:
                        a2a = None
                        if backend != "naive":
                            a2a = (backend, int(batch)*(int(seq) if phase == "prefill" else 1), int(cfg["hidden_size"]),
                                   int(cfg["num_experts"]), int(cfg["num_experts_per_tok"]), ep)
                        price = _collective_price(timing, kernel, tp, dp, moe_collective_rule, phase, a2a)
                        network_records.append(price)
                    continue
                if kernel.name.endswith(".split_k_reduce"):
                    seconds, activity = 0.0, ZERO_ACTIVITY
                elif mla and kernel.op in ("qkv_proj", "rope", "attention", "output_proj"):
                    seconds, activity = timing._price_mla_kernel(kernel, phase=phase, batch=int(batch), seq=int(seq))
                elif kernel.op == "MLP":
                    row = price_moe_layer_trace(timing, [kernel], grouped_traffic)[0]
                    seconds, activity = row.latency_s, row.activity
                elif kernel.kind == "gemm":
                    from program.vtp import gemm_weight_bytes
                    seconds, activity = _dense_gemm(timing, kernel.gemm, gemm_weight_bytes(timing, kernel))
                elif kernel.kind == "attention":
                    layer = kernel.writes[0].tensor.split(".")[0]
                    fi = flash_indices.get(layer, 0)
                    flash_indices[layer] = fi+1
                    if fi >= len(flash_records):
                        raise ValueError("MoE flash trace/pricing launch count differs")
                    record = flash_records[fi]

                    floor = timing.kernel_min_seconds()
                    vector = sum(record.vector_ops.get(x, 0) for x in ("fadd", "fmul", "hadd", "hmul"))
                    vector += 2*sum(record.vector_ops.get(x, 0) for x in ("ffma", "hfma"))
                    seconds = record.time_s if floor is None else max(record.time_s, floor)
                    activity = OperationActivity(useful_tensor_flops=record.useful_tensor_flops,
                        issued_tensor_flops=record.issued_tensor_flops, vector_flops=vector,
                        smem=MemoryTraffic(record.smem_read_bytes+record.diagnostics.get("l1_spill_read_bytes", 0),
                                           record.smem_write_bytes+record.diagnostics.get("l1_spill_write_bytes", 0)),
                        l2=MemoryTraffic(record.l2_read_bytes, record.l2_write_bytes),
                        hbm=MemoryTraffic(kernel.read_bytes, kernel.write_bytes),
                        smem_staging_bytes=record.smem_staging_bytes,
                        smem_staging_energy_bytes=record.smem_staging_bytes if settings.get("OPMODEL_SMEM_STAGING") else 0)
                else:
                    seconds, activity = timing._price_kernels([kernel], record=False)
                if simulate:
                    activity = replace(activity, hbm=MemoryTraffic(res.hbm_read[index], res.hbm_write[index]))
                if l2_bypass:
                    activity = l2_bypassed(kernel.name, activity)
                if seconds < -1e-12:
                    raise ValueError("MoE re-timing produced negative kernel time")
                seconds = max(0., seconds)
                compute += seconds
                segment += seconds
                activities += activity
                if counts_roles is not None:
                    role = counts_role(kernel)
                    counts_roles[role] = counts_roles.get(role, ZERO_ACTIVITY) + activity
                priced = PricedMoeKernel(kernel.name, seconds, kernel.read_bytes, kernel.write_bytes,
                    activity.useful_tensor_flops if kernel.kind in ("gemm", "grouped_gemm") else kernel.flops,
                    activity.issued_tensor_flops if kernel.kind in ("gemm", "grouped_gemm") else kernel.flops,
                    activity)
                all_rows.append(priced)
                if kernel.op == "MLP":
                    rows.append(priced)
            stages.append(segment)
            if stage_keys is None:
                stage_keys = keys
            elif keys != stage_keys:
                raise ValueError("MoE synchronized rank traces have different collective order")
            dynamic = price_activity(activities, timing.hw_config).total_j
            if counts_roles is not None and simulate:
                from ncu_counts import memcpy_activity
                timing.__dict__.setdefault("counts_memcpy_by_phase", {})[phase] = (memcpy_activity(ks, res), True)
                timing.__dict__.setdefault("counts_moe_roles_by_phase", {}).setdefault(phase, []).append(counts_roles)
            rank_prices.append(PricedMoeRank(rank, compute, activities, tuple(all_rows), tuple(rows), tuple(stages), dynamic))
            if key is not None:
                priced_classes[key] = rank_prices[-1]

    clocks = [0.]*world
    placed_collectives = []
    for index, price in enumerate(network_records):
        for rank in range(world):
            clocks[rank] += rank_prices[rank].stages[index]
        is_dp = price.event.name.startswith("moe.")
        groups = (tuple(tuple(d*tp+t for d in range(dp)) for t in range(tp)) if is_dp else
                  tuple(tuple(d*tp+t for t in range(tp)) for d in range(dp)))
        for group in groups:
            end = max(clocks[r] for r in group)+price.seconds
            for r in group:
                clocks[r] = end
            root = group[price.event.root] if price.event.root is not None else None
            event = replace(price.event, participants=group, root=root, network_energy_owner=min(group))
            placed_collectives.append((replace(price, event=event), 1))
    latency = max(clocks[r.rank]+r.stages[-1] for r in rank_prices)
    activity = sum((r.activity for r in rank_prices), ZERO_ACTIVITY)
    static = float(getattr(timing.hw_config, "static_power_w", 0.))*world*latency
    if attention_time_rule == "counts_only":
        latency, static = None, None
    return PricedMoeStep(latency, tuple(rank_prices), activity, tuple(placed_collectives),
        sum(p.network_joules*n for p, n in placed_collectives),
        sum(r.dynamic_j for r in rank_prices), static,
        dict(moe_collective_rule=moe_collective_rule, collective_buffers=mode,
             moe_tile_rule=tile_rule, moe_dp=dp, tp=tp, ep=ep, step_simulation=simulate,
             expert_layout="expert_parallel" if ep > 1 else ("tensor_parallel" if world > 1 else "single_rank"),
             expert_tp=1 if ep > 1 else world,
             all2all_backend=backend if dp > 1 else None,
             memory_time_rule=settings.get("RAPID_MEMORY_TIME_RULE"), replay_steps=replay_steps,
             moe_grouped_traffic=grouped_traffic,
             moe_routing_policy=cfg.get("moe_routing_policy", "multinomial_expected"),
             expert_imbalance_factor=cfg.get("expert_imbalance_factor", 1.0),
             moe_routing_step=moe_routing_step, routing_replay="same declared routing step",
             attention_time_rule=attention_time_rule,
             sharing_classes=policy.sharing_classes, capacity_bytes=capacity,
             physical_l2_bytes=cache_spec.physical_bytes if policy.sharing_classes else None,
             private_copies=str(policy.private_copies) if policy.sharing_classes else None,
             traffic="logical HBM spans; optional CTA operand L2/SMEM proxy; no grouped CTA replay", latency="blocking collectives, concurrent rank stages",
             network_energy_rule="gather uses dense VTP link rule; other collectives use per directed payload link byte",
             collective_memory_scope="network energy only; markers affect cache residency and compute evictions",
             source_energy_scope="GPU compute, approximated device-memory traffic and network link energy; collective GPU instructions and traffic excluded by declared network-only policy",

             **({"l2_bypass": True} if l2_bypass else {})))

def price_moe_step(timing, cfg, phase, batch, seq, tile_rule, **kwargs):
    result = price_moe_rank_step(timing, cfg, phase, batch, seq, tile_rule, **kwargs)
    return result.latency_s, [row for rank in result.ranks for row in rank.moe_kernels]
