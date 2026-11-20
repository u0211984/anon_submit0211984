from copy import deepcopy
from dataclasses import replace

from activity import MemoryTraffic, OperationActivity, ZERO_ACTIVITY
from opmodel import settings
from timing_model import DirectionTiming, OperationTiming

def enabled(timing):
    return (int(timing.tp) > 1 and str(timing.run_type).lower() == "inference"
            and timing.implementation == "fused"
            and timing.implementation_options["flash_source"] == "vllm"
            and timing.implementation_options["tp_allreduce"] == "vllm_size_rule")

def validate(timing):
    if int(timing.tp) == 1:
        return
    if not enabled(timing):
        raise ValueError("VTP TP>1 requires fused inference with flash_source=vllm and tp_allreduce=vllm_size_rule")
    if int(timing.pp) > 1:
        raise ValueError("VTP does not describe pipeline parallelism pp > 1")
    if (int(getattr(timing, "ep", 1)) > 1 or bool(getattr(timing, "use_moe", False))) and _mla(timing):
        raise ValueError("VTP describes dense MLA only; DeepSeek MoE layers at tp > 1 belong to the per-rank vLLM MoE "
                         "source path (the DeepSeek MoE layer description is separate work, not VTP)")
    if int(getattr(timing, "ep", 1)) > 1 or bool(getattr(timing, "use_moe", False)):
        raise ValueError("VTP does not describe ep > 1 or MoE layers of operator-model families; "
                         "use model_type qwen3_moe (vLLM source MoE path, per-rank multi-GPU)")
    if int(timing.cp) != 1:
        raise ValueError("VTP does not describe context parallelism")
    if bool(getattr(timing, "tp_sp", False)):
        raise ValueError("VTP does not describe sequence parallelism")
    mla = _mla(timing)
    if mla:
        if str(timing.model_type).lower() != "deepseek_v3":
            raise ValueError("VTP describes MLA only for model_type deepseek_v3")
        if not timing._mla_vllm_trace():
            raise ValueError("VTP MLA requires implementation.options.mla_inference = vllm_trace")
        heads = int(timing.num_heads)
        if heads % int(timing.tp) or heads // int(timing.tp) < 2:
            raise ValueError("VTP MLA requires tp to divide num_heads with at least 2 heads per rank")
        if int(timing.intermediate_size) % int(timing.tp):
            raise ValueError("VTP tp must divide intermediate_size")
    elif str(timing.model_type).lower() not in ("llama", "qwen2", "qwen3"):
        raise ValueError("VTP describes only dense llama, qwen2 and qwen3, and dense MLA (deepseek_v3)")
    for name in (() if mla else ("num_heads", "kv_heads", "intermediate_size")):
        if int(getattr(timing, name)) % int(timing.tp):
            raise ValueError(f"VTP tp must divide {name}; replicated KV heads are not described")
    if (64 * -(-int(timing.vocab_size) // 64)) % int(timing.tp):
        raise ValueError("VTP vocabulary padded to 64 must split evenly across tp")
    if settings.get("RAPID_L2_COLLECTIVE_BUFFERS") not in ("occupy_l2", "leave_l2"):
        raise ValueError("VTP tp > 1 requires hardware.l2_model.collective_buffers = occupy_l2 | leave_l2")
    dim = timing.hw_config.network_layout.parallelism_map["tp"]
    topology = str(dim.topology_type).lower().replace("_", "").replace("-", "")

    if int(timing.tp) > 2 and topology not in ("fullyconnected", "fc", "switch"):
        raise ValueError("VTP world size > 2 requires NVLink P2P between all GPU pairs (FullyConnected or Switch)")
    if str(timing.run_type).lower() != "inference":
        raise ValueError("VTP describes inference only")
    if (float(timing.precision.activations) != 2 or float(timing.precision.parameters) != 2
            or float(timing.precision.kv_cache) != 2):
        raise ValueError("VTP traces require 16-bit activations, parameters and KV cache")

def _mla(timing):
    return str(getattr(timing, "attention_type", "mha")).lower() == "mla"

def _op(kernel):
    if kernel.op == "MLP":
        return {"gate_up_proj": "ffn1", "silu_and_mul": "gelu", "down_proj": "ffn2"}.get(kernel.name, "ffn2")
    return kernel.op

_GEMM_NAME = "__vtp_gemm__"

def gemm_weight_bytes(timing, kernel):
    quant = getattr(getattr(timing, "precision", None), "linear_weights", None)
    if quant is None or kernel.kind != "gemm" or not kernel.gemm:
        return None
    _, k, n = map(int, kernel.gemm)
    weight = sum(float(a.nbytes) for a in kernel.reads if a.weight)
    per = weight / float(k * n)
    if per == float(timing.precision.activations):
        return None
    if per != float(quant):
        raise ValueError(f"{kernel.name}: weight operand {per} B/element is neither tensor_format nor linear_weights")
    return per

def _gemm(timing, shape, weight_bytes=None):
    cache = timing.__dict__.setdefault("_vtp_gemm_cache", {})
    shape = tuple(shape)
    key = (shape, timing.O) if weight_bytes is None else (shape, timing.O, float(weight_bytes))
    if key not in cache:
        name = _GEMM_NAME
        old = timing._gemm_activity_by_name.pop(name, None)
        seconds = timing.get_gemm_time(*shape, name, **({} if weight_bytes is None else {"weight_bytes": weight_bytes}))[0]
        activity = timing._gemm_activity_by_name.pop(name, ZERO_ACTIVITY)
        if old is not None:
            timing._gemm_activity_by_name[name] = old
        cache[key] = seconds, activity
    return cache[key]

def _rank_key(kernels):
    from program import trace_identity
    if not trace_identity.enabled():
        return None
    try:
        return trace_identity.TraceKey(kernels)
    except Exception:
        return None

def apply(timing, transformer_timings, *, phase, tokens, batch, past):
    import kernel_trace_vllm as kv
    import l2_sim
    import sim_settings

    validate(timing)
    tp, layers = int(timing.tp), int(timing.num_layers)
    cfg = timing._l2_sim_config()
    seq = int(tokens) // int(batch) if phase == "prefill" else int(past)
    eng = kv.VllmEngine(sm_count=int(timing.core.num_bundle), **dict(timing.engine_settings or {}))
    mode = settings.get("RAPID_L2_COLLECTIVE_BUFFERS")
    spec = sim_settings.CacheSpec.from_hw(timing.hw_config)
    policy = sim_settings.SimPolicy.current(spec)
    simulate = policy.step_simulation
    capacity = int(timing._l2_capacity_bytes()) if simulate else 0
    dim = timing.hw_config.network_layout.parallelism_map["tp"]
    topology = str(dim.topology_type).lower().replace("_", "").replace("-", "")
    full = topology in ("fullyconnected", "fc", "switch")
    broadcast_prefix = 0.0
    ar_policy = timing.network_model.vllm_allreduce
    ar_policy.pop("input_broadcast", None)
    if timing.implementation_options["vllm_input_broadcast"] == "source_ring":
        from program.vllm_input_broadcast import simple_flash_input_broadcasts, ring_broadcast_seconds
        if timing.hw_config.sw_config.nccl_latency_rule != "tuning_formula":
            raise ValueError("vllm_input_broadcast=source_ring requires tuning_formula")
        if not timing.flash_attention or eng.sampling not in ("random_seed", "random", "greedy"):
            raise ValueError("vllm_input_broadcast=source_ring requires simple FlashAttention sampling")
        if phase not in ("prefill", "decode") or int(batch) not in (1, 4):
            raise ValueError("vllm_input_broadcast=source_ring supports simple b1/b4 only")

        if phase == "prefill":
            if int(tokens) % int(batch):
                raise ValueError("prefill token count must be divisible by batch")
            table_columns, queries = 0, int(tokens)
        else:
            table_columns = kv.decode_attention_layout(cfg, int(batch), int(past) + 1, eng)[2] // eng.block_size
            queries = int(batch)
        sizes = simple_flash_input_broadcasts(batch=int(batch), padded_batch=int(batch),
            query_tokens=queries, block_table_columns=table_columns, selected_tokens=int(batch))

        bandwidth = ar_policy["bandwidth"]
        broadcast_prefix = ring_broadcast_seconds(sizes, tp,
            base=float(timing.hw_config.sw_config.collective_launch_overhead),
            latency=ar_policy["latency"], bandwidth=bandwidth)
        ar_policy["input_broadcast"] = dict(phase=phase, participants=tp, sizes=sizes,
                                              duration_seconds=broadcast_prefix)
    layer_ops = {"layernorm1", "qkv_proj", "rope", "attention", "output_proj", "layernorm2", "ffn1", "gelu", "ffn2"}
    ranks, rank_times, summaries, offsets, launch_counts = [], [], [], [], []

    attn = transformer_timings["attention"].forward
    attn_activity = attn.activity
    cache_write_time, cache_write = timing._kv_cache_write_cost(int(tokens))
    if cache_write is not None:
        def sub_mem(a, b):
            return MemoryTraffic(max(0, a.read_bytes-b.read_bytes), max(0, a.write_bytes-b.write_bytes))
        attn_activity = replace(attn_activity, hbm=sub_mem(attn_activity.hbm, cache_write.hbm),
                                l2=sub_mem(attn_activity.l2, cache_write.l2))
    mla = _mla(timing)
    classes = {}
    for rank in range(tp):
        ks = l2_sim.vllm_launch_trace(cfg, phase, batch, seq, sm_count=eng.sm_count, engine=eng, tp=tp, rank=rank,
                                      collective_buffers=mode, fully_connected=full, tp_allreduce=timing.implementation_options["tp_allreduce"])

        key = _rank_key(ks)
        if key is not None and key in classes:
            launch, summary, endpoint, row, row_times = classes[key]
            launch_counts.append(dict(launch, rank=rank))
            summaries.append(dict(deepcopy(summary), rank=rank))
            offsets.append(dict(endpoint))
            ranks.append(dict(row))
            rank_times.append(dict(row_times))
            continue
        from program.vllm_allreduce import choose_allreduce
        custom = sum(k.kind == "collective" and not k.name.endswith(".ipc_copy")
                     and k.name != "logits.gather" and choose_allreduce(
                         timing.implementation_options["tp_allreduce"], tp, full, int(k.reads[0].nbytes),
                         phase == "decode" and eng.cuda_graph).algorithm.startswith("custom_") for k in ks)
        local = [k for k in ks if k.kind != "collective"]
        outside = [k for k in local if k.name.startswith(("graph.copy_", "sampler.")) or k.name == "meta.query_start_loc_sub"]

        attention_adjust = 0

        if phase == "decode" and timing.flash_attention and not mla:
            attention_adjust = (layers * timing._decode_flash_launch_count
                                - sum(k.kind == "attention" for k in local))
        priced_local = len(local) + attention_adjust
        graph_count = (priced_local - len(outside) + custom
                       if getattr(timing, "_decode_graph_pricing", False) else 0)
        copies = sum(k.kind == "memcpy" for k in local)
        launch_counts.append({"rank": rank, "stream_kernels": priced_local - copies + custom - graph_count,
                              "graph_kernels": graph_count, "stream_memcpy_launch_charges": copies,
                              "custom_allreduce_kernels": custom})
        if simulate:
            rules = l2_sim.vllm_rules(cfg, phase, tp=tp, rank=rank, collective_buffers=mode)
            kwargs = {"gemm_min_tile_m": policy.gemm_min_tile_m, "steady_state": policy.steady_state, "address_plan": policy.address_plan}
            if policy.sharing_classes:
                kwargs.update(sharing_classes=True, physical_capacity_bytes=spec.physical_bytes,
                              private_copies=policy.private_copies)
                capacity = spec.shared_capacity_bytes

            res = l2_sim.simulate(ks, capacity, rules, operand_passes=timing._trace_gemm_passes(ks, name=_GEMM_NAME),
                                  **kwargs)
        activities, times, endpoint_offsets = {}, {}, {}
        attention_seen = set()
        for index, kernel in enumerate(ks):
            if kernel.kind == "collective":
                if kernel.name in ("embedding.all_reduce", "logits.gather"):
                    endpoint_offsets[kernel.op + "_f"] = times.get(kernel.op, 0.)
                continue
            op = _op(kernel)
            if mla and kernel.kind in ("batched_gemm", "attention"):
                seconds, activity = timing._price_mla_kernel(kernel, phase=phase, batch=int(batch), seq=seq)
            elif kernel.kind == "gemm":
                wb = gemm_weight_bytes(timing, kernel)
                seconds, activity = _gemm(timing, kernel.gemm) if wb is None else _gemm(timing, kernel.gemm, wb)
                if kernel.flops or len(kernel.reads) > 2:
                    epilogue = replace(kernel, kind="gemm_epilogue", reads=kernel.reads[2:], writes=(), gemm=None)
                    dt, da = timing._price_kernels([epilogue], record=False)
                    seconds, activity = seconds + dt, activity + da
            elif kernel.kind == "attention":
                layer = kernel.writes[0].tensor.split(".")[0]
                first = layer not in attention_seen
                attention_seen.add(layer)
                seconds, activity = (attn.compute_time - cache_write_time, attn_activity) if first else (0., ZERO_ACTIVITY)
            else:
                seconds, activity = timing._price_kernels([kernel], record=False)
            if simulate:
                activity = replace(activity, hbm=MemoryTraffic(res.hbm_read[index], res.hbm_write[index]))
            activities[op] = activities.get(op, ZERO_ACTIVITY) + activity
            times[op] = times.get(op, 0.) + seconds
        summaries.append({"rank": rank, "kernel_count": sum(k.kind != "collective" for k in ks),
                          "collective_count": sum(k.kind == "collective" and not k.name.endswith(".ipc_copy") for k in ks),
                          "activity": sum(activities.values(), ZERO_ACTIVITY).to_dict(),
                          "by_operation": {op: a.to_dict() for op, a in activities.items()},
                          "collective_buffers": mode})
        offsets.append(endpoint_offsets)
        if broadcast_prefix:
            times["embedding"] = times.get("embedding", 0.0) + broadcast_prefix
            endpoint_offsets["embedding_f"] = endpoint_offsets.get("embedding_f", 0.0) + broadcast_prefix
        ranks.append({op: a.scaled(1/layers if op in layer_ops else 1) for op, a in activities.items()})
        rank_times.append({op: t/(layers if op in layer_ops else 1) for op, t in times.items()})
        if key is not None:
            classes[key] = (launch_counts[-1], deepcopy(summaries[-1]), dict(offsets[-1]), dict(ranks[-1]),
                            dict(rank_times[-1]))
    for op in ranks[0]:
        old = transformer_timings.get(op, OperationTiming(op, DirectionTiming(0.)))
        records = tuple(row.get(op, ZERO_ACTIVITY) for row in ranks)
        durations = tuple(row.get(op, 0.) for row in rank_times)
        transformer_timings[op] = replace(old, forward=replace(old.forward, activity=records[0],
            compute_time=max(durations), notes={**old.forward.notes, "vtp_rank_activity": [a.to_dict() for a in records]}))
    timing._vtp_rank_activities = {op: tuple(row.get(op, ZERO_ACTIVITY) for row in ranks) for op in ranks[0]}
    timing._vtp_rank_durations = {op: tuple(row.get(op, 0.) for row in rank_times) for op in ranks[0]}
    for dest, src in (("embedding_f", "embedding"), ("linear_softmax_f", "linear_softmax")):
        timing._vtp_rank_activities[dest] = timing._vtp_rank_activities.pop(src)
        timing._vtp_rank_durations[dest] = timing._vtp_rank_durations.pop(src)
    timing._vtp_rank_activities["MLP"] = tuple(sum((row.get(op, ZERO_ACTIVITY) for op in ("ffn1", "gelu", "ffn2")), ZERO_ACTIVITY) for row in ranks)
    timing._vtp_rank_durations["MLP"] = tuple(sum(row.get(op, 0.) for op in ("ffn1", "gelu", "ffn2")) for row in rank_times)
    timing._vtp_endpoint_offsets = offsets
    timing._vtp_launch_counts = launch_counts
    timing.resolved_settings.setdefault("vtp", {})[phase] = {"tp": tp, "ranks": summaries,
        "batch": int(batch), "tokens": int(tokens), "past": int(past),
        "scope": "latest described template, including peak-memory probes; execution totals are phase_per_rank",
        "collective_buffers": mode, "collective_scope": settings.registry()["RAPID_L2_COLLECTIVE_BUFFERS"].source}
    if "input_broadcast" in ar_policy:
        timing.resolved_settings["vtp"][phase]["input_broadcast"] = dict(ar_policy["input_broadcast"])

def update_breakdown(timing, timings, breakdown):
    if not enabled(timing):
        return
    layer_ops = ("layernorm1", "qkv_proj", "rope", "attention", "output_proj", "layernorm2", "ffn1", "gelu", "ffn2")
    breakdown["transformer_time_f"] = sum(timings[op].total_forward_time() for op in layer_ops if op in timings)
    for op in ("embedding", "linear_softmax"):
        if op in timings:
            calls = getattr(timing, "_endpoint_collectives", {}).get(op + "_f", ())
            comm = sum(call.duration_seconds for call in calls)
            breakdown[op + "_f"] = timings[op].forward.compute_time + comm

            timing._vtp_rank_durations[op + "_f"] = tuple(t + comm for t in timing._vtp_rank_durations[op + "_f"])
            timing.__dict__.setdefault("_vtp_rank_collectives", {})[op + "_f"] = tuple(
                tuple(replace(call, offset_seconds=offsets[op + "_f"]) for call in calls)
                for offsets in timing._vtp_endpoint_offsets)

            timing._set_endpoint_collectives(op + "_f", timing._vtp_rank_collectives[op + "_f"][0])
