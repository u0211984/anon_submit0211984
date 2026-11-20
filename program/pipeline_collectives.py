from program.ir import CollectiveOp, TransferOp

FOLDED_SECONDS = "pipeline_cp_collective_seconds"

def prepare_pipeline_collectives(program, network_model, hw):
    if (program.meta.misc.get("granularity") != "pipeline"
            or program.dp_count != 1 or "cp" in program.layout.axis_order):
        return
    calls = [op for op in program.ops if isinstance(op, CollectiveOp)
             and op.is_dp and "cp" in op.axes
             and op.participants > 1 and op.size_bytes > 0]
    if not calls:
        return

    layout = hw.network_layout
    dimension = layout.parallelism_map["cp"]

    formula = program.meta.misc.get("nccl_latency_rule") in ("tuning_formula", "tuning_protocol")
    if (formula and dimension is layout.parallelism_map.get("pp")
            and any(isinstance(op, TransferOp) and op.src_device != op.dst_device
                    for op in program.ops)):
        raise NotImplementedError("tuning_formula cannot share an AstraSim dimension with P2P traffic")

    bandwidth, latency = layout.link_for_parallelism("cp")
    durations = {}
    prices = {}
    for call in calls:
        key = (call.coll, call.size_bytes, call.participants)
        if key not in prices:
            prices[key] = float(network_model.collective(
                kind=call.coll, size_bytes=call.size_bytes,
                participants=call.participants, ib=bandwidth, ll=latency,
                local_bytes=0.0, local_ops=0.0, axis="cp",
                debug_label=f"{call.name}_pipeline_cp",
            ))
        durations[call.uid] = prices[key]
    program.meta.misc[FOLDED_SECONDS] = durations
