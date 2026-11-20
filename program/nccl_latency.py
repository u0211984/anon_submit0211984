from timing_model import CollectiveType
from decimal import Decimal

PROTOCOL_SOURCE = (
    "NCCL v2.27.7 src/graph/tuning.cc:132-145 Ring baseLat us=(6.6,14.0,8.4), "
    "NVLINK hwLat us=(0.6,1.9,3.4); :158-174 Ampere single-node llMaxBw=87.7 GB/s, "
    "perChMaxRingLL128Bw=20 GB/s; :250-252 nsteps; :266-280 busBw and LL 0.5 / LL128 0.92 caps; "
    ":320-355 algBw=nRanks/nsteps and latency; :457-473 LL128 enable; :561-587 time; "
    "src/graph/topo.h:18 SM80_NVLINK_BW=20 GB/s. Single-node homogeneous SM80 NVLink contract; "
    "graph product=declared links_per_gpu*20 GB/s; nChannels from sw_param.nccl_tuning.n_channels "
    "or equivalent links-per-channel graph (not measured NCCL graph); "
    "Wire flags: src/include/device.h:71-83 ncclLLFifoLine (data/flag pairs), "
    ":106-108 NCCL_LL128_DATAELEMS/NCCL_LL128_LINEELEMS (15/16)."
)

def protocol_candidates(kind, participants, size_bytes, *, links, ll128_enabled=True, n_channels=None):
    import math
    n = int(participants)
    if n != participants or n <= 1 or size_bytes <= 0:
        raise ValueError("tuning_protocol requires positive bytes and at least two ranks")
    if not math.isfinite(links) or links <= 0 or int(links) != links:
        raise ValueError("tuning_protocol requires integral positive links_per_gpu")
    channels = links if n_channels is None else n_channels
    if not math.isfinite(channels) or channels <= 0 or int(channels) != channels:
        raise ValueError("tuning_protocol requires integral positive n_channels")
    if kind not in (CollectiveType.ALL_REDUCE, CollectiveType.ALL_GATHER,
                    CollectiveType.REDUCE_SCATTER):
        raise NotImplementedError(f"tuning_protocol unsupported collective {kind}")
    steps = (2 * (n - 1) if kind is CollectiveType.ALL_REDUCE else
             n - 1 if kind in (CollectiveType.ALL_GATHER, CollectiveType.REDUCE_SCATTER) else n)
    bus = links * 20e9
    rows = []
    for protocol, base_us, step_us, bandwidth, wire in (
            ("LL", 6.6, .6, min(87.7e9, .5 * bus), 2.),
            ("LL128", 14., 1.9, min(.92 * bus, channels * 20e9), 16 / 15),
            ("Simple", 8.4, 3.4, bus, 1.)):
        if protocol == "LL128" and not ll128_enabled:
            continue
        latency = (base_us + steps * step_us) * 1e-6
        alg_bw = bandwidth * n / steps
        transfer = float(size_bytes) / alg_bw
        rows.append(dict(protocol=protocol, latency_seconds=latency,
                         transfer_seconds=transfer, total_seconds=latency + transfer,
                         wire_multiplier=wire, alg_bandwidth_bytes_per_second=alg_bw,
                         bus_bandwidth_bytes_per_second=bandwidth, nsteps=steps,
                         nChannels=int(channels), bwIntra_GBs=links*20./channels, ll128_enabled=ll128_enabled,
                         source=PROTOCOL_SOURCE))
    return rows

def _protocol_params(hw, axis, participants):
    params = dimension_params(hw, axis, participants)
    dim = hw.network_layout.parallelism_map[params["axis"]]

    links = getattr(dim, "links_per_gpu", None)
    if links is None:
        raise ValueError("tuning_protocol requires topology.links_per_gpu for the single-node A100 NVLink contract")
    if getattr(hw.network_layout, "faulty_links", ()):
        raise NotImplementedError("tuning_protocol does not model faulty NVLink graphs")
    declaration = getattr(getattr(hw, "sw_config", None), "nccl_tuning", {})
    unknown = set(declaration) - {"n_channels", "node_count", "min_comp_cap", "max_comp_cap", "type_intra", "type_inter"}
    if unknown:
        raise ValueError(f"unknown sw_param.nccl_tuning keys: {sorted(unknown)}")
    nodes = declaration.get("node_count", 1)
    min_cc = declaration.get("min_comp_cap", 80)
    max_cc = declaration.get("max_comp_cap", 80)
    if nodes != 1 or min_cc != 80 or max_cc != 80:
        raise NotImplementedError("tuning_protocol requires a homogeneous single-node SM80 declaration")
    intra = declaration.get("type_intra", "NVL")
    inter = declaration.get("type_inter", "LOC")
    if intra not in ("NVL", "NVB"):
        raise NotImplementedError("tuning_protocol requires NVLink intra paths (NVL/NVB)")

    ll128 = inter in ("LOC", "NVL", "NVB", "PIX", "PXB")
    if inter not in ("LOC", "NVL", "NVB", "PIX", "PXB", "PXN", "PHB", "SYS", "NET"):
        raise ValueError("unknown NCCL type_inter path")
    params.update(links=float(links), n_channels=declaration.get("n_channels", int(links)),
                  ll128_enabled=ll128, source=PROTOCOL_SOURCE,
                  topology_contract=dict(node_count=nodes, min_comp_cap=min_cc, max_comp_cap=max_cc,
                                         type_intra=intra, type_inter=inter))
    return params

def protocol_model(hw, axis, kind, participants, size_bytes):
    params = _protocol_params(hw, axis, participants)
    if params["algorithms"][kind.value] != "ring":
        raise NotImplementedError("tuning_protocol implements ring only")

    model_bytes = size_bytes * participants if kind is CollectiveType.ALL_GATHER else size_bytes
    result = min(protocol_candidates(kind, participants, model_bytes, links=params["links"],
                                   ll128_enabled=params["ll128_enabled"], n_channels=params["n_channels"]), key=lambda row: row["total_seconds"])
    return dict(result, model_bytes=model_bytes, topology_contract=params["topology_contract"])

def protocol_for_call(program, call):
    table = program.meta.misc["nccl_latency_params"]
    params = table.get(f"{call.interconnect}:{call.participants}", table.get(call.interconnect))
    if params is None:
        raise NotImplementedError(f"tuning_protocol missing dimension for {call.interconnect!r}")
    if params["algorithms"][call.coll.value] != "ring":
        raise NotImplementedError("tuning_protocol implements ring only")
    model_bytes = call.size_bytes * call.participants if call.coll is CollectiveType.ALL_GATHER else call.size_bytes
    result = min(protocol_candidates(call.coll, call.participants, model_bytes,
                 links=params["links"], ll128_enabled=params["ll128_enabled"], n_channels=params.get("n_channels")),
                 key=lambda row: row["total_seconds"])
    result = dict(result, model_bytes=model_bytes)
    if "topology_contract" in params:
        result = dict(result, topology_contract=params["topology_contract"])
    identity = str(call.uid) if hasattr(call, 'uid') else f"embedded:{call.name}:{call.interconnect}:{call.participants}:{call.size_bytes}"
    program.meta.misc.setdefault("nccl_protocol_selections", {})[identity] = dict(
        result, label=call.name, participants=call.participants, size_bytes=call.size_bytes,
        axis=call.interconnect)
    return result

def resolve_dimension(hw, axis):
    layout = getattr(hw, "network_layout", None)
    if axis is None and len(getattr(layout, "dimensions", ())) == 1:
        axis = layout.dimensions[0].parallelisms[0]
    dim = getattr(layout, "parallelism_map", {}).get(axis)
    if axis == "dp":
        sch = getattr(hw, "sch_config", None)
        dp = int(getattr(getattr(sch, "train", None), "dp", 1) or 1)
        cp = int(getattr(sch, "cp", 1) or 1)
        if dp == 1 and cp > 1:
            dim = getattr(layout, "parallelism_map", {}).get("cp")
            axis = "cp"
    if dim is None:
        raise NotImplementedError(f"tuning_formula requires one network dimension for axis {axis!r}")
    return axis, dim

def dimension_params(hw, axis, participants, *, algorithm_ranks=None):
    from astrasim_lib.config_generation import _normalize_topology_name, resolve_collective_for_dimension
    axis, dim = resolve_dimension(hw, axis)
    topo = _normalize_topology_name(dim.topology_type)
    if topo not in ("Ring", "FullyConnected", "Switch"):
        raise NotImplementedError(f"tuning_formula does not support topology {topo}")
    algorithm_ranks = participants if algorithm_ranks is None else algorithm_ranks
    if topo == "Ring" and algorithm_ranks <= 2:
        topo = "FullyConnected"
    backend = getattr(hw, "execution_backend", None)
    astrasim = getattr(backend, "astrasim", None)
    coll = getattr(astrasim, "collectives", None)
    algorithms = {}
    for kind in CollectiveType:
        if kind is CollectiveType.PIPELINE:
            continue

        algorithms[kind.value] = ("ring" if (kind is CollectiveType.GATHER or
            str(getattr(backend, "model", "analytical")).lower() == "analytical") else
            resolve_collective_for_dimension(hw, dim, topo, kind.value.replace("_", "-"),
                getattr(coll, kind.value, "auto"), algorithm_ranks, [axis], force_nccl_ring=True))
    return {"axis": axis, "latency": float(dim.latency), "algorithms": algorithms}

def formula_seconds(base, kind, participants, latency, algorithm, *, exact=False):
    if algorithm != "ring":
        raise NotImplementedError(f"tuning_formula NCCL latency algorithm {algorithm!r}; only ring is implemented")
    n = int(participants)
    steps = (2 * (n - 1) if kind is CollectiveType.ALL_REDUCE else
             n - 1 if kind in (CollectiveType.ALL_GATHER, CollectiveType.REDUCE_SCATTER) else n)
    result = Decimal(str(base)) + steps * Decimal(str(latency))
    return result if exact else float(result)

def launch_seconds(program, call, *, exact=False):
    base = float(program.meta.misc.get("collective_launch_overhead", 0.0))
    if (program.meta.misc.get("nccl_latency_rule") == "tuning_protocol"
            and call.coll is not CollectiveType.GATHER):
        seconds = protocol_for_call(program, call)["latency_seconds"]
        return Decimal(str(seconds)) if exact else seconds
    if program.meta.misc.get("nccl_latency_rule", "astrasim_per_hop") not in ("tuning_formula", "tuning_protocol"):
        return base
    table = program.meta.misc["nccl_latency_params"]
    params = table.get(f"{call.interconnect}:{call.participants}", table.get(call.interconnect))
    if params is None:
        raise NotImplementedError(f"tuning_formula missing dimension for {call.interconnect!r}")
    return formula_seconds(base, call.coll, call.participants, params["latency"],
                           params["algorithms"][call.coll.value], exact=exact)

def enabled(program):
    return bool(program.meta.misc.get("collective_launch_overhead", 0.0) or
                program.meta.misc.get("nccl_latency_rule") in ("tuning_formula", "tuning_protocol"))

def stamp(program, hw):
    from sim_settings import nccl_latency_rule
    rule = nccl_latency_rule(hw)
    if rule not in ("tuning_formula", "tuning_protocol"):
        return
    program.meta.misc["nccl_latency_rule"] = rule

    from program.ir import CollectiveOp, ComputeOp
    from program.collective_launch import is_nccl_call
    from program.vllm_allreduce import op_choice
    params = {}
    for op in program.ops:
        calls = (op,) if isinstance(op, CollectiveOp) else op.embedded_collectives if isinstance(op, ComputeOp) else ()
        for call in calls:
            if is_nccl_call(call.coll, participants=call.participants, size_bytes=call.size_bytes,
                            algorithm=op_choice(program, call).algorithm):
                key = f"{call.interconnect}:{call.participants}"
                native = isinstance(call, CollectiveOp) or (
                    call.coll is CollectiveType.GATHER and program.meta.misc.get("granularity") != "pipeline")
                algorithm_ranks = len(program.compute_devices()) * max(1, program.dp_count) if native else call.participants
                resolved = dimension_params(hw, call.interconnect, call.participants,
                                            algorithm_ranks=algorithm_ranks)
                if rule == "tuning_protocol":
                    resolved.update(_protocol_params(hw, call.interconnect, call.participants))
                if key in params and params[key] != resolved:
                    raise NotImplementedError(
                        f"tuning_formula ambiguous algorithms: {params[key]['algorithms']} versus {resolved['algorithms']}")
                params[key] = resolved
    program.meta.misc["nccl_latency_params"] = params
    if rule == "tuning_protocol":
        for op in program.ops:
            calls = (op,) if isinstance(op, CollectiveOp) else op.embedded_collectives if isinstance(op, ComputeOp) else ()
            for call in calls:
                if call.coll is not CollectiveType.GATHER and is_nccl_call(call.coll, participants=call.participants, size_bytes=call.size_bytes,
                                algorithm=op_choice(program, call).algorithm):
                    protocol_for_call(program, call)
