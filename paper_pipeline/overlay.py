from __future__ import annotations

import copy
from typing import Optional

FC = "FC"

def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst

def _dim_size(dim: dict, parallelism: dict) -> int:
    if isinstance(dim.get("size"), int):
        return int(dim["size"])
    n = 1
    for axis in dim.get("parallelisms") or []:
        v = parallelism.get(axis, 1)
        n *= int(v) if isinstance(v, (int, float)) else 1
    return n

def overlay_hardware(hw: dict, settings: dict, lit, live, inter_node: Optional[str] = None,
                     fixed_latency_key: bool = False, entry_id: Optional[str] = None, curve_entries=()):
    hw = copy.deepcopy(hw)
    applied, assumptions = [], []
    hw = lit.transform(hw, live)
    applied.append(f"a100_literature.transform({live.sku}, {live.tier}, {live.run_type})")
    inf = hw.get("inference") or {}
    if inf.pop("phase_network_overrides", None) is not None:
        applied.append("F2: phase_network_overrides removed")
    net = settings.get("network") or {}
    nw = hw.get("network") or {}
    if "overlap" in nw:
        nw["overlap"] = dict(net.get("overlap") or {"tp_overlap": 0.0, "tp_sp_overlap": 0.0, "cp_overlap": 0.0})
        applied.append("F3: overlaps 0")
    par = hw.get("parallelism") or {}
    multi = False
    for dim in nw.get("dimensions") or []:
        n = _dim_size(dim, par)
        topo = dim.setdefault("topology", {})
        if lit._is_nvlink(dim):
            if n > 1:
                multi = True

            kind = str((net.get("topology_by_entry") or {}).get(entry_id, net.get("topology", "FullyConnected")))
            if kind == "FullyConnected":
                topo["type"] = FC
                topo["bandwidth"] = float(net["nvlink_bandwidth_Bps"]) / max(1, n - 1)
            elif kind == "Switch":
                topo["type"] = "Switch"
                topo["bandwidth"] = float(net["nvlink_bandwidth_Bps"])
            else:
                raise ValueError(f"network.topology {kind!r}: FullyConnected or Switch")
            topo["latency"] = float(net["per_step_latency_s"])
            topo["util"] = float(net["util"])
            topo["energy_per_bit"] = float(net["energy_per_bit"])
            payload_policy = net.get("nvlink_link_energy_rule_by_entry") or {}
            if entry_id in payload_policy:
                topo["link_energy_rule"] = payload_policy[entry_id]
            if entry_id in (net.get("nvlink_physical_inputs_for") or []):
                for key in ("links_per_gpu", "lanes_per_link", "lane_rate_bps"):
                    topo[key] = float(net[key])
            dim.pop("collective_override", None)
            applied.append(f"{dim.get('id')}: C1-C3, C6 NVLink {topo['type']} (n={n}, {topo['bandwidth']:.4g} B/s "
                           f"{'per peer link' if topo['type'] == FC else 'per GPU port'})")
        elif inter_node:
            spec = (net.get("inter_node") or {}).get(inter_node)
            if spec is None:
                raise ValueError(f"no inter-node definition {inter_node!r} in settings network.inter_node")
            multi = multi or n > 1
            topo.update(type=FC, bandwidth=float(spec["bandwidth_Bps_per_gpu"]), latency=float(spec["per_step_latency_s"]),
                        util=float(net["util"]), energy_per_bit=float(spec["energy_per_bit"]))
            if inter_node == "kareus_training":
                topo["link_energy_rule"] = spec["link_energy_rule"]
            dim.pop("collective_override", None)
            applied.append(f"{dim.get('id')}: inter-node {inter_node}")
            assumptions.append("EFA p4d EFA link (6.25 GB/s/GPU, 164 us/step); 8 pJ/bit is an assumption without a source"
                               if inter_node == "kareus_training" else
                               "C12 InfiniBand energy per bit 20 pJ/bit is an assumption")
    if multi and fixed_latency_key:
        hw.setdefault("sw_param", {})["collective_launch_overhead"] = float(net["collective_launch_overhead_s"])
        applied.append("C1: sw_param.collective_launch_overhead 8.4 us")
    elif multi:
        assumptions.append("C1 8.4 us per-collective launch overhead not applied (sw_param.collective_launch_overhead not merged yet)")
    if multi:
        nw["link_energy_rule"] = net["link_energy_rule"]
        hw.setdefault("sw_param", {})["nccl_latency_rule"] = net["nccl_latency_rule"]
        applied.append(f"network.link_energy_rule {net['link_energy_rule']}; nccl_latency_rule {net['nccl_latency_rule']}")

    for key in ("allreduce_model", "allreduce_curve", "allreduce_include_unverified", "allreduce_timing_basis"):
        hw.setdefault("sw_param", {}).pop(key, None)
        if multi and entry_id in curve_entries:
            hw["sw_param"][key] = net[key]
    if multi and entry_id in curve_entries:
        applied.append("NCCL_CURVE: registry-scoped measured all-reduce duration")

    hw.setdefault("sw_param", {}).pop("allreduce_rsag_rule", None)
    if (multi and entry_id in curve_entries and live.run_type == "training"
            and net.get("allreduce_rsag_rule", "network_model") != "network_model"):
        hw["sw_param"]["allreduce_rsag_rule"] = net["allreduce_rsag_rule"]
        applied.append(f"NCCL_CURVE_SCOPE.extension_rsag: allreduce_rsag_rule {net['allreduce_rsag_rule']}")
    sw = hw.get("sw_param") or {}
    if sw.get("gemm_activity_backend") == "timing":
        sw.pop("gemm_activity_backend")
        applied.append("sw_param.gemm_activity_backend 'timing' dropped (obsolete)")
    eb = hw.get("execution_backend")
    if multi and isinstance(eb, dict):
        eb["model"] = net.get("execution_backend", "astrasim")
        astrasim = eb.setdefault("astrasim", {})
        astrasim["collectives"] = {k: net.get("collective_algorithm", "auto") for k in (astrasim.get("collectives") or {})}
        astrasim.setdefault("sys_options", {})["active_chunks_per_dimension"] = int(net["active_chunks_per_dimension"])
        applied.append("C3-C5: astrasim, collectives auto, 8 chunks")
    _merge(hw, settings.get("hardware") or {})
    applied.append("settings hardware block")
    return hw, applied, assumptions

def overlay_model(model: dict, settings: dict, framework: str, gemv_option: Optional[dict] = None):
    model = copy.deepcopy(model)
    mp = model["model_param"]
    mp.setdefault("attention", {})["use_rope"] = True
    impl = copy.deepcopy((settings.get("model") or {}).get("implementation") or {})
    if framework:
        impl["framework"] = framework
    if gemv_option:
        impl.setdefault("gemm_library", {})["gemv"] = dict(gemv_option)
    old = mp.get("implementation")
    if isinstance(old, str) and old not in ("fused", "eager"):
        raise ValueError(f"unexpected implementation {old!r}")
    mp.pop("framework", None)
    mp["implementation"] = impl
    applied = ["C7 use_rope", f"implementation (settings) + framework {framework}"]
    if mp.get("model_type") == "qwen2" and "qwen2_bias_mode" in (settings.get("model") or {}):
        mp["qwen2_bias_mode"] = settings["model"]["qwen2_bias_mode"]
        applied.append("Qwen2 bias dispatch (settings, designation QWEN2_BIAS)")

    model_settings = settings.get("model") or {}
    is_moe = (mp.get("model_type") in {"qwen3_moe", "mixtral", "deepseek_moe", "deepseek_v2", "deepseek_v3"}
              or int((mp.get("moe") or {}).get("num_experts", mp.get("num_experts", 1))) > 1)
    if is_moe and "moe_tile_rule" in model_settings:
        mp["moe_tile_rule"] = model_settings["moe_tile_rule"]
        applied.append("MoE tile dispatch (settings, designation MOE_TILE)")
    source_moe = (mp.get("model_type") == "qwen3_moe"
                  or (mp.get("model_type") in {"deepseek_v2", "deepseek_v3"}
                      and int((mp.get("moe") or {}).get("num_experts", 1)) > 1))
    if source_moe and "moe_source_collective_rule" in model_settings:
        mp["moe_source_collective_rule"] = model_settings["moe_source_collective_rule"]
        applied.append("MoE all-reduce timing (settings, designation MOE_ALLREDUCE_NETWORK)")
    return model, applied + (["C10 gemv option"] if gemv_option else [])
