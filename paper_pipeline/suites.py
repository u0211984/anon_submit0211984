from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def _rows(p: Path):
    return list(csv.DictReader(open(p, newline="")))

def single_gpu(e: dict) -> list:
    d = ROOT / "paper_pipeline" / "inputs" / "single_gpu"
    runs = []
    for r in _rows(d / "cases.csv"):
        qwen = r["alias"].startswith("qwen")
        b, i, o = int(r["batch"]), int(r["input_tokens"]), int(r["output_tokens"])
        refs = [dict(quantity="prefill_energy", measured=r["measured_prefill_energy_j"], unit="J", predicted_from="prefill.total_j")]
        if o > 1:
            refs.append(dict(quantity="decode_energy", measured=r["measured_decode_energy_j"], unit="J",
                             predicted_from="decode.total_j", label="cumulative decode"))
        runs.append(dict(cell=r["cell"], description=f"{r['alias']} b{b} in{i} out{o}", figure="energy_validation",
                         panel="qwen" if qwen else f"llama_{r['profile']}", group=r["alias"],
                         phase="decode" if o > 1 else "prefill", hardware_entry="a100_pcie80_1065",
                         model_src=str(d / "model_configs" / f"{r['cell']}.yaml"), framework=e["framework"]["inference"],
                         batch=b, seq_len=i, refs=refs, model=r["alias"]))
    return runs

def qwen72b(e: dict) -> list:
    s = ROOT / "suite/historical_validation/inference/energy_latency_v6"
    base = {}
    for b in _rows(s / "data/baselines.csv"):
        base.setdefault(b["case_id"], []).append(b)
    q = {"prefill_energy": ("prefill.total_j", "J"), "decode_energy": ("decode.total_j", "J"),
         "prefill_latency": ("prefill.time_s", "s"), "decode_latency": ("decode.time_s", "s")}
    runs = []
    for c in _rows(s / "cases.csv"):
        refs = [dict(quantity=b["metric"], measured=b["measured_value"], unit=q[b["metric"]][1], predicted_from=q[b["metric"]][0],
                     label="F11 parked: prefill reference uncorrected" if b["metric"].startswith("prefill") else "")
                for b in base.get(c["case_id"], []) if b["metric"] in q]
        runs.append(dict(cell=c["case_id"], description=f"Qwen2.5-72B TP{c['tp']} b{c['batch_size']} in{c['input_seq_len']} out{c['output_seq_len']}",
                         figure="energy_validation", panel="qwen72b", group="qwen72b", phase="decode",
                         hardware_src=str(ROOT / c["hardware_config"]),
                         model_src=str(ROOT / e["workload"]["model_configs"] / Path(c["model_config"]).name),
                         framework=e["framework"]["inference"], live=("A100-SXM4-80GB", "paper", "inference"),
                         batch=int(c["batch_size"]), refs=refs, model="qwen25_72b", tp=int(c["tp"])))
    return runs

def _el_rescales():
    out = {}
    for fig, name in (("13", "el_f13_fig7b_reconciliation.json"), ("6", "el_f6_x2_reconciliation.json")):
        p = ROOT / "paper_pipeline/inputs/energylens" / name
        if not p.exists():
            raise FileNotFoundError(f"EnergyLens rescale factors missing: {p}")
        data = json.loads(p.read_text())
        out[fig] = ({int(d["batch"]): float(d["k_B"]) for d in data["per_batch"]}, name)
    return out

def el_f13_rescale_kB(value: float, batch: int, k: dict) -> float:
    return value * k[batch]

el_f6_rescale_x2 = el_f13_rescale_kB

def energylens(e: dict) -> list:
    s = ROOT / "paper_pipeline/inputs/energylens"
    pb = {}
    for b in _rows(s / "paper_baselines.csv"):
        if b["metric"] == "energy_per_request" and b["component"] == "total":
            pb.setdefault(b["case_id"], {})[b["series"]] = b
    resc = _el_rescales()
    runs = []
    for c in _rows(s / "cases.csv"):
        ser = pb.get(c["case_id"], {})
        if "measured" not in ser:
            continue
        m = float(ser["measured"]["value"])
        elv = float(ser["energylens"]["value"]) if "energylens" in ser else ""
        label = ""
        fig, b = str(c["figure"]).lstrip("0"), int(c["batch_size"])
        if fig in resc and c["phase"] == "decode":
            k, src = resc[fig]
            m = el_f13_rescale_kB(m, b, k)
            elv = el_f13_rescale_kB(elv, b, k) if elv != "" else ""
            label = f"rescaled ({'F12' if fig == '13' else 'F13'}, x{k[b]:.3f}, {src})"
        llama = "qwen3" not in c["case_id"]

        moe_rules = (e["workload"].get("moe_rules") or {}) if not llama else {}
        model_src = str(ROOT / c["model_config"])
        if moe_rules:
            model_src = str(ROOT / e["workload"]["moe_model_configs"] / Path(c["model_config"]).name)
        extra = {"implementation_options": moe_rules["implementation"]["options"]} if moe_rules.get("implementation") else {}
        runs.append(dict(cell=c["case_id"], description=f"{c['model']} {c['phase']} b{b} isl{c['input_seq_len']} osl{c['output_seq_len']} TP{c['tp']}",
                         figure="energylens", panel=f"F{fig}", group=f"{c['model']} {c['phase']}",
                         phase=c["phase"], hardware_src=str(ROOT / c["hardware_config"]), model_src=model_src,
                         framework=moe_rules.get("framework", e["framework"]["inference"]),
                         live=("A100-SXM4-80GB", "energylens", "inference"),
                         gemv_option=e["workload"]["rules"]["trtllm_gemv"] if llama else moe_rules.get("trtllm_gemv"),
                         batch=b, model=c["model"], **extra,
                         refs=[dict(quantity="energy_per_request", measured=m, unit="J", predicted_from=f"{c['phase']}.total_j/batch",
                                    label=label, reference_model_value=elv, reference_model_name="EnergyLens")]))
    return runs

def kareus(e: dict) -> list:
    s = ROOT / "suite/historical_validation/training/energy_latency"
    cases = {c["case_id"]: c for c in _rows(s / "cases.csv")}
    times = sorted((b for b in _rows(s / "data/baselines.csv") if b["metric"] == "fastest_point_time"),
                   key=lambda b: (int(b["figure"]), b["panel"]))
    curves = [p for p in _rows(s / "data/measured_points_raw.csv") if p["method"] == "Megatron-LM + Perseus"]
    import re as _re
    runs, seen = [], {}
    for bt in times:
        c = cases[bt["case_id"]]
        fig, pan = bt["figure"], bt["panel"]
        curve = sorted((float(p["time_s"]), float(p["energy_j"]))
                       for p in curves if p["figure"] == fig and p["panel"] == pan and p["workload_id"] == bt["workload_id"])
        dup = seen.get(bt["workload_id"])
        seen.setdefault(bt["workload_id"], f"{fig}{pan}")
        label = f"duplicate of {dup} (not counted, F15)" if dup else ""
        mm = _re.match(r"(llama|qwen)\w*?_tp(\d+)_cp(\d+)_mbs(\d+)_s(\d+)", c["case_id"])
        short = (f"{fig}{pan} {'L' if mm.group(1) == 'llama' else 'Q'} TP{mm.group(2)}/CP{mm.group(3)} "
                 f"{mm.group(5)}/{mm.group(4)}") if mm else f"{fig}{pan}"
        refs = [dict(quantity="train_energy_at_pred_time", measured="", unit="J", predicted_from="train.total_j",
                     curve=curve, label=label),
                dict(quantity="time_per_step", measured=bt["measured_value"], unit="s", predicted_from="train.time_s",
                     label=("fastest point; " + label).strip("; "))]
        model_src = ROOT / c["model_config"]
        if c["case_id"].startswith("qwen"):
            model_src = ROOT / e["workload"]["qwen_model_configs"] / model_src.name
        runs.append(dict(cell=f"{c['case_id']}_{fig}{pan}", description=short, figure="training", panel=f"{fig}{pan}",
                         group="kareus_training", phase="train", hardware_src=str(ROOT / c["hardware_config"]),
                         model_src=str(model_src), framework=e["framework"]["train"],
                         live=("A100-SXM4-40GB", "paper", "training"), inter_node="kareus_training",
                         batch=int(c["batch_size"]), refs=refs, model=c["case_id"].split("_tp")[0], duplicate=bool(dup)))
    return runs

LOADERS = {"single_gpu": single_gpu, "qwen72b": qwen72b, "energylens": energylens, "kareus": kareus}

def interpolate_at(curve, t):
    if not curve:
        return None, "no_curve"
    if t <= curve[0][0]:
        return curve[0][1], "fast_endpoint_clamp"
    if t >= curve[-1][0]:
        return curve[-1][1], "slow_endpoint_clamp"
    for (t0, e0), (t1, e1) in zip(curve, curve[1:]):
        if t0 <= t <= t1:
            return e0 + (e1 - e0) * (t - t0) / (t1 - t0) if t1 > t0 else e0, "interpolated"
    return None, "no_segment"
