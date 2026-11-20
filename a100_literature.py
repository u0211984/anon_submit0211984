from __future__ import annotations

import csv
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent

REFERENCE_CLOCK_HZ = 1.41e9
VF_SCALE = {1.065e9: 0.633, 1.275e9: 0.816, 1.41e9: 1.0}
VF_VOLT_MV = {1.065e9: 725.0, 1.275e9: 825.0, 1.41e9: 913.0}
PJ_PER_BYTE_1410 = {"DRAM": 104.88, "SRAM-L2": 37.68, "SRAM-L1": 12.72}

HBM_CONTROL_PJ_PER_BIT_1410 = 8.47
HBM_DATAPATH_PJ_PER_BIT = 4.64
REGISTER_PJ_PER_BYTE = 0.0
TENSOR_PJ_PER_FLOP_1410 = 0.70
VECTOR_PJ_PER_FLOP_1410 = 12.97
PADDED_TENSOR_PJ_PER_FLOP_1410 = 0.37
STATIC_POWER_W = 54.0
BUSY_POWER_W = 0.0
L2_BANDWIDTH_1410 = 4.3e12
L1_BANDWIDTH_1410 = 14.8e12
MEMORY_BANDWIDTH_FRACTION = 0.8
NETWORK_UTIL_SXM = 0.8
MATRIX_UTIL = 1.0
LAUNCH_FLOOR_S = 5e-6
DECODE_STEP_OVERHEAD_S = 0.72e-3
PREFILL_OVERHEAD_S = 1.59e-3
CUBLAS_GEMV_MAX_N = 2304
ENERGYLENS_CLOCK_HZ = 1.41e9
ENERGYLENS_NVSWITCH_PER_GPU_BPS = 300e9
DRAM_SPEC_BPS = {
    "A100-PCIe-80GB": 1.935e12,
    "A100-SXM4-80GB": 2.039e12,
    "A100-SXM4-40GB": 1.555e12,
}
SECTOR_BYTES = 32
NVLINK_PJ_PER_BIT = 8.0
NVLINK_PJ_PER_BIT_SENSITIVITY = (5.0, 10.0)

FITTED_NETWORK_KEYS = ("system_power_w", "static_power_per_link_w")

def _r(value: float) -> float:
    return float(f"{value:.12g}")

def vf_scale(clock_hz: float) -> float:
    f = float(clock_hz)
    for fc, s in VF_SCALE.items():
        if abs(f - fc) < 1.0:
            return s
    pts = sorted(VF_SCALE)
    for lo, hi in zip(pts, pts[1:]):
        if lo < f < hi:
            v = VF_VOLT_MV[lo] + (VF_VOLT_MV[hi] - VF_VOLT_MV[lo]) * (f - lo) / (hi - lo)
            w = (v * v - VF_VOLT_MV[lo] ** 2) / (VF_VOLT_MV[hi] ** 2 - VF_VOLT_MV[lo] ** 2)
            return VF_SCALE[lo] + (VF_SCALE[hi] - VF_SCALE[lo]) * w
    raise ValueError(f"clock {f:.6g} Hz is outside the cited V-f range {pts[0]:.6g}-{pts[-1]:.6g} Hz")

def per_byte_energy(level: str, clock_hz: float) -> float:
    if level == "SRAM-R":
        return _r(REGISTER_PJ_PER_BYTE * 1e-12)
    return _r(PJ_PER_BYTE_1410[level] * 1e-12 * vf_scale(clock_hz))

def core_energy(clock_hz: float) -> Dict[str, float]:
    s = vf_scale(clock_hz)
    return {
        "nominal_energy_per_flop": _r(TENSOR_PJ_PER_FLOP_1410 * 1e-12 * s),
        "tensor_energy_per_flop": _r(TENSOR_PJ_PER_FLOP_1410 * 1e-12 * s),
        "vector_energy_per_flop": _r(VECTOR_PJ_PER_FLOP_1410 * 1e-12 * s),
        "padded_tensor_energy_per_flop": _r(PADDED_TENSOR_PJ_PER_FLOP_1410 * 1e-12 * s),
    }

def launch_floor_cycles(clock_hz: float) -> int:
    return int(round(LAUNCH_FLOOR_S * float(clock_hz)))

@dataclass(frozen=True)
class Live:
    pattern: str
    sku: str
    tier: str
    run_type: str = ""

LIVE: Tuple[Live, ...] = (
    Live("configs/hardware-config/a100_80GB*.yaml", "A100-SXM4-80GB", "prices"),
    Live("validation_scripts/validation_configs/hardware-config/a100_80GB*.yaml", "A100-SXM4-80GB", "prices"),
    Live("validation_scripts/validation_configs/hardware-config/A100_SXM4_80GB_*.yaml", "A100-SXM4-80GB", "prices"),
    Live("webui/config_seeds/hardware/A100_SXM4_80GB.yaml", "A100-SXM4-80GB", "prices"),
    Live("webui/config_seeds/hardware/A100_PCIe_80GB.yaml", "A100-PCIe-80GB", "prices"),

    Live("suite/historical_validation/inference/traffic_flops_v6/configs/*.hardware.yaml", "A100-PCIe-80GB", "paper", "inference"),
    Live("suite/historical_validation/inference/energy_latency_v6/configs/*.hardware.yaml", "A100-SXM4-80GB", "paper", "inference"),
    Live("suite/historical_validation/training/energy_latency/configs/*.hardware.yaml", "A100-SXM4-40GB", "paper", "training"),
    Live("paper_pipeline/inputs/energylens/configs/*.hardware.yaml", "A100-SXM4-80GB", "energylens", "inference"),
    Live("suite/training-energy-validation-kareus/templates/A100_SXM4_80GB_8gpu_collective_inferred_1275mhz.yaml",
         "A100-SXM4-80GB", "paper", "inference"),

    Live("paper_pipeline/hardware/base/a100_80gb_pcie_1065mhz_paper_v6.yaml", "A100-PCIe-80GB", "paper", "inference"),
    Live("paper_pipeline/hardware/base/a100_80gb_pcie_1065mhz_paper_v6_train_*.yaml", "A100-PCIe-80GB", "paper", "training"),

    Live("paper_pipeline/hardware/base/a100_sxm4_80gb_400w_inference.yaml", "A100-SXM4-80GB", "paper", "inference"),
)

FROZEN: Tuple[Tuple[str, str], ...] = (
    ("outputs/*", "historical run outputs"),
    ("configs/legacy/*", "legacy RAPID-LLM config (3.024 GHz, not an A100 V-f point)"),
    ("opmodel/*", "op-model's own hardware schema; RAPID energy pricing does not read it"),
    ("probe_work/*", "scratch probes"),
    ("tests/fixtures/audit_train/*", "training energy audit: the paper training config, pinned as the "
                                     "input of the accounting regressions (tests/test_train_energy_audit.py)"),
    ("*/data/original_configs/*", "byte copies of the configs a study was derived from"),
    ("*/data/source/*", "source snapshots"),
    ("suite/historical_validation/inference/traffic_flops/configs/*", "retained batches (pre-v6)"),
    ("suite/historical_validation/inference/traffic_flops_v[345]*/configs/*", "retained batches (pre-v6)"),
    ("suite/historical_validation/inference/traffic_flops_step/configs/*", "retained batches"),
    ("suite/historical_validation/inference/traffic_flops_v6lit*/configs/*", "lit-stage experiment copies (lit1..lit6)"),
    ("suite/historical_validation/inference/energy_latency/configs/*", "retained batches (pre-v6)"),
    ("suite/historical_validation/inference/energy_latency_v[35]/configs/*", "retained batches (pre-v6)"),
    ("suite/historical_validation/inference/energy_latency_step/configs/*", "retained batches"),
    ("suite/historical_validation/inference/energy_latency_v6lit*/configs/*", "lit-stage experiment copies"),
    ("suite/historical_validation/training/energy_latency_*/configs/*", "lit7p / trainfix ladder study copies"),
    ("suite/energylens/configs/*", "retained EnergyLens batches (v1)"),
    ("suite/energylens_step/configs/*", "retained EnergyLens batches"),
    ("suite/energylens_v[35]/configs/*", "retained EnergyLens batches"),
    ("suite/energylens_v6/configs_lit*/*", "lit-stage EnergyLens experiment copies"),
    ("suite/energylens_v6_*/*", "EnergyLens config backups"),
    ("suite/training-energy-validation-kareus/configs/*", "retained Kareus batch inputs"),
    ("suite/hardware_execution_*/configs/*", "retained hardware-execution analysis batches"),
    ("suite/energy_latency_decoupling/*/configs/*", "retained decoupling analysis batches"),
    ("suite/energy_latency_decoupling/*/*/configs/*", "retained decoupling analysis batches"),
)

def _is_a100(text: str) -> bool:
    return re.search(r"^\s*num_bundles:\s*108\s*$", text, re.M) is not None and "tech_param" in text

def tracked_a100_configs() -> List[str]:
    out = subprocess.run(["git", "ls-files", "*.yaml", "*.yml"], cwd=ROOT, capture_output=True, text=True, check=True)
    paths = []
    for rel in out.stdout.split():
        p = ROOT / rel
        if p.is_file() and _is_a100(p.read_text(errors="replace")):
            paths.append(rel)
    return sorted(paths)

def classify(rel: str) -> Tuple[Optional[Live], Optional[str]]:
    for live in LIVE:
        if fnmatch.fnmatch(rel, live.pattern):
            return live, None
    for pattern, reason in FROZEN:
        if fnmatch.fnmatch(rel, pattern):
            return None, reason
    return None, None

def _memory_energy(block: Dict[str, Any]) -> Optional[float]:
    if block.get("dynamic_energy_per_byte") is not None:
        return float(block["dynamic_energy_per_byte"])
    if block.get("dynamic_energy_per_sector") is not None:
        return float(block["dynamic_energy_per_sector"]) / float(block["sector_size_bytes"])
    if block.get("dynamic_energy_per_bit") is not None:
        return float(block["dynamic_energy_per_bit"]) * 8.0
    return None

def _to_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    m = re.fullmatch(r"([0-9.eE+-]+)\s*([KMGT]?)B?", text)
    if not m:
        return float(text)
    return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[m.group(2)]

def expectations(hw: Dict[str, Any], live: Live) -> List[Tuple[str, Any, float]]:
    tp, core = hw["tech_param"], hw["tech_param"]["core"]
    clock = ENERGYLENS_CLOCK_HZ if live.tier == "energylens" else float(core["operating_frequency"])
    rows: List[Tuple[str, Any, float]] = []
    if live.tier == "energylens":
        rows.append(("core.operating_frequency", core.get("operating_frequency"), ENERGYLENS_CLOCK_HZ))
    for key, value in core_energy(clock).items():
        rows.append((f"core.{key}", core.get(key), value))
    for level in ("DRAM", "SRAM-L2", "SRAM-L1", "SRAM-R"):
        rows.append((f"{level}.energy_per_byte", _memory_energy(tp[level]), per_byte_energy(level, clock)))
    rows.append(("static_power_w", hw.get("static_power_w"), STATIC_POWER_W))
    rows.append(("busy_power_w", hw.get("busy_power_w", 0.0), BUSY_POWER_W))
    for dim in (hw.get("network") or {}).get("dimensions") or []:
        topo = dim.get("topology") or {}
        for key in FITTED_NETWORK_KEYS:
            rows.append((f"network.{dim.get('id')}.{key}", float(topo.get(key, 0.0) or 0.0), 0.0))
        if _is_nvlink(dim):
            rows.append((f"network.{dim.get('id')}.energy_per_bit", topo.get("energy_per_bit"), NVLINK_PJ_PER_BIT * 1e-12))
    if live.tier == "prices":
        return rows
    f_over = clock / REFERENCE_CLOCK_HZ
    rows += [
        ("core.matrix_unit.util", core["matrix_unit"].get("util"), MATRIX_UTIL),
        ("core.device_fixed_overhead_cycles", core.get("device_fixed_overhead_cycles"), launch_floor_cycles(clock)),
        ("SRAM-L2.bandwidth*util", _to_float(tp["SRAM-L2"]["bandwidth"]) * float(tp["SRAM-L2"].get("util", 1.0)), L2_BANDWIDTH_1410 * f_over),
        ("SRAM-L1.bandwidth*util", _to_float(tp["SRAM-L1"]["bandwidth"]) * float(tp["SRAM-L1"].get("util", 1.0)), L1_BANDWIDTH_1410 * f_over),
        ("DRAM.bandwidth*util", _to_float(tp["DRAM"]["bandwidth"]) * float(tp["DRAM"].get("util", 1.0)),
         MEMORY_BANDWIDTH_FRACTION * DRAM_SPEC_BPS[live.sku]),
    ]
    for level in ("DRAM", "SRAM-L2", "SRAM-L1"):
        if tp[level].get("latency_cycles") is not None:
            rows.append((f"{level}.latency", float(tp[level]["latency"]), float(tp[level]["latency_cycles"]) / clock))
    if live.run_type == "inference":
        sw = hw.get("sw_param", {})
        rows.append(("sw_param.inference_step_overhead", sw.get("inference_step_overhead"), DECODE_STEP_OVERHEAD_S))
        rows.append(("sw_param.inference_prefill_overhead", sw.get("inference_prefill_overhead"), PREFILL_OVERHEAD_S))
    if live.sku.startswith("A100-SXM4"):
        for dim in (hw.get("network") or {}).get("dimensions") or []:
            rows.append((f"network.{dim['id']}.util", dim["topology"].get("util"), NETWORK_UTIL_SXM))
    if live.tier == "energylens":
        n = int(hw.get("parallelism", {}).get("tp", 1))
        dim0 = hw["network"]["dimensions"][0]
        if n > 1:
            topo = dim0["topology"]
            rows.append(("network.dim0.type", topo.get("type"), "FC"))
            rows.append(("network.dim0.per_gpu_bandwidth", _to_float(topo["bandwidth"]) * (n - 1), ENERGYLENS_NVSWITCH_PER_GPU_BPS))
            for kind in ("all_reduce", "all_gather", "reduce_scatter"):
                rows.append((f"network.dim0.{kind}", (dim0.get("collective_override") or {}).get(kind), "direct"))
    return rows

def _is_nvlink(dim: Dict[str, Any]) -> bool:
    return "tp" in (dim.get("parallelisms") or [])

def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, str):
        return str(actual) == expected
    if actual is None:
        return False
    try:
        a = float(actual)
    except (TypeError, ValueError):
        return False
    return abs(a - float(expected)) <= 1e-9 * max(1.0, abs(float(expected))) if expected else abs(a) <= 1e-30

def check_config(rel: str, live: Live) -> List[str]:
    hw = yaml.safe_load((ROOT / rel).read_text())
    return [f"{rel}: {name} = {actual!r}, literature {expected!r}"
            for name, actual, expected in expectations(hw, live) if not _matches(actual, expected)]

def check_training_overlay() -> List[str]:
    study = ROOT / "suite/historical_validation/training/energy_latency"
    bad = [f"{study.name}/cases.csv {r['case_id']}: static_overlay_w_per_gpu = {r['static_overlay_w_per_gpu']}"
           for r in csv.DictReader((study / "cases.csv").open(newline=""))
           if float(r["static_overlay_w_per_gpu"]) != STATIC_POWER_W]
    scoring = json.loads((study / "setup.json").read_text())["scoring"]["training_energy"]
    if f"*{STATIC_POWER_W!r}*" not in scoring.replace(" ", ""):
        bad.append(f"{study.name}/setup.json scoring.training_energy does not use {STATIC_POWER_W} W: {scoring}")
    return bad

def check_all() -> List[str]:
    problems: List[str] = []
    for rel in tracked_a100_configs():
        live, frozen = classify(rel)
        if live is not None:
            problems += check_config(rel, live)
        elif frozen is None:
            problems.append(f"{rel}: A100 config is neither LIVE nor FROZEN in a100_literature.py")
    return problems + check_training_overlay()

def _set_memory_energy(block: Dict[str, Any], value: float) -> None:
    for key in ("dynamic_energy_per_bit", "dynamic_energy_per_byte", "dynamic_energy_per_sector"):
        block.pop(key, None)
    if block.get("sector_size_bytes") is not None:
        assert int(block["sector_size_bytes"]) == SECTOR_BYTES, block
        block["dynamic_energy_per_sector"] = _r(value * SECTOR_BYTES)
    else:
        block["dynamic_energy_per_byte"] = value

def transform(hw: Dict[str, Any], live: Live) -> Dict[str, Any]:
    tp, core = hw["tech_param"], hw["tech_param"]["core"]
    if live.tier == "energylens":
        old = float(core["operating_frequency"])
        core["operating_frequency"] = ENERGYLENS_CLOCK_HZ
    clock = float(core["operating_frequency"])
    for key, value in core_energy(clock).items():
        core[key] = value
    for level in ("DRAM", "SRAM-L2", "SRAM-L1", "SRAM-R"):
        _set_memory_energy(tp[level], per_byte_energy(level, clock))
    hw["static_power_w"] = STATIC_POWER_W
    hw.pop("busy_power_w", None)
    for dim in (hw.get("network") or {}).get("dimensions") or []:
        topo = dim.setdefault("topology", {})
        for key in FITTED_NETWORK_KEYS:
            topo.pop(key, None)
        if _is_nvlink(dim):
            topo["energy_per_bit"] = _r(NVLINK_PJ_PER_BIT * 1e-12)
    if live.tier == "prices":
        return hw
    f_over = clock / REFERENCE_CLOCK_HZ
    core["matrix_unit"]["util"] = MATRIX_UTIL
    core["device_fixed_overhead_cycles"] = launch_floor_cycles(clock)
    for level, bw in (("SRAM-L2", L2_BANDWIDTH_1410), ("SRAM-L1", L1_BANDWIDTH_1410)):
        tp[level]["bandwidth"] = _r(bw * f_over)
        tp[level]["util"] = 1.0
    tp["DRAM"]["bandwidth"] = _r(MEMORY_BANDWIDTH_FRACTION * DRAM_SPEC_BPS[live.sku])
    tp["DRAM"]["util"] = 1.0
    for level in ("DRAM", "SRAM-L2", "SRAM-L1"):
        if tp[level].get("latency_cycles") is not None:
            tp[level]["latency"] = _r(float(tp[level]["latency_cycles"]) / clock)
    if live.run_type == "inference":
        hw.setdefault("sw_param", {})["inference_step_overhead"] = DECODE_STEP_OVERHEAD_S
        hw["sw_param"]["inference_prefill_overhead"] = PREFILL_OVERHEAD_S
    if live.sku.startswith("A100-SXM4"):
        for dim in (hw.get("network") or {}).get("dimensions") or []:
            dim["topology"]["util"] = NETWORK_UTIL_SXM
    if live.tier == "energylens":
        n = int(hw.get("parallelism", {}).get("tp", 1))
        if n > 1:
            dim0 = hw["network"]["dimensions"][0]
            assert dim0["topology"]["type"] == "FC", dim0
            dim0["topology"]["bandwidth"] = _r(ENERGYLENS_NVSWITCH_PER_GPU_BPS / (n - 1))
    return hw

_PRICE_LINE = "{indent}{key}: {value!r} # J; a100_literature.py ({ref})\n"

def _patch_prices_text(text: str, clock: float, nvlink_dims: Tuple[int, ...] = ()) -> str:
    lines = text.splitlines(keepends=True)
    out: List[str] = []
    section: List[Tuple[int, str]] = []
    core_done = False
    dim_index = -1
    for line in lines:
        if re.match(r"^\s*-\s*id:", line) and [k for _, k in section][:2] == ["network", "dimensions"]:
            dim_index += 1
        m = re.match(r"^(\s*)([A-Za-z0-9_\-]+):(.*)$", line)
        if not m or line.lstrip().startswith("#"):
            out.append(line)
            continue
        indent, key = len(m.group(1)), m.group(2)
        while section and section[-1][0] >= indent:
            section.pop()
        path = [k for _, k in section] + [key]
        section.append((indent, key))
        pad = m.group(1)
        if path[:2] == ["tech_param", "core"] and len(path) == 3:
            if key in ("tensor_energy_per_flop", "vector_energy_per_flop", "padded_tensor_energy_per_flop"):
                continue
            if key == "nominal_energy_per_flop":
                for k, v in core_energy(clock).items():
                    out.append(_PRICE_LINE.format(indent=pad, key=k, value=v, ref="Antepara SC'25 x V^2"))
                core_done = True
                continue
        if (len(path) == 3 and path[0] == "tech_param" and path[1] in ("DRAM", "SRAM-L2", "SRAM-L1", "SRAM-R")
                and key in ("dynamic_energy_per_bit", "dynamic_energy_per_byte", "dynamic_energy_per_sector")):
            out.append(_PRICE_LINE.format(indent=pad, key="dynamic_energy_per_byte",
                                          value=per_byte_energy(path[1], clock), ref="Antepara SC'25 x V^2"))
            continue
        if path == ["static_power_w"] or path == ["busy_power_w"]:
            continue
        if path[:2] == ["network", "dimensions"] and key in FITTED_NETWORK_KEYS:
            continue
        if path[:2] == ["network", "dimensions"] and key == "energy_per_bit" and dim_index in nvlink_dims:
            out.append(_PRICE_LINE.format(indent=pad, key="energy_per_bit", value=_r(NVLINK_PJ_PER_BIT * 1e-12),
                                          ref="NVLink 8 pJ/bit, Dally GTC China 2020"))
            continue
        out.append(line)
    assert core_done, "tech_param.core.nominal_energy_per_flop not found"
    if out and not out[-1].endswith("\n"):
        out[-1] += "\n"
    out.append(f"static_power_w: {STATIC_POWER_W!r} # W/GPU; a100_literature.py (Antepara SC'25 P_const)\n")
    return "".join(out)

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _refresh_study_hashes(study: Path) -> None:
    cases = study / "cases.csv"
    if cases.is_file():
        rows = list(csv.DictReader(cases.open(newline="")))
        fields = list(rows[0].keys())
        for r in rows:
            for col, ref in (("hardware_sha256", "hardware_config"), ("model_sha256", "model_config")):
                if col in r and r.get(ref):
                    r[col] = _sha(ROOT / r[ref])
        with cases.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    setup = study / "setup.json"
    if setup.is_file():
        s = json.loads(setup.read_text())
        base = ROOT / "suite/historical_validation"
        for e in s.get("source_snapshots", []):
            p = base / e.get("pinned", "")
            if e.get("sha256") and p.is_file() and _sha(p) != e["sha256"]:
                e["sha256"] = _sha(p)
                e["note"] = "rewritten from a100_literature.py"
        setup.write_text(json.dumps(s, indent=2) + "\n")

def _apply_training_overlay() -> None:
    study = ROOT / "suite/historical_validation/training/energy_latency"
    rows = list(csv.DictReader((study / "cases.csv").open(newline="")))
    for r in rows:
        r["static_overlay_w_per_gpu"] = repr(STATIC_POWER_W)
    with (study / "cases.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    s = json.loads((study / "setup.json").read_text())
    s["scoring"]["training_energy"] = re.sub(r"\*\s*[0-9.]+\s*\*", f"*{STATIC_POWER_W!r}*",
                                             s["scoring"]["training_energy"], count=1)
    (study / "setup.json").write_text(json.dumps(s, indent=2) + "\n")

def apply_all() -> List[str]:
    written, studies = [], set()
    for rel in tracked_a100_configs():
        live, _ = classify(rel)
        if live is None:
            continue
        path = ROOT / rel
        if live.tier == "prices":
            hw = yaml.safe_load(path.read_text())
            nv = tuple(i for i, d in enumerate((hw.get("network") or {}).get("dimensions") or []) if _is_nvlink(d))
            path.write_text(_patch_prices_text(path.read_text(), float(hw["tech_param"]["core"]["operating_frequency"]), nv))
        else:
            path.write_text(yaml.safe_dump(transform(yaml.safe_load(path.read_text()), live), sort_keys=False))
        written.append(rel)
        if path.parent.name == "configs":
            studies.add(path.parent.parent)
    _apply_training_overlay()
    for study in sorted(studies):
        _refresh_study_hashes(study)
    return written

def main(argv: Iterable[str]) -> int:
    argv = list(argv)
    if argv == ["apply"]:
        for rel in apply_all():
            print("wrote", rel)
        argv = ["check"]
    if argv == ["check"]:
        problems = check_all()
        print("\n".join(problems) if problems else "all live A100 configs match the literature table")
        return 1 if problems else 0
    print("usage: python a100_literature.py check|apply")
    return 2

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
