from __future__ import annotations

import json
import os
from pathlib import Path

from paper_pipeline.scorers.gpu_work import job_idle_power, load_idle_power

DESIGNATION = "STATIC_IDLE"
WIP = ""
ENERGY_QUANTITIES = frozenset(("energy_per_step", "prefill_energy", "decode_energy", "total_energy",
                               "train_energy_at_pred_time", "energy_per_request"))

def resolve_static_power(rule: str, run: dict, idle_spec: dict | None, root: Path) -> dict:
    if rule not in ("measured_idle", "antepara_54w"):
        raise ValueError(f"unknown static_power_rule {rule!r}")
    count = int(run.get("n_gpus") or run.get("tp") or 1)
    if count <= 0:
        raise ValueError("GPU count must be positive")
    result = dict(rule=rule, designation=DESIGNATION, status="final", reason="Measured idle where available; otherwise Antepara 54 W per GPU",
                  selected_gpus=[], measured_power_w_by_gpu={}, source="antepara_54w",
                  power_w_per_gpu=54.0, job_power_w=54.0 * count)
    if rule == "measured_idle" and idle_spec:
        idle = load_idle_power(idle_spec, root)
        total = job_idle_power(run, idle)
        result.update(source=idle_spec["path"], job_power_w=total, power_w_per_gpu=total / count)
        if idle["mode"] == "per_cell":
            result["measurement_source"] = idle["cells"][str(run["cell"])]["source"]
        if idle["mode"] == "per_gpu":
            selected = run.get("gpus")
            if isinstance(selected, str):
                selected = json.loads(selected) if selected.strip().startswith("[") else selected.split(",")
            selected = [int(g) for g in selected]
            powers = dict(zip(map(int, idle["gpus"]), map(float, idle["locked"]["power_W"])))
            result.update(selected_gpus=selected, measured_power_w_by_gpu={str(g): powers[g] for g in selected})
        else:
            selected = run.get("gpus")
            if isinstance(selected, str):
                selected = json.loads(selected) if selected.strip().startswith("[") else selected.split(",")
            selected = [int(g) for g in selected] if selected else []
            if selected and len(selected) != 1:
                raise ValueError("fixed idle power requires one selected GPU")
            result.update(selected_gpus=selected, measured_power_w_by_gpu={str(selected[0]): total} if selected else {})
    return result

def mark_prediction(record: dict) -> dict:
    version = record.get("schema_version")
    if version is not None and version != 7:
        raise ValueError(f"unsupported energy report schema_version {version!r}")
    fields = []
    def add(mapping, prefix, names):
        if isinstance(mapping, dict):
            fields.extend(f"{prefix}{name}" for name in names if name in mapping and mapping[name] is not None)

    add(record, "", ("static_power_w", "total_energy_uj", "modeled_energy_uj", "per_device_energy_uj"))
    for name in ("breakdown_uj", "resource_breakdown_uj"):
        add(record.get(name), f"{name}.", ("static",))
    for phase, values in (record.get("phases") or {}).items():
        add(values, f"phases.{phase}.", ("total_j", "static_j", "static"))
    for phase in (record.get("phase_energy_uj") or {}):
        fields.append(f"phase_energy_uj.{phase}")

    def component_report(report, prefix):
        add(report, prefix, ("total_energy_j", "closure_error_j"))
        static = ((report or {}).get("components") or {}).get("static")
        add(static, prefix + "components.static.",
            ("energy_j", "coefficient", "activity_priced_energy_j", "accounting_residual_j"))
    component_report(record.get("component_energy"), "component_energy.")
    for phase, report in (record.get("phase_component_energy") or {}).items():
        component_report(report, f"phase_component_energy.{phase}.")
    record["energy_wip"] = WIP
    record["energy_wip_fields"] = fields
    return record

def mark_run_energy_report(run_dir, policy=None):
    path = Path(run_dir) / "output" / "LLM" / "energy_report.json"
    if path.is_file():
        record = json.loads(path.read_text())
        if policy is not None:
            record["static_power_policy"] = policy
        mark_prediction(record)
        temporary = path.with_name(path.name + ".wip.tmp")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
