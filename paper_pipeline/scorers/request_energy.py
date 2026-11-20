from __future__ import annotations

import json
from pathlib import Path

from paper_pipeline.scorers.common import cell_row, write_cells, profile_ok
from paper_pipeline.scorers.gpu_work import NO_GPU_WORK
from paper_pipeline.suites import interpolate_at

def _field(j: dict, spec: str, batch: int):
    div = spec.endswith("/batch")
    phase, key = spec.replace("/batch", "").split(".")
    v = float(j["phases"][phase][key])
    return v / batch if div else v

def score(entry: str, runs: list, pred_dir: Path, out_csv: Path, variant: str = "default", assumptions: str = "",
          profile_required: bool = False) -> int:
    rows = []
    for r in runs:
        p = Path(pred_dir) / f"{r['cell']}.json"
        j = json.loads(p.read_text()) if p.exists() else None
        if j is not None and j.get("assert_16bit", "ok") != "ok":
            j = None
        if j is not None and not profile_ok(j, variant, profile_required):
            j = None
        for ref in r["refs"]:
            base = dict(entry=entry, figure=r["figure"], panel=r["panel"], group=r["group"], cell=r["cell"],
                        description=r["description"], variant=variant, quantity=ref["quantity"], unit=ref["unit"],
                        label="; ".join(x for x in (NO_GPU_WORK, ref.get("label", "")) if x),
                        gpu_work_basis=NO_GPU_WORK,
                        measured_reference_secondary=ref.get("measured", ""),
                        assumptions="; ".join(x for x in (assumptions, r.get("assumptions", "")) if x),
                        reference_model_value=ref.get("reference_model_value", ""),
                        reference_model_name=ref.get("reference_model_name", ""))
            if r.get("excluded"):
                rows.append(cell_row(**dict(base, label=r["excluded"]), status="excluded"))
                continue
            if r.get("unsupported"):
                rows.append(cell_row(**base, status="unsupported"))
                continue
            if j is None:
                rows.append(cell_row(**base, status="no_gpu_work_basis"))
                continue
            if ref["quantity"] == "train_energy_at_pred_time":
                t = float(j["phases"]["train"]["time_s"])
                m, how = interpolate_at(ref["curve"], t)
                base["label"] = "; ".join(x for x in (how.replace("_", " "), base["label"]) if x)
                base["measured_reference_secondary"] = m
                rows.append(cell_row(**base, predicted=float(j["phases"]["train"]["total_j"]),
                                     status="no_gpu_work_basis"))
                continue
            rows.append(cell_row(**base, predicted=_field(j, ref["predicted_from"], r["batch"]),
                                 status="no_gpu_work_basis"))
    write_cells(rows, out_csv)
    return sum(x["status"] == "ok" for x in rows)
