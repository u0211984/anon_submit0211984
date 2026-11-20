from __future__ import annotations

import csv
from pathlib import Path
from paper_pipeline.static_power import ENERGY_QUANTITIES, WIP

CELL_FIELDS = ["entry", "figure", "panel", "group", "cell", "description", "variant", "quantity", "unit", "measured",
               "predicted", "ratio", "signed_err_pct", "abs_err_pct", "flag", "status", "label", "assumptions",
               "reference_model_value", "reference_model_name", "commit", "busy_fraction",
               "gpu_work_basis", "measured_wall_s_secondary", "measured_raw_energy_J_secondary",
               "measured_reference_secondary", "idle_power_W", "gpu_gap_s", "energy_wip"]
FLAG_LEVELS = (5.0, 10.0)

def flag(abs_err_pct):
    if abs_err_pct is None:
        return ""
    return "gt10" if abs_err_pct > FLAG_LEVELS[1] else ("gt5" if abs_err_pct > FLAG_LEVELS[0] else "ok")

def cell_row(**kw) -> dict:
    r = {k: kw.get(k, "") for k in CELL_FIELDS}
    if r["quantity"] in ENERGY_QUANTITIES:
        r["energy_wip"] = WIP
    m, p = kw.get("measured"), kw.get("predicted")
    if r["status"] in ("", None):
        r["status"] = "ok" if (m not in (None, "") and p not in (None, "")) else "missing"
    if r["status"] in ("ok", "duplicate", "ok_unverified_counters") and m not in (None, "") and p not in (None, ""):
        ratio = float(p) / float(m)
        r.update(ratio=ratio, signed_err_pct=100.0 * (ratio - 1.0), abs_err_pct=abs(100.0 * (ratio - 1.0)))
        r["flag"] = flag(r["abs_err_pct"])
    return r

def write_cells(rows, out_csv: Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELL_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return out_csv

def profile_ok(j: dict, variant: str, required: bool) -> bool:
    prof = j.get("settings_profile")
    if prof is None:
        return not required
    if isinstance(prof, str):
        prof = {"profile": prof}
    if prof.get("profile") == "paper":
        return True
    allowed = {part.split("=", 1)[0] for part in (variant or "").split("+") if "=" in part}
    diff = prof.get("differing") or prof.get("differing_keys") or []
    names = {str(k).split(".")[-1].lower() for k in diff}
    return bool(names) and names <= {a.lower() for a in allowed}
