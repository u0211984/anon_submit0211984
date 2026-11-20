from __future__ import annotations

import json
from pathlib import Path

from paper_pipeline.scorers.common import cell_row, write_cells, profile_ok
from paper_pipeline.scorers.gpu_work import (measured_basis, GPU_WORK, NO_GPU_WORK, NODE_RAW, NODE_RAW_BASIS,
                                             node_raw_basis, node_raw_prediction)

def describe(r: dict) -> str:
    bits = [r["model"], r["phase"], f"b{r['batch']}", f"s{r['seq_len']}"]
    if r.get("parallel"):
        bits.append(str(r["parallel"]))
    return " ".join(bits)

def combine(points, pred_dir: Path, cell: str, phase: str, variant: str = "default", profile_required: bool = False):
    t = e = 0.0
    for tag, _kv, w in points:
        p = Path(pred_dir) / (f"{cell}__{tag}.json" if tag else f"{cell}.json")
        if not p.exists():
            return None
        j = json.loads(p.read_text())
        if j.get("assert_16bit", "ok") != "ok":
            return None
        if not profile_ok(j, variant, profile_required):
            return None
        ph = j["phases"][phase]
        t += w * float(ph["time_s"])
        e += w * float(ph["total_j"])
    return t, e

def score(entry: str, figure: str, runs: list, per_step: dict, points_of, pred_dir: Path, out_csv: Path,
          variant: str = "default", assumptions: str = "", profile_required: bool = False,
          idle_power: dict | None = None, basis: str | None = None) -> int:
    if basis not in (None, NODE_RAW):
        raise ValueError(f"unknown score basis {basis!r}")
    rows = []
    for r in runs:
        measured = (node_raw_basis if basis == NODE_RAW else measured_basis)(r, per_step, idle_power)
        mt, me = measured.pop("gpu_work_time_s"), measured.pop("gpu_work_energy_J")
        has_basis = measured["gpu_work_basis"] in (GPU_WORK, NODE_RAW_BASIS)
        label = "" if has_basis else NO_GPU_WORK
        if has_basis and basis is None and measured["gpu_gap_s"] < 0:
            label = "kernel union exceeds measured wall time; signed gap retained"
        base = dict(entry=entry, figure=figure, panel=r["phase"], group=r["model"], cell=r["cell"],
                    description=describe(r), variant=variant, assumptions=assumptions,
                    busy_fraction=r.get("busy_fraction", ""), **measured)
        if r.get("excluded"):
            for q, m, u in (("time_per_step", mt, "s"), ("energy_per_step", me, "J")):
                rows.append(cell_row(**base, quantity=q, unit=u, measured=m, status="excluded", label=r["excluded"]))
            continue
        if r.get("unsupported"):
            for q, m, u in (("time_per_step", mt, "s"), ("energy_per_step", me, "J")):
                rows.append(cell_row(**base, quantity=q, unit=u, measured=m, status="unsupported", label=r["unsupported"]))
            continue
        got = combine(points_of(r), pred_dir, r["cell"], r["phase"], variant, profile_required)
        if got is not None and basis == NODE_RAW:
            got = node_raw_prediction(got[0], got[1], measured)
        for i, (q, m, u) in enumerate((("time_per_step", mt, "s"), ("energy_per_step", me, "J"))):
            rows.append(cell_row(**base, quantity=q, unit=u, measured=m, predicted=None if got is None else got[i],
                                 label=label, status="no_gpu_work_basis" if not has_basis else
                                 ("missing" if got is None else "")))
    write_cells(rows, out_csv)
    return sum(r["status"] == "ok" for r in rows) // 2
