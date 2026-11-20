from __future__ import annotations

import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

BANDS = ("ok", "gt5", "gt10", "missing", "unsupported", "no_gpu_work_basis")
BAND_HEADERS = ("ok (≤5%)", ">5% to 10%", ">10%", "missing", "unsupported", "no GPU-work basis")

def _read(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        dummy_comment = False
        for line in stream:
            if not line.startswith("#"):
                return list(csv.DictReader(itertools.chain([line], stream))), dummy_comment
            dummy_comment |= "DUMMY" in line.upper()
        return [], dummy_comment

def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None

def _sort_key(row):
    value = _number(row.get("abs_err_pct"))
    return (value is None, -value if value is not None else 0)

def _band(row):
    status = row.get("status", "")
    if status == "ok_unverified_counters":
        return status
    if status in ("missing", "unsupported", "no_gpu_work_basis"):
        return status
    flag = row.get("flag", "")
    if flag in BANDS[:3]:
        return flag

    return "missing"

def _escape(value):
    return str(value if value is not None else "").replace("\\", "\\\\").replace("|", "\\|").replace("\r", " ").replace("\n", "<br>")

def _table(lines, headers, rows):
    lines.append("| " + " | ".join(_escape(x) for x in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_escape(x) for x in row) + " |")
    lines.append("")

def _counts(rows, bands=BANDS):
    counter = Counter(_band(row) for row in rows)
    return [counter[band] for band in bands]

def _details(row):
    values = [row.get(k, "") for k in ("description", "quantity", "measured", "predicted", "signed_err_pct", "label", "assumptions")]
    return values

def _variant_details(row):
    if row is None:
        return "—"
    parts = [f"status: {row.get('status', '')}", f"flag: {row.get('flag', '')}",
             f"measured: {row.get('measured', '')}", f"predicted: {row.get('predicted', '')}",
             f"signed error (%): {row.get('signed_err_pct', '')}", f"label: {row.get('label', '')}",
             f"assumptions: {row.get('assumptions', '')}"]
    return _red("; ".join(parts)) if row.get("energy_wip") else "; ".join(parts)

def make_report(out_root: Path):
    root = Path(out_root)
    rows = []
    dummy_comment = False
    permissions = []
    for entry in sorted(root.iterdir()):
        if entry.name in ("#snapshot", ".snapshot", "figures", "flags") or not entry.is_dir():
            continue
        path = entry / "scores" / "cells.csv"
        if not path.is_file():
            continue
        permission = entry / "hold_exception.json"
        if permission.is_file():
            permissions.append(json.loads(permission.read_text()))
        input_rows, input_dummy = _read(path)
        rows.extend(input_rows)
        dummy_comment |= input_dummy
    dummy_value = any("DUMMY" in str(value).upper() for row in rows for value in row.values() if value is not None)

    all_rows = rows
    excluded = [r for r in rows if r.get("status") == "excluded"]
    rows = [r for r in rows if r.get("status") != "excluded"]
    rows.sort(key=_sort_key)
    dest = root / "flags"
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "cells_all.csv"
    fields = []
    for row in rows:
        for field in row:
            if field is not None and field not in fields:
                fields.append(field)
    if not fields:
        fields = ["entry", "figure", "panel", "group", "cell", "description", "variant", "quantity", "unit", "measured", "predicted", "ratio", "signed_err_pct", "abs_err_pct", "flag", "status", "label", "assumptions", "reference_model_value", "reference_model_name"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        if dummy_comment and not dummy_value:
            stream.write("# DUMMY\n")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows + excluded)

    figures = defaultdict(list)
    for row in rows:
        figures[row.get("figure") or "(unspecified)"].append(row)
    has_unverified = any(r.get("status") == "ok_unverified_counters" for r in rows)
    bands = (*BANDS, "ok_unverified_counters") if has_unverified else BANDS
    headers = (*BAND_HEADERS, "computed; counters unverified") if has_unverified else BAND_HEADERS
    lines = ["# DUMMY — Flags" if dummy_comment or dummy_value else "# Flags", "", "Error bands are exclusive; >5% to 10% includes 10%. Counts are score rows, including variants.", "", "## Overall", ""]
    if has_unverified:
        lines.extend(["Computed rows with unverified Hopper counters are shown separately from validated error bands; their numerical errors remain in cells_all.csv.", ""])
    for permission in permissions:
        lines.extend([f"HOLD exception: entry `{_escape(permission['entry'])}`, designation `{_escape(permission['designation'])}`. "
                      f"Decision: {_escape(permission['decision'])}. Global hold remains active for other entries.", ""])
    basis_counts = Counter((row.get("entry"), row["gpu_work_basis"]) for row in all_rows
                           if row.get("gpu_work_basis"))
    if basis_counts:
        lines.extend(["## GPU-work basis", "",
                      "Paper TIME uses measured kernel-union time. Paper ENERGY subtracts "
                      "(wall time minus kernel-union time) times the sum of measured idle watts for the job's GPUs. "
                      "No-context idle readings are a lower bound on gap power. "
                      "Rows with no GPU-work basis have no primary ratio. "
                      "Wall time and raw/reference energy remain in explicitly secondary CSV columns.", ""])
        _table(lines, ["Entry", "Basis", "Score rows (including variants)"],
               ([entry, basis, n] for (entry, basis), n in sorted(basis_counts.items())))
    _table(lines, ["Figure", "Cells", *headers],
           ([figure, len(group), *_counts(group, bands)] for figure, group in sorted(figures.items())))
    for figure, group in sorted(figures.items()):
        lines.extend([f"## {_escape(figure)}", "", "### Bands by quantity", ""])
        quantities = defaultdict(list)
        for row in group:
            quantities[row.get("quantity") or "(unspecified)"].append(row)
        _table(lines, ["Quantity", "Cells", *headers],
               ([quantity, len(items), *_counts(items, bands)] for quantity, items in sorted(quantities.items())))
        unverified = [row for row in group if row.get("status") == "ok_unverified_counters"]
        if unverified:
            lines.extend(["### Computed cells with unverified Hopper counters", ""])
            _table(lines, ["Entry", "Cell", "Variant", "Quantity", "Measured", "Predicted", "Signed error (%)", "Status"],
                   ([row.get(k, "") for k in ("entry", "cell", "variant", "quantity", "measured", "predicted", "signed_err_pct", "status")] for row in unverified))
        lines.extend(["### Worst ten cells", ""])
        _table(lines, ["Description", "Quantity", "Measured", "Predicted", "Signed error (%)", "Label", "Assumptions"],
               (_details(row) for row in group[:10]))
        unscored = [row for row in group if _band(row) in ("missing", "unsupported", "no_gpu_work_basis")]
        lines.extend(["### Missing and unsupported cells (including no GPU-work basis)", ""])
        _table(lines, ["Entry", "Panel", "Group", "Cell", "Variant", "Quantity", "Status", "Description", "Label", "Assumptions"],
               ([row.get(k, "") for k in ("entry", "panel", "group", "cell", "variant", "quantity", "status", "description", "label", "assumptions")] for row in unscored))
        busy = [r for r in group if r.get("busy_fraction") not in (None, "")
                and r.get("quantity") == "energy_per_step"]
        if busy:
            lines.extend(["### Measured busy fraction per cell (reported without adjustment)", ""])
            _table(lines, ["Entry", "Cell", "Description", "Variant", "Busy fraction"],
                   ([r.get(k, "") for k in ("entry", "cell", "description", "variant", "busy_fraction")]
                    for r in busy))
        lines.extend(["### Flag sheet", "", "Assumptions:", ""])
        assumptions = sorted({item.strip() for row in group for item in (row.get("assumptions") or "").split(";") if item.strip()})
        lines.extend([f"- {_escape(item)}" for item in assumptions] or ["- None"])
        lines.extend(["", "Labels:", ""])
        labels = sorted({row.get("label", "").strip() for row in group if row.get("label", "").strip()})
        lines.extend([f"- {_escape(item)}" for item in labels] or ["- None"])
        lines.append("")
        variants = sorted({row.get("variant") or "default" for row in group})
        if len(variants) > 1:
            lines.extend(["### Variants side by side", ""])

            grouped = defaultdict(lambda: defaultdict(list))
            for row in group:
                key = tuple(row.get(k, "") for k in ("entry", "panel", "group", "cell", "quantity"))
                grouped[key][row.get("variant") or "default"].append(row)
            side_rows = []
            for key, by_variant in sorted(grouped.items()):
                for index in range(max(map(len, by_variant.values()))):
                    side_rows.append([*key, *(_variant_details(by_variant[v][index] if index < len(by_variant[v]) else None) for v in variants)])
            _table(lines, ["Entry", "Panel", "Group", "Cell", "Quantity", *variants], side_rows)
    if excluded:
        by_rule = {}
        for r in excluded:
            rid = (r.get("label") or "").split(":")[0] or "?"
            by_rule.setdefault(rid, set()).add((r.get("entry"), r.get("cell")))
        lines.extend(["", "## Cells excluded by scope designations (not predicted, not scored, in no band, figure or worst-ten list)", "",
                      "| Designation | Cells | Entries | Reason |", "| --- | --- | --- | --- |"])
        for rid, cells in sorted(by_rule.items()):
            reason = next((r.get("label", "") for r in excluded if (r.get("label") or "").startswith(rid)), "")
            ents = sorted({e for e, _ in cells})
            lines.append(f"| {rid} | {len(cells)} | {', '.join(ents)} | {reason.split(':', 1)[-1].strip().replace('|', '/')} |")
    md_path = dest / "FLAGS.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [csv_path, md_path]
