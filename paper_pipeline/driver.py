#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EG = ROOT.parent
FOREIGN_TREES = ()
FORBIDDEN_ENV = ("RAPID_", "OPMODEL_", "COUNTS_")
KEEP_ENV = ("PATH", "LD_LIBRARY_PATH", "PYTHONHASHSEED", "HOME", "LANG", "TMPDIR", "MPLCONFIGDIR",
            "PYTHONDONTWRITEBYTECODE")
LOCK = HERE / "inputs.lock.json"
HOLD = "energy scoring and reruns are on hold (designations.yaml hold.active); only `plan` runs"

BASE_MODELLING_KEYS = ("sw_param.kernel_launch_overhead", "sw_param.dp_zero_stage", "sw_param.dp_microbatch",
                       "tech_param.core.tensor_latency_cycles")

REMOVED_ENV = sorted(k for k in os.environ if k.startswith(FORBIDDEN_ENV))
for _k in REMOVED_ENV:
    del os.environ[_k]

FOREIGN_SITE = [p for p in sys.path if p.endswith("site-packages") and not p.startswith(sys.prefix)]
sys.path[:] = [str(ROOT), str(ROOT / "opmodel" / "src")] + [
    p for p in sys.path if p and (p.endswith("site-packages") or p.startswith(sys.base_prefix))]
import yaml  # noqa: E402

def sha256(path: Path, n: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n] if n else h.hexdigest()

def load_yaml(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)

def resolve(p: str, paths: dict) -> Path:
    q = Path(p.format(**paths))
    return q if q.is_absolute() else ROOT / q

def inside_tree(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False

def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()

def foreign_modules() -> list:
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = str(Path(f).resolve())
        if "/site-packages/" in p or p.startswith(str(ROOT) + "/"):
            continue
        if p.startswith(FOREIGN_TREES):
            bad.append((name, p))
    return bad

def import_from_tree(name: str):
    mod = importlib.import_module(name)
    f = getattr(mod, "__file__", None)
    if not f or not inside_tree(Path(f)):
        raise RuntimeError(f"module {name} has origin {f!r}, not inside {ROOT}")
    bad = foreign_modules()
    if bad:
        raise RuntimeError(f"modules loaded from another code tree: {bad[:5]}")
    return mod

def clean_env() -> dict:
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return env

def merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            merge(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst

def dig(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d

class Report:
    def __init__(self):
        self.lines, self.errors, self.blocks = [], [], []

    def ok(self, msg):
        self.lines.append(f"  ok      {msg}")

    def warn(self, msg):
        self.lines.append(f"  WARN    {msg}")

    def err(self, msg):
        self.lines.append(f"  ERROR   {msg}")
        self.errors.append(msg)

    def block(self, msg):
        self.lines.append(f"  BLOCKED {msg}")
        self.blocks.append(msg)

class Inputs:
    def __init__(self, lock: bool):
        self.lock = lock
        self.pinned = json.loads(LOCK.read_text())["inputs"] if LOCK.exists() else {}
        self.seen = {}

    def check(self, rep: Report, label: str, path: Path, quiet: bool = False) -> bool:
        if not path.exists():
            rep.err(f"{label}: missing {path}")
            return False
        d = sha256(path)
        key = str(path.resolve().relative_to(ROOT)) if inside_tree(path) else str(path)
        self.seen[key] = {"sha256": d, "inside_tree": inside_tree(path)}
        where = "" if inside_tree(path) else " (outside the tree)"
        if self.lock:
            msg = f"{label}: {path.name} sha256 {d[:16]}{where} (pinning)"
        elif key not in self.pinned:
            rep.block(f"{label}: {path} is not pinned in inputs.lock.json (review, then plan --lock)")
            return True
        elif self.pinned[key] != d:
            rep.err(f"{label}: {path} changed since pinned ({self.pinned[key][:16]} -> {d[:16]})")
            return False
        else:
            msg = f"{label}: {path.name} sha256 {d[:16]}{where} matches the lock"
        if not quiet:
            rep.ok(msg)
        return True

def valid_designation(r) -> bool:
    return isinstance(r, dict) and bool(str(r.get("decision") or "").strip()) and bool(r.get("date")) and "value" in r

def open_designations(ids, designations: dict):
    return [i for i in ids if not valid_designation(designations.get(i))]

def check_settings(rep: Report, settings: dict, designations: dict) -> dict:
    bad = [k for k, v in designations.items() if not valid_designation(v)]
    if bad:
        rep.err(f"designations.yaml: entries without date/decision/value: {bad}")
    exceptions = hold_state().get("exceptions") or []
    if not isinstance(exceptions, list):
        rep.err("hold.exceptions must be a list of entry/designation mappings")
        exceptions = []
    for exception in exceptions:
        if (not isinstance(exception, dict) or not exception.get("entry")
                or not valid_designation(designations.get(exception.get("designation")))):
            rep.err(f"hold exception without a valid designation: {exception}")
    need = settings.get("designations_needed") or {}
    out = {k: open_designations(v or [], designations) for k, v in need.items()}
    rep.ok(f"settings/paper.yaml parsed: blocks {sorted(settings)}; sha256 {sha256(HERE / 'settings/paper.yaml', 16)}")
    for k, v in out.items():
        rep.lines.append(f"  info    designations_needed.{k}: {'open ' + ', '.join(v) if v else 'all ruled'}")
    nulls = [k for k, v in (settings.get("network") or {}).items() if v is None]
    if nulls:
        rep.lines.append(f"  info    network values still null (multi-GPU entries): {', '.join(nulls)}")
    return out

def hardware_architecture(hw: dict):
    core = (hw.get("tech_param") or {}).get("core") or {}
    arch = core.get("architecture", "sm80")
    if arch not in ("sm80", "sm90"):
        raise ValueError(f"unsupported hardware architecture {arch!r}")
    return arch

def entry_architecture(entry: dict):
    spec = load_yaml(HERE / "hardware" / f"{entry['hardware']}.yaml")
    bases = spec.get("base") or {}
    arches = {hardware_architecture(load_yaml(ROOT / base)) for base in bases.values()}
    if len(arches) != 1:
        raise ValueError(f"hardware {entry['hardware']}: expected one architecture across base descriptions")
    return arches.pop()

def apply_a100_latency(hw, settings):
    if "a100_latency_source" not in settings or hardware_architecture(hw) == "sm90":
        return
    from paper_pipeline.a100_latency import apply_latency_source
    policy = apply_latency_source(hw, settings["a100_latency_source"])
    hw.setdefault("metadata", {})["a100_latency_source"] = policy

def apply_static_power(hw, entry, run, settings):
    if "static_power_rule" not in settings or hardware_architecture(hw) == "sm90":
        return
    from paper_pipeline.static_power import resolve_static_power
    spec = ((entry.get("workload") or {}).get("gpu_work_idle_power")
            or (settings.get("static_power_idle_by_hardware") or {}).get(entry.get("hardware")))
    job = dict(run or {})
    if not job.get("n_gpus") and not job.get("tp"):
        par = hw.get("parallelism") or {}
        job["n_gpus"] = int(par.get("tp", 1)) * int(par.get("cp", 1)) * int(par.get("pp", 1))
        if (run or {}).get("phase") == "train":
            job["n_gpus"] *= int((par.get("train") or {}).get("dp", 1))
    policy = resolve_static_power(settings["static_power_rule"], job, spec, ROOT)
    hw["static_power_w"] = policy["power_w_per_gpu"]
    hw.setdefault("metadata", {})["static_power_policy"] = policy

def allreduce_curve_applies(spec: dict, run: dict) -> bool:
    box = load_yaml(HERE / "registry.yaml")["network_allreduce_curve"]["box"]
    if (spec.get("sku"), spec.get("gpus"), spec.get("interconnect")) != (box["sku"], box["gpus"], box["interconnect"]):
        return False
    return all(v == float(box["sm_clock_mhz"]) for v in measured_clock(run)["per_gpu"])

def megatron_layout(entry: dict, run: dict):
    rules = (entry.get("workload") or {}).get("rules") or {}
    if run.get("phase") != "train" or rules.get("train_layout") != MEGATRON_PER_ROW:
        return None
    if not str(run.get("framework") or "").startswith("megatron-core"):
        raise ValueError(f"train_layout {MEGATRON_PER_ROW} describes Megatron-Core rows, not {run.get('framework')!r}")
    tp, pp, dp, zero, micro, n, mbs, gb = (int(run[k]) for k in ("tp", "pp", "dp", "dp_zero_stage", "microbatches",
                                                                   "n_gpus", "batch", "global_batch"))
    if min(tp, pp, dp, micro, mbs) < 1 or tp * pp * dp != n:
        raise ValueError(f"cell {run.get('cell')}: tp {tp} x pp {pp} x dp {dp} != n_gpus {n}")
    if zero not in (0, 1):
        raise ValueError(f"cell {run.get('cell')}: dp_zero_stage {zero}; Megatron DDP (0) or distributed optimizer (1)")
    if pp == 1 and micro != 1:
        raise ValueError(f"cell {run.get('cell')}: without a pipeline the harness runs one micro-batch per step")
    if gb != mbs * micro * dp:
        raise ValueError(f"cell {run.get('cell')}: global_batch {gb} != micro-batch {mbs} x {micro} x dp {dp}")
    pp_dimension = rules.get("pp_network_dimension", "shared")
    if pp_dimension not in ("shared", PP_OWN_DIMENSION):
        raise ValueError(f"workload.rules.pp_network_dimension {pp_dimension!r}: shared | {PP_OWN_DIMENSION}")
    if pp_dimension == PP_OWN_DIMENSION and pp > 1 and dp > 1:
        raise ValueError(f"cell {run.get('cell')}: pp_network_dimension {PP_OWN_DIMENSION} with pp > 1 and dp > 1")

    raw = run.get("disable_bucketing")
    if raw in (None, ""):
        raw = rules.get("megatron_ddp_disable_bucketing", False)
    if str(raw).strip().lower() not in ("true", "false", "1", "0"):
        raise ValueError(f"cell {run.get('cell')}: disable_bucketing {raw!r} is not a boolean")
    one_bucket = str(raw).strip().lower() in ("true", "1") and dp > 1
    return dict(tp=tp, pp=pp, dp=dp, dp_zero_stage=zero, microbatches=micro, global_batch=gb,
                pp_network_dimension=pp_dimension, dp_buckets="one" if one_bucket else "per_reducer")

def megatron_parallelism(hw: dict, layout: dict):
    par = hw["parallelism"]
    par.update(tp=layout["tp"], pp=layout["pp"], cp=1, tp_sp=False, mb=layout["microbatches"])
    par["train"].update(dp=layout["dp"], ep=1)
    dims = (hw.get("network") or {}).get("dimensions") or []
    if len(dims) != 1:
        raise ValueError("Megatron per-row layout expects one scale-up network dimension in the base")
    dims[0]["parallelisms"] = ["tp", "cp", "ep", "pp", "dp"]
    dims[0]["size"] = "auto"

def megatron_software(hw: dict, layout: dict):
    sw = hw.setdefault("sw_param", {})
    sw["dp_zero_stage"] = layout["dp_zero_stage"]
    sw["dp_microbatch"] = "last_mb"
    if layout["pp"] > 1:
        sw["pipeline_schedule"] = "1f1b"
        sw["pipeline_interleave"] = 1
    hw["execution_backend"]["astrasim"]["mode"] = "full_astrasim_flattened"
    if layout["pp"] > 1 and layout.get("pp_network_dimension") == PP_OWN_DIMENSION:
        pp_own_dimension(hw)

def pp_own_dimension(hw: dict):
    dims = hw["network"]["dimensions"]
    if len(dims) != 1 or "pp" not in dims[0]["parallelisms"]:
        raise ValueError("pp_network_dimension own_logical_dimension expects the one overlaid scale-up dimension")
    pp_dim = copy.deepcopy(dims[0])
    pp_dim["id"] = f"{dims[0].get('id', 'dim0')}_pp"
    pp_dim["parallelisms"] = ["pp"]
    dims[0]["parallelisms"] = [axis for axis in dims[0]["parallelisms"] if axis != "pp"]
    dims.append(pp_dim)

def build_hardware(rep: Report, inputs: Inputs, hw_id: str, base_key: str, run_type: str, settings: dict, run=None,
                   entry_id=None):
    spec = load_yaml(HERE / "hardware" / f"{hw_id}.yaml")
    if not inputs.check(rep, f"hardware entry {hw_id}", HERE / "hardware" / f"{hw_id}.yaml", quiet=True):
        return None
    if spec.get("base") is None:
        rep.block(f"hardware {hw_id}: not buildable yet (clock {spec.get('clock_hz')}, base {spec.get('base')}); open H3, H6")
        return None
    if base_key not in spec["base"]:
        rep.block(f"hardware {hw_id}.{base_key}: no base YAML; open H3")
        return None
    base = ROOT / spec["base"][base_key]
    want = (spec.get("base_sha256") or {}).get(base_key)
    if not want:
        rep.err(f"hardware {hw_id}.{base_key}: no base_sha256 pinned in hardware/{hw_id}.yaml")
        return None
    if not inputs.check(rep, f"hardware {hw_id}.{base_key}", base, quiet=True):
        return None
    if sha256(base, 16) != want:
        rep.err(f"hardware {hw_id}.{base_key}: sha256 {sha256(base, 16)} != {want}")
        return None
    hw = load_yaml(base)
    if hardware_architecture(hw) == "sm90":
        entries = load_yaml(HERE / "registry.yaml")["entries"]
        entry = next((e for e in entries if e["id"] == entry_id), None)
        if entry is None or entry.get("scorer") != "counts_join":
            rep.err(f"hardware {hw_id}: sm90 supports COUNTS only; energy and timing scoring are refused")
            return None

        rep.ok(f"hardware {hw_id}.{base_key}: base sha256 {want}; sm90 COUNTS only (no energy transform)")
        return hw
    dynamic = spec.get("clock_hz") == "per_run_measured"
    if (dynamic and run is None) or (not dynamic and not isinstance(spec.get("clock_hz"), (int, float))):
        rep.block(f"hardware {hw_id}: not buildable yet (clock {spec.get('clock_hz')}, base {spec.get('base')}); open H3, H6")
        return None
    base_clock = spec.get("base_clock_hz") if dynamic else spec["clock_hz"]
    if float(hw["tech_param"]["core"]["operating_frequency"]) != float(base_clock):
        rep.err(f"hardware {hw_id}.{base_key}: base clock {hw['tech_param']['core']['operating_frequency']} != {spec['clock_hz']}")
        return None
    held = [k for k in BASE_MODELLING_KEYS if dig(hw, k) is not None]
    if held:
        rep.block(f"hardware {hw_id}.{base_key}: modelling settings still inside the base YAML ({', '.join(held)}); open H5")
    entry = next((e for e in load_yaml(HERE / "registry.yaml")["entries"] if e["id"] == entry_id),
                 {"hardware": hw_id})
    layout = None
    if dynamic:
        if run["phase"] == "train":
            layout = megatron_layout(entry, run)
            if layout is None:
                rep.block("per-run FSDP hardware is not implemented; FSDP2 training remains blocked")
                return None
        stats = measured_clock(run)
        hw["tech_param"]["core"]["operating_frequency"] = stats["mean"] * 1e6
        par = hw["parallelism"]
        if run["phase"] != "train":
            par["tp"] = int(run["n_gpus"])
            par["train"]["dp"] = 1
        else:
            megatron_parallelism(hw, layout)
    lit = import_from_tree("a100_literature")
    live = lit.Live(pattern="", sku=spec["sku"], tier=spec["literature_tier"], run_type=run_type)
    if spec.get("gpus", 1) != 1:
        from paper_pipeline import overlay as O
        fixed = bool((settings.get("network") or {}).get("collective_launch_overhead_key_available"))
        curve = (entry_id,) if run is not None and allreduce_curve_applies(spec, run) else ()
        hw, _applied, _ass = O.overlay_hardware(hw, settings, lit, live, spec.get("inter_node"), fixed, entry_id, curve)
    else:
        hw = lit.transform(hw, live)
        merge(hw, settings.get("hardware") or {})
    if layout is not None:
        megatron_software(hw, layout)
    apply_static_power(hw, entry, run, settings)
    apply_a100_latency(hw, settings)
    rep.ok(f"hardware {hw_id}.{base_key}: base sha256 {want}; transform({spec['sku']}, {spec['literature_tier']}, "
           f"{run_type}) + settings applied in memory")
    return hw

def check_sha256sums(rep: Report, sums: Path, must_list: Path):
    entries = [l.split(None, 1) for l in sums.read_text().splitlines() if l.strip()]
    if not entries:
        rep.err(f"frozen set {sums.parent.name}: SHA256SUMS is empty")
        return
    names = {n.strip().lstrip("*") for _, n in entries}
    if must_list.name not in names:
        rep.err(f"frozen set {sums.parent.name}: {must_list.name} is not listed in SHA256SUMS")
    bad = [n.strip() for d, n in entries
           if not (sums.parent / n.strip().lstrip("*")).exists() or sha256(sums.parent / n.strip().lstrip("*")) != d]
    (rep.err if bad else rep.ok)(f"frozen set {sums.parent.name}: SHA256SUMS ({len(entries)} files) "
                                 f"{'mismatch: ' + ', '.join(bad) if bad else 'all match'}")

def require_columns(rep: Report, label: str, header, cols) -> bool:
    missing = [c for c in cols if c not in (header or [])]
    if missing:
        rep.err(f"{label}: columns missing {missing}")
    return not missing

PHASES = {"prefill", "decode", "train", "infer"}

def runs_from_manifest(rep: Report, inputs: Inputs, label: str, path: Path, framework: dict,
                       scored_statuses=("scored",)):
    rd = csv.DictReader(open(path))
    if not require_columns(rep, label, rd.fieldnames, ("tag", "status", "per_kernel_csv")):
        return []
    rows = [r for r in rd if r.get("status") in scored_statuses]
    runs = []
    for r in rows:
        tag = r["tag"]
        parts = tag.split("_")
        try:
            i = next(j for j, t in enumerate(parts) if t in PHASES)
            b, s = parts[i + 1].split("x")
            model, phase, suffix = "_".join(parts[:i]), parts[i], "_".join(parts[i + 2:])
            b, s = int(b), int(s)
        except (StopIteration, IndexError, ValueError):
            rep.err(f"{label}: cannot parse tag {tag!r}")
            continue
        kern = Path(r["per_kernel_csv"])
        kern = kern if kern.is_absolute() else ROOT / kern
        inputs.check(rep, f"{label} {tag} per-kernel csv", kern, quiet=True)
        fw = framework.get(phase) if "vllm" in label else framework.get("train" if phase == "train" else "pytorch_inference")
        if not fw:
            rep.err(f"{label} {tag}: no framework for phase {phase}")
        runs.append(dict(set=label, model=model, phase=phase, batch=b, seq_len=s, variant=suffix, framework=fw,
                         source=str(kern)))
    rep.ok(f"{label}: {len(rows)} scored captures in {path.name} (per-kernel csvs pinned)")
    return runs

MEGATRON_PER_ROW = "megatron_per_row"
PP_OWN_DIMENSION = "own_logical_dimension"

def grid_cells(path: Path) -> dict:
    grid = json.loads(Path(path).read_text())
    out = {}
    for g in grid["groups"]:
        for c in g["cells"]:
            out[c["cell"]] = dict(group=g["group"], model=g["model"], phase="train", batch=str(c["mb"]),
                                  seq_len=str(c["seq"]), grad_ckpt="0", tp=str(g["tp"]), pp=str(g["pp"]),
                                  dp=str(g["dp"]), dp_zero_stage=str(int(bool(g["dist_opt"]))),
                                  microbatches=str(g["microbatches"]), n_gpus=str(len(g["gpus"])),
                                  gpus=json.dumps(g["gpus"]))
    return out

def runs_from_csv(rep: Report, label: str, m: dict, path: Path, cols: dict, framework: dict, paths: dict | None = None):
    rd = csv.DictReader(open(path))
    flat = [x for c in cols.values() for x in (c if isinstance(c, list) else [c])]
    if not require_columns(rep, label, rd.fieldnames, flat + list((m.get("filter") or {}).keys())):
        return []
    filt = m.get("filter") or {}
    every = list(rd)
    selected = [all(r.get(k) == v for k, v in filt.items()) for r in every]
    rows = [r for r, sel in zip(every, selected) if sel]
    exp = m.get("expect_rows")
    if exp is None:
        rep.err(f"{label}: registry gives no expect_rows")
    elif len(rows) != exp:
        rep.err(f"{label}: {len(rows)} rows after filter {filt}, expected {exp}")
    else:
        rep.ok(f"{label}: {len(rows)} rows (filter {filt}), as expected")

    kept = m.get("report_other_statuses")
    grid = {}
    if kept:
        grid = grid_cells(resolve(kept["grid"], paths or {}))
        others = [r for r, sel in zip(every, selected) if not sel]
        if len(others) != kept.get("expect_rows"):
            rep.err(f"{label}: {len(others)} rows outside filter {filt}, expected {kept.get('expect_rows')}")
        else:
            rep.ok(f"{label}: {len(others)} rows with other measured statuses kept as measured (not scored)")
    runs = []
    for r, sel in zip(every, selected):
        if not sel:
            if not kept:
                continue
            cell = r[cols["cell"]]
            if cell not in grid:
                rep.err(f"{label}: cell {cell} is not in the grid")
                continue
            run = {k: r.get(c, "") for k, c in cols.items() if isinstance(c, str)}
            run.update({k: v for k, v in grid[cell].items() if k in cols})
            run["measured_status"] = r.get("status", "")
            run["excluded"] = (f"measured status {run['measured_status']}: kept as measured, not predicted or scored "
                               f"({kept['designation']})")
            run["framework"] = framework.get(run["phase"])
            runs.append(run)
            continue
        if kept:
            cell = r[cols["cell"]]
            want = {k: v for k, v in grid.get(cell, {}).items() if k in cols}
            got = {k: r[cols[k]] for k in want}
            if not want or want != got:
                rep.err(f"{label}: cell {cell} disagrees with the grid: {got} vs {want}")
        run = {k: ("_".join(r[x] for x in c) if isinstance(c, list) else r[c]) for k, c in cols.items()}
        if run["phase"] not in PHASES:
            rep.err(f"{label}: unknown phase {run['phase']!r}")
        run["framework"] = framework.get(run["phase"])
        if not run["framework"]:
            rep.err(f"{label}: no framework for phase {run['phase']}")
        if "sm_mhz_per_gpu" in run:
            try:
                run["clock_mhz"] = measured_clock(run)
            except (ValueError, KeyError) as ex:
                rep.err(f"{label} cell {run['cell']}: {ex}")
        runs.append(run)
    dup = sorted({r['cell'] for r in runs if sum(x['cell'] == r['cell'] for x in runs) > 1}) if 'cell' in cols else []
    if dup:
        rep.err(f"{label}: cell ids not unique ({len(dup)}, e.g. {dup[:3]}); make workload.columns.cell a column list")
    return runs

def measured_clock(run: dict) -> dict:
    clocks = [float(v) for v in run["sm_mhz_per_gpu"].split(";")]
    if len(clocks) != int(run["n_gpus"]) or not all(math.isfinite(v) and v > 0 for v in clocks):
        raise ValueError("sm_mhz_per_gpu must contain one positive finite clock per GPU")
    return dict(mean=sum(clocks) / len(clocks), min=min(clocks), max=max(clocks), per_gpu=clocks)

def hold_state() -> dict:
    return (load_yaml(HERE / "designations.yaml") or {}).get("hold") or {}

def hold_active() -> bool:
    return bool(hold_state().get("active", True))

def hold_exception(entry_id: str):
    document = load_yaml(HERE / "designations.yaml") or {}
    for exception in (document.get("hold") or {}).get("exceptions") or []:
        if not isinstance(exception, dict) or exception.get("entry") != entry_id:
            continue
        designation_id = exception.get("designation")
        designation = (document.get("designations") or {}).get(designation_id)
        if valid_designation(designation):
            return dict(entry=entry_id, designation=designation_id, **designation)
    return None

def hold_blocks(entry_id: str = None) -> bool:
    return hold_active() and (entry_id is None or hold_exception(entry_id) is None)

def record_hold_exception(entry_id: str, root: Path):
    exception = hold_exception(entry_id) if hold_active() else None
    if exception:
        root.mkdir(parents=True, exist_ok=True)
        (root / "hold_exception.json").write_text(json.dumps(exception, indent=1, default=str) + "\n")
        print(f"hold exception: {entry_id}, designation {exception['designation']}: {exception['decision']}")

def hold_note() -> str:
    h = hold_state()
    return "energy scoring and reruns on hold; lifted by setting hold.active: false in designations.yaml"

PREVIEW_IGNORED = ("HOLD", "flash rebuild")

def preview_state() -> dict:
    return (load_yaml(HERE / "designations.yaml") or {}).get("preview") or {}

def preview_allowed() -> bool:
    p = preview_state()
    return bool(p.get("allowed")) and bool(str(p.get("words") or "").strip())

def preview_blockers(blocked: list) -> list:
    return [b for b in blocked if not b.startswith(PREVIEW_IGNORED)]

SANITY = {"on": False}

def out_base(head: str, preview: bool) -> Path:
    tag = (f"preview_sanity_{head}" if SANITY["on"] else f"preview_{head}") if preview else head
    return ROOT / "paper_pipeline" / "out" / tag

def entry_output(base: Path, e: dict) -> Path:
    name = e.get("output_directory", e["id"])
    if not isinstance(name, str) or name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"invalid output_directory for {e['id']}: {name!r}")
    return base / name

def sanity_subset(runs: list) -> list:
    seen, keep = set(), []
    for r in runs:
        if r.get("unsupported") or r.get("excluded"):
            keep.append(r)
            continue
        if r["phase"] in seen:
            keep.append(dict(r, unsupported="not run in the sanity pass"))
            continue
        seen.add(r["phase"])
        keep.append(r)
    return keep

def preview_note(head: str) -> str:
    return (f"PREVIEW at commit {head[:12]}: flash attention on the OLD model (code and settings file as they stand at "
            f"this commit; the structural rebuild is not in); not the bulk run")

def hardware_key(run: dict) -> tuple:
    if run["phase"] != "train":
        return ("inference", "inference")
    if run.get("variant") == "nockpt" or str(run.get("grad_ckpt", "0")) in ("0", "False", "false", ""):
        return ("train_norecompute", "training")
    return ("train_fullrecompute", "training")

def variants_for(e: dict, reg: dict, multi: bool):
    import itertools
    applies = []
    for v in reg.get("variants") or []:
        if v.get("status") == "pending":
            continue
        scope = v.get("applies_to", "all")
        if scope == "all" or (scope == "multi_gpu" and multi) or (scope == "single_gpu" and not multi) \
                or (isinstance(scope, list) and e["id"] in scope):
            applies.append(v)
    if not applies:
        return [("default", {})]
    out = []
    for combo in itertools.product(*[list(v["alternatives"].items()) for v in applies]):
        name = "+".join(f"{v['id']}={alt}" for v, (alt, _val) in zip(applies, combo))
        overlay = {v["switch"]: val for v, (_alt, val) in zip(applies, combo)}
        out.append((name, overlay))
    return out

def apply_overlay(settings: dict, overlay: dict) -> dict:
    s = copy.deepcopy(settings)
    for dotted, val in overlay.items():
        d = s
        keys = dotted.split(".")
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val
    if dig(s, "model.implementation.kernel_rules.flash_rebuild") is True:
        rules = s["model"]["implementation"]["kernel_rules"]
        rules.pop("fa_tile_util_fwd", None)
        rules.pop("fa_tile_util_bwd", None)
        rules.pop("fa_prefill_scale", None)
        s["flag_sheet"] = [line for line in s.get("flag_sheet", []) if not line.startswith("C8/F6")]
        s["flag_sheet"].append("FLASH_STRUCTURAL: flash_rebuild true, structural model at compute utilisation 1.0")
    return s

def flash_ready(e, settings):
    if settings.get("flash_rebuild_available"):
        return True
    reg = load_yaml(HERE / "registry.yaml")
    multi = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1
    return all(dig(apply_overlay(settings, overlay), "model.implementation.kernel_rules.flash_rebuild") is True
               for _, overlay in variants_for(e, reg, multi))

MAX_WORKERS = 32
MIN_AVAILABLE_GB = 24.0

class MemoryGuard:
    def __init__(self, min_gb: float = MIN_AVAILABLE_GB):
        import threading
        self.min_gb, self.low_gb, self.waits, self.lock = min_gb, float("inf"), 0, threading.Lock()

    @staticmethod
    def available_gb() -> float:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
        return float("inf")

    def wait(self):
        import time
        while True:
            gb = self.available_gb()
            with self.lock:
                self.low_gb = min(self.low_gb, gb)
            if gb >= self.min_gb:
                return
            with self.lock:
                self.waits += 1
            time.sleep(10)

def run_points(e: dict, run: dict):
    from paper_pipeline import workload as W
    if run["phase"] != "decode":
        return [("", None, 1.0)]
    rules = e["workload"]["rules"]
    column = rules.get("decode_kv_column")
    if column and rules["decode_context"] != "fixed_seq":
        raise ValueError("decode_kv_column requires decode_context fixed_seq")
    context = int(run[column]) if column else int(run["seq_len"])
    return W.decode_points(rules["decode_context"], context, run)

def run_configs(rep, inputs, e: dict, run: dict, paths: dict, settings: dict, out_dir: Path, tag: str = "", kv=None):
    from paper_pipeline import workload as W
    key, run_type = hardware_key(run)
    hw = build_hardware(rep, inputs, e["hardware"], key, run_type, settings, run, e["id"])
    if hw is None:
        return None
    if int((hw.get("parallelism") or {}).get("tp", 1)) > 1:
        choice = dig(hw, "l2_model.collective_buffers")
        if choice not in ("occupy_l2", "leave_l2"):
            raise ValueError("tp > 1 requires explicit hardware.l2_model.collective_buffers")
    rules = e["workload"].get("rules") or {}

    merge(hw, e["workload"].get("hardware_overlay") or {})
    apply_static_power(hw, e, run, settings)
    apply_a100_latency(hw, settings)
    if run["phase"] != "train" and rules.get("inference_precision"):
        prec = hw.setdefault("sw_param", {}).setdefault("precision", {})
        for k in ("tensor_format", "kv_cache", "parameters", "stats"):
            prec[k] = rules["inference_precision"]
    impl = copy.deepcopy((settings.get("model") or {}).get("implementation") or {})
    merge(impl, rules.get("implementation") or {})
    impl["framework"] = run["framework"]
    family = run.get("family", "llama")
    extra = {}
    arch = None
    if family in {"qwen2", "qwen3"}:
        _, arch = architecture_of(load_yaml(HERE / "registry.yaml"), run["model"])
        extra = W.published_inputs(arch)
        extra["qwen2_bias_mode"] = settings["model"].get("qwen2_bias_mode")
    if arch and rules.get("qwen_config_source") == "published":
        cfg = extra["published_cfg"]
    else:
        cfg = W.load_hf_config(resolve(e["workload"]["model_configs"], paths), run["model"], architecture=arch)
    layout = megatron_layout(e, run)
    batch = layout["global_batch"] if layout else int(run["batch"])
    if layout and layout["dp_buckets"] != "per_reducer":
        impl.setdefault("options", {})["megatron_dp_buckets"] = layout["dp_buckets"]
    model = W.model_yaml(cfg, run["phase"], batch, int(run["seq_len"]), kv, impl, family, **extra)
    if run["phase"] != "train" and rules.get("engine_max_model_len_column"):
        limit = int(run[rules["engine_max_model_len_column"]])
        if limit <= 0:
            raise ValueError("engine max_model_len must be positive")
        model["model_param"]["engine"] = {"max_model_len": limit}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{run['cell']}__{tag}" if tag else f"{run['cell']}"
    hp, mp = out_dir / f"{stem}.hardware.yaml", out_dir / f"{stem}.model.yaml"
    hp.write_text(yaml.safe_dump(hw, sort_keys=False))
    mp.write_text(yaml.safe_dump(model, sort_keys=False))
    return hp, mp

def predict(e: dict, run: dict, hp: Path, mp: Path, out_json: Path, commit: str, timeout: int | None = None):
    timeout = int(timeout or (e.get("harness") or {}).get("timeout_s") or 1800)
    cmd = [sys.executable, "-I", "-B", str(HERE / "predict_energy.py"), "--hardware", str(hp), "--model", str(mp),
           "--phase", run["phase"], "--out", str(out_json), "--commit", commit]
    exception = hold_exception(e["id"]) if hold_active() else None
    if exception:
        cmd += ["--hold-exception", json.dumps(exception, default=str)]
    static_policy = ((load_yaml(hp).get("metadata") or {}).get("static_power_policy") if hp.is_file() else None)
    cmd += ["--run-metadata", json.dumps({"entry": e["id"], "cell": run.get("cell"),
                                        "clock_mhz": run.get("clock_mhz"), "static_power": static_policy})]
    result = subprocess.run(cmd, env=clean_env(), capture_output=True, text=True, timeout=timeout)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.with_suffix(".stdout.log").write_text(result.stdout)
    out_json.with_suffix(".stderr.log").write_text(result.stderr)
    return result

def request_configs(rep, inputs, e: dict, run: dict, settings: dict, out_dir: Path):
    from paper_pipeline import overlay as O

    rules = e["workload"].get("rules") or {}
    settings = copy.deepcopy(settings)
    merge(settings.setdefault("hardware", {}).setdefault("sw_param", {}), rules.get("sw_param") or {})
    merge(settings.setdefault("model", {}).setdefault("implementation", {}), rules.get("implementation") or {})
    if run.get("implementation_options"):
        merge(settings["model"]["implementation"], {"options": dict(run["implementation_options"])})
    if run.get("hardware_entry"):
        hw = build_hardware(rep, inputs, run["hardware_entry"], "inference", "inference", settings, entry_id=e["id"])
        if hw is None:
            return None
        h_ass = []
    else:
        if not inputs.check(rep, f"study hardware {run['cell']}", Path(run["hardware_src"]), quiet=True):
            return None
        source_hw = load_yaml(Path(run["hardware_src"]))
        if hardware_architecture(source_hw) == "sm90":
            rep.err("sm90 supports COUNTS only; study energy and timing scoring are refused")
            return None
        lit = import_from_tree("a100_literature")
        live = lit.Live(pattern="", sku=run["live"][0], tier=run["live"][1], run_type=run["live"][2])
        fixed = bool((settings.get("network") or {}).get("collective_launch_overhead_key_available"))

        hw, _applied, h_ass = O.overlay_hardware(source_hw, settings, lit, live,
                                                 run.get("inter_node"), fixed, e["id"], ())
    if int((hw.get("parallelism") or {}).get("tp", 1)) > 1:
        if dig(hw, "l2_model.collective_buffers") not in ("occupy_l2", "leave_l2"):
            raise ValueError("tp > 1 requires explicit hardware.l2_model.collective_buffers")
    if not inputs.check(rep, f"model {run['cell']}", Path(run["model_src"]), quiet=True):
        return None
    src_model = load_yaml(Path(run["model_src"]))
    mt = str(src_model["model_param"].get("model_type", "")).lower()
    if run.get("family") and mt != run["family"]:
        rep.err(f"{run['cell']}: study model_type {mt!r} differs from the published family {run['family']!r}; not relabelled")
        return None
    model, _m = O.overlay_model(src_model, settings, run["framework"], run.get("gemv_option"))

    for key, value in (rules.get("model_param") or {}).items():
        model["model_param"][key] = value
    apply_static_power(hw, e, run, settings)
    apply_a100_latency(hw, settings)
    run["assumptions"] = "; ".join(h_ass)
    out_dir.mkdir(parents=True, exist_ok=True)
    hp, mp = out_dir / f"{run['cell']}.hardware.yaml", out_dir / f"{run['cell']}.model.yaml"
    hp.write_text(yaml.safe_dump(hw, sort_keys=False))
    mp.write_text(yaml.safe_dump(model, sort_keys=False))
    return hp, mp

def apply_unsupported_labels(rep, e: dict, runs: list):
    for u in e["workload"].get("unsupported") or []:
        hit = [r for r in runs if all(str(r.get(k, "")) == str(v) for k, v in u.items() if k != "reason")]
        for r in hit:
            r["unsupported"] = u["reason"]
        if hit:
            rep.lines.append(f"  info    {len(hit)} run(s) labelled unsupported, will stop rather than run: "
                             f"{u['model']} {u['phase']} ({u['reason']})")

def plan_request_entry(rep, inputs, e, settings, n_err, n_blk):
    from paper_pipeline import suites
    if not flash_ready(e, settings):
        rep.block("flash rebuild: structural flash-attention model not merged / switched on yet (FLASH_STRUCTURAL)")
    if load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1 and \
            not (settings.get("network") or {}).get("collective_launch_overhead_key_available"):
        rep.block("C1: sw_param.collective_launch_overhead not available; multi-GPU runs wait for it")
    for cls in e.get("blocked_on") or []:
        rep.block(f"waits for port class: {cls}")
    if "predict" not in (e.get("harness") or {}):
        rep.block("no prediction harness yet; open H2")
    if hold_blocks(e["id"]):
        rep.block(f"HOLD: {hold_note()}")
    try:
        runs = suites.LOADERS[e["loader"]](e)
    except Exception as ex:
        rep.err(f"loader {e['loader']}: {type(ex).__name__}: {ex}")
        runs = []
    if not runs:
        rep.err("no runs found")
    apply_scope(rep, e, runs, load_yaml(HERE / "registry.yaml"))
    apply_unsupported_labels(rep, e, runs)
    apply_architecture_rule(rep, e, runs, load_yaml(HERE / "registry.yaml"), settings)
    nrefs = sum(len(r["refs"]) for r in runs)
    rep.ok(f"loader {e['loader']}: {len(runs)} runs, {nrefs} reference quantities")
    for r in runs:
        for k in ("hardware_src", "model_src"):
            if r.get(k):
                inputs.check(rep, f"{k} {r['cell']}", Path(r[k]), quiet=True)
    first = next((r for r in runs if not r.get("unsupported")), None)
    if first:
        import tempfile
        reg = load_yaml(HERE / "registry.yaml")
        multi = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1
        for name, overlay in variants_for(e, reg, multi):
            with tempfile.TemporaryDirectory() as td:
                made = request_configs(rep, inputs, e, dict(first), apply_overlay(settings, overlay), Path(td))
            (rep.ok if made else rep.err)(f"config overlay of the first supported run ({first['cell']}, {name}) "
                                         f"{'builds' if made else 'failed'}")
    by = {}
    for r in runs:
        by[(r["panel"], r["phase"])] = by.get((r["panel"], r["phase"]), 0) + 1
    for (pan, ph), n in sorted(by.items()):
        rep.lines.append(f"  would run  {n:4d} x {pan:24s} {ph:8s} framework {e['framework']}")
    errs, blks = rep.errors[n_err:], rep.blocks[n_blk:]
    status = "ERROR" if errs else ("BLOCKED" if blks else "READY (dry)")
    partial = {}
    for r in runs:
        if r.get("unsupported"):
            partial[r["unsupported"]] = partial.get(r["unsupported"], 0) + 1
    return dict(id=e["id"], status=status, runs=len(runs), run_list=runs, errors=errs, blocked=blks, partial=partial)

def architecture_of(reg: dict, model: str):
    key = (reg.get("model_aliases") or {}).get(model, model)
    return key, (reg.get("architectures") or {}).get(key)

def described(arch: dict, settings: dict):
    fams = settings.get("described_families") or {}
    d = fams.get(arch["family"])
    if d is None:
        extra = []
        if any(arch["projection_bias"].values()):
            extra.append("projection biases " + "/".join(k for k, v in arch["projection_bias"].items() if v))
        if arch.get("qk_norm"):
            extra.append("q/k RMSNorm")
        if arch.get("moe"):
            extra.append("MoE")
        return f"{arch['family']} description ({', '.join(extra) or arch['hf_architecture']})"
    if any(arch["projection_bias"].values()) != bool(d.get("projection_bias")) or bool(arch.get("qk_norm")) != bool(d.get("qk_norm")):
        return f"{arch['family']} variant not described ({arch['hf_architecture']})"
    return None

def apply_scope(rep, e: dict, runs: list, reg: dict):
    counts = {}

    for rule in sorted(reg.get("scope") or [], key=lambda r: r["kind"] != "omit_entry_sets_phases"):
        if rule["kind"] == "omit_entry_sets_phases":
            if e["id"] in rule["entries"]:
                omitted = [r for r in runs if r.get("set") in rule["sets"] and r["phase"] in rule["phases"]]
                runs[:] = [r for r in runs if not (r.get("set") in rule["sets"] and r["phase"] in rule["phases"])]
                if omitted:
                    rep.lines.append(f"  info    {len(omitted)} cell(s) omitted by scope designation {rule['designation']} "
                                     "(not predicted, not scored, not reported)")
            continue
        if rule["kind"] not in ("exclude_models", "exclude_prefill_prompt_tokens"):
            rep.err(f"registry scope: unknown rule kind {rule['kind']!r} (designation {rule.get('designation')})")
            continue
        for r in runs:
            if r.get("excluded"):
                continue
            hit = False
            if rule["kind"] == "exclude_models":
                hit = (reg.get("model_aliases") or {}).get(r["model"], r["model"]) in rule["models"]
            elif rule["kind"] == "exclude_prefill_prompt_tokens":
                hit = r["phase"] == "prefill" and int(r.get("seq_len") or -1) in [int(x) for x in rule["tokens"]]
            if hit:
                r["excluded"] = f"{rule['designation']}: {rule['reason']}"
                counts[rule["designation"]] = counts.get(rule["designation"], 0) + 1
    for rid, n in sorted(counts.items()):
        rep.lines.append(f"  info    {n} cell(s) excluded by scope designation {rid} (not predicted, not scored)")
    return counts

def apply_architecture_rule(rep, e: dict, runs: list, reg: dict, settings: dict):
    reasons = {}
    for r in runs:
        key, arch = architecture_of(reg, r["model"])
        if arch is None:
            rep.err(f"no architecture entry for model {r['model']!r} (registry architectures / model_aliases)")
            continue
        r["family"], r["arch_key"] = arch["family"], key
        why = described(arch, settings)
        if why and not r.get("unsupported") and not r.get("excluded"):
            r["unsupported"] = why
            reasons[why] = reasons.get(why, 0) + 1
    for why, n in sorted(reasons.items()):
        (rep.block if n == len(runs) else lambda m: rep.lines.append("  info    " + m))(
            f"{n} of {len(runs)} run(s) unsupported until the {why}")
    for m in e.get("models") or []:
        key, arch = architecture_of(reg, m)
        if arch is None:
            rep.err(f"no architecture entry for model {m!r}")
        elif described(arch, settings):
            rep.block(f"model {m}: {described(arch, settings)}")
    return reasons

ALLOWED_KEYS = {"id", "title", "status", "hardware", "workload", "framework", "measured", "harness", "scorer",
                "outputs", "notes", "exceptions", "figure", "assumptions", "blocked_on", "kind", "loader", "models",
                "output_directory"}

def gpu_work_idle_power(e):
    from paper_pipeline.scorers.gpu_work import load_idle_power
    return load_idle_power((e.get("workload") or {}).get("gpu_work_idle_power"), ROOT)

def gpu_work_summary(e, runs):
    from paper_pipeline.scorers.gpu_work import GPU_WORK, NO_GPU_WORK, NODE_RAW, NODE_RAW_BASIS
    if e.get("scorer") not in ("energy_ratio", "request_energy"):
        return "not applicable (counts or prediction-only study)"
    if (e.get("workload") or {}).get("score_basis") == NODE_RAW:
        kept = sum(bool(r.get("measured_status")) for r in runs)
        return f"{NODE_RAW_BASIS}: {len(runs) - kept}; other measured status, kept and not scored: {kept}"
    count = sum(r.get("gpu_kernel_ms_per_step") not in (None, "") for r in runs)
    return f"{GPU_WORK}: {count}; {NO_GPU_WORK}: {len(runs) - count}"

def plan_entry(rep: Report, inputs: Inputs, e: dict, paths: dict, settings: dict, open_needed: dict, designations: dict):
    n_err, n_blk = len(rep.errors), len(rep.blocks)
    rep.lines.append(f"\n[{e['id']}] {e['title']}  (registry status: {e['status']})")
    for path in e.get("workload", {}).get("input_pins") or []:
        inputs.check(rep, f"{e['id']} workload input", resolve(path, paths))
    spec = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml")

    staged = e.get("status") == "blocked" and bool(e.get("blocked_on")) and e.get("scorer") == "counts_join"
    deferred_base = (staged and spec.get("integration_source") and isinstance(spec.get("base"), dict)
                     and any(not (ROOT / base).is_file() for base in spec["base"].values()))
    if deferred_base:
        inputs.check(rep, f"hardware entry {e['hardware']}", HERE / "hardware" / f"{e['hardware']}.yaml")
        rep.block(f"hardware {e['hardware']}: base supplied by {spec['integration_source']}")
    if not deferred_base and isinstance(spec.get("base"), dict) and spec["base"]:
        for base in sorted(set(spec["base"].values())):
            inputs.check(rep, f"hardware {e['hardware']} base", ROOT / base, quiet=True)
        try:
            if entry_architecture(e) == "sm90" and e.get("scorer") != "counts_join":
                raise ValueError("sm90 supports COUNTS only; energy and timing scoring are refused")
        except (ValueError, OSError) as exc:
            rep.err(str(exc))
            return dict(id=e['id'], status='ERROR', runs=0, run_list=[],
                        errors=rep.errors[n_err:], blocked=[], partial={})
    permission = hold_exception(e["id"]) if hold_active() else None
    if permission:
        rep.ok(f"HOLD exception: {permission['designation']} for {e['id']}: {permission['decision']}")
    extra = set(e) - ALLOWED_KEYS
    if extra:
        rep.err(f"keys not allowed in a registry entry: {sorted(extra)} (put them under exceptions with a D1 id)")
    if e["id"] == "counts_h100_fullstep":
        bundled = ROOT / "opmodel/src/opmodel/configs/gemm_catalogs/sm90_evidence/lt_heur_forced_h100.csv"
        for _name, overlay in variants_for(e, load_yaml(HERE / "registry.yaml"), False):
            forced = overlay.get("model.implementation.gemm_library.forced_templates")
            if forced:
                reference = resolve(forced, paths)
                available = inputs.check(rep, "H100 LT-HEUR bundled evidence", bundled)
                available = inputs.check(rep, "H100 LT-HEUR reference", reference) and available
                if available and sha256(bundled) != sha256(reference):
                    rep.err("H100 LT-HEUR reference differs from the frozen SM90 bundled evidence")
    exc_open = open_designations(e.get("exceptions") or [], designations)
    if exc_open:
        rep.block(f"exceptions without a designation: {exc_open}")
    multi = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1
    need = list(open_needed.get("all_entries", [])) + (list(open_needed.get("multi_gpu_entries", [])) if multi else [])
    if need:
        rep.block(f"open settings designations: {', '.join(need)}")
    if e["status"] != "admitted":
        rep.block(f"registry status is {e['status']!r}")
    if not flash_ready(e, settings):
        rep.block("flash rebuild: structural flash-attention model not merged / switched on yet (FLASH_STRUCTURAL)")
    if multi and not (settings.get("network") or {}).get("collective_launch_overhead_key_available"):
        rep.block("C1: sw_param.collective_launch_overhead not available; multi-GPU runs wait for it")
    for cls in e.get("blocked_on") or []:
        rep.block(f"waits for port class: {cls}")

    if e.get("kind") == "request_cases":
        return plan_request_entry(rep, inputs, e, settings, n_err, n_blk)
    runs = []
    for m in e.get("measured") or []:
        label = f"measured.{m['set']}"
        if "manifest" in m:
            p = resolve(m["manifest"], paths)
            planned_missing = m.get("planned") and not p.exists()
            if planned_missing and staged:
                rep.block(f"{label}: planned captures not yet available at {p}")
            elif planned_missing:
                rep.err(f"{label}: planned input requires blocked counts status and blocked_on")
            available = not planned_missing and inputs.check(rep, label, p)
            if available:
                runs += runs_from_manifest(rep, inputs, m["set"], p, e["framework"], m.get("scored_statuses", ("scored",)))
                if m.get("provenance"):
                    inputs.check(rep, f"{label} provenance", resolve(m["provenance"], paths))

                    with p.open(newline="") as stream:
                        for row in csv.DictReader(stream):
                            if row["status"] in m.get("scored_statuses", ("scored",)):
                                inputs.check(rep, f"{label} {row['tag']} raw capture",
                                             p.parent / "cap" / f"{row['tag']}.csv", quiet=True)
            if available or (planned_missing and staged):
                for pending in m.get("pending_runs") or []:
                    captured = any(r["set"] == m["set"] and not r.get("variant") and not r.get("missing")
                                   and all(r.get(k) == pending[k]
                                   for k in ("model", "phase", "batch", "seq_len")) for r in runs)
                    if pending.get("variant"):
                        row = dict(pending, set=m["set"], source="", framework=e["framework"][pending["phase"]],
                                   capture_tag=f"{pending['model']}_{pending['phase']}_{pending['batch']}x{pending['seq_len']}")
                        if not captured:
                            row["missing"] = "not captured yet"
                        runs.append(row)
                    elif not captured:
                        runs.append(dict(pending, set=m["set"], variant=pending.get("variant", ""), source="",
                                         framework=e["framework"][pending["phase"]], missing="not captured yet"))
                        rep.lines.append(f"  info    {m['set']} {pending['model']} {pending['phase']}: not captured yet")
        elif "csv" in m:
            p = resolve(m["csv"], paths)
            if inputs.check(rep, label, p):
                runs += runs_from_csv(rep, m["set"], m, p, e["workload"]["columns"], e["framework"], paths)
                if m.get("sha256sums"):
                    check_sha256sums(rep, resolve(m["sha256sums"], paths), p)
                elif m.get("frozen"):
                    rep.err(f"{label}: frozen set without SHA256SUMS")
        elif "capture_dir" in m:
            d = resolve(m["capture_dir"], paths)
            n = len(list(d.glob("*_kernels.csv"))) if d.exists() else 0
            if n == 0 and not m.get("optional"):
                rep.err(f"{label}: no captures in {d}")
            else:
                rep.lines.append(f"  info    {label}: {n} captures in {d}{' (optional, not used yet)' if m.get('optional') else ''}")
        else:
            rep.err(f"{label}: no manifest, csv or capture_dir")
    if not runs:
        if e.get("blocked_on") and not e.get("measured"):
            rep.block("no runs defined yet (generator pending, see blocked_on)")
        else:
            rep.err("no runs found")

    apply_scope(rep, e, runs, load_yaml(HERE / "registry.yaml"))
    apply_architecture_rule(rep, e, runs, load_yaml(HERE / "registry.yaml"), settings)
    if e.get("scorer") == "energy_ratio":
        from paper_pipeline.scorers.gpu_work import measured_basis, node_raw_basis, NODE_RAW
        basis_of = node_raw_basis if e["workload"].get("score_basis") == NODE_RAW else measured_basis
        if e["workload"].get("score_basis") not in (None, NODE_RAW):
            rep.err(f"{e['id']}: unknown workload.score_basis {e['workload']['score_basis']!r}")
        idle_spec = e["workload"].get("gpu_work_idle_power")
        idle = None
        if idle_spec and inputs.check(rep, f"{e['id']} GPU-work idle power", resolve(idle_spec["path"], paths)):
            try:
                idle = gpu_work_idle_power(e)
                if idle["mode"] == "per_cell":
                    for cell, measurement in sorted(idle["cells"].items()):
                        source = measurement.get("source")
                        if not source:
                            raise ValueError(f"cell {cell} idle power has no source")
                        inputs.check(rep, f"{e['id']} cell {cell} idle baseline", resolve(source, paths))
            except (ValueError, KeyError, OSError) as ex:
                rep.err(f"{e['id']} idle-power input: {ex}")
        for run in runs:
            try:
                basis = basis_of(run, e["workload"]["measured_per_step"], idle)
                if basis["gpu_gap_s"] != "" and basis["gpu_gap_s"] < 0:
                    rep.warn(f"{e['id']} cell {run['cell']}: kernel union exceeds measured wall time; signed gap retained")
            except (ValueError, KeyError, TypeError) as ex:
                rep.err(f"{e['id']} cell {run['cell']} GPU-work reference: {ex}")
    apply_unsupported_labels(rep, e, runs)
    mc = resolve(e["workload"]["model_configs"], paths)
    for mm in sorted({r["model"] for r in runs}):
        _, arch = architecture_of(load_yaml(HERE / "registry.yaml"), mm)
        qwen = arch and arch["family"] in {"qwen2", "qwen3", "qwen3_moe"}
        if qwen:
            inputs.check(rep, f"published model config {mm}", ROOT / arch["hf_config"], quiet=True)
        short = mc / f"{mm}.json"
        published_only = qwen and e["workload"].get("rules", {}).get("qwen_config_source") == "published"
        if not published_only and (short.exists() or not qwen):
            inputs.check(rep, f"model config {mm}", short, quiet=True)
    rep.ok(f"model configs: {len({r['model'] for r in runs})} models in {mc}")

    outside = []
    for k, h in (e.get("harness") or {}).items():
        if k == "timeout_s":
            continue
        p = resolve(h, paths)
        inputs.check(rep, f"harness.{k}", p)
        if not inside_tree(p):
            outside.append(k)
    if outside:
        rep.block(f"harness scripts outside the tree: {', '.join(outside)}")
    energy = e.get("scorer") in ("energy_ratio", "request_energy", "sweep", "counts_join")
    if energy and e.get("scorer") != "counts_join" and "predict" not in (e.get("harness") or {}):
        rep.block("no prediction harness yet; open H2")
    if energy and hold_blocks(e["id"]):
        rep.block(f"HOLD: {hold_note()}")

    spec = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml")
    keys = {hardware_key(r) for r in runs}
    if deferred_base:
        pass
    elif (spec.get("clock_hz") == "per_run_measured" or
          (e.get("workload", {}).get("gpu_work_idle_power") or {}).get("mode") == "per_cell") and spec.get("base") is not None:
        for run in runs:
            if run.get("measured_status"):
                continue
            key, run_type = hardware_key(run)
            try:
                build_hardware(rep, inputs, e["hardware"], key, run_type, settings, run, e["id"])
            except (ValueError, KeyError) as ex:
                rep.err(f"hardware {e['hardware']} cell {run['cell']}: {ex}")
    elif spec.get("base") is None or not isinstance(spec.get("clock_hz"), (int, float)):
        build_hardware(rep, inputs, e["hardware"], "inference", "inference", settings, entry_id=e["id"])
    else:
        for key, run_type in sorted(keys):
            build_hardware(rep, inputs, e["hardware"], key, run_type, settings, entry_id=e["id"])

    by = {}
    for r in runs:
        k = (r.get("set", "runs"), r["phase"], r.get("framework"))
        by[k] = by.get(k, 0) + 1
    for (s, ph, fw), n in sorted(by.items(), key=str):
        rep.lines.append(f"  would run  {n:4d} x {s:12s} {ph:8s} framework {fw}")
    errs, blks = rep.errors[n_err:], rep.blocks[n_blk:]
    status = "ERROR" if errs else ("BLOCKED" if blks else "READY (dry)")
    partial = {}
    for r in runs:
        if r.get("unsupported"):
            partial[r["unsupported"]] = partial.get(r["unsupported"], 0) + 1
    return dict(id=e["id"], status=status, runs=len(runs), run_list=runs, errors=errs, blocked=blks, partial=partial)

def cmd_plan(args):
    rep = Report()
    head = git("rev-parse", "HEAD")
    dirty = [l for l in git("status", "--porcelain").splitlines() if l]
    rep.lines.append(f"tree {ROOT} @ {head} ({git('rev-parse', '--abbrev-ref', 'HEAD')}); "
                     f"{'DIRTY: ' + str(len(dirty)) + ' paths (run/score/plot would refuse)' if dirty else 'clean'}")
    for p in FOREIGN_SITE:
        rep.warn(f"interpreter loads third-party packages from another tree's venv: {p} (libraries only; open H4)")
    rep.lines.append(f"  info    environment: {'removed from this process: ' + ', '.join(REMOVED_ENV) if REMOVED_ENV else 'no RAPID_/OPMODEL_/COUNTS_ variables'}")
    settings = load_yaml(HERE / "settings/paper.yaml")
    designations = (load_yaml(HERE / "designations.yaml") or {}).get("designations") or {}
    reg = load_yaml(HERE / "registry.yaml")
    ids = [e["id"] for e in reg["entries"]]
    unknown = [x for x in args.entries if x not in ids]
    if unknown:
        print(f"unknown registry entries: {unknown}; known: {ids}", file=sys.stderr)
        return 2
    inputs = Inputs(lock=args.lock)
    for p in ("settings/paper.yaml", "registry.yaml", "designations.yaml"):
        inputs.check(rep, p, HERE / p)
    for p in settings.get("input_pins") or []:
        inputs.check(rep, "paper package input", Path(p) if Path(p).is_absolute() else ROOT / p)
    open_needed = check_settings(rep, settings, designations)
    results = [plan_entry(rep, inputs, e, reg["paths"], settings, open_needed, designations)
               for e in reg["entries"] if not args.entries or e["id"] in args.entries]
    bad = foreign_modules()
    if bad:
        rep.err(f"modules loaded from another code tree: {bad[:5]}")
    print("\n".join(rep.lines))
    print("\nsummary:")
    for r in results:
        e = next(x for x in reg["entries"] if x["id"] == r["id"])
        multi = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1
        nv = len(variants_for(e, reg, multi))
        print(f"  {r['id']:22s} {r['status']:12s} {r['runs']:4d} runs x {nv} variant(s)  errors {len(r['errors'])}  "
              f"blocked by: {'; '.join(sorted(set(b.split(':')[0] if b.startswith('HOLD') else b for b in r['blocked']))) or '-'}")
        print(f"  {'':22s} {gpu_work_summary(e, r.get('run_list', []))}")
        for why, n in sorted((r.get("partial") or {}).items()):
            print(f"  {'':22s} {n} of {r['runs']} cells wait for: {why}")
    print(f"  inputs: {len(inputs.seen)} files ({sum(not v['inside_tree'] for v in inputs.seen.values())} outside the tree); "
          f"errors {len(rep.errors)}")
    if getattr(args, "preview", False):
        print(f"\npreview (runs past HOLD and 'flash rebuild' only; permission in designations.yaml: "
              f"{'GIVEN' if preview_allowed() else 'not given'}; outputs to paper_pipeline/out/preview_<commit>/):")
        for r in results:
            if preview_blockers(r["blocked"]) or r["errors"]:
                continue
            e = next(x for x in reg["entries"] if x["id"] == r["id"])
            multi = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1
            nv = len(variants_for(e, reg, multi))
            uns = sum(1 for x in r["run_list"] if x.get("unsupported") and not x.get("excluded"))
            exc = sum(1 for x in r["run_list"] if x.get("excluded"))
            print(f"  {r['id']:22s} {r['runs'] - uns - exc:4d} cells run, {uns:3d} unsupported (reported), "
                  f"{exc:3d} excluded by scope designations, x {nv} variant(s)")
    if args.lock:
        if args.entries:
            print("--lock needs the full registry (no ENTRY selection)", file=sys.stderr)
            return 2
        LOCK.write_text(json.dumps(dict(note="pinned by driver.py plan --lock",
                                        inputs={k: v["sha256"] for k, v in sorted(inputs.seen.items())}), indent=1) + "\n")
        print(f"  pinned {len(inputs.seen)} inputs in {LOCK.relative_to(ROOT)}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(commit=head, dirty=bool(dirty), entries=results, inputs=inputs.seen,
                                                   errors=rep.errors, blocked=rep.blocks), indent=1) + "\n")
    return 1 if rep.errors else 0

def cmd_manifest(_args):
    rows = list(csv.DictReader(open(HERE / "manifest.csv")))
    by = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    print(f"{len(rows)} manifest rows: " + ", ".join(f"{k} {v}" for k, v in sorted(by.items())))
    return 0

def _entry_runs(entry_id: str):
    settings = load_yaml(HERE / "settings/paper.yaml")
    designations = (load_yaml(HERE / "designations.yaml") or {}).get("designations") or {}
    reg = load_yaml(HERE / "registry.yaml")
    e = next((x for x in reg["entries"] if x["id"] == entry_id), None)
    if e is None:
        raise SystemExit(f"unknown registry entry {entry_id!r}")
    rep, inputs = Report(), Inputs(lock=False)
    res = plan_entry(rep, inputs, e, reg["paths"], settings, check_settings(Report(), settings, designations), designations)
    multi = load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1
    if SANITY["on"]:
        res["run_list"] = sanity_subset(res["run_list"])
    return e, reg, settings, rep, inputs, res, variants_for(e, reg, multi)

def _jobs(e, res):
    for run in res["run_list"]:
        if run.get("unsupported") or run.get("excluded"):
            continue
        if e.get("kind") == "request_cases":
            yield run, "", None
            continue
        for tag, kv, _w in run_points(e, run):
            yield run, tag, kv

def _configs(inputs, e, run, reg, settings, out, tag, kv):
    if e.get("kind") == "request_cases":
        return request_configs(Report(), inputs, e, run, settings, out)
    return run_configs(Report(), inputs, e, run, reg["paths"], settings, out, tag, kv)

VARIANT_ROWS = {"multi_gpu42_vllm": (32, "energy_per_step"),
                "multi_gpu32_locked1065": (32, "energy_per_step"),
                "kareus_training": (14, "train_energy_at_pred_time"),
                "megatron_multi_a100": (33, "energy_per_step")}

VARIANT_LATENCY = {}

def variant_row_summary(e, res, variants, settings, root=None, reg=None):
    if e["id"] not in VARIANT_ROWS:
        return
    expected, _quantity = VARIANT_ROWS[e["id"]]
    if reg is None:
        reg = load_yaml(HERE / "registry.yaml")
    required_variants = variants_for(e, reg, True)
    energy_rules = {dig(apply_overlay(settings, overlay), "network.link_energy_rule")
                    for _, overlay in required_variants}
    required_latency = VARIANT_LATENCY.get(e["id"], "tuning_formula")
    runs = [r for r in res["run_list"] if not r.get("unsupported") and not r.get("excluded")]
    if len(runs) != expected or len({r["cell"] for r in runs}) != expected:
        raise ValueError(f"{e['id']} requires exactly {expected} distinct labelled rows")
    choices, product = [], []
    for name, overlay in variants:
        scoped = apply_overlay(settings, overlay)
        choice = dig(scoped, "hardware.l2_model.collective_buffers")
        if choice not in ("occupy_l2", "leave_l2"):
            raise ValueError(f"{name}: missing explicit collective_buffers choice")
        choices.append(choice)
        energy = dig(scoped, "network.link_energy_rule")
        latency = dig(scoped, "network.nccl_latency_rule")
        if latency != required_latency:
            raise ValueError(f"{name}: requires nccl_latency_rule {required_latency}")
        product.append((choice, energy))
        points_found = 0
        if root is not None:
            for run in runs:
                for tag, _kv, _w in run_points(e, run):
                    stem = f"{run['cell']}__{tag}" if tag else run["cell"]
                    path = root / name / "pred" / f"{stem}.json"
                    if not path.exists():
                        continue
                    rec = json.loads(path.read_text())
                    got = dig(rec, "settings.settings.RAPID_L2_COLLECTIVE_BUFFERS.value")
                    if got != choice:
                        raise ValueError(f"{path}: collective_buffers runtime value {got!r} != {choice!r}")
                    for key, expected_value in (("RAPID_LINK_ENERGY_RULE", energy), ("RAPID_NCCL_LATENCY_RULE", latency)):
                        got = dig(rec, f"settings.settings.{key}.value")
                        if got != expected_value:
                            raise ValueError(f"{path}: {key} runtime value {got!r} != {expected_value!r}")
                    if dig(rec, "settings.settings.RAPID_FLASH_REBUILD.value") is not True:
                        raise ValueError(f"{path}: structural flash rebuild was not active")
                    if e["id"] == "multi_gpu42_vllm" and \
                            dig(rec, "pipeline_settings.hold_exception.designation") != "MGPU_SCORE":
                        raise ValueError(f"{path}: missing scoped hold permission")
                    points_found += 1
        suffix = f", {points_found} runtime points verified" if root is not None else ""
        print(f"{e['id']}: {name}: rows per variant = {len(runs)}{suffix}")

    if sorted(set(choices)) != ["leave_l2", "occupy_l2"] or choices.count("leave_l2") != choices.count("occupy_l2"):
        raise ValueError(f"{e['id']} requires both collective_buffers variants, equally crossed")
    expected_product = {(buffer, energy) for buffer in ("occupy_l2", "leave_l2") for energy in energy_rules}
    if len(product) != len(expected_product) or set(product) != expected_product:
        raise ValueError(f"{e['id']} requires the complete collective_buffers x link_energy_rule product, once each")

def score_assumptions(e, settings, overlay):
    scoped = apply_overlay(settings, overlay)
    notes = (e.get("assumptions") or []) + scoped.get("flag_sheet", [])
    sw = (e.get("workload", {}).get("hardware_overlay") or {}).get("sw_param") or {}
    for key in ("inference_step_overhead", "inference_prefill_overhead"):
        if key in sw:
            notes.append(f"workload overlay: sw_param.{key} = {sw[key]} s (time only)")
    if load_yaml(HERE / "hardware" / f"{e['hardware']}.yaml").get("gpus", 0) != 1:
        notes += scoped.get("flag_sheet_multi_gpu", [])
    permission = hold_exception(e["id"]) if hold_active() else None
    if permission:
        notes += [f"HOLD exception: entry {e['id']}, designation {permission['designation']}"]
    return "; ".join(notes)

ASTRA_NOT_BUILT = "AstraSim is not built in this tree: run reproduce/setup.sh"

def astra_sim_ready(root=ROOT):
    a = root / "astra-sim"
    return ((a / "build").is_dir()
            and (a / "extern" / "graph_frontend" / "chakra" / "schema" / "protobuf" / "et_def_pb2.py").is_file())

def _require_astra(e):
    hw = e.get("hardware")
    if not hw or not (HERE / "hardware" / f"{hw}.yaml").is_file():
        return
    if load_yaml(HERE / "hardware" / f"{hw}.yaml").get("gpus", 0) != 1 and not astra_sim_ready():
        raise SystemExit(ASTRA_NOT_BUILT)

def cmd_run(args):
    e, reg, settings, rep, inputs, res, variants = _entry_runs(args.entry)
    head = git("rev-parse", "HEAD")
    if args.plumbing_test:
        out = Path(args.out or "").resolve()
        scratch = (ROOT / "build/tmp").resolve()
        if not args.out or (inside_tree(out) and not out.is_relative_to(scratch)):
            raise SystemExit("--plumbing-test needs --out DIR outside the tree or under build/tmp")
        if res["errors"]:
            raise SystemExit(f"plan errors: {res['errors']}")
        if e.get("scorer") == "counts_join":
            from paper_pipeline import counts_stage
            selected = [r for r in res["run_list"] if not r.get("unsupported")
                        and (not args.phase or r["phase"] == args.phase)
                        and (not args.cell or counts_stage._cell(r) == args.cell)]
            if not selected:
                raise SystemExit("no supported counts run for the requested phase/cell")
            vname, overlay = variants[0]
            return counts_stage.run(e, selected[:1], settings, vname, overlay, out, head, 1)
        want = args.phase
        run, tag, kv = next((j for j in _jobs(e, res) if (not want or j[0]["phase"] == want)
                             and (not args.cell or str(j[0]["cell"]) == args.cell)), (None, None, None))
        if run is None:
            raise SystemExit(f"no supported run{' for phase ' + want if want else ''}")
        vname, overlay = variants[0]
        made = _configs(inputs, e, run, reg, apply_overlay(settings, overlay), out / "configs", tag, kv)
        if made is None:
            raise SystemExit("hardware not buildable for this entry")
        stem = f"{run['cell']}__{tag}" if tag else run["cell"]
        _require_astra(e)
        p = predict(e, run, *made, out / f"{stem}.json", head)
        if p.returncode != 0:
            print(p.stdout[-2000:], p.stderr[-4000:], file=sys.stderr)
            return 1
        j = json.loads((out / f"{stem}.json").read_text())
        print(f"plumbing test {e['id']} [{vname}] {stem} ({run.get('model')} {run['phase']} b{run.get('batch')} s{run.get('seq_len', '')}): "
              f"JSON written; phases {sorted(j['phases'])}; settings recorded: {bool(j.get('settings'))}; "
              f"commit {j['commit'][:12]}; elapsed {getattr(p, 'elapsed', 'n/a')}")
        return 0
    dirty = [l for l in git("status", "--porcelain").splitlines() if l]
    preview = bool(getattr(args, "preview", False))
    if preview:
        if not preview_allowed():
            print("refusing the preview: designations.yaml preview.allowed is not set", file=sys.stderr)
            return 2
        blk = preview_blockers(res["blocked"])
        if blk or res["errors"] or dirty:
            print(f"refusing to preview {e['id']}: blocked={blk}, errors={res['errors']}, dirty={len(dirty)}", file=sys.stderr)
            return 2
    elif hold_blocks(e["id"]) or res["blocked"] or res["errors"] or dirty:
        print(f"refusing to run {e['id']}: hold={hold_active()}, blocked={res['blocked']}, errors={res['errors']}, "
              f"dirty={len(dirty)} paths", file=sys.stderr)
        return 2
    base = out_base(head, preview)
    variant_row_summary(e, res, variants, settings)
    _require_astra(e)
    record_hold_exception(e["id"], entry_output(base, e))
    if e.get("scorer") == "counts_join":
        from paper_pipeline import counts_stage
        out = entry_output(base, e)
        results = [counts_stage.run(e, res["run_list"], settings, name, overlay, out, head, args.workers)
                   for name, overlay in variants]
        return int(any(results))
    jobs = list(_jobs(e, res))
    failed = 0
    from concurrent.futures import ThreadPoolExecutor

    def one(job):
        (vname, overlay), (run, tag, kv) = job
        out = entry_output(base, e) / vname
        made = _configs(inputs, e, run, reg, apply_overlay(settings, overlay), out / "configs", tag, kv)
        if made is None:
            return 1
        stem = f"{run['cell']}__{tag}" if tag else run["cell"]
        return predict(e, run, *made, out / "pred" / f"{stem}.json", head).returncode
    work = [(v, j) for v in variants for j in jobs]
    workers = min(int(args.workers), MAX_WORKERS)
    guard = MemoryGuard()

    def guarded(job):
        guard.wait()
        return one(job)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        failed = sum(rc != 0 for rc in ex.map(guarded, work))
    print(f"memory: lowest MemAvailable seen {guard.low_gb:.1f} GB; waited {guard.waits} times")
    print(f"{e['id']}: {len(work)} predictions ({len(jobs)} points x {len(variants)} variants), {failed} failed")
    variant_row_summary(e, res, variants, settings, entry_output(base, e))
    return 1 if failed else 0

def cmd_score(args):
    e, reg, settings, rep, inputs, res, variants = _entry_runs(args.entry)
    preview = bool(getattr(args, "preview", False))
    dirty = [l for l in git("status", "--porcelain").splitlines() if l]
    if preview:
        if not preview_allowed():
            print("refusing the preview: designations.yaml preview.allowed is not set", file=sys.stderr)
            return 2
        if preview_blockers(res["blocked"]) or res["errors"] or dirty:
            print(f"refusing to score the preview of {e['id']}", file=sys.stderr)
            return 2
    else:
        if hold_blocks(e["id"]):
            print(f"refusing to score: {hold_note()}", file=sys.stderr)
            return 2
        if res["blocked"] or res["errors"] or dirty:
            print(f"refusing to score {e['id']}: blocked={res['blocked']}, errors={res['errors']}, dirty={len(dirty)}",
                  file=sys.stderr)
            return 2
    head0 = git("rev-parse", "HEAD")
    if preview:
        settings = dict(settings, flag_sheet=[preview_note(head0)] + list(settings.get("flag_sheet") or []))
    stale = [p for p in (entry_output(out_base(head0, preview), e)).glob("*/pred/*.json")
             if json.loads(p.read_text()).get("commit") != head0]
    if stale:
        print(f"refusing to score {e['id']}: {len(stale)} predictions carry another commit (e.g. {stale[0]})", file=sys.stderr)
        return 2
    from paper_pipeline.scorers import energy_ratio
    from paper_pipeline.scorers.common import write_cells
    import csv as _csv
    head = git("rev-parse", "HEAD")
    root = entry_output(out_base(head, preview), e)
    variant_row_summary(e, res, variants, settings, root)
    record_hold_exception(e["id"], root)
    if e.get("scorer") == "counts_join":
        from paper_pipeline import counts_stage
        rc = counts_stage.score(e, res["run_list"], root, [name for name, _ in variants], head)
        if preview and (root / "scores" / "cells.csv").exists():
            rows = list(_csv.DictReader(open(root / "scores" / "cells.csv")))
            for r in rows:
                r["assumptions"] = "; ".join(x for x in (preview_note(head), r.get("assumptions", "")) if x)
            write_cells(rows, root / "scores" / "cells.csv")
        return rc
    rows = []
    for vname, _ov in variants:
        tmp = root / vname / "scores" / "cells.csv"
        if e.get("kind") == "request_cases":
            from paper_pipeline.scorers import request_energy
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                for run in res["run_list"]:
                    if not run.get("unsupported"):
                        request_configs(Report(), inputs, e, run, apply_overlay(settings, _ov), Path(td))
            request_energy.score(e["id"], res["run_list"], root / vname / "pred", tmp, variant=vname,
                                 assumptions=score_assumptions(e, settings, _ov),
                                 profile_required=bool(settings.get("settings_profile_required")))
            rows += list(_csv.DictReader(open(tmp)))
            continue
        energy_ratio.score(e["id"], e.get("figure", e["id"]), res["run_list"], e["workload"]["measured_per_step"],
                           lambda r: run_points(e, r), root / vname / "pred", tmp, variant=vname,
                           assumptions=score_assumptions(e, settings, _ov),
                           profile_required=bool(settings.get("settings_profile_required")),
                           idle_power=gpu_work_idle_power(e), basis=e["workload"].get("score_basis"))
        rows += list(_csv.DictReader(open(tmp)))
    for r in rows:
        r["commit"] = head0
    write_cells(rows, root / "scores" / "cells.csv")
    if e["id"] in VARIANT_ROWS:
        expected, quantity = VARIANT_ROWS[e["id"]]
        for name, _ in variants:
            count = sum(r["variant"] == name and r["quantity"] == quantity and r["status"] != "excluded" for r in rows)
            if count != expected:
                raise ValueError(f"{name}: score summary has {count} rows, expected {expected}")
            print(f"score summary {name}: rows per variant = {count}")
    from paper_pipeline.flags.report import make_report
    make_report(out_base(head, preview))
    print(f"{e['id']}: {len(rows)} cell rows ({len(variants)} variants) -> {root / 'scores' / 'cells.csv'}")
    return 0

def cmd_rehearse(args):
    import random
    from paper_pipeline.scorers import energy_ratio, request_energy
    from paper_pipeline.scorers.common import write_cells
    from paper_pipeline.plots.plot_all import plot_all
    from paper_pipeline.flags.report import make_report
    from paper_pipeline.plots import dummy
    import csv as _csv
    out = Path(args.out).resolve()
    if inside_tree(out):
        raise SystemExit("--out must be outside the tree")
    rnd = random.Random(20261113)
    reg = load_yaml(HERE / "registry.yaml")
    for e0 in reg["entries"]:
        if e0.get("scorer") not in ("energy_ratio", "request_energy"):
            continue
        e, reg, settings, rep, inputs, res, variants = _entry_runs(e0["id"])
        rows = []
        for vname, _ov in variants[:2]:
            pred = out / e["id"] / vname / "pred"
            pred.mkdir(parents=True, exist_ok=True)
            for run in res["run_list"]:
                f = rnd.uniform(0.82, 1.18)
                if run.get("measured_status"):
                    continue
                if e.get("kind") == "request_cases":
                    phases = {}
                    for ref in run["refs"]:
                        ph = ref["predicted_from"].split(".")[0]
                        meas = float(ref["measured"]) if ref.get("measured") not in (None, "") else 1.0
                        if ref["quantity"] == "train_energy_at_pred_time" and ref.get("curve"):
                            meas = ref["curve"][0][1]
                        val = meas * f * (run["batch"] if ref["predicted_from"].endswith("/batch") else 1)
                        key = "time_s" if ".time_s" in ref["predicted_from"] else "total_j"
                        phases.setdefault(ph, {"time_s": 1.0, "total_j": 1.0})[key] = val
                    (pred / f"{run['cell']}.json").write_text(json.dumps({"commit": "DUMMY", "phases": phases}))
                else:
                    steps = float(run[e["workload"]["measured_per_step"]["steps"]])
                    mt = float(run[e["workload"]["measured_per_step"]["runtime"]]) / steps
                    me = float(run[e["workload"]["measured_per_step"]["energy"]]) / steps
                    for tag, _kv, _w in run_points(e, run):
                        stem = f"{run['cell']}__{tag}" if tag else run["cell"]
                        (pred / f"{stem}.json").write_text(json.dumps({"commit": "DUMMY", "phases": {
                            run["phase"]: {"time_s": mt * f, "total_j": me * f}}}))
            tmp = out / e["id"] / vname / "scores" / "cells.csv"
            if e.get("kind") == "request_cases":
                request_energy.score(e["id"], res["run_list"], pred, tmp, variant=vname, assumptions="DUMMY")
            else:
                energy_ratio.score(e["id"], e.get("figure", e["id"]), res["run_list"], e["workload"]["measured_per_step"],
                                   lambda r, e=e: run_points(e, r), pred, tmp, variant=vname, assumptions="DUMMY",
                                   idle_power=gpu_work_idle_power(e), basis=e["workload"].get("score_basis"))
            rows += list(_csv.DictReader(open(tmp)))
        write_cells(rows, out / e["id"] / "scores" / "cells.csv")
        print(f"rehearsal {e['id']}: {len(rows)} cell rows ({min(2, len(variants))} variants)")

    tmpd = out / "_dummy_src"
    dummy.write_dummy(tmpd)
    for name in ("counts_fullstep", "cs_interactivity", "cs_disagg"):
        src = tmpd / name
        if src.exists():
            import shutil
            shutil.copytree(src, out / name, dirs_exist_ok=True)
    import shutil
    shutil.rmtree(tmpd, ignore_errors=True)
    pdfs = plot_all(out)
    make_report(out)
    print(f"rehearsal: {len(pdfs)} figures in {out / 'figures'}; flag report {out / 'flags' / 'FLAGS.md'}")
    return 0

def cmd_run_all(args):
    preview = bool(getattr(args, "preview", False))
    SANITY["on"] = bool(getattr(args, "sanity", False)) and preview
    if preview and not preview_allowed():
        print("refusing the preview: designations.yaml preview.allowed is not set", file=sys.stderr)
        return 2
    if not preview and hold_active():
        print(f"refusing: {hold_note()}", file=sys.stderr)
        return 2
    reg = load_yaml(HERE / "registry.yaml")
    head = git("rev-parse", "HEAD")
    done, skipped = [], []
    for e in reg["entries"]:
        _e, _reg, _s, _rep, _inp, res, _v = _entry_runs(e["id"])
        blk = preview_blockers(res["blocked"]) if preview else res["blocked"]
        if blk or res["errors"]:
            skipped.append((e["id"], blk + res["errors"]))
            continue
        rc = cmd_run(argparse.Namespace(entry=e["id"], plumbing_test=False, out=None, phase=None, cell=None,
                                        workers=args.workers, preview=preview))
        rc = rc or cmd_score(argparse.Namespace(entry=e["id"], preview=preview))
        (done if rc == 0 else skipped).append((e["id"], [] if rc == 0 else [f"run/score exit {rc}"]))
    out = out_base(head, preview)
    from paper_pipeline.plots.plot_all import plot_all
    from paper_pipeline.flags.report import make_report
    pdfs = plot_all(out)
    make_report(out)
    if preview:
        renamed = []
        for p in pdfs:
            q = Path(p).with_name(f"PREVIEW_{head[:12]}_{Path(p).name}")
            Path(p).rename(q)
            renamed.append(q)
        pdfs = renamed
        fl = out / "flags" / "FLAGS.md"
        fl.write_text(f"# {preview_note(head)}\n\n" + fl.read_text())
    with open(out / "flags" / "FLAGS.md", "a") as f:
        f.write("\n## Entries not run at this commit\n\n| Entry | Why |\n| --- | --- |\n")
        for eid, why in skipped:
            f.write(f"| {eid} | {'; '.join(why).replace('|', '/')} |\n")
    print(f"run-all at {head}: {len(done)} entries done, {len(skipped)} skipped; {len(pdfs)} figures; flags {out / 'flags'}")
    for eid, why in skipped:
        print(f"  skipped {eid}: {'; '.join(why)[:300]}")
    return 0 if not skipped else 3

def cmd_hold(_args):
    print(HOLD, file=sys.stderr)
    return 2

def main(argv=None):
    ap = argparse.ArgumentParser(description="Paper pipeline driver.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("entries", nargs="*")
    p.add_argument("--json")
    p.add_argument("--lock", action="store_true")
    p.add_argument("--preview", action="store_true", help="also list what a labelled preview would run")
    p.set_defaults(fn=cmd_plan)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    q = sub.add_parser("run")
    q.add_argument("entry")
    q.add_argument("--plumbing-test", action="store_true", help="ONE prediction into --out (outside the tree); value not reported")
    q.add_argument("--out")
    q.add_argument("--phase", choices=("prefill", "decode", "train"), help="plumbing test: pick a run of this phase")
    q.add_argument("--cell", help="plumbing test: pick this cell id")
    q.add_argument("--workers", type=int, default=32, help="parallel CPU prediction processes (no GPU; capped at 32)")
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("score")
    q.add_argument("entry")
    q.set_defaults(fn=cmd_score)
    q = sub.add_parser("run-all")
    q.add_argument("--workers", type=int, default=32, help="parallel CPU prediction processes per entry (no GPU; capped at 32)")
    q.add_argument("--preview", action="store_true", help="labelled PREVIEW: entries blocked only by HOLD / flash rebuild")
    q.add_argument("--sanity", action="store_true", help="with --preview: one supported cell per (entry, phase), end to end")
    q.set_defaults(fn=cmd_run_all)
    q = sub.add_parser("rehearse")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_rehearse)
    q = sub.add_parser("plot")
    q.add_argument("rest", nargs="*")
    q.set_defaults(fn=cmd_hold)
    args = ap.parse_args(argv)
    return args.fn(args)

if __name__ == "__main__":
    sys.exit(main())
