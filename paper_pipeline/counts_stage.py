from __future__ import annotations

import copy
import csv
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from paper_pipeline import driver as D
from paper_pipeline import workload as W
from paper_pipeline.scorers.common import cell_row, write_cells

HARNESS = Path(__file__).resolve().parent / "harness"
QUANTITIES = ("hbm_bytes", "l2_bytes", "smem_bytes", "tensor_flops")
PRED_FIELDS = ("variant", "workload", "model", "phase", "op_class", "role", "counter", "value")
ALIASES = ("m12_fused", "m12_eager_l2sim", "m12_eager_l2cls", "m12_eager_l2cls_epi",
           "m12_fused_vctx", "m12_fused_l2cls_epi")
CAPTURE_OPTIONS = {"pytorch_v1": "--capture-root", "vllm_v1": "--vllm-capture-root",
                   "pytorch_v2": "--capture-v2-root", "vllm_v2": "--vllm-capture-v2-root"}
CAPTURE_FORMATS = {"vllm_qwen_v2": "vllm_v2"}

def _path(value):
    paths = D.load_yaml(D.HERE / "registry.yaml")["paths"] if "{" in str(value) else {}
    return D.resolve(str(value), paths).resolve()

def _source(run):
    if CAPTURE_FORMATS.get(run["set"], run["set"]) not in CAPTURE_OPTIONS:
        raise ValueError(f"unknown counts capture set: {run['set']}")
    return run["set"].split("_")[0]

def _tag(run):
    suffix = "_" + run["variant"] if run.get("variant") else ""
    return f"{run['model']}_{run['phase']}_{run['batch']}x{run['seq_len']}{suffix}"

def _cell(run):
    value = f"{run['set']}__{_tag(run)}"
    if "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid counts cell: {value!r}")
    return value

def _write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def _read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def _execute(script, args):
    env = D.clean_env()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for key in ("TMPDIR", "MPLCONFIGDIR"):
        if os.environ.get(key):
            env[key] = os.environ[key]

    target = D.HERE.parent / "counts_run_one.py" if script == "counts_run_one.py" else HARNESS / script
    return subprocess.run([sys.executable, "-I", str(target), *map(str, args)],
                          env=env, capture_output=True, text=True, timeout=1800)

def _checked(script, args, log):
    p = _execute(script, args)
    log.write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError(f"{script} exited {p.returncode}; see {log}")

def _configs(entry, run, settings, directory):
    source = _source(run)
    rules = entry["workload"]["rules"]
    settings = copy.deepcopy(settings)
    D.merge(settings.setdefault("hardware", {}).setdefault("sw_param", {}), rules.get("sw_param") or {})
    impl = copy.deepcopy(settings["model"]["implementation"])
    D.merge(impl, rules["implementation"][source])
    if run.get("framework") and impl["framework"] != run["framework"]:
        raise ValueError(f"{_cell(run)}: framework disagrees with the registry implementation rule")
    impl.setdefault("kernel_rules", {})["prefill_logits_last_token"] = rules["prefill_logits_last_token"]
    checkpointed = (run["phase"] == "train" and run["model"] == "Llama-3.2-3B"
                    and run.get("variant") != "nockpt")
    base = ("train_fullrecompute" if checkpointed else "train_norecompute") if run["phase"] == "train" else "inference"
    rep = D.Report()
    hw = D.build_hardware(rep, D.Inputs(lock=False), entry["hardware"], base,
                          "training" if run["phase"] == "train" else "inference", settings, entry_id=entry["id"])
    if hw is None or rep.errors:
        raise ValueError(f"hardware not buildable: {rep.errors + rep.blocks}")
    if D.hardware_architecture(hw) == "sm90":
        hardware_settings = settings.get("hardware") or {}
        for key in ("sw_param", "l2_model"):
            D.merge(hw.setdefault(key, {}), hardware_settings.get(key) or {})
        if (hw.get("metadata") or {}).get("counts_only") is not True:
            raise ValueError("sm90 counts prediction requires metadata.counts_only: true")
    directory.mkdir(parents=True, exist_ok=True)
    hp, ip = directory / "hardware.yaml", directory / "implementation.yaml"
    hp.write_text(yaml.safe_dump(hw, sort_keys=False))
    ip.write_text(yaml.safe_dump(impl, sort_keys=False))
    family = run.get("family", "llama")
    extra = {}
    arch = None
    if family in {"qwen2", "qwen3", "qwen3_moe"}:
        _, arch = D.architecture_of(D.load_yaml(D.HERE / "registry.yaml"), run["model"])
        extra = W.published_inputs(arch)
        if family == "qwen3_moe":
            extra["moe_rules"] = dict(rules.get("moe") or {},
                                      collective_rule=settings["model"].get("moe_source_collective_rule"))
            if run.get("moe_tile_rule"):
                extra["moe_rules"]["moe_tile_rule"] = run["moe_tile_rule"]
        else:
            extra["qwen2_bias_mode"] = settings["model"].get("qwen2_bias_mode")
    cfg = W.load_hf_config(_path(entry["workload"]["model_configs"]), run["model"], architecture=arch)
    kv = rules["decode_kv_tokens_on_entry"][source] if run["phase"] == "decode" else None
    model = W.model_yaml(cfg, run["phase"], run["batch"], run["seq_len"], kv, impl, family, **extra)
    mp = directory / "model.yaml"
    mp.write_text(yaml.safe_dump(model, sort_keys=False))
    if arch:
        facts = directory / "hf_configs"
        facts.mkdir(exist_ok=True)
        (facts / f"{run['model']}.json").write_text(json.dumps(extra["published_cfg"]) + "\n")
    return hp, ip, mp

def run(entry, runs, settings, variant_name, overlay, out_dir, commit, workers):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    effective = D.apply_overlay(settings, overlay)
    config_dir = _path(entry["workload"]["model_configs"])
    guard = D.MemoryGuard()
    cells = [_cell(r) for r in runs]
    if len(cells) != len(set(cells)):
        raise ValueError("duplicate counts run identities")

    def one(r):
        cell = _cell(r)
        state = dict(cell=cell, capture_set=r["set"], commit=commit, status="ok")
        if r.get("excluded"):
            return {**state, "status": "excluded", "reason": r["excluded"]}, []
        if r.get("unsupported"):
            return {**state, "status": "unsupported", "reason": r["unsupported"]}, []
        if r.get("missing") and not r.get("predict"):
            return {**state, "status": "missing", "reason": r["missing"]}, []
        raw = out_dir / "raw" / variant_name / cell
        raw.mkdir(parents=True, exist_ok=True)
        prediction = raw / "prediction.json"
        prediction.unlink(missing_ok=True)
        try:
            hp, ip, mp = _configs(entry, r, effective, raw / "configs")
            run_config_dir = mp.parent / "hf_configs" if (mp.parent / "hf_configs").is_dir() else config_dir
            args = [r["model"], r["phase"], r["batch"], r["seq_len"], prediction, raw / "work",
                    "--out-dir", raw / "configs", "--config-dir", run_config_dir,
                    "--hardware-yaml", hp, "--implementation-yaml", ip, "--model-yaml", mp]
            if r.get("variant") == "nockpt":
                args.append("--nockpt")
            if r["phase"] == "decode":
                args += ["--decode-kv-on-entry", entry["workload"]["rules"]["decode_kv_tokens_on_entry"][_source(r)]]
            guard.wait()
            _checked("counts_run_one.py", args, raw / "run.log")
            data = json.loads(prediction.read_text())
            native = data.get("native_gemm_timing")
            if not isinstance(native, dict) or set(native) - {"flash_tile"}:
                raise ValueError(f"native_gemm_timing contains unsupported reasons: {native}")
            data.update(commit=commit, capture_set=r["set"], pipeline_variant=variant_name)
            prediction.write_text(json.dumps(data) + "\n")
            (raw / "TREE_COMMIT.txt").write_text(commit + "\n")

            view = raw / "aggregate_input"
            view.mkdir(exist_ok=True)
            model = yaml.safe_load(Path(data["model_config"]).read_text())
            impl = model["model_param"].get("implementation")
            if isinstance(impl, dict):
                model["model_param"]["implementation"] = impl["kind"]
            mp = view / "model.yaml"
            mp.write_text(yaml.safe_dump(model, sort_keys=False))
            adapted = {**data, "model_config": str(mp)}
            (view / "prediction.json").write_text(json.dumps(adapted) + "\n")
            (view / "TREE_COMMIT.txt").write_text(commit + "\n")
            _checked("aggregate.py", ["--raw-dir", view, "--config-dir", run_config_dir,
                     "--out-dir", raw, "m12_pipeline"], raw / "aggregate.log")
            rows = _read_csv(raw / "pred_m12_pipeline.csv")

            phase = data["phases"][r["phase"]]
            assert phase["counts_scope"] == "ncu_kernels_excluding_memcpy", phase.get("counts_scope")
            a = phase["counts_activity"]
            totals = {"hbm_read": a["hbm"]["read_bytes"],
                      "hbm_write": a["hbm"]["write_bytes"],
                      "l2": a["l2"]["read_bytes"] + a["l2"]["write_bytes"],
                      "smem": a["smem"]["read_bytes"] + a["smem"]["write_bytes"]}
            for row in rows:
                is_total = row["op_class"] == "total" and not row["role"]
                row["counts_scope"] = "ncu_kernels_excluding_memcpy" if is_total else "full_traffic_diagnostic"
                if is_total and row["counter"] in totals:
                    row["value"] = totals[row["counter"]]
            for row in rows:
                row.update(variant=variant_name, cell=cell, capture_set=r["set"], commit=commit)
            return state, rows
        except (ValueError, KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return {**state, "status": "failed", "reason": str(exc)}, []

    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), D.MAX_WORKERS))) as pool:
        results = list(pool.map(one, runs))
    states = [state for state, _ in results]
    _write_csv(out_dir / f"pred_{variant_name}.csv", [r for _, rows in results for r in rows],
               [*PRED_FIELDS, "cell", "capture_set", "commit", "counts_scope"])
    (out_dir / f"run_status_{variant_name}.json").write_text(json.dumps(states, indent=2) + "\n")
    failed = sum(s["status"] == "failed" for s in states)
    print(f"{entry['id']} [{variant_name}]: {len(states)} runs, {failed} failed")
    return int(bool(failed))

def _capture(entry, run):
    item = next(m for m in entry["measured"] if m["set"] == run["set"])
    root = _path(item["manifest"]).parent if "manifest" in item else _path(item["capture_dir"]).parent
    return root

def _figure_for_run(entry, run, predictions, work, evidence_dir=None):
    tag = run.get("capture_tag") or _tag(run)
    root = _capture(entry, run)
    kernels = root / "cap" / f"{tag}_kernels.csv"
    raw_csv = root / "cap" / f"{tag}.csv"
    if not kernels.is_file() or not raw_csv.is_file():
        return []
    manifest = root / "MANIFEST.csv"
    records = ([r for r in _read_csv(manifest) if r["tag"] == tag] if manifest.exists()
               else [dict(tag=tag, status="scored")])
    if not records:
        return []
    pred_dir = work / "pred"

    for alias in ALIASES:
        rows = [{k: r[k] for k in PRED_FIELDS} for r in predictions]
        for row in rows:
            row["variant"] = alias

            if run["model"] != "Llama-3.2-3B" and run.get("variant") == "nockpt":
                row["workload"] = row["workload"].removesuffix("_nockpt")
        _write_csv(pred_dir / f"pred_{alias}.csv", rows, PRED_FIELDS)
    args = []
    for capture_set, option in CAPTURE_OPTIONS.items():
        view = work / capture_set
        (view / "cap").mkdir(parents=True)
        selected = records if capture_set == CAPTURE_FORMATS.get(run["set"], run["set"]) else []
        _write_csv(view / "MANIFEST.csv", [{"tag": r["tag"], "status": r["status"]} for r in selected],
                   ["tag", "status"])
        if selected:
            for src in (kernels, raw_csv):
                (view / "cap" / src.name).symlink_to(src)
        args += [option, view]
    output = work / "figure_counts.csv"
    config_dir = _path(entry["workload"]["model_configs"])
    if run.get("family") in {"qwen2", "qwen3", "qwen3_moe"}:
        _, arch = D.architecture_of(D.load_yaml(D.HERE / "registry.yaml"), run["model"])
        config_dir = work / "hf_configs"
        config_dir.mkdir()
        (config_dir / f"{run['model']}.json").write_text(json.dumps(W.published_inputs(arch)["published_cfg"]) + "\n")
    _checked("make_figure_csv.py", [*args, "--architecture", D.entry_architecture(entry), "--config-dir", config_dir,
             "--prediction-dir", pred_dir, "--out-dir", work / "join", "--output", output], work / "figure.log")
    if evidence_dir is not None:
        for summary in (work / "join").glob("*_hopper.json"):
            evidence_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(summary, evidence_dir / summary.name)
    return _read_csv(output)

def _unjoined_predictions(rows):
    if not rows:
        return {}
    counts = {(r["op_class"], r["counter"]): float(r["value"]) for r in rows if not r["role"]}
    return {"hbm_bytes": counts[("total", "hbm_read")] + counts[("total", "hbm_write")],
            "l2_bytes": counts[("total", "l2")], "smem_bytes": counts[("total", "smem")],
            "tensor_flops": sum(counts[(cls, counter)] for cls in ("gemm", "attention")
                                for counter in ("tensor_useful", "tensor_padding"))}

def _hopper_verification():
    designation = (D.load_yaml(D.HERE / "designations.yaml").get("designations") or {}).get("HOPPER_COUNTERS_VERIFIED") or {}
    if designation.get("value") is not True:
        return False, "HOPPER_COUNTERS_VERIFIED=false: counter semantics UNCONFIRMED"
    if not D.valid_designation(designation) or not all(designation.get(k) for k in ("verified_by", "verified_at", "evidence_path")):
        raise ValueError("HOPPER_COUNTERS_VERIFIED=true requires verified_by, verified_at and evidence_path")
    evidence = _path(designation["evidence_path"])
    if not evidence.is_file():
        raise ValueError(f"HOPPER_COUNTERS_VERIFIED evidence missing: {evidence}")
    return True, (f"HOPPER_COUNTERS_VERIFIED: {designation['verified_by']} at {designation['verified_at']}; "
                  f"evidence={evidence}; sha256={D.sha256(evidence)}")

def score(entry, runs, out_dir, variant_names, commit):
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures, cells = [], []
    hopper = D.entry_architecture(entry) == "sm90"
    verified, verification = _hopper_verification() if hopper else (False, "")
    for variant in variant_names:
        path = out_dir / f"pred_{variant}.csv"
        predictions = _read_csv(path) if path.exists() else []
        if any(r.get("commit") != commit for r in predictions):
            raise ValueError(f"{path}: predictions carry another commit")
        by_cell = {}
        for r in predictions:
            by_cell.setdefault(r["cell"], []).append(r)
        for run in runs:
            cell = _cell(run)
            joined = []
            if not run.get("unsupported") and not run.get("excluded") and by_cell.get(cell):
                with tempfile.TemporaryDirectory(prefix="counts-join-", dir=out_dir) as td:
                    joined = _figure_for_run(entry, run, by_cell[cell], Path(td),
                                            out_dir / "hopper_evidence" / variant / cell)
            quantities = {r["count"]: r for r in joined}
            unjoined = _unjoined_predictions(by_cell.get(cell, [])) if not joined else {}
            for row in joined:
                current = {k: v for k, v in row.items() if not k.startswith("pred_") or k == "pred_final"}
                if hopper:
                    complete = current.get("measured", "") != "" and current.get("pred_final", "") != ""
                    current.update(status=("ok" if verified else "ok_unverified_counters") if complete else "missing",
                                   counter_verification=verification)
                figures.append({**current, "cell": cell, "variant": variant, "commit": commit})
            for quantity in QUANTITIES:
                row = quantities.get(quantity, {})
                measured, predicted = row.get("measured", ""), row.get("pred_final", unjoined.get(quantity, ""))
                if quantity == "l2_bytes" and row.get("l2_definition") != "delivered":
                    measured = ""
                status = "excluded" if run.get("excluded") else ("unsupported" if run.get("unsupported") else (
                    "ok" if measured != "" and predicted != "" else "missing"))
                assumptions = run.get("unsupported") or run.get("missing", "")
                if hopper:
                    if status == "ok" and not verified:
                        status = "ok_unverified_counters"
                    assumptions = "; ".join(filter(None, (assumptions, verification,
                                           "Hopper kernel roles unassigned; packed-half normalization UNCONFIRMED")))

                result = cell_row(entry=entry["id"], figure=entry.get("figure", "counts"), panel=run["phase"],
                    group=run["model"], cell=cell, description=_tag(run), variant=variant, quantity=quantity,
                    unit="FLOP" if quantity == "tensor_flops" else "bytes", measured=measured, predicted=predicted,
                    status="missing" if measured != "" and float(measured) == 0 else status,
                    label=run.get("excluded") or run["set"], assumptions=assumptions, commit=commit)
                result["status"] = status
                cells.append(result)
                if quantity not in quantities:
                    figures.append(dict(cell=cell, variant=variant, commit=commit, capture_set=run["set"].split("_")[-1],
                        measured_source=_source(run), model=run["model"], phase=run["phase"], batch=run["batch"],
                        seq_len=run["seq_len"], count=quantity, measured="", pred_final=predicted, status=status))
    fields = list(dict.fromkeys(k for r in figures for k in r)) or ["cell", "variant", "commit", "count"]
    _write_csv(out_dir / "figure_counts.csv", figures, fields)
    write_cells(cells, out_dir / "scores" / "cells.csv")
    print(f"{entry['id']}: {len(cells)} count cells -> {out_dir / 'scores' / 'cells.csv'}")
    return 0
