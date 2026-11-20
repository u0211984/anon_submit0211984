#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_ENV = ("RAPID_", "OPMODEL_", "COUNTS_")

def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--phase", required=True, choices=("prefill", "decode", "train"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--hold-exception", help="validated entry permission recorded by the driver")
    ap.add_argument("--run-metadata", default="{}")
    a = ap.parse_args(argv)
    leaked = sorted(k for k in os.environ if k.startswith(FORBIDDEN_ENV))
    if leaked:
        raise SystemExit(f"refusing: model switches in the environment: {leaked}")
    sys.path[:] = [str(ROOT), str(ROOT / "opmodel" / "src")] + [
        p for p in sys.path if p and (p.endswith("site-packages") or p.startswith(sys.base_prefix))]
    out = Path(a.out).resolve()
    run_dir = out.parent / (out.stem + ".rundir")
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    import config
    import train_timing, inference_timing, base_timing, llm_execution, opmodel
    for mod in (config, train_timing, inference_timing, base_timing, llm_execution, opmodel):
        f = str(Path(mod.__file__).resolve())
        if not f.startswith(str(ROOT) + "/"):
            raise SystemExit(f"refusing: {mod.__name__} imported from {f}, not from {ROOT}")
    from train_timing import TimeCalculationLLM
    from inference_timing import TimeCalculationLLMInference

    hwc = config.parse_config(a.hardware, config_type="hardware")
    mc = config.parse_config(a.model, config_type="LLM")
    config.validate_configs(hwc, mc)
    exp_dir = str(run_dir / "output" / "LLM")
    os.makedirs(exp_dir, exist_ok=True)
    phases = {}
    decode_only = False
    if a.phase == "train":
        tc = TimeCalculationLLM(hwc, mc, "LLM", output_dir=exp_dir)
        t = tc.calc_time_llm()
        phases["train"] = (tc.execution_energy, float(t))
        extra = {"total_time_s": float(t)}
    else:
        tc = TimeCalculationLLMInference(hwc, mc, "LLM", output_dir=exp_dir)
        decode_only = tc.decode_only()
        if decode_only and a.phase != "decode":
            raise SystemExit("refusing: inference_param.decode_only needs --phase decode (prefill is not simulated)")
        timing = tc.calc_total_inference_time()
        if not decode_only:
            phases["prefill"] = (tc.prefill_execution_energy, float(timing["prefill_time"]))
        if a.phase == "decode":
            phases["decode"] = (tc.decode_execution_energy, float(timing["decode_time"]))
        extra = {k: float(v) for k, v in timing.items() if isinstance(v, (int, float))}
    native = dict(getattr(tc, "native_gemm_timing", None) or {})
    rec = {
        "native_gemm_timing": native,
        "settings_profile": getattr(tc, "settings_profile", None),
        "commit": a.commit, "phase": a.phase, "measured_phase": a.phase,
        "hardware_config": a.hardware, "hardware_sha256": sha256(a.hardware),
        "model_config": a.model, "model_sha256": sha256(a.model),
        "settings": getattr(tc, "resolved_settings", None),
        "pipeline_settings": {"hold_exception": json.loads(a.hold_exception) if a.hold_exception else None,
                              "run": json.loads(a.run_metadata)},
        "modules": {m.__name__: str(Path(m.__file__).resolve()) for m in (config, train_timing, inference_timing,
                                                                          base_timing, llm_execution, opmodel)},
        "extra": extra, "phases": {},
    }
    if decode_only:
        rec["decode_only"] = True
        rec["phases_not_simulated"] = {"prefill": "inference_param.decode_only: prefill time and energy not simulated; "
                                                  "its KV-cache stores not charged; decode steps attend to the prompt "
                                                  "as after a prefill"}
    for name, (res, t) in phases.items():
        g = res.gpu
        rec["phases"][name] = {"time_s": t, "total_j": res.total_j, "static_j": g.static_j,
                               "dynamic": g.dynamic.to_dict(),
                               "communication_j": res.communication.total_j, "activity": g.activity.to_dict()}
        if str(tc.model_type).lower() == "qwen3_moe":
            rec["phases"][name]["coverage"] = res.coverage.to_dict()
    bad = {k: v for k, v in native.items() if k != "flash_tile"}
    rec["assert_16bit"] = "ok" if not bad else f"FAILED: {bad}"
    from paper_pipeline.static_power import mark_prediction, mark_run_energy_report
    mark_prediction(rec)
    mark_run_energy_report(run_dir, rec["pipeline_settings"]["run"].get("static_power"))
    from paper_pipeline.reporting import sanitize_run_artifacts, without_register_traffic
    sanitize_run_artifacts(run_dir)
    rec = without_register_traffic(rec)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1, default=str) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    if bad:
        raise SystemExit(f"assert 16-bit failed: native GEMM timing {bad}")

if __name__ == "__main__":
    main()
