from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import runpy
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "opmodel/src"))
sys.path.insert(0, str(ROOT))

from ncu_counts import add_counts, counts_only_projection
from train_timing import TimeCalculationLLM
from config import parse_bandwidth_string

DRY_FLAG = "--plumbing-dry-run-tensor-latency-cycles"
DRY_STAMP = "PLUMBING DRY RUN - NOT FOR RESULTS"

def _counts_only_mode(args):
    if "--hardware-yaml" not in args:
        return False
    raw = yaml.safe_load(Path(args[args.index("--hardware-yaml") + 1]).read_text())
    core = raw.get("tech_param", {}).get("core", {})
    flag = (raw.get("metadata") or {}).get("counts_only", False)
    if not isinstance(flag, bool):
        raise ValueError("metadata.counts_only must be a boolean")
    if core.get("architecture") == "sm90" and flag is not True:
        raise ValueError("sm90 counts dispatch requires metadata.counts_only: true")
    if flag and core.get("architecture") != "sm90":
        raise ValueError("metadata.counts_only requires architecture sm90")
    return flag

def _check_tile_selection_inputs(args, tensor_override=None):
    raw = yaml.safe_load(Path(args[args.index("--hardware-yaml") + 1]).read_text())
    required = ["tech_param.core.operating_frequency", "tech_param.core.tensor_latency_cycles",
                "tech_param.core.matrix_unit.nominal_flop_rate_per_mcu",
                "tech_param.core.vector_unit.nominal_flop_rate_per_bundle"]
    for level in ("DRAM", "SRAM-L2", "SRAM-L1", "SRAM-R"):
        required.append(f"tech_param.{level}.bandwidth")
        block = raw.get("tech_param", {}).get(level, {})
        latency_key = "latency_cycles" if block.get("latency_cycles") is not None else "latency"
        required.append(f"tech_param.{level}.{latency_key}")
    missing = []
    for path in required:
        value = raw
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if path == "tech_param.core.tensor_latency_cycles" and tensor_override is not None:
            value = tensor_override
        if path.endswith(".bandwidth") and not isinstance(value, bool):
            try:
                value = parse_bandwidth_string(value)
            except (ValueError, TypeError, AssertionError):
                value = None
        allow_zero = path in ("tech_param.SRAM-R.latency", "tech_param.SRAM-R.latency_cycles")
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero)):
            missing.append(path)
    if missing:
        raise ValueError("H100 tile-selection inputs lack cited values: " + ", ".join(missing))

def main():
    args = sys.argv[1:]
    tensor_override = None
    if DRY_FLAG in args:
        if args.count(DRY_FLAG) != 1:
            raise ValueError(f"{DRY_FLAG} must be supplied at most once")
        index = args.index(DRY_FLAG)
        try:
            tensor_override = float(args[index + 1])
        except (IndexError, ValueError):
            raise ValueError(f"{DRY_FLAG} requires a positive finite number") from None
        if not math.isfinite(tensor_override) or tensor_override <= 0:
            raise ValueError(f"{DRY_FLAG} requires a positive finite number")
        args = args[:index] + args[index + 2:]
    harness = ROOT / "paper_pipeline/harness/counts_run_one.py"
    if args[:1] == ["--harness"]:
        harness, args = Path(args[1]).resolve(), args[2:]
    if len(args) < 6:
        raise SystemExit("usage: counts_run_one.py [--harness PATH] MODEL PHASE BATCH SEQ OUT_JSON RUN_DIR [options]")
    output = Path(args[4]).resolve()
    counts_only = _counts_only_mode(args)
    if tensor_override is not None and not counts_only:
        raise ValueError(f"{DRY_FLAG} requires H100 counts-only mode")
    if counts_only:
        _check_tile_selection_inputs(args, tensor_override)
    if tensor_override is not None:
        print(f"{DRY_STAMP}: tensor_latency_cycles={tensor_override}", file=sys.stderr)
    observations = {}
    original = TimeCalculationLLM._apply_l2_simulation

    def observe(self, *a, **kw):
        result = original(self, *a, **kw)
        observations.update(getattr(self, "counts_memcpy_by_phase", {}))
        return result

    original_moe = TimeCalculationLLM.price_moe_source_step

    def observe_moe(self, *a, **kw):
        result = original_moe(self, *a, **kw)
        observations.update(getattr(self, "counts_memcpy_by_phase", {}))
        return result

    def execute(harness_args):
        with patch.object(TimeCalculationLLM, "_apply_l2_simulation", observe), \
                patch.object(TimeCalculationLLM, "price_moe_source_step", observe_moe), \
                patch.object(sys, "argv", [str(harness), *harness_args]):
            runpy.run_path(str(harness), run_name="__main__")

    if counts_only:
        output.parent.mkdir(parents=True, exist_ok=True)

        previous_cwd = Path.cwd()
        with TemporaryDirectory(prefix="counts_internal_", dir=output.parent) as internal:
            internal_output = Path(internal) / "prediction.json"
            internal_args = list(args)
            internal_args[4:6] = [str(internal_output), str(Path(internal) / "work")]
            if tensor_override is not None:
                hw_index = internal_args.index("--hardware-yaml") + 1
                raw = yaml.safe_load(Path(internal_args[hw_index]).read_text())
                raw["tech_param"]["core"]["tensor_latency_cycles"] = tensor_override
                temporary_hw = Path(internal) / "hardware.yaml"
                temporary_hw.write_text(yaml.safe_dump(raw))
                internal_args[hw_index] = str(temporary_hw)
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    execute(internal_args)
                document = counts_only_projection(add_counts(json.loads(internal_output.read_text()), observations))
                if tensor_override is not None:
                    document["plumbing_dry_run"] = {"warning": DRY_STAMP,
                                                    "tensor_latency_cycles": tensor_override}
            finally:
                os.chdir(previous_cwd)
    else:
        execute(args)
        document = add_counts(json.loads(output.read_text()), observations)
    if counts_only:
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", prefix=".counts_", suffix=".tmp", dir=output.parent,
                                    delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(document) + "\n")
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    else:
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(document) + "\n")
        temporary.replace(output)

if __name__ == "__main__":
    main()
