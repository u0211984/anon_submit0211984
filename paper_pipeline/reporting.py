from __future__ import annotations

import json
import os
from pathlib import Path

def without_register_traffic(value):
    if isinstance(value, dict):
        result = {key: without_register_traffic(item) for key, item in value.items()}
        register = value.get("register")
        if (isinstance(register, dict) and {"read_bytes", "write_bytes"} <= register.keys()
                and {"smem", "l2", "hbm"} <= value.keys()):
            result.pop("register", None)
        return result
    if isinstance(value, list):
        return [without_register_traffic(item) for item in value]
    if isinstance(value, tuple):
        return tuple(without_register_traffic(item) for item in value)
    return value

def sanitize_run_artifacts(run_dir):
    root = Path(run_dir) / "output" / "LLM"
    for name in ("energy_report.json", "decode_transformer_results.txt", "transformer_timings.yaml"):
        path = root / name
        if not path.is_file():
            continue
        if name.endswith(".yaml"):
            import yaml
            data = yaml.safe_load(path.read_text())
            content = yaml.safe_dump(without_register_traffic(data), sort_keys=True)
        else:
            data = json.loads(path.read_text())
            content = json.dumps(without_register_traffic(data), indent=2, sort_keys=True) + "\n"
        temporary = path.with_name(path.name + ".paper.tmp")
        temporary.write_text(content)
        os.replace(temporary, path)
