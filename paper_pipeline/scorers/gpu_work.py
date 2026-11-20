from __future__ import annotations

import json
import math
from pathlib import Path

GPU_WORK = "GPU-work basis"
NO_GPU_WORK = "no GPU-work basis"

def load_idle_power(spec: dict | None, root: Path) -> dict | None:
    if not spec:
        return None
    data = json.loads((root / spec["path"]).read_text())
    return dict(data, **spec)

def _finite(value, name, *, positive=False):
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"invalid {name}: {value!r}")
    return result

def job_idle_power(run: dict, idle: dict) -> float:
    if idle["mode"] == "per_cell":
        if int(run.get("n_gpus") or run.get("tp") or 1) != 1:
            raise ValueError("per-cell idle power is a single-GPU measurement")
        cell = str(run.get("cell", ""))
        try:
            measurement = idle["cells"][cell]
        except KeyError as exc:
            raise ValueError(f"no measured idle power for cell {cell}") from exc
        if not measurement.get("source"):
            raise ValueError(f"cell {cell} idle power has no source")
        return _finite(measurement["power_W"], f"cell {cell} idle power", positive=True)
    if idle["mode"] == "fixed":
        if int(run.get("n_gpus") or run.get("tp") or 1) != 1:
            raise ValueError("fixed idle power is a single-GPU measurement")
        return _finite(idle["power_W"], "idle power", positive=True)
    if idle["mode"] != "per_gpu":
        raise ValueError(f"unknown idle-power mode {idle['mode']!r}")
    selected = run.get("gpus")
    if isinstance(selected, str):
        selected = json.loads(selected) if selected.strip().startswith("[") else selected.split(",")
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError("per-GPU idle power requires the job's gpus indices")
    selected = [int(g) for g in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate job GPU indices")
    expected = run.get("n_gpus") or run.get("tp")
    if expected and len(selected) != int(expected):
        raise ValueError("job GPU indices disagree with GPU count")
    indices, powers = idle["gpus"], idle["locked"]["power_W"]
    if len(indices) != len(powers) or len(set(indices)) != len(indices):
        raise ValueError("invalid per-GPU idle measurement")
    by_gpu = dict(zip(map(int, indices), powers))
    try:
        return sum(_finite(by_gpu[g], f"GPU {g} idle power", positive=True) for g in selected)
    except KeyError as exc:
        raise ValueError(f"no measured idle power for GPU {exc.args[0]}") from exc

def measured_basis(run: dict, per_step: dict, idle_power: dict | None) -> dict:
    steps = _finite(run[per_step["steps"]], "steps", positive=True)
    wall = _finite(run[per_step["runtime"]], "wall time", positive=True) / steps
    raw = _finite(run[per_step["energy"]], "raw energy", positive=True) / steps
    result = dict(gpu_work_basis=NO_GPU_WORK, measured_wall_s_secondary=wall,
                  measured_raw_energy_J_secondary=raw, idle_power_W="", gpu_gap_s="",
                  gpu_work_time_s="", gpu_work_energy_J="")
    kernel = run.get("gpu_kernel_ms_per_step")
    if kernel in (None, ""):
        return result
    kernel = _finite(kernel, "kernel-union time", positive=True) / 1000.0
    if idle_power is None:
        raise ValueError("kernel-union measurement requires a pinned idle-power input")
    idle = job_idle_power(run, idle_power)

    gap = wall - kernel
    energy = raw - gap * idle
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("GPU-work energy must be finite and positive")
    result.update(gpu_work_basis=GPU_WORK, idle_power_W=idle, gpu_gap_s=gap,
                  gpu_work_time_s=kernel, gpu_work_energy_J=energy)
    return result

NODE_RAW = "node_raw"
NODE_RAW_BASIS = "raw wall and raw energy, all node GPUs (MEGATRON_MULTI_SCORE)"

def _gpu_list(value, name):
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith(("[", "{")) else value.split(",")
    if isinstance(value, dict):
        value = list(value)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name}: no GPU indices")
    out = [int(g) for g in value]
    if len(set(out)) != len(out):
        raise ValueError(f"{name}: duplicate GPU indices")
    return out

def node_raw_basis(run: dict, per_step: dict, idle_power: dict | None) -> dict:
    result = dict(gpu_work_basis="", measured_wall_s_secondary="", measured_raw_energy_J_secondary="",
                  idle_power_W="", gpu_gap_s="", gpu_work_time_s="", gpu_work_energy_J="")
    if run.get("measured_status"):
        return result
    if idle_power is None or idle_power.get("mode") != "per_gpu":
        raise ValueError("node_raw scoring requires the node's per-GPU idle-power measurement")
    indices, powers = idle_power["gpus"], idle_power["locked"]["power_W"]
    if len(indices) != len(powers):
        raise ValueError("invalid per-GPU idle measurement")
    node = dict(zip(_gpu_list(indices, "idle gpus"), powers))
    job = _gpu_list(run["gpus"], "job gpus")
    if len(job) != int(run["n_gpus"]) or not set(job) <= set(node):
        raise ValueError(f"job GPUs {job} are not {run['n_gpus']} GPUs of the node {sorted(node)}")
    steps = _finite(run[per_step["steps"]], "steps", positive=True)
    wall = _finite(run[per_step["runtime"]], "wall time", positive=True) / steps
    raw = _finite(run[per_step["energy"]], "raw energy", positive=True) / steps
    per_gpu = json.loads(run[per_step["energy_per_gpu"]])
    if sorted(_gpu_list(per_gpu, "energy GPUs")) != sorted(node):
        raise ValueError(f"raw energy covers GPUs {sorted(per_gpu)}, not every node GPU {sorted(node)}")
    total = sum(_finite(v, "per-GPU energy", positive=True) for v in per_gpu.values())
    if abs(total / steps - raw) > 1e-9 * raw:
        raise ValueError(f"raw energy {raw * steps!r} J is not the sum over the node's GPUs {total!r} J")
    if per_step.get("wall_ms") and abs(_finite(run[per_step["wall_ms"]], "wall_ms") / 1000.0 - wall) > 1e-9 * wall:
        raise ValueError("runtime / steps disagrees with the row's wall time per step")
    unused = sum((_finite(node[g], f"GPU {g} idle power", positive=True) for g in node if g not in job), 0.0)
    result.update(gpu_work_basis=NODE_RAW_BASIS, measured_wall_s_secondary=wall, measured_raw_energy_J_secondary=raw,
                  idle_power_W=unused, gpu_work_time_s=wall, gpu_work_energy_J=raw)
    return result

def node_raw_prediction(time_s: float, energy_j: float, basis: dict):
    return time_s, energy_j + float(basis["idle_power_W"]) * time_s
