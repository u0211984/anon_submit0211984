from copy import deepcopy

from activity import MemoryTraffic, OperationActivity

def memcpy_activity(kernels, result):
    read = write = l2_read = l2_write = 0.0
    for kernel, rd, wr in zip(kernels, result.hbm_read, result.hbm_write):
        if kernel.kind == "memcpy":
            read += rd
            write += wr
            l2_read += kernel.read_bytes + float(getattr(kernel, "l2_reread", 0.0))
            l2_write += kernel.write_bytes
    return OperationActivity(hbm=MemoryTraffic(read, write), l2=MemoryTraffic(l2_read, l2_write))

def phase_counts(activity, memcpy, *, memcpy_l2_in_activity):
    original = activity.to_dict() if isinstance(activity, OperationActivity) else activity
    counts = deepcopy(original)
    physical = {level: getattr(memcpy, level).to_dict() for level in ("hbm", "l2")}
    excluded = deepcopy(physical)
    if not memcpy_l2_in_activity:
        excluded["l2"] = MemoryTraffic().to_dict()
    for level, traffic in excluded.items():
        for direction, value in traffic.items():
            remaining = counts[level][direction] - value
            if remaining < -1e-6:
                raise ValueError(f"memcpy {level}.{direction} exceeds booked activity")
            counts[level][direction] = max(0.0, remaining)
    return {"counts_scope": "ncu_kernels_excluding_memcpy", "counts_activity": counts,
            "memcpy_bytes": physical, "excluded_memcpy_bytes": excluded}

def add_counts(document, observations):
    for phase, record in document["phases"].items():
        if phase not in observations:
            raise ValueError(f"no step-simulation memcpy observation for {phase}")
        memcpy, included = observations[phase]
        record.update(phase_counts(record["activity"], memcpy, memcpy_l2_in_activity=included))
    return document

def counts_only_projection(document):
    def scalar(value):
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValueError("counts projection requires scalar field values")
        return value

    def activity(value):
        if value is None:
            return None
        out = {key: scalar(value[key]) for key in (
            "useful_tensor_flops", "issued_tensor_flops", "vector_flops", "smem_staging_bytes"
        ) if key in value}
        for level in ("register", "smem", "l2", "hbm"):
            if level in value:
                out[level] = traffic(value[level])
        return out

    def traffic(value):
        return {key: scalar(value[key]) for key in ("read_bytes", "write_bytes") if key in value}

    def memory_levels(value):
        return {level: traffic(value[level]) for level in ("hbm", "l2") if level in value}

    result = {key: scalar(document[key]) for key in (
        "model", "phase", "batch", "seq_len", "nockpt", "hardware_config", "model_config", "tree"
    ) if key in document}
    result["metric_scope"] = "counts_only"
    if "native_gemm_timing" in document:
        reasons = document["native_gemm_timing"]
        if not isinstance(reasons, dict) or set(reasons) - {"flash_tile"}:
            raise ValueError("unsupported native GEMM timing reasons")
        if any(type(value) is not int or value < 0 for value in reasons.values()):
            raise ValueError("native GEMM timing reason counts must be nonnegative integers")
        result["native_gemm_timing"] = dict(reasons)
    if "modules" in document:
        result["modules"] = {k: v for k, v in document["modules"].items()
                             if isinstance(k, str) and isinstance(v, str)}
    result["events"] = []
    for event in document.get("events", []):
        projected = {key: scalar(event[key]) for key in ("kind", "cls", "name", "inst") if key in event}
        if "activity" in event:
            projected["activity"] = activity(event["activity"])
        if "timings" in event:
            projected["timings"] = {
                name: {direction: activity(parts[direction]) for direction in (
                    "forward", "backward", "forward_eager_trace", "backward_eager_trace"
                ) if direction in parts}
                for name, parts in event["timings"].items()
            }
        result["events"].append(projected)
    result["phases"] = {}
    for name, phase in document["phases"].items():
        projected = {key: activity(phase[key]) for key in ("activity", "counts_activity") if key in phase}
        if "counts_scope" in phase:
            projected["counts_scope"] = scalar(phase["counts_scope"])
        for key in ("memcpy_bytes", "excluded_memcpy_bytes"):
            if key in phase:
                projected[key] = memory_levels(phase[key])
        if "by_operation" in phase:
            projected["by_operation"] = {
                op: {"activity": activity(record["activity"])}
                for op, record in phase["by_operation"].items() if "activity" in record
            }
        result["phases"][name] = projected
    return result
