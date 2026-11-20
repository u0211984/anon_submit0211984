import json
import math
from pathlib import Path

def normalize_dtype(value):
    value = str(value).strip().lower()
    return {"bf16": "bfloat16", "fp32": "float32", "float": "float32",
            "fp16": "float16", "half": "float16"}.get(value, value)

def load_curve(path, *, include_unverified=False, timing_basis="kernel_union"):
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if timing_basis == "kernel_union":
        if data.get("status") != "ready" or data.get("timing_basis") != "kernel_union":
            raise ValueError("measured all-reduce curve requires status=ready and timing_basis=kernel_union")
        points, seconds_key = data.get("points", []), "kernel_seconds"
    elif timing_basis == "cuda_event_large_message_proxy":
        if data.get("timing_basis") != "cuda_event_interval":
            raise ValueError("CUDA event proxy requires timing_basis=cuda_event_interval source")
        points, seconds_key = data.get("event_interval_points", []), "event_interval_seconds"
    else:
        raise ValueError(f"unknown all-reduce timing basis {timing_basis!r}")
    series = {}
    for point in points:
        if timing_basis == "cuda_event_large_message_proxy" and (
                point.get("path") != "nccl" or point.get("status") != "ok"
                or point.get("bytes", 0) < 2**20):
            continue
        if not include_unverified and point.get("clock_verified") is not True:
            continue
        path_name = point["path"]
        tp = point["tp"]
        size = point["bytes"]
        seconds = float(point[seconds_key])
        if (path_name not in ("nccl", "custom") or isinstance(tp, bool) or int(tp) != tp or tp < 2
                or isinstance(size, bool) or int(size) != size or size <= 0
                or not math.isfinite(seconds) or seconds <= 0):
            raise ValueError("invalid measured all-reduce curve point")
        key = series_key(path_name, tp, point["dtype"])
        series.setdefault(key, []).append((int(size), seconds))
    if not series:
        raise ValueError("measured all-reduce curve has no eligible points")
    for points in series.values():
        points.sort()
        if len({size for size, _ in points}) != len(points):
            raise ValueError("duplicate measured all-reduce curve size")
    return series

def series_key(path, tp, dtype):
    return f"{path}:{int(tp)}:{normalize_dtype(dtype)}"

def curve_seconds(series, path, tp, dtype, size):
    key = series_key(path, tp, dtype)
    if key not in series:
        raise ValueError(f"measured all-reduce curve missing series {key!r}")
    points = series[key]
    if size <= points[0][0]:
        return points[0][1]
    if size >= points[-1][0]:
        top_size, top_seconds = points[-1]
        return top_seconds * (size / top_size)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if size <= x1:
            fraction = math.log(size / x0) / math.log(x1 / x0)
            return math.exp(math.log(y0) + fraction * math.log(y1 / y0))
    raise AssertionError("unreachable")

RSAG_RULES = ("network_model", "allreduce_bus_bandwidth")
_RSAG = ("reduce_scatter", "all_gather")

def _kind(coll):
    return str(getattr(coll, "value", coll))

def nccl_tests_size(coll, participants, size_bytes):
    return size_bytes * participants if _kind(coll) == "all_gather" else size_bytes

def call_dtype(policy, interconnect="tp", coll="all_reduce"):
    kind = _kind(coll)
    if kind in _RSAG:
        if interconnect != "dp" or policy.get("rsag_rule") != "allreduce_bus_bandwidth":
            return None
        return policy.get("dp_reduce_scatter_dtype" if kind == "reduce_scatter" else "dp_all_gather_dtype")
    if kind != "all_reduce":
        return None
    if interconnect == "tp":
        return policy["allreduce_dtype"]
    if interconnect == "dp":
        return policy.get("dp_allreduce_dtype")
    return None

def measured_seconds(policy, choice, participants, size, interconnect="tp", coll="all_reduce"):
    if not policy or policy.get("allreduce_model") != "measured_curve":
        return None
    kind = _kind(coll)
    path = "custom" if choice.algorithm.startswith("custom_") else "nccl"
    series = policy["allreduce_curve_series"]
    dtype = call_dtype(policy, interconnect, kind)
    if dtype is None or (kind in _RSAG and path != "nccl"):
        return None
    size_s = nccl_tests_size(kind, participants, size)
    if policy.get("allreduce_timing_basis", "kernel_union") == "cuda_event_large_message_proxy":
        if path != "nccl" or size_s < 2**20 or series_key(path, participants, dtype) not in series:
            return None
    if series_key(path, participants, dtype) not in series and (path == "custom" or kind in _RSAG):
        return None
    allreduce = curve_seconds(series, path, participants, dtype, size_s)
    if kind == "all_reduce":
        return allreduce

    n = int(participants)
    bus_bandwidth = 2 * (n - 1) / n * size_s / allreduce
    return (n - 1) / n * size_s / bus_bandwidth

def fallback_reason(policy, choice, participants, size, interconnect="tp", coll="all_reduce"):
    path = "custom" if choice.algorithm.startswith("custom_") else "nccl"
    proxy = policy.get("allreduce_timing_basis", "kernel_union") == "cuda_event_large_message_proxy"
    if proxy and path != "nccl":
        return "custom_path_not_measured"
    key = series_key(path, participants, call_dtype(policy, interconnect, coll))
    if key not in policy["allreduce_curve_series"]:
        return "missing_series:" + key
    if proxy and nccl_tests_size(coll, participants, size) < 2**20:
        return "below_proxy_minimum"
    return None

def record_usage(timing, policy, records):
    if not policy or policy.get("allreduce_model") != "measured_curve":
        return
    from types import SimpleNamespace
    if not hasattr(timing, "resolved_settings"):
        timing.resolved_settings = {}
    usage = timing.resolved_settings.setdefault("allreduce_usage", {
        "scope": "evaluated logical collectives including DP multiplicity; decode samples only",
        "curve_collectives": 0, "fallback_collectives": 0, "fallback_reasons": {},
    })
    for record in records:
        choice = SimpleNamespace(algorithm=record["algorithm"])
        interconnect = record.get("interconnect", "tp")
        coll = record.get("coll", "all_reduce")
        seconds = measured_seconds(policy, choice, record["participants"], record["size_bytes"], interconnect, coll)
        count = record["count"] * record["multiplicity"]
        if seconds is not None:
            usage["curve_collectives"] += count
        else:
            reason = fallback_reason(policy, choice, record["participants"], record["size_bytes"], interconnect, coll)
            usage["fallback_collectives"] += count
            reasons = usage["fallback_reasons"]
            reasons[reason] = reasons.get(reason, 0) + count
    _usage_status(usage)

def merge_usage(timing, sample_usage):
    if sample_usage is None:
        return
    usage = timing.resolved_settings["allreduce_usage"]
    for key in ("curve_collectives", "fallback_collectives"):
        usage[key] += sample_usage[key]
    for reason, count in sample_usage["fallback_reasons"].items():
        usage["fallback_reasons"][reason] = usage["fallback_reasons"].get(reason, 0) + count
    _usage_status(usage)

def _usage_status(usage):
    applied, fallback = usage["curve_collectives"], usage["fallback_collectives"]
    usage["status"] = "partially_applied" if applied and fallback else "applied" if applied else "not_applied"
    if applied:
        usage.pop("not_applied_reason", None)
    else:
        usage["not_applied_reason"] = "native_fallbacks" if fallback else "no_eligible_collectives"

def is_measured_record(policy, record):
    from types import SimpleNamespace
    choice = SimpleNamespace(algorithm=record["algorithm"])
    return measured_seconds(policy, choice, record["participants"], record["size_bytes"],
                            record.get("interconnect", "tp"), record.get("coll", "all_reduce")) is not None

def program_policy(program):
    misc = program.meta.misc
    return misc.get("vllm_allreduce") or misc.get("allreduce_curve") or {}

def is_measured_call(program, call):
    from timing_model import CollectiveType
    policy = program_policy(program)
    interconnect = getattr(call, "interconnect", None)
    coll = getattr(call, "coll", None)
    if (policy.get("allreduce_model") != "measured_curve"
            or coll not in (CollectiveType.ALL_REDUCE, CollectiveType.REDUCE_SCATTER, CollectiveType.ALL_GATHER)
            or interconnect not in ("tp", "dp") or call_dtype(policy, interconnect, coll) is None
            or call.participants <= 1 or call.size_bytes <= 0):
        return False
    from program.vllm_allreduce import op_choice
    return measured_seconds(policy, op_choice(program, call), call.participants, call.size_bytes,
                            interconnect, coll) is not None

TRAINING_USAGE_SCOPE = "training: TP all-reduce calls on every issuing rank, including DP multiplicity"
TRAINING_DP_USAGE_SCOPE = ("training: TP and data-parallel gradient all-reduce calls on every issuing rank, "
                           "including DP multiplicity")
TRAINING_RSAG_USAGE_SCOPE = ("training: TP and data-parallel gradient all-reduce calls, and data-parallel "
                             "reduce-scatter / all-gather calls (allreduce_rsag_rule allreduce_bus_bandwidth), on "
                             "every issuing rank, including DP multiplicity")

def training_records(program):
    from program.ir import CollectiveOp, ComputeOp
    from timing_model import CollectiveType
    policy = program_policy(program)
    axes = ("tp", "dp") if policy.get("dp_allreduce_dtype") else ("tp",)
    rsag = policy.get("rsag_rule") == "allreduce_bus_bandwidth"
    records = []
    for op in program.ops:
        calls = (op,) if isinstance(op, CollectiveOp) else (
            op.embedded_collectives if isinstance(op, ComputeOp) else ())
        for call in calls:
            if call.participants <= 1:
                continue
            if call.coll is CollectiveType.ALL_REDUCE:
                if call.interconnect not in axes:
                    continue
            elif not (rsag and call.interconnect == "dp"
                      and call.coll in (CollectiveType.REDUCE_SCATTER, CollectiveType.ALL_GATHER)):
                continue
            if call.interconnect == "dp" and call.size_bytes <= 0:
                continue
            record = dict(algorithm="nccl", count=1, multiplicity=max(1, program.dp_count),
                          participants=call.participants, size_bytes=call.size_bytes, label=call.name)
            if call.interconnect != "tp":
                record["interconnect"] = call.interconnect
            if call.coll is not CollectiveType.ALL_REDUCE:
                record["coll"] = call.coll.value
            records.append(record)
    return tuple(records)

def training_usage_scope(program):
    if any("coll" in record for record in training_records(program)):
        return TRAINING_RSAG_USAGE_SCOPE
    return TRAINING_DP_USAGE_SCOPE if program_policy(program).get("dp_allreduce_dtype") else TRAINING_USAGE_SCOPE
