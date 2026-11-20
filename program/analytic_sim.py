from __future__ import annotations

from program.allreduce_curve import is_measured_call

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from typing import Any, Collection, Dict, List, Mapping, Optional, Set, Tuple

from program.ir import CollectiveOp, CollectiveType, ComputeOp, Program, TransferOp
from program.collective_launch import LaunchAccounting, is_nccl_call, is_nccl_collective, launch_issue_consumers
from program.nccl_latency import enabled as launch_enabled, launch_seconds as call_launch_seconds
from program.vllm_allreduce import op_choice, custom_seconds

__all__ = [
    "COMM_DURATIONS_KEY",
    "CoarseEvalResult",
    "comm_durations",
    "evaluate",
    "evaluate_detailed",
]

_ROOT_COMM_IS_UNTIMED = True

COMM_DURATIONS_KEY = "comm_durations"

def comm_durations(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
    *,
    collective_keys: Optional[Collection[str]] = None,
    operation_uids: Optional[Collection[int]] = None,
) -> Tuple[float, ...]:
    if not isinstance(program, Program):
        raise TypeError(
            f"comm_durations expects a Program (got {type(program).__name__})"
        )
    out: List[float] = [0.0] * len(program.ops)
    for op in program.ops:
        if operation_uids is not None and op.uid not in operation_uids:
            continue
        if isinstance(op, ComputeOp):
            continue
        if collective_keys is not None and (
            not isinstance(op, CollectiveOp) or op.comm_key not in collective_keys
        ):
            continue

        if _ROOT_COMM_IS_UNTIMED and not op.deps and not (isinstance(op, CollectiveOp) and str(op.comm_key).startswith("zm_")):
            continue
        size = float(op.size_bytes)
        if size <= 0:
            continue
        if isinstance(op, CollectiveOp):
            choice = op_choice(program, op)
            from program.allreduce_curve import measured_seconds, program_policy
            policy = program_policy(program)
            measured = (measured_seconds(policy, choice, op.participants, size, op.interconnect, op.coll)
                        if op.coll in (CollectiveType.ALL_REDUCE, CollectiveType.REDUCE_SCATTER,
                                       CollectiveType.ALL_GATHER) and op.interconnect in ("tp", "dp")
                        and op.participants > 1 else None)
            if measured is not None:
                out[op.uid] = measured
                continue
            if choice.algorithm.startswith("custom_"):
                out[op.uid] = custom_seconds(choice, size, op.participants,
                                             program.meta.misc["vllm_allreduce"])
                continue
        kind = op.coll if isinstance(op, CollectiveOp) else op.comm_type
        if kind is None:
            raise ValueError(f"Comm op {op.name!r} (uid {op.uid}) has no CollectiveType")

        if isinstance(op, CollectiveOp) and str(op.comm_key).startswith("zm_"):
            axes = [a for a in program.layout.axis_order if a in op.axes
                    and int(program.layout.axis_sizes.get(a, 1)) > 1]
            if kind is CollectiveType.ALL_GATHER:
                axes.reverse()
            phase_size = size
            total = 0.0
            for phase_axis in axes:
                participants = int(program.layout.axis_sizes[phase_axis])
                ib, ll = interconnect_params[phase_axis]
                total += float(network_model.collective(
                    kind=kind, size_bytes=phase_size, participants=participants,
                    ib=ib, ll=ll, local_bytes=0.0, local_ops=0.0,
                    axis=phase_axis, debug_label=f"{op.name}_{phase_axis}_conversion",
                ))
                if kind is CollectiveType.REDUCE_SCATTER:
                    phase_size /= participants
                elif kind is CollectiveType.ALL_GATHER:
                    phase_size *= participants
            out[op.uid] = total
            continue
        axis = op.interconnect
        if not axis or axis not in interconnect_params:
            raise ValueError(f"Invalid interconnect type: {axis}")
        ib, ll = interconnect_params[axis]
        if (program.meta.misc.get("nccl_latency_rule") in ("tuning_formula", "tuning_protocol")
                and isinstance(op, CollectiveOp)
                and is_nccl_call(kind, participants=op.participants, size_bytes=size,
                                 algorithm=op_choice(program, op).algorithm)):
            ll = 0.0
        out[op.uid] = float(
            network_model.collective(
                kind=kind,
                size_bytes=size,
                participants=op.participants,
                ib=ib,
                ll=ll,
                local_bytes=0.0,
                local_ops=0.0,
                debug_label=f"{op.name}_conversion",
                **({"axis": axis} if program.meta.misc.get("nccl_latency_rule") in ("tuning_formula", "tuning_protocol") else {}),
            )
        )
    return tuple(out)

@dataclass
class CoarseEvalResult:
    total_time: float
    finish_times: List[float]

def _device_count(program: Program) -> int:
    count = max(1, len(program.devices))
    for op in program.ops:
        if isinstance(op, ComputeOp):
            count = max(count, int(op.device) + 1)
    return count

def evaluate_detailed(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
) -> CoarseEvalResult:
    if not isinstance(program, Program):
        raise TypeError(f"evaluate expects a Program (got {type(program).__name__})")
    if program.meta.misc.get("granularity") != "pipeline":
        raise RuntimeError(
            "Analytical evaluation requires a PIPELINE program "
            "(build(..., granularity=Granularity.PIPELINE))."
        )

    ops = program.ops
    durations = list(comm_durations(program, network_model, interconnect_params))
    for op in ops:
        if isinstance(op, ComputeOp):
            durations[op.uid] = float(op.duration[0])

    program.meta.misc[COMM_DURATIONS_KEY] = tuple(
        0.0 if isinstance(op, ComputeOp) else durations[op.uid] for op in ops
    )

    launch_seconds = float(program.meta.misc.get("collective_launch_overhead", 0.0))
    original_count = len(ops)
    if launch_enabled(program):
        ops = list(ops)
        accounting = LaunchAccounting(launch_seconds)
        issue_consumers = launch_issue_consumers(program)
        for op in program.ops:
            if isinstance(op, ComputeOp):
                for coll in op.embedded_collectives:
                    if (not is_measured_call(program, coll) and is_nccl_call(coll.coll, participants=coll.participants,
                                    size_bytes=coll.size_bytes, algorithm=op_choice(program, coll).algorithm)):
                        seconds = call_launch_seconds(program, coll)
                        durations[op.uid] += seconds
                        for dp_idx in range(max(int(program.dp_count), 1)):
                            accounting.add(program.rank_for(op.device, dp_idx), label=coll.name, seconds=seconds)
                continue
            if is_measured_call(program, op) or not is_nccl_collective(op, algorithm=op_choice(program, op).algorithm):
                continue
            seconds = call_launch_seconds(program, op)
            uid = len(ops)
            launch = ComputeOp(uid=uid, name=op.name + "_nccl_launch",
                               device=op.device, duration=(seconds,), deps=op.deps)
            ops.append(launch)
            durations.append(seconds)
            ops[op.uid] = replace(op, deps=(uid,))

            consumer_ids = issue_consumers.get(op.uid, ())
            for consumer_id in consumer_ids:
                consumer = ops[consumer_id]
                ops[consumer.uid] = replace(consumer, deps=(*consumer.deps, uid))
            for dp_idx in range(max(int(program.dp_count), 1)):
                accounting.add(program.rank_for(op.device, dp_idx), label=op.comm_key or op.name, seconds=seconds)
        program.meta.misc["collective_launch_stats"] = accounting.records(emitted=False)

    succs: List[List[int]] = [[] for _ in ops]
    for op in ops:
        for dep in op.deps:
            succs[dep].append(op.uid)
    for entry in succs:
        entry.sort()

    time: float = 0
    counter = 0
    heap: List[Tuple[float, int, int]] = []
    ready: List[int] = []
    done: Set[int] = set()
    scheduled: Set[int] = set()
    finish: Dict[int, float] = {}
    gpu_free = [True] * _device_count(program)

    for op in ops:
        if not op.deps:
            if launch_enabled(program) and isinstance(op, ComputeOp):
                ready.append(op.uid)
            else:
                heappush(heap, (durations[op.uid], counter, op.uid))
                scheduled.add(op.uid)
                counter += 1
    for candidate in ready[:]:
        op = ops[candidate]
        if gpu_free[int(op.device)]:
            heappush(heap, (durations[candidate], counter, candidate))
            scheduled.add(candidate)
            counter += 1
            gpu_free[int(op.device)] = False
            ready.remove(candidate)

    while heap:
        time, _, uid = heappop(heap)
        done.add(uid)
        scheduled.discard(uid)
        finish[uid] = time

        for child in succs[uid]:
            if child in done or child in scheduled or child in ready:
                continue
            if all(dep in done for dep in ops[child].deps):
                ready.append(child)

        if isinstance(ops[uid], ComputeOp):
            gpu_free[int(ops[uid].device)] = True

        for candidate in ready[:]:
            op = ops[candidate]
            if isinstance(op, ComputeOp):
                if gpu_free[int(op.device)]:
                    heappush(heap, (time + durations[candidate], counter, candidate))
                    scheduled.add(candidate)
                    counter += 1
                    gpu_free[int(op.device)] = False
                    ready.remove(candidate)
            else:
                heappush(heap, (time + durations[candidate], counter, candidate))
                scheduled.add(candidate)
                counter += 1
                ready.remove(candidate)

    return CoarseEvalResult(
        total_time=time,
        finish_times=[finish.get(op.uid, -1) for op in ops[:original_count]],
    )

def evaluate(
    program: Program,
    network_model: Any,
    interconnect_params: Mapping[str, Tuple[float, float]],
) -> float:
    return evaluate_detailed(program, network_model, interconnect_params).total_time
