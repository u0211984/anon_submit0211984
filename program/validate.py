from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from program.ir import CollectiveOp, ComputeOp, GroupKey, Program, TransferOp
from timing_model import CollectiveType

class ProgramInvariantError(ValueError):
    pass

class GroupRaceWarning(UserWarning):
    pass

def _fail(invariant: str, message: str) -> None:
    raise ProgramInvariantError(f"{invariant}: {message}")

def validate_program(
    program: Program,
    *,
    check_races: bool = True,
    check_group_membership: bool = False,
) -> None:
    ops = program.ops
    devices: Set[int] = set(program.devices)
    if len(devices) != len(program.devices):
        _fail("V1", "Program.devices contains duplicate device ids")
    dp_count = max(int(program.dp_count), 1)

    checked_groups: Dict[GroupKey, Set[int]] = {}
    for idx, op in enumerate(ops):
        uid = op.uid
        if uid != idx:
            _fail("V1", f"ops[{idx}].uid == {uid}; uids must be dense list indices")
        for dep in op.deps:
            if not (0 <= int(dep) < uid):
                _fail(
                    "V1",
                    f"op {uid} ('{op.name}') has dep {dep}; every dep uid must "
                    f"be a valid uid < {uid}",
                )
        if isinstance(op, TransferOp):
            if not (0 <= op.producer < uid):
                _fail(
                    "V1",
                    f"TransferOp {uid} ('{op.name}') producer {op.producer} must "
                    f"be a valid uid < {uid}",
                )
            for consumer in op.consumers:
                if not (0 <= int(consumer) < len(ops)):
                    _fail("V1", f"TransferOp {uid} consumer {consumer} is not a valid uid")
                if isinstance(ops[int(consumer)], TransferOp):
                    _fail("V1", f"TransferOp {uid} consumer {consumer} is itself a TransferOp")

            if op.src_device not in devices or op.dst_device not in devices:
                _fail(
                    "V3",
                    f"TransferOp {uid} ('{op.name}') endpoints "
                    f"({op.src_device} -> {op.dst_device}) must be in Program.devices",
                )
            producer = ops[op.producer]
            if isinstance(producer, TransferOp):
                _fail("V3", f"TransferOp {uid} producer {op.producer} is a TransferOp")
            if producer.device != op.src_device:
                _fail(
                    "V3",
                    f"TransferOp {uid} ('{op.name}') producer {op.producer} lives on "
                    f"device {producer.device}, not src_device {op.src_device}",
                )
            for consumer_uid in op.consumers:
                consumer = ops[int(consumer_uid)]
                if consumer.device not in (op.src_device, op.dst_device):
                    _fail(
                        "V3",
                        f"TransferOp {uid} consumer {consumer_uid} on device "
                        f"{consumer.device} is on neither endpoint "
                        f"({op.src_device} -> {op.dst_device})",
                    )

            if (
                op.src_device == op.dst_device
                and int(op.size_bytes) > 0
                and getattr(op, "moe_component", None)
            ):
                _fail(
                    "V8",
                    f"TransferOp {uid} ('{op.name}') is a same-device p2p carrying "
                    f"{op.size_bytes} bytes for MoE component "
                    f"{op.moe_component!r}; same-device transfers are elided at "
                    "emission, so this silently DROPS the payload "
                    "(ext_moe_flat.md blocker 4)",
                )
        elif isinstance(op, CollectiveOp):
            if op.coll == CollectiveType.PIPELINE:
                _fail("V2", f"CollectiveOp {uid} ('{op.name}') has PIPELINE comm type")
            if op.group is not None:
                members = checked_groups.get(op.group)
                if members is None:
                    registered = program.groups.get(op.group)
                    if registered is None:
                        _fail(
                            "V2",
                            f"CollectiveOp {uid} ('{op.name}') references unregistered "
                            f"group {op.group}",
                        )
                    members = set(op.group.members)
                    if not members <= devices:
                        _fail(
                            "V2",
                            f"group {op.group} of op {uid} has members outside "
                            f"Program.devices {sorted(devices)}",
                        )
                    checked_groups[op.group] = members
                if op.device not in members:
                    _fail(
                        "V2",
                        f"CollectiveOp {uid} ('{op.name}') on device {op.device} is "
                        f"not a member of its group {op.group.members}",
                    )

            if (op.label is None) != bool(op.is_dp):
                _fail(
                    "V5",
                    f"CollectiveOp {uid} ('{op.name}'): is_dp={op.is_dp} but "
                    f"label={op.label!r} (is_dp must hold exactly when unlabeled)",
                )
            if op.group is not None and op.label is None:
                _fail(
                    "V5",
                    f"CollectiveOp {uid} ('{op.name}') has a group but no label; "
                    "wire gid interning is label-driven",
                )
            if op.label is not None and op.group is None:
                _fail(
                    "V5",
                    f"CollectiveOp {uid} ('{op.name}') has label {op.label!r} but "
                    "no group",
                )
        elif isinstance(op, ComputeOp):
            if len(op.duration) not in (1, dp_count):
                _fail(
                    "V4",
                    f"ComputeOp {uid} ('{op.name}') duration has length "
                    f"{len(op.duration)} but dp_count={dp_count} (must be 1 or dp_count)",
                )
    for key in program.groups:
        if tuple(sorted(key.members)) != key.members:
            _fail("V2", f"GroupKey members must be sorted: {key}")

    if check_group_membership:
        _check_group_instantiation(program)

    for op in ops:
        if not isinstance(op, TransferOp):
            continue
        if (
            op.src_device == op.dst_device
            and int(op.size_bytes) > 0
            and getattr(op, "moe_component", None)
        ):
            _fail(
                "V8",
                f"TransferOp {op.uid} ('{op.name}') is a same-device p2p carrying "
                f"{op.size_bytes} bytes for MoE component "
                f"{op.moe_component!r}; same-device transfers are elided at "
                "emission, so this silently DROPS the payload "
                "(ext_moe_flat.md blocker 4)",
            )

    if check_races:
        _warn_group_races(program)

def _check_group_instantiation(program: Program) -> None:
    seen: Dict[Tuple[GroupKey, str, int, object], Set[int]] = defaultdict(set)
    for op in program.ops:
        if isinstance(op, CollectiveOp) and op.group is not None:
            key = (op.group, str(op.label), int(op.size_bytes), op.coll)
            seen[key].add(int(op.device))
    for (group, label, size, coll), devices in seen.items():
        missing = sorted(set(group.members) - devices)
        if missing:
            _fail(
                "V7",
                f"collective '{label}' ({coll.name}, {size} bytes) of group "
                f"{group.axis}{list(group.members)} is instantiated on "
                f"{sorted(devices)} but NOT on {missing}; every member device of "
                "a communicator must issue the group's collectives or that "
                "group deadlocks",
            )

def _warn_group_races(program: Program) -> None:
    ops = program.ops

    consumer_transfers: Dict[int, List[TransferOp]] = defaultdict(list)
    for op in ops:
        if isinstance(op, TransferOp):
            for consumer in op.consumers:
                consumer_transfers[int(consumer)].append(op)

    pred_list: List[List[int]] = []
    for uid, op in enumerate(ops):
        if isinstance(op, TransferOp):
            pred_list.append([int(op.producer)])
            continue
        entry = [int(d) for d in op.deps]
        entry.extend(int(t.producer) for t in consumer_transfers.get(uid, ()))
        pred_list.append(entry)

    device_of: List[int] = [
        int(op.src_device) if isinstance(op, TransferOp) else int(op.device)
        for op in ops
    ]

    def has_path(src: int, dst: int) -> bool:
        device = device_of[dst]
        stack = [dst]
        seen: Set[int] = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen or cur < src:
                continue
            seen.add(cur)
            for pred in pred_list[cur]:
                if device_of[pred] == device or pred == src:
                    stack.append(pred)
        stack = [dst]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur == src:
                return True
            if cur in seen or cur < src:
                continue
            seen.add(cur)
            stack.extend(pred_list[cur])
        return False

    per_device_group: Dict[Tuple[GroupKey, int], List[int]] = defaultdict(list)
    for op in ops:
        if isinstance(op, CollectiveOp) and op.group is not None:
            per_device_group[(op.group, op.device)].append(op.uid)

    succ_count: Dict[int, int] = defaultdict(int)
    for op in ops:
        for dep in op.deps:
            succ_count[int(dep)] += 1

    for (group, device), uids in per_device_group.items():
        for earlier, later in zip(uids, uids[1:]):
            if not succ_count[earlier]:
                continue
            if not has_path(earlier, later):
                warnings.warn(
                    GroupRaceWarning(
                        f"V6: collectives {earlier} ('{ops[earlier].name}') and "
                        f"{later} ('{ops[later].name}') of group {group} on device "
                        f"{device} have no dependency path between them; their "
                        "relative issue order is a scheduling accident that member "
                        "devices may resolve differently (AstraSim matches "
                        "collectives by per-rank issue order)"
                    ),
                    stacklevel=3,
                )
