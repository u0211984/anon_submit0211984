from __future__ import annotations

from program.allreduce_curve import is_measured_call

import itertools
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from timing_model import CollectiveType

from astrasim_lib.et_utils import (
    chakra_encode,
    new_comm_node,
    new_comp_node,
    new_recv_node,
    new_send_node,
    pb,
    write_et_node,
)
from program.ir import CollectiveOp, ComputeOp, Program, TransferOp
from program.collective_launch import LaunchAccounting, is_nccl_call, is_nccl_collective, launch_issue_consumers
from program.nccl_latency import enabled as launch_enabled, launch_seconds as call_launch_seconds
from program.pipeline_collectives import FOLDED_SECONDS
from program.vllm_allreduce import op_choice, custom_seconds
from program.validate import validate_program

_WIRE_GROUP_BASE_ID = 1000

_SYNC_TAG_BASE = 1_000_000

class EmissionError(RuntimeError):
    pass

@dataclass
class EmittedBundle:
    et_prefix: str
    rank_ids: List[int]
    manifest_path: str
    comm_groups: Dict[str, List[int]] = field(default_factory=dict)
    comm_groups_path: Optional[str] = None

    zero_moe_operation_groups: Optional[List[Dict[str, Any]]] = None

def _collective_enum(coll: CollectiveType) -> int:
    if coll is None:
        raise ValueError("Collective comm_type is required")
    if not isinstance(coll, CollectiveType):
        raise TypeError(f"comm_type must be CollectiveType (got {type(coll).__name__})")
    if coll == CollectiveType.PIPELINE:
        raise ValueError("Pipeline comm_type should not be mapped to a collective enum")
    mapping = {
        CollectiveType.ALL_REDUCE: pb.ALL_REDUCE,
        CollectiveType.ALL_GATHER: pb.ALL_GATHER,
        CollectiveType.REDUCE_SCATTER: pb.REDUCE_SCATTER,
        CollectiveType.ALL_TO_ALL: pb.ALL_TO_ALL,
    }
    return mapping[coll]

class _Trace:
    def __init__(self, rank: int, path: str) -> None:
        self.rank = rank
        self.path = path
        self.nodes: List["pb.Node"] = []

        self.control_ids: Set[int] = set()
        self.launch_ids: Set[int] = set()

    @property
    def next_id(self) -> int:
        return len(self.nodes)

    def append_node(self, node: "pb.Node", *, control: bool = False, launch: bool = False) -> None:
        if control:
            self.control_ids.add(len(self.nodes))
        if launch:
            self.launch_ids.add(len(self.nodes))
        self.nodes.append(node)

    def renumber_control_priority(self) -> None:
        if not self.nodes:
            return

        control_nodes: List["pb.Node"] = []
        regular_nodes: List["pb.Node"] = []
        launch_nodes: List["pb.Node"] = []
        for index, node in enumerate(self.nodes):
            if index in self.control_ids:
                control_nodes.append(node)
            elif index in self.launch_ids:
                launch_nodes.append(node)
            else:
                regular_nodes.append(node)

        if not control_nodes and not launch_nodes:
            return

        id_map: Dict[int, int] = {}
        new_order: List["pb.Node"] = []

        for node in control_nodes + regular_nodes + launch_nodes:
            new_id = len(new_order)
            id_map[int(node.id)] = new_id
            node.id = new_id
            new_order.append(node)

        for node in new_order:
            remapped = [id_map.get(int(dep), int(dep)) for dep in node.ctrl_deps]
            node.ctrl_deps[:] = remapped

        self.nodes = new_order
        self.control_ids = set(range(len(control_nodes)))
        self.launch_ids = set(range(len(control_nodes) + len(regular_nodes), len(new_order)))

    def write(self) -> None:
        with open(self.path, "wb") as fh:
            chakra_encode(fh, pb.GlobalMetadata(version="0.0.4"))
            for node in self.nodes:
                write_et_node(fh, node)

def emit_chakra(program: Program, output_dir: str) -> EmittedBundle:
    validate_program(program, check_races=False)
    os.makedirs(output_dir, exist_ok=True)

    ops = program.ops
    launch_seconds = float(program.meta.misc.get("collective_launch_overhead", 0.0))
    charge_launch = launch_enabled(program)
    folded_seconds = program.meta.misc.get(FOLDED_SECONDS, {})
    launch_accounting = LaunchAccounting(launch_seconds)
    launch_ids = {}
    energy_launch_nodes = []
    endpoint_launch_chains = []
    protocol_readiness = {}
    protocol_ordinals = {}
    dp_count = max(int(program.dp_count), 1)
    devices = program.devices
    device_index = {device: idx for idx, device in enumerate(devices)}
    ns_initial = program.num_stages_initial()
    compute_devices = program.compute_devices()

    def rank_for(device: int, dp_idx: int) -> int:
        idx = device_index.get(device)
        if idx is None:
            raise EmissionError(f"Device {device} is not in Program.devices")
        return dp_idx * ns_initial + idx

    layout = program.layout
    _dp_sibling_cache: Dict[Tuple[int, Tuple[str, ...]], Tuple[int, ...]] = {}

    def _intern_dp_group(siblings: Tuple[int, ...]) -> str:
        members = tuple(
            sorted(
                rank_for(device, dp_idx)
                for device in siblings
                for dp_idx in range(dp_count)
            )
        )
        gid = members_to_gid.get(members)
        if gid is None:
            gid = str(next(gid_counter))
            members_to_gid[members] = gid
            gid_members[gid] = list(members)
        return gid

    _dependents: Optional[set] = None

    def _has_dependents(uid: int) -> bool:
        nonlocal _dependents
        if _dependents is None:
            found = set()
            for other in ops:
                for dep in getattr(other, "deps", ()):
                    found.add(int(dep))
                producer = getattr(other, "producer", None)
                if producer is not None:
                    found.add(int(producer))
                for consumer in getattr(other, "consumers", ()) or ():
                    found.add(int(consumer))
            _dependents = found
        return int(uid) in _dependents

    def _dp_group_devices(op: "CollectiveOp") -> Tuple[int, ...]:
        companions = tuple(a for a in (op.axes or ()) if a != "dp")
        key = (int(op.device), companions)
        cached = _dp_sibling_cache.get(key)
        if cached is not None:
            return cached

        live = tuple(a for a in companions if a in layout.axis_order)
        if not live or not layout.axis_order:
            result = (int(op.device),)
        else:
            own = layout.coords_of(int(op.device))
            companions = live
            fixed = {a: v for a, v in own.items() if a not in companions}
            result = tuple(
                sorted(
                    d
                    for d in devices
                    if all(layout.coords_of(int(d)).get(a) == v for a, v in fixed.items())
                )
            )
            if int(op.device) not in result:  # pragma: no cover - defensive
                raise EmissionError(
                    f"Collective {op.name!r}: owning device {op.device} is not in its "
                    f"own communicator {result}"
                )
        _dp_sibling_cache[key] = result
        return result

    traces: Dict[int, _Trace] = {}
    for dp_idx in range(dp_count):
        for device in compute_devices:
            rank = rank_for(device, dp_idx)
            traces[rank] = _Trace(rank, os.path.join(output_dir, f"llm_graph.{rank}.et"))

    def _trace_for(rank: int, context: str) -> _Trace:
        trace = traces.get(rank)
        if trace is None:
            raise EmissionError(
                f"{context}: rank {rank} has no trace (device outside the compute "
                "stage set — the legacy converter would KeyError here)"
            )
        return trace

    label_tokens: Dict[str, set] = defaultdict(set)
    for op in ops:
        if isinstance(op, CollectiveOp) and op.label is not None and op.group is not None:
            for dp_idx in range(dp_count):
                members_tuple = tuple(sorted(rank_for(d, dp_idx) for d in op.group.members))
                label_tokens[op.label].add((op.group.axis, dp_idx, members_tuple))

    label_dp_gid: Dict[Tuple[str, int], str] = {}
    members_to_gid: Dict[Tuple[int, ...], str] = {}
    gid_members: Dict[str, List[int]] = {}

    gid_axes: Dict[str, set] = defaultdict(set)
    gid_labels: Dict[str, set] = defaultdict(set)
    gid_counter = itertools.count(start=_WIRE_GROUP_BASE_ID)
    for label in sorted(label_tokens.keys()):
        for axis, dp_idx, members_tuple in sorted(label_tokens[label]):
            gid = members_to_gid.get(members_tuple)
            if gid is None:
                gid = str(next(gid_counter))
                members_to_gid[members_tuple] = gid
                if gid not in gid_members:
                    gid_members[gid] = list(members_tuple)
            existing = label_dp_gid.get((label, dp_idx))
            if existing is not None and existing != gid:
                raise EmissionError(
                    f"collective label '{label}' maps to two different "
                    f"communicator member sets at dp index {dp_idx}: gid "
                    f"{existing} (members {gid_members.get(existing)}) vs gid "
                    f"{gid} (members {gid_members.get(gid)}). One label must "
                    "identify exactly one communicator per dp index."
                )
            label_dp_gid[(label, dp_idx)] = gid
            gid_axes[gid].add(str(axis))
            gid_labels[gid].add(str(label))

    gather_groups = {}
    for candidate in ops:
        if not isinstance(candidate, ComputeOp):
            continue
        for call in candidate.embedded_collectives:
            if call.coll is CollectiveType.GATHER:
                if candidate.work is None:
                    raise EmissionError("embedded gather needs a semantic work item")
                key = (candidate.work, call.name)
                gather_groups.setdefault(key, []).append(candidate)
    gather_tags = {}
    tag_counter = len(ops) + 1000000
    for key, members in gather_groups.items():
        members.sort(key=lambda item: device_index[item.device])
        for member in members[1:]:
            gather_tags[(key, member.device)] = tag_counter
            tag_counter += 1

    et_ids: Dict[Tuple[int, int], int] = {}

    group_records: Dict[str, Dict[int, List[Tuple[Any, int, int]]]] = defaultdict(lambda: defaultdict(list))

    cross_rank_syncs: List[Tuple[int, int]] = []

    def _resolve_deps(
        dep_uids: Any, own_rank: int, dp_idx: int, *, owner: str, owner_uid: int
    ) -> List[int]:
        deps: List[int] = []

        def _resolve(uid: int) -> None:
            dep_op = ops[uid]
            if isinstance(dep_op, TransferOp):
                if dep_op.src_device != dep_op.dst_device:
                    return
                if rank_for(dep_op.src_device, dp_idx) != own_rank:
                    cross_rank_syncs.append((owner_uid, uid))
                    return
                for inner in dep_op.deps:
                    _resolve(int(inner))
                return
            dep_rank = rank_for(dep_op.device, dp_idx)
            if dep_rank != own_rank:
                cross_rank_syncs.append((owner_uid, uid))
                return
            key = (uid, dep_rank)
            if key not in et_ids:
                raise EmissionError(
                    f"op {owner_uid} ('{owner}') depends on op {uid} "
                    f"('{dep_op.name}') which has no ET node on rank {dep_rank} "
                    "(skipped or not yet emitted)"
                )
            deps.append(et_ids[key])

        for dep in dep_uids:
            _resolve(int(dep))
        unique: List[int] = []
        for dep in deps:
            if dep not in unique:
                unique.append(dep)
        return unique

    def _resolved_deps(op: Any, dp_idx: int) -> List[int]:
        return _resolve_deps(
            op.deps,
            rank_for(op.device, dp_idx),
            dp_idx,
            owner=op.name,
            owner_uid=op.uid,
        )

    for op in ops:
        if isinstance(op, TransferOp):
            continue
        if isinstance(op, ComputeOp):
            for dp_idx in range(dp_count):
                rank = rank_for(op.device, dp_idx)
                trace = _trace_for(rank, f"compute op {op.uid}")
                unique_deps = _resolved_deps(op, dp_idx)
                duration_sec = float(op.duration[dp_idx] if len(op.duration) > 1 else op.duration[0])
                duration_micros = int(round(duration_sec * 1e6)) if duration_sec else 0
                gather = [c for c in op.embedded_collectives if c.coll is CollectiveType.GATHER]
                if gather and program.meta.misc.get("granularity") != "pipeline":
                    if len(gather) != 1:
                        raise EmissionError("only one embedded gather per endpoint is supported")
                    call = gather[0]
                    group_key = (op.work, call.name)
                    members = gather_groups[group_key]
                    if len(members) != call.participants:
                        raise EmissionError("gather endpoint requires one placed op per participant")
                    pre_us = int(round(float(call.offset_seconds) * 1e6))
                    tail_us = int(round(max(0.0, duration_sec - call.offset_seconds - call.duration_seconds) * 1e6))
                    pre = new_comp_node(trace.next_id, f"{op.name}_{op.uid}_gather_pre", max(pre_us, 0))
                    pre.ctrl_deps.extend(unique_deps)
                    trace.append_node(pre)
                    deps = [pre.id]
                    launch_us = launch_accounting.add(rank, label=call.name, emit=True, seconds=call_launch_seconds(program, call, exact=True)) if charge_launch else 0
                    if launch_us:
                        launch = new_comp_node(trace.next_id, f"{call.name}_{op.uid}_nccl_launch", launch_us)
                        launch.ctrl_deps.extend(deps)
                        trace.append_node(launch, launch=True)
                        deps = [launch.id]
                    root = rank_for(members[0].device, dp_idx)
                    wire_ids = []
                    if rank == root:
                        for member in members[1:]:
                            src = rank_for(member.device, dp_idx)
                            node = new_recv_node(trace.next_id, f"{call.name}_recv_{src}", int(call.size_bytes), src,
                                                 gather_tags[(group_key, member.device)])
                            node.ctrl_deps.extend(deps)
                            trace.append_node(node)
                            wire_ids.append(node.id)
                    else:
                        node = new_send_node(trace.next_id, f"{call.name}_send_{rank}", int(call.size_bytes), root,
                                             gather_tags[(group_key, op.device)])
                        node.ctrl_deps.extend(deps)
                        trace.append_node(node)
                        wire_ids.append(node.id)
                    tail = new_comp_node(trace.next_id, f"{op.name}_{op.uid}_gather_tail", tail_us)
                    tail.ctrl_deps.extend(wire_ids or deps)
                    trace.append_node(tail)
                    et_ids[(op.uid, rank)] = tail.id
                    continue
                embedded = [c for c in op.embedded_collectives
                            if charge_launch and not is_measured_call(program, c) and is_nccl_call(
                                c.coll, participants=c.participants, size_bytes=c.size_bytes,
                                algorithm=op_choice(program, c).algorithm)]

                for coll in embedded:
                    launch_us = launch_accounting.add(rank, label=coll.name, emit=True, seconds=call_launch_seconds(program, coll, exact=True))

                    if not launch_us:
                        continue
                    launch = new_comp_node(trace.next_id, f"{coll.name}_{op.uid}_nccl_launch",
                                           launch_us)
                    launch.ctrl_deps.extend(unique_deps)
                    trace.append_node(launch, launch=True)
                    launch_ids.setdefault((op.uid, rank), launch.id)
                    unique_deps = [launch.id]
                node_id = trace.next_id
                comp_node = new_comp_node(node_id, f"{op.name}_{op.uid}", max(duration_micros, 0))
                comp_node.ctrl_deps.extend(unique_deps)
                trace.append_node(comp_node)
                et_ids[(op.uid, rank)] = node_id

                for device in op.launch_devices if embedded else ():
                    other_rank = rank_for(device, dp_idx)
                    other_trace = _trace_for(other_rank, f"endpoint launch {op.uid}")
                    chain = []
                    for coll in embedded:
                        launch_us = launch_accounting.add(other_rank, label=coll.name, emit=True, seconds=call_launch_seconds(program, coll, exact=True))
                        if not launch_us:
                            continue
                        launch = new_comp_node(other_trace.next_id,
                            f"{coll.name}_{op.uid}_nccl_launch", launch_us)
                        if chain:
                            launch.ctrl_deps.append(chain[-1])
                        other_trace.append_node(launch, launch=True)
                        chain.append(launch.id)
                    if chain:
                        target = next((later.uid for later in ops[op.uid + 1:]
                                       if not isinstance(later, TransferOp)
                                       and later.device == device), None)
                        endpoint_launch_chains.append((other_rank, chain, target))
            continue

        singleton_dp = op.is_dp and len(_dp_group_devices(op)) * dp_count <= 1
        if singleton_dp and op.uid not in folded_seconds and not _has_dependents(op.uid):
            continue
        for dp_idx in range(dp_count):
            rank = rank_for(op.device, dp_idx)
            trace = _trace_for(rank, f"collective op {op.uid}")
            unique_deps = _resolved_deps(op, dp_idx)
            node_id = trace.next_id
            choice = op_choice(program, op)
            if op.uid in folded_seconds:
                if charge_launch and is_nccl_collective(op, algorithm=choice.algorithm):
                    launch_us = launch_accounting.add(
                        rank, label=op.comm_key or op.name, emit=True,
                        seconds=call_launch_seconds(program, op, exact=True))
                    if launch_us:
                        launch = new_comp_node(node_id, f"{op.name}_{op.uid}_nccl_launch", launch_us)
                        launch.ctrl_deps.extend(unique_deps)
                        trace.append_node(launch, launch=True)
                        launch_ids[(op.uid, rank)] = launch.id
                        node_id = trace.next_id
                        unique_deps = [launch.id]
                node = new_comp_node(node_id, f"{op.name}_{op.uid}_pipeline_cp",
                                     max(0, int(round(folded_seconds[op.uid] * 1e6))))
                node.ctrl_deps.extend(unique_deps)
                trace.append_node(node)
                et_ids[(op.uid, rank)] = node_id
                continue
            if is_measured_call(program, op):
                from program.allreduce_curve import measured_seconds, program_policy
                seconds = measured_seconds(program_policy(program), choice,
                                           op.participants, op.size_bytes, op.interconnect, op.coll)
                measured = "allreduce" if op.coll is CollectiveType.ALL_REDUCE else op.coll.value
                node = new_comp_node(node_id, f"{op.name}_{op.uid}_measured_{measured}",
                                     int(round(seconds * 1e6)))
                node.ctrl_deps.extend(unique_deps)
                trace.append_node(node)
                et_ids[(op.uid, rank)] = node_id
                if op.label is not None:
                    gid = label_dp_gid.get((op.label, dp_idx))
                    members = tuple(gid_members.get(gid, ()))
                else:
                    members = tuple(sorted(rank_for(device, replica)
                        for device in _dp_group_devices(op) for replica in range(dp_count)))
                if len(members) != op.participants or rank not in members:
                    raise EmissionError("measured all-reduce requires all communicator ranks to be placed")
                ordinal_key = (members, rank)
                ordinal = protocol_ordinals.get(ordinal_key, 0)
                protocol_ordinals[ordinal_key] = ordinal + 1
                protocol_readiness.setdefault((members, ordinal), {})[rank] = (
                    node, tuple(unique_deps), (op.coll, op.size_bytes))
                continue
            if choice.algorithm.startswith("custom_"):
                seconds = custom_seconds(choice, op.size_bytes, op.participants,
                                         program.meta.misc["vllm_allreduce"])
                node = new_comp_node(node_id, f"{op.name}_{op.uid}_{choice.algorithm}",
                                     int(round(seconds * 1e6)))
                node.ctrl_deps.extend(unique_deps)
                trace.append_node(node)
                et_ids[(op.uid, rank)] = node_id
                continue
            if op.is_dp and op.size_bytes <= 0 and program.meta.misc.get("megatron_dp_buckets") == "one":
                node = new_comp_node(node_id, f"{op.name}_{op.uid}_dp{dp_idx}_empty", 0)
                node.ctrl_deps.extend(unique_deps)
                trace.append_node(node)
                et_ids[(op.uid, rank)] = node_id
                continue
            if (program.meta.misc.get("nccl_latency_rule") == "tuning_protocol"
                    and op.coll in (CollectiveType.ALL_REDUCE, CollectiveType.ALL_GATHER, CollectiveType.REDUCE_SCATTER)
                    and is_nccl_collective(op, algorithm=choice.algorithm)):
                from program.nccl_latency import protocol_for_call
                model = protocol_for_call(program, op)
                launch_us = launch_accounting.add(
                    rank, label=op.comm_key or op.name, emit=True,
                    seconds=call_launch_seconds(program, op, exact=True))
                if launch_us:
                    launch = new_comp_node(trace.next_id, f"{op.name}_{op.uid}_nccl_launch", launch_us)
                    launch.ctrl_deps.extend(unique_deps)
                    trace.append_node(launch, launch=True)
                    launch_ids[(op.uid, rank)] = launch.id
                    unique_deps = [launch.id]

                node = new_comp_node(trace.next_id, f"{op.name}_{op.uid}_nccl_{model['protocol']}",
                                     max(0, int(round(model["transfer_seconds"] * 1e6))))
                node.ctrl_deps.extend(unique_deps)
                trace.append_node(node)
                et_ids[(op.uid, rank)] = node.id
                if op.label is not None:
                    gid = label_dp_gid.get((op.label, dp_idx))
                    members = tuple(gid_members.get(gid, ()))
                else:
                    members = tuple(sorted(rank_for(device, replica)
                        for device in _dp_group_devices(op) for replica in range(dp_count)))
                if len(members) != op.participants or rank not in members:
                    raise EmissionError("tuning_protocol requires all native communicator ranks to be placed")
                ordinal_key = (members, rank)
                ordinal = protocol_ordinals.get(ordinal_key, 0)
                protocol_ordinals[ordinal_key] = ordinal + 1
                protocol_readiness.setdefault((members, ordinal), {})[rank] = (
                    node, tuple(unique_deps), (op.coll, op.size_bytes))
                continue
            if op.label is not None:
                comm_name = op.label
                gid = label_dp_gid.get((op.label, dp_idx))
                members = gid_members.get(gid) if gid else None
                if members is not None and len(members) <= 1:
                    noop_node = new_comp_node(node_id, f"{comm_name}_noop", 0)
                    noop_node.ctrl_deps.extend(unique_deps)
                    trace.append_node(noop_node)
                    et_ids[(op.uid, rank)] = node_id
                    continue
                comm_node = new_comm_node(node_id, comm_name, _collective_enum(op.coll), op.size_bytes)
                if gid:
                    comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=str(gid)))
                    group_records[str(gid)][rank].append(
                        (comm_node, _collective_enum(op.coll), int(op.size_bytes))
                    )
            elif singleton_dp:
                noop = new_comp_node(node_id, f"{op.name}_{op.uid}_dp{dp_idx}_noop", 0)
                noop.ctrl_deps.extend(unique_deps)
                trace.append_node(noop)
                et_ids[(op.uid, rank)] = node_id
                continue
            else:
                comm_name = f"{op.name}_{op.uid}_dp{dp_idx}"
                comm_node = new_comm_node(node_id, comm_name, _collective_enum(op.coll), op.size_bytes)
                siblings = _dp_group_devices(op)
                if len(siblings) * dp_count > 1:
                    if len(siblings) == 1:
                        gid = str(device_index[op.device] + 1)
                    else:
                        gid = _intern_dp_group(siblings)
                    comm_node.attr.append(pb.AttributeProto(name="pg_name", string_val=gid))
                    group_records[gid][rank].append(
                        (comm_node, _collective_enum(op.coll), int(op.size_bytes))
                    )
                    gid_axes[gid].update(op.axes or ("dp",))
            energy_launch = None
            if charge_launch and is_nccl_collective(op, algorithm=choice.algorithm):
                launch_us = launch_accounting.add(rank, label=op.comm_key or op.name, emit=True, seconds=call_launch_seconds(program, op, exact=True))
                if launch_us:
                    launch_id = node_id
                    launch_node = new_comp_node(launch_id, f"{op.name}_{op.uid}_nccl_launch", launch_us)
                    energy_launch = launch_node
                    launch_node.ctrl_deps.extend(unique_deps)
                    trace.append_node(launch_node, launch=True)
                    launch_ids[(op.uid, rank)] = launch_id
                    node_id = trace.next_id
                    comm_node.id = node_id
                    unique_deps = [launch_id]
            comm_node.ctrl_deps.extend(unique_deps)
            trace.append_node(comm_node)
            energy_launch_nodes.append((rank, comm_node, energy_launch))
            et_ids[(op.uid, rank)] = node_id

    if charge_launch:
        issue_consumers = launch_issue_consumers(program)
        for op in ops:
            if not isinstance(op, CollectiveOp):
                continue
            consumer_ids = issue_consumers.get(op.uid, ())
            for consumer_id in consumer_ids:
                consumer = ops[consumer_id]
                for dp_idx in range(dp_count):
                    rank = rank_for(op.device, dp_idx)
                    launch_id = launch_ids.get((op.uid, rank))
                    if launch_id is None:
                        continue
                    consumer_rank = rank_for(consumer.device, dp_idx)
                    if consumer_rank != rank:
                        raise EmissionError(f"overlap launch and consumer are on different ranks: {op.name}")
                    key = (consumer.uid, rank)
                    target_id = launch_ids.get(key, et_ids[key])
                    target = traces[rank].nodes[target_id]
                    if launch_id not in target.ctrl_deps:
                        target.ctrl_deps.append(launch_id)

    transfers = [op for op in ops if isinstance(op, TransferOp)]

    events: List[Tuple[str, TransferOp]] = []
    for transfer in transfers:
        if transfer.src_device == transfer.dst_device:
            continue
        events.append(("send", transfer))
        events.append(("recv", transfer))

    send_ids: Dict[Tuple[int, int], int] = {}
    recv_ids: Dict[Tuple[int, int], int] = {}
    for kind, transfer in events:
        is_control = transfer.is_control
        size = 1 if is_control else transfer.size_bytes
        for dp_idx in range(dp_count):
            src_rank = rank_for(transfer.src_device, dp_idx)
            dst_rank = rank_for(transfer.dst_device, dp_idx)
            if kind == "send":
                trace = _trace_for(src_rank, f"transfer {transfer.uid} send")
                name = (
                    f"{transfer.name}_send_control"
                    if is_control
                    else f"{transfer.name}_send_dp{dp_idx}"
                )
                node_id = trace.next_id
                node = new_send_node(node_id, name, size, dst_rank, transfer.uid)
                producer_key = (transfer.producer, src_rank)
                if producer_key not in et_ids:
                    raise EmissionError(
                        f"transfer {transfer.uid} ('{transfer.name}'): producer op "
                        f"{transfer.producer} has no ET node on rank {src_rank}"
                    )
                node.ctrl_deps.append(et_ids[producer_key])

                for extra in _resolve_deps(
                    transfer.deps,
                    src_rank,
                    dp_idx,
                    owner=transfer.name,
                    owner_uid=transfer.uid,
                ):
                    if extra not in node.ctrl_deps:
                        node.ctrl_deps.append(extra)
                trace.append_node(node, control=is_control)
                send_ids[(transfer.uid, dp_idx)] = node_id
            else:
                trace = _trace_for(dst_rank, f"transfer {transfer.uid} recv")
                name = (
                    f"{transfer.name}_recv_control"
                    if is_control
                    else f"{transfer.name}_recv_dp{dp_idx}"
                )
                node_id = trace.next_id
                node = new_recv_node(node_id, name, size, src_rank, transfer.uid)
                trace.append_node(node, control=is_control)
                recv_ids[(transfer.uid, dp_idx)] = node_id

    for transfer in transfers:
        if transfer.src_device == transfer.dst_device:
            continue
        for consumer_uid in transfer.consumers:
            consumer = ops[consumer_uid]
            on_dst = consumer.device == transfer.dst_device
            for dp_idx in range(dp_count):
                rank = rank_for(consumer.device, dp_idx)
                consumer_key = (consumer_uid, rank)
                if consumer_key not in et_ids:
                    raise EmissionError(
                        f"transfer {transfer.uid} ('{transfer.name}'): consumer op "
                        f"{consumer_uid} has no ET node on rank {rank}"
                    )
                node = traces[rank].nodes[launch_ids.get(consumer_key, et_ids[consumer_key])]
                wire_ids = recv_ids if on_dst else send_ids
                wire_id = wire_ids.get((transfer.uid, dp_idx))
                if wire_id is None:
                    raise EmissionError(
                        f"transfer {transfer.uid} ('{transfer.name}'): no "
                        f"{'RECV' if on_dst else 'SEND'} node exists for consumer "
                        f"{consumer_uid} at dp {dp_idx}"
                    )
                if wire_id not in node.ctrl_deps:
                    node.ctrl_deps.append(wire_id)

    protocol_tag = _SYNC_TAG_BASE + len(set(cross_rank_syncs)) * dp_count
    for (members, ordinal), arrivals in sorted(protocol_readiness.items()):
        if set(arrivals) != set(members):
            raise EmissionError(f"tuning_protocol missing communicator arrivals for invocation {ordinal}")
        if len({entry[2] for entry in arrivals.values()}) != 1:
            raise EmissionError("tuning_protocol communicator collective order differs across ranks")
        for source in members:
            for destination in members:
                if source == destination:
                    continue
                source_trace, destination_trace = traces[source], traces[destination]
                send = new_send_node(source_trace.next_id,
                    f"nccl_protocol_ready_{ordinal}_{source}_{destination}_send_control",
                    1, destination, protocol_tag)
                send.ctrl_deps.extend(arrivals[source][1])
                source_trace.append_node(send, control=True)
                recv = new_recv_node(destination_trace.next_id,
                    f"nccl_protocol_ready_{ordinal}_{source}_{destination}_recv_control",
                    1, source, protocol_tag)
                recv.ctrl_deps.extend(arrivals[destination][1])
                destination_trace.append_node(recv, control=True)
                arrivals[source][0].ctrl_deps.append(send.id)
                arrivals[destination][0].ctrl_deps.append(recv.id)
                protocol_tag += 1

    for index, (consumer_uid, dep_uid) in enumerate(sorted(set(cross_rank_syncs))):
        consumer = ops[consumer_uid]
        dep_op = ops[dep_uid]
        src_device = (
            dep_op.src_device if isinstance(dep_op, TransferOp) else dep_op.device
        )

        consumer_is_transfer = isinstance(consumer, TransferOp)
        if consumer_is_transfer:
            if consumer.src_device == consumer.dst_device:
                raise EmissionError(
                    f"cross-rank sync into transfer {consumer_uid} "
                    f"('{consumer.name}') is impossible: the transfer is "
                    "same-device and therefore elided, so it has no ET node"
                )
            consumer_device = consumer.src_device
        else:
            consumer_device = consumer.device
        tag = _SYNC_TAG_BASE + index
        for dp_idx in range(dp_count):
            src_rank = rank_for(src_device, dp_idx)
            dst_rank = rank_for(consumer_device, dp_idx)
            if src_rank == dst_rank:  # pragma: no cover - defensive
                raise EmissionError(
                    f"cross-rank sync {consumer_uid} <- {dep_uid} resolved to the "
                    f"same rank {src_rank}; it should have been a ctrl_dep"
                )
            before = len(cross_rank_syncs)
            anchors = _resolve_deps(
                (dep_uid,), src_rank, dp_idx, owner=dep_op.name, owner_uid=dep_uid
            )
            if len(cross_rank_syncs) != before or not anchors:
                raise EmissionError(
                    f"cross-rank sync {consumer_uid} ('{consumer.name}') <- "
                    f"{dep_uid} ('{dep_op.name}') cannot be carried: resolving the "
                    f"dep on its own rank {src_rank} yielded {anchors!r} and "
                    f"{len(cross_rank_syncs) - before} further cross-rank dep(s). "
                    "Anchor the requirement on the consumer's device instead "
                    "(INTERFACES §4.4)."
                )
            send_trace = _trace_for(src_rank, f"cross-rank sync {consumer_uid} send")
            send_id = send_trace.next_id
            send_node = new_send_node(
                send_id,
                f"cross_rank_sync_{consumer_uid}_{dep_uid}_send_control",
                1,
                dst_rank,
                tag,
            )
            send_node.ctrl_deps.extend(anchors)
            send_trace.append_node(send_node, control=True)

            recv_trace = _trace_for(dst_rank, f"cross-rank sync {consumer_uid} recv")
            recv_id = recv_trace.next_id
            recv_node = new_recv_node(
                recv_id,
                f"cross_rank_sync_{consumer_uid}_{dep_uid}_recv_control",
                1,
                src_rank,
                tag,
            )
            recv_trace.append_node(recv_node, control=True)
            if consumer_is_transfer:
                target_id = send_ids.get((consumer_uid, dp_idx))
            else:
                target_id = launch_ids.get((consumer_uid, dst_rank), et_ids.get((consumer_uid, dst_rank)))
            if target_id is None:
                raise EmissionError(
                    f"cross-rank sync {consumer_uid} ('{consumer.name}') has no ET "
                    f"node on rank {dst_rank}"
                )
            consumer_node = recv_trace.nodes[target_id]
            if recv_id not in consumer_node.ctrl_deps:
                consumer_node.ctrl_deps.append(recv_id)

    for rank, chain, target in reversed(endpoint_launch_chains):
        trace = traces[rank]
        if target is None:
            originals = [node for node in trace.nodes if node.id not in trace.launch_ids]
            trace.nodes[chain[0]].ctrl_deps.extend(node.id for node in originals)
            continue
        target_id = launch_ids.get((target, rank), et_ids.get((target, rank)))
        if target_id is None:
            raise EmissionError(f"endpoint launch target {target} has no node on rank {rank}")
        node = trace.nodes[target_id]
        trace.nodes[chain[0]].ctrl_deps.extend(node.ctrl_deps)
        node.ctrl_deps[:] = [chain[-1]]

    if charge_launch:
        program.meta.misc["collective_launch_stats"] = launch_accounting.records(emitted=True)

    _check_no_lost_successors(ops, et_ids, traces, dp_count)

    for trace in traces.values():
        trace.renumber_control_priority()

    _check_group_order_postcondition(group_records, gid_members, dp_count, ns_initial)

    zero_moe_groups = None
    if (program.meta.misc.get("sharding_policy") == "zero_moe" or
            any(isinstance(op, CollectiveOp) and (op.comm_key or "").startswith("zm_") for op in ops)):
        rank_ids_sorted = sorted(traces)
        rank_index = {rank: index for index, rank in enumerate(rank_ids_sorted)}
        rank_count = len(rank_ids_sorted)
        names = {pb.ALL_REDUCE: "ALL_REDUCE", pb.ALL_GATHER: "ALL_GATHER",
                 pb.REDUCE_SCATTER: "REDUCE_SCATTER", pb.ALL_TO_ALL: "ALL_TO_ALL"}
        zero_moe_groups = []
        for gid, per_rank in sorted(group_records.items()):
            members = gid_members.get(gid)
            if members is None:
                stage_index = int(gid) - 1
                members = [index * ns_initial + stage_index for index in range(dp_count)]
            for rank, records in sorted(per_rank.items()):
                for ordinal, (node, kind, size) in enumerate(sorted(records, key=lambda entry: int(entry[0].id))):
                    zero_moe_groups.append({"operation_id": int(node.id) * rank_count + rank_index[rank],
                        "rank": rank, "group_id": str(gid), "ordinal": ordinal,
                        "collective_type": names[kind], "collective_size_bytes": size,
                        "members": list(members)})
        for rank, trace in traces.items():
            remap = {int(node.id): int(node.id) * rank_count + rank_index[rank] for node in trace.nodes}
            for node in trace.nodes:
                node.id = remap[int(node.id)]
                node.ctrl_deps[:] = [remap[int(dep)] for dep in node.ctrl_deps]
        with open(os.path.join(output_dir, "zero_moe_traffic_groups.json"), "w") as handle:
            json.dump(zero_moe_groups, handle, sort_keys=True, separators=(",", ":"))

    energy_launch_durations = {}
    for rank, comm_node, launch_node in energy_launch_nodes:
        energy_launch_durations.setdefault(str(comm_node.id), {})[rank] = (
            float(launch_node.duration_micros) * 1e-6 if launch_node is not None else 0.0)
    program.meta.misc["energy_launch_durations"] = energy_launch_durations

    for trace in traces.values():
        trace.write()

    et_prefix = os.path.join(output_dir, "llm_graph")
    rank_ids = sorted(traces.keys())

    manifest_path = _write_manifest(output_dir, traces, rank_ids)
    comm_groups, comm_groups_path = _write_comm_groups(output_dir, dp_count, rank_ids, gid_members)
    _write_comm_axes(output_dir, comm_groups, gid_axes, gid_labels)

    return EmittedBundle(
        et_prefix=et_prefix,
        rank_ids=rank_ids,
        manifest_path=manifest_path,
        comm_groups=comm_groups,
        comm_groups_path=comm_groups_path,
        zero_moe_operation_groups=zero_moe_groups,
    )

def _check_no_lost_successors(
    ops: Any,
    et_ids: Dict[Tuple[int, int], int],
    traces: Dict[int, "_Trace"],
    dp_count: int,
) -> None:
    has_ir_successor: Set[int] = set()
    pending: List[int] = []

    def _mark(dep_uids: Any) -> None:
        for raw in dep_uids:
            dep = int(raw)
            if dep in has_ir_successor:
                continue
            has_ir_successor.add(dep)
            dep_op = ops[dep]
            if (
                isinstance(dep_op, TransferOp)
                and dep_op.src_device == dep_op.dst_device
            ):
                pending.append(dep)

    for uid, _rank in et_ids:
        _mark(ops[uid].deps)
    for op in ops:
        if isinstance(op, TransferOp) and op.src_device != op.dst_device:
            _mark(op.deps)
    while pending:
        _mark(ops[pending.pop()].deps)

    wired: Dict[int, Set[int]] = {
        rank: {int(dep) for node in trace.nodes for dep in node.ctrl_deps}
        for rank, trace in traces.items()
    }
    lost: List[str] = []
    for (uid, rank), node_id in sorted(et_ids.items()):
        if uid not in has_ir_successor:
            continue
        if node_id in wired.get(rank, ()):
            continue
        op = ops[uid]
        lost.append(f"{op.name!r} (uid {uid}, rank {rank}, node {node_id})")
    if lost:
        raise EmissionError(
            f"{len(lost)} op(s) have successors in the Program but NONE in the "
            "emitted trace — an ordering constraint was dropped at emission, "
            "which no other gate can see (it moves only the AstraSim wall "
            f"clock). Examples: {', '.join(lost[:6])}"
        )

def _check_group_order_postcondition(
    group_records: Dict[str, Dict[int, List[Tuple[Any, int, int]]]],
    gid_members: Dict[str, List[int]],
    dp_count: int,
    ns_initial: int,
) -> None:
    for gid, per_rank in group_records.items():
        members = gid_members.get(gid)
        if members is None:
            stage_idx = int(gid) - 1
            members = [dp_idx * ns_initial + stage_idx for dp_idx in range(dp_count)]
        non_members = set(per_rank.keys()) - set(members)
        if non_members:
            raise EmissionError(
                f"group-order postcondition violated for communicator group "
                f"{gid} (members {members}): collectives were recorded on "
                f"non-member ranks {sorted(non_members)}. Ops are stamped "
                "with a pg_name of a communicator their rank does not "
                "belong to, which deadlocks AstraSim silently."
            )
        sequences: Dict[int, List[Tuple[int, int]]] = {}
        for rank in members:
            records = per_rank.get(rank, [])
            records_sorted = sorted(records, key=lambda item: int(item[0].id))
            sequences[rank] = [(ctype, size) for _node, ctype, size in records_sorted]
        reference_rank = members[0]
        reference = sequences[reference_rank]
        for rank in members[1:]:
            if sequences[rank] != reference:
                detail = "; ".join(
                    f"rank {r}: {sequences[r]}" for r in members
                )
                raise EmissionError(
                    f"group-order postcondition violated for communicator group "
                    f"{gid} (members {members}): member ranks would issue this "
                    f"group's collectives in different orders, which deadlocks "
                    f"AstraSim silently. Sequences: {detail}"
                )

def _write_manifest(output_dir: str, traces: Dict[int, _Trace], rank_ids: List[int]) -> str:
    def _manifest_op_key(op: List) -> tuple:
        try:
            kind = op[0]
        except Exception:
            kind = None
        if kind == "COMP":
            return (0, int(op[1]) if len(op) > 1 else 0, 0)
        if kind == "COMM":
            size_val = int(op[2]) if len(op) > 2 else 0
            ctype_val = int(op[1]) if len(op) > 1 else -1
            return (1, size_val, ctype_val)
        if kind == "SEND":
            return (2, int(op[1]) if len(op) > 1 else 0, 0)
        if kind == "RECV":
            return (3, int(op[1]) if len(op) > 1 else 0, 0)
        return (9, 0, 0)

    manifest_ranks: Dict[str, List[List]] = {}
    for rank, trace in sorted(traces.items()):
        rank_ops: List[List] = []
        for node in trace.nodes:
            t = int(node.type)
            if t == pb.COMP_NODE:
                rank_ops.append(["COMP", int(node.duration_micros or 0)])
            elif t == pb.COMM_COLL_NODE:
                ctype = None
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_type":
                        which = attr.WhichOneof("value")
                        ctype = int(getattr(attr, which)) if which else None
                    elif attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["COMM", int(ctype or -1), int(csize or 0), None])
            elif t == pb.COMM_SEND_NODE:
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["SEND", int(csize or 0)])
            elif t == pb.COMM_RECV_NODE:
                csize = None
                for attr in node.attr:
                    if attr.name == "comm_size":
                        which = attr.WhichOneof("value")
                        csize = int(getattr(attr, which)) if which else None
                rank_ops.append(["RECV", int(csize or 0)])
        manifest_ranks[str(int(rank))] = sorted(rank_ops, key=_manifest_op_key)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump(
            {
                "version": "df-astrasim-manifest/1",
                "npus": len(rank_ids),
                "ranks": manifest_ranks,
            },
            mf,
            sort_keys=True,
            separators=(",", ":"),
        )
    return manifest_path

def _write_comm_groups(
    output_dir: str,
    dp_count: int,
    rank_ids: List[int],
    gid_members: Dict[str, List[int]],
) -> Tuple[Dict[str, List[int]], Optional[str]]:
    dp = int(dp_count)
    if not rank_ids:
        return {}, None
    rank_ids = sorted(rank_ids)
    total_ranks = len(rank_ids)

    groups: Dict[str, List[int]] = {}
    if dp > 1:
        if total_ranks % dp != 0:
            raise ValueError(f"Cannot partition {total_ranks} ranks into {dp} stages evenly")
        num_stages = total_ranks // dp
        for stage_idx in range(num_stages):
            groups[str(stage_idx + 1)] = [
                dp_idx * num_stages + stage_idx for dp_idx in range(dp)
            ]

    for gid, members in gid_members.items():
        groups[str(gid)] = list(sorted(members))

    if not groups:
        return {}, None

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comm_groups.json")
    with open(path, "w") as fh:
        json.dump(groups, fh, indent=2)
    return groups, path

def _write_comm_axes(
    output_dir: str,
    comm_groups: Dict[str, List[int]],
    gid_axes: Dict[str, Any],
    gid_labels: Dict[str, Any],
) -> Optional[str]:
    if not comm_groups:
        return None
    entries: Dict[str, Dict[str, List[str]]] = {}
    for gid in sorted(comm_groups, key=lambda g: (len(g), g)):
        axes = sorted(str(a) for a in gid_axes.get(gid, ()))
        labels = sorted(str(la) for la in gid_labels.get(gid, ()))
        if not axes:
            axes = ["dp"]
        entries[str(gid)] = {"axes": axes, "labels": labels}
    path = os.path.join(output_dir, "comm_axes.json")
    with open(path, "w") as fh:
        json.dump(
            {"version": "df-astrasim-comm-axes/1", "groups": entries},
            fh,
            indent=2,
            sort_keys=True,
        )
    return path
