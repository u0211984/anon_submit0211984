from __future__ import annotations

import heapq
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from timing_model import CollectiveType

from program.groups import canonical_axis_label
from program.ir import (
    CollectiveOp,
    CommGroup,
    ComputeOp,
    GroupKey,
    Op,
    OpRole,
    Program,
    ProgramMeta,
    TransferOp,
)
from program.ir import Direction as IRDirection
from program.placement import (
    LEGACY_PLACEMENT,
    BlockExpander,
    CommStep,
    ComputeStep,
    ExpandedChain,
    Granularity,
    Placement,
    PlacementPolicy,
    StepKind,
)
from program.policies.gradaccum import GradAccumPolicy, grad_accum_policy_for
from program.policies.overlap import (
    AxisFractionOverlap,
    OverlapAnchor,
    OverlapDecl,
    OverlapPolicy,
)
from program.policies.recompute import RecomputePolicy
from program.policies.routing import MoERoutingPolicy, ep_sync_requirements
from program.policies.sharding import ShardingContext, ShardingPolicy
from program.schedule.policy import Schedule, SchedulePolicy
from program.types import DP_AXIS, CommKey, DeviceId
from program.work import (
    AttachMode,
    ByteSource,
    ByteSplit,
    DepClass,
    Direction,
    SyncAnchor,
    SyncKey,
    SyncPhase,
    SyncRequirement,
    WorkItem,
    WorkKind,
    WorkSet,
    enumerate_work,
)
from program.workload import FrozenWorkload

__all__ = [
    "BuildError",
    "SyncOrder",
    "build",
    "check_granularity_preconditions",
    "restrict_work_for",
]

def _pricing_axis(axes: Sequence[str]) -> Optional[str]:
    if not axes:
        return None
    axes = tuple(axes)
    cached = _PRICING_MEMO.get(axes)
    if cached is None:
        cached = DP_AXIS if DP_AXIS in axes else canonical_axis_label(axes)
        _PRICING_MEMO[axes] = cached
    return cached

_PRICING_MEMO: Dict[Tuple[str, ...], str] = {}

class BuildError(ValueError):
    pass

CROSS_LAYER_KEY: CommKey = "cross_layer"

_PHASE_RANK: Mapping[SyncPhase, int] = {
    SyncPhase.FWD_ENTRY: 0,
    SyncPhase.FWD: 1,
    SyncPhase.GRAD: 2,
    SyncPhase.BWD_ENTRY: 3,
    SyncPhase.BWD: 4,
}

@dataclass(frozen=True)
class SyncOrder:
    name: str = "phase_major"

    def key(self, req: SyncRequirement) -> Tuple[Any, ...]:
        key = req.key
        return (
            _PHASE_RANK[key.phase],
            -1 if key.microbatch is None else int(key.microbatch),
            -1 if key.layer is None else int(key.layer),
            str(key.comm_key),
        )

_OrderKey = Tuple[int, int, int, int, int]

_PHASE_CHAIN = 0
_PHASE_XFER = 1
_PHASE_SYNC = 2

_BLOCK_KINDS: Tuple[WorkKind, ...] = (WorkKind.LAYER, WorkKind.RECOMPUTE)

_R5_GRADIENT_KINDS: Tuple[WorkKind, ...] = (
    WorkKind.LAYER,
    WorkKind.EMBEDDING,
    WorkKind.SOFTMAX,
)

def restrict_work_for(
    granularity: Granularity,
    work: WorkSet,
    *,
    directions: Optional[Sequence[Direction]] = None,
) -> WorkSet:
    items = work.items
    if granularity is Granularity.BLOCK:
        items = tuple(item for item in items if item.kind in _BLOCK_KINDS)
    if directions is not None:
        allowed = tuple(directions)
        items = tuple(item for item in items if item.direction in allowed)
    if items is work.items:
        return work
    return WorkSet(items=items)

def check_granularity_preconditions(granularity: Granularity, fw: FrozenWorkload) -> None:
    if granularity is not Granularity.BLOCK:
        return
    degrees = fw.spec.degrees
    if int(degrees.dp) != 1 or int(degrees.pp) != 1:
        raise BuildError(
            "Granularity.BLOCK requires a single-replica, single-stage workload "
            f"(got dp={degrees.dp}, pp={degrees.pp}). BLOCK's device space is the "
            "(tp,cp,ep) sublayout, which carries no dp and no pp axis; a block "
            "run is a single-replica measurement (legacy dp_override=1). Build "
            "the block spec with dp=pp=1, or use Granularity.FLAT."
        )

_ROLE_BY_KIND: Mapping[WorkKind, OpRole] = {
    WorkKind.EMBEDDING: OpRole.EMBEDDING,
    WorkKind.SOFTMAX: OpRole.SOFTMAX,
    WorkKind.LAYER: OpRole.TRANSFORMER_LAYER,
    WorkKind.RECOMPUTE: OpRole.TRANSFORMER_LAYER,
    WorkKind.OPTIMIZER: OpRole.OPTIMIZER,
}

def build(
    fw: FrozenWorkload,
    *,
    granularity: Granularity,
    sharding: ShardingPolicy,
    schedule_policy: SchedulePolicy,
    recompute: RecomputePolicy,
    routing: Optional[MoERoutingPolicy] = None,
    overlap: OverlapPolicy = AxisFractionOverlap(),
    grad_accum: Optional[GradAccumPolicy] = None,
    dp_count: Optional[int] = None,
    label: str = "",
    gmap_workdir: Optional[str] = None,
    placement_policy: PlacementPolicy = LEGACY_PLACEMENT,
    sync_order: SyncOrder = SyncOrder(),
    validate: bool = True,
    check_group_membership: bool = True,
    directions: Optional[Sequence[Direction]] = None,
) -> Program:
    builder = _Builder(
        fw=fw,
        granularity=granularity,
        sharding=sharding,
        schedule_policy=schedule_policy,
        recompute=recompute,
        routing=routing,
        overlap=overlap,
        grad_accum=grad_accum,
        dp_count=dp_count,
        label=label,
        gmap_workdir=gmap_workdir,
        placement_policy=placement_policy,
        sync_order=sync_order,
        directions=directions,
    )
    return builder.run(
        validate=validate, check_group_membership=check_group_membership
    )

class _Builder:
    def __init__(
        self,
        *,
        fw: FrozenWorkload,
        granularity: Granularity,
        sharding: ShardingPolicy,
        schedule_policy: SchedulePolicy,
        recompute: RecomputePolicy,
        routing: Optional[MoERoutingPolicy],
        overlap: OverlapPolicy,
        grad_accum: Optional[GradAccumPolicy],
        dp_count: Optional[int],
        label: str,
        gmap_workdir: Optional[str],
        placement_policy: PlacementPolicy,
        sync_order: SyncOrder,
        directions: Optional[Sequence[Direction]] = None,
    ) -> None:
        self._fw = fw
        self._granularity = granularity
        self._sharding = sharding
        self._schedule_policy = schedule_policy
        self._recompute = recompute
        self._routing = routing
        self._overlap = overlap
        self._grad_accum_in = grad_accum
        self._dp_count_in = dp_count
        self._label = label
        self._gmap_workdir = gmap_workdir
        self._placement_policy = placement_policy
        self._sync_order = sync_order
        self._directions = directions

        self._nodes: List[Op] = []
        self._orders: List[_OrderKey] = []
        self._succs: List[List[int]] = []
        self._entry_names: List[Optional[str]] = []

        self._cross_slots_fwd: List[int] = []
        self._cross_slots_bwd: List[int] = []
        self._cross_slots_other: List[int] = []

        self._edges: Set[Tuple[int, int]] = set()
        self._edge_classes: Dict[Tuple[int, int], Set[DepClass]] = {}

        self._chains: Dict[WorkItem, Tuple[ExpandedChain, ...]] = {}
        self._chain_nodes: Dict[Tuple[WorkItem, int], List[int]] = {}
        self._chain_by_device: Dict[Tuple[WorkItem, int], ExpandedChain] = {}

        self._sync_nodes: Dict[SyncKey, List[int]] = {}

        self._reducer_keys: Set[SyncKey] = set()

        self._after_update_keys: Set[SyncKey] = set()

        self._r5_edges: List[Tuple[int, int]] = []

        self._r4_edges: List[Tuple[int, int]] = []

        self._edge_recorder: Optional[List[Tuple[int, int]]] = None

        self._overlap_sites: List[Tuple[int, OverlapDecl, Optional[int]]] = []
        self._xfer_seq = 0

        self._cross_transfers: List[Tuple[int, WorkItem, int]] = []
        self._cross_layer_bytes: Optional[float] = None
        self._dropped_requirements: List[str] = []

    def run(self, *, validate: bool, check_group_membership: bool = False) -> Program:
        fw = self._fw

        check_granularity_preconditions(self._granularity, fw)
        full_work = enumerate_work(fw, self._recompute)
        self._work: WorkSet = restrict_work_for(
            self._granularity, full_work, directions=self._directions
        )
        self._work_restricted = len(self._work) != len(full_work)
        self._schedule: Schedule = self._schedule_policy.schedule(fw, self._work)
        self._schedule.check_permutation(self._work)
        self._placement = Placement(
            fw, self._granularity, self._schedule.layers, policy=self._placement_policy
        )
        self._devices: Tuple[DeviceId, ...] = self._placement.devices()
        self._device_pos: Dict[int, int] = {
            int(device): index for index, device in enumerate(self._devices)
        }
        self._grad_accum = (
            self._grad_accum_in
            if self._grad_accum_in is not None
            else grad_accum_policy_for(fw)
        ).for_schedule(self._schedule)

        self._expand_all()
        self._apply_r2()

        self._apply_r3()

        if getattr(self._schedule_policy, "blocking_p2p", False):
            self._apply_blocking_p2p()

        self._edge_recorder = self._r4_edges
        try:
            self._apply_r4(self._collect_requirements())
        finally:
            self._edge_recorder = None

        self._apply_r5()

        self._realize_overlap()

        return self._finish(
            validate=validate, check_group_membership=check_group_membership
        )

    def _order(
        self, work: Optional[WorkItem], device: int, phase: int, a: int, b: int = 0
    ) -> _OrderKey:
        slot = -1 if work is None else self._schedule.index_of(work)
        position = self._device_pos.get(int(device))
        if position is None:
            raise BuildError(
                f"device {device} is not one of placement.devices() {self._devices}"
            )
        return (slot, position, phase, a, b)

    def _add_op(
        self, op: Op, order: _OrderKey, entry_name: Optional[str] = None
    ) -> int:
        self._nodes.append(op)
        self._orders.append(order)
        self._succs.append([])
        self._entry_names.append(entry_name)
        return op.uid

    def _next_nid(self) -> int:
        return len(self._nodes)

    def _add_dep(self, dep: int, node: int, dep_class: DepClass) -> None:
        if dep == node:
            raise BuildError(f"self-dependency on node {dep}")
        key = (dep, node)
        if key not in self._edges:
            self._edges.add(key)
            if dep_class is not DepClass.DATA_FLOW:
                self._edge_classes[key] = {dep_class}
            self._nodes[node].deps.append(dep)
            self._succs[dep].append(node)
            if self._edge_recorder is not None:
                self._edge_recorder.append(key)
        else:
            classes = self._edge_classes.get(key)
            if classes is not None:
                classes.add(dep_class)
            elif dep_class is not DepClass.DATA_FLOW:
                self._edge_classes[key] = {DepClass.DATA_FLOW, dep_class}

    def _classes(self, dep: int, node: int) -> Set[DepClass]:
        key = (dep, node)
        classes = self._edge_classes.get(key)
        if classes is not None:
            return set(classes)
        return {DepClass.DATA_FLOW}

    def _reparent_dep(self, dep: int, node: int, *, like: Tuple[int, int]) -> None:
        for dep_class in self._classes(*like):
            self._add_dep(dep, node, dep_class)

    def _drop_dep(self, dep: int, node: int) -> None:
        key = (dep, node)
        if key not in self._edges:
            return
        self._edges.discard(key)
        self._edge_classes.pop(key, None)
        self._nodes[node].deps.remove(dep)
        self._succs[dep].remove(node)

    def _reaches(self, source: int, target: int, *, slot_floor: Optional[int] = None) -> bool:
        if source == target:
            return True
        nodes = self._nodes
        orders = self._orders
        stack = [target]
        seen: Set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for dep in nodes[current].deps:
                if dep == source:
                    return True
                if dep in seen:
                    continue
                if slot_floor is not None and orders[dep][0] < slot_floor:
                    continue
                stack.append(dep)
        return False

    def _reaches_forward(self, source: int, target: int) -> bool:
        if source == target:
            return True
        succs = self._succs
        stack = [source]
        seen: Set[int] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            for succ in succs[current]:
                if succ == target:
                    return True
                if succ not in seen:
                    stack.append(succ)
        return False

    def _expand_all(self) -> None:
        expander = BlockExpander(
            self._fw, self._placement, overlap=self._overlap, routing=self._routing
        )
        for slot in self._schedule.ordered_slots:
            work = slot.work
            chains = expander.expand(work)
            self._chains[work] = chains
            for chain in chains:
                device = int(chain.device)
                if (work, device) in self._chain_nodes:
                    raise BuildError(
                        f"{work!r} expanded twice onto device {device}; "
                        "Placement.devices_for must be injective on "
                        "(WorkItem, cluster_rank) (P5)"
                    )
                self._chain_by_device[(work, device)] = chain
                self._chain_nodes[(work, device)] = self._materialize_chain(work, chain)

    def _materialize_chain(self, work: WorkItem, chain: ExpandedChain) -> List[int]:
        device = int(chain.device)
        slot = self._schedule.index_of(work)
        position = self._device_pos.get(device)
        if position is None:
            raise BuildError(
                f"device {device} is not one of placement.devices() {self._devices}"
            )
        direction = (
            IRDirection.FORWARD
            if work.direction is Direction.FORWARD
            else IRDirection.BACKWARD
        )
        kind_role = _ROLE_BY_KIND[work.kind]
        is_moe = bool(
            work.layer is not None and self._fw.spec.is_moe_layer(work.layer)
        )

        suffix = chain.name_suffix
        nodes = self._nodes
        orders = self._orders
        succs = self._succs
        entry_names = self._entry_names
        microbatch = work.microbatch
        layer = work.layer
        compute_new = ComputeOp.__new__
        nids: List[int] = []
        for step in chain.steps:
            order = (slot, position, _PHASE_CHAIN, step.index, 0)
            if isinstance(step, ComputeStep):
                entry_name = step.entry_name
                role = (
                    OpRole.JOIN
                    if step.kind is StepKind.JOIN
                    else (OpRole.GEMM if entry_name is not None else kind_role)
                )
                nid = len(nodes)

                op = compute_new(ComputeOp)
                op.uid = nid
                op.name = step.name + suffix if entry_name is not None else step.name
                op.device = device
                op.duration = (step.duration,)
                op.deps = []
                op.role = role
                op.direction = direction
                op.mem_kind = step.mem_kind
                op.recompute = step.recompute
                op.param_gather = step.param_gather
                op.micro_batch = microbatch
                op.layer = layer
                op.is_moe_layer = is_moe
                op.work = work
                op.activity = step.activity
                op.invocation_duration = step.invocation_duration
                op.invocation_activity = step.invocation_activity
                op.split_head = False
                op.embedded_collectives = getattr(step, "embedded_collectives", ())
                op.launch_devices = getattr(step, "launch_devices", ())
                op.launch_shadow = getattr(step, "launch_shadow", False)
                nodes.append(op)
                orders.append(order)
                succs.append([])
                entry_names.append(entry_name)
                nids.append(nid)
            elif isinstance(step, CommStep):
                nids.append(self._add_comm_step(work, chain, step, order, nids))
            else:  # pragma: no cover - ChainStep is a closed union
                raise BuildError(f"Unsupported chain step {step!r}")

        edges = self._edges
        recorder = self._edge_recorder
        for step, nid in zip(chain.steps, nids):
            node_deps = nodes[nid].deps
            for local in step.deps:
                dep_nid = nids[local]
                key = (dep_nid, nid)
                if key not in edges:
                    edges.add(key)
                    node_deps.append(dep_nid)
                    succs[dep_nid].append(nid)
                    if recorder is not None:  # pragma: no cover - R4-only
                        recorder.append(key)

                dep_node = nodes[dep_nid]
                if isinstance(dep_node, TransferOp) and nid not in dep_node.consumers:
                    dep_node.consumers.append(nid)

        for step, nid in zip(chain.steps, nids):
            if not isinstance(step, CommStep):
                continue
            for ref in step.extra_consumers:
                target = self._chain_step_nid(work, int(ref.device), ref.index)
                self._link_extra_consumer(nid, target)
        return nids

    def _chain_step_nid(self, work: WorkItem, device: int, index: int) -> int:
        nids = self._chain_nodes.get((work, device))
        if nids is None:
            raise BuildError(
                f"{work!r} has no expansion on device {device}; a StepRef must "
                "point at a sibling chain of the same expansion (P4: the hot "
                "rank is min(members) and expands first)"
            )
        return nids[index]

    def _link_extra_consumer(self, nid: int, target: int) -> None:
        node = self._nodes[nid]
        if isinstance(node, TransferOp):
            node.consumers.append(target)
            src_device = int(node.src_device)
        else:
            src_device = int(node.device)
        if src_device != int(self._nodes[target].device):
            self._cross_slots_other.append(self._orders[nid][0])
        self._add_dep(nid, target, DepClass.DATA_FLOW)

    def _add_comm_step(
        self,
        work: WorkItem,
        chain: ExpandedChain,
        step: CommStep,
        order: _OrderKey,
        prefix: Sequence[int],
    ) -> int:
        spec = step.spec
        device = int(chain.device)
        if spec.kind is CollectiveType.PIPELINE:
            dst = device
            for ref in step.extra_consumers:
                dst = int(ref.device)
                break
            producer = prefix[step.deps[0]] if step.deps else -1
            if producer < 0:
                raise BuildError(
                    f"block-template transfer {spec.key!r} has no producer step"
                )
            nid = self._add_op(
                TransferOp(
                    uid=self._next_nid(),

                    name=f"{spec.key}_rank{device}_to_rank{dst}",
                    src_device=device,
                    dst_device=dst,

                    size_bytes=int(spec.size_bytes),
                    comm_type=spec.kind,
                    producer=producer,
                    deps=[],
                    consumers=[],
                    moe_component=spec.moe_component,
                    participants=int(spec.participants),
                    interconnect=canonical_axis_label(spec.axes) if spec.axes else None,
                ),
                order,
            )
            if dst != device:
                self._cross_slots_other.append(order[0])
            self._add_dep(producer, nid, DepClass.DATA_FLOW)
            return nid

        is_dp = "dp" in tuple(spec.axes)
        group = (
            None
            if is_dp
            else self._placement.communicators.group_for(spec.axes, DeviceId(device))
        )
        axes = tuple(spec.axes)
        nid = self._add_op(
            CollectiveOp(
                uid=self._next_nid(),
                name=spec.key,
                device=device,
                coll=spec.kind,
                size_bytes=float(spec.size_bytes),
                participants=int(spec.participants),
                interconnect=_pricing_axis(axes),
                is_dp=is_dp,
                label=None,
                group=group,
                deps=[],
                comm_key=spec.key,
                axes=axes,
                work=work,
            ),
            order,
        )
        if step.overlap is not None:
            producer = None
            for local in step.deps:
                candidate = prefix[local]
                if isinstance(self._nodes[candidate], ComputeOp):
                    producer = candidate
            self._overlap_sites.append((nid, step.overlap, producer))
        return nid

    def _cross_layer_size(self) -> float:
        if self._cross_layer_bytes is None:
            if CROSS_LAYER_KEY not in self._fw.spec.comm:
                raise BuildError(
                    f"Comm key {CROSS_LAYER_KEY!r} is required for a cross-device "
                    "data-flow edge but is not declared in WorkloadSpec.comm "
                    f"(declared: {sorted(self._fw.spec.comm)})"
                )
            shards = self._placement.activation_shard_size()
            if self._placement.granularity is Granularity.PIPELINE:
                shards = max(1, int(self._fw.degrees.tp)) * max(1, int(self._fw.degrees.cp))
            self._cross_layer_bytes = ByteSource(
                key=CROSS_LAYER_KEY, split=ByteSplit.CEIL_DIV_CLUSTER
            ).bytes_for(self._fw, shards)
        return self._cross_layer_bytes

    def _cross_layer_spec(self):
        return self._fw.spec.comm.require(CROSS_LAYER_KEY)

    def _apply_r2(self) -> None:
        work = self._work
        shape = self._fw.spec.shape
        layers = int(shape.num_layers)
        for b in range(int(shape.micro_batches)):
            forward = [work.get(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b)]
            forward.extend(
                work.get(WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer)
                for layer in range(layers)
            )
            forward.append(work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))
            sequence = [item for item in forward if item is not None]
            for producer, consumer in zip(sequence, sequence[1:]):
                self._link(producer, consumer, transfer=True)

            if not self._fw.spec.run.include_backward:
                continue

            previous = work.get(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b)
            softmax_f = work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b)
            if softmax_f is not None and previous is not None:
                self._link(softmax_f, previous, transfer=False)
            for layer in reversed(range(layers)):
                backward = work.get(
                    WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=layer
                )
                if backward is None:
                    continue
                remat = work.get(
                    WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=b, layer=layer
                )
                entry = remat if remat is not None else backward
                if previous is not None:
                    self._link(previous, entry, transfer=True)
                if remat is not None:
                    layer_f = work.get(
                        WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer
                    )
                    if layer_f is not None:
                        self._link(layer_f, remat, transfer=False)
                    self._link(remat, backward, transfer=False)
                previous = backward
            embedding_b = work.get(
                WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b
            )
            if previous is not None and embedding_b is not None:
                self._link(previous, embedding_b, transfer=True)

    def _link(self, producer: WorkItem, consumer: WorkItem, *, transfer: bool) -> None:
        producers = self._chains[producer]
        consumers = self._chains[consumer]

        size = (
            0.0
            if self._placement.same_stage(producer, consumer)
            else self._cross_layer_size()
        )
        for prod, cons in _pair(producers, consumers, producer, consumer):
            src = int(prod.device)
            dst = int(cons.device)
            producer_nid = self._chain_nodes[(producer, src)][prod.exit]
            consumer_nid = self._chain_nodes[(consumer, dst)][cons.entry]
            if not transfer:
                if src != dst:
                    raise BuildError(
                        f"{producer!r} -> {consumer!r} is a same-device data-flow "
                        f"link but the chains are on {src} and {dst}"
                    )
                self._add_dep(producer_nid, consumer_nid, DepClass.DATA_FLOW)
                continue
            self._emit_cross_layer(producer, prod, producer_nid, cons, consumer_nid, size)

    def _emit_cross_layer(
        self,
        producer: WorkItem,
        prod: ExpandedChain,
        producer_nid: int,
        cons: ExpandedChain,
        consumer_nid: int,
        size: float,
    ) -> None:
        src = int(prod.device)
        dst = int(cons.device)
        order = self._order(producer, src, _PHASE_XFER, self._xfer_seq)
        self._xfer_seq += 1
        spec = self._cross_layer_spec()
        nid = self._add_op(
            TransferOp(
                uid=self._next_nid(),
                name=CROSS_LAYER_KEY,
                src_device=src,
                dst_device=dst,
                size_bytes=int(size),
                comm_type=CollectiveType.PIPELINE,
                producer=producer_nid,
                deps=[],
                consumers=[consumer_nid],
                participants=int(spec.participants),
                interconnect=canonical_axis_label(spec.axes) if spec.axes else None,
            ),
            order,
        )
        if src != dst:
            self._cross_transfers.append((nid, producer, src))

            (
                self._cross_slots_fwd
                if producer.direction is Direction.FORWARD
                else self._cross_slots_bwd
            ).append(order[0])
        self._add_dep(producer_nid, nid, DepClass.DATA_FLOW)

        anchor = self._nearest_compute(prod, producer_nid)
        if anchor is not None and anchor != producer_nid:
            self._add_dep(anchor, nid, DepClass.DATA_FLOW)
        self._add_dep(nid, consumer_nid, DepClass.DATA_FLOW)

    def _nearest_compute(self, chain: ExpandedChain, exit_nid: int) -> Optional[int]:
        if isinstance(self._nodes[exit_nid], ComputeOp):
            return None
        nids = self._chain_nodes[(chain.work, int(chain.device))]
        for index in range(chain.exit, -1, -1):
            if isinstance(self._nodes[nids[index]], ComputeOp):
                return nids[index]
        return None

    def _apply_r3(self) -> None:
        nodes = self._nodes
        orders = self._orders

        self._cross_slots_fwd.sort()
        self._cross_slots_bwd.sort()
        self._cross_slots_other.sort()
        for (dep_nid, node_nid) in self._edges:
            if orders[dep_nid][0] > orders[node_nid][0]:
                raise BuildError(
                    "D2 violated before R3: edge "
                    f"{nodes[dep_nid].name!r} (slot {orders[dep_nid][0]}) -> "
                    f"{nodes[node_nid].name!r} (slot {orders[node_nid][0]}) "
                    "runs backwards in schedule slots. R3's reachability bound "
                    "assumes the R1+R2 graph is slot-monotone (see _reaches)."
                )
        for dep in self._schedule.implied_deps(self._placement.devices_for):
            device = int(dep.device)
            before = self._chain_nodes.get((dep.before, device))
            after = self._chain_nodes.get((dep.after, device))
            if before is None or after is None:  # pragma: no cover - projection
                raise BuildError(
                    f"ScheduleDep {dep.before!r} -> {dep.after!r} names device "
                    f"{device}, which is not where both are placed"
                )
            source = before[self._chain_by_device[(dep.before, device)].exit]
            target = after[self._chain_by_device[(dep.after, device)].entry]
            floor = orders[source][0]
            if orders[target][0] < floor:  # pragma: no cover - see D2
                raise BuildError(
                    "D2 violated by an implied_deps pair: "
                    f"{dep.before!r} (slot {floor}) is scheduled AFTER "
                    f"{dep.after!r} (slot {orders[target][0]}) on device "
                    f"{device}; Schedule.implied_deps must return a monotone "
                    "slot sequence."
                )
            if self._r3_implied(source, target, floor):
                continue
            self._add_dep(source, target, DepClass.SCHEDULE)

    def _apply_blocking_p2p(self) -> None:
        projection = self._schedule.device_projection(self._placement.devices_for)
        next_item: Dict[Tuple[WorkItem, int], WorkItem] = {}
        for device, items in projection.items():
            for before, after in zip(items, items[1:]):
                next_item[(before, int(device))] = after
        for nid, producer, src in self._cross_transfers:
            node = self._nodes[nid]
            if int(getattr(node, "size_bytes", 0) or 0) <= 0:
                continue
            following = next_item.get((producer, int(src)))
            if following is None:
                continue
            chain = self._chain_by_device.get((following, int(src)))
            if chain is None:
                continue
            target = self._chain_nodes[(following, int(src))][chain.entry]
            if self._orders[target][0] < self._orders[nid][0]:  # pragma: no cover - D2
                raise BuildError("blocking p2p edge would run backwards in schedule slots")
            self._add_dep(nid, target, DepClass.SCHEDULE)
            if target not in node.consumers:
                node.consumers.append(target)

    def _r3_implied(self, source: int, target: int, floor: int) -> bool:
        nodes = self._nodes
        orders = self._orders
        if source in nodes[target].deps:
            return True

        ceiling = orders[target][0]
        lo_f = bisect_left(self._cross_slots_fwd, floor)
        hi_f = bisect_right(self._cross_slots_fwd, ceiling)
        lo_b = bisect_left(self._cross_slots_bwd, floor)
        hi_b = bisect_right(self._cross_slots_bwd, ceiling)
        lo_o = bisect_left(self._cross_slots_other, floor)
        hi_o = bisect_right(self._cross_slots_other, ceiling)
        has_fwd = hi_f > lo_f
        has_bwd = hi_b > lo_b
        if not (hi_o > lo_o) and not (has_fwd and has_bwd):
            device = nodes[target].device
            stack = [target]
            seen: Set[int] = set()
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                for dep in nodes[current].deps:
                    if dep == source:
                        return True
                    if dep in seen:
                        continue
                    if orders[dep][0] < floor:
                        continue
                    dep_node = nodes[dep]
                    dep_device = (
                        dep_node.src_device
                        if isinstance(dep_node, TransferOp)
                        else dep_node.device
                    )
                    if dep_device != device:
                        continue
                    stack.append(dep)
            return False

        floor_slot = floor
        budget = 96
        stack = [target]
        seen: Set[int] = set()
        decisive = True
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if len(seen) > budget:
                decisive = False
                break
            for dep in nodes[current].deps:
                if dep == source:
                    return True
                if dep in seen:
                    continue
                if orders[dep][0] < floor_slot:
                    continue
                stack.append(dep)
        if decisive:
            return False

        succs = self._succs
        stack = [source]
        fseen: Set[int] = set()
        while stack:
            current = stack.pop()
            if current in fseen:
                continue
            fseen.add(current)
            for succ in succs[current]:
                if succ == target:
                    return True
                if succ in fseen:
                    continue
                if orders[succ][0] > ceiling:
                    continue
                stack.append(succ)
        return False

    def _collect_requirements(self) -> Tuple[SyncRequirement, ...]:
        ctx = ShardingContext(
            fw=self._fw,
            work=self._work,
            grad_accum=self._grad_accum,
            stages=self._schedule.layers,

            order=self._schedule.order(),
        )
        out: List[SyncRequirement] = list(self._sharding.workload_requirements(ctx))
        for item in self._work:
            out.extend(self._sharding.requirements(item, ctx))
            if self._routing is not None:
                out.extend(ep_sync_requirements(item, ctx))

        out.sort(key=self._sync_order.key)
        return tuple(out)

    def _apply_r4(self, requirements: Sequence[SyncRequirement]) -> None:
        resolved: Set[SyncKey] = set()
        for rank, req in enumerate(requirements):
            if req.mode is AttachMode.AFTER:
                for anchor in req.anchors:
                    if isinstance(anchor, SyncKey) and anchor not in resolved:
                        raise BuildError(
                            f"SyncRequirement {req.key!r} is AFTER({anchor!r}), which "
                            "is not strictly earlier in SyncOrder"
                        )
            devices = self._placement.devices_for_sync(req)
            groups = self._placement.communicators.groups_for(req, self._placement)
            if len(groups) != len(devices):  # pragma: no cover - defensive
                raise BuildError(
                    f"groups_for({req.key!r}) returned {len(groups)} entries for "
                    f"{len(devices)} instance devices"
                )
            size = req.bytes.bytes_for(self._fw, len(devices))
            instances: List[int] = []
            for instance, (device, group) in enumerate(zip(devices, groups)):
                attached = self._attach(req, rank, instance, int(device), group, size)
                if attached is not None:
                    instances.append(attached)
            if instances:
                self._sync_nodes[req.key] = instances
                if req.is_reducer:
                    self._reducer_keys.add(req.key)
                if req.after_update:
                    self._after_update_keys.add(req.key)
                resolved.add(req.key)
            else:
                self._dropped_requirements.append(f"{req.key!r} ({req.origin})")

    def _attach(
        self,
        req: SyncRequirement,
        rank: int,
        instance: int,
        device: int,
        group: Optional[GroupKey],
        size: float,
    ) -> Optional[int]:
        anchors = [self._resolve_anchor(anchor, device) for anchor in req.anchors]
        anchors = [pair for pair in anchors if pair[0] or pair[1]]
        if not anchors:
            return None

        axes = tuple(req.axes)
        nid = self._add_op(
            CollectiveOp(
                uid=self._next_nid(),
                name=self._sync_name(req, instance),
                device=device,
                coll=req.kind,
                size_bytes=float(size),
                participants=int(req.participants),
                interconnect=_pricing_axis(axes),
                is_dp=bool(req.is_dp),
                label=None,
                group=group,
                deps=[],
                comm_key=req.comm_key,
                axes=axes,
                work=req.place_on,
            ),
            self._order(req.place_on, device, _PHASE_SYNC, rank, instance),
        )
        mode = req.mode
        for entries, exits in anchors:
            if mode is AttachMode.BEFORE:
                for entry in entries:
                    for dep in list(self._nodes[entry].deps):
                        self._add_dep(dep, nid, DepClass.SYNC)
                        dep_node = self._nodes[dep]
                        if (
                            isinstance(dep_node, TransferOp)
                            and int(dep_node.dst_device) == int(device)
                            and nid not in dep_node.consumers
                        ):
                            dep_node.consumers.append(nid)
                    self._add_dep(nid, entry, DepClass.SYNC)
            elif mode is AttachMode.AFTER:
                for exit_nid in exits:
                    self._add_dep(exit_nid, nid, DepClass.SYNC)
            elif mode is AttachMode.PARALLEL_TO:
                for entry in entries:
                    for dep in list(self._nodes[entry].deps):
                        self._add_dep(dep, nid, DepClass.SYNC)
                        dep_node = self._nodes[dep]
                        if (
                            isinstance(dep_node, TransferOp)
                            and int(dep_node.dst_device) == int(device)
                            and nid not in dep_node.consumers
                        ):
                            dep_node.consumers.append(nid)
                for exit_nid in exits:
                    for succ in list(self._succs[exit_nid]):
                        if succ == nid:
                            continue
                        if self._classes(exit_nid, succ) & req.via:
                            self._add_dep(nid, succ, DepClass.SYNC)
            elif mode is AttachMode.OVERLAP_WITH:  # pragma: no cover - unused
                raise BuildError(
                    "AttachMode.OVERLAP_WITH on a SyncRequirement is realized by "
                    "the OverlapDecl (§4.5), not by an attach; no production "
                    "policy declares it"
                )
            else:  # pragma: no cover - AttachMode is closed
                from typing import assert_never

                assert_never(mode)

        for consumer in req.consumers:
            consumer_entries, _consumer_exits = self._resolve_anchor(consumer, device)
            if not consumer_entries:
                raise BuildError(
                    f"SyncRequirement {req.key!r} ({req.origin}) declares consumer "
                    f"{consumer!r}, which has no chain reachable from device {device}"
                )
            for entry in consumer_entries:
                if entry == nid:  # pragma: no cover - defensive
                    continue
                self._add_dep(nid, entry, DepClass.SYNC)
        return nid

    def _sync_name(self, req: SyncRequirement, instance: int) -> str:
        key = req.key
        parts = [str(key.comm_key), key.phase.name.lower()]
        if key.microbatch is not None:
            parts.append(f"b{key.microbatch}")
        if key.layer is not None:
            parts.append(f"l{key.layer}")
        if instance:
            parts.append(f"rank{instance}")
        return "_".join(parts)

    def _resolve_anchor(
        self, anchor: SyncAnchor, device: int
    ) -> Tuple[List[int], List[int]]:
        if isinstance(anchor, SyncKey):
            nodes = self._sync_nodes.get(anchor)
            if not nodes:
                return ([], [])
            local = [nid for nid in nodes if self._nodes[nid].device == device]
            chosen = local if local else list(nodes)
            return (chosen, list(chosen))
        chains = self._chains.get(anchor)
        if not chains:
            return ([], [])
        local_chains = [chain for chain in chains if int(chain.device) == device]
        chosen_chains = local_chains if local_chains else list(chains)
        entries: List[int] = []
        exits: List[int] = []
        for chain in chosen_chains:
            nids = self._chain_nodes[(anchor, int(chain.device))]
            entries.append(nids[chain.entry])
            exits.append(nids[chain.exit])
        return (entries, exits)

    def _apply_r5(self) -> None:
        if not self._fw.spec.run.include_backward:
            return
        optimizers = [
            item for item in self._work if item.kind is WorkKind.OPTIMIZER
        ]
        if not optimizers:
            return
        producers = [
            item
            for item in self._work
            if item.direction is Direction.BACKWARD
            and item.kind in _R5_GRADIENT_KINDS
        ]

        reducers: Dict[Tuple[int, int], List[int]] = {}
        after_update: Dict[Tuple[int, int], List[int]] = {}
        for key, nids in self._sync_nodes.items():
            if key in self._reducer_keys:
                bucket = reducers
            elif key in self._after_update_keys:
                bucket = after_update
            else:
                continue
            for nid in nids:
                node = self._nodes[nid]
                work = getattr(node, "work", None)
                if work is None:
                    continue
                bucket.setdefault(
                    (int(self._schedule.stage_of(work)), int(node.device)), []
                ).append(nid)

        producers_by_stage: Dict[int, List[WorkItem]] = {}
        for item in sorted(producers, key=WorkItem.sort_key):
            producers_by_stage.setdefault(
                int(self._schedule.stage_of(item)), []
            ).append(item)
        for optimizer in sorted(optimizers, key=WorkItem.sort_key):
            stage = int(self._schedule.stage_of(optimizer))
            for chain in self._chains.get(optimizer, ()):
                device = int(chain.device)
                target = self._chain_nodes[(optimizer, device)][chain.entry]

                target_slot = self._orders[target][0]
                r5_sources: List[Tuple[WorkItem, int]] = []
                for item in producers_by_stage.get(stage, ()):
                    entry = self._chain_by_device.get((item, device))
                    if entry is None:
                        continue
                    r5_sources.append(
                        (item, self._chain_nodes[(item, device)][entry.exit])
                    )
                r5b_sources = sorted(reducers.get((stage, device), ()))
                for _item, source in r5_sources:
                    if source == target:
                        continue
                    if self._orders[source][0] < target_slot:
                        continue
                    if self._reaches_forward(source, target):
                        continue
                    self._add_dep(source, target, DepClass.DATA_FLOW)
                    self._r5_edges.append((source, target))

                for source in r5b_sources:
                    if source == target:
                        continue
                    if self._reaches_forward(source, target):
                        continue
                    if self._reaches_forward(target, source):
                        raise BuildError(
                            f"R5b would close a cycle: OPTIMIZER node {target} "
                            f"already reaches its own gradient reducer {source} "
                            f"(stage {stage}, device {device})"
                        )
                    self._add_dep(source, target, DepClass.DATA_FLOW)
                    self._r5_edges.append((source, target))

                exit_nid = self._chain_nodes[(optimizer, device)][chain.exit]

                for sink in sorted(after_update.get((stage, device), ())):
                    if sink == exit_nid or self._reaches(exit_nid, sink):
                        continue
                    if self._reaches(sink, exit_nid):
                        raise BuildError(
                            f"R5c would close a cycle: post-update collective "
                            f"{sink} already reaches OPTIMIZER node {exit_nid} "
                            f"(stage {stage}, device {device})"
                        )
                    self._add_dep(exit_nid, sink, DepClass.DATA_FLOW)
                    self._r5_edges.append((exit_nid, sink))

    def _realize_overlap(self) -> None:
        producer_sites: Dict[int, List[int]] = {}
        producer_decl: Dict[int, OverlapDecl] = {}
        for nid, decl, producer in self._overlap_sites:
            if decl.anchor is OverlapAnchor.PRODUCER:
                if producer is None:
                    raise BuildError(
                        f"OverlapDecl(PRODUCER) on {self._nodes[nid].name!r} has no "
                        "producing ComputeStep in its chain"
                    )
                producer_sites.setdefault(producer, []).append(nid)
                producer_decl[producer] = decl
            else:
                self._split_collective(nid, decl)
        for producer in sorted(producer_sites):
            self._split_compute(producer, producer_sites[producer], producer_decl[producer])

    def _split_compute(
        self, compute: int, collectives: Sequence[int], decl: OverlapDecl
    ) -> None:
        node = self._nodes[compute]
        if not isinstance(node, ComputeOp):  # pragma: no cover - defensive
            raise BuildError("PRODUCER overlap must split a ComputeOp")
        duration = float(node.duration[0])
        if duration <= 0.0:
            return

        for coll in collectives:
            self._nodes[coll].launch_overlap_consumer = compute

        if decl.is_hoist:
            for coll in collectives:
                for dep in list(self._nodes[compute].deps):
                    self._reparent_dep(dep, coll, like=(dep, compute))
                self._drop_dep(compute, coll)
                for succ in list(self._succs[coll]):
                    self._reparent_dep(compute, succ, like=(coll, succ))
            return

        head_duration = duration * (1.0 - decl.fraction)
        tail_duration = duration * decl.fraction
        if head_duration <= 0.0 or tail_duration <= 0.0:
            return

        slot, device_index, _, a, _ = self._orders[compute]
        original_activity = node.activity
        head = self._add_op(
            ComputeOp(
                uid=self._next_nid(),
                name=f"{node.name}_head",
                device=node.device,
                duration=(head_duration,),
                deps=[],
                role=node.role,
                direction=node.direction,
                mem_kind=node.mem_kind,
                recompute=node.recompute,
                param_gather=node.param_gather,
                micro_batch=node.micro_batch,
                layer=node.layer,
                is_moe_layer=node.is_moe_layer,
                work=node.work,
                activity=original_activity.scaled(1.0 - decl.fraction),
                invocation_duration=(node.invocation_duration * (1.0 - decl.fraction)
                                     if node.invocation_duration is not None else None),
                invocation_activity=(node.invocation_activity.scaled(1.0 - decl.fraction)
                                     if node.invocation_activity is not None else None),

                split_head=True,
            ),
            (slot, device_index, _PHASE_CHAIN, a, -1),
            entry_name=self._entry_names[compute],
        )
        node.duration = (tail_duration,)
        node.activity = original_activity.scaled(decl.fraction)
        if node.invocation_duration is not None:
            node.invocation_duration *= decl.fraction
        if node.invocation_activity is not None:
            node.invocation_activity = node.invocation_activity.scaled(decl.fraction)
        for dep in list(node.deps):
            for dep_class in self._classes(dep, compute):
                self._add_dep(dep, head, dep_class)

            dep_node = self._nodes[dep]
            if isinstance(dep_node, TransferOp):
                dep_node.consumers = [
                    head if consumer == compute else consumer
                    for consumer in dep_node.consumers
                ]
            self._drop_dep(dep, compute)
        self._add_dep(head, compute, DepClass.DATA_FLOW)
        for coll in collectives:
            self._reparent_dep(head, coll, like=(compute, coll))
            self._drop_dep(compute, coll)
            for succ in list(self._succs[coll]):
                self._reparent_dep(compute, succ, like=(coll, succ))

    def _split_collective(self, coll: int, decl: OverlapDecl) -> None:
        node = self._nodes[coll]
        if not isinstance(node, CollectiveOp):  # pragma: no cover - defensive
            raise BuildError("CONSUMER overlap must split a CollectiveOp")
        blocking = [
            succ
            for succ in self._succs[coll]
            if self._is_blocking_consumer(succ, decl)
        ]
        if not blocking:
            return
        total = float(node.size_bytes)
        block_bytes = math.ceil(total * (1.0 - decl.fraction))
        ovlp_bytes = max(0.0, total - block_bytes)
        preds = list(self._nodes[coll].deps)
        succs = list(self._succs[coll])

        if decl.is_hoist or total <= 0 or block_bytes <= 0:
            node.launch_overlap_consumers = tuple(blocking)

            for consumer in blocking:
                for pred in preds:
                    self._reparent_dep(pred, consumer, like=(pred, coll))
                self._drop_dep(coll, consumer)
            for succ in succs:
                if succ in blocking:
                    continue
                for consumer in blocking:
                    self._reparent_dep(consumer, succ, like=(coll, succ))
            return

        node.size_bytes = float(block_bytes)
        node.name = f"{node.name}_block"
        ovlp: Optional[int] = None
        if ovlp_bytes > 0:
            slot, device_index, _, a, _ = self._orders[coll]
            ovlp = self._add_op(
                CollectiveOp(
                    uid=self._next_nid(),
                    name=f"{node.comm_key}_ovlp",
                    device=node.device,
                    coll=node.coll,
                    size_bytes=float(ovlp_bytes),
                    participants=node.participants,
                    interconnect=_pricing_axis(node.axes),
                    is_dp=node.is_dp,
                    label=None,
                    group=node.group,
                    deps=[],
                    comm_key=node.comm_key,
                    axes=node.axes,
                    work=node.work,
                    launch_continuation=True,
                ),
                (slot, device_index, _PHASE_CHAIN, a, 1),
            )
            self._add_dep(coll, ovlp, DepClass.DATA_FLOW)
        for succ in succs:
            if succ in blocking:
                continue
            carried = self._classes(coll, succ)
            self._drop_dep(coll, succ)
            for consumer in blocking:
                for dep_class in carried:
                    self._add_dep(consumer, succ, dep_class)
            for dep_class in carried:
                self._add_dep(coll if ovlp is None else ovlp, succ, dep_class)

    def _is_blocking_consumer(self, nid: int, decl: OverlapDecl) -> bool:
        node = self._nodes[nid]
        if not isinstance(node, ComputeOp):
            return False
        target = decl.blocking_consumer
        if target is None:  # pragma: no cover - OverlapDecl rejects it
            return False

        return target.lower() in str(self._entry_names[nid] or "").lower()

    def _finish(self, *, validate: bool, check_group_membership: bool = False) -> Program:
        uid_of = self._kahn()

        nodes = self._nodes
        ops: List[Op] = [None] * len(nodes)  # type: ignore[list-item]
        for nid, op in enumerate(nodes):
            uid = uid_of[nid]
            op.uid = uid
            op.deps = tuple([uid_of[dep] for dep in op.deps])
            if isinstance(op, CollectiveOp):
                if op.launch_overlap_consumer is not None:
                    op.launch_overlap_consumer = uid_of[op.launch_overlap_consumer]
                op.launch_overlap_consumers = tuple(uid_of[c] for c in op.launch_overlap_consumers)
            if isinstance(op, TransferOp):
                op.producer = uid_of[op.producer]
                op.consumers = tuple([uid_of[c] for c in op.consumers])
            ops[uid] = op
        groups: Dict[GroupKey, CommGroup] = {}
        labels = _LabelInterner()
        for op in ops:
            if isinstance(op, CollectiveOp) and op.group is not None:
                label = labels.label_for(op.comm_key or op.name, op.group)
                op.label = label
                groups.setdefault(op.group, CommGroup(key=op.group, label=label))

        devices = tuple(int(device) for device in self._devices)
        dp_count = (
            int(self._dp_count_in)
            if self._dp_count_in is not None
            else int(self._fw.spec.run.retime_dp_count(self._fw.spec.degrees))
        )
        meta = ProgramMeta(
            label=self._label,
            misc={
                "granularity": self._granularity.name.lower(),
                "program_order": "kahn(slot,device,intra)",
                "duration_revision": int(self._fw.durations.revision),
                **({"vtp_per_rank": True} if self._fw.spec.rank_activities else {}),
                "num_stages_initial": len(devices),
                "schedule_policy": self._schedule.policy,
                "sync_order": self._sync_order.name,
                "sharding_policy": self._sharding.name,
                "grad_accum_policy": self._grad_accum.name,
                "recompute_policy": self._recompute.name,
                "overlap_policy": self._overlap.name,
                "routing_policy": None if self._routing is None else self._routing.name,
                "placement_policy": self._placement_policy.name,
                "layer_assignment_contiguous": self._schedule.layers.contiguous_layers(),
                "work_restricted": self._work_restricted,
                "work_directions": (
                    None
                    if self._directions is None
                    else tuple(d.name for d in self._directions)
                ),

                "schedule_edges": tuple(
                    sorted(
                        (uid_of[source], uid_of[target])
                        for (source, target), classes in self._edge_classes.items()
                        if DepClass.SCHEDULE in classes
                    )
                ),

                "r5_edges": tuple(
                    sorted(
                        (uid_of[source], uid_of[target])
                        for (source, target) in self._r5_edges
                        if (source, target) in self._edges
                    )
                ),

                "r4_edges": tuple(
                    sorted(
                        (uid_of[source], uid_of[target])
                        for (source, target) in self._r4_edges
                        if (source, target) in self._edges
                    )
                ),
                "dropped_requirements": tuple(self._dropped_requirements),
                "gmap_workdir": self._gmap_workdir,
            },
        )
        program = Program(
            layout=self._placement.layout,
            dp_count=max(1, dp_count),
            devices=devices,
            ops=ops,
            groups=groups,
            meta=meta,
        )
        if validate:
            from program.validate import validate_program

            validate_program(
                program,
                check_races=True,
                check_group_membership=check_group_membership,
            )
        return program

    def _kahn(self) -> List[int]:
        nodes = self._nodes
        orders = self._orders
        succs_arr = self._succs
        by_rank: List[int] = sorted(
            range(len(nodes)), key=lambda nid: (orders[nid], nid)
        )
        rank_of: List[int] = [0] * len(nodes)
        for rank, nid in enumerate(by_rank):
            rank_of[nid] = rank
        indegree: List[int] = [len(node.deps) for node in nodes]
        heap: List[int] = [
            rank_of[nid] for nid in range(len(nodes)) if indegree[nid] == 0
        ]
        heapq.heapify(heap)
        uid_of: List[int] = [-1] * len(nodes)
        assigned = 0
        while heap:
            nid = by_rank[heapq.heappop(heap)]
            uid_of[nid] = assigned
            assigned += 1
            for succ in succs_arr[nid]:
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    heapq.heappush(heap, rank_of[succ])
        if assigned != len(nodes):
            stuck = [nid for nid in range(len(nodes)) if uid_of[nid] < 0][:8]
            raise BuildError(
                f"dependency cycle: {len(nodes) - assigned} ops never "
                f"became ready (e.g. {[nodes[n].name for n in stuck]})"
            )
        return uid_of

def _pair(
    producers: Sequence[ExpandedChain],
    consumers: Sequence[ExpandedChain],
    producer_work: WorkItem,
    consumer_work: WorkItem,
) -> Tuple[Tuple[ExpandedChain, ExpandedChain], ...]:
    if len(producers) == len(consumers):
        return tuple(zip(producers, consumers))
    if len(producers) == 1:
        return tuple((producers[0], cons) for cons in consumers)
    if len(consumers) == 1:
        return tuple((prod, consumers[0]) for prod in producers)
    raise BuildError(
        f"cannot pair {len(producers)} chains of {producer_work!r} with "
        f"{len(consumers)} chains of {consumer_work!r}"
    )

class _LabelInterner:
    def __init__(self) -> None:
        self._labels: Dict[Tuple[str, Tuple[int, ...]], str] = {}
        self._suffix: Dict[str, int] = {}

    def label_for(self, base: str, group: GroupKey) -> str:
        key = (str(base), tuple(group.members))
        label = self._labels.get(key)
        if label is None:
            count = self._suffix.get(str(base), 0)
            label = str(base) if count == 0 else f"{base}_{count}"
            self._suffix[str(base)] = count + 1
            self._labels[key] = label
        return label
