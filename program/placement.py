from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from memory_estimation import mem_kind_from_op_name
from activity import OperationActivity, ZERO_ACTIVITY

from program.ir import EmbeddedCollective
from program.block import BlockTemplate, GemmEntry
from program.groups import CommunicatorFactory, GroupError, canonical_axis_label
from program.axes import (
    CLUSTER_AXES as _REGISTRY_CLUSTER_AXES,
    PIPELINE_AXES,
    REPLICA_AXES,
    activation_sharding_axes,
    axis_sizes_from,
)
from program.layout import CANONICAL_AXES, RankLayout, cluster_coords
from program.types import DP_AXIS, AxisName, CommKey, Coords, DeviceId, LayerId, StageId
from program.work import (
    Direction,
    SyncSpread,
    WorkItem,
    WorkKind,
    activity_for,
    duration_for,
    duration_key,
    mem_kind,
)
from program.workload import CommSpec, CommSpecTable, FrozenWorkload

if TYPE_CHECKING:  # pragma: no cover - typing only
    from program.policies.overlap import OverlapDecl, OverlapPolicy
    from program.policies.routing import MoERoutingPolicy
    from program.schedule.policy import LayerAssignment
    from program.work import SyncRequirement

class PlacementError(ValueError):
    pass

class Granularity(Enum):
    PIPELINE = auto()
    FLAT = auto()
    BLOCK = auto()

_GRANULARITY_AXES: Mapping[Granularity, Tuple[AxisName, ...]] = MappingProxyType(
    {

        Granularity.PIPELINE: PIPELINE_AXES + REPLICA_AXES,
        Granularity.FLAT: CANONICAL_AXES,
        Granularity.BLOCK: _REGISTRY_CLUSTER_AXES,
    }
)

CLUSTER_AXES: Tuple[AxisName, ...] = _REGISTRY_CLUSTER_AXES

@dataclass(frozen=True, eq=False)
class DeviceCoord:
    coords: Coords

    def __post_init__(self) -> None:
        normalized = {str(axis): int(value) for axis, value in dict(self.coords).items()}
        object.__setattr__(self, "coords", MappingProxyType(normalized))

    def with_(self, **axes: int) -> "DeviceCoord":
        merged = dict(self.coords)
        merged.update({str(axis): int(value) for axis, value in axes.items()})
        return DeviceCoord(merged)

    def of(self, axis: AxisName) -> int:
        if axis not in self.coords:
            raise PlacementError(
                f"DeviceCoord has no coordinate for axis {axis!r} "
                f"(has {sorted(self.coords)})"
            )
        return int(self.coords[axis])

    def _key(self) -> Tuple[Tuple[str, int], ...]:
        return tuple(sorted(self.coords.items()))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, DeviceCoord) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        inner = ",".join(f"{axis}={value}" for axis, value in self._key())
        return f"DeviceCoord({inner})"

class WorkSpread(Enum):
    CLUSTER_RANK_0 = auto()
    PER_CLUSTER_RANK = auto()

@dataclass(frozen=True)
class PlacementPolicy:
    name: str
    spread_by_kind: Mapping[WorkKind, WorkSpread]

    def __post_init__(self) -> None:
        missing = [kind for kind in WorkKind if kind not in self.spread_by_kind]
        if missing:
            raise PlacementError(
                f"PlacementPolicy {self.name!r} does not declare a spread for "
                f"{[kind.name for kind in missing]}"
            )
        object.__setattr__(
            self, "spread_by_kind", MappingProxyType(dict(self.spread_by_kind))
        )

    def spread_for(self, kind: WorkKind) -> WorkSpread:
        return self.spread_by_kind[kind]

LEGACY_PLACEMENT = PlacementPolicy(
    name="legacy",
    spread_by_kind={
        WorkKind.EMBEDDING: WorkSpread.CLUSTER_RANK_0,
        WorkKind.SOFTMAX: WorkSpread.CLUSTER_RANK_0,
        WorkKind.LAYER: WorkSpread.PER_CLUSTER_RANK,
        WorkKind.RECOMPUTE: WorkSpread.PER_CLUSTER_RANK,
        WorkKind.OPTIMIZER: WorkSpread.PER_CLUSTER_RANK,
    },
)

def canonical_layout(axis_sizes: Mapping[AxisName, int]) -> RankLayout:
    sizes = {axis: max(1, int(axis_sizes.get(axis, 1))) for axis in CANONICAL_AXES}
    seed = RankLayout(
        axis_order=CANONICAL_AXES,
        axis_sizes=sizes,
        axis_strides={},
    )
    return seed.subset(CANONICAL_AXES)

class Placement:
    def __init__(
        self,
        fw: FrozenWorkload,
        granularity: Granularity,
        layers: "LayerAssignment",
        *,
        policy: PlacementPolicy = LEGACY_PLACEMENT,
    ) -> None:
        if not isinstance(granularity, Granularity):
            raise PlacementError(
                f"granularity must be a Granularity (got {type(granularity).__name__})"
            )
        degrees = fw.spec.degrees
        full = fw.spec.layout
        if full is None or not full.axis_order:
            full = canonical_layout(
                {
                    **axis_sizes_from(degrees),
                }
            )

        sizes = full.axis_sizes
        self._sizes: Dict[AxisName, int] = {}
        for axis in CANONICAL_AXES:
            self._sizes[axis] = (
                max(1, int(sizes[axis])) if axis in sizes else degrees.of(axis)
            )
        cluster = degrees.cluster_size()
        layout_cluster = 1
        for _axis in CLUSTER_AXES:
            layout_cluster *= max(1, int(self._sizes[_axis]))
        if layout_cluster != cluster:
            raise PlacementError(
                "Inconsistent tensor/context/expert parallel factors: layout says "
                + ", ".join(f"{a}={self._sizes[a]}" for a in CLUSTER_AXES) + " "
                f"(product {layout_cluster}) but the degrees say {cluster}"
            )

        self._fw = fw
        self._degrees = degrees
        self._granularity = granularity
        self._layers = layers
        self._policy = policy
        self._full_layout = full

        self._physical_pipeline = (
            granularity is Granularity.PIPELINE
            and fw.spec.run.include_backward and fw.spec.run.zero_stage >= 2
            and fw.spec.shape.has_moe_layers
        )
        axes = _GRANULARITY_AXES[granularity]
        self._layout = full if granularity is Granularity.FLAT or self._physical_pipeline else full.subset(axes)

        self._device_axes: Tuple[AxisName, ...] = tuple(
            axis for axis in self._layout.axis_order if axis != DP_AXIS
        )

        self._group_layout = self._layout.subset(self._device_axes)

        self._cluster_size = 1 if granularity is Granularity.PIPELINE and not self._physical_pipeline else cluster
        self._communicators = CommunicatorFactory(self._group_layout)

    @property
    def granularity(self) -> Granularity:
        return self._granularity

    @property
    def policy(self) -> PlacementPolicy:
        return self._policy

    @property
    def layers(self) -> "LayerAssignment":
        return self._layers

    @property
    def layout(self) -> RankLayout:
        return self._layout

    @property
    def full_layout(self) -> RankLayout:
        return self._full_layout

    @property
    def group_layout(self) -> RankLayout:
        return self._group_layout

    @property
    def communicators(self) -> CommunicatorFactory:
        return self._communicators

    def cluster_size(self) -> int:
        return self._cluster_size

    def activation_shard_size(self) -> int:
        if self._granularity is Granularity.PIPELINE and not self._physical_pipeline:
            return 1

        size = 1
        for _axis in activation_sharding_axes():
            size *= max(1, int(self._sizes[_axis]))
        return max(1, size)

    def num_stages(self) -> int:
        return int(self._degrees.pp)

    def devices(self) -> Tuple[DeviceId, ...]:
        if not self._device_axes:
            return (DeviceId(0),)
        ranges = [range(self._layout.axis_sizes[axis]) for axis in self._device_axes]
        found = set()
        for combo in itertools.product(*ranges):
            found.add(self._layout.linearize(dict(zip(self._device_axes, combo))))
        return tuple(DeviceId(device) for device in sorted(found))

    def coords_of(self, device: DeviceId) -> DeviceCoord:
        return DeviceCoord(self._layout.coords_of(int(device)))

    def device_of(self, coord: DeviceCoord) -> DeviceId:
        return DeviceId(self._layout.linearize(coord.coords))

    def device_for(self, stage: StageId, cluster_rank: int = 0) -> DeviceId:
        rank = int(cluster_rank)
        if rank < 0 or rank >= self._cluster_size:
            raise PlacementError(
                f"cluster_rank {rank} is out of range for cluster_size "
                f"{self._cluster_size}"
            )
        stage_id = int(stage)
        if stage_id < 0 or stage_id >= self.num_stages():
            raise PlacementError(
                f"stage {stage_id} is out of range for pp={self.num_stages()}"
            )
        coords = cluster_coords(
            self._layout.axis_order,
            rank,
            stage_id,
            sizes=self._sizes,
        )
        return DeviceId(self._layout.linearize(coords))

    def stage_device(self, stage: StageId) -> DeviceId:
        return self.device_for(stage, 0)

    def cluster_devices(self, stage: StageId) -> Tuple[DeviceId, ...]:
        return tuple(self.device_for(stage, rank) for rank in range(self._cluster_size))

    def cluster_rank_of(self, device: DeviceId) -> int:
        coords = self._layout.coords_of(int(device))
        for rank in range(self._cluster_size):
            candidate = cluster_coords(
                self._layout.axis_order,
                rank,
                int(coords.get("pp", 0)),
                sizes=self._sizes,
            )
            if all(coords.get(axis, 0) == value for axis, value in candidate.items()):
                return rank
        raise PlacementError(f"Device {device} is not a cluster rank of this layout")

    def stage_of(self, work: WorkItem) -> StageId:
        kind = work.kind
        if kind is WorkKind.EMBEDDING:
            return StageId(0)
        if kind is WorkKind.SOFTMAX:
            return StageId(self.num_stages() - 1)
        if kind is WorkKind.LAYER or kind is WorkKind.RECOMPUTE:
            if work.layer is None:
                raise PlacementError(f"{kind.name} WorkItem carries no layer: {work!r}")
            return StageId(int(self._layers.stage_of(work.layer)))
        if kind is WorkKind.OPTIMIZER:
            if work.stage is None:
                raise PlacementError(
                    "OPTIMIZER WorkItem carries no stage; its identity IS per-stage "
                    "(BUG_LEDGER 10b: one fused node per stage, priced over the "
                    "layers that stage owns)"
                )
            return StageId(int(work.stage))
        raise PlacementError(f"Unhandled WorkKind {kind!r}")

    def devices_for(self, work: WorkItem) -> Tuple[DeviceId, ...]:
        stage = self.stage_of(work)
        if (self._granularity is Granularity.FLAT and self._fw.spec.rank_activities
                and work.kind in (WorkKind.EMBEDDING, WorkKind.SOFTMAX)):
            return self.cluster_devices(stage)

        if (work.kind in (WorkKind.EMBEDDING, WorkKind.SOFTMAX)
                and self._fw.spec.run.include_backward and self._fw.spec.run.zero_stage >= 2
                and self._fw.spec.shape.has_moe_layers):
            return self.cluster_devices(stage)
        if self._policy.spread_for(work.kind) is WorkSpread.CLUSTER_RANK_0:
            return (self.device_for(stage, 0),)
        return self.cluster_devices(stage)

    def coords_for(self, work: WorkItem) -> Tuple[DeviceCoord, ...]:
        return tuple(self.coords_of(device) for device in self.devices_for(work))

    def devices_for_sync(self, req: "SyncRequirement") -> Tuple[DeviceId, ...]:
        spread = req.spread
        place_on = req.place_on
        if spread is SyncSpread.STAGE:
            return (self.stage_device(self.stage_of(place_on)),)
        if spread is SyncSpread.CLUSTER_RANK_0:
            return (self.devices_for(place_on)[0],)
        if spread is SyncSpread.PER_CLUSTER_RANK:
            devices = self.cluster_devices(self.stage_of(place_on))
            if len(devices) < self._cluster_size:
                raise PlacementError(
                    f"SyncSpread.PER_CLUSTER_RANK for {req.key!r} resolved to "
                    f"{len(devices)} device(s) but cluster_size is "
                    f"{self._cluster_size}; a per-cluster-rank collective must "
                    "exist on every cluster rank of its stage"
                )
            return devices
        raise PlacementError(f"Unhandled SyncSpread {spread!r}")

    def same_stage(self, a: WorkItem, b: WorkItem) -> bool:
        return self.stage_of(a) == self.stage_of(b)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Placement(granularity={self._granularity.name}, "
            f"devices={len(self.devices())}, cluster_size={self._cluster_size}, "
            f"policy={self._policy.name!r})"
        )

class StepKind(Enum):
    COMPUTE = auto()
    COMM = auto()

    JOIN = auto()

@dataclass(frozen=True, slots=True)
class StepRef:
    device: DeviceId
    index: int

@dataclass(frozen=True, slots=True)
class ComputeStep:
    index: int
    name: str
    duration: float
    deps: Tuple[int, ...] = ()
    kind: StepKind = StepKind.COMPUTE
    entry_name: Optional[str] = None
    mem_kind: Any = None
    param_gather: bool = False
    recompute: bool = False
    activity: OperationActivity = ZERO_ACTIVITY
    invocation_duration: Optional[float] = None
    invocation_activity: Optional[OperationActivity] = None
    embedded_collectives: Tuple[EmbeddedCollective, ...] = ()
    launch_devices: Tuple[DeviceId, ...] = ()
    launch_shadow: bool = False

@dataclass(frozen=True, slots=True)
class CommStep:
    index: int
    comm_key: CommKey
    spec: CommSpec
    deps: Tuple[int, ...] = ()
    kind: StepKind = StepKind.COMM

    extra_consumers: Tuple[StepRef, ...] = ()

    overlap: Optional["OverlapDecl"] = None

    @property
    def placement(self) -> str:
        return self.spec.placement

ChainStep = Union[ComputeStep, CommStep]

@dataclass(frozen=True)
class ExpandedChain:
    work: WorkItem
    device: DeviceId
    cluster_rank: int
    steps: Tuple[ChainStep, ...]
    entry: int
    exit: int

    name_suffix: str = ""

    def __post_init__(self) -> None:
        if not self.steps:
            raise PlacementError(f"Expansion of {self.work!r} produced no steps")
        for position, step in enumerate(self.steps):
            if step.index != position:
                raise PlacementError(
                    f"ExpandedChain step {position} carries index {step.index}"
                )
            for dep in step.deps:
                if dep < 0 or dep >= len(self.steps):
                    raise PlacementError(
                        f"ExpandedChain step {position} depends on out-of-range {dep}"
                    )

    def compute_steps(self) -> Tuple[ComputeStep, ...]:
        return tuple(step for step in self.steps if isinstance(step, ComputeStep))

    def comm_steps(self) -> Tuple[CommStep, ...]:
        return tuple(step for step in self.steps if isinstance(step, CommStep))

    @property
    def entry_step(self) -> ChainStep:
        return self.steps[self.entry]

    @property
    def exit_step(self) -> ChainStep:
        return self.steps[self.exit]

def split_comm_keys(
    comm_keys: Sequence[CommKey],
    specs: Mapping[CommKey, CommSpec],
) -> Tuple[Tuple[CommKey, ...], Tuple[CommKey, ...]]:
    pre: List[CommKey] = []
    post: List[CommKey] = []
    for key in comm_keys:
        spec = specs[key] if key in specs else None
        if spec is None:
            raise PlacementError(
                f"Block template references comm key {key!r} which is not registered"
            )
        if spec.placement == "pre":
            pre.append(key)
        elif spec.placement == "post":
            post.append(key)
        else:  # pragma: no cover - CommSpec.__post_init__ already rejects these
            raise PlacementError(
                f"Unsupported comm placement {spec.placement!r} for key {key!r}"
            )
    return tuple(pre), tuple(post)

def comm_parallel_groups(
    comm_keys: Sequence[CommKey],
    specs: Mapping[CommKey, CommSpec],
) -> Tuple[Tuple[CommKey, ...], ...]:
    grouped: List[List[CommKey]] = []
    index: Dict[str, int] = {}
    for key in comm_keys:
        spec = specs[key]
        group = spec.parallel_group
        if not group:
            grouped.append([key])
            continue
        if group not in index:
            index[group] = len(grouped)
            grouped.append([key])
        else:
            grouped[index[group]].append(key)
    return tuple(tuple(group) for group in grouped)

_WORK_NAME: Mapping[Tuple[WorkKind, Direction], str] = MappingProxyType(
    {
        (WorkKind.EMBEDDING, Direction.FORWARD): "embedding",
        (WorkKind.EMBEDDING, Direction.BACKWARD): "embedding_b",
        (WorkKind.SOFTMAX, Direction.FORWARD): "linear_softmax",
        (WorkKind.SOFTMAX, Direction.BACKWARD): "linear_softmax_b",
        (WorkKind.LAYER, Direction.FORWARD): "layer",
        (WorkKind.LAYER, Direction.BACKWARD): "layer_b",
        (WorkKind.RECOMPUTE, Direction.FORWARD): "layer_recompute",
        (WorkKind.RECOMPUTE, Direction.BACKWARD): "layer_recompute_b",
        (WorkKind.OPTIMIZER, Direction.FORWARD): "optimizer",
        (WorkKind.OPTIMIZER, Direction.BACKWARD): "optimizer",
    }
)

def work_name(work: WorkItem) -> str:
    stem = _WORK_NAME[(work.kind, work.direction)]
    parts: List[str] = [stem]
    if work.stage is not None:
        parts.append(f"stage{work.stage}")
    if work.layer is not None:
        parts.append(f"l{work.layer}")
    if work.microbatch is not None:
        parts.append(f"mb{work.microbatch}")
    return "_".join(parts)

class BlockExpander:
    def __init__(
        self,
        fw: FrozenWorkload,
        placement: Placement,
        overlap: Optional["OverlapPolicy"] = None,
        routing: Optional["MoERoutingPolicy"] = None,
    ) -> None:
        self._fw = fw
        self._placement = placement
        self._overlap = overlap
        self._routing = routing

        self._template_memo: Dict[
            Tuple[WorkKind, Direction, LayerId], Tuple[ExpandedChain, ...]
        ] = {}

    def _specs_for(self, work: WorkItem) -> CommSpecTable:
        if work.layer is None:
            raise PlacementError(
                f"{work.kind.name} WorkItem has no layer, so it resolves no block "
                f"comm keys: {work!r}"
            )
        return self._fw.spec.block_comm(work.layer)

    def expand(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        if self._placement.granularity is Granularity.PIPELINE:
            return self._expand_single(work)
        if work.kind is WorkKind.LAYER or work.kind is WorkKind.RECOMPUTE:
            return self._expand_template(work)
        return self._expand_single(work)

    def _expand_single(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        duration = float(duration_for(work, self._fw, self._placement.layers))
        activity = activity_for(work, self._fw, self._placement.layers)
        kind_of_memory = mem_kind(work)
        stem = work_name(work)
        devices = self._placement.devices_for(work)
        calls = ()
        if work.kind is WorkKind.OPTIMIZER:
            keys = []
            stage = int(work.stage)
            if stage == 0:
                keys.append("optimizer_embedding")
            if stage == max(0, int(self._fw.degrees.pp) - 1):
                keys.append("optimizer_softmax")
            offset = duration - sum(self._fw.durations.get_or(key, 0.0) for key in keys)
            segments = []
            for key in keys:
                segments.extend(replace(call, offset_seconds=offset + call.offset_seconds)
                                for call in self._fw.spec.embedded_collectives.get(key, ()))
                offset += self._fw.durations.get_or(key, 0.0)
            calls = tuple(segments)
        else:
            calls = self._fw.spec.embedded_collectives.get(duration_key(work, self._fw), ())
        rank_calls = self._fw.spec.rank_collectives.get(duration_key(work, self._fw))
        if rank_calls and self._placement.granularity is Granularity.PIPELINE:
            calls = rank_calls[0]
            rank_duration = self._fw.spec.rank_durations.get(duration_key(work, self._fw))
            if rank_duration:
                duration = max(rank_duration)
        chains: List[ExpandedChain] = []
        for cluster_rank, device in enumerate(devices):
            name = stem if len(devices) == 1 else f"{stem}_rank{cluster_rank}"
            key = duration_key(work, self._fw)
            rank_activity = self._fw.spec.rank_activities.get(key)
            rank_duration = self._fw.spec.rank_durations.get(key)
            step = ComputeStep(
                index=0,
                name=name,
                duration=(rank_duration[cluster_rank] if rank_duration and len(devices) > 1 else duration),
                deps=(),
                mem_kind=kind_of_memory,
                recompute=(work.kind is WorkKind.RECOMPUTE),
                activity=(rank_activity[cluster_rank] if rank_activity and len(devices) > 1 else activity),
                embedded_collectives=(rank_calls[cluster_rank] if rank_calls and
                                      self._placement.granularity is Granularity.FLAT else calls),
                launch_devices=tuple(
                    d for d in self._placement.cluster_devices(self._placement.stage_of(work))
                    if d not in devices
                ) if calls and work.kind in (WorkKind.EMBEDDING, WorkKind.SOFTMAX) else (),
            )
            chains.append(
                ExpandedChain(
                    work=work,
                    device=device,
                    cluster_rank=cluster_rank,
                    steps=(step,),
                    entry=0,
                    exit=0,
                )
            )
        return tuple(chains)

    def _expand_template(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        if work.layer is None:
            raise PlacementError(f"{work.kind.name} WorkItem carries no layer: {work!r}")
        memo_key = (work.kind, work.direction, work.layer)
        cached = self._template_memo.get(memo_key)
        if cached is not None:
            return self._instantiate_template(cached, work)
        result = self._expand_template_uncached(work)
        self._template_memo[memo_key] = result
        return result

    def _instantiate_template(
        self, cached: Tuple[ExpandedChain, ...], work: WorkItem
    ) -> Tuple[ExpandedChain, ...]:
        wname = work_name(work)
        chain_new = ExpandedChain.__new__
        chain_set = object.__setattr__
        out: List[ExpandedChain] = []
        for chain in cached:
            replica = chain_new(ExpandedChain)
            chain_set(replica, "work", work)
            chain_set(replica, "device", chain.device)
            chain_set(replica, "cluster_rank", chain.cluster_rank)
            chain_set(replica, "steps", chain.steps)
            chain_set(replica, "entry", chain.entry)
            chain_set(replica, "exit", chain.exit)
            chain_set(replica, "name_suffix", f"_{wname}_rank{chain.cluster_rank}")
            out.append(replica)
        return tuple(out)

    def _expand_template_uncached(self, work: WorkItem) -> Tuple[ExpandedChain, ...]:
        template = self._fw.spec.block_template(work.layer)
        if not template.entries:
            raise PlacementError(
                f"BlockTemplate for layer {work.layer} has no GEMM entries"
            )
        direction = work.direction
        entries = (
            template.entries
            if direction is Direction.FORWARD
            else tuple(reversed(template.entries))
        )

        specs = self._specs_for(work)
        devices = self._placement.devices_for(work)

        joins: Dict[Tuple[str, DeviceId], StepRef] = {}
        chains: List[ExpandedChain] = []
        for cluster_rank, device in enumerate(devices):
            steps = self._expand_chain(
                work, device, cluster_rank, entries, direction, joins, specs
            )
            chains.append(
                ExpandedChain(
                    work=work,
                    device=device,
                    cluster_rank=cluster_rank,
                    steps=steps,
                    entry=0,
                    exit=len(steps) - 1,
                    name_suffix=f"_{work_name(work)}_rank{cluster_rank}",
                )
            )
        return tuple(chains)

    def _expand_chain(
        self,
        work: WorkItem,
        device: DeviceId,
        cluster_rank: int,
        entries: Sequence[GemmEntry],
        direction: Direction,
        joins: Dict[Tuple[str, DeviceId], StepRef],
        specs: CommSpecTable,
    ) -> Tuple[ChainStep, ...]:
        steps: List[ChainStep] = []
        previous: Optional[int] = None
        direction_name = "forward" if direction is Direction.FORWARD else "backward"

        for position, entry in enumerate(entries):
            cfg = entry.direction(direction_name)
            rank_activity = self._fw.spec.rank_activities.get(entry.name)
            rank_duration = self._fw.spec.rank_durations.get(entry.name)
            invocation = cfg.invocations.get(
                "recompute" if work.kind is WorkKind.RECOMPUTE else direction_name
            )
            if cfg.duration is None:
                raise PlacementError(
                    f"Missing duration for transformer entry {entry.name!r} in "
                    f"direction {direction_name!r}"
                )
            pre_keys, post_keys = split_comm_keys(cfg.comm_keys, specs)

            for key in pre_keys:
                previous = self._append_comm(steps, key, previous, specs)

            compute_index = len(steps)
            steps.append(
                ComputeStep(
                    index=compute_index,

                    name=f"{entry.name}_{direction_name}",
                    duration=float(rank_duration[cluster_rank] if rank_duration else cfg.duration),
                    deps=() if previous is None else (previous,),
                    entry_name=entry.name,
                    mem_kind=mem_kind_from_op_name(entry.name),
                    param_gather=(position == 0),
                    recompute=(work.kind is WorkKind.RECOMPUTE),
                    activity=rank_activity[cluster_rank] if rank_activity else cfg.activity,
                    invocation_duration=invocation[0] if invocation is not None else None,
                    invocation_activity=invocation[1] if invocation is not None else None,
                )
            )
            previous = compute_index

            for group in comm_parallel_groups(post_keys, specs):
                head = specs[group[0]]
                if head.parallel_group and head.moe_component:
                    previous = self._append_moe_group(
                        steps, group, previous, device, joins, specs
                    )
                else:
                    for key in group:
                        previous = self._append_comm(steps, key, previous, specs)

        if not steps:
            raise PlacementError("Transformer expansion produced no steps")
        return tuple(steps)

    def _append_comm(
        self,
        steps: List[ChainStep],
        key: CommKey,
        previous: Optional[int],
        specs: CommSpecTable,
    ) -> int:
        spec = specs.require(key)
        index = len(steps)
        steps.append(
            CommStep(
                index=index,
                comm_key=key,
                spec=spec,
                deps=() if previous is None else (previous,),
                overlap=self._declare_overlap(spec),
            )
        )
        return index

    def _declare_overlap(self, spec: CommSpec) -> Optional["OverlapDecl"]:
        if self._overlap is None:
            return None
        return self._overlap.declare(spec, self._fw)

    def _append_moe_group(
        self,
        steps: List[ChainStep],
        group: Sequence[CommKey],
        previous: Optional[int],
        device: DeviceId,
        joins: Dict[Tuple[str, DeviceId], StepRef],
        specs: CommSpecTable,
    ) -> int:
        base_key: Optional[CommKey] = None
        residual_key: Optional[CommKey] = None
        parallel_group: Optional[str] = None
        routing_mode: Optional[str] = None

        from program.policies.routing import MoEComponent, component_of, routing_for_mode

        for key in group:
            spec = specs.require(key)
            parallel_group = parallel_group or spec.parallel_group
            routing_mode = routing_mode or spec.moe_routing_mode

            component = component_of(spec)
            if component is MoEComponent.BASE_ALL_TO_ALL:
                base_key = key
            elif component is MoEComponent.RESIDUAL_P2P:
                residual_key = key
            else:
                raise PlacementError(
                    f"Comm key {key!r} is in parallel group "
                    f"{spec.parallel_group!r} but declares no MoE component"
                )
        if not parallel_group or not routing_mode:
            raise PlacementError(
                f"MoE parallel comm group {tuple(group)} is missing required metadata "
                "(parallel_group / moe_routing_mode)"
            )
        if self._routing is None:
            raise PlacementError(
                f"Comm group {tuple(group)} declares moe_routing_mode "
                f"{routing_mode!r} but no MoERoutingPolicy was supplied to the "
                "BlockExpander"
            )

        if routing_for_mode(routing_mode) != self._routing:
            raise PlacementError(
                f"Comm group {tuple(group)} declares moe_routing_mode "
                f"{routing_mode!r} but the supplied MoERoutingPolicy is "
                f"{self._routing.name!r}"
            )

        axes = tuple(self._routing.routing_axes())
        members = self._placement.communicators.members(axes, device)
        hot_device = DeviceId(self._routing.hot_device(members))
        token = (parallel_group, hot_device)

        base_index: Optional[int] = None
        if base_key is not None:
            base_index = self._append_comm(steps, base_key, previous, specs)

        residual_index: Optional[int] = None
        extra_consumers: Tuple[StepRef, ...] = ()
        if device != hot_device and residual_key is not None:
            hot_join = joins.get(token)
            if hot_join is None:
                raise PlacementError(
                    f"No hot-rank join for MoE parallel group {parallel_group!r} at "
                    f"device {hot_device} (routing axes {canonical_axis_label(axes)}); "
                    "hot ranks are min(members) and must expand first"
                )
            residual_index = len(steps)
            steps.append(
                CommStep(
                    index=residual_index,
                    comm_key=residual_key,
                    spec=specs.require(residual_key),
                    deps=() if previous is None else (previous,),
                    extra_consumers=(hot_join,),
                    overlap=self._declare_overlap(specs.require(residual_key)),
                )
            )

        join_deps: List[int] = []
        if base_index is not None:
            join_deps.append(base_index)
        if residual_index is not None:
            join_deps.append(residual_index)
        if not join_deps and previous is not None:
            join_deps.append(previous)

        join_index = len(steps)
        steps.append(
            ComputeStep(
                index=join_index,
                name=f"{parallel_group}_join_device{int(device)}",
                duration=0.0,
                deps=tuple(join_deps),
                kind=StepKind.JOIN,
            )
        )
        if device == hot_device:
            joins[token] = StepRef(device=device, index=join_index)
        return join_index

__all__ = [
    "BlockExpander",
    "ChainStep",
    "CommStep",
    "ComputeStep",
    "CLUSTER_AXES",
    "CommunicatorFactory",
    "DeviceCoord",
    "ExpandedChain",
    "Granularity",
    "GroupError",
    "LEGACY_PLACEMENT",
    "Placement",
    "PlacementError",
    "PlacementPolicy",
    "StepKind",
    "StepRef",
    "WorkSpread",
    "canonical_layout",
    "comm_parallel_groups",
    "split_comm_keys",
    "work_name",
]
