from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from functools import total_ordering
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from timing_model import CollectiveType
from activity import OperationActivity, ZERO_ACTIVITY

from program.axes import axes_needing_gradient_reduction
from program.types import DP_AXIS, AxisName, CommKey, LayerId, MicroBatch, StageId
from program.workload import CommSpec, FrozenWorkload, ceil_div

_DP_COMPANION_AXES: Tuple[str, ...] = axes_needing_gradient_reduction()

if TYPE_CHECKING:  # pragma: no cover - typing only; L3 must not be imported at L1
    from program.schedule.policy import LayerAssignment

try:
    from memory_estimation import MemKind
except Exception:  # pragma: no cover - defensive
    MemKind = None  # type: ignore[assignment]

__all__ = [
    "WorkError",
    "SyncError",
    "WorkKind",
    "Direction",
    "WorkItem",
    "WorkRef",
    "WorkSet",
    "enumerate_work",
    "duration_key",
    "duration_for",
    "optimizer_duration",
    "activity_for",
    "optimizer_activity",
    "stage_layers",
    "mem_kind",
    "is_moe",
    "block_prefix",
    "AttachMode",
    "attach_mode_arity",
    "DepClass",
    "VIA_DATA_FLOW",
    "VIA_NON_DATA_FLOW",
    "VIA_ALL",
    "SyncSpread",
    "ByteSplit",
    "ByteSource",
    "SyncPhase",
    "SyncKey",
    "SyncAnchor",
    "SyncRequirement",
]

class WorkError(ValueError):
    pass

class SyncError(ValueError):
    pass

class WorkKind(Enum):
    EMBEDDING = auto()
    LAYER = auto()
    RECOMPUTE = auto()
    SOFTMAX = auto()
    OPTIMIZER = auto()

class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()

_UNSET = -1

@total_ordering
@dataclass(frozen=True)
class WorkItem:
    kind: WorkKind
    direction: Direction
    microbatch: Optional[MicroBatch] = None
    layer: Optional[LayerId] = None

    stage: Optional[StageId] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, WorkKind):
            raise WorkError("WorkItem.kind must be a WorkKind")
        if not isinstance(self.direction, Direction):
            raise WorkError("WorkItem.direction must be a Direction")
        if self.kind is WorkKind.OPTIMIZER:
            if self.stage is None:
                raise WorkError("WorkItem(OPTIMIZER) requires a stage")
            if self.microbatch is not None or self.layer is not None:
                raise WorkError("WorkItem(OPTIMIZER) carries neither microbatch nor layer")
        else:
            if self.stage is not None:
                raise WorkError(
                    f"WorkItem({self.kind.name}) must not carry a stage "
                    "(placement is derived at L2)"
                )
            if self.microbatch is None:
                raise WorkError(f"WorkItem({self.kind.name}) requires a microbatch")
        if self.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE) and self.layer is None:
            raise WorkError(f"WorkItem({self.kind.name}) requires a layer")
        if self.kind in (WorkKind.EMBEDDING, WorkKind.SOFTMAX) and self.layer is not None:
            raise WorkError(f"WorkItem({self.kind.name}) must not carry a layer")
        if self.kind is WorkKind.RECOMPUTE and self.direction is not Direction.FORWARD:
            raise WorkError("WorkItem(RECOMPUTE) is a forward-direction rematerialization")

    def sort_key(self) -> Tuple[int, int, int, int, int]:
        return (
            self.kind.value,
            self.direction.value,
            _UNSET if self.microbatch is None else int(self.microbatch),
            _UNSET if self.layer is None else int(self.layer),
            _UNSET if self.stage is None else int(self.stage),
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, WorkItem):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        parts = [self.kind.name, self.direction.name[:3]]
        if self.microbatch is not None:
            parts.append(f"b{self.microbatch}")
        if self.layer is not None:
            parts.append(f"l{self.layer}")
        if self.stage is not None:
            parts.append(f"s{self.stage}")
        return "W(" + ",".join(parts) + ")"

def _workitem_hash(self: WorkItem) -> int:
    cached = self.__dict__.get("_cached_hash")
    if cached is None:
        cached = hash(
            (self.kind, self.direction, self.microbatch, self.layer, self.stage)
        )
        object.__setattr__(self, "_cached_hash", cached)
    return cached

WorkItem.__hash__ = _workitem_hash  # type: ignore[method-assign]

WorkRef = WorkItem

@dataclass(frozen=True)
class WorkSet:
    items: Tuple[WorkItem, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.items, key=WorkItem.sort_key))
        if len(set(ordered)) != len(ordered):
            raise WorkError("WorkSet contains duplicate WorkItems")
        object.__setattr__(self, "items", ordered)
        object.__setattr__(self, "_index", frozenset(ordered))

    def __iter__(self) -> Iterator[WorkItem]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __contains__(self, item: Any) -> bool:
        return item in self.__dict__["_index"]

    def get(
        self,
        kind: WorkKind,
        direction: Direction,
        *,
        microbatch: Optional[MicroBatch] = None,
        layer: Optional[LayerId] = None,
        stage: Optional[StageId] = None,
    ) -> Optional[WorkItem]:
        candidate = WorkItem(
            kind=kind, direction=direction, microbatch=microbatch, layer=layer, stage=stage
        )
        return candidate if candidate in self else None

    def require(
        self,
        kind: WorkKind,
        direction: Direction,
        *,
        microbatch: Optional[MicroBatch] = None,
        layer: Optional[LayerId] = None,
        stage: Optional[StageId] = None,
    ) -> WorkItem:
        found = self.get(
            kind, direction, microbatch=microbatch, layer=layer, stage=stage
        )
        if found is None:
            raise WorkError(
                f"No WorkItem {kind.name}/{direction.name} "
                f"(microbatch={microbatch}, layer={layer}, stage={stage}) in this WorkSet"
            )
        return found

    def layers(self, direction: Direction, microbatch: MicroBatch) -> Tuple[WorkItem, ...]:
        return tuple(
            item
            for item in self.items
            if item.kind is WorkKind.LAYER
            and item.direction is direction
            and item.microbatch == microbatch
        )

    def of_kind(self, kind: WorkKind) -> Tuple[WorkItem, ...]:
        return tuple(item for item in self.items if item.kind is kind)

def is_moe(item: WorkItem, fw: FrozenWorkload) -> bool:
    if item.layer is None:
        return False
    return fw.spec.is_moe_layer(item.layer)

_DURATION_KEYS: Dict[Tuple[WorkKind, Direction], Tuple[str, str]] = {
    (WorkKind.EMBEDDING, Direction.FORWARD): ("embedding_f", "embedding_f"),
    (WorkKind.EMBEDDING, Direction.BACKWARD): ("embedding_b", "embedding_b"),
    (WorkKind.SOFTMAX, Direction.FORWARD): ("linear_softmax_f", "linear_softmax_f"),
    (WorkKind.SOFTMAX, Direction.BACKWARD): ("linear_softmax_b", "linear_softmax_b"),
    (WorkKind.LAYER, Direction.FORWARD): ("transformer_f_dense", "transformer_f_moe"),
    (WorkKind.LAYER, Direction.BACKWARD): ("transformer_b_dense", "transformer_b_moe"),
    (WorkKind.RECOMPUTE, Direction.FORWARD): ("transformer_f_dense", "transformer_f_moe"),
    (WorkKind.OPTIMIZER, Direction.BACKWARD): ("optimizer", "optimizer_moe"),
}

_DURATION_FALLBACK: Dict[str, str] = {
    "transformer_f_dense": "transformer_f",
    "transformer_f_moe": "transformer_f",
    "transformer_b_dense": "transformer_b",
    "transformer_b_moe": "transformer_b",
    "optimizer_moe": "optimizer",
}

def duration_key(item: WorkItem, fw: FrozenWorkload) -> str:
    entry = _DURATION_KEYS.get((item.kind, item.direction))
    if entry is None:
        raise WorkError(f"No duration key for {item!r}")
    dense_key, moe_key = entry
    return moe_key if is_moe(item, fw) else dense_key

def _duration_by_key(key: str, fw: FrozenWorkload) -> float:
    fallback_key = _DURATION_FALLBACK.get(key)
    fallback = 0.0 if fallback_key is None else fw.durations.get_or(fallback_key, 0.0)
    return fw.durations.get_or(key, fallback)

def _activity_by_key(key: str, fw: FrozenWorkload) -> OperationActivity:
    fallback_key = _DURATION_FALLBACK.get(key)
    fallback = ZERO_ACTIVITY if fallback_key is None else fw.activities.get(
        fallback_key, ZERO_ACTIVITY
    )
    return fw.activities.get(key, fallback)

def stage_layers(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> Tuple[LayerId, ...]:
    if item.stage is None:
        raise WorkError(f"{item!r} carries no stage, so it owns no layers")
    if layers is None:
        from program.policies.sharding import ContiguousStages

        layers = ContiguousStages.legacy(fw.shape.num_layers, fw.degrees.pp)
    return tuple(LayerId(int(layer)) for layer in layers.layers_of(StageId(int(item.stage))))

def optimizer_duration(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> float:
    per_layer_dense = _duration_by_key("optimizer", fw)
    per_layer_moe = _duration_by_key("optimizer_moe", fw)
    total = 0.0
    for layer in stage_layers(item, fw, layers):
        total += per_layer_moe if fw.is_moe_layer(layer) else per_layer_dense
    stage = int(item.stage)
    if stage == 0:
        total += _duration_by_key("optimizer_embedding", fw)
    if stage == max(0, int(fw.degrees.pp) - 1):
        total += _duration_by_key("optimizer_softmax", fw)
    return total

def optimizer_activity(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> OperationActivity:
    per_layer_dense = _activity_by_key("optimizer", fw)
    per_layer_moe = _activity_by_key("optimizer_moe", fw)
    total = ZERO_ACTIVITY
    for layer in stage_layers(item, fw, layers):
        total = total + (
            per_layer_moe if fw.is_moe_layer(layer) else per_layer_dense
        )
    stage = int(item.stage)
    if stage == 0:
        total = total + _activity_by_key("optimizer_embedding", fw)
    if stage == max(0, int(fw.degrees.pp) - 1):
        total = total + _activity_by_key("optimizer_softmax", fw)
    return total

def duration_for(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> float:
    if item.kind is WorkKind.OPTIMIZER:
        return optimizer_duration(item, fw, layers)
    return _duration_by_key(duration_key(item, fw), fw)

def activity_for(
    item: WorkItem,
    fw: FrozenWorkload,
    layers: Optional["LayerAssignment"] = None,
) -> OperationActivity:
    if item.kind is WorkKind.OPTIMIZER:
        return optimizer_activity(item, fw, layers)
    return _activity_by_key(duration_key(item, fw), fw)

_MEM_KIND_NAMES: Dict[WorkKind, str] = {
    WorkKind.EMBEDDING: "EMBEDDING",
    WorkKind.SOFTMAX: "SOFTMAX",
    WorkKind.LAYER: "TRANSFORMER",
    WorkKind.RECOMPUTE: "TRANSFORMER",
    WorkKind.OPTIMIZER: "OPTIMIZER",
}

def mem_kind(item: WorkItem) -> Optional[Any]:
    if MemKind is None:  # pragma: no cover - defensive
        return None
    name = _MEM_KIND_NAMES.get(item.kind)
    if name is None:
        return None
    return MemKind[name]

def block_prefix(fw: FrozenWorkload) -> str:
    return "vit_block" if fw.spec.shape.model_type.startswith("vit") else "transformer_layer"

def enumerate_work(fw: FrozenWorkload, recompute: Any) -> WorkSet:
    shape = fw.spec.shape
    run = fw.spec.run
    items: list[WorkItem] = []

    for b in range(shape.micro_batches):
        items.append(WorkItem(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b))
        for layer in range(shape.num_layers):
            items.append(WorkItem(WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer))
        items.append(WorkItem(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))

        if not run.include_backward:
            continue

        items.append(WorkItem(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b))
        for layer in range(shape.num_layers):
            items.append(WorkItem(WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=layer))
            if recompute.materializes(layer, b, fw):
                items.append(
                    WorkItem(WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=b, layer=layer)
                )
        items.append(WorkItem(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b))

    if (
        run.include_backward
        and run.include_optimizer
        and fw.durations.get_or("optimizer", 0.0) > 0.0
    ):
        for stage in range(fw.spec.degrees.pp):
            items.append(WorkItem(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=StageId(stage)))

    return WorkSet(items=tuple(items))

def backward_entry(item: WorkItem, work: WorkSet) -> WorkItem:
    if item.kind is not WorkKind.LAYER or item.direction is not Direction.BACKWARD:
        return item
    candidate = work.get(
        WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=item.microbatch, layer=item.layer
    )
    return candidate if candidate is not None else item

class AttachMode(Enum):
    BEFORE = auto()
    AFTER = auto()
    PARALLEL_TO = auto()
    OVERLAP_WITH = auto()

def attach_mode_arity(mode: AttachMode) -> Optional[int]:
    if mode is AttachMode.BEFORE:
        return 1
    if mode is AttachMode.AFTER:
        return 1
    if mode is AttachMode.OVERLAP_WITH:
        return 1
    if mode is AttachMode.PARALLEL_TO:
        return None
    from typing import assert_never

    assert_never(mode)

class DepClass(Enum):
    DATA_FLOW = auto()
    SCHEDULE = auto()
    SYNC = auto()

VIA_DATA_FLOW: FrozenSet[DepClass] = frozenset({DepClass.DATA_FLOW})
VIA_NON_DATA_FLOW: FrozenSet[DepClass] = frozenset({DepClass.SCHEDULE, DepClass.SYNC})
VIA_ALL: FrozenSet[DepClass] = frozenset(DepClass)

class SyncSpread(Enum):
    STAGE = auto()
    CLUSTER_RANK_0 = auto()
    PER_CLUSTER_RANK = auto()

class ByteSplit(Enum):
    WHOLE = auto()
    CEIL_DIV_CLUSTER = auto()

@dataclass(frozen=True)
class ByteSource:
    key: CommKey
    split: ByteSplit = ByteSplit.WHOLE

    def bytes_for(self, fw: FrozenWorkload, instances: int) -> float:
        count = int(instances)
        if count < 1:
            raise SyncError(
                f"ByteSource({self.key!r}).bytes_for needs instances >= 1 (got {instances})"
            )
        spec = fw.spec.comm.require(self.key)
        if self.split is ByteSplit.WHOLE:
            return float(spec.size_bytes)
        if self.split is ByteSplit.CEIL_DIV_CLUSTER:
            return ceil_div(spec.size_bytes, count)
        from typing import assert_never

        assert_never(self.split)

class SyncPhase(Enum):
    FWD_ENTRY = auto()
    FWD = auto()
    BWD_ENTRY = auto()
    BWD = auto()
    GRAD = auto()

@dataclass(frozen=True)
class SyncKey:
    comm_key: CommKey
    phase: SyncPhase
    microbatch: Optional[MicroBatch] = None
    layer: Optional[LayerId] = None

    def sort_key(self) -> Tuple[int, int, int, str]:
        return (
            self.phase.value,
            _UNSET if self.microbatch is None else int(self.microbatch),
            _UNSET if self.layer is None else int(self.layer),
            self.comm_key,
        )

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, SyncKey):
            return NotImplemented
        return self.sort_key() < other.sort_key()

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        parts = [self.comm_key, self.phase.name]
        if self.microbatch is not None:
            parts.append(f"b{self.microbatch}")
        if self.layer is not None:
            parts.append(f"l{self.layer}")
        return "S(" + ",".join(parts) + ")"

SyncAnchor = Union[WorkItem, SyncKey]

@dataclass(frozen=True)
class SyncRequirement:
    key: SyncKey

    bytes: ByteSource
    kind: CollectiveType
    axes: Tuple[AxisName, ...]
    participants: int

    place_on: WorkItem
    spread: SyncSpread = SyncSpread.CLUSTER_RANK_0

    mode: AttachMode = AttachMode.AFTER
    anchors: Tuple[SyncAnchor, ...] = ()
    via: FrozenSet[DepClass] = VIA_ALL

    consumers: Tuple[WorkItem, ...] = ()

    overlap: Optional[Any] = None

    is_reducer: bool = False

    after_update: bool = False

    origin: str = ""

    def __post_init__(self) -> None:
        if not self.anchors:
            raise SyncError(f"SyncRequirement {self.key!r} has no anchors")
        arity = attach_mode_arity(self.mode)
        if arity is not None and len(self.anchors) != arity:
            raise SyncError(
                f"SyncRequirement {self.key!r} mode {self.mode.name} admits exactly "
                f"{arity} anchor(s), got {len(self.anchors)}"
            )
        if self.mode is not AttachMode.PARALLEL_TO and self.via != VIA_ALL:
            raise SyncError(
                f"SyncRequirement {self.key!r} sets via={sorted(c.name for c in self.via)} "
                f"but mode {self.mode.name} ignores it (silent-misuse guard)"
            )
        if self.kind is CollectiveType.PIPELINE:
            raise SyncError(
                f"SyncRequirement {self.key!r} has kind PIPELINE; a p2p transfer is a "
                "data-flow TransferOp (R2), not a SyncRequirement"
            )
        if DP_AXIS in tuple(self.axes) and tuple(self.axes) != (DP_AXIS,):
            companions = tuple(a for a in self.axes if a != DP_AXIS)

            allowed = (*_DP_COMPANION_AXES, "ep") if str(self.comm_key).startswith(("zm_", "mdp_")) else _DP_COMPANION_AXES
            unknown = [a for a in companions if a not in allowed]
            if unknown:
                raise SyncError(
                    f"SyncRequirement {self.key!r} declares axes {tuple(self.axes)}: "
                    f"'dp' may only be combined with an in-device-space axis "
                    f"{_DP_COMPANION_AXES}, got {unknown}"
                )
        if self.mode is AttachMode.AFTER:
            pass
        elif any(isinstance(anchor, SyncKey) for anchor in self.anchors):
            raise SyncError(
                f"SyncRequirement {self.key!r} mode {self.mode.name} may only anchor on "
                "WorkItems; only AFTER may chain onto another SyncKey"
            )
        if any(not isinstance(item, WorkItem) for item in self.consumers):
            raise SyncError(
                f"SyncRequirement {self.key!r} declares a non-WorkItem consumer: "
                f"{list(self.consumers)!r}"
            )
        object.__setattr__(self, "anchors", tuple(self.anchors))
        object.__setattr__(self, "consumers", tuple(self.consumers))
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "participants", int(self.participants))

    @classmethod
    def from_spec(
        cls,
        spec: CommSpec,
        *,
        key: SyncKey,
        place_on: WorkItem,
        mode: AttachMode,
        anchors: Sequence[SyncAnchor],
        spread: SyncSpread = SyncSpread.CLUSTER_RANK_0,
        via: FrozenSet[DepClass] = VIA_ALL,
        consumers: Sequence[WorkItem] = (),
        split: ByteSplit = ByteSplit.WHOLE,
        overlap: Optional[Any] = None,
        is_reducer: bool = False,
        after_update: bool = False,
        origin: str = "",
    ) -> "SyncRequirement":
        if key.comm_key != spec.key:
            raise SyncError(
                f"SyncKey.comm_key {key.comm_key!r} disagrees with CommSpec.key {spec.key!r}"
            )
        return cls(
            key=key,
            bytes=ByteSource(key=spec.key, split=split),
            kind=spec.kind,
            axes=spec.axes,
            participants=spec.participants,
            place_on=place_on,
            spread=spread,
            mode=mode,
            anchors=tuple(anchors),
            via=via,
            consumers=tuple(consumers),
            overlap=overlap,
            is_reducer=is_reducer,
            after_update=after_update,
            origin=origin,
        )

    @property
    def comm_key(self) -> CommKey:
        return self.bytes.key

    @property
    def is_dp(self) -> bool:
        return DP_AXIS in tuple(self.axes)

    @property
    def dp_companion_axes(self) -> Tuple[str, ...]:
        return tuple(axis for axis in self.axes if axis != DP_AXIS)

    def size_bytes(self, fw: FrozenWorkload, instances: int = 1) -> float:
        return self.bytes.bytes_for(fw, instances)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"Sync({self.key!r},{self.mode.name},on={self.place_on!r},"
            f"anchors={list(self.anchors)},spread={self.spread.name})"
        )
