from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from program.policies.sharding import ContiguousStages
from program.types import DeviceId, MicroBatch, StageId
from program.work import Direction, WorkItem, WorkKind, WorkSet
from program.workload import FrozenWorkload

__all__ = [
    "ScheduleError",
    "LayerAssignment",
    "ScheduleSlot",
    "ScheduleDep",
    "Schedule",
    "SchedulePolicy",
]

class ScheduleError(ValueError):
    pass

@dataclass(frozen=True)
class LayerAssignment(ContiguousStages):
    @classmethod
    def explicit(
        cls, stage_of_layer: Sequence[int], num_stages: Optional[int] = None
    ) -> "LayerAssignment":
        mapping = tuple(StageId(int(stage)) for stage in stage_of_layer)
        if any(int(stage) < 0 for stage in mapping):
            raise ScheduleError(
                f"LayerAssignment.explicit got a negative stage in {tuple(mapping)}"
            )
        stages = int(num_stages) if num_stages is not None else (
            max(int(stage) for stage in mapping) + 1 if mapping else 1
        )
        if mapping and max(int(stage) for stage in mapping) >= stages:
            raise ScheduleError(
                f"LayerAssignment.explicit: stage {max(int(s) for s in mapping)} "
                f"is out of range for num_stages={stages}"
            )
        return cls(stage_of_layer=mapping, num_stages=stages)

    def contiguous_layers(self) -> bool:
        for stage in range(self.num_stages):
            layers = self.layers_of(StageId(stage))
            if not layers:
                continue
            if tuple(range(layers[0], layers[-1] + 1)) != layers:
                return False
        return True

@dataclass(frozen=True)
class ScheduleSlot:
    index: int
    stage: StageId
    work: WorkItem

@dataclass(frozen=True)
class ScheduleDep:
    before: WorkItem
    after: WorkItem
    device: DeviceId
    reason: str = "device_serialization"

_GRADIENT_KINDS = (WorkKind.LAYER, WorkKind.EMBEDDING, WorkKind.SOFTMAX)

@dataclass(frozen=True)
class Schedule:
    policy: str
    layers: LayerAssignment
    slots: Tuple[ScheduleSlot, ...]

    def __post_init__(self) -> None:
        slots = tuple(self.slots)
        object.__setattr__(self, "slots", slots)
        indices = [int(slot.index) for slot in slots]

        if sorted(indices) != list(range(len(slots))):
            raise ScheduleError(
                f"Schedule.slots indices must be dense 0..{len(slots) - 1} and "
                f"unique (got {sorted(indices)})"
            )
        by_work: Dict[WorkItem, int] = {}
        stage_by_work: Dict[WorkItem, StageId] = {}
        for slot in slots:
            if slot.work in by_work:
                raise ScheduleError(
                    f"Schedule contains {slot.work!r} twice (S2: every WorkItem "
                    "appears exactly once)"
                )
            by_work[slot.work] = int(slot.index)
            stage_by_work[slot.work] = StageId(int(slot.stage))
        object.__setattr__(self, "_by_work", by_work)
        object.__setattr__(self, "_stage_by_work", stage_by_work)
        object.__setattr__(
            self, "_ordered", tuple(sorted(slots, key=lambda s: int(s.index)))
        )

    @property
    def ordered_slots(self) -> Tuple[ScheduleSlot, ...]:
        return self.__dict__["_ordered"]

    def order(self) -> Tuple[WorkItem, ...]:
        return tuple(slot.work for slot in self.ordered_slots)

    def index_of(self, work: WorkItem) -> int:
        found = self.__dict__["_by_work"].get(work)
        if found is None:
            raise ScheduleError(f"{work!r} is not scheduled")
        return found

    def stage_of(self, work: WorkItem) -> StageId:
        found = self.__dict__["_stage_by_work"].get(work)
        if found is None:
            raise ScheduleError(f"{work!r} is not scheduled")
        return found

    def order_for(self, stage: StageId) -> Tuple[WorkItem, ...]:
        return tuple(
            slot.work
            for slot in self.ordered_slots
            if int(slot.stage) == int(stage)
        )

    def __len__(self) -> int:
        return len(self.slots)

    def last_microbatch_of(self) -> MicroBatch:
        for slot in reversed(self.ordered_slots):
            work = slot.work
            if (
                work.direction is Direction.BACKWARD
                and work.kind in _GRADIENT_KINDS
                and work.microbatch is not None
            ):
                return MicroBatch(int(work.microbatch))
        return MicroBatch(0)

    def device_projection(
        self, devices_for: Callable[[WorkItem], Sequence[DeviceId]]
    ) -> Mapping[DeviceId, Tuple[WorkItem, ...]]:
        projection: Dict[DeviceId, List[WorkItem]] = {}
        for slot in self.ordered_slots:
            for device in devices_for(slot.work):
                projection.setdefault(DeviceId(int(device)), []).append(slot.work)
        return {device: tuple(items) for device, items in projection.items()}

    def implied_deps(
        self, devices_for: Callable[[WorkItem], Sequence[DeviceId]]
    ) -> Tuple[ScheduleDep, ...]:
        deps: List[ScheduleDep] = []
        projection = self.device_projection(devices_for)
        for device in sorted(projection, key=int):
            items = projection[device]
            for before, after in zip(items, items[1:]):
                deps.append(ScheduleDep(before=before, after=after, device=device))
        deps.sort(
            key=lambda dep: (
                self.index_of(dep.after),
                int(dep.device),
                self.index_of(dep.before),
            )
        )
        return tuple(deps)

    def check_permutation(self, work: WorkSet) -> None:
        scheduled = set(self.__dict__["_by_work"])
        declared = set(work.items)
        missing = declared - scheduled
        extra = scheduled - declared
        if missing or extra:
            raise ScheduleError(
                f"Schedule({self.policy!r}) is not a permutation of the WorkSet: "
                f"{len(missing)} unscheduled {sorted(missing, key=WorkItem.sort_key)[:4]}, "
                f"{len(extra)} unknown {sorted(extra, key=WorkItem.sort_key)[:4]}"
            )

class SchedulePolicy(Protocol):
    name: str

    def layer_assignment(self, fw: FrozenWorkload) -> LayerAssignment:
        pass

    def schedule(self, fw: FrozenWorkload, work: WorkSet) -> Schedule:
        pass
