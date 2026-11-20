from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from program.schedule.policy import (
    LayerAssignment,
    Schedule,
    ScheduleError,
    ScheduleSlot,
)
from program.types import StageId
from program.work import Direction, WorkItem, WorkKind, WorkSet
from program.workload import FrozenWorkload

__all__ = ["GPipeSchedule"]

@dataclass(frozen=True)
class GPipeSchedule:
    name: str = "gpipe"

    def layer_assignment(self, fw: FrozenWorkload) -> LayerAssignment:
        return LayerAssignment.contiguous(
            int(fw.spec.shape.num_layers), int(fw.spec.degrees.pp)
        )

    def schedule(self, fw: FrozenWorkload, work: WorkSet) -> Schedule:
        layers = self.layer_assignment(fw)
        shape = fw.spec.shape
        order: List[WorkItem] = []

        def take(item: Optional[WorkItem]) -> None:
            if item is not None:
                order.append(item)

        for b in range(int(shape.micro_batches)):
            take(work.get(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b))
            for layer in range(int(shape.num_layers)):
                take(
                    work.get(
                        WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=layer
                    )
                )
            take(work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))

        for b in reversed(range(int(shape.micro_batches))):
            take(work.get(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b))
            for layer in reversed(range(int(shape.num_layers))):
                take(
                    work.get(
                        WorkKind.RECOMPUTE,
                        Direction.FORWARD,
                        microbatch=b,
                        layer=layer,
                    )
                )
                take(
                    work.get(
                        WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=layer
                    )
                )
            take(work.get(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b))

        for stage in range(int(fw.spec.degrees.pp)):
            take(
                work.get(
                    WorkKind.OPTIMIZER, Direction.BACKWARD, stage=StageId(stage)
                )
            )

        if len(order) != len(work.items):
            raise ScheduleError(
                f"GPipeSchedule placed {len(order)} of {len(work.items)} WorkItems; "
                "every WorkItem must appear exactly once (S3)"
            )

        slots = tuple(
            ScheduleSlot(index=index, stage=layers.stage_of_work(item), work=item)
            for index, item in enumerate(order)
        )
        schedule = Schedule(policy=self.name, layers=layers, slots=slots)
        schedule.check_permutation(work)
        return schedule
