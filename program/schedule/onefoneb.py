from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from program.schedule.policy import (
    LayerAssignment,
    Schedule,
    ScheduleError,
    ScheduleSlot,
)
from program.types import StageId
from program.work import Direction, WorkItem, WorkKind, WorkSet
from program.workload import FrozenWorkload

__all__ = ["OneFOneBSchedule"]

@dataclass(frozen=True)
class OneFOneBSchedule:
    name: str = "1f1b"
    blocking_p2p: bool = True
    first_stage_layers: int = 0

    def layer_assignment(self, fw: FrozenWorkload) -> LayerAssignment:
        count, stages = int(fw.spec.shape.num_layers), int(fw.spec.degrees.pp)
        if self.first_stage_layers:
            first = int(self.first_stage_layers)
            if stages < 2 or first < 1 or count - first < stages - 1:
                raise ScheduleError("first_pipeline_stage_layers requires pp>=2 and at least one layer on each stage")
            tail = LayerAssignment.contiguous(count - first, stages - 1)
            return LayerAssignment.explicit([0] * first + [int(s) + 1 for s in tail.stage_of_layer], stages)
        return LayerAssignment.contiguous(count, stages)

    def schedule(self, fw: FrozenWorkload, work: WorkSet) -> Schedule:
        layers = self.layer_assignment(fw)
        num_stages = int(fw.spec.degrees.pp)
        num_mb = int(fw.spec.shape.micro_batches)
        num_layers = int(fw.spec.shape.num_layers)

        def unit(stage: int, b: int, forward: bool) -> List[WorkItem]:
            items: List[Optional[WorkItem]] = []
            stage_layers = [l for l in range(num_layers) if int(layers.stage_of(l)) == stage]
            if forward:
                items.append(work.get(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=b))
                for l in stage_layers:
                    items.append(work.get(WorkKind.LAYER, Direction.FORWARD, microbatch=b, layer=l))
                items.append(work.get(WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b))
            else:
                items.append(work.get(WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=b))
                for l in reversed(stage_layers):
                    items.append(work.get(WorkKind.RECOMPUTE, Direction.FORWARD, microbatch=b, layer=l))
                    items.append(work.get(WorkKind.LAYER, Direction.BACKWARD, microbatch=b, layer=l))
                items.append(work.get(WorkKind.EMBEDDING, Direction.BACKWARD, microbatch=b))

            return [it for it in items if it is not None and int(layers.stage_of_work(it)) == stage]

        seqs: Dict[int, List[tuple]] = {}
        for s in range(num_stages):
            warm = min(num_stages - s - 1, num_mb)
            seq = [(True, b) for b in range(warm)]
            for i in range(num_mb - warm):
                seq.append((True, warm + i))
                seq.append((False, i))
            seq.extend((False, b) for b in range(num_mb - warm, num_mb))
            seqs[s] = seq

        placed = set()
        cursor = {s: 0 for s in range(num_stages)}
        order: List[WorkItem] = []

        def ready(s: int, fwd: bool, b: int) -> bool:
            if fwd:
                return s == 0 or (s - 1, True, b) in placed
            return (s, True, b) in placed and (s == num_stages - 1 or (s + 1, False, b) in placed)

        remaining = sum(len(v) for v in seqs.values())
        while remaining:
            progressed = False
            for s in range(num_stages):
                if cursor[s] >= len(seqs[s]):
                    continue
                fwd, b = seqs[s][cursor[s]]
                if not ready(s, fwd, b):
                    continue
                order.extend(unit(s, b, fwd))
                placed.add((s, fwd, b))
                cursor[s] += 1
                remaining -= 1
                progressed = True
            if not progressed:  # pragma: no cover - 1F1B is deadlock-free
                raise ScheduleError("OneFOneBSchedule: no stage can make progress")

        for stage in range(num_stages):
            item = work.get(WorkKind.OPTIMIZER, Direction.BACKWARD, stage=StageId(stage))
            if item is not None:
                order.append(item)

        if len(order) != len(work.items):
            raise ScheduleError(
                f"OneFOneBSchedule placed {len(order)} of {len(work.items)} WorkItems; "
                "every WorkItem must appear exactly once (S3)"
            )
        slots = tuple(
            ScheduleSlot(index=index, stage=layers.stage_of_work(item), work=item)
            for index, item in enumerate(order)
        )
        schedule = Schedule(policy=self.name, layers=layers, slots=slots)
        schedule.check_permutation(work)
        return schedule
