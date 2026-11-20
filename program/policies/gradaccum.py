from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

from program.types import MicroBatch
from program.workload import (
    CommSpec,
    DpMicrobatchMode,
    FrozenWorkload,
    GradAccumCycle,
)

__all__ = ["GradAccumPolicy", "grad_accum_policy_for"]

@dataclass(frozen=True)
class GradAccumPolicy:
    name: str = "legacy"
    dp: int = 1

    cp: int = 1

    dense_ep: int = 1
    zero_stage: int = 0
    cycle: GradAccumCycle = GradAccumCycle.FINAL
    mode: DpMicrobatchMode = DpMicrobatchMode.EVERY_MB

    last_microbatch: int = 0

    def emits(self, spec: CommSpec, microbatch: Optional[MicroBatch]) -> bool:
        if self.dp * max(1, int(self.cp)) * max(1, int(self.dense_ep)) <= 1:
            return False
        if self.cycle is GradAccumCycle.NONFINAL:
            return spec.ga_required_every_cycle
        if spec.ga_required_every_cycle:
            return True
        if self.mode is not DpMicrobatchMode.LAST_MB or self.zero_stage >= 3:
            return True
        return microbatch == self.last_microbatch

    def for_schedule(self, schedule: Any) -> "GradAccumPolicy":
        resolved = schedule.last_microbatch_of()
        return replace(self, last_microbatch=int(resolved))

def grad_accum_policy_for(fw: FrozenWorkload) -> GradAccumPolicy:
    from program.workload import DpGradSync
    """The single construction site: every field is a typed WorkloadSpec read."""
    run = fw.spec.run
    return GradAccumPolicy(
        name="legacy",
        dp=int(fw.spec.degrees.dp),
        cp=int(getattr(fw.spec.degrees, "cp", 1) or 1),

        dense_ep=int(fw.spec.degrees.ep) if run.include_backward and fw.spec.shape.has_moe_layers and (
            run.zero_stage >= 2 or run.dp_grad_sync is not DpGradSync.LEGACY) else 1,
        zero_stage=int(run.zero_stage),
        cycle=run.grad_accum_cycle,
        mode=run.dp_microbatch_mode,
    )
