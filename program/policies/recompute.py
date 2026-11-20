from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from program.types import LayerId, MicroBatch
from program.workload import FrozenWorkload

__all__ = [
    "RecomputeError",
    "RecomputePolicy",
    "NoRecompute",
    "FullRecompute",
    "recompute_policy_for",
]

class RecomputeError(ValueError):
    pass

class RecomputePolicy(Protocol):
    name: str

    def materializes(
        self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload
    ) -> bool: ...

@dataclass(frozen=True)
class NoRecompute(RecomputePolicy):
    name: str = "none"

    def materializes(self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload) -> bool:
        return False

@dataclass(frozen=True)
class FullRecompute(RecomputePolicy):
    name: str = "full"

    def materializes(self, layer: LayerId, microbatch: MicroBatch, fw: FrozenWorkload) -> bool:
        return True

def recompute_policy_for(
    fw: FrozenWorkload,
    granularity: Any = None,
    *,
    block_expanded: Optional[bool] = None,
) -> RecomputePolicy:
    run = fw.spec.run
    enabled = run.include_backward and run.full_recomputation
    return FullRecompute() if enabled else NoRecompute()
