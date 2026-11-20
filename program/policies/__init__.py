from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from program.policies.gradaccum import GradAccumPolicy, grad_accum_policy_for
from program.policies.overlap import (
    AxisFractionOverlap,
    NoOverlap,
    OverlapAnchor,
    OverlapDecl,
    OverlapPolicy,
    overlap_policy_for,
)
from program.policies.recompute import (
    FullRecompute,
    NoRecompute,
    RecomputePolicy,
    recompute_policy_for,
)
from program.policies.routing import (
    ROUTING_MODES,
    AxisRouting,
    MoERoutingPolicy,
    routing_for_mode,
    routing_policy_for,
)
from program.policies.sharding import (
    DDP,
    SHARDING_POLICIES,
    ContiguousStages,
    NullSharding,
    ShardingContext,
    ShardingPolicy,
    StagePartition,
    ZeRO1,
    ZeRO2,
    ZeRO3,
    sharding_policy_for,
)

__all__ = [
    "PolicyBundle",
    "policies_for",

    "ShardingPolicy",
    "ShardingContext",
    "StagePartition",
    "ContiguousStages",
    "NullSharding",
    "DDP",
    "ZeRO1",
    "ZeRO2",
    "ZeRO3",
    "SHARDING_POLICIES",
    "sharding_policy_for",

    "GradAccumPolicy",
    "grad_accum_policy_for",

    "RecomputePolicy",
    "NoRecompute",
    "FullRecompute",
    "recompute_policy_for",

    "MoERoutingPolicy",
    "AxisRouting",
    "ROUTING_MODES",
    "routing_for_mode",
    "routing_policy_for",

    "OverlapPolicy",
    "OverlapAnchor",
    "OverlapDecl",
    "NoOverlap",
    "AxisFractionOverlap",
    "overlap_policy_for",
]

@dataclass(frozen=True)
class PolicyBundle:
    sharding: ShardingPolicy
    grad_accum: GradAccumPolicy
    recompute: RecomputePolicy
    overlap: OverlapPolicy
    routing: Optional[MoERoutingPolicy] = None

    @property
    def names(self) -> dict:
        return {
            "sharding": self.sharding.name,
            "grad_accum": self.grad_accum.name,
            "recompute": self.recompute.name,
            "overlap": self.overlap.name,
            "routing": None if self.routing is None else self.routing.name,
        }

def policies_for(
    spec: Any,
    *,
    granularity: Any = None,
    block_expanded: Optional[bool] = None,
) -> PolicyBundle:
    from program.placement import Granularity
    from program.workload import WorkloadSpec

    fw = spec.freeze() if isinstance(spec, WorkloadSpec) else spec

    from program.policies.sharding import MoEZeRO
    sharding = (
        NullSharding()
        if granularity is Granularity.BLOCK
        else MoEZeRO() if fw.spec.run.include_backward and fw.spec.run.zero_stage >= 2 and fw.spec.shape.has_moe_layers
        else sharding_policy_for(fw.spec.run, fw.spec.degrees)
    )
    return PolicyBundle(
        sharding=sharding,
        grad_accum=grad_accum_policy_for(fw),
        recompute=recompute_policy_for(
            fw, granularity, block_expanded=block_expanded
        ),
        overlap=overlap_policy_for(fw),
        routing=routing_policy_for(fw),
    )
