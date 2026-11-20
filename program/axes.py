from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Mapping, Sequence, Tuple

AxisName = str

__all__ = [
    "AxisName",
    "AxisRole",
    "AxisSpec",
    "AXES",
    "AXIS_BY_NAME",
    "CANONICAL_AXES",
    "CLUSTER_AXES",
    "PIPELINE_AXES",
    "REPLICA_AXES",
    "DP_AXIS",
    "axes_with_role",
    "axes_needing_gradient_reduction",
    "activation_sharding_axes",
    "axis_sizes_from",
    "cluster_strides",
]

class AxisRole(Enum):
    CLUSTER = auto()

    PIPELINE = auto()

    REPLICA = auto()

@dataclass(frozen=True)
class AxisSpec:
    name: AxisName
    role: AxisRole

    degree_attr: str

    label: str

    shards_work: bool = False

    replicates_parameters: bool = False

    shards_activations: bool = False

    def __post_init__(self) -> None:
        if self.replicates_parameters and not self.shards_work:
            raise ValueError(
                f"AxisSpec({self.name!r}) replicates parameters without sharding "
                "work; such a rank would be pure duplicate compute"
            )

AXES: Tuple[AxisSpec, ...] = (
    AxisSpec("tp", AxisRole.CLUSTER, "tp", "tensor parallelism", shards_work=False,
             replicates_parameters=False, shards_activations=True),
    AxisSpec("cp", AxisRole.CLUSTER, "cp", "context parallelism", shards_work=True,
             replicates_parameters=True, shards_activations=True),
    AxisSpec("ep", AxisRole.CLUSTER, "ep", "expert parallelism", shards_work=True,
             replicates_parameters=False, shards_activations=False),
    AxisSpec("pp", AxisRole.PIPELINE, "pp", "pipeline parallelism"),
    AxisSpec("dp", AxisRole.REPLICA, "dp", "data parallelism", shards_work=True),
)

AXIS_BY_NAME: Mapping[AxisName, AxisSpec] = {axis.name: axis for axis in AXES}

def axes_with_role(*roles: AxisRole) -> Tuple[AxisName, ...]:
    wanted = set(roles)
    return tuple(axis.name for axis in AXES if axis.role in wanted)

CANONICAL_AXES: Tuple[AxisName, ...] = tuple(axis.name for axis in AXES)
CLUSTER_AXES: Tuple[AxisName, ...] = axes_with_role(AxisRole.CLUSTER)
PIPELINE_AXES: Tuple[AxisName, ...] = axes_with_role(AxisRole.PIPELINE)
REPLICA_AXES: Tuple[AxisName, ...] = axes_with_role(AxisRole.REPLICA)

DP_AXIS: AxisName = REPLICA_AXES[0]

def axes_needing_gradient_reduction() -> Tuple[AxisName, ...]:
    return tuple(
        axis.name
        for axis in AXES
        if axis.role is AxisRole.CLUSTER
        and axis.shards_work
        and axis.replicates_parameters
    )

def activation_sharding_axes() -> Tuple[AxisName, ...]:
    return tuple(axis.name for axis in AXES if axis.shards_activations)

def axis_sizes_from(degrees: object) -> Dict[AxisName, int]:
    return {
        axis.name: max(1, int(getattr(degrees, axis.degree_attr, 1) or 1))
        for axis in AXES
    }

def cluster_strides(
    axis_order: Sequence[AxisName], sizes: Mapping[AxisName, int]
) -> Dict[AxisName, int]:
    strides: Dict[AxisName, int] = {}
    running = 1
    for name in CLUSTER_AXES:
        if name not in axis_order:
            continue
        strides[name] = running
        running *= max(1, int(sizes.get(name, 1)))
    return strides
