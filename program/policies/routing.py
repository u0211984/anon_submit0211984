from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from program.axes import CLUSTER_AXES
from program.types import AxisName, Coords

__all__ = [
    "RoutingError",
    "MoEComponent",
    "MoERoutingPolicy",
    "AxisRouting",
    "EP_ROUTING",
    "TP_EP_ROUTING",
    "ROUTING_MODES",
    "routing_for_mode",
    "routing_policy_for",
]

class RoutingError(ValueError):
    pass

class MoEComponent(Enum):
    BASE_ALL_TO_ALL = "base_all_to_all"
    RESIDUAL_P2P = "residual_p2p"

MOE_COMPONENTS: Mapping[str, MoEComponent] = {c.value: c for c in MoEComponent}

def component_of(spec: Any) -> Optional[MoEComponent]:
    name = spec.moe_component
    if name is None:
        return None
    if name not in MOE_COMPONENTS:
        raise RoutingError(
            f"MoE parallel comm group contains unsupported component {name!r} "
            f"for key {spec.key!r} (known: {sorted(MOE_COMPONENTS)})"
        )
    return MOE_COMPONENTS[name]

class MoERoutingPolicy(Protocol):
    name: str

    def routing_axes(self) -> Tuple[AxisName, ...]:
        pass

    def hot_coords(self, coords: Coords) -> Coords:
        pass

    def join_token(self, coords: Coords) -> Tuple[Any, ...]:
        pass

@dataclass(frozen=True)
class AxisRouting(MoERoutingPolicy):
    name: str
    axes: Tuple[AxisName, ...]

    cluster_axes: Tuple[AxisName, ...] = CLUSTER_AXES

    def __post_init__(self) -> None:
        if not self.axes:
            raise RoutingError(f"AxisRouting {self.name!r} declares no routing axes")
        unknown = [a for a in self.axes if a not in self.cluster_axes]
        if unknown:
            raise RoutingError(
                f"AxisRouting {self.name!r} routes over {unknown} which are not "
                f"cluster axes {list(self.cluster_axes)}"
            )

    def routing_axes(self) -> Tuple[AxisName, ...]:
        return tuple(self.axes)

    def hot_coords(self, coords: Coords) -> Dict[AxisName, int]:
        return {**dict(coords), **{axis: 0 for axis in self.axes}}

    def is_hot(self, coords: Coords) -> bool:
        return all(int(dict(coords).get(axis, 0)) == 0 for axis in self.axes)

    def join_token(self, coords: Coords) -> Tuple[Any, ...]:
        base = dict(coords)
        rest = tuple(a for a in self.cluster_axes if a not in self.axes)
        return (self.name,) + tuple(int(base.get(axis, 0)) for axis in rest)

    def hot_device(self, members: Sequence[int]) -> int:
        if not members:
            raise RoutingError(f"AxisRouting {self.name!r}: empty routing group")
        return min(int(m) for m in members)

EP_ROUTING = AxisRouting(name="ep", axes=("ep",))

TP_EP_ROUTING = AxisRouting(name="tp_ep", axes=("tp", "ep"))

ROUTING_MODES: Mapping[str, AxisRouting] = {
    EP_ROUTING.name: EP_ROUTING,
    TP_EP_ROUTING.name: TP_EP_ROUTING,
}

def routing_for_mode(mode: str) -> AxisRouting:
    if mode not in ROUTING_MODES:
        raise RoutingError(
            f"Unsupported MoE routing mode {mode!r} (known: {sorted(ROUTING_MODES)})"
        )
    return ROUTING_MODES[mode]

def routing_policy_for(fw: Any) -> Optional[AxisRouting]:
    declared = {
        spec.moe_routing_mode
        for spec in fw.spec.all_comm_specs()
        if spec.moe_routing_mode is not None
    }
    if len(declared) > 1:
        raise RoutingError(
            f"Workload declares conflicting MoE routing modes {sorted(declared)}"
        )
    if declared:
        return routing_for_mode(next(iter(declared)))
    if fw.spec.degrees.ep > 1:
        return EP_ROUTING
    return None

EP_SYNC_KEYS: Mapping[bool, str] = {
    False: "transformer_dense_ep_sync",
    True: "transformer_moe_ep_sync",
}

def ep_sync_requirements(work: Any, ctx: Any) -> Tuple[Any, ...]:
    from program.work import (
        AttachMode,
        Direction,
        SyncKey,
        SyncPhase,
        SyncRequirement,
        SyncSpread,
        WorkKind,
        is_moe,
    )

    if ctx.fw.spec.run.include_backward and ctx.fw.spec.run.zero_stage >= 2 and ctx.fw.spec.shape.has_moe_layers:
        return ()
    if work.kind is not WorkKind.LAYER or work.direction is not Direction.BACKWARD:
        return ()

    if ctx.fw.spec.degrees.ep <= 1:
        return ()
    key = EP_SYNC_KEYS[bool(is_moe(work, ctx.fw))]
    spec = ctx.spec_for(key)
    if spec is None or not ctx.grad_accum.emits(spec, work.microbatch):
        return ()
    return (
        SyncRequirement.from_spec(
            spec,
            key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer),
            place_on=work,
            mode=AttachMode.AFTER,
            anchors=(work,),
            spread=SyncSpread.PER_CLUSTER_RANK,

            is_reducer=True,
            origin="S12",
        ),
    )
