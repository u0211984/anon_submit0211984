from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from types import MappingProxyType
from typing import Mapping, Optional, Protocol

from program.types import AxisName
from program.workload import CommSpec, FrozenWorkload

__all__ = [
    "OverlapError",
    "OverlapAnchor",
    "OverlapDecl",
    "OverlapPolicy",
    "NoOverlap",
    "AxisFractionOverlap",
    "AsyncDgradOverlap",
    "overlap_policy_for",
]

class OverlapError(ValueError):
    pass

class OverlapAnchor(Enum):
    PRODUCER = auto()

    CONSUMER = auto()

@dataclass(frozen=True)
class OverlapDecl:
    fraction: float
    anchor: OverlapAnchor
    blocking_consumer: Optional[str] = None
    """For CONSUMER anchoring: the ``BlockTemplate`` entry name that must wait
    for the blocking part. Today this is the substring test
    ``"attention" in name.lower()`` (transforms.py:224) against a
    TEMPLATE-SUPPLIED name; here it is DATA on the policy. The default
    ``"attention"`` reproduces today exactly."""

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, OverlapAnchor):
            raise OverlapError("OverlapDecl.anchor must be an OverlapAnchor")
        if float(self.fraction) <= 0.0:
            raise OverlapError(
                f"OverlapDecl.fraction must be > 0 (got {self.fraction}); "
                "absence of overlap is declared by returning None"
            )
        if self.anchor is OverlapAnchor.CONSUMER and self.blocking_consumer is None:
            raise OverlapError(
                "OverlapDecl(anchor=CONSUMER) requires a blocking_consumer "
                "(the BlockTemplate entry that waits on the blocking part)"
            )
        object.__setattr__(self, "fraction", float(self.fraction))

    @property
    def is_hoist(self) -> bool:
        return self.fraction >= 1.0

class OverlapPolicy(Protocol):
    name: str

    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        pass

@dataclass(frozen=True)
class NoOverlap(OverlapPolicy):
    name: str = "none"

    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        return None

_DEFAULT_ANCHOR_BY_AXIS: Mapping[AxisName, OverlapAnchor] = MappingProxyType(
    {"tp": OverlapAnchor.PRODUCER, "cp": OverlapAnchor.CONSUMER}
)

@dataclass(frozen=True)
class AxisFractionOverlap(OverlapPolicy):
    name: str = "axis_fraction"
    anchor_by_axis: Mapping[AxisName, OverlapAnchor] = field(
        default_factory=lambda: _DEFAULT_ANCHOR_BY_AXIS
    )

    blocking_consumer: str = "attention"

    def overlap_axis(self, spec: CommSpec) -> AxisName:
        return spec.axes[0]

    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        axis = self.overlap_axis(spec)
        anchor = self.anchor_by_axis.get(axis)
        if anchor is None:
            return None
        fraction = fw.spec.overlap.fraction(axis)
        if fraction <= 0.0:
            return None
        return OverlapDecl(
            fraction=fraction,
            anchor=anchor,
            blocking_consumer=(
                self.blocking_consumer if anchor is OverlapAnchor.CONSUMER else None
            ),
        )

@dataclass(frozen=True)
class AsyncDgradOverlap(OverlapPolicy):
    name: str = "async_dgrad"

    def declare(self, spec: CommSpec, fw: FrozenWorkload) -> Optional[OverlapDecl]:
        fraction = float((spec.extra or {}).get("async_dgrad_overlap_fraction", 0.0) or 0.0)

        if fraction <= 0.0:
            return None
        return OverlapDecl(fraction=min(fraction, 0.999999), anchor=OverlapAnchor.PRODUCER)

def overlap_policy_for(fw: FrozenWorkload) -> OverlapPolicy:
    if any(value > 0.0 for value in fw.spec.overlap.by_axis.values()):
        return AxisFractionOverlap()

    if any(
        float((spec.extra or {}).get("async_dgrad_overlap_fraction", 0.0) or 0.0) > 0.0
        for spec in fw.all_comm_specs()
    ):
        return AsyncDgradOverlap()
    return NoOverlap()
