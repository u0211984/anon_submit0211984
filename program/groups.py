from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from program.ir import GroupKey
from program.layout import CANONICAL_AXES, RankLayout
from program.types import DP_AXIS, AxisName, DeviceId

class GroupError(ValueError):
    pass

def canonical_axis_label(axes: Sequence[AxisName]) -> str:
    seen = {str(axis) for axis in axes}
    ordered = [axis for axis in CANONICAL_AXES if axis in seen]
    ordered.extend(sorted(axis for axis in seen if axis not in CANONICAL_AXES))
    if not ordered:
        raise GroupError("A communicator needs at least one axis")
    return "+".join(ordered)

@dataclass(frozen=True)
class CommunicatorFactory:
    layout: RankLayout

    _memo: Dict[Tuple, object] = field(
        default_factory=dict, compare=False, repr=False, hash=False
    )

    def members(self, axes: Sequence[AxisName], anchor: DeviceId) -> Tuple[DeviceId, ...]:
        memo_key = (tuple(axes), int(anchor))
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached

        spanned = self._spanned_axes(axes)
        base = self.layout.coords_of(int(anchor))
        if not spanned:
            result: Tuple[DeviceId, ...] = (DeviceId(int(anchor)),)
        else:
            ranges = [range(self.layout.axis_sizes[axis]) for axis in spanned]
            found: List[int] = []
            for combo in itertools.product(*ranges):
                coords: Dict[AxisName, int] = dict(base)
                coords.update(zip(spanned, combo))
                found.append(self.layout.linearize(coords))
            result = tuple(DeviceId(rank) for rank in sorted(set(found)))

        self._memo[memo_key] = result
        return result

    def group_for(self, axes: Sequence[AxisName], anchor: DeviceId) -> GroupKey:
        memo_key = ("group_for", tuple(axes), int(anchor))
        cached = self._memo.get(memo_key)
        if cached is None:
            cached = GroupKey(
                axis=canonical_axis_label(axes),
                members=self.members(axes, anchor),
            )
            self._memo[memo_key] = cached
        return cached  # type: ignore[return-value]

    def groups_for(
        self, req: "SyncRequirement", placement: "Placement"  # noqa: F821
    ) -> Tuple[Optional[GroupKey], ...]:
        devices = placement.devices_for_sync(req)
        if req.is_dp:
            return tuple(None for _ in devices)
        return tuple(self.group_for(req.axes, device) for device in devices)

    def spans(self, axes: Sequence[AxisName]) -> int:
        span = 1
        for axis in self._spanned_axes(axes):
            span *= self.layout.axis_sizes[axis]
        return span

    def partition(self, axes: Sequence[AxisName]) -> Tuple[Tuple[DeviceId, ...], ...]:
        seen: Dict[Tuple[DeviceId, ...], None] = {}
        for device in range(self.layout.num_ranks()):
            seen.setdefault(self.members(axes, DeviceId(device)), None)
        return tuple(sorted(seen, key=lambda group: group[0]))

    def _spanned_axes(self, axes: Sequence[AxisName]) -> Tuple[AxisName, ...]:
        if not axes:
            raise GroupError("A communicator needs at least one axis")
        spanned: List[AxisName] = []
        for axis in axes:
            name = str(axis)
            if name == DP_AXIS:
                raise GroupError(
                    "'dp' is not a communicator axis of a device layout: dp "
                    "replication is stamped at emission over pre-dp device ids "
                    "(ir.py:93-104). A dp requirement carries group=None, "
                    "is_dp=True (SyncRequirement.is_dp)."
                )
            if name not in CANONICAL_AXES:
                raise GroupError(
                    f"Unknown communicator axis {name!r} (canonical axes: {CANONICAL_AXES})"
                )
            if name not in self.layout.axis_order:
                continue
            if self.layout.axis_sizes[name] <= 1:
                continue
            if name in spanned:
                raise GroupError(f"Duplicate communicator axis {name!r} in {tuple(axes)!r}")
            spanned.append(name)

        return tuple(axis for axis in CANONICAL_AXES if axis in spanned)

def dedupe_groups(groups: Iterable[GroupKey]) -> Tuple[GroupKey, ...]:
    seen: Dict[Tuple[str, Tuple[DeviceId, ...]], GroupKey] = {}
    for group in groups:
        seen.setdefault((group.axis, group.members), group)
    return tuple(seen.values())

__all__ = [
    "CommunicatorFactory",
    "GroupError",
    "canonical_axis_label",
    "dedupe_groups",
]
