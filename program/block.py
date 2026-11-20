from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from activity import OperationActivity, ZERO_ACTIVITY

from timing_model import CollectiveType

@dataclass(frozen=True)
class CommMeta:
    name: str
    size_bytes: Any
    kind: CollectiveType
    participants: int
    interconnect: Optional[str]
    local_comp_time: float = 0.0
    tp_shard: bool = False

    ga_required_every_cycle: bool = False

    placement: str = "post"
    parallel_group: Optional[str] = None
    moe_component: Optional[str] = None
    moe_routing_mode: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy(cls, name: str, data: Mapping[str, Any]) -> "CommMeta":
        kind = data.get("type")
        if kind is not None and not isinstance(kind, CollectiveType):
            raise TypeError(
                f"comm_metadata[{name!r}]['type'] must be CollectiveType "
                f"(got {type(kind).__name__})"
            )
        known = {
            "size",
            "type",
            "participants",
            "interconnect_type",
            "local_comp_time",
            "tp_shard",
            "ga_required_every_cycle",
            "placement",
            "parallel_group",
            "moe_component",
            "moe_routing_mode",
        }
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            name=name,
            size_bytes=data.get("size", 0),
            kind=kind,
            participants=int(data.get("participants", 1) or 0),
            interconnect=data.get("interconnect_type"),
            local_comp_time=float(data.get("local_comp_time", 0) or 0.0),
            tp_shard=bool(data.get("tp_shard", False)),
            ga_required_every_cycle=bool(data.get("ga_required_every_cycle", False)),
            placement=str(data.get("placement", "post")),
            parallel_group=data.get("parallel_group"),
            moe_component=data.get("moe_component"),
            moe_routing_mode=data.get("moe_routing_mode"),
            extra=extra,
        )

def comm_metadata_from_legacy(raw: Mapping[str, Mapping[str, Any]]) -> Dict[str, CommMeta]:
    return {name: CommMeta.from_legacy(name, data) for name, data in (raw or {}).items()}

@dataclass(frozen=True)
class GemmDirection:
    duration: Optional[float]
    comm_keys: Tuple[str, ...] = ()
    activity: OperationActivity = ZERO_ACTIVITY

    invocations: Mapping[str, Tuple[float, OperationActivity]] = field(default_factory=dict)

@dataclass(frozen=True)
class GemmEntry:
    name: str
    forward: GemmDirection
    backward: GemmDirection

    def direction(self, direction_name: str) -> GemmDirection:
        if direction_name == "forward":
            return self.forward
        if direction_name == "backward":
            return self.backward
        raise ValueError(f"Unknown direction {direction_name!r}")

@dataclass(frozen=True)
class BlockTemplate:
    entries: Tuple[GemmEntry, ...]
    comm_metadata: Mapping[str, CommMeta]

    @classmethod
    def from_gemm_entries(
        cls,
        gemm_entries: Any,
        comm_metadata: Mapping[str, Mapping[str, Any]],
    ) -> "BlockTemplate":
        entries = []
        for idx, entry in enumerate(gemm_entries):
            name = entry.get("name", f"g{idx}")

            def _direction(cfg: Mapping[str, Any]) -> GemmDirection:
                duration = cfg.get("duration")
                return GemmDirection(
                    duration=float(duration) if duration is not None else None,
                    comm_keys=tuple(cfg.get("comm_keys", []) or []),
                    activity=cfg.get("activity", ZERO_ACTIVITY),
                    invocations=cfg.get("invocations", {}),
                )

            entries.append(
                GemmEntry(
                    name=name,
                    forward=_direction(entry.get("forward", {}) or {}),
                    backward=_direction(entry.get("backward", {}) or {}),
                )
            )
        return cls(
            entries=tuple(entries),
            comm_metadata=comm_metadata_from_legacy(comm_metadata),
        )
