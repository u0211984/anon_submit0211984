from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from timing_model import CollectiveType
from activity import OperationActivity, ZERO_ACTIVITY

from program.layout import RankLayout

OpUid = int
DeviceId = int

class OpRole(Enum):
    GENERIC = auto()
    TRANSFORMER_LAYER = auto()
    EMBEDDING = auto()
    SOFTMAX = auto()
    OPTIMIZER = auto()

    GEMM = auto()
    JOIN = auto()

class Direction(Enum):
    FORWARD = auto()
    BACKWARD = auto()

@dataclass(frozen=True)
class GroupKey:
    axis: str
    members: Tuple[DeviceId, ...]

@dataclass
class CommGroup:
    key: GroupKey
    label: str

@dataclass(frozen=True, slots=True)
class EmbeddedCollective:
    name: str
    coll: CollectiveType
    size_bytes: float
    participants: int
    interconnect: str
    offset_seconds: float
    duration_seconds: float

@dataclass(slots=True)
class ComputeOp:
    uid: OpUid
    name: str
    device: DeviceId
    duration: Tuple[float, ...]
    deps: Tuple[OpUid, ...] = ()
    role: OpRole = OpRole.GENERIC
    direction: Direction = Direction.FORWARD
    mem_kind: Optional[Any] = None
    recompute: bool = False
    param_gather: bool = False
    micro_batch: Optional[int] = None
    layer: Optional[int] = None
    is_moe_layer: bool = False

    work: Optional[Any] = None
    activity: OperationActivity = ZERO_ACTIVITY

    invocation_duration: Optional[float] = None
    invocation_activity: Optional[OperationActivity] = None

    split_head: bool = False
    embedded_collectives: Tuple[EmbeddedCollective, ...] = ()

    launch_devices: Tuple[DeviceId, ...] = ()

    launch_shadow: bool = False

@dataclass(slots=True)
class CollectiveOp:
    uid: OpUid
    name: str
    device: DeviceId
    coll: CollectiveType

    size_bytes: float
    participants: int = 0
    interconnect: Optional[str] = None

    is_dp: bool = False

    label: Optional[str] = None
    group: Optional[GroupKey] = None
    deps: Tuple[OpUid, ...] = ()

    comm_key: Optional[str] = None

    axes: Tuple[str, ...] = ()

    work: Optional[Any] = None

    launch_continuation: bool = False

    launch_overlap_consumer: Optional[OpUid] = None
    launch_overlap_consumers: Tuple[OpUid, ...] = ()

@dataclass(slots=True)
class TransferOp:
    uid: OpUid
    name: str
    src_device: DeviceId
    dst_device: DeviceId
    size_bytes: int
    comm_type: Optional[CollectiveType]
    producer: OpUid
    deps: Tuple[OpUid, ...] = ()

    consumers: Tuple[OpUid, ...] = ()

    moe_component: Optional[str] = None

    participants: int = 0
    interconnect: Optional[str] = None

    @property
    def is_control(self) -> bool:
        return self.size_bytes == 0 or self.comm_type != CollectiveType.PIPELINE

Op = Union[ComputeOp, CollectiveOp, TransferOp]

@dataclass
class ProgramMeta:
    label: str = ""
    misc: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Program:
    layout: RankLayout
    dp_count: int

    devices: Tuple[DeviceId, ...]
    ops: List[Op]
    groups: Dict[GroupKey, CommGroup]
    meta: ProgramMeta

    def num_stages_initial(self) -> int:
        return int(self.meta.misc.get("num_stages_initial", len(self.devices)))

    def compute_devices(self) -> Tuple[DeviceId, ...]:
        stored = self.meta.misc.get("compute_devices")
        if stored is None:
            return self.devices
        return tuple(stored)

    def device_index(self) -> Dict[DeviceId, int]:
        return {device: idx for idx, device in enumerate(self.devices)}

    def rank_for(self, device: DeviceId, dp_idx: int) -> int:
        return dp_idx * self.num_stages_initial() + self.device_index()[device]

    def validate(self) -> None:
        from program.validate import validate_program

        validate_program(self)

class ProgramBuilder:
    def __init__(
        self,
        layout: Optional[RankLayout] = None,
        dp_count: int = 1,
        meta: Optional[ProgramMeta] = None,
    ) -> None:
        self.layout = layout if layout is not None else RankLayout((), {}, {})
        self.dp_count = max(int(dp_count), 1)
        self.meta = meta if meta is not None else ProgramMeta()
        self._ops: List[Op] = []
        self._groups: Dict[GroupKey, CommGroup] = {}

    def _next_uid(self) -> OpUid:
        return len(self._ops)

    def add_compute(
        self,
        name: str,
        device: DeviceId,
        duration: Union[float, Sequence[float]],
        deps: Sequence[OpUid] = (),
        **fields: Any,
    ) -> OpUid:
        if isinstance(duration, (int, float)):
            duration_tuple = (float(duration),)
        else:
            duration_tuple = tuple(float(v) for v in duration)
        op = ComputeOp(
            uid=self._next_uid(),
            name=name,
            device=int(device),
            duration=duration_tuple,
            deps=tuple(deps),
            **fields,
        )
        self._ops.append(op)
        return op.uid

    def add_collective(
        self,
        name: str,
        device: DeviceId,
        coll: CollectiveType,
        size_bytes: int,
        deps: Sequence[OpUid] = (),
        **fields: Any,
    ) -> OpUid:
        op = CollectiveOp(
            uid=self._next_uid(),
            name=name,
            device=int(device),
            coll=coll,
            size_bytes=int(size_bytes),
            deps=tuple(deps),
            **fields,
        )
        self._ops.append(op)
        return op.uid

    def add_transfer(
        self,
        name: str,
        src_device: DeviceId,
        dst_device: DeviceId,
        size_bytes: int,
        producer: OpUid,
        comm_type: Optional[CollectiveType] = CollectiveType.PIPELINE,
        consumers: Sequence[OpUid] = (),
        **fields: Any,
    ) -> OpUid:
        op = TransferOp(
            uid=self._next_uid(),
            name=name,
            src_device=int(src_device),
            dst_device=int(dst_device),
            size_bytes=int(size_bytes),
            comm_type=comm_type,
            producer=int(producer),
            deps=(int(producer),),
            consumers=tuple(consumers),
            **fields,
        )
        self._ops.append(op)
        return op.uid

    def op(self, uid: OpUid) -> Op:
        return self._ops[uid]

    def group(self, axis: str, members: Sequence[DeviceId], label: str) -> GroupKey:
        key = GroupKey(axis=str(axis), members=tuple(sorted(int(m) for m in members)))
        if key not in self._groups:
            self._groups[key] = CommGroup(key=key, label=label)
        return key

    def finish(
        self,
        devices: Optional[Sequence[DeviceId]] = None,
        *,
        num_stages_initial: Optional[int] = None,
        compute_devices: Optional[Sequence[DeviceId]] = None,
        validate: bool = True,
    ) -> Program:
        if devices is None:
            seen: List[DeviceId] = []
            for op in self._ops:
                if isinstance(op, TransferOp):
                    candidates = (op.src_device, op.dst_device)
                else:
                    candidates = (op.device,)
                for device in candidates:
                    if device not in seen:
                        seen.append(device)
            for key in self._groups:
                for device in key.members:
                    if device not in seen:
                        seen.append(device)
            devices = sorted(seen)
        devices_tuple = tuple(int(d) for d in devices)
        misc = dict(self.meta.misc)
        misc["num_stages_initial"] = (
            int(num_stages_initial) if num_stages_initial is not None else len(
                tuple(compute_devices) if compute_devices is not None else devices_tuple
            )
        )
        if compute_devices is not None:
            misc["compute_devices"] = tuple(int(d) for d in compute_devices)
        meta = ProgramMeta(label=self.meta.label, misc=misc)
        program = Program(
            layout=self.layout,
            dp_count=self.dp_count,
            devices=devices_tuple,
            ops=list(self._ops),
            groups=dict(self._groups),
            meta=meta,
        )
        if validate:
            program.validate()
        return program
