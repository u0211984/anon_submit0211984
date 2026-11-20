from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from timing_model import CollectiveType
from activity import OperationActivity

from program.block import BlockTemplate, CommMeta
from program.layout import RankLayout
from program.axes import CANONICAL_AXES, axis_sizes_from
from program.ir import EmbeddedCollective
from program.types import AxisName, CommKey, LayerId

__all__ = [
    "WorkloadError",
    "RunType",
    "GradAccumCycle",
    "DpMicrobatchMode",
    "DpGradSync",
    "ParallelDegrees",
    "ModelShape",
    "CommSpec",
    "CommSpecTable",
    "DurationTable",
    "FrozenDurations",
    "RunPolicy",
    "OverlapSpec",
    "BlockTemplates",
    "WorkloadSpec",
    "FrozenWorkload",
    "DURATION_KEYS",
]

class WorkloadError(ValueError):
    pass

class RunType(Enum):
    TRAINING = auto()
    INFERENCE = auto()

    @classmethod
    def parse(cls, value: Any) -> "RunType":
        if isinstance(value, RunType):
            return value
        text = str(value).strip().lower()
        if text == "inference":
            return cls.INFERENCE
        if text in ("training", "train"):
            return cls.TRAINING
        raise WorkloadError(f"Unknown run_type {value!r} (expected 'training' or 'inference')")

class GradAccumCycle(Enum):
    FINAL = auto()
    NONFINAL = auto()

    @classmethod
    def parse(cls, value: Any) -> "GradAccumCycle":
        if isinstance(value, GradAccumCycle):
            return value
        text = str(value).strip().lower()
        if text == "nonfinal":
            return cls.NONFINAL
        if text == "final":
            return cls.FINAL
        raise WorkloadError(
            f"Unknown grad_accum_cycle {value!r} (expected 'final' or 'nonfinal')"
        )

class DpMicrobatchMode(Enum):
    EVERY_MB = auto()
    LAST_MB = auto()

    @classmethod
    def parse(cls, value: Any) -> "DpMicrobatchMode":
        if isinstance(value, DpMicrobatchMode):
            return value
        text = str(value).strip().lower()
        if text == "last_mb":
            return cls.LAST_MB
        if text == "every_mb":
            return cls.EVERY_MB
        raise WorkloadError(
            f"Unknown dp_microbatch_mode {value!r} (expected 'every_mb' or 'last_mb')"
        )

class DpGradSync(Enum):
    LEGACY = "legacy"
    MEGATRON_END_OF_BACKWARD = "megatron_end_of_backward"
    MEGATRON_PER_WORK_ITEM = "megatron_per_work_item"

    @classmethod
    def parse(cls, value: Any) -> "DpGradSync":
        if isinstance(value, DpGradSync):
            return value
        text = str(value).strip().lower()
        for member in cls:
            if member.value == text:
                return member
        raise WorkloadError(
            f"Unknown dp_grad_sync {value!r} (expected one of {[m.value for m in cls]})"
        )

@dataclass(frozen=True)
class ParallelDegrees:
    tp: int
    cp: int
    ep: int
    pp: int
    dp: int

    def cluster_size(self) -> int:
        return max(1, self.tp * self.cp * self.ep)

    def of(self, axis: AxisName) -> int:
        table = axis_sizes_from(self)
        if axis not in table:
            raise WorkloadError(
                f"Unknown parallelism axis {axis!r} (known: {sorted(table)})"
            )
        return table[axis]

    def __post_init__(self) -> None:
        for name in CANONICAL_AXES:
            value = self.__dict__[name]
            if not isinstance(value, int) or isinstance(value, bool):
                raise WorkloadError(f"ParallelDegrees.{name} must be an int (got {value!r})")
            if value < 1:
                raise WorkloadError(f"ParallelDegrees.{name} must be >= 1 (got {value})")

@dataclass(frozen=True)
class ModelShape:
    num_layers: int
    micro_batches: int
    model_type: str
    moe_layer_mask: Tuple[bool, ...] = ()

    def is_moe_layer(self, layer: LayerId) -> bool:
        if not self.moe_layer_mask:
            return False
        if layer < 0 or layer >= len(self.moe_layer_mask):
            raise WorkloadError(
                f"Layer {layer} is out of range for {len(self.moe_layer_mask)} layers"
            )
        return bool(self.moe_layer_mask[layer])

    @property
    def has_moe_layers(self) -> bool:
        return any(self.moe_layer_mask)

    def __post_init__(self) -> None:
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise WorkloadError(
                f"ModelShape.num_layers must be a positive int (got {self.num_layers!r}); "
                "this is the schedule.py:186 hazard — a wrong input object used to "
                "silently produce num_layers=0"
            )
        if not isinstance(self.micro_batches, int) or self.micro_batches <= 0:
            raise WorkloadError(
                f"ModelShape.micro_batches must be a positive int (got {self.micro_batches!r}); "
                "this is the schedule.py:186 hazard — a wrong input object used to "
                "silently produce mb=0"
            )
        if self.moe_layer_mask and len(self.moe_layer_mask) != self.num_layers:
            raise WorkloadError(
                f"ModelShape.moe_layer_mask has {len(self.moe_layer_mask)} entries "
                f"but num_layers is {self.num_layers}"
            )
        object.__setattr__(self, "model_type", str(self.model_type).lower())
        object.__setattr__(self, "moe_layer_mask", tuple(bool(v) for v in self.moe_layer_mask))

COMM_PLACEMENTS: Tuple[str, ...] = ("pre", "post")

def _declare_axes(
    key: CommKey,
    interconnect: Optional[Any],
    routing_mode: Optional[Any],
) -> Tuple[AxisName, ...]:
    if routing_mode is not None:
        from program.policies.routing import routing_for_mode

        return routing_for_mode(str(routing_mode)).routing_axes()
    if interconnect is None:
        raise WorkloadError(
            f"comm_metadata[{key!r}] has neither 'interconnect_type' nor "
            "'moe_routing_mode'; the communicator axis cannot be declared"
        )

    text = str(interconnect)
    if "*" in text:
        return tuple(part for part in text.split("*") if part)
    return (text,)

@dataclass(frozen=True)
class CommSpec:
    key: CommKey
    size_bytes: float
    kind: CollectiveType
    axes: Tuple[AxisName, ...]
    participants: int
    ga_required_every_cycle: bool = False
    tp_shard: bool = False
    placement: str = "post"

    local_comp_time: float = 0.0
    parallel_group: Optional[str] = None
    moe_component: Optional[str] = None
    moe_routing_mode: Optional[str] = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CollectiveType):
            raise WorkloadError(
                f"CommSpec[{self.key!r}].kind must be a CollectiveType "
                f"(got {type(self.kind).__name__})"
            )
        if self.placement not in COMM_PLACEMENTS:
            raise WorkloadError(
                f"CommSpec[{self.key!r}].placement must be one of {COMM_PLACEMENTS} "
                f"(got {self.placement!r})"
            )
        if not self.axes:
            raise WorkloadError(f"CommSpec[{self.key!r}] declares no communicator axes")
        if int(self.participants) < 1:
            raise WorkloadError(
                f"CommSpec[{self.key!r}].participants must be >= 1 (got {self.participants})"
            )
        object.__setattr__(self, "size_bytes", float(self.size_bytes))
        object.__setattr__(self, "participants", int(self.participants))
        object.__setattr__(self, "axes", tuple(str(a) for a in self.axes))

    @classmethod
    def from_legacy(cls, key: CommKey, data: Mapping[str, Any]) -> "CommSpec":
        if "type" not in data:
            raise WorkloadError(f"comm_metadata[{key!r}] has no 'type' (CollectiveType)")
        kind = data["type"]
        if not isinstance(kind, CollectiveType):
            raise WorkloadError(
                f"comm_metadata[{key!r}]['type'] must be a CollectiveType "
                f"(got {type(kind).__name__})"
            )
        interconnect = data.get("interconnect_type")
        routing_mode = data.get("moe_routing_mode")
        axes = _declare_axes(key, interconnect, routing_mode)

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
            key=key,
            size_bytes=float(data.get("size", 0) or 0.0),
            kind=kind,
            axes=axes,
            participants=int(data.get("participants", 1) or 1),
            ga_required_every_cycle=bool(data.get("ga_required_every_cycle", False)),
            tp_shard=bool(data.get("tp_shard", False)),
            placement=str(data.get("placement", "post")),
            local_comp_time=float(data.get("local_comp_time", 0) or 0.0),
            parallel_group=data.get("parallel_group"),
            moe_component=data.get("moe_component"),
            moe_routing_mode=(None if routing_mode is None else str(routing_mode)),
            extra=extra,
        )

    @classmethod
    def from_comm_meta(cls, meta: CommMeta) -> "CommSpec":
        if not isinstance(meta, CommMeta):
            raise WorkloadError(
                f"CommSpec.from_comm_meta expects a CommMeta (got {type(meta).__name__})"
            )
        if not isinstance(meta.kind, CollectiveType):
            raise WorkloadError(
                f"BlockTemplate comm entry {meta.name!r} has no CollectiveType "
                f"(got {type(meta.kind).__name__})"
            )
        return cls(
            key=meta.name,
            size_bytes=float(meta.size_bytes or 0.0),
            kind=meta.kind,
            axes=_declare_axes(meta.name, meta.interconnect, meta.moe_routing_mode),
            participants=max(1, int(meta.participants or 1)),
            ga_required_every_cycle=bool(meta.ga_required_every_cycle),
            tp_shard=bool(meta.tp_shard),
            placement=str(meta.placement),
            local_comp_time=float(meta.local_comp_time or 0.0),
            parallel_group=meta.parallel_group,
            moe_component=meta.moe_component,
            moe_routing_mode=(
                None if meta.moe_routing_mode is None else str(meta.moe_routing_mode)
            ),
            extra=dict(meta.extra or {}),
        )

class CommSpecTable(Mapping[CommKey, CommSpec]):
    __slots__ = ("_data",)

    def __init__(self, specs: Any = ()) -> None:
        data: Dict[CommKey, CommSpec] = {}
        items = specs.items() if isinstance(specs, Mapping) else specs
        for entry in items:
            if isinstance(entry, CommSpec):
                spec = entry
                key = spec.key
            else:
                key, spec = entry
            if not isinstance(spec, CommSpec):
                raise WorkloadError(
                    f"CommSpecTable[{key!r}] must be a CommSpec (got {type(spec).__name__})"
                )
            if key != spec.key:
                raise WorkloadError(
                    f"CommSpecTable key {key!r} disagrees with CommSpec.key {spec.key!r}"
                )
            if key in data:
                raise WorkloadError(f"Duplicate CommSpec key {key!r}")
            data[key] = spec
        self._data = data

    def __getitem__(self, key: CommKey) -> CommSpec:
        return self._data[key]

    def __iter__(self) -> Iterator[CommKey]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CommSpecTable({list(self._data)})"

    def require(self, key: CommKey) -> CommSpec:
        if key not in self._data:
            raise WorkloadError(
                f"Comm key {key!r} is not registered. Known keys: {sorted(self._data)}"
            )
        return self._data[key]

    def keys_with_axis(self, axis: AxisName) -> Tuple[CommKey, ...]:
        return tuple(k for k, s in self._data.items() if axis in s.axes)

    @classmethod
    def from_legacy(cls, raw: Mapping[str, Mapping[str, Any]]) -> "CommSpecTable":
        return cls(
            [(name, CommSpec.from_legacy(name, data)) for name, data in (raw or {}).items()]
        )

    @classmethod
    def from_block_template(cls, template: BlockTemplate) -> "CommSpecTable":
        return cls(
            [
                (name, CommSpec.from_comm_meta(meta))
                for name, meta in (template.comm_metadata or {}).items()
            ]
        )

DURATION_KEYS: Tuple[str, ...] = (
    "embedding_f",
    "embedding_b",
    "linear_softmax_f",
    "linear_softmax_b",
    "transformer_f",
    "transformer_b",
    "transformer_f_dense",
    "transformer_b_dense",
    "transformer_f_moe",
    "transformer_b_moe",

    "optimizer",
    "optimizer_moe",
)

_BLOCK_WRITEBACK_KEYS: Tuple[str, ...] = (
    "transformer_f",
    "transformer_b",
    "transformer_f_dense",
    "transformer_b_dense",
    "transformer_f_moe",
    "transformer_b_moe",
)

@dataclass(frozen=True)
class FrozenDurations(Mapping[str, float]):
    revision: int
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))

    def __getitem__(self, key: str) -> float:
        if key not in self.values:
            raise WorkloadError(
                f"Duration key {key!r} is not present. Known keys: {sorted(self.values)}"
            )
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def get_or(self, key: str, default: float = 0.0) -> float:
        value = self.values.get(key)
        return float(value) if value is not None else float(default)

class DurationTable:
    __slots__ = ("_values", "_revision")

    def __init__(self, values: Mapping[str, float]) -> None:
        cleaned: Dict[str, float] = {}
        for key, value in (values or {}).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric = float(value)
            if numeric < 0.0:
                raise WorkloadError(f"Duration {key!r} is negative ({numeric})")
            cleaned[str(key)] = numeric
        self._values = cleaned
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    def snapshot(self) -> FrozenDurations:
        return FrozenDurations(revision=self._revision, values=dict(self._values))

    def write_block_timings(
        self,
        *,
        dense_forward: Optional[float] = None,
        dense_backward: Optional[float] = None,
        moe_forward: Optional[float] = None,
        moe_backward: Optional[float] = None,
    ) -> int:
        for name, value in (
            ("dense_forward", dense_forward),
            ("dense_backward", dense_backward),
            ("moe_forward", moe_forward),
            ("moe_backward", moe_backward),
        ):
            if value is not None and float(value) < 0.0:
                raise WorkloadError(f"AstraSim transformer time {name} must be >= 0 (got {value})")

        staged = dict(self._values)
        if dense_forward is not None:
            staged["transformer_f"] = float(dense_forward)
            staged["transformer_f_dense"] = float(dense_forward)
        if dense_backward is not None:
            staged["transformer_b"] = float(dense_backward)
            staged["transformer_b_dense"] = float(dense_backward)
        if moe_forward is not None:
            staged["transformer_f_moe"] = float(moe_forward)
        if moe_backward is not None:
            staged["transformer_b_moe"] = float(moe_backward)

        self._values = staged
        self._revision += 1
        return self._revision

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"DurationTable(rev={self._revision}, keys={sorted(self._values)})"

@dataclass(frozen=True)
class RunPolicy:
    run_type: RunType
    grad_accum_cycle: GradAccumCycle
    dp_microbatch_mode: DpMicrobatchMode
    zero_stage: int
    pipeline_interleave: int = 1
    full_recomputation: bool = False
    dp_grad_sync: DpGradSync = DpGradSync.LEGACY

    def __post_init__(self) -> None:
        if not isinstance(self.run_type, RunType):
            raise WorkloadError("RunPolicy.run_type must be a RunType")
        if not isinstance(self.grad_accum_cycle, GradAccumCycle):
            raise WorkloadError("RunPolicy.grad_accum_cycle must be a GradAccumCycle")
        if not isinstance(self.dp_microbatch_mode, DpMicrobatchMode):
            raise WorkloadError("RunPolicy.dp_microbatch_mode must be a DpMicrobatchMode")
        if int(self.zero_stage) < 0:
            raise WorkloadError(f"RunPolicy.zero_stage must be >= 0 (got {self.zero_stage})")
        if int(self.pipeline_interleave) < 1:
            raise WorkloadError(
                f"RunPolicy.pipeline_interleave must be >= 1 (got {self.pipeline_interleave})"
            )
        if not isinstance(self.dp_grad_sync, DpGradSync):
            raise WorkloadError("RunPolicy.dp_grad_sync must be a DpGradSync")

    @property
    def include_backward(self) -> bool:
        return self.run_type is not RunType.INFERENCE

    @property
    def include_optimizer(self) -> bool:
        return self.grad_accum_cycle is not GradAccumCycle.NONFINAL

    @property
    def is_nonfinal_grad_accum_cycle(self) -> bool:
        return self.grad_accum_cycle is GradAccumCycle.NONFINAL

    def effective_dp(self, degrees: ParallelDegrees) -> int:
        return 1 if self.run_type is RunType.INFERENCE else max(1, degrees.dp)

    def retime_dp_count(self, degrees: ParallelDegrees) -> int:
        return self.effective_dp(degrees)

    def interleave_scale(self, degrees: ParallelDegrees, shape: ModelShape) -> float:
        v = int(self.pipeline_interleave)
        pp = int(degrees.pp)
        mb = int(shape.micro_batches)
        if v <= 1 or pp <= 1:
            return 1.0
        return (mb + (pp - 1) / float(v)) / float(mb + pp - 1)

@dataclass(frozen=True)
class OverlapSpec:
    parallelism_mode: Any
    by_axis: Mapping[AxisName, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        cleaned: Dict[AxisName, float] = {}
        for axis, value in (self.by_axis or {}).items():
            fraction = float(value)
            if fraction < 0.0:
                raise WorkloadError(f"OverlapSpec.by_axis[{axis!r}] must be >= 0 (got {fraction})")
            cleaned[str(axis)] = fraction
        object.__setattr__(self, "by_axis", cleaned)

    def fraction(self, axis: AxisName) -> float:
        value = self.by_axis.get(axis)
        return float(value) if value is not None else 0.0

    @property
    def mode_label(self) -> str:
        mode = self.parallelism_mode
        if isinstance(mode, Enum):
            return str(mode.value).lower()
        return str(mode).lower()

    @classmethod
    def from_legacy(
        cls,
        parallelism_mode: Any,
        *,
        tp_overlap: float = 0.0,
        tp_sp_overlap: float = 0.0,
        cp_overlap: float = 0.0,
    ) -> "OverlapSpec":
        probe = cls(parallelism_mode=parallelism_mode, by_axis={})
        label = probe.mode_label
        if label == "tensor_sequence":
            tp_value = float(tp_sp_overlap)
        elif label in ("tensor", "tensor_context", "tensor_context_hybrid"):
            tp_value = float(tp_overlap)
        else:
            tp_value = 0.0
        cp_value = float(cp_overlap) if label in ("context", "tensor_context", "tensor_context_hybrid") else 0.0
        by_axis: Dict[AxisName, float] = {}
        if tp_value > 0.0:
            by_axis["tp"] = tp_value
        if cp_value > 0.0:
            by_axis["cp"] = cp_value
        return cls(parallelism_mode=parallelism_mode, by_axis=by_axis)

@dataclass(frozen=True)
class BlockTemplates:
    dense: BlockTemplate
    moe: Optional[BlockTemplate] = None

    dense_comm: Optional[CommSpecTable] = None

    moe_comm: Optional[CommSpecTable] = None

    def __post_init__(self) -> None:
        if not isinstance(self.dense, BlockTemplate):
            raise WorkloadError("BlockTemplates.dense must be a BlockTemplate")
        if self.moe is not None and not isinstance(self.moe, BlockTemplate):
            raise WorkloadError("BlockTemplates.moe must be a BlockTemplate or None")
        for field_name, table in (("dense_comm", self.dense_comm), ("moe_comm", self.moe_comm)):
            if table is not None and not isinstance(table, CommSpecTable):
                raise WorkloadError(
                    f"BlockTemplates.{field_name} must be a CommSpecTable or None"
                )
        if self.dense_comm is None:
            object.__setattr__(
                self, "dense_comm", CommSpecTable.from_block_template(self.dense)
            )
        if self.moe is None:
            if self.moe_comm is not None:
                raise WorkloadError(
                    "BlockTemplates.moe_comm was supplied without a moe BlockTemplate"
                )
        elif self.moe_comm is None:
            object.__setattr__(
                self, "moe_comm", CommSpecTable.from_block_template(self.moe)
            )

    def template_for(self, is_moe_layer: bool) -> BlockTemplate:
        if is_moe_layer:
            if self.moe is None:
                raise WorkloadError(
                    "A MoE layer was requested but BlockTemplates.moe is None"
                )
            return self.moe
        return self.dense

    def comm_for(self, is_moe_layer: bool) -> CommSpecTable:
        if is_moe_layer:
            if self.moe_comm is None:
                raise WorkloadError(
                    "A MoE layer was requested but BlockTemplates.moe_comm is None"
                )
            return self.moe_comm
        if self.dense_comm is None:  # pragma: no cover - __post_init__ fills it
            raise WorkloadError("BlockTemplates.dense_comm is None")
        return self.dense_comm

    def tables(self) -> Tuple[CommSpecTable, ...]:
        out = [self.comm_for(False)]
        if self.moe_comm is not None:
            out.append(self.moe_comm)
        return tuple(out)

REQUIRED_MISC_KEYS: Tuple[str, ...] = (
    "dp_microbatch_mode",
    "dp_zero_stage",
    "dp_grad_sync",
    "full_recomputation",
    "moe_layer_mask",
    "model_type",
    "num_batch",
    "num_layer",
)

def _require_misc(misc_metadata: Mapping[str, Any], key: str) -> Any:
    if key not in misc_metadata:
        raise WorkloadError(
            f"misc_metadata is missing required key {key!r} "
            f"(required: {list(REQUIRED_MISC_KEYS)}; produced by "
            "train_timing._prepare_execution_graphs — INTERFACES §1.7). W1 "
            "forbids a silent default here."
        )
    return misc_metadata[key]

@dataclass(frozen=True)
class WorkloadSpec:
    degrees: ParallelDegrees
    shape: ModelShape
    run: RunPolicy

    comm: CommSpecTable
    blocks: BlockTemplates
    overlap: OverlapSpec
    layout: Optional[RankLayout]
    interconnect: Mapping[str, Tuple[float, float]]
    granularity_hint: Any

    durations: DurationTable = field(compare=False)

    activities: Mapping[str, OperationActivity] = field(default_factory=dict)
    rank_activities: Mapping[str, Tuple[OperationActivity, ...]] = field(default_factory=dict)
    rank_durations: Mapping[str, Tuple[float, ...]] = field(default_factory=dict)
    rank_collectives: Mapping[str, Tuple[Tuple[EmbeddedCollective, ...], ...]] = field(default_factory=dict)

    embedded_collectives: Mapping[str, Tuple[EmbeddedCollective, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.degrees, ParallelDegrees):
            raise WorkloadError("WorkloadSpec.degrees must be a ParallelDegrees")
        if not isinstance(self.shape, ModelShape):
            raise WorkloadError("WorkloadSpec.shape must be a ModelShape")
        if not isinstance(self.run, RunPolicy):
            raise WorkloadError("WorkloadSpec.run must be a RunPolicy")
        if not isinstance(self.comm, CommSpecTable):
            raise WorkloadError("WorkloadSpec.comm must be a CommSpecTable")
        if not isinstance(self.blocks, BlockTemplates):
            raise WorkloadError("WorkloadSpec.blocks must be a BlockTemplates")
        if not isinstance(self.overlap, OverlapSpec):
            raise WorkloadError("WorkloadSpec.overlap must be an OverlapSpec")
        if not isinstance(self.durations, DurationTable):
            raise WorkloadError("WorkloadSpec.durations must be a DurationTable")
        activities = dict(self.activities or {})
        invalid = {
            key: type(value).__name__
            for key, value in activities.items()
            if not isinstance(key, str) or not isinstance(value, OperationActivity)
        }
        if invalid:
            raise WorkloadError(
                "WorkloadSpec.activities must map string keys to OperationActivity "
                f"values (invalid: {invalid})"
            )
        if self.shape.has_moe_layers and self.blocks.moe is None:
            raise WorkloadError(
                "WorkloadSpec.shape declares MoE layers but blocks.moe is None"
            )
        object.__setattr__(self, "interconnect", dict(self.interconnect or {}))
        object.__setattr__(self, "activities", MappingProxyType(activities))
        rank_activities = {k: tuple(v) for k, v in self.rank_activities.items()}
        rank_durations = {k: tuple(float(x) for x in v) for k, v in self.rank_durations.items()}
        if any(len(v) != self.degrees.tp or not all(isinstance(x, OperationActivity) for x in v)
               for v in rank_activities.values()):
            raise WorkloadError("rank_activities requires one OperationActivity per TP rank")
        if any(len(v) != self.degrees.tp or any(not math.isfinite(x) or x < 0 for x in v)
               for v in rank_durations.values()):
            raise WorkloadError("rank_durations requires one nonnegative duration per TP rank")
        object.__setattr__(self, "rank_activities", MappingProxyType(rank_activities))
        object.__setattr__(self, "rank_durations", MappingProxyType(rank_durations))
        rank_collectives = {k: tuple(tuple(calls) for calls in ranks)
                            for k, ranks in self.rank_collectives.items()}
        if any(len(ranks) != self.degrees.tp or
               any(not isinstance(call, EmbeddedCollective) for calls in ranks for call in calls)
               for ranks in rank_collectives.values()):
            raise WorkloadError("rank_collectives requires one collective tuple per TP rank")
        object.__setattr__(self, "rank_collectives", MappingProxyType(rank_collectives))
        object.__setattr__(self, "embedded_collectives", MappingProxyType({
            key: tuple(calls) for key, calls in self.embedded_collectives.items()
        }))
        self._check_pipeline_block_key_disjointness()

    def _check_pipeline_block_key_disjointness(self) -> None:
        for table in self.blocks.tables():
            for key, block_spec in table.items():
                pipeline_spec = self.comm.get(key)
                if pipeline_spec is None:
                    continue
                if (
                    pipeline_spec.size_bytes != block_spec.size_bytes
                    or pipeline_spec.kind is not block_spec.kind
                    or pipeline_spec.axes != block_spec.axes
                    or pipeline_spec.participants != block_spec.participants
                ):
                    raise WorkloadError(
                        f"Comm key {key!r} is declared both pipeline-level and "
                        "block-level with different content "
                        f"(pipeline: {pipeline_spec.size_bytes} bytes / "
                        f"{pipeline_spec.kind.name} / {pipeline_spec.axes}; block: "
                        f"{block_spec.size_bytes} bytes / {block_spec.kind.name} / "
                        f"{block_spec.axes}). Pipeline and block comm namespaces "
                        "must not shadow each other."
                    )

    def is_moe_layer(self, layer: LayerId) -> bool:
        return self.shape.is_moe_layer(layer)

    def cluster_size(self) -> int:
        return self.degrees.cluster_size()

    def block_template(self, layer: LayerId) -> BlockTemplate:
        return self.blocks.template_for(self.shape.is_moe_layer(layer))

    def block_comm(self, layer: LayerId) -> CommSpecTable:
        return self.blocks.comm_for(self.shape.is_moe_layer(layer))

    def all_comm_specs(self) -> Tuple[CommSpec, ...]:
        out: List[CommSpec] = list(self.comm.values())
        for table in self.blocks.tables():
            out.extend(table.values())
        return tuple(out)

    def freeze(self) -> "FrozenWorkload":
        return FrozenWorkload(spec=self, durations=self.durations.snapshot())

    def with_(self, **changes: Any) -> "WorkloadSpec":
        return replace(self, **changes)

    @classmethod
    def from_timing(
        cls,
        *,
        tp: int,
        cp: int,
        ep: int,
        pp: int,
        dp: int,
        run_type: Any,
        comp_times: Mapping[str, Any],
        comm_metadata: Mapping[str, Mapping[str, Any]],
        misc_metadata: Mapping[str, Any],
        blocks: BlockTemplates,
        overlap: OverlapSpec,
        interconnect: Mapping[str, Tuple[float, float]],
        grad_accum_cycle: Any = GradAccumCycle.FINAL,
        pipeline_interleave: int = 1,
        layout: Optional[RankLayout] = None,
        granularity_hint: Any = None,
        durations: Optional[DurationTable] = None,
        activities: Optional[Mapping[str, OperationActivity]] = None,
        rank_activities: Optional[Mapping[str, Tuple[OperationActivity, ...]]] = None,
        rank_durations: Optional[Mapping[str, Tuple[float, ...]]] = None,
        rank_collectives: Optional[Mapping[str, Tuple[Tuple[EmbeddedCollective, ...], ...]]] = None,
        embedded_collectives: Optional[Mapping[str, Tuple[EmbeddedCollective, ...]]] = None,
    ) -> "WorkloadSpec":
        run = RunPolicy(
            run_type=RunType.parse(run_type),
            grad_accum_cycle=GradAccumCycle.parse(grad_accum_cycle),
            dp_microbatch_mode=DpMicrobatchMode.parse(
                _require_misc(misc_metadata, "dp_microbatch_mode")
            ),
            zero_stage=int(_require_misc(misc_metadata, "dp_zero_stage") or 0),
            pipeline_interleave=max(1, int(pipeline_interleave or 1)),
            full_recomputation=bool(_require_misc(misc_metadata, "full_recomputation")),
            dp_grad_sync=DpGradSync.parse(_require_misc(misc_metadata, "dp_grad_sync")),
        )
        declared = max(1, int(dp or 1))
        degrees = ParallelDegrees(
            tp=max(1, int(tp or 1)),
            cp=max(1, int(cp or 1)),
            ep=max(1, int(ep or 1)),
            pp=max(1, int(pp or 1)),
            dp=1 if run.run_type is RunType.INFERENCE else declared,
        )
        shape = ModelShape(
            num_layers=int(_require_misc(misc_metadata, "num_layer") or 0),
            micro_batches=int(_require_misc(misc_metadata, "num_batch") or 0),
            model_type=str(_require_misc(misc_metadata, "model_type") or ""),
            moe_layer_mask=tuple(
                bool(value)
                for value in (_require_misc(misc_metadata, "moe_layer_mask") or [])
            ),
        )
        return cls(
            degrees=degrees,
            shape=shape,
            run=run,
            comm=CommSpecTable.from_legacy(comm_metadata or {}),
            blocks=blocks,
            overlap=overlap,
            layout=layout,
            interconnect=dict(interconnect or {}),
            granularity_hint=granularity_hint,
            durations=(
                durations if durations is not None else DurationTable(comp_times or {})
            ),
            activities=dict(activities or {}),
            rank_activities=dict(rank_activities or {}),
            rank_durations=dict(rank_durations or {}),
            rank_collectives=dict(rank_collectives or {}),
            embedded_collectives=dict(embedded_collectives or {}),
        )

    def for_block(self, *, layout: Optional[RankLayout], moe: bool) -> "WorkloadSpec":
        if moe and self.blocks.moe is None:
            raise WorkloadError(
                "for_block(moe=True) needs BlockTemplates.moe, which is None"
            )
        blocks = (
            BlockTemplates(dense=self.blocks.moe, dense_comm=self.blocks.moe_comm)
            if moe
            else BlockTemplates(dense=self.blocks.dense, dense_comm=self.blocks.dense_comm)
        )
        return replace(
            self,
            degrees=ParallelDegrees(
                tp=self.degrees.tp,
                cp=self.degrees.cp,
                ep=self.degrees.ep,
                pp=1,
                dp=1,
            ),
            shape=ModelShape(
                num_layers=1,
                micro_batches=1,
                model_type=self.shape.model_type,
                moe_layer_mask=(),
            ),
            run=replace(self.run, full_recomputation=False),
            blocks=blocks,
            layout=layout,
        )

@dataclass(frozen=True)
class FrozenWorkload:
    spec: WorkloadSpec
    durations: FrozenDurations

    @property
    def degrees(self) -> ParallelDegrees:
        return self.spec.degrees

    @property
    def shape(self) -> ModelShape:
        return self.spec.shape

    @property
    def run(self) -> RunPolicy:
        return self.spec.run

    @property
    def comm(self) -> CommSpecTable:
        return self.spec.comm

    @property
    def blocks(self) -> BlockTemplates:
        return self.spec.blocks

    @property
    def layout(self) -> Optional[RankLayout]:
        return self.spec.layout

    @property
    def activities(self) -> Mapping[str, OperationActivity]:
        return self.spec.activities

    def is_moe_layer(self, layer: LayerId) -> bool:
        return self.spec.is_moe_layer(layer)

    def cluster_size(self) -> int:
        return self.spec.cluster_size()

    def block_comm(self, layer: LayerId) -> CommSpecTable:
        return self.spec.block_comm(layer)

    def all_comm_specs(self) -> Tuple[CommSpec, ...]:
        return self.spec.all_comm_specs()

def ceil_div(total: float, divisor: int) -> float:
    return float(math.ceil(float(total) / float(max(1, int(divisor)))))
