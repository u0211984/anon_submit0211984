from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Protocol, Sequence, Tuple

from program.policies.gradaccum import GradAccumPolicy
from program.axes import axes_needing_gradient_reduction
from program.types import CommKey, LayerId, StageId
from program.work import (
    AttachMode,
    Direction,
    SyncKey,
    SyncPhase,
    SyncRequirement,
    SyncSpread,
    VIA_ALL,
    VIA_DATA_FLOW,
    VIA_NON_DATA_FLOW,
    WorkItem,
    WorkKind,
    WorkSet,
    backward_entry,
    is_moe,
)
from program.workload import CommSpec, DpGradSync, FrozenWorkload, WorkloadError

__all__ = [
    "StagePartition",
    "ContiguousStages",
    "ShardingContext",
    "ShardingPolicy",
    "NullSharding",
    "DDP",
    "ZeRO1",
    "ZeRO2",
    "ZeRO3",
    "MegatronDP",
    "SHARDING_POLICIES",
    "sharding_policy_for",
]

class StagePartition(Protocol):
    def same_stage(self, a: WorkItem, b: WorkItem) -> bool: ...

@dataclass(frozen=True)
class ContiguousStages:
    stage_of_layer: Tuple[StageId, ...]
    num_stages: int

    @classmethod
    def legacy(cls, num_layers: int, pp: int) -> "ContiguousStages":
        stage_count = max(1, int(pp))
        base = num_layers // stage_count
        remainder = num_layers % stage_count
        mapping: List[StageId] = []
        for stage_idx in range(stage_count):
            count = base + (1 if stage_idx < remainder else 0)
            mapping.extend(StageId(stage_idx) for _ in range(count))
        return cls(stage_of_layer=tuple(mapping), num_stages=stage_count)

    @classmethod
    def contiguous(cls, num_layers: int, pp: int) -> "ContiguousStages":
        return cls.legacy(num_layers, pp)

    def stage_of(self, layer: LayerId) -> StageId:
        if layer < 0 or layer >= len(self.stage_of_layer):
            raise WorkloadError(
                f"Layer index {layer} is out of bounds for "
                f"{len(self.stage_of_layer)} layers."
            )
        return self.stage_of_layer[layer]

    def layers_of(self, stage: StageId) -> Tuple[LayerId, ...]:
        return tuple(
            layer
            for layer, assigned in enumerate(self.stage_of_layer)
            if int(assigned) == int(stage)
        )

    def min_layer(self, stage: StageId) -> Optional[LayerId]:
        layers = self.layers_of(stage)
        return layers[0] if layers else None

    def stage_of_work(self, work: WorkItem) -> StageId:
        if work.kind is WorkKind.EMBEDDING:
            return StageId(0)
        if work.kind is WorkKind.SOFTMAX:
            return StageId(self.num_stages - 1)
        if work.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE):
            return self.stage_of(work.layer)  # type: ignore[arg-type]
        if work.kind is WorkKind.OPTIMIZER:
            return StageId(int(work.stage))  # type: ignore[arg-type]
        raise WorkloadError(f"No stage rule for {work!r}")

    def same_stage(self, a: WorkItem, b: WorkItem) -> bool:
        return self.stage_of_work(a) == self.stage_of_work(b)

@dataclass(frozen=True)
class ShardingContext:
    fw: FrozenWorkload
    work: WorkSet
    grad_accum: GradAccumPolicy
    stages: StagePartition

    order: Tuple[WorkItem, ...] = ()

    def next_after(
        self, hosts: Sequence[WorkItem], kind: WorkKind, direction: Direction
    ) -> Optional[WorkItem]:
        if not self.order or not hosts:
            return None
        index = {item: position for position, item in enumerate(self.order)}
        try:
            after = max(index[host] for host in hosts)
        except KeyError:  # pragma: no cover - a host outside the schedule
            return None
        for item in self.order[after + 1 :]:
            if item.kind is kind and item.direction is direction:
                return item
        return None

    def spec_for(self, key: CommKey) -> Optional[CommSpec]:
        return self.fw.spec.comm.get(key)

    def emits(self, key: CommKey, microbatch: Optional[int]) -> Optional[CommSpec]:
        spec = self.spec_for(key)
        if spec is None:
            return None
        return spec if self.grad_accum.emits(spec, microbatch) else None

    def same_stage(self, a: WorkItem, b: WorkItem) -> bool:
        return self.stages.same_stage(a, b)

    def layer_item(self, direction: Direction, microbatch: int, layer: LayerId) -> WorkItem:
        return self.work.require(
            WorkKind.LAYER, direction, microbatch=microbatch, layer=layer
        )

    def backward_entry(self, microbatch: int, layer: LayerId) -> WorkItem:
        return backward_entry(
            self.layer_item(Direction.BACKWARD, microbatch, layer), self.work
        )

def _spread_for(spec: CommSpec) -> SyncSpread:
    return SyncSpread.PER_CLUSTER_RANK

class ShardingPolicy(Protocol):
    name: str

    def requirements(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Sequence[SyncRequirement]:
        pass

    def workload_requirements(
        self, ctx: ShardingContext
    ) -> Sequence[SyncRequirement]:
        pass

@dataclass(frozen=True)
class NullSharding(ShardingPolicy):
    name: str = "none"

    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        return ()

    def workload_requirements(self, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        return ()

_REDUCER_KEYS: Mapping[WorkKind, Tuple[CommKey, CommKey]] = {

    WorkKind.EMBEDDING: ("embedding", "embedding"),
    WorkKind.SOFTMAX: ("softmax", "softmax"),
    WorkKind.LAYER: ("transformer_dense", "transformer_moe"),
}

@dataclass(frozen=True)
class DDP(ShardingPolicy):
    name: str = "ddp"

    def reducer_key(self, work: WorkItem, ctx: ShardingContext) -> Optional[CommKey]:
        if work.direction is not Direction.BACKWARD:
            return None
        entry = _REDUCER_KEYS.get(work.kind)
        if entry is None:
            return None
        dense_key, moe_key = entry
        return moe_key if is_moe(work, ctx.fw) else dense_key

    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        key = self.reducer_key(work, ctx)
        if key is None:
            return ()
        spec = ctx.emits(key, work.microbatch)
        if spec is None:
            return ()
        reducer = SyncRequirement.from_spec(
            spec,
            key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer),
            place_on=work,
            mode=AttachMode.AFTER,
            anchors=(work,),
            spread=_spread_for(spec),

            is_reducer=True,
            origin=_REDUCER_ORIGIN[work.kind],
        )
        return (reducer,) + tuple(self.after_reducer(reducer, work, ctx))

    def after_reducer(
        self, reducer: SyncRequirement, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        return ()

    def workload_requirements(self, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        return ()

_REDUCER_ORIGIN: Mapping[WorkKind, str] = {
    WorkKind.EMBEDDING: "S2",
    WorkKind.LAYER: "S4",
    WorkKind.SOFTMAX: "S9",
}

@dataclass(frozen=True)
class ZeRO1(DDP):
    name: str = "zero1"

_ZERO2_GATHER_KEYS: Mapping[WorkKind, CommKey] = {
    WorkKind.EMBEDDING: "zero2_embedding_gather",
    WorkKind.LAYER: "zero2_transformer_gather",
    WorkKind.SOFTMAX: "zero2_softmax_gather",
}

_ZERO2_ORIGIN: Mapping[WorkKind, str] = {
    WorkKind.EMBEDDING: "S3",
    WorkKind.LAYER: "S5",
    WorkKind.SOFTMAX: "S10",
}

@dataclass(frozen=True)
class ZeRO2(DDP):
    name: str = "zero2"

    def after_reducer(
        self, reducer: SyncRequirement, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        key = _ZERO2_GATHER_KEYS.get(work.kind)
        if key is None:
            return ()
        spec = ctx.emits(key, work.microbatch)
        if spec is None:
            return ()
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer),
                place_on=work,
                mode=AttachMode.AFTER,
                anchors=(reducer.key,),
                spread=_spread_for(spec),

                after_update=True,
                origin=_ZERO2_ORIGIN[work.kind],
            ),
        )

ZERO3_EMBEDDING_KEY: CommKey = "zero3_embedding_gather"
ZERO3_TRANSFORMER_KEY: CommKey = "zero3_transformer_gather"
ZERO3_SOFTMAX_KEY: CommKey = "zero3_softmax_gather"

@dataclass(frozen=True)
class ZeRO3(ZeRO2):
    name: str = "zero3"
    prefetch_depth: int = 1

    def __post_init__(self) -> None:
        if int(self.prefetch_depth) < 1:
            raise WorkloadError(
                f"ZeRO3.prefetch_depth must be >= 1 (got {self.prefetch_depth})"
            )

    def _via(self, host: WorkItem, target: WorkItem, ctx: ShardingContext):
        return VIA_ALL if ctx.same_stage(host, target) else VIA_DATA_FLOW

    def _prefetch_attach(
        self,
        target: WorkItem,
        candidates: Sequence[WorkItem],
        ctx: ShardingContext,
    ) -> Tuple[WorkItem, AttachMode, Any]:
        for host in candidates:
            if ctx.same_stage(host, target):
                return host, AttachMode.PARALLEL_TO, self._via(host, target, ctx)
        return target, AttachMode.BEFORE, VIA_ALL

    def _forward_prefetch_window(
        self, microbatch: int, layer: LayerId, ctx: ShardingContext
    ) -> Tuple[WorkItem, ...]:
        return tuple(
            ctx.layer_item(Direction.FORWARD, microbatch, candidate)
            for candidate in range(layer - int(self.prefetch_depth), layer)
        )

    def _backward_prefetch_window(
        self, microbatch: int, layer: LayerId, ctx: ShardingContext
    ) -> Tuple[WorkItem, ...]:
        out: List[WorkItem] = []
        for ahead in range(layer + int(self.prefetch_depth), layer, -1):
            host = self._backward_host_at(microbatch, ahead, ctx)
            if host not in out:
                out.append(host)
        return tuple(out)

    def _cross_device_forward_hosts(
        self, microbatch: int, ctx: ShardingContext
    ) -> Tuple[WorkItem, ...]:
        hosts: List[WorkItem] = []
        for layer in range(1, ctx.fw.spec.shape.num_layers):
            host = ctx.layer_item(Direction.FORWARD, microbatch, layer - 1)
            target = ctx.layer_item(Direction.FORWARD, microbatch, layer)
            if not ctx.same_stage(host, target):
                hosts.append(host)
        return tuple(hosts)

    def _backward_host(self, microbatch: int, layer: LayerId, ctx: ShardingContext) -> WorkItem:
        return self._backward_host_at(
            microbatch, layer + int(self.prefetch_depth), ctx
        )

    def _backward_host_at(
        self, microbatch: int, ahead: LayerId, ctx: ShardingContext
    ) -> WorkItem:
        if ahead > ctx.fw.spec.shape.num_layers - 1:
            return ctx.work.require(
                WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=microbatch
            )
        return ctx.layer_item(Direction.BACKWARD, microbatch, ahead)

    def _cross_device_backward_hosts(
        self, microbatch: int, ctx: ShardingContext
    ) -> Tuple[WorkItem, ...]:
        hosts: List[WorkItem] = []
        for layer in reversed(range(ctx.fw.spec.shape.num_layers)):
            host = self._backward_host(microbatch, layer, ctx)
            target = ctx.backward_entry(microbatch, layer)
            if not ctx.same_stage(host, target):
                hosts.append(host)
        return tuple(hosts)

    def workload_requirements(self, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        out: List[SyncRequirement] = []

        spec = ctx.emits(ZERO3_EMBEDDING_KEY, 0)
        if spec is not None:
            entry = ctx.work.require(WorkKind.EMBEDDING, Direction.FORWARD, microbatch=0)
            out.append(
                SyncRequirement.from_spec(
                    spec,
                    key=SyncKey(ZERO3_EMBEDDING_KEY, SyncPhase.FWD_ENTRY, 0),
                    place_on=entry,
                    mode=AttachMode.BEFORE,
                    anchors=(entry,),
                    spread=_spread_for(spec),
                    origin="S1",
                )
            )

        spec = ctx.emits(ZERO3_SOFTMAX_KEY, 0)
        if spec is not None and ctx.fw.spec.run.include_backward:
            last_mb = ctx.fw.spec.shape.micro_batches - 1
            anchor = ctx.work.require(
                WorkKind.SOFTMAX, Direction.FORWARD, microbatch=last_mb
            )
            out.append(
                SyncRequirement.from_spec(
                    spec,
                    key=SyncKey(ZERO3_SOFTMAX_KEY, SyncPhase.BWD_ENTRY, 0),
                    place_on=ctx.work.require(
                        WorkKind.SOFTMAX, Direction.BACKWARD, microbatch=0
                    ),
                    mode=AttachMode.PARALLEL_TO,
                    anchors=(anchor,),
                    via=VIA_ALL,
                    spread=_spread_for(spec),
                    origin="S13",
                )
            )
        return tuple(out)

    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        out: List[SyncRequirement] = list(super().requirements(work, ctx))
        handler = _ZERO3_HANDLERS.get((work.kind, work.direction))
        if handler is not None:
            out.extend(handler(self, work, ctx))
        return tuple(out)

    def _forward_embedding(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        transformer = ctx.emits(ZERO3_TRANSFORMER_KEY, work.microbatch)
        if transformer is None:
            return ()
        out: List[SyncRequirement] = []
        b = work.microbatch

        for layer in range(min(int(self.prefetch_depth), ctx.fw.spec.shape.num_layers)):
            target = ctx.layer_item(Direction.FORWARD, b, layer)
            out.append(
                SyncRequirement.from_spec(
                    transformer,
                    key=SyncKey(ZERO3_TRANSFORMER_KEY, SyncPhase.FWD, b, layer),
                    place_on=work,
                    mode=AttachMode.PARALLEL_TO,
                    anchors=(work,),
                    via=self._via(work, target, ctx),
                    spread=_spread_for(transformer),
                    origin="S7",
                )
            )

        if b + 1 < ctx.fw.spec.shape.micro_batches:
            embedding = ctx.emits(ZERO3_EMBEDDING_KEY, b + 1)
            hosts = self._cross_device_forward_hosts(b, ctx)
            if embedding is not None and hosts:
                next_embedding = ctx.next_after(
                    hosts, WorkKind.EMBEDDING, Direction.FORWARD
                )
                out.append(
                    SyncRequirement.from_spec(
                        embedding,
                        key=SyncKey(ZERO3_EMBEDDING_KEY, SyncPhase.FWD, b + 1),
                        place_on=work,
                        mode=AttachMode.PARALLEL_TO,
                        anchors=hosts,
                        via=VIA_NON_DATA_FLOW,

                        consumers=() if next_embedding is None else (next_embedding,),
                        spread=_spread_for(embedding),
                        origin="S6",
                    )
                )
        return tuple(out)

    def _forward_layer(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        layer = int(work.layer)
        depth = int(self.prefetch_depth)
        if layer < depth:
            return ()
        spec = ctx.emits(ZERO3_TRANSFORMER_KEY, work.microbatch)
        if spec is None:
            return ()
        host, mode, via = self._prefetch_attach(
            work, self._forward_prefetch_window(work.microbatch, layer, ctx), ctx
        )
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_TRANSFORMER_KEY, SyncPhase.FWD, work.microbatch, layer),
                place_on=work,
                mode=mode,
                anchors=(host,),
                via=via,
                spread=_spread_for(spec),
                origin="S8",
            ),
        )

    def _forward_softmax(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        spec = ctx.emits(ZERO3_SOFTMAX_KEY, work.microbatch)
        if spec is None:
            return ()
        host = ctx.layer_item(
            Direction.FORWARD, work.microbatch, ctx.fw.spec.shape.num_layers - 1
        )
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_SOFTMAX_KEY, SyncPhase.FWD, work.microbatch),
                place_on=host,
                mode=AttachMode.PARALLEL_TO,
                anchors=(host,),
                via=VIA_ALL,
                spread=_spread_for(spec),
                origin="S11",
            ),
        )

    def _backward_softmax(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        b = int(work.microbatch)
        if b == 0:
            return ()
        spec = ctx.emits(ZERO3_SOFTMAX_KEY, b)
        if spec is None:
            return ()
        hosts = self._cross_device_backward_hosts(b, ctx)
        if not hosts:
            return ()

        consumer = ctx.next_after(hosts, WorkKind.SOFTMAX, Direction.BACKWARD)
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_SOFTMAX_KEY, SyncPhase.BWD, b),
                place_on=ctx.work.require(
                    WorkKind.SOFTMAX, Direction.FORWARD, microbatch=b
                ),
                mode=AttachMode.PARALLEL_TO,
                anchors=hosts,
                via=VIA_NON_DATA_FLOW,
                consumers=() if consumer is None else (consumer,),
                spread=_spread_for(spec),
                origin="S14",
            ),
        )

    def _backward_layer(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        spec = ctx.emits(ZERO3_TRANSFORMER_KEY, work.microbatch)
        if spec is None:
            return ()
        layer = int(work.layer)
        target = ctx.backward_entry(work.microbatch, layer)
        host, mode, via = self._prefetch_attach(
            target, self._backward_prefetch_window(work.microbatch, layer, ctx), ctx
        )
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_TRANSFORMER_KEY, SyncPhase.BWD, work.microbatch, layer),
                place_on=target,
                mode=mode,
                anchors=(host,),
                via=via,
                spread=_spread_for(spec),
                origin="S15",
            ),
        )

    def _backward_embedding(
        self, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        spec = ctx.emits(ZERO3_EMBEDDING_KEY, work.microbatch)
        if spec is None:
            return ()
        host = ctx.layer_item(Direction.BACKWARD, work.microbatch, 0)
        return (
            SyncRequirement.from_spec(
                spec,
                key=SyncKey(ZERO3_EMBEDDING_KEY, SyncPhase.BWD, work.microbatch),
                place_on=host,
                mode=AttachMode.PARALLEL_TO,
                anchors=(host,),
                via=VIA_ALL,
                spread=_spread_for(spec),
                origin="S16",
            ),
        )

_ZERO3_HANDLERS: Mapping[Tuple[WorkKind, Direction], Any] = {
    (WorkKind.EMBEDDING, Direction.FORWARD): ZeRO3._forward_embedding,
    (WorkKind.LAYER, Direction.FORWARD): ZeRO3._forward_layer,
    (WorkKind.SOFTMAX, Direction.FORWARD): ZeRO3._forward_softmax,
    (WorkKind.SOFTMAX, Direction.BACKWARD): ZeRO3._backward_softmax,
    (WorkKind.LAYER, Direction.BACKWARD): ZeRO3._backward_layer,
    (WorkKind.EMBEDDING, Direction.BACKWARD): ZeRO3._backward_embedding,
}

@dataclass(frozen=True)
class MoEZeRO:
    name: str = "zero_moe"

    def workload_requirements(self, ctx):
        return ()

    def requirements(self, work, ctx):
        if work.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE):
            label = "moe_layer" if is_moe(work, ctx.fw) else "dense_layer"
        elif work.kind is WorkKind.EMBEDDING:
            label = "embedding"
        elif work.kind is WorkKind.SOFTMAX:
            label = "softmax"
        else:
            return ()
        out = []
        for role in ("dense", "expert"):
            prefix = f"zm_{label}_{role}"
            if work.kind is WorkKind.RECOMPUTE or work.direction is Direction.FORWARD or work.direction is Direction.BACKWARD:
                spec = ctx.emits(prefix + "_gather", work.microbatch)
                if spec is not None:
                    phase = (SyncPhase.BWD_ENTRY if work.kind is WorkKind.RECOMPUTE else
                             SyncPhase.FWD if work.direction is Direction.FORWARD else SyncPhase.BWD)
                    out.append(SyncRequirement.from_spec(spec,
                        key=SyncKey(spec.key, phase, work.microbatch, work.layer),
                        place_on=work, mode=AttachMode.BEFORE, anchors=(work,),
                        spread=_spread_for(spec), origin="ZeRO+MoE"))
            if work.kind is WorkKind.RECOMPUTE or work.direction is not Direction.BACKWARD:
                continue
            cp = ctx.emits(prefix + "_a_cp", work.microbatch)
            rs = ctx.emits(prefix + "_b_rs", work.microbatch)
            predecessor = work
            if cp is not None:
                cp_key = SyncKey(cp.key, SyncPhase.GRAD, work.microbatch, work.layer)
                out.append(SyncRequirement.from_spec(cp, key=cp_key, place_on=work,
                    mode=AttachMode.AFTER, anchors=(work,), spread=_spread_for(cp),
                    is_reducer=rs is None, origin="ZeRO+MoE"))
                predecessor = cp_key
            if rs is not None:
                rs_key = SyncKey(rs.key, SyncPhase.GRAD, work.microbatch, work.layer)
                out.append(SyncRequirement.from_spec(rs, key=rs_key, place_on=work,
                    mode=AttachMode.AFTER, anchors=(predecessor,), spread=_spread_for(rs),
                    is_reducer=True, origin="ZeRO+MoE"))
                predecessor = rs_key
            ag = ctx.emits(prefix + "_c_update_ag", work.microbatch)

            if ag is not None and ctx.fw.spec.run.include_optimizer and work.microbatch == ctx.grad_accum.last_microbatch:
                out.append(SyncRequirement.from_spec(ag,
                    key=SyncKey(ag.key, SyncPhase.GRAD, work.microbatch, work.layer),
                    place_on=work, mode=AttachMode.AFTER, anchors=(predecessor,),
                    spread=_spread_for(ag), after_update=True, origin="ZeRO+MoE"))
        return tuple(out)

SHARDING_POLICIES: Mapping[int, Any] = {0: DDP, 1: ZeRO1, 2: ZeRO2, 3: ZeRO3}

_GRADIENT_DIRECTIONS = frozenset({Direction.BACKWARD})

@dataclass(frozen=True)
class MegatronDP(DDP):
    name: str = "megatron_dp"
    end_of_backward: bool = True
    distributed_optimizer: bool = True

    def _stage_last_backward(self, work: WorkItem, ctx: ShardingContext) -> WorkItem:
        if not ctx.order:
            raise WorkloadError("MegatronDP end_of_backward needs the schedule's work order")
        for item in reversed(ctx.order):
            if (item.direction in _GRADIENT_DIRECTIONS and item.kind in _REDUCER_KEYS
                    and ctx.same_stage(item, work)):
                return item
        raise WorkloadError(f"no backward work on the stage of {work!r}")

    def requirements(self, work: WorkItem, ctx: ShardingContext) -> Tuple[SyncRequirement, ...]:
        grouped = self._group_requirements(work, ctx)
        if grouped is not None:
            return grouped
        key = self.reducer_key(work, ctx)
        if key is None:
            return ()
        spec = ctx.emits(key, work.microbatch)
        if spec is None:
            return ()
        anchor = self._stage_last_backward(work, ctx) if self.end_of_backward else work
        reducer = SyncRequirement.from_spec(
            spec,
            key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer),
            place_on=anchor,
            mode=AttachMode.AFTER,
            anchors=(anchor,),
            spread=_spread_for(spec),
            is_reducer=True,
            origin=_REDUCER_ORIGIN[work.kind],
        )
        return (reducer,) + tuple(self.after_reducer(reducer, work, ctx))

    def after_reducer(
        self, reducer: SyncRequirement, work: WorkItem, ctx: ShardingContext
    ) -> Tuple[SyncRequirement, ...]:
        if not self.distributed_optimizer:
            return ()
        return ZeRO2.after_reducer(self, reducer, work, ctx)

    def _group_requirements(self, work: WorkItem, ctx: ShardingContext) -> Optional[Tuple[SyncRequirement, ...]]:
        if work.direction is not Direction.BACKWARD or work.kind not in _REDUCER_KEYS:
            return None
        label = {WorkKind.EMBEDDING: "embedding", WorkKind.SOFTMAX: "softmax"}.get(
            work.kind, "moe_layer" if is_moe(work, ctx.fw) else "dense_layer")
        keys = [f"mdp_{label}_{role}" for role in ("dense", "expert")]
        if not any(ctx.spec_for(key) is not None for key in keys):
            return None
        out = []
        for key in keys:
            spec = ctx.emits(key, work.microbatch)
            if spec is None:
                continue
            anchor = self._stage_last_backward(work, ctx) if self.end_of_backward else work
            reducer = SyncRequirement.from_spec(
                spec, key=SyncKey(key, SyncPhase.GRAD, work.microbatch, work.layer), place_on=anchor,
                mode=AttachMode.AFTER, anchors=(anchor,), spread=_spread_for(spec), is_reducer=True,
                origin=_REDUCER_ORIGIN[work.kind])
            out.append(reducer)
            gather = ctx.emits(key + "_gather", work.microbatch) if self.distributed_optimizer else None
            if gather is not None:
                out.append(SyncRequirement.from_spec(
                    gather, key=SyncKey(key + "_gather", SyncPhase.GRAD, work.microbatch, work.layer),
                    place_on=work, mode=AttachMode.AFTER, anchors=(reducer.key,), spread=_spread_for(gather),
                    after_update=True, origin=_ZERO2_ORIGIN[work.kind]))
        return tuple(out)

def _reduction_companion_degree(degrees: Any) -> int:
    total = 1
    for axis in axes_needing_gradient_reduction():
        total *= max(1, int(getattr(degrees, axis, 1) or 1))
    return total

def sharding_policy_for(run: Any, degrees: Any) -> ShardingPolicy:
    megatron_moe = run.dp_grad_sync is not DpGradSync.LEGACY and int(getattr(degrees, "ep", 1) or 1) > 1
    if (int(degrees.dp) * _reduction_companion_degree(degrees) <= 1 and not megatron_moe) or not run.include_backward:
        return NullSharding()
    stage = int(run.zero_stage)
    if stage < 0:
        raise WorkloadError(f"zero_stage must be >= 0 (got {stage})")
    sync = run.dp_grad_sync
    if sync is not DpGradSync.LEGACY:
        if stage > 1:
            raise WorkloadError("MegatronDP describes dp_zero_stage 0 (DDP) and 1 (distributed optimizer)")
        return MegatronDP(end_of_backward=sync is DpGradSync.MEGATRON_END_OF_BACKWARD,
                          distributed_optimizer=stage == 1)
    return SHARDING_POLICIES[min(stage, 3)]()
