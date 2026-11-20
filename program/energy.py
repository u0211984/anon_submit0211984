from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite
from typing import Any, Iterable, Mapping, Tuple

from activity import OperationActivity, ZERO_ACTIVITY
from program.ir import ComputeOp, Program

def _nonnegative(value: float, name: str) -> float:
    parsed = float(value)
    if not isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and >= 0 (got {value!r})")
    return parsed

def _activity_mismatches(
    actual: OperationActivity, expected: OperationActivity
) -> list[str]:
    pairs = (
        ("useful_tensor_flops", actual.useful_tensor_flops, expected.useful_tensor_flops),
        ("issued_tensor_flops", actual.issued_tensor_flops, expected.issued_tensor_flops),
        ("vector_flops", actual.vector_flops, expected.vector_flops),
        ("register.read_bytes", actual.register.read_bytes, expected.register.read_bytes),
        ("register.write_bytes", actual.register.write_bytes, expected.register.write_bytes),
        ("smem.read_bytes", actual.smem.read_bytes, expected.smem.read_bytes),
        ("smem.write_bytes", actual.smem.write_bytes, expected.smem.write_bytes),
        ("smem_staging_bytes", actual.smem_staging_bytes, expected.smem_staging_bytes),
        ("smem_staging_energy_bytes", actual.smem_staging_energy_bytes, expected.smem_staging_energy_bytes),
        ("l2.read_bytes", actual.l2.read_bytes, expected.l2.read_bytes),
        ("l2.write_bytes", actual.l2.write_bytes, expected.l2.write_bytes),
        ("hbm.read_bytes", actual.hbm.read_bytes, expected.hbm.read_bytes),
        ("hbm.write_bytes", actual.hbm.write_bytes, expected.hbm.write_bytes),
        ("l2_bypass_bytes", actual.l2_bypass_bytes, expected.l2_bypass_bytes),
        ("atomic_rmw_read_bytes", actual.atomic_rmw_read_bytes, expected.atomic_rmw_read_bytes),
    )
    return [
        name
        for name, observed, reference in pairs
        if not isclose(observed, reference, rel_tol=1e-12, abs_tol=1e-12)
    ]

@dataclass(frozen=True)
class ActivityEnergy:
    tensor_j: float = 0.0
    vector_j: float = 0.0
    register_j: float = 0.0
    smem_j: float = 0.0
    l2_j: float = 0.0
    hbm_j: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "tensor_j",
            "vector_j",
            "register_j",
            "smem_j",
            "l2_j",
            "hbm_j",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))

    @property
    def compute_j(self) -> float:
        return self.tensor_j + self.vector_j

    @property
    def memory_j(self) -> float:
        return self.register_j + self.smem_j + self.l2_j + self.hbm_j

    @property
    def total_j(self) -> float:
        return self.compute_j + self.memory_j

    def scale(self, factor: float) -> "ActivityEnergy":
        factor = _nonnegative(factor, "factor")
        return ActivityEnergy(
            tensor_j=self.tensor_j * factor,
            vector_j=self.vector_j * factor,
            register_j=self.register_j * factor,
            smem_j=self.smem_j * factor,
            l2_j=self.l2_j * factor,
            hbm_j=self.hbm_j * factor,
        )

    def __add__(self, other: "ActivityEnergy") -> "ActivityEnergy":
        if not isinstance(other, ActivityEnergy):
            return NotImplemented
        return ActivityEnergy(
            tensor_j=self.tensor_j + other.tensor_j,
            vector_j=self.vector_j + other.vector_j,
            register_j=self.register_j + other.register_j,
            smem_j=self.smem_j + other.smem_j,
            l2_j=self.l2_j + other.l2_j,
            hbm_j=self.hbm_j + other.hbm_j,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "tensor_j": self.tensor_j,
            "vector_j": self.vector_j,
            "register_j": self.register_j,
            "smem_j": self.smem_j,
            "l2_j": self.l2_j,
            "hbm_j": self.hbm_j,
            "compute_j": self.compute_j,
            "memory_j": self.memory_j,
            "dynamic_j": self.total_j,
        }

ZERO_ACTIVITY_ENERGY = ActivityEnergy()

@dataclass(frozen=True)
class ProgramExecution:
    program: Program
    wall_time_s: float
    repetitions: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.program, Program):
            raise TypeError("ProgramExecution.program must be a Program")
        object.__setattr__(
            self, "wall_time_s", _nonnegative(self.wall_time_s, "wall_time_s")
        )
        if isinstance(self.repetitions, bool) or int(self.repetitions) < 1:
            raise ValueError("ProgramExecution.repetitions must be an integer >= 1")
        object.__setattr__(self, "repetitions", int(self.repetitions))

@dataclass(frozen=True)
class ProgramEnergyResult:
    activity: OperationActivity = ZERO_ACTIVITY
    dynamic: ActivityEnergy = ZERO_ACTIVITY_ENERGY
    static_j: float = 0.0
    wall_time_s: float = 0.0
    device_count: int = 0
    by_operation_activity: Mapping[str, OperationActivity] = field(default_factory=dict)
    by_operation: Mapping[str, ActivityEnergy] = field(default_factory=dict)

    busy_j: float = 0.0
    busy_s: float = 0.0

    accounting_by_operation_activity: Mapping[str, OperationActivity] | None = None
    accounting_by_operation: Mapping[str, ActivityEnergy] | None = None
    by_rank_activity: Mapping[str, OperationActivity] = field(default_factory=dict)
    by_rank: Mapping[str, ActivityEnergy] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.activity, OperationActivity):
            raise TypeError("ProgramEnergyResult.activity must be OperationActivity")
        if not isinstance(self.dynamic, ActivityEnergy):
            raise TypeError("ProgramEnergyResult.dynamic must be ActivityEnergy")
        object.__setattr__(self, "static_j", _nonnegative(self.static_j, "static_j"))
        object.__setattr__(self, "busy_j", _nonnegative(self.busy_j, "busy_j"))
        object.__setattr__(self, "busy_s", _nonnegative(self.busy_s, "busy_s"))
        object.__setattr__(
            self, "wall_time_s", _nonnegative(self.wall_time_s, "wall_time_s")
        )
        if isinstance(self.device_count, bool) or int(self.device_count) < 0:
            raise ValueError("ProgramEnergyResult.device_count must be an integer >= 0")
        object.__setattr__(self, "device_count", int(self.device_count))
        by_operation_activity = dict(self.by_operation_activity)
        if not all(
            isinstance(value, OperationActivity)
            for value in by_operation_activity.values()
        ):
            raise TypeError(
                "ProgramEnergyResult.by_operation_activity values must be "
                "OperationActivity"
            )
        object.__setattr__(self, "by_operation_activity", by_operation_activity)
        if by_operation_activity:
            attributed_activity = sum(
                by_operation_activity.values(), ZERO_ACTIVITY
            )
            mismatched_activity = _activity_mismatches(
                attributed_activity, self.activity
            )
            if mismatched_activity:
                raise ValueError(
                    "operation activity does not reconcile with total activity "
                    f"for {mismatched_activity}: "
                    f"operations={attributed_activity.to_dict()}, "
                    f"total={self.activity.to_dict()}"
                )
        by_operation = dict(self.by_operation)
        if not all(isinstance(value, ActivityEnergy) for value in by_operation.values()):
            raise TypeError("ProgramEnergyResult.by_operation values must be ActivityEnergy")
        object.__setattr__(self, "by_operation", by_operation)
        attributed = sum(by_operation.values(), ZERO_ACTIVITY_ENERGY)
        component_pairs = (
            ("tensor_j", attributed.tensor_j, self.dynamic.tensor_j),
            ("vector_j", attributed.vector_j, self.dynamic.vector_j),
            ("register_j", attributed.register_j, self.dynamic.register_j),
            ("smem_j", attributed.smem_j, self.dynamic.smem_j),
            ("l2_j", attributed.l2_j, self.dynamic.l2_j),
            ("hbm_j", attributed.hbm_j, self.dynamic.hbm_j),
        )
        mismatched = [
            name
            for name, actual, expected in component_pairs

            if not isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-18)
        ]
        if mismatched:
            raise ValueError(
                "operation energy does not reconcile with dynamic components "
                f"{mismatched}: operations={attributed.to_dict()}, "
                f"dynamic={self.dynamic.to_dict()}"
            )

    @property
    def total_j(self) -> float:
        return self.dynamic.total_j + self.static_j + self.busy_j

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy_scope": "system",
            "device_count": self.device_count,
            "wall_time_s": self.wall_time_s,
            "activity": self.activity.to_dict(),
            "dynamic": self.dynamic.to_dict(),
            "static_j": self.static_j,
            **({"busy_j": self.busy_j, "busy_s": self.busy_s} if self.busy_j else {}),
            "total_j": self.total_j,
            "by_operation_activity": {
                name: value.to_dict()
                for name, value in self.by_operation_activity.items()
            },
            "by_operation": {
                name: value.to_dict() for name, value in self.by_operation.items()
            },
            **({"by_rank_activity": {name: value.to_dict() for name, value in self.by_rank_activity.items()},
                "by_rank": {name: value.to_dict() for name, value in self.by_rank.items()}}
               if self.by_rank_activity else {}),
        }

def tensor_activity_energy(activity: OperationActivity, core: Any) -> float:
    tensor = float(core.tensor_energy)
    padded = float(getattr(core, "padded_tensor_energy", tensor))
    useful = min(activity.useful_tensor_flops, activity.issued_tensor_flops)
    return useful * tensor + (activity.issued_tensor_flops - useful) * padded

def price_activity(activity: OperationActivity, hw_config: Any) -> ActivityEnergy:
    if not isinstance(activity, OperationActivity):
        raise TypeError("activity must be OperationActivity")
    tech = hw_config.tech_config
    return ActivityEnergy(

        tensor_j=tensor_activity_energy(activity, tech.core),
        vector_j=activity.vector_flops * float(tech.core.vector_energy),
        register_j=activity.register.total_bytes
        * float(tech.SRAMR.dynamic_energy_per_byte_value),

        smem_j=(activity.smem.total_bytes + activity.smem_staging_energy_bytes)
        * float(tech.SRAML1.dynamic_energy_per_byte_value),
        l2_j=activity.l2.total_bytes
        * float(tech.SRAML2.dynamic_energy_per_byte_value),
        hbm_j=activity.hbm.total_bytes
        * float(tech.DRAM.dynamic_energy_per_byte_value),
    )

def _require_flat(program: Program) -> None:
    granularity = str(program.meta.misc.get("granularity", "")).lower()
    if granularity != "flat":
        raise ValueError(
            "GPU energy replay requires a FLAT Program so physical placement and "
            f"execution multiplicity are explicit (got {granularity or 'unspecified'!r})"
        )

def _physical_device_count(program: Program) -> int:
    return len(program.compute_devices()) * max(1, int(program.dp_count))

def l2_bypassed(name: str, activity: OperationActivity) -> OperationActivity:
    if activity.l2.total_bytes == 0.0 and activity.hbm.total_bytes > 0.0:
        raise ValueError(f"l2_bypass: operation {name!r} has DRAM bytes but no L2 bytes, so its L1-miss traffic "
                         "is not described")
    return activity.l2_bypassed()

def _l2_bypassed(op: ComputeOp, activity: OperationActivity) -> OperationActivity:
    return l2_bypassed(op.name, activity)

def _program_activity(
    program: Program, *, l2_bypass: bool = False,
) -> Tuple[OperationActivity, dict[str, OperationActivity],
           dict[str, OperationActivity] | None, dict[str, OperationActivity]]:
    _require_flat(program)
    dp_count = max(1, int(program.dp_count))
    total = ZERO_ACTIVITY
    by_operation: dict[str, OperationActivity] = {}
    accounting: dict[str, OperationActivity] = {}
    has_invocations = False
    by_rank = {}
    for op in program.ops:
        if not isinstance(op, ComputeOp):
            continue
        if op.launch_shadow:
            continue
        activity = (_l2_bypassed(op, op.activity) if l2_bypass else op.activity).scaled(dp_count)
        if program.meta.misc.get("vtp_per_rank") or program.meta.misc.get("megatron_per_rank"):
            rank = str(int(op.device))
            by_rank[rank] = by_rank.get(rank, ZERO_ACTIVITY) + activity
        total = total + activity
        accounting[op.name] = accounting.get(op.name, ZERO_ACTIVITY) + activity
        has_invocations |= op.invocation_activity is not None
        reported = op.invocation_activity if op.invocation_activity is not None else op.activity
        if l2_bypass:
            reported = _l2_bypassed(op, reported)
        by_operation[op.name] = by_operation.get(op.name, ZERO_ACTIVITY) + reported.scaled(dp_count)
    return total, by_operation, accounting if has_invocations else None, by_rank

def replay_program_sequence(
    executions: Iterable[ProgramExecution], hw_config: Any
) -> ProgramEnergyResult:
    entries = tuple(executions)
    if not entries:
        raise ValueError("replay_program_sequence requires at least one execution")

    from sim_settings import l2_bypass as _l2_bypass_switch
    l2_bypass = _l2_bypass_switch(hw_config)
    device_count: int | None = None
    total_activity = ZERO_ACTIVITY
    by_operation_activity: dict[str, OperationActivity] = {}
    accounting_by_operation_activity: dict[str, OperationActivity] = {}
    by_rank_activity: dict[str, OperationActivity] = {}
    has_invocations = False
    wall_time_s = 0.0
    for execution in entries:
        count = _physical_device_count(execution.program)
        if device_count is None:
            device_count = count
        elif count != device_count:
            raise ValueError(
                "all execution segments must use the same physical GPU count "
                f"({device_count} != {count})"
            )
        activity, by_operation, accounting, rank_activity = _program_activity(execution.program, l2_bypass=l2_bypass)
        has_invocations |= accounting is not None
        factor = execution.repetitions
        total_activity = total_activity + activity.scaled(factor)
        wall_time_s += execution.wall_time_s * factor
        for rank, value in rank_activity.items():
            by_rank_activity[rank] = by_rank_activity.get(rank, ZERO_ACTIVITY) + value.scaled(factor)
        for name, value in by_operation.items():
            by_operation_activity[name] = (
                by_operation_activity.get(name, ZERO_ACTIVITY) + value.scaled(factor)
            )
        for name, value in (accounting if accounting is not None else by_operation).items():
            accounting_by_operation_activity[name] = (
                accounting_by_operation_activity.get(name, ZERO_ACTIVITY) + value.scaled(factor)
            )

    dynamic = price_activity(total_activity, hw_config)
    by_operation_energy = {
        name: price_activity(activity, hw_config)
        for name, activity in sorted(by_operation_activity.items())
    }
    static_power_w = _nonnegative(
        getattr(hw_config, "static_power_w", 0.0) or 0.0,
        "hardware_config.static_power_w",
    )
    resolved_devices = int(device_count or 0)
    busy_power_w = _nonnegative(
        getattr(hw_config, "busy_power_w", 0.0) or 0.0,
        "hardware_config.busy_power_w",
    )
    return ProgramEnergyResult(
        activity=total_activity,
        dynamic=dynamic,
        static_j=static_power_w * resolved_devices * wall_time_s,
        wall_time_s=wall_time_s,
        device_count=resolved_devices,
        by_operation_activity=by_operation_activity,
        by_operation=by_operation_energy,
        by_rank_activity=by_rank_activity,
        by_rank={rank: price_activity(value, hw_config) for rank, value in by_rank_activity.items()},
        accounting_by_operation_activity=(accounting_by_operation_activity if has_invocations else None),
        accounting_by_operation=({name: price_activity(activity, hw_config)
                                  for name, activity in sorted(accounting_by_operation_activity.items())}
                                 if has_invocations else None),
        busy_j=busy_power_w * resolved_devices * wall_time_s,
        busy_s=wall_time_s if busy_power_w else 0.0,
    )

def replay_program_energy(
    program: Program,
    hw_config: Any,
    *,
    wall_time_s: float,
    repetitions: int = 1,
) -> ProgramEnergyResult:
    return replay_program_sequence(
        (
            ProgramExecution(
                program=program,
                wall_time_s=wall_time_s,
                repetitions=repetitions,
            ),
        ),
        hw_config,
    )

__all__ = [
    "ActivityEnergy",
    "ProgramEnergyResult",
    "ProgramExecution",
    "price_activity",
    "replay_program_energy",
    "replay_program_sequence",
]
