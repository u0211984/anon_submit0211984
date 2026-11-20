from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isclose
import json
import os
import re
from typing import Any, Mapping, Sequence

from activity import ZERO_ACTIVITY
from program.energy import ActivityEnergy, ProgramEnergyResult, ZERO_ACTIVITY_ENERGY
from program.network_energy import CommunicationEnergyResult

@dataclass(frozen=True)
class EnergyCoverage:
    gpu_activity: str = "complete"
    gpu_static: str = "complete"
    communication: str = "unavailable"
    missing: tuple[str, ...] = ("communication",)

    def __post_init__(self) -> None:
        valid = {"complete", "partial", "unavailable"}
        for name in ("gpu_activity", "gpu_static", "communication"):
            value = getattr(self, name)
            if value not in valid:
                raise ValueError(f"invalid {name} coverage {value!r}")

    @property
    def complete(self) -> bool:
        return not self.missing and all(
            value == "complete"
            for value in (self.gpu_activity, self.gpu_static, self.communication)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_activity": self.gpu_activity,
            "gpu_static": self.gpu_static,
            "communication": self.communication,
            "missing": list(self.missing),
            "complete": self.complete,
        }

@dataclass(frozen=True)
class ExecutionEnergyResult:
    gpu: ProgramEnergyResult
    communication: CommunicationEnergyResult = field(
        default_factory=CommunicationEnergyResult
    )
    coverage: EnergyCoverage = field(default_factory=EnergyCoverage)
    execution_mode: str = "unknown"

    collective_launch_records: tuple[Mapping[str, Any], ...] = ()
    tp_allreduce_records: tuple[Mapping[str, Any], ...] = ()

    @property
    def total_j(self) -> float:
        return self.gpu.total_j + self.communication.total_j

    @property
    def dynamic_j(self) -> float:
        return self.gpu.dynamic.total_j + self.communication.total_j

    def component_energy_report(self, hardware: Any = None) -> dict[str, Any]:
        from program.component_energy import component_energy_report
        return component_energy_report(self, hardware)

    def to_report(self, *, phases: Mapping[str, "ExecutionEnergyResult"] | None = None) -> dict[str, Any]:
        phases = dict(phases or {})
        scale = 1e6
        resource_breakdown = {
            "tensor": self.gpu.dynamic.tensor_j * scale,
            "vector": self.gpu.dynamic.vector_j * scale,
            "register": self.gpu.dynamic.register_j * scale,
            "smem": self.gpu.dynamic.smem_j * scale,
            "l2": self.gpu.dynamic.l2_j * scale,
            "hbm": self.gpu.dynamic.hbm_j * scale,
            "communication": self.communication.total_j * scale,
            "static": self.gpu.static_j * scale,
        }
        breakdown = {
            "compute": self.gpu.dynamic.compute_j * scale,
            "memory": self.gpu.dynamic.memory_j * scale,
            "communication": self.communication.total_j * scale,
            "static": self.gpu.static_j * scale,
        }
        if self.gpu.busy_j:
            resource_breakdown["busy"] = self.gpu.busy_j * scale
            breakdown["busy"] = self.gpu.busy_j * scale
        reconciled = sum(breakdown.values())
        if not isclose(reconciled, self.total_j * scale, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(
                "execution energy does not reconcile: "
                f"breakdown={reconciled}, total={self.total_j * scale}"
            )
        resource_total = sum(resource_breakdown.values())
        if not isclose(
            resource_total, self.total_j * scale, rel_tol=1e-12, abs_tol=1e-6
        ):
            raise ValueError(
                "resource energy does not reconcile: "
                f"breakdown={resource_total}, total={self.total_j * scale}"
            )

        def _operation_rows() -> list[dict[str, Any]]:
            rows = []
            for name, energy in sorted(self.gpu.by_operation.items()):
                row = {"name": name}
                row.update({key.replace("_j", "_uj"): value * scale for key, value in energy.to_dict().items()})
                activity = self.gpu.by_operation_activity.get(name)
                if activity is not None:
                    row["activity"] = activity.to_dict()
                rows.append(row)
            return rows

        pricing_semantics = {
            "gpu_dynamic": "physical_activity_times_hardware_coefficient",
            "tensor_activity": "issued_flops",
            "vector_activity": "logical_flops",
            "memory_activity": "read_plus_write_bytes_per_level",
            "gpu_static": "static_power_times_device_count_times_completion_time",
        }

        report = {
            "schema_version": 7,
            "component_energy": self.component_energy_report(),
            "energy_unit": "uJ",
            "energy_scope": "system",
            "operation_energy_scope": "system",
            "execution_mode": self.execution_mode,
            "device_count": self.gpu.device_count,
            "completion_time_s": self.gpu.wall_time_s,
            "total_energy_uj": self.total_j * scale,
            "modeled_energy_uj": self.total_j * scale,
            "total_energy_complete": self.coverage.complete,
            "unmodeled_energy_components": list(self.coverage.missing),
            "coverage": self.coverage.to_dict(),
            "per_device_energy_uj": (
                self.total_j * scale / self.gpu.device_count
                if self.gpu.device_count
                else 0.0
            ),
            "static_idle_topup_uj": 0.0,
            "network_energy_uj": self.communication.total_j * scale,
            "breakdown_uj": breakdown,
            "resource_breakdown_uj": resource_breakdown,
            "pricing_semantics": pricing_semantics,
            "gpu_activity": self.gpu.activity.to_dict(),
            "by_operation": _operation_rows(),
            "communication": self.communication.to_dict(),
            "network_energy_by_phase_uj": {
                name: phase.communication.total_j * scale for name, phase in phases.items()
            },
            "phase_gpu_activity": {
                name: phase.gpu.activity.to_dict() for name, phase in phases.items()
            },
            "phase_energy_uj": {
                name: phase.total_j * scale for name, phase in phases.items()
            },
        }
        if phases:
            report["phases"] = {
                name: phase.to_report()["breakdown_uj"] for name, phase in phases.items()
            }
            if any(phase.gpu.by_rank_activity for phase in phases.values()):
                report["phase_per_rank"] = {
                    name: {rank: {"activity": activity.to_dict(),
                                  "dynamic_energy_j": phase.gpu.by_rank[rank].to_dict()}
                           for rank, activity in phase.gpu.by_rank_activity.items()}
                    for name, phase in phases.items()
                }
        if self.gpu.by_rank_activity:
            report["per_rank"] = {
                rank: {"activity": activity.to_dict(), "dynamic_energy_j": self.gpu.by_rank[rank].to_dict()}
                for rank, activity in self.gpu.by_rank_activity.items()
            }
        if self.collective_launch_records:
            report["collective_launch"] = _launch_summary(self.collective_launch_records)
            report["phase_collective_launch"] = {
                name: _launch_summary(phase.collective_launch_records)
                for name, phase in phases.items()
            } if phases else {"training": report["collective_launch"]}
        if self.tp_allreduce_records:
            report["tp_allreduce"] = _tp_allreduce_summary(self.tp_allreduce_records)
            report["phase_tp_allreduce"] = {
                name: _tp_allreduce_summary(phase.tp_allreduce_records)
                for name, phase in phases.items()
            } if phases else {"training": report["tp_allreduce"]}
        return report

def _tp_allreduce_summary(records):
    counts = {"nccl": 0, "custom_one_shot": 0, "custom_two_shot": 0}
    for row in records:
        algorithm = row["algorithm"]
        counts[algorithm] = counts.get(algorithm, 0) + row["count"] * row.get("multiplicity", 1.0)
    return {"counts": counts, "records": list(records)}

def _launch_summary(records):
    return {
        "count": sum(row["count"] * row.get("multiplicity", 1.0) for row in records),
        "exact_seconds": sum(row["exact_seconds"] * row.get("multiplicity", 1.0) for row in records),
        "astrasim_emitted_seconds": sum((row["emitted_seconds"] or 0.0) * row.get("multiplicity", 1.0) for row in records),
        "records": list(records),
    }

def combine_execution_energy(
    parts: Sequence[tuple[ExecutionEnergyResult, float]],
    *,
    execution_mode: str | None = None,
) -> ExecutionEnergyResult:
    if not parts:
        raise ValueError("combine_execution_energy requires at least one result")
    device_count = parts[0][0].gpu.device_count
    activity = ZERO_ACTIVITY
    dynamic = ZERO_ACTIVITY_ENERGY
    static_j = 0.0
    busy_j = 0.0
    busy_s = 0.0
    wall_time_s = 0.0
    by_operation: dict[str, ActivityEnergy] = {}
    by_operation_activity = {}
    by_rank_activity = {}
    by_rank = {}
    accounting_by_operation = {}
    accounting_by_operation_activity = {}
    has_invocations = any(result.gpu.accounting_by_operation is not None for result, _ in parts)
    communication = CommunicationEnergyResult()
    missing: set[str] = set()
    statuses = {"gpu_activity": set(), "gpu_static": set(), "communication": set()}
    launch_records = []
    tp_allreduce_records = []
    for result, factor_raw in parts:
        factor = float(factor_raw)
        if factor < 0.0:
            raise ValueError("execution-energy factor must be nonnegative")
        if result.gpu.device_count != device_count:
            raise ValueError("execution-energy parts must use the same physical GPU count")
        activity = activity + result.gpu.activity.scaled(factor)
        dynamic = dynamic + result.gpu.dynamic.scale(factor)
        static_j += result.gpu.static_j * factor
        busy_j += result.gpu.busy_j * factor
        busy_s += result.gpu.busy_s * factor
        wall_time_s += result.gpu.wall_time_s * factor
        for name, energy in result.gpu.by_operation.items():
            by_operation[name] = by_operation.get(name, ZERO_ACTIVITY_ENERGY) + energy.scale(factor)
        for rank, value in result.gpu.by_rank_activity.items():
            by_rank_activity[rank] = by_rank_activity.get(rank, ZERO_ACTIVITY) + value.scaled(factor)
        for rank, value in result.gpu.by_rank.items():
            by_rank[rank] = by_rank.get(rank, ZERO_ACTIVITY_ENERGY) + value.scale(factor)
        for name, op_activity in result.gpu.by_operation_activity.items():
            by_operation_activity[name] = (
                by_operation_activity.get(name, ZERO_ACTIVITY)
                + op_activity.scaled(factor)
            )
        if has_invocations:
            for name, energy in (result.gpu.accounting_by_operation or result.gpu.by_operation).items():
                accounting_by_operation[name] = accounting_by_operation.get(name, ZERO_ACTIVITY_ENERGY) + energy.scale(factor)
            for name, op_activity in (result.gpu.accounting_by_operation_activity or result.gpu.by_operation_activity).items():
                accounting_by_operation_activity[name] = accounting_by_operation_activity.get(name, ZERO_ACTIVITY) + op_activity.scaled(factor)
        communication = communication + result.communication.scaled(factor)
        launch_records.extend(
            dict(row, multiplicity=row.get("multiplicity", 1.0) * factor)
            for row in result.collective_launch_records
        )
        tp_allreduce_records.extend(
            dict(row, multiplicity=row.get("multiplicity", 1.0) * factor)
            for row in result.tp_allreduce_records
        )
        missing.update(result.coverage.missing)
        for name in statuses:
            statuses[name].add(getattr(result.coverage, name))
    gpu = ProgramEnergyResult(
        activity=activity,
        dynamic=dynamic,
        static_j=static_j,
        wall_time_s=wall_time_s,
        device_count=device_count,
        by_operation_activity=by_operation_activity,
        by_operation=by_operation,
        by_rank_activity=by_rank_activity,
        by_rank=by_rank,
        accounting_by_operation_activity=accounting_by_operation_activity if has_invocations else None,
        accounting_by_operation=accounting_by_operation if has_invocations else None,
        busy_j=busy_j,
        busy_s=busy_s,
    )

    def _status(values: set[str]) -> str:
        if values == {"complete"}:
            return "complete"
        if values == {"unavailable"}:
            return "unavailable"
        return "partial"

    coverage = EnergyCoverage(
        gpu_activity=_status(statuses["gpu_activity"]),
        gpu_static=_status(statuses["gpu_static"]),
        communication=_status(statuses["communication"]),
        missing=tuple(sorted(missing)),
    )
    return ExecutionEnergyResult(
        gpu=gpu,
        communication=communication,
        coverage=coverage,
        execution_mode=execution_mode or parts[0][0].execution_mode,
        collective_launch_records=tuple(launch_records),
        tp_allreduce_records=tuple(tp_allreduce_records),
    )

def add_idle_interval(
    result: "ExecutionEnergyResult", seconds: float, static_power_w: float
) -> "ExecutionEnergyResult":
    seconds = float(seconds)
    if seconds <= 0.0:
        return result
    gpu = replace(
        result.gpu,
        static_j=result.gpu.static_j
        + float(static_power_w) * result.gpu.device_count * seconds,
        wall_time_s=result.gpu.wall_time_s + seconds,
    )
    return replace(result, gpu=gpu)

def apply_inference_gpu_accounting(
    result: ExecutionEnergyResult,
    records: Sequence[Mapping[str, Any]],
    *,
    layer_count: int,
) -> ExecutionEnergyResult:
    return result

def apply_training_graph_accounting(
    result: ExecutionEnergyResult,
    *,
    pipeline_stages: int,
    data_parallel: int,
) -> ExecutionEnergyResult:
    if result.gpu.by_rank_activity:
        return result
    stages = max(1, int(pipeline_stages))
    replicas = max(1, int(data_parallel))
    if result.gpu.device_count % (stages * replicas):
        raise ValueError(
            "training device count must be divisible by pipeline stages x "
            f"data-parallel replicas: devices={result.gpu.device_count}, "
            f"stages={stages}, dp={replicas}"
        )
    stage_width = max(1, result.gpu.device_count // (stages * replicas))
    rank_pattern = re.compile(r"(?:^|_)rank\d+(?:_|$)")

    by_operation_activity = {}
    by_operation = {}
    activity = ZERO_ACTIVITY
    dynamic = ZERO_ACTIVITY_ENERGY
    names = set(result.gpu.by_operation_activity) | set(result.gpu.by_operation)

    for name in sorted(names):
        factor = 1.0 if rank_pattern.search(name) else float(stage_width)
        op_activity = result.gpu.by_operation_activity.get(name, ZERO_ACTIVITY).scaled(
            factor
        )
        op_energy = result.gpu.by_operation.get(name, ZERO_ACTIVITY_ENERGY).scale(
            factor
        )
        by_operation_activity[name] = op_activity
        by_operation[name] = op_energy
        accounting_activity = (result.gpu.accounting_by_operation_activity or result.gpu.by_operation_activity).get(name, ZERO_ACTIVITY).scaled(factor)
        accounting_energy = (result.gpu.accounting_by_operation or result.gpu.by_operation).get(name, ZERO_ACTIVITY_ENERGY).scale(factor)
        activity = activity + accounting_activity
        dynamic = dynamic + accounting_energy

    gpu = ProgramEnergyResult(
        activity=activity,
        dynamic=dynamic,
        static_j=result.gpu.static_j,
        wall_time_s=result.gpu.wall_time_s,
        device_count=result.gpu.device_count,
        by_operation_activity=by_operation_activity,
        by_operation=by_operation,
        busy_j=result.gpu.busy_j,
        busy_s=result.gpu.busy_s,
    )
    return ExecutionEnergyResult(
        gpu=gpu,
        communication=result.communication,
        coverage=result.coverage,
        execution_mode=result.execution_mode,
        collective_launch_records=result.collective_launch_records,
        tp_allreduce_records=result.tp_allreduce_records,
    )

def write_energy_report(
    output_dir: str,
    result: ExecutionEnergyResult,
    *,
    phases: Mapping[str, ExecutionEnergyResult] | None = None,
    metadata: Mapping[str, Any] | None = None,
    hardware: Any = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    report = result.to_report(phases=phases)
    report["component_energy"] = result.component_energy_report(hardware)
    if phases:
        report["phase_component_energy"] = {
            name: phase.component_energy_report(hardware) for name, phase in phases.items()
        }
    report.update(dict(metadata or {}))
    path = os.path.join(output_dir, "energy_report.json")
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)
    return path

__all__ = [
    "EnergyCoverage",
    "ExecutionEnergyResult",
    "combine_execution_energy",
    "apply_inference_gpu_accounting",
    "apply_training_graph_accounting",
    "write_energy_report",
]
