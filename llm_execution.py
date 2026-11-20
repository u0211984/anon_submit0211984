from __future__ import annotations

import os
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple, TYPE_CHECKING

from astrasim_lib import run_astra_simulation_only_onepath
from astrasim_lib.fault_projection import FaultProjectionResult, FaultSpace
from astrasim_lib.layout_utils import axis_layout_from_descriptor
from program import _env_flag
from program.energy import replay_program_energy
from program.execution_energy import EnergyCoverage, ExecutionEnergyResult
from activity import ZERO_ACTIVITY
from program.ir import CollectiveOp, ComputeOp, Program, TransferOp
from program.layout import RankLayout
from program.network_energy import (
    CommunicationEnergyResult,
    communication_energy_from_astrasim,
    link_dynamic_from_program_transfers,
)
from program.placement import Granularity
from program.work import Direction, WorkKind
from program.workload import WorkloadSpec
from util import log_message

if TYPE_CHECKING:
    from train_timing import TimeCalculationLLM

class ExecutionMode(Enum):
    ANALYTICAL = "analytical"
    HYBRID = "hybrid"
    FULL_ASTRASIM_HIERARCHICAL = "full_astrasim_hierarchical"
    FULL_ASTRASIM_FLATTENED = "full_astrasim_flattened"

@dataclass
class ExecutionResult:
    total_time: float
    energy: Optional[ExecutionEnergyResult] = None

@dataclass
class TransformerTimings:
    forward: float
    backward: float

class LLMExecutionDispatcher:
    def __init__(
        self,
        time_calc: "TimeCalculationLLM",
        workload: WorkloadSpec,
    ) -> None:
        if not isinstance(workload, WorkloadSpec):
            raise TypeError(
                "LLMExecutionDispatcher requires a typed program.workload.WorkloadSpec "
                f"(got {type(workload).__name__}); build it with WorkloadSpec.from_timing"
            )
        self.time_calc = time_calc

        self.no_data_parallel = bool(workload.run.is_nonfinal_grad_accum_cycle)

        self._network_dimensions: Tuple[Any, ...] = tuple()
        self._transformer_rank_layout: Optional[Dict[str, Any]] = None
        self._pipeline_rank_layout: Optional[Dict[str, Any]] = None
        self._axis_dimension_map: Dict[str, int] = {}
        self._first_dim_optimize_cfg: Optional[Dict[str, Any]] = None
        self._rank_layout = self._build_rank_layout_descriptor(workload)

        self.workload = workload.with_(layout=_layout_of(self._rank_layout))
        self.interconnect_params: Dict[str, Tuple[float, float]] = dict(
            self.workload.interconnect
        )

        self.flat_program: Optional[Program] = None
        self._memory_flat_program: Optional[Program] = None

        self.pipeline_program: Optional[Program] = None

        self._astra_energy_runs: List[Tuple[str, Program, Any, float]] = []
        self._astra_timing_runs: List[Tuple[str, Program, float]] = []

        self._comm_sizes_converted = False

        self._transformer_stage_dp_faults: Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]] = {}
        self._transformer_stage_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_stage_moe_timings: Dict[Tuple[int, int], TransformerTimings] = {}
        self._transformer_baseline_timings: Optional[TransformerTimings] = None
        self._transformer_moe_baseline_timings: Optional[TransformerTimings] = None
        self._fault_space: Optional[FaultSpace] = None
        self._fault_projections: Dict[str, FaultProjectionResult] = {}
        self._initialize_fault_mappings()

    def _build_rank_layout_descriptor(self, workload: WorkloadSpec) -> Dict[str, Any]:
        hw_config = getattr(self.time_calc, "hw_config", None)
        layout = getattr(hw_config, "network_layout", None)
        dimensions = getattr(layout, "dimensions", None) if layout is not None else None
        if not dimensions and not getattr(hw_config, "counts_only", False):
            return {}
        self._network_dimensions = tuple(dimensions)

        degrees = workload.degrees
        axis_sizes: Dict[str, int] = {
            "tp": degrees.tp,
            "cp": degrees.cp,

            "ep": max(1, int(getattr(self.time_calc, "ep", 1) or 1)),
            "pp": degrees.pp,
            "dp": max(1, int(getattr(self.time_calc, "dp", 1) or 1)),
        }

        enforce_layout = self.time_calc.execution_mode in {
            ExecutionMode.HYBRID,
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
        }
        if not dimensions:
            if any(size != 1 for size in axis_sizes.values()) or any(
                getattr(degrees, axis) != 1 for axis in ("tp", "cp", "ep", "pp", "dp")
            ):
                raise ValueError("counts-only hardware without network requires single-device runtime degrees")
            axes = ("tp", "cp", "ep", "pp", "dp")
            rank_layout = RankLayout(axes, dict(axis_sizes), {axis: 1 for axis in axes})
            optimize_cfg = None
        else:
            rank_layout, optimize_cfg = RankLayout.from_network_layout(
                layout, axis_sizes, enforce_layout
            )
        self._first_dim_optimize_cfg = optimize_cfg

        def _subset_descriptor(allowed: Sequence[str]) -> Optional[Dict[str, Any]]:
            subset = rank_layout.subset(allowed)
            if not subset.axis_order:
                return None
            return subset.descriptor()

        self._transformer_rank_layout = _subset_descriptor(["tp", "cp", "ep"])

        zero_moe = (workload.run.include_backward and workload.run.zero_stage >= 2
                    and workload.shape.has_moe_layers)
        self._pipeline_rank_layout = (rank_layout.descriptor() if zero_moe
                                      else _subset_descriptor(["pp", "dp"]))

        return rank_layout.descriptor()

    def _log_fault_summary(self, axis_order: Sequence[str], axis_sizes: Mapping[str, int]) -> None:
        if not axis_order:
            return
        space = self._fault_space
        if space is None or not space.entries:
            return
        network_dims = getattr(self, "_network_dimensions", tuple())
        if not network_dims:
            log_message("[RAPID-LLM][faults] Hardware dimensions unavailable; skipping fault summary.")
            return

        def coords_to_dict(coords: Tuple[Tuple[str, int], ...]) -> Dict[str, int]:
            return {name: value for name, value in coords}

        entries = space.entries
        for dim_index, dim in enumerate(network_dims):
            axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
            if not axes:
                continue
            replication_axes: List[str] = []
            for future_dim in network_dims[dim_index + 1 :]:
                replication_axes.extend(
                    str(axis).strip().lower() for axis in getattr(future_dim, "parallelisms", ()) if axis
                )
            total_clusters = 1
            for axis in replication_axes:
                total_clusters *= max(1, axis_sizes.get(axis, 1))
            axes_label = ", ".join(axes) or "<none>"
            table_rows: List[Tuple[str, str, float]] = []
            for entry in entries:
                if not any(axis in axes for axis in entry.affected_axes):
                    continue
                src_dict = coords_to_dict(entry.src_coords)
                dst_dict = coords_to_dict(entry.dst_coords)
                src_label = f"{entry.original[0]} (" + ", ".join(f"{axis}={src_dict.get(axis, 0)}" for axis in axes) + ")"
                dst_label = f"{entry.original[1]} (" + ", ".join(f"{axis}={dst_dict.get(axis, 0)}" for axis in axes) + ")"
                higher_coords = " ".join(f"{axis}={src_dict.get(axis, 0)}" for axis in replication_axes)
                table_rows.append((higher_coords, f"{src_label} <-> {dst_label}", float(entry.original[2])))

            if not table_rows:
                log_message(f"  • dim{dim_index} ({axes_label}) → Affected HW clusters: 0 / {total_clusters}", category="faults")
                continue

            log_message(
                f"  • dim{dim_index} ({axes_label}) → Affected HW clusters: {len(table_rows)} / {total_clusters or 1}",
                category="faults",
            )
            header_axes = " ".join(replication_axes) if replication_axes else ""
            log_message(f"  {header_axes:<12} | {'source ↔ dest':<40} | derate", category="faults")
            log_message(f"  {'-' * max(len(header_axes),12)}-+-{'-' * 40}-+-------", category="faults")

            for higher_str, pair, derate in table_rows:
                log_message(
                    f"  {higher_str:<12} | {pair:<40} | {derate:>6.2f}",
                    category="faults",
                )
            log_message(f"  {'-' * max(len(header_axes),12)}-+-{'-' * 40}-+-------", category="faults")

    def _initialize_fault_mappings(self) -> None:
        hw_config = getattr(self.time_calc, "hw_config", None)
        network_layout = getattr(hw_config, "network_layout", None)
        faulty_links: Tuple[Tuple[int, int, float], ...] = tuple(
            getattr(network_layout, "faulty_links", ()) or ()
        )
        if faulty_links and self.time_calc.execution_mode in {ExecutionMode.ANALYTICAL, ExecutionMode.HYBRID}:
            raise ValueError("Faulty links require full AstraSim execution; analytical/hybrid modes are not supported.")
        axis_layout = axis_layout_from_descriptor(self._rank_layout)
        axis_dim_map = self._axis_to_dimension_map(network_layout)
        self._axis_dimension_map = dict(axis_dim_map)
        self._fault_space = FaultSpace(
            axis_layout,
            faulty_links,
            axis_to_dimension=axis_dim_map,
        )
        self._fault_projections = self._build_fault_projections_from_space(self._fault_space)
        self._validate_fault_coverage()
        self._transformer_stage_dp_faults = self._build_transformer_stage_dp_fault_map()
        axis_order = self._rank_layout.get("axis_order", []) if isinstance(self._rank_layout, dict) else []
        axis_sizes = self._rank_layout.get("axis_sizes", {}) if isinstance(self._rank_layout, dict) else {}
        self._log_fault_summary(axis_order, axis_sizes)

    def _axis_to_dimension_map(self, network_layout) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        if network_layout is None:
            return mapping
        dimensions = getattr(network_layout, "dimensions", None)
        if not dimensions:
            return mapping
        for idx, dim in enumerate(dimensions):
            for axis in getattr(dim, "parallelisms", ()) or ():
                normalized = str(axis).strip().lower()
                if normalized:
                    mapping[normalized] = idx
        return mapping

    def _build_fault_projections_from_space(
        self,
        space: FaultSpace,
    ) -> Dict[str, FaultProjectionResult]:
        projections: Dict[str, FaultProjectionResult] = {}
        if not space.entries:
            return projections

        global_axes = tuple(space.layout.axis_order)
        if global_axes:
            projections["global"] = space.project(global_axes)

        transformer_axes: Tuple[str, ...] = tuple(
            self._transformer_rank_layout.get("axis_order", ())
        ) if self._transformer_rank_layout else tuple()
        if transformer_axes:
            projections["transformer"] = space.project(transformer_axes)

        pipeline_axes: Tuple[str, ...] = tuple(
            self._pipeline_rank_layout.get("axis_order", ())
        ) if self._pipeline_rank_layout else tuple()
        if pipeline_axes:
            projections["pipeline"] = space.project(pipeline_axes)

        return projections

    def _validate_fault_coverage(self) -> None:
        if self._fault_space is None:
            return
        covered: Set[Tuple[int, int, float]] = set()
        for label in ("transformer", "pipeline"):
            proj = self._fault_projections.get(label)
            if proj:
                covered.update(proj.covered_originals)
        uncovered = [
            entry.original
            for entry in self._fault_space.entries
            if entry.original not in covered
        ]
        if uncovered:
            formatted = ", ".join(str(item) for item in uncovered)
            raise ValueError(
                "Faulty links do not map to any transformer or pipeline axis subset: "
                f"{formatted}"
            )

    def _axis_value_from_coords(self, coords: Tuple[Tuple[str, int], ...], axis: str) -> int:
        for name, value in coords:
            if name == axis:
                return int(value)
        layout = self._rank_layout or {}
        if isinstance(layout, dict):
            layout_axes = layout.get("axis_order", [])
            if axis not in layout_axes:
                return 0
            axis_sizes = layout.get("axis_sizes", {})
        else:
            axis_sizes = {}
        if axis_sizes.get(axis, 1) <= 1:
            return 0
        raise ValueError(f"Axis '{axis}' not present in coordinate tuple for faulty link.")

    def _build_transformer_stage_dp_fault_map(self) -> Dict[Tuple[int, int], Tuple[Tuple[int, int, float], ...]]:
        projection = self._fault_projection_for("transformer")
        if not projection:
            return {}
        stage_dp_faults: Dict[Tuple[int, int], List[Tuple[int, int, float]]] = {}
        for detail in projection.entries:
            src_stage = self._axis_value_from_coords(detail.src_coords, "pp")
            dst_stage = self._axis_value_from_coords(detail.dst_coords, "pp")
            if src_stage != dst_stage:
                raise ValueError(
                    "Transformer faulty link spans multiple pipeline stages; "
                    "hierarchical execution requires faults limited to a single stage."
                )
            src_dp = self._axis_value_from_coords(detail.src_coords, "dp")
            dst_dp = self._axis_value_from_coords(detail.dst_coords, "dp")
            affected_dp = {src_dp, dst_dp}
            for dp_idx in affected_dp:
                stage_dp_faults.setdefault((dp_idx, src_stage), []).append(detail.remapped)
        return {key: tuple(links) for key, links in stage_dp_faults.items()}

    def _fault_projection_for(self, label: str) -> Optional[FaultProjectionResult]:
        return self._fault_projections.get(label)

    def _fault_links_for(self, label: str) -> Tuple[Tuple[int, int, float], ...]:
        projection = self._fault_projection_for(label)
        if projection is None:
            return tuple()
        return projection.remapped_links

    def _fault_override(self, label: str) -> Optional[Tuple[Tuple[int, int, float], ...]]:
        if self._fault_space is None:
            return None
        return self._fault_links_for(label)

    def _build_program(
        self,
        granularity: Granularity,
        *,
        label: str,
        workload: Optional[WorkloadSpec] = None,
        gmap_workdir: Optional[str] = None,
        directions: Optional[Tuple[Direction, ...]] = None,
    ) -> Program:
        from program.build import build
        from program.policies import policies_for
        from program.schedule.gpipe import GPipeSchedule
        from program.schedule.onefoneb import OneFOneBSchedule
        from program.validate import GroupRaceWarning
        from program import megatron

        spec = workload if workload is not None else self.workload
        fw = spec.freeze()
        bundle = policies_for(fw, granularity=granularity)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GroupRaceWarning)
            program = build(
                fw,
                granularity=granularity,
                sharding=bundle.sharding,

                schedule_policy=(
                    OneFOneBSchedule(first_stage_layers=(
                        int(self.time_calc.implementation_options.get("first_pipeline_stage_layers", 0))
                        if granularity is not Granularity.BLOCK else 0))
                    if str(getattr(self.time_calc, "pipeline_schedule", "gpipe")) == "1f1b"
                    and str(getattr(self.time_calc, "run_type", "training")) == "training"
                    else GPipeSchedule()
                ),
                recompute=bundle.recompute,
                routing=bundle.routing,
                overlap=bundle.overlap,
                grad_accum=bundle.grad_accum,
                placement_policy=megatron.placement(self.time_calc),
                label=label,
                gmap_workdir=gmap_workdir,
                directions=directions,

                validate=bool(getattr(self.time_calc, "validate_graph", True)),
            )
        launch = float(getattr(self.time_calc.hw_config.sw_config, "collective_launch_overhead", 0.0))
        if launch:
            program.meta.misc["collective_launch_overhead"] = launch
        policy = getattr(getattr(self.time_calc, "network_model", None), "vllm_allreduce", None)
        if policy:
            program.meta.misc["vllm_allreduce"] = dict(policy)
        curve = getattr(getattr(self.time_calc, "network_model", None), "allreduce_curve", None)
        if curve and not policy:
            program.meta.misc["allreduce_curve"] = dict(curve)
        if granularity is Granularity.FLAT:
            from program import megatron
            if megatron.enabled(self.time_calc):
                megatron.apply(program, self.time_calc)
        from program.nccl_latency import stamp
        stamp(program, self.time_calc.hw_config)
        return program

    def _emission_program(
        self,
        granularity: Granularity,
        *,
        label: str,
        artifact_dir: str,
        workload: Optional[WorkloadSpec] = None,
        directions: Optional[Tuple[Direction, ...]] = None,
    ) -> Program:
        from program.mapping import apply_first_dim_mapping

        optimize_cfg = dict(self._first_dim_optimize_cfg) if self._first_dim_optimize_cfg else None
        gmap_workdir = (
            artifact_dir if (optimize_cfg and self.time_calc.persist_astrasim_artifacts) else None
        )
        program = self._build_program(
            granularity,
            label=label,
            workload=workload,
            gmap_workdir=gmap_workdir,
            directions=directions,
        )
        return apply_first_dim_mapping(
            program, optimize_2dmap=optimize_cfg, workdir=gmap_workdir
        )

    def _interleave_scale(self) -> float:
        return self.workload.run.interleave_scale(
            self.workload.degrees, self.workload.shape
        )

    def _pipeline_label(self) -> str:
        return "pipeline_no_dp" if self.no_data_parallel else "pipeline"

    def run(self, mode: ExecutionMode) -> ExecutionResult:
        self._astra_energy_runs = []
        self._astra_timing_runs = []

        if (self.workload.run.include_backward and self.workload.run.zero_stage >= 2
                and self.workload.shape.has_moe_layers
                and self.workload.degrees.cp > 1):
            raise ValueError(
                "ZeRO+MoE: training ZeRO>=2 requires cp=1; "
                "combined CP gradient/state sharding is not implemented."
            )
        if mode == ExecutionMode.ANALYTICAL:
            result = self._run_pipeline_with_analytical_comm(ExecutionMode.ANALYTICAL)
        elif mode == ExecutionMode.HYBRID:
            result = self._run_hybrid()
        elif mode == ExecutionMode.FULL_ASTRASIM_HIERARCHICAL:
            result = self._run_full_astrasim_hierarchical()
        elif mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            result = self._run_full_astrasim_flattened()
        else:
            raise ValueError(f"Unknown execution mode {mode!r}")
        return self._attach_energy(result, mode)

    @staticmethod
    def _collective_signature(op: CollectiveOp) -> Tuple[Any, ...]:
        return (
            op.comm_key,
            op.coll.value,
            int(round(float(op.size_bytes))),
            int(op.participants),
            tuple(op.axes),
            getattr(getattr(op, "work", None), "direction", None),
        )

    def _block_scale(
        self,
        block_program: Program,
        flat_program: Program,
        shared: frozenset = frozenset(),
    ) -> float:
        block = Counter(
            self._collective_signature(op)
            for op in block_program.ops
            if isinstance(op, CollectiveOp)
        )
        unique = Counter(
            {signature: count for signature, count in block.items() if signature not in shared}
        )
        if unique:
            block = unique
        if not block:
            return 0.0
        flat = Counter(
            self._collective_signature(op)
            for op in flat_program.ops
            if isinstance(op, CollectiveOp)
        )
        ratios = []
        for signature, count in block.items():
            available = flat.get(signature, 0)

            if available == 0:
                continue
            if available % count:
                raise RuntimeError(
                    "cannot reconcile hierarchical transformer communication "
                    f"with FLAT execution for signature {signature}: "
                    f"block={count}, flat={available}"
                )
            ratios.append(available // count)
        if not ratios:
            return 0.0
        if len(set(ratios)) != 1:
            raise RuntimeError(
                "hierarchical transformer collective signatures have inconsistent "
                f"execution multiplicities: {ratios}"
            )
        return float(ratios[0])

    def _pipeline_dp_slices(
        self, priced: CommunicationEnergyResult
    ) -> CommunicationEnergyResult:
        tc = self.time_calc
        tp = max(1, int(getattr(tc, "tp", 1) or 1))
        cp = max(1, int(getattr(tc, "cp", 1) or 1))
        dp = max(1, int(getattr(tc, "dp", 1) or 1))
        factor = tp * (dp * cp - 1) / (dp - 1) if dp > 1 else 1.0
        if factor == 1.0 and tp * cp == 1:
            return priced
        from program.link_energy import resolve_rule

        def scale_collective(item):
            if str(item.collective_type).upper() == "P2P":
                return item.scaled(tp * cp)
            if factor == 1.0:
                return item
            scaled = item.scaled(factor)
            dimension = tc.hw_config.network_layout.dimensions[item.dimension_index]
            if resolve_rule(tc.hw_config, dimension) == "line_rate_while_active":
                scaled = replace(scaled, link_dynamic_j=item.link_dynamic_j * tp * cp,
                                 completion_time_ns=item.completion_time_ns * tp * cp)
            return scaled

        return CommunicationEnergyResult(tuple(scale_collective(item) for item in priced.collectives))

    @staticmethod
    def _baseline_energy_program(program: Program) -> Program:
        meta = getattr(program, "meta", None)
        misc = getattr(meta, "misc", {})
        keys = [key for key in ("vllm_allreduce", "allreduce_curve")
                if (misc.get(key) or {}).get("allreduce_model") == "measured_curve"]
        if not keys:
            return program
        from copy import deepcopy
        baseline = deepcopy(program)
        for key in keys:
            baseline.meta.misc[key]["allreduce_model"] = "astrasim"
        return baseline

    def _record_astra_energy_run(
        self, scope: str, program: Program, astra_result: Any,
        artifact_dir: str, *, faulty_links_override=None, rank_layout=None,
    ) -> None:
        if not hasattr(self, "_astra_timing_runs"):
            self._astra_timing_runs = []
        self._astra_timing_runs.append((scope, program, 1.0))
        from program.allreduce_curve import is_measured_call
        calls = (call for op in program.ops for call in
                 (op.embedded_collectives if isinstance(op, ComputeOp) else (op,)))
        if any(is_measured_call(program, call) for call in calls):
            program = self._baseline_energy_program(program)
            astra_result = run_astra_simulation_only_onepath(
                program, self.time_calc,
                os.path.join(artifact_dir, "communication_baseline"),
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
                faulty_links_override=faulty_links_override,
                rank_layout=rank_layout,
            )
        self._astra_energy_runs.append((scope, program, astra_result, 1.0))

    def _communication_energy(
        self, mode: ExecutionMode, flat_program: Program
    ) -> Tuple[CommunicationEnergyResult, EnergyCoverage]:
        flat_program = self._baseline_energy_program(flat_program)
        from program.link_energy import resolve_rule, line_rate_bps, LinkEnergyConfigError

        traffic_axes = set()
        for op in flat_program.ops:
            calls = op.embedded_collectives if isinstance(op, ComputeOp) else (op,)
            for call in calls:
                if float(getattr(call, "size_bytes", 0)) <= 0:
                    continue
                if isinstance(call, TransferOp):
                    if call.src_device == call.dst_device:
                        continue
                elif int(getattr(call, "participants", 0)) <= 1:
                    continue
                traffic_axes.update(str(getattr(call, "interconnect", "") or "").split("*"))
        for dimension in self.time_calc.hw_config.network_layout.dimensions:
            if (int(getattr(dimension, "size", 1)) > 1
                    and traffic_axes.intersection(getattr(dimension, "parallelisms", ()))):
                if resolve_rule(self.time_calc.hw_config, dimension) == "line_rate_while_active":
                    line_rate_bps(dimension)
        has_communication = any(
            isinstance(op, CollectiveOp)
            or (isinstance(op, TransferOp) and op.size_bytes > 0)
            for op in flat_program.ops
        )
        if not has_communication:
            return CommunicationEnergyResult(), EnergyCoverage(
                communication="complete", missing=()
            )
        if mode == ExecutionMode.ANALYTICAL:
            if flat_program.meta.misc.get("nccl_latency_rule") == "tuning_protocol":
                from program.network_energy import protocol_energy_from_program
                return protocol_energy_from_program(flat_program, self.time_calc.hw_config), EnergyCoverage(
                    communication="partial", missing=("communication:non_protocol_traffic",))
            return CommunicationEnergyResult(), EnergyCoverage(
                communication="unavailable", missing=("communication",)
            )

        result = CommunicationEnergyResult()
        missing: List[str] = []
        block_signatures: Dict[str, set] = {}
        for scope, program, _, _ in self._astra_energy_runs:
            if scope.startswith("transformer"):
                block_signatures[scope] = {
                    self._collective_signature(op)
                    for op in program.ops
                    if isinstance(op, CollectiveOp)
                }

        def _shared_with_other_blocks(scope: str) -> frozenset:
            direction = scope.rsplit("_", 1)[-1]
            others = set()
            for other, signatures in block_signatures.items():
                if other != scope and other.rsplit("_", 1)[-1] == direction:
                    others |= signatures
            return frozenset(block_signatures.get(scope, set()) & others)

        zero_moe_graph = getattr(getattr(flat_program, "meta", None), "misc", {}).get("sharding_policy") == "zero_moe"
        for scope, program, astra_result, scale in self._astra_energy_runs:
            zero_moe_program = getattr(getattr(program, "meta", None), "misc", {}).get("sharding_policy") == "zero_moe"
            stats = getattr(astra_result, "link_stats", None)
            if not stats:
                if getattr(getattr(program, "meta", None), "misc", {}).get("nccl_latency_rule") == "tuning_protocol":
                    stats = {}
                else:
                    missing.append(f"communication:{scope}")
                    continue
            if scope.startswith("transformer"):
                if zero_moe_graph:
                    want_moe = scope.startswith("transformer_moe_")
                    direction = Direction.FORWARD if scope.endswith("_forward") else Direction.BACKWARD
                    consumers = {
                        op.work for op in flat_program.ops
                        if isinstance(op, ComputeOp) and op.work is not None
                        and op.work.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE)
                        and op.work.direction is direction
                        and bool(op.is_moe_layer) == want_moe
                    }
                    scale = float(len(consumers) * max(1, int(flat_program.dp_count)))
                else:
                    scale = self._block_scale(
                        program, flat_program, _shared_with_other_blocks(scope)
                    ) * max(
                        1, int(flat_program.dp_count)
                    )
            try:
                meta = getattr(getattr(program, "meta", None), "misc", {})
                priced = communication_energy_from_astrasim(
                    stats, self.time_calc.hw_config,
                    zero_moe_dimension_reconcile=zero_moe_program,
                    collective_launch_overhead=meta.get("collective_launch_overhead", 0.0),
                    launch_durations=meta.get("energy_launch_durations"),
                )
                if meta.get("nccl_latency_rule") == "tuning_protocol":
                    from program.network_energy import protocol_energy_from_program
                    priced = priced + protocol_energy_from_program(program, self.time_calc.hw_config)
            except LinkEnergyConfigError:
                raise
            except ValueError as exc:
                missing.append(f"communication:{scope}:unreconciled")
                warnings.warn(
                    f"communication energy unavailable for {scope}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            if scope == "pipeline" and not zero_moe_program:
                priced = self._pipeline_dp_slices(priced)
            result = result + priced.scaled(scale)

        observed_dimensions = {
            item.dimension_index for item in result.collectives

            if not zero_moe_graph or str(item.collective_type).upper() == "P2P"
        }
        transfer_axes = {
            str(op.interconnect or "").strip().lower()
            for op in flat_program.ops
            if isinstance(op, TransferOp) and op.size_bytes > 0 and op.src_device != op.dst_device
        }
        for dimension_index, dimension in enumerate(
            self.time_calc.hw_config.network_layout.dimensions
        ):
            parallelisms = {
                str(axis).strip().lower()
                for axis in getattr(dimension, "parallelisms", ())
            }
            expected_axes = transfer_axes.intersection(parallelisms)
            if expected_axes and dimension_index not in observed_dimensions:
                if "pp" in expected_axes:
                    durations = None
                    if resolve_rule(self.time_calc.hw_config, dimension) == "line_rate_while_active":
                        from program.analytic_sim import comm_durations
                        uids = {op.uid for op in flat_program.ops if isinstance(op, TransferOp)
                                and op.size_bytes > 0 and op.interconnect == "pp"}
                        times = comm_durations(flat_program, self.time_calc.network_model,
                                               self.interconnect_params, operation_uids=uids)
                        durations = {uid: times[uid] for uid in uids}
                    recovered = link_dynamic_from_program_transfers(
                        flat_program.ops,
                        self.time_calc.hw_config,
                        dimension_index,
                        axis="pp",
                        durations=durations,
                    )
                    if recovered.collectives:
                        result = result + recovered.scaled(
                            max(1, int(flat_program.dp_count))
                        )
                        expected_axes.remove("pp")

                        if (
                            float(getattr(dimension, "system_power_w", 0.0) or 0.0) > 0.0
                            or float(getattr(dimension, "static_power_per_link_w", 0.0) or 0.0) > 0.0
                        ):
                            missing.append(
                                f"communication:dimension{dimension_index}:pp:active_power_unmeasured"
                            )
                if expected_axes:
                    axes = ",".join(sorted(expected_axes))
                    missing.append(
                        f"communication:dimension{dimension_index}:{axes}"
                    )

        if mode == ExecutionMode.HYBRID:
            missing.append("communication:pipeline_analytical")
        status = "complete" if not missing else ("partial" if result.collectives else "unavailable")
        return result, EnergyCoverage(
            communication=status,
            missing=tuple(sorted(set(missing))),
        )

    def _attach_energy(
        self, result: ExecutionResult, mode: ExecutionMode
    ) -> ExecutionResult:
        flat_program = self.build_flat_program_for_memory()
        gpu = replay_program_energy(
            flat_program,
            self.time_calc.hw_config,
            wall_time_s=result.total_time,
        )
        communication, communication_coverage = self._communication_energy(mode, flat_program)
        from program.tied_sync_energy import tied_sync_energy
        tied_energy, tied_dimension = (tied_sync_energy(
            self.workload, self.time_calc.hw_config,
            tp=self.time_calc.tp, cp=self.time_calc.cp, dp=self.time_calc.dp,
            element_bytes=self.time_calc.precision.grad_communication,
        ) if not self.no_data_parallel
            else (CommunicationEnergyResult(), None))
        if tied_energy.collectives:
            communication += tied_energy
            missing = set(communication_coverage.missing)
            if any(float(getattr(tied_dimension, key, 0.0) or 0.0) > 0 for key in
                   ('system_power_w', 'static_power_per_link_w')):
                missing.add('communication:tied_sync_active_network_power')
            status = communication_coverage.communication
            if missing or status == 'unavailable':
                status = 'partial'
            communication_coverage = replace(communication_coverage, communication=status,
                                             missing=tuple(sorted(missing)))
        if flat_program.meta.misc.get("vtp_per_rank"):
            from program.vtp_gather import logical_gathers, gather_energy, gather_coverage

            communication = CommunicationEnergyResult(tuple(
                row for row in communication.collectives if str(row.collective_type).upper() != "P2P"))
            gather_records = logical_gathers(flat_program)
            gather_dimension = flat_program.meta.misc["vllm_allreduce"]["dimension_index"]
            communication += gather_energy(gather_records, self.time_calc.hw_config, gather_dimension)
        from program.vllm_allreduce import logical_records
        ar_records = logical_records(flat_program)
        from program.allreduce_curve import record_usage
        record_usage(self.time_calc, flat_program.meta.misc.get("vllm_allreduce"), ar_records)
        if flat_program.meta.misc.get("allreduce_curve"):
            from program.allreduce_curve import training_records, training_usage_scope
            record_usage(self.time_calc, flat_program.meta.misc["allreduce_curve"],
                         training_records(flat_program))
            self.time_calc.resolved_settings["allreduce_usage"]["scope"] = training_usage_scope(flat_program)
        if flat_program.meta.misc.get("vtp_per_rank") and mode == ExecutionMode.ANALYTICAL:
            from program.vllm_allreduce_energy import analytical_nccl_allreduce_energy
            policy = flat_program.meta.misc["vllm_allreduce"]
            if any(row["algorithm"] == "nccl" and row["size_bytes"] > 0 for row in ar_records):
                from program.link_energy import resolve_rule
                dim = self.time_calc.hw_config.network_layout.dimensions[policy["dimension_index"]]
                if resolve_rule(self.time_calc.hw_config, dim) == "line_rate_while_active":
                    from program.analytic_sim import comm_durations

                    uids = {op.uid for op in flat_program.ops if isinstance(op, CollectiveOp)}
                    times = comm_durations(flat_program, self.time_calc.network_model,
                                           self.interconnect_params, operation_uids=uids)
                    ar_records = logical_records(flat_program, active_durations=times)
            if flat_program.meta.misc.get("nccl_latency_rule") != "tuning_protocol":
                communication += analytical_nccl_allreduce_energy(ar_records, self.time_calc.hw_config, policy)

            dim = self.time_calc.hw_config.network_layout.dimensions[policy["dimension_index"]]
            active = any(float(getattr(dim, key, 0.) or 0.) > 0 for key in
                         ("system_power_w", "static_power_per_link_w"))
            communication_coverage = EnergyCoverage(communication="partial" if active else "complete",
                missing=("communication:analytical_active_network_power",) if active else ())
        if flat_program.meta.misc.get("vtp_per_rank"):
            dimension = self.time_calc.hw_config.network_layout.dimensions[gather_dimension]
            communication_coverage = gather_coverage(communication_coverage, gather_records, dimension)
            from program.vllm_input_broadcast import input_broadcast_energy
            policy = flat_program.meta.misc["vllm_allreduce"]
            inputs = policy.get("input_broadcast")
            communication += input_broadcast_energy(inputs, self.time_calc.hw_config, policy,
                                                     max(1, flat_program.dp_count))
            omission = ("communication:input_broadcast_protocol_buffers_and_control" if inputs else
                        "communication:vllm_worker_input_broadcast")
            communication_coverage = EnergyCoverage(communication="partial",
                missing=tuple(sorted(set(communication_coverage.missing).union({omission}))))
        if ar_records:
            from program.vllm_allreduce_energy import custom_allreduce_energy
            custom_energy = custom_allreduce_energy(
                ar_records, self.time_calc.hw_config, flat_program.meta.misc["vllm_allreduce"])
            communication = communication + custom_energy
            if custom_energy.collectives:
                communication_coverage = EnergyCoverage(
                    communication="partial",
                    missing=tuple(sorted(set(communication_coverage.missing).union({
                        "communication:custom_allreduce_polling_and_control",
                    }))),
                )
        missing_activity = tuple(
            sorted(
                {
                    f"gpu_activity:{op.name}"
                    for op in flat_program.ops
                    if isinstance(op, ComputeOp)
                    and not op.launch_shadow
                    and max(op.duration, default=0.0) > 0.0
                    and op.activity == ZERO_ACTIVITY
                    and not (flat_program.meta.misc.get("vllm_allreduce") and op.embedded_collectives and sum(
                        call.duration_seconds for call in op.embedded_collectives
                    ) == max(op.duration, default=0.0))
                }
            )
        )
        missing = tuple(
            sorted(set(communication_coverage.missing).union(missing_activity))
        )
        coverage = EnergyCoverage(
            gpu_activity="partial" if missing_activity else "complete",
            gpu_static="complete",
            communication=communication_coverage.communication,
            missing=missing,
        )
        return ExecutionResult(
            total_time=result.total_time,
            energy=ExecutionEnergyResult(
                gpu=gpu,
                communication=communication,
                coverage=coverage,
                execution_mode=mode.value,
                collective_launch_records=self._collective_launch_records(flat_program),
                tp_allreduce_records=ar_records,
            ),
        )

    def _collective_launch_records(self, flat_program: Program) -> tuple:
        from sim_settings import nccl_latency_rule
        if (not getattr(self.time_calc.hw_config.sw_config, "collective_launch_overhead", 0.0)
                and nccl_latency_rule(self.time_calc.hw_config) not in ("tuning_formula", "tuning_protocol")):
            return ()
        runs = list(getattr(self, "_astra_timing_runs", [
            (scope, program, scale) for scope, program, _, scale in self._astra_energy_runs
        ]))
        if self.pipeline_program is not None and not any(
            program is self.pipeline_program for _, program, _ in runs
        ):
            runs.append(("pipeline_analytical", self.pipeline_program, 1.0))
        records = []
        for scope, program, scale in runs:
            if scope.startswith("transformer"):
                want_moe = scope.startswith("transformer_moe_")
                direction = Direction.FORWARD if scope.endswith("_forward") else Direction.BACKWARD
                consumers = {
                    op.work for op in flat_program.ops
                    if isinstance(op, ComputeOp) and op.work is not None
                    and op.work.kind in (WorkKind.LAYER, WorkKind.RECOMPUTE)
                    and op.work.direction is direction
                    and bool(op.is_moe_layer) == want_moe
                }
                scale *= len(consumers) * max(1, int(flat_program.dp_count))
            elif scope.startswith("pipeline") and program.meta.misc.get("sharding_policy") != "zero_moe":
                scale *= self.workload.degrees.tp * self.workload.degrees.cp * self.workload.degrees.ep
            if scale:
                records.extend(
                    dict(row, scope=scope, rank=int(rank), multiplicity=scale)
                    for rank, row in program.meta.misc.get("collective_launch_stats", {}).items()
                )
        return tuple(records)

    def _collect_block_timings(
        self,
        timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings],
    ) -> Any:
        from program.retime import BlockTimings

        return BlockTimings(
            dense=timings or self._transformer_baseline_timings,
            moe=moe_timings or self._transformer_moe_baseline_timings,
            stage_dense=dict(self._transformer_stage_timings),
            stage_moe=dict(self._transformer_stage_moe_timings),
        )

    def _run_pipeline_with_analytical_comm(
        self,
        declared_mode: ExecutionMode,
        pipeline_program: Optional[Program] = None,
    ) -> ExecutionResult:
        from program import analytic_sim

        if declared_mode == ExecutionMode.HYBRID:
            filename = "/hybrid_graph_no_dp" if self.no_data_parallel else "/hybrid_graph"
        else:
            filename = "/analytical_graph_no_dp" if self.no_data_parallel else "/analytical_graph"

        if pipeline_program is None:
            pipeline_program = self._build_program(
                Granularity.PIPELINE, label=self._pipeline_label()
            )
        self.pipeline_program = pipeline_program

        from program.nccl_latency import enabled as launch_enabled, stamp
        if declared_mode is ExecutionMode.ANALYTICAL and launch_enabled(pipeline_program):
            self._stamp_analytical_layer_launches(pipeline_program)
            stamp(pipeline_program, self.time_calc.hw_config)

        total_time = analytic_sim.evaluate(
            pipeline_program,
            self.time_calc.network_model,
            self.interconnect_params,
        )

        self._comm_sizes_converted = True

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            from program.viz import save_program_graph

            save_program_graph(pipeline_program, self.time_calc.output_dir, filename)

        total_time *= self._interleave_scale()
        from program import megatron
        if megatron.kernel_timing(self.time_calc):
            total_time = megatron.serial_time(self.build_flat_program_for_memory())
        return ExecutionResult(total_time=total_time)

    def _stamp_analytical_layer_launches(self, program: Program) -> None:
        from program.collective_launch import is_nccl_call
        from program.ir import EmbeddedCollective

        for op in program.ops:
            work = getattr(op, "work", None)
            if not isinstance(op, ComputeOp) or work is None or work.kind not in (WorkKind.LAYER, WorkKind.RECOMPUTE):
                continue
            template = self.workload.block_template(work.layer)
            table = self.workload.block_comm(work.layer)
            direction = "forward" if work.direction is Direction.FORWARD else "backward"
            calls = []
            for entry in template.entries:
                for key in entry.direction(direction).comm_keys:
                    spec = table[key]
                    from program.vllm_allreduce import choice_for
                    choice = choice_for(program.meta.misc.get("vllm_allreduce"), spec.kind,
                                        spec.participants, spec.size_bytes, "*".join(spec.axes))
                    if is_nccl_call(spec.kind, participants=spec.participants, size_bytes=spec.size_bytes,
                                    algorithm=choice.algorithm):
                        calls.append(EmbeddedCollective(
                            name=key, coll=spec.kind, size_bytes=spec.size_bytes,
                            participants=spec.participants, interconnect="*".join(spec.axes),
                            offset_seconds=0.0, duration_seconds=0.0,
                        ))
            op.embedded_collectives = tuple(calls)

    def _run_hybrid(self) -> ExecutionResult:
        from program.retime import apply_block_timings

        transformer_time, moe_transformer_time = self._run_transformer_astrasim()

        pipeline_program = self._build_program(
            Granularity.PIPELINE, label=self._pipeline_label()
        )
        if transformer_time is not None or moe_transformer_time is not None:
            self._write_block_durations(transformer_time, moe_transformer_time)
            apply_block_timings(
                pipeline_program,
                self._collect_block_timings(transformer_time, moe_transformer_time),
                self.workload.run.retime_dp_count(self.workload.degrees),
            )
        return self._run_pipeline_with_analytical_comm(
            ExecutionMode.HYBRID, pipeline_program=pipeline_program
        )

    def _run_full_astrasim_hierarchical(self) -> ExecutionResult:
        from program.retime import apply_block_timings

        transformer_time, moe_transformer_time = self._run_transformer_astrasim()

        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_hier")

        pipeline_program = self._emission_program(
            Granularity.PIPELINE, label=self._pipeline_label(), artifact_dir=artifact_dir
        )
        if transformer_time is not None or moe_transformer_time is not None:
            self._write_block_durations(transformer_time, moe_transformer_time)
            apply_block_timings(
                pipeline_program,
                self._collect_block_timings(transformer_time, moe_transformer_time),
                self.workload.run.retime_dp_count(self.workload.degrees),
            )
        self.pipeline_program = pipeline_program

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            from program.viz import save_program_graph

            filename = (
                "/pipeline_graph_hierarchical_no_dp"
                if self.no_data_parallel
                else "/pipeline_graph_hierarchical"
            )
            save_program_graph(pipeline_program, self.time_calc.output_dir, filename)

        astra_result = run_astra_simulation_only_onepath(
            pipeline_program,
            self.time_calc,
            artifact_dir,
            persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            faulty_links_override=self._fault_override("pipeline"),
            rank_layout=self._pipeline_rank_layout or None,
        )
        per_rank_sec, max_sec = astra_result
        self._record_astra_energy_run(
            "pipeline", pipeline_program, astra_result, artifact_dir=artifact_dir,
            faulty_links_override=self._fault_override("pipeline"),
            rank_layout=self._pipeline_rank_layout or None,
        )
        if max_sec <= 0:
            raise RuntimeError("AstraSim pipeline execution returned non-positive duration Re-run with sw_param.validate_graph: true (or RAPID_VALIDATE_GRAPH=1) for build-time graph diagnostics.")
        max_sec *= self._interleave_scale()
        return ExecutionResult(total_time=max_sec)

    def _run_full_astrasim_flattened(self) -> ExecutionResult:
        artifact_dir = self.time_calc.output_dir
        if self.time_calc.persist_astrasim_artifacts:
            artifact_dir = os.path.join(self.time_calc.output_dir, "astra_flat")

        program = self._emission_program(
            Granularity.FLAT,
            label="flat_no_dp" if self.no_data_parallel else "flat",
            artifact_dir=artifact_dir,
        )
        self.flat_program = program

        if _env_flag("RAPID_VISUALIZE_GRAPHS"):
            from program.viz import save_program_graph

            filename = (
                "/pipeline_graph_pre_flatten_no_dp"
                if self.no_data_parallel
                else "/pipeline_graph_pre_flatten"
            )
            save_program_graph(
                self._build_program(Granularity.PIPELINE, label=self._pipeline_label()),
                self.time_calc.output_dir,
                filename,
            )

        astra_result = run_astra_simulation_only_onepath(
            program,
            self.time_calc,
            artifact_dir,
            persist_artifacts=self.time_calc.persist_astrasim_artifacts,
            rank_layout=self._rank_layout or None,
        )
        per_rank_sec, max_sec = astra_result
        self._record_astra_energy_run(
            "flattened", program, astra_result, artifact_dir=artifact_dir,
            rank_layout=self._rank_layout or None,
        )

        if not per_rank_sec:
            raise RuntimeError("AstraSim flattened execution returned no per-rank timings Re-run with sw_param.validate_graph: true (or RAPID_VALIDATE_GRAPH=1) for build-time graph diagnostics.")

        effective_dp = self.workload.run.effective_dp(self.workload.degrees)
        expected_rank_count = effective_dp * len(program.compute_devices())

        if expected_rank_count == 1:
            if len(per_rank_sec) > 2:
                raise RuntimeError(
                    "AstraSim rank count mismatch for flattened execution: "
                    f"expected {expected_rank_count}, got {len(per_rank_sec)}"
                )
            per_rank_sec = per_rank_sec[:1]
        if len(per_rank_sec) != expected_rank_count:
            raise RuntimeError(
                "AstraSim rank count mismatch for flattened execution: "
                f"expected {expected_rank_count}, got {len(per_rank_sec)}"
            )

        if max_sec <= 0:
            raise RuntimeError("AstraSim flattened execution returned non-positive duration Re-run with sw_param.validate_graph: true (or RAPID_VALIDATE_GRAPH=1) for build-time graph diagnostics.")

        return ExecutionResult(total_time=max_sec * self._interleave_scale())

    def build_flat_program_for_memory(self) -> Program:
        if self._memory_flat_program is not None:
            return self._memory_flat_program

        program = self.flat_program
        if program is None:
            program = self._build_program(
                Granularity.FLAT,
                label="flat_no_dp" if self.no_data_parallel else "flat",
            )
        self._stamp_memory_comm_durations(program)
        self._memory_flat_program = program
        return program

    def _stamp_memory_comm_durations(self, program: Program) -> None:
        from program.analytic_sim import COMM_DURATIONS_KEY, comm_durations

        if not self._comm_sizes_converted:
            program.meta.misc.pop(COMM_DURATIONS_KEY, None)
            return
        program.meta.misc[COMM_DURATIONS_KEY] = comm_durations(
            program,
            self.time_calc.network_model,
            self.interconnect_params,
            collective_keys=set(self.workload.comm),
        )

    def _build_transformer_block_programs(
        self, *, moe: bool, label: str
    ) -> Tuple[Optional[Program], Optional[Program]]:
        block_workload = self.workload.for_block(
            layout=_layout_of(self._transformer_rank_layout), moe=moe
        )
        forward = self._build_program(
            Granularity.BLOCK,
            label=f"{label}_forward",
            workload=block_workload,
            directions=(Direction.FORWARD,),
        )
        backward = None
        if self.workload.run.include_backward:
            backward = self._build_program(
                Granularity.BLOCK,
                label=f"{label}_backward",
                workload=block_workload,
                directions=(Direction.BACKWARD,),
            )
        return forward, backward

    def _run_transformer_astrasim(
        self,
    ) -> Tuple[Optional[TransformerTimings], Optional[TransformerTimings]]:
        blocks = self.workload.blocks
        has_moe = blocks.moe is not None

        persist = self.time_calc.persist_astrasim_artifacts
        os.makedirs(self.time_calc.output_dir, exist_ok=True)
        self._transformer_stage_timings = {}
        self._transformer_stage_moe_timings = {}
        self._transformer_baseline_timings = None
        self._transformer_moe_baseline_timings = None

        forward_program, backward_program = self._build_transformer_block_programs(
            moe=False, label="block_dense"
        )

        baseline_fwd_dir, baseline_bwd_dir = self._transformer_artifact_dirs(label=None, persist=persist)
        baseline_timings, baseline_fwd_per_rank, baseline_bwd_per_rank = self._execute_transformer_run(
            baseline_fwd_dir,
            baseline_bwd_dir,
            forward_program=forward_program,
            backward_program=backward_program,
            faulty_links_override=(),
            energy_scope="transformer_dense",
        )
        self._transformer_baseline_timings = baseline_timings
        self.time_calc.transformer_astrasim_per_rank_forward = baseline_fwd_per_rank
        self.time_calc.transformer_astrasim_per_rank_backward = baseline_bwd_per_rank
        self.time_calc.transformer_astrasim_time_forward = baseline_timings.forward
        self.time_calc.transformer_astrasim_time_backward = baseline_timings.backward

        stage_dp_faults = self._transformer_stage_dp_faults
        for fault_index, ((dp_idx, stage_id), fault_links) in enumerate(sorted(stage_dp_faults.items())):
            variant = f"fault{fault_index}_dp{dp_idx}_stage{stage_id}"
            stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=variant, persist=persist)
            stage_timings, _, _ = self._execute_transformer_run(
                stage_fwd_dir,
                stage_bwd_dir,
                forward_program=forward_program,
                backward_program=backward_program,
                faulty_links_override=fault_links,
            )
            self._transformer_stage_timings[(dp_idx, stage_id)] = stage_timings

        moe_timings: Optional[TransformerTimings] = None
        if has_moe:
            moe_forward_program, moe_backward_program = self._build_transformer_block_programs(
                moe=True, label="block_moe"
            )
            moe_fwd_dir, moe_bwd_dir = self._transformer_artifact_dirs(label="moe", persist=persist)
            moe_timings, moe_fwd_per_rank, moe_bwd_per_rank = self._execute_transformer_run(
                moe_fwd_dir,
                moe_bwd_dir,
                forward_program=moe_forward_program,
                backward_program=moe_backward_program,
                faulty_links_override=(),
                energy_scope="transformer_moe",
            )
            self._transformer_moe_baseline_timings = moe_timings
            self.time_calc.transformer_astrasim_per_rank_forward_moe = moe_fwd_per_rank
            self.time_calc.transformer_astrasim_per_rank_backward_moe = moe_bwd_per_rank
            self.time_calc.transformer_astrasim_time_forward_moe = moe_timings.forward
            self.time_calc.transformer_astrasim_time_backward_moe = moe_timings.backward
            for fault_index, ((dp_idx, stage_id), fault_links) in enumerate(sorted(stage_dp_faults.items())):
                variant = f"moe_fault{fault_index}_dp{dp_idx}_stage{stage_id}"
                stage_fwd_dir, stage_bwd_dir = self._transformer_artifact_dirs(label=variant, persist=persist)
                stage_timings, _, _ = self._execute_transformer_run(
                    stage_fwd_dir,
                    stage_bwd_dir,
                    forward_program=moe_forward_program,
                    backward_program=moe_backward_program,
                    faulty_links_override=fault_links,
                )
                self._transformer_stage_moe_timings[(dp_idx, stage_id)] = stage_timings

        return baseline_timings, moe_timings

    def _transformer_artifact_dirs(self, label: Optional[str], persist: bool) -> Tuple[str, str]:
        if not persist:
            base_dir = self.time_calc.output_dir
            os.makedirs(base_dir, exist_ok=True)
            return base_dir, base_dir
        base_dir = os.path.join(self.time_calc.output_dir, "astra_hier")
        os.makedirs(base_dir, exist_ok=True)
        if label is None:
            fwd_dir = os.path.join(base_dir, "fwd")
            bwd_dir = os.path.join(base_dir, "bwd")
        else:
            fwd_dir = os.path.join(base_dir, f"{label}_fwd")
            bwd_dir = os.path.join(base_dir, f"{label}_bwd")
        os.makedirs(fwd_dir, exist_ok=True)
        os.makedirs(bwd_dir, exist_ok=True)
        return fwd_dir, bwd_dir

    def _execute_transformer_run(
        self,
        artifact_dir_fwd: str,
        artifact_dir_bwd: str,
        *,
        forward_program: Optional[Program],
        backward_program: Optional[Program],
        faulty_links_override: Optional[Tuple[Tuple[int, int, float], ...]],
        energy_scope: Optional[str] = None,
    ) -> Tuple[TransformerTimings, Optional[List[float]], Optional[List[float]]]:
        fwd_per_rank = None
        bwd_per_rank = None
        fwd_max = 0
        bwd_max = 0

        if forward_program is not None:
            fwd_result = run_astra_simulation_only_onepath(
                forward_program,
                self.time_calc,
                artifact_dir_fwd,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
                faulty_links_override=faulty_links_override,
                rank_layout=self._transformer_rank_layout or None,
            )
            fwd_per_rank, fwd_max = fwd_result
            if energy_scope is not None:
                self._record_astra_energy_run(
                    f"{energy_scope}_forward", forward_program, fwd_result,
                    artifact_dir=artifact_dir_fwd,
                    faulty_links_override=faulty_links_override,
                    rank_layout=self._transformer_rank_layout or None,
                )
            if fwd_max <= 0:
                raise RuntimeError("AstraSim transformer forward execution returned non-positive duration")

        if backward_program is not None:
            bwd_result = run_astra_simulation_only_onepath(
                backward_program,
                self.time_calc,
                artifact_dir_bwd,
                persist_artifacts=self.time_calc.persist_astrasim_artifacts,
                faulty_links_override=faulty_links_override,
                rank_layout=self._transformer_rank_layout or None,
            )
            bwd_per_rank, bwd_max = bwd_result
            if energy_scope is not None:
                self._record_astra_energy_run(
                    f"{energy_scope}_backward", backward_program, bwd_result,
                    artifact_dir=artifact_dir_bwd,
                    faulty_links_override=faulty_links_override,
                    rank_layout=self._transformer_rank_layout or None,
                )
            if bwd_max < 0:
                raise RuntimeError("AstraSim transformer backward execution returned non-positive duration")

        return TransformerTimings(forward=fwd_max, backward=bwd_max), fwd_per_rank, bwd_per_rank

    def _write_block_durations(
        self,
        timings: Optional[TransformerTimings],
        moe_timings: Optional[TransformerTimings] = None,
    ) -> None:
        if timings is None and moe_timings is None:
            return
        baseline = timings or self._transformer_baseline_timings
        moe_baseline = moe_timings or self._transformer_moe_baseline_timings
        self.workload.durations.write_block_timings(
            dense_forward=None if baseline is None else baseline.forward,
            dense_backward=None if baseline is None else baseline.backward,
            moe_forward=None if moe_baseline is None else moe_baseline.forward,
            moe_backward=None if moe_baseline is None else moe_baseline.backward,
        )

def _layout_of(descriptor: Optional[Mapping[str, Any]]) -> Optional[RankLayout]:
    if not descriptor or not descriptor.get("axis_order"):
        return None
    return RankLayout(
        axis_order=tuple(descriptor["axis_order"]),
        axis_sizes=dict(descriptor["axis_sizes"]),
        axis_strides=dict(descriptor["axis_strides"]),
    )
