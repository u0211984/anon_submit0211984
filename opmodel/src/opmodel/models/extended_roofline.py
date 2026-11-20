from __future__ import annotations

import math
import csv
import atexit
import json
import os
import socket
import sys
import uuid
from collections import Counter, OrderedDict
from dataclasses import dataclass, fields
from itertools import accumulate
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from opmodel import settings as _settings
from opmodel.memo import SettingsMemo, hardware_token, replace, typed

from opmodel.api import DType, EngineKind, LocalOp, MemoryAccess, OpKind, OpProfile, TensorRole
from opmodel.energy import apply_calibrated_energy_model
from opmodel.estimator import DispatchingOpModel
from opmodel.hardware import HardwareSpec, MemoryLevel
from opmodel.models.roofline import (
    AttentionEstimator,
    CopyEstimator,
    ElementwiseEstimator,
    EmbeddingEstimator,
    NormEstimator,
    ReductionEstimator,
    SoftmaxEstimator,
    _make_profile,
    _matmul_engine,
)
from opmodel.models.simple_energy import estimate_energy
from opmodel.ops import (
    dtype_nbytes,
    footprint_from_tensors,
    get_tensors,
    parse_batched_gemm,
    parse_gemm,
)
from opmodel.split_k import normalize_split_k, split_k_counts, split_k_lengths

class ExtendedRooflineModel(DispatchingOpModel):
    def __init__(self) -> None:
        super().__init__(
            {
                OpKind.GEMM: ExtendedGemmEstimator(batched=False),
                OpKind.BATCHED_GEMM: ExtendedGemmEstimator(batched=True),
                OpKind.ATTENTION_PREFILL: AttentionEstimator(prefill=True),
                OpKind.ATTENTION_DECODE: AttentionEstimator(prefill=False),
                OpKind.SOFTMAX: SoftmaxEstimator(),
                OpKind.LAYERNORM: NormEstimator(layernorm=True),
                OpKind.RMSNORM: NormEstimator(layernorm=False),
                OpKind.ELEMENTWISE: ElementwiseEstimator(),
                OpKind.REDUCTION: ReductionEstimator(),
                OpKind.EMBEDDING: EmbeddingEstimator(),
                OpKind.COPY: CopyEstimator(),
            }
        )

@dataclass(frozen=True)
class GemmProblemSpec:
    batch: int
    m: int
    n: int
    k: int
    input_dtype: DType
    output_dtype: DType
    weight_is_batched: bool
    beta_zero: bool
    epilogue_reads_c: bool
    transpose_a: bool
    transpose_b: bool

    a_bytes: float | None = None
    b_bytes: float | None = None

def _operand_bytes(problem: "GemmProblemSpec", dtype_bytes: float) -> tuple[float, float]:
    return (dtype_bytes if problem.a_bytes is None else problem.a_bytes,
            dtype_bytes if problem.b_bytes is None else problem.b_bytes)

def _mixed_operands(problem: "GemmProblemSpec") -> bool:
    return problem.a_bytes is not None or problem.b_bytes is not None

@dataclass(frozen=True)
class GemmKernelSpec:
    cta_m: int
    cta_n: int
    cta_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    num_warp_tile_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    pipeline_stages: int
    warps_per_cta: int
    threads_per_cta: int
    registers_per_thread: int
    max_concurrent_ctas_per_sm: int | None
    slice_k: bool

    order_sweep_axis: str | None = None
    order_strip_width: int = 0
    order_serpentine: bool = False

    store_evict_first: bool | None = None

    split_k_slices: int = 1
    split_k_reduction: str = "none"
    split_k_partial_bytes: int = 2

    split_k: int = 1
    reduction_scheme: str | int = "none"
    architecture: str = "sm80"

    def __post_init__(self) -> None:
        normalize_split_k(self.split_k, self.reduction_scheme)

@dataclass(frozen=True)
class GemmKernelTemplate:
    name: str
    cta_m: int
    cta_n: int
    cta_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    num_warp_tile_k: int
    mma_m: int
    mma_n: int
    mma_k: int
    pipeline_stages: int
    warps_per_cta: int
    registers_per_thread: int
    source: str
    dtypes: tuple[DType, ...]
    split_k: int = 1
    reduction_scheme: str | int = "none"
    architecture: str = "sm80"

    def __post_init__(self) -> None:
        normalize_split_k(self.split_k, self.reduction_scheme)

@dataclass(frozen=True)
class GemmCandidateEvaluation:
    template: GemmKernelTemplate
    kernel: GemmKernelSpec
    profile: OpProfile
    selection_energy_j: float
    cheap_rank: int
    cheap_score: float

@dataclass(frozen=True)
class GridAccounting:
    blocks_m: int
    blocks_n: int
    k_stages: int
    cta_count: int
    useful_flops: float
    issued_flops: float
    tile_efficiency: float

@dataclass(frozen=True)
class L2CapacityEstimate:
    capacity_bytes: int | None
    working_set_bytes: int
    miss_fraction: float
    a_dram_read_bytes: int
    b_dram_read_bytes: int
    a_capacity_miss_bytes: int
    b_capacity_miss_bytes: int

@dataclass(frozen=True)
class L2ReuseFootprint:
    peak_working_set_bytes: int
    weighted_working_set_bytes: float
    miss_fraction: float
    k_window_stages: int
    k_window_count: int
    full_wave_full_window_bytes: int
    full_wave_tail_window_bytes: int
    last_wave_full_window_bytes: int
    last_wave_tail_window_bytes: int

@dataclass(frozen=True)
class L2ReuseDistanceEstimate:
    capacity_estimate: L2CapacityEstimate
    a_within_wave_repeated_bytes: int
    a_cross_wave_repeated_bytes: int
    b_within_wave_repeated_bytes: int
    b_cross_wave_repeated_bytes: int
    within_wave_miss_fraction: float
    a_cross_wave_reuse_distance_bytes: float
    b_cross_wave_reuse_distance_bytes: float
    a_cross_wave_miss_fraction: float
    b_cross_wave_miss_fraction: float

@dataclass(frozen=True)
class TrafficAccounting:
    memory_access: MemoryAccess
    a_logical_bytes: int
    b_logical_bytes: int
    c_read_logical_bytes: int
    d_store_logical_bytes: int
    a_l2_requested_bytes: int
    b_l2_requested_bytes: int
    a_dram_unique_bytes: int
    b_dram_unique_bytes: int
    a_dram_read_bytes: int
    b_dram_read_bytes: int
    c_read_transaction_bytes: int
    d_store_transaction_bytes: int
    l2_requested_bytes: int
    l2_capacity_bytes: int | None
    l2_operation_working_set_bytes: int
    l2_reuse_working_set_bytes: int
    l2_weighted_reuse_working_set_bytes: float
    l2_capacity_miss_fraction: float
    l2_k_window_stages: int
    l2_k_window_count: int
    l2_full_wave_full_window_bytes: int
    l2_full_wave_tail_window_bytes: int
    l2_last_wave_full_window_bytes: int
    l2_last_wave_tail_window_bytes: int
    a_within_wave_repeated_bytes: int
    a_cross_wave_repeated_bytes: int
    b_within_wave_repeated_bytes: int
    b_cross_wave_repeated_bytes: int
    l2_within_wave_miss_fraction: float
    a_cross_wave_reuse_distance_bytes: float
    b_cross_wave_reuse_distance_bytes: float
    a_cross_wave_miss_fraction: float
    b_cross_wave_miss_fraction: float
    dram_capacity_miss_bytes: int
    dram_unique_bytes: int
    dram_total_bytes: int
    smem_read_bytes: int
    smem_write_bytes: int
    sector_size_bytes: int
    line_size_bytes: int

    @property
    def smem_total_bytes(self) -> int:
        return self.smem_read_bytes + self.smem_write_bytes

@dataclass(frozen=True)
class OccupancyResult:
    num_sms: int
    resident_ctas_per_sm: int
    total_resident_ctas: int
    ctas_per_wave: int
    wave_count: int
    full_wave_ctas: int
    last_wave_ctas: int
    last_wave_busy_sms: int
    last_wave_lazy_sms: int
    last_wave_busy_ctas_per_sm: int
    last_wave_lazy_ctas_per_sm: int
    tail_efficiency: float
    limiting_factors: tuple[str, ...]

@dataclass(frozen=True)
class WavePipelineResult:
    active_ctas: int
    busy_sms: int
    lazy_sms: int
    busy_ctas_per_sm: int
    lazy_ctas_per_sm: int
    start_cycles: float
    work_cycles: float
    end_cycles: float
    total_cycles: float
    sm_stage_cycles: float
    sm_last_stage_cycles: float
    smem_group_cycles: float
    math_group_cycles: float
    math_issue_group_cycles: float
    math_latency_group_cycles: float
    memory_full_stage_cycles: float
    memory_last_stage_cycles: float
    l2_full_stage_cycles: float
    l2_last_stage_cycles: float
    dram_full_stage_cycles: float
    dram_last_stage_cycles: float
    epilogue_global_cycles: float
    epilogue_smem_cycles: float
    slice_k_extra_cycles: float
    epilogue_smem_bytes: float
    epilogue_l2_bytes: float
    epilogue_dram_bytes: float
    l2_epilogue_cycles: float
    dram_epilogue_cycles: float
    exposed_l2_cycles: float
    exposed_dram_cycles: float

@dataclass(frozen=True)
class TimelineResult:
    kernel_cycles: float
    cta_cycles: float
    full_wave: WavePipelineResult
    last_wave: WavePipelineResult
    prologue_cycles: float
    work_cycles: float
    stage_cycles: float
    epilogue_cycles: float
    compute_stage_cycles: float
    mma_issue_cycles: float
    mma_dependency_penalty_cycles: float
    mma_ilp_efficiency: float
    smem_stage_cycles: float
    global_load_issue_cycles: float
    l2_service_cycles: float
    dram_service_cycles: float
    exposed_l2_cycles: float
    exposed_dram_cycles: float
    compute_active_cycles: float
    smem_active_cycles: float
    l2_active_cycles: float
    dram_active_cycles: float
    epilogue_smem_bytes: float
    epilogue_l2_bytes: float
    epilogue_dram_bytes: float
    groups_k: int
    last_stage_groups_k: int
    last_stage_k: int
    memory_pipeline_groups: int
    last_memory_pipeline_stages: int
    compute_issue_utilization: float
    compute_latency_utilization: float
    compute_active_utilization: float
    smem_utilization: float
    l2_utilization: float
    dram_utilization: float
    compute_smem_overlap: float
    compute_l2_overlap: float
    compute_dram_overlap: float

_GEMM_SELECTION_OBJECTIVES = frozenset(("latency", "energy"))
_GEMM_SELECTION_BACKENDS = frozenset(("extended_roofline",))
_DEFAULT_GEMM_SELECTION_SHORTLIST_SIZE = 12
_EXPLICIT_KERNEL_ATTRS = frozenset(
    (
        "cta_tile_m",
        "cta_tile_n",
        "cta_tile_k",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
        "num_warp_tile_k",
        "num_warp_tile_K",
        "mma_m",
        "mma_n",
        "mma_k",
        "pipeline_stages",
        "warps_per_cta",
        "threads_per_cta",
        "registers_per_thread",
        "resident_ctas_per_sm",
        "max_concurrent_block",
        "slice_k",
        "sliceK",
    )
)

_BF16_FP16_DTYPES = (DType.BF16, DType.FP16)

_GEMM_CATALOG_DIR = Path(__file__).resolve().parent.parent / "configs" / "gemm_catalogs"
_GEMM_CATALOG_FILES = {"default": "sm80_default.yaml", "cublas_sm80": "cublas_sm80.yaml", "cutlass_sm80_16816": "cutlass_sm80_16816.yaml", "cublaslt_sm80_bf16": "cublaslt_sm80_bf16.yaml", "cublaslt_sm80_bf16_splitk": "cublaslt_sm80_bf16_splitk.yaml"}
_GEMM_CATALOG_CACHE: dict[str, tuple[GemmKernelTemplate, ...]] = {}

_LTLIST_SPLIT_K_WORKSPACE_BYTES = 8519680

_FORCED_CACHE: dict[str, dict] = {}
_FORCED_HITS: Counter = Counter()
_FORCED_FALLBACKS: Counter = Counter()

def reset_forced_template_diagnostics() -> None:
    _FORCED_HITS.clear()
    _FORCED_FALLBACKS.clear()
    globals().pop("_TRACE_SPLIT_K_BY_PHASE", None)

def forced_template_diagnostics() -> dict:
    return {"forced": _FORCED_HITS["forced"], "fallback": _FORCED_HITS["fallback"],
            "split_k": _FORCED_HITS["split_k"],
            "split_k_by_scheme": {k: _FORCED_HITS["scheme_" + k] for k in ("inplace", "compute_type", "output_type")},
            "trace_split_k_by_phase": globals().get("_TRACE_SPLIT_K_BY_PHASE", {}),
            "fallback_shapes": [{"shape": list(k), "hits": v} for k, v in sorted(_FORCED_FALLBACKS.items())]}

def _write_forced_template_diagnostics() -> None:
    directory = os.environ.get("OPMODEL_GEMM_DIAGNOSTICS_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        filename = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex}.json"
        report = {**forced_template_diagnostics(), "argv": sys.argv}
        (path / filename).write_text(json.dumps(report, sort_keys=True) + "\n")

atexit.register(_write_forced_template_diagnostics)

def _load_forced_templates(path: str) -> dict:
    if path not in _FORCED_CACHE:
        rows: dict = {}
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if row["primary_for_row"] != "1":
                    continue
                key = tuple(int(row[k]) for k in ("dim1", "dim2", "dim3", "batch"))
                rows.setdefault(key, []).append(row)
        _FORCED_CACHE[path] = rows
    return _FORCED_CACHE[path]

def _forced_template(row: Mapping[str, str]) -> GemmKernelTemplate:
    m, n, k = (int(row[x]) for x in ("model_cta_m", "model_cta_n", "model_cta_k"))

    raw_m, raw_n = int(row["tile_m"]), int(row["tile_n"])
    warp_raw_m = min(raw_m // 2, 64)
    warp_raw_n = min(raw_n // 2, 64)
    warp_m, warp_n = ((warp_raw_n, warp_raw_m) if row["model_axes_swapped_from_cublas"] == "1"
                      else (warp_raw_m, warp_raw_n))
    reg = row["registers_per_thread"]
    source = row["reg_source_rows"]
    if not reg:
        if (raw_m, raw_n, k, int(row["stages"])) != (128, 64, 64, 6):
            raise ValueError(f"missing register evidence for {row['shape_id']}")
        reg = "146"
        source = "CUTLASS stand-in: paper_audit/cutlass_regs/cutlass_sm80_16816_resources.csv:34 (128x64x32 s6 TN, REG 146)"
    return GemmKernelTemplate(
        name="lt_heur_" + row["shape_id"], cta_m=m, cta_n=n, cta_k=k,
        warp_m=warp_m, warp_n=warp_n, warp_k=k, num_warp_tile_k=1,
        mma_m=16, mma_n=8, mma_k=16, pipeline_stages=int(row["stages"]),
        warps_per_cta=(m // warp_m) * (n // warp_n), registers_per_thread=int(reg),
        source=source, dtypes=(DType.BF16,),
        split_k=int(row["splitK"]), reduction_scheme=int(row["reduction_scheme"]))

def resolve_forced_template_for_shape(shape, role="forward", *, hardware=None):
    forced_path = _settings.get("OPMODEL_GEMM_FORCED_TEMPLATES")
    if not forced_path:
        return None, []
    if hardware is not None and hardware.architecture == "sm90":
        from . import sm90_gemm
        return sm90_gemm.resolve_forced(shape, role, hardware)
    rows = _load_forced_templates(str(forced_path)).get(tuple(shape), [])
    transposes = {"forward": ("T", "N"), "dgrad": ("N", "N"),
                  "wgrad": ("N", "T")}
    if role in transposes:
        rows = [r for r in rows if (r["cublas_transa"], r["cublas_transb"]) == transposes[role]]
    elif len(rows) > 1:
        rows = []
    return (_forced_template(rows[0]) if rows else None), rows

def _resolve_forced_template(problem: GemmProblemSpec, role="forward", *, hardware=None):
    if not _settings.get("OPMODEL_GEMM_FORCED_TEMPLATES"):
        return None, None
    shape = (problem.m, problem.k, problem.n, problem.batch)
    template, rows = resolve_forced_template_for_shape(shape, role, hardware=hardware)
    status = "forced" if rows else "fallback"
    _FORCED_HITS[status] += 1
    metadata = {"status": status, "shape": list(shape), "role": role,
                "source_rows": [r["shape_id"] for r in rows],
                "split_k_requested": int(rows[0]["splitK"]) if rows else None,
                "split_k_modelled": template.split_k if template else None}
    if rows:
        if template.split_k > 1:
            _FORCED_HITS["split_k"] += 1
            scheme = normalize_split_k(template.split_k, template.reduction_scheme)[1]
            _FORCED_HITS["scheme_" + scheme] += 1
        return template, metadata
    _FORCED_FALLBACKS[shape] += 1
    return None, metadata

def gemm_catalog_names() -> tuple[str, ...]:
    return tuple(_GEMM_CATALOG_FILES)

def _load_catalog_yaml(text: str):
    import yaml

    loader = yaml.CSafeLoader if getattr(yaml, "__with_libyaml__", False) else yaml.SafeLoader
    return yaml.load(text, Loader=loader)

def load_gemm_catalog(name: str) -> tuple[GemmKernelTemplate, ...]:
    if name not in _GEMM_CATALOG_CACHE:
        if name not in _GEMM_CATALOG_FILES:
            raise ValueError(f"unknown GEMM catalog {name!r}; known {gemm_catalog_names()}")
        data = _load_catalog_yaml((_GEMM_CATALOG_DIR / _GEMM_CATALOG_FILES[name]).read_text())
        _GEMM_CATALOG_CACHE[name] = tuple(
            GemmKernelTemplate(
                str(t["name"]), *(int(x) for x in t["cta"]), *(int(x) for x in t["warp"]),
                int(t["num_warp_tile_k"]), *(int(x) for x in t["mma"]), int(t["stages"]), int(t["warps"]),
                int(t["registers_per_thread"]), str(t["source"]), tuple(DType(str(d)) for d in t["dtypes"]),
                split_k=t.get("split_k", 1), reduction_scheme=t.get("reduction_scheme", "none"),
            )
            for t in data["templates"]
        )
    return _GEMM_CATALOG_CACHE[name]

class ExtendedGemmEstimator:
    def __init__(self, *, batched: bool) -> None:
        self._batched = batched

    def estimate(self, op: LocalOp, hardware: HardwareSpec) -> OpProfile:
        problem = _problem_spec(op, batched=self._batched)
        footprint = footprint_from_tensors(op)
        engine = _matmul_engine(problem.input_dtype, hardware)
        if engine != EngineKind.TENSOR:
            return _make_profile(
                hardware=hardware,
                dtype=problem.input_dtype,
                flops=float(2 * problem.batch * problem.m * problem.n * problem.k),
                hbm_read=footprint.input_bytes + footprint.weight_bytes,
                hbm_write=footprint.output_bytes,
                engine=engine,
                footprint=footprint,
                implementation=_implementation_name(self._batched),
                diagnostics={"tensor_fallback": engine.value},
            )

        if _should_select_gemm_kernel(op.attrs, hardware):
            return _estimate_selected_gemm_kernel(
                op=op,
                problem=problem,
                footprint=footprint,
                hardware=hardware,
                batched=self._batched,
            )

        warnings: list[str] = []
        kernel = _kernel_spec(op.attrs, hardware, warnings)
        return _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=self._batched,
            warnings=warnings,
        )

def _estimate_gemm_with_kernel(
    *,
    problem: GemmProblemSpec,
    footprint,
    kernel: GemmKernelSpec,
    hardware: HardwareSpec,
    batched: bool,
    warnings: list[str],
) -> OpProfile:
    if kernel.split_k > 1:
        return _estimate_split_k_gemm(
            problem=problem, footprint=footprint, kernel=kernel, hardware=hardware,
            batched=batched, warnings=warnings,
        )
    clock_hz = _clock_hz(hardware, warnings)
    grid = _grid_accounting(problem, kernel)
    occupancy = _occupancy(problem, kernel, grid, hardware, warnings)
    traffic = _traffic_accounting(
        problem, kernel, grid, occupancy, hardware, warnings
    )
    traffic = _split_k_hbm_read(problem, kernel, traffic, hardware)
    timeline = _timeline(problem, kernel, grid, traffic, occupancy, hardware, clock_hz, warnings)
    if kernel.architecture == "sm90" and "sram" in hardware.memory_levels:
        from .sm90_gemm import instruction_counts
        traffic = replace(traffic, memory_access=replace(traffic.memory_access,
            sram_write_bytes=(traffic.memory_access.sram_write_bytes or 0)
                             + instruction_counts(problem, kernel).epilogue_write_bytes))
    fixed_overhead_cycles = float(hardware.compute.device_fixed_overhead_cycles or 0)
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    bottlenecks = _classify_bottlenecks(
        problem,
        grid,
        traffic,
        occupancy,
        timeline,
        hardware,
        clock_hz,
        fixed_overhead_cycles,
    )

    latency_s = total_device_cycles / clock_hz
    flops_per_cycle = (
        grid.useful_flops / total_device_cycles if total_device_cycles else 0.0
    )
    tflops_per_s = (grid.useful_flops / latency_s / 1.0e12) if latency_s else 0.0
    energy_breakdown = estimate_energy(
        flops=grid.useful_flops,
        memory_access=traffic.memory_access,
        engine=EngineKind.TENSOR,
        dtype=problem.input_dtype,
        hardware=hardware,
        latency_s=latency_s,
    )

    diagnostics = _diagnostics(
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        occupancy=occupancy,
        timeline=timeline,
        bottlenecks=bottlenecks,
        warnings=tuple(warnings),
        clock_hz=clock_hz,
        latency_s=latency_s,
        flops_per_cycle=flops_per_cycle,
        tflops_per_s=tflops_per_s,
        hardware=hardware,
        fixed_overhead_cycles=fixed_overhead_cycles,
    )
    profile = OpProfile(
        latency_s=latency_s,
        energy_j=energy_breakdown.total_j,
        flops=grid.useful_flops,
        engine=EngineKind.TENSOR,
        footprint=footprint,
        memory_access=traffic.memory_access,
        energy_breakdown=energy_breakdown,
        implementation=_implementation_name(batched),
        diagnostics=diagnostics,
    )
    return apply_calibrated_energy_model(profile, hardware)

def _sum_memory_access(accesses: list[MemoryAccess]) -> MemoryAccess:
    return MemoryAccess(**{
        f.name: (sum(getattr(a, f.name) or 0 for a in accesses)
                 if any(getattr(a, f.name) is not None for a in accesses) else None)
        for f in fields(MemoryAccess)
    })

def _estimate_split_k_gemm(*, problem, footprint, kernel, hardware, batched, warnings,
                           _selection_terms: bool = False) -> OpProfile:
    s, scheme = normalize_split_k(kernel.split_k, kernel.reduction_scheme)
    lengths = split_k_lengths(problem.k, s, kernel.cta_k, scheme)
    counts = split_k_counts(problem.batch, problem.m, problem.n, problem.k,
                            s, scheme, int(dtype_nbytes(problem.output_dtype)))
    plain = replace(kernel, split_k=1, reduction_scheme="none", split_k_slices=1)
    partial_dtype = DType.FP32 if scheme == "compute_type" else problem.output_dtype
    reduction_dtype = partial_dtype
    clock_hz = _clock_hz(hardware, warnings)
    slice_traffic = []
    slice_grids = []
    slice_footprints = []
    priced_lengths = {}
    for length in lengths:
        if not length:
            slice_footprints.append(0)
            continue
        if length not in priced_lengths:
            marker = len(warnings)
            sub = replace(problem, k=length, output_dtype=partial_dtype,
                          epilogue_reads_c=False, beta_zero=True)
            g = _grid_accounting(sub, plain)
            occ = _occupancy(sub, plain, g, hardware, warnings)
            t = _traffic_accounting(sub, plain, g, occ, hardware, warnings)
            priced_lengths[length] = g, t, tuple(warnings[marker:])
        else:
            g, t, repeated_warnings = priced_lengths[length]
            warnings.extend(repeated_warnings)
        slice_traffic.append(t)
        slice_grids.append(g)
        slice_footprints.append(t.a_dram_unique_bytes + t.b_dram_unique_bytes
                                + t.d_store_transaction_bytes)

    representative = replace(problem, k=max(lengths), output_dtype=partial_dtype,
                             epilogue_reads_c=False, beta_zero=True)
    base_grid = _grid_accounting(representative, plain)
    useful = float(2 * problem.batch * problem.m * problem.n * problem.k)
    issued = sum(g.issued_flops for g in slice_grids)
    grid = replace(base_grid, cta_count=base_grid.cta_count * s,
                   useful_flops=useful, issued_flops=issued,
                   tile_efficiency=useful / issued if issued else 0.0)
    occupancy = _occupancy(representative, plain, grid, hardware, warnings)
    summable = (
        "a_logical_bytes", "b_logical_bytes", "a_l2_requested_bytes", "b_l2_requested_bytes",
        "a_dram_unique_bytes", "b_dram_unique_bytes", "a_dram_read_bytes", "b_dram_read_bytes",
        "smem_read_bytes", "smem_write_bytes", "dram_capacity_miss_bytes",
    )
    traffic = replace(slice_traffic[0], **{
        name: sum(getattr(t, name) for t in slice_traffic) for name in summable
    })

    out_problem = replace(problem, epilogue_reads_c=False)
    out_grid = _grid_accounting(out_problem, plain)
    out_occ = _occupancy(out_problem, plain, out_grid, hardware, warnings)
    out_traffic = _traffic_accounting(out_problem, plain, out_grid, out_occ, hardware, warnings)
    d_tx = out_traffic.d_store_transaction_bytes
    partial_tx = slice_traffic[0].d_store_transaction_bytes
    capacity = _effective_l2_capacity(hardware)

    if scheme == "inplace":
        d_misses = [capacity is None or max(slice_footprints[i - 1:i + 1]) > capacity
                    for i in range(1, s)]
        miss_reads = sum(d_tx for miss in d_misses if miss)
        gemm_reads = miss_reads
        gemm_writes = d_tx + miss_reads
        l2_extra_read = (s - 1) * d_tx
        l2_store = s * d_tx
        reduction_read = reduction_write = reduction_l2_read = reduction_l2_write = 0
    else:
        gemm_reads = 0
        gemm_writes = s * partial_tx
        l2_extra_read = 0
        l2_store = s * partial_tx

        reduction_read = sum(partial_tx for i in range(s)
                             if capacity is None or
                             sum(slice_footprints[i + 1:]) + s * partial_tx + d_tx > capacity)
        reduction_write = d_tx
        reduction_l2_read, reduction_l2_write = s * partial_tx, d_tx
    c_tx = d_tx if problem.epilogue_reads_c else 0
    operand_read = traffic.a_dram_read_bytes + traffic.b_dram_read_bytes
    operand_l2 = traffic.a_l2_requested_bytes + traffic.b_l2_requested_bytes
    has_l2 = "l2" in hardware.memory_levels
    gemm_access = MemoryAccess(
        hbm_read_bytes=operand_read + gemm_reads + (c_tx if scheme == "inplace" else 0),
        hbm_write_bytes=gemm_writes,
        l2_read_bytes=operand_l2 + l2_extra_read + (c_tx if scheme == "inplace" else 0) if has_l2 else None,
        l2_write_bytes=l2_store if has_l2 else None,
        sram_read_bytes=sum(t.memory_access.sram_read_bytes or 0 for t in slice_traffic)
                        if "sram" in hardware.memory_levels else None,
        sram_write_bytes=(sum(t.memory_access.sram_write_bytes or 0 for t in slice_traffic)
                          if kernel.architecture == "sm90" else 0)
                         if "sram" in hardware.memory_levels else None,
    )
    if kernel.architecture == "sm90" and "sram" in hardware.memory_levels:
        from .sm90_gemm import instruction_counts
        gemm_access = replace(gemm_access, sram_write_bytes=(gemm_access.sram_write_bytes or 0)
            + instruction_counts(problem, kernel).epilogue_write_bytes)
    traffic = replace(traffic, memory_access=gemm_access,
        c_read_transaction_bytes=l2_extra_read + (c_tx if scheme == "inplace" else 0),
        d_store_transaction_bytes=l2_store,
        l2_requested_bytes=(gemm_access.l2_read_bytes or 0) + (gemm_access.l2_write_bytes or 0),
        dram_total_bytes=gemm_access.hbm_read_bytes + gemm_access.hbm_write_bytes,
        dram_unique_bytes=operand_read + gemm_writes)
    epilogue_hbm = gemm_reads + gemm_writes + (c_tx if scheme == "inplace" else 0)
    epilogue_delivered = l2_extra_read + l2_store + (c_tx if scheme == "inplace" else 0)
    epilogue_memory = (max(0, epilogue_delivered - epilogue_hbm) if has_l2 else 0,
                       epilogue_hbm)
    timing_traffic, timing_slices = traffic, slice_traffic

    if not _selection_terms:
        timeline = _timeline_uncached(representative, plain, grid, timing_traffic, occupancy, hardware,
                                      clock_hz, warnings, epilogue_memory=epilogue_memory)
    fixed = float(hardware.compute.device_fixed_overhead_cycles or 0)
    gemm_cycles, wave_groups = _split_k_wave_cycles(
        problem=representative, kernel=plain, lengths=lengths, grid=grid,
        traffic=timing_traffic, slice_traffic=timing_slices, occupancy=occupancy,
        hardware=hardware, clock_hz=clock_hz, warnings=warnings,
        epilogue_memory=epilogue_memory,
    )
    ordered_epilogue_bound = 0.0
    if scheme == "inplace":
        mainloop_cycles = 0.0
        epilogue_cycles = 0.0
        nonempty_traffic = iter(timing_slices)

        slice_inputs: dict = {}
        for index, length in enumerate(lengths):
            t = next(nonempty_traffic) if length else None
            case = (length, index == 0, bool(d_misses[index - 1]) if index else None)
            inputs = slice_inputs.get(case)
            if inputs is None:
                if t is None:
                    t = _scale_split_k_traffic(traffic, 0)
                marker = len(warnings)
                sub = replace(representative, k=length)
                one_grid = _grid_accounting(sub, plain)
                tile_count = one_grid.cta_count
                one_grid = replace(one_grid, cta_count=1,
                                   useful_flops=one_grid.useful_flops / tile_count,
                                   issued_flops=one_grid.issued_flops / tile_count)
                one_traffic = replace(_scale_split_k_traffic(t, 1 / tile_count),
                                      d_store_transaction_bytes=d_tx / tile_count)

                epi_hbm = d_tx + c_tx if index == 0 else 2 * d_tx * d_misses[index - 1]
                epi_delivered = d_tx + c_tx if index == 0 else 2 * d_tx
                epi_l2 = max(0, epi_delivered - epi_hbm) if has_l2 else 0
                one_epilogue = (epi_l2 / tile_count, epi_hbm / tile_count)
                one_occ = _occupancy(sub, plain, one_grid, hardware, warnings)
                inputs = slice_inputs[case] = (sub, one_grid, one_traffic, one_occ, one_epilogue,
                                               tuple(warnings[marker:]))
            else:
                warnings.extend(inputs[5])
            sub, one_grid, one_traffic, one_occ, one_epilogue, _ = inputs
            one = _timeline(sub, plain, one_grid, one_traffic, one_occ, hardware,
                            clock_hz, warnings, epilogue_memory=one_epilogue)
            if length:
                mainloop_cycles = max(mainloop_cycles, one.kernel_cycles - one.epilogue_cycles)
            epilogue_cycles += one.epilogue_cycles
        ordered_epilogue_bound = mainloop_cycles + epilogue_cycles
        gemm_cycles = max(gemm_cycles, ordered_epilogue_bound)

    reduction_access = MemoryAccess(
        hbm_read_bytes=reduction_read + (c_tx if counts.extra_launches else 0),
        hbm_write_bytes=reduction_write,
        l2_read_bytes=reduction_l2_read + (c_tx if counts.extra_launches else 0) if has_l2 else None,
        l2_write_bytes=reduction_l2_write if has_l2 else None,
    )
    reduction = _make_profile(
        hardware=hardware, dtype=reduction_dtype, flops=counts.reduction_flops,
        memory_access=reduction_access, engine=EngineKind.VECTOR,
        footprint=replace(footprint, workspace_bytes=counts.workspace_bytes),
        implementation="roofline.elementwise.split_k_reduction",
    )
    reduction_s = reduction.latency_s + counts.extra_launches * fixed / clock_hz
    latency_s = (gemm_cycles + fixed) / clock_hz + reduction_s
    if _selection_terms:
        return latency_s, issued, useful, useful + counts.reduction_flops
    access = _sum_memory_access([gemm_access, reduction_access])
    timeline = replace(timeline, kernel_cycles=gemm_cycles)
    energy = estimate_energy(flops=useful, memory_access=access, engine=EngineKind.TENSOR,
                             dtype=problem.input_dtype, hardware=hardware, latency_s=latency_s)
    vector_energy = estimate_energy(flops=counts.reduction_flops, engine=EngineKind.VECTOR,
                                    dtype=reduction_dtype, hardware=hardware)
    energy = replace(energy, compute_j=energy.compute_j + vector_energy.compute_j)
    diagnostics = _diagnostics(
        problem=problem, kernel=kernel, grid=grid, traffic=traffic, occupancy=occupancy,
        timeline=timeline,
        bottlenecks=_classify_bottlenecks(representative, grid, traffic, occupancy,
                                         timeline, hardware, clock_hz, fixed),
        warnings=tuple(dict.fromkeys(warnings)), clock_hz=clock_hz, latency_s=latency_s,
        flops_per_cycle=useful / (latency_s * clock_hz),
        tflops_per_s=useful / latency_s / 1e12, hardware=hardware, fixed_overhead_cycles=fixed,
    )
    diagnostics["split_k"] = {
        "slices": s, "reduction_scheme": scheme, "k_lengths": lengths,
        "workspace_bytes": counts.workspace_bytes, "partial_bytes": counts.partial_bytes,
        "tensor_flops": useful, "issued_tensor_flops": issued,
        "reduction_flops": counts.reduction_flops,
        "gemm_hbm_bytes": gemm_access.hbm_read_bytes + gemm_access.hbm_write_bytes,
        "reduction_hbm_bytes": reduction_access.hbm_read_bytes + reduction_access.hbm_write_bytes,
        "partial_write_bytes": counts.partial_write_bytes,
        "partial_read_bytes": counts.partial_read_bytes,
        "output_read_bytes": counts.output_read_bytes,
        "output_write_bytes": counts.output_write_bytes,
        "operand_smem_read_bytes": traffic.smem_read_bytes,
        "reduction_smem_bytes": 0, "kernel_launches": 1 + counts.extra_launches,
        "gemm_latency_s": (gemm_cycles + fixed) / clock_hz,
        "reduction_latency_s": reduction_s,
        "reduction_compute_dtype": reduction_dtype.value,
        "semaphore_control_traffic": "excluded_runtime_dependent",
        "wave_groups": wave_groups,
        "ordered_epilogue_bound_cycles": ordered_epilogue_bound,
    }
    diagnostics["kernel_trace"] = (
        {"name": "gemm.split_k", "kind": "gemm", "slices": s,
         "latency_s": (gemm_cycles + fixed) / clock_hz + (reduction_s if not counts.extra_launches else 0)},
    ) + (({"name": "gemm.split_k.reduce", "kind": "pointwise",
           "flops": counts.reduction_flops, "latency_s": reduction_s},) if counts.extra_launches else ())
    diagnostics["predicted_elapsed_cycles"] = latency_s * clock_hz
    diagnostics["total_device_cycles"] = latency_s * clock_hz
    diagnostics["predicted_latency_s"] = latency_s
    diagnostics["modeled_device_cycles"] = gemm_cycles + reduction.latency_s * clock_hz
    diagnostics["device_fixed_overhead_cycles"] = fixed * (1 + counts.extra_launches)
    diagnostics["device_fixed_overhead_s"] = fixed * (1 + counts.extra_launches) / clock_hz
    diagnostics["device_fixed_overhead_fraction"] = (
        fixed * (1 + counts.extra_launches) / (latency_s * clock_hz))
    diagnostics["transaction_bytes"].update({
        "dram_total": access.hbm_read_bytes + access.hbm_write_bytes,
        "l2_requested": (access.l2_read_bytes or 0) + (access.l2_write_bytes or 0),
        "partial_write": counts.partial_write_bytes, "partial_read": counts.partial_read_bytes,
        "output_read": counts.output_read_bytes, "output_write": counts.output_write_bytes,
    })
    diagnostics["active_cycles"]["dram"] = (access.hbm_read_bytes + access.hbm_write_bytes) / (
        hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz)
    if has_l2:
        diagnostics["active_cycles"]["l2"] = (
            (access.l2_read_bytes or 0) + (access.l2_write_bytes or 0)) / (
                hardware.memory_levels["l2"].bandwidth_bytes_per_s / clock_hz)
    profile = OpProfile(latency_s=latency_s, energy_j=energy.total_j,
        flops=useful + counts.reduction_flops, engine=EngineKind.MIXED,
        footprint=replace(footprint, workspace_bytes=counts.workspace_bytes),
        memory_access=access, energy_breakdown=energy,
        implementation=_implementation_name(batched), diagnostics=diagnostics)
    return apply_calibrated_energy_model(profile, hardware)

def _scale_split_k_traffic(traffic: TrafficAccounting, scale: float) -> TrafficAccounting:
    names = ("a_l2_requested_bytes", "b_l2_requested_bytes", "a_dram_read_bytes",
             "b_dram_read_bytes", "c_read_transaction_bytes", "d_store_transaction_bytes",
             "l2_requested_bytes", "dram_total_bytes", "smem_read_bytes", "smem_write_bytes")
    return replace(traffic, **{name: getattr(traffic, name) * scale for name in names})

def _split_k_wave_cycles(*, problem, kernel, lengths, grid, traffic, slice_traffic,
                         occupancy, hardware, clock_hz, warnings, epilogue_memory=None):
    tiles = grid.cta_count // len(lengths)
    cpw = occupancy.ctas_per_wave
    by_length = dict(zip((x for x in lengths if x), slice_traffic))
    start = 0
    cycles = 0.0
    groups = []
    while start < grid.cta_count:
        index = start // tiles
        length = lengths[index]
        end_index = index + 1
        while end_index < len(lengths) and lengths[end_index] == length:
            end_index += 1
        complete_waves = (end_index * tiles - start) // cpw
        ct = (complete_waves * cpw if complete_waves else
              min(cpw, grid.cta_count - start))
        covered_end = _ceil_div(start + ct, tiles)
        length = max(lengths[index:covered_end])
        sub = replace(problem, k=length)
        g = replace(_grid_accounting(sub, kernel), cta_count=ct)
        occ = _occupancy(sub, kernel, g, hardware, warnings)

        parts = []
        for j in range(index, min(len(lengths), _ceil_div(start + ct, tiles))):
            overlap = min(start + ct, (j + 1) * tiles) - max(start, j * tiles)
            if lengths[j] and overlap:
                parts.append((by_length[lengths[j]], overlap / tiles))
        names = ("a_l2_requested_bytes", "b_l2_requested_bytes", "a_dram_read_bytes",
                 "b_dram_read_bytes", "smem_read_bytes", "smem_write_bytes",
                 "l2_requested_bytes", "dram_total_bytes")
        changes = {name: sum(getattr(t, name) * scale for t, scale in parts) for name in names}

        changes["d_store_transaction_bytes"] = traffic.d_store_transaction_bytes * ct / grid.cta_count
        changes["c_read_transaction_bytes"] = traffic.c_read_transaction_bytes * ct / grid.cta_count
        local = replace(traffic, **changes)
        local_epilogue = (tuple(value * ct / grid.cta_count for value in epilogue_memory)
                          if epilogue_memory is not None else None)

        t = _timeline_uncached(sub, kernel, g, local, occ, hardware, clock_hz, warnings,
                               epilogue_memory=local_epilogue)
        if not length:
            group_cycles = t.epilogue_cycles
        else:
            group_cycles = t.kernel_cycles
        cycles += group_cycles
        groups.append({"cta_count": ct, "waves": occ.wave_count,
                       "max_k": length, "cycles": group_cycles})
        start += ct
    return cycles, tuple(groups)

def evaluate_gemm_template_candidates(
    op: LocalOp,
    hardware: HardwareSpec,
    *,
    shortlist_size: int | None = None,
) -> tuple[GemmCandidateEvaluation, ...]:
    if op.kind not in (OpKind.GEMM, OpKind.BATCHED_GEMM):
        raise ValueError("GEMM template candidates require a GEMM op")
    problem = _problem_spec(op, batched=op.kind == OpKind.BATCHED_GEMM)
    engine = _matmul_engine(problem.input_dtype, hardware)
    if engine != EngineKind.TENSOR or hardware.kind != "gpu":
        return ()
    backend = _selection_backend(op.attrs)
    size = (
        _positive_int_value(shortlist_size, "shortlist_size")
        if shortlist_size is not None
        else _selection_shortlist_size(op.attrs)
    )
    return _evaluate_gemm_template_candidates(
        problem=problem,
        footprint=footprint_from_tensors(op),
        hardware=hardware,
        batched=op.kind == OpKind.BATCHED_GEMM,
        objective=_selection_objective(op.attrs),
        backend=backend,
        shortlist_size=size,
    )

def select_gemm_template_candidate(
    candidates: tuple[GemmCandidateEvaluation, ...],
    *,
    objective: str = "latency",
) -> GemmCandidateEvaluation:
    objective = _validate_selection_objective(objective)
    if not candidates:
        if objective == "latency":
            raise ValueError("no legal GEMM catalog template")
        raise ValueError("No GEMM template candidates to select from")
    if _settings.get("OPMODEL_GEMM_SELECTION_RULE") == "useful_fraction":
        if objective != "latency":
            raise ValueError("useful_fraction requires latency objective; no prices")

        def useful_fraction_key(candidate):
            d = candidate.profile.diagnostics
            split = d.get("split_k", {})
            issued = split.get("issued_tensor_flops", d.get("issued_flops", candidate.profile.flops))
            useful = split.get("tensor_flops", d.get("useful_flops", candidate.profile.flops))
            return candidate.profile.latency_s * issued / useful
        return min(candidates, key=useful_fraction_key)
    return min(candidates, key=lambda candidate: _candidate_objective_key(candidate, objective))

def _estimate_selected_gemm_kernel(
    *,
    op: LocalOp,
    problem: GemmProblemSpec,
    footprint,
    hardware: HardwareSpec,
    batched: bool,
) -> OpProfile:
    objective = _selection_objective(op.attrs)
    backend = _selection_backend(op.attrs)
    shortlist_size = _selection_shortlist_size(op.attrs)
    template, forced_metadata = _resolve_forced_template(problem, op.attrs.get("gemm_role", "forward"), hardware=hardware)
    if template is not None:
        profile = _estimate_gemm_with_kernel(
            problem=problem, footprint=footprint, kernel=_kernel_from_template(template),
            hardware=hardware, batched=batched, warnings=[])
        diagnostics = dict(profile.diagnostics)
        diagnostics["gemm_forced"] = forced_metadata
        diagnostics["gemm_selection"] = {"enabled": True, "objective": objective, "backend": backend,
                                         "selected_template": _template_diagnostics(template)}
        return replace(profile, diagnostics=diagnostics)
    candidates = _evaluate_gemm_template_candidates(
        problem=problem,
        footprint=footprint,
        hardware=hardware,
        batched=batched,
        objective=objective,
        backend=backend,
        shortlist_size=shortlist_size,
    )
    if not candidates:
        if objective == "latency":
            raise ValueError("no legal GEMM catalog template")
        warnings = ["gemm_selection_no_legal_catalog_template"]
        kernel = _kernel_spec(op.attrs, hardware, warnings)
        profile = _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=batched,
            warnings=warnings,
        )
        result = _with_gemm_selection_diagnostics(
            profile,
            {
                "enabled": False,
                "reason": "no_legal_catalog_template",
                "objective": objective,
                "backend": backend,
                "catalog_size": len(_kernel_template_catalog(problem.input_dtype, hardware)),
                "legal_candidates": 0,
                "shortlist_size": 0,
            },
        )
        if forced_metadata is not None:
            result = replace(result, diagnostics={**result.diagnostics, "gemm_forced": forced_metadata})
        return result

    selected = select_gemm_template_candidate(candidates, objective=objective)
    result = _with_gemm_selection_diagnostics(
        selected.profile,
        _gemm_selection_metadata(
            selected=selected,
            candidates=candidates,
            objective=objective,
            backend=backend,
            catalog_size=len(_kernel_template_catalog(problem.input_dtype, hardware)),
            legal_count=len(_legal_kernel_templates(problem, hardware)),
        ),
    )
    if forced_metadata is not None:
        result = replace(result, diagnostics={**result.diagnostics, "gemm_forced": forced_metadata})
    return result

def _evaluate_gemm_template_candidates(
    *,
    problem: GemmProblemSpec,
    footprint,
    hardware: HardwareSpec,
    batched: bool,
    objective: str,
    backend: str,
    shortlist_size: int,
) -> tuple[GemmCandidateEvaluation, ...]:
    if backend != "extended_roofline":
        raise ValueError(f"Unsupported GEMM selection backend: {backend}")
    legal = _legal_kernel_templates(problem, hardware)
    ranked = (legal if objective == "latency" else sorted(
        legal,
        key=lambda template: _cheap_template_score(template, problem, hardware, objective),
    ))
    evaluations: list[GemmCandidateEvaluation] = []
    for rank, template in enumerate(ranked if objective == "latency" else ranked[:shortlist_size], start=1):
        kernel = _kernel_from_template(template)
        profile = _estimate_gemm_with_kernel(
            problem=problem,
            footprint=footprint,
            kernel=kernel,
            hardware=hardware,
            batched=batched,
            warnings=[],
        )
        evaluations.append(
            GemmCandidateEvaluation(
                template=template,
                kernel=kernel,
                profile=profile,
                selection_energy_j=_selection_energy_j(profile, problem, hardware),
                cheap_rank=rank,
                cheap_score=0.0 if objective == "latency" else _cheap_template_score(template, problem, hardware, objective),
            )
        )
    return tuple(evaluations)

def _should_select_gemm_kernel(attrs: Mapping[str, Any], hardware: HardwareSpec) -> bool:
    if hardware.kind != "gpu":
        return False
    return not any(name in attrs for name in _EXPLICIT_KERNEL_ATTRS)

def _selection_objective(attrs: Mapping[str, Any]) -> str:
    return _validate_selection_objective(str(attrs.get("gemm_selection_objective", "latency")))

def _validate_selection_objective(value: str) -> str:
    if value not in _GEMM_SELECTION_OBJECTIVES:
        raise ValueError(
            "gemm_selection_objective must be one of: "
            + ", ".join(sorted(_GEMM_SELECTION_OBJECTIVES))
        )
    return value

def _selection_backend(attrs: Mapping[str, Any]) -> str:
    backend = str(attrs.get("gemm_selection_backend", "extended_roofline"))
    if backend not in _GEMM_SELECTION_BACKENDS:
        raise ValueError(
            "gemm_selection_backend must be one of: "
            + ", ".join(sorted(_GEMM_SELECTION_BACKENDS))
        )
    return backend

def _selection_shortlist_size(attrs: Mapping[str, Any]) -> int:
    return _positive_int_value(
        attrs.get("gemm_selection_shortlist_size", _DEFAULT_GEMM_SELECTION_SHORTLIST_SIZE),
        "gemm_selection_shortlist_size",
    )

def _positive_int_value(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result

def _kernel_template_catalog(
    dtype: DType, hardware: HardwareSpec
) -> tuple[GemmKernelTemplate, ...]:
    if hardware.kind != "gpu":
        return ()

    catalog = _settings.get("OPMODEL_GEMM_CATALOG")
    if hardware.architecture == "sm90":
        from . import sm90_gemm
        return sm90_gemm.catalog(dtype, hardware, catalog)
    if catalog not in ("", "default"):
        templates = load_gemm_catalog(catalog)

        held = _CATALOG_BY_DTYPE.get((id(templates), dtype))
        if held is None or held[0] is not templates:
            held = _CATALOG_BY_DTYPE[(id(templates), dtype)] = (
                templates, tuple(template for template in templates if dtype in template.dtypes))
        return held[1]

    min_tile = _settings.get("OPMODEL_MIN_GEMM_TILE")
    return tuple(
        template
        for template in load_gemm_catalog("default")
        if dtype in template.dtypes
        and min(template.cta_m, template.cta_n) >= min_tile
    )

_CATALOG_BY_DTYPE: dict = {}
_LEGAL_TEMPLATES_MEMO = SettingsMemo(4096)
_TRAFFIC_MEMO = SettingsMemo(40_000)

def _legal_kernel_templates(
    problem: GemmProblemSpec, hardware: HardwareSpec
) -> tuple[GemmKernelTemplate, ...]:
    catalog = _kernel_template_catalog(problem.input_dtype, hardware)

    def legal():
        return tuple(template for template in catalog if _template_is_legal(template, problem, hardware))

    if not any(held[1] is catalog for held in _CATALOG_BY_DTYPE.values()):
        return legal()
    held, templates = _LEGAL_TEMPLATES_MEMO.get_or_compute(
        (id(catalog), typed(problem), hardware_token(hardware)), lambda: (catalog, legal()))
    return templates if held is catalog else legal()

def _allocated_registers_per_cta(
    registers_per_thread: int, threads_per_cta: int, hardware: HardwareSpec
) -> int:
    unit = hardware.compute.register_allocation_per_thread
    return _ceil_div(registers_per_thread, unit) * unit * 32 * _ceil_div(threads_per_cta, 32)

def _template_is_legal(
    template: GemmKernelTemplate,
    problem: GemmProblemSpec,
    hardware: HardwareSpec,
) -> bool:
    if template.split_k > 1 and _settings.get("OPMODEL_GEMM_CATALOG") == "cublaslt_sm80_bf16_splitk":
        workspace = split_k_counts(problem.batch, problem.m, problem.n, problem.k,
                                   template.split_k, template.reduction_scheme,
                                   int(dtype_nbytes(problem.output_dtype))).workspace_bytes
        if workspace > _LTLIST_SPLIT_K_WORKSPACE_BYTES:
            return False
    if problem.input_dtype not in hardware.compute.tensor_flops_per_s:
        return False
    if template.architecture == "sm90":
        if template.pipeline_stages < (2 if _settings.get("OPMODEL_SM90_MIN_TWO_STAGES") else 1):
            return False

        if _settings.get("OPMODEL_SM90_ACCUMULATOR_LIMIT"):
            if template.cta_m * template.cta_n > 256 * 255:
                return False
        return True
    if not _divisible(template.cta_m, template.mma_m):
        return False
    if not _divisible(template.cta_n, template.mma_n):
        return False
    if not _divisible(template.cta_k, template.mma_k):
        return False
    if not _divisible(template.warp_m, template.mma_m):
        return False
    if not _divisible(template.warp_n, template.mma_n):
        return False
    if not _divisible(template.warp_k, template.mma_k):
        return False
    if not _divisible(template.cta_m, template.warp_m):
        return False
    if not _divisible(template.cta_n, template.warp_n):
        return False
    if not _divisible(template.cta_k, template.warp_k):
        return False
    spatial_warps = (template.cta_m // template.warp_m) * (
        template.cta_n // template.warp_n
    )
    if spatial_warps != template.warps_per_cta:
        return False
    threads_per_cta = template.warps_per_cta * 32
    if hardware.compute.max_warps_per_sm is not None:
        if template.warps_per_cta > hardware.compute.max_warps_per_sm:
            return False
    if hardware.compute.registers_per_sm is not None:
        registers_per_cta = _allocated_registers_per_cta(template.registers_per_thread, threads_per_cta, hardware)
        if registers_per_cta > hardware.compute.registers_per_sm:
            return False
    shared_bytes = _shared_memory_bytes_per_cta(problem, _kernel_from_template(template))
    for capacity in (hardware.compute.shared_memory_bytes_per_sm,
                     hardware.compute.shared_memory_bytes_per_block):
        if capacity is not None and shared_bytes > capacity:
            return False
    if hardware.compute.max_ctas_per_sm is not None and hardware.compute.max_ctas_per_sm <= 0:
        return False
    return True

def _cheap_template_score(
    template: GemmKernelTemplate,
    problem: GemmProblemSpec,
    hardware: HardwareSpec,
    objective: str,
) -> float:
    kernel = _kernel_from_template(template)
    grid = _grid_accounting(problem, kernel)
    occupancy = _occupancy(problem, kernel, grid, hardware, [])
    problem_aspect = problem.m / max(problem.n, 1)
    tile_aspect = template.cta_m / max(template.cta_n, 1)
    aspect_penalty = abs(math.log2(problem_aspect / tile_aspect))
    wave_penalty = 1.0 - occupancy.tail_efficiency
    underfill_penalty = max(0, occupancy.ctas_per_wave - grid.cta_count) / max(
        occupancy.ctas_per_wave, 1
    )
    k_penalty = 0.2 if problem.k < template.cta_k else 0.0
    tile_efficiency_penalty = 1.0 - grid.tile_efficiency
    objective_penalty = 0.0
    if objective == "energy":
        smem_bytes = _shared_memory_bytes_per_cta(problem, kernel)
        objective_penalty = smem_bytes / max(1, template.pipeline_stages) / 131072.0
    return (
        4.0 * tile_efficiency_penalty
        + 1.2 * underfill_penalty
        + 0.7 * wave_penalty
        + 0.35 * aspect_penalty
        + k_penalty
        + 0.2 * objective_penalty
    )

def _kernel_from_template(template: GemmKernelTemplate) -> GemmKernelSpec:
    return GemmKernelSpec(
        cta_m=template.cta_m,
        cta_n=template.cta_n,
        cta_k=template.cta_k,
        warp_m=template.warp_m,
        warp_n=template.warp_n,
        warp_k=template.warp_k,
        num_warp_tile_k=template.num_warp_tile_k,
        mma_m=template.mma_m,
        mma_n=template.mma_n,
        mma_k=template.mma_k,
        pipeline_stages=template.pipeline_stages,
        warps_per_cta=template.warps_per_cta,
        threads_per_cta=template.warps_per_cta * 32,
        registers_per_thread=template.registers_per_thread,
        max_concurrent_ctas_per_sm=1 if template.architecture == "sm90" else None,
        slice_k=False,
        split_k=template.split_k,
        reduction_scheme=template.reduction_scheme,
        architecture=template.architecture,
    )

def _selection_energy_j(
    profile: OpProfile, problem: GemmProblemSpec, hardware: HardwareSpec
) -> float:
    issued_flops = float(profile.diagnostics.get("issued_flops", profile.flops))
    tensor_flops = profile.diagnostics.get("split_k", {}).get("tensor_flops", profile.flops)
    extra_flops = max(0.0, issued_flops - tensor_flops)
    return profile.energy_j + extra_flops * hardware.compute.tensor_energy_j_per_flop.get(
        problem.input_dtype, 0.0
    )

def _candidate_objective_key(
    candidate: GemmCandidateEvaluation, objective: str
) -> tuple[float, ...]:
    tile_efficiency = float(candidate.profile.diagnostics.get("tile_efficiency", 0.0))
    if objective == "energy":
        return (
            candidate.selection_energy_j,
            candidate.profile.latency_s,
            -tile_efficiency,
            candidate.cheap_rank,
        )

    return (candidate.profile.latency_s,)

def _with_gemm_selection_diagnostics(
    profile: OpProfile, metadata: Mapping[str, Any]
) -> OpProfile:
    diagnostics = dict(profile.diagnostics)
    diagnostics["gemm_selection"] = dict(metadata)
    return replace(profile, diagnostics=diagnostics)

def _gemm_selection_metadata(
    *,
    selected: GemmCandidateEvaluation,
    candidates: tuple[GemmCandidateEvaluation, ...],
    objective: str,
    backend: str,
    catalog_size: int,
    legal_count: int,
) -> dict[str, Any]:
    metadata = {
        "enabled": True,
        "objective": objective,
        "backend": backend,
        "catalog_size": catalog_size,
        "legal_candidates": legal_count,
        "shortlist_size": len(candidates),
        "selected_template": _template_diagnostics(selected.template),
        "selected_rank": selected.cheap_rank,
        "selected_latency_s": selected.profile.latency_s,
        "selected_energy_j": selected.profile.energy_j,
    }
    if objective == "energy":
        metadata.update(selection_energy_j=selected.selection_energy_j, cheap_score=selected.cheap_score)
    return metadata

def _template_diagnostics(template: GemmKernelTemplate) -> dict[str, Any]:
    return {
        **({"architecture": "sm90", "model_status": "H100 model output, not yet scored",
            "stage_policy": "CUTLASS_v3.5.1_StageCountAutoCarveout",
            "occupancy_rule": "approved_1_CTA_per_SM",
            "split_candidate_status": "observed-only for algo 66"}
           if template.architecture == "sm90" else {}),
        **({"split_k": template.split_k,
            "reduction_scheme": normalize_split_k(template.split_k, template.reduction_scheme)[1]}
           if template.split_k > 1 else {}),
        "name": template.name,
        "source": template.source,
        "cta_tile": {"m": template.cta_m, "n": template.cta_n, "k": template.cta_k},
        "warp_tile": {
            "m": template.warp_m,
            "n": template.warp_n,
            "k": template.warp_k,
        },
        "num_warp_tile_k": template.num_warp_tile_k,
        "mma_shape": {"m": template.mma_m, "n": template.mma_n, "k": template.mma_k},
        "pipeline_stages": template.pipeline_stages,
        "warps_per_cta": template.warps_per_cta,
        "threads_per_cta": template.warps_per_cta * 32,
        "registers_per_thread": template.registers_per_thread,
        "dtypes": tuple(dtype.value for dtype in template.dtypes),
    }

def _divisible(numerator: int, denominator: int) -> bool:
    return denominator > 0 and numerator % denominator == 0

def _problem_spec(op: LocalOp, *, batched: bool) -> GemmProblemSpec:
    if batched:
        batch, m, n, k, dtype = parse_batched_gemm(op)
        weights = get_tensors(op, TensorRole.WEIGHT)
        outputs = get_tensors(op, TensorRole.OUTPUT)
        weight_is_batched = bool(weights and len(weights[0].shape) == 3)
    else:
        m, n, k, dtype = parse_gemm(op)
        batch = 1
        outputs = get_tensors(op, TensorRole.OUTPUT)
        weight_is_batched = True
    output_dtype = outputs[0].dtype if outputs else dtype
    weight_bytes = op.attrs.get("weight_bytes")
    if weight_bytes is not None and not float(weight_bytes) > 0:
        raise ValueError("weight_bytes must be positive")
    beta_zero = bool(op.attrs.get("beta_zero", True))
    epilogue_reads_c = bool(op.attrs.get("epilogue_reads_c", not beta_zero))
    return GemmProblemSpec(
        batch=batch,
        m=m,
        n=n,
        k=k,
        input_dtype=dtype,
        output_dtype=output_dtype,
        weight_is_batched=weight_is_batched,
        beta_zero=beta_zero,
        epilogue_reads_c=epilogue_reads_c,
        transpose_a=bool(op.attrs.get("transpose_a", False)),
        transpose_b=bool(op.attrs.get("transpose_b", False)),
        b_bytes=None if weight_bytes is None else float(weight_bytes),
    )

def _kernel_spec(
    attrs: Mapping[str, Any],
    hardware: HardwareSpec,
    warnings: list[str],
) -> GemmKernelSpec:
    default_mma = hardware.compute.fma_dims or (16, 8, 16)
    if hardware.compute.fma_dims is None:
        warnings.append("default_mma_shape_used")
    mma_m = _positive_int_attr(attrs, "mma_m", default_mma[0])
    mma_n = _positive_int_attr(attrs, "mma_n", default_mma[1])
    mma_k = _positive_int_attr(attrs, "mma_k", default_mma[2])
    cta_m = _positive_int_attr(attrs, "cta_tile_m", 128)
    cta_n = _positive_int_attr(attrs, "cta_tile_n", 128)
    cta_k = _positive_int_attr(attrs, "cta_tile_k", max(mma_k, 32))
    warp_m = _positive_int_attr(attrs, "warp_tile_m", min(cta_m, 64))
    warp_n = _positive_int_attr(attrs, "warp_tile_n", min(cta_n, 64))
    warp_k = _positive_int_attr(attrs, "warp_tile_k", cta_k)
    num_warp_tile_k = _positive_int_attr(
        attrs,
        "num_warp_tile_k",
        _positive_int_attr(attrs, "num_warp_tile_K", 1)
        if "num_warp_tile_K" in attrs
        else 1,
    )
    pipeline_stages = _positive_int_attr(attrs, "pipeline_stages", 3)
    warps_per_cta = _positive_int_attr(attrs, "warps_per_cta", 4)
    threads_per_cta = _positive_int_attr(attrs, "threads_per_cta", warps_per_cta * 32)
    registers_per_thread = _positive_int_attr(attrs, "registers_per_thread", 64)
    max_concurrent_ctas_per_sm = _optional_positive_int_attr(
        attrs, "resident_ctas_per_sm"
    )
    if max_concurrent_ctas_per_sm is None:
        max_concurrent_ctas_per_sm = _optional_positive_int_attr(
            attrs, "max_concurrent_block"
        )
    return GemmKernelSpec(
        cta_m=cta_m,
        cta_n=cta_n,
        cta_k=cta_k,
        warp_m=warp_m,
        warp_n=warp_n,
        warp_k=warp_k,
        num_warp_tile_k=num_warp_tile_k,
        mma_m=mma_m,
        mma_n=mma_n,
        mma_k=mma_k,
        pipeline_stages=pipeline_stages,
        warps_per_cta=warps_per_cta,
        threads_per_cta=threads_per_cta,
        registers_per_thread=registers_per_thread,
        max_concurrent_ctas_per_sm=max_concurrent_ctas_per_sm,
        slice_k=bool(attrs.get("slice_k", attrs.get("sliceK", False))),
        store_evict_first=(bool(attrs["store_evict_first"]) if "store_evict_first" in attrs else None),
        split_k_slices=max(1, int(attrs.get("split_k_slices", 1) or 1)),
        split_k_reduction=str(attrs.get("split_k_reduction", "none") or "none"),
        split_k_partial_bytes=int(attrs.get("split_k_partial_bytes", 2) or 2),
        split_k=attrs.get("split_k", 1),
        reduction_scheme=attrs.get("reduction_scheme", "none"),
        **_launch_order_from_attrs(attrs),
    )

def _launch_order_from_attrs(attrs: Mapping[str, Any]) -> dict:
    r = attrs.get("cta_raster")
    if not isinstance(r, Mapping):
        return {}
    rows = str(r.get("rows_dim", "m"))
    other = "n" if rows == "m" else "m"
    x_axis = other if r.get("transpose") else rows
    if r.get("swap"):
        x_axis = "n" if x_axis == "m" else "m"
    y_axis = "n" if x_axis == "m" else "m"
    family = str(r.get("family", "cutlass"))
    if family == "panel":
        return dict(order_sweep_axis=y_axis, order_strip_width=max(1, int(r.get("panel", 1))),
                    order_serpentine=False)
    return dict(order_sweep_axis=x_axis, order_strip_width=1 << max(0, int(r.get("log_tile", 0) or 0)),
                order_serpentine=bool(r.get("serpentine", False)))

def _grid_accounting(problem: GemmProblemSpec, kernel: GemmKernelSpec) -> GridAccounting:
    blocks_m = _ceil_div(problem.m, kernel.cta_m)
    blocks_n = _ceil_div(problem.n, kernel.cta_n)
    k_stages = _ceil_div(problem.k, kernel.cta_k)
    total_k_stages = k_stages
    if kernel.split_k > 1:
        k_stages = _ceil_div(max(split_k_lengths(problem.k, kernel.split_k, kernel.cta_k,
                                               kernel.reduction_scheme)), kernel.cta_k)
    cta_count = problem.batch * blocks_m * blocks_n
    cta_count *= kernel.split_k
    useful_flops = float(2 * problem.batch * problem.m * problem.n * problem.k)
    issued_flops = float(
        2
        * problem.batch
        * blocks_m
        * blocks_n
        * total_k_stages
        * kernel.cta_m
        * kernel.cta_n
        * kernel.cta_k
    )
    if kernel.architecture == "sm90":
        from .sm90_gemm import instruction_counts
        issued_flops = float(instruction_counts(problem, kernel).issued_flops)
    return GridAccounting(
        blocks_m=blocks_m,
        blocks_n=blocks_n,
        k_stages=k_stages,
        cta_count=cta_count,
        useful_flops=useful_flops,
        issued_flops=issued_flops,
        tile_efficiency=useful_flops / issued_flops if issued_flops else 0.0,
    )

def _smem_load_per_cta_elements(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting | None = None,
) -> int:
    k_stages = grid.k_stages if grid is not None else _ceil_div(problem.k, kernel.cta_k)
    warp_m_tiles = _ceil_div(kernel.cta_m, kernel.warp_m)
    warp_n_tiles = _ceil_div(kernel.cta_n, kernel.warp_n)
    warp_k_tiles = _ceil_div(kernel.cta_k, kernel.warp_k)
    return (
        (kernel.warp_m + kernel.warp_n)
        * warp_m_tiles
        * warp_n_tiles
        * k_stages
        * warp_k_tiles
        * kernel.warp_k
    )

def _shared_memory_bytes_per_cta(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
) -> int:
    if _mixed_operands(problem):
        if kernel.architecture == "sm90":
            raise NotImplementedError("weight-only operand bytes are described for the sm80 GEMM model only")
        a_b, b_b = _operand_bytes(problem, dtype_nbytes(problem.input_dtype))
        stage_operand_bytes = _ceil_scalar_bytes(kernel.cta_m * kernel.cta_k * a_b + kernel.cta_k * kernel.cta_n * b_b)
    else:
        stage_operand_bytes = _ceil_scalar_bytes(
            (kernel.cta_m * kernel.cta_k + kernel.cta_k * kernel.cta_n)
            * dtype_nbytes(problem.input_dtype)
        )
    if kernel.architecture == "sm90":
        return (stage_operand_bytes + 16) * kernel.pipeline_stages + 1
    return stage_operand_bytes * kernel.pipeline_stages

def _smem_read_bytes_per_cta(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting | None = None,
) -> int:
    if _mixed_operands(problem):
        a_b, b_b = _operand_bytes(problem, dtype_nbytes(problem.input_dtype))
        k_stages = grid.k_stages if grid is not None else _ceil_div(problem.k, kernel.cta_k)
        common = (_ceil_div(kernel.cta_m, kernel.warp_m) * _ceil_div(kernel.cta_n, kernel.warp_n) * k_stages
                  * _ceil_div(kernel.cta_k, kernel.warp_k) * kernel.warp_k)
        return _ceil_scalar_bytes(kernel.warp_m * common * a_b + kernel.warp_n * common * b_b)
    return _ceil_scalar_bytes(
        _smem_load_per_cta_elements(problem, kernel, grid)
        * dtype_nbytes(problem.input_dtype)
    )

def _count_periodic_indices(
    *, start: int, count: int, period: int, target: int
) -> int:
    if count <= 0:
        return 0
    first_offset = (target - start) % period
    if first_offset >= count:
        return 0
    return 1 + (count - 1 - first_offset) // period

def _cyclic_distinct_panel_counts(
    *, start: int, request_count: int, panel_count: int
) -> tuple[int, int]:
    distinct = min(panel_count, max(0, request_count))
    if distinct == 0:
        return 0, 0
    tail_offset = (panel_count - 1 - start) % panel_count
    tail_count = 1 if tail_offset < distinct else 0
    return distinct - tail_count, tail_count

DEFAULT_LAUNCH_ORDER = ("m", 8, True)

@dataclass(frozen=True)
class LaunchOrder:
    sweep_axis: str
    strip_width: int
    serpentine: bool
    legacy: bool
    padded: bool = False

def _launch_order(kernel: GemmKernelSpec, grid: GridAccounting) -> LaunchOrder:
    mode = _settings.get("OPMODEL_HBM_ORDER")
    if mode == "cutlass_identity_swizzle":
        n = grid.blocks_n
        return LaunchOrder("m", 8 if n >= 6 else 4 if n >= 3 else 2 if n >= 2 else 1, False, False, True)
    if mode == "legacy":
        return LaunchOrder("m", grid.blocks_n, False, True)
    if mode == "rowmajor_general":
        return LaunchOrder("m", grid.blocks_n, False, False)
    if mode == "launch" and kernel.order_sweep_axis in ("m", "n"):
        axis, width, serp = kernel.order_sweep_axis, kernel.order_strip_width, kernel.order_serpentine
    elif mode in ("launch", "default"):
        axis, width, serp = _settings.get("OPMODEL_HBM_DEFAULT_ORDER")
    else:
        raise ValueError(f"OPMODEL_HBM_ORDER={mode!r}")
    strip_tiles = grid.blocks_n if axis == "m" else grid.blocks_m
    return LaunchOrder(axis, max(1, min(int(width), strip_tiles)), bool(serp), False)

def _order_tile(cta: int, order: LaunchOrder, grid: GridAccounting) -> tuple[int, int]:
    sweep_tiles = grid.blocks_m if order.sweep_axis == "m" else grid.blocks_n
    strip_tiles = grid.blocks_n if order.sweep_axis == "m" else grid.blocks_m
    width = order.strip_width
    strip = cta // (sweep_tiles * width)
    local = cta - strip * sweep_tiles * width
    w = width if order.padded else min(width, strip_tiles - strip * width)
    x, y = local // w, strip * width + local % w
    if order.serpentine and strip % 2 == 1:
        x = sweep_tiles - 1 - x
    return (x, y) if order.sweep_axis == "m" else (y, x)

def _launch_slots_per_batch(grid: GridAccounting, order: LaunchOrder) -> int:
    if order.padded:
        return grid.blocks_m * order.strip_width * _ceil_div(grid.blocks_n, order.strip_width)
    return grid.blocks_m * grid.blocks_n

def _valid_launch_tiles(
    problem: GemmProblemSpec, grid: GridAccounting, order: LaunchOrder, start: int, count: int,
) -> Iterator[tuple[int, int, int, int]]:
    per_batch = _launch_slots_per_batch(grid, order)
    end = start + max(0, count)
    if order.padded:
        end = min(end, problem.batch * per_batch)
    for slot in range(start, end):
        batch, local = divmod(slot, per_batch)
        mi, ni = _order_tile(local, order, grid)
        if not order.padded or (mi < grid.blocks_m and ni < grid.blocks_n):
            yield slot, batch, mi, ni

_GEOMETRY_CACHE: "OrderedDict[tuple, _LaunchGeometry]" = OrderedDict()
_GEOMETRY_CACHE_TILE_BUDGET = 6_000_000

class _ReuseEvents:
    __slots__ = ("within_counts", "cross_counts", "cls", "pair", "pair_previous", "pair_wave")

    def __init__(self, within_counts, cross_counts, cls, pair, pair_previous, pair_wave):
        self.within_counts = within_counts
        self.cross_counts = cross_counts
        self.cls = cls
        self.pair = pair
        self.pair_previous = pair_previous
        self.pair_wave = pair_wave

class _LaunchGeometry:
    def __init__(self, blocks_m: int, blocks_n: int, order: LaunchOrder) -> None:
        grid = GridAccounting(blocks_m, blocks_n, 0, 0, 0.0, 0.0, 0.0)
        self.blocks_m, self.blocks_n, self.order = blocks_m, blocks_n, order
        self.per_batch = _launch_slots_per_batch(grid, order)
        mi_list, ni_list, slot_list, valid_before = [], [], [], [0]
        for local in range(self.per_batch):
            mi, ni = _order_tile(local, order, grid)
            if not order.padded or (mi < blocks_m and ni < blocks_n):
                mi_list.append(mi)
                ni_list.append(ni)
                slot_list.append(local)
            valid_before.append(len(mi_list))
        self.mi, self.ni, self.slot = mi_list, ni_list, slot_list
        self.valid_before = valid_before
        last_m, last_n = blocks_m - 1, blocks_n - 1

        self.tail_m_before = [0, *accumulate(int(mi == last_m and ni != last_n) for mi, ni in zip(mi_list, ni_list))]
        self.tail_n_before = [0, *accumulate(int(mi != last_m and ni == last_n) for mi, ni in zip(mi_list, ni_list))]
        self.tail_mn_before = [0, *accumulate(int(mi == last_m and ni == last_n) for mi, ni in zip(mi_list, ni_list))]
        self._panel_counts: dict = {}
        self._reuse_events: dict = {}
        self.entries = 6 * len(mi_list) + self.per_batch

    def segments(self, batch_count: int, start: int, count: int):
        end = start + max(0, count)
        if self.order.padded:
            end = min(end, batch_count * self.per_batch)
        if end <= start:
            return
        per = self.per_batch
        first, last = start // per, (end - 1) // per
        for batch in range(first, last + 1):
            lo = start - batch * per if batch == first else 0
            hi = end - batch * per if batch == last else per
            yield batch, self.valid_before[lo], self.valid_before[hi]

    def panel_counts(self, batch_count: int, weight_is_batched: bool, start: int, count: int) -> tuple[int, int, int, int]:
        key = (batch_count if self.order.padded else 0, weight_is_batched, start, count)
        counts = self._panel_counts.get(key)
        if counts is None:
            last_m, last_n = self.blocks_m - 1, self.blocks_n - 1
            a_panels = a_tail = b_panels = b_tail = 0
            shared_b: set = set()
            for _, c0, c1 in self.segments(batch_count, start, count):
                m_tiles = set(self.mi[c0:c1])
                a_panels += len(m_tiles)
                a_tail += last_m in m_tiles
                n_tiles = set(self.ni[c0:c1])
                if weight_is_batched:
                    b_panels += len(n_tiles)
                    b_tail += last_n in n_tiles
                else:
                    shared_b |= n_tiles
            if not weight_is_batched:
                b_panels, b_tail = len(shared_b), int(last_n in shared_b)
            counts = self._panel_counts[key] = (a_panels - a_tail, a_tail, b_panels - b_tail, b_tail)
            self.entries += 8
        return counts

    def valid_count(self, batch_count: int, start: int, count: int) -> int:
        return sum(c1 - c0 for _, c0, c1 in self.segments(batch_count, start, count))

    def output_class_counts(self, batch_count: int, start: int, count: int) -> tuple[int, int, int, int]:
        total = tail_m = tail_n = tail_mn = 0
        for _, c0, c1 in self.segments(batch_count, start, count):
            total += c1 - c0
            tail_m += self.tail_m_before[c1] - self.tail_m_before[c0]
            tail_n += self.tail_n_before[c1] - self.tail_n_before[c0]
            tail_mn += self.tail_mn_before[c1] - self.tail_mn_before[c0]
        return total - tail_m - tail_n - tail_mn, tail_m, tail_n, tail_mn

    def reuse_events(self, batch_count: int, weight_is_batched: bool, ctas_per_wave: int) -> tuple:
        key = (batch_count, weight_is_batched, ctas_per_wave)
        events = self._reuse_events.get(key)
        if events is not None:
            return events
        valid = len(self.mi)
        slot = (np.arange(batch_count, dtype=np.int64)[:, None] * self.per_batch
                + np.asarray(self.slot, dtype=np.int64)[None, :]).reshape(-1)
        wave = slot // ctas_per_wave
        wave_span = int(wave[-1]) + 1 if len(wave) else 1
        batch_item = np.repeat(np.arange(batch_count, dtype=np.int64), valid)
        out = []
        for tiles, blocks, batched in ((self.mi, self.blocks_m, True), (self.ni, self.blocks_n, weight_is_batched)):
            tile = np.tile(np.asarray(tiles, dtype=np.int64), batch_count)
            panel = batch_item * blocks + tile if batched else tile
            by_panel = np.argsort(panel, kind="stable")
            sorted_panel, sorted_wave = panel[by_panel], wave[by_panel]
            seen_sorted = np.zeros(len(panel), dtype=bool)
            previous_sorted = np.zeros(len(panel), dtype=np.int64)
            if len(panel) > 1:
                seen_sorted[1:] = sorted_panel[1:] == sorted_panel[:-1]
                previous_sorted[1:] = sorted_wave[:-1]
            seen = np.empty(len(panel), dtype=bool)
            previous = np.empty(len(panel), dtype=np.int64)
            seen[by_panel] = seen_sorted
            previous[by_panel] = previous_sorted
            tail = tile == blocks - 1
            within = seen & (previous == wave)
            cross = seen & (previous != wave)
            cls = tail[cross].astype(np.uint8)
            code, pair = np.unique(previous[cross] * wave_span + wave[cross], return_inverse=True)
            out.append(_ReuseEvents(
                within_counts=(int(np.count_nonzero(within & ~tail)), int(np.count_nonzero(within & tail))),
                cross_counts=(int(np.count_nonzero(cls == 0)), int(np.count_nonzero(cls == 1))),
                cls=cls, pair=pair.astype(np.int64, copy=False).reshape(-1),
                pair_previous=code // wave_span, pair_wave=code % wave_span))
            self.entries += 9 * len(cls) + 16 * len(code)
        events = self._reuse_events[key] = tuple(out)
        return events

def _launch_geometry(blocks_m: int, blocks_n: int, order: LaunchOrder) -> _LaunchGeometry:
    key = (blocks_m, blocks_n, order)
    geometry = _GEOMETRY_CACHE.get(key)
    if geometry is None:
        geometry = _GEOMETRY_CACHE[key] = _LaunchGeometry(blocks_m, blocks_n, order)
        while len(_GEOMETRY_CACHE) > 1 and sum(g.entries for g in _GEOMETRY_CACHE.values()) > _GEOMETRY_CACHE_TILE_BUDGET:
            _GEOMETRY_CACHE.popitem(last=False)
    else:
        _GEOMETRY_CACHE.move_to_end(key)
    return geometry

def _panel_k_sum(extent: int, k_lengths: tuple[int, ...], dtype_bytes: float, unit: int) -> int:
    return sum(_sector_round_bytes(extent * k_len * dtype_bytes, unit) * n for k_len, n in Counter(k_lengths).items())

def _reference_ordered_wave_panel_counts(
    *, problem: GemmProblemSpec, grid: GridAccounting, order: LaunchOrder, start_cta: int, active_ctas: int
) -> tuple[int, int, int, int]:
    a_panels, b_panels = set(), set()
    for _, batch, mi, ni in _valid_launch_tiles(problem, grid, order, start_cta, active_ctas):
        a_panels.add((batch, mi))
        b_panels.add((batch if problem.weight_is_batched else 0, ni))
    a_tail = sum(1 for _, mi in a_panels if mi == grid.blocks_m - 1)
    b_tail = sum(1 for _, ni in b_panels if ni == grid.blocks_n - 1)
    return len(a_panels) - a_tail, a_tail, len(b_panels) - b_tail, b_tail

def _ordered_wave_panel_counts(
    *, problem: GemmProblemSpec, grid: GridAccounting, order: LaunchOrder, start_cta: int, active_ctas: int
) -> tuple[int, int, int, int]:
    return _launch_geometry(grid.blocks_m, grid.blocks_n, order).panel_counts(
        problem.batch, problem.weight_is_batched, start_cta, active_ctas)

def _representative_wave_panel_counts(
    *,
    problem: GemmProblemSpec,
    grid: GridAccounting,
    start_cta: int,
    active_ctas: int,
    order: LaunchOrder | None = None,
) -> tuple[int, int, int, int]:
    if active_ctas <= 0:
        return 0, 0, 0, 0
    if order is not None and not order.legacy:
        return _ordered_wave_panel_counts(
            problem=problem, grid=grid, order=order, start_cta=start_cta, active_ctas=active_ctas
        )

    end_cta = start_cta + active_ctas - 1
    first_a_panel = start_cta // grid.blocks_n
    last_a_panel = end_cta // grid.blocks_n
    a_panel_count = last_a_panel - first_a_panel + 1
    a_tail_panels = _count_periodic_indices(
        start=first_a_panel,
        count=a_panel_count,
        period=grid.blocks_m,
        target=grid.blocks_m - 1,
    )
    a_full_panels = a_panel_count - a_tail_panels

    if not problem.weight_is_batched:
        b_full_panels, b_tail_panels = _cyclic_distinct_panel_counts(
            start=start_cta % grid.blocks_n,
            request_count=active_ctas,
            panel_count=grid.blocks_n,
        )
        return a_full_panels, a_tail_panels, b_full_panels, b_tail_panels

    ctas_per_batch = grid.blocks_m * grid.blocks_n
    first_batch = start_cta // ctas_per_batch
    last_batch = end_cta // ctas_per_batch
    if first_batch == last_batch:
        b_full_panels, b_tail_panels = _cyclic_distinct_panel_counts(
            start=start_cta % grid.blocks_n,
            request_count=active_ctas,
            panel_count=grid.blocks_n,
        )
        return a_full_panels, a_tail_panels, b_full_panels, b_tail_panels

    first_batch_requests = ctas_per_batch - start_cta % ctas_per_batch
    last_batch_requests = end_cta % ctas_per_batch + 1
    interior_batches = max(0, last_batch - first_batch - 1)
    first_full, first_tail = _cyclic_distinct_panel_counts(
        start=start_cta % grid.blocks_n,
        request_count=first_batch_requests,
        panel_count=grid.blocks_n,
    )
    last_full, last_tail = _cyclic_distinct_panel_counts(
        start=0,
        request_count=last_batch_requests,
        panel_count=grid.blocks_n,
    )
    b_full_panels = first_full + last_full + interior_batches * (grid.blocks_n - 1)
    b_tail_panels = first_tail + last_tail + interior_batches
    return a_full_panels, a_tail_panels, b_full_panels, b_tail_panels

def _representative_l2_window_bytes(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    start_cta: int,
    active_ctas: int,
    k_lengths: tuple[int, ...],
    dtype_bytes: float,
    line_size: int,
    order: LaunchOrder | None = None,
) -> int:
    a_full, a_tail, b_full, b_tail = _representative_wave_panel_counts(
        problem=problem,
        grid=grid,
        start_cta=start_cta,
        active_ctas=active_ctas,
        order=order,
    )
    tail_m = problem.m - (grid.blocks_m - 1) * kernel.cta_m
    tail_n = problem.n - (grid.blocks_n - 1) * kernel.cta_n
    a_b, b_b = _operand_bytes(problem, dtype_bytes)

    def panel_bytes(full_count: int, tail_count: int, full_extent: int, tail_extent: int, nbytes: float) -> int:
        return (full_count * _panel_k_sum(full_extent, k_lengths, nbytes, line_size)
                + tail_count * _panel_k_sum(tail_extent, k_lengths, nbytes, line_size))

    return panel_bytes(a_full, a_tail, kernel.cta_m, tail_m, a_b) + panel_bytes(
        b_full, b_tail, kernel.cta_n, tail_n, b_b
    )

def _l2_reuse_footprint(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    k_tiles: tuple[int, ...],
    capacity_bytes: int | None,
    dtype_bytes: float,
    line_size: int,
    order: LaunchOrder | None = None,
) -> L2ReuseFootprint:
    k_window_stages = max(1, min(grid.k_stages, kernel.pipeline_stages))
    k_window_count = _ceil_div(grid.k_stages, k_window_stages)
    full_k_window_count = max(0, k_window_count - 1)
    full_k_lengths = k_tiles[:k_window_stages]
    tail_k_lengths = k_tiles[full_k_window_count * k_window_stages :]

    full_wave_count = max(0, occupancy.wave_count - 1)
    last_wave_start = full_wave_count * occupancy.ctas_per_wave
    wave_classes = (
        (
            full_wave_count,
            0,
            occupancy.full_wave_ctas,
            "full",
        ),
        (1, last_wave_start, occupancy.last_wave_ctas, "last"),
    )
    if order is not None and order.padded:
        slots = problem.batch * _launch_slots_per_batch(grid, order)
        wave_classes = tuple(
            (1, start, min(occupancy.ctas_per_wave, slots - start),
             "last" if start + occupancy.ctas_per_wave >= slots else "full")
            for start in range(0, slots, occupancy.ctas_per_wave)
        )
    k_classes = (
        (full_k_window_count, full_k_lengths, "full"),
        (1, tail_k_lengths, "tail"),
    )

    footprints = {
        ("full", "full"): 0,
        ("full", "tail"): 0,
        ("last", "full"): 0,
        ("last", "tail"): 0,
    }
    weighted_footprint = 0.0
    weighted_miss_fraction = 0.0
    total_weight = 0.0
    peak_working_set = 0

    geometry = (_launch_geometry(grid.blocks_m, grid.blocks_n, order)
                if order is not None and not order.legacy else None)
    tail_m = problem.m - (grid.blocks_m - 1) * kernel.cta_m
    tail_n = problem.n - (grid.blocks_n - 1) * kernel.cta_n
    k_sums: dict = {}
    for wave_occurrences, start_cta, active_ctas, wave_name in wave_classes:
        for k_occurrences, k_lengths, k_name in k_classes:
            if wave_occurrences <= 0 or k_occurrences <= 0 or not k_lengths:
                continue
            if geometry is None:
                working_set = _representative_l2_window_bytes(
                    problem=problem,
                    kernel=kernel,
                    grid=grid,
                    start_cta=start_cta,
                    active_ctas=active_ctas,
                    k_lengths=k_lengths,
                    dtype_bytes=dtype_bytes,
                    line_size=line_size,
                    order=order,
                )
            else:
                if k_name not in k_sums:
                    a_b, b_b = _operand_bytes(problem, dtype_bytes)
                    k_sums[k_name] = (
                        tuple(_panel_k_sum(extent, k_lengths, nbytes, line_size)
                              for extent, nbytes in ((kernel.cta_m, a_b), (tail_m, a_b),
                                                     (kernel.cta_n, b_b), (tail_n, b_b))),
                        sum(k_lengths))
                a_full, a_tail, b_full, b_tail = ((0, 0, 0, 0) if active_ctas <= 0 else geometry.panel_counts(
                    problem.batch, problem.weight_is_batched, start_cta, active_ctas))
                a_full_bytes, a_tail_bytes, b_full_bytes, b_tail_bytes = k_sums[k_name][0]
                working_set = ((a_full * a_full_bytes + a_tail * a_tail_bytes)
                               + (b_full * b_full_bytes + b_tail * b_tail_bytes))
            footprints[(wave_name, k_name)] = working_set
            peak_working_set = max(peak_working_set, working_set)
            if geometry is not None and order.padded:
                weight = float(wave_occurrences * k_occurrences
                               * geometry.valid_count(problem.batch, start_cta, active_ctas) * k_sums[k_name][1])
            else:
                weight = float(
                    wave_occurrences
                    * k_occurrences
                    * (sum(1 for _ in _valid_launch_tiles(problem, grid, order, start_cta, active_ctas))
                       if order is not None and order.padded else active_ctas)
                    * sum(k_lengths)
                )
            miss_fraction = 0.0
            if capacity_bytes is not None and working_set > capacity_bytes:
                miss_fraction = _clamp01(1.0 - capacity_bytes / working_set)
            weighted_footprint += weight * working_set
            weighted_miss_fraction += weight * miss_fraction
            total_weight += weight

    return L2ReuseFootprint(
        peak_working_set_bytes=peak_working_set,
        weighted_working_set_bytes=(weighted_footprint / total_weight if total_weight else 0.0),
        miss_fraction=(weighted_miss_fraction / total_weight if total_weight else 0.0),
        k_window_stages=k_window_stages,
        k_window_count=k_window_count,
        full_wave_full_window_bytes=footprints[("full", "full")],
        full_wave_tail_window_bytes=footprints[("full", "tail")],
        last_wave_full_window_bytes=footprints[("last", "full")],
        last_wave_tail_window_bytes=footprints[("last", "tail")],
    )

def _l2_capacity_estimate(
    *,
    a_requested_bytes: int,
    b_requested_bytes: int,
    a_unique_bytes: int,
    b_unique_bytes: int,
    working_set_bytes: int,
    capacity_bytes: int | None,
    sector_size_bytes: int,
    capacity_miss_fraction: float | None = None,
) -> L2CapacityEstimate:
    if capacity_miss_fraction is None:
        miss_fraction = 0.0
        if capacity_bytes is not None and working_set_bytes > capacity_bytes:
            miss_fraction = _clamp01(1.0 - capacity_bytes / working_set_bytes)
    else:
        miss_fraction = _clamp01(capacity_miss_fraction)

    def actual_read(requested: int, unique: int) -> int:
        repeated = max(0, requested - unique)
        capacity_miss = min(
            repeated,
            _sector_round_bytes(repeated * miss_fraction, sector_size_bytes),
        )
        return min(requested, unique + capacity_miss)

    a_dram_read = actual_read(a_requested_bytes, a_unique_bytes)
    b_dram_read = actual_read(b_requested_bytes, b_unique_bytes)
    return L2CapacityEstimate(
        capacity_bytes=capacity_bytes,
        working_set_bytes=working_set_bytes,
        miss_fraction=miss_fraction,
        a_dram_read_bytes=a_dram_read,
        b_dram_read_bytes=b_dram_read,
        a_capacity_miss_bytes=a_dram_read - a_unique_bytes,
        b_capacity_miss_bytes=b_dram_read - b_unique_bytes,
    )

def _hybrid_l2_reuse_distance_estimate(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    reuse_footprint: L2ReuseFootprint,
    k_tiles: tuple[int, ...],
    a_requested_bytes: int,
    b_requested_bytes: int,
    a_unique_bytes: int,
    b_unique_bytes: int,
    c_read_bytes: int,
    d_store_bytes: int,
    operation_working_set_bytes: int,
    capacity_bytes: int | None,
    dtype_bytes: float,
    sector_size_bytes: int,
    line_size_bytes: int,
    order: LaunchOrder | None = None,
) -> L2ReuseDistanceEstimate:
    if order is not None and not order.legacy:
        return _ordered_reuse_distance_estimate(**{k: v for k, v in locals().items()})
    a_repeated = max(0, a_requested_bytes - a_unique_bytes)
    b_repeated = max(0, b_requested_bytes - b_unique_bytes)
    a_cross_share = 0.0
    b_cross_share = 0.0
    if occupancy.wave_count > 1:
        wave_boundaries = occupancy.wave_count - 1
        a_group_aligned_boundaries = (
            (grid.cta_count - 1)
            // math.lcm(occupancy.ctas_per_wave, grid.blocks_n)
        )
        a_cross_events = max(0, wave_boundaries - a_group_aligned_boundaries)
        a_repeat_events = problem.batch * grid.blocks_m * max(0, grid.blocks_n - 1)
        if a_repeat_events > 0:
            a_cross_share = _clamp01(a_cross_events / a_repeat_events)
        if b_repeated > 0:
            b_cross_share = _clamp01(grid.blocks_n / occupancy.ctas_per_wave)

    def split_repeated(repeated: int, cross_share: float) -> tuple[int, int]:
        cross = min(
            repeated,
            _sector_round_bytes(repeated * cross_share, sector_size_bytes),
        )
        return repeated - cross, cross

    a_within, a_cross = split_repeated(a_repeated, a_cross_share)
    b_within, b_cross = split_repeated(b_repeated, b_cross_share)

    full_wave_all_k_bytes = 0
    wave_output_bytes = 0
    if occupancy.wave_count > 1:
        full_wave_all_k_bytes = _representative_l2_window_bytes(
            problem=problem,
            kernel=kernel,
            grid=grid,
            start_cta=0,
            active_ctas=occupancy.full_wave_ctas,
            k_lengths=k_tiles,
            dtype_bytes=dtype_bytes,
            line_size=line_size_bytes,
        )
        wave_output_bytes = _sector_round_bytes(
            (c_read_bytes + d_store_bytes)
            * occupancy.full_wave_ctas
            / max(1, grid.cta_count),
            line_size_bytes,
        )
    a_cross_distance = float(full_wave_all_k_bytes + wave_output_bytes)
    b_wave_gap = max(1.0, grid.blocks_n / max(1, occupancy.ctas_per_wave))
    b_cross_distance = a_cross_distance * b_wave_gap

    capacity_active = (
        capacity_bytes is not None
        and operation_working_set_bytes > capacity_bytes
    )

    def reuse_distance_miss_fraction(distance: float) -> float:
        if not capacity_active or capacity_bytes is None or distance <= capacity_bytes:
            return 0.0

        operation_pressure = _clamp01(
            1.0 - capacity_bytes / operation_working_set_bytes
        )
        reuse_interval_pressure = _clamp01(1.0 - capacity_bytes / distance)
        return operation_pressure * reuse_interval_pressure

    a_cross_miss_fraction = reuse_distance_miss_fraction(a_cross_distance)
    b_cross_miss_fraction = reuse_distance_miss_fraction(b_cross_distance)
    within_miss_fraction = reuse_footprint.miss_fraction

    def actual_read(
        *, unique: int, requested: int, within: int, cross: int, cross_fraction: float
    ) -> tuple[int, int]:
        within_miss = _sector_round_bytes(
            within * within_miss_fraction, sector_size_bytes
        )
        cross_miss = _sector_round_bytes(cross * cross_fraction, sector_size_bytes)
        capacity_miss = min(max(0, requested - unique), within_miss + cross_miss)
        read = min(requested, unique + capacity_miss)
        return read, read - unique

    a_dram_read, a_capacity_miss = actual_read(
        unique=a_unique_bytes,
        requested=a_requested_bytes,
        within=a_within,
        cross=a_cross,
        cross_fraction=a_cross_miss_fraction,
    )
    b_dram_read, b_capacity_miss = actual_read(
        unique=b_unique_bytes,
        requested=b_requested_bytes,
        within=b_within,
        cross=b_cross,
        cross_fraction=b_cross_miss_fraction,
    )
    repeated_total = a_repeated + b_repeated
    capacity_miss_total = a_capacity_miss + b_capacity_miss
    capacity_estimate = L2CapacityEstimate(
        capacity_bytes=capacity_bytes,
        working_set_bytes=operation_working_set_bytes,
        miss_fraction=(
            capacity_miss_total / repeated_total if repeated_total else 0.0
        ),
        a_dram_read_bytes=a_dram_read,
        b_dram_read_bytes=b_dram_read,
        a_capacity_miss_bytes=a_capacity_miss,
        b_capacity_miss_bytes=b_capacity_miss,
    )
    return L2ReuseDistanceEstimate(
        capacity_estimate=capacity_estimate,
        a_within_wave_repeated_bytes=a_within,
        a_cross_wave_repeated_bytes=a_cross,
        b_within_wave_repeated_bytes=b_within,
        b_cross_wave_repeated_bytes=b_cross,
        within_wave_miss_fraction=within_miss_fraction,
        a_cross_wave_reuse_distance_bytes=a_cross_distance,
        b_cross_wave_reuse_distance_bytes=b_cross_distance,
        a_cross_wave_miss_fraction=a_cross_miss_fraction,
        b_cross_wave_miss_fraction=b_cross_miss_fraction,
    )

def _interval_miss_fraction(
    distance: float, capacity: int | None, active: bool, operation_pressure: float, serpentine: bool
) -> float:
    if not active or capacity is None or distance <= 0:
        return 0.0
    if _settings.get("OPMODEL_HBM_MISS_SHAPE") == "recency":
        if serpentine:
            frac = 0.0 if 2.0 * distance <= capacity else 1.0 - capacity / (2.0 * distance)
        else:
            frac = 1.0 if distance > capacity else 0.0
        if not _settings.get("OPMODEL_HBM_SINGLE_MISS"):
            frac *= operation_pressure
        return _clamp01(frac)
    if serpentine:
        span = 2.0 * distance
        if span <= capacity:
            return 0.0
        frac = (span - capacity - capacity * math.log(span / capacity)) / span
    else:
        if distance <= capacity:
            return 0.0
        frac = 1.0 - capacity / distance
    if not _settings.get("OPMODEL_HBM_SINGLE_MISS"):
        frac *= operation_pressure
    return _clamp01(frac)

def _reference_padded_reuse_distance_estimate(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    reuse_footprint: L2ReuseFootprint,
    k_tiles: tuple[int, ...],
    a_requested_bytes: int,
    b_requested_bytes: int,
    a_unique_bytes: int,
    b_unique_bytes: int,
    c_read_bytes: int,
    d_store_bytes: int,
    operation_working_set_bytes: int,
    capacity_bytes: int | None,
    dtype_bytes: float,
    sector_size_bytes: int,
    line_size_bytes: int,
    order: LaunchOrder,
) -> L2ReuseDistanceEstimate:
    cpw = max(1, occupancy.ctas_per_wave)
    slots = problem.batch * _launch_slots_per_batch(grid, order)
    wave_count = _ceil_div(slots, cpw)
    wave_bytes = []
    evict_first = _settings.get("OPMODEL_HBM_EF_STORES") and kernel.store_evict_first
    for wave in range(wave_count):
        start, count = wave * cpw, min(cpw, slots - wave * cpw)
        operands = _representative_l2_window_bytes(
            problem=problem, kernel=kernel, grid=grid, start_cta=start,
            active_ctas=count, k_lengths=k_tiles, dtype_bytes=dtype_bytes,
            line_size=line_size_bytes, order=order,
        )
        outputs = 0
        if not evict_first:
            for _, _, mi, ni in _valid_launch_tiles(problem, grid, order, start, count):
                ml = min(kernel.cta_m, problem.m - mi * kernel.cta_m)
                nl = min(kernel.cta_n, problem.n - ni * kernel.cta_n)
                outputs += _sector_round_bytes(ml * nl * dtype_nbytes(problem.output_dtype), sector_size_bytes)
            outputs *= 1 + int(problem.epilogue_reads_c)
        wave_bytes.append(float(operands + outputs))
    prefix = [0.0]
    for size in wave_bytes:
        prefix.append(prefix[-1] + size)

    capacity_active = capacity_bytes is not None and operation_working_set_bytes > capacity_bytes
    op_pressure = (_clamp01(1.0 - capacity_bytes / operation_working_set_bytes)
                   if capacity_active else 0.0)

    last = ({}, {})
    within = [0, 0]
    cross = [0, 0]
    distance_sum = [0.0, 0.0]
    miss_sum = [0.0, 0.0]
    panel_sizes = {}
    operand_nbytes = _operand_bytes(problem, dtype_bytes)
    for slot, batch, mi, ni in _valid_launch_tiles(problem, grid, order, 0, slots):
        wave = slot // cpw
        for operand, key, extent in (
            (0, (batch, mi), min(kernel.cta_m, problem.m - mi * kernel.cta_m)),
            (1, (batch if problem.weight_is_batched else 0, ni),
             min(kernel.cta_n, problem.n - ni * kernel.cta_n)),
        ):
            nbytes = operand_nbytes[operand]
            if (extent, nbytes) not in panel_sizes:
                panel_sizes[(extent, nbytes)] = sum(
                    _sector_round_bytes(extent * kl * nbytes, sector_size_bytes) for kl in k_tiles)
            size = panel_sizes[(extent, nbytes)]
            previous = last[operand].get(key)
            if previous is not None:
                if previous == wave:
                    within[operand] += size
                else:
                    distance = prefix[wave] - prefix[previous]
                    fraction = _interval_miss_fraction(distance, capacity_bytes, capacity_active, op_pressure, False)
                    cross[operand] += size
                    distance_sum[operand] += size * distance
                    miss_sum[operand] += size * fraction
            last[operand][key] = wave
    requested = (a_requested_bytes, b_requested_bytes)
    unique = (a_unique_bytes, b_unique_bytes)
    reads, misses = [], []
    for i in range(2):
        repeated = max(0, requested[i] - unique[i])
        if within[i] + cross[i] != repeated:
            raise ValueError("CUTLASS padded panel requests do not match operand traffic")
        miss = min(repeated, _sector_round_bytes(
            within[i] * reuse_footprint.miss_fraction + miss_sum[i], sector_size_bytes))
        misses.append(miss)
        reads.append(unique[i] + miss)
    repeated_total = sum(within) + sum(cross)
    return L2ReuseDistanceEstimate(
        capacity_estimate=L2CapacityEstimate(
            capacity_bytes=capacity_bytes, working_set_bytes=operation_working_set_bytes,
            miss_fraction=sum(misses) / repeated_total if repeated_total else 0.0,
            a_dram_read_bytes=reads[0], b_dram_read_bytes=reads[1],
            a_capacity_miss_bytes=misses[0], b_capacity_miss_bytes=misses[1],
        ),
        a_within_wave_repeated_bytes=within[0], a_cross_wave_repeated_bytes=cross[0],
        b_within_wave_repeated_bytes=within[1], b_cross_wave_repeated_bytes=cross[1],
        within_wave_miss_fraction=reuse_footprint.miss_fraction,
        a_cross_wave_reuse_distance_bytes=distance_sum[0] / cross[0] if cross[0] else 0.0,
        b_cross_wave_reuse_distance_bytes=distance_sum[1] / cross[1] if cross[1] else 0.0,
        a_cross_wave_miss_fraction=miss_sum[0] / cross[0] if cross[0] else 0.0,
        b_cross_wave_miss_fraction=miss_sum[1] / cross[1] if cross[1] else 0.0,
    )

def _padded_reuse_distance_estimate(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    reuse_footprint: L2ReuseFootprint,
    k_tiles: tuple[int, ...],
    a_requested_bytes: int,
    b_requested_bytes: int,
    a_unique_bytes: int,
    b_unique_bytes: int,
    c_read_bytes: int,
    d_store_bytes: int,
    operation_working_set_bytes: int,
    capacity_bytes: int | None,
    dtype_bytes: float,
    sector_size_bytes: int,
    line_size_bytes: int,
    order: LaunchOrder,
) -> L2ReuseDistanceEstimate:
    arguments = dict(locals())
    last_m, last_n = grid.blocks_m - 1, grid.blocks_n - 1
    tail_m, tail_n = problem.m - last_m * kernel.cta_m, problem.n - last_n * kernel.cta_n
    if not (0 < tail_m <= kernel.cta_m and 0 < tail_n <= kernel.cta_n):
        return _reference_padded_reuse_distance_estimate(**arguments)
    geometry = _launch_geometry(grid.blocks_m, grid.blocks_n, order)
    cpw = max(1, occupancy.ctas_per_wave)
    slots = problem.batch * _launch_slots_per_batch(grid, order)
    wave_count = _ceil_div(slots, cpw)
    wave_bytes = []
    evict_first = _settings.get("OPMODEL_HBM_EF_STORES") and kernel.store_evict_first

    a_b, b_b = _operand_bytes(problem, dtype_bytes)
    window = tuple(_panel_k_sum(extent, k_tiles, nbytes, line_size_bytes)
                   for extent, nbytes in ((kernel.cta_m, a_b), (tail_m, a_b), (kernel.cta_n, b_b), (tail_n, b_b)))
    out_bytes = dtype_nbytes(problem.output_dtype)
    output = tuple(_sector_round_bytes(ml * nl * out_bytes, sector_size_bytes)
                   for ml, nl in ((kernel.cta_m, kernel.cta_n), (tail_m, kernel.cta_n),
                                  (kernel.cta_m, tail_n), (tail_m, tail_n)))
    for wave in range(wave_count):
        start, count = wave * cpw, min(cpw, slots - wave * cpw)
        a_full, a_tail, b_full, b_tail = ((0, 0, 0, 0) if count <= 0 else geometry.panel_counts(
            problem.batch, problem.weight_is_batched, start, count))
        operands = ((a_full * window[0] + a_tail * window[1]) + (b_full * window[2] + b_tail * window[3]))
        outputs = 0
        if not evict_first:
            outputs = sum(n * size for n, size in zip(
                geometry.output_class_counts(problem.batch, start, count), output))
            outputs *= 1 + int(problem.epilogue_reads_c)
        wave_bytes.append(float(operands + outputs))
    prefix = [0.0]
    for size in wave_bytes:
        prefix.append(prefix[-1] + size)

    capacity_active = capacity_bytes is not None and operation_working_set_bytes > capacity_bytes
    op_pressure = (_clamp01(1.0 - capacity_bytes / operation_working_set_bytes)
                   if capacity_active else 0.0)

    panel_size = ((_panel_k_sum(kernel.cta_m, k_tiles, a_b, sector_size_bytes),
                   _panel_k_sum(tail_m, k_tiles, a_b, sector_size_bytes)),
                  (_panel_k_sum(kernel.cta_n, k_tiles, b_b, sector_size_bytes),
                   _panel_k_sum(tail_n, k_tiles, b_b, sector_size_bytes)))
    within = [0, 0]
    cross = [0, 0]
    distance_sum = [0.0, 0.0]
    miss_sum = [0.0, 0.0]
    prefix_array = np.array(prefix, dtype=np.float64)
    events = geometry.reuse_events(problem.batch, problem.weight_is_batched, cpw)
    for operand in range(2):
        ev, sizes = events[operand], panel_size[operand]
        within[operand] = ev.within_counts[0] * sizes[0] + ev.within_counts[1] * sizes[1]
        cross[operand] = ev.cross_counts[0] * sizes[0] + ev.cross_counts[1] * sizes[1]
        if not len(ev.cls):
            continue
        pair_distance = prefix_array[ev.pair_wave] - prefix_array[ev.pair_previous]
        pair_fraction = np.array([
            _interval_miss_fraction(distance, capacity_bytes, capacity_active, op_pressure, False)
            for distance in pair_distance.tolist()], dtype=np.float64)
        size = np.array([float(sizes[0]), float(sizes[1])], dtype=np.float64)[ev.cls]
        zero = np.zeros(1, dtype=np.float64)
        distance_sum[operand] = float(np.cumsum(np.concatenate((zero, size * pair_distance[ev.pair])))[-1])
        miss_sum[operand] = float(np.cumsum(np.concatenate((zero, size * pair_fraction[ev.pair])))[-1])
    requested = (a_requested_bytes, b_requested_bytes)
    unique = (a_unique_bytes, b_unique_bytes)
    reads, misses = [], []
    for i in range(2):
        repeated = max(0, requested[i] - unique[i])
        if within[i] + cross[i] != repeated:
            raise ValueError("CUTLASS padded panel requests do not match operand traffic")
        miss = min(repeated, _sector_round_bytes(
            within[i] * reuse_footprint.miss_fraction + miss_sum[i], sector_size_bytes))
        misses.append(miss)
        reads.append(unique[i] + miss)
    repeated_total = sum(within) + sum(cross)
    return L2ReuseDistanceEstimate(
        capacity_estimate=L2CapacityEstimate(
            capacity_bytes=capacity_bytes, working_set_bytes=operation_working_set_bytes,
            miss_fraction=sum(misses) / repeated_total if repeated_total else 0.0,
            a_dram_read_bytes=reads[0], b_dram_read_bytes=reads[1],
            a_capacity_miss_bytes=misses[0], b_capacity_miss_bytes=misses[1],
        ),
        a_within_wave_repeated_bytes=within[0], a_cross_wave_repeated_bytes=cross[0],
        b_within_wave_repeated_bytes=within[1], b_cross_wave_repeated_bytes=cross[1],
        within_wave_miss_fraction=reuse_footprint.miss_fraction,
        a_cross_wave_reuse_distance_bytes=distance_sum[0] / cross[0] if cross[0] else 0.0,
        b_cross_wave_reuse_distance_bytes=distance_sum[1] / cross[1] if cross[1] else 0.0,
        a_cross_wave_miss_fraction=miss_sum[0] / cross[0] if cross[0] else 0.0,
        b_cross_wave_miss_fraction=miss_sum[1] / cross[1] if cross[1] else 0.0,
    )

def _ordered_reuse_distance_estimate(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    reuse_footprint: L2ReuseFootprint,
    k_tiles: tuple[int, ...],
    a_requested_bytes: int,
    b_requested_bytes: int,
    a_unique_bytes: int,
    b_unique_bytes: int,
    c_read_bytes: int,
    d_store_bytes: int,
    operation_working_set_bytes: int,
    capacity_bytes: int | None,
    dtype_bytes: float,
    sector_size_bytes: int,
    line_size_bytes: int,
    order: LaunchOrder,
) -> L2ReuseDistanceEstimate:
    if order.padded:
        return _padded_reuse_distance_estimate(**locals())
    sweep_is_m = order.sweep_axis == "m"
    sweep_tiles = grid.blocks_m if sweep_is_m else grid.blocks_n
    strip_tiles = grid.blocks_n if sweep_is_m else grid.blocks_m
    width = order.strip_width
    strips = _ceil_div(strip_tiles, width)
    a_repeated = max(0, a_requested_bytes - a_unique_bytes)
    b_repeated = max(0, b_requested_bytes - b_unique_bytes)
    x_repeated, y_repeated = (a_repeated, b_repeated) if sweep_is_m else (b_repeated, a_repeated)
    x_batches = problem.batch if (sweep_is_m or problem.weight_is_batched) else 1

    cpw = max(1, occupancy.ctas_per_wave)
    wave_cross_share = 0.0
    strip_share = 0.0
    y_cross_share = 0.0
    full_wave_all_k_bytes = 0
    wave_output_bytes = 0
    if strip_tiles > 1:
        strip_share = _clamp01((strips - 1) / (strip_tiles - 1))
    if occupancy.wave_count > 1:
        wave_boundaries = occupancy.wave_count - 1
        aligned = (grid.cta_count - 1) // math.lcm(cpw, width)
        x_repeat_events = x_batches * sweep_tiles * max(0, strip_tiles - 1)
        if x_repeat_events > 0:
            wave_cross_share = _clamp01(max(0, wave_boundaries - aligned) / x_repeat_events)
        if y_repeated > 0:
            y_cross_share = _clamp01(width / cpw)
        full_wave_all_k_bytes = _representative_l2_window_bytes(
            problem=problem, kernel=kernel, grid=grid, start_cta=0, active_ctas=occupancy.full_wave_ctas,
            k_lengths=k_tiles, dtype_bytes=dtype_bytes, line_size=line_size_bytes, order=order,
        )
        wave_output_bytes = _sector_round_bytes(
            (c_read_bytes + d_store_bytes) * occupancy.full_wave_ctas / max(1, grid.cta_count), line_size_bytes
        )

    if _settings.get("OPMODEL_HBM_EF_STORES") and kernel.store_evict_first:
        wave_output_bytes = 0
    wave_bytes = float(full_wave_all_k_bytes + wave_output_bytes)

    strip_distance = wave_bytes * (sweep_tiles * width / cpw) if occupancy.wave_count > 1 else 0.0
    y_distance = wave_bytes * max(1.0, width / cpw)

    def split(repeated: int, share: float) -> int:
        return min(repeated, _sector_round_bytes(repeated * share, sector_size_bytes))

    x_strip = split(x_repeated, strip_share)

    if _settings.get("OPMODEL_HBM_STRIP_IN_WAVE") and occupancy.wave_count > 1:
        gap = sweep_tiles * width
        r = max(1, occupancy.resident_ctas_per_sm)
        burst = max(1, occupancy.num_sms)
        x_strip_in_wave = split(x_strip, max(0.0, 1.0 - gap / burst))
        x_strip -= x_strip_in_wave
        strip_distance = (wave_bytes / r) * max(1.0, gap / burst)
    x_wave = split(max(0, x_repeated - x_strip), wave_cross_share)
    x_within = x_repeated - x_strip - x_wave
    y_cross = split(y_repeated, y_cross_share)
    y_within = y_repeated - y_cross

    capacity_active = capacity_bytes is not None and operation_working_set_bytes > capacity_bytes
    op_pressure = (
        _clamp01(1.0 - capacity_bytes / operation_working_set_bytes) if capacity_active else 0.0
    )
    f_wave = _interval_miss_fraction(wave_bytes, capacity_bytes, capacity_active, op_pressure, False)
    f_strip = _interval_miss_fraction(strip_distance, capacity_bytes, capacity_active, op_pressure,
                                      order.serpentine)
    f_y = _interval_miss_fraction(y_distance, capacity_bytes, capacity_active, op_pressure, False)
    within_miss_fraction = reuse_footprint.miss_fraction

    def miss(within: int, parts: list[tuple[int, float]], requested: int, unique: int) -> tuple[int, int]:
        m = _sector_round_bytes(within * within_miss_fraction, sector_size_bytes)
        m += sum(_sector_round_bytes(b * f, sector_size_bytes) for b, f in parts)
        cap_miss = min(max(0, requested - unique), m)
        read = min(requested, unique + cap_miss)
        return read, read - unique

    x_req, x_uni = (a_requested_bytes, a_unique_bytes) if sweep_is_m else (b_requested_bytes, b_unique_bytes)
    y_req, y_uni = (b_requested_bytes, b_unique_bytes) if sweep_is_m else (a_requested_bytes, a_unique_bytes)
    x_read, x_miss = miss(x_within, [(x_wave, f_wave), (x_strip, f_strip)], x_req, x_uni)
    y_read, y_miss = miss(y_within, [(y_cross, f_y)], y_req, y_uni)
    x_cross = x_wave + x_strip
    x_frac = (x_wave * f_wave + x_strip * f_strip) / x_cross if x_cross else 0.0
    x_dist = (x_wave * wave_bytes + x_strip * strip_distance) / x_cross if x_cross else 0.0
    if sweep_is_m:
        a = (x_read, x_miss, x_within, x_cross, x_dist, x_frac)
        b = (y_read, y_miss, y_within, y_cross, y_distance, f_y)
    else:
        a = (y_read, y_miss, y_within, y_cross, y_distance, f_y)
        b = (x_read, x_miss, x_within, x_cross, x_dist, x_frac)
    repeated_total = a_repeated + b_repeated
    capacity_estimate = L2CapacityEstimate(
        capacity_bytes=capacity_bytes,
        working_set_bytes=operation_working_set_bytes,
        miss_fraction=((a[1] + b[1]) / repeated_total if repeated_total else 0.0),
        a_dram_read_bytes=a[0],
        b_dram_read_bytes=b[0],
        a_capacity_miss_bytes=a[1],
        b_capacity_miss_bytes=b[1],
    )
    return L2ReuseDistanceEstimate(
        capacity_estimate=capacity_estimate,
        a_within_wave_repeated_bytes=a[2],
        a_cross_wave_repeated_bytes=a[3],
        b_within_wave_repeated_bytes=b[2],
        b_cross_wave_repeated_bytes=b[3],
        within_wave_miss_fraction=within_miss_fraction,
        a_cross_wave_reuse_distance_bytes=a[4],
        b_cross_wave_reuse_distance_bytes=b[4],
        a_cross_wave_miss_fraction=a[5],
        b_cross_wave_miss_fraction=b[5],
    )

def _l2_bypass(hardware: HardwareSpec) -> bool:
    return bool(getattr(hardware, "l2_bypass", False))

def _effective_l2_capacity(hardware: HardwareSpec) -> int | None:
    if _l2_bypass(hardware):
        return 0
    l2 = hardware.memory_levels.get("l2")
    cap = l2.size_bytes if l2 is not None else None
    eff_bytes = _settings.get("OPMODEL_L2_EFFECTIVE_MIB")
    if cap is not None and eff_bytes is not None:
        cap = int(eff_bytes)
    return cap

def _split_k_hbm_read(problem: GemmProblemSpec, kernel: GemmKernelSpec, traffic, hardware: HardwareSpec,
                      traffic_fn=None):
    gz = kernel.split_k_slices
    if (not _settings.get("OPMODEL_HBM_SPLITK") or gz <= 1
            or _settings.get("OPMODEL_HBM_ORDER") == "legacy"):
        return traffic
    k_tiles = _ceil_div(problem.k, kernel.cta_k)
    per = _ceil_div(k_tiles, gz) * kernel.cta_k
    ks, rem = [], problem.k
    while rem > 0:
        ks.append(min(rem, per))
        rem -= ks[-1]
    dt_a, dt_b = _operand_bytes(problem, dtype_nbytes(problem.input_dtype))
    part = problem.batch * problem.m * problem.n * kernel.split_k_partial_bytes
    operand_read, foot = 0, []
    for kk in ks:
        sub = replace(problem, k=kk)
        g = _grid_accounting(sub, kernel)
        occ = _occupancy(sub, kernel, g, hardware, [])
        t = (traffic_fn or _traffic_accounting)(sub, kernel, g, occ, hardware, [])
        operand_read += t.memory_access.hbm_read_bytes
        foot.append(problem.batch * problem.m * kk * dt_a
                    + (problem.batch if problem.weight_is_batched else 1) * kk * problem.n * dt_b + part)
    cap = _effective_l2_capacity(hardware)
    miss = lambda d: cap is None or d > cap  # noqa: E731
    reduce_read = 0
    partial_write = 0
    if kernel.split_k_reduction == "separate":
        for i in range(len(ks)):
            if miss(sum(foot[i + 1:]) + foot[i] / 2.0):
                reduce_read += part

        partial_write = len(ks) * part
    elif kernel.split_k_reduction == "inplace":
        if _settings.get("OPMODEL_HBM_SPLITK_WRITES"):
            g0 = _grid_accounting(replace(problem, k=ks[0]), kernel)
            occ0 = _occupancy(replace(problem, k=ks[0]), kernel, g0, hardware, [])
            tiles = max(1, problem.batch * g0.blocks_m * g0.blocks_n)
            cpw = max(1, occ0.ctas_per_wave)
            crossings = sum(len({(s * tiles + t) // cpw for s in range(len(ks))}) - 1 for t in range(tiles))
            if miss(foot[0] * max(1.0, cpw / tiles)):
                reduce_read = part * crossings / tiles
                partial_write = part * crossings / tiles
        else:
            for i in range(1, len(ks)):
                if miss((foot[i - 1] + foot[i]) / 2.0):
                    reduce_read += part
    access = replace(traffic.memory_access, hbm_read_bytes=operand_read + reduce_read)

    if _settings.get("OPMODEL_HBM_SPLITK_WRITES"):
        access = replace(access, hbm_write_bytes=(traffic.memory_access.hbm_write_bytes or 0) + partial_write)
    return replace(traffic, memory_access=access)

def _traffic_accounting(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    hardware: HardwareSpec,
    warnings: list[str],
) -> TrafficAccounting:
    def compute():
        appended: list[str] = []
        return _traffic_accounting_uncached(problem, kernel, grid, occupancy, hardware, appended), tuple(appended)

    traffic, appended = _TRAFFIC_MEMO.get_or_compute(
        (typed(problem), typed(kernel), typed(grid), typed(occupancy), hardware_token(hardware)), compute)
    warnings.extend(appended)
    return traffic

def _traffic_accounting_uncached(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    occupancy: OccupancyResult,
    hardware: HardwareSpec,
    warnings: list[str],
) -> TrafficAccounting:
    dtype_bytes = dtype_nbytes(problem.input_dtype)
    out_dtype_bytes = dtype_nbytes(problem.output_dtype)
    if _mixed_operands(problem) and kernel.architecture == "sm90":
        raise NotImplementedError("weight-only operand bytes are described for the sm80 GEMM model only")
    a_bytes, b_bytes = _operand_bytes(problem, dtype_bytes)
    l2_level = hardware.memory_levels.get("l2")
    transaction_level = l2_level or hardware.memory_levels["hbm"]
    sector_size = transaction_level.sector_size_bytes or 32
    line_size = transaction_level.line_size_bytes or max(128, sector_size)
    if transaction_level.sector_size_bytes is None:
        warnings.append(f"{transaction_level.name}_sector_size_default_32B")
    if transaction_level.line_size_bytes is None:
        warnings.append(f"{transaction_level.name}_line_size_default_{line_size}B")

    k_tiles = _tile_lengths(problem.k, kernel.cta_k)

    m_groups = _tile_groups(problem.m, kernel.cta_m)
    n_groups = _tile_groups(problem.n, kernel.cta_n)
    k_groups = _tile_groups(problem.k, kernel.cta_k)
    a_unique_tx = sum(
        _sector_round_bytes(m_len * k_len * a_bytes, sector_size) * m_cnt * k_cnt
        for m_len, m_cnt in m_groups
        for k_len, k_cnt in k_groups
    )
    b_unique_one_batch_tx = sum(
        _sector_round_bytes(k_len * n_len * b_bytes, sector_size) * n_cnt * k_cnt
        for n_len, n_cnt in n_groups
        for k_len, k_cnt in k_groups
    )
    a_requested_tx = problem.batch * grid.blocks_n * a_unique_tx
    b_requested_tx = problem.batch * grid.blocks_m * b_unique_one_batch_tx
    a_unique_dram_tx = problem.batch * a_unique_tx
    b_unique_dram_tx = (
        problem.batch if problem.weight_is_batched else 1
    ) * b_unique_one_batch_tx

    d_store_tx = problem.batch * sum(
        _sector_round_bytes(m_len * n_len * out_dtype_bytes, sector_size) * m_cnt * n_cnt
        for m_len, m_cnt in m_groups
        for n_len, n_cnt in n_groups
    )
    c_read_tx = d_store_tx if problem.epilogue_reads_c else 0
    l2_read = a_requested_tx + b_requested_tx + c_read_tx if l2_level is not None else None
    l2_write = d_store_tx if l2_level is not None else None
    l2_to_smem_bytes = a_requested_tx + b_requested_tx if l2_level is not None else None
    l2_capacity = l2_level.size_bytes if l2_level is not None else None

    eff_bytes = _settings.get("OPMODEL_L2_EFFECTIVE_MIB")
    if l2_capacity is not None and eff_bytes is not None:
        l2_capacity = int(eff_bytes)
    order = _launch_order(kernel, grid)
    reuse_footprint = _l2_reuse_footprint(
        problem=problem,
        kernel=kernel,
        grid=grid,
        occupancy=occupancy,
        k_tiles=k_tiles,
        capacity_bytes=l2_capacity,
        dtype_bytes=dtype_bytes,
        line_size=line_size,
        order=order,
    )

    representative_batches = max(1, problem.batch)
    operation_working_set = (
        a_unique_dram_tx // representative_batches
        + b_unique_one_batch_tx
        + max(c_read_tx, d_store_tx) // representative_batches
    )
    if l2_level is None or _l2_bypass(hardware):
        hbm_read = a_requested_tx + b_requested_tx + c_read_tx
        hbm_write = d_store_tx
        capacity_estimate = L2CapacityEstimate(
            capacity_bytes=None,
            working_set_bytes=reuse_footprint.peak_working_set_bytes,
            miss_fraction=1.0,
            a_dram_read_bytes=a_requested_tx,
            b_dram_read_bytes=b_requested_tx,
            a_capacity_miss_bytes=max(0, a_requested_tx - a_unique_dram_tx),
            b_capacity_miss_bytes=max(0, b_requested_tx - b_unique_dram_tx),
        )
        reuse_distance_estimate = L2ReuseDistanceEstimate(
            capacity_estimate=capacity_estimate,
            a_within_wave_repeated_bytes=max(0, a_requested_tx - a_unique_dram_tx),
            a_cross_wave_repeated_bytes=0,
            b_within_wave_repeated_bytes=max(0, b_requested_tx - b_unique_dram_tx),
            b_cross_wave_repeated_bytes=0,
            within_wave_miss_fraction=1.0,
            a_cross_wave_reuse_distance_bytes=0.0,
            b_cross_wave_reuse_distance_bytes=0.0,
            a_cross_wave_miss_fraction=1.0,
            b_cross_wave_miss_fraction=1.0,
        )
        warnings.append("l2_level_absent_dram_uses_cta_requested_traffic" if l2_level is None
                        else "l2_bypass_dram_uses_cta_requested_traffic")
    else:
        if l2_capacity is None:
            warnings.append("l2_size_bytes_absent_dram_uses_unique_first_touch")
        reuse_distance_estimate = _hybrid_l2_reuse_distance_estimate(
            problem=problem,
            kernel=kernel,
            grid=grid,
            occupancy=occupancy,
            reuse_footprint=reuse_footprint,
            k_tiles=k_tiles,
            a_requested_bytes=a_requested_tx,
            b_requested_bytes=b_requested_tx,
            a_unique_bytes=a_unique_dram_tx,
            b_unique_bytes=b_unique_dram_tx,
            c_read_bytes=c_read_tx,
            d_store_bytes=d_store_tx,
            operation_working_set_bytes=operation_working_set,
            capacity_bytes=l2_capacity,
            dtype_bytes=dtype_bytes,
            sector_size_bytes=sector_size,
            line_size_bytes=line_size,
            order=order,
        )
        capacity_estimate = reuse_distance_estimate.capacity_estimate
        hbm_read = (
            capacity_estimate.a_dram_read_bytes
            + capacity_estimate.b_dram_read_bytes
            + c_read_tx
        )
        hbm_write = d_store_tx

    if _mixed_operands(problem):
        stage_operand_bytes = _ceil_scalar_bytes(
            kernel.cta_m * kernel.cta_k * a_bytes + kernel.cta_k * kernel.cta_n * b_bytes)
    else:
        stage_operand_bytes = _ceil_scalar_bytes(
            (kernel.cta_m * kernel.cta_k + kernel.cta_k * kernel.cta_n) * dtype_bytes
        )
    smem_write = grid.cta_count * grid.k_stages * stage_operand_bytes
    smem_read = grid.cta_count * _smem_read_bytes_per_cta(problem, kernel, grid)
    if kernel.architecture == "sm90":
        from .sm90_gemm import instruction_counts
        smem_read = instruction_counts(problem, kernel).operand_read_bytes

        if not _settings.get("OPMODEL_SMEM_STAGING"):
            smem_write = 0
    sram_read = smem_read if "sram" in hardware.memory_levels else None

    sram_write = (smem_write if kernel.architecture == "sm90" else 0) if "sram" in hardware.memory_levels else None

    return TrafficAccounting(
        memory_access=MemoryAccess(
            hbm_read_bytes=hbm_read,
            hbm_write_bytes=hbm_write,
            l2_read_bytes=(l2_to_smem_bytes + c_read_tx) if l2_level is not None else None,
            l2_write_bytes=d_store_tx if l2_level is not None else None,
            sram_read_bytes=sram_read,
            sram_write_bytes=sram_write,
        ),
        a_logical_bytes=_ceil_scalar_bytes(problem.batch * problem.m * problem.k * a_bytes),
        b_logical_bytes=_ceil_scalar_bytes(
            (problem.batch if problem.weight_is_batched else 1)
            * problem.k
            * problem.n
            * b_bytes
        ),
        c_read_logical_bytes=_ceil_scalar_bytes(
            problem.batch * problem.m * problem.n * out_dtype_bytes
        )
        if problem.epilogue_reads_c
        else 0,
        d_store_logical_bytes=_ceil_scalar_bytes(
            problem.batch * problem.m * problem.n * out_dtype_bytes
        ),
        a_l2_requested_bytes=a_requested_tx,
        b_l2_requested_bytes=b_requested_tx,
        a_dram_unique_bytes=a_unique_dram_tx,
        b_dram_unique_bytes=b_unique_dram_tx,
        a_dram_read_bytes=capacity_estimate.a_dram_read_bytes,
        b_dram_read_bytes=capacity_estimate.b_dram_read_bytes,
        c_read_transaction_bytes=c_read_tx,
        d_store_transaction_bytes=d_store_tx,
        l2_requested_bytes=l2_to_smem_bytes or 0,
        l2_capacity_bytes=l2_capacity,
        l2_operation_working_set_bytes=operation_working_set,
        l2_reuse_working_set_bytes=reuse_footprint.peak_working_set_bytes,
        l2_weighted_reuse_working_set_bytes=reuse_footprint.weighted_working_set_bytes,
        l2_capacity_miss_fraction=capacity_estimate.miss_fraction,
        l2_k_window_stages=reuse_footprint.k_window_stages,
        l2_k_window_count=reuse_footprint.k_window_count,
        l2_full_wave_full_window_bytes=reuse_footprint.full_wave_full_window_bytes,
        l2_full_wave_tail_window_bytes=reuse_footprint.full_wave_tail_window_bytes,
        l2_last_wave_full_window_bytes=reuse_footprint.last_wave_full_window_bytes,
        l2_last_wave_tail_window_bytes=reuse_footprint.last_wave_tail_window_bytes,
        a_within_wave_repeated_bytes=(
            reuse_distance_estimate.a_within_wave_repeated_bytes
        ),
        a_cross_wave_repeated_bytes=reuse_distance_estimate.a_cross_wave_repeated_bytes,
        b_within_wave_repeated_bytes=(
            reuse_distance_estimate.b_within_wave_repeated_bytes
        ),
        b_cross_wave_repeated_bytes=reuse_distance_estimate.b_cross_wave_repeated_bytes,
        l2_within_wave_miss_fraction=(
            reuse_distance_estimate.within_wave_miss_fraction
        ),
        a_cross_wave_reuse_distance_bytes=(
            reuse_distance_estimate.a_cross_wave_reuse_distance_bytes
        ),
        b_cross_wave_reuse_distance_bytes=(
            reuse_distance_estimate.b_cross_wave_reuse_distance_bytes
        ),
        a_cross_wave_miss_fraction=(
            reuse_distance_estimate.a_cross_wave_miss_fraction
        ),
        b_cross_wave_miss_fraction=(
            reuse_distance_estimate.b_cross_wave_miss_fraction
        ),
        dram_capacity_miss_bytes=(
            capacity_estimate.a_capacity_miss_bytes
            + capacity_estimate.b_capacity_miss_bytes
        ),
        dram_unique_bytes=(
            a_unique_dram_tx + b_unique_dram_tx + c_read_tx + hbm_write
        ),
        dram_total_bytes=hbm_read + hbm_write,
        smem_read_bytes=smem_read,
        smem_write_bytes=smem_write if kernel.architecture == "sm90" else 0,
        sector_size_bytes=sector_size,
        line_size_bytes=line_size,
    )

def _occupancy(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    hardware: HardwareSpec,
    warnings: list[str],
) -> OccupancyResult:
    num_sms = hardware.compute.num_sms or 1
    if hardware.compute.num_sms is None:
        warnings.append("num_sms_default_1")

    registers_per_cta = (0 if kernel.architecture == "sm90" else
                         _allocated_registers_per_cta(kernel.registers_per_thread, kernel.threads_per_cta, hardware))
    if hardware.compute.registers_per_sm is not None and registers_per_cta > hardware.compute.registers_per_sm:
        raise ValueError("GEMM CTA exceeds register capacity")
    shared_memory_bytes_per_cta = _shared_memory_bytes_per_cta(problem, kernel)
    for capacity in (hardware.compute.shared_memory_bytes_per_sm,
                     hardware.compute.shared_memory_bytes_per_block):
        if capacity is not None and shared_memory_bytes_per_cta > capacity:
            raise ValueError("GEMM CTA exceeds shared memory capacity")

    if kernel.architecture == "sm90":
        resident = 1
        limiting_factors = ("approved_sm90_1_cta_per_sm",)
    elif kernel.max_concurrent_ctas_per_sm is not None:
        limits = {"kernel_observed_max_concurrent_block": kernel.max_concurrent_ctas_per_sm}
        for name, capacity, required in (
            ("cta", hardware.compute.max_ctas_per_sm, 1),
            ("warp", hardware.compute.max_warps_per_sm, kernel.warps_per_cta),
            ("register", hardware.compute.registers_per_sm, registers_per_cta),
            ("shared_memory", hardware.compute.shared_memory_bytes_per_sm, shared_memory_bytes_per_cta),
        ):
            if capacity is not None:
                limits[name] = capacity // max(1, required)
        resident = min(limits.values())
        if resident < 1:
            raise ValueError("hardware resources cannot hold a GEMM CTA")
        limiting_factors = tuple(name for name, value in limits.items() if value == resident)
    else:
        max_ctas_limit = hardware.compute.max_ctas_per_sm or 1
        if hardware.compute.max_ctas_per_sm is None:
            warnings.append("max_ctas_per_sm_default_1")
        max_warps_limit = max(1, (hardware.compute.max_warps_per_sm or kernel.warps_per_cta) // kernel.warps_per_cta)
        if hardware.compute.max_warps_per_sm is None:
            warnings.append("max_warps_per_sm_default_kernel_warps")
        if hardware.compute.registers_per_sm is None:
            register_limit = max_ctas_limit
            warnings.append("registers_per_sm_absent")
        else:
            register_limit = max(1, hardware.compute.registers_per_sm // max(1, registers_per_cta))
        if hardware.compute.shared_memory_bytes_per_sm is None:
            smem_limit = max_ctas_limit
            warnings.append("shared_memory_bytes_per_sm_absent")
        else:
            smem_limit = max(
                1,
                hardware.compute.shared_memory_bytes_per_sm
                // max(1, shared_memory_bytes_per_cta),
            )
        limits = {
            "cta": max_ctas_limit,
            "warp": max_warps_limit,
            "register": register_limit,
            "shared_memory": smem_limit,
        }
        resident = max(1, min(limits.values()))
        min_limit = min(limits.values())
        limiting_factors = tuple(name for name, value in limits.items() if value == min_limit)
    ctas_per_wave = num_sms * resident
    wave_count = max(1, _ceil_div(grid.cta_count, ctas_per_wave))
    final_wave_ctas = grid.cta_count - (wave_count - 1) * ctas_per_wave
    tail_efficiency = final_wave_ctas / ctas_per_wave if ctas_per_wave else 1.0
    lazy_ctas_per_sm, busy_remainder = divmod(final_wave_ctas, num_sms)
    busy_ctas_per_sm = lazy_ctas_per_sm + (1 if busy_remainder else 0)
    busy_sms = busy_remainder if busy_remainder else (num_sms if final_wave_ctas else 0)
    lazy_sms = num_sms - busy_sms
    return OccupancyResult(
        num_sms=num_sms,
        resident_ctas_per_sm=resident,
        total_resident_ctas=ctas_per_wave,
        ctas_per_wave=ctas_per_wave,
        wave_count=wave_count,
        full_wave_ctas=ctas_per_wave,
        last_wave_ctas=final_wave_ctas,
        last_wave_busy_sms=busy_sms,
        last_wave_lazy_sms=lazy_sms,
        last_wave_busy_ctas_per_sm=busy_ctas_per_sm,
        last_wave_lazy_ctas_per_sm=lazy_ctas_per_sm,
        tail_efficiency=tail_efficiency,
        limiting_factors=limiting_factors,
    )

_TIMELINE_MEMO = SettingsMemo(256)

def _timeline(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
    epilogue_memory=None,
) -> TimelineResult:
    held = (problem, kernel, grid, traffic, occupancy, hardware, epilogue_memory)

    def compute():
        appended: list[str] = []
        return held, _timeline_uncached(problem, kernel, grid, traffic, occupancy, hardware, clock_hz, appended,
                                        epilogue_memory), tuple(appended)

    stored, timeline, appended = _TIMELINE_MEMO.get_or_compute(
        (*map(id, held), type(clock_hz), clock_hz), compute)
    for mine, theirs in zip(held, stored):
        if mine is not theirs:
            return _timeline_uncached(problem, kernel, grid, traffic, occupancy, hardware, clock_hz, warnings,
                                      epilogue_memory)
    warnings.extend(appended)
    return timeline

def _timeline_uncached(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    hardware: HardwareSpec,
    clock_hz: float,
    warnings: list[str],
    epilogue_memory=None,
) -> TimelineResult:
    peak_compute_flops_per_cycle = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_hbm_bw_per_cycle = hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz
    l2_level = hardware.memory_levels.get("l2")

    l2_timed = l2_level is not None and not _l2_bypass(hardware)
    peak_l2_bw_per_cycle = (
        l2_level.bandwidth_bytes_per_s / clock_hz if l2_timed else 0.0
    )
    peak_smem_bw_per_cycle = _smem_bandwidth_per_cycle(hardware, clock_hz, warnings)

    shared_latency = _shared_latency_cycles(hardware, clock_hz, warnings)
    l2_latency = _memory_latency_cycles(l2_level, clock_hz, "l2")
    dram_latency = _memory_latency_cycles(
        hardware.memory_levels["hbm"], clock_hz, "hbm"
    )
    tensor_latency = _tensor_latency_cycles(kernel, hardware, warnings)

    groups_k = _k_groups(kernel.cta_k, kernel)
    last_stage_k = (problem.k - 1) % kernel.cta_k + 1 if problem.k else 0
    last_stage_groups_k = _k_groups(last_stage_k, kernel)
    memory_pipeline_groups = _ceil_div(grid.k_stages, kernel.pipeline_stages)
    last_memory_pipeline_stages = (grid.k_stages - 1) % kernel.pipeline_stages + 1

    load_stage_count = max(1, grid.cta_count * grid.k_stages)
    avg_l2_load_bytes_per_cta_stage = (
        (traffic.a_l2_requested_bytes + traffic.b_l2_requested_bytes) / load_stage_count
        if l2_timed
        else 0.0
    )
    avg_dram_load_bytes_per_cta_stage = (
        (traffic.a_dram_read_bytes + traffic.b_dram_read_bytes) / load_stage_count
    )

    full_wave = _wave_pipeline(
        epilogue_memory=epilogue_memory,
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        active_ctas=occupancy.full_wave_ctas,
        busy_sms=occupancy.num_sms,
        lazy_sms=0,
        busy_ctas_per_sm=occupancy.resident_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.resident_ctas_per_sm,
        peak_compute_flops_per_cycle=peak_compute_flops_per_cycle,
        peak_smem_bw_per_cycle=peak_smem_bw_per_cycle,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        shared_latency_cycles=shared_latency,
        tensor_latency_cycles=tensor_latency,
        l2_latency_cycles=l2_latency,
        dram_latency_cycles=dram_latency,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )
    last_wave = _wave_pipeline(
        epilogue_memory=epilogue_memory,
        problem=problem,
        kernel=kernel,
        grid=grid,
        traffic=traffic,
        active_ctas=occupancy.last_wave_ctas,
        busy_sms=occupancy.last_wave_busy_sms,
        lazy_sms=occupancy.last_wave_lazy_sms,
        busy_ctas_per_sm=occupancy.last_wave_busy_ctas_per_sm,
        lazy_ctas_per_sm=occupancy.last_wave_lazy_ctas_per_sm,
        peak_compute_flops_per_cycle=peak_compute_flops_per_cycle,
        peak_smem_bw_per_cycle=peak_smem_bw_per_cycle,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        shared_latency_cycles=shared_latency,
        tensor_latency_cycles=tensor_latency,
        l2_latency_cycles=l2_latency,
        dram_latency_cycles=dram_latency,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
    )

    full_wave_count = max(0, occupancy.wave_count - 1)
    kernel_cycles = full_wave.total_cycles * full_wave_count + last_wave.total_cycles
    prologue_cycles = full_wave.start_cycles * full_wave_count + last_wave.start_cycles
    work_cycles = full_wave.work_cycles * full_wave_count + last_wave.work_cycles
    epilogue_cycles = full_wave.end_cycles * full_wave_count + last_wave.end_cycles
    cta_cycles = full_wave.total_cycles / max(1, occupancy.resident_ctas_per_sm)
    stage_cycles = full_wave.work_cycles / max(1, grid.k_stages)

    compute_active = grid.issued_flops / max(peak_compute_flops_per_cycle, 1.0e-12)
    epilogue_smem_bytes = (
        full_wave.epilogue_smem_bytes * full_wave_count + last_wave.epilogue_smem_bytes
    )
    epilogue_l2_bytes = (
        full_wave.epilogue_l2_bytes * full_wave_count + last_wave.epilogue_l2_bytes
    )
    epilogue_dram_bytes = (
        full_wave.epilogue_dram_bytes * full_wave_count + last_wave.epilogue_dram_bytes
    )
    smem_active = (traffic.smem_total_bytes + epilogue_smem_bytes) / max(
        peak_smem_bw_per_cycle, 1.0e-12
    )
    l2_active = (
        traffic.l2_requested_bytes / max(peak_l2_bw_per_cycle, 1.0e-12)
        if l2_timed
        else 0.0
    )
    dram_active = traffic.dram_total_bytes / max(peak_hbm_bw_per_cycle, 1.0e-12)
    compute_issue_util = _clamp01(
        full_wave.math_issue_group_cycles / max(full_wave.math_group_cycles, 1.0e-12)
    )
    mma_ilp_efficiency = _clamp01(
        full_wave.math_issue_group_cycles / max(full_wave.math_latency_group_cycles, 1.0e-12)
    )
    compute_latency_util = mma_ilp_efficiency
    compute_active_util = _clamp01(compute_active / max(kernel_cycles, 1.0e-12))
    smem_util = _clamp01(smem_active / max(kernel_cycles, 1.0e-12))
    l2_util = _clamp01(l2_active / max(kernel_cycles, 1.0e-12))
    dram_util = _clamp01(dram_active / max(kernel_cycles, 1.0e-12))
    compute_smem_overlap = _overlap_ratio(
        full_wave.math_group_cycles, full_wave.smem_group_cycles
    )
    compute_l2_overlap = (
        min(1.0, full_wave.work_cycles / full_wave.l2_full_stage_cycles)
        if full_wave.l2_full_stage_cycles > 0.0
        else 0.0
    )
    compute_dram_overlap = (
        min(1.0, full_wave.work_cycles / full_wave.dram_full_stage_cycles)
        if full_wave.dram_full_stage_cycles > 0.0
        else 0.0
    )
    return TimelineResult(
        kernel_cycles=kernel_cycles,
        cta_cycles=cta_cycles,
        full_wave=full_wave,
        last_wave=last_wave,
        prologue_cycles=prologue_cycles,
        work_cycles=work_cycles,
        stage_cycles=stage_cycles,
        epilogue_cycles=epilogue_cycles,
        compute_stage_cycles=full_wave.sm_stage_cycles,
        mma_issue_cycles=full_wave.math_issue_group_cycles,
        mma_dependency_penalty_cycles=max(
            0.0, full_wave.math_group_cycles - full_wave.math_issue_group_cycles
        ),
        mma_ilp_efficiency=mma_ilp_efficiency,
        smem_stage_cycles=full_wave.smem_group_cycles,
        global_load_issue_cycles=full_wave.memory_full_stage_cycles,
        l2_service_cycles=full_wave.l2_full_stage_cycles,
        dram_service_cycles=full_wave.dram_full_stage_cycles,
        exposed_l2_cycles=full_wave.exposed_l2_cycles,
        exposed_dram_cycles=full_wave.exposed_dram_cycles,
        compute_active_cycles=compute_active,
        smem_active_cycles=smem_active,
        l2_active_cycles=l2_active,
        dram_active_cycles=dram_active,
        epilogue_smem_bytes=epilogue_smem_bytes,
        epilogue_l2_bytes=epilogue_l2_bytes,
        epilogue_dram_bytes=epilogue_dram_bytes,
        groups_k=groups_k,
        last_stage_groups_k=last_stage_groups_k,
        last_stage_k=last_stage_k,
        memory_pipeline_groups=memory_pipeline_groups,
        last_memory_pipeline_stages=last_memory_pipeline_stages,
        compute_issue_utilization=compute_issue_util,
        compute_latency_utilization=compute_latency_util,
        compute_active_utilization=compute_active_util,
        smem_utilization=smem_util,
        l2_utilization=l2_util,
        dram_utilization=dram_util,
        compute_smem_overlap=compute_smem_overlap,
        compute_l2_overlap=compute_l2_overlap,
        compute_dram_overlap=compute_dram_overlap,
    )

def _wave_pipeline(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    active_ctas: int,
    busy_sms: int,
    lazy_sms: int,
    busy_ctas_per_sm: int,
    lazy_ctas_per_sm: int,
    peak_compute_flops_per_cycle: float,
    peak_smem_bw_per_cycle: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
    shared_latency_cycles: float,
    tensor_latency_cycles: float,
    l2_latency_cycles: float,
    dram_latency_cycles: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    groups_k: int,
    last_stage_groups_k: int,
    last_stage_k: int,
    memory_pipeline_groups: int,
    last_memory_pipeline_stages: int,
    epilogue_memory=None,
) -> WavePipelineResult:
    if active_ctas <= 0:
        return WavePipelineResult(
            active_ctas=0,
            busy_sms=0,
            lazy_sms=busy_sms + lazy_sms,
            busy_ctas_per_sm=0,
            lazy_ctas_per_sm=0,
            start_cycles=0.0,
            work_cycles=0.0,
            end_cycles=0.0,
            total_cycles=0.0,
            sm_stage_cycles=0.0,
            sm_last_stage_cycles=0.0,
            smem_group_cycles=0.0,
            math_group_cycles=0.0,
            math_issue_group_cycles=0.0,
            math_latency_group_cycles=0.0,
            memory_full_stage_cycles=0.0,
            memory_last_stage_cycles=0.0,
            l2_full_stage_cycles=0.0,
            l2_last_stage_cycles=0.0,
            dram_full_stage_cycles=0.0,
            dram_last_stage_cycles=0.0,
            epilogue_global_cycles=0.0,
            epilogue_smem_cycles=0.0,
            slice_k_extra_cycles=0.0,
            epilogue_smem_bytes=0.0,
            epilogue_l2_bytes=0.0,
            epilogue_dram_bytes=0.0,
            l2_epilogue_cycles=0.0,
            dram_epilogue_cycles=0.0,
            exposed_l2_cycles=0.0,
            exposed_dram_cycles=0.0,
        )

    total_sms = max(1, busy_sms + lazy_sms)
    per_sm_smem_bw = peak_smem_bw_per_cycle / total_sms
    per_smsp_compute = peak_compute_flops_per_cycle / total_sms / 4.0
    active_warps = max(0, busy_ctas_per_sm) * kernel.warps_per_cta

    effective_warp_m = min(kernel.warp_m, kernel.cta_m)
    effective_warp_n = min(kernel.warp_n, kernel.cta_n)
    effective_warp_k = _effective_warp_k(kernel)
    dtype_bytes = dtype_nbytes(problem.input_dtype)
    if _mixed_operands(problem):
        a_b, b_b = _operand_bytes(problem, dtype_bytes)
        warptile_smem_to_reg_bytes = effective_warp_m * effective_warp_k * a_b + effective_warp_n * effective_warp_k * b_b
    else:
        warptile_smem_to_reg_bytes = (
            (effective_warp_m + effective_warp_n) * effective_warp_k * dtype_bytes
        )
    smsp_warps = _ceil_div(active_warps, 4) if active_warps else 0

    smem_group_cycles = (
        max(active_warps * warptile_smem_to_reg_bytes / max(per_sm_smem_bw, 1.0e-12),
            shared_latency_cycles)
        if active_warps > 0
        else 0.0
    )

    concurrent_mma = (
        smsp_warps
        * _ceil_div(effective_warp_m, kernel.mma_m)
        * _ceil_div(effective_warp_n, kernel.mma_n)
    )
    mma_k_iters = max(1, _ceil_div(effective_warp_k, kernel.mma_k))
    math_issue_group_cycles = (
        mma_k_iters
        * concurrent_mma
        * (2 * kernel.mma_m * kernel.mma_n * kernel.mma_k)
        / max(per_smsp_compute, 1.0e-12)
        if concurrent_mma > 0
        else 0.0
    )
    if concurrent_mma > 0 and (kernel.mma_m, kernel.mma_n, kernel.mma_k) == (1, 1, 1):
        math_latency_group_cycles = math_issue_group_cycles
    else:
        math_latency_group_cycles = mma_k_iters * tensor_latency_cycles if concurrent_mma > 0 else 0.0
    math_group_cycles = max(math_issue_group_cycles, math_latency_group_cycles)
    if kernel.architecture == "sm90":
        from .sm90_gemm import tile_instruction_counts
        tile = tile_instruction_counts(min(problem.m, kernel.cta_m),
            min(problem.n, kernel.cta_n), effective_warp_k,
            swapped=problem.m < 64 and problem.m <= problem.n,
            input_bytes=dtype_bytes, output_bytes=dtype_nbytes(problem.output_dtype),
            nominal_n=(kernel.cta_m if problem.m < 64 and problem.m <= problem.n else kernel.cta_n))
        per_sm_compute = per_smsp_compute * 4
        smem_group_cycles = max(
            busy_ctas_per_sm * tile.operand_read_bytes / max(per_sm_smem_bw, 1.0e-12),
            shared_latency_cycles)
        math_issue_group_cycles = busy_ctas_per_sm * tile.issued_flops / max(per_sm_compute, 1.0e-12)
        math_latency_group_cycles = mma_k_iters * tensor_latency_cycles
        math_group_cycles = max(math_issue_group_cycles, math_latency_group_cycles)
    per_group_cycles = max(smem_group_cycles, math_group_cycles)
    math_tail_cycles = (
        math_latency_group_cycles * (1.0 + 1.0 / max(1, kernel.pipeline_stages - 1))
        if math_group_cycles > smem_group_cycles
        else 0.0
    )
    sm_stage_cycles = groups_k * per_group_cycles + math_tail_cycles
    sm_last_stage_cycles = last_stage_groups_k * per_group_cycles + math_tail_cycles

    full_memory_stages = min(kernel.pipeline_stages, grid.k_stages)
    last_stage_fraction = last_stage_k / max(1, kernel.cta_k)
    last_memory_stage_equiv = max(
        last_stage_fraction,
        float(max(0, last_memory_pipeline_stages - 1)) + last_stage_fraction,
    )
    (
        memory_full_stage_cycles,
        l2_full_stage_cycles,
        dram_full_stage_cycles,
    ) = _memory_pipeline_stage_cycles(
        active_ctas=active_ctas,
        stage_equivalent=float(full_memory_stages),
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        l2_latency_cycles=l2_latency_cycles,
        dram_latency_cycles=dram_latency_cycles,
    )
    (
        memory_full_steady_cycles,
        l2_full_steady_cycles,
        dram_full_steady_cycles,
    ) = _memory_pipeline_stage_bandwidth_cycles(
        active_ctas=active_ctas,
        stage_equivalent=float(full_memory_stages),
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
    )
    (
        memory_last_stage_cycles,
        l2_last_stage_cycles,
        dram_last_stage_cycles,
    ) = _memory_pipeline_stage_cycles(
        active_ctas=active_ctas,
        stage_equivalent=last_memory_stage_equiv,
        avg_l2_load_bytes_per_cta_stage=avg_l2_load_bytes_per_cta_stage,
        avg_dram_load_bytes_per_cta_stage=avg_dram_load_bytes_per_cta_stage,
        peak_l2_bw_per_cycle=peak_l2_bw_per_cycle,
        peak_hbm_bw_per_cycle=peak_hbm_bw_per_cycle,
        l2_latency_cycles=l2_latency_cycles,
        dram_latency_cycles=dram_latency_cycles,
    )

    sm_path_cycles = (
        sm_stage_cycles * max(0, grid.k_stages - 1) + sm_last_stage_cycles
        if grid.k_stages > 1
        else sm_last_stage_cycles
    )
    memory_path_cycles = sm_stage_cycles + sm_last_stage_cycles + (
        memory_full_steady_cycles * max(0, memory_pipeline_groups - 1)
        + memory_last_stage_cycles
    )
    work_cycles = max(sm_path_cycles, memory_path_cycles) if grid.k_stages > 1 else sm_path_cycles
    start_cycles = (
        memory_full_stage_cycles
        if grid.k_stages >= kernel.pipeline_stages
        else memory_last_stage_cycles
    )

    wave_fraction = active_ctas / max(1, grid.cta_count)
    l2_epilogue_bytes = 0.0
    dram_epilogue_bytes = wave_fraction * (
        traffic.d_store_transaction_bytes + traffic.c_read_transaction_bytes
    )
    if epilogue_memory is not None:
        l2_epilogue_bytes, dram_epilogue_bytes = (
            wave_fraction * value for value in epilogue_memory)
    output_dtype_bytes = dtype_nbytes(problem.output_dtype)
    warptile_reg_to_smem_bytes = effective_warp_m * effective_warp_n * output_dtype_bytes
    epilogue_smem_cycles = (
        max(active_warps * warptile_reg_to_smem_bytes / max(per_sm_smem_bw, 1.0e-12),
            shared_latency_cycles)
        if active_warps > 0
        else 0.0
    )
    slice_k_ld_cycles = epilogue_smem_cycles if kernel.slice_k else 0.0
    slice_k_st_cycles = (
        max(
            active_warps
            * warptile_reg_to_smem_bytes
            / max(per_sm_smem_bw * kernel.num_warp_tile_k, 1.0e-12),
            shared_latency_cycles,
        )
        if kernel.slice_k and active_warps > 0
        else 0.0
    )
    slice_k_extra_cycles = slice_k_ld_cycles + slice_k_st_cycles
    epilogue_smem_bytes = wave_fraction * traffic.d_store_transaction_bytes
    if kernel.slice_k:
        epilogue_smem_bytes += wave_fraction * traffic.d_store_transaction_bytes * (
            1.0 + 1.0 / max(1, kernel.num_warp_tile_k)
        )
    from .effective_roofline import cutlass_epilogue_smem_bytes
    _cutlass_epi = cutlass_epilogue_smem_bytes(kernel, grid, problem)
    if _cutlass_epi is not None:
        epilogue_smem_bytes = wave_fraction * _cutlass_epi
    l2_epilogue_cycles = _service_cycles(
        l2_epilogue_bytes, peak_l2_bw_per_cycle, l2_latency_cycles
    )
    dram_epilogue_cycles = _service_cycles(
        dram_epilogue_bytes, peak_hbm_bw_per_cycle, dram_latency_cycles
    )
    epilogue_global_cycles = l2_epilogue_cycles + dram_epilogue_cycles
    end_cycles = epilogue_smem_cycles + slice_k_extra_cycles + epilogue_global_cycles

    l2_pipeline_cycles = (
        l2_full_steady_cycles * max(0, memory_pipeline_groups - 1) + l2_last_stage_cycles
    )
    dram_pipeline_cycles = (
        dram_full_steady_cycles * max(0, memory_pipeline_groups - 1) + dram_last_stage_cycles
    )
    exposed_l2_cycles = max(0.0, l2_pipeline_cycles - sm_path_cycles)
    exposed_dram_cycles = max(0.0, dram_pipeline_cycles - sm_path_cycles)

    return WavePipelineResult(
        active_ctas=active_ctas,
        busy_sms=busy_sms,
        lazy_sms=lazy_sms,
        busy_ctas_per_sm=busy_ctas_per_sm,
        lazy_ctas_per_sm=lazy_ctas_per_sm,
        start_cycles=start_cycles,
        work_cycles=work_cycles,
        end_cycles=end_cycles,
        total_cycles=start_cycles + work_cycles + end_cycles,
        sm_stage_cycles=sm_stage_cycles,
        sm_last_stage_cycles=sm_last_stage_cycles,
        smem_group_cycles=smem_group_cycles,
        math_group_cycles=math_group_cycles,
        math_issue_group_cycles=math_issue_group_cycles,
        math_latency_group_cycles=math_latency_group_cycles,
        memory_full_stage_cycles=memory_full_stage_cycles,
        memory_last_stage_cycles=memory_last_stage_cycles,
        l2_full_stage_cycles=l2_full_stage_cycles,
        l2_last_stage_cycles=l2_last_stage_cycles,
        dram_full_stage_cycles=dram_full_stage_cycles,
        dram_last_stage_cycles=dram_last_stage_cycles,
        epilogue_global_cycles=epilogue_global_cycles,
        epilogue_smem_cycles=epilogue_smem_cycles,
        slice_k_extra_cycles=slice_k_extra_cycles,
        epilogue_smem_bytes=epilogue_smem_bytes,
        epilogue_l2_bytes=l2_epilogue_bytes,
        epilogue_dram_bytes=dram_epilogue_bytes,
        l2_epilogue_cycles=l2_epilogue_cycles,
        dram_epilogue_cycles=dram_epilogue_cycles,
        exposed_l2_cycles=exposed_l2_cycles,
        exposed_dram_cycles=exposed_dram_cycles,
    )

def _memory_pipeline_stage_cycles(
    *,
    active_ctas: int,
    stage_equivalent: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
    l2_latency_cycles: float,
    dram_latency_cycles: float,
) -> tuple[float, float, float]:
    l2_bytes = active_ctas * stage_equivalent * avg_l2_load_bytes_per_cta_stage
    dram_bytes = active_ctas * stage_equivalent * avg_dram_load_bytes_per_cta_stage
    l2_cycles = _service_cycles(l2_bytes, peak_l2_bw_per_cycle, l2_latency_cycles)
    dram_cycles = _service_cycles(dram_bytes, peak_hbm_bw_per_cycle, dram_latency_cycles)
    return max(l2_cycles, dram_cycles), l2_cycles, dram_cycles

def _memory_pipeline_stage_bandwidth_cycles(
    *,
    active_ctas: int,
    stage_equivalent: float,
    avg_l2_load_bytes_per_cta_stage: float,
    avg_dram_load_bytes_per_cta_stage: float,
    peak_l2_bw_per_cycle: float,
    peak_hbm_bw_per_cycle: float,
) -> tuple[float, float, float]:
    l2_bytes = active_ctas * stage_equivalent * avg_l2_load_bytes_per_cta_stage
    dram_bytes = active_ctas * stage_equivalent * avg_dram_load_bytes_per_cta_stage
    l2_cycles = (
        l2_bytes / peak_l2_bw_per_cycle
        if l2_bytes > 0.0 and peak_l2_bw_per_cycle > 0.0
        else 0.0
    )
    dram_cycles = (
        dram_bytes / peak_hbm_bw_per_cycle
        if dram_bytes > 0.0 and peak_hbm_bw_per_cycle > 0.0
        else 0.0
    )
    return max(l2_cycles, dram_cycles), l2_cycles, dram_cycles

def _service_cycles(bytes_count: float, bandwidth_bytes_per_cycle: float, latency_cycles: float) -> float:
    if bytes_count <= 0.0 or bandwidth_bytes_per_cycle <= 0.0:
        return 0.0
    return max(bytes_count / bandwidth_bytes_per_cycle, latency_cycles)

def _effective_warp_k(kernel: GemmKernelSpec) -> int:
    if kernel.slice_k:
        return max(1, _ceil_div(kernel.cta_k, kernel.num_warp_tile_k))
    return min(kernel.warp_k, kernel.cta_k)

def _k_groups(k_extent: int, kernel: GemmKernelSpec) -> int:
    if k_extent <= 0:
        return 0
    k_per_warp_tile = _ceil_div(k_extent, kernel.num_warp_tile_k)
    return max(1, _ceil_div(k_per_warp_tile, kernel.warp_k))

def _classify_bottlenecks(
    problem: GemmProblemSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    timeline: TimelineResult,
    hardware: HardwareSpec,
    clock_hz: float,
    fixed_overhead_cycles: float,
) -> tuple[str, tuple[str, ...]]:
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    fixed_overhead_fraction = fixed_overhead_cycles / max(total_device_cycles, 1.0e-12)
    metrics = _roofline_metrics(
        problem,
        grid,
        traffic,
        timeline,
        hardware,
        clock_hz,
        total_device_cycles,
    )
    ceiling_util = metrics["ceiling_utilization"]
    labels = {
        "compute": "compute_roof_limited",
        "smem": "smem_bandwidth_limited",
        "l2": "l2_bandwidth_limited",
        "dram": "dram_bandwidth_limited",
    }
    candidates = {
        labels[name]: value
        for name, value in ceiling_util.items()
        if value is not None and math.isfinite(value)
    }
    epilogue_ratio = timeline.epilogue_cycles / max(timeline.kernel_cycles, 1.0e-12)
    if epilogue_ratio > 0.2:
        primary = "epilogue_limited"
    elif candidates:
        primary = max(candidates.items(), key=lambda item: item[1])[0]
    else:
        primary = "compute_roof_limited"
    roofline_primary = primary

    secondary: list[str] = []
    if fixed_overhead_fraction >= 0.5:
        primary = "fixed_overhead_limited"
        secondary.append(roofline_primary)
    elif fixed_overhead_fraction >= 0.2:
        secondary.append("fixed_overhead_limited")
    if timeline.compute_latency_utilization < 0.8:
        secondary.append("compute_latency_limited")
    if timeline.mma_ilp_efficiency < 0.8:
        secondary.append("low_mma_ilp")
    if timeline.compute_smem_overlap < 0.7:
        secondary.append("poor_compute_smem_overlap")
    if timeline.exposed_l2_cycles > 0.1 * max(timeline.full_wave.work_cycles, 1.0e-12):
        secondary.append("l2_latency_exposed")
    if timeline.exposed_dram_cycles > 0.1 * max(timeline.full_wave.work_cycles, 1.0e-12):
        secondary.append("dram_latency_exposed")
    if 0.0 < timeline.compute_l2_overlap < 0.7:
        secondary.append("poor_compute_l2_overlap")
    if 0.0 < timeline.compute_dram_overlap < 0.7:
        secondary.append("poor_compute_dram_overlap")
    if occupancy.wave_count <= 1 or occupancy.tail_efficiency < 0.6:
        secondary.append("insufficient_cta_waves")
    if grid.tile_efficiency < 0.9:
        secondary.append("edge_tile_predication_loss")
    if problem.epilogue_reads_c or epilogue_ratio > 0.2:
        secondary.append("epilogue_limited")
    for name, value in ceiling_util.items():
        if value is not None and value >= 0.8:
            secondary.append(f"{name}_ceiling_near_saturation")
    return primary, tuple(dict.fromkeys(item for item in secondary if item != primary))

def _global_factors(
    *,
    useful: float,
    issued: float,
    kernel_cycles: float,
    total_device_cycles: float,
) -> dict[str, float]:
    return {
        "useful_flop_efficiency": useful / max(issued, 1.0e-12),
        "kernel_scope_efficiency": kernel_cycles / max(total_device_cycles, 1.0e-12),
        "total_device_overhead_fraction": max(0.0, total_device_cycles - kernel_cycles)
        / max(total_device_cycles, 1.0e-12),
    }

def _phase_factors(timeline: TimelineResult) -> dict[str, float]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    return {
        "prologue_cycles": timeline.prologue_cycles,
        "work_cycles": timeline.work_cycles,
        "epilogue_cycles": timeline.epilogue_cycles,
        "prologue_fraction": timeline.prologue_cycles / kernel_cycles,
        "work_fraction": timeline.work_cycles / kernel_cycles,
        "epilogue_fraction": timeline.epilogue_cycles / kernel_cycles,
    }

def _compute_factors(
    *,
    useful: float,
    issued: float,
    achieved: float,
    peak_compute: float,
    timeline: TimelineResult,
) -> dict[str, float]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    math_group_cycles = max(
        timeline.mma_issue_cycles + timeline.mma_dependency_penalty_cycles,
        1.0e-12,
    )
    return {
        "raw_peak_flop_per_cycle": peak_compute,
        "achieved_flop_per_cycle": achieved,
        "compute_roof_utilization": achieved / max(peak_compute, 1.0e-12),
        "useful_flop_efficiency": useful / max(issued, 1.0e-12),
        "compute_active_cycles": timeline.compute_active_cycles,
        "compute_active_fraction": timeline.compute_active_cycles / kernel_cycles,
        "mma_issue_cycles": timeline.mma_issue_cycles,
        "mma_dependency_penalty_cycles": timeline.mma_dependency_penalty_cycles,
        "mma_issue_fraction_of_math_group": timeline.mma_issue_cycles
        / math_group_cycles,
        "mma_dependency_fraction_of_math_group": timeline.mma_dependency_penalty_cycles
        / math_group_cycles,
        "mma_ilp_efficiency": timeline.mma_ilp_efficiency,
        "compute_issue_utilization": timeline.compute_issue_utilization,
        "compute_latency_utilization": timeline.compute_latency_utilization,
    }

def _memory_factors(
    *,
    q_bytes: float,
    peak_bytes_per_cycle: float,
    active_cycles: float,
    utilization: float,
    overlap: float | None,
    exposed_cycles: float,
    epilogue_bytes: float,
    useful: float,
    achieved: float,
    kernel_cycles: float,
) -> dict[str, float | None]:
    kernel = max(kernel_cycles, 1.0e-12)
    if q_bytes <= 0.0 or peak_bytes_per_cycle <= 0.0:
        return {
            "arithmetic_intensity": None,
            "raw_bound": None,
            "achieved": achieved,
            "ceiling_utilization": None,
            "bytes": q_bytes,
            "active_cycles_at_peak": None,
            "active_fraction_of_kernel": None,
            "timeline_utilization": utilization,
            "exposed_cycles": exposed_cycles,
            "exposed_fraction_of_kernel": exposed_cycles / kernel,
            "hidden_fraction_of_service": None,
            "compute_overlap": overlap,
            "epilogue_bytes": epilogue_bytes,
            "epilogue_byte_fraction": None,
        }

    raw_bound = peak_bytes_per_cycle * useful / q_bytes
    return {
        "arithmetic_intensity": useful / q_bytes,
        "raw_bound": raw_bound,
        "achieved": achieved,
        "ceiling_utilization": achieved / max(raw_bound, 1.0e-12),
        "bytes": q_bytes,
        "active_cycles_at_peak": active_cycles,
        "active_fraction_of_kernel": active_cycles / kernel,
        "timeline_utilization": utilization,
        "exposed_cycles": exposed_cycles,
        "exposed_fraction_of_kernel": exposed_cycles / kernel,
        "hidden_fraction_of_service": max(
            0.0, 1.0 - exposed_cycles / max(active_cycles, 1.0e-12)
        ),
        "compute_overlap": overlap,
        "epilogue_bytes": epilogue_bytes,
        "epilogue_byte_fraction": epilogue_bytes / max(q_bytes, 1.0e-12),
    }

def _critical_path_factors(timeline: TimelineResult) -> dict[str, float]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    math_group_cycles = max(
        timeline.mma_issue_cycles + timeline.mma_dependency_penalty_cycles,
        1.0e-12,
    )
    return {
        "prologue_fraction": timeline.prologue_cycles / kernel_cycles,
        "work_fraction": timeline.work_cycles / kernel_cycles,
        "epilogue_fraction": timeline.epilogue_cycles / kernel_cycles,
        "exposed_l2_fraction": timeline.exposed_l2_cycles / kernel_cycles,
        "exposed_dram_fraction": timeline.exposed_dram_cycles / kernel_cycles,
        "max_exposed_memory_fraction": max(
            timeline.exposed_l2_cycles,
            timeline.exposed_dram_cycles,
        )
        / kernel_cycles,
        "mma_dependency_fraction_of_math_group": timeline.mma_dependency_penalty_cycles
        / math_group_cycles,
        "compute_inactive_fraction": max(0.0, 1.0 - timeline.compute_active_utilization),
    }

def _classify_attribution_bottleneck(
    timeline: TimelineResult,
) -> dict[str, float | str]:
    kernel_cycles = max(timeline.kernel_cycles, 1.0e-12)
    math_group_cycles = max(
        timeline.mma_issue_cycles + timeline.mma_dependency_penalty_cycles,
        1.0e-12,
    )
    scores = {
        "prologue_dominated": timeline.prologue_cycles / kernel_cycles,
        "epilogue_dominated": timeline.epilogue_cycles / kernel_cycles,
        "l2_exposed": timeline.exposed_l2_cycles / kernel_cycles,
        "dram_exposed": timeline.exposed_dram_cycles / kernel_cycles,
        "mma_dependency_limited": timeline.mma_dependency_penalty_cycles
        / math_group_cycles,
        "compute_underoccupied": max(0.0, 1.0 - timeline.compute_active_utilization),
    }
    primary = max(scores, key=scores.__getitem__)
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    secondary, secondary_score = (
        sorted_scores[1] if len(sorted_scores) > 1 else (primary, scores[primary])
    )
    return {
        "primary": primary,
        "primary_score": scores[primary],
        "secondary": secondary,
        "secondary_score": secondary_score,
        **scores,
    }

def _deprecated_effective_bounds(
    *,
    useful: float,
    q_smem: float,
    q_l2: float,
    q_dram: float,
    peak_compute: float,
    peak_smem: float,
    peak_l2: float,
    peak_dram: float,
    timeline: TimelineResult,
) -> dict[str, float | None]:
    return {
        "compute": peak_compute
        * timeline.compute_issue_utilization
        * timeline.compute_latency_utilization
        * timeline.compute_active_utilization,
        "smem": peak_smem * timeline.smem_utilization * (useful / q_smem)
        if q_smem
        else None,
        "l2": peak_l2 * timeline.l2_utilization * (useful / q_l2)
        if q_l2 and peak_l2
        else None,
        "dram": peak_dram * timeline.dram_utilization * (useful / q_dram)
        if q_dram
        else None,
    }

def _validate_roofline_attribution(metrics: Mapping[str, Any]) -> tuple[str, ...]:
    warnings: list[str] = []
    raw = metrics["raw_bounds"]
    achieved = metrics["achieved"]["kernel_scope"]
    for name, bound in raw.items():
        if bound is None or bound <= 0.0:
            continue
        utilization = achieved / bound
        if utilization > 1.05:
            warnings.append(
                f"{name} utilization exceeds raw bound by more than 5%; "
                "check traffic accounting, useful/issued FLOPs, or peak rates"
            )

    breakdown = metrics["factor_breakdown"]
    global_factors = breakdown["global"]
    useful_efficiency = global_factors["useful_flop_efficiency"]
    if useful_efficiency > 1.000001:
        warnings.append(
            f"useful FLOP efficiency is {useful_efficiency:.6g}, expected <= 1"
        )

    phase = breakdown["phase"]
    phase_sum = (
        phase["prologue_fraction"]
        + phase["work_fraction"]
        + phase["epilogue_fraction"]
    )
    if abs(phase_sum - 1.0) > 1.0e-6:
        warnings.append(f"phase fractions sum to {phase_sum:.6g}, expected 1")

    for name in ("smem", "l2", "dram"):
        factors = breakdown[name]
        active_fraction = factors["active_fraction_of_kernel"]
        timeline_utilization = factors["timeline_utilization"]
        if active_fraction is not None:
            expected = min(1.0, active_fraction)
            if abs(expected - timeline_utilization) > 1.0e-6:
                warnings.append(
                    f"{name} active fraction {active_fraction:.6g} does not match "
                    f"timeline utilization {timeline_utilization:.6g}"
                )
        active_cycles = factors["active_cycles_at_peak"]
        exposed_cycles = factors["exposed_cycles"]
        if (
            active_cycles is not None
            and exposed_cycles is not None
            and exposed_cycles > active_cycles + 1.0e-6
        ):
            warnings.append(
                f"{name} exposed cycles exceed active service cycles; "
                "check memory overlap accounting"
            )
    return tuple(warnings)

def _roofline_metrics(
    problem: GemmProblemSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    timeline: TimelineResult,
    hardware: HardwareSpec,
    clock_hz: float,
    total_device_cycles: float,
) -> dict[str, Any]:
    q_smem = traffic.smem_total_bytes + timeline.epilogue_smem_bytes
    q_l2 = traffic.l2_requested_bytes
    q_dram = traffic.dram_total_bytes
    useful = grid.useful_flops
    issued = grid.issued_flops
    achieved_kernel = useful / max(timeline.kernel_cycles, 1.0e-12)
    achieved_total_device = useful / max(total_device_cycles, 1.0e-12)
    peak_compute = hardware.compute.tensor_flops_per_s[problem.input_dtype] / clock_hz
    peak_smem = _smem_bandwidth_per_cycle(hardware, clock_hz, [])
    peak_l2 = (
        hardware.memory_levels["l2"].bandwidth_bytes_per_s / clock_hz
        if "l2" in hardware.memory_levels
        else 0.0
    )
    peak_dram = hardware.memory_levels["hbm"].bandwidth_bytes_per_s / clock_hz

    raw_bounds = {
        "compute": peak_compute,
        "smem": peak_smem * (useful / q_smem) if q_smem else None,
        "l2": peak_l2 * (useful / q_l2) if q_l2 and peak_l2 else None,
        "dram": peak_dram * (useful / q_dram) if q_dram else None,
    }
    ceiling_utilization = {
        name: achieved_kernel / value if value and value > 0.0 else None
        for name, value in raw_bounds.items()
    }
    total_ceiling_utilization = {
        name: achieved_total_device / value if value and value > 0.0 else None
        for name, value in raw_bounds.items()
    }
    smem_factors = _memory_factors(
        q_bytes=q_smem,
        peak_bytes_per_cycle=peak_smem,
        active_cycles=timeline.smem_active_cycles,
        utilization=timeline.smem_utilization,
        overlap=timeline.compute_smem_overlap,
        exposed_cycles=0.0,
        epilogue_bytes=timeline.epilogue_smem_bytes,
        useful=useful,
        achieved=achieved_kernel,
        kernel_cycles=timeline.kernel_cycles,
    )
    l2_factors = _memory_factors(
        q_bytes=q_l2,
        peak_bytes_per_cycle=peak_l2,
        active_cycles=timeline.l2_active_cycles,
        utilization=timeline.l2_utilization,
        overlap=timeline.compute_l2_overlap,
        exposed_cycles=timeline.exposed_l2_cycles,
        epilogue_bytes=timeline.epilogue_l2_bytes,
        useful=useful,
        achieved=achieved_kernel,
        kernel_cycles=timeline.kernel_cycles,
    )
    dram_factors = _memory_factors(
        q_bytes=q_dram,
        peak_bytes_per_cycle=peak_dram,
        active_cycles=timeline.dram_active_cycles,
        utilization=timeline.dram_utilization,
        overlap=timeline.compute_dram_overlap,
        exposed_cycles=timeline.exposed_dram_cycles,
        epilogue_bytes=timeline.epilogue_dram_bytes,
        useful=useful,
        achieved=achieved_kernel,
        kernel_cycles=timeline.kernel_cycles,
    )
    return {
        "raw_bounds": raw_bounds,
        "achieved": {
            "kernel_scope": achieved_kernel,
            "device_scope": achieved_total_device,
        },
        "ceiling_utilization": ceiling_utilization,
        "total_ceiling_utilization": total_ceiling_utilization,
        "factor_breakdown": {
            "global": _global_factors(
                useful=useful,
                issued=issued,
                kernel_cycles=timeline.kernel_cycles,
                total_device_cycles=total_device_cycles,
            ),
            "phase": _phase_factors(timeline),
            "compute": _compute_factors(
                useful=useful,
                issued=issued,
                achieved=achieved_kernel,
                peak_compute=peak_compute,
                timeline=timeline,
            ),
            "smem": smem_factors,
            "l2": l2_factors,
            "dram": dram_factors,
            "critical_path": _critical_path_factors(timeline),
            "bottleneck_classification": _classify_attribution_bottleneck(timeline),
        },
        "effective_bounds_deprecated": _deprecated_effective_bounds(
            useful=useful,
            q_smem=q_smem,
            q_l2=q_l2,
            q_dram=q_dram,
            peak_compute=peak_compute,
            peak_smem=peak_smem,
            peak_l2=peak_l2,
            peak_dram=peak_dram,
            timeline=timeline,
        ),
    }

def _diagnostics(
    *,
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    timeline: TimelineResult,
    bottlenecks: tuple[str, tuple[str, ...]],
    warnings: tuple[str, ...],
    clock_hz: float,
    latency_s: float,
    flops_per_cycle: float,
    tflops_per_s: float,
    hardware: HardwareSpec,
    fixed_overhead_cycles: float,
) -> dict[str, Any]:
    primary, secondary = bottlenecks
    total_device_cycles = timeline.kernel_cycles + fixed_overhead_cycles
    fixed_overhead_fraction = fixed_overhead_cycles / max(total_device_cycles, 1.0e-12)
    smem_load_per_cta_elements = _smem_load_per_cta_elements(problem, kernel, grid)
    smem_read_bytes_per_cta = _smem_read_bytes_per_cta(problem, kernel, grid)
    shared_memory_bytes_per_cta = _shared_memory_bytes_per_cta(problem, kernel)
    q_smem = traffic.smem_total_bytes + timeline.epilogue_smem_bytes
    q_l2 = traffic.l2_requested_bytes
    q_dram = traffic.dram_total_bytes
    useful = grid.useful_flops
    total_q = q_smem + q_l2 + q_dram
    roofline_metrics = _roofline_metrics(
        problem,
        grid,
        traffic,
        timeline,
        hardware,
        clock_hz,
        total_device_cycles,
    )
    attribution_warnings = tuple(
        f"roofline_attribution: {message}"
        for message in _validate_roofline_attribution(roofline_metrics)
    )
    all_warnings = warnings + attribution_warnings
    launch_diagnostics = {}
    order = _launch_order(kernel, grid)
    if order.padded:
        slots = problem.batch * _launch_slots_per_batch(grid, order)
        launch_diagnostics = {
            "launch_order": "cutlass_identity_swizzle",
            "launch_slots": slots,
            "valid_ctas": grid.cta_count,
            "nop_launch_slots": slots - grid.cta_count,
            "memory_launch_waves": _ceil_div(slots, occupancy.ctas_per_wave),
        }
    return {
        **launch_diagnostics,
        "problem": {
            "batch": problem.batch,
            "m": problem.m,
            "n": problem.n,
            "k": problem.k,
            "input_dtype": problem.input_dtype.value,
            "output_dtype": problem.output_dtype.value,
            "beta_zero": problem.beta_zero,
            "epilogue_reads_c": problem.epilogue_reads_c,
            "transpose_a": problem.transpose_a,
            "transpose_b": problem.transpose_b,
        },
        "kernel": {
            "cta_tile": {"m": kernel.cta_m, "n": kernel.cta_n, "k": kernel.cta_k},
            "warp_tile": {"m": kernel.warp_m, "n": kernel.warp_n, "k": kernel.warp_k},
            "num_warp_tile_k": kernel.num_warp_tile_k,
            "mma_shape": {"m": kernel.mma_m, "n": kernel.mma_n, "k": kernel.mma_k},
            "pipeline_stages": kernel.pipeline_stages,
            "warps_per_cta": kernel.warps_per_cta,
            "threads_per_cta": kernel.threads_per_cta,
            "registers_per_thread": kernel.registers_per_thread,
            "smem_load_per_cta_elements": smem_load_per_cta_elements,
            "smem_read_bytes_per_cta": smem_read_bytes_per_cta,
            "shared_memory_bytes_per_cta": shared_memory_bytes_per_cta,
            "max_concurrent_ctas_per_sm": kernel.max_concurrent_ctas_per_sm,
            "slice_k": kernel.slice_k,
        },
        "predicted_elapsed_cycles": total_device_cycles,
        "modeled_device_cycles": timeline.kernel_cycles,
        "device_fixed_overhead_cycles": fixed_overhead_cycles,
        "total_device_cycles": total_device_cycles,
        "device_fixed_overhead_s": fixed_overhead_cycles / clock_hz,
        "device_fixed_overhead_fraction": fixed_overhead_fraction,
        "predicted_flop_per_cycle": flops_per_cycle,
        "modeled_flop_per_cycle": (
            grid.useful_flops / timeline.kernel_cycles if timeline.kernel_cycles else 0.0
        ),
        "predicted_tflops_per_s": tflops_per_s,
        "predicted_latency_s": latency_s,
        "clock_hz": clock_hz,
        "cta_count": grid.cta_count,
        "cta_grid": {
            "m": grid.blocks_m,
            "n": grid.blocks_n,
            "k_stages": grid.k_stages,
        },
        "cta_waves": occupancy.wave_count,
        "resident_ctas_per_sm": occupancy.resident_ctas_per_sm,
        "ctas_per_wave": occupancy.ctas_per_wave,
        "tail_efficiency": occupancy.tail_efficiency,
        "occupancy_limiting_factors": occupancy.limiting_factors,
        "wave_shape": {
            "full_wave_ctas": occupancy.full_wave_ctas,
            "last_wave_ctas": occupancy.last_wave_ctas,
            "last_wave_busy_sms": occupancy.last_wave_busy_sms,
            "last_wave_lazy_sms": occupancy.last_wave_lazy_sms,
            "last_wave_busy_ctas_per_sm": occupancy.last_wave_busy_ctas_per_sm,
            "last_wave_lazy_ctas_per_sm": occupancy.last_wave_lazy_ctas_per_sm,
        },
        "useful_flops": grid.useful_flops,
        "issued_flops": grid.issued_flops,
        "tile_efficiency": grid.tile_efficiency,
        "logical_bytes": {
            "a": traffic.a_logical_bytes,
            "b": traffic.b_logical_bytes,
            "c_read": traffic.c_read_logical_bytes,
            "d_store": traffic.d_store_logical_bytes,
        },
        "transaction_bytes": {
            "a_l2_requested": traffic.a_l2_requested_bytes,
            "b_l2_requested": traffic.b_l2_requested_bytes,
            "a_dram_unique": traffic.a_dram_unique_bytes,
            "b_dram_unique": traffic.b_dram_unique_bytes,
            "a_dram_read": traffic.a_dram_read_bytes,
            "b_dram_read": traffic.b_dram_read_bytes,
            "c_read": traffic.c_read_transaction_bytes,
            "d_store": traffic.d_store_transaction_bytes,
            "l2_requested": traffic.l2_requested_bytes,
            "l2_capacity": traffic.l2_capacity_bytes,
            "l2_operation_working_set": traffic.l2_operation_working_set_bytes,
            "l2_reuse_working_set": traffic.l2_reuse_working_set_bytes,
            "l2_weighted_reuse_working_set": (
                traffic.l2_weighted_reuse_working_set_bytes
            ),
            "l2_capacity_miss_fraction": traffic.l2_capacity_miss_fraction,
            "l2_k_window_stages": traffic.l2_k_window_stages,
            "l2_k_window_count": traffic.l2_k_window_count,
            "l2_reuse_windows": {
                "full_wave_full_k": traffic.l2_full_wave_full_window_bytes,
                "full_wave_tail_k": traffic.l2_full_wave_tail_window_bytes,
                "last_wave_full_k": traffic.l2_last_wave_full_window_bytes,
                "last_wave_tail_k": traffic.l2_last_wave_tail_window_bytes,
            },
            "l2_reuse_distance": {
                "a_within_wave_repeated": traffic.a_within_wave_repeated_bytes,
                "a_cross_wave_repeated": traffic.a_cross_wave_repeated_bytes,
                "b_within_wave_repeated": traffic.b_within_wave_repeated_bytes,
                "b_cross_wave_repeated": traffic.b_cross_wave_repeated_bytes,
                "within_wave_miss_fraction": traffic.l2_within_wave_miss_fraction,
                "a_cross_wave_bytes": traffic.a_cross_wave_reuse_distance_bytes,
                "b_cross_wave_bytes": traffic.b_cross_wave_reuse_distance_bytes,
                "a_cross_wave_miss_fraction": traffic.a_cross_wave_miss_fraction,
                "b_cross_wave_miss_fraction": traffic.b_cross_wave_miss_fraction,
            },
            "dram_capacity_miss": traffic.dram_capacity_miss_bytes,
            "dram_unique": traffic.dram_unique_bytes,
            "dram_total": traffic.dram_total_bytes,
            "smem_read": traffic.smem_read_bytes,
            "smem_write": traffic.smem_write_bytes,
            "epilogue_smem": timeline.epilogue_smem_bytes,
            "epilogue_l2": timeline.epilogue_l2_bytes,
            "epilogue_dram": timeline.epilogue_dram_bytes,
            "sector_size": traffic.sector_size_bytes,
            "line_size": traffic.line_size_bytes,
        },
        "wave_pipeline": {
            "full": _wave_pipeline_diagnostics(timeline.full_wave),
            "last": _wave_pipeline_diagnostics(timeline.last_wave),
        },
        "pipeline_components": {
            "groups_k": timeline.groups_k,
            "last_stage_groups_k": timeline.last_stage_groups_k,
            "last_stage_k": timeline.last_stage_k,
            "memory_pipeline_groups": timeline.memory_pipeline_groups,
            "last_memory_pipeline_stages": timeline.last_memory_pipeline_stages,
        },
        "active_cycles": {
            "compute": timeline.compute_active_cycles,
            "smem": timeline.smem_active_cycles,
            "l2": timeline.l2_active_cycles,
            "dram": timeline.dram_active_cycles,
        },
        "stage_cycles": {
            "compute": timeline.compute_stage_cycles,
            "mma_issue": timeline.mma_issue_cycles,
            "mma_dependency_penalty": timeline.mma_dependency_penalty_cycles,
            "smem": timeline.smem_stage_cycles,
            "global_load_issue": timeline.global_load_issue_cycles,
            "l2_service": timeline.l2_service_cycles,
            "dram_service": timeline.dram_service_cycles,
            "exposed_l2": timeline.exposed_l2_cycles,
            "exposed_dram": timeline.exposed_dram_cycles,
            "stage": timeline.stage_cycles,
            "prologue": timeline.prologue_cycles,
            "work": timeline.work_cycles,
            "epilogue": timeline.epilogue_cycles,
            "cta": timeline.cta_cycles,
        },
        "utilization": {
            "compute_issue": timeline.compute_issue_utilization,
            "compute_latency_adjusted": timeline.compute_latency_utilization,
            "compute_active": timeline.compute_active_utilization,
            "smem": timeline.smem_utilization,
            "l2": timeline.l2_utilization,
            "dram": timeline.dram_utilization,
            "mma_ilp_efficiency": timeline.mma_ilp_efficiency,
        },
        "overlap": {
            "compute_smem": timeline.compute_smem_overlap,
            "compute_l2": timeline.compute_l2_overlap,
            "compute_dram": timeline.compute_dram_overlap,
        },
        "operational_intensity": {
            "smem_flop_per_byte": useful / q_smem if q_smem else None,
            "l2_flop_per_byte": useful / q_l2 if q_l2 else None,
            "dram_flop_per_byte": useful / q_dram if q_dram else None,
            "merged_flop_per_byte": useful / total_q if total_q else None,
        },
        "roofline_bounds_flop_per_cycle": roofline_metrics[
            "effective_bounds_deprecated"
        ],
        "roofline_effective_bounds_deprecated_flop_per_cycle": roofline_metrics[
            "effective_bounds_deprecated"
        ],
        "roofline_raw_bounds_flop_per_cycle": roofline_metrics["raw_bounds"],
        "roofline_achieved_flop_per_cycle": roofline_metrics["achieved"],
        "roofline_ceiling_utilization": roofline_metrics["ceiling_utilization"],
        "roofline_total_ceiling_utilization": roofline_metrics[
            "total_ceiling_utilization"
        ],
        "roofline_factor_breakdown": roofline_metrics["factor_breakdown"],
        "memory_level_latencies_s": {
            "hbm": hardware.memory_levels["hbm"].latency_s,
            "l2": hardware.memory_levels["l2"].latency_s
            if "l2" in hardware.memory_levels
            else None,
            "sram": hardware.memory_levels["sram"].latency_s
            if "sram" in hardware.memory_levels
            else None,
            "register": None,
        },
        "primary_bottleneck": primary,
        "secondary_bottlenecks": secondary,
        "warnings": all_warnings,
        "assumptions": (
            "deterministic_tiled_gemm_access_pattern",
            "row_major_cta_launch_order_for_l2_reuse",
            "representative_full_last_wave_k_window_l2_capacity_model",
            "analytical_a_b_cross_wave_reuse_distance_buckets",
            "bounded_l2_capacity_miss_approximation",
            "dram_unique_is_compulsory_lower_bound",
            "dram_total_includes_capacity_misses_and_output_writeback",
            "artifact_style_full_last_wave_pipeline",
        )
        + (
            ("fixed_device_overhead_calibrated_from_small_gemm_residuals",)
            if fixed_overhead_cycles > 0.0
            else ("no_fixed_device_overhead_configured",)
        )
        + (
            "shared_memory_bank_conflict_factor_1",
            "four_smsps_per_sm_assumed",
        ),
        "debug_trace": _debug_trace(
            problem,
            kernel,
            grid,
            traffic,
            occupancy,
            timeline,
            primary,
            secondary,
            fixed_overhead_cycles,
            total_device_cycles,
        ),
    }

def _wave_pipeline_diagnostics(wave: WavePipelineResult) -> dict[str, Any]:
    return {
        "active_ctas": wave.active_ctas,
        "busy_sms": wave.busy_sms,
        "lazy_sms": wave.lazy_sms,
        "busy_ctas_per_sm": wave.busy_ctas_per_sm,
        "lazy_ctas_per_sm": wave.lazy_ctas_per_sm,
        "start_cycles": wave.start_cycles,
        "work_cycles": wave.work_cycles,
        "end_cycles": wave.end_cycles,
        "total_cycles": wave.total_cycles,
        "sm_stage_cycles": wave.sm_stage_cycles,
        "sm_last_stage_cycles": wave.sm_last_stage_cycles,
        "smem_group_cycles": wave.smem_group_cycles,
        "math_group_cycles": wave.math_group_cycles,
        "math_issue_group_cycles": wave.math_issue_group_cycles,
        "math_latency_group_cycles": wave.math_latency_group_cycles,
        "memory_full_stage_cycles": wave.memory_full_stage_cycles,
        "memory_last_stage_cycles": wave.memory_last_stage_cycles,
        "l2_full_stage_cycles": wave.l2_full_stage_cycles,
        "l2_last_stage_cycles": wave.l2_last_stage_cycles,
        "dram_full_stage_cycles": wave.dram_full_stage_cycles,
        "dram_last_stage_cycles": wave.dram_last_stage_cycles,
        "epilogue_global_cycles": wave.epilogue_global_cycles,
        "epilogue_smem_cycles": wave.epilogue_smem_cycles,
        "slice_k_extra_cycles": wave.slice_k_extra_cycles,
        "epilogue_smem_bytes": wave.epilogue_smem_bytes,
        "epilogue_l2_bytes": wave.epilogue_l2_bytes,
        "epilogue_dram_bytes": wave.epilogue_dram_bytes,
        "l2_epilogue_cycles": wave.l2_epilogue_cycles,
        "dram_epilogue_cycles": wave.dram_epilogue_cycles,
        "exposed_l2_cycles": wave.exposed_l2_cycles,
        "exposed_dram_cycles": wave.exposed_dram_cycles,
    }

def _debug_trace(
    problem: GemmProblemSpec,
    kernel: GemmKernelSpec,
    grid: GridAccounting,
    traffic: TrafficAccounting,
    occupancy: OccupancyResult,
    timeline: TimelineResult,
    primary: str,
    secondary: tuple[str, ...],
    fixed_overhead_cycles: float,
    total_device_cycles: float,
) -> str:
    return "\n".join(
        (
            f"GEMM shape: B={problem.batch}, M={problem.m}, N={problem.n}, K={problem.k}",
            f"CTA tile: {kernel.cta_m}x{kernel.cta_n}x{kernel.cta_k}",
            f"CTA count: {grid.cta_count}",
            f"K stages: {grid.k_stages}",
            f"resident CTAs per SM: {occupancy.resident_ctas_per_sm}",
            f"CTA waves: {occupancy.wave_count}",
            f"tail efficiency: {occupancy.tail_efficiency:.4g}",
            f"full wave cycles: {timeline.full_wave.total_cycles:.4g}",
            f"last wave cycles: {timeline.last_wave.total_cycles:.4g}",
            f"modeled device cycles: {timeline.kernel_cycles:.4g}",
            f"fixed overhead cycles: {fixed_overhead_cycles:.4g}",
            f"total device cycles: {total_device_cycles:.4g}",
            f"A logical bytes: {traffic.a_logical_bytes}",
            f"B logical bytes: {traffic.b_logical_bytes}",
            f"D store bytes: {traffic.d_store_logical_bytes}",
            f"L2 requested bytes: {traffic.l2_requested_bytes}",
            f"DRAM unique bytes: {traffic.dram_unique_bytes}",
            f"DRAM capacity miss bytes: {traffic.dram_capacity_miss_bytes}",
            f"DRAM total bytes: {traffic.dram_total_bytes}",
            f"SMEM bytes: {traffic.smem_total_bytes + timeline.epilogue_smem_bytes:.4g}",
            f"compute stage cycles: {timeline.compute_stage_cycles:.4g}",
            f"SMEM stage cycles: {timeline.smem_stage_cycles:.4g}",
            f"L2 service cycles: {timeline.l2_service_cycles:.4g}",
            f"DRAM service cycles: {timeline.dram_service_cycles:.4g}",
            f"exposed L2 cycles: {timeline.exposed_l2_cycles:.4g}",
            f"exposed DRAM cycles: {timeline.exposed_dram_cycles:.4g}",
            f"compute active utilization: {timeline.compute_active_utilization:.4g}",
            f"SMEM active utilization: {timeline.smem_utilization:.4g}",
            f"L2 active utilization: {timeline.l2_utilization:.4g}",
            f"DRAM active utilization: {timeline.dram_utilization:.4g}",
            f"compute/L2 overlap: {timeline.compute_l2_overlap:.4g}",
            f"compute/DRAM overlap: {timeline.compute_dram_overlap:.4g}",
            f"primary bottleneck: {primary}",
            f"secondary bottlenecks: {', '.join(secondary) if secondary else 'none'}",
        )
    )

def _implementation_name(batched: bool) -> str:
    return "extended_roofline.batched_gemm" if batched else "extended_roofline.gemm"

def _positive_int_attr(attrs: Mapping[str, Any], name: str, default: int) -> int:
    value = attrs.get(name, default)
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result

def _optional_positive_int_attr(attrs: Mapping[str, Any], name: str) -> int | None:
    value = attrs.get(name)
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result

def _clock_hz(hardware: HardwareSpec, warnings: list[str]) -> float:
    if hardware.compute.clock_hz is None:
        warnings.append("clock_hz_default_1GHz")
        return 1.0e9
    return hardware.compute.clock_hz

def _smem_bandwidth_per_cycle(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    sram = hardware.memory_levels.get("sram")
    if sram is not None:
        return sram.bandwidth_bytes_per_s / clock_hz
    if hardware.compute.num_sms is not None:
        warnings.append("sram_level_absent_smem_bandwidth_inferred_from_shared_banks")
        return float(hardware.compute.num_sms * 32 * 4)
    warnings.append("sram_level_absent_smem_bandwidth_inferred_from_hbm")
    return hardware.memory_levels["hbm"].bandwidth_bytes_per_s * 8.0 / clock_hz

def _shared_latency_cycles(
    hardware: HardwareSpec, clock_hz: float, warnings: list[str]
) -> float:
    sram = hardware.memory_levels.get("sram")
    if sram is not None and sram.latency_cycles is not None:
        return float(sram.latency_cycles)
    if sram is not None and sram.latency_s > 0.0:
        return sram.latency_s * clock_hz
    raise ValueError("sram latency is required for GEMM timing")

def _memory_latency_cycles(
    level: MemoryLevel | None, clock_hz: float, name: str
) -> float:
    if level is None:
        raise ValueError(f"{name} memory level and latency are required for GEMM timing")
    if level.latency_cycles is not None:
        return float(level.latency_cycles)
    if level.latency_s > 0.0:
        return level.latency_s * clock_hz
    raise ValueError(f"{name} latency is required for GEMM timing")

def _tensor_latency_cycles(
    kernel: GemmKernelSpec, hardware: HardwareSpec, warnings: list[str]
) -> float:
    if hardware.compute.tensor_latency_cycles is not None:
        return float(hardware.compute.tensor_latency_cycles)
    raise ValueError("tensor latency is required for GEMM timing")

def _append_warning_once(warnings: list[str], warning: str) -> None:
    if warning not in warnings:
        warnings.append(warning)

def _tile_lengths(total: int, tile: int) -> tuple[int, ...]:
    full, rem = divmod(total, tile)
    values = [tile] * full
    if rem:
        values.append(rem)
    return tuple(values) or (0,)

def _tile_groups(total: int, tile: int) -> tuple:
    full, rem = divmod(total, tile)
    groups = []
    if full:
        groups.append((tile, full))
    if rem:
        groups.append((rem, 1))
    return tuple(groups) or ((0, 1),)

def _sector_round_bytes(value: float, sector_size: int) -> int:
    scalar = _ceil_scalar_bytes(value)
    if scalar == 0:
        return 0
    return _ceil_div(scalar, sector_size) * sector_size

def _ceil_scalar_bytes(value: float) -> int:
    return int(math.ceil(value))

def _ceil_div(numerator: int | float, denominator: int | float) -> int:
    return int(math.ceil(numerator / denominator))

def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value

def _overlap_ratio(x_cycles: float, y_cycles: float) -> float:
    denominator = max(x_cycles, y_cycles)
    if denominator <= 0.0:
        return 0.0
    return min(x_cycles, y_cycles) / denominator
