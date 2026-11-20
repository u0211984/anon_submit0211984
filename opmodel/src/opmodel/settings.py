from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

@dataclass(frozen=True)
class Setting:
    name: str
    default: Any
    parse: Callable[[str], Any]
    yaml: str
    source: str

_REGISTRY: Dict[str, Setting] = {}
_EMPTY: Mapping[str, Any] = {}
_YAML: ContextVar = ContextVar("opmodel_settings_yaml", default=None)
_LATEST: list = [_EMPTY]
_OVERRIDES: list = []
_RECORDERS: list = []

def register(name: str, default: Any, parse: Callable[[str], Any], yaml: str, source: str) -> None:
    _REGISTRY[name] = Setting(name, default, parse, yaml, source)

def registry() -> Mapping[str, Setting]:
    return dict(_REGISTRY)

def is_registered(name: str) -> bool:
    return name in _REGISTRY

def get(name: str) -> Any:
    value = _resolve(name)
    if _RECORDERS:
        for reads in _RECORDERS:
            if name not in reads:
                reads[name] = value
    return value

@contextmanager
def recording() -> Iterator[Dict[str, Any]]:
    reads: Dict[str, Any] = {}
    _RECORDERS.append(reads)
    try:
        yield reads
    finally:
        for index in range(len(_RECORDERS) - 1, -1, -1):
            if _RECORDERS[index] is reads:
                del _RECORDERS[index]
                break

def unchanged(reads: Mapping[str, Any]) -> bool:
    for name, value in reads.items():
        current = get(name)
        if current is not value and (type(current) is not type(value) or repr(current) != repr(value)):
            return False
    return True

def _resolve(name: str) -> Any:
    for layer in reversed(_OVERRIDES):
        if name in layer:
            return layer[name]
    setting = _REGISTRY.get(name)
    if setting is None:
        raise KeyError(f"setting {name!r} is not registered (Rapid-LLM's RAPID_* settings register when its module "
                       "sim_settings is imported; every module that reads one imports it)")
    raw = os.environ.get(name)
    if raw is not None:
        return setting.parse(raw)
    layer = _YAML.get()
    if layer is None:
        layer = _LATEST[0]
    if name in layer:
        return layer[name]
    return setting.default

def origin(name: str) -> str:
    for layer in reversed(_OVERRIDES):
        if name in layer:
            return "override"
    if os.environ.get(name) is not None:
        return "env"
    if name in active():
        return "yaml"
    return "default"

def check(values: Mapping[str, Any]) -> Dict[str, Any]:
    unknown = [k for k in values if k not in _REGISTRY]
    if unknown:
        raise KeyError(f"unknown settings {unknown}")
    return dict(values)

@contextmanager
def use(layer: Mapping[str, Any]) -> Iterator[None]:
    token = _YAML.set(layer)
    try:
        yield
    finally:
        _YAML.reset(token)

def active() -> Mapping[str, Any]:
    layer = _YAML.get()
    return _LATEST[0] if layer is None else layer

def set_latest(layer: Mapping[str, Any]) -> None:
    _LATEST[0] = layer

@contextmanager
def override(values: Mapping[str, Any]) -> Iterator[None]:
    unknown = [k for k in values if k not in _REGISTRY]
    if unknown:
        raise KeyError(f"unknown settings {unknown}")
    _OVERRIDES.append(dict(values))
    try:
        yield
    finally:
        _OVERRIDES.pop()

def _jsonable(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value

def resolved(*, exclude_prefixes: tuple[str, ...] = ()) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(_REGISTRY):
        if name.startswith(exclude_prefixes):
            continue
        s = _REGISTRY[name]
        try:
            value = _jsonable(get(name))
        except (ValueError, TypeError) as exc:
            value = f"<invalid: {exc}>"
        out[name] = {"value": value, "from": origin(name), "yaml": s.yaml, "source": s.source}
    return out

def flag_one(raw: str) -> bool:
    return raw == "1"

def as_str(raw: str) -> str:
    return raw

def int_or_zero(raw: str) -> int:
    return int(raw or 0)

def float_or_zero(raw: str) -> float:
    return float(raw or 0)

def l2_effective_bytes(raw: str) -> Optional[float]:
    return float(raw) * 1024 * 1024 if raw else None

def launch_order(raw: str) -> tuple:
    axis, width, serp = (x.strip() for x in raw.split(","))
    if axis not in ("m", "n"):
        raise ValueError(f"OPMODEL_HBM_DEFAULT_ORDER axis must be m or n, got {axis!r}")
    return (axis, int(width), serp == "1")

register("OPMODEL_HBM_ORDER", "legacy", as_str, "hw:l2_model.gemm_tile.order",
         "HBM tiling v2; legacy default unchanged; cutlass_identity_swizzle = GemmIdentityThreadblockSwizzle<8>, CUTLASS v3.5.1 f7b19de32c5d1f3cedfc735c2849f12b537522ee include/cutlass/gemm/threadblock/threadblock_swizzle.h:110-139 and include/cutlass/gemm/kernel/gemm.h:208-215")
register("OPMODEL_HBM_SINGLE_MISS", False, flag_one, "hw:l2_model.gemm_tile.single_miss",
         "HBM v2: drop v1's operation-pressure factor")
register("OPMODEL_HBM_MISS_SHAPE", "overflow", as_str, "hw:l2_model.gemm_tile.miss_shape",
         "HBM v2 step B2b: overflow (1 - C/d) or recency (A100 L2 probes, results/l2_probe)")
register("OPMODEL_L2_EFFECTIVE_MIB", None, l2_effective_bytes, "hw:l2_model.capacity",
         "value in bytes; None = physical L2; effective = physical / cache.partitions")
register("OPMODEL_HBM_DEFAULT_ORDER", ("m", 8, True), launch_order, "hw:l2_model.gemm_tile.default_launch_order",
         "strips of 8 N-tiles walked along M, serpentine (our reading of cuBLAS on A100) [no public source]")
register("OPMODEL_HBM_WS_ORDER", False, flag_one, "hw:l2_model.gemm_tile.ws_order",
         "HBM v5 step B6: launch order axes swapped with the weight-stationary transpose")
register("OPMODEL_HBM_EF_STORES", False, flag_one, "hw:l2_model.gemm_tile.ef_stores",
         "HBM v5 step B2c: evict-first output stores (SASS STG.E.EF)")
register("OPMODEL_HBM_STRIP_IN_WAVE", False, flag_one, "hw:l2_model.gemm_tile.strip_in_wave",
         "HBM v5 step B4: panel returns one strip later")
register("OPMODEL_HBM_SPLITK", False, flag_one, "hw:l2_model.gemm_tile.splitk",
         "HBM v5 step B3: split-K slice reads")
register("OPMODEL_HBM_SPLITK_WRITES", False, flag_one, "hw:l2_model.gemm_tile.splitk_writes",
         "HBM v5 step B5: split-K partial-result writes")
register("OPMODEL_RASTER_WEIGHT_STATIONARY", True, flag_one, "hw:l2_model.gemm_tile.raster_weight_stationary",
         "larger operand stationary (NCU, A100 PCIe, vLLM prefill 7B QKV 101.7 vs 100.7 MB)")
register("OPMODEL_HBM_READ_ECC", False, flag_one, "hw:l2_model.gemm_tile.dram_read_ecc",
         "DRAM reads x 9/8 on GDDR (observed L4/T4/A10) [no public source]")

register("OPMODEL_GEMM_CATALOG", "default", as_str, "model:implementation.gemm_library.catalog",
         "tile template catalog: default, cublas_sm80, or cutlass_sm80_16816")
def _gemm_selection_rule(raw: str) -> str:
    value = raw.strip().lower()
    if value not in ("min_latency", "useful_fraction"):
        raise ValueError(f"invalid GEMM selection rule: {value!r}")
    return value

register("OPMODEL_GEMM_SELECTION_RULE", "min_latency", _gemm_selection_rule,
         "model:implementation.gemm_library.selection_rule", "GEMM candidate selection rule")
register("OPMODEL_SM90_MIN_TWO_STAGES", True, flag_one, "model:implementation.gemm_library.sm90_min_two_stages",
         "CUTLASS f7b19de3 sm90_mma_tma_gmma_ss_warpspecialized.hpp:126 requires Stages >= 2")
register("OPMODEL_SM90_ACCUMULATOR_LIMIT", True, flag_one, "model:implementation.gemm_library.sm90_accumulator_limit",
         "CUTLASS f7b19de3 sm90_gemm_tma_warpspecialized_cooperative.hpp:283-294 has two consumer warpgroups; captured distributed_runs/h100_counts/launches/05a0baefeecb40ffa5a00444f687e1a4/results/session/vllm_v2/cap/Llama-3.1-8B_prefill_1x1024.csv:490 block (384,1,1); PTX WGMMA D is a register fragment; 255 registers per thread")
register("OPMODEL_SM90_CUTLASS_N_SELECTOR", True, flag_one, "model:implementation.gemm_library.sm90_cutlass_n_selector",
         "CUTLASS f7b19de3 cute/arch/mma_sm90.hpp:852-927 default BF16 SS selector")
register("OPMODEL_GEMM_FORCED_TEMPLATES", None, lambda raw: raw.strip() or None,
         "model:implementation.gemm_library.forced_templates", "optional logical-shape template CSV")
register("OPMODEL_MIN_GEMM_TILE", 64, int_or_zero, "model:implementation.gemm_library.min_tile",
         "NCU, cuBLAS sm80 FP16 Llama batch-1 decode / 32-token prefill: no 32-wide tiles [no public source]")
def _tile_padding(raw: str) -> str:
    if raw.strip() != "cta":
        raise ValueError("OPMODEL_TILE_PADDING / gemm_library.tile_padding 'mma' is OBSOLETE: "
                         f"tile padding is always 'cta'; got {raw!r}")
    return "cta"

register("OPMODEL_TILE_PADDING", "cta", _tile_padding, "model:implementation.gemm_library.tile_padding",
         "cta: full warp tile per live warp (NCU batch-1 decode: 64x useful work on 64-row tiles); the old 'mma' rule is "
         "OBSOLETE and raises")
register("OPMODEL_SMEM_STAGING_LATENCY", True, flag_one, "model:implementation.gemm_library.smem_staging_latency",
         "sm80: ldmatrix reads and cp.async staging writes share SMEM banks (time only)")
register("OPMODEL_SMEM_STAGING", False, flag_one, "model:implementation.gemm_library.smem_staging",
         "charge cp.async staging as SMEM writes (NCU ldgsts 21-32% of Llama SMEM bytes)")
register("OPMODEL_SMEM_EPILOGUE", "none", lambda raw: raw.strip().lower(), "model:implementation.gemm_library.epilogue_smem",
         "cutlass = CUTLASS 2.x tensor-op epilogue SMEM (effective_roofline._roofline_epilogue citations)")
register("OPMODEL_SMEM_CLEAR_STAGE", False, flag_one, "model:implementation.gemm_library.clear_last_stage",
         "CUTLASS kClearLastStage prologue (basis: NCU counter match only) [weak source]")
register("OPMODEL_GEMV_RULE", "cublas", as_str, "model:implementation.gemm_library.gemv.rule",
         "cublas = max_n rule; trtllm = TensorRT-LLM 1.0 cudaCoreGemm for M <= max_m (nsys, measure_trtllm)")
register("OPMODEL_GEMV_MAX_N", 2304, int_or_zero, "model:implementation.gemm_library.gemv.max_n",
         "cuBLAS 12.8 F.linear M=1 picks GEMV for N <= 2304 (probe_work/cublas_choice/gemv_boundary.json) [own profile]")
register("OPMODEL_GEMV_MAX_M", 4, lambda raw: int(raw or 4), "model:implementation.gemm_library.gemv.max_m",
         "TRT-LLM 1.0 as profiled (v0.14 gemmPlugin.cpp: 6)")
register("OPMODEL_GEMV_STREAM_BW", 0.0, float_or_zero, "model:implementation.gemm_library.gemv.stream_bw",
         "bytes/s, 0 = off; calibrated on TRT-LLM cudaCoreGemm (time only) [fitted, no public source]; legacy, no paper entry uses it")
register("OPMODEL_GEMV_STREAM_HBM", False, flag_one, "model:implementation.gemm_library.gemv.stream_hbm",
         "GEMV-rule linears stream at the hardware file's DRAM rate (DRAM.bandwidth x util, as every memory-bound kernel; "
         "A100: a100_literature [RAP] 0.8 x spec) + launch overhead (time only); no GEMV-specific number")
register("OPMODEL_GEMV_IDEAL_ROOFLINE", False, flag_one, "model:implementation.gemm_library.gemv.ideal_roofline",
         "cuBLAS GEMV timing-only cold-stream ideal roofline using configured peak HBM and vector rates")
