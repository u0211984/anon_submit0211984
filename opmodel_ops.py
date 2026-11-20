from __future__ import annotations

import math
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

@dataclass(frozen=True)
class OpModelResult:
    latency_s: float
    energy_j: float
    flops: float
    implementation: str
    memory_accesses: tuple[int, int, int, int]
    tile_dims: tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]
    energy_breakdown: Mapping[str, float]
    diagnostics: Mapping[str, Any]

class OpModelEstimator:
    def __init__(
        self,
        *,
        hw_config: Any,
        core: Any,
        mem_layer: list[Any],
        precision_bytes: float,
        run_type: str,
    ) -> None:
        self.hw_config = hw_config
        self.core = core
        self.mem_layer = mem_layer
        self.precision_bytes = float(precision_bytes)
        self.run_type = str(run_type or "training").lower()
        self.gemm_selection_objective = str(
            getattr(getattr(hw_config, "sw_config", None), "gemm_selection_objective", "latency")
            or "latency"
        ).lower()
        self._api, self._hardware_mod, self._registry = self._load_opmodel()
        self.provenance = self._opmodel_provenance()
        self.dtype = self._dtype_from_precision_bytes(self.precision_bytes)
        self.hardware = self._build_hardware_spec()
        self.model = self._registry.create_model("effective_roofline")

    @staticmethod
    def _load_opmodel():
        try:
            from opmodel import api
            from opmodel import hardware as hardware_mod
            from opmodel import registry
            return api, hardware_mod, registry
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Rapid-LLM requires the declared opmodel dependency. Install "
                "the in-tree ./opmodel package in the repository virtual environment."
            ) from exc

    @staticmethod
    def _opmodel_provenance() -> dict[str, str]:
        import opmodel

        module_path = Path(opmodel.__file__).resolve()
        try:
            package_version = version("opmodel")
        except PackageNotFoundError:
            package_version = "unknown"

        revision = "unknown"
        repository = module_path.parents[2]
        git_dir = repository / ".git"
        if git_dir.is_dir():
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref: "):
                ref_path = git_dir / head.removeprefix("ref: ")
                if ref_path.is_file():
                    revision = ref_path.read_text().strip()
            elif head:
                revision = head
        return {
            "backend": "opmodel",
            "version": package_version,
            "revision": revision,
            "module_path": str(module_path),
        }

    def _dtype_from_precision_bytes(self, precision_bytes: float):
        dtype = self._api.DType
        if precision_bytes <= 0.5:
            return dtype.INT4
        if precision_bytes <= 1.0:
            return dtype.FP8
        if precision_bytes <= 2.0:
            return dtype.BF16
        return dtype.FP32

    def _build_hardware_spec(self):
        hardware = self._hardware_mod
        dtype = self._api.DType
        clock_hz = float(self.core.operating_freq)
        tensor_flops = float(self.core.get_throughput("matrix"))
        vector_flops = float(self.core.get_throughput("vector"))
        tensor_energy = float(self.core.tensor_energy_per_flop)
        vector_energy = float(self.core.vector_energy_per_flop)
        num_sms = int(self.core.num_bundle)
        if getattr(self.core, "max_ctas_per_sm", None) is None:
            raise ValueError("hardware tech_param.core.max_ctas_per_sm is required")
        shared_bytes = int(getattr(self.mem_layer[1], "size", 0) / num_sms)
        register_count = int(getattr(self.mem_layer[0], "size", 0) / num_sms / 4)
        architecture = str(getattr(getattr(getattr(self.hw_config, "tech_config", None),
                                          "core", None), "architecture", None) or "sm80")
        shared_per_block = getattr(self.core, "shared_memory_bytes_per_block", None)
        if architecture == "sm90" and shared_per_block is None:
            raise ValueError("sm90 stage policy requires hardware shared_memory_bytes_per_block")
        compute = hardware.ComputeUnit(
            clock_hz=clock_hz,
            vector_flops_per_s={
                dtype.FP32: vector_flops,
                dtype.BF16: vector_flops,
                dtype.FP16: vector_flops,
            },
            tensor_flops_per_s={
                dtype.FP32: tensor_flops,
                dtype.TF32: tensor_flops,
                dtype.BF16: tensor_flops,
                dtype.FP16: tensor_flops,
                dtype.FP8: tensor_flops,
                dtype.INT8: tensor_flops,
                dtype.INT4: tensor_flops,
            },
            vector_energy_j_per_flop={
                dtype.FP32: vector_energy,
                dtype.BF16: vector_energy,
                dtype.FP16: vector_energy,
            },
            tensor_energy_j_per_flop={
                dtype.FP32: tensor_energy,
                dtype.TF32: tensor_energy,
                dtype.BF16: tensor_energy,
                dtype.FP16: tensor_energy,
                dtype.FP8: tensor_energy,
                dtype.INT8: tensor_energy,
                dtype.INT4: tensor_energy,
            },
            num_sms=num_sms,
            fma_dims=self._fma_dims(),
            dataflow=self._dataflow(),

            max_ctas_per_sm=int(self.core.max_ctas_per_sm),
            max_warps_per_sm=int(getattr(self.core, "max_warps_per_sm", None) or 64),
            registers_per_sm=register_count or None,
            shared_memory_bytes_per_sm=shared_bytes or None,

            shared_memory_bytes_per_block=int(shared_per_block or 163 * 1024),

            register_allocation_per_thread=int(getattr(self.core, "register_allocation_per_thread", None) or 8),
            tensor_latency_cycles=self._optional_nonnegative_cycles(self.core.tensor_latency_cycles),
            device_fixed_overhead_cycles=self._optional_nonnegative_int(
                self.core.device_fixed_overhead_cycles
            ),
        )
        from sim_settings import l2_bypass
        bypass = {}
        if l2_bypass(self.hw_config):
            import dataclasses
            if "l2_bypass" not in {f.name for f in dataclasses.fields(hardware.HardwareSpec)}:
                raise ImportError(f"tech_param.SRAM-L2.l2_bypass needs an opmodel with HardwareSpec.l2_bypass; the "
                                  f"imported opmodel ({hardware.__file__}) has none. Put this tree's opmodel/src "
                                  "first on PYTHONPATH.")
            bypass = {"l2_bypass": True}
        return hardware.HardwareSpec(
            name="rapid_opmodel_gpu",
            kind="gpu",
            compute=compute,
            memory_levels=self._memory_levels(),
            static_power_w=float(getattr(self.hw_config, "static_power_w", 0.0) or 0.0),
            architecture=architecture,
            **bypass,
        )

    def _fma_dims(self) -> tuple[int, int, int] | None:
        dims = tuple(int(x) for x in getattr(self.core, "FMA_dims", ()) or ())
        if len(dims) >= 3:
            return dims[:3]
        if len(dims) == 2:
            return (16, 8, 16)
        return None

    def _dataflow(self) -> str | None:
        aliases = {
            "ost": "output_stationary",
            "wst": "weight_stationary",
            "ast": "activation_stationary",
            "best": "output_stationary",
        }
        value = str(getattr(self.core, "dataflow", "") or "").lower()
        return aliases.get(value, value or None)

    def _memory_levels(self) -> Mapping[str, Any]:
        hardware = self._hardware_mod
        names = {0: "register", 1: "sram", 2: "l2", 3: "hbm"}
        levels: dict[str, Any] = {}
        from sim_settings import CacheSpec
        l2_line = CacheSpec.from_hw(self.hw_config).line_bytes
        for index, layer in enumerate(self.mem_layer):
            if layer is None:
                continue
            name = names.get(index, f"level{index}")
            levels[name] = hardware.MemoryLevel(
                name=name,
                size_bytes=self._optional_int(getattr(layer, "size", None)),
                bandwidth_bytes_per_s=float(layer.get_throughput()),
                energy_j_per_byte=float(
                    getattr(layer, "dynamic_energy_per_byte", 0.0) or 0.0
                ),
                energy_j_per_sector=self._optional_float(
                    getattr(layer, "dynamic_energy_per_sector", None)
                ),
                sector_size_bytes=self._optional_int(
                    getattr(layer, "sector_size_bytes", None)
                ),

                line_size_bytes=l2_line if name == "l2" else None,
                latency_s=float(layer.get_latency()),
                latency_cycles=self._optional_float(
                    getattr(layer, "latency_cycles", None)
                ),
            )
        if "hbm" not in levels:
            raise ValueError("op-model hardware requires an HBM memory level")
        return levels

    def predict_batched_gemm(
        self,
        *,
        dim1: int,
        dim2: int,
        dim3: int,
        batch: int = 1,
        name: str = "",
        allow_gemv: bool = True,
        gemm_role: str | None = None,
        weight_bytes: float | None = None,
    ) -> OpModelResult:
        if min(dim1, dim2, dim3) <= 0:
            raise ValueError(f"GEMM dimensions must be positive: {(dim1, dim2, dim3)}")
        batch = max(1, int(math.ceil(float(batch))))
        weight_shape: tuple[int, ...]
        if batch > 1 and "attention" in str(name).lower():
            weight_shape = (batch, int(dim2), int(dim3))
        else:
            weight_shape = (int(dim2), int(dim3))
        if weight_bytes is not None:
            if len(weight_shape) != 2:
                raise ValueError("weight_bytes describes a 2-D (linear-layer) weight, not a batched attention operand")
            allow_gemv = False
        api = self._api
        op = api.LocalOp(
            name=name or "gemm",
            kind=api.OpKind.BATCHED_GEMM,
            phase=self._phase(),
            tensors=(
                api.TensorSpec(api.TensorRole.INPUT, (batch, int(dim1), int(dim2)), self.dtype),
                api.TensorSpec(api.TensorRole.WEIGHT, weight_shape, self.dtype),
                api.TensorSpec(api.TensorRole.OUTPUT, (batch, int(dim1), int(dim3)), self.dtype),
            ),
            attrs={"gemm_selection_objective": self.gemm_selection_objective,
                   **({"gemm_role": gemm_role} if gemm_role is not None else {}),
                   **({"weight_bytes": float(weight_bytes)} if weight_bytes is not None else {})},
        )
        from opmodel import settings as _settings
        split_catalog = _settings.get("OPMODEL_GEMM_CATALOG") == "cublaslt_sm80_bf16_splitk"
        gemv = allow_gemv and "attention" not in str(name).lower() and self._is_cublas_gemv(
            dim1=int(dim1), dim3=int(dim3), batch=batch, weight_rank=len(weight_shape), name=str(name),
            architecture=self.hardware.architecture)
        if split_catalog and gemv:
            with _settings.override({"OPMODEL_GEMM_CATALOG": "cublaslt_sm80_bf16"}):
                result = self._result(self.model.predict(op, self.hardware))
        else:
            result = self._result(self.model.predict(op, self.hardware))
        if gemv:
            reg, _smem, l2, hbm = result.memory_accesses
            result = replace(
                result,
                flops=0.0,
                memory_accesses=(reg, 0, l2, hbm),
                diagnostics={**dict(result.diagnostics), "gemv": True},
            )
        return result

    def predict_gemm_with_template(self, *, dim1: int, dim2: int, dim3: int, template: Any,
                                   name: str = "") -> OpModelResult:
        from dataclasses import replace as _replace
        from opmodel.models import effective_roofline as _effective
        from opmodel.models import extended_roofline as _extended
        from opmodel.ops import footprint_from_tensors
        if min(dim1, dim2, dim3) <= 0:
            raise ValueError(f"GEMM dimensions must be positive: {(dim1, dim2, dim3)}")
        api = self._api
        op = api.LocalOp(
            name=name or "gemm", kind=api.OpKind.BATCHED_GEMM, phase=self._phase(),
            tensors=(
                api.TensorSpec(api.TensorRole.INPUT, (1, int(dim1), int(dim2)), self.dtype),
                api.TensorSpec(api.TensorRole.WEIGHT, (int(dim2), int(dim3)), self.dtype),
                api.TensorSpec(api.TensorRole.OUTPUT, (1, int(dim1), int(dim3)), self.dtype),
            ),
            attrs={"gemm_selection_objective": self.gemm_selection_objective},
        )
        problem = _extended._problem_spec(op, batched=True)
        kernel = _extended._kernel_from_template(_replace(template, split_k=1, reduction_scheme="none"))
        profile = _effective._estimate_gemm_with_kernel(
            problem=problem, footprint=footprint_from_tensors(op), kernel=kernel, hardware=self.hardware,
            batched=True, warnings=[])
        return self._result(profile)

    def selected_gemm_template(self, *, dim1, dim2, dim3, name="", role="forward", allow_gemv=True):
        from opmodel.models.extended_roofline import _kernel_template_catalog
        result = self.predict_batched_gemm(dim1=dim1, dim2=dim2, dim3=dim3,
                                           name=name, gemm_role=role, allow_gemv=allow_gemv)
        if result.diagnostics.get("gemv"):
            return None
        selected = result.diagnostics["gemm_selection"]["selected_template"]["name"]
        return next(t for t in _kernel_template_catalog(self.dtype, self.hardware) if t.name == selected)

    @staticmethod
    def _is_cublas_gemv(*, dim1: int, dim3: int, batch: int, weight_rank: int, name: str = "", architecture: str = "sm80") -> bool:
        from opmodel import settings as _settings

        if architecture == "sm90":
            return False
        if _settings.get("OPMODEL_GEMV_RULE") == "trtllm":
            lowered = name.lower()
            is_lm_head = "linear_softmax" in lowered or "lm_head" in lowered or "logits" in lowered

            max_m = _settings.get("OPMODEL_GEMV_MAX_M")
            return weight_rank == 2 and batch == 1 and 1 <= dim1 <= max_m and not is_lm_head
        max_n = _settings.get("OPMODEL_GEMV_MAX_N")
        return max_n > 0 and weight_rank == 2 and dim1 == 1 and batch == 1 and dim3 <= max_n

    def predict_softmax(
        self, *, rows: int, row_size: int, name: str, phase: str, dtype_bytes: float | None = None
    ) -> OpModelResult:
        if rows <= 0 or row_size <= 0:
            raise ValueError("softmax rows and row_size must be positive")
        api = self._api
        attrs: dict[str, Any] = {"row_size": int(row_size)}
        shape = (int(rows), int(row_size))

        dtype = self.dtype if dtype_bytes is None else self._dtype_from_precision_bytes(float(dtype_bytes))
        op = api.LocalOp(
            name=name or "softmax",
            kind=api.OpKind.SOFTMAX,
            phase=self._phase(phase),
            tensors=(
                api.TensorSpec(api.TensorRole.INPUT, shape, dtype),
                api.TensorSpec(api.TensorRole.OUTPUT, shape, dtype),
            ),
            attrs=attrs,
        )
        return self._result(self.model.predict(op, self.hardware))

    def predict_flash_attention(
        self,
        *,
        batch: int,
        query_heads: int,
        kv_heads: int,
        seq_q: int,
        seq_kv: int,
        head_dim: int,
        value_dim: int,
        tile_size: int,
        name: str = "flash_attention_prefill",
    ) -> OpModelResult:
        raise ValueError("predict_flash_attention is obsolete; use the FlashAttention rebuild (flash_attention_adapter.py + flash_attention.py)")

    def _phase(self, phase: str | None = None):
        api_phase = self._api.Phase
        normalized = str(phase or self.run_type).lower()
        if normalized == "prefill":
            return api_phase.PREFILL
        if normalized == "decode":
            return api_phase.DECODE
        if self.run_type == "inference":
            return api_phase.INFERENCE
        return api_phase.TRAIN_FWD

    def _result(self, profile: Any) -> OpModelResult:
        memory = profile.memory_access
        breakdown = profile.energy_breakdown
        diagnostics = dict(profile.diagnostics)
        issued_flops = float(diagnostics.get("issued_flops", profile.flops))
        return OpModelResult(
            latency_s=float(profile.latency_s),
            energy_j=float(profile.energy_j),
            flops=issued_flops,
            implementation=str(profile.implementation),
            memory_accesses=(
                int((memory.register_read_bytes or 0) + (memory.register_write_bytes or 0)),
                int((memory.sram_read_bytes or 0) + (memory.sram_write_bytes or 0)),
                int((memory.l2_read_bytes or 0) + (memory.l2_write_bytes or 0)),
                int((memory.hbm_read_bytes or 0) + (memory.hbm_write_bytes or 0)),
            ),
            tile_dims=self._tile_dims(profile.diagnostics),
            energy_breakdown={
                "compute": float(breakdown.compute_j),
                "hbm": float(breakdown.hbm_j),
                "l2": float(breakdown.l2_j),
                "sram": float(breakdown.sram_j),
                "register": float(breakdown.register_j),
                "static": float(breakdown.static_j),
            },
            diagnostics=diagnostics,
        )

    @staticmethod
    def _tile_dims(diagnostics: Mapping[str, Any]):
        kernel = diagnostics.get("kernel", {}) if isinstance(diagnostics, Mapping) else {}
        cta = kernel.get("cta_tile", {}) if isinstance(kernel, Mapping) else {}
        if isinstance(cta, Mapping) and cta:
            tile = (
                int(cta.get("m", 0) or 0),
                int(cta.get("k", 0) or 0),
                int(cta.get("n", 0) or 0),
            )
        elif isinstance(kernel, Mapping):
            tile = (
                int(kernel.get("block_q", 0) or 0),
                int(kernel.get("head_dim", 0) or 0),
                int(kernel.get("block_k", 0) or 0),
            )
        else:
            tile = (0, 0, 0)
        return ((0, 0, 0), (0, 0, 0), tile)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        return parsed if parsed > 0 else None

    @staticmethod
    def _optional_nonnegative_cycles(value: Any) -> float | None:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0:
            raise ValueError("cycle counts must be finite and non-negative")
        return int(parsed) if parsed.is_integer() else parsed

    @staticmethod
    def _optional_nonnegative_int(value: Any) -> int | None:
        if value is None:
            return None
        parsed = int(value)
        if parsed < 0:
            raise ValueError("cycle counts must be non-negative")
        return parsed

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)
