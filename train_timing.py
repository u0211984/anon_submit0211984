import math
import os
import warnings
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, Tuple, Optional, List, Sequence
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from contextlib import contextmanager
from activity import MemoryTraffic, OperationActivity, ZERO_ACTIVITY
from llm_execution import ExecutionMode, LLMExecutionDispatcher
from program import _env_flag
from program.ir import EmbeddedCollective
from program.collective_launch import is_nccl_call
from program.block import BlockTemplate
from program.workload import BlockTemplates, OverlapSpec, WorkloadSpec
from program.execution_energy import (
    apply_training_graph_accounting,
    combine_execution_energy,
    write_energy_report,
)
import llm_util
import kernel_trace
import sim_settings
from opmodel import settings as _settings
from memory_estimation import MemoryEstimator
from base_timing import TimeCalculation
from timing_model import CollectiveType, CommSpec, DirectionTiming, OperationTiming, OperationGroup
import yaml

def _vllm_pointwise() -> bool:
    import os as _os
    return _env_flag("RAPID_VLLM_POINTWISE")

def _prefill_logits_last_token() -> bool:
    import os as _os
    return _env_flag("RAPID_PREFILL_LOGITS_LAST_TOKEN")

def _decode_batched_attention() -> bool:
    import os as _os
    return _env_flag("RAPID_DECODE_BATCHED_ATTENTION")

class ParallelismMode(Enum):
    TENSOR = "tensor"
    TENSOR_SEQUENCE = "tensor_sequence"
    CONTEXT = "context"
    TENSOR_CONTEXT = "tensor_context"
    TENSOR_CONTEXT_HYBRID = "tensor_context_hybrid"
    SINGLE = "single"

SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT = kernel_trace.SILU_FWD_FLOPS

SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT = 20
GELU_FORWARD_FLOPS_PER_ELEMENT = 10
GELU_BACKWARD_FLOPS_PER_ELEMENT = 20

LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT = 7
LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT = 14
LAYER_NORM_FORWARD_MEM_ACCESSES = 2
LAYER_NORM_BACKWARD_MEM_ACCESSES = 4
SOFTMAX_FORWARD_FLOPS_PER_ELEMENT = 7
SOFTMAX_FORWARD_MEM_ACCESSES = 4
SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT = 4
SOFTMAX_BACKWARD_MEM_ACCESSES = 3

ALL_REDUCE = CollectiveType.ALL_REDUCE
ALL_GATHER = CollectiveType.ALL_GATHER
REDUCE_SCATTER = CollectiveType.REDUCE_SCATTER

CROSS_ENTROPY_TRAIN_BYTES = 4.0
ALL_TO_ALL = CollectiveType.ALL_TO_ALL
PIPELINE = CollectiveType.PIPELINE

COMM_RULE_DEFAULT_KEY = "__default__"
COMMUNICATION_RULES: Dict[
    ParallelismMode, Dict[str, Dict[str, Optional[Dict[str, object]]]]
] = {
    ParallelismMode.TENSOR: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm1': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm2': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'qkv_proj': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'attention': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'output_proj': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'MLP': {
            'forward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_REDUCE, 'participants': 'tp', 'interconnect': 'tp'},
        },
    },
    ParallelismMode.TENSOR_SEQUENCE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
        'layernorm1': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm2': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'qkv_proj': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'attention': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'output_proj': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'MLP': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
    },
    ParallelismMode.CONTEXT: {
        COMM_RULE_DEFAULT_KEY: {
            'forward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'cp', 'interconnect': 'cp'},
        },
        'attention': {'forward': None, 'backward': {'kind': REDUCE_SCATTER, 'participants': 'cp', 'interconnect': 'cp'}},
        'output_proj': {'forward': None, 'backward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'}},
        'qkv_proj': {'forward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'}, 'backward': None},
    },
    ParallelismMode.TENSOR_CONTEXT_HYBRID: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
        'layernorm1': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'layernorm2': {
            'forward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'MLP': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
        'attention': {'forward': None, 'backward': {'kind': REDUCE_SCATTER, 'participants': 'cp', 'interconnect': 'cp'}},
        'output_proj': {
            'forward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
            'backward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'},
        },
        'qkv_proj': {
            'forward': {'kind': ALL_GATHER, 'participants': 'cp', 'interconnect': 'cp'},
            'backward': {'kind': REDUCE_SCATTER, 'participants': 'tp', 'interconnect': 'tp'},
        },
    },
    ParallelismMode.SINGLE: {
        COMM_RULE_DEFAULT_KEY: {'forward': None, 'backward': None},
    },
}

COMMUNICATION_RULES[ParallelismMode.TENSOR_CONTEXT] = {
    op: {
        direction: (
            COMMUNICATION_RULES[ParallelismMode.CONTEXT].get(op, {}).get(direction) or rule
            if op != COMM_RULE_DEFAULT_KEY else rule
        )
        for direction, rule in directions.items()
    }
    for op, directions in COMMUNICATION_RULES[ParallelismMode.TENSOR].items()
}

_MOE_ROUTER_RULE = {
    'forward': None,
    'backward': None,
}
_MOE_DISPATCH_RULE = {
    'forward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
    'backward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
}
_MOE_COMBINE_RULE = {
    'forward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
    'backward': {
        'kind': ALL_TO_ALL,
        'participants': 'ep',
        'interconnect': 'ep',
    },
}
MOE_COMMUNICATION_RULES: Dict[
    ParallelismMode, Dict[str, Dict[str, Optional[Dict[str, object]]]]
] = {
    mode: {
        'router': _MOE_ROUTER_RULE,
        'moe_dispatch': _MOE_DISPATCH_RULE,
        'moe_combine': _MOE_COMBINE_RULE,
    }
    for mode in ParallelismMode
}
class GemmType(Enum):
    ATTENTION_SCORE = "attention_score"
    ATTENTION_OUTPUT = "attention_output"
    MLA_DOWN_PROJ = "mla_down_proj"
    QKV = "qkv"
    OUT_PROJ = "out_proj"
    FFN1 = "ffn1"
    FFN2 = "ffn2"
    LINEAR_SOFTMAX = "linear_softmax"
    LAYER_NORM_1 = "layer_norm1"

@dataclass(frozen=True)
class GemmShardSpec:
    batch: int
    m: int
    k: int
    n: int
    shard_batch: int
    shard_m: int
    shard_k: int
    shard_n: int
    batch_scale: float

    def output_elements(self) -> int:
        return int(self.shard_batch * self.shard_m * self.shard_n)

@dataclass(frozen=True)
class MoEImbalanceProfile:
    hot_expert_factor: float
    cold_expert_factor: float
    hot_rank_factor: float
    total_experts: int
    experts_per_rank: int

@dataclass(frozen=True)
class MoECommDecomposition:
    total_bytes: int
    base_all_to_all_bytes: int
    residual_total_bytes: int
    residual_per_sender_bytes: int
    residual_sender_count: int
    routing_group: int
    routing_mode: str
    hot_rank_factor: float
    cold_rank_factor: float

class TimeCalculationLLM(TimeCalculation):
    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
        execution_mode = self._derive_execution_mode(hw_config)
        astra_policy = self._map_execution_mode_to_policy(execution_mode)

        super().__init__(
            hw_config,
            model_config,
            mode,
            astra_policy_override=astra_policy,
        )

        self.output_dir = os.path.abspath(output_dir) if output_dir else os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)
        overlap_cfg = getattr(getattr(hw_config, "network_layout", None), "overlap_config", None)
        self.tp_overlap = float(overlap_cfg.tp_overlap)
        self.tp_sp_overlap = float(overlap_cfg.tp_sp_overlap)
        self.cp_overlap = float(overlap_cfg.cp_overlap)
        self._generate_graphs = _env_flag("RAPID_VISUALIZE_GRAPHS")
        self.persist_astrasim_artifacts = _env_flag("RAPID_PERSIST_ASTRASIM_ARTIFACTS")
        self._debug_memory = _env_flag("RAPID_DEBUG_MEMORY")
        self._memory_breakdown_debug = None
        self.execution_mode = execution_mode

        self.model_type = self.model.model_type
        self.qwen2_bias_mode = getattr(self.model, "qwen2_bias_mode", None)
        self.max_position_embeddings = getattr(self.model, "max_position_embeddings", None)
        self.tied_embeddings = getattr(self.model, "tied_embeddings", True)
        self.disable_embedding_unembedding = bool(
            getattr(self.model, "disable_embedding_unembedding", False)
        )
        self.disable_kv_cache = str(getattr(self, "mode", "")).upper() == "VIT"

        if self.disable_kv_cache and self.execution_mode == ExecutionMode.FULL_ASTRASIM_FLATTENED:
            raise ValueError(
                "ViT inference does not support execution_backend.astrasim.mode='full_astrasim_flattened'."
            )

        self.estimate_memory = bool(
            getattr(hw_config.sw_config, "estimate_memory", True)
        )

        self.validate_graph = bool(
            getattr(hw_config.sw_config, "validate_graph", False)
        ) or _env_flag("RAPID_VALIDATE_GRAPH")
        self.memory_capacity_exceeded = False
        self.memory_capacity_violation_gb = 0.0
        self.zero3_ephemeral_peak_bytes = 0.0

        self.workload: Optional[WorkloadSpec] = None
        self.workload_no_dp: Optional[WorkloadSpec] = None
        self.pipeline_interconnect: Optional[Dict[str, Tuple[float, float]]] = None

        self.transformer_blocks: Optional[BlockTemplates] = None
        self.transformer_analytical_time_forward: Optional[float] = None
        self.transformer_analytical_time_backward: Optional[float] = None
        self.transformer_analytical_time_backward_combined: Optional[float] = None
        self.transformer_astrasim_time_forward: Optional[float] = None
        self.transformer_astrasim_time_backward: Optional[float] = None
        self.transformer_astrasim_time_forward_moe: Optional[float] = None
        self.transformer_astrasim_time_backward_moe: Optional[float] = None
        self.transformer_astrasim_per_rank_forward: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_backward: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_forward_moe: Optional[List[float]] = None
        self.transformer_astrasim_per_rank_backward_moe: Optional[List[float]] = None
        if self.run_type == "training":
            self._training_allreduce_curve()
        if self.l2_bypass:
            self._validate_l2_bypass()

    def _validate_l2_bypass(self) -> None:
        def refuse(reason):
            raise NotImplementedError(f"tech_param.SRAM-L2.l2_bypass: {reason}")
        core = self.hw_config.tech_config.core
        if getattr(self.hw_config, "counts_only", False):
            refuse("counts-only hardware is not described")
        if str(getattr(core, "architecture", None) or "sm80") != "sm80":
            refuse("described for sm80 only")
        if self.resolved_settings["gemm_timing_backend"] != "opmodel":
            refuse("requires sw_param.gemm_timing_backend: opmodel")
        if self.num_levels != 4:
            refuse("requires the register / SMEM / L2 / DRAM memory hierarchy")
        if _settings.get("RAPID_FLASH_SPILLS"):
            refuse("FlashAttention spill traffic (kernel_rules.flash_spills) is not described")
        moe = bool(getattr(self, "use_moe", False))
        mla = str(getattr(self, "attention_type", "mha")).lower() == "mla"
        if self.run_type == "training":
            from program import megatron
            if not (megatron.kernel_timing(self) and megatron.flattened(self)):
                refuse("training is described for the Megatron-Core kernel description only: implementation fused, "
                       "l2_model.step_simulation true, implementation.options.megatron_timing=kernel_trace, "
                       "execution_backend.astrasim.mode=full_astrasim_flattened")
            if mla and not megatron.mla_trace(self):
                refuse("MLA training is described for implementation.options.mla_training=megatron_trace only")
        elif self.run_type == "inference":
            if kernel_trace.implementation(self) != "fused" or self.implementation_options.get("flash_source") != "vllm":
                refuse("inference is described for the vLLM (fused) kernel description only")
            if not bool(getattr(self.model, "inference_decode_only", False)):
                refuse("inference is described for decode only (inference_param.decode_only: true); prefill is not")
            if moe or mla or self._source_moe():
                if not (moe and mla and str(getattr(self, "model_type", "")).lower() in ("deepseek_v2", "deepseek_v3")
                        and self._source_moe()):
                    refuse("MoE / MLA inference is described for DeepSeek MoE + MLA on the vLLM source MoE path only "
                           "(mla_inference=vllm_trace); dense MLA and qwen3_moe are not")
                if any(int(getattr(self, axis, 1) or 1) != 1 for axis in ("tp", "cp", "pp")):
                    refuse("MoE inference is described for tp = cp = pp = 1 (one GPU or attention DP x EP, moe_dp >= 1)")
                if self._implementation_option("vllm_all2all_backend", "naive") != "naive":
                    refuse("MoE inference is described for the naive all-to-all backend (vllm_all2all_backend: naive)")
                if str(getattr(self.model, "moe_grouped_traffic", "logical")) != "cta_operand_proxy":
                    refuse("MoE inference is described for moe_grouped_traffic: cta_operand_proxy (L1-miss bytes)")
            elif any(int(getattr(self, axis, 1) or 1) != 1 for axis in ("tp", "cp", "pp", "moe_dp")):
                refuse("dense inference is described for one GPU (tp = cp = pp = moe_dp = 1)")
        else:
            refuse(f"run_type {self.run_type!r} is not described")

    def _training_allreduce_curve(self):
        if _settings.get("RAPID_ALLREDUCE_MODEL") != "measured_curve" or (int(self.tp) <= 1 and int(self.dp) <= 1):
            return
        if self.tp_sp and int(self.tp) > 1:
            raise ValueError("measured TP all-reduce curve: sequence parallelism issues reduce-scatter/all-gather")
        from program.link_energy import resolve_rule
        from program.allreduce_curve import load_curve, series_key
        for axis in (("tp",) if int(self.tp) > 1 else ()) + (("dp",) if int(self.dp) > 1 else ()):
            dimension = self.hw_config.network_layout.parallelism_map[axis]
            if resolve_rule(self.hw_config, dimension) != "per_payload_byte":
                raise ValueError("measured all-reduce requires per_payload_byte link energy")
        sw = self.hw_config.sw_config
        basis = _settings.get("RAPID_ALLREDUCE_TIMING_BASIS")
        series = load_curve(_settings.get("RAPID_ALLREDUCE_CURVE"),
                            include_unverified=_settings.get("RAPID_ALLREDUCE_INCLUDE_UNVERIFIED"),
                            timing_basis=basis)
        key = series_key("nccl", self.tp, sw.allreduce_dtype)
        if basis == "kernel_union" and key not in series:
            raise ValueError(f"measured all-reduce curve missing series {key!r}")
        self.network_model.allreduce_curve = dict(
            allreduce_model="measured_curve", allreduce_curve_series=series,
            allreduce_dtype=sw.allreduce_dtype, allreduce_timing_basis=basis)
        if int(self.dp) > 1:
            self.network_model.allreduce_curve["dp_allreduce_dtype"] = sw.grad_allreduce_dtype
            if _settings.get("RAPID_ALLREDUCE_RSAG_RULE") == "allreduce_bus_bandwidth":
                self.network_model.allreduce_curve.update(
                    rsag_rule="allreduce_bus_bandwidth", dp_reduce_scatter_dtype=sw.grad_allreduce_dtype,
                    dp_all_gather_dtype=sw.param_dtype)

    def _sequence_parallel_degree(self) -> int:
        degree = 1
        if self.tp_sp and self.tp > 1:
            degree *= self.tp
        if self.cp > 1:
            degree *= self.cp
        return degree

    def get_parallelism_mode(self):
        if self.tp_sp and self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR_SEQUENCE
        elif self.tp > 1 and self.cp == 1:
            return ParallelismMode.TENSOR
        elif self.cp > 1 and self.tp == 1:
            return ParallelismMode.CONTEXT
        elif self.cp > 1 and self.tp > 1:
            return (ParallelismMode.TENSOR_CONTEXT_HYBRID if self.tp_sp
                    else ParallelismMode.TENSOR_CONTEXT)
        else:
            return ParallelismMode.SINGLE

    def is_moe_layer(self, layer_idx: int) -> bool:
        mask = getattr(self, "moe_layer_mask", None)
        if not mask:
            return False
        if layer_idx < 0 or layer_idx >= len(mask):
            return False
        return bool(mask[layer_idx])

    @property
    def experts_per_gpu(self) -> int:
        if not self.use_moe or self.moe_num_experts <= 1:
            return 0
        moe_ranks = self._moe_routing_group()
        if self.moe_num_experts % moe_ranks != 0:
            if self._moe_allow_padding() and _env_flag("RAPID_ALLOW_MOE_EXPERT_PADDING"):
                if not getattr(self, "_moe_expert_padding_warned", False):
                    warnings.warn(
                        "MoE expert count is not divisible by the routing group; "
                        "padding experts for inference. Set RAPID_ALLOW_MOE_EXPERT_PADDING=0 "
                        "to enforce divisibility.",
                        stacklevel=2,
                    )
                    self._moe_expert_padding_warned = True
                return max(1, math.ceil(self.moe_num_experts / moe_ranks))
            raise ValueError(
                "moe_num_experts must be divisible by the MoE routing group size. "
                f"moe_num_experts={self.moe_num_experts}, moe_group={moe_ranks}, "
                f"tp={self.tp}, ep={self.ep}."
            )
        return max(1, math.ceil(self.moe_num_experts / moe_ranks))

    @experts_per_gpu.setter
    def experts_per_gpu(self, value: int) -> None:
        self._experts_per_gpu_base_value = value

    def _moe_tokens_owner(self, batch_size: int, seq_len: int) -> int:
        seq_divisor = max(1, int(self.cp)) if self.cp > 1 else 1
        tokens = int(batch_size) * int(math.ceil(float(seq_len) / float(seq_divisor)))
        return tokens * self._moe_dp_attention_groups()

    def _moe_dp_attention_groups(self) -> int:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        if run_type != "inference" or not _env_flag("RAPID_INFER_DP_ATTENTION"):
            return 1
        return max(1, int(getattr(self, "moe_dp", 1) or 1))

    def _moe_tp_replicated_combine(self) -> bool:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        return (
            run_type == "inference"
            and _env_flag("RAPID_MOE_TP_REPLICATED_COMBINE")
            and int(self.tp) > 1
            and int(getattr(self, "moe_dp", 1) or 1) == 1
        )

    def _moe_forward_comm(self, tokens_owner: int, tokens_dispatched_pool: int, experts_per_rank: int, label: str):
        if self._moe_tp_replicated_combine():
            comb_bytes = int(math.ceil(self.precision.activations * float(tokens_owner) * self.hidden_dim))
            comb_time = self.network_model.collective(
                kind=ALL_REDUCE,
                size_bytes=comb_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label=f"{label}_moe_combine_tp_all_reduce",
                axis="tp",
            )
            return 0, 0.0, comb_bytes, comb_time
        if _env_flag("RAPID_MOE_ACTIVE_EXPERTS"):
            groups = self._moe_dp_attention_groups()
            tokens_dispatched_pool = int(tokens_owner) * groups * int(self.moe_top_k)
        disp_bytes = self._moe_dispatch_bytes_per_rank(tokens_dispatched_pool)
        disp_time = self._moe_comm_time(
            self._moe_comm_decomposition(disp_bytes, experts_per_rank), debug_label=f"{label}_moe_dispatch_f", axis=None
        )
        comb_time = self._moe_comm_time(
            self._moe_comm_decomposition(disp_bytes, experts_per_rank), debug_label=f"{label}_moe_combine_f", axis=None
        )
        return disp_bytes, disp_time, disp_bytes, comb_time

    def _moe_permute_kernels_time(self, tokens_group: int) -> float:
        if not _env_flag("RAPID_MOE_PERMUTE_KERNELS"):
            return 0.0
        groups = self._moe_dp_attention_groups()
        share = max(1, int(self._moe_local_share_divisor()))
        slots_local = math.ceil(float(tokens_group) * groups * int(self.moe_top_k) / share)
        mem_bytes = 2.0 * (slots_local + float(tokens_group)) * self.hidden_dim * self.precision.activations

        return self.roofline(0, mem_bytes, name="pointwise-router-f", mem_level=self.num_levels - 1) + 3 * self.O

    def _moe_dispatch_bytes_per_rank(self, tokens_dispatched_pool: int) -> int:
        groups = self._moe_dp_attention_groups()
        tokens = float(tokens_dispatched_pool)
        if _env_flag("RAPID_INFER_DP_ATTENTION") and str(
            getattr(getattr(self, "model", None), "run_type", "training")
        ).lower() == "inference":
            tokens = tokens / float(groups * max(1, int(self.tp)))
        return int(math.ceil(self.precision.activations * tokens * self.hidden_dim))

    def _moe_tokens_dispatched(self, tokens_owner: int) -> int:
        return int(tokens_owner) * int(self.moe_top_k)

    def _moe_tokens_shared(self, tokens_owner: int) -> int:
        return int(tokens_owner) // self._moe_dp_attention_groups() * int(self.n_shared_experts)

    def _moe_routing_group(self) -> int:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        if run_type == "inference":
            return max(1, int(self.tp) * int(self.ep))
        return max(1, int(self.ep))

    def _moe_local_share_divisor(self) -> int:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        if run_type == "inference":
            return max(1, self._moe_routing_group())
        return 1

    def _moe_allow_padding(self) -> bool:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        return run_type == "inference"

    def _moe_expert_imbalance_factor(self) -> float:
        factor = float(getattr(self, "expert_imbalance_factor", 1.0) or 1.0)
        if factor < 1.0:
            raise ValueError(
                "MoE expert_imbalance_factor must be >= 1.0. "
                "Use 1.0 to preserve the current perfectly balanced MoE behavior; "
                "values > 1.0 model one routed expert as hotter than the balanced average expert load."
            )
        return factor

    def _moe_imbalance_profile(self, experts_per_rank: int) -> MoEImbalanceProfile:
        experts_per_rank = max(1, int(experts_per_rank))
        total_experts = max(1, int(self.moe_num_experts))
        hot_expert_factor = self._moe_expert_imbalance_factor()
        if hot_expert_factor <= 1.0 or total_experts <= 1:
            return MoEImbalanceProfile(
                hot_expert_factor=1.0,
                cold_expert_factor=1.0,
                hot_rank_factor=1.0,
                total_experts=total_experts,
                experts_per_rank=experts_per_rank,
            )
        if hot_expert_factor > float(total_experts):
            raise ValueError(
                "MoE expert_imbalance_factor cannot exceed moe_num_experts under the "
                "one-hot-expert assumption. The simulator models one routed expert as hot "
                "and the remaining routed experts as uniformly colder. "
                f"expert_imbalance_factor={hot_expert_factor}, moe_num_experts={total_experts}."
            )
        cold_expert_factor = (float(total_experts) - hot_expert_factor) / float(total_experts - 1)
        hot_rank_factor = (
            hot_expert_factor + float(max(0, experts_per_rank - 1)) * cold_expert_factor
        ) / float(experts_per_rank)
        return MoEImbalanceProfile(
            hot_expert_factor=hot_expert_factor,
            cold_expert_factor=cold_expert_factor,
            hot_rank_factor=hot_rank_factor,
            total_experts=total_experts,
            experts_per_rank=experts_per_rank,
        )

    def _moe_comm_routing_mode(self) -> str:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        if run_type == "inference":
            return "tp_ep"
        return "ep"

    def _moe_comm_decomposition(self, total_bytes: int, experts_per_rank: int) -> MoECommDecomposition:
        total_bytes = int(math.ceil(float(total_bytes or 0.0)))
        routing_group = max(1, int(self._moe_routing_group()))
        routing_mode = self._moe_comm_routing_mode()
        profile = self._moe_imbalance_profile(experts_per_rank)
        cold_rank_factor = float(profile.cold_expert_factor)
        hot_rank_factor = float(profile.hot_rank_factor)
        if total_bytes <= 0:
            return MoECommDecomposition(
                total_bytes=0,
                base_all_to_all_bytes=0,
                residual_total_bytes=0,
                residual_per_sender_bytes=0,
                residual_sender_count=max(0, routing_group - 1),
                routing_group=routing_group,
                routing_mode=routing_mode,
                hot_rank_factor=hot_rank_factor,
                cold_rank_factor=cold_rank_factor,
            )
        if routing_group <= 1 or profile.hot_expert_factor <= 1.0:
            return MoECommDecomposition(
                total_bytes=total_bytes,
                base_all_to_all_bytes=total_bytes,
                residual_total_bytes=0,
                residual_per_sender_bytes=0,
                residual_sender_count=max(0, routing_group - 1),
                routing_group=routing_group,
                routing_mode=routing_mode,
                hot_rank_factor=1.0,
                cold_rank_factor=1.0,
            )

        base_all_to_all_bytes = int(math.floor(float(total_bytes) * max(0.0, min(1.0, cold_rank_factor))))
        base_all_to_all_bytes = max(0, min(total_bytes, base_all_to_all_bytes))
        residual_total_bytes = max(0, total_bytes - base_all_to_all_bytes)
        residual_sender_count = max(1, routing_group - 1) if residual_total_bytes > 0 else max(0, routing_group - 1)
        residual_per_sender_bytes = (
            int(math.ceil(float(residual_total_bytes) / float(residual_sender_count)))
            if residual_total_bytes > 0 and residual_sender_count > 0
            else 0
        )
        return MoECommDecomposition(
            total_bytes=total_bytes,
            base_all_to_all_bytes=base_all_to_all_bytes,
            residual_total_bytes=residual_total_bytes,
            residual_per_sender_bytes=residual_per_sender_bytes,
            residual_sender_count=residual_sender_count,
            routing_group=routing_group,
            routing_mode=routing_mode,
            hot_rank_factor=hot_rank_factor,
            cold_rank_factor=cold_rank_factor,
        )

    def _moe_comm_time(
        self,
        decomposition: MoECommDecomposition,
        *,
        debug_label: str,
        axis: Optional[str],
    ) -> float:
        base_time = 0.0
        if decomposition.base_all_to_all_bytes > 0 and decomposition.routing_group > 1:
            base_time = self.network_model.collective(
                kind=ALL_TO_ALL,
                size_bytes=decomposition.base_all_to_all_bytes,
                participants=decomposition.routing_group,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0,
                debug_label=f"{debug_label}_base_all_to_all",
                axis=axis,
            )
        residual_time = 0.0
        if decomposition.residual_per_sender_bytes > 0:
            residual_time = self.network_model.collective(
                kind=PIPELINE,
                size_bytes=decomposition.residual_per_sender_bytes,
                participants=2,
                ib=self.links["ep"].bandwidth,
                ll=self.links["ep"].latency,
                local_bytes=0,
                debug_label=f"{debug_label}_residual_p2p",
                axis=None,
            )
        return max(base_time, residual_time)

    def _megatron_moe_dispatch_specs(self, direction: str) -> Tuple[CommSpec, ...]:
        from program import megatron
        if not (bool(getattr(self, "use_moe", False)) and int(self.ep) > 1 and megatron.enabled(self)):
            return ()
        opts = self.implementation_options
        tokens = int(self.micro_batch) * int(self.seq_len) // max(1, int(self.cp))
        specs = []
        if opts.get("megatron_moe_count_allgather", True) and direction == "forward":
            specs.append(CommSpec(name=f"moe_dispatch_{direction}_token_count_all_gather", kind=ALL_GATHER,
                                  size_bytes=8 * int(self.moe_num_experts), participants=int(self.ep),
                                  interconnect="ep", extra={"placement": "post"}))
        if opts.get("megatron_moe_probs_alltoall", True):
            specs.append(CommSpec(name=f"moe_dispatch_{direction}_probs_all_to_all", kind=ALL_TO_ALL,
                                  size_bytes=int(math.ceil(float(self.precision.activations) * tokens * int(self.moe_top_k))),
                                  participants=int(self.ep), interconnect="ep", extra={"placement": "post"}))
        return tuple(specs)

    def _make_moe_comm_specs(
        self,
        op_name: str,
        direction: str,
        total_bytes: float,
    ) -> Tuple[CommSpec, ...]:
        if self._moe_tp_replicated_combine():
            if op_name == "moe_dispatch" or not total_bytes or direction != "forward":
                return ()
            return (
                CommSpec(
                    name=f"{op_name}_{direction}_tp_all_reduce",
                    kind=ALL_REDUCE,
                    size_bytes=int(total_bytes),
                    participants=int(self.tp),
                    interconnect="tp",
                    extra={"placement": "post", "moe_component": "tp_replicated_combine"},
                ),
            )
        decomposition = self._moe_comm_decomposition(total_bytes, self.experts_per_gpu)
        if decomposition.total_bytes <= 0:
            return ()
        parallel_group = f"{op_name}_{direction}_moe_parallel_group"
        specs: List[CommSpec] = []
        if decomposition.base_all_to_all_bytes > 0 and decomposition.routing_group > 1:
            specs.append(
                CommSpec(
                    name=f"{op_name}_{direction}_base_all_to_all",
                    kind=ALL_TO_ALL,
                    size_bytes=decomposition.base_all_to_all_bytes,
                    participants=decomposition.routing_group,
                    interconnect="ep",
                    extra={
                        "placement": "post",
                        "parallel_group": parallel_group,
                        "moe_component": "base_all_to_all",
                        "moe_routing_mode": decomposition.routing_mode,
                    },
                )
            )
        if decomposition.residual_per_sender_bytes > 0:
            specs.append(
                CommSpec(
                    name=f"{op_name}_{direction}_residual_p2p",
                    kind=PIPELINE,
                    size_bytes=decomposition.residual_per_sender_bytes,
                    participants=2,
                    interconnect="ep",
                    extra={
                        "placement": "post",
                        "parallel_group": parallel_group,
                        "moe_component": "residual_p2p",
                        "moe_routing_mode": decomposition.routing_mode,
                    },
                )
            )
        return tuple(specs)

    def _moe_apply_expert_imbalance(self, tokens_local: int, experts_per_rank: int) -> int:
        tokens_local = int(tokens_local)
        experts_per_rank = max(1, int(experts_per_rank))
        profile = self._moe_imbalance_profile(experts_per_rank)
        if profile.hot_expert_factor <= 1.0 or tokens_local <= 0:
            return tokens_local
        tokens_per_expert = int(tokens_local // experts_per_rank)
        if tokens_per_expert <= 0:
            return tokens_local
        hot_tokens = max(tokens_per_expert, int(math.ceil(float(tokens_per_expert) * profile.hot_expert_factor)))
        cold_tokens = max(0, int(math.floor(float(tokens_per_expert) * profile.cold_expert_factor)))
        inflated_tokens = hot_tokens + cold_tokens * max(0, experts_per_rank - 1)
        return max(tokens_local, inflated_tokens)

    def _moe_tokens_local(self, tokens_dispatched: int) -> int:
        share_divisor = self._moe_local_share_divisor()
        tokens_local = int(math.ceil(float(tokens_dispatched) / float(max(1, share_divisor))))
        experts_per_rank = self.experts_per_gpu
        if experts_per_rank > 0:
            return self._moe_apply_expert_imbalance(tokens_local, experts_per_rank)
        return tokens_local

    def _moe_balanced_routed_tokens_per_expert(
        self,
        batch_size: int,
        seq_len: int,
        *,
        allow_padding: bool = False,
    ) -> Tuple[int, int, int, int, int]:
        tokens_owner = self._moe_tokens_owner(batch_size, seq_len)
        tokens_dispatched = self._moe_tokens_dispatched(tokens_owner)
        share_divisor = self._moe_local_share_divisor()
        if share_divisor <= 0:
            share_divisor = 1
        if allow_padding:
            tokens_dispatched = int(math.ceil(float(tokens_dispatched) / float(share_divisor)) * share_divisor)
        else:
            if tokens_dispatched % share_divisor != 0:
                raise ValueError(
                    "MoE routed tokens must divide evenly across the MoE routing group for batched expert GEMMs. "
                    f"tokens_dispatched={tokens_dispatched}, share_divisor={share_divisor}, "
                    f"tp={self.tp}, ep={self.ep}, tokens_owner={tokens_owner}, "
                    f"batch_size={batch_size}, seq_len={seq_len}, top_k={self.moe_top_k}, cp={self.cp}."
                )
        tokens_local = tokens_dispatched // share_divisor
        experts_per_rank = self.experts_per_gpu
        if experts_per_rank <= 0:
            raise ValueError(
                "MoE experts_per_rank must be positive when use_moe is enabled. "
                f"experts_per_rank={experts_per_rank}, moe_num_experts={self.moe_num_experts}, "
                f"tp={self.tp}, ep={self.ep}."
            )
        if allow_padding:
            if tokens_local % experts_per_rank != 0:
                tokens_local = int(
                    math.ceil(float(tokens_local) / float(experts_per_rank)) * experts_per_rank
                )
                tokens_dispatched = tokens_local * share_divisor
        else:
            if tokens_local % experts_per_rank != 0:
                raise ValueError(
                    "MoE routed tokens per rank must divide evenly across experts for batched expert GEMMs. "
                    f"tokens_local={tokens_local}, experts_per_rank={experts_per_rank} "
                    f"(moe_num_experts={self.moe_num_experts}, moe_group={self._moe_routing_group()}). "
                    f"tokens_owner={tokens_owner}, tokens_dispatched={tokens_dispatched}, "
                    f"batch_size={batch_size}, seq_len={seq_len}, top_k={self.moe_top_k}, cp={self.cp}."
                )
        tokens_per_expert = tokens_local // experts_per_rank
        return tokens_owner, tokens_dispatched, tokens_local, experts_per_rank, tokens_per_expert

    def _moe_routed_tokens_per_expert(
        self,
        batch_size: int,
        seq_len: int,
        *,
        allow_padding: bool = False,
    ) -> Tuple[int, int, int, int, int]:
        (
            tokens_owner,
            _tokens_dispatched_balanced,
            tokens_local_balanced,
            experts_per_rank,
            _tokens_per_expert_balanced,
        ) = self._moe_balanced_routed_tokens_per_expert(
            batch_size,
            seq_len,
            allow_padding=allow_padding,
        )
        share_divisor = max(1, self._moe_local_share_divisor())
        tokens_local = self._moe_apply_expert_imbalance(tokens_local_balanced, experts_per_rank)
        tokens_dispatched = tokens_local * share_divisor

        tokens_per_expert = int(math.ceil(float(tokens_local) / float(experts_per_rank)))
        return tokens_owner, tokens_dispatched, tokens_local, experts_per_rank, tokens_per_expert

    def _moe_active_expert_bucket(
        self,
        tokens_owner: int,
        experts_per_rank: int,
    ) -> Optional[Tuple[int, int, str]]:
        if not _env_flag("RAPID_MOE_ACTIVE_EXPERTS"):
            return None
        total_experts = int(self.moe_num_experts)
        top_k = int(self.moe_top_k)
        experts_per_rank = int(experts_per_rank)
        if total_experts <= 1 or top_k <= 0 or experts_per_rank <= 0:
            return None
        share_divisor = max(1, int(self._moe_local_share_divisor()))
        slots_local = int(math.ceil(float(tokens_owner) * top_k / share_divisor))
        if slots_local <= 0:
            return None
        p_hit = min(1.0, float(top_k) / float(total_experts))
        t_src = float(slots_local) * total_experts / (top_k * experts_per_rank)
        expected_active = experts_per_rank * (1.0 - (1.0 - p_hit) ** t_src)
        active = max(1, min(experts_per_rank, slots_local, int(round(expected_active))))
        tokens_per_active = int(math.ceil(float(slots_local) / active))
        return (active, tokens_per_active, "uniform")

    def _moe_routed_compute_buckets(
        self,
        tokens_per_expert_balanced: int,
        experts_per_rank: int,
    ) -> List[Tuple[int, int, str]]:
        tokens_per_expert_balanced = int(tokens_per_expert_balanced)
        experts_per_rank = max(1, int(experts_per_rank))
        if tokens_per_expert_balanced <= 0:
            return []
        profile = self._moe_imbalance_profile(experts_per_rank)
        if profile.hot_expert_factor <= 1.0 or experts_per_rank <= 1:
            return [(experts_per_rank, tokens_per_expert_balanced, "uniform")]

        tokens_local_balanced = tokens_per_expert_balanced * experts_per_rank
        target_local_tokens = self._moe_apply_expert_imbalance(tokens_local_balanced, experts_per_rank)
        hot_tokens = max(
            tokens_per_expert_balanced,
            int(math.ceil(float(tokens_per_expert_balanced) * profile.hot_expert_factor)),
        )
        cold_tokens = max(
            0,
            int(math.floor(float(tokens_per_expert_balanced) * profile.cold_expert_factor)),
        )
        remaining_experts = max(0, experts_per_rank - 1)
        if remaining_experts > 0 and (hot_tokens + cold_tokens * remaining_experts) < target_local_tokens:
            cold_tokens = int(
                math.ceil(float(max(0, target_local_tokens - hot_tokens)) / float(remaining_experts))
            )

        buckets: List[Tuple[int, int, str]] = [(1, hot_tokens, "hot")]
        if remaining_experts > 0 and cold_tokens > 0:
            buckets.append((remaining_experts, cold_tokens, "cold"))
        return buckets

    def _param_stats_per_rank(
        self,
        hidden_dim: int,
        intermediate_size: int,
        vocab_size: int,
    ) -> Tuple[float, float, float, float, float]:
        tp = max(1, self.tp)
        pp = max(1, self.pp)

        params_per_layer_per_rank = sum(
            self.layer_params_per_rank(hidden_dim, intermediate_size, moe=False).values()
        )

        total_transformer_params = params_per_layer_per_rank * self.num_layers
        if pp == 1:
            transformer_params_local = total_transformer_params
        else:
            layers_per_stage = math.ceil(self.num_layers / pp)
            transformer_params_local = params_per_layer_per_rank * layers_per_stage
            transformer_params_local = min(transformer_params_local, total_transformer_params)

        if self.disable_embedding_unembedding:
            embedding_params = 0.0
            output_params = 0.0
        elif self._is_vit_model():
            embedding_params = float(
                llm_util.vit_patch_embed_param_count(
                    hidden_dim=hidden_dim,
                    patch_size=self.patch_size,
                    in_chans=self.in_chans,
                )
            )
            output_params = float(
                llm_util.vit_head_param_count(
                    hidden_dim=hidden_dim,
                    num_classes=getattr(self, "num_classes", 0),
                )
            )
        else:
            embedding_params = vocab_size * hidden_dim / tp
            output_params = 0.0 if self.tied_embeddings else embedding_params

        if pp == 1:
            total_params_per_rank = transformer_params_local + embedding_params + output_params
        else:
            total_params_per_rank = transformer_params_local + (embedding_params / pp) + (output_params / pp)

        max_layer_params = max(params_per_layer_per_rank, embedding_params, output_params)
        return (
            total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params,
            output_params,
        )

    def get_kv_shard_bytes(self) -> float:
        tp = max(1, int(self.tp))
        cp = max(1, int(self.cp))
        if self.attention_type == "mla":
            head_share = 1.0 / tp
        else:
            kv_heads = max(1, int(self.kv_heads))
            head_share = math.ceil(kv_heads / tp) / kv_heads
        return self.get_kv_size_bytes() * head_share / cp

    def get_kv_size_bytes(self) -> int:
        if getattr(self, "disable_kv_cache", False):
            return 0
        if self.attention_type == "mla":
            per_token_bytes = llm_util.attention_kv_cache_token_bytes(
                "mla",
                batch_size=self.micro_batch,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
                precision_bytes=self.precision.kv_cache,
                kv_lora_rank=self.kv_lora_rank,
                num_heads=self.num_heads,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
            )
            return int(math.ceil(float(self.seq_len) * float(per_token_bytes)))
        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        total_elements = 2 * self.seq_len * self.micro_batch * head_dim * self.kv_heads
        return total_elements * self.precision.kv_cache

    def _attention_param_components(self, hidden_dim: int) -> Tuple[int, int]:
        if self.attention_type == "mla":
            params = llm_util.mla_attention_param_sizes(
                hidden_dim,
                self.num_heads,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
            )
            return int(params["d_proj"] + params["u_proj_q"] + params["u_proj_kv"]), int(
                params["out_proj"]
            )
        _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
            hidden_dim,
            self.num_heads,
            self.kv_heads,
            head_dim=getattr(self, "head_dim", None),
        )
        qkv_params = int(hidden_dim * (q_size + 2 * kv_size))
        output_params = int(q_size * hidden_dim)
        return qkv_params, output_params

    def _attention_param_components_per_rank(self, hidden_dim: int) -> Tuple[float, float]:
        if self.attention_type == "mla":
            groups = llm_util.mla_attention_param_groups(
                hidden_dim,
                self.num_heads,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                self.qk_rope_head_dim,
                self.v_head_dim,
            )
            qkv_params = (
                groups["d_proj_q"]
                + groups["d_proj_kv"]
                + (
                    groups["u_proj_q_nope"]
                    + groups["u_proj_q_rope"]
                    + groups["u_proj_k_nope"]
                    + groups["u_proj_v"]
                )
                / float(max(1, self.tp))
            )
            output_params = groups["out_proj"] / float(max(1, self.tp))
            return float(qkv_params), float(output_params)
        qkv_params, output_params = self._attention_param_components(hidden_dim)
        return (
            float(qkv_params) / float(max(1, self.tp)),
            float(output_params) / float(max(1, self.tp)),
        )

    def _attention_param_total_per_rank(self, hidden_dim: int) -> float:
        qkv_params, output_params = self._attention_param_components_per_rank(hidden_dim)
        return float(qkv_params + output_params)

    @staticmethod
    def _derive_execution_mode(hw_config) -> ExecutionMode:
        backend = getattr(hw_config, "execution_backend", None)
        if not backend or getattr(backend, "model", "analytical").lower() != "astrasim":
            return ExecutionMode.ANALYTICAL

        mode_str = "hybrid"
        astra_cfg = getattr(backend, "astrasim", None)
        if astra_cfg and getattr(astra_cfg, "mode", None):
            mode_str = str(astra_cfg.mode).lower()

        for candidate in ExecutionMode:
            if candidate.value == mode_str:
                return candidate
        print(f"[WARN] Unknown execution mode '{mode_str}', defaulting to 'hybrid'.")
        return ExecutionMode.HYBRID

    @staticmethod
    def _map_execution_mode_to_policy(mode: ExecutionMode) -> str:
        if mode == ExecutionMode.ANALYTICAL:
            return 'analytical'
        if mode in (
            ExecutionMode.FULL_ASTRASIM_HIERARCHICAL,
            ExecutionMode.FULL_ASTRASIM_FLATTENED,
        ):
            return 'full'

        return 'hybrid'

    @staticmethod
    def _expand_gemm_descriptor(gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        if len(gemm) == 3:
            return 1, gemm[0], gemm[1], gemm[2]
        if len(gemm) == 4:
            return gemm[0], gemm[1], gemm[2], gemm[3]
        raise ValueError(f"Unsupported GEMM descriptor length: {len(gemm)}")

    def _shard_gemm_descriptor(self, gemm: Tuple[int, ...], gemm_type: Optional[GemmType]) -> GemmShardSpec:
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for GEMM sharding")

        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        shard_batch = batch
        shard_m = m
        shard_k = k
        shard_n = n
        batch_scale = 1.0

        tp = max(1, int(self.tp))
        cp = max(1, int(self.cp))
        mode = self.get_parallelism_mode()

        if mode in (ParallelismMode.TENSOR, ParallelismMode.TENSOR_SEQUENCE):
            if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
                shard_batch = math.ceil(batch / tp)
                batch_scale = 1.0 / tp
            elif gemm_type == GemmType.MLA_DOWN_PROJ:
                pass
            elif gemm_type in (GemmType.QKV, GemmType.FFN1):
                shard_n = math.ceil(n / tp)
            elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
                shard_k = math.ceil(k / tp)
            elif gemm_type == GemmType.LINEAR_SOFTMAX:
                shard_n = math.ceil(n / max(1, tp * cp))
            else:
                raise ValueError(f"Unsupported GEMM type for tensor parallelism: {gemm_type}")
        elif mode == ParallelismMode.CONTEXT:
            shard_m = math.ceil(m / cp)
            if gemm_type == GemmType.ATTENTION_SCORE:
                shard_n = math.ceil(n / cp)
            elif gemm_type == GemmType.ATTENTION_OUTPUT:
                shard_k = math.ceil(k / cp)
            elif gemm_type not in (
                GemmType.MLA_DOWN_PROJ,
                GemmType.QKV,
                GemmType.OUT_PROJ,
                GemmType.FFN1,
                GemmType.FFN2,
                GemmType.LINEAR_SOFTMAX,
            ):
                raise ValueError(f"Unsupported GEMM type for context parallelism: {gemm_type}")
        elif mode in (ParallelismMode.TENSOR_CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            shard_m = math.ceil(m / cp)
            if gemm_type == GemmType.ATTENTION_SCORE:
                shard_batch = math.ceil(batch / tp)
                batch_scale = 1.0 / tp
                shard_n = math.ceil(n / cp)
            elif gemm_type == GemmType.ATTENTION_OUTPUT:
                shard_batch = math.ceil(batch / tp)
                batch_scale = 1.0 / tp
                shard_k = math.ceil(k / cp)
            elif gemm_type == GemmType.MLA_DOWN_PROJ:
                pass
            elif gemm_type in (GemmType.QKV, GemmType.FFN1, GemmType.LINEAR_SOFTMAX):
                shard_n = math.ceil(n / tp)
            elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
                shard_k = math.ceil(k / tp)
            else:
                raise ValueError(f"Unsupported GEMM type for hybrid parallelism: {gemm_type}")
        elif mode == ParallelismMode.SINGLE:
            pass
        else:
            raise ValueError(f"Unsupported parallelism mode: {mode}")

        return GemmShardSpec(
            batch=batch,
            m=m,
            k=k,
            n=n,
            shard_batch=shard_batch,
            shard_m=shard_m,
            shard_k=shard_k,
            shard_n=shard_n,
            batch_scale=batch_scale,
        )

    def _gemm_output_elements(self, gemm: Tuple[int, ...], gemm_type: Optional[GemmType]) -> int:
        spec = self._shard_gemm_descriptor(gemm, gemm_type)
        return spec.output_elements()

    @classmethod
    def _effective_dims(cls, gemm: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        batch, m, k, n = cls._expand_gemm_descriptor(gemm)
        return batch, batch * m, k, n

    def _is_vit_model(self) -> bool:
        return llm_util.is_vit_style(getattr(self, "model_type", ""))

    def _uses_gated_mlp(self) -> bool:
        return llm_util.uses_gated_mlp(
            self.model_type,
            swiglu_mlp=getattr(self, "swiglu_mlp", False),
        )

    def _ffn1_output_dim(self, intermediate_size: int) -> int:
        return 2 * intermediate_size if self._uses_gated_mlp() else intermediate_size

    def _dense_ffn_param_components_per_rank(
        self, hidden_dim: int, intermediate_size: int
    ) -> Tuple[float, float]:
        tp = float(max(1, int(self.tp)))
        ffn1_params = float(self._ffn1_output_dim(intermediate_size)) * float(hidden_dim) / tp
        ffn2_params = float(intermediate_size) * float(hidden_dim) / tp
        return ffn1_params, ffn2_params

    def _moe_ffn_param_components_per_rank(
        self, hidden_dim: int, moe_intermediate_size: int
    ) -> Tuple[float, float]:
        tp = max(1, int(self.tp))
        ffn_proj_factor = 3 if self._uses_gated_mlp() else 2
        expert_param_size = ffn_proj_factor * moe_intermediate_size * hidden_dim
        moe_group = max(1, int(self._moe_routing_group()))
        per_expert = (expert_param_size / tp) if bool(getattr(self, "tp_ep", True)) else expert_param_size
        experts_per_rank = (self.moe_num_experts / moe_group) + self.n_shared_experts
        expert_params = float(per_expert) * float(experts_per_rank)
        router_params = float(hidden_dim) * float(self.moe_num_experts)
        return expert_params, router_params

    def layer_params_per_rank(
        self, hidden_dim: int, intermediate_size: int, *, moe: bool = False
    ) -> Dict[str, float]:
        qkv, output = self._attention_param_components_per_rank(hidden_dim)
        params: Dict[str, float] = {"qkv": float(qkv), "output": float(output)}
        family = str(self.model_type).lower()
        if family == "qwen2":
            params["qkv"] += (self.num_heads + 2 * self.kv_heads) * self.head_dim
        elif family == "qwen3":
            params["qkv"] += 2 * self.head_dim
        if moe:
            experts, router = self._moe_ffn_param_components_per_rank(hidden_dim, intermediate_size)
            params["experts"] = experts
            params["router"] = router
        else:
            ffn1, ffn2 = self._dense_ffn_param_components_per_rank(hidden_dim, intermediate_size)
            params["ffn1"] = ffn1
            params["ffn2"] = ffn2
        return params

    def zero_moe_layer_param_groups(self, hidden_dim: int, intermediate_size: int, *, moe: bool = False) -> Tuple[float, float]:
        params = self.layer_params_per_rank(hidden_dim, intermediate_size, moe=moe)
        if not moe:
            return sum(params.values()), 0.0
        routed_count = self.moe_num_experts / max(1, int(self._moe_routing_group()))
        total_count = routed_count + self.n_shared_experts
        routed = params["experts"] * routed_count / total_count if total_count else 0.0
        return sum(params.values()) - routed, routed

    def _vit_sequence_shard_degree(self) -> int:
        if not self._is_vit_model():
            return 1
        mode = self.get_parallelism_mode()
        if mode in (
            ParallelismMode.TENSOR_SEQUENCE,
            ParallelismMode.CONTEXT,
            ParallelismMode.TENSOR_CONTEXT,
            ParallelismMode.TENSOR_CONTEXT_HYBRID,
        ):
            return max(1, int(self._sequence_parallel_degree()))
        return 1

    def _vit_pool_token_count(self) -> int:
        if not self._is_vit_model():
            return 0
        seq_degree = self._vit_sequence_shard_degree()
        pooled_tokens = max(1, int(self.seq_len) - int(getattr(self, "num_prefix_tokens", 0)))
        return max(1, int(math.ceil(float(pooled_tokens) / float(seq_degree))))

    def _vit_patch_embed_gemm(self, batch: int, hidden_dim: int) -> Tuple[int, int, int, int]:
        return (
            int(batch),
            int(getattr(self, "num_patches", 0)),
            int(getattr(self, "patch_dim", 0)),
            int(hidden_dim),
        )

    def _vit_pool_time(self, *, batch: int, hidden_dim: int, backward: bool) -> Tuple[float, float]:
        call_key = "vit_pool_b" if backward else "vit_pool_f"
        self._set_endpoint_collectives(call_key, ())
        if not self._is_vit_model():
            return 0.0, 0.0
        output_elements = float(batch) * float(hidden_dim)
        reduce_tokens = float(self._vit_pool_token_count())
        flops = output_elements * reduce_tokens
        mem = self.precision.activations * (output_elements * (reduce_tokens + 1.0))
        time = self.roofline(
            flops,
            mem,
            name="vit_pool_b" if backward else "vit_pool_f",
            mem_level=self.num_levels - 1,
        ) + self.O
        seq_degree = self._vit_sequence_shard_degree()
        if seq_degree > 1:
            pooled_bytes = math.ceil(
                float(batch) * float(hidden_dim) * float(self.precision.activations)
            )
            if pooled_bytes > 0:
                axis_hint = "cp" if self.cp > 1 else "tp"
                comm_time = self.network_model.collective(
                    kind=ALL_REDUCE,
                    size_bytes=pooled_bytes,
                    participants=seq_degree,
                    ib=self.links[axis_hint].bandwidth,
                    ll=self.links[axis_hint].latency,
                    local_bytes=0,
                    debug_label="vit_pool_b" if backward else "vit_pool_f",
                    axis=axis_hint,
                )
                self._set_endpoint_collectives(call_key, self._endpoint_collective(
                    call_key, ALL_REDUCE, pooled_bytes, comm_time, time,
                    axis=axis_hint, participants=seq_degree,
                ))
                time += comm_time
        return time, mem

    def get_tensor_reduction_time(
        self,
        total_bytes: int,
        kind: CollectiveType,
        name: str,
        participants: Optional[int] = None,
    ) -> float:
        if not total_bytes:
            return 0.0

        if not participants:
            participants = int(self.tp)

        reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=name or "comm",
            axis="tp",
        )
        return reduction_time

    def _flash_attention_backward_communication(self, batch_size, seq_len, kv_heads, head_dim):
        kv_width = float(kv_heads) * float(head_dim) / max(1, self.tp)
        attention_size_b = (
            self.precision.activations * batch_size * seq_len * kv_width * 2
            if int(self.cp) > 1 else 0
        )
        kind = REDUCE_SCATTER
        participants = self.cp
        axis_hint = "cp"
        attention_backward_reduction_time = 0
        if attention_size_b > 0:
            attention_backward_reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=attention_size_b,
            participants=participants,
            ib=self.links["cp"].bandwidth,
            ll=self.links["cp"].latency,
            local_bytes=0,
            debug_label= "attention_backward_reduction",
            axis=axis_hint,
        )
        return attention_backward_reduction_time, attention_size_b

    @staticmethod
    def _normalize_gemm_type(gemm_type: Optional[GemmType]) -> Optional[GemmType]:
        if gemm_type is None or isinstance(gemm_type, GemmType):
            return gemm_type
        if isinstance(gemm_type, str):
            try:
                return GemmType(gemm_type)
            except ValueError as exc:
                raise ValueError(f"Unknown GEMM type: {gemm_type!r}") from exc
        raise TypeError(f"Unsupported gemm type specifier: {gemm_type!r}")

    def _grad_accum_time(self, elements: float, name: str) -> float:
        if elements <= 0:
            return 0.0
        if getattr(self, "gradient_accumulation_steps", 1) <= 1:
            return 0.0
        mem_bytes = float(elements) * (
            2.0 * float(self.precision.gradients) + float(self.precision.grad_communication)
        )
        time_s = self.roofline(0, mem_bytes, name=name, mem_level=self.num_levels - 1)
        overhead = getattr(self, "grad_acc_overhead", 0.0) or 0.0
        if overhead:
            time_s += float(overhead)
        return time_s

    def _grad_accum_elems_for_gemm(
        self,
        gemm_type: Optional[GemmType],
        *,
        k: int,
        n: int,
        shard_k: int,
        shard_n: int,
    ) -> int:
        if getattr(self, "gradient_accumulation_steps", 1) <= 1:
            return 0
        if gemm_type is None:
            return int(k * n)
        if gemm_type == GemmType.MLA_DOWN_PROJ:
            return int(k * n)
        if gemm_type in (GemmType.QKV, GemmType.FFN1, GemmType.LINEAR_SOFTMAX):
            return int(k * shard_n)
        if gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
            return int(shard_k * n)
        return 0

    def parallelism_gemm_forward(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        *,
        decode: bool = False,
    ) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_forward(gemm, name, gemm_type, decode=decode)
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_forward(gemm, name, gemm_type, decode=decode)
        elif parallelism_mode in (ParallelismMode.TENSOR_CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            return self._tensor_context_hybrid_gemm_forward(gemm, name, gemm_type, decode=decode)
        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_forward(gemm, name, gemm_type, decode=decode)
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")

    def parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Any:
        parallelism_mode = self.get_parallelism_mode()
        if parallelism_mode == ParallelismMode.TENSOR or parallelism_mode == ParallelismMode.TENSOR_SEQUENCE:
            return self._tensor_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode == ParallelismMode.CONTEXT:
            return self._context_parallelism_gemm_backward(gemm, name, gemm_type)
        elif parallelism_mode in (ParallelismMode.TENSOR_CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            return self._tensor_context_hybrid_gemm_backward(gemm, name, gemm_type)

        elif parallelism_mode == ParallelismMode.SINGLE:
            return self.single_gpu_gemm_backward(gemm, name, gemm_type)
        else:
            raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")

    def _attention_gemm_time(self, dims, n_gemms: int, name: str, gemm_type, *, decode: bool):
        m, k, n = (int(x) for x in dims)
        repeats = n_gemms
        gemm_time, _, _, mem_accesses = self.get_gemm_time(
            m, k, n, name, disable_overhead=True, activity_batch=repeats,
        )
        if not (decode and _decode_batched_attention()):
            return gemm_time * repeats, mem_accesses
        n_gemms = max(1, int(math.ceil(float(n_gemms))))
        backend = getattr(self, "_opmodel_backend", None) or getattr(self, "_gemm_activity_backend", None)
        if backend is None:
            raise RuntimeError("RAPID_DECODE_BATCHED_ATTENTION requires the op-model backend")
        ctx = n if gemm_type == GemmType.ATTENTION_SCORE else k
        num_sms = int(self.core.num_bundle)
        splits = max(1, min(math.ceil(num_sms / n_gemms), math.ceil(ctx / 128), 128))
        chunk = math.ceil(ctx / splits)
        dm, dk, dn = (m, k, chunk) if gemm_type == GemmType.ATTENTION_SCORE else (m, chunk, n)

        fixed = 0.0
        if getattr(self.core, "device_fixed_overhead_cycles", None) and self.core.operating_freq > 0:
            fixed = float(self.core.device_fixed_overhead_cycles) / float(self.core.operating_freq)
        body = max(0.0, float(backend.predict_batched_gemm(dim1=dm, dim2=dk, dim3=dn, batch=n_gemms * splits, name=name).latency_s) - fixed)
        if gemm_type == GemmType.ATTENTION_SCORE:
            return body, mem_accesses

        t = body - self.O
        if splits > 1:
            partial_bytes = n_gemms * splits * m * (n + 1) * 4 + n_gemms * m * n * float(self.precision.activations)
            t += self.vector_roofline(0.0, partial_bytes, mem_level=self.num_levels - 1) + self.O
        return t, mem_accesses

    def get_gemm_time(self, dim1, dim2, dim3, name="",
                      flashattn_enable=False, disable_overhead=False, read_bytes_l2=0,
                      write_bytes_l2=0, original=False, batch=1, activity_batch=None, gemm_role=None,
                      weight_bytes=None):
        if getattr(self, "_capture_operator_gemms", False) and not flashattn_enable:
            repeat = max(1, int(math.ceil(float(
                (batch if activity_batch is None else activity_batch) or 1))))
            self._operator_gemm_shapes.setdefault(str(name), []).append(
                (int(dim1), int(dim2), int(dim3), repeat))
        return super().get_gemm_time(
            dim1, dim2, dim3, name, flashattn_enable=flashattn_enable,
            disable_overhead=disable_overhead, read_bytes_l2=read_bytes_l2,
            write_bytes_l2=write_bytes_l2, original=original, batch=batch,
            activity_batch=activity_batch,
            **({"gemm_role": gemm_role} if gemm_role is not None else {}),
            **({"weight_bytes": weight_bytes} if weight_bytes is not None else {}))

    def single_gpu_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, decode: bool = False) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_flops = 2 * batch * m * k * n
        mem_accesses = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            gemm_time = self._attention_gemm_time((m, k, n), batch, name, gemm_type, decode=decode)[0] + self.O
        else :
            gemm_time,_,_, mem_accesses = self.get_gemm_time(m, k, n, name)
        return gemm_time, 0, 0, total_flops, mem_accesses

    def single_gpu_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            grad_time_act = self.get_gemm_time(m, k, n, name, disable_overhead=True, activity_batch=batch, gemm_role="dgrad")[0] * batch + self.O
            grad_time_wt = self.get_gemm_time(k, m, n, name, disable_overhead=True, activity_batch=batch, gemm_role="wgrad")[0] * batch + self.O
        else :
            grad_time_act = self.get_gemm_time(m, n, k, name, gemm_role="dgrad")[0]
            grad_time_wt = self.get_gemm_time(k, m, n, name, gemm_role="wgrad")[0]
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=k,
            n=n,
            shard_k=k,
            shard_n=n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        return gemm_time, 0, 0

    def _tensor_context_hybrid_gemm_forward(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        *,
        decode: bool = False,
    ) -> Tuple[float, float, float, float, Any]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)

        participants = 0
        total_bytes = 0
        reduction_time = 0
        total_flops = 2 * batch * m * k * n
        mem_accesses: Any = []

        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid forward GEMM")

        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        axis_hint = None

        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            if gemm_type == GemmType.ATTENTION_SCORE:
                dims = (shard_spec.shard_m, shard_spec.k, shard_spec.n if not decode else shard_spec.shard_n)
            else:
                dims = (shard_spec.shard_m, shard_spec.k if not decode else shard_spec.shard_k, shard_spec.n)
            gemm_time, mem_accesses = self._attention_gemm_time(
                dims, batch * shard_spec.batch_scale, name, gemm_type, decode=decode
            )
            gemm_time = gemm_time + self.O
        elif gemm_type == GemmType.MLA_DOWN_PROJ:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
            )
        elif gemm_type == GemmType.QKV:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
            if decode:
                total_bytes = shard_spec.shard_m * shard_spec.k * self.precision.activations
            else:
                total_bytes = self.get_kv_shard_bytes()
            kind = ALL_GATHER
            participants = self.cp
            axis_hint = "cp"
        elif gemm_type == GemmType.OUT_PROJ:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_k,
                shard_spec.n,
                name,
            )
            total_bytes = math.ceil(self.precision.activations * shard_spec.shard_m * n)
            kind = COMMUNICATION_RULES[self.get_parallelism_mode()]["output_proj"]["forward"]["kind"]
            participants = self.tp
            axis_hint = "tp"
        elif gemm_type == GemmType.FFN2:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_k,
                shard_spec.n,
                name,
            )
            total_bytes = math.ceil(self.precision.activations * shard_spec.shard_m * n)
            kind = COMMUNICATION_RULES[self.get_parallelism_mode()]["MLP"]["forward"]["kind"]
            participants = self.tp
            axis_hint = "tp"
        elif gemm_type == GemmType.FFN1:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")

        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=name or "comm",
            axis=axis_hint,
        )
        return gemm_time, reduction_time, total_bytes, total_flops, mem_accesses
    def _tensor_context_hybrid_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)

        participants = 0
        total_bytes = 0
        reduction_time = 0

        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-context hybrid backward GEMM")

        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        axis_hint = None

        if gemm_type == GemmType.ATTENTION_SCORE:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
                activity_batch=batch * shard_spec.batch_scale,
                gemm_role="dgrad",
            )[0] * batch * shard_spec.batch_scale + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
                activity_batch=batch * shard_spec.batch_scale,
                gemm_role="wgrad",
            )[0] * batch * shard_spec.batch_scale + self.O

            total_bytes = self.precision.activations * k * n * batch * 2 / self.tp
            kind = REDUCE_SCATTER
            participants = self.cp
            axis_hint = "cp"
        elif gemm_type == GemmType.ATTENTION_OUTPUT:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                activity_batch=batch * shard_spec.batch_scale,
                gemm_role="dgrad",
            )[0] * batch * shard_spec.batch_scale
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                activity_batch=batch * shard_spec.batch_scale,
                gemm_role="wgrad",
            )[0] * batch * shard_spec.batch_scale
        elif gemm_type == GemmType.MLA_DOWN_PROJ:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
        elif gemm_type == GemmType.QKV:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
                gemm_role="wgrad",
            )[0]

            total_bytes = math.ceil(self.precision.activations * shard_spec.shard_m * k)
            kind = COMMUNICATION_RULES[self.get_parallelism_mode()]["qkv_proj"]["backward"]["kind"]
            participants = self.tp
            axis_hint = "tp"

            self.__dict__.setdefault("_colpar_wgrad_time", {})[name] = float(grad_time_wt)
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
            total_bytes = self.get_kv_shard_bytes()
            kind = ALL_GATHER
            participants = self.cp
            axis_hint = "cp"
        elif gemm_type == GemmType.FFN2:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
        elif gemm_type == GemmType.FFN1:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
                gemm_role="wgrad",
            )[0]
            total_bytes = math.ceil(self.precision.activations * shard_spec.shard_m * k)
            kind = COMMUNICATION_RULES[self.get_parallelism_mode()]["MLP"]["backward"]["kind"]
            participants = self.tp
            axis_hint = "tp"

            self.__dict__.setdefault("_colpar_wgrad_time", {})[name] = float(grad_time_wt)
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
            total_bytes = math.ceil(self.precision.activations * shard_spec.shard_m * shard_spec.shard_k) * self.cp * self.tp
            kind = ALL_GATHER
            participants = self.cp * self.tp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=shard_spec.k,
            n=shard_spec.n,
            shard_k=shard_spec.shard_k,
            shard_n=shard_spec.shard_n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=name or "comm",
            axis=axis_hint,
        )
        return gemm_time, reduction_time, total_bytes
    def _tensor_parallelism_gemm_forward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, decode: bool = False) -> Tuple[float, float]:
        tp_mode = self.get_parallelism_mode()
        if gemm_type == GemmType.LINEAR_SOFTMAX:
            comm_kind_fwd = ALL_GATHER
        else:
            comm_kind_fwd = ALL_REDUCE if tp_mode == ParallelismMode.TENSOR else REDUCE_SCATTER
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        size_bytes = 0
        total_bytes = 0
        reduction_time = 0
        total_flops = 2 * batch * m * k * n
        mem_accesses = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel forward GEMM")

        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            gemm_time, mem_accesses = self._attention_gemm_time(
                (shard_spec.shard_m, shard_spec.k, shard_spec.n),
                batch * shard_spec.batch_scale, name, gemm_type, decode=decode,
            )
            gemm_time = gemm_time + self.O
        elif gemm_type == GemmType.MLA_DOWN_PROJ:
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
            )
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_k,
                shard_spec.n,
                name,
            )
            size_bytes = math.ceil(self.precision.activations * m * n)
            participants = self.tp
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            gemm_time,_,_, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.shard_n,
                name,
            )
            size_bytes = math.ceil(self.precision.activations * m * n)
            participants = self.tp * self.cp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")

        if size_bytes > 0:
            if getattr(self.network_model, "vllm_allreduce", None):
                size_bytes *= batch
            total_bytes = size_bytes
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_fwd, participants=participants, name=name)

        return gemm_time, reduction_time, total_bytes, total_flops, mem_accesses

    def _tensor_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        seq_degree = self._sequence_parallel_degree()
        act_bytes = 0
        total_bytes = 0
        if gemm_type == GemmType.LINEAR_SOFTMAX:
            comm_kind_bwd = ALL_REDUCE
        else:
            comm_kind_bwd = ALL_REDUCE if seq_degree == 1 else REDUCE_SCATTER
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for tensor-parallel backward GEMM")

        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
                activity_batch=batch * shard_spec.batch_scale,
                gemm_role="dgrad",
            )[0] * batch * shard_spec.batch_scale + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
                activity_batch=batch * shard_spec.batch_scale,
                gemm_role="wgrad",
            )[0] * batch * shard_spec.batch_scale + self.O
        elif gemm_type == GemmType.MLA_DOWN_PROJ:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
        elif gemm_type in (GemmType.QKV, GemmType.FFN1):
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
                gemm_role="wgrad",
            )[0]

            act_bytes = math.ceil(self.precision.activations * m * k)
            participants = self.tp

            self.__dict__.setdefault("_colpar_wgrad_time", {})[name] = float(grad_time_wt)
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN2):
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.shard_k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.shard_k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
        elif gemm_type == GemmType.LINEAR_SOFTMAX:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.shard_n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.shard_n,
                name,
                gemm_role="wgrad",
            )[0]
            act_bytes = math.ceil(self.precision.activations * m * k)
            participants = self.tp * self.cp
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=shard_spec.k,
            n=shard_spec.n,
            shard_k=shard_spec.shard_k,
            shard_n=shard_spec.shard_n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        reduction_time = 0
        if act_bytes > 0:
            total_bytes = act_bytes
            reduction_time = self.get_tensor_reduction_time(total_bytes, kind=comm_kind_bwd, participants=participants, name=name)

        return gemm_time, reduction_time, total_bytes
    def _context_parallelism_gemm_forward(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        decode: bool = False,
    ) -> Tuple[float, float, float, float, Any]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        total_flops = 2 * batch * m * k * n
        mem_accesses: Any = []
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel forward GEMM")
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type in (GemmType.ATTENTION_SCORE, GemmType.ATTENTION_OUTPUT):
            gemm_time, mem_accesses = self._attention_gemm_time(
                (shard_spec.shard_m, shard_spec.k, shard_spec.n if not decode else shard_spec.shard_n),
                batch, name, gemm_type, decode=decode,
            )
            gemm_time = gemm_time + self.O
        elif gemm_type == GemmType.MLA_DOWN_PROJ:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
            )
        elif gemm_type == GemmType.QKV:
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
            )
            total_bytes = self.get_kv_shard_bytes() if not decode else shard_spec.shard_m * shard_spec.k * self.precision.activations
        elif gemm_type in (GemmType.OUT_PROJ, GemmType.FFN1, GemmType.FFN2):
            gemm_time, _, _, mem_accesses = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.k,
                shard_spec.n,
                name,
            )
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        if gemm_type == GemmType.QKV:
            kind = ALL_GATHER
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label=name or "comm",
                axis="cp",
            )

        return gemm_time, reduction_time, total_bytes, total_flops, mem_accesses

    def _context_parallelism_gemm_backward(self, gemm: Tuple[int, ...], name: str, gemm_type: Optional[GemmType] = None, comm_after: bool = False) -> Tuple[float, float]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        total_bytes = 0
        reduction_time = 0
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            raise ValueError("gemm_type is required for context-parallel backward GEMM")
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        if gemm_type == GemmType.ATTENTION_SCORE:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
                activity_batch=batch,
                gemm_role="dgrad",
            )[0] * batch + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
                activity_batch=batch,
                gemm_role="wgrad",
            )[0] * batch + self.O
            total_bytes = self.precision.activations * k * n * batch * 2
            kind = REDUCE_SCATTER
        elif gemm_type == GemmType.ATTENTION_OUTPUT:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                disable_overhead=True,
                activity_batch=batch,
                gemm_role="dgrad",
            )[0] * batch + self.O
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                disable_overhead=True,
                activity_batch=batch,
                gemm_role="wgrad",
            )[0] * batch + self.O
        elif gemm_type in (GemmType.MLA_DOWN_PROJ, GemmType.QKV, GemmType.FFN1, GemmType.FFN2):
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
        elif gemm_type == GemmType.OUT_PROJ:
            grad_time_act = self.get_gemm_time(
                shard_spec.shard_m,
                shard_spec.n,
                shard_spec.k,
                name,
                gemm_role="dgrad",
            )[0]
            grad_time_wt = self.get_gemm_time(
                shard_spec.k,
                shard_spec.shard_m,
                shard_spec.n,
                name,
                gemm_role="wgrad",
            )[0]
            total_bytes = self.get_kv_shard_bytes()
            kind = ALL_GATHER
        else:
            raise ValueError(f"Unsupported gemm type: {gemm_type}")
        gemm_time = grad_time_act + grad_time_wt
        grad_accum_elems = self._grad_accum_elems_for_gemm(
            gemm_type,
            k=shard_spec.k,
            n=shard_spec.n,
            shard_k=shard_spec.shard_k,
            shard_n=shard_spec.shard_n,
        )
        gemm_time += self._grad_accum_time(grad_accum_elems, f"{name}_grad_accum")
        if total_bytes > 0:
            reduction_time = self.network_model.collective(
                kind=kind,
                size_bytes=total_bytes,
                participants=self.cp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label=name or "comm",
                axis="cp",
            )
        return gemm_time, reduction_time, total_bytes

    def _scale_mem_accesses(self, mem_accesses: Any, scale: int) -> Dict[str, float]:
        if scale <= 0:
            return {}
        levels = self._mem_levels(mem_accesses)
        if scale == 1:
            return levels
        return {k: v * scale for k, v in levels.items()}

    def _fused_batched_comm_time(
        self,
        total_bytes: int,
        gemm_type: Optional[GemmType],
        *,
        direction: str,
    ) -> float:
        total_bytes = int(math.ceil(float(total_bytes or 0.0)))
        if total_bytes <= 0:
            return 0.0
        gemm_type = self._normalize_gemm_type(gemm_type)
        if gemm_type is None:
            return 0.0
        mode = self.get_parallelism_mode()
        kind = None
        participants = None
        if mode in (ParallelismMode.TENSOR, ParallelismMode.TENSOR_SEQUENCE):
            if direction == "forward" and gemm_type == GemmType.FFN2:
                kind = ALL_REDUCE if mode == ParallelismMode.TENSOR else REDUCE_SCATTER
                participants = self.tp
            elif direction == "backward" and gemm_type == GemmType.FFN1:
                kind = ALL_REDUCE if mode == ParallelismMode.TENSOR else REDUCE_SCATTER
                participants = self.tp
        elif mode in (ParallelismMode.TENSOR_CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            if direction == "forward" and gemm_type == GemmType.FFN2:
                kind = COMMUNICATION_RULES[mode]["MLP"][direction]["kind"]
                participants = self.tp
            elif direction == "backward" and gemm_type == GemmType.FFN1:
                kind = COMMUNICATION_RULES[mode]["MLP"][direction]["kind"]
                participants = self.tp
        if not kind or not participants or participants <= 1:
            return 0.0
        return self.network_model.collective(
            kind=kind,
            size_bytes=total_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=f"batched_{gemm_type.value}_{direction}",
            axis="tp",
        )

    def _batched_gemm_forward_compute(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        *,
        use_tp: bool = True,
    ) -> Tuple[float, float, int, float, Dict[str, float]]:
        prev = getattr(self, "_gemv_disabled", False)
        self._gemv_disabled = True
        moe_tile = self._moe_grouped_gemm_min_tile()

        tile_override = {"OPMODEL_MIN_GEMM_TILE": int(moe_tile)} if moe_tile is not None else {}
        try:
            with _settings.override(tile_override):
                return self._batched_gemm_forward_compute_unguarded(gemm, name, gemm_type, use_tp=use_tp)
        finally:
            self._gemv_disabled = prev

    def _moe_grouped_gemm_min_tile(self) -> Optional[int]:
        if not self._is_inference_run_type():
            return None
        return int(_settings.get("RAPID_MOE_GEMM_MIN_TILE"))

    def rapid_moe_tile_selector(self, *, counts, hidden_size, intermediate_size):
        active = [int(n) for n in counts if int(n) > 0]
        if not active:
            raise ValueError("rapid_grouped requires at least one routed slot")
        backend = self._opmodel_backend or self._gemm_activity_backend
        if backend is None:
            raise NotImplementedError("rapid_grouped MoE tile selection requires the operator-model backend")
        cache = self.__dict__.setdefault("_rapid_moe_tile_cache", {})
        key = (max(active), len(active), int(hidden_size), int(intermediate_size), self._moe_grouped_gemm_min_tile())
        if key in cache:
            return cache[key]
        result = self._rapid_moe_w13_selection(backend, max(active), hidden_size, intermediate_size)
        bm, bk, bn = result.tile_dims[-1]
        if min(bm, bn, bk) <= 0:
            raise ValueError("operator-model grouped GEMM did not export a CTA tile")
        cache[key] = int(bm), int(bn), int(bk), 1
        return cache[key]

    def rapid_moe_selected_template(self, *, counts, hidden_size, intermediate_size):
        active = [int(n) for n in counts if int(n) > 0]
        if not active:
            raise ValueError("rapid_grouped requires at least one routed slot")
        backend = self._opmodel_backend or self._gemm_activity_backend
        if backend is None:
            raise NotImplementedError("rapid_grouped MoE tile selection requires the operator-model backend")
        result, catalog = self._rapid_moe_w13_selection(backend, max(active), hidden_size, intermediate_size,
                                                         return_catalog=True)
        name = result.diagnostics["gemm_selection"]["selected_template"]["name"]
        return next(t for t in catalog if t.name == name), backend

    def _rapid_moe_w13_selection(self, backend, rows, hidden_size, intermediate_size, *, return_catalog=False):
        from opmodel.models.extended_roofline import _kernel_template_catalog
        minimum = self._moe_grouped_gemm_min_tile()
        override = {"OPMODEL_MIN_GEMM_TILE": int(minimum)} if minimum is not None else {}
        if _settings.get("OPMODEL_GEMM_CATALOG") == "cublaslt_sm80_bf16_splitk":
            override["OPMODEL_GEMM_CATALOG"] = "cublaslt_sm80_bf16"
        with _settings.override(override):
            result = backend.predict_batched_gemm(
                dim1=int(rows), dim2=int(hidden_size), dim3=2 * int(intermediate_size),
                batch=1, name="moe_w13", allow_gemv=False)
            catalog = _kernel_template_catalog(backend.dtype, backend.hardware) if return_catalog else None
        if int(dict(result.diagnostics.get("split_k") or {}).get("kernel_launches", 1)) != 1:
            raise ValueError("rapid_grouped selected a split-K template; fused_moe has no split-K")
        return (result, catalog) if return_catalog else result

    def _is_inference_run_type(self) -> bool:
        return str(getattr(getattr(self, "model", None), "run_type", "training")).lower() == "inference"

    def _batched_gemm_forward_compute_unguarded(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        *,
        use_tp: bool = True,
    ) -> Tuple[float, float, int, float, Dict[str, float]]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if batch <= 0 or m <= 0 or k <= 0 or n <= 0:
            return 0.0, 0.0, 0, 0.0, {}
        with self._repeated_activity(batch, name):
            if use_tp:
                result = self.parallelism_gemm_forward(
                    (m, k, n),
                    name,
                    gemm_type=gemm_type,
                )
                if len(result) != 5:
                    raise ValueError(f"Unsupported parallelism_gemm_forward return length: {len(result)}")
                per_time, per_comm_time, per_comm_bytes, per_flops, per_mem = result
            else:
                per_time, per_comm_time, per_comm_bytes, per_flops, per_mem = self.single_gpu_gemm_forward(
                    (m, k, n),
                    name,
                    gemm_type=gemm_type,
                )
        scale = int(batch)
        comm_bytes = int(math.ceil(float(per_comm_bytes or 0.0))) * scale
        comm_time = per_comm_time * scale
        if use_tp and comm_bytes and scale > 1:
            fused_time = self._fused_batched_comm_time(comm_bytes, gemm_type, direction="forward")
            if fused_time:
                comm_time = fused_time
        compute_time = per_time * scale

        if scale > 1 and _env_flag("RAPID_MOE_GROUPED_GEMM"):
            compute_time = max(0.0, compute_time - (scale - 1) * float(self.O or 0.0))
            grouped_bw = float(_settings.get("RAPID_MOE_GROUPED_BW"))
            if grouped_bw > 0.0:
                act_bytes = float(self.precision.activations)
                total_bytes = scale * (k * n + m * (k + n)) * act_bytes
                total_flops = scale * 2.0 * m * k * n

                if total_flops / float(self.th_matrix) < total_bytes / grouped_bw:
                    one_kernel = float(self.O or 0.0) + total_bytes / grouped_bw
                    compute_time = min(compute_time, one_kernel)
        return (
            compute_time,
            comm_time,
            comm_bytes,
            per_flops * scale,
            self._scale_mem_accesses(per_mem, scale),
        )

    def _batched_gemm_backward_compute(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: Optional[GemmType] = None,
        *,
        use_tp: bool = True,
    ) -> Tuple[float, float, int]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if batch <= 0 or m <= 0 or k <= 0 or n <= 0:
            return 0.0, 0.0, 0

        with self._repeated_activity(batch, name, f"{name}_grad_accum"):
            if use_tp:
                per_time, per_comm_time, per_comm_bytes = self.parallelism_gemm_backward(
                    (m, k, n),
                    name,
                    gemm_type=gemm_type,
                )
            else:
                per_time, per_comm_time, per_comm_bytes = self.single_gpu_gemm_backward(
                    (m, k, n),
                    name,
                    gemm_type=gemm_type,
                )
        scale = int(batch)
        comm_bytes = int(math.ceil(float(per_comm_bytes or 0.0))) * scale
        comm_time = per_comm_time * scale
        if use_tp and comm_bytes and scale > 1:
            fused_time = self._fused_batched_comm_time(comm_bytes, gemm_type, direction="backward")
            if fused_time:
                comm_time = fused_time
        return per_time * scale, comm_time, comm_bytes

    def get_moe_ffn_f(
        self,
        gemm,
        name,
        gemm_type: Optional[GemmType] = None,
        *,
        batch_size: int,
        seq_len: int,
        allow_padding: Optional[bool] = None,
    ):
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm)
        if allow_padding is None:
            allow_padding = self._moe_allow_padding()
        (
            tokens_owner,
            _tokens_dispatched,
            _tokens_local,
            experts_per_rank,
            tokens_per_expert_balanced,
        ) = self._moe_balanced_routed_tokens_per_expert(batch_size, seq_len, allow_padding=allow_padding)
        tokens_shared = self._moe_tokens_shared(tokens_owner)

        use_tp_sharded = bool(getattr(self, "tp_ep", True))
        gemm_time = 0.0
        tp_comm_time = 0.0
        tp_comm_bytes = 0
        total_flops = 0.0
        mem_accesses_total: Dict[str, float] = {}
        buckets = self._moe_routed_compute_buckets(
            tokens_per_expert_balanced,
            experts_per_rank,
        )
        active = self._moe_active_expert_bucket(tokens_owner, experts_per_rank)
        if active is not None and len(buckets) == 1 and buckets[0][2] == "uniform":
            buckets = [active]
        for bucket_batch, bucket_tokens_per_expert, bucket_label in buckets:
            m_per_expert = bucket_tokens_per_expert * max(1, int(self.cp))
            gemm_override = (bucket_batch, m_per_expert, k, n)
            bucket_name = name if bucket_label == "uniform" else f"{name}_{bucket_label}"
            (
                bucket_gemm_time,
                bucket_tp_comm_time,
                bucket_tp_comm_bytes,
                bucket_total_flops,
                bucket_mem_accesses,
            ) = self._batched_gemm_forward_compute(
                gemm_override,
                bucket_name,
                gemm_type=gemm_type,
                use_tp=use_tp_sharded,
            )
            gemm_time += bucket_gemm_time
            tp_comm_time += bucket_tp_comm_time
            tp_comm_bytes += bucket_tp_comm_bytes
            total_flops += bucket_total_flops
            mem_accesses_total = self._combine_mem(mem_accesses_total, bucket_mem_accesses)

        if tokens_shared > 0:
            shared_m = (tokens_owner // self._moe_dp_attention_groups()) * max(1, int(self.cp))
            shared_name = f"{name}_shared" if name else "moe_ffn_shared"
            gemm_override_shared = (int(self.n_shared_experts), shared_m, k, n)
            (
                shared_gemm_time,
                shared_comm_time,
                shared_comm_bytes,
                shared_flops,
                mem_accesses_shared,
            ) = self._batched_gemm_forward_compute(
                gemm_override_shared,
                shared_name,
                gemm_type=gemm_type,
                use_tp=use_tp_sharded,
            )
            gemm_time += shared_gemm_time
            tp_comm_time += shared_comm_time
            tp_comm_bytes += shared_comm_bytes
            total_flops += shared_flops
            mem_accesses_total = self._combine_mem(mem_accesses_total, mem_accesses_shared)

        return gemm_time, tp_comm_time, tp_comm_bytes, total_flops, mem_accesses_total

    def get_moe_ffn_b(
        self,
        gemm,
        name,
        gemm_type,
        *,
        batch_size: int,
        seq_len: int,
        allow_padding: Optional[bool] = None,
    ):
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm)
        if allow_padding is None:
            allow_padding = self._moe_allow_padding()
        (
            tokens_owner,
            _tokens_dispatched,
            _tokens_local,
            experts_per_rank,
            tokens_per_expert_balanced,
        ) = self._moe_balanced_routed_tokens_per_expert(batch_size, seq_len, allow_padding=allow_padding)
        tokens_shared = self._moe_tokens_shared(tokens_owner)

        use_tp_sharded = bool(getattr(self, "tp_ep", True))
        gemm_time = 0.0
        tp_comm_time = 0.0
        tp_comm_bytes = 0
        for bucket_batch, bucket_tokens_per_expert, bucket_label in self._moe_routed_compute_buckets(
            tokens_per_expert_balanced,
            experts_per_rank,
        ):
            m_per_expert = bucket_tokens_per_expert * max(1, int(self.cp))
            gemm_override = (bucket_batch, m_per_expert, k, n)
            bucket_name = name if bucket_label == "uniform" else f"{name}_{bucket_label}"
            bucket_gemm_time, bucket_tp_comm_time, bucket_tp_comm_bytes = self._batched_gemm_backward_compute(
                gemm_override,
                bucket_name,
                gemm_type=gemm_type,
                use_tp=use_tp_sharded,
            )
            gemm_time += bucket_gemm_time
            tp_comm_time += bucket_tp_comm_time
            tp_comm_bytes += bucket_tp_comm_bytes
        if tokens_shared > 0:
            shared_m = (tokens_owner // self._moe_dp_attention_groups()) * max(1, int(self.cp))
            shared_name = f"{name}_shared" if name else "moe_ffn_shared"
            gemm_override_shared = (int(self.n_shared_experts), shared_m, k, n)
            shared_gemm_time, shared_comm_time, shared_comm_bytes = self._batched_gemm_backward_compute(
                gemm_override_shared,
                shared_name,
                gemm_type=gemm_type,
                use_tp=use_tp_sharded,
            )
            gemm_time += shared_gemm_time
            tp_comm_time += shared_comm_time
            tp_comm_bytes += shared_comm_bytes

        return gemm_time, tp_comm_time, tp_comm_bytes

    def _set_endpoint_collectives(self, key, calls):
        from sim_settings import nccl_latency_rule
        if (key not in ("optimizer_embedding", "optimizer_softmax")
                and nccl_latency_rule(self.hw_config) not in ("tuning_formula", "tuning_protocol")
                and float(getattr(self.hw_config.sw_config, "collective_launch_overhead", 0.0)) <= 0
                and not getattr(getattr(self, "network_model", None), "vllm_allreduce", None)):
            return
        if not hasattr(self, "_endpoint_collectives"):
            self._endpoint_collectives = {}
        self._endpoint_collectives[key] = tuple(calls)

    def _endpoint_collective(self, name, kind, size, duration, offset, *, axis="tp", participants=None):
        participants = int(self.tp) if participants is None else int(participants)
        if not is_nccl_call(kind, participants=participants, size_bytes=size):
            return ()
        return (EmbeddedCollective(name, kind, float(size), participants, axis,
                                   max(0.0, float(offset)), float(duration)),)

    def get_embedding_f(self, vocab_size, seq_len, hidden_dim):
        self._set_endpoint_collectives("embedding_f", ())
        if self._is_vit_model():
            batch = self._effective_transformer_batch()
            patch_gemm = self._vit_patch_embed_gemm(batch, hidden_dim)
            gemm_time, _, _, _, _ = self.single_gpu_gemm_forward(
                patch_gemm,
                "vit_patch_embed_f",
            )
            input_image_bytes = (
                float(batch)
                * float(getattr(self, "image_size", (1, 1))[0])
                * float(getattr(self, "image_size", (1, 1))[1])
                * float(getattr(self, "in_chans", 3))
                * float(self.precision.activations)
            )
            token_bytes = float(batch) * float(seq_len) * float(hidden_dim) * float(self.precision.activations)
            prep_time = self.vector_roofline(
                0.0,
                token_bytes,
                name="vit_embedding_f",
                mem_level=self.num_levels - 1,
            ) + self.O
            embedding_mem = input_image_bytes + token_bytes
            embedding_time = gemm_time + prep_time
            if self.h2d_bandwidth and self.h2d_bandwidth > 0:
                embedding_time += input_image_bytes / self.h2d_bandwidth
            return embedding_time, embedding_mem

        batch = self._effective_transformer_batch()
        embedding_mem = vocab_size * hidden_dim * self.precision.activations + seq_len * batch * hidden_dim * self.precision.activations
        if self.run_type == "inference" and _env_flag("RAPID_EMBEDDING_GATHER"):
            embedding_mem = 2 * seq_len * batch * hidden_dim * self.precision.activations
        embedding_allreduce_time = 0.0
        if self.run_type == "training" and self._implementation_option("vocab_parallel_embedding"):
            tokens = float(batch) * math.ceil(float(seq_len) / max(1, int(self.cp)))
            act = float(self.precision.activations)
            embedding_mem = tokens * hidden_dim * act / max(1, int(self.tp)) + tokens * hidden_dim * act
            if getattr(self, "tp_sp", False):
                embedding_allreduce_time = self._training_tp_collective(
                    REDUCE_SCATTER, tokens * hidden_dim * act, "embedding_f_reduce_scatter"
                )
            else:
                embedding_allreduce_time = self._training_tp_collective(
                    ALL_REDUCE, tokens * hidden_dim * act, "embedding_f_allreduce"
                )
        elif self.run_type == "training":
            embedding_mem = 2 * batch * seq_len * hidden_dim * self.precision.activations
        if getattr(self.network_model, "vllm_allreduce", None):
            tokens = float(batch) * math.ceil(float(seq_len) / max(1, int(self.cp)))
            act = float(self.precision.activations)
            embedding_allreduce_time = self._training_tp_collective(
                ALL_REDUCE, tokens * hidden_dim * act, "embedding_f_allreduce"
            )
        embedding_time = self.vector_roofline(
            0,
            embedding_mem,
            name="embedding_f",
            mem_level=self.num_levels - 1,
        ) + self.O
        embedding_time = self.kernel_floor(embedding_time)
        embedding_time += embedding_allreduce_time
        if self.h2d_bandwidth and self.h2d_bandwidth > 0:
            embedding_transfer_time = embedding_mem / self.h2d_bandwidth
        else:
            embedding_transfer_time = 0.0
        if self.debug:
            print(
                "Embedding_mem: {:,}, transfer_time: {:.6f}".format(
                    int(embedding_mem / 1e9), embedding_transfer_time
                )
            )
        self._set_endpoint_collectives("embedding_f", self._endpoint_collective(
            "embedding_f_reduce_scatter" if getattr(self, "tp_sp", False) else "embedding_f_allreduce",
            REDUCE_SCATTER if getattr(self, "tp_sp", False) else ALL_REDUCE,
            tokens * hidden_dim * act if ((self.run_type == "training" and self._implementation_option("vocab_parallel_embedding")) or getattr(self.network_model, "vllm_allreduce", None)) else 0,
            embedding_allreduce_time,
            embedding_time + embedding_transfer_time - embedding_allreduce_time,
        ))
        return embedding_time + embedding_transfer_time, embedding_mem

    def _implementation_option(self, name: str, default=True):
        return getattr(self, "implementation_options", {}).get(name, default)

    def _async_dgrad_fraction(self, op_name: str, backward_compute_time: float) -> float:
        if self.run_type != "training" or not self._implementation_option("async_dgrad_overlap") or backward_compute_time <= 0.0:
            return 0.0
        name = {"qkv_proj": "qkv_projection_b", "MLP": "ffn_b"}.get(op_name)
        wgrad = float(getattr(self, "_colpar_wgrad_time", {}).get(name, 0.0) or 0.0)
        return max(0.0, min(1.0, wgrad / backward_compute_time))

    def _training_head_tokens(self, gemm) -> int:
        b, m, _k, _n = self._expand_gemm_descriptor(gemm)
        return int(b) * int(math.ceil(float(m) / max(1, int(self.cp))))

    def _training_head_shard_gemm(self, gemm):
        b, m, k, n = self._expand_gemm_descriptor(gemm)
        return (
            int(b),
            int(math.ceil(float(m) / max(1, int(self.cp)))),
            int(k),
            int(math.ceil(float(n) / max(1, int(self.tp)))),
        )

    def _training_tp_collective(self, kind, size_bytes: float, label: str) -> float:
        if int(self.tp) <= 1 or size_bytes <= 0:
            return 0.0
        return self.network_model.collective(
            kind=kind,
            size_bytes=int(math.ceil(size_bytes)),
            participants=int(self.tp),
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=label,
            axis="tp",
        )

    def _training_head_ce_allreduce_calls(self, gemm):
        tokens = self._training_head_tokens(gemm)
        if self._implementation_option("megatron_loss", "one_pass") == "native_fused":
            sizes = (CROSS_ENTROPY_TRAIN_BYTES * tokens, 2 * CROSS_ENTROPY_TRAIN_BYTES * tokens)
            return [(size, self._training_tp_collective(ALL_REDUCE, size, "lm_head_ce_allreduce")) for size in sizes]
        width = CROSS_ENTROPY_TRAIN_BYTES if self._implementation_option("fp32_loss") else float(self.precision.activations)
        size = width * tokens
        seconds = self._training_tp_collective(ALL_REDUCE, size, "lm_head_ce_allreduce")
        return [(size, seconds)] * 3

    def _training_head_ce_allreduce_time(self, gemm) -> float:
        calls = self._training_head_ce_allreduce_calls(gemm)
        if len(calls) == 3:
            return 3.0 * calls[0][1]
        return sum(seconds for _, seconds in calls)

    def _training_head_dgrad_allreduce_time(self, gemm) -> float:
        _b, _m, k, _n = self._expand_gemm_descriptor(gemm)
        size = float(self.precision.activations) * self._training_head_tokens(gemm) * float(k)
        if getattr(self, "tp_sp", False):
            return self._training_tp_collective(REDUCE_SCATTER, size, "lm_head_dgrad_reduce_scatter")
        return self._training_tp_collective(ALL_REDUCE, size, "lm_head_dgrad_allreduce")

    def _training_head_input_gather_time(self, gemm) -> float:
        if not getattr(self, "tp_sp", False):
            return 0.0
        _b, _m, k, _n = self._expand_gemm_descriptor(gemm)
        per_rank = float(self.precision.activations) * math.ceil(self._training_head_tokens(gemm) / max(1, int(self.tp))) * float(k)
        return self._training_tp_collective(ALL_GATHER, per_rank, "lm_head_input_all_gather")

    def get_linear_softmax_f(self, gemm):
        self._set_endpoint_collectives("linear_softmax_f", ())
        if self._is_vit_model():
            batch = self._effective_transformer_batch()
            total_time = 0.0
            total_mem = 0.0
            ln_f, ln_comm, ln_bytes = self.get_layernorm_f(
                batch=batch,
                seq_len=self.seq_len,
                d_model=self.hidden_dim,
                name="pointwise-final-norm-f",
            )
            self._set_endpoint_collectives("linear_softmax_f", self._endpoint_collective(
                "vit_final_norm_f_all_gather", ALL_GATHER, ln_bytes, ln_comm, ln_f,
            ))
            total_time += ln_f + ln_comm
            total_mem += float(batch) * float(self.seq_len) * float(self.hidden_dim) * float(self.precision.stats)

            pool_time, pool_mem = self._vit_pool_time(batch=batch, hidden_dim=self.hidden_dim, backward=False)
            self._set_endpoint_collectives("linear_softmax_f",
                getattr(self, "_endpoint_collectives", {}).get("linear_softmax_f", ())
                + tuple(replace(call, offset_seconds=total_time + call.offset_seconds)
                        for call in getattr(self, "_endpoint_collectives", {}).get("vit_pool_f", ())))
            total_time += pool_time
            total_mem += pool_mem

            if getattr(self, "num_classes", 0) > 0:
                head_gemm = (batch, 1, self.hidden_dim, int(self.num_classes))
                gemm_time, _, _, _, _ = self.single_gpu_gemm_forward(
                    head_gemm,
                    "vit_head_f",
                    gemm_type=GemmType.LINEAR_SOFTMAX,
                )
                total_time += gemm_time
                total_mem += (
                    float(batch)
                    * float(self.num_classes)
                    * float(self.precision.activations)
                )
            else:
                total_mem += (
                    float(batch)
                    * float(self.hidden_dim)
                    * float(self.precision.activations)
                )

            return total_time, total_mem

        _fb, _fm, _fk, _fn = self._expand_gemm_descriptor(gemm)
        final_norm_time = self._final_norm_time(int(_fb) * int(_fm), "forward")
        selected_sampler = self._sourced_vllm_rule("RAPID_VLLM_SELECTED_SAMPLER")
        if selected_sampler:
            gemm = (int(self._effective_transformer_batch()), _fk, 64 * math.ceil(_fn / 64))
        elif self.run_type == "inference" and _prefill_logits_last_token():
            _b, _m, _k, _n = self._expand_gemm_descriptor(gemm)
            gemm = (_b, 1, _k, _n)
            if _b == 1 and _m > 1:
                gemm = (int(self._effective_transformer_batch()), _k, _n)
        if self.run_type == "inference" and _env_flag("RAPID_INFER_VOCAB_PARALLEL_HEAD"):
            _b, _m, _k, _n = self._expand_gemm_descriptor(gemm)
            gemm = (_b * _m, _k, _n)
        _, effective_m, k, n = self._effective_dims(gemm)

        head_share = 1.0 if self.run_type == "inference" else 1.0 / (self.tp * self.cp)
        if self.run_type == "inference":
            head_share *= float(self._moe_dp_attention_groups())
        vocab_parallel = (
            self.run_type == "inference" and int(self.tp) > 1 and (
                _env_flag("RAPID_INFER_VOCAB_PARALLEL_HEAD") or self._implementation_option("vocab_parallel_head_gather"))
        )
        gather_time = 0.0
        if vocab_parallel:
            _b, _m, _k, _n = self._expand_gemm_descriptor(gemm)
            vtp_head = self._implementation_option("vocab_parallel_head_gather")
            shard_n = (64 * -(-int(_n) // 64)) // int(self.tp) if vtp_head else int(math.ceil(float(_n) / float(self.tp)))
            gemm = (_b, _m, _k, shard_n)
            head_share = float(self.tp) * float(self._moe_dp_attention_groups())
            gather_bytes = int(math.ceil(float(_b) * float(_m) * shard_n * self.precision.activations))
            if vtp_head:
                from program.vtp_gather import gather_seconds
                from sim_settings import nccl_latency_rule
                gather_latency = 0.0 if nccl_latency_rule(self.hw_config) in ("tuning_formula", "tuning_protocol") else self.links["tp"].latency
                gather_time = gather_seconds(self.tp, gather_bytes, self.links["tp"].bandwidth, gather_latency,
                                             self.network_model.vllm_allreduce["per_peer_links"])
                if int(self.tp) > 1:
                    gather_time = self.kernel_floor(gather_time)
            else:
                gather_time = self.network_model.collective(
                kind=ALL_GATHER,
                size_bytes=gather_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label="lm_head_vocab_gather",
                axis="tp",
            )
        timed_gemm = gemm
        if self.run_type == "training" and self._implementation_option("vocab_parallel_head"):
            timed_gemm = self._training_head_shard_gemm(gemm)
            head_share = 1.0
        with self._repeated_activity(head_share, "linear_softmax_f"):
            gemm_time, _, _, _, _ = self.single_gpu_gemm_forward(timed_gemm, "linear_softmax_f", gemm_type=GemmType.LINEAR_SOFTMAX)
        head_gemm_time = gemm_time
        ce_allreduce_time = input_gather_time = 0.0
        gemm_time += gather_time
        if self.run_type == "training" and self._implementation_option("vocab_parallel_head"):
            ce_allreduce_time = self._training_head_ce_allreduce_time(gemm)
            gemm_time += ce_allreduce_time
            input_gather_time = self._training_head_input_gather_time(gemm)
            gemm_time += input_gather_time

        head_tp = self.tp if self.run_type != "training" or self._implementation_option("vocab_parallel_head") else 1
        elements = effective_m * n / (head_tp * self.cp)
        point_flop = elements * SOFTMAX_FORWARD_FLOPS_PER_ELEMENT

        ce_bytes = CROSS_ENTROPY_TRAIN_BYTES if self.run_type == "training" and self._implementation_option("fp32_loss") else self.precision.activations
        point_mem = (
            ce_bytes
            * elements
            * SOFTMAX_FORWARD_MEM_ACCESSES
        )
        softmax_shape = dict(
            rows=math.ceil(effective_m / max(1, self.cp)),
            row_size=math.ceil(n / max(1, head_tp)),
            name="linear_softmax_f",
            phase="prefill" if self.run_type == "inference" else "train_fwd",
        )
        if self.run_type == "training":
            softmax_shape["dtype_bytes"] = ce_bytes
        from program import vtp
        if selected_sampler:
            trace = self._sourced_vllm_trace(int(self._effective_transformer_batch()))
            kernels = [replace(kernel, activity_name="pointwise-linear-softmax-f")
                       for kernel in trace if kernel.op == "linear_softmax"
                       and kernel.name not in ("fused_add_rms_norm", "lm_head", "logits.gather")]
            point_time, point_activity = self._price_kernels(kernels)
            point_flop, point_mem = point_activity.vector_flops, point_activity.hbm.total_bytes
        elif self._eager():
            point_flop, point_mem, point_time = 0.0, 0.0, 0.0
        elif vtp.enabled(self):
            point_flop, point_mem, point_time = 0.0, 0.0, 0.0
        elif self._opmodel_backend is not None:
            softmax_result = self.get_opmodel_softmax(**softmax_shape)
            point_flop = softmax_result.flops
            point_mem = softmax_result.memory_accesses[-1]
            point_time = softmax_result.latency_s
            if getattr(self, "_decode_graph_pricing", False):
                compute = self._opmodel_backend.hardware.compute
                fixed = float(compute.device_fixed_overhead_cycles or 0) / float(compute.clock_hz)
                point_time += self.O - fixed
        else:
            activity_from_opmodel = self._gemm_activity_backend is not None
            point_time = self.vector_roofline(
                point_flop,
                point_mem,
                name="" if activity_from_opmodel else "pointwise-linear-softmax-f",
                mem_level=self.num_levels - 1,
            ) + 4 * self.O
            if activity_from_opmodel:
                self.get_opmodel_softmax(**softmax_shape)

        if self.debug:
            print(
                "Linear Softmax (f) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (f) point_time: {:,}\n".format(point_time))

        calls = ()
        offset = final_norm_time
        if self.run_type == "training" and self._implementation_option("vocab_parallel_head"):
            if getattr(self, "tp_sp", False):
                _b, _m, k, _n = self._expand_gemm_descriptor(gemm)
                per_rank = (float(self.precision.activations)
                            * math.ceil(self._training_head_tokens(gemm) / max(1, int(self.tp))) * float(k))
                calls += self._endpoint_collective("lm_head_input_all_gather", ALL_GATHER,
                                                   per_rank, input_gather_time, offset)
                offset += input_gather_time
            offset += head_gemm_time
            ce_calls = self._training_head_ce_allreduce_calls(gemm)
            for index, (size, seconds) in enumerate(ce_calls):
                duration = ce_allreduce_time / 3.0 if len(ce_calls) == 3 else seconds
                calls += self._endpoint_collective(f"lm_head_ce_allreduce_{index}", ALL_REDUCE,
                    ce_bytes * self._training_head_tokens(gemm) if len(ce_calls) == 3 else size, duration, offset)
                offset += duration
        elif vocab_parallel:
            calls += self._endpoint_collective("lm_head_vocab_gather", CollectiveType.GATHER if vtp_head else ALL_GATHER,
                                               gather_bytes, gather_time, offset + head_gemm_time)
        self._set_endpoint_collectives("linear_softmax_f", calls)
        return final_norm_time + gemm_time + point_time, point_mem

    def get_linear_softmax_b(self, gemm):
        self._set_endpoint_collectives("linear_softmax_b", ())
        if self._is_vit_model():
            batch = self._effective_transformer_batch()
            total_time = 0.0
            if getattr(self, "num_classes", 0) > 0:
                head_gemm = (batch, 1, self.hidden_dim, int(self.num_classes))
                gemm_time, _, _ = self.single_gpu_gemm_backward(
                    head_gemm,
                    "vit_head_b",
                    gemm_type=GemmType.LINEAR_SOFTMAX,
                )
                total_time += gemm_time

            pool_time, _ = self._vit_pool_time(batch=batch, hidden_dim=self.hidden_dim, backward=True)
            pool_calls = tuple(replace(call, offset_seconds=total_time + call.offset_seconds)
                for call in getattr(self, "_endpoint_collectives", {}).get("vit_pool_b", ()))
            total_time += pool_time

            ln_b, ln_comm, ln_bytes = self.get_layernorm_b(
                batch=batch,
                seq_len=self.seq_len,
                d_model=self.hidden_dim,
                name="pointwise-final-norm-b",
                accum_name="final_norm_grad_accum",
            )
            self._set_endpoint_collectives("linear_softmax_b", pool_calls + self._endpoint_collective(
                "vit_final_norm_b_all_gather", ALL_GATHER, ln_bytes, ln_comm, total_time + ln_b,
            ))
            total_time += ln_b + ln_comm
            return total_time

        _fb, _fm, _fk, _fn = self._expand_gemm_descriptor(gemm)
        final_norm_time = self._final_norm_time(int(_fb) * int(_fm), "backward")

        _, effective_m, k, n = self._effective_dims(gemm)

        head_share = 1.0 if self.run_type == "inference" else 1.0 / (self.tp * self.cp)
        timed_gemm = gemm
        dgrad_allreduce_time = 0.0
        if self.run_type == "training" and self._implementation_option("vocab_parallel_head"):
            timed_gemm = self._training_head_shard_gemm(gemm)
            head_share = 1.0
            dgrad_allreduce_time = self._training_head_dgrad_allreduce_time(gemm)
        with self._repeated_activity(head_share, "linear_softmax_b"):
            gemm_time, _, _ = self.single_gpu_gemm_backward(timed_gemm, "linear_softmax_b", gemm_type=GemmType.LINEAR_SOFTMAX)
        if self.run_type == "training" and self._implementation_option("vocab_parallel_head"):
            _b, _m, k, _n = self._expand_gemm_descriptor(gemm)
            size = float(self.precision.activations) * self._training_head_tokens(gemm) * float(k)
            self._set_endpoint_collectives("linear_softmax_b", self._endpoint_collective(
                "lm_head_dgrad_reduce_scatter" if getattr(self, "tp_sp", False) else "lm_head_dgrad_allreduce",
                REDUCE_SCATTER if getattr(self, "tp_sp", False) else ALL_REDUCE,
                size, dgrad_allreduce_time, final_norm_time + gemm_time,
            ))
        gemm_time += dgrad_allreduce_time
        head_tp = self.tp if self.run_type != "training" or self._implementation_option("vocab_parallel_head") else 1
        elements = effective_m * n / (head_tp * self.cp)
        point_flop = elements * SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT

        ce_bytes = CROSS_ENTROPY_TRAIN_BYTES if self.run_type == "training" and self._implementation_option("fp32_loss") else self.precision.activations
        point_mem = ce_bytes * elements * SOFTMAX_BACKWARD_MEM_ACCESSES

        point_time = 0.0 if self._eager() else self.vector_roofline(
            point_flop,
            point_mem,
            name="pointwise-linear-softmax-b",
            mem_level=self.num_levels - 1,
        ) + 4 * self.O

        if self.debug:
            print(
                "Linear Softmax (b) point_flop: {:,}, point_mem: {:,}".format(
                    int(point_flop / 1e9), int(point_mem / 1e9)
                )
            )
            print("Linear Softmax (b) point_time: {:,}\n".format(point_time))

        return final_norm_time + gemm_time + point_time
    def get_scale_softmax_f(self, gemm):
        batch, m, _, n = self._expand_gemm_descriptor(gemm)
        elements = math.ceil(batch * m * n / (self.tp * self.cp))
        flops = elements * (SOFTMAX_FORWARD_FLOPS_PER_ELEMENT + 1)
        mem = self.precision.activations * elements * (SOFTMAX_FORWARD_MEM_ACCESSES )

        time = self.vector_roofline(
            flops,
            mem,
            name="pointwise-scale-softmax-f",
            mem_level=self.num_levels - 1,
        ) + self.O

        return time

    def get_scale_softmax_b(self, gemm):
        batch, m, _, n = self._expand_gemm_descriptor(gemm)
        elements = math.ceil(batch * m * n / (self.tp * self.cp))
        flops = elements * (SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT + 1)
        mem = self.precision.activations * elements * (SOFTMAX_BACKWARD_MEM_ACCESSES)

        time = self.vector_roofline(
            flops,
            mem,
            name="pointwise-scale_softmax-b",
            mem_level=self.num_levels - 1,
        ) +  self.O

        if self.debug:
            print(
                "Scale Softmax (b) flop: {:,}, mem: {:,}".format(
                    int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Scale Softmax(b) time: {:,}".format(time))

        return time

    def get_residual_f(self, tensor_shape):
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements_m = m
        mode = self.get_parallelism_mode()
        if mode in (
            ParallelismMode.TENSOR_SEQUENCE,
            ParallelismMode.CONTEXT,
            ParallelismMode.TENSOR_CONTEXT,
            ParallelismMode.TENSOR_CONTEXT_HYBRID,
        ):
            seq_degree = max(1, int(self._sequence_parallel_degree()))
            elements_m = math.ceil(m / seq_degree)
        elements = batch * elements_m * n

        flops = 2 * elements
        mem = self.precision.activations * elements * 3
        _vllm_pw = self.run_type == "inference" and _vllm_pointwise()
        time = self.vector_roofline(
            flops, mem, name="pointwise-residual-f", mem_level=self.num_levels - 1,
            active_bundles=(min(int(batch) * int(elements_m), self.core.num_bundle) if _vllm_pw else None),
        ) + (0 if _vllm_pw else 1) * self.O

        if self.debug:
            print(
                "Residual (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (f) time: {:,}\n".format(time))

        return time

    def get_residual_b(self, tensor_shape):
        batch, m, _, n = self._expand_gemm_descriptor(tensor_shape)
        elements_m = m
        mode = self.get_parallelism_mode()
        if mode in (
            ParallelismMode.TENSOR_SEQUENCE,
            ParallelismMode.CONTEXT,
            ParallelismMode.TENSOR_CONTEXT,
            ParallelismMode.TENSOR_CONTEXT_HYBRID,
        ):
            seq_degree = max(1, int(self._sequence_parallel_degree()))
            elements_m = math.ceil(m / seq_degree)
        elements = batch * elements_m * n

        flops = elements

        mem = self.precision.activations * elements * 3
        time = self.vector_roofline(flops, mem, name="pointwise-residual-b", mem_level=self.num_levels - 1) + self.O

        if self.debug:
            print(
                "Residual (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(flops / 1e9), int(mem / 1e9)
                )
            )
            print("Residual (b) time: {:,}\n".format(time))

        return time

    def get_gelu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * hidden * 2

        time = self.vector_roofline(compute_flops, mem_bytes, name="pointwise-gelu-f", mem_level=self.num_levels - 1) + 2 * self.O

        if self.debug:
            print(
                "GELU (f) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (f) time: {:,}\n".format(time))

        return time
    def get_gelu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        compute_flops = elements * hidden * GELU_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * hidden * 3

        time = self.vector_roofline(compute_flops, mem_bytes, name="pointwise-gelu-b", mem_level=self.num_levels - 1) + 3 * self.O

        if self.debug:
            print(
                "GELU (b) elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int(elements / 1e6), int(compute_flops / 1e9), int(mem_bytes / 1e9)
                )
            )
            print("GELU (b) time: {:,}\n".format(time))
        return time

    def get_swiglu_f(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        if self.run_type == "inference" and self._eager():
            d = replace(self._trace_dims(int(elements), 1), intermediate=int(gate_hidden))
            ks = kernel_trace.swiglu_eager("g", "u", "m", int(elements), d, float(SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT))
            return self._price_kernels(ks)[0]
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_FORWARD_FLOPS_PER_ELEMENT + gate_hidden
        )
        reads = 2 * gate_hidden
        writes = gate_hidden
        mem_bytes = self.precision.activations * elements * (reads + writes)
        _vllm_pw = self.run_type == "inference" and _vllm_pointwise()
        time = self.vector_roofline(
            compute_flops, mem_bytes, name="pointwise-swiglu-f", mem_level=self.num_levels - 1,
            active_bundles=(min(int(elements), self.core.num_bundle) if _vllm_pw else None),
        ) + (1 if _vllm_pw else 2) * self.O
        time = self.kernel_floor(time, 1 if _vllm_pw else 2)
        if self.debug:
            print(
                "SwiGLU (f) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (f) time: {:,}\n".format(time))
        return time

    def get_swiglu_b(self, tensor_shape):
        _, elements, _, hidden = self._effective_dims(tensor_shape)
        gate_hidden = max(hidden // 2, 1)
        compute_flops = elements * (
            gate_hidden * SWIGLU_SILU_BACKWARD_FLOPS_PER_ELEMENT + 2 * gate_hidden
        )
        reads = 2 * gate_hidden
        reads += gate_hidden
        writes = 2 * gate_hidden
        mem_bytes = self.precision.activations * elements * (reads + writes)
        time = self.vector_roofline(compute_flops, mem_bytes, name="pointwise-swiglu-b", mem_level=self.num_levels - 1) + 3 * self.O
        if self.debug:
            print(
                "SwiGLU (b) gate elements: {:,}, flops: {:,}, mem: {:,}".format(
                    int((elements * gate_hidden) / 1e6),
                    int(compute_flops / 1e9),
                    int(mem_bytes / 1e9),
                )
            )
            print("SwiGLU (b) time: {:,}\n".format(time))
        return time

    def _layernorm_gather_elements(self, batch, seq_len, d_model) -> int:
        shards = max(1, int(self.tp)) * max(1, int(self.cp))
        return batch * math.ceil(seq_len / shards) * d_model

    def _eager(self) -> bool:
        if kernel_trace.implementation(self) != "eager":
            return False
        if str(self.run_type).lower() != "inference":
            if float(self.precision.parameters) != 4.0 or float(self.precision.tensor_format) != 2.0:
                raise ValueError("implementation: eager training is described for fp32 parameters under 16-bit autocast")
            if int(getattr(self, "gradient_accumulation_steps", 1) or 1) != 1 or int(getattr(self, "dp", 1)) != 1:
                raise ValueError("implementation: eager training is described for one GPU without gradient accumulation")
            if not bool(getattr(self, "tied_embeddings", False)) and self.model_type not in ("qwen2", "qwen3"):
                raise ValueError("implementation: eager training is described for tied embeddings")
        if (int(self.tp) != 1 or int(self.cp) != 1 or int(getattr(self, "pp", 1)) != 1
                or bool(getattr(self, "use_moe", False)) or not llm_util.is_llama_style(self.model_type)
                or str(getattr(self, "attention_type", "mha")).lower() not in ("mha", "gqa")):
            raise ValueError("implementation: eager is described for single-GPU dense Llama-style MHA/GQA only")
        return True

    def _eager_training(self) -> bool:
        return str(self.run_type).lower() == "training" and self._eager()

    def _validate_eager_training_l2_trace(self) -> None:
        if str(self.model_type).strip().lower() not in ("llama", "qwen2", "qwen3"):
            raise ValueError("RAPID_L2_RESIDENCY eager training requires the Llama model family or described dense qwen2/qwen3")
        if int(getattr(self, "ep", 1)) != 1:
            raise ValueError("RAPID_L2_RESIDENCY eager training trace requires expert parallelism ep=1")
        if not bool(getattr(self, "flash_attention", False)):
            raise ValueError("RAPID_L2_RESIDENCY eager training trace requires attention.use_flashattention=true")
        if not bool(getattr(self, "use_rope", False)):
            raise ValueError("RAPID_L2_RESIDENCY eager training trace requires attention.use_rope=true")
        if float(self.precision.gradients) != 4.0:
            raise ValueError("RAPID_L2_RESIDENCY eager training trace requires fp32 gradients")
        if float(self.precision.optimizer_states) != 4.0:
            raise ValueError("RAPID_L2_RESIDENCY eager training trace requires fp32 optimizer states")
        if float(self.precision.master_parameters) != 0.0:
            raise ValueError("RAPID_L2_RESIDENCY eager training trace requires no separate master parameters")
        if bool(getattr(self, "disable_embedding_unembedding", False)):
            raise ValueError("RAPID_L2_RESIDENCY eager training trace requires embedding and unembedding")
        if bool(getattr(self, "gradient_clipping", False)):
            raise ValueError("RAPID_L2_RESIDENCY eager training trace does not describe gradient clipping")

    def _trace_dims(self, tokens: int, batch: int, past: int = 0, head_tokens: int = 0) -> "kernel_trace.LayerDims":
        head_dim = int(getattr(self, "head_dim", None) or self.hidden_dim // self.num_heads)
        return kernel_trace.LayerDims(
            tokens=int(tokens), batch=max(1, int(batch)), hidden=int(self.hidden_dim), heads=int(self.num_heads),
            kv_heads=int(getattr(self, "kv_heads", self.num_heads)), head_dim=head_dim,
            intermediate=int(self.intermediate_size), vocab=int(self.vocab_size),
            act_bytes=float(self.precision.activations), weight_bytes=float(self.precision.parameters),
            kv_bytes=float(self.precision.kv_cache), past=int(past), head_tokens=int(head_tokens),
            model_type=str(self.model_type).lower(),
            qwen2_bias_mode=getattr(self, "qwen2_bias_mode", None),
            tied_embeddings=bool(getattr(self, "tied_embeddings", False)),
        )

    def _price_kernels(self, kernels, record: bool = True):
        total_time = 0.0
        total = ZERO_ACTIVITY
        for k in kernels:
            moved = k.read_bytes + k.write_bytes
            kwargs = {}
            if k.name in ("rms_norm", "fused_add_rms_norm", "q_a_layernorm", "kv_a_layernorm"):
                rows = int(k.writes[-1].nbytes / k.reads[-1].nbytes)
                kwargs["active_bundles"] = min(rows, self.core.num_bundle)
            elif k.name in ("q_norm", "k_norm"):
                rows = int(k.writes[0].nbytes / (self.precision.activations * self.head_dim))
                kwargs["active_bundles"] = min(rows, self.core.num_bundle)
            if getattr(self, "l2_bypass", False):
                moved += float(getattr(k, "l2_reread", 0.0))
            kernel_time = self.vector_roofline(k.flops, moved, name="", mem_level=self.num_levels - 1, **kwargs)
            if (_settings.get("RAPID_SMALLGRID_TIMING") and self.num_levels >= 4
                    and (not _settings.get("RAPID_SMALLGRID_DECODE_ONLY")
                         or getattr(self, "_pricing_decode_template", False))
                    and (k.name not in ("rms_norm", "fused_add_rms_norm")
                         or (self.run_type == "inference" and float(self.precision.activations) == 2.0))):
                from smallgrid_timing import estimate

                pass_level = self.num_levels - 1 if getattr(self, "l2_bypass", False) else 2
                extra = estimate(k, sms=self.core.num_bundle,
                                 l2_latency=self.mem_layer[pass_level].get_latency(),
                                 smem_latency=self.mem_layer[1].get_latency(),
                                 l2_bw=self.mem_layer[pass_level].get_throughput(),
                                 vector_rate=self.core.get_throughput("vector"),
                                 architecture=getattr(self.hw_config.tech_config.core, "architecture", None),
                                 clock_hz=self.core.operating_freq)
                if extra is not None:
                    kernel_time = max(kernel_time, extra)
            kernel_time += (
                0.0 if k.kind == "gemm_epilogue" else (
                    self.stream_launch_overhead_s if getattr(self, "_decode_graph_pricing", False)

                    and (k.name.startswith(("graph.copy_", "sampler.")) or k.name == "meta.query_start_loc_sub") else self.O))
            if k.kind not in ("gemm_epilogue", "memcpy"):
                kernel_time = self.kernel_floor(kernel_time)
            total_time += kernel_time
            traffic = MemoryTraffic(read_bytes=k.read_bytes, write_bytes=k.write_bytes)
            l2 = MemoryTraffic(read_bytes=k.read_bytes + float(getattr(k, "l2_reread", 0.0)), write_bytes=k.write_bytes)
            act = OperationActivity(vector_flops=float(k.flops), hbm=traffic, l2=l2)
            if record and k.activity_name:
                reg = self._roofline_activity_by_name
                reg[k.activity_name] = reg.get(k.activity_name, ZERO_ACTIVITY) + act
            total = total + act
        return total_time, total

    def _mla_vllm_trace(self) -> bool:
        return (str(getattr(self, "attention_type", "mha")).lower() == "mla"
                and str(getattr(self, "run_type", "training")).lower() == "inference"
                and self._implementation_option("mla_inference", "vllm_trace") == "vllm_trace")

    def _price_mla_kernel(self, kernel, *, phase: str, batch: int, seq: int):
        import kernel_trace_vllm_mla as kvm
        from program.vtp import _gemm
        k = kernel
        if k.kind == "gemm":
            from program.vtp import gemm_weight_bytes
            wb = gemm_weight_bytes(self, k)
            return _gemm(self, k.gemm) if wb is None else _gemm(self, k.gemm, wb)
        if k.kind == "batched_gemm":
            cache = self.__dict__.setdefault("_mla_bmm_cache", {})
            key = (k.gemm, k.batch, self.O)
            if key not in cache:
                name = "__mla_attention_bmm__"
                old = self._gemm_activity_by_name.pop(name, None)
                t = self.get_gemm_time(*k.gemm, name, batch=k.batch)[0]
                cache[key] = t, self._gemm_activity_by_name.pop(name, ZERO_ACTIVITY)
                if old is not None:
                    self._gemm_activity_by_name[name] = old
            return cache[key]
        dims = kvm.MlaDims.from_cfg(self._l2_sim_config(), int(self.tp))
        sms = int(self.core.num_bundle)
        if k.name == "attention.flash_fwd":
            import flash_attention_adapter

            seconds, act, _source = flash_attention_adapter.mla_prefill(self, int(batch), int(seq),
                                                                        dims.H * max(1, int(self.tp)),
                                                                        dims.N + dims.P, sms)
            return seconds, act
        if isinstance(k, kvm.MlaStage1Kernel):
            levels = [0.0] * int(self.num_levels)
            levels[min(2, self.num_levels - 1)] += k.l2_bytes
            levels[-1] = k.unique_bytes
            issued = k.tensor_flops * (k.head_groups * kvm.TRITON_BLOCK_H) / dims.H

            rate_flops = (max(issued, k.tensor_flops)
                          if self._implementation_option("mla_decode_stage1_rows", "local_heads") == "block_h"
                          else k.tensor_flops)
            seconds = self.roofline(rate_flops, levels, compute_unit="matrix",
                                    active_bundles=min(int(k.ctas), sms)) + self.O
            seconds = self.kernel_floor(seconds)
            writes = k.write_bytes
            act = OperationActivity(useful_tensor_flops=k.tensor_flops, issued_tensor_flops=max(issued, k.tensor_flops),
                                    vector_flops=k.flops,
                                    l2=MemoryTraffic(read_bytes=k.l2_bytes - writes, write_bytes=writes),
                                    hbm=MemoryTraffic(read_bytes=k.unique_bytes - writes, write_bytes=writes))
            return seconds, act
        return self._price_kernels([k], record=False)

    def _mla_vllm_op_timings(self, *, phase: str, batch: int, seq: int, backward_zero: bool = False):
        from program import vtp
        if int(self.cp) != 1 or int(getattr(self, "pp", 1)) != 1 or (int(self.tp) != 1 and not vtp.enabled(self)):
            raise NotImplementedError("the vLLM MLA description is single-GPU (tp = cp = pp = 1) or VTP (tp > 1)")
        if getattr(self.hw_config.tech_config.core, "architecture", None) == "sm90":
            raise NotImplementedError("the vLLM MLA description is the sm80 TRITON_MLA path; H100 FlashMLA is not described")

        if (float(self.precision.activations) != 2.0 or float(self.precision.parameters) != 2.0
                or float(self.precision.kv_cache) != 2.0):
            raise NotImplementedError("the vLLM MLA description is bf16 (activations, weights, latent cache)")
        import kernel_trace_vllm as kv
        tp = int(self.tp)
        cfg = dict(self._l2_sim_config(), num_hidden_layers=1)
        sms = int(self.core.num_bundle)
        eng = kv.VllmEngine(sm_count=sms, **dict(getattr(self, "engine_settings", None) or {}))
        trace = kv.vllm_step_trace(cfg, phase, int(batch), int(seq), eng, tp=tp, rank=0) if tp > 1 else \
            kv.vllm_step_trace(cfg, phase, int(batch), int(seq), eng)
        ops = {}
        for k in trace:
            if k.op not in ("qkv_proj", "rope", "attention", "output_proj") or k.kind == "collective":
                continue
            seconds, act = self._price_mla_kernel(k, phase=phase, batch=int(batch), seq=int(seq))
            if k.kind == "gemm":
                flops = 2.0 * k.gemm[0] * k.gemm[1] * k.gemm[2]
            elif k.kind == "batched_gemm":
                flops = 2.0 * k.batch * k.gemm[0] * k.gemm[1] * k.gemm[2]
            elif k.name == "attention.flash_fwd":
                flops = act.useful_tensor_flops + act.vector_flops
            elif hasattr(k, "tensor_flops"):
                flops = k.tensor_flops + k.flops
            else:
                flops = float(k.flops)
            entry = ops.setdefault(k.op, [0.0, ZERO_ACTIVITY, 0.0, []])
            entry[0] += seconds
            entry[1] = entry[1] + act
            entry[2] += flops
            entry[3].append(k.name)
        out = {}
        for op, (seconds, act, flops, names) in ops.items():
            comm_time, comm_bytes = 0.0, 0
            if op == "output_proj" and tp > 1:
                tokens = int(batch) * (int(seq) if phase == "prefill" else 1)
                comm_bytes = int(math.ceil(float(self.precision.activations) * tokens * int(self.hidden_dim)))
                label = "decode_output_projection_f" if phase == "decode" else "output_projection_f"
                comm_time = self.get_tensor_reduction_time(comm_bytes, kind=ALL_REDUCE, participants=tp, name=label)
            fwd = DirectionTiming(compute_time=seconds, flops=flops, activity=act, comm_time=comm_time,
                                  comm_bytes=comm_bytes,
                                  memory_accesses={f"L{self.num_levels - 1}": act.hbm.total_bytes},
                                  notes={"description": "kernel_trace_vllm_mla", "kernels": names})
            out[op] = OperationTiming(op, forward=fwd,
                                      backward=DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0)
                                      if backward_zero else None)
        return out

    def _source_moe(self) -> bool:
        mt = str(getattr(self, "model_type", "")).lower()
        if mt == "qwen3_moe":
            return True
        return (mt in ("deepseek_v2", "deepseek_v3") and bool(getattr(self, "use_moe", False))
                and str(getattr(self, "run_type", "training")).lower() == "inference"
                and str(getattr(self, "attention_type", "mha")).lower() == "mla" and self._mla_vllm_trace())

    def _vllm_norms(self) -> bool:
        return (self.run_type == "inference" and kernel_trace.implementation(self) == "fused"
                and self._implementation_option("flash_source", None) == "vllm"
                and (str(self.model_type).lower() in ("llama", "qwen2", "qwen3")
                     or (self._mla_vllm_trace() and str(self.model_type).lower() == "deepseek_v3"))
                and not bool(getattr(self, "use_moe", False))
                and (str(getattr(self, "attention_type", "mha")).lower() in ("mha", "gqa") or self._mla_vllm_trace())

                and int(self.cp) == 1 and not bool(getattr(self, "tp_sp", False))
                and float(self.precision.activations) == 2.0
                and float(self.precision.parameters) == 2.0)

    def _vllm_norm_f(self, *, batch: int, seq_len: int, role: str) -> float:
        import kernel_trace_vllm as vkt

        cfg = self._l2_sim_config()
        key = (int(batch), int(seq_len), tuple(sorted(cfg.items())))
        cache = self.__dict__.setdefault("_vllm_norm_trace_cache", {})
        if key not in cache:
            cache[key] = tuple(k for k in vkt.vllm_step_trace(cfg, "prefill", int(batch), int(seq_len),
                               eng=vkt.VllmEngine(sm_count=int(self.core.num_bundle)))
                               if k.name in ("rms_norm", "fused_add_rms_norm"))
        kernels = [k for k in cache[key] if k.op == role]
        if not kernels:
            raise ValueError(f"vLLM trace has no norm kernels for {role!r}")
        seconds, activity = self._price_kernels(kernels, record=False)
        if role in ("layernorm1", "layernorm2"):
            scale = 1.0 / int(self.num_layers)
            seconds, activity = seconds * scale, activity.scaled(scale)
            name = "pointwise-layernorm-f"
        else:
            name = "pointwise-final-norm-f"
            factor = float(max(1, int(self.tp)) * max(1, int(self.cp))) * float(self._moe_dp_attention_groups())
            activity = activity.scaled(factor)
        reg = self._roofline_activity_by_name
        reg[name] = reg.get(name, ZERO_ACTIVITY) + activity
        return seconds

    def _sourced_optimizer_enabled(self) -> bool:
        if (not _settings.get("RAPID_OPTIMIZER_LAUNCH_TRACE")
                or str(self.run_type).lower() != "training"
                or kernel_trace.implementation(self) != "eager"):
            return False
        if (str(self.model_type).lower() not in ("llama", "qwen2", "qwen3")
                or any(int(getattr(self, degree, 1)) != 1 for degree in ("tp", "cp", "pp", "dp", "ep"))
                or bool(getattr(self, "use_moe", False))
                or int(getattr(self, "gradient_accumulation_steps", 1) or 1) != 1
                or bool(getattr(self, "disable_embedding_unembedding", False))
                or bool(getattr(self, "gradient_clipping", False))):
            return False
        return (float(self.precision.parameters) == 4.0
                and float(self.precision.gradients) == 4.0
                and float(self.precision.optimizer_states) == 4.0
                and float(self.precision.master_parameters) == 0.0)

    def _sourced_optimizer_timing(self):
        import kernel_trace_train
        kernels = kernel_trace_train.optimizer_step(self._trace_dims(1, 1), int(self.num_layers))
        seconds, activity = self._price_kernels(kernels, record=False)
        if kernel_trace.l2_residency_enabled():
            activity = replace(activity, hbm=self._sourced_optimizer_hbm)
        return seconds, activity

    def _gemm_n_parts(self, gemm_type) -> Optional[Tuple[int, ...]]:
        if not self._eager():
            return None
        head_dim = int(getattr(self, "head_dim", None) or self.hidden_dim // self.num_heads)
        if gemm_type == GemmType.QKV:
            return (int(self.num_heads) * head_dim, int(self.kv_heads) * head_dim, int(self.kv_heads) * head_dim)
        if gemm_type == GemmType.FFN1:
            return (int(self.intermediate_size), int(self.intermediate_size))
        return None

    def _gemm_forward_parts(self, gemm, name, gemm_type, **kwargs):
        result = self._gemm_forward_parts_base(gemm, name, gemm_type, **kwargs)
        if (self.model_type not in ("qwen2", "qwen3") or self.run_type != "inference"
                or gemm_type != GemmType.QKV):
            return result

        b, m, _, _ = self._expand_gemm_descriptor(gemm)
        d = self._trace_dims(b * m, b)
        if self._eager():
            trace = kernel_trace.layer_trace(0, d, "eager", "x", "y", 4.0)
        else:
            import kernel_trace_vllm as vkt
            cfg = self._l2_sim_config()
            cfg["num_hidden_layers"] = 1
            trace = vkt.vllm_step_trace(cfg, "prefill", b, m,
                                      eng=vkt.VllmEngine(sm_count=int(self.core.num_bundle)))
        extras = []
        for k in trace:
            if k.op != "qkv_proj":
                continue
            if k.kind == "gemm":
                if k.flops or len(k.reads) > 2:
                    extras.append(replace(k, kind="gemm_epilogue", reads=k.reads[2:], writes=(), gemm=None))
            else:
                extras.append(k)
        seconds, activity = self._price_kernels(extras, record=False)
        self._gemm_activity_by_name[name] = self._gemm_activity_by_name.get(name, ZERO_ACTIVITY) + activity
        ret = list(result)
        ret[0] += seconds
        if len(ret) == 5:
            ret[3] += activity.vector_flops
            ret[4] = self._combine_mem(ret[4], {"L2": activity.l2.total_bytes,
                                               f"L{self.num_levels - 1}": activity.hbm.total_bytes})
        return tuple(ret)

    def _gemm_forward_parts_base(self, gemm, name, gemm_type, **kwargs):
        parts = self._gemm_n_parts(gemm_type)
        if parts is None:
            return self.parallelism_gemm_forward(gemm, name, gemm_type=gemm_type, **kwargs)
        b, m, k, n = self._expand_gemm_descriptor(gemm)
        if sum(parts) != n:
            raise ValueError(f"eager split {parts} does not add up to the fused GEMM width {n}")
        tot_t = tot_r = tot_s = tot_f = 0.0
        tot_mem = None
        for part in parts:
            sub = (b, m, k, part) if len(gemm) == 4 else (m, k, part)
            ret = self.parallelism_gemm_forward(sub, name, gemm_type=gemm_type, **kwargs)
            t, r, sz = ret[0], ret[1], ret[2]
            f, mem = (ret[3], ret[4]) if len(ret) == 5 else (0.0, [])
            tot_t += t; tot_r += r; tot_s += sz or 0; tot_f += f or 0.0
            tot_mem = mem if tot_mem is None else self._combine_mem(tot_mem, mem)
        return tot_t, tot_r, tot_s, tot_f, tot_mem

    def _gemm_backward_parts(self, gemm, name, gemm_type):
        parts = self._gemm_n_parts(gemm_type)
        if parts is None:
            return self.parallelism_gemm_backward(gemm, name, gemm_type=gemm_type)
        b, m, k, n = self._expand_gemm_descriptor(gemm)
        if sum(parts) != n:
            raise ValueError(f"eager split {parts} does not add up to the fused GEMM width {n}")
        tot_t = tot_r = tot_s = 0.0
        for part in parts:
            sub = (b, m, k, part) if len(gemm) == 4 else (m, k, part)
            t, r, sz = self.parallelism_gemm_backward(sub, name, gemm_type=gemm_type)[:3]
            tot_t += t; tot_r += r; tot_s += sz or 0
        return tot_t, tot_r, tot_s

    def _apply_eager_training(self, transformer_timings, *, batch: int, seq: int) -> None:
        import kernel_trace_train as kt2
        layers = int(self.num_layers)
        d = self._trace_dims(int(batch) * int(seq), int(batch))
        d = replace(d, act_bytes=kt2.BF16, weight_bytes=kt2.F32, kv_bytes=kt2.BF16)
        opts = kt2.TrainOpts(softmax_reads=3, sm_count=int(self.core.num_bundle),
                             loss_bytes=kt2.F32 if self._implementation_option("fp32_loss") else self.precision.activations)
        checkpoint = bool(getattr(self, "full_recomputation", False))
        if checkpoint and (not getattr(self, "kernel_description_resolved", False)
                           or self._implementation_option("flash_source", None) != "dao"):
            raise ValueError("checkpoint training requires the eager kernel description")
        trace = kt2.train_step_trace(d, layers, opts, checkpoint=checkpoint)
        gemm_ops = {"qkv_proj", "output_proj", "ffn1", "ffn2", "linear_softmax"}
        layer_ops = {"layernorm1", "qkv_proj", "attention", "output_proj", "layernorm2", "ffn1", "gelu", "ffn2"}
        if checkpoint:
            invocations = {op: {phase: DirectionTiming(0.0) for phase in ("forward", "recompute")}
                           for op in layer_ops}
            from collections import Counter
            original = Counter((k.op, k.kind, k.gemm) for k in trace
                               if k.name.startswith("forward.") and k.op in layer_ops
                               and k.kind in ("gemm", "attention"))
            replay = Counter((k.op, k.kind, k.gemm) for k in trace
                             if k.name.startswith("recompute.") and k.op in layer_ops
                             and k.kind in ("gemm", "attention"))
            if replay.keys() - original.keys():
                raise NotImplementedError("checkpoint recomputation introduced an undescribed compute kernel")
            for op in sorted({key[0] for key in original}):
                ratios = {(count + replay[key]) / (2.0 * count)
                          for key, count in original.items() if key[0] == op}
                if len(ratios) != 1:
                    raise NotImplementedError("checkpoint compute shapes need separate pricing")
                factor = ratios.pop()
                timing = transformer_timings[op]
                dt = timing.forward
                base = self._direction_activity(op, "forward", dt)
                invocations[op] = {
                    "forward": replace(dt, activity=base),
                    "recompute": replace(dt, compute_time=dt.compute_time * (2 * factor - 1),
                                         flops=dt.flops * (2 * factor - 1),
                                         activity=base.scaled(2 * factor - 1)),
                }
                transformer_timings[op] = replace(timing, forward=replace(
                    dt, compute_time=dt.compute_time * factor,
                    activity=base.scaled(factor)))
        grouped: Dict[Tuple[str, str], list] = {}
        for k in trace:
            if (k.op == "attention"
                    and k.name.endswith(("flash.gqa_sum_dk", "flash.gqa_sum_dv"))):
                continue
            if k.kind == "gemm" and k.op in gemm_ops and (k.flops or len(k.reads) > 2):
                k = replace(k, kind="gemm_epilogue", reads=k.reads[2:], writes=(), gemm=None)
            if k.kind == "attention" or (k.kind == "gemm" and k.op in gemm_ops) or k.op == "optimizer":
                continue
            op = "attention" if k.op == "rope" else k.op
            direction = "backward" if k.name.startswith("backward.") else "forward"
            grouped.setdefault((op, direction), []).append(k)
        for (op, direction), ks in grouped.items():
            if op not in transformer_timings:
                raise KeyError(f"eager training trace op {op!r} has no model op")
            t, act = self._price_kernels(ks, record=False)
            if op in layer_ops:
                consumers = layers * (2 if checkpoint and direction == "forward" else 1)
                t, act = t / consumers, act.scaled(1.0 / consumers)
            timing = transformer_timings[op]
            dt = getattr(timing, direction)
            if dt is None:
                continue
            notes = dict(getattr(dt, "notes", None) or {})
            notes["eager_trace_activity"] = act.to_dict()
            if op in ("layernorm1", "layernorm2", "gelu", "embedding"):
                new = replace(dt, compute_time=t, activity=act, notes=notes)
            else:
                base = self._direction_activity(op, direction, dt)
                new = replace(dt, compute_time=dt.compute_time + t, activity=base + act, notes=notes)
            transformer_timings[op] = replace(timing, **{direction: new})

        if checkpoint:
            separate = {}
            for k in trace:
                if k.kind == "gemm" and k.op in gemm_ops and (k.flops or len(k.reads) > 2):
                    k = replace(k, kind="gemm_epilogue", reads=k.reads[2:], writes=(), gemm=None)
                op = "attention" if k.op == "rope" else k.op
                if (op not in layer_ops or k.name.startswith("backward.") or k.kind == "attention"
                        or (k.kind == "gemm" and k.op in gemm_ops)):
                    continue
                phase = "recompute" if k.name.startswith("recompute.") else "forward"
                separate.setdefault((op, phase), []).append(k)
            for (op, phase), ks in separate.items():
                t, act = self._price_kernels(ks, record=False)
                t, act = t / layers, act.scaled(1.0 / layers)
                dt = invocations[op][phase]
                invocations[op][phase] = replace(dt, compute_time=dt.compute_time + t,
                                                activity=dt.activity + act,
                                                notes={"eager_trace_activity": act.to_dict()})
            for op, records in invocations.items():
                timing = transformer_timings[op]
                records = {
                    phase: replace(record,
                                   flops=record.activity.useful_tensor_flops + record.activity.vector_flops,
                                   memory_accesses=self._mem_levels([
                                       getattr(record.activity, level).total_bytes
                                       for level in ("register", "smem", "l2", "hbm")]))
                    for phase, record in records.items()
                }
                transformer_timings[op] = replace(timing, metadata={**timing.metadata,
                                                                  "checkpoint_invocations": records})

    def _final_norm_time(self, rows: int, direction: str) -> float:
        if self._eager_training():
            return 0.0
        if direction == "forward" and self._vllm_norms():
            return self._vllm_norm_f(batch=1, seq_len=int(rows), role="linear_softmax")
        if direction == "forward" and self.run_type == "inference" and self._eager():
            d = self._trace_dims(int(rows), 1)
            ks = kernel_trace.rmsnorm_eager("linear_softmax", "x", "y", int(rows), d, "pointwise-final-norm-f", "w")
            return self._price_kernels(ks)[0]
        batch, seq = 1, int(rows)
        names = ("pointwise-final-norm-f",) if direction == "forward" else ("pointwise-final-norm-b", "final_norm_grad_accum")
        factor = 1.0
        if self.run_type == "inference":
            factor = float(max(1, int(self.tp)) * max(1, int(self.cp))) * float(self._moe_dp_attention_groups())
        with self._repeated_activity(factor, *names):
            if direction == "forward":
                t, _, _ = self.get_layernorm_f(batch=batch, seq_len=seq, d_model=self.hidden_dim, name=names[0])
            else:
                t, _, _ = self.get_layernorm_b(batch=batch, seq_len=seq, d_model=self.hidden_dim, name=names[0],
                                               accum_name=names[1])
        return t

    def _l2_capacity_bytes(self) -> float:
        eff = _settings.get("OPMODEL_L2_EFFECTIVE_MIB")
        if eff is not None:
            return float(eff)
        if self.num_levels < 4:
            raise ValueError("RAPID_L2_RESIDENCY needs a memory hierarchy with an L2 level (register, SMEM, L2, HBM)")
        return float(self.memory_hierarchy.mem_layer[2].size)

    def _apply_l2_residency(self, transformer_timings, *, tokens: int, batch: int, past: int, head_tokens: int) -> None:
        if not kernel_trace.l2_residency_enabled():
            return
        if bool(getattr(self, "use_moe", False)):
            raise NotImplementedError("MoE source replay uses price_moe_source_step; legacy operator templates cannot own its rank activity")
        if not past and int(head_tokens) != int(batch):
            raise ValueError("RAPID_L2_RESIDENCY describes last-token logits: set RAPID_PREFILL_LOGITS_LAST_TOKEN=1")
        self._apply_l2_simulation(transformer_timings, phase="decode" if past else "prefill", tokens=tokens,
                                  batch=batch, past=past)

    def _l2_sim_config(self) -> dict:
        head_dim = int(getattr(self, "head_dim", None) or self.hidden_dim // self.num_heads)
        cfg = {"hidden_size": int(self.hidden_dim), "num_attention_heads": int(self.num_heads),
                "num_key_value_heads": int(getattr(self, "kv_heads", self.num_heads)), "head_dim": head_dim,
                "intermediate_size": int(self.intermediate_size), "vocab_size": int(self.vocab_size),
                "num_hidden_layers": int(self.num_layers), "tie_word_embeddings": bool(getattr(self, "tied_embeddings", False)),
                "model_type": str(self.model_type).lower(), "qwen2_bias_mode": getattr(self, "qwen2_bias_mode", None)}
        if getattr(self, "max_position_embeddings", None) is not None:
            cfg["max_position_embeddings"] = int(self.max_position_embeddings)
        if str(getattr(self, "attention_type", "mha")).lower() == "mla":
            cfg.update(attention_type="mla", q_lora_rank=getattr(self, "q_lora_rank", None),
                       kv_lora_rank=int(self.kv_lora_rank), qk_nope_head_dim=int(self.qk_nope_head_dim),
                       qk_rope_head_dim=int(self.qk_rope_head_dim), v_head_dim=int(self.v_head_dim),
                       mla_kv_reads=self._implementation_option("mla_kv_reads", "per_head_group"),
                       mla_prefill_pad_v=bool(self._implementation_option("mla_prefill_pad_v", True)))
        if self._source_moe() and str(self.model_type).lower() != "qwen3_moe":
            moe = self.model.moe
            cfg.update(moe_intermediate_size=int(self.moe_intermediate_size),
                       num_experts=int(self.moe_num_experts), num_experts_per_tok=int(self.moe_top_k),
                       n_shared_experts=int(self.n_shared_experts or 0),
                       first_k_dense_replace=int(moe.first_k_dense_replace), moe_layer_freq=int(moe.moe_layer_freq),
                       scoring_func=moe.scoring_func, topk_method=moe.topk_method, n_group=moe.n_group,
                       topk_group=moe.topk_group, norm_topk_prob=moe.norm_topk_prob,
                       routed_scaling_factor=moe.routed_scaling_factor,
                       vllm_use_fused_moe_grouped_topk=bool(self._implementation_option(
                           "vllm_use_fused_moe_grouped_topk", True)),
                       moe_dp=int(self.moe_dp), moe_tile_rule=self.model.moe_tile_rule,
                       moe_routing_policy=moe.routing_policy, expert_imbalance_factor=moe.expert_imbalance_factor,
                       moe_grouped_traffic=self.model.moe_grouped_traffic)
        if str(self.model_type).lower() == "qwen3_moe":
            cfg.update(moe_intermediate_size=int(self.moe_intermediate_size),
                       num_experts=int(self.moe_num_experts), num_experts_per_tok=int(self.moe_top_k),
                       moe_dp=int(self.moe_dp), moe_tile_rule=self.model.moe_tile_rule,
                       moe_routing_policy=self.model.moe.routing_policy,
                       expert_imbalance_factor=self.model.moe.expert_imbalance_factor,
                       moe_grouped_traffic=self.model.moe_grouped_traffic)
        if "moe_dp" in cfg:
            backend = self._implementation_option("vllm_all2all_backend", "naive")
            if backend != "naive":
                cfg["vllm_all2all_backend"] = backend

        linear_weights = getattr(self.precision, "linear_weights", None)
        if linear_weights is not None:
            cfg["linear_weight_bytes"] = float(linear_weights)
        return cfg

    def price_moe_source_step(self, phase, batch, seq, **options):
        if not self._source_moe():
            raise ValueError("price_moe_source_step requires a vLLM source MoE model (qwen3_moe, DeepSeek V2/V3 MoE)")
        from moe_trace_pricing import price_moe_rank_step
        cfg = self._l2_sim_config()
        tile_rule = options.pop("moe_tile_rule", self.model.moe_tile_rule)
        return price_moe_rank_step(self, cfg, phase, batch, seq, tile_rule, **options)

    def _apply_l2_simulation(self, transformer_timings, *, phase: str, tokens: int, batch: int, past: int) -> None:
        import l2_sim
        if int(self.tp) > 1 and str(self.run_type).lower() == "inference":
            from program import vtp
            vtp.apply(self, transformer_timings, phase=phase, tokens=tokens, batch=batch, past=past)
            return
        training = str(self.run_type).lower() == "training"
        impl = kernel_trace.implementation(self)
        if training and impl == "fused":
            from program import megatron
            megatron.validate(self)
            return
        if getattr(self, "l2_bypass", False):
            return
        if impl == "eager":
            self._eager()
            if training:
                self._validate_eager_training_l2_trace()
        else:
            if training:
                raise NotImplementedError("RAPID_L2_RESIDENCY with the fused (vLLM) description is described for "
                                          "inference only")
            if (int(self.tp) != 1 or int(self.cp) != 1 or int(getattr(self, "pp", 1)) != 1
                    or bool(getattr(self, "use_moe", False)) or not llm_util.is_llama_style(self.model_type)
                    or self._is_vit_model() or (str(getattr(self, "attention_type", "mha")).lower() not in ("mha", "gqa")
                                                and not self._mla_vllm_trace())):
                raise ValueError("RAPID_L2_RESIDENCY is described for single-GPU dense Llama-style MHA/GQA layers "
                                 "and MLA under mla_inference=vllm_trace only")
        checkpoint = training and bool(getattr(self, "full_recomputation", False))
        if float(self.precision.activations) != 2.0 or (not training and (float(self.precision.kv_cache) != 2.0
                                                                        or float(self.precision.parameters) != 2.0)):
            raise ValueError("the L2 simulation's traces are described for 16-bit activations, weights (inference) and KV cache")

        impl_now = kernel_trace.implementation(self)
        flash_source = self._implementation_option("flash_source", None)
        if (not getattr(self, "kernel_description_resolved", False)
                or (impl_now, flash_source) not in (("fused", "vllm"), ("eager", "dao"))):
            raise ValueError("RAPID_L2_RESIDENCY requires a resolvable kernel description")
        engine_settings = dict(getattr(self, "engine_settings", None) or {})
        if engine_settings and impl_now != "fused":
            raise ValueError("model_param.engine applies to the fused kernel description only")
        cfg = self._l2_sim_config()
        sm_count = int(self.core.num_bundle)

        spec = sim_settings.CacheSpec.from_hw(self.hw_config)
        policy = sim_settings.SimPolicy.current(spec)
        sharing = policy.sharing_classes
        physical, copies = 0, None
        if sharing:
            if self.num_levels < 4:
                raise ValueError("RAPID_L2_SHARING_CLASSES needs a memory hierarchy with an L2 level (register, SMEM, "
                                 "L2, HBM)")
            physical = int(self.memory_hierarchy.mem_layer[2].size)
            copies = policy.private_copies
        from opmodel import settings as _gemm_settings
        forced_path = _gemm_settings.get("OPMODEL_GEMM_FORCED_TEMPLATES")
        selected_split_k = _gemm_settings.get("OPMODEL_GEMM_CATALOG") == "cublaslt_sm80_bf16_splitk"
        split_backend = self._opmodel_backend or self._gemm_activity_backend
        key = (forced_path, impl, tuple(sorted(engine_settings.items())), training, checkpoint, self._implementation_option("fp32_loss"), phase, int(batch), int(tokens), int(past), tuple(sorted(cfg.items())),
               self._l2_capacity_bytes(), sm_count, physical, spec.partitions, copies, policy.gemm_min_tile_m, policy.steady_state, policy.address_plan)
        if split_backend is None:
            raise ValueError("the L2 simulation's GEMM operand passes require op-model GEMM pricing")

        key = (key, repr(split_backend.hardware), split_backend.gemm_selection_objective,
               bool(getattr(self, "_gemv_disabled", False)),
               tuple((k, repr(_gemm_settings.get(k))) for k in sorted(_gemm_settings.registry())))
        cache = getattr(TimeCalculationLLM, "_l2_sim_cache", None)
        if cache is None:
            cache = TimeCalculationLLM._l2_sim_cache = {}
        if key not in cache:
            if training:
                seq = int(tokens) // max(1, int(batch))
                ks = l2_sim.train_launch_trace(cfg, int(batch), seq, sm_count=sm_count, checkpoint=checkpoint,
                                               loss_bytes=4.0 if self._implementation_option("fp32_loss") else self.precision.activations)
                rules = l2_sim.train_rules(cfg, checkpoint=checkpoint)
            elif impl == "fused":
                seq = int(tokens) // max(1, int(batch)) if phase == "prefill" else int(past)
                import kernel_trace_vllm as _kv
                ks = l2_sim.vllm_launch_trace(cfg, phase, int(batch), seq, sm_count=sm_count,
                                              engine=_kv.VllmEngine(sm_count=sm_count, **engine_settings))
                rules = l2_sim.vllm_rules(cfg, phase)
            else:
                seq = int(tokens) // max(1, int(batch)) if phase == "prefill" else int(past)
                ks = l2_sim.inference_launch_trace(cfg, phase, int(batch), seq, sm_count=sm_count)
                rules = l2_sim.inference_rules(cfg, phase)
            if forced_path:
                from opmodel.models.extended_roofline import resolve_forced_template_for_shape
                def resolve_trace_forced(shape, role="forward"):
                    return resolve_forced_template_for_shape(shape, role, hardware=split_backend.hardware)
                ks, split_counts = l2_sim.expand_forced_split_k(ks, resolve_trace_forced)
            elif selected_split_k:
                def select_trace_kernel(kernel):
                    return split_backend.selected_gemm_template(
                        dim1=kernel.gemm[0], dim2=kernel.gemm[1], dim3=kernel.gemm[2],
                        name=kernel.name, role=l2_sim.gemm_trace_role(kernel),
                        allow_gemv=not getattr(self, "_gemv_disabled", False))
                ks, split_counts = l2_sim.expand_selected_split_k(ks, select_trace_kernel)
            else:
                split_counts = {}

            passes = self._trace_gemm_passes(ks, named=selected_split_k, roles=True)
            if sharing:
                res = l2_sim.simulate(ks, physical // int(spec.partitions), rules, sharing_classes=True,
                                      physical_capacity_bytes=physical, private_copies=copies,
                                      gemm_min_tile_m=policy.gemm_min_tile_m, steady_state=policy.steady_state, address_plan=policy.address_plan,
                                      operand_passes=passes)
            else:
                res = l2_sim.simulate(ks, int(self._l2_capacity_bytes()), rules, gemm_min_tile_m=policy.gemm_min_tile_m,
                                      steady_state=policy.steady_state, address_plan=policy.address_plan,
                                      operand_passes=passes)
            cache[key] = (ks, res, split_counts)
        ks, res, split_counts = cache[key]
        self.__dict__.setdefault("split_k_trace_counts_by_phase", {})[phase] = split_counts
        if forced_path or selected_split_k:
            from opmodel.models import extended_roofline as _extended_roofline
            if not hasattr(_extended_roofline, "_TRACE_SPLIT_K_BY_PHASE"):
                _extended_roofline._TRACE_SPLIT_K_BY_PHASE = {}
            _extended_roofline._TRACE_SPLIT_K_BY_PHASE[phase] = split_counts

        from ncu_counts import memcpy_activity
        self.__dict__.setdefault("counts_memcpy_by_phase", {})[phase] = (
            memcpy_activity(ks, res), training and impl == "eager")
        layers = int(self.num_layers)
        layer_ops = {"layernorm1", "qkv_proj", "rope", "attention", "output_proj", "layernorm2", "ffn1", "gelu", "ffn2", "MLP"}
        sim = {}
        invocation_sim = {}
        sourced_optimizer = self._sourced_optimizer_enabled()
        if sourced_optimizer:
            self._sourced_optimizer_hbm = MemoryTraffic()
        for k, rd, wr in zip(ks, res.hbm_read, res.hbm_write):
            name = k.name
            if k.op == "optimizer":
                if sourced_optimizer:
                    self._sourced_optimizer_hbm = self._sourced_optimizer_hbm + MemoryTraffic(rd, wr)
                continue
            direction = "backward" if name.startswith("backward.") else "forward"
            op = l2_sim.trace_op_name(k)
            if op == "rope" and (training or "rope" not in transformer_timings):
                op = "attention"
            if op not in transformer_timings:
                continue
            acc = sim.setdefault((op, direction), [0.0, 0.0, 0.0, 0.0])
            acc[0] += rd; acc[1] += wr
            other_read = other_write = 0.0
            if k.kind not in ("gemm", "attention"):
                acc[2] += rd; acc[3] += wr
                other_read, other_write = rd, wr
            if checkpoint and op in layer_ops and direction == "forward":
                invocation = "recompute" if name.startswith("recompute.") else "forward"
                dst = invocation_sim.setdefault((op, invocation), [0.0] * 4)
                for i, value in enumerate((rd, wr, other_read, other_write)):
                    dst[i] += value
        for (op, direction), (srd, swr, ord_, owr) in sim.items():
            scale = 1.0 / layers if op in layer_ops else 1.0
            if checkpoint and op in layer_ops and direction == "forward":
                scale /= 2.0
            timing = transformer_timings[op]
            dt = getattr(timing, direction)
            if dt is None:
                continue
            act = self._direction_activity(op, direction, dt)
            write = swr * scale
            read = srd * scale
            notes = dict(getattr(dt, "notes", None) or {})
            notes["l2_sim"] = {"sim_read": read, "sim_write": write,
                               "model_hbm": act.hbm.read_bytes + act.hbm.write_bytes}
            if "eager_trace_activity" in notes:
                tr = dict(notes["eager_trace_activity"])
                tr["hbm"] = {"read_bytes": ord_ * scale, "write_bytes": owr * scale}
                notes["eager_trace_activity"] = tr
            new = replace(dt, activity=replace(act, hbm=MemoryTraffic(read_bytes=read, write_bytes=write)), notes=notes)
            transformer_timings[op] = replace(timing, **{direction: new})

        for (op, invocation), (srd, swr, ord_, owr) in invocation_sim.items():
            timing = transformer_timings[op]
            records = dict(timing.metadata["checkpoint_invocations"])
            dt = records[invocation]
            scale = 1.0 / layers
            notes = dict(dt.notes)
            notes["l2_sim"] = {"sim_read": srd * scale, "sim_write": swr * scale,
                               "model_hbm": dt.activity.hbm.total_bytes}
            if "eager_trace_activity" in notes:
                tr = dict(notes["eager_trace_activity"])
                tr["hbm"] = {"read_bytes": ord_ * scale, "write_bytes": owr * scale}
                notes["eager_trace_activity"] = tr
            hbm = MemoryTraffic(read_bytes=srd * scale, write_bytes=swr * scale)
            records[invocation] = replace(dt, notes=notes,
                                         memory_accesses={**dt.memory_accesses, f"L{self.num_levels - 1}": hbm.total_bytes},
                                         activity=replace(dt.activity, hbm=hbm))
            transformer_timings[op] = replace(timing, metadata={**timing.metadata,
                                                              "checkpoint_invocations": records})

    def _trace_gemm_passes(self, kernels, *, named=False, name="__l2sim_gemm_probe__", kinds=("gemm",),
                           roles=False) -> dict:
        import l2_sim
        out = {}
        for index, kernel in enumerate(kernels):
            if kernel.kind not in kinds or not kernel.gemm:
                continue
            batch = int(getattr(kernel, "batch", 1) or 1) if kernel.kind == "batched_gemm" else 1
            role = l2_sim.gemm_trace_role(kernel) if roles else None
            from program.vtp import gemm_weight_bytes
            passes = self._gemm_operand_passes(*kernel.gemm, name=kernel.name if named else name,
                                               role=role, batch=batch,
                                               **({} if (wb := gemm_weight_bytes(self, kernel)) is None else {"weight_bytes": wb}))
            if any(p > 1.0 for p in passes):
                out[index] = passes
        return out

    def _gemm_operand_passes(self, m: int, k: int, n: int, *, name: str, role=None, batch: int = 1,
                             weight_bytes=None):
        cache = self.__dict__.setdefault("_gemm_operand_passes_cache", {})
        gemv_disabled = bool(getattr(self, "_gemv_disabled", False))
        key = (int(m), int(k), int(n), str(name), role, int(batch), gemv_disabled)
        wb = {} if weight_bytes is None else {"weight_bytes": float(weight_bytes)}
        if wb:
            key += (wb["weight_bytes"],)
        if key not in cache:
            import l2_sim
            backend = self._opmodel_backend or self._gemm_activity_backend
            if backend is None:
                raise ValueError("GEMM operand passes require op-model GEMM pricing")
            result = backend.predict_batched_gemm(
                dim1=int(m), dim2=int(k), dim3=int(n), batch=max(1, int(batch)), name=str(name),
                allow_gemv=not gemv_disabled, **({"gemm_role": role} if role is not None else {}), **wb)
            cache[key] = l2_sim.gemm_passes_from_traffic(dict(result.diagnostics)["transaction_bytes"])
        return cache[key]

    def _sourced_vllm_rule(self, setting: str) -> bool:
        raw_model = getattr(self, "_raw_model_config", None)
        framework = getattr(getattr(raw_model, "model_config", raw_model), "framework", None)
        return (bool(_env_flag(setting)) and self.run_type == "inference"
                and framework in (None, "vllm-0.10.2-v0")
                and kernel_trace.implementation(self) == "fused"
                and self._implementation_option("flash_source", None) == "vllm"
                and ((str(self.model_type).lower() in ("llama", "qwen2", "qwen3")
                      and str(getattr(self, "attention_type", "mha")).lower() in ("mha", "gqa"))

                     or (self._mla_vllm_trace() and setting != "RAPID_VLLM_ATTN_OUTPUT_ZEROS"
                         and str(self.model_type).lower() in ("llama", "deepseek_v3")))
                and not bool(getattr(self, "use_moe", False))
                and int(self.tp) == 1 and int(self.cp) == 1
                and float(self.precision.activations) == 2.0
                and float(self.precision.parameters) == 2.0)

    def _sourced_vllm_trace(self, batch: int):
        import kernel_trace_vllm
        cfg = dict(self._l2_sim_config(), num_hidden_layers=1)
        engine = kernel_trace_vllm.VllmEngine(
            sm_count=int(self.core.num_bundle), **dict(getattr(self, "engine_settings", None) or {}))
        return kernel_trace_vllm.vllm_step_trace(cfg, "prefill", int(batch), 1,
                                               engine, tp=int(self.tp), rank=0)

    def _vllm_attn_output_zero_cost(self, token_rows: int):
        if not self._sourced_vllm_rule("RAPID_VLLM_ATTN_OUTPUT_ZEROS"):
            return 0.0, None
        head_dim = int(getattr(self, "head_dim", None) or self.hidden_dim // self.num_heads)
        output_bytes = (float(token_rows) * (int(self.num_heads) // int(self.tp))
                        * head_dim * float(self.precision.activations))
        kernel = kernel_trace.Kernel(
            "attention", "attn_out_zeros", "pointwise", (),
            (kernel_trace.Access("attention.o", output_bytes),))
        return self._price_kernels((kernel,), record=False)

    def _kv_cache_write_cost(self, token_rows: int):
        import os as _os
        if self.run_type == "inference" and self._eager():
            past = int(getattr(self, "_eager_kv_past", 0) or 0)
            batch = int(getattr(self, "_eager_kv_batch", 1) or 1)
            d = self._trace_dims(token_rows, batch, past=past)
            ks = [kernel_trace.kv_update_eager("k", "k", "kc0", "kc", d), kernel_trace.kv_update_eager("v", "v", "vc0", "vc", d)]
            return self._price_kernels(ks, record=False)
        if self.run_type != "inference" or not _env_flag("RAPID_KV_CACHE_WRITE"):
            return 0.0, None
        if str(getattr(self, "attention_type", "mha")).lower() == "mla":
            return 0.0, None
        kv_heads_local = max(1, math.ceil(int(self.kv_heads) / max(1, int(self.tp))))
        head_dim = int(getattr(self, "head_dim", None) or self.hidden_dim // self.num_heads)
        kv_bytes = float(token_rows) * 2 * kv_heads_local * head_dim * float(self.precision.kv_cache)
        hbm_bytes = 2.0 * kv_bytes
        seconds = self.vector_roofline(
            0.0, hbm_bytes, mem_level=self.num_levels - 1,
            active_bundles=min(int(token_rows), self.core.num_bundle),
        ) + self.O
        seconds = self.kernel_floor(seconds)
        level = self.num_levels - 1

        traffic = MemoryTraffic(read_bytes=kv_bytes, write_bytes=kv_bytes)
        activity = OperationActivity(hbm=traffic, l2=traffic)
        return seconds, activity

    def _with_kv_cache_write(self, op: OperationTiming, token_rows: int) -> Tuple[OperationTiming, float]:
        seconds, activity = self._kv_cache_write_cost(token_rows)
        zero_seconds, zero_activity = self._vllm_attn_output_zero_cost(token_rows)
        if zero_seconds:
            seconds += zero_seconds
            activity = (activity or ZERO_ACTIVITY) + zero_activity
        if not seconds:
            return op, 0.0
        fwd = op.forward

        base = self._direction_activity(op.name, "forward", fwd)
        fwd = replace(fwd, compute_time=fwd.compute_time + seconds, activity=base + activity)
        return replace(op, forward=fwd), seconds

    def _inference_rope_timing(self, qkv_gemm: Tuple[int, ...]) -> Optional[OperationTiming]:
        if self.run_type != "inference" or not getattr(self, "use_rope", False):
            return None

        if _env_flag("RAPID_ROPE_FUSED_KV_WRITE"):
            return None
        if str(self.model_type).lower() not in ("llama", "qwen2", "qwen3") or self.attention_type not in ("mha", "gqa"):
            raise ValueError("attention.use_rope currently supports llama-style MHA/GQA inference only")
        if self.cp != 1:
            raise ValueError("attention.use_rope requires cp=1; inference CP decode is not modeled")
        tp = int(self.tp)
        q_heads = int(self.num_heads)
        kv_heads = int(self.kv_heads)
        head_dim = int(self.head_dim)
        if q_heads % tp or kv_heads % tp or kv_heads < tp:
            raise ValueError("attention.use_rope requires whole Q and K heads on every TP rank")
        if head_dim % 2:
            raise ValueError("attention.use_rope requires an even full-rotation head_dim")

        spec = self._shard_gemm_descriptor(qkv_gemm, GemmType.QKV)
        local_q_heads = q_heads // tp
        local_k_heads = kv_heads // tp
        expected_width = (local_q_heads + 2 * local_k_heads) * head_dim
        if spec.shard_n != expected_width:
            raise ValueError("attention.use_rope QKV shard does not preserve whole Q/K heads")
        token_rows = spec.shard_batch * spec.shard_m
        if self._eager():
            d = self._trace_dims(token_rows, spec.shard_batch)
            ks = (kernel_trace.rope_eager("q", "qr", token_rows, local_q_heads, d, "cos", "sin")
                  + kernel_trace.rope_eager("k", "kr", token_rows, local_k_heads, d, "cos", "sin"))
            t, act = self._price_kernels(ks, record=False)
            return OperationTiming("rope", forward=DirectionTiming(
                compute_time=t, flops=act.vector_flops,
                memory_accesses={f"L{self.num_levels - 1}": act.hbm.read_bytes + act.hbm.write_bytes},
                activity=act, notes={"token_rows": token_rows, "implementation": "eager"}), backward=None)
        rotated_elements = token_rows * (local_q_heads + local_k_heads) * head_dim
        flops = 3 * rotated_elements
        qk_bytes = rotated_elements * self.precision.activations
        cache_bytes = token_rows * head_dim * self.precision.activations
        position_bytes = token_rows * 8
        read_bytes = qk_bytes + cache_bytes + position_bytes
        write_bytes = qk_bytes
        hbm_bytes = read_bytes + write_bytes
        compute_time = self.vector_roofline(
            flops,
            hbm_bytes,
            mem_level=self.num_levels - 1,
            active_bundles=min(token_rows, self.core.num_bundle),
        ) + self.O
        compute_time = self.kernel_floor(compute_time)
        return OperationTiming(
            "rope",
            forward=DirectionTiming(
                compute_time=compute_time,
                flops=flops,
                memory_accesses={f"L{self.num_levels - 1}": hbm_bytes},

                activity=OperationActivity(
                    vector_flops=flops,
                    hbm=MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes),
                    l2=MemoryTraffic(read_bytes=read_bytes, write_bytes=write_bytes),
                ),
                notes={"token_rows": token_rows, "rotated_elements": rotated_elements},
            ),
            backward=None,
        )

    def get_layernorm_f(self, batch, seq_len, d_model, comm_after=False, name="pointwise-layernorm-f"):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()

        if self.run_type == "inference" and self._eager():
            rows = int(batch) * int(seq_len)
            ks = kernel_trace.rmsnorm_eager("layernorm", "x", "y", rows, self._trace_dims(rows, batch),
                                            "pointwise-layernorm-f", "w")
            return self._price_kernels(ks)[0], 0.0, 0
        elements = batch * math.ceil(seq_len / seq_degree) * d_model

        compute_flops = elements * (kernel_trace.RMSNORM_FLOPS if kernel_trace.uses_rmsnorm(self.model_type)
                                    else LAYER_NORM_FORWARD_FLOPS_PER_ELEMENT)

        mem_bytes = (
            self.precision.activations * elements * LAYER_NORM_FORWARD_MEM_ACCESSES
            + self.precision.stats * 2 * (elements / max(1, d_model))
        )
        _vllm_pw = self.run_type == "inference" and _vllm_pointwise()
        compute_time = self.vector_roofline(
            compute_flops,
            mem_bytes,
            name=name,
            mem_level=self.num_levels - 1,
            active_bundles=(min(int(batch) * math.ceil(seq_len / seq_degree), self.core.num_bundle) if _vllm_pw else None),
        ) + (1 if _vllm_pw else 3) * self.O
        compute_time = self.kernel_floor(compute_time, 1 if _vllm_pw else 3)
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            per_rank_bytes = self.precision.activations * self._layernorm_gather_elements(
                batch, seq_len, d_model
            )
            total_bytes = int(math.ceil(per_rank_bytes))
            reduction_time = self.network_model.collective(
                kind=ALL_GATHER,
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label="layernorm_f_all_gather",
                axis="tp",
            )
        else:
            reduction_time = 0.0
            total_bytes = 0

        return compute_time, reduction_time, total_bytes

    def get_layernorm_b(self, batch, seq_len, d_model, type = Optional, name="pointwise-layernorm-b",
                        accum_name="layernorm_grad_accum"):
        tp_mode = self.get_parallelism_mode()
        seq_degree = self._sequence_parallel_degree()

        elements = batch * math.ceil(seq_len / seq_degree) * d_model
        compute_flops = elements * LAYER_NORM_BACKWARD_FLOPS_PER_ELEMENT

        mem_bytes = (
            self.precision.activations * elements * LAYER_NORM_BACKWARD_MEM_ACCESSES
            + self.precision.stats * 2 * (elements / max(1, d_model))
        )

        compute_time = self.vector_roofline(
            compute_flops,
            mem_bytes,
            name=name,
            mem_level=self.num_levels - 1,
        ) + 4 * self.O
        compute_time += self._grad_accum_time(2 * d_model, accum_name)
        if tp_mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID) :
            per_rank_bytes = self.precision.activations * self._layernorm_gather_elements(
                batch, seq_len, d_model
            )
            total_bytes = int(math.ceil(per_rank_bytes))
            reduction_time = self.network_model.collective(
                kind=ALL_GATHER,
                size_bytes=total_bytes,
                participants=self.tp,
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                local_bytes=0,
                debug_label="layernorm_b_all_gather",
                axis="tp",
            )

        else:
            reduction_time = 0.0
            total_bytes = 0

        return compute_time, reduction_time, total_bytes
    def get_router_f(self, gemm_router, gemm_ffn1, *, batch_size: int, seq_len: int):
        tokens_owner = self._moe_tokens_owner(batch_size, seq_len) // self._moe_dp_attention_groups()
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm_router)
        gemm_time = self.get_gemm_time(tokens_owner, k, n, "router_f")[0]
        elements = tokens_owner * n
        flops = elements * SOFTMAX_FORWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * SOFTMAX_FORWARD_MEM_ACCESSES

        compute_time = gemm_time + self.roofline(flops, mem_bytes, name="pointwise-router-f", mem_level=self.num_levels - 1) + self.O
        compute_time += self._moe_permute_kernels_time(tokens_owner)

        return compute_time, 0.0, 0
    def get_router_b(self, gemm_router, gemm_ffn1, *, batch_size: int, seq_len: int):
        tokens_owner = self._moe_tokens_owner(batch_size, seq_len)
        _batch, _m, k, n = self._expand_gemm_descriptor(gemm_router)

        grad_time_act = self.get_gemm_time(tokens_owner, n, k, "router_b", gemm_role="dgrad")[0]
        grad_time_wt = self.get_gemm_time(k, tokens_owner, n, "router_b", gemm_role="wgrad")[0]
        gemm_time = grad_time_act + grad_time_wt

        elements = tokens_owner * n
        flops = elements * SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT
        mem_bytes = self.precision.activations * elements * SOFTMAX_BACKWARD_MEM_ACCESSES

        compute_time = gemm_time + self.roofline(flops, mem_bytes, name="pointwise-router-b", mem_level=self.num_levels - 1) + self.O
        return compute_time, 0.0, 0

    def get_embedding_b(self, vocab_size, seq_len, hidden_dim):
        if self._is_vit_model():
            batch = self._effective_transformer_batch()
            patch_gemm = self._vit_patch_embed_gemm(batch, hidden_dim)
            gemm_time, _, _ = self.single_gpu_gemm_backward(
                patch_gemm,
                "vit_patch_embed_b",
            )
            token_bytes = float(batch) * float(seq_len) * float(hidden_dim) * float(self.precision.gradients)
            prep_time = self.vector_roofline(
                0.0,
                token_bytes,
                name="vit_embedding_b",
                mem_level=self.num_levels - 1,
            ) + self.O
            return gemm_time + prep_time

        batch = self._effective_transformer_batch()

        embedding_params = math.ceil((vocab_size * hidden_dim) / max(1, self.tp))
        embedding_mem = embedding_params * self.precision.gradients + seq_len * batch * hidden_dim * self.precision.gradients
        embedding_mem_time = self.vector_roofline(
            0,
            embedding_mem,
            name="embedding_b",
            mem_level=self.num_levels - 1,
        ) + self.O
        embedding_mem_time += self._grad_accum_time(embedding_params, "embedding_grad_accum")

        if self.debug:
            print("(gr) Embedding_mem: {:,}".format(int(embedding_mem / 1e9)))
        return embedding_mem_time

    def get_inter_layer_comm_latency_llm(self, batch_size, hidden_dim, seq_len):
        w = 0
        w_size = 0
        if self.pp > 1:
            w_size = self.precision.activations * batch_size * hidden_dim * seq_len
            if self.run_type == "training" and not getattr(self, "tp_sp", False):
                w_size *= max(1, int(self.tp))

            rank_bytes = math.ceil(w_size / (max(1, int(self.tp)) * max(1, int(self.cp))))
            transfer_time = rank_bytes / self.links["pp"].bandwidth + self.links["pp"].latency
            mem_time = self.roofline(0, 2 * w_size, name="inter_layer", mem_level=self.num_levels - 1)

            w = mem_time + transfer_time
        return w, w_size

    def get_data_parallel_reduction_sizes(self, d, intermediate_size, *, moe: bool = False):
        if int(getattr(self, "dp", 1) or 1) * max(1, int(getattr(self, "cp", 1) or 1)) <= 1:
            return 0
        params = self.layer_params_per_rank(d, intermediate_size, moe=moe)
        grad_comm = self.precision.grad_communication
        if not moe:
            return sum(
                math.ceil(grad_comm * params[k]) for k in ("qkv", "output", "ffn1", "ffn2")
            )

        return (
            math.ceil(grad_comm * params["qkv"])
            + math.ceil(grad_comm * params["output"])
            + math.ceil(grad_comm * (params["experts"] + params["router"]))
        )

    def get_data_parallel_reduction_llm(self, d, intermediate_size, *, moe: bool = False):
        if self.use_moe and self._zero_moe_update_sharded() and str(getattr(self.model, "run_type", "training")).lower() != "inference":
            dense, expert = self.zero_moe_layer_param_groups(d, intermediate_size, moe=moe)
            return self._zero_moe_apply_grad(dense, self.dp * self.ep) + self._zero_moe_apply_grad(expert, self.dp)
        params = self.layer_params_per_rank(d, intermediate_size, moe=moe)
        order = ("qkv", "output", "ffn1", "ffn2") if not moe else ("qkv", "output", "experts", "router")
        grad_shard = self.dp if (self.zero_stage >= 1 and self.dp > 1) else 1

        with self._repeated_activity(1.0 / grad_shard, "apply_grad", "pointwise-grad-clipping"):
            apply_grad_time = sum(self.apply_grad(int(params[k])) for k in order)

        if grad_shard > 1:
            apply_grad_time /= grad_shard

        return apply_grad_time

    def get_embedding_apply_grad_llm(self, hidden_dim, vocab_size):
        params, _ = self._embedding_softmax_params_per_rank(hidden_dim, vocab_size)
        return self._apply_grad_sharded(params)

    def get_softmax_apply_grad_llm(self, hidden_dim, vocab_size):
        _, params = self._embedding_softmax_params_per_rank(hidden_dim, vocab_size)
        return self._apply_grad_sharded(params)

    def _zero_moe_update_sharded(self) -> bool:
        if self.zero_stage >= 2:
            return True
        return self.zero_stage >= 1 and int(self.dp) * int(self.ep) > 1

    def _apply_grad_sharded(self, params: float) -> float:
        if params <= 0:
            return 0.0

        if self.use_moe and self._zero_moe_update_sharded() and str(getattr(self.model, "run_type", "training")).lower() != "inference":
            return self._zero_moe_apply_grad(params, self.dp * self.ep)
        grad_shard = self.dp if (self.zero_stage >= 1 and self.dp > 1) else 1

        with self._repeated_activity(1.0 / grad_shard, "apply_grad", "pointwise-grad-clipping"):
            t = self.apply_grad(int(params))
        return t / grad_shard if grad_shard > 1 else t

    def _zero_moe_apply_grad(self, params: float, shards: int) -> float:
        if params <= 0:
            return 0.0
        factor = 1.0 / max(1, int(shards))
        with self._repeated_activity(factor, "apply_grad", "pointwise-grad-clipping"):
            return self.apply_grad(int(params)) * factor

    def _embedding_softmax_params_per_rank(
        self, hidden_dim: int, vocab_size: int
    ) -> Tuple[float, float]:
        if getattr(self, "disable_embedding_unembedding", False):
            return 0.0, 0.0
        if self._is_vit_model():
            embedding = float(
                llm_util.vit_patch_embed_param_count(
                    hidden_dim=hidden_dim,
                    patch_size=self.patch_size,
                    in_chans=self.in_chans,
                )
            )
            output = float(
                llm_util.vit_head_param_count(
                    hidden_dim=hidden_dim,
                    num_classes=getattr(self, "num_classes", 0),
                )
            )
            return embedding, output
        embedding = float(vocab_size) * float(hidden_dim) / float(max(1, int(self.tp)))
        output = 0.0 if self.tied_embeddings else embedding
        return embedding, output

    def _combine_mem(self, *args):
            combined = {}
            for d in args:
                mem_levels = self._mem_levels(d)
                for k, v in mem_levels.items():
                    combined[k] = combined.get(k, 0.0) + (v or 0.0)
            return combined

    def _mem_levels(self, arr):
        if isinstance(arr, MappingABC):
            return {str(k): float(v) for k, v in arr.items()}
        if isinstance(arr, (int, float)):
            return {f"L{i}": (float(arr) if i == self.num_levels - 1 else 0.0) for i in range(self.num_levels)}
        if isinstance(arr, SequenceABC) and not isinstance(arr, (str, bytes)):
            return {f"L{i}": float(v) for i, v in enumerate(arr)}
        return {}

    def _mem_levels_to_list(self, arr: Any) -> List[int]:
        levels = self._mem_levels(arr)
        return [
            int(math.ceil(float(levels.get(f"L{i}", 0.0) or 0.0)))
            for i in range(self.num_levels)
        ]

    @contextmanager
    def _repeated_activity(self, factor: float, *names: str):
        registries = (
            self._gemm_activity_by_name,
            self._roofline_activity_by_name,
        )
        prior = [{name: reg.pop(name, ZERO_ACTIVITY) for name in names} for reg in registries]
        try:
            yield
        finally:
            for reg, before in zip(registries, prior):
                for name in names:
                    fresh = reg.pop(name, ZERO_ACTIVITY)
                    total = before[name] + fresh.scaled(float(factor))
                    if total != ZERO_ACTIVITY:
                        reg[name] = total

    def _gemm_activity(self, name: str) -> OperationActivity:
        return getattr(self, "_gemm_activity_by_name", {}).pop(name, ZERO_ACTIVITY)

    def _roofline_activity(self, *names: str) -> OperationActivity:
        registry = getattr(self, "_roofline_activity_by_name", {})
        activity = ZERO_ACTIVITY
        for name in names:
            activity = activity + registry.pop(name, ZERO_ACTIVITY)
        return activity

    def _vector_activity(
        self, flops: float, memory_accesses: Any
    ) -> OperationActivity:
        levels = self._mem_levels(memory_accesses)
        traffic = [
            MemoryTraffic(read_bytes=float(levels.get(f"L{i}", 0.0) or 0.0))
            for i in range(self.num_levels)
        ]
        traffic += [MemoryTraffic()] * max(0, 4 - len(traffic))
        return OperationActivity(
            vector_flops=float(flops or 0.0),
            register=traffic[0],
            smem=traffic[1],
            l2=traffic[2],
            hbm=traffic[self.num_levels - 1],
        )

    def _direction_activity(
        self,
        op_name: str,
        direction: str,
        timing: DirectionTiming,
    ) -> OperationActivity:
        suffix = "f" if direction == "forward" else "b"
        gemm_names = {
            "qkv_proj": (
                f"qkv_projection_{suffix}",
                f"decode_qkv_proj_{suffix}",
                f"mla_qkv_proj_{suffix}",
            ),
            "attention": (
                f"attention_score_{suffix}",
                f"attention_output_{suffix}",
                f"decode_attention_score_{suffix}",
                f"decode_attention_output_{suffix}",
                f"mla_attention_score_{suffix}",
                f"mla_attention_output_{suffix}",
            ),
            "output_proj": (
                f"output_projection_{suffix}",
                f"decode_output_projection_{suffix}",
                f"mla_output_proj_{suffix}",
            ),
            "ffn1": (f"ffn_{suffix}", f"ffn1_{suffix}", f"decode_ffn1_{suffix}"),
            "ffn2": (f"ffn2_{suffix}", f"decode_ffn2_{suffix}"),
            "router": (f"router_{suffix}",),
            "linear_softmax": (f"linear_softmax_{suffix}",),
        }.get(op_name, ())
        roofline_names = {
            "attention": (
                f"pointwise-scale-softmax-{suffix}",
                f"pointwise-scale_softmax-{suffix}",
                f"mla_attention_scale_softmax_{suffix}",
                f"flash_attention_softmax_{suffix}",
                f"flash_attention_initial_load_{suffix}",
                f"flash_attention_hbm_{suffix}",
            ),
            "router": (f"pointwise-router-{suffix}",),
            "gelu": (f"pointwise-gelu-{suffix}", f"pointwise-swiglu-{suffix}"),
            "layernorm1": (f"pointwise-layernorm-{suffix}", f"pointwise-residual-{suffix}"),
            "layernorm2": (f"pointwise-layernorm-{suffix}", f"pointwise-residual-{suffix}"),
            "embedding": (f"embedding_{suffix}",),
            "linear_softmax": (f"pointwise-linear-softmax-{suffix}", f"pointwise-final-norm-{suffix}",
                               "final_norm_grad_accum" if suffix == "b" else f"pointwise-final-norm-{suffix}"),
        }.get(op_name, ())

        accum = self._grad_accum_activity(op_name, direction, gemm_names)
        if timing.activity != ZERO_ACTIVITY:
            return timing.activity if accum == ZERO_ACTIVITY else timing.activity + accum
        activity = ZERO_ACTIVITY
        for name in gemm_names:
            activity = activity + self._gemm_activity(name)
        activity = activity + self._roofline_activity(*roofline_names)
        if activity != ZERO_ACTIVITY:
            return activity if accum == ZERO_ACTIVITY else activity + accum
        if accum != ZERO_ACTIVITY:
            return accum
        if timing.flops or timing.memory_accesses:
            fallback = self._vector_activity(timing.flops, timing.memory_accesses)

            if op_name != "attention" and fallback.l2 == MemoryTraffic() and fallback.hbm != MemoryTraffic():
                fallback = replace(fallback, l2=fallback.hbm)
            return fallback
        return ZERO_ACTIVITY

    def _add_autocast_weight_casts(self, transformer_timings, hidden_dim, intermediate_size, vocab_size, use_moe_layer) -> None:
        if str(getattr(self.model, "run_type", "training")).lower() == "inference":
            return
        p_bytes, c_bytes = float(self.precision.parameters), float(self.precision.tensor_format)
        g_bytes = float(self.precision.gradients)
        if p_bytes == c_bytes:
            return
        lp = self.layer_params_per_rank(hidden_dim, intermediate_size, moe=bool(use_moe_layer))
        if use_moe_layer:
            per_op = {"qkv_proj": lp.get("qkv", 0.0), "output_proj": lp.get("output", 0.0),
                      "ffn1": float(lp.get("experts", 0.0)) + float(lp.get("router", 0.0)), "ffn2": 0.0}
        else:
            per_op = {"qkv_proj": lp.get("qkv", 0.0), "output_proj": lp.get("output", 0.0),
                      "ffn1": lp.get("ffn1", 0.0), "ffn2": lp.get("ffn2", 0.0)}
        per_op["linear_softmax"] = math.ceil(float(vocab_size) * float(hidden_dim) / max(1, int(self.tp)))
        if getattr(self, "disable_embedding_unembedding", False):
            per_op["linear_softmax"] = 0.0

        def cast(op, direction, params):
            timing = getattr(op, direction)
            if timing is None or params <= 0:
                return timing
            read_b, write_b = (params * p_bytes, params * c_bytes) if direction == "forward" else (params * c_bytes, params * g_bytes)
            seconds = self.roofline(0, read_b + write_b, f"autocast-cast-{direction}", mem_level=self.num_levels - 1) + self.O
            self._roofline_activity(f"autocast-cast-{direction}")
            base = self._direction_activity(op.name, direction, timing)

            moved = MemoryTraffic(read_bytes=read_b, write_bytes=write_b)
            act = base + OperationActivity(hbm=moved, l2=moved)
            return replace(timing, compute_time=timing.compute_time + seconds, activity=act)

        for name, params in per_op.items():
            op = transformer_timings.get(name)
            if op is None or not params:
                continue
            transformer_timings[name] = replace(op, forward=cast(op, "forward", float(params)),
                                                backward=cast(op, "backward", float(params)))

    def _grad_accum_activity(self, op_name: str, direction: str, gemm_names: Tuple[str, ...]) -> OperationActivity:
        if direction != "backward":
            return ZERO_ACTIVITY

        names = tuple(f"{g}_grad_accum" for g in gemm_names) + {
            "embedding": ("embedding_grad_accum",),
        }.get(op_name, ())
        return self._roofline_activity(*names) if names else ZERO_ACTIVITY

    def _mla_component_output_elements(
        self,
        gemm: Tuple[int, ...],
        gemm_type: GemmType,
        *,
        visible_cp: bool = False,
    ) -> int:
        spec = self._shard_gemm_descriptor(gemm, gemm_type)
        elements = spec.output_elements()
        if visible_cp and self.cp > 1:
            elements *= int(self.cp)
        return int(elements)

    def _mla_query_state_bytes(self, gemm_shapes: Dict[str, Tuple[int, ...]], precision_bytes: int) -> int:
        if "Q_absorb" in gemm_shapes:
            score_spec = self._shard_gemm_descriptor(
                gemm_shapes["attention_score"],
                GemmType.ATTENTION_SCORE,
            )
            elements = int(score_spec.shard_batch * score_spec.shard_m * score_spec.k)
        elif "U_proj_q" in gemm_shapes:
            elements = self._mla_component_output_elements(
                gemm_shapes["U_proj_q"],
                GemmType.QKV,
            )
        else:
            score_spec = self._shard_gemm_descriptor(
                gemm_shapes["attention_score"],
                GemmType.ATTENTION_SCORE,
            )
            elements = int(score_spec.shard_batch * score_spec.shard_m * score_spec.k)
        return int(math.ceil(float(elements) * float(precision_bytes)))

    def _mla_cache_state_bytes(
        self,
        gemm_shapes: Dict[str, Tuple[int, ...]],
        precision_bytes: int,
        *,
        visible_cp: bool,
    ) -> int:
        if "U_proj_kv" in gemm_shapes:
            score_spec = self._shard_gemm_descriptor(
                gemm_shapes["attention_score"],
                GemmType.ATTENTION_SCORE,
            )
            output_spec = self._shard_gemm_descriptor(
                gemm_shapes["attention_output"],
                GemmType.ATTENTION_OUTPUT,
            )
            key_tokens = score_spec.n if visible_cp else score_spec.shard_n
            value_tokens = output_spec.k if visible_cp else output_spec.shard_k
            elements = int(
                (score_spec.shard_batch * key_tokens * score_spec.k)
                + (output_spec.shard_batch * value_tokens * output_spec.n)
            )
        else:
            elements = self._mla_component_output_elements(
                gemm_shapes["D_proj_kv"],
                GemmType.MLA_DOWN_PROJ,
                visible_cp=visible_cp,
            )
        return int(math.ceil(float(elements) * float(precision_bytes)))

    def _mla_latent_state_bytes(
        self,
        gemm_shapes: Dict[str, Tuple[int, ...]],
        precision_bytes: int,
        *,
        visible_cp: bool,
    ) -> int:
        elements = self._mla_component_output_elements(
            gemm_shapes["D_proj_kv"],
            GemmType.MLA_DOWN_PROJ,
            visible_cp=visible_cp,
        )
        return int(math.ceil(float(elements) * float(precision_bytes)))

    def _mla_attention_ctx_bytes(
        self,
        gemm_shapes: Dict[str, Tuple[int, ...]],
        precision_bytes: int,
    ) -> int:
        elements = self._mla_component_output_elements(
            gemm_shapes["attention_output"],
            GemmType.ATTENTION_OUTPUT,
        )
        return int(math.ceil(float(elements) * float(precision_bytes)))

    def _mla_component_forward_stats(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: GemmType,
    ) -> Tuple[float, float, Dict[str, float]]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if batch <= 0 or m <= 0 or k <= 0 or n <= 0:
            return 0.0, 0.0, {}
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        _, _, _, mem_accesses = self.get_gemm_time(
            shard_spec.shard_m,
            shard_spec.shard_k,
            shard_spec.shard_n,
            name,
            disable_overhead=True,
        )
        local_flops = (
            2.0
            * float(shard_spec.shard_batch)
            * float(shard_spec.shard_m)
            * float(shard_spec.shard_k)
            * float(shard_spec.shard_n)
        )
        global_flops = 2.0 * float(batch) * float(m) * float(k) * float(n)
        return local_flops, global_flops, self._scale_mem_accesses(
            mem_accesses,
            int(shard_spec.shard_batch),
        )

    def _mla_component_backward_stats(
        self,
        gemm: Tuple[int, ...],
        name: str,
        gemm_type: GemmType,
    ) -> Tuple[float, float, Dict[str, float]]:
        batch, m, k, n = self._expand_gemm_descriptor(gemm)
        if batch <= 0 or m <= 0 or k <= 0 or n <= 0:
            return 0.0, 0.0, {}
        shard_spec = self._shard_gemm_descriptor(gemm, gemm_type)
        _, _, _, grad_act_mem = self.get_gemm_time(
            shard_spec.shard_m,
            shard_spec.shard_n,
            shard_spec.shard_k,
            f"{name}_grad_act",
            disable_overhead=True,
            gemm_role="dgrad",
        )
        _, _, _, grad_wt_mem = self.get_gemm_time(
            shard_spec.shard_k,
            shard_spec.shard_m,
            shard_spec.shard_n,
            f"{name}_grad_wt",
            disable_overhead=True,
            gemm_role="wgrad",
        )
        local_flops = (
            4.0
            * float(shard_spec.shard_batch)
            * float(shard_spec.shard_m)
            * float(shard_spec.shard_k)
            * float(shard_spec.shard_n)
        )
        global_flops = 4.0 * float(batch) * float(m) * float(k) * float(n)
        return (
            local_flops,
            global_flops,
            self._combine_mem(
                self._scale_mem_accesses(grad_act_mem, int(shard_spec.shard_batch)),
                self._scale_mem_accesses(grad_wt_mem, int(shard_spec.shard_batch)),
            ),
        )

    def _mla_reduce_kind_for_tp(self) -> Optional[CollectiveType]:
        mode = self.get_parallelism_mode()
        if mode in (ParallelismMode.TENSOR, ParallelismMode.TENSOR_CONTEXT):
            return ALL_REDUCE
        if mode in (ParallelismMode.TENSOR_SEQUENCE, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            return REDUCE_SCATTER
        return None

    def _mla_collective_time(
        self,
        *,
        kind: Optional[CollectiveType],
        size_bytes: int,
        participants: int,
        axis: Optional[str],
        name: str,
    ) -> float:
        size_bytes = int(math.ceil(float(size_bytes or 0.0)))
        participants = int(participants or 0)
        if kind is None or size_bytes <= 0 or participants <= 1:
            return 0.0
        return self.network_model.collective(
            kind=kind,
            size_bytes=size_bytes,
            participants=participants,
            ib=self.links["tp"].bandwidth,
            ll=self.links["tp"].latency,
            local_bytes=0,
            debug_label=name,
            axis=axis,
        )

    def _mla_projection_forward_comm(self, gemm_shapes: Dict[str, Tuple[int, ...]], *, decode: bool) -> Tuple[int, float]:
        mode = self.get_parallelism_mode()
        if mode not in (ParallelismMode.CONTEXT, ParallelismMode.TENSOR_CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            return 0, 0.0
        size_bytes = (
            self._mla_query_state_bytes(gemm_shapes, self.precision.activations)
            if decode
            else self._mla_latent_state_bytes(
                gemm_shapes,
                self.precision.activations,
                visible_cp=True,
            )
        )
        comm_time = self._mla_collective_time(
            kind=ALL_GATHER,
            size_bytes=size_bytes,
            participants=int(self.cp),
            axis="cp",
            name="mla_qkv_proj_cp_f",
        )
        return size_bytes, comm_time

    def _mla_projection_backward_comm(self, gemm_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[int, float]:
        kind = self._mla_reduce_kind_for_tp()
        if kind is None:
            return 0, 0.0
        d_proj_q_spec = self._shard_gemm_descriptor(gemm_shapes["D_proj_q"], GemmType.MLA_DOWN_PROJ)

        size_bytes = int(
            math.ceil(
                float(self.precision.activations)
                * float(d_proj_q_spec.shard_m)
                * float(d_proj_q_spec.shard_batch)
                * float(
                    int(self.q_lora_rank or 0)
                    + int(self.kv_lora_rank)
                    + int(self.qk_rope_head_dim)
                )
            )
        )
        comm_time = self._mla_collective_time(
            kind=kind,
            size_bytes=size_bytes,
            participants=int(self.tp),
            axis="tp",
            name="mla_qkv_proj_tp_b",
        )
        return size_bytes, comm_time

    def _mla_attention_backward_comm(self, gemm_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[int, float]:
        mode = self.get_parallelism_mode()
        if mode not in (ParallelismMode.CONTEXT, ParallelismMode.TENSOR_CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            return 0, 0.0
        size_bytes = self._mla_latent_state_bytes(
            gemm_shapes,
            self.precision.activations,
            visible_cp=True,
        )
        comm_time = self._mla_collective_time(
            kind=REDUCE_SCATTER,
            size_bytes=size_bytes,
            participants=int(self.cp),
            axis="cp",
            name="mla_attention_cp_b",
        )
        return size_bytes, comm_time

    def _mla_output_forward_comm(self, gemm_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[int, float]:
        kind = self._mla_reduce_kind_for_tp()
        if kind is None:
            return 0, 0.0
        spec = self._shard_gemm_descriptor(gemm_shapes["output_proj"], GemmType.OUT_PROJ)
        size_bytes = int(
            math.ceil(
                float(self.precision.activations)
                * float(spec.batch)
                * float(spec.shard_m)
                * float(spec.n)
            )
        )
        comm_time = self._mla_collective_time(
            kind=kind,
            size_bytes=size_bytes,
            participants=int(self.tp),
            axis="tp",
            name="mla_output_proj_tp_f",
        )
        return size_bytes, comm_time

    def _mla_output_backward_comm(self, gemm_shapes: Dict[str, Tuple[int, ...]]) -> Tuple[int, float]:
        mode = self.get_parallelism_mode()
        if mode not in (ParallelismMode.CONTEXT, ParallelismMode.TENSOR_CONTEXT, ParallelismMode.TENSOR_CONTEXT_HYBRID):
            return 0, 0.0
        size_bytes = self._mla_latent_state_bytes(
            gemm_shapes,
            self.precision.activations,
            visible_cp=True,
        )
        comm_time = self._mla_collective_time(
            kind=ALL_GATHER,
            size_bytes=size_bytes,
            participants=int(self.cp),
            axis="cp",
            name="mla_output_proj_cp_b",
        )
        return size_bytes, comm_time

    def _mla_composite_direction(
        self,
        *,
        stage_name: str,
        components: Sequence[Tuple[str, Tuple[int, ...], GemmType]],
        backward: bool,
        grad_accum_elems: int = 0,
        comm_bytes: int = 0,
        comm_time: float = 0.0,
        extra_local_flops: float = 0.0,
        extra_global_flops: float = 0.0,
        memory_override: Optional[Dict[str, float]] = None,
    ) -> DirectionTiming:
        total_global_flops = 0.0
        total_local_flops = 0.0
        total_mem: Dict[str, float] = {}
        for component_name, gemm, gemm_type in components:
            if backward:
                local_flops, global_flops, mem_accesses = self._mla_component_backward_stats(
                    gemm,
                    component_name,
                    gemm_type,
                )
            else:
                local_flops, global_flops, mem_accesses = self._mla_component_forward_stats(
                    gemm,
                    component_name,
                    gemm_type,
                )
            total_local_flops += local_flops
            total_global_flops += global_flops
            total_mem = self._combine_mem(total_mem, mem_accesses)
        total_local_flops += float(extra_local_flops)
        total_global_flops += float(extra_global_flops)
        if memory_override is not None:
            total_mem = dict(memory_override)

        compute_time = 0.0
        if total_local_flops > 0.0:
            compute_time = self.roofline(
                total_local_flops,
                self._mem_levels_to_list(total_mem),
                name=stage_name,
            ) + self.O
        if backward and grad_accum_elems > 0:
            compute_time += self._grad_accum_time(int(grad_accum_elems), f"{stage_name}_grad_accum")

        return DirectionTiming(
            compute_time=compute_time,
            comm_time=float(comm_time or 0.0),
            comm_bytes=int(math.ceil(float(comm_bytes or 0.0))),
            flops=total_global_flops,
            memory_accesses=dict(total_mem),
        )

    def _mla_softmax_direction(
        self,
        gemm: Tuple[int, ...],
        *,
        backward: bool,
        name: str,
    ) -> DirectionTiming:
        batch, m, _, n = self._expand_gemm_descriptor(gemm)
        elements = math.ceil(float(batch) * float(m) * float(n) / float(max(1, self.tp * self.cp)))
        if backward:
            flops = float(elements) * float(SOFTMAX_BACKWARD_FLOPS_PER_ELEMENT + 1)
            mem_bytes = float(self.precision.activations) * float(elements) * float(
                SOFTMAX_BACKWARD_MEM_ACCESSES
            )
        else:
            flops = float(elements) * float(SOFTMAX_FORWARD_FLOPS_PER_ELEMENT + 1)
            mem_bytes = float(self.precision.activations) * float(elements) * float(
                SOFTMAX_FORWARD_MEM_ACCESSES
            )
        compute_time = self.roofline(
            flops,
            mem_bytes,
            name=name,
            mem_level=self.num_levels - 1,
        ) + self.O
        return DirectionTiming(
            compute_time=compute_time,
            comm_time=0.0,
            comm_bytes=0,
            flops=flops,
            memory_accesses=dict(self._mem_levels(mem_bytes)),
        )

    def _mla_param_stage_sizes(self, hidden_dim: int) -> Dict[str, int]:
        return llm_util.mla_attention_param_groups(
            hidden_dim,
            self.num_heads,
            self.q_lora_rank,
            self.kv_lora_rank,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            self.v_head_dim,
        )

    def _build_mla_qkv_projection_timing(
        self,
        gemm_shapes: Dict[str, Tuple[int, ...]],
        *,
        include_backward: bool,
        hidden_dim: int,
        decode: bool = False,
    ) -> OperationTiming:
        components = [
            ("mla_d_proj_q", gemm_shapes["D_proj_q"], GemmType.MLA_DOWN_PROJ),
            ("mla_d_proj_kv", gemm_shapes["D_proj_kv"], GemmType.MLA_DOWN_PROJ),
            ("mla_u_proj_q", gemm_shapes["U_proj_q"], GemmType.QKV),
        ]
        if "U_proj_kv" in gemm_shapes:
            components.append(("mla_u_proj_kv", gemm_shapes["U_proj_kv"], GemmType.QKV))
        if "Q_absorb" in gemm_shapes:
            components.append(("mla_q_absorb", gemm_shapes["Q_absorb"], GemmType.QKV))
        forward_comm_bytes, forward_comm_time = self._mla_projection_forward_comm(
            gemm_shapes,
            decode=decode,
        )
        forward = self._mla_composite_direction(
            stage_name="mla_qkv_proj_f",
            components=tuple(components),
            backward=False,
            comm_bytes=forward_comm_bytes,
            comm_time=forward_comm_time,
        )
        backward_dir = DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0)
        if include_backward:
            backward_comm_bytes, backward_comm_time = self._mla_projection_backward_comm(gemm_shapes)
            backward_dir = self._mla_composite_direction(
                stage_name="mla_qkv_proj_b",
                components=tuple(components),
                backward=True,
                grad_accum_elems=self._mla_param_stage_sizes(hidden_dim)["projection_stage"],
                comm_bytes=backward_comm_bytes,
                comm_time=backward_comm_time,
            )
        return OperationTiming("qkv_proj", forward=forward, backward=backward_dir)

    def _build_mla_attention_timings(
        self,
        gemm_shapes: Dict[str, Tuple[int, ...]],
        *,
        include_backward: bool,
        hidden_dim: int,
        decode: bool = False,
    ) -> Dict[str, OperationTiming]:
        score_components = (
            ("mla_attention_score", gemm_shapes["attention_score"], GemmType.ATTENTION_SCORE),
        )
        output_components = (
            ("mla_attention_output", gemm_shapes["attention_output"], GemmType.ATTENTION_OUTPUT),
        )

        flash_enabled = bool(self.flash_attention) and not decode

        score_forward_mem = None
        score_backward_mem = None
        output_forward_mem = None
        output_backward_mem = None
        flash_recompute_local_flops = 0.0
        flash_recompute_global_flops = 0.0
        flash_softmax_recompute = 0.0
        if flash_enabled:
            score_forward_mem = self._mem_levels(
                self._mla_query_state_bytes(gemm_shapes, self.precision.activations)
                + self._mla_cache_state_bytes(
                    gemm_shapes,
                    self.precision.activations,
                    visible_cp=True,
                )
            )
            score_backward_mem = self._mem_levels(
                self._mla_query_state_bytes(gemm_shapes, self.precision.activations)
                + (2 * self._mla_cache_state_bytes(
                    gemm_shapes,
                    self.precision.activations,
                    visible_cp=True,
                ))
            )
            output_forward_mem = self._mem_levels(
                self._mla_attention_ctx_bytes(gemm_shapes, self.precision.activations)
            )
            output_backward_mem = self._mem_levels(
                2 * self._mla_attention_ctx_bytes(gemm_shapes, self.precision.activations)
            )
            if include_backward and not self.full_recomputation:
                for component_name, gemm, gemm_type in score_components:
                    local_flops, global_flops, _ = self._mla_component_forward_stats(
                        gemm,
                        f"{component_name}_flash_recompute",
                        gemm_type,
                    )
                    flash_recompute_local_flops += local_flops
                    flash_recompute_global_flops += global_flops
                batch, m, _, n = self._expand_gemm_descriptor(gemm_shapes["attention_score"])
                flash_softmax_recompute = float(
                    math.ceil(float(batch) * float(m) * float(n) / float(max(1, self.tp * self.cp)))
                ) * float(SOFTMAX_FORWARD_FLOPS_PER_ELEMENT + 1)

        backward_comm_bytes = 0
        backward_comm_time = 0.0
        if include_backward:
            backward_comm_bytes, backward_comm_time = self._mla_attention_backward_comm(gemm_shapes)

        attention_score_forward = self._mla_composite_direction(
            stage_name="mla_attention_score_f",
            components=score_components,
            backward=False,
            memory_override=score_forward_mem,
        )
        attention_score_backward = DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0)
        if include_backward:
            attention_score_backward = self._mla_composite_direction(
                stage_name="mla_attention_score_b",
                components=score_components,
                backward=True,
                grad_accum_elems=0,
                comm_bytes=backward_comm_bytes,
                comm_time=backward_comm_time,
                extra_local_flops=flash_recompute_local_flops,
                extra_global_flops=flash_recompute_global_flops,
                memory_override=score_backward_mem,
            )

        softmax_forward = self._mla_softmax_direction(
            gemm_shapes["attention_score"],
            backward=False,
            name="mla_attention_scale_softmax_f",
        )
        softmax_backward = DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0)
        if include_backward:
            softmax_backward = self._mla_softmax_direction(
                gemm_shapes["attention_score"],
                backward=True,
                name="mla_attention_scale_softmax_b",
            )
            if flash_enabled and not self.full_recomputation:
                batch, m, _, n = self._expand_gemm_descriptor(gemm_shapes["attention_score"])
                local_elements = math.ceil(
                    float(batch) * float(m) * float(n) / float(max(1, self.tp * self.cp))
                )
                flash_softmax_recompute_time = self.roofline(
                    flash_softmax_recompute,
                    float(self.precision.activations)
                    * float(local_elements)
                    * float(SOFTMAX_FORWARD_MEM_ACCESSES),
                    name="mla_attention_scale_softmax_recompute_b",
                    mem_level=self.num_levels - 1,
                ) + self.O
                softmax_backward = DirectionTiming(
                    compute_time=softmax_backward.compute_time + flash_softmax_recompute_time,
                    comm_time=softmax_backward.comm_time,
                    comm_bytes=softmax_backward.comm_bytes,
                    flops=softmax_backward.flops + flash_softmax_recompute,
                    memory_accesses=dict(
                        self._combine_mem(
                            softmax_backward.memory_accesses,
                            self._mem_levels(
                                float(self.precision.activations)
                                * float(local_elements)
                                * float(SOFTMAX_FORWARD_MEM_ACCESSES)
                            ),
                        )
                    ),
                )

        attention_output_forward = self._mla_composite_direction(
            stage_name="mla_attention_output_f",
            components=output_components,
            backward=False,
            memory_override=output_forward_mem,
        )
        attention_output_backward = DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0)
        if include_backward:
            attention_output_backward = self._mla_composite_direction(
                stage_name="mla_attention_output_b",
                components=output_components,
                backward=True,
                grad_accum_elems=self._mla_param_stage_sizes(hidden_dim)["attention_stage"],
                memory_override=output_backward_mem,
            )

        attention_forward_mem = self._combine_mem(
            attention_score_forward.memory_accesses,
            softmax_forward.memory_accesses,
            attention_output_forward.memory_accesses,
        )
        attention_forward = DirectionTiming(
            compute_time=(
                attention_score_forward.compute_time
                + softmax_forward.compute_time
                + attention_output_forward.compute_time
            ),
            comm_time=attention_score_forward.comm_time,
            comm_bytes=attention_score_forward.comm_bytes,
            flops=(
                attention_score_forward.flops
                + softmax_forward.flops
                + attention_output_forward.flops
            ),
            memory_accesses=dict(self._mem_levels(attention_forward_mem)),
        )
        attention_backward = DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0)
        if include_backward:
            attention_backward_mem = self._combine_mem(
                attention_score_backward.memory_accesses,
                softmax_backward.memory_accesses,
                attention_output_backward.memory_accesses,
            )
            attention_backward = DirectionTiming(
                compute_time=(
                    attention_score_backward.compute_time
                    + softmax_backward.compute_time
                    + attention_output_backward.compute_time
                ),
                comm_time=attention_score_backward.comm_time,
                comm_bytes=attention_score_backward.comm_bytes,
                flops=(
                    attention_score_backward.flops
                    + softmax_backward.flops
                    + attention_output_backward.flops
                ),
                memory_accesses=dict(self._mem_levels(attention_backward_mem)),
            )

        return {
            "attention_score": OperationTiming(
                "attention_score",
                forward=attention_score_forward,
                backward=attention_score_backward,
            ),
            "attention_scale_softmax": OperationTiming(
                "attention_scale_softmax",
                forward=softmax_forward,
                backward=softmax_backward,
            ),
            "attention_output": OperationTiming(
                "attention_output",
                forward=attention_output_forward,
                backward=attention_output_backward,
            ),
            "attention": OperationTiming(
                "attention",
                forward=attention_forward,
                backward=attention_backward,
            ),
        }

    def _build_mla_output_projection_timing(
        self,
        gemm_shapes: Dict[str, Tuple[int, ...]],
        *,
        include_backward: bool,
        hidden_dim: int,
    ) -> OperationTiming:
        components = []
        if "U_proj_v" in gemm_shapes:
            components.append(("mla_u_proj_v", gemm_shapes["U_proj_v"], GemmType.QKV))
        components.append(("mla_output_proj", gemm_shapes["output_proj"], GemmType.OUT_PROJ))
        forward_comm_bytes, forward_comm_time = self._mla_output_forward_comm(gemm_shapes)
        forward = self._mla_composite_direction(
            stage_name="mla_output_proj_f",
            components=tuple(components),
            backward=False,
            comm_bytes=forward_comm_bytes,
            comm_time=forward_comm_time,
        )
        backward_dir = DirectionTiming(compute_time=0.0, comm_time=0.0, comm_bytes=0)
        if include_backward:
            backward_comm_bytes, backward_comm_time = self._mla_output_backward_comm(gemm_shapes)
            backward_dir = self._mla_composite_direction(
                stage_name="mla_output_proj_b",
                components=tuple(components),
                backward=True,
                grad_accum_elems=self._mla_param_stage_sizes(hidden_dim)["output_stage"],
                comm_bytes=backward_comm_bytes,
                comm_time=backward_comm_time,
            )
        return OperationTiming("output_proj", forward=forward, backward=backward_dir)

    _ACTIVITY_BIND_ORDER = (
        "layernorm1", "qkv_proj", "attention", "output_proj", "layernorm2",
        "MLP", "optimizer", "router", "moe_dispatch", "moe_combine",
    )

    def compute_all_gemm_and_node_times(self, *args, **kwargs):
        capture_gemms = (kernel_trace.l2_residency_enabled()
                         and str(self.run_type).lower() == "training"
                         and kernel_trace.implementation(self) == "fused")
        prior_capture = getattr(self, "_capture_operator_gemms", False)
        if capture_gemms:
            self._capture_operator_gemms = True
            self._operator_gemm_shapes = {}
        registries = (self._gemm_activity_by_name, self._roofline_activity_by_name)
        prior = [dict(reg) for reg in registries]
        for reg in registries:
            reg.clear()
        try:
            transformer_timings, node_breakdown = self._compute_template_timings(
                *args, **kwargs
            )
            names = [n for n in self._ACTIVITY_BIND_ORDER if n in transformer_timings]
            names += [n for n in transformer_timings if n not in names]
            for name in names:
                op = transformer_timings[name]
                forward = op.forward
                backward = op.backward
                if forward is not None:
                    forward = replace(
                        forward,
                        activity=self._direction_activity(name, "forward", forward),
                    )
                if backward is not None:
                    backward = replace(
                        backward,
                        activity=self._direction_activity(name, "backward", backward),
                    )
                transformer_timings[name] = replace(op, forward=forward, backward=backward)
            if kernel_trace.l2_residency_enabled():
                b, s = self._template_dims
                if str(self.run_type).lower() == "training":
                    self._apply_l2_simulation(transformer_timings, phase="train", tokens=b * s, batch=b, past=0)
                else:
                    head_rows = b if _prefill_logits_last_token() else b * s
                    self._apply_l2_residency(transformer_timings, tokens=b * s, batch=b, past=0, head_tokens=head_rows)
            else:
                from program import vtp
                if vtp.enabled(self):
                    b, s = self._template_dims
                    vtp.apply(self, transformer_timings, phase="prefill", tokens=b*s, batch=b, past=0)
            from program import vtp
            vtp.update_breakdown(self, transformer_timings, node_breakdown)
        finally:
            if capture_gemms:
                self._capture_operator_gemms = prior_capture
            for reg, before in zip(registries, prior):
                reg.clear()
                reg.update(before)
        return transformer_timings, node_breakdown

    def _compute_template_timings(
        self,
        batch_size,
        vocab_size,
        hidden_dim,
        seq_len,
        num_heads,
        kv_heads,
        intermediate_size,
        num_SMs,
        use_moe_override: Optional[bool] = None,
    ):
        self._template_dims = (int(batch_size), int(seq_len))
        self._eager_kv_past, self._eager_kv_batch = 0, int(batch_size)
        use_moe_layer = self.use_moe if use_moe_override is None else bool(use_moe_override)
        gemm_ctx = self
        if use_moe_layer != self.use_moe:
            from types import SimpleNamespace

            gemm_ctx = SimpleNamespace(
                use_moe=use_moe_layer,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_intermediate_size=intermediate_size,
                model_type=self.model_type,
                head_dim=getattr(self, "head_dim", None),
                attention_type=getattr(self, "attention_type", "mha"),
                q_lora_rank=getattr(self, "q_lora_rank", None),
                kv_lora_rank=getattr(self, "kv_lora_rank", None),
                qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
                qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
                v_head_dim=getattr(self, "v_head_dim", None),
                run_type=str(
                    getattr(
                        self,
                        "run_type",
                        getattr(getattr(self, "model", None), "run_type", "training"),
                    )
                ).lower(),
            )

        gemm_shapes = llm_util.process_gemm_shapes(
            gemm_ctx,
            batch_size,
            seq_len,
            hidden_dim,
            num_heads,
            kv_heads,
            intermediate_size,
            vocab_size,
        )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        gemm_linear = gemm_shapes["linear"]
        gemm_router = gemm_shapes["router"]
        mla_attention = self.attention_type == "mla"
        mla_include_backward = str(
            getattr(
                self,
                "run_type",
                getattr(getattr(self, "model", None), "run_type", "training"),
            )
        ).lower() != "inference"

        transformer_timings: Dict[str, OperationTiming] = {}
        moe_tokens_owner: Optional[int] = None
        moe_tokens_dispatched: Optional[int] = None
        moe_tokens_local: Optional[int] = None
        moe_tokens_shared: Optional[int] = None

        def _extract_forward(ret: Sequence[Any]) -> Tuple[float, float, float, float, Any]:
            if len(ret) == 5:
                time, reduction, size, flops, mem = ret
                return time, reduction, size, flops, mem
            if len(ret) == 3:
                time, reduction, size = ret
                return time, reduction, size, 0.0, []
            raise ValueError(f"Unsupported return length: {len(ret)}")

        mla_ops = (self._mla_vllm_op_timings(phase="prefill", batch=int(batch_size), seq=int(seq_len), backward_zero=True)
                   if mla_attention and self._mla_vllm_trace() else None)

        if mla_ops is not None:
            transformer_timings["qkv_proj"] = mla_ops["qkv_proj"]
        elif mla_attention:
            transformer_timings["qkv_proj"] = self._build_mla_qkv_projection_timing(
                gemm_shapes,
                include_backward=mla_include_backward,
                hidden_dim=hidden_dim,
            )
        else:
            qkv_proj_gemm_f, qkv_proj_reduction_f, qkv_proj_size_f, qkv_proj_flops_f, qkv_proj_mem_f = _extract_forward(
                self._gemm_forward_parts(
                    gemm_qkv_proj,
                    "qkv_projection_f",
                    GemmType.QKV,
                )
            )
            qkv_proj_gemm_b, qkv_proj_reduction_b, qkv_proj_size_b = self._gemm_backward_parts(
                gemm_qkv_proj,
                "qkv_projection_b",
                GemmType.QKV,
            )
            qkv_forward = DirectionTiming(
                compute_time=qkv_proj_gemm_f,
                comm_time=qkv_proj_reduction_f,
                comm_bytes=int(qkv_proj_size_f or 0),
                flops=qkv_proj_flops_f,
                memory_accesses=dict(self._mem_levels(qkv_proj_mem_f)),
                activity=self._gemm_activity("qkv_projection_f"),
            )
            qkv_backward = DirectionTiming(
                compute_time=qkv_proj_gemm_b,
                comm_time=qkv_proj_reduction_b,
                comm_bytes=int(qkv_proj_size_b or 0),
                activity=self._gemm_activity("qkv_projection_b"),
            )
            transformer_timings["qkv_proj"] = OperationTiming(
                "qkv_proj",
                forward=qkv_forward,
                backward=qkv_backward,
            )

        rope_timing = mla_ops["rope"] if mla_ops is not None else self._inference_rope_timing(gemm_qkv_proj)
        if rope_timing is not None:
            transformer_timings["rope"] = rope_timing

        if mla_ops is not None:
            transformer_timings["attention"] = mla_ops["attention"]
            self.resolved_settings.setdefault("attention_paths", {})["forward"] = {
                "attention_path": "vllm_mla_prefill_fa2_d192", "source": "kernel_trace_vllm_mla"}
        elif mla_attention:
            transformer_timings.update(
                self._build_mla_attention_timings(
                    gemm_shapes,
                    include_backward=mla_include_backward,
                    hidden_dim=hidden_dim,
                )
            )
        elif not self.flash_attention:
            attn_score_gemm_f, attn_score_reduction_f, attn_score_size_f, attn_score_flops_f, attn_score_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_attention_score, "attention_score_f", gemm_type=GemmType.ATTENTION_SCORE)
            )
            attn_score_gemm_b, attn_score_reduction_b, attn_score_size_b = self.parallelism_gemm_backward(
                gemm_attention_score, "attention_score_b", gemm_type=GemmType.ATTENTION_SCORE
            )
            attn_score_forward = DirectionTiming(
                compute_time=attn_score_gemm_f,
                comm_time=attn_score_reduction_f,
                comm_bytes=int(attn_score_size_f or 0),
                flops=attn_score_flops_f,
                memory_accesses=dict(self._mem_levels(attn_score_mem_f)),
            )
            attn_score_backward = DirectionTiming(
                compute_time=attn_score_gemm_b,
                comm_time=attn_score_reduction_b,
                comm_bytes=int(attn_score_size_b or 0),
            )
            transformer_timings["attention_score"] = OperationTiming(
                "attention_score",
                forward=attn_score_forward,
                backward=attn_score_backward,
            )

            attention_scale_softmax_f = self.get_scale_softmax_f(gemm=gemm_attention_score)
            attention_scale_softmax_b = self.get_scale_softmax_b(gemm=gemm_attention_score)

            transformer_timings["attention_scale_softmax"] = OperationTiming(
                name="attention_scale_softmax",
                forward=DirectionTiming(
                    compute_time=attention_scale_softmax_f,
                    comm_time=0.0,
                    comm_bytes=0,
                ),
                backward=DirectionTiming(
                    compute_time=attention_scale_softmax_b,
                    comm_time=0.0,
                    comm_bytes=0,
                ),
            )

            attn_out_gemm_f, attn_out_reduction_f, attn_out_size_f, attn_out_flops_f, attn_out_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_attention_output, "attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT)
            )
            attn_out_gemm_b, attn_out_reduction_b, attn_out_size_b = self.parallelism_gemm_backward(
                gemm_attention_output, "attention_output_b", gemm_type=GemmType.ATTENTION_OUTPUT
            )
            attn_out_forward = DirectionTiming(
                compute_time=attn_out_gemm_f,
                comm_time=attn_out_reduction_f,
                comm_bytes=int(attn_out_size_f or 0),
                flops=attn_out_flops_f,
                memory_accesses=dict(self._mem_levels(attn_out_mem_f)),
            )
            attn_out_backward = DirectionTiming(
                compute_time=attn_out_gemm_b,
                comm_time=attn_out_reduction_b,
                comm_bytes=int(attn_out_size_b or 0),
            )
            transformer_timings["attention_output"] = OperationTiming(
                "attention_output",
                forward=attn_out_forward,
                backward=attn_out_backward,
            )

            attention_forward_compute = (
                attn_score_forward.compute_time
                + transformer_timings["attention_scale_softmax"].forward.compute_time
                + attn_out_forward.compute_time
            )
            attention_forward_comm = attn_score_forward.comm_time + attn_out_forward.comm_time
            attention_backward_compute = (
                attn_score_backward.compute_time
                + transformer_timings["attention_scale_softmax"].backward.compute_time
                + attn_out_backward.compute_time
            )
            attention_backward_comm = attn_score_backward.comm_time + attn_out_backward.comm_time
            attention_comm_bytes_f = attn_score_size_f + attn_out_size_f
            attention_comm_bytes_b = attn_score_size_b + attn_out_size_b
            attention_mem = self._combine_mem(attn_score_mem_f, attn_out_mem_f)
            transformer_timings["attention"] = OperationTiming(
                "attention",
                forward=DirectionTiming(
                    compute_time=attention_forward_compute,
                    comm_time=attention_forward_comm,
                    comm_bytes=int(attention_comm_bytes_f or 0),
                    memory_accesses=dict(self._mem_levels(attention_mem)),
                    flops=(attn_score_flops_f + attn_out_flops_f),
                    activity=(
                        self._gemm_activity("attention_score_f")
                        + self._gemm_activity("attention_output_f")
                        + self._roofline_activity(
                            "pointwise-scale-softmax-f",
                            "pointwise-scale_softmax-f",
                        )
                    ),
                ),
                backward=DirectionTiming(
                    compute_time=attention_backward_compute,
                    comm_time=attention_backward_comm,
                    comm_bytes=int(attention_comm_bytes_b or 0),
                    activity=(
                        self._gemm_activity("attention_score_b")
                        + self._gemm_activity("attention_output_b")
                        + self._roofline_activity(
                            "pointwise-scale-softmax-b",
                            "pointwise-scale_softmax-b",
                        )
                    ),
                ),
            )
        else:
            import flash_attention_adapter
            cp_flash_training = int(self.cp) > 1 and str(self.run_type).lower() == "training"
            flash_forward = (flash_attention_adapter.training_all_gather if cp_flash_training
                             else flash_attention_adapter.forward)

            attention_gemm_f, flash_activity, flash_source = flash_forward(
                self, batch_size, hidden_dim, seq_len, num_heads, kv_heads, num_SMs)
            attention_reduction_f, attention_size_f, attention_mem = 0, 0, 0
            flash_notes = {"attention_path": "flash_rebuild_fwd", "source": flash_source,
                           "dtype": "bf16", "dtype_source": "16-bit config; subtype unavailable",
                           **flash_attention_adapter.source_notes(flash_source)}
            self.resolved_settings.setdefault("attention_paths", {})["forward"] = flash_notes
            attention_gemm_b, attention_reduction_b, attention_size_b = 0.0, 0, 0
            flash_backward_activity, flash_backward_notes = ZERO_ACTIVITY, {}
            if str(self.run_type).lower() == "training":
                if cp_flash_training:
                    attention_gemm_b, flash_backward_activity, backward_source = flash_attention_adapter.training_all_gather(
                        self, batch_size, hidden_dim, seq_len, num_heads, kv_heads, num_SMs, op="bwd")

                    attention_reduction_b, attention_size_b = self._flash_attention_backward_communication(
                        batch_size, seq_len, kv_heads, self.head_dim or hidden_dim // num_heads)
                else:
                    attention_gemm_b, flash_backward_activity, backward_source = flash_attention_adapter.backward(
                        self, batch_size, hidden_dim, seq_len, num_heads, kv_heads, num_SMs)
                flash_backward_notes = {"attention_path": "flash_rebuild_bwd", "source": backward_source,
                                        "dtype": "bf16", "dtype_source": "16-bit config; subtype unavailable",
                                        **flash_attention_adapter.source_notes(backward_source)}
                self.resolved_settings["attention_paths"]["backward"] = flash_backward_notes
                if self.full_recomputation:
                    self.resolved_settings["attention_paths"]["recompute"] = dict(flash_notes)
            transformer_timings["attention"] = OperationTiming(
                "attention",
                forward=DirectionTiming(
                    compute_time=attention_gemm_f,
                    comm_time=attention_reduction_f,
                    comm_bytes=int(attention_size_f or 0),
                    memory_accesses=dict(self._mem_levels(attention_mem)),
                    activity=flash_activity, notes=flash_notes,
                ),
                backward=DirectionTiming(
                    compute_time=attention_gemm_b,
                    comm_time=attention_reduction_b,
                    comm_bytes=int(attention_size_b or 0),
                    activity=flash_backward_activity, notes=flash_backward_notes,
                ),
            )

        if "attention" in transformer_timings:
            paths = self.resolved_settings.setdefault("attention_paths", {})
            paths.setdefault("forward", {"attention_path": "legacy_mla" if mla_attention else "legacy_nonflash"})
            if str(self.run_type).lower() == "training":
                paths.setdefault("backward", {"attention_path": "legacy_mla" if mla_attention else "legacy_nonflash"})
            transformer_timings["attention"], _ = self._with_kv_cache_write(
                transformer_timings["attention"], int(batch_size) * int(seq_len)
            )

        if mla_ops is not None:
            transformer_timings["output_proj"] = mla_ops["output_proj"]
        elif mla_attention:
            transformer_timings["output_proj"] = self._build_mla_output_projection_timing(
                gemm_shapes,
                include_backward=mla_include_backward,
                hidden_dim=hidden_dim,
            )
        else:
            out_proj_gemm_f, out_proj_reduction_f, out_proj_size_f, out_proj_flops_f, out_proj_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_output_proj, "output_projection_f", gemm_type=GemmType.OUT_PROJ)
            )
            out_proj_gemm_b, out_proj_reduction_b, out_proj_size_b = self.parallelism_gemm_backward(
                gemm_output_proj, "output_projection_b", gemm_type=GemmType.OUT_PROJ
            )
            transformer_timings["output_proj"] = OperationTiming(
                "output_proj",
                forward=DirectionTiming(
                    compute_time=out_proj_gemm_f,
                    comm_time=out_proj_reduction_f,
                    comm_bytes=int(out_proj_size_f or 0),
                    flops=out_proj_flops_f,
                    memory_accesses=dict(self._mem_levels(out_proj_mem_f)),
                    activity=self._gemm_activity("output_projection_f"),
                ),
                backward=DirectionTiming(
                    compute_time=out_proj_gemm_b,
                    comm_time=out_proj_reduction_b,
                    comm_bytes=int(out_proj_size_b or 0),
                    activity=self._gemm_activity("output_projection_b"),
                ),
            )
        if not use_moe_layer:
            ffn1_gemm_f, ffn1_reduction_f, ffn1_size_f, ffn1_flops_f, ffn1_mem_f = _extract_forward(
                self._gemm_forward_parts(gemm_ffn1, "ffn_f", GemmType.FFN1)
            )
            ffn1_gemm_b, ffn1_reduction_b, ffn1_size_b = self._gemm_backward_parts(
                gemm_ffn1, "ffn_b", GemmType.FFN1
            )
            ffn2_gemm_f, ffn2_reduction_f, ffn2_size_f, ffn2_flops_f, ffn2_mem_f = _extract_forward(
                self.parallelism_gemm_forward(gemm_ffn2, "ffn2_f", gemm_type=GemmType.FFN2)
            )
            ffn2_gemm_b, ffn2_reduction_b, ffn2_size_b = self.parallelism_gemm_backward(
                gemm_ffn2, "ffn2_b", gemm_type=GemmType.FFN2
            )

        else:
            allow_padding = self._moe_allow_padding()
            (
                moe_tokens_owner,
                moe_tokens_dispatched_balanced,
                moe_tokens_local_balanced,
                experts_per_rank,
                _tokens_per_expert_balanced,
            ) = self._moe_balanced_routed_tokens_per_expert(batch_size, seq_len, allow_padding=allow_padding)
            moe_tokens_local = self._moe_apply_expert_imbalance(
                moe_tokens_local_balanced,
                experts_per_rank,
            )
            moe_tokens_shared = self._moe_tokens_shared(moe_tokens_owner)
            router_gemm_time_f , router_reduction_f, router_size_f, _, _ = _extract_forward(
                self.get_router_f(gemm_router, gemm_ffn1, batch_size=batch_size, seq_len=seq_len)
            )
            router_gemm_time_b , router_reduction_b, router_size_b= self.get_router_b(
                gemm_router, gemm_ffn1, batch_size=batch_size, seq_len=seq_len
            )
            moe_group = self._moe_routing_group()
            axis = "ep" if moe_group == self.ep else None
            dispatch_fwd_bytes, _disp_t, combine_fwd_bytes, _comb_t = self._moe_forward_comm(
                moe_tokens_owner // self._moe_dp_attention_groups(), moe_tokens_dispatched_balanced, experts_per_rank, "prefill"
            )
            dispatch_bwd_bytes = int(
                math.ceil(self.precision.activations * moe_tokens_dispatched_balanced * hidden_dim)
            )
            dispatch_fwd_time = _disp_t if self._moe_tp_replicated_combine() else self._moe_comm_time(
                self._moe_comm_decomposition(dispatch_fwd_bytes, experts_per_rank),
                debug_label="moe_dispatch_f",
                axis=axis,
            )
            dispatch_bwd_time = self._moe_comm_time(
                self._moe_comm_decomposition(dispatch_bwd_bytes, experts_per_rank),
                debug_label="moe_dispatch_b",
                axis=axis,
            )
            ffn1_gemm_f, ffn1_reduction_f, ffn1_size_f, ffn1_flops_f, ffn1_mem_f = _extract_forward(
                self.get_moe_ffn_f(
                    gemm_ffn1,
                    "ffn1_f",
                    gemm_type=GemmType.FFN1,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    allow_padding=allow_padding,
                )
            )
            ffn1_gemm_b, ffn1_reduction_b, ffn1_size_b = self.get_moe_ffn_b(
                gemm_ffn1,
                "ffn1_b",
                gemm_type=GemmType.FFN1,
                batch_size=batch_size,
                seq_len=seq_len,
                allow_padding=allow_padding,
            )
            combine_fwd_time = _comb_t if self._moe_tp_replicated_combine() else self._moe_comm_time(
                self._moe_comm_decomposition(dispatch_fwd_bytes, experts_per_rank),
                debug_label="moe_combine_f",
                axis=axis,
            )
            combine_bwd_time = self._moe_comm_time(
                self._moe_comm_decomposition(dispatch_bwd_bytes, experts_per_rank),
                debug_label="moe_combine_b",
                axis=axis,
            )
            ffn2_gemm_f, ffn2_reduction_f, ffn2_size_f, ffn2_flops_f, ffn2_mem_f = _extract_forward(
                self.get_moe_ffn_f(
                    gemm_ffn2,
                    "ffn2_f",
                    gemm_type=GemmType.FFN2,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    allow_padding=allow_padding,
                )
            )
            ffn2_gemm_b, ffn2_reduction_b, ffn2_size_b = self.get_moe_ffn_b(
                gemm_ffn2,
                "ffn2_b",
                gemm_type=GemmType.FFN2,
                batch_size=batch_size,
                seq_len=seq_len,
                allow_padding=allow_padding,
            )
        if use_moe_layer:
            transformer_timings["router"] = OperationTiming(
                "router",
                forward=DirectionTiming(
                    compute_time=router_gemm_time_f,
                    comm_time=router_reduction_f,
                    comm_bytes=int(router_size_f or 0),
                ),
                backward=DirectionTiming(
                    compute_time=router_gemm_time_b,
                    comm_time=router_reduction_b,
                    comm_bytes=int(router_size_b or 0),
                ),
            )
            transformer_timings["moe_dispatch"] = OperationTiming(
                "moe_dispatch",
                forward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=dispatch_fwd_time,
                    comm_bytes=int(dispatch_fwd_bytes or 0),
                ),
                backward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=dispatch_bwd_time,
                    comm_bytes=int(dispatch_bwd_bytes or 0),
                ),
            )
            transformer_timings["moe_combine"] = OperationTiming(
                "moe_combine",
                forward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=combine_fwd_time,
                    comm_bytes=int(combine_fwd_bytes or 0),
                ),
                backward=DirectionTiming(
                    compute_time=0.0,
                    comm_time=combine_bwd_time,
                    comm_bytes=int(dispatch_bwd_bytes or 0),
                ),
            )

        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=DirectionTiming(
                compute_time=ffn1_gemm_f,
                comm_time=ffn1_reduction_f,
                comm_bytes=int(ffn1_size_f or 0),
                flops=ffn1_flops_f,
                memory_accesses=dict(self._mem_levels(ffn1_mem_f)),
                activity=(
                    self._gemm_activity("ffn1_f")
                    + self._gemm_activity("ffn_f")
                ),
            ),
            backward=DirectionTiming(
                compute_time=ffn1_gemm_b,
                comm_time=ffn1_reduction_b,
                comm_bytes=int(ffn1_size_b or 0),
                activity=(
                    self._gemm_activity("ffn1_b")
                    + self._gemm_activity("ffn_b")
                ),
            ),
        )

        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=DirectionTiming(
                compute_time=ffn2_gemm_f,
                comm_time=ffn2_reduction_f,
                comm_bytes=int(ffn2_size_f or 0),
                flops=ffn2_flops_f,
                memory_accesses=dict(self._mem_levels(ffn2_mem_f)),
                activity=self._gemm_activity("ffn2_f"),
            ),
            backward=DirectionTiming(
                compute_time=ffn2_gemm_b,
                comm_time=ffn2_reduction_b,
                comm_bytes=int(ffn2_size_b or 0),
                activity=self._gemm_activity("ffn2_b"),
            ),
        )

        with self._repeated_activity(float(self._moe_dp_attention_groups()), "embedding_f"):
            embedding_f, embedding_mem = self.get_embedding_f(
                vocab_size=vocab_size,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
            )
        embedding_b = self.get_embedding_b(
            vocab_size=vocab_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
        )
        if self.disable_embedding_unembedding:
            embedding_f = 0.0
            embedding_b = 0.0
            embedding_mem = {}
        transformer_timings["embedding"] = OperationTiming(
            "embedding",
            forward=DirectionTiming(
                compute_time=embedding_f,
                comm_time=0.0,
                comm_bytes=0,
                memory_accesses=dict(self._mem_levels(embedding_mem)),
                activity=self._roofline_activity("embedding_f", "vit_embedding_f")
                + self._gemm_activity("vit_patch_embed_f"),
            ),
            backward=DirectionTiming(
                compute_time=embedding_b,
                comm_time=0.0,
                comm_bytes=0,
                activity=self._roofline_activity("embedding_b", "vit_embedding_b")
                + self._gemm_activity("vit_patch_embed_b"),
            ),
        )

        residual1_f = 0.0 if self._vllm_norms() else self.get_residual_f(tensor_shape=gemm_output_proj)
        residual1_b = self.get_residual_b(tensor_shape=gemm_output_proj)
        if self._vllm_norms():
            layernorm1_f = self._vllm_norm_f(batch=batch_size, seq_len=seq_len, role="layernorm1")
            layernorm1_reduction_f, LN1_comm_bytes_f = 0.0, 0
        else:
            layernorm1_f, layernorm1_reduction_f, LN1_comm_bytes_f = self.get_layernorm_f(
                batch=batch_size,
                seq_len=seq_len,
                d_model=hidden_dim,
            )
        layernorm1_b, layernorm1_reduction_b, LN1_comm_bytes_b = self.get_layernorm_b(
            batch=batch_size,
            seq_len=seq_len,
            d_model=hidden_dim,
            type=GemmType.LAYER_NORM_1,
        )
        transformer_timings["layernorm1"] = OperationTiming(
            "layernorm1",
            forward=DirectionTiming(
                compute_time=layernorm1_f + residual1_f,
                comm_time=layernorm1_reduction_f,
                comm_bytes=int(LN1_comm_bytes_f or 0),
                activity=self._roofline_activity(
                    "pointwise-layernorm-f", "pointwise-residual-f"
                ),
            ),
            backward=DirectionTiming(
                compute_time=layernorm1_b + residual1_b,
                comm_time=layernorm1_reduction_b,
                comm_bytes=int(LN1_comm_bytes_b or 0),

                activity=self._roofline_activity(
                    "pointwise-layernorm-b", "pointwise-residual-b", "layernorm_grad_accum"
                ),
            ),
        )

        residual2_f = 0.0 if self._vllm_norms() else self.get_residual_f(tensor_shape=gemm_ffn2)
        residual2_b = self.get_residual_b(tensor_shape=gemm_ffn2)
        if self._vllm_norms():
            layernorm2_f = self._vllm_norm_f(batch=batch_size, seq_len=seq_len, role="layernorm2")
            layernorm2_reduction_f, LN2_comm_bytes_f = 0.0, 0
        else:
            layernorm2_f, layernorm2_reduction_f, LN2_comm_bytes_f = self.get_layernorm_f(
                batch=batch_size,
                seq_len=seq_len,
                d_model=hidden_dim,
            )
        layernorm2_b, layernorm2_reduction_b, LN2_comm_bytes_b = self.get_layernorm_b(
            batch=batch_size,
            seq_len=seq_len,
            d_model=hidden_dim,
        )
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=DirectionTiming(
                compute_time=layernorm2_f + residual2_f,
                comm_time=layernorm2_reduction_f,
                comm_bytes=int(LN2_comm_bytes_f or 0),
                activity=self._roofline_activity(
                    "pointwise-layernorm-f", "pointwise-residual-f"
                ),
            ),
            backward=DirectionTiming(
                compute_time=layernorm2_b + residual2_b,
                comm_time=layernorm2_reduction_b,
                comm_bytes=int(LN2_comm_bytes_b or 0),

                activity=self._roofline_activity(
                    "pointwise-layernorm-b", "pointwise-residual-b", "layernorm_grad_accum"
                ),
            ),
        )

        ffn1_spec = self._shard_gemm_descriptor(gemm_ffn1, GemmType.FFN1)
        ffn1_activation_shape = (ffn1_spec.shard_m, ffn1_spec.k, ffn1_spec.shard_n)
        ffn1_activation_shape_shared = None
        if use_moe_layer:
            tokens_owner = moe_tokens_owner if moe_tokens_owner is not None else self._moe_tokens_owner(batch_size, seq_len)
            tokens_local = moe_tokens_local if moe_tokens_local is not None else self._moe_tokens_local(
                self._moe_tokens_dispatched(tokens_owner)
            )
            tokens_shared = moe_tokens_shared if moe_tokens_shared is not None else self._moe_tokens_shared(tokens_owner)
            use_tp_sharded = bool(getattr(self, "tp_ep", True))
            ffn1_n = ffn1_spec.shard_n if use_tp_sharded else ffn1_spec.n
            ffn1_activation_shape = (tokens_local, ffn1_spec.k, ffn1_n)
            if tokens_shared > 0:
                ffn1_activation_shape_shared = (tokens_shared, ffn1_spec.k, ffn1_n)

        if self._uses_gated_mlp():
            act_f = self.get_swiglu_f(tensor_shape=ffn1_activation_shape)
            act_b = self.get_swiglu_b(tensor_shape=ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_swiglu_f(tensor_shape=ffn1_activation_shape_shared)
                act_b += self.get_swiglu_b(tensor_shape=ffn1_activation_shape_shared)
        else:
            act_f = self.get_gelu_f(tensor_shape=ffn1_activation_shape)
            act_b = self.get_gelu_b(tensor_shape=ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_gelu_f(tensor_shape=ffn1_activation_shape_shared)
                act_b += self.get_gelu_b(tensor_shape=ffn1_activation_shape_shared)
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=DirectionTiming(
                compute_time=act_f,
                comm_time=0.0,
                comm_bytes=0,
                activity=self._roofline_activity(
                    "pointwise-gelu-f", "pointwise-swiglu-f"
                ),
            ),
            backward=DirectionTiming(
                compute_time=act_b,
                comm_time=0.0,
                comm_bytes=0,
                activity=self._roofline_activity(
                    "pointwise-gelu-b", "pointwise-swiglu-b"
                ),
            ),
        )

        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(gemm=gemm_linear)

        linear_softmax_b = (
            self.get_linear_softmax_b(gemm=gemm_linear) if mla_include_backward else 0.0
        )
        if self.disable_embedding_unembedding:
            linear_softmax_f = 0.0
            linear_softmax_b = 0.0
            linear_softmax_mem = {}
        transformer_timings["linear_softmax"] = OperationTiming(
            "linear_softmax",
            forward=DirectionTiming(
                compute_time=linear_softmax_f,
                comm_time=0.0,
                comm_bytes=0,
                memory_accesses=dict(self._mem_levels(linear_softmax_mem)),
                activity=(
                    self._gemm_activity("linear_softmax_f")
                    + self._roofline_activity("pointwise-linear-softmax-f", "pointwise-final-norm-f")
                ),
            ),
            backward=DirectionTiming(
                compute_time=linear_softmax_b,
                comm_time=0.0,
                comm_bytes=0,
                activity=(
                    self._gemm_activity("linear_softmax_b")
                    + self._roofline_activity("pointwise-linear-softmax-b", "pointwise-final-norm-b", "final_norm_grad_accum")
                ),
            ),
        )

        if self._eager_training():
            self._apply_eager_training(transformer_timings, batch=batch_size, seq=seq_len)
        else:
            self._add_autocast_weight_casts(transformer_timings, hidden_dim, intermediate_size, vocab_size, use_moe_layer)

        mlp_group = OperationGroup(
            "MLP",
            operations=(
                transformer_timings["ffn1"],
                transformer_timings["gelu"],
                transformer_timings["ffn2"],
            ),
        )

        transformer_time_f = (
            transformer_timings["qkv_proj"].total_forward_time()
            + (transformer_timings["rope"].total_forward_time() if "rope" in transformer_timings else 0.0)
            + transformer_timings["attention"].total_forward_time()
            + transformer_timings["output_proj"].total_forward_time()
            + mlp_group.forward_total_time()
            + transformer_timings["layernorm1"].total_forward_time()
            + transformer_timings["layernorm2"].total_forward_time()
        )
        transformer_time_b = (
            transformer_timings["qkv_proj"].total_backward_time()
            + transformer_timings["attention"].total_backward_time()
            + transformer_timings["output_proj"].total_backward_time()
            + mlp_group.backward_total_time()
            + transformer_timings["layernorm1"].total_backward_time()
            + transformer_timings["layernorm2"].total_backward_time()
        )
        if use_moe_layer:
            for op_name in ("router", "moe_dispatch", "moe_combine"):
                if op_name in transformer_timings:
                    transformer_time_f += transformer_timings[op_name].total_forward_time()
                    transformer_time_b += transformer_timings[op_name].total_backward_time()

        transformer_time_b_combined = transformer_time_b

        node_breakdown = {
            "transformer_time_f": transformer_time_f,
            "transformer_time_b": transformer_time_b,
            "transformer_time_b_combined": transformer_time_b_combined,
            "embedding_f": transformer_timings["embedding"].total_forward_time(),
            "embedding_b": transformer_timings["embedding"].total_backward_time(),
            "linear_softmax_f": transformer_timings["linear_softmax"].total_forward_time(),
            "linear_softmax_b": transformer_timings["linear_softmax"].total_backward_time(),
        }
        if self._generate_graphs:
            results_path = os.path.join(self.output_dir, "transformer_timings.yaml")
            with open(results_path, "w", encoding="utf-8") as results_file:
                yaml.dump(
                    {
                        "transformer_results": {
                            name: timing.to_dict() for name, timing in transformer_timings.items()
                        },
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    sort_keys=True,
                )

        return transformer_timings, node_breakdown

    def _effective_transformer_batch(self) -> int:
        run_type = str(getattr(getattr(self, "model", None), "run_type", "training")).lower()
        if run_type == "inference":
            if self.pp > 1:
                return self.micro_batch
            return self.batch_size
        if self.pp > 1:
            return self.micro_batch
        dp_dense = self.dp * self.ep if self.use_moe else self.dp
        if dp_dense > 1:
            return self.mini_batch
        return self.batch_size

    def _build_comm_metadata(
        self,
        reduction_sizes_dense: float,
        reduction_sizes_moe: float,
        local_comp_dense: float,
        local_comp_moe: float,
        embedding_size: int,
        softmax_size: int,
        cross_layer_bytes: int,
        transformer_dense_ep_sync_bytes: float = 0.0,
        transformer_moe_ep_sync_bytes: float = 0.0,
        zero2_embedding_gather_bytes: float = 0.0,
        zero2_transformer_gather_bytes: float = 0.0,
        zero2_softmax_gather_bytes: float = 0.0,
        zero3_embedding_gather_bytes: float = 0.0,
        zero3_transformer_gather_bytes: float = 0.0,
        zero3_softmax_gather_bytes: float = 0.0,
    ) -> Dict[str, Dict[str, Any]]:
        from program import megatron as _megatron
        grad_collective = REDUCE_SCATTER if ((self.zero_stage >= 2 and self.dp > 1)
                                             or _megatron.distributed_optimizer(self)) else ALL_REDUCE
        def _dp_comm_flag(required_every_cycle: bool) -> Dict[str, bool]:
            return {"ga_required_every_cycle": bool(required_every_cycle)}

        grad_cp = max(1, int(self.cp))
        grad_axis = "dp" if grad_cp == 1 else "dp*cp"
        grad_participants = int(self.dp) * grad_cp

        metadata = {
            'transformer_dense': {
                'size': reduction_sizes_dense,
                'type': grad_collective,
                'participants': grad_participants,
                'interconnect_type': grad_axis,
                'local_comp_time': local_comp_dense,
                **_dp_comm_flag(False),
            },
            'transformer_moe': {
                'size': reduction_sizes_moe,
                'type': grad_collective,
                'participants': grad_participants,
                'interconnect_type': grad_axis,
                'local_comp_time': local_comp_moe,
                **_dp_comm_flag(False),
            },
            'embedding': {
                'size': embedding_size,
                'type': grad_collective,
                'participants': grad_participants,
                'interconnect_type': grad_axis,
                'local_comp_time': 0,
                **_dp_comm_flag(False),
            },
            'softmax': {
                'size': softmax_size,
                'type': grad_collective,
                'participants': grad_participants,
                'interconnect_type': grad_axis,
                'local_comp_time': 0,
                **_dp_comm_flag(False),
            },
            'cross_layer': {
                'size': cross_layer_bytes,
                'type': PIPELINE,
                'participants': 2,
                'interconnect_type': 'pp',
                'local_comp_time': 0
            }
        }
        if transformer_dense_ep_sync_bytes and self.ep > 1:
            metadata['transformer_dense_ep_sync'] = {
                'size': int(math.ceil(transformer_dense_ep_sync_bytes)),
                'type': grad_collective,
                'participants': self.ep,
                'interconnect_type': 'ep',
                'local_comp_time': 0,
            }
        if transformer_moe_ep_sync_bytes and self.ep > 1:
            metadata['transformer_moe_ep_sync'] = {
                'size': int(math.ceil(transformer_moe_ep_sync_bytes)),
                'type': grad_collective,
                'participants': self.ep,
                'interconnect_type': 'ep',
                'local_comp_time': 0,
            }
        if zero2_embedding_gather_bytes:
            metadata['zero2_embedding_gather'] = {
                'size': int(math.ceil(zero2_embedding_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                **_dp_comm_flag(False),
            }
        if zero2_transformer_gather_bytes:
            metadata['zero2_transformer_gather'] = {
                'size': int(math.ceil(zero2_transformer_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                **_dp_comm_flag(False),
            }
        if zero2_softmax_gather_bytes:
            metadata['zero2_softmax_gather'] = {
                'size': int(math.ceil(zero2_softmax_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                **_dp_comm_flag(False),
            }
        if zero3_embedding_gather_bytes:
            metadata['zero3_embedding_gather'] = {
                'size': int(math.ceil(zero3_embedding_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                **_dp_comm_flag(True),
            }
        if zero3_transformer_gather_bytes:
            metadata['zero3_transformer_gather'] = {
                'size': int(math.ceil(zero3_transformer_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                'tp_shard': True,
                **_dp_comm_flag(True),
            }
        if zero3_softmax_gather_bytes:
            metadata['zero3_softmax_gather'] = {
                'size': int(math.ceil(zero3_softmax_gather_bytes)),
                'type': ALL_GATHER,
                'participants': self.dp,
                'interconnect_type': 'dp',
                'local_comp_time': 0,
                **_dp_comm_flag(True),
            }

        if self.use_moe and self.zero_stage >= 2 and str(getattr(self.model, "run_type", "training")).lower() != "inference":
            dense_layer = self.zero_moe_layer_param_groups(self.hidden_dim, self.intermediate_size)
            moe_layer = self.zero_moe_layer_param_groups(self.hidden_dim, self.moe_intermediate_size, moe=True)
            emb, head = self._embedding_softmax_params_per_rank(self.hidden_dim, self.vocab_size)
            groups = {"dense_layer": dense_layer, "moe_layer": moe_layer,
                      "embedding": (emb, 0.0), "softmax": (head, 0.0)}
            for label, counts in groups.items():
                for role, count in zip(("dense", "expert"), counts):
                    if count <= 0:
                        continue
                    shards = max(1, int(self.dp) * (int(self.ep) if role == "dense" else 1))
                    axis = "dp*ep" if role == "dense" and self.ep > 1 else "dp"
                    prefix = f"zm_{label}_{role}"
                    def add(suffix, kind, size, participants, network_axis, each_cycle=False):
                        metadata[prefix + suffix] = {"size": int(math.ceil(size)), "type": kind,
                            "participants": participants, "interconnect_type": network_axis,
                            "local_comp_time": 0, "ga_required_every_cycle": each_cycle}
                    if self.cp > 1:
                        add("_a_cp", ALL_REDUCE, count * self.precision.grad_communication, self.cp, "cp", True)
                    if shards > 1:
                        add("_b_rs", REDUCE_SCATTER, count * self.precision.grad_communication, shards, axis, True)
                        if self.zero_stage == 2:
                            add("_c_update_ag", ALL_GATHER, count * self.precision.parameters / shards, shards, axis)
                        else:
                            add("_gather", ALL_GATHER, count * self.precision.parameters / shards, shards, axis, True)

        if self.use_moe and _megatron.dp_grad_sync(self) != "legacy":
            dense_group, expert_group = _megatron.data_parallel_groups(self)
            dense_layer = self.zero_moe_layer_param_groups(self.hidden_dim, self.intermediate_size)
            moe_layer = self.zero_moe_layer_param_groups(self.hidden_dim, self.moe_intermediate_size, moe=True)
            emb, head = self._embedding_softmax_params_per_rank(self.hidden_dim, self.vocab_size)
            groups = {"dense_layer": dense_layer, "moe_layer": moe_layer, "embedding": (emb, 0.0), "softmax": (head, 0.0)}
            for label, counts in groups.items():
                for role, count in zip(("dense", "expert"), counts):
                    shards = dense_group if role == "dense" else expert_group
                    if count <= 0 or shards <= 1:
                        continue
                    axis = "dp*ep" if role == "dense" else "dp"
                    key = f"mdp_{label}_{role}"
                    metadata[key] = {"size": int(math.ceil(count * self.precision.grad_communication)),
                                     "type": grad_collective, "participants": shards, "interconnect_type": axis,
                                     "local_comp_time": 0, **_dp_comm_flag(False)}
                    if _megatron.distributed_optimizer(self):
                        metadata[key + "_gather"] = {"size": int(math.ceil(count * self.precision.parameters / shards)),
                                                     "type": ALL_GATHER, "participants": shards,
                                                     "interconnect_type": axis, "local_comp_time": 0,
                                                     **_dp_comm_flag(False)}
        return metadata

    def _build_interconnect_params(self) -> Dict[str, Tuple[float, float]]:
        return {
            'dp': (self.links["dp"].bandwidth, self.links["dp"].latency),
            'ep': (self.links["ep"].bandwidth, self.links["ep"].latency),
            'pp': (self.links["pp"].bandwidth, self.links["pp"].latency),
            'tp': (self.links["tp"].bandwidth, self.links["tp"].latency),
            'cp': (self.links["cp"].bandwidth, self.links["cp"].latency),
        }

    def _prepare_execution_graphs(
        self,
        *,
        node_breakdown: Dict[str, float],
        transformer_timings: Dict[str, OperationTiming],
        moe_node_breakdown: Optional[Dict[str, float]] = None,
        moe_transformer_timings: Optional[Dict[str, OperationTiming]] = None,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_size: int,
        vocab_size: int,
        include_pipeline_backward: bool,
        include_transformer_backward: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        zero2_embedding_gather_bytes: float = 0.0,
        zero2_transformer_gather_bytes: float = 0.0,
        zero2_softmax_gather_bytes: float = 0.0,
        zero3_embedding_gather_bytes: float = 0.0,
        zero3_transformer_gather_bytes: float = 0.0,
        zero3_softmax_gather_bytes: float = 0.0,
    ) -> Tuple[WorkloadSpec, Optional[WorkloadSpec]]:
        need_no_dp_variant = getattr(self, "gradient_accumulation_steps", 1) > 1

        if not include_pipeline_backward and not include_transformer_backward:
            reduction_sizes_dense = 0.0
            reduction_sizes_moe = 0.0
            local_comp_dense = 0.0
            local_comp_moe = 0.0
        else:
            reduction_sizes_dense = self.get_data_parallel_reduction_sizes(hidden_dim, intermediate_size, moe=False)
            local_comp_dense = self.get_data_parallel_reduction_llm(hidden_dim, intermediate_size, moe=False)

            self._roofline_activity("apply_grad", "pointwise-grad-clipping")
            reduction_sizes_moe = reduction_sizes_dense
            local_comp_moe = local_comp_dense
            if self.use_moe:
                reduction_sizes_moe = self.get_data_parallel_reduction_sizes(
                    hidden_dim,
                    self.moe_intermediate_size,
                    moe=True,
                )
                local_comp_moe = self.get_data_parallel_reduction_llm(
                    hidden_dim,
                    self.moe_intermediate_size,
                    moe=True,
                )

                self._roofline_activity("apply_grad", "pointwise-grad-clipping")

        embedding_params, output_params = self._embedding_softmax_params_per_rank(
            hidden_dim, vocab_size
        )
        embedding_size = math.ceil(self.precision.grad_communication * embedding_params)
        softmax_size = math.ceil(self.precision.grad_communication * output_params)
        cross_layer_bytes = self.get_inter_layer_comm_latency_llm(batch_size, hidden_dim, seq_len)[1]

        ep_dense_sync_bytes_dense = 0
        ep_dense_sync_bytes_moe = 0
        from program import megatron as _megatron_ep

        if (include_transformer_backward and self.use_moe and self.ep > 1
                and _megatron_ep.dp_grad_sync(self) == "legacy"):
            grad_comm = self.precision.grad_communication
            dense = self.layer_params_per_rank(hidden_dim, intermediate_size, moe=False)
            moe_params = self.layer_params_per_rank(hidden_dim, self.moe_intermediate_size, moe=True)
            qkv_size = math.ceil(grad_comm * dense["qkv"])
            output_size = math.ceil(grad_comm * dense["output"])
            ep_dense_sync_bytes_dense = int(
                qkv_size
                + output_size
                + math.ceil(grad_comm * dense["ffn1"])
                + math.ceil(grad_comm * dense["ffn2"])
            )
            router_size = math.ceil(grad_comm * moe_params["router"])
            shared_size = 0
            if self.n_shared_experts > 0:
                moe_group = max(1, int(self._moe_routing_group()))
                experts_per_rank = (self.moe_num_experts / moe_group) + self.n_shared_experts
                shared_fraction = float(self.n_shared_experts) / float(experts_per_rank)
                shared_size = math.ceil(grad_comm * moe_params["experts"] * shared_fraction)
            ep_dense_sync_bytes_moe = int(qkv_size + output_size + router_size + shared_size)

        comm_metadata = self._build_comm_metadata(
            reduction_sizes_dense=reduction_sizes_dense,
            reduction_sizes_moe=reduction_sizes_moe,
            local_comp_dense=local_comp_dense,
            local_comp_moe=local_comp_moe,
            embedding_size=embedding_size,
            softmax_size=softmax_size,
            cross_layer_bytes=cross_layer_bytes,
            transformer_dense_ep_sync_bytes=ep_dense_sync_bytes_dense,
            transformer_moe_ep_sync_bytes=ep_dense_sync_bytes_moe,
            zero2_embedding_gather_bytes=zero2_embedding_gather_bytes,
            zero2_transformer_gather_bytes=zero2_transformer_gather_bytes,
            zero2_softmax_gather_bytes=zero2_softmax_gather_bytes,
            zero3_embedding_gather_bytes=zero3_embedding_gather_bytes,
            zero3_transformer_gather_bytes=zero3_transformer_gather_bytes,
            zero3_softmax_gather_bytes=zero3_softmax_gather_bytes,
        )
        from program import megatron as _megatron
        grad_collective = REDUCE_SCATTER if ((self.zero_stage >= 2 and self.dp > 1)
                                             or _megatron.distributed_optimizer(self)) else ALL_REDUCE

        parallelism_mode = self.get_parallelism_mode()
        def _clone_comm_rules(rules: Dict[str, Dict[str, Optional[Dict[str, str]]]]) -> Dict[str, Dict[str, Optional[Dict[str, str]]]]:
            cloned: Dict[str, Dict[str, Optional[Dict[str, str]]]] = {}
            for op_name, directions in rules.items():
                cloned[op_name] = {}
                for direction, spec in directions.items():
                    cloned[op_name][direction] = dict(spec) if spec else None
            return cloned

        def _resolve_comm_rules(
            mode: ParallelismMode,
            use_moe_layer: bool,
        ) -> Dict[str, Dict[str, Optional[Dict[str, str]]]]:
            base_rules = COMMUNICATION_RULES.get(mode) or {}
            if not use_moe_layer:
                return base_rules
            overrides = MOE_COMMUNICATION_RULES.get(mode)
            if not overrides:
                return base_rules
            merged = _clone_comm_rules(base_rules)
            for op_name, directions in overrides.items():
                merged.setdefault(op_name, {})
                for direction, spec in directions.items():
                    merged[op_name][direction] = dict(spec) if spec else None
            return merged

        graph_ep = self.ep if self.use_moe else 1
        participants_lookup_base = {
            'tp': int(getattr(self, 'tp', 0) or 0),
            'cp': int(getattr(self, 'cp', 0) or 0),
            'ep': int(graph_ep or 0),
            'dp': int(getattr(self, 'dp', 0) or 0),
            'pp': int(getattr(self, 'pp', 0) or 0),
        }
        group_members = {
            "MLP": ("ffn1", "gelu", "ffn2"),
        }
        def _build_transformer_template(
            transformer_timings_local: Dict[str, OperationTiming],
            *,
            use_moe_layer: bool,
            ep_dense_sync_bytes: int,
        ) -> BlockTemplate:
            participants_lookup = dict(participants_lookup_base)
            if use_moe_layer:
                participants_lookup["ep"] = int(self._moe_routing_group())
            transformer_operation_entries: List[Dict[str, Any]] = []
            transformer_comm_metadata: Dict[str, Dict[str, Any]] = {}
            rules_by_mode = _resolve_comm_rules(parallelism_mode, use_moe_layer)

            def _register_specs(specs: Sequence[CommSpec]) -> List[str]:
                names: List[str] = []
                for spec in specs:
                    existing = transformer_comm_metadata.get(spec.name)
                    extra = dict(spec.extra or {})
                    if existing:
                        if (
                            existing["size"] != spec.size_bytes
                            or existing["type"] != spec.kind
                            or existing["participants"] != spec.participants
                            or existing["interconnect_type"] != spec.interconnect
                        ):
                            raise ValueError(
                                f"Conflicting CommSpec registration for '{spec.name}' "
                                f"(existing={existing}, new={spec})"
                            )
                        for key, value in extra.items():
                            if key in existing and existing[key] != value:
                                raise ValueError(
                                    f"Conflicting CommSpec metadata for '{spec.name}' on key '{key}' "
                                    f"(existing={existing[key]!r}, new={value!r})"
                                )
                            existing.setdefault(key, value)
                    else:
                        transformer_comm_metadata[spec.name] = {
                            "size": spec.size_bytes,
                            "type": spec.kind,
                            "participants": spec.participants,
                            "interconnect_type": spec.interconnect,
                        }
                        transformer_comm_metadata[spec.name].update(extra)
                    names.append(spec.name)
                return names

            def _make_ep_dense_sync_specs(
                op_name: str,
                direction: str,
                total_bytes: int,
            ) -> Tuple[CommSpec, ...]:
                bytes_int = int(math.ceil(float(total_bytes or 0.0)))
                if bytes_int <= 0 or self.ep <= 1:
                    return ()
                name = f"ep_dense_sync_{op_name}_{direction}"
                return (
                    CommSpec(
                        name=name,
                        kind=grad_collective,
                        size_bytes=bytes_int,
                        participants=int(self.ep),
                        interconnect="ep",
                        extra={
                            "scope": "ep",
                            "direction": direction,
                            "op_name": op_name,
                            "parallelism_mode": parallelism_mode.value,
                            "placement": "post",
                        },
                    ),
                )

            def _build_direction_entry(
                op_name: str,
                direction: str,
                timing: Optional[DirectionTiming],
            ) -> Dict[str, Any]:
                if timing is None:
                    return {"duration": 0.0, "reduction": 0.0, "comm_keys": []}
                if op_name in {"moe_dispatch", "moe_combine"}:
                    specs = list(self._make_moe_comm_specs(op_name, direction, timing.comm_bytes))
                    if op_name == "moe_dispatch" and specs:
                        extra = self._megatron_moe_dispatch_specs(direction)
                        specs = ([s for s in extra if "token_count" in s.name] + specs
                                 + [s for s in extra if "token_count" not in s.name])
                else:
                    specs = list(_make_comm_specs(op_name, direction, timing.comm_bytes, rules_by_mode))
                if (
                    include_transformer_backward
                    and direction == "backward"
                    and op_name == "layernorm1"
                    and ep_dense_sync_bytes > 0

                    and not (self.use_moe and self.zero_stage >= 2)
                ):
                    specs.extend(_make_ep_dense_sync_specs(op_name, direction, ep_dense_sync_bytes))
                if direction == "backward" and op_name in ("qkv_proj", "MLP"):
                    fraction = self._async_dgrad_fraction(op_name, timing.compute_time)
                    if fraction > 0.0:
                        specs = [
                            replace(spec, extra={**dict(spec.extra or {}), "async_dgrad_overlap_fraction": fraction})
                            if spec.interconnect == "tp" else spec
                            for spec in specs
                        ]
                return {
                    "duration": timing.compute_time,
                    "reduction": timing.comm_time,
                    "comm_keys": _register_specs(specs),
                    "activity": self._direction_activity(op_name, direction, timing),
                }

            def _make_comm_specs(
                op_name: str,
                direction: str,
                total_bytes: float,
                rules: Dict[str, Dict[str, Optional[Dict[str, str]]]],
            ) -> Tuple[CommSpec, ...]:
                bytes_int = int(math.ceil(float(total_bytes or 0.0)))
                if bytes_int <= 0:
                    return ()
                rule_spec = rules.get(op_name) or rules.get(COMM_RULE_DEFAULT_KEY)
                if not rule_spec:
                    raise ValueError(
                        f"Missing communication rule for op '{op_name}' in mode '{parallelism_mode.value}' "
                        f"with non-zero comm_bytes ({bytes_int})"
                    )
                rule = rule_spec.get(direction)
                if not rule:
                    return ()
                kind = rule.get("kind")
                scope = rule.get("participants")
                if kind is None or not scope:
                    return ()
                if not isinstance(kind, CollectiveType):
                    raise TypeError(
                        f"Comm rule for '{op_name}' must use CollectiveType (got {type(kind).__name__})"
                    )
                kind_enum = kind
                participants = participants_lookup.get(scope, 0)
                if participants <= 1:
                    return ()
                interconnect = rule.get("interconnect", scope)
                count_raw = rule.get("count", 1)
                try:
                    count = int(count_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Comm rule for '{op_name}' has non-integer count ({count_raw!r})"
                    ) from exc
                if count <= 0:
                    return ()
                if count == 1:
                    sizes = [bytes_int]
                else:
                    per_size = bytes_int // count
                    if per_size * count != bytes_int:
                        raise ValueError(
                            f"Comm bytes for '{op_name}' ({bytes_int}) must be divisible by count={count}"
                        )
                    sizes = [per_size for _ in range(count)]
                placement_spec = rule.get("placement", None)
                placements: List[str] = []
                if placement_spec is None:
                    placements = ["post" for _ in range(len(sizes))]
                elif isinstance(placement_spec, str):
                    placements = [placement_spec for _ in range(len(sizes))]
                else:
                    try:
                        placements = list(placement_spec)
                    except TypeError as exc:
                        raise ValueError(
                            f"Comm rule for '{op_name}' has invalid placement ({placement_spec!r})"
                        ) from exc
                    if len(placements) != len(sizes):
                        raise ValueError(
                            f"Comm rule for '{op_name}' placement length mismatch "
                            f"(expected {len(sizes)}, got {len(placements)})"
                        )
                specs: List[CommSpec] = []
                for idx, size in enumerate(sizes):
                    if size <= 0:
                        continue
                    placement = placements[idx] if idx < len(placements) else "post"
                    if placement not in {"pre", "post"}:
                        raise ValueError(
                            f"Comm rule for '{op_name}' has invalid placement '{placement}' "
                            "(expected 'pre' or 'post')"
                        )
                    name = f"{op_name}_{direction}_{kind_enum.value}"
                    if len(sizes) > 1:
                        name = f"{name}_part{idx}"
                    specs.append(
                        CommSpec(
                            name=name,
                            kind=kind_enum,
                            size_bytes=size,
                            participants=participants,
                            interconnect=interconnect,
                            extra={
                                "scope": scope,
                                "direction": direction,
                                "op_name": op_name,
                                "parallelism_mode": parallelism_mode.value,
                                "placement": placement,
                            },
                        )
                    )
                return tuple(specs)

            def _build_group_operation(name: str, members: Tuple[str, ...]) -> OperationTiming:
                ops = tuple(transformer_timings_local[m] for m in members)
                group = OperationGroup(name, ops)
                forward_memory = self._combine_mem(*(op.forward.memory_accesses for op in ops))
                forward_flops = sum(op.forward.flops for op in ops)
                forward_dir = DirectionTiming(
                    compute_time=group.forward_compute_time(),
                    comm_time=group.forward_comm_time(),
                    comm_bytes=group.forward_comm_bytes(),
                    flops=forward_flops,
                    memory_accesses=forward_memory,
                    activity=group.forward_activity(),
                )
                if include_transformer_backward:
                    backward_memory = self._combine_mem(
                        *(op.backward.memory_accesses for op in ops if op.backward is not None)
                    )
                    backward_flops = sum((op.backward.flops if op.backward else 0.0) for op in ops)
                    backward_dir = DirectionTiming(
                        compute_time=group.backward_compute_time(),
                        comm_time=group.backward_comm_time(),
                        comm_bytes=group.backward_comm_bytes(),
                        flops=backward_flops,
                        memory_accesses=backward_memory,
                        activity=group.backward_activity(),
                    )
                else:
                    backward_dir = None
                metadata = {}
                if all("checkpoint_invocations" in op.metadata for op in ops):
                    metadata["checkpoint_invocations"] = {
                        phase: DirectionTiming(
                            compute_time=sum(op.metadata["checkpoint_invocations"][phase].compute_time for op in ops),
                            activity=sum((op.metadata["checkpoint_invocations"][phase].activity for op in ops),
                                         ZERO_ACTIVITY),
                        ) for phase in ("forward", "recompute")
                    }
                return OperationTiming(name, forward_dir, backward_dir, metadata=metadata)

            def _resolve_operation(name: str) -> OperationTiming:
                if name in group_members:
                    return _build_group_operation(name, group_members[name])
                return transformer_timings_local[name]

            if parallelism_mode not in (
                ParallelismMode.CONTEXT,
                ParallelismMode.TENSOR_CONTEXT,
                ParallelismMode.TENSOR_CONTEXT_HYBRID,
                ParallelismMode.TENSOR,
                ParallelismMode.TENSOR_SEQUENCE,
                ParallelismMode.SINGLE,
            ):
                raise ValueError(f"Unsupported parallelism mode: {parallelism_mode}")
            op_names: Sequence[str] = ("layernorm1", "qkv_proj", "attention", "output_proj", "layernorm2", "MLP")
            op_names = list(op_names)
            if "rope" in transformer_timings_local:
                op_names.insert(op_names.index("qkv_proj") + 1, "rope")
            if "optimizer" in transformer_timings_local:
                op_names.append("optimizer")
            if use_moe_layer and "MLP" in op_names:
                mlp_index = op_names.index("MLP")
                for offset, op_name in enumerate(("router", "moe_dispatch")):
                    if op_name not in op_names:
                        if op_name not in transformer_timings_local:
                            raise KeyError(f"Missing transformer timing for operation '{op_name}'")
                        op_names.insert(mlp_index + offset, op_name)
                if "moe_combine" not in op_names:
                    if "moe_combine" not in transformer_timings_local:
                        raise KeyError("Missing transformer timing for operation 'moe_combine'")
                    mlp_index = op_names.index("MLP")
                    op_names.insert(mlp_index + 1, "moe_combine")

            for key in op_names:
                try:
                    op_timing = _resolve_operation(key)
                except KeyError as exc:
                    raise KeyError(f"Missing transformer timing for operation '{key}'") from exc

                entry = {
                    "name": key,
                    "forward": _build_direction_entry(key, "forward", op_timing.forward),
                    "backward": _build_direction_entry(
                        key, "backward", op_timing.backward if include_transformer_backward else None
                    ),
                }

                if "checkpoint_invocations" in op_timing.metadata:
                    entry["forward"]["invocations"] = {
                        phase: (record.compute_time, record.activity)
                        for phase, record in op_timing.metadata["checkpoint_invocations"].items()
                    }

                if not include_transformer_backward:
                    entry["backward"] = {"duration": 0.0, "reduction": 0.0, "comm_keys": []}

                transformer_operation_entries.append(entry)

            return BlockTemplate.from_gemm_entries(
                transformer_operation_entries,
                transformer_comm_metadata,
            )

        transformer_template = _build_transformer_template(
            transformer_timings,
            use_moe_layer=False,
            ep_dense_sync_bytes=ep_dense_sync_bytes_dense,
        )
        moe_transformer_template: Optional[BlockTemplate] = None
        moe_layer_mask = list(getattr(self, "moe_layer_mask", []) or [])
        has_moe_layers = bool(self.use_moe and any(moe_layer_mask))
        if has_moe_layers:
            if not moe_transformer_timings or not moe_node_breakdown:
                raise ValueError("MoE layer schedule requires moe_transformer_timings and moe_node_breakdown.")
            moe_transformer_template = _build_transformer_template(
                moe_transformer_timings,
                use_moe_layer=True,
                ep_dense_sync_bytes=ep_dense_sync_bytes_moe,
            )

        transformer_blocks = BlockTemplates(
            dense=transformer_template,
            moe=moe_transformer_template,
        )

        transformer_blocks_no_dp: Optional[BlockTemplates] = None
        if need_no_dp_variant and include_transformer_backward and self.use_moe and self.ep > 1:
            dense_template_no_dp = _build_transformer_template(
                transformer_timings,
                use_moe_layer=False,
                ep_dense_sync_bytes=0,
            )
            moe_template_no_dp: Optional[BlockTemplate] = None
            if has_moe_layers:
                if not moe_transformer_timings or not moe_node_breakdown:
                    raise ValueError("MoE layer schedule requires moe_transformer_timings and moe_node_breakdown.")
                moe_template_no_dp = _build_transformer_template(
                    moe_transformer_timings,
                    use_moe_layer=True,
                    ep_dense_sync_bytes=0,
                )
            transformer_blocks_no_dp = BlockTemplates(
                dense=dense_template_no_dp,
                moe=moe_template_no_dp,
            )

        dense_transformer_f = node_breakdown.get('transformer_time_f', 0.0)
        dense_transformer_b = node_breakdown.get('transformer_time_b', 0.0) if include_pipeline_backward else 0.0
        moe_transformer_f = dense_transformer_f
        moe_transformer_b = dense_transformer_b
        if has_moe_layers and moe_node_breakdown:
            moe_transformer_f = moe_node_breakdown.get('transformer_time_f', dense_transformer_f)
            moe_transformer_b = (
                moe_node_breakdown.get('transformer_time_b', dense_transformer_b)
                if include_pipeline_backward
                else 0.0
            )

        optimizer_per_layer_dense = self.get_data_parallel_reduction_llm(
            hidden_dim, intermediate_size
        )
        optimizer_activity_dense = self._roofline_activity(
            "apply_grad", "pointwise-grad-clipping"
        )
        optimizer_per_layer_moe = optimizer_per_layer_dense
        optimizer_activity_moe = optimizer_activity_dense
        if has_moe_layers:
            optimizer_per_layer_moe = self.get_data_parallel_reduction_llm(
                hidden_dim,
                self.moe_intermediate_size,
                moe=True,
            )
            optimizer_activity_moe = self._roofline_activity(
                "apply_grad", "pointwise-grad-clipping"
            )

        optimizer_embedding = (
            self.get_embedding_apply_grad_llm(hidden_dim, vocab_size)
            if include_pipeline_backward
            else 0.0
        )
        optimizer_embedding_activity = self._roofline_activity(
            "apply_grad", "pointwise-grad-clipping"
        )
        optimizer_softmax = (
            self.get_softmax_apply_grad_llm(hidden_dim, vocab_size)
            if include_pipeline_backward
            else 0.0
        )
        optimizer_softmax_activity = self._roofline_activity(
            "apply_grad", "pointwise-grad-clipping"
        )
        self._set_endpoint_collectives("optimizer_embedding", ())
        self._set_endpoint_collectives("optimizer_softmax", ())
        if (
            include_pipeline_backward
            and self.run_type == "training"
            and int(self.pp) > 1
            and getattr(self, "tied_embeddings", False)
            and not self._is_vit_model()
        ):
            divisor = getattr(self.model, "make_vocab_size_divisible_by", None)
            sync_vocab_size = int(vocab_size)
            if divisor is not None:
                multiple = int(divisor) * max(1, int(self.tp))
                sync_vocab_size = ((sync_vocab_size + multiple - 1) // multiple) * multiple
            emb_params_rank = math.ceil(float(sync_vocab_size) * float(hidden_dim) / max(1, int(self.tp)))
            rank_bytes = float(self.precision.grad_communication) * emb_params_rank
            group_bytes = (
                rank_bytes
                * max(1, int(self.tp)) * max(1, int(self.cp))
            )
            embedding_group_ar = self.network_model._analytical_collective(
                kind=ALL_REDUCE,
                size_bytes=float(math.ceil(rank_bytes)),
                participants=2,
                ib=self.links["pp"].bandwidth,
                ll=self.links["pp"].latency,
                debug_label="embedding_group_allreduce",
            )
            embedding_sync = self._endpoint_collective(
                "embedding_group_allreduce", ALL_REDUCE, group_bytes, embedding_group_ar,
                0.0, axis="pp", participants=2,
            )
            self._set_endpoint_collectives("optimizer_embedding", embedding_sync)
            self._set_endpoint_collectives("optimizer_softmax", embedding_sync)
            optimizer_embedding += embedding_group_ar
            last_copy_update = self._apply_grad_sharded(emb_params_rank)
            optimizer_softmax += embedding_group_ar + last_copy_update
            optimizer_softmax_activity = optimizer_softmax_activity + self._roofline_activity(
                "apply_grad", "pointwise-grad-clipping"
            )

        if include_pipeline_backward and self._sourced_optimizer_enabled():
            optimizer_total, optimizer_activity_total = self._sourced_optimizer_timing()
            optimizer_per_layer_dense = optimizer_per_layer_moe = optimizer_total / int(self.num_layers)
            optimizer_activity_dense = optimizer_activity_moe = optimizer_activity_total.scaled(1.0 / int(self.num_layers))
            optimizer_embedding = optimizer_softmax = 0.0
            optimizer_embedding_activity = optimizer_softmax_activity = ZERO_ACTIVITY

        def _endpoint_activity(name: str, direction: str) -> OperationActivity:
            operation = transformer_timings.get(name)
            timing = getattr(operation, direction, None) if operation is not None else None
            return timing.activity if timing is not None else ZERO_ACTIVITY

        endpoint_activities = {
            "embedding_f": _endpoint_activity("embedding", "forward"),
            "embedding_b": _endpoint_activity("embedding", "backward"),
            "linear_softmax_f": _endpoint_activity("linear_softmax", "forward"),
            "linear_softmax_b": _endpoint_activity("linear_softmax", "backward"),
            "optimizer": optimizer_activity_dense,
            "optimizer_moe": optimizer_activity_moe,
            "optimizer_embedding": optimizer_embedding_activity,
            "optimizer_softmax": optimizer_softmax_activity,
        }

        comp_times = {
            "embedding_f": node_breakdown.get('embedding_f', 0.0),
            "embedding_b": node_breakdown.get('embedding_b', 0.0) if include_pipeline_backward else 0.0,
            "linear_softmax_f": node_breakdown.get('linear_softmax_f', 0.0),
            "linear_softmax_b": node_breakdown.get('linear_softmax_b', 0.0) if include_pipeline_backward else 0.0,
            "transformer_f": dense_transformer_f,
            "transformer_b": dense_transformer_b,
            "transformer_f_dense": dense_transformer_f,
            "transformer_b_dense": dense_transformer_b,
            "transformer_f_moe": moe_transformer_f,
            "transformer_b_moe": moe_transformer_b,

            "optimizer": optimizer_per_layer_dense,
            "optimizer_moe": optimizer_per_layer_moe,

            "optimizer_embedding": optimizer_embedding,
            "optimizer_softmax": optimizer_softmax,
        }
        comp_times_no_dp = None
        if need_no_dp_variant:
            comp_times_no_dp = dict(comp_times)
            if include_pipeline_backward and self.use_moe and self.ep > 1:
                gemm_shapes_local = gemm_shapes
                if gemm_shapes_local is None:
                    gemm_shapes_local = llm_util.process_gemm_shapes(
                        self,
                        batch_size=batch_size,
                        seq_len=seq_len,
                        d_model=hidden_dim,
                        num_heads=self.num_heads,
                        kv_heads=self.kv_heads,
                        intermediate_size=intermediate_size,
                        vocab_size=vocab_size,
                    )
                gemm_linear = gemm_shapes_local.get("linear") if gemm_shapes_local else None
                if gemm_linear is not None:
                    comp_times_no_dp["linear_softmax_b"] = self.get_linear_softmax_b(
                        gemm_linear,
                    )
                comp_times_no_dp["embedding_b"] = self.get_embedding_b(
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    hidden_dim=hidden_dim,
                )
        from program import megatron as _megatron_dp
        misc_metadata = {

            "num_batch": self.mb if self.pp > 1 else 1,
            "num_layer": self.num_layers,
            "dp_zero_stage": self.zero_stage,
            "full_recomputation": self.full_recomputation,
            "dp_microbatch_mode": getattr(self, "dp_microbatch", "every_mb"),
            "dp_grad_sync": _megatron_dp.dp_grad_sync(self),
            "moe_layer_mask": moe_layer_mask,
            "model_type": self.model_type,
        }
        misc_metadata_final = dict(misc_metadata)
        misc_metadata_final["grad_accum_cycle"] = "final"
        misc_metadata_nonfinal = dict(misc_metadata)
        misc_metadata_nonfinal["grad_accum_cycle"] = "nonfinal"

        interconnect_params = self._build_interconnect_params()
        overlap_spec = OverlapSpec.from_legacy(
            self.get_parallelism_mode(),
            tp_overlap=getattr(self, "tp_overlap", 0.0),
            tp_sp_overlap=getattr(self, "tp_sp_overlap", 0.0),
            cp_overlap=getattr(self, "cp_overlap", 0.0),
        )

        def _workload(
            *,
            comp: Dict[str, Any],
            misc: Dict[str, Any],
            blocks: BlockTemplates,
            grad_accum_cycle: str,
        ) -> WorkloadSpec:
            return WorkloadSpec.from_timing(
                tp=self.tp,
                cp=self.cp,
                ep=graph_ep,
                pp=self.pp,
                dp=self.dp,
                run_type=getattr(self, "run_type", "training"),
                grad_accum_cycle=grad_accum_cycle,
                pipeline_interleave=int(getattr(self, "pipeline_interleave", 1) or 1),
                comp_times=comp,
                comm_metadata=comm_metadata,
                misc_metadata=misc,
                blocks=blocks,
                overlap=overlap_spec,
                interconnect=interconnect_params,
                activities=endpoint_activities,
                rank_activities=getattr(self, "_vtp_rank_activities", {}),
                rank_durations=getattr(self, "_vtp_rank_durations", {}),
                rank_collectives=getattr(self, "_vtp_rank_collectives", {}),
                embedded_collectives={
                    key: calls for key, calls in getattr(self, "_endpoint_collectives", {}).items()
                    if comp.get(key, 0.0) > 0.0
                    and not (self.disable_embedding_unembedding and key.startswith(("embedding_", "linear_softmax_")))
                },
            )

        workload = _workload(
            comp=comp_times,
            misc=misc_metadata_final,
            blocks=transformer_blocks,
            grad_accum_cycle="final",
        )
        workload_no_dp = None
        if need_no_dp_variant:
            workload_no_dp = _workload(
                comp=comp_times_no_dp or comp_times,
                misc=misc_metadata_nonfinal,
                blocks=transformer_blocks_no_dp or transformer_blocks,
                grad_accum_cycle="nonfinal",
            )

        return workload, workload_no_dp

    def _build_training_graphs_and_memory_data(self):
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        seq_len = self.seq_len
        num_heads = self.num_heads
        intermediate_size = self.intermediate_size
        kv_heads = self.kv_heads

        (
            total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params_per_rank,
            output_params_per_rank,
        ) = self._param_stats_per_rank(hidden_dim, intermediate_size, vocab_size)

        param_bytes = total_params_per_rank * self.precision.parameters
        transformer_param_layer_bytes = params_per_layer_per_rank * self.precision.parameters
        embedding_param_bytes = embedding_params_per_rank * self.precision.parameters
        softmax_param_bytes = output_params_per_rank * self.precision.parameters

        zero_shards = max(1, int(self.dp))
        from program import megatron as _megatron

        if (self.zero_stage == 2 and self.dp > 1) or _megatron.distributed_optimizer(self):
            zero2_embedding_gather_bytes = embedding_param_bytes / zero_shards
            zero2_transformer_gather_bytes = transformer_param_layer_bytes / zero_shards
            zero2_softmax_gather_bytes = softmax_param_bytes / zero_shards
        else:
            zero2_embedding_gather_bytes = 0.0
            zero2_transformer_gather_bytes = 0.0
            zero2_softmax_gather_bytes = 0.0

        if self.zero_stage >= 3 and self.dp > 1:
            zero3_embedding_gather_bytes = embedding_param_bytes / zero_shards
            zero3_transformer_gather_bytes = transformer_param_layer_bytes / zero_shards
            zero3_softmax_gather_bytes = softmax_param_bytes / zero_shards
        else:
            zero3_embedding_gather_bytes = 0.0
            zero3_transformer_gather_bytes = 0.0
            zero3_softmax_gather_bytes = 0.0

        self.zero3_ephemeral_peak_bytes = (
            max_layer_params * self.precision.parameters
            if self.zero_stage >= 3 and self.dp > 1
            else 0.0
        )

        num_SMs = self.hw_config.tech_config.core.num_bundles
        transformer_timings, node_breakdown = self.compute_all_gemm_and_node_times(
            batch_size,
            vocab_size,
            hidden_dim,
            seq_len,
            num_heads,
            kv_heads,
            intermediate_size,
            num_SMs,
            use_moe_override=False,
        )
        moe_transformer_timings = None
        moe_node_breakdown = None
        if self.use_moe:
            moe_transformer_timings, moe_node_breakdown = self.compute_all_gemm_and_node_times(
                batch_size,
                vocab_size,
                hidden_dim,
                seq_len,
                num_heads,
                kv_heads,
                self.moe_intermediate_size,
                num_SMs,
                use_moe_override=True,
            )

        mem_estimator = MemoryEstimator(self)
        memory_data = mem_estimator.build_memory_data(
            mode="training",
            batch_size=batch_size,
            seq_len=seq_len,
            zero3_ephemeral_peak_bytes=self.zero3_ephemeral_peak_bytes,
        )

        transformer_act_layer = memory_data.get("activation_mem_per_layer", 0.0)
        transformer_act_layer_inf = memory_data.get("activation_mem_per_layer_inference", 0.0)
        transformer_static_layer = memory_data.get("static_mem_per_layer", 0.0)
        gradient_mem_layer = memory_data.get("gradient_mem_per_layer", 0.0)
        optimizer_mem_layer = memory_data.get("optimizer_mem_per_layer", 0.0)
        weight_memory_layer = memory_data.get("weight_mem_per_layer", 0.0)
        transformer_mem_layer = memory_data.get("total_mem_per_layer", 0.0)
        if self.use_moe:
            dense_layers = max(0, self.num_layers - self.num_moe_layers)
            moe_layers = max(0, self.num_moe_layers)
            total_layers = max(1, dense_layers + moe_layers)
            act_dense = memory_data.get("activation_mem_per_layer_dense", transformer_act_layer)
            act_moe = memory_data.get("activation_mem_per_layer_moe", act_dense)
            act_inf_dense = memory_data.get("activation_mem_per_layer_inference_dense", transformer_act_layer_inf)
            act_inf_moe = memory_data.get("activation_mem_per_layer_inference_moe", act_inf_dense)
            static_dense = memory_data.get("static_mem_per_layer_dense", transformer_static_layer)
            static_moe = memory_data.get("static_mem_per_layer_moe", static_dense)
            grad_dense = memory_data.get("gradient_mem_per_layer_dense", gradient_mem_layer)
            grad_moe = memory_data.get("gradient_mem_per_layer_moe", grad_dense)
            opt_dense = memory_data.get("optimizer_mem_per_layer_dense", optimizer_mem_layer)
            opt_moe = memory_data.get("optimizer_mem_per_layer_moe", opt_dense)
            weight_dense = memory_data.get("weight_mem_per_layer_dense", weight_memory_layer)
            weight_moe = memory_data.get("weight_mem_per_layer_moe", weight_dense)
            total_dense = memory_data.get("total_mem_per_layer_dense", transformer_mem_layer)
            total_moe = memory_data.get("total_mem_per_layer_moe", total_dense)
            transformer_act_layer = (act_dense * dense_layers + act_moe * moe_layers) / total_layers
            transformer_act_layer_inf = (act_inf_dense * dense_layers + act_inf_moe * moe_layers) / total_layers
            transformer_static_layer = (static_dense * dense_layers + static_moe * moe_layers) / total_layers
            gradient_mem_layer = (grad_dense * dense_layers + grad_moe * moe_layers) / total_layers
            optimizer_mem_layer = (opt_dense * dense_layers + opt_moe * moe_layers) / total_layers
            weight_memory_layer = (weight_dense * dense_layers + weight_moe * moe_layers) / total_layers
            transformer_mem_layer = (total_dense * dense_layers + total_moe * moe_layers) / total_layers

        if self._debug_memory:
            layers_per_device = max(1, math.ceil(self.num_layers / max(1, self.pp)))
            to_gib = lambda bytes_val: float(bytes_val) / float(1024 ** 3)
            self._memory_breakdown_debug = {
                "layers_per_device": layers_per_device,
                "zero_stage": self.zero_stage,
                "activation_gib": to_gib(transformer_act_layer * layers_per_device),
                "weight_gib": to_gib(weight_memory_layer * layers_per_device),
                "gradient_gib": to_gib(gradient_mem_layer * layers_per_device),
                "optimizer_gib": to_gib(optimizer_mem_layer * layers_per_device),
                "static_gib": to_gib(transformer_static_layer * layers_per_device),
                "total_layer_gib": to_gib(transformer_mem_layer * layers_per_device),
            }
            if self.zero_stage >= 3 and self.zero3_ephemeral_peak_bytes:
                self._memory_breakdown_debug["zero3_ephemeral_gib"] = to_gib(
                    self.zero3_ephemeral_peak_bytes
                )
            extra_static = memory_data.get("extra_static_bytes_per_device", {}) or {}
            if extra_static:
                self._memory_breakdown_debug["extra_static_gib_max"] = to_gib(
                    max(extra_static.values())
                )

        workload, workload_no_dp = self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            moe_node_breakdown=moe_node_breakdown,
            moe_transformer_timings=moe_transformer_timings,
            batch_size=batch_size,
            seq_len=seq_len,
            hidden_dim=hidden_dim,

            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            include_pipeline_backward=True,
            include_transformer_backward=True,
            zero2_embedding_gather_bytes=zero2_embedding_gather_bytes,
            zero2_transformer_gather_bytes=zero2_transformer_gather_bytes,
            zero2_softmax_gather_bytes=zero2_softmax_gather_bytes,
            zero3_embedding_gather_bytes=zero3_embedding_gather_bytes,
            zero3_transformer_gather_bytes=zero3_transformer_gather_bytes,
            zero3_softmax_gather_bytes=zero3_softmax_gather_bytes,
        )

        self.transformer_blocks = workload.blocks
        self.transformer_analytical_time_forward = node_breakdown['transformer_time_f']

        self.transformer_analytical_time_backward_combined = node_breakdown['transformer_time_b_combined']
        self.transformer_analytical_time_backward = self.transformer_analytical_time_backward_combined
        if self.use_moe and moe_node_breakdown and self.num_moe_layers == self.num_layers:
            self.transformer_analytical_time_forward = moe_node_breakdown.get(
                "transformer_time_f",
                self.transformer_analytical_time_forward,
            )
            self.transformer_analytical_time_backward_combined = moe_node_breakdown.get(
                "transformer_time_b_combined",
                self.transformer_analytical_time_backward_combined,
            )
            self.transformer_analytical_time_backward = self.transformer_analytical_time_backward_combined

        self.workload = workload
        self.workload_no_dp = workload_no_dp
        self.pipeline_interconnect = dict(workload.interconnect)

        return mem_estimator, memory_data

    def calc_time_llm(self):
        mem_estimator, memory_data = self._build_training_graphs_and_memory_data()

        mode = self.execution_mode
        time_fw_bw_no_dp: Optional[float] = None
        if self.gradient_accumulation_steps > 1:
            if not self.workload_no_dp:
                raise RuntimeError("Gradient accumulation steps > 1 requires a non-final accumulation workload")
            dispatcher_no_dp = LLMExecutionDispatcher(self, self.workload_no_dp)
            try:
                result_no_dp = dispatcher_no_dp.run(mode)
            except NotImplementedError as exc:
                raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc
            time_fw_bw_no_dp = result_no_dp.total_time

        dispatcher = LLMExecutionDispatcher(self, self.workload)
        try:
            result = dispatcher.run(mode)
        except NotImplementedError as exc:
            raise NotImplementedError(f"{exc}. Selected execution mode '{mode.value}'.") from exc
        time_fw_bw = result.total_time

        if self.estimate_memory:
            memory_program = dispatcher.build_flat_program_for_memory()
            mem_estimator.simulate_peak(
                memory_program,
                memory_data,
                mode="training",
                filename="memory_graph_training",
            )

        if self._generate_graphs:
            print(f"Analytical transformer forward time: {self.transformer_analytical_time_forward:.4f}s")
            print(f"Analytical transformer backward time: {self.transformer_analytical_time_backward:.4f}s")
            if self.transformer_astrasim_time_forward is not None and self.transformer_astrasim_time_backward is not None:
                print(f"Actual transformer forward time: {self.transformer_astrasim_time_forward:.4f}s")
                print(f"Actual transformer backward time: {self.transformer_astrasim_time_backward:.4f}s")

        if self.gradient_accumulation_steps > 1:
            if time_fw_bw_no_dp is None:
                raise RuntimeError("Missing non-final accumulation timing for gradient accumulation computation")
            self.tot_time = (
                time_fw_bw_no_dp * (self.gradient_accumulation_steps - 1) + time_fw_bw
            )
        else:
            self.tot_time = time_fw_bw
        energy_parts = []
        if self.gradient_accumulation_steps > 1:
            if result_no_dp.energy is None:
                raise RuntimeError("non-final training execution did not return energy")
            energy_parts.append(
                (
                    apply_training_graph_accounting(
                        result_no_dp.energy, pipeline_stages=self.pp, data_parallel=self.dp
                    ),
                    self.gradient_accumulation_steps - 1,
                )
            )
        if result.energy is None:
            raise RuntimeError("final training execution did not return energy")
        energy_parts.append(
            (
                apply_training_graph_accounting(
                    result.energy, pipeline_stages=self.pp, data_parallel=self.dp
                ),
                1,
            )
        )
        self.execution_energy = combine_execution_energy(
            energy_parts,

            execution_mode=(result.energy.execution_mode if self.use_moe and self.zero_stage >= 2 and str(getattr(self.model, "run_type", "training")).lower() != "inference" else mode.value),
        )
        write_energy_report(
            self.output_dir,
            self.execution_energy,
            hardware=self.hw_config,
            metadata={
                "run_type": "training",
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "final_accumulation_executions": 1,
                "nonfinal_accumulation_executions": max(
                    0, self.gradient_accumulation_steps - 1
                ),
                "gpu_activity_model": self.activity_model_provenance(),
            },
        )
        return self.tot_time

    def estimate_memory_only(self) -> Dict[str, Any]:
        mem_estimator, memory_data = self._build_training_graphs_and_memory_data()

        dispatcher = LLMExecutionDispatcher(self, self.workload)
        memory_program = dispatcher.build_flat_program_for_memory()
        _, training_peak_gb = mem_estimator.simulate_peak(
            memory_program,
            memory_data,
            mode="training",
            filename="memory_graph_training",
        )

        return {
            "peak_gb": float(training_peak_gb),
            "checkpoint_accounting": self.memory_checkpoint_accounting,
            "capacity_gb": getattr(self, "memory_capacity_per_device_gb", None),
            "headroom_gb": getattr(self, "memory_headroom_gb", None),
            "capacity_exceeded": bool(getattr(self, "memory_capacity_exceeded", False)),
            "capacity_violation_gb": float(getattr(self, "memory_capacity_violation_gb", 0.0) or 0.0),
            "output_dir": self.output_dir,
        }

    def _simulate_with_memory(
        self,
        memory_program: Any,
        memory_data: Dict[str, Any],
        mode: str = "training",
        filename: Optional[str] = None,
    ) -> Tuple[float, float]:
        from program.memory_sim import simulate_memory

        time_with_memory, peak_mem = simulate_memory(
            memory_program,
            memory_data,
            mode,
            self.output_dir,
            filename or "memory_graph",
        )
        self.memory_peak_gb = peak_mem
        self.memory_checkpoint_accounting = memory_data.get("checkpoint_accounting", "legacy")

        hardware_mem_bytes = getattr(self.DRAM, "size", None)
        if hardware_mem_bytes is None and hasattr(self.hw_config, "tech_config"):
            tech_cfg = self.hw_config.tech_config
            if hasattr(tech_cfg, "DRAM"):
                hardware_mem_bytes = getattr(tech_cfg.DRAM, "size", None)

        if hardware_mem_bytes is not None:
            hardware_mem_gib = float(hardware_mem_bytes) / float(1024 ** 3)
            self.memory_capacity_per_device_gb = hardware_mem_gib
            mem_delta = hardware_mem_gib - peak_mem
            self.memory_headroom_gb = mem_delta
            memory_dir = os.path.join(self.output_dir, "memory-summary")
            os.makedirs(memory_dir, exist_ok=True)
            info_lines = [
                f"Simulation mode: {mode}",
                f"Hardware memory capacity (per gpu): {hardware_mem_gib:.2f} GiB",
                f"Simulated peak memory usage(per gpu): {peak_mem:.2f} GiB",
            ]
            if self.zero_stage >= 3 and self.zero3_ephemeral_peak_bytes:
                info_lines.append(
                    "ZeRO-3 ephemeral param gather (per gpu): {:.2f} GiB".format(
                        self.zero3_ephemeral_peak_bytes / float(1024 ** 3)
                    )
                )
            if mem_delta < 0:
                info_lines.append(f"[WARN] Peak memory exceeds capacity by {abs(mem_delta):.2f} GiB")
                self.memory_capacity_exceeded = True
                self.memory_capacity_violation_gb = max(self.memory_capacity_violation_gb, abs(mem_delta))
            else:
                info_lines.append(f"Remaining memory headroom: {mem_delta:.2f} GiB")
            info_path = os.path.join(memory_dir, "memory_capacity_comparison.txt")
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write("\n".join(info_lines) + "\n")
            if self._debug_memory:
                breakdown = self._memory_breakdown_debug or {}
                print("[memory] Memory summary (per device):")
                print("  Capacity: {:.2f} GiB".format(hardware_mem_gib))
                print("  Simulated peak usage: {:.2f} GiB".format(peak_mem))
                print("  Headroom: {:.2f} GiB".format(mem_delta))
                if breakdown:
                    print("  Layers per device: {}".format(breakdown.get("layers_per_device")))
                    print("  Breakdown (GiB, approx per device):")

                    print("    Weights: {:.2f}".format(breakdown.get("weight_gib", 0.0)))
                    print("    Gradients: {:.2f}".format(breakdown.get("gradient_gib", 0.0)))
                    print("    Optimizer: {:.2f}".format(breakdown.get("optimizer_gib", 0.0)))
                    print("    Static buffers: {:.2f}".format(breakdown.get("static_gib", 0.0)))

        else:
            self.memory_capacity_per_device_gb = None
            self.memory_headroom_gb = None
            if self._debug_memory:
                print("[memory] Memory summary (per device):")
                print("  Capacity: unknown")
                print("  Simulated peak usage: {:.6f} GiB".format(peak_mem))

        return time_with_memory, peak_mem

    def get_time(self):
        return self.tot_time

    def memory_capacity_warning(self) -> Optional[str]:
        violation = max(0.0, getattr(self, "memory_capacity_violation_gb", 0.0))
        if violation > 0.0:
            return (
                f"[WARN] Peak memory exceeds capacity by {violation:.2f} GiB. "
                "Please change parallelism settings for realistic results."
            )
        return None
