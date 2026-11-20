import functools
import math
import types
import os
import pickle
import sys
import config
import shutil
import itertools
from dataclasses import replace
from typing import Dict, NamedTuple, Optional

import sim_settings
from opmodel import settings as _settings

from activity import MemoryTraffic, OperationActivity, OperationTimingResult, ZERO_ACTIVITY

import util
from hw_component import Core, MemoryHierarchy, Network, DRAM
from astrasim_lib import run_cache_astrasim
from tile import AccessBytes, TiledGEMM, formatBytes
from timing_model import CollectiveType

class LinkInfo(NamedTuple):
    bandwidth: float
    latency: float

def _active_axis_sizes(hw_config, run_type: str) -> Dict[str, int]:
    sch = getattr(hw_config, "sch_config", None)
    if sch is None:
        return {"tp": 1, "cp": 1, "ep": 1, "pp": 1, "dp": 1}
    normalized = str(run_type or "training").lower()
    if normalized == "inference":
        dp = 1
        ep = int(sch.inference.moe_dp)
    else:
        dp = int(sch.train.dp)
        ep = int(sch.train.ep)
    return {
        "tp": int(sch.tp),
        "cp": int(sch.cp),
        "ep": ep,
        "pp": int(sch.pp),
        "dp": dp,
    }

def _refresh_network_layout(hw_config, axis_sizes: Dict[str, int]):
    layout = getattr(hw_config, "network_layout", None)
    if layout is None or not getattr(layout, "dimensions", None):
        return layout
    dimensions = []
    for dim in layout.dimensions:
        axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ()) or ()]
        if not axes:
            dimensions.append(dim)
            continue
        expected = 1
        for axis in axes:
            expected *= max(1, int(axis_sizes.get(axis, 1)))
        size_value = int(getattr(dim, "size", expected) or expected)
        if expected != size_value:
            dimensions.append(replace(dim, size=expected))
        else:
            dimensions.append(dim)
    return config._build_network_layout_config(
        dimensions,
        getattr(layout, "faulty_links", ()) or (),
        getattr(layout, "overlap_config", None),
        getattr(layout, "link_energy_rule", None),
    )

class Parallelism:
    def __init__(self, exp_config, run_type: str = "training"):
        sch = exp_config.sch_config
        self.pp = sch.pp
        self.mb = sch.mb
        self.tp = sch.tp
        self.cp = sch.cp
        self.tp_sp = sch.tp_sp
        self.run_type = str(run_type or "training").lower()
        if self.run_type == "inference":
            self.dp = 1
            self.ep = int(sch.inference.moe_dp)
            self.tp_ep = False
            self.replica_count = int(sch.inference.replica_count)
            self.moe_dp = int(sch.inference.moe_dp)
        else:
            self.dp = int(sch.train.dp)
            self.ep = int(sch.train.ep)
            self.tp_ep = bool(sch.train.tp_ep)
            self.replica_count = 1
            self.moe_dp = 1

class model_GEMM:
    def __init__(self, exp_config):
        self.M = exp_config.model_config.M
        self.K = exp_config.model_config.K
        self.N = exp_config.model_config.N
        self.backward = exp_config.model_config.backward
        self.gemm_shard_axis = str(exp_config.model_config.gemm_shard_axis).strip().lower()

class model_LLM:
    def __init__(self, exp_config):
        self.global_batch_size = exp_config.model_config.global_batch_size
        self.gradient_accumulation_steps = exp_config.model_config.gradient_accumulation_steps
        self.vocab_size = exp_config.model_config.vocab_size
        self.num_layers = exp_config.model_config.num_layers
        self.hidden_dim = exp_config.model_config.hidden_dim
        self.seq_len = exp_config.model_config.seq_len
        self.decode_len = exp_config.model_config.decode_len
        self.num_heads = exp_config.model_config.num_heads
        self.head_dim = getattr(exp_config.model_config, "head_dim", None)
        self.tied_embeddings = exp_config.model_config.tied_embeddings
        self.make_vocab_size_divisible_by = getattr(exp_config.model_config, "make_vocab_size_divisible_by", None)
        self.gradient_clipping = bool(getattr(exp_config.model_config, "gradient_clipping", False))
        self.implementation = str(getattr(exp_config.model_config, "implementation", "fused") or "fused")
        self.implementation_options = dict(getattr(exp_config.model_config, "implementation_options", None) or {})
        self.kernel_description_resolved = getattr(exp_config.model_config, "kernel_description_resolved", False)
        self.engine_settings = getattr(exp_config.model_config, "engine", None)
        self.model_type = exp_config.model_config.model_type
        self.qwen2_bias_mode = getattr(exp_config.model_config, "qwen2_bias_mode", None)
        self.max_position_embeddings = getattr(exp_config.model_config, "max_position_embeddings", None)
        self.intermediate_size = exp_config.model_config.intermediate_size
        self.n_tokens = exp_config.model_config.n_tokens
        self.run_type = exp_config.model_config.run_type
        self.disable_embedding_unembedding = bool(
            getattr(exp_config.model_config, "disable_embedding_unembedding", False)
        )
        self.attention_type = exp_config.model_config.attention.attention_type
        self.kv_heads = (
            exp_config.model_config.attention.kv_heads
            if hasattr(exp_config.model_config.attention, "kv_heads")
            else None
        )
        self.use_flashattention = getattr(exp_config.model_config.attention, "use_flashattention", False)
        self.use_rope = getattr(exp_config.model_config.attention, "use_rope", False)
        self.attention_tile_size = getattr(exp_config.model_config.attention, "attention_tile_size", None)
        self.kv_lora_rank = getattr(exp_config.model_config.attention, "kv_lora_rank", None)
        self.q_lora_rank = getattr(exp_config.model_config.attention, "q_lora_rank", None)
        self.qk_nope_head_dim = getattr(exp_config.model_config.attention, "qk_nope_head_dim", None)
        self.qk_rope_head_dim = getattr(exp_config.model_config.attention, "qk_rope_head_dim", None)
        self.v_head_dim = getattr(exp_config.model_config.attention, "v_head_dim", None)
        self.image_size = getattr(exp_config.model_config, "image_size", None)
        self.patch_size = getattr(exp_config.model_config, "patch_size", None)
        self.in_chans = getattr(exp_config.model_config, "in_chans", 3)
        self.num_classes = getattr(exp_config.model_config, "num_classes", 0)
        self.num_prefix_tokens = getattr(exp_config.model_config, "num_prefix_tokens", 0)
        self.num_patches = getattr(exp_config.model_config, "num_patches", 0)
        self.patch_dim = getattr(exp_config.model_config, "patch_dim", 0)
        self.swiglu_mlp = getattr(exp_config.model_config, "swiglu_mlp", False)

        self.moe_num_experts = int(getattr(exp_config.model_config, "num_experts", 1))
        self.moe_tile_rule = getattr(exp_config.model_config, "moe_tile_rule", None)

        self.moe_routing_policy = getattr(getattr(exp_config.model_config, "moe", None), "routing_policy",
                                          "multinomial_expected")
        if (str(self.model_type).lower() == "qwen3_moe"
                or (str(self.model_type).lower() in ("deepseek_v2", "deepseek_v3")
                    and int(getattr(getattr(exp_config.model_config, "moe", None), "num_experts", 1) or 1) > 1)):
            self.moe = exp_config.model_config.moe
            self.moe_source_collective_rule = exp_config.model_config.moe_source_collective_rule
            self.moe_source_replay_steps = exp_config.model_config.moe_source_replay_steps
            self.moe_grouped_traffic = exp_config.model_config.moe_grouped_traffic
        self.moe_top_k = int(getattr(exp_config.model_config, "top_k", 1))
        self.moe_intermediate_size = int(
            getattr(exp_config.model_config, "moe_intermediate_size", self.intermediate_size)
        )
        self.n_shared_experts = int(getattr(exp_config.model_config, "n_shared_experts", 0))
        self.expert_imbalance_factor = float(
            getattr(exp_config.model_config, "expert_imbalance_factor", 1.0)
        )
        self.moe_layer_freq = int(getattr(exp_config.model_config, "moe_layer_freq", 1))
        self.first_k_dense_replace = int(getattr(exp_config.model_config, "first_k_dense_replace", 0))
        self.moe_layer_mask = list(getattr(exp_config.model_config, "moe_layer_mask", []))
        self.use_moe = bool(getattr(exp_config.model_config, "use_moe", False))

        inference_cfg = getattr(exp_config, "inference_config", None)
        if str(self.run_type).lower() == "inference":
            if inference_cfg is None:
                raise ValueError("Inference configuration not found for inference run_type")
            self.inference_sample_every = inference_cfg.sample_every
            self.inference_decode_only = bool(getattr(inference_cfg, "decode_only", False))
        else:
            self.inference_sample_every = -1
            self.inference_decode_only = False

class NetworkModel:
    def __init__(self, hw_config, precision, kernel_overhead, roofline_cb, astra_policy: Optional[str] = None):
        self.hw_config = hw_config
        self.precision = precision
        self.O = kernel_overhead
        self._roofline = roofline_cb
        self._astra_policy = astra_policy or "analytical"

    def _astra_collective(self, kind: str, participants: int, size_bytes: int, axis: Optional[str] = None) -> float:
        part = int(participants)
        if part <= 1 or size_bytes <= 0:
            return 0.0
        byte_count = int(math.ceil(size_bytes))
        axes_filter = [str(axis).lower()] if axis else None
        hw = self.hw_config
        zero_axis = None
        from sim_settings import nccl_latency_rule
        if nccl_latency_rule(hw) in ("tuning_formula", "tuning_protocol"):
            from program.nccl_latency import dimension_params, formula_seconds
            params = dimension_params(hw, axis, part)
            formula_seconds(0, CollectiveType(kind), part, params["latency"], params["algorithms"][kind])
            zero_axis = params["axis"]
            axes_filter = [zero_axis]

        _, max_sec = run_cache_astrasim(
            hw,
            comm=kind,
            npus_count=part,
            size_bytes=byte_count,
            axes_filter=axes_filter,
            transform_2d_to_1d=True,
            **({"nccl_latency_zero_axis": zero_axis} if zero_axis is not None else {}),
        )
        return float(max_sec)

    def collective(
        self,
        *,
        kind: CollectiveType,
        size_bytes: float,
        participants: int,
        ib: float,
        ll: float,
        local_bytes: float = 0.0,
        local_ops: float = 0.0,
        debug_label: str = "",
        axis: Optional[str] = None,
    ) -> float:
        if participants is None:
            return 0.0

        part = int(participants)
        if part <= 1:
            return 0.0

        if size_bytes <= 0:
            return 0.0

        policy = getattr(self, "vllm_allreduce", None)
        if policy and kind is CollectiveType.ALL_REDUCE and axis == "tp":
            from program.vllm_allreduce import choice_for, custom_seconds
            choice = choice_for(policy, kind, part, size_bytes, axis)
            from program.allreduce_curve import measured_seconds
            measured = measured_seconds(policy, choice, part, size_bytes)
            if measured is not None:
                return measured
            if choice.algorithm.startswith("custom_"):
                return custom_seconds(choice, size_bytes, part, policy)
        curve = getattr(self, "allreduce_curve", None)
        if curve and not policy and axis in ("tp", "dp") and kind in (
                CollectiveType.ALL_REDUCE, CollectiveType.REDUCE_SCATTER, CollectiveType.ALL_GATHER):
            from program.vllm_allreduce import AllReduceChoice
            from program.allreduce_curve import measured_seconds
            measured = measured_seconds(curve, AllReduceChoice("nccl"), part, size_bytes, axis, kind)
            if measured is not None:
                return measured
        from sim_settings import nccl_latency_rule
        from program.collective_launch import is_nccl_call
        if (nccl_latency_rule(self.hw_config) == "tuning_protocol"
                and kind in (CollectiveType.ALL_REDUCE, CollectiveType.ALL_GATHER, CollectiveType.REDUCE_SCATTER)
                and is_nccl_call(kind, participants=part, size_bytes=size_bytes)):
            from program.nccl_latency import protocol_model
            model = protocol_model(self.hw_config, axis, kind, part, size_bytes)

            return model["transfer_seconds"]
        if nccl_latency_rule(self.hw_config) in ("tuning_formula", "tuning_protocol") and is_nccl_call(
                kind, participants=part, size_bytes=size_bytes):
            from program.nccl_latency import dimension_params, formula_seconds
            params = dimension_params(self.hw_config, axis, part)
            formula_seconds(0, kind, part, params["latency"], params["algorithms"][kind.value])
            ll = 0.0
        if policy and kind is CollectiveType.ALL_REDUCE and axis == "tp":
            if self._astra_policy not in {"hybrid", "full"}:
                return self._analytical_all_reduce(size_bytes, part, ib, ll, debug_label) + self.O

        if not isinstance(kind, CollectiveType):
            raise TypeError(f"collective kind must be CollectiveType (got {type(kind).__name__})")
        kind_value = kind.value
        collective_ops = {
            CollectiveType.ALL_REDUCE,
            CollectiveType.REDUCE_SCATTER,
            CollectiveType.ALL_GATHER,
            CollectiveType.ALL_TO_ALL,
        }

        network_bytes = float(math.ceil(size_bytes))
        local_bytes_int = int(math.ceil(local_bytes)) if local_bytes else 0

        if kind == CollectiveType.PIPELINE:
            network_time = self._analytical_point_to_point(network_bytes, ib, ll)
        elif self._astra_policy in {"hybrid", "full"}:
            if kind in collective_ops or kind == CollectiveType.PIPELINE:
                npus = 2 if kind == CollectiveType.PIPELINE else part
                axis_filter = axis if kind != CollectiveType.PIPELINE else None
                network_time = self._astra_collective(kind_value, npus, network_bytes, axis_filter)
            else:
                raise ValueError(f"Unsupported collective operation: {kind_value}")
        else:
            network_time = self._analytical_collective(
                kind=kind,
                size_bytes=network_bytes,
                participants=part,
                ib=ib,
                ll=ll,
                debug_label=debug_label,
            )

        local_time = 0.0

        overhead_kinds = {
            CollectiveType.ALL_REDUCE,
            CollectiveType.REDUCE_SCATTER,
            CollectiveType.ALL_GATHER,
        }
        overhead = self.O if (network_bytes and kind in overhead_kinds) else 0.0

        return network_time + local_time + overhead

    def _analytical_collective(
        self,
        *,
        kind: CollectiveType,
        size_bytes: float,
        participants: int,
        ib: float,
        ll: float,
        debug_label: str,
    ) -> float:
        if kind == CollectiveType.ALL_REDUCE:
            return self._analytical_all_reduce(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.REDUCE_SCATTER:
            return self._analytical_reduce_scatter(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.ALL_GATHER:
            return self._analytical_all_gather(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.ALL_TO_ALL:
            return self._analytical_all_to_all(size_bytes, participants, ib, ll, debug_label)
        if kind == CollectiveType.PIPELINE:
            return self._analytical_point_to_point(size_bytes, ib, ll)

        raise ValueError(f"Unsupported collective operation: {kind.value}")

    def _analytical_all_reduce(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        per_rank = size_bytes / participants
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",

            mem_level=0,
        )
        data_transfer = ((per_rank / ib) + mem_access + ll) * 2 * (participants - 1)
        prep_comp = per_rank
        prep_mem = int(math.ceil(3 * size_bytes / participants))

        data_prep = (
            self._roofline(prep_comp, prep_mem, name=f"{label}-prep", mem_level=0)
        ) * (participants - 1)
        return data_transfer + data_prep

    def _analytical_reduce_scatter(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        per_rank = size_bytes / participants
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",

            mem_level=0,
        )
        data_transfer = ((per_rank / ib) + mem_access + ll) * (participants - 1)
        prep_comp = per_rank
        prep_mem = int(math.ceil(3 * size_bytes / participants))
        data_prep = (
            self._roofline(prep_comp, prep_mem, name=f"{label}-prep", mem_level=0) + self.O
        ) * (participants - 1)
        return data_transfer + data_prep

    def _analytical_all_gather(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        mem_access = self._roofline(
            0,
            int(math.ceil(2 * size_bytes / participants)),
            name=f"{label}-mem",

            mem_level=0,
        )
        data_transfer = ((size_bytes / ib) + mem_access + ll) * (participants - 1)
        return data_transfer

    def _analytical_all_to_all(self, size_bytes, participants, ib, ll, label):
        if ib == 0:
            return float("inf")
        return ((size_bytes / participants) / ib + ll) * (participants - 1)

    def _analytical_point_to_point(self, size_bytes, ib, ll):
        if size_bytes <= 0:
            return 0.0
        if ib == 0:
            return float("inf")
        return size_bytes / ib + ll

def _is_attention_activity_name(name: str) -> bool:
    n = str(name or "").lower()
    return n.startswith(("flash_attention", "pointwise-scale-softmax", "pointwise-scale_softmax", "mla_attention",
                         "attention", "decode_attention"))

def _in_settings_layer(fn):
    @functools.wraps(fn)
    def method(self, *args, **kwargs):
        layer = self.__dict__.get("_settings_layer")
        if layer is None or _settings.active() is layer:
            return fn(self, *args, **kwargs)
        with _settings.use(layer):
            return fn(self, *args, **kwargs)
    method._in_settings_layer = True
    return method

class _SettingsScoped(type):
    def __new__(mcls, name, bases, namespace):
        for key, value in list(namespace.items()):
            if isinstance(value, types.FunctionType) and not (key.startswith("__") and key.endswith("__")) \
                    and not getattr(value, "_in_settings_layer", False):
                namespace[key] = _in_settings_layer(value)
        return super().__new__(mcls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        hw_config = kwargs["hw_config"] if "hw_config" in kwargs else args[0]
        model_config = kwargs["model_config"] if "model_config" in kwargs else (args[1] if len(args) > 1 else None)
        obj = cls.__new__(cls)
        obj._settings_layer = sim_settings.layer(hw_config, model_config)
        with _settings.use(obj._settings_layer):
            obj.__init__(*args, **kwargs)
        _settings.set_latest(obj._settings_layer)
        return obj

class TimeCalculation(metaclass=_SettingsScoped):
    def __init__(self, hw_config, model_config, mode, *, astra_policy_override: Optional[str] = None):
        self.O = hw_config.sw_config.kernel_launch_overhead
        self.precision = hw_config.sw_config.precision
        self.precision_bytes = self.precision.activations
        self.h2d_bandwidth = getattr(hw_config.sw_config, "h2d_bandwidth", -1)
        self.zero_stage = getattr(hw_config.sw_config, "dp_zero_stage", 0)
        self.full_recomputation = getattr(hw_config.sw_config, "full_recomputation", False)
        self.dp_microbatch = getattr(hw_config.sw_config, "dp_microbatch", "every_mb")
        self.pipeline_interleave = int(getattr(hw_config.sw_config, "pipeline_interleave", 1) or 1)
        self.pipeline_schedule = str(getattr(hw_config.sw_config, "pipeline_schedule", "gpipe") or "gpipe")
        self.grad_acc_overhead = float(getattr(hw_config.sw_config, "grad_acc_overhead", 0.0) or 0.0)
        self.attached = True

        run_type = "training"
        if str(mode).upper() in {"LLM", "VIT"}:
            model_cfg = getattr(model_config, "model_config", None)
            run_type = str(getattr(model_cfg, "run_type", "training")).lower()
        self.run_type = run_type
        if getattr(self.precision, "linear_weights", None) is not None and run_type != "inference":
            raise NotImplementedError("precision.linear_weights (weight-only quantization) is an inference option")

        axis_sizes = _active_axis_sizes(hw_config, run_type)
        setattr(hw_config, "active_parallelism", dict(axis_sizes))
        setattr(hw_config, "active_run_type", run_type)
        refreshed_layout = _refresh_network_layout(hw_config, axis_sizes)
        if refreshed_layout is not None:
            hw_config.network_layout = refreshed_layout

        if "_settings_layer" not in self.__dict__:
            self._settings_layer = sim_settings.layer(hw_config, model_config)
        with _settings.use(self._settings_layer):
            self.resolved_settings = sim_settings.resolution(hw_config, model_config)

        self.hw_config = hw_config

        self.l2_bypass = sim_settings.l2_bypass(hw_config)
        if self.l2_bypass:
            if str(mode).upper() != "LLM":
                raise NotImplementedError(f"tech_param.SRAM-L2.l2_bypass is described for LLM runs only (mode {mode!r})")
            self.resolved_settings["l2_bypass"] = True
        self.core = Core(hw_config)
        if self.core.device_fixed_overhead_cycles is not None:
            if self.core.operating_freq <= 0.0:
                raise ValueError("fixed overhead cycles require a positive core frequency")
            self.O = self.core.device_fixed_overhead_cycles / self.core.operating_freq
        self.stream_launch_overhead_s = self.O
        self.resolved_settings["kernel_launch"]["stream_overhead_s"] = self.O
        self.th_matrix = self.core.get_throughput("matrix")
        self.th_vector = self.core.get_throughput("vector")
        self.th = self.th_matrix
        self.FMA_dims = self.core.FMA_dims
        self.dataflow = self.core.dataflow

        self._local_op_backend_name = str(
            getattr(hw_config.sw_config, "local_op_backend", "extended_roofline")
        ).lower()
        self.resolved_settings["gemm_timing_backend"] = str(
            getattr(hw_config.sw_config, "gemm_timing_backend", "extended_roofline")
        ).lower()
        self._extended_gemm_backend = None
        self._opmodel_backend = None
        self._gemm_activity_backend = None

        self.native_gemm_timing: Dict[str, int] = {}

        self._inference_kernel_energy_records = []

        self.memory_hierarchy = MemoryHierarchy(hw_config, core=self.core)
        self.num_levels = self.memory_hierarchy.num_levels
        self.mem_layer = self.memory_hierarchy.mem_layer
        self.tile_space = None
        self.H2Dbw = self.h2d_bandwidth

        if self._local_op_backend_name == "opmodel" or getattr(hw_config, "counts_only", False):
            from opmodel_ops import OpModelEstimator
            self._opmodel_backend = OpModelEstimator(
                hw_config=hw_config, core=self.core, mem_layer=self.mem_layer,
                precision_bytes=self.precision_bytes, run_type=run_type,
            )
        else:
            if self.resolved_settings["gemm_timing_backend"] == "extended_roofline":
                import extended_timing
                self._extended_gemm_backend = extended_timing.create(hw_config)

            from opmodel_ops import OpModelEstimator
            self._gemm_activity_backend = OpModelEstimator(
                hw_config=hw_config, core=self.core, mem_layer=self.mem_layer,
                precision_bytes=self.precision_bytes, run_type=run_type,
            )

        level = 0
        mem_config = hw_config.memory_hierarchy.mem_hr[level]
        self.DRAM = DRAM(hw_config, mem_config, level, self.core)
        self.network = Network(hw_config)

        self.links = {
            "dp": LinkInfo(*self.network.get_link("dp")),
            "ep": LinkInfo(*self.network.get_link("ep")),
            "pp": LinkInfo(*self.network.get_link("pp")),
            "tp": LinkInfo(*self.network.get_link("tp")),
            "cp": LinkInfo(*self.network.get_link("cp")),
        }

        par = Parallelism(hw_config, run_type=run_type)
        self.mode = mode
        self.pp = par.pp
        self.mb = par.mb
        self.dp = par.dp
        self.ep = par.ep

        self.tp = par.tp
        self.cp = par.cp

        self.tp_sp = par.tp_sp
        self.tp_ep = par.tp_ep
        self.replica_count = par.replica_count
        self.moe_dp = par.moe_dp
        self.num_workers = self._derive_num_workers(hw_config)

        self.tot_flop = 0
        self.tot_mem = 0
        self.tot_time = 0
        self.debug = False

        default_policy = 'analytical'
        eb = getattr(hw_config, "execution_backend", None)
        if eb and getattr(eb, "model", "analytical") == "astrasim":
            default_policy = 'hybrid'
        self._astra_policy = astra_policy_override or default_policy

        self.network_model = NetworkModel(
            hw_config,
            self.precision_bytes,
            self.O,
            self.roofline,
            astra_policy=self._astra_policy,
        )

        self._gemm_activity_by_name: Dict[str, OperationActivity] = {}
        self._roofline_activity_by_name: Dict[str, OperationActivity] = {}

        model_class = self.get_model_class(mode)
        self.model = model_class(model_config)

        if mode == "GEMM":
            self.M = self.model.M
            self.K = self.model.K
            self.N = self.model.N
            self.gemm_shard_axis = self.model.gemm_shard_axis

        if str(mode).upper() in {"LLM", "VIT"}:
            self.global_batch_size = self.model.global_batch_size
            self.gradient_accumulation_steps = self.model.gradient_accumulation_steps
            self.batch_size = self.global_batch_size // self.gradient_accumulation_steps
            self.vocab_size = self.model.vocab_size
            self.num_layers = self.model.num_layers
            self.hidden_dim = self.model.hidden_dim
            self.seq_len = self.model.seq_len
            self.num_heads = self.model.num_heads
            self.head_dim = (
                int(self.model.head_dim)
                if getattr(self.model, "head_dim", None) is not None
                else self.hidden_dim // self.num_heads
            )
            self.intermediate_size = self.model.intermediate_size
            self.n_tokens = self.model.n_tokens
            if self.run_type == "inference":
                dp_dense = 1
            else:
                dp_dense = self.dp * self.ep if bool(getattr(self.model, "use_moe", False)) else self.dp
            self.mini_batch = math.ceil(self.batch_size / dp_dense)
            self.micro_batch = math.ceil(self.mini_batch / self.mb) if self.pp > 1 else self.mini_batch
            self.attention_type = self.model.attention_type
            self.flash_attention = getattr(self.model, "use_flashattention", False)
            self.use_rope = bool(getattr(self.model, "use_rope", False))
            self.implementation = config._coerce_implementation(_settings.get("RAPID_IMPLEMENTATION"))
            self.implementation_options = {
                key: _settings.get(sim_settings.option_setting(key)) for key in sim_settings.IMPLEMENTATION_OPTIONS
            }
            sim_settings.validate_implementation_options(self.implementation_options)
            self.kernel_description_resolved = getattr(self.model, "kernel_description_resolved", False)
            self.engine_settings = getattr(self.model, "engine", getattr(self, "engine_settings", None))
            if getattr(hw_config, "counts_only", False):
                self.engine_settings = self.model.engine_settings
            self.kv_heads = self.model.kv_heads if hasattr(self.model, "kv_heads") else self.num_heads
            self.attention_tile_size = getattr(self.model, "attention_tile_size", None)
            self.kv_lora_rank = getattr(self.model, "kv_lora_rank", None)
            self.q_lora_rank = getattr(self.model, "q_lora_rank", None)
            self.qk_nope_head_dim = getattr(self.model, "qk_nope_head_dim", None)
            self.qk_rope_head_dim = getattr(self.model, "qk_rope_head_dim", None)
            self.v_head_dim = getattr(self.model, "v_head_dim", None)
            self.image_size = getattr(self.model, "image_size", None)
            self.patch_size = getattr(self.model, "patch_size", None)
            self.in_chans = getattr(self.model, "in_chans", 3)
            self.num_classes = getattr(self.model, "num_classes", 0)
            self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
            self.num_patches = getattr(self.model, "num_patches", 0)
            self.patch_dim = getattr(self.model, "patch_dim", 0)
            self.swiglu_mlp = getattr(self.model, "swiglu_mlp", False)
            raw_num_experts = getattr(self.model, "moe_num_experts", 1)
            raw_top_k = getattr(self.model, "moe_top_k", 1)
            self.moe_num_experts = max(1, int(raw_num_experts))
            self.moe_top_k = max(1, int(raw_top_k))
            self.moe_intermediate_size = int(
                getattr(self.model, "moe_intermediate_size", self.intermediate_size)
            )
            self.n_shared_experts = int(getattr(self.model, "n_shared_experts", 0))
            self.expert_imbalance_factor = float(
                getattr(self.model, "expert_imbalance_factor", 1.0)
            )
            self.moe_layer_freq = int(getattr(self.model, "moe_layer_freq", 1))
            self.first_k_dense_replace = int(getattr(self.model, "first_k_dense_replace", 0))
            self.moe_layer_mask = list(getattr(self.model, "moe_layer_mask", []))
            self.use_moe = bool(getattr(self.model, "use_moe", False))
            self.num_moe_layers = sum(self.moe_layer_mask)

    def _derive_num_workers(self, hw_config) -> int:
        layout = getattr(hw_config, "network_layout", None)
        if layout and getattr(layout, "dimensions", None):
            total = 1
            for dim in layout.dimensions:
                try:
                    size = int(dim.size)
                except (TypeError, ValueError):
                    size = 1
                if size < 1:
                    size = 1
                total *= size
            if total >= 1:
                return int(total)

        active = getattr(hw_config, "active_parallelism", None)
        if isinstance(active, dict) and active:
            total = 1
            for axis in ("dp", "pp", "tp", "cp", "ep"):
                total *= max(1, int(active.get(axis, 1) or 1))
            return max(1, total)

        sch = getattr(hw_config, "sch_config", None)
        if sch is None:
            return 1
        axis_sizes = _active_axis_sizes(hw_config, getattr(self, "run_type", "training"))
        total = 1
        for axis in ("dp", "pp", "tp", "cp", "ep"):
            total *= max(1, int(axis_sizes.get(axis, 1) or 1))
        return max(1, total)

    def get_model_class(self, model_type):
        model_classes = {
            "GEMM": model_GEMM,
            "LLM": model_LLM,
            "VIT": model_LLM,
        }
        key = str(model_type).upper()
        if key not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model_classes[key]

    def roofline(self, flop, mem_access_, name="", util=1, info=False, mem_level=None, flashattn_enable=False, activity_kind="vector", compute_unit=None, active_bundles=None):
        if flashattn_enable:
            raise ValueError("flashattn_enable is obsolete; use the flash attention rebuild")

        if isinstance(mem_access_, (int, float)):
            mem_access = [int(mem_access_)]
            if mem_level is None:
                raise ValueError(
                    f"roofline({name!r}): a scalar mem_access needs an explicit "
                    f"mem_level (0 = fastest .. {self.num_levels - 1} = DRAM). "
                    "Pass mem_level=self.num_levels-1 for anything that streams "
                    "a full tensor from memory."
                )
        elif isinstance(mem_access_, list):
            mem_access = mem_access_
        else:
            print(mem_access_)
            print("mem_access_ should be integer or list, wrong input", flush=True)
            sys.exit(0)

        if compute_unit is None:
            compute_unit = "vector" if activity_kind == "vector" else "matrix"
        active_fraction = 1.0
        if active_bundles is not None:
            active_fraction = min(
                1.0,
                max(1, int(active_bundles)) / max(1, int(self.core.num_bundle)),
            )
        throughput = self.core.get_throughput(compute_unit) * util * active_fraction

        if mem_level is not None:
            levels_to_compute = [(mem_level, mem_access[0])]
        else:
            num_level = len(mem_access)
            try:
                assert mem_access[num_level - 1] > 0, f"mem_access: {mem_access_}"
            except Exception as e:
                print(
                    "{}: Number of accesses to the last level of memory hierarchy cannot be zero:\n {}".format(
                        name, e
                    ),
                    flush=True,
                )
                sys.exit(0)
            levels_to_compute = [(i, mem_access[i]) for i in range(num_level)]

        timed_levels = levels_to_compute
        if getattr(self, "l2_bypass", False) and self.num_levels >= 4:
            timed_levels = self._l2_bypass_levels(levels_to_compute)
        times = []
        for level_idx, num_mem in timed_levels:
            mem_bw = self.mem_layer[level_idx].get_throughput() * (
                1.0 if level_idx == self.num_levels - 1 else active_fraction
            )
            mem_latency = self.mem_layer[level_idx].get_latency()
            inflection_point = float("inf") if mem_bw == 0 else throughput / mem_bw

            comp_int = float("inf") if num_mem == 0 else flop / num_mem
            if comp_int < inflection_point:
                level_time = (float("inf") if (mem_bw == 0 or num_mem == 0) else (num_mem / mem_bw)) + mem_latency
            else:
                level_time = float("inf") if (throughput == 0) else (flop / throughput)

            times.append(level_time)

        max_time = max(times)

        if name and activity_kind is not None:
            traffic = [MemoryTraffic() for _ in range(max(4, self.num_levels))]
            for level_idx, num_mem in levels_to_compute:
                traffic[level_idx] = traffic[level_idx] + MemoryTraffic(
                    read_bytes=float(num_mem)
                )

            hbm_level = self.num_levels - 1
            if (self.num_levels >= 4 and traffic[2] == MemoryTraffic() and traffic[hbm_level] != MemoryTraffic()
                    and not _is_attention_activity_name(name)):
                traffic[2] = traffic[hbm_level]
            activity = OperationActivity(
                vector_flops=float(flop) if activity_kind == "vector" else 0.0,
                useful_tensor_flops=float(flop) if activity_kind == "tensor" else 0.0,
                issued_tensor_flops=float(flop) if activity_kind == "tensor" else 0.0,
                register=traffic[0],
                smem=traffic[1],
                l2=traffic[2],
                hbm=traffic[self.num_levels - 1],
            )
            self._roofline_activity_by_name[name] = (
                self._roofline_activity_by_name.get(name, ZERO_ACTIVITY) + activity
            )

        return max_time

    def _l2_bypass_levels(self, levels):
        l2_level, dram_level = 2, self.num_levels - 1
        l2_bytes = sum(num for level, num in levels if level == l2_level)
        timed = [(level, num) for level, num in levels if level not in (l2_level, dram_level)]
        dram_bytes = sum(num for level, num in levels if level == dram_level)
        if any(level in (l2_level, dram_level) for level, _ in levels):
            timed.append((dram_level, l2_bytes if l2_bytes else dram_bytes))
        return timed

    def kernel_min_seconds(self):
        opts = getattr(self, "implementation_options", None) or {}
        if opts.get("kernel_min_duration", "off") == "off":
            return None

        return float(self.stream_launch_overhead_s)

    def kernel_floor(self, seconds, launches=1):
        floor = self.kernel_min_seconds()
        if floor is None:
            return seconds
        return max(seconds, launches * floor)

    def vector_roofline(
        self,
        flop,
        mem_access_,
        name="",
        util=1,
        info=False,
        mem_level=None,
        flashattn_enable=False,
        active_bundles=None,
    ):
        return self.roofline(
            flop,
            mem_access_,
            name=name,
            util=util,
            info=info,
            mem_level=mem_level,
            flashattn_enable=flashattn_enable,
            compute_unit="vector",
            active_bundles=active_bundles,
        )

    def matrix_roofline(
        self,
        flop,
        mem_access_,
        name="",
        util=1,
        info=False,
        mem_level=None,
        flashattn_enable=False,
        active_bundles=None,
    ):
        return self.roofline(
            flop,
            mem_access_,
            name=name,
            util=util,
            info=info,
            mem_level=mem_level,
            flashattn_enable=flashattn_enable,
            compute_unit="matrix",
            active_bundles=active_bundles,
        )

    def get_opmodel_softmax(
        self, *, rows: int, row_size: int, name: str, phase: str, dtype_bytes=None
    ):
        backend = self._opmodel_backend or self._gemm_activity_backend
        if backend is None:
            raise RuntimeError("softmax op-model requires an opmodel timing or activity backend")
        extra = {} if dtype_bytes is None else {"dtype_bytes": float(dtype_bytes)}
        result = backend.predict_softmax(
            rows=int(rows), row_size=int(row_size), name=name, phase=phase, **extra
        )
        self._record_inference_kernel_energy(result, name)
        memory = tuple(float(value or 0.0) for value in result.memory_accesses)

        def traffic(level: int) -> MemoryTraffic:
            value = memory[level] if level < len(memory) else 0.0
            return MemoryTraffic(read_bytes=value, write_bytes=0.0)

        activity = OperationActivity(
            vector_flops=float(result.flops),
            register=traffic(0),
            smem=traffic(1),
            l2=traffic(2),
            hbm=traffic(3),
        )
        if name:
            self._gemm_activity_by_name[name] = (
                self._gemm_activity_by_name.get(name, ZERO_ACTIVITY) + activity
            )
        return result

    def _inference_kernel_energy_marker(self) -> int:
        return len(self._inference_kernel_energy_records)

    def _inference_kernel_energy_since(self, marker: int):
        return list(self._inference_kernel_energy_records[int(marker):])

    def _record_inference_kernel_energy(self, result, name: str) -> None:
        breakdown = dict(getattr(result, "energy_breakdown", {}) or {})
        self._inference_kernel_energy_records.append(
            {
                "name": str(name or "opmodel"),
                "compute_unit": "tensor",
                "kernel_static_j": max(
                    0.0, float(breakdown.get("static", 0.0) or 0.0)
                ),
                "compute_j": max(0.0, float(breakdown.get("compute", 0.0) or 0.0)),
                "memory_j_by_level": {
                    "L0": max(0.0, float(breakdown.get("register", 0.0) or 0.0)),
                    "L1": max(0.0, float(breakdown.get("sram", 0.0) or 0.0)),
                    "L2": max(0.0, float(breakdown.get("l2", 0.0) or 0.0)),
                    f"L{self.num_levels - 1}": max(
                        0.0, float(breakdown.get("hbm", 0.0) or 0.0)
                    ),
                },
            }
        )

    def activity_model_provenance(self) -> Dict[str, str]:
        backend = self._opmodel_backend or self._gemm_activity_backend
        if backend is not None:
            return dict(backend.provenance)
        return {"backend": "timing"}

    @staticmethod
    def _opmodel_gemm_activity(
        result,
        *,
        dim1: int,
        dim2: int,
        dim3: int,
        batch: int,
    ) -> OperationActivity:
        memory = tuple(float(value or 0.0) for value in result.memory_accesses)

        def traffic(level: int) -> MemoryTraffic:
            value = memory[level] if level < len(memory) else 0.0
            return MemoryTraffic(read_bytes=value, write_bytes=0.0)

        useful = float(2 * int(batch) * int(dim1) * int(dim2) * int(dim3))
        if dict(getattr(result, "diagnostics", {}) or {}).get("gemv"):
            return OperationActivity(
                useful_tensor_flops=0.0,
                issued_tensor_flops=0.0,
                vector_flops=useful,
                register=traffic(0),
                smem=traffic(1),
                l2=traffic(2),
                hbm=traffic(3),
            )
        split_k = dict(getattr(result, "diagnostics", {}) or {}).get("split_k", {})
        split_scale = useful / float(split_k["tensor_flops"]) if split_k else 1.0
        return OperationActivity(
            useful_tensor_flops=useful,
            issued_tensor_flops=(float(split_k["issued_tensor_flops"]) * split_scale
                                 if split_k else float(result.flops)),
            vector_flops=float(split_k.get("reduction_flops", 0)) * split_scale,
            register=traffic(0),
            smem=traffic(1),
            l2=traffic(2),
            hbm=traffic(3),
        )

    def get_gemm_time(self, dim1, dim2, dim3, name="",
                    flashattn_enable=False, disable_overhead=False, read_bytes_l2=0, write_bytes_l2=0, original=False, batch=1,
                    activity_batch=None, gemm_role=None, weight_bytes=None):
        if flashattn_enable:
            raise ValueError("flashattn_enable is obsolete; use the flash attention rebuild")
        batch = max(1, int(math.ceil(float(batch or 1))))
        activity_batch = (
            batch
            if activity_batch is None
            else max(1, int(math.ceil(float(activity_batch))))
        )
        gemm_backend = self._opmodel_backend
        unified_gemm = getattr(self.hw_config.sw_config, "gemm_timing_backend", "extended_roofline") == "opmodel"
        if unified_gemm:
            gemm_backend = gemm_backend or self._gemm_activity_backend
        wb = {} if weight_bytes is None else {"weight_bytes": float(weight_bytes)}
        if wb and (not unified_gemm or gemm_backend is None):
            raise NotImplementedError("weight-only GEMM operand bytes need sw_param.gemm_timing_backend: opmodel")
        if gemm_backend is not None:
            result = gemm_backend.predict_batched_gemm(
                dim1=int(dim1), dim2=int(dim2), dim3=int(dim3),
                batch=batch, name=name, allow_gemv=not getattr(self, "_gemv_disabled", False),
                **({"gemm_role": gemm_role} if gemm_role is not None else {}), **wb,
            )
            self._last_gemm_diagnostics = dict(getattr(result, "diagnostics", {}) or {})
            memory = tuple(float(value or 0.0) for value in result.memory_accesses)

            scale = activity_batch / batch
            activity_result = replace(result, flops=result.flops * scale,
                memory_accesses=tuple(value * scale for value in result.memory_accesses),
                energy_breakdown={key: value * scale for key, value in result.energy_breakdown.items()})
            if not unified_gemm and activity_batch != batch:
                activity_result = gemm_backend.predict_batched_gemm(
                    dim1=int(dim1), dim2=int(dim2), dim3=int(dim3), batch=activity_batch,
                    name=name, allow_gemv=not getattr(self, "_gemv_disabled", False),
                    **({"gemm_role": gemm_role} if gemm_role is not None else {}),
                )
            self._record_inference_kernel_energy(activity_result, name)
            self._last_gemm_activity = self._opmodel_gemm_activity(
                activity_result,
                dim1=int(dim1),
                dim2=int(dim2),
                dim3=int(dim3),
                batch=activity_batch,
            )
            if name:
                self._gemm_activity_by_name[name] = (
                    self._gemm_activity_by_name.get(name, ZERO_ACTIVITY)
                    + self._last_gemm_activity
                )
            compute = gemm_backend.hardware.compute
            fixed = float(compute.device_fixed_overhead_cycles or 0) / float(compute.clock_hz)
            launches = int(dict(getattr(result, "diagnostics", {}) or {}).get("split_k", {}).get("kernel_launches", 1))
            if unified_gemm:
                latency = (result.latency_s - launches * fixed) + (0.0 if disable_overhead else launches * self.O)
                if not wb:
                    latency = self._gemv_stream_time(
                        latency, dim1, dim2, dim3, name=name, batch=batch,
                        flashattn_enable=flashattn_enable, disable_overhead=disable_overhead,
                    )
                if not disable_overhead:
                    latency = self.kernel_floor(latency, launches)
            else:
                latency = result.latency_s
                if getattr(self, "_decode_graph_pricing", False):
                    latency += launches * (self.O - fixed)
            return latency, 0, result.tile_dims, memory

        best_time = float("inf")
        best_choice = None
        best_mem_access = None
        best_rw_access = None
        best_gemm = None
        best_roofline_inputs = None
        best_metric = float("inf")

        selection_cache = self.__dict__.setdefault("_gemm_selection_cache", {})
        selection_key = None
        if not self.debug:
            try:
                selection_key = pickle.dumps(
                    (dim1, dim2, dim3, self.precision_bytes, self.num_levels, disable_overhead,
                     self.O, self.core, self.mem_layer), protocol=4)
            except (TypeError, pickle.PickleError, AttributeError):
                pass
        cached = selection_cache.get(selection_key) if selection_key is not None else None
        if cached is not None:
            (best_time, best_choice, best_mem_access, best_rw_access,
             best_gemm, best_roofline_inputs) = cached

        candidates = () if cached is not None else TiledGEMM.enumerate_candidates(self.core, self.mem_layer, dim1, dim2, dim3, self.precision_bytes, original=False)
        for gemm in candidates:
            if self.debug:
                print("===============================================================")
                print(f"inner_code: {gemm._inner_code}")
                print("===============================================================")

            GEMM_flop = gemm.GEMM_flop
            rw_accesses = gemm.mem_accesses

            mem_access = rw_accesses.totals()

            mem_access_per_sm = list(mem_access)
            reuse_M = (dim1 + gemm.l2_M - 1) // gemm.l2_M
            reuse_K = (dim2 + gemm.l2_K - 1) // gemm.l2_K
            reuse_N = (dim3 + gemm.l2_N - 1) // gemm.l2_N
            eff_sm = min(self.core.num_bundle, reuse_M * reuse_K * reuse_N)
            if eff_sm > 0:
                mem_access_per_sm[1] = mem_access_per_sm[1] / eff_sm

            sm_util = 1
            gemm_waves = self.mem_layer[1].calc_waves_per_sm(dim1, dim2, dim3, gemm.l2_M, gemm.l2_K, gemm.l2_N)
            if gemm_waves != -1:
                full_waves = math.floor(gemm_waves)
                partial_wave = gemm_waves - full_waves
                sm_util = (full_waves + partial_wave * partial_wave) / gemm_waves

            GEMM_time = self.roofline(
                GEMM_flop,
                mem_access_per_sm,
                name=name,
                util=sm_util,
                flashattn_enable=flashattn_enable,
                compute_unit="matrix",
                activity_kind=None,
            )
            if not disable_overhead:
                GEMM_time = GEMM_time + self.O

            tile_dims = (
                (gemm.l0_M, gemm.l0_K, gemm.l0_N),
                (gemm.l1_M, gemm.l1_K, gemm.l1_N),
                (gemm.l2_M, gemm.l2_K, gemm.l2_N),
            )
            key = (gemm._inner_code, tile_dims)

            metric = math.hypot(mem_access[3], mem_access[2])

            if (GEMM_time < best_time) or (GEMM_time == best_time and metric < best_metric):
                best_time = GEMM_time
                best_choice = key
                best_mem_access = mem_access
                best_rw_access = rw_accesses
                best_gemm = gemm
                best_metric = metric
                best_roofline_inputs = (GEMM_flop, mem_access_per_sm, sm_util)

        if cached is None and selection_key is not None and best_choice is not None:
            if len(selection_cache) >= 256:
                selection_cache.clear()
            selection_cache[selection_key] = (best_time, best_choice, best_mem_access,
                best_rw_access, best_gemm, best_roofline_inputs)

        mem_access = best_mem_access  # type: ignore

        if self.debug:
            print(repr(best_gemm))
            print(
                f"{name}: Best Time: {best_time * 1e3:,} ms, Best Inner: {best_choice[0]}, Best Tile: {best_choice[1]}\n"
            )

        best_inner_code = best_choice[0]  # type: ignore[index]
        best_tile_dims = best_choice[1]  # type: ignore[index]

        def _traffic(level: int) -> MemoryTraffic:
            if best_rw_access is None or level >= len(best_rw_access.reads):
                return MemoryTraffic()
            return MemoryTraffic(
                best_rw_access.reads[level], best_rw_access.writes[level]
            )

        self._last_gemm_activity = OperationActivity(
            useful_tensor_flops=float(best_gemm.GEMM_flop),
            issued_tensor_flops=float(best_gemm.GEMM_flop),
            register=_traffic(0),
            smem=_traffic(1),
            l2=_traffic(2),
            hbm=_traffic(3),
        ).scaled(activity_batch) if best_gemm is not None else ZERO_ACTIVITY

        backend = getattr(self, "_extended_gemm_backend", None)
        if backend is not None:
            result = backend.gemm_timing_result(
                dim1, dim2, dim3, self.precision_bytes, batch=batch,
            )
            if result is None:
                why = getattr(backend, "last_fallback_reason", "unknown")
                self.native_gemm_timing[why] = self.native_gemm_timing.get(why, 0) + 1
            if result is not None:
                util = max(float(getattr(self.core, "util", 1.0) or 1.0), 1e-9)
                best_time = result.latency_s / util
                self._last_gemm_activity = result.activity.scaled(
                    activity_batch / batch
                )
                if not disable_overhead:
                    best_time = best_time + self.O

        best_time = self._gemv_stream_time(
            best_time, dim1, dim2, dim3, name=name, batch=batch,
            flashattn_enable=flashattn_enable, disable_overhead=disable_overhead,
        )

        if self._gemm_activity_backend is not None:
            activity_profile = self._gemm_activity_backend.predict_batched_gemm(
                dim1=int(dim1), dim2=int(dim2), dim3=int(dim3),
                batch=activity_batch, name=name, allow_gemv=not getattr(self, "_gemv_disabled", False),
            )
            self._record_inference_kernel_energy(activity_profile, name)
            self._last_gemm_activity = self._opmodel_gemm_activity(
                activity_profile,
                dim1=int(dim1),
                dim2=int(dim2),
                dim3=int(dim3),
                batch=activity_batch,
            )

        if name:
            self._gemm_activity_by_name[name] = (
                self._gemm_activity_by_name.get(name, ZERO_ACTIVITY)
                + self._last_gemm_activity
            )

        return best_time, best_inner_code, best_tile_dims, mem_access

    def _gemv_stream_time(self, best_time, dim1, dim2, dim3, *, name, batch, flashattn_enable, disable_overhead):
        stream_bw = float(_settings.get("OPMODEL_GEMV_STREAM_BW"))
        ideal = bool(_settings.get("OPMODEL_GEMV_IDEAL_ROOFLINE"))
        stream_hbm = bool(_settings.get("OPMODEL_GEMV_STREAM_HBM"))
        if (stream_bw > 0) + ideal + stream_hbm > 1:
            raise ValueError("GEMV stream_bw, stream_hbm and ideal_roofline are mutually exclusive")
        if (stream_bw <= 0 and not ideal and not stream_hbm) or flashattn_enable or getattr(self, "_gemv_disabled", False):
            return best_time
        if "attention" in str(name).lower():
            return best_time
        from opmodel_ops import OpModelEstimator
        b = max(1, int(math.ceil(float(batch or 1))))
        backend = self._opmodel_backend or self._gemm_activity_backend
        architecture = getattr(getattr(backend, "hardware", None), "architecture", "sm80") if ideal else "sm80"
        if not OpModelEstimator._is_cublas_gemv(dim1=int(dim1), dim3=int(dim3), batch=b, weight_rank=2, name=str(name), architecture=architecture):
            return best_time
        if ideal and _settings.get("OPMODEL_GEMV_RULE") != "cublas":
            return best_time
        p = float(self.precision_bytes)
        stream_bytes = b * (float(dim2) * float(dim3) + float(dim1) * float(dim2) + float(dim1) * float(dim3)) * p
        if ideal:
            stream_bw = float(self.mem_layer[-1].get_throughput())
            vector_rate = float(self.core.get_throughput("vector"))
            if stream_bw <= 0 or vector_rate <= 0:
                raise ValueError("ideal GEMV roofline needs positive HBM and vector rates")
            work = 2.0 * b * float(dim1) * float(dim2) * float(dim3)
            return (0.0 if disable_overhead else float(self.O)) + max(stream_bytes / stream_bw, work / vector_rate)
        if stream_hbm:
            stream_bw = float(self.mem_layer[-1].get_throughput())
            if stream_bw <= 0:
                raise ValueError("GEMV stream_hbm needs a positive DRAM rate")
        return (0.0 if disable_overhead else float(self.O)) + stream_bytes / stream_bw

    def get_gemm_timing_result(
        self, dim1, dim2, dim3, name="", flashattn_enable=False,
        disable_overhead=False, read_bytes_l2=0, write_bytes_l2=0, original=False,
    ) -> OperationTimingResult:
        latency, _, _, _ = self.get_gemm_time(
            dim1, dim2, dim3, name=name,
            flashattn_enable=flashattn_enable,
            disable_overhead=disable_overhead,
            read_bytes_l2=read_bytes_l2,
            write_bytes_l2=write_bytes_l2,
            original=original,
        )
        return OperationTimingResult(
            latency_s=latency,
            activity=getattr(self, "_last_gemm_activity", ZERO_ACTIVITY),
        )

    def get_tile_size(self, lid):
        memory = self.mem_layer[lid]
        memory.calc_tile_dim()
        tile_dim = memory.get_tile_dim()
        return tile_dim, tile_dim, tile_dim

    def get_num_accesses(self, level, dim1, dim2, dim3, tile_dim, num_repeat, name, r):
        tile1, tile2, tile3 = tile_dim

        orig_size = tile1 * tile2 + tile1 * tile3 + tile2 * tile3
        short_tile_cond = [0, 0, 0]

        if tile1 > dim1:
            tile1 = dim1
            short_tile_cond[0] = 1
        if tile2 > dim2:
            tile2 = dim2
            short_tile_cond[1] = 1
        if tile3 > dim3:
            tile3 = dim3
            short_tile_cond[2] = 1

        if short_tile_cond[2] == 0 and (short_tile_cond[0] | short_tile_cond[1]) == 1:
            if level <= 1:
                tile3 = math.floor((orig_size - tile1 * tile2) / (tile1 + tile2))
            else:
                tile3 = math.floor((orig_size - tile1 * tile2) / tile2)
            if tile3 > dim3:
                tile3 = dim3

        elif short_tile_cond[0] == 0 and (short_tile_cond[1] | short_tile_cond[2]) == 1:
            if level <= 1:
                tile1 = math.floor((orig_size - tile3 * tile2) / (tile3 + tile2))
            else:
                tile1 = math.floor((orig_size - tile3 * tile2) / tile2)
            if tile1 > dim1:
                tile1 = dim1
        elif short_tile_cond[1] == 0 and (short_tile_cond[0] & short_tile_cond[2]) == 1:
            if level <= 1:
                tile2 = math.floor((orig_size - tile3 * tile1) / (tile3 + tile1))
            else:
                tile2 = math.floor((orig_size) / (tile1 + tile3))
            if tile2 > dim2:
                tile2 = dim2

        reload_A = 1
        reload_B = 1
        reload_C = 1

        if tile1 > 0 and tile2 > 0 and tile3 > 0:
            reload_A = math.ceil(dim3 / tile3)
            reload_B = math.ceil(dim1 / tile1)

            reload_C = 1 if level > 1 else math.ceil(dim2 / tile2)

        num_repeat = r[0] * r[1] * r[2]

        if level == 2:
            num_mem = (
                num_repeat * (dim1 * dim2 * reload_A + dim2 * dim3 * reload_B)
                + r[0] * dim1 * r[2] * dim3 * reload_C
            ) * self.precision_bytes
        else:
            num_mem = (
                num_repeat
                * (
                    dim1 * dim2 * reload_A
                    + dim2 * dim3 * reload_B
                    + dim1 * dim3 * reload_C
                )
                * self.precision_bytes
            )

        if self.debug:
            print(name)
            print(
                "Matrix dimension at Level {}: {:,} x {:,} x {:,}".format(
                    level, dim1, dim2, dim3
                )
            )
            print(
                "Tile dimension at Level {}: {:,} x {:,} x {:,}".format(
                    level - 1, tile1, tile2, tile3
                )
            )
            print(
                "reload_A: {}, reload_B: {}, reload_C: {}".format(
                    reload_A, reload_B, reload_C
                )
            )
            print("num_repeat: {}".format(num_repeat))
            print("Bytes Accessed: {:,}".format(num_mem))
            print("")

        return num_mem, tile1, tile2, tile3

    def GEMM(self, order_dims, tile_dims, name):
        dim1_ = order_dims[0]
        dim2_ = order_dims[1]
        dim3_ = order_dims[2]

        dim1 = dim1_
        dim2 = dim2_
        dim3 = dim3_

        GEMM_flop = dim1 * dim3 * (dim2 + dim2 - 1)

        num_accesses = [0] * self.num_levels
        r1, r2, r3 = 1, 1, 1

        num_repeat = 1
        for level in range(self.num_levels - 1, 0, -1):
            repeat = (r1, r2, r3)
            num_accesses[level], tile1, tile2, tile3 = self.get_num_accesses(
                level,
                dim1,
                dim2,
                dim3,
                tile_dims[level - 1],
                num_repeat,
                name,
                repeat,
            )
            try:
                num_repeat *= (
                    math.ceil(dim1 / tile1)
                    * math.ceil(dim2 / tile2)
                    * math.ceil(dim3 / tile3)
                )
                r1 *= math.ceil(dim1 / tile1)
                r2 *= math.ceil(dim2 / tile2)
                r3 *= math.ceil(dim3 / tile3)
            except:
                num_repeat *= 1

            dim1 = tile1 if tile1 != 0 else dim1
            dim2 = tile2 if tile2 != 0 else dim2
            dim3 = tile3 if tile3 != 0 else dim3

            reuse = 1
            dim1 = dim1_
            dim2 = dim2_
            dim3 = dim3_

            if self.dataflow == "none":
                reuse = 1
            elif self.dataflow == "best":
                reuse = max(
                    math.ceil(dim1 / self.FMA_dims[0]),
                    math.ceil(dim3 / self.FMA_dims[0]),
                    math.ceil(dim2 / self.FMA_dims[1]),
                )
            elif self.dataflow == "wst":
                reuse = math.ceil(dim1 / self.FMA_dims[0])
            elif self.dataflow == "ast":
                reuse = math.ceil(dim3 / self.FMA_dims[0])
            elif self.dataflow == "ost":
                reuse = math.ceil(dim2 / self.FMA_dims[1])
            else:
                raise NotImplementedError()

            num_accesses[0] = GEMM_flop * ((2 * reuse + 1) / (2 * reuse)) * 1/(2 * self.FMA_dims[0]) * self.precision_bytes

        return GEMM_flop, num_accesses

    def get_cf(self, m, k, n):
        GEMM_time = self.get_gemm_time(m, k, n, "Cf")
        point_time = self._gemm_pointwise_time(m, n, "Cf")
        return GEMM_time[0] + point_time

    def _gemm_pointwise_time(self, m, n, name):
        point_flop = m * n * 5

        point_mem = self.precision_bytes * m * n * (3 * 3 + 2 * 2)

        point_time = (
            self.roofline(point_flop, point_mem, name=f"pointwise_{name}", mem_level=0)
            + 5 * self.O
        )

        if self.debug:
            gigaByte = 1024 * 1024 * 1024
            print(
                "Hidden point_flop: {:,}, point_mem: {:,}\n".format(
                    int(point_flop / 1e9), int(point_mem / gigaByte)
                )
            )
            print("Hidden point_time: {:,}\n".format(point_time))

        return point_time

    def grad_clipping(self, num_params: int) -> float:
        norm_comp = num_params * 2
        clip_comp = num_params * 2
        clip_mem = num_params * 2 * self.precision.gradients

        gradclip_mem = clip_mem
        gradclip_comp = norm_comp + clip_comp

        return self.roofline(
            gradclip_comp,
            gradclip_mem,
            name="pointwise-grad-clipping",
            mem_level=self.num_levels - 1,
        )

    def apply_grad(self, num_params: int) -> float:
        from kernel_trace_train import ADAMW_FLOPS
        bytes_per_param_read = (
            self.precision.parameters
            + self.precision.gradients
            + 2 * self.precision.optimizer_states
            + self.precision.master_parameters
        )
        bytes_per_param_write = (
            self.precision.parameters
            + 2 * self.precision.optimizer_states
            + self.precision.master_parameters
        )

        apply_grad_comp = num_params * ADAMW_FLOPS
        apply_grad_mem = num_params * (bytes_per_param_read + bytes_per_param_write)

        apply_grad_time = self.roofline(apply_grad_comp, apply_grad_mem, name="apply_grad", mem_level=self.num_levels-1)

        clip_time = self.grad_clipping(num_params) if bool(getattr(self.model, "gradient_clipping", False)) else 0.0
        return apply_grad_time + clip_time

    def get_dist_gemm_forward(self, m, k, n, name, batch=1):
        if self.tp <= 1:
            gemm_time = self.get_gemm_time(m, k, n, name)[0] * batch
            point_time = self._gemm_pointwise_time(m, n, name)
            return gemm_time + point_time, 0.0
        if self.gemm_shard_axis == "row":
            gemm_time = self.get_gemm_time(m, k // self.tp, n, name)[0]
            gemm_time *= batch
            point_time = self._gemm_pointwise_time(m, n, name)
            total_bytes = math.ceil(self.precision_bytes * m * n) * batch
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_REDUCE,
                size_bytes=total_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time + point_time, reduction_time
        if self.gemm_shard_axis == "col":
            gemm_time = self.get_gemm_time(m, k, n // self.tp, name)[0]
            gemm_time *= batch
            point_time = self._gemm_pointwise_time(m, n, name)
            total_bytes = math.ceil(self.precision_bytes * m * n) * batch
            shard_bytes = math.ceil(total_bytes / self.tp)
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_GATHER,
                size_bytes=shard_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time + point_time, reduction_time
        raise ValueError(f"Unsupported GEMM shard axis: {self.gemm_shard_axis}")

    def get_dist_gemm_backward(self, m, k, n, name, batch=1):
        if self.tp <= 1:
            grad_act_time, _, _, _ = self.get_gemm_time(m, n, k, name + "_act")
            grad_wt_time, _, _, _ = self.get_gemm_time(k, m, n, name + "_wt")
            return (grad_act_time + grad_wt_time) * batch, 0.0
        if self.gemm_shard_axis == "row":
            grad_wt_time, _, _, _ = self.get_gemm_time(k // self.tp, m, n, name + "_wt")
            grad_act_time, _, _, _ = self.get_gemm_time(m, n, k // self.tp, name + "_act")
            gemm_time = (grad_wt_time + grad_act_time) * batch
            total_bytes = math.ceil(self.precision_bytes * m * k) * batch
            shard_bytes = math.ceil(total_bytes / self.tp)
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_GATHER,
                size_bytes=shard_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time, reduction_time
        if self.gemm_shard_axis == "col":
            grad_wt_time, _, _, _ = self.get_gemm_time(k, m, n // self.tp, name + "_wt")
            grad_act_time, _, _, _ = self.get_gemm_time(m, n // self.tp, k, name + "_act")
            gemm_time = (grad_wt_time + grad_act_time) * batch
            total_bytes = math.ceil(self.precision_bytes * m * k) * batch
            reduction_time = self.network_model.collective(
                kind=CollectiveType.ALL_REDUCE,
                size_bytes=total_bytes,
                participants=int(self.tp),
                ib=self.links["tp"].bandwidth,
                ll=self.links["tp"].latency,
                debug_label=name or "comm",
            )
            return gemm_time, reduction_time
        raise ValueError(f"Unsupported GEMM shard axis: {self.gemm_shard_axis}")

    def get_data_parallel_reduction(self, k, n, name):
        if not self.dp or self.dp <= 1:
            return 0.0
        if self.gemm_shard_axis == "row":
            param_count = (k // self.tp) * n if self.tp > 1 else k * n
        elif self.gemm_shard_axis == "col":
            param_count = k * (n // self.tp) if self.tp > 1 else k * n
        else:
            raise ValueError(f"Unsupported GEMM shard axis: {self.gemm_shard_axis}")
        total_bytes = math.ceil(self.precision_bytes * param_count)
        reduction_time = self.network_model.collective(
            kind=CollectiveType.ALL_REDUCE,
            size_bytes=total_bytes,
            participants=int(self.dp),
            ib=self.links["dp"].bandwidth,
            ll=self.links["dp"].latency,
            debug_label=name or "comm",
        )
        apply_grad_time = self.apply_grad(int(math.ceil(param_count)))
        if self.debug:
            print(f"reduction_time_wt_dp: {reduction_time}")
            print(f"apply_grad_time: {apply_grad_time}")
        return reduction_time + apply_grad_time
