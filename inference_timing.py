import math
import os
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Mapping
from train_timing import (
    _decode_batched_attention,
    LLMExecutionDispatcher,
    TimeCalculationLLM,
    GemmType,
)
from memory_estimation import MemoryEstimator
from simulate_inference_graph import DecodeSample, InferenceConfig, InferenceEngine
from program.execution_energy import (
    add_idle_interval,
    apply_inference_gpu_accounting,
    combine_execution_energy,
    write_energy_report,
)
import llm_util
import kernel_trace
from opmodel import settings as _settings
from program import _env_flag
import json
from timing_model import DirectionTiming, OperationTiming, OperationGroup

def convert_prefix(value: float) -> float:
    if value == 0:
        return "0"
    if value < 0:
        return f"{value:.2f}"
    if value > 1:
        prefixes = ["", "k", "M", "G"]
        index = min(int(math.log10(value) // 3), len(prefixes) - 1)
        scaled_value = value / (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"
    else:
        prefixes = ["", "m", "µ", "n"]
        index = min(int(-(math.log10(value) // 3)), len(prefixes) - 1)
        scaled_value = value * (1000 ** index)
        return f"{scaled_value:.2f}{prefixes[index]}"

class TimeCalculationLLMInference(TimeCalculationLLM):
    def __init__(self, hw_config, model_config, mode, output_dir: Optional[str] = None):
        super().__init__(hw_config, model_config, mode, output_dir)
        self._raw_model_config = model_config
        source_moe = self._source_moe()
        if source_moe:
            if (self.implementation != "fused"
                    or self.implementation_options["flash_source"] != "vllm"):
                raise ValueError("vLLM source MoE inference requires the fused vLLM kernel description; no fallback")
            if not self.model.moe_source_collective_rule:
                raise ValueError("vLLM source MoE inference requires explicit moe_source_collective_rule")
            self._validate_moe_source_scope()
        from program import vtp
        if not source_moe and self.tp > 1 and (vtp.enabled(self) or kernel_trace.l2_residency_enabled()
                            or _settings.get("RAPID_L2_COLLECTIVE_BUFFERS") is not None):
            from program.vtp import validate
            validate(self)
        if self.run_type == "inference" and self.implementation_options["tp_allreduce"] == "vllm_size_rule" and self.tp > 1:
            if self.tp_sp:
                raise ValueError("vLLM 0.10.2 TP all-reduce model does not describe sequence parallelism")

            from copy import deepcopy
            self.hw_config = deepcopy(self.hw_config)
            self.hw_config.vllm_nccl_ring = True
            self.network_model.hw_config = self.hw_config
            dimension = self.hw_config.network_layout.parallelism_map["tp"]
            topology = str(dimension.topology_type).lower().replace("_", "").replace("-", "")
            if getattr(self.hw_config.network_layout, "faulty_links", ()):
                raise ValueError("vLLM IPC all-reduce does not model faulty links")

            sw = getattr(self.hw_config, "sw_config", None)
            curve_series = None
            if _settings.get("RAPID_ALLREDUCE_MODEL") == "measured_curve":
                from program.link_energy import resolve_rule
                if resolve_rule(self.hw_config, dimension) != "per_payload_byte":
                    raise ValueError("measured all-reduce requires per_payload_byte link energy")
                from program.allreduce_curve import load_curve, series_key
                curve_series = load_curve(_settings.get("RAPID_ALLREDUCE_CURVE"),
                                          include_unverified=_settings.get("RAPID_ALLREDUCE_INCLUDE_UNVERIFIED"),
                                          timing_basis=_settings.get("RAPID_ALLREDUCE_TIMING_BASIS"))
                key = series_key("nccl", self.tp, sw.allreduce_dtype)
                if _settings.get("RAPID_ALLREDUCE_TIMING_BASIS") == "kernel_union" and key not in curve_series:
                    raise ValueError(f"measured all-reduce curve missing series {key!r}")
                usage = self.resolved_settings["allreduce_usage"]
                usage["not_applied_reason"] = ("no_eligible_collectives" if key in curve_series
                                               else "missing_series:" + key)
            self.network_model.vllm_allreduce = dict(
                tp_allreduce=self.implementation_options["tp_allreduce"],
                fully_connected=topology in {"fullyconnected", "fc", "switch"},
                per_peer_links=topology in {"fullyconnected", "fc"},
                cuda_graph=False, bandwidth=self.links["tp"].bandwidth,
                latency=self.links["tp"].latency, kernel_overhead=self.O,
                barrier_timing=self.implementation_options["vllm_custom_barrier_timing"],
                sync_energy=self.implementation_options["vllm_custom_sync_energy"],
                hbm_bandwidth=self.mem_layer[self.num_levels - 1].get_throughput(),
                precision_bytes=self.precision.activations,
                dimension_index=self.hw_config.network_layout.dimensions.index(dimension),
            )
            if curve_series is not None:
                self.network_model.vllm_allreduce.update(
                    allreduce_model="measured_curve", allreduce_curve_series=curve_series,
                    allreduce_dtype=sw.allreduce_dtype,
                    allreduce_timing_basis=_settings.get("RAPID_ALLREDUCE_TIMING_BASIS"))
        elif (self.tp > 1 and _settings.get("RAPID_ALLREDUCE_MODEL") == "measured_curve"
              and _settings.get("RAPID_ALLREDUCE_TIMING_BASIS") == "kernel_union"):
            raise ValueError("measured all-reduce curves require vLLM inference TP > 1")

    def _validate_moe_source_scope(self):
        if str(getattr(self, "model_type", "")).lower() in ("deepseek_v2", "deepseek_v3"):
            return self._validate_deepseek_moe_scope()
        if (not self.use_moe or self.n_shared_experts or
                len(self.moe_layer_mask) != self.num_layers or not all(self.moe_layer_mask)):
            raise ValueError("qwen3_moe source inference requires every layer to use routed experts and no shared experts")
        if not self.flash_attention or not self.use_rope:
            raise ValueError("qwen3_moe source inference requires FlashAttention and RoPE")
        if self.disable_embedding_unembedding or getattr(self, "disable_kv_cache", False):
            raise ValueError("qwen3_moe source inference requires embeddings, logits, and KV cache")
        if self.seq_len > self.max_position_embeddings:
            raise ValueError("qwen3_moe context exceeds max_position_embeddings; scaled RoPE is not described")
        if self.tp_sp:
            raise ValueError("qwen3_moe source inference does not describe sequence parallelism")
        for name in ("fuse_norm", "fuse_rope", "fuse_activation", "fuse_residual_add",
                     "vocab_parallel_head_gather"):
            if not self.implementation_options[name]:
                raise ValueError(f"qwen3_moe source inference requires implementation option {name}=true")

    def _validate_deepseek_moe_scope(self):
        moe = self.model.moe
        if str(getattr(self, "attention_type", "mha")).lower() != "mla" or not self._mla_vllm_trace():
            raise ValueError("DeepSeek MoE source inference requires MLA with implementation.options.mla_inference=vllm_trace")
        missing = [f for f in ("scoring_func", "topk_method", "n_group", "topk_group", "norm_topk_prob",
                               "routed_scaling_factor") if getattr(moe, f) is None]
        if missing:
            raise ValueError(f"DeepSeek MoE source inference requires model_param.moe.{missing} (published config)")
        if self.disable_embedding_unembedding or getattr(self, "disable_kv_cache", False):
            raise ValueError("DeepSeek MoE source inference requires embeddings, logits and KV cache")
        if self.tp_sp:
            raise ValueError("DeepSeek MoE source inference does not describe sequence parallelism")
        tp = int(self.tp)
        if tp > 1 and (int(self.num_heads) % tp or int(self.num_heads) // tp < 2):
            raise ValueError("DeepSeek MoE at tp > 1 needs tp | num_heads and at least 2 heads per rank")
        for name in ("fuse_norm", "fuse_rope", "fuse_activation", "fuse_residual_add", "vocab_parallel_head_gather"):
            if not self.implementation_options[name]:
                raise ValueError(f"DeepSeek MoE source inference requires implementation option {name}=true")

    @contextmanager
    def inference_phase_network_override(self, phase: str):
        if getattr(getattr(self, "network_model", None), "vllm_allreduce", None):
            yield
            return
        inference_config = getattr(self.hw_config, "inference_config", None)
        overrides = getattr(inference_config, "phase_network_overrides", {}) or {}
        override = overrides.get(str(phase).strip().lower())
        if override is None:
            yield
            return

        original_layout = self.hw_config.network_layout
        original_network = self.network
        astra_config = getattr(self.hw_config.execution_backend, "astrasim", None)
        original_collectives = (
            getattr(astra_config, "collectives", None) if astra_config else None
        )
        dimensions = list(original_layout.dimensions)
        if not dimensions:
            raise ValueError("inference network override requires a network dimension")
        base_dimension = dimensions[0]
        dimension_updates = {}
        for field_name in (
            "topology_type",
            "bandwidth",
            "util",
            "latency",
            "energy_per_bit",
            "system_power_w",
            "static_power_per_link_w",
        ):
            value = getattr(override, field_name, None)
            if value is not None:
                dimension_updates[field_name] = value
        collective_override = dict(base_dimension.collective_override or {})
        collective_override.update(dict(override.collective_override or {}))
        dimension_updates["collective_override"] = collective_override
        phase_dimension = replace(base_dimension, **dimension_updates)
        dimensions[0] = phase_dimension
        parallelism_map = {
            axis: (phase_dimension if dimension is base_dimension else dimension)
            for axis, dimension in original_layout.parallelism_map.items()
        }
        self.hw_config.network_layout = replace(
            original_layout,
            dimensions=tuple(dimensions),
            parallelism_map=parallelism_map,
        )
        if original_collectives is not None:
            collective_updates = {}
            for name, algorithm in collective_override.items():
                attr = str(name).strip().lower().replace("-", "_")
                if hasattr(original_collectives, attr):
                    collective_updates[attr] = str(algorithm)
            if collective_updates:
                astra_config.collectives = replace(
                    original_collectives, **collective_updates
                )
        self.network = type(original_network)(self.hw_config)
        try:
            yield
        finally:
            self.hw_config.network_layout = original_layout
            if astra_config is not None and original_collectives is not None:
                astra_config.collectives = original_collectives
            self.network = original_network

    def _build_decode_transformer_results(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        use_moe_layer: bool,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> Tuple[Dict[str, OperationTiming], Dict[str, float]]:
        if (self.flash_attention and str(getattr(self, "attention_type", "mha")).lower() != "mla"
                and int(self.cp) != 1):
            raise NotImplementedError("FlashAttention rebuild CP>1 needs a query-position rule")

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads

        token_bytes = llm_util.attention_kv_cache_token_bytes(
            getattr(self, "attention_type", "mha"),
            batch_size=batch_size,
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
            kv_lora_rank=getattr(self, "kv_lora_rank", None),
            num_heads=getattr(self, "num_heads", None),
            qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
            v_head_dim=getattr(self, "v_head_dim", None),
        )
        intermediate_size = self.moe_intermediate_size if use_moe_layer else self.intermediate_size
        gemm_ctx = self
        if use_moe_layer != self.use_moe:
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
        if gemm_shapes is None or use_moe_layer != self.use_moe:
            gemm_shapes = llm_util.process_decode_gemm_shapes(
                gemm_ctx,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )

        gemm_qkv_proj = gemm_shapes["qkv_proj"]
        gemm_attention_score = gemm_shapes["attention_score"]
        gemm_attention_output = gemm_shapes["attention_output"]
        gemm_output_proj = gemm_shapes["output_proj"]
        gemm_ffn1 = gemm_shapes["ffn1"]
        gemm_ffn2 = gemm_shapes["ffn2"]
        run_type = str(
            getattr(
                self,
                "run_type",
                getattr(getattr(self, "model", None), "run_type", "training"),
            )
        ).lower()
        mla_decode = str(getattr(self, "attention_type", "mha")).lower() == "mla" and run_type == "inference"

        mla_ops = (self._mla_vllm_op_timings(phase="decode", batch=int(batch_size), seq=max(0, int(total_seq_len) - 1))
                   if mla_decode and self._mla_vllm_trace() else None)

        output_seq_len = 1

        kernel_energy_marker = self._inference_kernel_energy_marker()
        transformer_timings: Dict[str, OperationTiming] = {}

        def _make_forward(
            op_name: str,
            compute_time: float,
            comm_time: float,
            comm_bytes: float,
            *,
            flops: float = 0.0,
            memory: Optional[Mapping[str, float]] = None,
        ) -> DirectionTiming:
            bytes_int = int(math.ceil(float(comm_bytes or 0.0)))
            timing = DirectionTiming(
                compute_time=compute_time,
                comm_time=comm_time,
                comm_bytes=bytes_int,
                flops=flops,
                memory_accesses=dict(memory) if memory else {},
            )
            return replace(
                timing,
                activity=self._direction_activity(op_name, "forward", timing),
            )

        if mla_ops is not None:
            transformer_timings["qkv_proj"] = mla_ops["qkv_proj"]
            qkv_op = transformer_timings["qkv_proj"].forward
            qkv_proj_time, qkv_proj_reduction, qkv_proj_size = qkv_op.compute_time, 0.0, 0
            qkv_proj_flops, qkv_proj_mem = qkv_op.flops, qkv_op.memory_accesses
        elif mla_decode:
            transformer_timings["qkv_proj"] = self._build_mla_qkv_projection_timing(
                gemm_shapes,
                include_backward=False,
                hidden_dim=self.hidden_dim,
                decode=True,
            )
            qkv_op = transformer_timings["qkv_proj"].forward
            qkv_proj_time = qkv_op.compute_time
            qkv_proj_reduction = qkv_op.comm_time
            qkv_proj_size = qkv_op.comm_bytes
            qkv_proj_flops = qkv_op.flops
            qkv_proj_mem = qkv_op.memory_accesses
        else:
            qkv_proj_time, qkv_proj_reduction, qkv_proj_size, qkv_proj_flops, qkv_proj_mem = self._gemm_forward_parts(
                gemm_qkv_proj, "decode_qkv_proj_f", GemmType.QKV, decode=True
            )
            transformer_timings["qkv_proj"] = OperationTiming(
                "qkv_proj",
                forward=_make_forward(
                    "qkv_proj",
                    compute_time=qkv_proj_time,
                    comm_time=qkv_proj_reduction,
                    comm_bytes=qkv_proj_size,
                    flops=qkv_proj_flops,
                    memory=self._mem_levels(qkv_proj_mem),
                ),
                backward=None,
            )

        rope_timing = mla_ops["rope"] if mla_ops is not None else self._inference_rope_timing(gemm_qkv_proj)
        if rope_timing is not None:
            transformer_timings["rope"] = rope_timing

        if mla_ops is not None:
            transformer_timings["attention"] = mla_ops["attention"]
            attention_score_time = transformer_timings["attention"].forward.compute_time
            attention_output_time = attention_scale_softmax_f = 0.0
            attention_score_reduction = attention_output_reduction = 0.0
            transformer_timings["attention_scale_softmax"] = OperationTiming(
                "attention_scale_softmax", forward=DirectionTiming(compute_time=0.0), backward=None)
            self.resolved_settings.setdefault("attention_paths", {})["decode"] = {
                "attention_path": "vllm_triton_mla", "source": "kernel_trace_vllm_mla"}
        elif mla_decode:
            transformer_timings.update(
                self._build_mla_attention_timings(
                    gemm_shapes,
                    include_backward=False,
                    hidden_dim=self.hidden_dim,
                    decode=True,
                )
            )
            attention_score_op = transformer_timings["attention_score"].forward
            attention_score_time = attention_score_op.compute_time
            attention_score_reduction = attention_score_op.comm_time
            attention_score_size = attention_score_op.comm_bytes
            attention_score_flops = attention_score_op.flops
            attention_score_mem = attention_score_op.memory_accesses
            attention_output_op = transformer_timings["attention_output"].forward
            attention_output_time = attention_output_op.compute_time
            attention_output_reduction = attention_output_op.comm_time
            attention_output_size = attention_output_op.comm_bytes
            attention_output_flops = attention_output_op.flops
            attention_output_mem = attention_output_op.memory_accesses
            attention_scale_softmax_f = transformer_timings["attention_scale_softmax"].forward.compute_time
        elif self.flash_attention:
            import flash_attention_adapter
            attention_forward_compute, flash_activity, flash_source = flash_attention_adapter.decode(
                self, batch_size, self.hidden_dim, total_seq_len, self.num_heads, self.kv_heads,
                self.hw_config.tech_config.core.num_bundles)
            flash_notes = {"attention_path": "flash_rebuild_decode", "source": flash_source}
            self.resolved_settings.setdefault("attention_paths", {})["decode"] = flash_notes
            attention_reduction = attention_comm_bytes = 0
            attention_flops = flash_activity.useful_tensor_flops + flash_activity.vector_flops
            attention_mem = {}
            attention_score_time = attention_forward_compute
            attention_output_time = attention_scale_softmax_f = 0.0
            attention_score_reduction = attention_output_reduction = 0.0
            attention_forward_time = attention_forward_compute
            transformer_timings["attention"] = OperationTiming(
                "attention", forward=DirectionTiming(compute_time=attention_forward_compute,
                    flops=attention_flops, activity=flash_activity, notes=flash_notes), backward=None)
            transformer_timings["attention_scale_softmax"] = OperationTiming(
                "attention_scale_softmax", forward=DirectionTiming(compute_time=0.0), backward=None)
        else:
            attention_score_time, attention_score_reduction, attention_score_size, attention_score_flops, attention_score_mem = self.parallelism_gemm_forward(
                gemm_attention_score, "decode_attention_score_f", gemm_type=GemmType.ATTENTION_SCORE, decode=True
            )
            attention_output_time, attention_output_reduction, attention_output_size, attention_output_flops, attention_output_mem = self.parallelism_gemm_forward(
                gemm_attention_output, "decode_attention_output_f", gemm_type=GemmType.ATTENTION_OUTPUT, decode=True
            )

            attention_scale_softmax_f = 0.0 if _decode_batched_attention() else self.get_scale_softmax_f(gemm_attention_score)

            attention_reduction = attention_score_reduction + attention_output_reduction
            attention_comm_bytes = attention_score_size + attention_output_size
            attention_forward_compute = attention_score_time + attention_scale_softmax_f + attention_output_time
            attention_forward_time = attention_forward_compute + attention_reduction
            attention_flops = (attention_score_flops or 0.0) + (attention_output_flops or 0.0)
            attention_mem = self._combine_mem(attention_score_mem, attention_output_mem)

            transformer_timings["attention"] = OperationTiming(
                "attention",
                forward=_make_forward(
                    "attention",
                    compute_time=attention_forward_compute,
                    comm_time=attention_reduction,
                    comm_bytes=attention_comm_bytes,
                    flops=attention_flops,
                    memory=attention_mem,
                ),
                backward=None,
            )

            attention_scale_softmax_op = OperationTiming(
                "attention_scale_softmax",
                forward=_make_forward(
                    "attention_scale_softmax",
                    compute_time=attention_scale_softmax_f,
                    comm_time=0.0,
                    comm_bytes=0.0,
                ),
                backward=None,
            )
            transformer_timings["attention_scale_softmax"] = attention_scale_softmax_op

        if mla_decode:
            attention_op = transformer_timings["attention"].forward
            attention_reduction = attention_op.comm_time
            attention_comm_bytes = attention_op.comm_bytes
            attention_forward_compute = attention_op.compute_time
            attention_forward_time = transformer_timings["attention"].total_forward_time()
            attention_flops = attention_op.flops
            attention_mem = attention_op.memory_accesses

        self._eager_kv_past, self._eager_kv_batch = max(0, int(total_seq_len) - 1), int(batch_size)
        transformer_timings["attention"], kv_write_time = self._with_kv_cache_write(
            transformer_timings["attention"], int(batch_size)
        )
        self._eager_kv_past, self._eager_kv_batch = 0, 1

        if mla_ops is not None:
            transformer_timings["output_proj"] = mla_ops["output_proj"]
            out_proj_op = transformer_timings["output_proj"].forward

            out_proj_time, out_proj_reduction, out_proj_size = (out_proj_op.compute_time, out_proj_op.comm_time,
                                                                out_proj_op.comm_bytes)
            out_proj_flops, out_proj_mem = out_proj_op.flops, out_proj_op.memory_accesses
        elif mla_decode:
            transformer_timings["output_proj"] = self._build_mla_output_projection_timing(
                gemm_shapes,
                include_backward=False,
                hidden_dim=self.hidden_dim,
            )
            out_proj_op = transformer_timings["output_proj"].forward
            out_proj_time = out_proj_op.compute_time
            out_proj_reduction = out_proj_op.comm_time
            out_proj_size = out_proj_op.comm_bytes
            out_proj_flops = out_proj_op.flops
            out_proj_mem = out_proj_op.memory_accesses
        else:
            out_proj_time, out_proj_reduction, out_proj_size, out_proj_flops, out_proj_mem = self.parallelism_gemm_forward(
                gemm_output_proj, "decode_output_projection_f", gemm_type=GemmType.OUT_PROJ
            )
            transformer_timings["output_proj"] = OperationTiming(
                "output_proj",
                forward=_make_forward(
                    "output_proj",
                    compute_time=out_proj_time,
                    comm_time=out_proj_reduction,
                    comm_bytes=out_proj_size,
                    flops=out_proj_flops,
                    memory=self._mem_levels(out_proj_mem),
                ),
                backward=None,
            )

        router_time_f = 0.0
        router_comm_f = 0.0
        router_bytes_f = 0.0
        dispatch_fwd_time = 0.0
        dispatch_fwd_bytes = 0.0
        combine_fwd_time = 0.0
        moe_tokens_local = None
        moe_tokens_shared = None
        if not use_moe_layer:
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem = self._gemm_forward_parts(
                gemm_ffn1, "decode_ffn1_f", GemmType.FFN1
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem = self.parallelism_gemm_forward(
                gemm_ffn2, "decode_ffn2_f", gemm_type=GemmType.FFN2
            )
        else:
            gemm_router = gemm_shapes.get("router")
            if gemm_router is None:
                raise KeyError("Missing decode GEMM shape for 'router'")
            allow_padding = self._moe_allow_padding()
            (
                tokens_owner,
                tokens_dispatched_balanced,
                _tokens_local_balanced,
                experts_per_rank,
                _tokens_per_expert_balanced,
            ) = self._moe_balanced_routed_tokens_per_expert(
                batch_size,
                output_seq_len,
                allow_padding=allow_padding,
            )
            moe_tokens_shared = self._moe_tokens_shared(tokens_owner)
            if _env_flag("RAPID_MOE_ACTIVE_EXPERTS"):
                share = max(1, int(self._moe_local_share_divisor()))
                moe_tokens_local = int(math.ceil(float(tokens_owner) * int(self.moe_top_k) / share))
            router_time_f, router_comm_f, router_bytes_f = self.get_router_f(
                gemm_router,
                gemm_ffn1,
                batch_size=batch_size,
                seq_len=output_seq_len,
            )
            moe_group = self._moe_routing_group()
            axis = None
            dispatch_fwd_bytes, dispatch_fwd_time, combine_fwd_bytes, combine_fwd_time = self._moe_forward_comm(
                tokens_owner // self._moe_dp_attention_groups(), tokens_dispatched_balanced, experts_per_rank, "decode"
            )
            ffn1_time, ffn1_reduction, ffn1_size, ffn1_flops, ffn1_mem = self.get_moe_ffn_f(
                gemm_ffn1,
                "decode_ffn1_f",
                gemm_type=GemmType.FFN1,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
            )
            ffn2_time, ffn2_reduction, ffn2_size, ffn2_flops, ffn2_mem = self.get_moe_ffn_f(
                gemm_ffn2,
                "decode_ffn2_f",
                gemm_type=GemmType.FFN2,
                batch_size=batch_size,
                seq_len=output_seq_len,
                allow_padding=allow_padding,
            )
            transformer_timings["router"] = OperationTiming(
                "router",
                forward=_make_forward(
                    "router",
                    compute_time=router_time_f,
                    comm_time=router_comm_f,
                    comm_bytes=router_bytes_f,
                ),
                backward=None,
            )
            transformer_timings["moe_dispatch"] = OperationTiming(
                "moe_dispatch",
                forward=_make_forward(
                    "moe_dispatch",
                    compute_time=0.0,
                    comm_time=dispatch_fwd_time,
                    comm_bytes=dispatch_fwd_bytes,
                ),
                backward=None,
            )
            transformer_timings["moe_combine"] = OperationTiming(
                "moe_combine",
                forward=_make_forward(
                    "moe_combine",
                    compute_time=0.0,
                    comm_time=combine_fwd_time,
                    comm_bytes=combine_fwd_bytes,
                ),
                backward=None,
            )

        transformer_timings["ffn1"] = OperationTiming(
            "ffn1",
            forward=_make_forward(
                "ffn1",
                compute_time=ffn1_time,
                comm_time=ffn1_reduction,
                comm_bytes=ffn1_size,
                flops=ffn1_flops,
                memory=self._mem_levels(ffn1_mem),
            ),
            backward=None,
        )
        transformer_timings["ffn2"] = OperationTiming(
            "ffn2",
            forward=_make_forward(
                "ffn2",
                compute_time=ffn2_time,
                comm_time=ffn2_reduction,
                comm_bytes=ffn2_size,
                flops=ffn2_flops,
                memory=self._mem_levels(ffn2_mem),
            ),
            backward=None,
        )

        ffn1_spec = self._shard_gemm_descriptor(gemm_ffn1, GemmType.FFN1)
        ffn1_activation_shape = (ffn1_spec.shard_m, ffn1_spec.k, ffn1_spec.shard_n)
        ffn1_activation_shape_shared = None
        if use_moe_layer and moe_tokens_local is not None:
            use_tp_sharded = bool(getattr(self, "tp_ep", True))
            ffn1_n = ffn1_spec.shard_n if use_tp_sharded else ffn1_spec.n
            ffn1_activation_shape = (moe_tokens_local, ffn1_spec.k, ffn1_n)
            if moe_tokens_shared and moe_tokens_shared > 0:
                ffn1_activation_shape_shared = (moe_tokens_shared, ffn1_spec.k, ffn1_n)
        if llm_util.is_llama_style(self.model_type):
            act_f = self.get_swiglu_f(ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_swiglu_f(ffn1_activation_shape_shared)
        else:
            act_f = self.get_gelu_f(ffn1_activation_shape)
            if ffn1_activation_shape_shared is not None:
                act_f += self.get_gelu_f(ffn1_activation_shape_shared)
        transformer_timings["gelu"] = OperationTiming(
            "gelu",
            forward=_make_forward("gelu", compute_time=act_f, comm_time=0.0, comm_bytes=0.0),
            backward=None,
        )

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        q_size = self.num_heads * head_dim
        output_proj_shape = (
            batch_size,
            output_seq_len,
            q_size,
            self.hidden_dim,
        )
        residual1_f = 0.0 if self._vllm_norms() else self.get_residual_f(output_proj_shape)
        if self._vllm_norms():
            layernorm1_f = self._vllm_norm_f(batch=batch_size, seq_len=output_seq_len, role="layernorm1")
            layernorm1_reduction, layernorm1_bytes = 0.0, 0
        else:
            layernorm1_f, layernorm1_reduction, layernorm1_bytes = self.get_layernorm_f(
                batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
            )
        transformer_timings["layernorm1"] = OperationTiming(
            "layernorm1",
            forward=_make_forward(
                "layernorm1",
                compute_time=layernorm1_f + residual1_f,
                comm_time=layernorm1_reduction,
                comm_bytes=layernorm1_bytes,
            ),
            backward=None,
        )

        ffn2_shape = (
            batch_size,
            output_seq_len,
            intermediate_size,
            self.hidden_dim,
        )
        residual2_f = 0.0 if self._vllm_norms() else self.get_residual_f(ffn2_shape)
        if self._vllm_norms():
            layernorm2_f = self._vllm_norm_f(batch=batch_size, seq_len=output_seq_len, role="layernorm2")
            layernorm2_reduction, layernorm2_bytes = 0.0, 0
        else:
            layernorm2_f, layernorm2_reduction, layernorm2_bytes = self.get_layernorm_f(
                batch=batch_size, seq_len=output_seq_len, d_model=self.hidden_dim
            )
        transformer_timings["layernorm2"] = OperationTiming(
            "layernorm2",
            forward=_make_forward(
                "layernorm2",
                compute_time=layernorm2_f + residual2_f,
                comm_time=layernorm2_reduction,
                comm_bytes=layernorm2_bytes,
            ),
            backward=None,
        )

        linear_shape = (
            batch_size,
            output_seq_len,
            self.hidden_dim,
            self.vocab_size,
        )
        linear_softmax_f, linear_softmax_mem = self.get_linear_softmax_f(linear_shape)
        if getattr(self, "disable_embedding_unembedding", False):
            linear_softmax_f = 0.0
            linear_softmax_mem = {}
        transformer_timings["linear_softmax"] = OperationTiming(
            "linear_softmax",
            forward=_make_forward(
                "linear_softmax",
                compute_time=linear_softmax_f,
                comm_time=0.0,
                comm_bytes=0.0,
                memory=self._mem_levels(linear_softmax_mem),
            ),
            backward=None,
        )

        mlp_group = OperationGroup(
            "MLP",
            operations=(
                transformer_timings["ffn1"],
                transformer_timings["gelu"],
                transformer_timings["ffn2"],
            ),
        )

        qkv_proj_forward = qkv_proj_time + qkv_proj_reduction
        rope_forward = transformer_timings["rope"].total_forward_time() if "rope" in transformer_timings else 0.0
        attention_forward = attention_score_time + attention_scale_softmax_f + attention_output_time + attention_reduction
        attention_forward += kv_write_time
        out_proj_forward = out_proj_time + out_proj_reduction
        mha_forward = qkv_proj_forward + attention_forward + out_proj_forward

        ffn1_forward = ffn1_time + ffn1_reduction
        ffn2_forward = ffn2_time + ffn2_reduction
        mlp_forward = ffn1_forward + act_f + ffn2_forward
        if use_moe_layer:
            mlp_forward += router_time_f + dispatch_fwd_time + combine_fwd_time

        layernorm1_forward = residual1_f + layernorm1_f
        layernorm2_forward = residual2_f + layernorm2_f

        transformer_forward = (
            mha_forward
            + rope_forward
            + mlp_forward
            + layernorm1_forward
            + layernorm1_reduction
            + layernorm2_forward
            + layernorm2_reduction
        )

        node_breakdown = {
            "transformer_time_f": transformer_forward,
            "transformer_time_b": 0.0,
            "linear_softmax_f": transformer_timings["linear_softmax"].total_forward_time(),
            "linear_softmax_b": 0.0,
            "embedding_f": 0.0,
            "embedding_b": 0.0,
        }

        node_breakdown["inference_kernel_energy_records"] = (
            self._inference_kernel_energy_since(kernel_energy_marker)
        )
        if kernel_trace.l2_residency_enabled():
            self._apply_l2_residency(transformer_timings, tokens=int(batch_size), batch=int(batch_size),
                                     past=max(0, int(total_seq_len) - 1), head_tokens=int(batch_size))
        else:
            from program import vtp
            if vtp.enabled(self):
                vtp.apply(self, transformer_timings, phase="decode", tokens=int(batch_size), batch=int(batch_size),
                          past=max(0, int(total_seq_len)-1))
        return transformer_timings, node_breakdown

    def decode_uses_cuda_graph(self, total_seq_len: int) -> bool:
        from kernel_trace_vllm import VllmEngine, decode_uses_cuda_graph
        return (self.run_type == "inference" and self.implementation == "fused"
                and self.implementation_options["flash_source"] == "vllm"
                and decode_uses_cuda_graph(total_seq_len, VllmEngine(sm_count=int(self.core.num_bundle), **dict(self.engine_settings or {}))))

    def decode_launch_overhead(self, total_seq_len: int) -> float:
        if (_settings.get("RAPID_KERNEL_LAUNCH_MODE") == "graph_when_captured"
                and self.decode_uses_cuda_graph(total_seq_len)):
            value = _settings.get("RAPID_GRAPH_LAUNCH_OVERHEAD_S")
            if value is None:
                raise ValueError("graph_when_captured requires sw_param.graph_launch_overhead_s")
            return value
        return self.stream_launch_overhead_s

    def prepare_decode_graphs(self, *, batch_size: int, total_seq_len: int, gemm_shapes=None):
        original = self.O
        original_decode_scope = getattr(self, "_pricing_decode_template", False)
        policy = getattr(self.network_model, "vllm_allreduce", None)
        original_policy = dict(policy) if policy else None
        mode = _settings.get("RAPID_KERNEL_LAUNCH_MODE")
        captured = self.decode_uses_cuda_graph(total_seq_len)
        self._decode_graph_pricing = mode == "graph_when_captured" and captured
        self.O = self.decode_launch_overhead(total_seq_len)
        if policy:
            policy["kernel_overhead"] = self.O
        self._pricing_decode_template = True
        try:
            result = self._prepare_decode_graphs(
                batch_size=batch_size, total_seq_len=total_seq_len, gemm_shapes=gemm_shapes)
            self._record_decode_launches(batch_size, total_seq_len, mode, captured)
            return result
        except BaseException:
            if original_policy is not None:
                policy.clear()
                policy.update(original_policy)
            raise
        finally:
            self.O = original
            self._decode_graph_pricing = False
            self._pricing_decode_template = original_decode_scope

    def _record_decode_launches(self, batch_size, total_seq_len, mode, captured):
        from program import vtp
        if vtp.enabled(self):
            ranks = self._vtp_launch_counts
        elif self._mla_vllm_trace():
            from kernel_trace_vllm import VllmEngine, vllm_step_trace
            trace = vllm_step_trace(self._l2_sim_config(), "decode", int(batch_size), max(0, int(total_seq_len) - 1),
                                    eng=VllmEngine(sm_count=int(self.core.num_bundle),
                                                   **dict(self.engine_settings or {})))
            count = sum(k.kind not in ("memcpy", "collective") for k in trace)
            outside = sum(k.name.startswith("sampler.") or k.name == "meta.query_start_loc_sub" for k in trace)
            graph = count - outside if mode == "graph_when_captured" and captured else 0
            ranks = [{"rank": 0, "stream_kernels": count - graph, "graph_kernels": graph,
                      "stream_memcpy_launch_charges": 0, "custom_allreduce_kernels": 0}]
        elif (self.implementation == "fused" and self.implementation_options["flash_source"] == "vllm"
              and not self.use_moe and self.flash_attention and self.tp == self.cp == self.pp == 1

              and str(getattr(self, "attention_type", "mha")).lower() != "mla"):
            pw = _env_flag("RAPID_VLLM_POINTWISE")
            per_layer = (4 + int(bool(getattr(self, "use_rope", True)))
                         + int(_env_flag("RAPID_KV_CACHE_WRITE"))
                         + (1 if pw else 2) + 2 * (1 if pw else 3)
                         + (0 if pw else 2) + self._decode_flash_launch_count)
            if self._sourced_vllm_rule("RAPID_VLLM_ATTN_OUTPUT_ZEROS"):
                per_layer += 1
            if self.model_type == "qwen3":
                per_layer += 2 + (2 if int(batch_size) > 1 else 0)
            elif self.model_type == "qwen2":
                from kernel_trace_vllm import VllmEngine, vllm_step_trace
                cfg = dict(self._l2_sim_config(), num_hidden_layers=1)
                per_layer += sum(k.op == "qkv_proj" and k.kind != "gemm"
                                 for k in vllm_step_trace(cfg, "prefill", int(batch_size), 1,
                                     eng=VllmEngine(sm_count=int(self.core.num_bundle))))
            if self._sourced_vllm_rule("RAPID_VLLM_SELECTED_SAMPLER"):
                trace = self._sourced_vllm_trace(int(batch_size))
                sampler_launches = sum(k.op == "linear_softmax" and
                    k.name not in ("fused_add_rms_norm", "lm_head", "logits.gather") for k in trace)
            else:
                sampler_launches = 1 if self._opmodel_backend is not None else 4
            head = 1 + (1 if pw else 3) + sampler_launches
            if getattr(self, "disable_embedding_unembedding", False):
                head = 0
            count = per_layer * int(self.num_layers) + head
            outside = (sum(k.name.startswith("sampler.") for k in trace)
                       if self._sourced_vllm_rule("RAPID_VLLM_SELECTED_SAMPLER") and head else 0)
            graph = count - outside if mode == "graph_when_captured" and captured else 0
            ranks = [{"rank": 0, "stream_kernels": count - graph, "graph_kernels": graph,
                      "stream_memcpy_launch_charges": 0, "custom_allreduce_kernels": 0}]
        else:
            ranks = None
        self.resolved_settings["kernel_launch"]["decode_templates"].append({
            "batch": int(batch_size), "kv_len": int(total_seq_len), "captured": captured,
            "mode": mode, "ranks": ranks,
            "count_scope": "priced decode template per rank; model launch terms, includes custom AR; "
                           "NCCL collective_launch_overhead excluded; null ranks = undescribed census"})

    def _prepare_decode_graphs(
        self,
        *,
        batch_size: int,
        total_seq_len: int,
        gemm_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    ):
        if getattr(self.network_model, "vllm_allreduce", None):
            self.network_model.vllm_allreduce["cuda_graph"] = bool((self.engine_settings or {}).get("cuda_graph", True))
        moe_layers_active = bool(self.use_moe and any(getattr(self, "moe_layer_mask", []) or []))
        moe_intermediate = self.moe_intermediate_size
        decode_gemm_shapes_moe = gemm_shapes
        if decode_gemm_shapes_moe is None:
            decode_gemm_shapes_moe = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=moe_intermediate if self.use_moe else self.intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )
        decode_gemm_shapes_dense = decode_gemm_shapes_moe
        if self.use_moe:
            dense_ctx = SimpleNamespace(
                use_moe=False,
                moe_num_experts=self.moe_num_experts,
                moe_top_k=self.moe_top_k,
                moe_intermediate_size=self.intermediate_size,
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
            decode_gemm_shapes_dense = llm_util.process_decode_gemm_shapes(
                dense_ctx,
                batch_size=batch_size,
                current_seq_len=total_seq_len,
                d_model=self.hidden_dim,
                num_heads=self.num_heads,
                kv_heads=self.kv_heads,
                intermediate_size=self.intermediate_size,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
            )

        transformer_timings, node_breakdown = self._build_decode_transformer_results(
            batch_size=batch_size,
            total_seq_len=total_seq_len,
            use_moe_layer=False,
            gemm_shapes=decode_gemm_shapes_dense,
        )
        if (getattr(self.network_model, "vllm_allreduce", None)
                and not getattr(self, "disable_embedding_unembedding", False)):
            from timing_model import CollectiveType
            size = batch_size * self.hidden_dim * self.precision.activations
            duration = self._training_tp_collective(
                CollectiveType.ALL_REDUCE, size, "embedding_f_allreduce")
            node_breakdown["embedding_f"] = duration
            self._set_endpoint_collectives("embedding_f", self._endpoint_collective(
                "embedding_f_allreduce", CollectiveType.ALL_REDUCE, size, duration, 0.0))
        from program import vtp
        vtp.update_breakdown(self, transformer_timings, node_breakdown)
        moe_transformer_timings = None
        moe_node_breakdown = None
        if moe_layers_active:
            moe_transformer_timings, moe_node_breakdown = self._build_decode_transformer_results(
                batch_size=batch_size,
                total_seq_len=total_seq_len,
                use_moe_layer=True,
                gemm_shapes=decode_gemm_shapes_moe,
            )

        output_act_bytes = decode_gemm_shapes_dense["qkv_proj"][0] * decode_gemm_shapes_dense["qkv_proj"][1] * self.precision_bytes

        energy = 0.0

        if self._generate_graphs:
            results_path = os.path.join(self.output_dir, "decode_transformer_results.txt")
            with open(results_path, "w", encoding="utf-8") as results_file:
                json.dump(
                    {
                        "transformer_results": {
                            name: timing.to_dict() for name, timing in transformer_timings.items()
                        },
                        "node_breakdown": node_breakdown,
                    },
                    results_file,
                    indent=2,
                    sort_keys=True,
                )

        self._last_inference_kernel_energy_records = list(
            node_breakdown.get("inference_kernel_energy_records", ())
        )
        return self._prepare_execution_graphs(
            node_breakdown=node_breakdown,
            transformer_timings=transformer_timings,
            moe_node_breakdown=moe_node_breakdown,
            moe_transformer_timings=moe_transformer_timings,
            batch_size=batch_size,
            seq_len=1,
            hidden_dim=self.hidden_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            include_pipeline_backward=False,
            include_transformer_backward=False,
            gemm_shapes=decode_gemm_shapes_moe if self.use_moe else decode_gemm_shapes_dense,
        ), energy

    def _moe_execution_energy(self, step):
        from program.energy import ProgramEnergyResult, price_activity
        from program.execution_energy import EnergyCoverage, ExecutionEnergyResult
        from program.network_energy import CollectiveCommunicationEnergy, CommunicationEnergyResult

        if step.latency_s is None or step.static_gpu_j is None:
            raise ValueError("counts-only MoE probes cannot produce an inference energy prediction")
        rank_activity = {str(rank.rank): rank.activity for rank in step.ranks}
        rank_energy = {rank: price_activity(activity, self.hw_config)
                       for rank, activity in rank_activity.items()}
        dynamic = price_activity(step.activity, self.hw_config)
        gpu = ProgramEnergyResult(
            activity=step.activity, dynamic=dynamic, static_j=step.static_gpu_j,
            wall_time_s=step.latency_s, device_count=len(step.ranks),
            by_operation_activity={"vllm_moe_source": step.activity},
            by_operation={"vllm_moe_source": dynamic},
            by_rank_activity=rank_activity, by_rank=rank_energy,
            busy_s=step.latency_s,
            busy_j=float(getattr(self.hw_config, "busy_power_w", 0.) or 0.)
                   * len(step.ranks) * step.latency_s,
        )
        communications = []
        for index, (price, multiplicity) in enumerate(step.collectives):
            event = price.event
            axis = "ep" if event.name.startswith("moe.") else "tp"
            dim = self.hw_config.network_layout.parallelism_map[axis]
            row = CollectiveCommunicationEnergy(
                operation_id=f"{index}:{event.name}",
                dimension_index=self.hw_config.network_layout.dimensions.index(dim),
                collective_type=event.kind, collective_size_bytes=event.payload_bytes,
                transmitted_bytes=int(price.transmitted_bytes), chunks=1,
                active_link_count=len(event.participants),
                serialization_time_ns=price.seconds*1e9, network_transfers=1,
                completion_time_ns=price.seconds*1e9,
                link_dynamic_j=price.network_joules, active_network_j=0.,
            )
            communications.append(row.scaled(multiplicity))
        missing = []
        if step.assumptions.get("moe_grouped_traffic", "logical") == "logical":
            missing.append("grouped_gemm_smem_and_cta_traffic")
        coverage = EnergyCoverage(gpu_activity="partial" if missing else "complete",
                                  communication="complete", missing=tuple(missing))
        result = ExecutionEnergyResult(gpu, CommunicationEnergyResult(tuple(communications)),
                                       coverage, execution_mode="vllm_moe_source")
        if not math.isclose(result.total_j-gpu.busy_j, step.total_j, rel_tol=1e-10, abs_tol=1e-15):
            raise ValueError("MoE source energy adapter did not conserve system energy")
        return result

    def _price_moe_inference_step(self, phase, batch, seq, *, routing_step=0):
        original = self.O
        original_decode_scope = getattr(self, "_pricing_decode_template", False)
        self.O = (self.decode_launch_overhead(seq+1) if phase == "decode"
                  else self.stream_launch_overhead_s)

        self._decode_graph_pricing = (phase == "decode" and self.decode_uses_cuda_graph(seq+1)
                                      and _settings.get("RAPID_KERNEL_LAUNCH_MODE") == "graph_when_captured")

        self._pricing_decode_template = phase == "decode"
        try:
            world = int(self.tp)*int(self.moe_dp)
            step = self.price_moe_source_step(
                phase, batch, seq, ep=world if self.implementation_options.get("vllm_enable_expert_parallel", False) else 1,
                moe_collective_rule=self.model.moe_source_collective_rule,
                replay_steps=self.model.moe_source_replay_steps,
                moe_routing_step=routing_step,
            )
        finally:
            self.O = original
            self._pricing_decode_template = original_decode_scope
            self._decode_graph_pricing = False
        self.resolved_settings.setdefault("moe_source", {})[phase] = dict(step.assumptions)
        return step, self._moe_execution_energy(step)

    def _calc_moe_prefill_time(self):
        prefill_len = self.seq_len-self.model.decode_len
        if prefill_len <= 0:
            raise ValueError("qwen3_moe inference requires a nonempty prompt")
        step, result = self._price_moe_inference_step(
            "prefill", self._effective_transformer_batch(), prefill_len)
        overhead = self._inference_prefill_overhead()
        if overhead:
            result = add_idle_interval(result, overhead, self._static_power_w())
        self.prefill_execution_energy = result
        return step.latency_s+overhead, result.total_j

    def _calc_moe_decode_time(self):
        from simulate_inference_graph import DecodeGraph
        forwards = max(0, self.model.decode_len-1)
        if not forwards:
            self.decode_execution_energy = None
            return 0., 0., []
        interval = self.model.inference_sample_every
        if interval == -1:
            interval = 2**31-1
        cfg = InferenceConfig(
            batch_size=self._effective_transformer_batch(),
            seq_len=self.seq_len-self.model.decode_len, decode_len=forwards,
            hidden_dim=self.hidden_dim, num_heads=self.num_heads, kv_heads=self.kv_heads,
            intermediate_size=self.moe_intermediate_size, vocab_size=self.vocab_size,
            use_moe=True, num_experts=self.moe_num_experts, top_k=self.moe_top_k,
            sample_every=interval,
        )
        graph = DecodeGraph(cfg, self.hw_config, self._raw_model_config, type(self))
        samples = []
        for index in graph._generate_sample_points():
            past = cfg.seq_len+index
            step, result = self._price_moe_inference_step(
                "decode", cfg.batch_size, past, routing_step=index)
            launch = dict(mode=_settings.get("RAPID_KERNEL_LAUNCH_MODE"),
                          captured=self.decode_uses_cuda_graph(past+1),
                          source="vllm_moe_source", per_rank_launches=[len(r.kernels) for r in step.ranks])
            samples.append(DecodeSample(index, past+1, step.latency_s, result.total_j,
                                        past+1, result, launch))

        parts = [(samples[0].energy_result, 1.)]
        for previous, current in zip(samples, samples[1:]):
            gap = current.step_id-previous.step_id
            transition = (current.kernel_launch["mode"] == "graph_when_captured"
                          and current.kernel_launch["captured"] != previous.kernel_launch["captured"])
            if transition and gap != 1:
                raise ValueError("MoE decode capture transition must have adjacent samples")
            parts.extend(((previous.energy_result, (gap-1)/2),
                          (current.energy_result, (gap+1)/2)))
        result = combine_execution_energy(parts, execution_mode="sampled_moe_decode")
        total_time = result.gpu.wall_time_s
        overhead = self._inference_step_overhead()*forwards
        if overhead:
            result = add_idle_interval(result, overhead, self._static_power_w())
        self.decode_execution_energy = result
        self.resolved_settings["kernel_launch"]["decode_templates"] = [
            dict(sample.kernel_launch, step_id=sample.step_id) for sample in samples]
        self.resolved_settings["moe_source"]["decode"]["sampling"] = {
            "rule": "discrete_sum_of_linear_interpolant", "sample_every": interval,
            "sample_steps": [sample.step_id for sample in samples], "forward_count": forwards,
        }
        return total_time+overhead, result.total_j, samples

    def decode_only(self) -> bool:
        return bool(getattr(self.model, "inference_decode_only", False))

    def calc_time(self) -> Tuple[float, float]:
        if self._source_moe():
            if self.decode_only():
                raise ValueError("inference_param.decode_only: the source-MoE prefill is not simulated")
            return self._calc_moe_prefill_time()
        if getattr(self.network_model, "vllm_allreduce", None):
            self.network_model.vllm_allreduce["cuda_graph"] = False
            self.network_model.vllm_allreduce["kernel_overhead"] = self.stream_launch_overhead_s
        batch_size = self._effective_transformer_batch()
        vocab_size = self.vocab_size
        hidden_dim = self.hidden_dim
        decode_len = self.model.decode_len
        prefill_len = self.seq_len - decode_len
        num_heads = self.num_heads
        intermediate_size = self.intermediate_size
        kv_heads = self.kv_heads

        total_time = 0.0
        total_energy = 0.0
        prefill_peak_gb = 0.0
        mem_estimator = MemoryEstimator(self)

        if prefill_len <= 0 or self.decode_only():
            print("Skipping prefill (inference_param.decode_only)" if self.decode_only() else "Skipping prefill")
            self.workload = None
            self.pipeline_interconnect = None
            self.transformer_blocks = None
            self.transformer_analytical_time_forward = None
            self.transformer_analytical_time_backward = None
        else:
            num_SMs = self.hw_config.tech_config.core.num_bundles
            kernel_energy_marker = self._inference_kernel_energy_marker()
            transformer_timings, node_breakdown = self.compute_all_gemm_and_node_times(
                batch_size,
                vocab_size,
                hidden_dim,
                prefill_len,
                num_heads,
                kv_heads,
                intermediate_size,
                num_SMs,
                use_moe_override=False,
            )
            node_breakdown["inference_kernel_energy_records"] = (
                self._inference_kernel_energy_since(kernel_energy_marker)
            )
            moe_transformer_timings = None
            moe_node_breakdown = None
            if self.use_moe and any(getattr(self, "moe_layer_mask", []) or []):
                moe_transformer_timings, moe_node_breakdown = self.compute_all_gemm_and_node_times(
                    batch_size,
                    vocab_size,
                    hidden_dim,
                    prefill_len,
                    num_heads,
                    kv_heads,
                    self.moe_intermediate_size,
                    num_SMs,
                    use_moe_override=True,
                )

            total_energy = 0.0

            head_dim = getattr(self, "head_dim", None)
            if head_dim is None:
                head_dim = hidden_dim // num_heads
            token_bytes = llm_util.attention_kv_cache_token_bytes(
                getattr(self, "attention_type", "mha"),
                batch_size=batch_size,
                kv_heads=self.kv_heads,
                head_dim=head_dim,
                precision_bytes=self.precision.kv_cache,
                kv_lora_rank=getattr(self, "kv_lora_rank", None),
                num_heads=getattr(self, "num_heads", None),
                qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
                qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
                v_head_dim=getattr(self, "v_head_dim", None),
            )
            if getattr(self, "disable_kv_cache", False):
                token_bytes = 0.0

            workload, _ = self._prepare_execution_graphs(
                node_breakdown=node_breakdown,
                transformer_timings=transformer_timings,
                moe_node_breakdown=moe_node_breakdown,
                moe_transformer_timings=moe_transformer_timings,
                batch_size=batch_size,
                seq_len=prefill_len,
                hidden_dim=hidden_dim,
                intermediate_size=intermediate_size,
                vocab_size=vocab_size,
                include_pipeline_backward=False,
                include_transformer_backward=False,
            )

            self.workload = workload
            self.pipeline_interconnect = dict(workload.interconnect)
            self.transformer_blocks = workload.blocks
            self.transformer_analytical_time_forward = node_breakdown.get("transformer_time_f")
            self.transformer_analytical_time_backward = None

            dispatcher = LLMExecutionDispatcher(self, workload)
            mode = self.execution_mode
            try:
                result = dispatcher.run(mode)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"{exc}. Selected execution mode '{mode.value}'."
                ) from exc

            total_time = result.total_time
            if result.energy is None:
                raise RuntimeError("prefill execution did not return energy")
            self.prefill_execution_energy = result.energy
            self.prefill_execution_energy = apply_inference_gpu_accounting(
                self.prefill_execution_energy,
                node_breakdown.get("inference_kernel_energy_records", ()),
                layer_count=self.num_layers,
            )
            total_energy = result.energy.total_j
            total_energy = self.prefill_execution_energy.total_j

            prefill_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=prefill_len,
                kv_cache_tokens=0 if getattr(self, "disable_kv_cache", False) else prefill_len,
            )
            prefill_program = dispatcher.build_flat_program_for_memory()
            _, prefill_peak_gb = mem_estimator.simulate_peak(
                prefill_program,
                prefill_memory_data,
                mode="inference",
                filename="memory_graph_prefill",
            )

        decode_peak_gb = prefill_peak_gb
        if decode_len > 0:
            decode_gemm_shapes = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=batch_size,
                current_seq_len=self.seq_len,
                d_model=hidden_dim,
                num_heads=num_heads,
                kv_heads=kv_heads,
                intermediate_size=self.moe_intermediate_size if self.use_moe else intermediate_size,
                vocab_size=vocab_size,
                model_type=self.model_type,
            )
            (decode_workload, _), _ = self.prepare_decode_graphs(
                batch_size=batch_size,
                total_seq_len=self.seq_len,
                gemm_shapes=decode_gemm_shapes,
            )
            decode_dispatcher = LLMExecutionDispatcher(self, decode_workload)
            decode_memory_program = decode_dispatcher.build_flat_program_for_memory()
            decode_memory_data = mem_estimator.build_memory_data(
                mode="inference",
                batch_size=batch_size,
                seq_len=1,
                gemm_shapes=decode_gemm_shapes,
                kv_cache_tokens=self.seq_len,
            )
            _, decode_peak_gb = mem_estimator.simulate_peak(
                decode_memory_program,
                decode_memory_data,
                mode="inference",
                filename="memory_graph_decode",
            )

        max_peak_gb = max(prefill_peak_gb, decode_peak_gb)
        self.memory_peak_gb = max_peak_gb

        hardware_mem_bytes = getattr(self.DRAM, "size", None)
        if hardware_mem_bytes is None and hasattr(self.hw_config, "tech_config"):
            tech_cfg = self.hw_config.tech_config
            if hasattr(tech_cfg, "DRAM"):
                hardware_mem_bytes = getattr(tech_cfg.DRAM, "size", None)

        if hardware_mem_bytes is not None:
            hardware_mem_gib = float(hardware_mem_bytes) / float(1024 ** 3)
            self.memory_capacity_per_device_gb = hardware_mem_gib
            mem_delta = hardware_mem_gib - max_peak_gb
            self.memory_headroom_gb = mem_delta
            self.memory_capacity_exceeded = mem_delta < 0
            self.memory_capacity_violation_gb = abs(mem_delta) if mem_delta < 0 else 0.0

            memory_dir = os.path.join(self.output_dir, "memory-summary")
            os.makedirs(memory_dir, exist_ok=True)
            info_lines = [
                "Simulation mode: inference",
                f"Hardware memory capacity (per gpu): {hardware_mem_gib:.2f} GiB",
                ("Prefill peak memory usage (per gpu): not simulated (inference_param.decode_only)"
                 if self.decode_only() else f"Prefill peak memory usage (per gpu): {prefill_peak_gb:.2f} GiB"),
                f"Final decode peak memory usage (per gpu): {decode_peak_gb:.2f} GiB",
                f"Max peak memory usage (per gpu): {max_peak_gb:.2f} GiB",
            ]
            if mem_delta < 0:
                info_lines.append(
                    f"[WARN] Peak memory exceeds capacity by {abs(mem_delta):.2f} GiB"
                )
            else:
                info_lines.append(f"Remaining memory headroom: {mem_delta:.2f} GiB")
            info_path = os.path.join(memory_dir, "memory_capacity_comparison.txt")
            with open(info_path, "w", encoding="utf-8") as info_file:
                info_file.write("\n".join(info_lines) + "\n")
        else:
            self.memory_capacity_per_device_gb = None
            self.memory_headroom_gb = None
            self.memory_capacity_exceeded = False
            self.memory_capacity_violation_gb = 0.0

        step_overhead = self._inference_prefill_overhead()
        if step_overhead > 0.0 and getattr(self, "prefill_execution_energy", None) is not None:
            total_time += step_overhead
            prefill_result = getattr(self, "prefill_execution_energy", None)
            if prefill_result is not None:
                self.prefill_execution_energy = add_idle_interval(
                    prefill_result, step_overhead, self._static_power_w()
                )
                total_energy = self.prefill_execution_energy.total_j
        return total_time, total_energy

    def _inference_step_overhead(self) -> float:
        sw = getattr(self.hw_config, "sw_config", None)
        return float(getattr(sw, "inference_step_overhead", 0.0) or 0.0)

    def _inference_prefill_overhead(self) -> float:
        sw = getattr(self.hw_config, "sw_config", None)
        value = getattr(sw, "inference_prefill_overhead", None)
        return self._inference_step_overhead() if value is None else float(value)

    def _static_power_w(self) -> float:
        return float(getattr(self.hw_config, "static_power_w", 0.0) or 0.0)

    def calc_decode_time(self) -> Tuple[float, List[DecodeSample]]:
        if self._source_moe():
            return self._calc_moe_decode_time()

        sample_every = self.model.inference_sample_every
        if sample_every == -1:
            sample_every = 2**31 - 1

        decode_len = max(0, self.model.decode_len - 1)
        if decode_len == 0:
            print("Skipping decode")
            self.decode_execution_energy = None
            return 0.0, 0.0, []

        inference_config = InferenceConfig(
            batch_size=self._effective_transformer_batch(),
            seq_len=self.seq_len - self.model.decode_len,
            decode_len=decode_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            kv_heads=self.kv_heads,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            use_moe=self.use_moe,
            num_experts=self.moe_num_experts,
            top_k=self.moe_top_k,
            sample_every=sample_every,
        )

        inference_engine = InferenceEngine(
            config=inference_config,
            hw_config=self.hw_config,
            model_config=self._raw_model_config,
            time_calc_cls=TimeCalculationLLMInference,
        )

        decode_time, decode_energy, decode_samples = inference_engine._build_decode_graph()
        from program.allreduce_curve import merge_usage
        for sample in decode_samples:
            merge_usage(self, sample.allreduce_usage)
        self.resolved_settings["kernel_launch"]["decode_templates"] = [
            {**sample.kernel_launch, "step_id": sample.step_id}
            for sample in decode_samples if sample.kernel_launch is not None]
        self.decode_execution_energy = inference_engine.decode_graph.execution_energy_result

        step_overhead = self._inference_step_overhead()
        if step_overhead > 0.0:
            idle_s = step_overhead * decode_len
            decode_time += idle_s
            if self.decode_execution_energy is not None:
                self.decode_execution_energy = add_idle_interval(
                    self.decode_execution_energy, idle_s, self._static_power_w()
                )
                decode_energy = self.decode_execution_energy.total_j
        return decode_time, decode_energy, decode_samples

    def calc_total_inference_time(self) -> dict:
        decode_only = self.decode_only()
        if decode_only:
            self.prefill_execution_energy = None
            if not self._source_moe():
                self.calc_time()
            prefill_time = prefill_energy = None
        else:
            prefill_time, prefill_energy = self.calc_time()

        decode_time, decode_energy, decode_samples = self.calc_decode_time()
        total_time = None if decode_only else prefill_time + decode_time

        time_to_first_token = prefill_time

        head_dim = getattr(self, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_dim // self.num_heads
        token_bytes = llm_util.attention_kv_cache_token_bytes(
            getattr(self, "attention_type", "mha"),
            batch_size=self._effective_transformer_batch(),
            kv_heads=self.kv_heads,
            head_dim=head_dim,
            precision_bytes=self.precision.kv_cache,
            kv_lora_rank=getattr(self, "kv_lora_rank", None),
            num_heads=getattr(self, "num_heads", None),
            qk_nope_head_dim=getattr(self, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(self, "qk_rope_head_dim", None),
            v_head_dim=getattr(self, "v_head_dim", None),
        )
        if getattr(self, "disable_kv_cache", False):
            token_bytes = 0.0
        prefill_len = self.seq_len - self.model.decode_len
        decode_len = self.model.decode_len
        num_layers = self.num_layers

        prefill_store_bytes = None if decode_only else token_bytes * prefill_len * num_layers
        decode_store_bytes = token_bytes * decode_len * num_layers
        decode_fetch_bytes = token_bytes * num_layers * (
            decode_len * (prefill_len + self.seq_len) // 2
        )

        def _to_gib(byte_val: int) -> str:
            gib_val = byte_val / (1024 ** 3)
            if gib_val > 1024:
                tib_val = gib_val / 1024
                return f"{tib_val:.1f} TiB"
            return f"{gib_val:.1f} GiB"

        if decode_samples:
            decode_rates = self._decode_token_rates(decode_samples, decode_len, decode_time, self.batch_size, step_overhead=self._inference_step_overhead())
        else:
            decode_rates = None

        if decode_only:
            print(f"[prefill] not simulated (inference_param.decode_only), [decode] time: {decode_time:.4f}s")
        else:
            print(
                f"[prefill] time: {prefill_time:.4f}s, "
                f"[decode] time: {decode_time:.4f}s, "
                f"[total] time: {total_time:.4f}s"
            )
        if not getattr(self, "disable_kv_cache", False):
            print(
                f"[kv-cache] prefill_store="
                f"{'not charged (decode_only)' if decode_only else _to_gib(prefill_store_bytes)}, "
                f"decode_store={_to_gib(decode_store_bytes)}, "
                f"decode_fetch={_to_gib(decode_fetch_bytes)}"
            )

        total_energy = decode_energy if decode_only else prefill_energy + decode_energy
        phase_energy = {}
        energy_parts = []
        prefill_result = getattr(self, "prefill_execution_energy", None)
        if prefill_result is not None:
            phase_energy["prefill"] = prefill_result
            energy_parts.append((prefill_result, 1.0))
        decode_result = getattr(self, "decode_execution_energy", None)
        if decode_result is not None:
            phase_energy["decode"] = decode_result
            energy_parts.append((decode_result, 1.0))
        if energy_parts:
            self.execution_energy = combine_execution_energy(
                energy_parts,
                execution_mode="inference",
            )
            total_energy = self.execution_energy.total_j
            write_energy_report(
                self.output_dir,
                self.execution_energy,
                hardware=self.hw_config,
                phases=phase_energy,
                metadata={
                    "run_type": "inference",
                    "prefill_generates_first_token": True,
                    "requested_output_tokens": self.model.decode_len,
                    "decode_forward_executions": max(0, self.model.decode_len - 1),
                    "time_to_first_token_s": time_to_first_token,
                    "gpu_activity_model": self.activity_model_provenance(),
                    "kernel_launch": self.resolved_settings["kernel_launch"],
                    **({"moe_source": self.resolved_settings["moe_source"]}
                       if self._source_moe() else {}),
                    **({"decode_only": True, "prefill": "not_simulated", "kv_cache_prefill_store": "not_charged"}
                       if decode_only else {}),
                },
            )

        decode_forwards = max(0, decode_len - 1)
        prefill_denom = self._effective_transformer_batch() * prefill_len
        decode_denom = self._effective_transformer_batch() * decode_forwards
        total_denom = self._effective_transformer_batch() * (prefill_len + decode_forwards)
        decode_energy_tok = decode_energy / decode_denom if decode_denom > 0 else 0.0
        if decode_only:
            print(f"[prefill] energy: not simulated (inference_param.decode_only)",
                  f"[decode] energy: {convert_prefix(decode_energy)}J, energy/tok: {convert_prefix(decode_energy_tok)}J")
        else:
            prefill_energy_tok = prefill_energy / prefill_denom if prefill_denom > 0 else 0.0
            total_energy_tok = total_energy / total_denom if total_denom > 0 else 0.0
            print(
                f"[prefill] energy: {convert_prefix(prefill_energy)}J, energy/tok: {convert_prefix(prefill_energy_tok)}J",
                f"[decode] energy: {convert_prefix(decode_energy)}J, energy/tok: {convert_prefix(decode_energy_tok)}J",
                f"[total] energy: {convert_prefix(total_energy)}J, energy/tok: {convert_prefix(total_energy_tok)}J",
            )

        return {
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "total_inference_time": total_time,
            "time_to_first_token": time_to_first_token,
            "kv_cache_prefill_store_bytes": prefill_store_bytes,
            "kv_cache_decode_store_bytes": decode_store_bytes,
            "kv_cache_decode_fetch_bytes": decode_fetch_bytes,
            "decode_tokens_per_s": decode_rates,
        }

    @staticmethod
    def _decode_token_rates(
        samples: List[DecodeSample],
        decode_len: int,
        total_decode_time: float,
        batch_size: int,
        step_overhead: float = 0.0,
    ) -> Dict[str, float]:
        if decode_len <= 0:
            return {}

        def token_time_at(step: int) -> float:
            if not samples:
                return 0.0
            if step <= samples[0].step_id:
                return samples[0].execution_time
            for idx in range(1, len(samples)):
                prev = samples[idx - 1]
                curr = samples[idx]
                if step <= curr.step_id:
                    gap = curr.step_id - prev.step_id
                    if gap <= 0:
                        return curr.execution_time
                    ratio = (step - prev.step_id) / gap
                    return prev.execution_time + ratio * (curr.execution_time - prev.execution_time)
            return samples[-1].execution_time

        def safe_rate(token_time: float) -> float:
            token_time = token_time + step_overhead
            if token_time <= 0.0:
                return 0.0
            return 1.0 / token_time

        last_step = max(decode_len - 1, 0)
        mid_step = decode_len // 2

        start_rate = safe_rate(token_time_at(0))
        mid_rate = safe_rate(token_time_at(mid_step))
        end_rate = safe_rate(token_time_at(last_step))

        overall_rate = 0.0
        if total_decode_time > 0.0:
            overall_rate = max(0, decode_len - 1) / total_decode_time

        return {
            "start": start_rate,
            "midpoint": mid_rate,
            "end": end_rate,
            "midpoint_step": mid_step,
            "overall": overall_rate,
        }

__all__ = ["TimeCalculationLLMInference"]
