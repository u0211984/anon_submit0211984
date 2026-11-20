import math
import os
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple

import llm_util
from program.layout import RankLayout, cluster_coords
from program import _env_flag

class MemKind(Enum):
    TRANSFORMER = "transformer"
    EMBEDDING = "embedding"
    SOFTMAX = "softmax"
    OPTIMIZER = "optimizer"
    LAYERNORM1 = "layernorm1"
    QKV_PROJ = "qkv_proj"
    ROPE = "rope"
    ATTENTION = "attention"
    OUTPUT_PROJ = "output_proj"
    LAYERNORM2 = "layernorm2"
    MLP = "MLP"
    ROUTER = "router"
    MOE_DISPATCH = "moe_dispatch"
    MOE_COMBINE = "moe_combine"

_OP_NAME_TO_MEM_KIND: Dict[str, MemKind] = {
    kind.value: kind for kind in MemKind
}

def mem_kind_from_op_name(op_name: str) -> MemKind:
    if op_name in _OP_NAME_TO_MEM_KIND:
        return _OP_NAME_TO_MEM_KIND[op_name]
    raise ValueError(f"Unsupported mem_kind op name: {op_name!r}")

TRANSFORMER_OP_KINDS: FrozenSet[MemKind] = frozenset(
    {
        MemKind.LAYERNORM1,
        MemKind.QKV_PROJ,
        MemKind.ROPE,
        MemKind.ATTENTION,
        MemKind.OUTPUT_PROJ,
        MemKind.LAYERNORM2,
        MemKind.MLP,
        MemKind.MOE_DISPATCH,
        MemKind.MOE_COMBINE,
        MemKind.TRANSFORMER,
    }
)

NON_TRANSFORMER_KINDS: FrozenSet[MemKind] = frozenset(
    {
        MemKind.EMBEDDING,
        MemKind.SOFTMAX,
        MemKind.OPTIMIZER,
    }
)

class MemoryEstimator:
    def __init__(self, time_calc: Any) -> None:
        self.time_calc = time_calc

    def build_memory_data(
        self,
        *,
        mode: str,
        batch_size: int,
        seq_len: int,
        gemm_shapes: Optional[Dict[str, Any]] = None,
        kv_cache_tokens: Optional[int] = None,
        zero3_ephemeral_peak_bytes: Optional[float] = None,
    ) -> Dict[str, Any]:
        tc = self.time_calc
        precision = tc.precision

        dp = max(1, int(getattr(tc, "dp", 1)))
        dp_layout = dp
        tp = max(1, int(getattr(tc, "tp", 1)))
        pp = max(1, int(getattr(tc, "pp", 1)))
        cp = max(1, int(getattr(tc, "cp", 1)))
        ep = max(1, int(getattr(tc, "ep", 1)))
        tp_ep = bool(getattr(tc, "tp_ep", True))
        moe_group = ep
        zero_stage = int(getattr(tc, "zero_stage", 0) or 0)
        flash_attention = bool(getattr(tc, "flash_attention", False))
        full_recomputation = bool(getattr(tc, "full_recomputation", False))
        use_moe = bool(getattr(tc, "use_moe", False))
        disable_embedding_unembedding = bool(
            getattr(tc, "disable_embedding_unembedding", False)
        )
        if mode == "inference":
            dp = 1
            tp_ep = False
            if use_moe and hasattr(tc, "_moe_routing_group"):
                moe_group = max(1, int(tc._moe_routing_group()))

        zero_moe = mode == "training" and use_moe and zero_stage >= 2
        if use_moe and zero_stage >= 2 and not zero_moe:
            raise NotImplementedError("MoE memory estimation with ZeRO-2/3 (dp_zero_stage >= 2) is not supported yet.")

        checkpoint_shards = cp * (tp if bool(getattr(tc, "tp_sp", False)) else 1)
        checkpoint_input_bytes = (
            float(batch_size) * math.ceil(float(seq_len) / checkpoint_shards)
            * float(tc.hidden_dim) * float(precision.activations)
            if mode == "training" and full_recomputation else 0.0
        )

        seq_len_eff = float(seq_len) / float(cp)

        (
            transformer_mem_layer_dense,
            transformer_act_layer_dense,
            transformer_act_layer_inf_dense,
            transformer_static_layer_dense,
            gradient_mem_layer_dense,
            optimizer_mem_layer_dense,
            weight_memory_layer_dense,
        ) = llm_util.get_transformer_mem_layer(
            dp=dp,
            tp=tp,
            cp=cp,
            pp=pp,
            mb=max(1, int(getattr(tc, "mb", 1))),
            batch_size=batch_size,
            hidden_dim=tc.hidden_dim,
            seq_len=seq_len_eff,
            intermediate_size=tc.intermediate_size,
            n_heads=tc.num_heads,
            kv_heads=tc.kv_heads,
            head_dim=getattr(tc, "head_dim", None),
            precision=precision,
            model_type=tc.model_type,
            zero_stage=zero_stage,
            flash_attention=flash_attention,
            full_recomputation=full_recomputation,
            attention_type=getattr(tc, "attention_type", "mha"),
            q_lora_rank=getattr(tc, "q_lora_rank", None),
            kv_lora_rank=getattr(tc, "kv_lora_rank", None),
            qk_nope_head_dim=getattr(tc, "qk_nope_head_dim", None),
            qk_rope_head_dim=getattr(tc, "qk_rope_head_dim", None),
            v_head_dim=getattr(tc, "v_head_dim", None),
            run_type=getattr(tc, "run_type", "training"),
            key_seq_len=(kv_cache_tokens if mode == "inference" and kv_cache_tokens is not None else seq_len),
            swiglu_mlp=getattr(tc, "swiglu_mlp", False),
        )
        transformer_mem_layer_moe = transformer_mem_layer_dense
        transformer_act_layer_moe = transformer_act_layer_dense
        transformer_act_layer_inf_moe = transformer_act_layer_inf_dense
        transformer_static_layer_moe = transformer_static_layer_dense
        gradient_mem_layer_moe = gradient_mem_layer_dense
        optimizer_mem_layer_moe = optimizer_mem_layer_dense
        weight_memory_layer_moe = weight_memory_layer_dense

        (
            _total_params_per_rank,
            max_layer_params,
            params_per_layer_per_rank,
            embedding_params_per_rank,
            output_params_per_rank,
        ) = tc._param_stats_per_rank(tc.hidden_dim, tc.intermediate_size, tc.vocab_size)

        if zero3_ephemeral_peak_bytes is None:
            if zero_stage >= 3 and dp > 1:
                zero3_ephemeral_peak_bytes = max_layer_params * precision.parameters
            else:
                zero3_ephemeral_peak_bytes = 0.0

        extra_static_bytes_per_device: Dict[int, float] = {}

        bytes_per_param_weight = 0.0
        bytes_per_param_grad = 0.0
        bytes_per_param_opt = 0.0
        if params_per_layer_per_rank:
            denom = float(params_per_layer_per_rank)
            bytes_per_param_weight = float(weight_memory_layer_dense) / denom
            if mode == "training":
                bytes_per_param_grad = float(gradient_mem_layer_dense) / denom
                bytes_per_param_opt = float(optimizer_mem_layer_dense) / denom

        per_param_static_bytes = (
            bytes_per_param_weight + bytes_per_param_grad + bytes_per_param_opt
        )
        embedding_static_bytes = float(embedding_params_per_rank) * per_param_static_bytes
        lm_head_static_bytes = float(output_params_per_rank) * per_param_static_bytes
        if disable_embedding_unembedding:
            embedding_static_bytes = 0.0
            lm_head_static_bytes = 0.0

        from program import megatron as _megatron
        megatron_state = (_megatron.static_bytes_per_param(tc)
                          if mode == "training" and not use_moe else None)
        if megatron_state is not None:
            layer_params = float(params_per_layer_per_rank)
            weight_memory_layer_dense, gradient_mem_layer_dense, optimizer_mem_layer_dense = (
                b * layer_params for b in megatron_state)
            transformer_static_layer_dense = (
                weight_memory_layer_dense + gradient_mem_layer_dense + optimizer_mem_layer_dense)
            transformer_mem_layer_dense = transformer_act_layer_dense + weight_memory_layer_dense
            transformer_mem_layer_moe = transformer_mem_layer_dense
            transformer_static_layer_moe = transformer_static_layer_dense
            gradient_mem_layer_moe = gradient_mem_layer_dense
            optimizer_mem_layer_moe = optimizer_mem_layer_dense
            weight_memory_layer_moe = weight_memory_layer_dense
            if not disable_embedding_unembedding:
                embedding_static_bytes = float(embedding_params_per_rank) * sum(megatron_state)
                lm_head_static_bytes = float(output_params_per_rank) * sum(megatron_state)

        if use_moe:
            moe_intermediate = int(getattr(tc, "moe_intermediate_size", tc.intermediate_size))
            (
                _,
                transformer_act_layer_moe,
                transformer_act_layer_inf_moe,
                _,
                _,
                _,
                _,
            ) = llm_util.get_transformer_mem_layer(
                dp=dp,
                tp=tp,
                cp=cp,
                pp=pp,
                mb=max(1, int(getattr(tc, "mb", 1))),
                batch_size=batch_size,
                hidden_dim=tc.hidden_dim,
                seq_len=seq_len_eff,
                intermediate_size=moe_intermediate,
                n_heads=tc.num_heads,
                kv_heads=tc.kv_heads,
                head_dim=getattr(tc, "head_dim", None),
                precision=precision,
                model_type=tc.model_type,
                zero_stage=zero_stage,
                flash_attention=flash_attention,
                full_recomputation=full_recomputation,
                attention_type=getattr(tc, "attention_type", "mha"),
                q_lora_rank=getattr(tc, "q_lora_rank", None),
                kv_lora_rank=getattr(tc, "kv_lora_rank", None),
                qk_nope_head_dim=getattr(tc, "qk_nope_head_dim", None),
                qk_rope_head_dim=getattr(tc, "qk_rope_head_dim", None),
                v_head_dim=getattr(tc, "v_head_dim", None),
                run_type=getattr(tc, "run_type", "training"),
                key_seq_len=(kv_cache_tokens if mode == "inference" and kv_cache_tokens is not None else seq_len),
                swiglu_mlp=getattr(tc, "swiglu_mlp", False),
            )
            ffn_proj_factor = 3 if llm_util.uses_gated_mlp(tc.model_type, swiglu_mlp=getattr(tc, "swiglu_mlp", False)) else 2
            if getattr(tc, "attention_type", "mha") == "mla":
                attention_params = llm_util.mla_attention_params_per_rank(
                    tc.hidden_dim,
                    tc.num_heads,
                    tc.q_lora_rank,
                    tc.kv_lora_rank,
                    tc.qk_nope_head_dim,
                    tc.qk_rope_head_dim,
                    tc.v_head_dim,
                    tp=tp,
                )["total"]
            else:
                _head_dim, q_size, kv_size = llm_util.attention_dim_sizes(
                    tc.hidden_dim,
                    tc.num_heads,
                    tc.kv_heads,
                    head_dim=getattr(tc, "head_dim", None),
                )
                attention_params = (
                    (tc.hidden_dim * (q_size + 2 * kv_size)) + (q_size * tc.hidden_dim)
                ) / float(max(1, tp))
            expert_param = ffn_proj_factor * moe_intermediate * tc.hidden_dim
            routed_experts = int(getattr(tc, "moe_num_experts", 1))
            shared_experts = int(getattr(tc, "n_shared_experts", 0))
            routed_experts_per_rank = routed_experts / float(max(1, moe_group))

            if tp_ep:
                routed_params_per_rank = (expert_param / float(max(1, tp))) * routed_experts_per_rank
            else:
                routed_params_per_rank = expert_param * routed_experts_per_rank
            if tp_ep:
                shared_params_per_rank = (expert_param / float(max(1, tp))) * shared_experts
            else:
                shared_params_per_rank = expert_param * shared_experts
            expert_params_per_rank = routed_params_per_rank + shared_params_per_rank
            router_params_per_rank = tc.hidden_dim * routed_experts
            params_per_layer_moe = (
                attention_params + expert_params_per_rank + router_params_per_rank
            )
            weight_memory_layer_moe = params_per_layer_moe * bytes_per_param_weight
            if mode == "training":
                gradient_mem_layer_moe = params_per_layer_moe * bytes_per_param_grad
                optimizer_mem_layer_moe = params_per_layer_moe * bytes_per_param_opt
            else:
                gradient_mem_layer_moe = 0.0
                optimizer_mem_layer_moe = 0.0
            transformer_static_layer_moe = weight_memory_layer_moe + gradient_mem_layer_moe + optimizer_mem_layer_moe

        zero_moe_gather_bytes = None
        if zero_moe:
            dense_group, expert_group = dp * ep, dp
            dense_params, _ = tc.zero_moe_layer_param_groups(
                tc.hidden_dim, tc.intermediate_size, moe=False)
            moe_dense_params, routed_params = tc.zero_moe_layer_param_groups(
                tc.hidden_dim, tc.moe_intermediate_size, moe=True)

            def state_bytes(dense, expert):
                shard = dense / dense_group + expert / expert_group
                weights = precision.parameters * (shard if zero_stage >= 3 else dense + expert)
                gradients = precision.gradients * shard

                optimizer = (2 * precision.optimizer_states + precision.activations
                             + precision.master_parameters) * shard
                return weights, gradients, optimizer

            weight_memory_layer_dense, gradient_mem_layer_dense, optimizer_mem_layer_dense = state_bytes(dense_params, 0)
            weight_memory_layer_moe, gradient_mem_layer_moe, optimizer_mem_layer_moe = state_bytes(moe_dense_params, routed_params)
            transformer_static_layer_dense = weight_memory_layer_dense + gradient_mem_layer_dense + optimizer_mem_layer_dense
            transformer_static_layer_moe = weight_memory_layer_moe + gradient_mem_layer_moe + optimizer_mem_layer_moe
            transformer_mem_layer_dense = transformer_act_layer_dense + weight_memory_layer_dense
            embedding_static_bytes = sum(state_bytes(embedding_params_per_rank, 0)) if not disable_embedding_unembedding else 0.0
            lm_head_static_bytes = sum(state_bytes(output_params_per_rank, 0)) if not disable_embedding_unembedding else 0.0

            def materialized_bytes(dense, expert):
                if zero_stage < 3:
                    return 0.0
                return precision.parameters * ((dense if dense_group > 1 else 0.0)
                                               + (expert if expert_group > 1 else 0.0))

            zero_moe_gather_bytes = {
                "dense": materialized_bytes(dense_params, 0),
                "moe": materialized_bytes(moe_dense_params, routed_params),
                "embedding": materialized_bytes(embedding_params_per_rank, 0) if not disable_embedding_unembedding else 0.0,
                "softmax": materialized_bytes(output_params_per_rank, 0) if not disable_embedding_unembedding else 0.0,
            }
            zero3_ephemeral_peak_bytes = max(zero_moe_gather_bytes.values())
            tc.zero3_ephemeral_peak_bytes = zero3_ephemeral_peak_bytes

        megatron_groups = (_megatron.static_bytes_per_param_groups(tc)
                           if mode == "training" and use_moe and not zero_moe else None)
        if megatron_groups is not None:
            w_b, g_b, opt_dense, opt_expert = megatron_groups
            dense_params, _ = tc.zero_moe_layer_param_groups(tc.hidden_dim, tc.intermediate_size, moe=False)
            moe_dense_params, routed_params = tc.zero_moe_layer_param_groups(
                tc.hidden_dim, tc.moe_intermediate_size, moe=True)
            weight_memory_layer_dense = w_b * dense_params
            gradient_mem_layer_dense = g_b * dense_params
            optimizer_mem_layer_dense = opt_dense * dense_params
            weight_memory_layer_moe = w_b * (moe_dense_params + routed_params)
            gradient_mem_layer_moe = g_b * (moe_dense_params + routed_params)
            optimizer_mem_layer_moe = opt_dense * moe_dense_params + opt_expert * routed_params
            transformer_static_layer_dense = weight_memory_layer_dense + gradient_mem_layer_dense + optimizer_mem_layer_dense
            transformer_static_layer_moe = weight_memory_layer_moe + gradient_mem_layer_moe + optimizer_mem_layer_moe
            transformer_mem_layer_dense = transformer_act_layer_dense + weight_memory_layer_dense
            if not disable_embedding_unembedding:
                embedding_static_bytes = float(embedding_params_per_rank) * (w_b + g_b + opt_dense)
                lm_head_static_bytes = float(output_params_per_rank) * (w_b + g_b + opt_dense)

        const_mem_offset_bytes = 0.0
        hw_config = getattr(tc, "hw_config", None)
        sw_config = getattr(hw_config, "sw_config", None) if hw_config is not None else None
        if sw_config is not None:
            const_mem_offset_bytes = float(getattr(sw_config, "const_mem_offset", 0.0) or 0.0)

        def _build_rank_layout() -> Optional[RankLayout]:
            hw_config = getattr(tc, "hw_config", None)
            layout = getattr(hw_config, "network_layout", None)
            dimensions = getattr(layout, "dimensions", None) if layout is not None else None
            if not dimensions:
                return None

            ep = max(1, int(getattr(tc, "ep", 1)))
            axis_sizes = {"tp": tp, "cp": cp, "ep": ep, "pp": pp, "dp": dp_layout}
            rank_layout, _ = RankLayout.from_network_layout(
                layout,
                axis_sizes,
                False,
                extract_optimize_2dmap=False,
                unsupported_axis_context="memory estimation",
                empty_axis_order_is_none=True,
            )
            return rank_layout

        def _hw_id_for_rank(stage_id: int, tp_rank: int, layout: Optional[RankLayout]):
            if layout is None:
                return stage_id * par_degree + tp_rank
            axis_sizes = layout.axis_sizes
            coords = cluster_coords(
                layout.axis_order,
                tp_rank,
                stage_id,
                sizes=axis_sizes,
            )
            return layout.linearize(coords)

        par_degree = max(1, tp * cp * ep)
        layout = None
        if embedding_static_bytes or lm_head_static_bytes or const_mem_offset_bytes:
            layout = _build_rank_layout()
        if embedding_static_bytes or lm_head_static_bytes:
            last_stage = max(0, pp - 1)
            for tp_rank in range(par_degree):
                if embedding_static_bytes:
                    hw_id = _hw_id_for_rank(0, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = extra_static_bytes_per_device.get(hw_id, 0.0) + embedding_static_bytes
                if lm_head_static_bytes:
                    hw_id = _hw_id_for_rank(last_stage, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = extra_static_bytes_per_device.get(hw_id, 0.0) + lm_head_static_bytes
        if const_mem_offset_bytes:
            for stage_id in range(pp):
                for tp_rank in range(par_degree):
                    hw_id = _hw_id_for_rank(stage_id, tp_rank, layout)
                    extra_static_bytes_per_device[hw_id] = (
                        extra_static_bytes_per_device.get(hw_id, 0.0) + const_mem_offset_bytes
                    )

        kv_cache_bytes_per_layer = 0.0
        if gemm_shapes is None:
            gemm_shapes_moe = llm_util.process_gemm_shapes(
                tc,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=tc.hidden_dim,
                num_heads=tc.num_heads,
                kv_heads=tc.kv_heads,
                intermediate_size=tc.intermediate_size,
                vocab_size=tc.vocab_size,
            )
        else:
            gemm_shapes_moe = gemm_shapes

        gemm_shapes_dense = gemm_shapes_moe
        if use_moe:
            from types import SimpleNamespace

            dense_ctx = SimpleNamespace(
                use_moe=False,
                moe_num_experts=int(getattr(tc, "moe_num_experts", 1)),
                moe_top_k=int(getattr(tc, "moe_top_k", 1)),
                moe_intermediate_size=tc.intermediate_size,
                model_type=tc.model_type,
                head_dim=getattr(tc, "head_dim", None),
                attention_type=getattr(tc, "attention_type", "mha"),
                q_lora_rank=getattr(tc, "q_lora_rank", None),
                kv_lora_rank=getattr(tc, "kv_lora_rank", None),
                qk_nope_head_dim=getattr(tc, "qk_nope_head_dim", None),
                qk_rope_head_dim=getattr(tc, "qk_rope_head_dim", None),
                v_head_dim=getattr(tc, "v_head_dim", None),
            )
            gemm_shapes_dense = llm_util.process_gemm_shapes(
                dense_ctx,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=tc.hidden_dim,
                num_heads=tc.num_heads,
                kv_heads=tc.kv_heads,
                intermediate_size=tc.intermediate_size,
                vocab_size=tc.vocab_size,
            )

        gemm_type_map = {
            "qkv_proj": "qkv",
            "attention_score": "attention_score",
            "attention_output": "attention_output",
            "output_proj": "out_proj",
            "ffn1": "ffn1",
            "ffn2": "ffn2",
            "linear": "linear_softmax",
            "router": "ffn1",
        }

        def _build_activation_maps(
            gemm_shapes_local: Dict[str, Any],
            transformer_act_layer_local: float,
            transformer_act_layer_inf_local: float,
            use_moe_local: bool,
        ) -> Tuple[Dict[MemKind, float], Dict[MemKind, float], Dict[MemKind, float]]:
            def _gemm_out_bytes(key: str, gemm_type: str) -> float:
                if key not in gemm_shapes_local:
                    raise RuntimeError(f"Missing GEMM shape for '{key}'")
                elements = tc._gemm_output_elements(gemm_shapes_local[key], gemm_type)
                return float(elements) * precision.activations

            qkv_bytes = _gemm_out_bytes("qkv_proj", gemm_type_map["qkv_proj"])
            attn_out_bytes = _gemm_out_bytes("attention_output", gemm_type_map["attention_output"])
            out_proj_bytes = _gemm_out_bytes("output_proj", gemm_type_map["output_proj"])
            ffn1_bytes = _gemm_out_bytes("ffn1", gemm_type_map["ffn1"])
            ffn2_bytes = _gemm_out_bytes("ffn2", gemm_type_map["ffn2"])
            attn_score_bytes = 0.0
            if not flash_attention:
                attn_score_bytes = _gemm_out_bytes("attention_score", gemm_type_map["attention_score"])

            softmax_out_bytes = 0.0
            if mode == "inference" and _env_flag("RAPID_PREFILL_LOGITS_LAST_TOKEN"):
                vocab_shard = math.ceil(float(tc.vocab_size) / float(max(1, tp * cp)))
                softmax_out_bytes = float(batch_size) * float(vocab_shard) * float(precision.activations)
            elif "linear" in gemm_shapes_local:
                softmax_out_bytes = _gemm_out_bytes("linear", gemm_type_map["linear"])
            else:
                tokens = float(batch_size) * float(seq_len_eff)
                vocab_shard = math.ceil(float(tc.vocab_size) / float(max(1, tp * cp)))
                softmax_out_bytes = tokens * float(vocab_shard) * float(precision.activations)

            if llm_util.is_vit_style(getattr(tc, "model_type", "")):
                embedding_out_bytes = (
                    float(batch_size)
                    * float(seq_len_eff)
                    * float(tc.hidden_dim)
                    * float(precision.activations)
                )
                vit_output_dim = llm_util.vit_output_dim(
                    hidden_dim=tc.hidden_dim,
                    num_classes=getattr(tc, "num_classes", 0),
                )
                softmax_out_bytes = (
                    float(batch_size)
                    * float(vit_output_dim)
                    * float(precision.activations)
                )
            else:
                embedding_out_bytes = out_proj_bytes

            base_outputs = {
                MemKind.LAYERNORM1: out_proj_bytes,
                MemKind.QKV_PROJ: qkv_bytes,
                MemKind.ROPE: 0.0,
                MemKind.ATTENTION: attn_out_bytes,
                MemKind.OUTPUT_PROJ: out_proj_bytes,
                MemKind.LAYERNORM2: out_proj_bytes,
                MemKind.MLP: ffn2_bytes,
                MemKind.EMBEDDING: 0.0 if disable_embedding_unembedding else embedding_out_bytes,
                MemKind.SOFTMAX: 0.0 if disable_embedding_unembedding else softmax_out_bytes,
                MemKind.OPTIMIZER: 0.0,
                MemKind.MOE_DISPATCH: 0.0,
                MemKind.MOE_COMBINE: 0.0,
            }
            parallelism_mode = None
            try:
                parallelism_mode = tc.get_parallelism_mode()
            except AttributeError:
                parallelism_mode = None
            if parallelism_mode is not None:
                mode_value = str(getattr(parallelism_mode, "value", parallelism_mode)).lower()
                if mode_value.endswith("tensor_sequence"):
                    seq_degree = max(1, int(tc._sequence_parallel_degree()))
                    hidden_full_bytes = (
                        float(batch_size)
                        * float(seq_len)
                        * float(tc.hidden_dim)
                        * float(precision.activations)
                    )
                    hidden_shard_bytes = (
                        float(batch_size)
                        * float(math.ceil(float(seq_len) / float(seq_degree)))
                        * float(tc.hidden_dim)
                        * float(precision.activations)
                    )
                    base_outputs[MemKind.OUTPUT_PROJ] = hidden_shard_bytes
                    base_outputs[MemKind.MLP] = hidden_shard_bytes
                    base_outputs[MemKind.LAYERNORM1] = hidden_full_bytes
                    base_outputs[MemKind.LAYERNORM2] = hidden_full_bytes

            transformer_fallback_bytes = transformer_act_layer_local
            if mode != "training":
                transformer_fallback_bytes = transformer_act_layer_inf_local
            mlp_bytes_base = base_outputs[MemKind.MLP] + ffn1_bytes
            if use_moe_local:
                seq_divisor = cp if cp > 1 else 1
                tokens_owner = float(batch_size) * math.ceil(float(seq_len) / float(seq_divisor))
                moe_scale = 1.0
                if tokens_owner > 0:
                    groups_fn = getattr(tc, "_moe_dp_attention_groups", None)
                    pool_groups = int(groups_fn()) if callable(groups_fn) else 1
                    tokens_dispatched = tokens_owner * pool_groups * float(getattr(tc, "moe_top_k", 1))

                    share_divisor = 1 if mode == "training" else max(1, moe_group)
                    tokens_local = math.ceil(tokens_dispatched / float(share_divisor))
                    if mode != "training":
                        experts_per_rank = int(
                            max(1, float(getattr(tc, "moe_num_experts", 1)) / float(max(1, moe_group)))
                        )
                        if experts_per_rank > 0 and (tokens_local % experts_per_rank) != 0:
                            tokens_local = math.ceil(tokens_local / float(experts_per_rank)) * experts_per_rank
                    else:
                        experts_per_rank = int(
                            max(1, float(getattr(tc, "moe_num_experts", 1)) / float(max(1, moe_group)))
                        )
                    apply_imbalance = getattr(tc, "_moe_apply_expert_imbalance", None)
                    if callable(apply_imbalance):
                        tokens_local = float(apply_imbalance(int(tokens_local), experts_per_rank))
                    moe_scale = (float(tokens_local) / float(tokens_owner)) + float(
                        getattr(tc, "n_shared_experts", 0)
                    )

                base_outputs[MemKind.MLP] *= moe_scale
                ffn1_bytes *= moe_scale
            mlp_bytes_scaled = base_outputs[MemKind.MLP] + ffn1_bytes
            if use_moe_local:
                transformer_fallback_bytes += float(mlp_bytes_scaled - mlp_bytes_base)
            base_outputs[MemKind.TRANSFORMER] = transformer_fallback_bytes

            persistent_bytes_by_kind = {kind: 0.0 for kind in base_outputs}
            transient_bytes_by_kind = {kind: 0.0 for kind in base_outputs}

            store_outputs = mode == "training" and not full_recomputation
            if store_outputs:
                for kind, bytes_val in base_outputs.items():
                    persistent_bytes_by_kind[kind] = float(bytes_val or 0.0)
                if attn_score_bytes:
                    persistent_bytes_by_kind[MemKind.ATTENTION] += float(attn_score_bytes)
                persistent_bytes_by_kind[MemKind.MLP] += float(ffn1_bytes)
            else:
                for kind, bytes_val in base_outputs.items():
                    transient_bytes_by_kind[kind] = float(bytes_val or 0.0)
                if attn_score_bytes:
                    transient_bytes_by_kind[MemKind.ATTENTION] += float(attn_score_bytes)
                transient_bytes_by_kind[MemKind.MLP] += float(ffn1_bytes)

            return base_outputs, persistent_bytes_by_kind, transient_bytes_by_kind

        _, persistent_bytes_by_kind_dense, transient_bytes_by_kind_dense = _build_activation_maps(
            gemm_shapes_dense,
            transformer_act_layer_dense,
            transformer_act_layer_inf_dense,
            False,
        )
        persistent_bytes_by_kind_moe = persistent_bytes_by_kind_dense
        transient_bytes_by_kind_moe = transient_bytes_by_kind_dense
        if use_moe:
            base_outputs_moe, persistent_bytes_by_kind_moe, transient_bytes_by_kind_moe = _build_activation_maps(
                gemm_shapes_moe,
                transformer_act_layer_moe,
                transformer_act_layer_inf_moe,
                True,
            )
            if mode == "training":
                transformer_act_layer_moe = float(
                    base_outputs_moe.get(MemKind.TRANSFORMER, transformer_act_layer_moe)
                )
            else:
                transformer_act_layer_inf_moe = float(
                    base_outputs_moe.get(MemKind.TRANSFORMER, transformer_act_layer_inf_moe)
                )

        selective_remat = None
        if mode == "training" and not full_recomputation:
            from program import megatron as _megatron_sr
            mla_up_bytes = 0.0
            if str(getattr(tc, "attention_type", "mha")).lower() == "mla":
                heads = int(tc.num_heads)
                nope, rope, v_dim = (int(tc.qk_nope_head_dim), int(tc.qk_rope_head_dim), int(tc.v_head_dim))

                mla_up_bytes = (float(batch_size) * float(seq_len_eff) * heads * (nope + rope + nope + v_dim)
                                * float(precision.activations))
            freed = _megatron_sr.selective_memory(
                tc, persistent_bytes_by_kind_dense.get(MemKind.LAYERNORM1, 0.0), mla_up_bytes)
            if freed is not None:
                maps = {}
                for layer_type, persistent, transient in (
                        ("dense", persistent_bytes_by_kind_dense, transient_bytes_by_kind_dense),
                        ("moe", persistent_bytes_by_kind_moe, transient_bytes_by_kind_moe)):
                    persistent, transient = dict(persistent), dict(transient)
                    for kind, nbytes in freed[layer_type].items():
                        if nbytes > persistent.get(kind, 0.0) + 1e-6:
                            raise ValueError(f"selective recompute frees {nbytes} bytes of {kind} but the "
                                             f"{layer_type} layer stores {persistent.get(kind, 0.0)}")
                        persistent[kind] = persistent.get(kind, 0.0) - nbytes
                        transient[kind] = transient.get(kind, 0.0) + nbytes
                    maps[layer_type] = persistent, transient
                persistent_bytes_by_kind_dense, transient_bytes_by_kind_dense = maps["dense"]
                persistent_bytes_by_kind_moe, transient_bytes_by_kind_moe = maps["moe"]
                selective_remat = _megatron_sr.selective_remat(tc, freed)
                tc.__dict__["_selective_recompute_memory"] = {
                    layer_type: {kind.value: nbytes for kind, nbytes in kinds.items()}
                    for layer_type, kinds in freed.items()}
        persistent_bytes_by_kind = persistent_bytes_by_kind_dense
        transient_bytes_by_kind = transient_bytes_by_kind_dense

        if mode != "training" and kv_cache_tokens:
            kv_tokens = int(kv_cache_tokens)
            if getattr(tc, "attention_type", "mha") == "mla":
                per_token_bytes = llm_util.mla_kv_cache_token_bytes(
                    batch_size=batch_size,
                    kv_lora_rank=getattr(tc, "kv_lora_rank", 0),
                    qk_rope_head_dim=getattr(tc, "qk_rope_head_dim", 0),
                    precision_bytes=precision.kv_cache,
                )
                kv_tokens_local = math.ceil(float(kv_tokens) / float(max(1, cp)))
                kv_cache_bytes_per_layer = (
                    float(per_token_bytes) * float(kv_tokens_local)
                )
            else:
                kv_heads = int(getattr(tc, "kv_heads", tc.num_heads))
                head_dim = getattr(tc, "head_dim", None)
                if head_dim is None:
                    head_dim = tc.hidden_dim // tc.num_heads
                kv_heads_per_tp = math.ceil(kv_heads / tp)
                kv_cache_bytes_per_layer = (
                    float(batch_size)
                    * float(kv_heads_per_tp)
                    * float(head_dim)
                    * float(kv_tokens)
                    * float(precision.kv_cache)
                    * 2.0
                )

        param_gather_bytes = 0.0
        if mode == "training" and zero3_ephemeral_peak_bytes and dp > 1 and zero_stage >= 3:
            param_gather_bytes = float(zero3_ephemeral_peak_bytes)

        if use_moe:
            transformer_mem_layer_moe = transformer_act_layer_moe + weight_memory_layer_moe

        return {
            "checkpoint_input_bytes": checkpoint_input_bytes,
            "checkpoint_accounting": "layer_input_lifetime_v1",
            "activation_mem_per_layer": transformer_act_layer_dense,
            "activation_mem_per_layer_inference": transformer_act_layer_inf_dense,
            "weight_mem_per_layer": weight_memory_layer_dense,
            "gradient_mem_per_layer": gradient_mem_layer_dense,
            "optimizer_mem_per_layer": optimizer_mem_layer_dense,
            "static_mem_per_layer": transformer_static_layer_dense,
            "total_mem_per_layer": transformer_mem_layer_dense,
            "activation_mem_per_layer_dense": transformer_act_layer_dense,
            "activation_mem_per_layer_moe": transformer_act_layer_moe,
            "activation_mem_per_layer_inference_dense": transformer_act_layer_inf_dense,
            "activation_mem_per_layer_inference_moe": transformer_act_layer_inf_moe,
            "weight_mem_per_layer_dense": weight_memory_layer_dense,
            "weight_mem_per_layer_moe": weight_memory_layer_moe,
            "gradient_mem_per_layer_dense": gradient_mem_layer_dense,
            "gradient_mem_per_layer_moe": gradient_mem_layer_moe,
            "optimizer_mem_per_layer_dense": optimizer_mem_layer_dense,
            "optimizer_mem_per_layer_moe": optimizer_mem_layer_moe,
            "static_mem_per_layer_dense": transformer_static_layer_dense,
            "static_mem_per_layer_moe": transformer_static_layer_moe,
            "total_mem_per_layer_dense": transformer_mem_layer_dense,
            "total_mem_per_layer_moe": transformer_mem_layer_moe,
            "persistent_bytes_by_kind": persistent_bytes_by_kind,
            "transient_bytes_by_kind": transient_bytes_by_kind,
            "persistent_bytes_by_kind_dense": persistent_bytes_by_kind_dense,
            "persistent_bytes_by_kind_moe": persistent_bytes_by_kind_moe,
            "transient_bytes_by_kind_dense": transient_bytes_by_kind_dense,
            "transient_bytes_by_kind_moe": transient_bytes_by_kind_moe,
            "extra_static_bytes_per_device": extra_static_bytes_per_device,
            "kv_cache_bytes_per_layer": kv_cache_bytes_per_layer,
            "zero3_ephemeral_peak_bytes": zero3_ephemeral_peak_bytes,
            "param_gather_bytes": param_gather_bytes,

            **({"zero_moe_gather_bytes": zero_moe_gather_bytes} if zero_moe else {}),
            **({"selective_recompute_remat": selective_remat} if selective_remat else {}),
        }

    def simulate_peak(
        self,
        memory_program: Any,
        memory_data: Dict[str, Any],
        *,
        mode: str,
        filename: Optional[str] = None,
    ) -> Any:
        from program.ir import Program

        if not isinstance(memory_program, Program) or (
            memory_program.meta.misc.get("granularity") != "flat"
        ):
            raise RuntimeError(
                "Memory simulation requires a FLAT program. "
                "Use LLMExecutionDispatcher.build_flat_program_for_memory()."
            )
        return self.time_calc._simulate_with_memory(
            memory_program,
            memory_data,
            mode=mode,
            filename=filename,
        )

def estimate_inference_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    return llm_util.estimate_inference_memory(
        exp_hw_config,
        exp_model_config,
        mode=mode,
        output_dir=output_dir,
        **overrides,
    )

def estimate_training_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    from train_timing import TimeCalculationLLM

    tc = TimeCalculationLLM(exp_hw_config, exp_model_config, mode, output_dir=output_dir)
    return tc.estimate_memory_only()

def estimate_memory(
    exp_hw_config,
    exp_model_config,
    *,
    mode: str = "LLM",
    output_dir: Optional[str] = None,
    **overrides,
) -> Dict[str, Any]:
    run_type = str(getattr(exp_model_config.model_config, "run_type", "training")).lower()
    if run_type == "inference":
        return estimate_inference_memory(
            exp_hw_config,
            exp_model_config,
            mode=mode,
            output_dir=output_dir,
            **overrides,
        )
    return estimate_training_memory(
        exp_hw_config,
        exp_model_config,
        mode=mode,
        output_dir=output_dir,
    )
