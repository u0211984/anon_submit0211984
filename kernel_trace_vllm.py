from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

from kernel_trace import Kernel, qwen2_bias_path, SILU_AND_MUL_FLOPS, RMSNORM_FLOPS, FUSED_ADD_RMSNORM_FLOPS

BF16, F32, I64, I32 = 2.0, 4.0, 8.0, 4.0
SAMPLING = ("random_seed", "random", "greedy")

@dataclass(frozen=True)
class VAccess:
    tensor: str
    nbytes: float
    weight: bool = False
    offset: float = 0.0

@dataclass(frozen=True)
class VllmEngine:
    cuda_graph: bool = True
    max_model_len: int = 2048
    block_size: int = 16
    sampling: str = "random_seed"
    min_tokens_penalty: bool = True
    n_stop_ids: int = 1
    sm_count: int = field(kw_only=True)

    def __post_init__(self):
        if not isinstance(self.sm_count, int) or isinstance(self.sm_count, bool) or self.sm_count <= 0:
            raise ValueError("sm_count must be a positive integer")

def _a(t, n, w=False, off=0.0):
    return VAccess(t, float(n), w, float(off))

def _k(op, name, kind, r, w, gemm=None, l2_reread=0.0, smallgrid_rows=0):
    return Kernel(op, name, kind, tuple(r), tuple(w), 0.0, gemm, name, float(l2_reread), smallgrid_rows)

def flash_num_splits(batch: int, heads: int, seqlen_q: int, seqlen_k: int, head_dim: int, sm_count: int) -> int:
    block_n = 256 if head_dim <= 64 else (128 if head_dim <= 128 else 64)
    n_blocks = -(-seqlen_k // block_n)
    work = batch * heads * (-(-seqlen_q // 64))
    sms = 2 * sm_count
    if work >= 0.8 * sms:
        return 1
    max_splits = min(128, sms, n_blocks)
    cdiv = lambda x, y: -(-x // y)  # noqa: E731
    elig = lambda s: s == 1 or cdiv(n_blocks, s) != cdiv(n_blocks, s - 1)  # noqa: E731
    eff = [(work * s / sms) / math.ceil(work * s / sms) if elig(s) else 0.0 for s in range(1, max_splits + 1)]
    best = max(eff) if eff else 1.0
    for s in range(1, max_splits + 1):
        if elig(s) and eff[s - 1] >= 0.85 * best:
            return s
    return 1

def _dims(cfg: dict):
    h, H = cfg["hidden_size"], cfg["num_attention_heads"]
    return (h, H, cfg["num_key_value_heads"], cfg.get("head_dim") or h // H, cfg["intermediate_size"],
            cfg["vocab_size"], cfg["num_hidden_layers"])

def decode_uses_cuda_graph(kv_len: int, eng: VllmEngine) -> bool:
    return bool(eng.cuda_graph) and int(kv_len) <= min(8192, int(eng.max_model_len))

def decode_attention_layout(cfg: dict, batch: int, kv_len: int, eng: VllmEngine) -> Tuple[int, int, int, int]:
    h, H, Hk, hd, _, _, _ = _dims(cfg)
    swapped = H > Hk
    hq, sq = (Hk, H // Hk) if swapped else (H, 1)
    bs = eng.block_size

    seq_to_capture = min(8192, int(eng.max_model_len))
    if decode_uses_cuda_graph(kv_len, eng):
        width = -(-seq_to_capture // bs)
    else:
        width = -(-kv_len // bs)
    seqlen_k = width * bs
    return hq, sq, seqlen_k, flash_num_splits(batch, hq, sq, seqlen_k, hd, eng.sm_count)

def local_kv_heads(kv_heads: int, tp: int) -> int:
    if kv_heads % tp == 0:
        return kv_heads // tp
    if tp % kv_heads == 0:
        return 1
    raise ValueError("tp must divide the KV heads or be a multiple of them")

def _vtp_pointwise_flops(kernels):
    result = []
    for k in kernels:
        if k.name == "rotary_embedding":
            k = replace(k, flops=3 * sum(a.nbytes for a in k.writes) / BF16)
        elif k.name == "silu_and_mul":
            k = replace(k, flops=SILU_AND_MUL_FLOPS * k.writes[0].nbytes / BF16)
        result.append(k)
    return result

def vllm_step_trace(cfg: dict, phase: str, batch: int = 1, seq: int = 1024,
                    eng: Optional[VllmEngine] = None, tp: int = 1, rank: int = 0,
                    *, moe_tile_rule=None, ep: int = 1, moe_chunk_size=None,
                    rapid_tile_selector=None, moe_rank=None, moe_token_counts=None,
                    moe_routing_step=0, moe_layer_cache=None) -> List[Kernel]:
    if eng is None:
        raise ValueError("eng with hardware sm_count is required")
    family = cfg.get("model_type", "llama")
    mla = str(cfg.get("attention_type", "")).lower() == "mla"
    if mla:
        if family not in ("llama", "deepseek_v3"):
            raise ValueError(f"vLLM MLA trace does not describe model_type {family!r}")

        family = "llama"
    if family not in ("llama", "qwen2", "qwen3", "qwen3_moe"):
        raise ValueError(f"vLLM trace does not describe model_type {family!r}")
    if family in ("qwen2", "qwen3", "qwen3_moe") and cfg.get("use_sliding_window", False):
        raise ValueError("vLLM Qwen trace does not describe sliding-window attention")
    if family in ("qwen3", "qwen3_moe") and cfg.get("attention_bias", False):
        raise ValueError("vLLM Qwen3 trace does not describe projection bias")
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase must be prefill or decode, got {phase!r}")
    if eng.sampling not in SAMPLING:
        raise ValueError(f"sampling must be one of {SAMPLING}")
    moe_spec = None
    from kernel_trace_vllm_moe import is_deepseek_moe
    deepseek_moe = mla and is_deepseek_moe(cfg)
    if family == "qwen3_moe" or deepseek_moe:
        import os
        from kernel_trace_vllm_moe import MoeSpec, moe_layer_trace, _integer, deepseek_moe_layers
        _integer(batch, "batch")
        _integer(seq, "seq", minimum=1 if phase == "prefill" else 0)
        chunk_size = moe_chunk_size if moe_chunk_size is not None else cfg.get("moe_chunk_size")
        if chunk_size is None:
            chunk_size = int(os.environ.get("VLLM_FUSED_MOE_CHUNK_SIZE", "32768"))
        moe_spec = MoeSpec.from_config(
            cfg, tp=tp, ep=ep,
            tile_rule=moe_tile_rule if moe_tile_rule is not None else cfg.get("moe_tile_rule"),
            chunk_size=chunk_size,
            rapid_tile_selector=rapid_tile_selector)

    dims_cfg = dict(cfg, intermediate_size=cfg["moe_intermediate_size"]) if moe_spec and not deepseek_moe else cfg
    moe_layer_set = frozenset(deepseek_moe_layers(cfg)) if deepseek_moe else frozenset()
    h, H, Hk, hd, I, V, L = _dims(dims_cfg)
    if tp < 1 or not 0 <= rank < tp:
        raise ValueError("tp must be positive and rank must be in [0, tp)")
    if mla and tp > 1:
        Hk = tp
    if H % tp or I % tp or (Hk % tp and not (moe_spec is not None and tp % Hk == 0)):
        raise ValueError("tp must divide attention heads, KV heads and intermediate size")
    H, Hk, I = H // tp, local_kv_heads(Hk, tp), I // tp
    b = int(batch)
    tied = bool(cfg.get("tie_word_embeddings", False))
    T = b * seq if phase == "prefill" else b
    S_new = seq if phase == "prefill" else 1
    past = 0 if phase == "prefill" else seq
    qn, kn = H * hd, Hk * hd
    a = BF16

    wq = cfg.get("linear_weight_bytes")
    if wq is not None and not mla:
        raise NotImplementedError("precision.linear_weights is described for the DeepSeek MLA trace only")
    wq = a if wq is None else float(wq)
    row_kv = kn * BF16
    bs = eng.block_size
    blocks = -(-(past + S_new) // bs)
    seq_stride = blocks * bs * row_kv
    V_pad = 64 * -(-V // 64)
    if V_pad % tp:
        raise ValueError("tp must divide padded vocabulary size")
    V_shard = V_pad // tp
    w_head = "w_embed" if tied else "w_head"
    graph = phase == "decode" and eng.cuda_graph
    ks: List[Kernel] = []

    if phase == "decode":
        if rank == 0:
            ks.append(_k("embedding", "meta.query_start_loc_sub", "pointwise",
                     [_a("query_start_loc", I32 * (b + 1))], [_a("query_start_loc_dec", I32 * (b + 1))]))
        local_cfg = dict(cfg, num_attention_heads=H, num_key_value_heads=Hk, head_dim=hd)
        hq, sq, seqlen_k, ns = decode_attention_layout(local_cfg, b, past + 1, eng)
        width = seqlen_k // bs
        if graph and rank == 0:
            for src, n in (("ids", I64 * b), ("positions", I64 * b), ("slot_mapping", I64 * b),
                           ("seq_lens", I32 * b), ("block_table", I32 * b * width)):
                ks.append(_k("embedding", f"graph.copy_{src}", "memcpy", [_a(src, n)], [_a("g." + src, n)]))
        pre = "g." if graph else ""
    else:
        pre = ""
    ids, pos, slots = pre + "ids", pre + "positions", pre + "slot_mapping"

    if tp > 1:
        ks.append(_k("embedding", "embedding.mask_input", "pointwise", [_a(ids, I64 * T)],
                     [_a("masked_ids", I64 * T), _a("input_mask", T)]))
    ks.append(_k("embedding", "embedding", "gather", [_a("masked_ids" if tp > 1 else ids, I64 * T), _a("w_embed.rows", a * T * h, True)],
                 [_a("resid.local" if tp > 1 else "resid", a * T * h)]))
    if tp > 1:
        ks.append(_k("embedding", "embedding.mask_output", "pointwise", [_a("resid.local", a * T * h), _a("input_mask", T)], [_a("resid.local", a * T * h)]))
        ks.append(_k("embedding", "embedding.all_reduce", "collective", [_a("resid.local", a * T * h)], [_a("resid", a * T * h)]))

    for layer in range(L):
        p = f"L{layer}"

        if layer == 0:
            y1 = f"{p}.y1"

            ks.append(replace(_k("layernorm1", "rms_norm", "pointwise", [_a("resid", a * T * h), _a(f"{p}.w_ln1", a * h, True)],
                         [_a(y1, a * T * h)]), flops=RMSNORM_FLOPS * T * h))
        else:
            y1 = f"L{layer - 1}.dn"

            ks.append(replace(_k("layernorm1", "fused_add_rms_norm", "pointwise",
                         [_a(y1, a * T * h), _a("resid", a * T * h), _a(f"{p}.w_ln1", a * h, True)],
                         [_a("resid", a * T * h), _a(y1, a * T * h)]), flops=FUSED_ADD_RMSNORM_FLOPS * T * h))
        if mla:
            import kernel_trace_vllm_mla as kvm
            o = f"{p}.o"
            ks.extend(kvm.attention_block(cfg, phase, layer=layer, b=b, T=T, past=past, S_new=S_new, y1=y1, o=o,
                                          pre=pre, block_size=bs, kv_reads=cfg.get("mla_kv_reads", "per_head_group"),
                                          pad_v=bool(cfg.get("mla_prefill_pad_v", True)), tp=tp))
            ow = H * int(cfg["v_head_dim"])
            ao = f"{p}.ao"
            ks.append(_k("output_proj", "o_proj", "gemm", [_a(o, a * T * ow), _a(f"{p}.wo", wq * ow * h, True)],
                         [_a(ao + ".local" if tp > 1 else ao, a * T * h)], gemm=(T, ow, h)))
            if tp > 1:
                ks.append(_k("output_proj", "o_proj.all_reduce", "collective", [_a(ao + ".local", a * T * h)],
                             [_a(ao, a * T * h)]))
            ks.append(replace(_k("layernorm2", "fused_add_rms_norm", "pointwise",
                         [_a(ao, a * T * h), _a("resid", a * T * h), _a(f"{p}.w_ln2", a * h, True)],
                         [_a("resid", a * T * h), _a(ao, a * T * h)]), flops=FUSED_ADD_RMSNORM_FLOPS * T * h))

            dn = f"{p}.dn.local" if tp > 1 else f"{p}.dn"
            if layer in moe_layer_set:
                ks.extend(moe_layer_trace(moe_spec, T, layer, ao, dn,
                                         rank=(rank if ep > 1 else 0) if moe_rank is None else moe_rank,
                                         token_counts=moe_token_counts, step=moe_routing_step,
                                     cache=moe_layer_cache))
            else:
                ks.append(_k("MLP", "gate_up_proj", "gemm", [_a(ao, a * T * h), _a(f"{p}.wgu", wq * h * 2 * I, True)],
                             [_a(f"{p}.gu", a * T * 2 * I)], gemm=(T, h, 2 * I)))
                ks.append(_k("MLP", "silu_and_mul", "pointwise", [_a(f"{p}.gu", a * T * 2 * I)], [_a(f"{p}.m", a * T * I)]))
                ks.append(_k("MLP", "down_proj", "gemm", [_a(f"{p}.m", a * T * I), _a(f"{p}.wd", wq * I * h, True)],
                             [_a(dn, a * T * h)], gemm=(T, I, h)))
            if tp > 1:
                ks.append(_k("MLP", "down_proj.all_reduce", "collective",
                             [_a(f"{p}.dn.local", a * T * h)], [_a(f"{p}.dn", a * T * h)]))
            continue
        qkv = f"{p}.qkv"
        n = qn + 2 * kn
        reads = [_a(y1, a * T * h), _a(f"{p}.wqkv", a * h * n, True)]
        flops = 0
        if cfg.get("model_type") == "qwen2":
            mode = qwen2_bias_path(cfg.get("qwen2_bias_mode"), T, h, n, torch_version="2.8")
            if mode == "cublaslt_epilogue":
                reads.append(_a(f"{p}.bqkv", a * n, True))
            elif mode == "addmm_copy_gemm":
                ks.append(_k("qkv_proj", "bias_copy_qkv", "pointwise", [_a(f"{p}.bqkv", a * n, True)], [_a(qkv, a * T * n)]))
                reads.append(_a(qkv, a * T * n))
            else:
                raise ValueError("qwen2_bias_mode must be cublaslt_epilogue or addmm_copy_gemm")
            flops = T * n
        ks.append(replace(_k("qkv_proj", "qkv_proj", "gemm", reads, [_a(qkv, a * T * n)], gemm=(T, h, n)), flops=flops))
        q, k, v = f"{qkv}.q", f"{qkv}.k", f"{qkv}.v"
        if family in ("qwen3", "qwen3_moe"):
            for name, n in (("q", qn), ("k", kn)):
                norm_input = f"{qkv}.{name}"
                if T > 1:
                    contiguous = f"{p}.{name}_contiguous"
                    ks.append(_k("qkv_proj", f"{name}_contiguous", "pointwise",
                                 [_a(norm_input, a * T * n)], [_a(contiguous, a * T * n)]))
                    norm_input = contiguous
                ks.append(replace(_k("qkv_proj", f"{name}_norm", "pointwise",
                             [_a(norm_input, a * T * n), _a(f"{p}.w_{name}norm", a * hd, True)],
                             [_a(f"{p}.{name}norm", a * T * n)]), flops=RMSNORM_FLOPS * T * n + 2 * T * n / hd))
            q, k = f"{p}.qnorm", f"{p}.knorm"

        cs_row = hd * a
        if family in ("qwen3", "qwen3_moe"):
            ks.append(_k("rope", "rotary_embedding", "pointwise",
                         [_a(pos, I64 * T), _a(q, a * T * qn), _a(k, a * T * kn),
                          _a("cos_sin_cache", cs_row * T, off=cs_row * past)],
                         [_a(q, a * T * qn), _a(k, a * T * kn)]))
        else:
            ks.append(_k("rope", "rotary_embedding", "pointwise",
                         [_a(pos, I64 * T), _a(f"{qkv}.qk", a * T * (qn + kn)),
                          _a("cos_sin_cache", cs_row * T, off=cs_row * past)],
                         [_a(f"{qkv}.qk", a * T * (qn + kn))]))

        o = f"{p}.o"
        ks.append(_k("attention", "attn_out_zeros", "pointwise", [], [_a(o, a * T * qn)]))

        kw = [_a(f"{p}.{c}cache", row_kv * S_new, off=j * seq_stride + past * row_kv) for c in "kv" for j in range(b)]
        ks.append(_k("attention", "reshape_and_cache_flash", "pointwise",
                     [_a(k, a * T * kn), _a(v, a * T * kn), _a(slots, I64 * T)], kw))
        if phase == "prefill":
            ks.append(_k("attention", "attention.flash_fwd", "attention",
                         [_a(q, a * T * qn), _a(k, a * T * kn), _a(v, a * T * kn)],
                         [_a(o, a * T * qn), _a(f"{p}.lse", F32 * H * T)]))
        else:
            kr = [_a(f"{p}.{c}cache", row_kv * (past + 1), off=j * seq_stride) for c in "kv" for j in range(b)]
            meta = [_a(pre + "block_table", I32 * b * width), _a(pre + "seq_lens", I32 * b)]
            lse = _a(f"{p}.lse", F32 * b * hq * sq)
            qr = [_a(q, a * T * qn)]
            if ns > 1:
                acc = [_a(f"{p}.lse_accum", F32 * ns * b * hq * sq), _a(f"{p}.out_accum", F32 * ns * b * hq * sq * hd)]
                ks.append(_k("attention", "attention.splitkv", "attention", qr + kr + meta, acc))
                ks.append(_k("attention", "attention.combine", "attention", acc, [_a(o, a * T * qn), lse]))
            else:
                ks.append(_k("attention", "attention.fwd_kvcache", "attention", qr + kr + meta, [_a(o, a * T * qn), lse]))
        ao = f"{p}.ao"
        ks.append(_k("output_proj", "o_proj", "gemm", [_a(o, a * T * qn), _a(f"{p}.wo", a * qn * h, True)],
                     [_a(ao + ".local" if tp > 1 else ao, a * T * h)], gemm=(T, qn, h)))
        if tp > 1:
            ks.append(_k("output_proj", "o_proj.all_reduce", "collective", [_a(ao + ".local", a * T * h)], [_a(ao, a * T * h)]))
        ks.append(replace(_k("layernorm2", "fused_add_rms_norm", "pointwise",
                     [_a(ao, a * T * h), _a("resid", a * T * h), _a(f"{p}.w_ln2", a * h, True)],
                     [_a("resid", a * T * h), _a(ao, a * T * h)]), flops=FUSED_ADD_RMSNORM_FLOPS * T * h))
        if moe_spec is not None:
            ks.extend(moe_layer_trace(moe_spec, T, layer, ao,
                                     f"{p}.dn.local" if tp > 1 else f"{p}.dn",
                                     rank=(rank if ep > 1 else 0) if moe_rank is None else moe_rank,
                                     token_counts=moe_token_counts, step=moe_routing_step,
                                     cache=moe_layer_cache))
            if tp > 1:
                ks.append(_k("MLP", "down_proj.all_reduce", "collective",
                             [_a(f"{p}.dn.local", a * T * h)], [_a(f"{p}.dn", a * T * h)]))
            continue
        ks.append(_k("MLP", "gate_up_proj", "gemm", [_a(ao, a * T * h), _a(f"{p}.wgu", a * h * 2 * I, True)],
                     [_a(f"{p}.gu", a * T * 2 * I)], gemm=(T, h, 2 * I)))
        ks.append(_k("MLP", "silu_and_mul", "pointwise", [_a(f"{p}.gu", a * T * 2 * I)], [_a(f"{p}.m", a * T * I)]))
        ks.append(_k("MLP", "down_proj", "gemm", [_a(f"{p}.m", a * T * I), _a(f"{p}.wd", a * I * h, True)],
                     [_a(f"{p}.dn.local" if tp > 1 else f"{p}.dn", a * T * h)], gemm=(T, I, h)))

        if tp > 1:
            dn = f"{p}.dn"
            ks.append(_k("MLP", "down_proj.all_reduce", "collective", [_a(dn + ".local", a * T * h)], [_a(dn, a * T * h)]))

    hid = f"L{L - 1}.dn"
    ks.append(replace(_k("linear_softmax", "fused_add_rms_norm", "pointwise",
                 [_a(hid, a * T * h), _a("resid", a * T * h), _a("w_lnf", a * h, True)],
                 [_a("resid", a * T * h), _a(hid, a * T * h)]), flops=FUSED_ADD_RMSNORM_FLOPS * T * h))

    sel = [_a(hid, a * h, off=(j * S_new + S_new - 1) * a * h) for j in range(b)]
    ks.append(_k("linear_softmax", "prune_hidden_states", "gather", sel + [_a("selected_token_indices", I64 * b)],
                 [_a("yf", a * b * h)]))
    ks.append(_k("linear_softmax", "lm_head", "gemm", [_a("yf", a * b * h), _a(w_head, a * h * V_shard, True)],
                 [_a("logits_shard" if tp > 1 else "logits", a * b * V_shard)], gemm=(b, h, V_shard)))
    def finish_trace():
        nonlocal ks
        if moe_spec is not None and moe_spec.attention_dp > 1:
            global_rank = rank if moe_rank is None else moe_rank
            dp_rank = global_rank // tp
            counts = tuple(moe_token_counts) if moe_token_counts is not None else (T,) * moe_spec.attention_dp
            view_offset = 2 * h * sum(counts[:dp_rank])
            suffix = ".dn.local" if tp > 1 else ".dn"
            def view_access(access):
                if (access.tensor.startswith("L") and access.tensor.endswith(suffix)
                        and (not moe_layer_set or int(access.tensor[1:].split(".")[0]) in moe_layer_set)):
                    return replace(access, offset=access.offset + view_offset)
                return access
            ks = [replace(k, reads=tuple(map(view_access, k.reads)), writes=tuple(map(view_access, k.writes))) for k in ks]
        return _vtp_pointwise_flops(ks) if tp > 1 else ks

    if tp > 1:
        ks.append(_k("linear_softmax", "logits.gather", "collective", [_a("logits_shard", a * b * V_shard)],
                     [_a(f"logits.recv{j}", a * b * V_shard) for j in range(tp)] if rank == 0 else []))
        if rank != 0:
            return finish_trace()

        ks.append(_k("linear_softmax", "logits.cat", "pointwise",
                     [_a(f"logits.recv{j}", a * b * V_shard) for j in range(tp)],
                     [_a("logits", a * b * V_pad)]))

    lv = float(V)
    if eng.min_tokens_penalty:
        n = b * eng.n_stop_ids
        ks.append(_k("linear_softmax", "sampler.min_tokens_index_put", "pointwise",
                     [_a("penalty_rows", I64 * n), _a("penalty_cols", I64 * n)], [_a("logits", a * n)]))
    ks.append(_k("linear_softmax", "sampler.to_fp32", "pointwise", [_a("logits", a * b * lv)], [_a("logits32", F32 * b * lv)]))
    ks.append(_k("linear_softmax", "sampler.div_temperature", "pointwise",
                 [_a("logits32", F32 * b * lv), _a("temperatures", F32 * b)], [_a("logits32", F32 * b * lv)]))

    ks.append(_k("linear_softmax", "sampler.softmax", "reduce", [_a("logits32", F32 * b * lv)], [_a("probs", F32 * b * lv)],
                 l2_reread=2 * F32 * b * lv, smallgrid_rows=b))
    ks.append(_k("linear_softmax", "sampler.log_softmax", "reduce", [_a("logits32", F32 * b * lv)],
                 [_a("logprobs", F32 * b * lv)], l2_reread=2 * F32 * b * lv, smallgrid_rows=b))
    ks.append(_k("linear_softmax", "sampler.indices_long", "pointwise", [_a("sample_indices", I32 * b)],
                 [_a("sample_indices64", I64 * b)]))
    if eng.sampling == "greedy":
        ks.append(_k("linear_softmax", "sampler.index_rows", "gather",
                     [_a("logprobs", F32 * b * lv), _a("sample_indices64", I64 * b)], [_a("rows_sel", F32 * b * lv)]))
        ks.append(_k("linear_softmax", "sampler.argmax", "reduce", [_a("rows_sel", F32 * b * lv)], [_a("tok", I64 * b)]))
    else:
        ks.append(_k("linear_softmax", "sampler.index_rows", "gather",
                     [_a("probs", F32 * b * lv), _a("sample_indices64", I64 * b)], [_a("rows_sel", F32 * b * lv)]))
        if eng.sampling == "random_seed":
            for j in range(b):
                ks.append(_k("linear_softmax", "sampler.exponential", "pointwise", [],
                             [_a("q_exp", F32 * lv, off=j * F32 * lv)]))
        else:
            ks.append(_k("linear_softmax", "sampler.exponential", "pointwise", [], [_a("q_exp", F32 * b * lv)]))
        ks.append(_k("linear_softmax", "sampler.div_q", "pointwise",
                     [_a("rows_sel", F32 * b * lv), _a("q_exp", F32 * b * lv)], [_a("rows_sel", F32 * b * lv)]))
        ks.append(_k("linear_softmax", "sampler.argmax", "reduce", [_a("rows_sel", F32 * b * lv)], [_a("tok", I64 * b)]))
    return finish_trace()

def kernels_per_layer(phase: str, cfg: dict, batch: int = 1, seq: int = 1024,
                      eng: Optional[VllmEngine] = None, **trace_options) -> int:
    if eng is None:
        raise ValueError("eng with hardware sm_count is required")
    if cfg.get("model_type") == "qwen3_moe":
        trace = vllm_step_trace(dict(cfg, num_hidden_layers=1), phase, batch, seq, eng,
                                **trace_options)
        return sum(k.op not in ("embedding", "linear_softmax") and k.kind != "collective" for k in trace)
    if phase == "prefill":
        return 11
    return 12 if decode_attention_layout(cfg, batch, seq + 1, eng)[3] > 1 else 11
