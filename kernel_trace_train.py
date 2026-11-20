from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from kernel_trace import Access, Kernel, LayerDims, qwen2_bias_path, SILU_FWD_FLOPS

F32, BF16, I64, BOOL = 4.0, 2.0, 8.0, 1.0
SILU_BWD_FLOPS = 9.0
LOGSOFTMAX_FWD_FLOPS, LOGSOFTMAX_BWD_FLOPS = 6.0, 4.0
ADAMW_FLOPS = 15.0

@dataclass(frozen=True)
class TrainOpts:
    flash_launches: bool = False
    softmax_reads: int = 1
    sm_count: int = field(kw_only=True)
    include_memcpy: bool = True
    loss_bytes: float = F32

    def __post_init__(self):
        if not isinstance(self.sm_count, int) or isinstance(self.sm_count, bool) or self.sm_count <= 0:
            raise ValueError("sm_count must be a positive integer")

NCU: Dict[str, Tuple[str, str]] = {}

def _ncu(role: str, cls: str, sub: str) -> None:
    NCU[role] = (cls, sub)

for _r, _c, _s in [
    ("embedding.gather", "gather", "vectorized_gather_kernel"),
    ("mask.arange", "setup", "arange_cuda_out"), ("mask.add_past", "setup", "CUDAFunctorOnSelf_add<long>"),
    ("mask.first_minus_1", "setup", "CUDAFunctorOnSelf_add<long>"), ("mask.diff_cat", "cat", "CatArrayBatchedCopy"),
    ("mask.diff_sub", "setup", "CUDAFunctor_add<long>"), ("mask.ne_1", "setup", "Compare"),
    ("mask.cumsum_cast", "direct_copy", "direct_copy_kernel"), ("mask.scan_init", "scan", "DeviceScanInitKernel"),
    ("mask.scan", "scan", "DeviceScanKernel"), ("mask.last_eq_0", "setup", "Compare"),
    ("rotary.pos_float", "direct_copy", "direct_copy_kernel"), ("rotary.gemv", "gemv", "gemmk1_kernel"),
    ("rotary.cat", "cat", "CatArrayBatchedCopy"), ("rotary.cos", "unary", "cos_kernel"),
    ("rotary.scale_cos", "scalar_mul", "AUnaryFunctor<float"), ("rotary.sin", "unary", "sin_kernel"),
    ("rotary.scale_sin", "scalar_mul", "AUnaryFunctor<float"),
    ("rmsnorm.pow", "pow", "pow_tensor_scalar"), ("rmsnorm.mean", "reduce_mean", "MeanOps"),
    ("rmsnorm.add_eps", "scalar_add", "CUDAFunctorOnSelf_add<float>"), ("rmsnorm.rsqrt", "unary", "rsqrt_kernel"),
    ("rmsnorm.mul_rs", "mul_f32", "BinaryFunctor<float"), ("rmsnorm.mul_weight", "mul_f32", "BinaryFunctor<float"),
    ("cast_weight", "cast_bf16", "bfloat16_copy"), ("cast_act", "cast_bf16", "bfloat16_copy"),
    ("gemm", "gemm", "gemm"), ("gemm_dX", "gemm", "gemm"), ("gemm_dW", "gemm", "gemm"),
    ("rope.mul_cos", "mul_f32", "BinaryFunctor<float"), ("rope.neg", "neg", "neg_kernel"),
    ("rope.cat", "cat", "CatArrayBatchedCopy"), ("rope.mul_sin", "mul_f32", "BinaryFunctor<float"),
    ("rope.add", "add_f32", "CUDAFunctor_add<f"),
    ("sdpa.cast_k", "cast_bf16", "bfloat16_copy"), ("sdpa.cast_q", "cast_bf16", "bfloat16_copy"),
    ("flash", "flash_fwd", "flash_fwd"), ("flash.splitkv", "flash_fwd", "flash_fwd_splitkv_kernel"),
    ("flash.combine", "flash_fwd", "flash_fwd_splitkv_combine"), ("flash.fwd", "flash_fwd", "flash_fwd_kernel"),
    ("residual_add", "add_f32", "CUDAFunctor_add<f"), ("silu", "silu", "silu_kernel"),
    ("gate_mul", "mul_bf16", "BinaryFunctor<c10::BFloat16"),
    ("logits_float", "direct_copy", "direct_copy_kernel"), ("label_pad_fill", "fill", "FillFunctor<long>"),
    ("label_pad_copy", "memcpy", ""), ("log_softmax", "softmax", "cunn_SoftMaxForward"),
    ("nll_loss", "nll", "nll_loss_forward"),

    ("loss_seed", "fill", "FillFunctor<float>"), ("nll_zeros", "fill", "FillFunctor<float>"),
    ("nll_loss", "nll", "nll_loss"), ("log_softmax_bwd", "softmax", "cunn_SoftMaxBackward"),
    ("logits_float_bwd", "cast_bf16", "bfloat16_copy"),
    ("tocopy_act", "direct_copy", "direct_copy_kernel"), ("tocopy_weight", "direct_copy", "direct_copy_kernel"),
    ("grad_accum_add", "add_f32", "CUDAFunctor_add<f"), ("qk_reshape_copy", "direct_copy", "direct_copy_kernel"),
    ("residual_grad_cast", "cast_bf16", "bfloat16_copy"),
    ("rmsnorm.mul_grad_w", "mul_f32", "BinaryFunctor<float"), ("rmsnorm.mul_grad_n", "mul_f32", "BinaryFunctor<float"),
    ("rmsnorm.sum_w", "reduce_sum_f32", "sum_functor<float"), ("rmsnorm.mul_grad_x_r", "mul_f32", "BinaryFunctor<float"),
    ("rmsnorm.mul_grad_rs", "mul_f32", "BinaryFunctor<float"), ("rmsnorm.sum_rs", "reduce_sum_f32", "sum_functor<float"),
    ("rmsnorm.accum_residual", "add_f32", "CUDAFunctor_add<f"), ("rmsnorm.pow3", "pow", "pow_tensor_scalar"),
    ("rmsnorm.mul_m05", "scalar_mul", "AUnaryFunctor<float"), ("rmsnorm.mul_rsqrt_grad", "mul_f32", "BinaryFunctor<float"),
    ("rmsnorm.mean_div", "scalar_div", "BUnaryFunctor"), ("rmsnorm.pow1_copy", "memcpy", ""),
    ("rmsnorm.mul_2x", "scalar_mul", "AUnaryFunctor<float"), ("rmsnorm.mul_pow_grad", "mul_f32", "BinaryFunctor<float"),
    ("rmsnorm.accum_pow", "add_f32", "CUDAFunctor_add<f"),
    ("gate_mul.grad_up", "mul_bf16", "BinaryFunctor<c10::BFloat16"),
    ("gate_mul.grad_silu", "mul_bf16", "BinaryFunctor<c10::BFloat16"), ("silu_bwd", "silu_bwd", "silu_backward"),
    ("flash_bwd", "flash_bwd", "flash_bwd"), ("flash.dot_do_o", "flash_bwd", "flash_bwd_dot_do_o"),
    ("flash.dq_dk_dv", "flash_bwd", "flash_bwd_dq_dk_dv"), ("flash.convert_dq", "flash_bwd", "flash_bwd_convert_dq"),
    ("flash.gqa_sum_dk", "reduce_sum_bf16", "ReduceOp<c10::BFloat16"),
    ("flash.gqa_sum_dv", "reduce_sum_bf16", "ReduceOp<c10::BFloat16"),
    ("sdpa.tocopy_q", "direct_copy", "direct_copy_kernel"), ("sdpa.tocopy_k", "direct_copy", "direct_copy_kernel"),
    ("rope.mul_sin_grad", "mul_f32", "BinaryFunctor<float"), ("rope.cast_rot", "cast_bf16", "bfloat16_copy"),
    ("rope.neg_bwd", "neg", "neg_kernel"), ("rope.slice2_zeros", "fill", "FillFunctor<c10::BFloat16>"),
    ("rope.slice2_copy", "direct_copy", "direct_copy_kernel"), ("rope.slice1_zeros", "fill", "FillFunctor<c10::BFloat16>"),
    ("rope.slice1_copy", "direct_copy", "direct_copy_kernel"), ("rope.accum_slices", "add_bf16", "CUDAFunctor_add<c10"),
    ("rope.mul_cos_grad", "mul_f32", "BinaryFunctor<float"), ("rope.cast_cos", "cast_bf16", "bfloat16_copy"),
    ("rope.accum_cos", "add_bf16", "CUDAFunctor_add<c10"),
    ("embedding.zeros", "fill", "FillFunctor<float>"), ("embedding.feature_kernel", "embedding_bwd", "embedding_backward_feature"),
    ("embedding.tied_accum", "add_f32", "CUDAFunctor_add<f"),
    ("adamw.step_count", "optimizer", "TensorListMetadata<1>"), ("adamw.fused", "optimizer", "FusedOptimizerTensorListMetadata<4>"),
]:
    _ncu(_r, _c, _s)

def role_of(k: Kernel) -> str:
    return k.name.split(".", 1)[1]

def _a(t: str, n: float, w: bool = False) -> Access:
    return Access(t, float(n), w)

def _k(op: str, direction: str, role: str, kind: str, reads, writes, flops: float = 0.0,
       gemm: Optional[Tuple[int, int, int]] = None, l2_reread: float = 0.0) -> Kernel:
    return Kernel(op, f"{direction}.{role}", kind, tuple(reads), tuple(writes), float(flops), gemm, f"train-{role}",
                  float(l2_reread))

def train_dims(cfg: dict, batch: int, seq: int) -> LayerDims:
    family = cfg.get("model_type", "llama")
    if family not in ("llama", "qwen2", "qwen3"):
        raise ValueError(f"training trace does not describe model_type {family!r}")
    if family in ("qwen2", "qwen3") and cfg.get("use_sliding_window", False):
        raise ValueError("training trace does not describe sliding-window attention")
    if family == "qwen3" and cfg.get("attention_bias", False):
        raise ValueError("training trace does not describe Qwen3 projection bias")
    if family == "qwen2":
        qwen2_bias_path(cfg.get("qwen2_bias_mode"), 2, 2, 2)
    h, H = cfg["hidden_size"], cfg["num_attention_heads"]
    return LayerDims(tokens=batch * seq, batch=batch, hidden=h, heads=H, kv_heads=cfg["num_key_value_heads"],
                     head_dim=cfg.get("head_dim") or h // H, intermediate=cfg["intermediate_size"],
                     vocab=cfg["vocab_size"], act_bytes=BF16, weight_bytes=F32, kv_bytes=BF16,
                     model_type=cfg.get("model_type", "llama"), tied_embeddings=cfg.get("tie_word_embeddings", True),
                     qwen2_bias_mode=cfg.get("qwen2_bias_mode"))

def rmsnorm_fwd(op: str, x: str, y: str, d: LayerDims, w: str, direction: str = "forward") -> List[Kernel]:
    T, e = d.tokens, float(d.tokens) * d.hidden
    t = f"{y}.tmp"
    k = lambda r, kind, rd, wr, f: _k(op, direction, f"rmsnorm.{r}", kind, rd, wr, f)  # noqa: E731
    return [
        k("pow", "pointwise", [_a(x, F32 * e)], [_a(t + ".sq", F32 * e)], e),
        k("mean", "reduce", [_a(t + ".sq", F32 * e)], [_a(t + ".var", F32 * T)], e),
        k("add_eps", "pointwise", [_a(t + ".var", F32 * T)], [_a(t + ".ve", F32 * T)], T),
        k("rsqrt", "pointwise", [_a(t + ".ve", F32 * T)], [_a(y + ".rs", F32 * T)], T),
        k("mul_rs", "pointwise", [_a(x, F32 * e), _a(y + ".rs", F32 * T)], [_a(y + ".n", F32 * e)], e),
        k("mul_weight", "pointwise", [_a(w, F32 * d.hidden, True), _a(y + ".n", F32 * e)],
          [_a(y, F32 * e)], e),
    ]

def linear_fwd(op: str, x: str, x_fp32: bool, w: str, y: str, n_in: int, n_out: int, d: LayerDims,
               direction: str = "forward", bias: Optional[str] = None, bias_mode: Optional[str] = None) -> List[Kernel]:
    T = d.tokens
    ks = []
    if bias is not None:
        bias_mode = qwen2_bias_path(bias_mode, T, n_in, n_out)
        ks.append(_k(op, direction, "cast_bias", "pointwise", [_a(bias, F32 * n_out, True)], [_a(bias + ".bf16", BF16 * n_out)]))
    ks += [_k(op, direction, "cast_weight", "pointwise", [_a(w, F32 * n_in * n_out, True)],
             [_a(w + ".bf16", BF16 * n_in * n_out)])]
    xb = x
    if x_fp32:
        xb = f"{x}.bf16@{w}"
        ks.append(_k(op, direction, "cast_act", "pointwise", [_a(x, F32 * T * n_in)], [_a(xb, BF16 * T * n_in)]))
    rd = [_a(xb, BF16 * T * n_in), _a(w + ".bf16", BF16 * n_in * n_out)]
    if bias is not None:
        if bias_mode == "cublaslt_epilogue":
            rd.append(_a(bias + ".bf16", BF16 * n_out))
        else:
            ks.append(_k(op, direction, "bias_expand_copy", "pointwise", [_a(bias + ".bf16", BF16 * n_out)], [_a(y, BF16 * T * n_out)]))
            rd.append(_a(y, BF16 * T * n_out))
    ks.append(_k(op, direction, "gemm", "gemm", rd,
                 [_a(y, BF16 * T * n_out)], float(T * n_out) if bias else 0.0, (T, n_in, n_out)))
    return ks

def qwen3_norm_fwd(x: str, y: str, w: str, heads: int, d: LayerDims, direction: str) -> List[Kernel]:
    nd = replace(d, tokens=d.tokens * heads, hidden=d.head_dim)
    e = float(nd.tokens * nd.hidden)
    ks = [_k("qkv_proj", direction, "qk_norm.cast_input", "pointwise", [_a(x, BF16 * e)], [_a(x + ".f32", F32 * e)])]
    inner = rmsnorm_fwd("qkv_proj", x + ".f32", y, nd, w, direction)
    ks += inner[:-1]
    ks.append(_k("qkv_proj", direction, "qk_norm.cast_normalized", "pointwise", [_a(y + ".n", F32 * e)], [_a(y + ".nb", BF16 * e)]))
    ks.append(_k("qkv_proj", direction, "qk_norm.mul_weight", "pointwise", [_a(w, F32 * d.head_dim, True), _a(y + ".nb", BF16 * e)], [_a(y, F32 * e)], e))
    return ks

def qwen3_norm_bwd(gy: str, x: str, y: str, w: str, gx: str, heads: int, d: LayerDims, opts: TrainOpts) -> List[Kernel]:
    nd = replace(d, tokens=d.tokens * heads, hidden=d.head_dim)
    e = float(nd.tokens * nd.hidden)

    ks = [_k("qkv_proj", "backward", "qk_norm.mul_grad_normalized", "pointwise", [_a(gy, F32 * e), _a(w, F32 * d.head_dim, True)], [_a(gx + ".gn32", F32 * e)], e),
          _k("qkv_proj", "backward", "qk_norm.mul_grad_weight", "pointwise", [_a(gy, F32 * e), _a(y + ".nb", BF16 * e)], [_a(gx + ".gwf", F32 * e)], e),
          _k("qkv_proj", "backward", "qk_norm.sum_weight", "reduce", [_a(gx + ".gwf", F32 * e)], [_a(w + ".grad", F32 * d.head_dim)], e),
          _k("qkv_proj", "backward", "qk_norm.cast_grad_bf16", "pointwise", [_a(gx + ".gn32", F32 * e)], [_a(gx + ".gnb", BF16 * e)]),
          _k("qkv_proj", "backward", "qk_norm.cast_grad_float", "pointwise", [_a(gx + ".gnb", BF16 * e)], [_a(gx + ".gn", F32 * e)])]
    inner = rmsnorm_bwd("qkv_proj", gy, x + ".f32", y, w, gx + ".f32", nd, False, opts)

    inner = [replace(k, reads=tuple(replace(a, tensor=a.tensor.replace(gx + ".f32.tmp.gn", gx + ".gn")) for a in k.reads)) for k in inner[3:]]
    ks += inner
    ks.append(_k("qkv_proj", "backward", "qk_norm.cast_input_grad", "pointwise", [_a(gx + ".f32", F32 * e)], [_a(gx, BF16 * e)]))
    return ks

def rope_fwd(x: str, out: str, heads: int, d: LayerDims, direction: str = "forward", input_bytes: float = BF16) -> List[Kernel]:
    e = float(d.tokens) * heads * d.head_dim
    cs = F32 * d.tokens * d.head_dim
    t = f"{out}.tmp"
    k = lambda r, rd, wr, f: _k("rope", direction, f"rope.{r}", "pointwise", rd, wr, f)  # noqa: E731
    return [
        k("mul_cos", [_a(x, input_bytes * e), _a("cos", cs)], [_a(t + ".c", F32 * e)], e),
        k("neg", [_a(x, input_bytes * e / 2)], [_a(t + ".neg", input_bytes * e / 2)], e / 2),
        k("cat", [_a(t + ".neg", input_bytes * e / 2), _a(x, input_bytes * e / 2)], [_a(t + ".rot", input_bytes * e)], 0.0),
        _k("rope", direction, "rope.mul_sin", "pointwise", [_a(t + ".rot", input_bytes * e), _a("sin", cs)],
           [_a(t + ".s", F32 * e)], e, l2_reread=(heads - 1) * cs),
        k("add", [_a(t + ".c", F32 * e), _a(t + ".s", F32 * e)], [_a(out, F32 * e)], e),
    ]

def flash_num_splits(d: LayerDims, sm_count: int) -> int:
    if d.head_dim <= 64:
        block_n = 256
    elif d.head_dim <= 128:
        block_n = 128
    else:
        block_n = 64
    seq = d.tokens // d.batch
    n_blocks = -(-seq // block_n)
    work = d.batch * d.heads * (-(-seq // 64))
    sms = 2 * sm_count
    if work >= 0.8 * sms:
        return 1
    max_splits = min(128, sms, n_blocks)
    cdiv = lambda a, b: -(-a // b)  # noqa: E731
    elig = lambda s: s == 1 or cdiv(n_blocks, s) != cdiv(n_blocks, s - 1)  # noqa: E731
    eff = []
    for s in range(1, max_splits + 1):
        if not elig(s):
            eff.append(0.0)
            continue
        waves = work * s / sms
        eff.append(waves / math.ceil(waves))
    best = max(eff) if eff else 1.0
    for s in range(1, max_splits + 1):
        if elig(s) and eff[s - 1] >= 0.85 * best:
            return s
    return 1

def attention_fwd(p: str, q: str, k: str, v: str, o: str, d: LayerDims, opts: TrainOpts,
                  direction: str = "forward") -> List[Kernel]:
    T, H, Hk, dh = d.tokens, d.heads, d.kv_heads, d.head_dim
    rd = [_a(q, BF16 * T * H * dh), _a(k, BF16 * T * Hk * dh), _a(v, BF16 * T * Hk * dh)]
    wr = [_a(o, BF16 * T * H * dh), _a(f"{p}.lse", F32 * H * T)]
    if not opts.flash_launches:
        return [_k("attention", direction, "flash", "attention", rd, wr)]
    s = flash_num_splits(d, opts.sm_count)
    if s == 1:
        return [_k("attention", direction, "flash.fwd", "attention", rd, wr)]
    acc = [_a(f"{p}.out_accum", F32 * s * H * T * dh), _a(f"{p}.lse_accum", F32 * s * H * T)]
    return [_k("attention", direction, "flash.splitkv", "attention", rd, acc),
            _k("attention", direction, "flash.combine", "attention", acc, wr)]

def residual_add(op: str, r: str, y: str, out: str, d: LayerDims, direction: str = "forward") -> Kernel:
    e = float(d.tokens) * d.hidden
    return _k(op, direction, "residual_add", "pointwise", [_a(r, F32 * e), _a(y, BF16 * e)], [_a(out, F32 * e)], e)

def activation_fwd(g: str, u: str, s: str, out: str, d: LayerDims,
                   direction: str = "forward") -> List[Kernel]:
    e = float(d.tokens) * d.intermediate
    return [_k("gelu", direction, "silu", "pointwise", [_a(g, BF16 * e)], [_a(s, BF16 * e)], SILU_FWD_FLOPS * e),
            _k("gelu", direction, "gate_mul", "pointwise", [_a(s, BF16 * e), _a(u, BF16 * e)], [_a(out, BF16 * e)], e)]

def activation_bwd(gy: str, g: str, u: str, s: str, gs: str, gg: str, gu: str,
                   d: LayerDims) -> List[Kernel]:
    e = float(d.tokens) * d.intermediate
    return [_k("gelu", "backward", "gate_mul.grad_up", "pointwise", [_a(gy, BF16 * e), _a(s, BF16 * e)], [_a(gu, BF16 * e)], e),
            _k("gelu", "backward", "gate_mul.grad_silu", "pointwise", [_a(gy, BF16 * e), _a(u, BF16 * e)], [_a(gs, BF16 * e)], e),
            _k("gelu", "backward", "silu_bwd", "pointwise", [_a(gs, BF16 * e), _a(g, BF16 * e)], [_a(gg, BF16 * e)], SILU_BWD_FLOPS * e)]

def layer_forward(layer: int, d: LayerDims, x_in: str, x_out: str, opts: TrainOpts = None,
                  direction: str = "forward") -> List[Kernel]:
    if opts is None:
        raise ValueError("opts with hardware sm_count is required")
    p = f"L{layer}"
    T, h, I = d.tokens, d.hidden, d.intermediate
    qn, kn = d.heads * d.head_dim, d.kv_heads * d.head_dim
    ks: List[Kernel] = []
    ks += rmsnorm_fwd("layernorm1", x_in, f"{p}.y1", d, f"{p}.w_ln1", direction)
    for nm, w, n in (("q", "wq", qn), ("k", "wk", kn), ("v", "wv", kn)):
        bias = f"{p}.b{nm}" if d.model_type == "qwen2" else None
        ks += linear_fwd("qkv_proj", f"{p}.y1", True, f"{p}.{w}", f"{p}.{nm}", h, n, d, direction, bias, d.qwen2_bias_mode)
        if d.model_type == "qwen3" and nm in ("q", "k"):
            ks += qwen3_norm_fwd(f"{p}.{nm}", f"{p}.{nm}n", f"{p}.w_{nm}norm", d.heads if nm == "q" else d.kv_heads, d, direction)
    qx, kx = (f"{p}.qn", f"{p}.kn") if d.model_type == "qwen3" else (f"{p}.q", f"{p}.k")
    rb = F32 if d.model_type == "qwen3" else BF16
    ks += rope_fwd(qx, f"{p}.qr", d.heads, d, direction, rb)
    ks += rope_fwd(kx, f"{p}.kr", d.kv_heads, d, direction, rb)
    ek, eq = float(T) * kn, float(T) * qn
    ks.append(_k("attention", direction, "sdpa.cast_k", "pointwise", [_a(f"{p}.kr", F32 * ek)], [_a(f"{p}.kb", BF16 * ek)]))
    ks.append(_k("attention", direction, "sdpa.cast_q", "pointwise", [_a(f"{p}.qr", F32 * eq)], [_a(f"{p}.qb", BF16 * eq)]))
    ks += attention_fwd(p, f"{p}.qb", f"{p}.kb", f"{p}.v", f"{p}.o", d, opts, direction)
    ks += linear_fwd("output_proj", f"{p}.o", False, f"{p}.wo", f"{p}.ao", qn, h, d, direction)
    ks.append(residual_add("layernorm1", x_in, f"{p}.ao", f"{p}.r1", d, direction))
    ks += rmsnorm_fwd("layernorm2", f"{p}.r1", f"{p}.y2", d, f"{p}.w_ln2", direction)
    ei = float(T) * I
    ks += linear_fwd("ffn1", f"{p}.y2", True, f"{p}.wg", f"{p}.g", h, I, d, direction)
    act = activation_fwd(f"{p}.g", f"{p}.u", f"{p}.s", f"{p}.m", d, direction)
    ks.append(act[0])
    ks += linear_fwd("ffn1", f"{p}.y2", True, f"{p}.wu", f"{p}.u", h, I, d, direction)
    ks.append(act[1])
    ks += linear_fwd("ffn2", f"{p}.m", False, f"{p}.wd", f"{p}.dn", I, h, d, direction)
    ks.append(residual_add("layernorm2", f"{p}.r1", f"{p}.dn", x_out, d, direction))
    return ks

def tocopy_bwd(op: str, g: str, out: str, n: float, role: str, direction: str = "backward") -> Kernel:
    return _k(op, direction, role, "pointwise", [_a(g, BF16 * n)], [_a(out, F32 * n)])

def linear_bwd(op: str, gy: str, x: str, x_cast: bool, w: str, gx: str, n_in: int, n_out: int, d: LayerDims,
               accumulate: bool = False, reshape_copy: bool = False, direction: str = "backward", bias: Optional[str] = None) -> List[Kernel]:
    T = d.tokens
    ks: List[Kernel] = []
    if reshape_copy:
        ks.append(_k(op, direction, "qk_reshape_copy", "pointwise", [_a(gy + ".t", BF16 * T * n_out)],
                     [_a(gy, BF16 * T * n_out)]))
    xb = f"{x}.bf16@{w}" if x_cast else x
    gxb = f"{gx}.bf16@{w}"
    ks.append(_k(op, direction, "gemm_dX", "gemm", [_a(gy, BF16 * T * n_out), _a(w + ".bf16", BF16 * n_in * n_out)],
                 [_a(gxb, BF16 * T * n_in)], 0.0, (T, n_out, n_in)))
    ks.append(_k(op, direction, "gemm_dW", "gemm", [_a(xb, BF16 * T * n_in), _a(gy, BF16 * T * n_out)],
                 [_a(w + ".grad.bf16", BF16 * n_in * n_out)], 0.0, (n_in, T, n_out)))
    if bias is not None:
        ks.append(_k(op, direction, "bias_grad_sum", "reduce", [_a(gy, BF16 * T * n_out)], [_a(bias + ".grad.bf16", BF16 * n_out)], T * n_out))
    if x_cast:
        part = f"{gx}.part@{w}" if accumulate else gx
        ks.append(tocopy_bwd(op, gxb, part, float(T) * n_in, "tocopy_act", direction))
        if accumulate:
            e = float(T) * n_in
            ks.append(_k(op, direction, "grad_accum_add", "pointwise", [_a(gx, F32 * e), _a(part, F32 * e)],
                         [_a(gx, F32 * e)], e))
    ks.append(tocopy_bwd(op, w + ".grad.bf16", w + ".grad", float(n_in) * n_out, "tocopy_weight", direction))
    if bias is not None:
        ks.append(tocopy_bwd(op, bias + ".grad.bf16", bias + ".grad", n_out, "tocopy_bias", direction))
    return ks

def rmsnorm_bwd(op: str, gy: str, x: str, y: str, w: str, gx: str, d: LayerDims, residual_in_buffer: bool,
                opts: TrainOpts, direction: str = "backward") -> List[Kernel]:
    T, h = d.tokens, d.hidden
    e = float(T) * h
    t = f"{gx}.tmp"
    k = lambda r, kind, rd, wr, f: _k(op, direction, f"rmsnorm.{r}", kind, rd, wr, f)  # noqa: E731
    ks = [
        k("mul_grad_w", "pointwise", [_a(gy, F32 * e), _a(w, F32 * h, True)], [_a(t + ".gn", F32 * e)], e),
        k("mul_grad_n", "pointwise", [_a(gy, F32 * e), _a(y + ".n", F32 * e)], [_a(t + ".gwf", F32 * e)], e),
        k("sum_w", "reduce", [_a(t + ".gwf", F32 * e)], [_a(w + ".grad", F32 * h)], e),
        k("mul_grad_x_r", "pointwise", [_a(t + ".gn", F32 * e), _a(x, F32 * e)], [_a(t + ".grsf", F32 * e)], e),
        k("mul_grad_rs", "pointwise", [_a(t + ".gn", F32 * e), _a(y + ".rs", F32 * T)], [_a(t + ".gx1", F32 * e)], e),
        k("sum_rs", "reduce", [_a(t + ".grsf", F32 * e)], [_a(t + ".grs", F32 * T)], e),
    ]
    if residual_in_buffer:
        ks.append(k("accum_residual", "pointwise", [_a(gx, F32 * e), _a(t + ".gx1", F32 * e)], [_a(gx, F32 * e)], e))
    ks += [
        k("pow3", "pointwise", [_a(y + ".rs", F32 * T)], [_a(t + ".rs3", F32 * T)], 2 * T),
        k("mul_m05", "pointwise", [_a(t + ".grs", F32 * T)], [_a(t + ".g05", F32 * T)], T),
        k("mul_rsqrt_grad", "pointwise", [_a(t + ".g05", F32 * T), _a(t + ".rs3", F32 * T)], [_a(t + ".gv", F32 * T)], T),
        k("mean_div", "pointwise", [_a(t + ".gv", F32 * T)], [_a(t + ".gsq", F32 * e)], e),
    ]
    if opts.include_memcpy:
        ks.append(k("pow1_copy", "memcpy", [_a(x, F32 * e)], [_a(t + ".x1", F32 * e)], 0.0))
    ks += [
        k("mul_2x", "pointwise", [_a(t + ".x1", F32 * e)], [_a(t + ".x2", F32 * e)], e),
        k("mul_pow_grad", "pointwise", [_a(t + ".gsq", F32 * e), _a(t + ".x2", F32 * e)], [_a(t + ".gx2", F32 * e)], e),
    ]
    if residual_in_buffer:
        ks.append(k("accum_pow", "pointwise", [_a(gx, F32 * e), _a(t + ".gx2", F32 * e)], [_a(gx, F32 * e)], e))
    else:
        ks.append(k("accum_pow", "pointwise", [_a(t + ".gx1", F32 * e), _a(t + ".gx2", F32 * e)], [_a(gx, F32 * e)], e))
    return ks

def rope_bwd(g: str, gx: str, heads: int, d: LayerDims, direction: str = "backward") -> List[Kernel]:
    e = float(d.tokens) * heads * d.head_dim
    cs = F32 * d.tokens * d.head_dim
    t = f"{gx}.tmp"
    k = lambda r, rd, wr, f: _k("rope", direction, f"rope.{r}", "pointwise", rd, wr, f)  # noqa: E731
    return [
        k("mul_sin_grad", [_a(g, F32 * e), _a("sin", cs)], [_a(t + ".grot32", F32 * e)], e),
        k("cast_rot", [_a(t + ".grot32", F32 * e)], [_a(t + ".grot", BF16 * e)], 0.0),
        k("neg_bwd", [_a(t + ".grot", BF16 * e / 2)], [_a(t + ".gx2", BF16 * e / 2)], e / 2),
        k("slice2_zeros", [], [_a(t + ".s2", BF16 * e)], 0.0),
        k("slice2_copy", [_a(t + ".gx2", BF16 * e / 2)], [_a(t + ".s2", BF16 * e / 2)], 0.0),
        k("slice1_zeros", [], [_a(t + ".s1", BF16 * e)], 0.0),
        k("slice1_copy", [_a(t + ".grot", BF16 * e / 2)], [_a(t + ".s1", BF16 * e / 2)], 0.0),
        k("accum_slices", [_a(t + ".s2", BF16 * e), _a(t + ".s1", BF16 * e)], [_a(gx + ".t", BF16 * e)], e),
        k("mul_cos_grad", [_a(g, F32 * e), _a("cos", cs)], [_a(t + ".gc32", F32 * e)], e),
        k("cast_cos", [_a(t + ".gc32", F32 * e)], [_a(t + ".gc", BF16 * e)], 0.0),
        k("accum_cos", [_a(gx + ".t", BF16 * e), _a(t + ".gc", BF16 * e)], [_a(gx + ".t", BF16 * e)], e),
    ]

def qwen3_rope_bwd(g: str, gx: str, heads: int, d: LayerDims) -> List[Kernel]:
    e = float(d.tokens * heads * d.head_dim)
    cs = F32 * d.tokens * d.head_dim
    t = gx + ".tmp"
    k = lambda r, rd, wr, f=0: _k("rope", "backward", "rope." + r, "pointwise", rd, wr, f)
    return [k("mul_sin_grad", [_a(g, F32*e), _a("sin", cs)], [_a(t+".grot", F32*e)], e),
            k("neg_bwd", [_a(t+".grot", F32*e/2)], [_a(t+".gx2", F32*e/2)], e/2),
            k("slice2_zeros", [], [_a(t+".s2", F32*e)]),
            k("slice2_copy", [_a(t+".gx2", F32*e/2)], [_a(t+".s2", F32*e/2)]),
            k("slice1_zeros", [], [_a(t+".s1", F32*e)]),
            k("slice1_copy", [_a(t+".grot", F32*e/2)], [_a(t+".s1", F32*e/2)]),
            k("accum_slices", [_a(t+".s2", F32*e), _a(t+".s1", F32*e)], [_a(gx, F32*e)], e),
            k("mul_cos_grad", [_a(g, F32*e), _a("cos", cs)], [_a(t+".gc", F32*e)], e),
            k("accum_cos", [_a(gx, F32*e), _a(t+".gc", F32*e)], [_a(gx, F32*e)], e)]

def attention_bwd(p: str, d: LayerDims, opts: TrainOpts, direction: str = "backward") -> List[Kernel]:
    T, H, Hk, dh = d.tokens, d.heads, d.kv_heads, d.head_dim
    eq, ek = float(T) * H * dh, float(T) * Hk * dh
    base_r = [_a(f"{p}.o", BF16 * eq), _a(f"{p}.go", BF16 * eq), _a(f"{p}.qb", BF16 * eq), _a(f"{p}.kb", BF16 * ek),
              _a(f"{p}.v", BF16 * ek), _a(f"{p}.lse", F32 * H * T)]
    outs = [_a(f"{p}.gqb", BF16 * eq), _a(f"{p}.gkb", BF16 * ek), _a(f"{p}.gv", BF16 * ek)]
    if not opts.flash_launches:
        return [_k("attention", direction, "flash_bwd", "attention", base_r, outs)]
    dsum, dqa = _a(f"{p}.dsum", F32 * H * T), _a(f"{p}.dq_accum", F32 * eq)
    ks = [_k("attention", direction, "flash.dot_do_o", "attention", [_a(f"{p}.go", BF16 * eq), _a(f"{p}.o", BF16 * eq)],
             [dsum, dqa]),
          _k("attention", direction, "flash.dq_dk_dv", "attention",
             [_a(f"{p}.qb", BF16 * eq), _a(f"{p}.kb", BF16 * ek), _a(f"{p}.v", BF16 * ek), _a(f"{p}.go", BF16 * eq),
              _a(f"{p}.lse", F32 * H * T), dsum, dqa],
             [dqa] + ([_a(f"{p}.gk_exp", BF16 * eq), _a(f"{p}.gv_exp", BF16 * eq)] if H != Hk else outs[1:])),
          _k("attention", direction, "flash.convert_dq", "attention", [dqa], [outs[0]])]
    if H != Hk:
        ks += [_k("attention", direction, "flash.gqa_sum_dk", "reduce", [_a(f"{p}.gk_exp", BF16 * eq)], [outs[1]], eq),
               _k("attention", direction, "flash.gqa_sum_dv", "reduce", [_a(f"{p}.gv_exp", BF16 * eq)], [outs[2]], eq)]
    return ks

def layer_backward(layer: int, d: LayerDims, g_out: str, g_in: str, x_in: str, opts: TrainOpts = None,
                   recompute: bool = False) -> List[Kernel]:
    if opts is None:
        raise ValueError("opts with hardware sm_count is required")
    p = f"L{layer}"
    T, h, I = d.tokens, d.hidden, d.intermediate
    qn, kn = d.heads * d.head_dim, d.kv_heads * d.head_dim
    e, ei = float(T) * h, float(T) * I
    B = "backward"
    ks: List[Kernel] = []

    ks.append(_k("layernorm2", B, "residual_grad_cast", "pointwise", [_a(g_out, F32 * e)], [_a(f"{p}.gdn", BF16 * e)]))
    if recompute:
        fwd = layer_forward(layer, d, x_in, f"L{layer + 1}.unused", opts, "recompute")
        ks += fwd[:-2]
    ks += linear_bwd("ffn2", f"{p}.gdn", f"{p}.m", False, f"{p}.wd", f"{p}.gm", I, h, d)
    act = activation_bwd(f"{p}.gm", f"{p}.g", f"{p}.u", f"{p}.s", f"{p}.gs", f"{p}.gg", f"{p}.gu", d)
    ks += act[:2]
    ks += linear_bwd("ffn1", f"{p}.gu", f"{p}.y2", True, f"{p}.wu", f"{p}.gy2", h, I, d)
    ks.append(act[2])
    ks += linear_bwd("ffn1", f"{p}.gg", f"{p}.y2", True, f"{p}.wg", f"{p}.gy2", h, I, d, accumulate=True)

    ks += rmsnorm_bwd("layernorm2", f"{p}.gy2", f"{p}.r1", f"{p}.y2", f"{p}.w_ln2", f"{p}.gr1", d, True, opts)
    ks.append(_k("layernorm1", B, "residual_grad_cast", "pointwise", [_a(f"{p}.gr1", F32 * e)], [_a(f"{p}.gao", BF16 * e)]))
    ks += linear_bwd("output_proj", f"{p}.gao", f"{p}.o", False, f"{p}.wo", f"{p}.go", qn, h, d)
    ks += attention_bwd(p, d, opts)
    ks.append(tocopy_bwd("attention", f"{p}.gqb", f"{p}.gqr", float(T) * qn, "sdpa.tocopy_q"))
    ks.append(tocopy_bwd("attention", f"{p}.gkb", f"{p}.gkr", float(T) * kn, "sdpa.tocopy_k"))
    if d.model_type == "qwen3":
        ks += qwen3_rope_bwd(f"{p}.gkr", f"{p}.gkn", d.kv_heads, d)
        ks += qwen3_rope_bwd(f"{p}.gqr", f"{p}.gqn", d.heads, d)
    else:
        ks += rope_bwd(f"{p}.gkr", f"{p}.gk", d.kv_heads, d)
        ks += rope_bwd(f"{p}.gqr", f"{p}.gq", d.heads, d)
    ks += linear_bwd("qkv_proj", f"{p}.gv", f"{p}.y1", True, f"{p}.wv", f"{p}.gy1", h, kn, d, bias=f"{p}.bv" if d.model_type == "qwen2" else None)
    if d.model_type == "qwen3":
        ks += qwen3_norm_bwd(f"{p}.gkn", f"{p}.k", f"{p}.kn", f"{p}.w_knorm", f"{p}.gk.t", d.kv_heads, d, opts)
    ks += linear_bwd("qkv_proj", f"{p}.gk", f"{p}.y1", True, f"{p}.wk", f"{p}.gy1", h, kn, d, True, True, bias=f"{p}.bk" if d.model_type == "qwen2" else None)
    if d.model_type == "qwen3":
        ks += qwen3_norm_bwd(f"{p}.gqn", f"{p}.q", f"{p}.qn", f"{p}.w_qnorm", f"{p}.gq.t", d.heads, d, opts)
    ks += linear_bwd("qkv_proj", f"{p}.gq", f"{p}.y1", True, f"{p}.wq", f"{p}.gy1", h, qn, d, True, True, bias=f"{p}.bq" if d.model_type == "qwen2" else None)
    ks += rmsnorm_bwd("layernorm1", f"{p}.gy1", x_in, f"{p}.y1", f"{p}.w_ln1", g_in, d, True, opts)
    return ks

def setup_fwd(d: LayerDims) -> List[Kernel]:
    T, h, B = d.tokens, d.hidden, d.batch
    S = T // B
    hd = d.head_dim
    f = "forward"
    k = lambda op, r, kind, rd, wr, fl=0.0: _k(op, f, r, kind, rd, wr, fl)  # noqa: E731
    return [
        k("embedding", "embedding.gather", "gather", [_a("w_embed.rows", F32 * T * h, True), _a("ids", I64 * T)],
          [_a("x0", F32 * T * h)]),
        k("attention", "mask.arange", "pointwise", [], [_a("cache_pos0", I64 * S)]),
        k("attention", "mask.add_past", "pointwise", [_a("cache_pos0", I64 * S)], [_a("cache_pos", I64 * S)], S),
        k("attention", "mask.first_minus_1", "pointwise", [_a("cache_pos", I64 * B)], [_a("pos_m1", I64 * B)], B),
        k("attention", "mask.diff_cat", "pointwise", [_a("pos_m1", I64 * B), _a("cache_pos", I64 * T)],
          [_a("pos_cat", I64 * (T + B))]),
        k("attention", "mask.diff_sub", "pointwise", [_a("pos_cat", I64 * 2 * T)], [_a("pos_diff", I64 * T)], T),
        k("attention", "mask.ne_1", "pointwise", [_a("pos_diff", I64 * T)], [_a("pos_ne", BOOL * T)], T),
        k("attention", "mask.cumsum_cast", "pointwise", [_a("pos_ne", BOOL * T)], [_a("pos_ne64", I64 * T)]),
        k("attention", "mask.scan_init", "scan", [], []),
        k("attention", "mask.scan", "scan", [_a("pos_ne64", I64 * T)], [_a("packed", I64 * T)], T),
        k("attention", "mask.last_eq_0", "pointwise", [_a("packed", I64 * B)], [_a("packed_eq", BOOL * B)], B),
        k("rope", "rotary.pos_float", "pointwise", [_a("cache_pos", I64 * S)], [_a("pos_f", F32 * S)]),
        k("rope", "rotary.gemv", "gemm", [_a("inv_freq", F32 * hd / 2), _a("pos_f", F32 * S)],
          [_a("freqs", F32 * B * S * hd / 2)], B * S * hd),
        k("rope", "rotary.cat", "pointwise", [_a("freqs", F32 * T * hd / 2)], [_a("emb", F32 * T * hd)]),
        k("rope", "rotary.cos", "pointwise", [_a("emb", F32 * T * hd)], [_a("cos0", F32 * T * hd)], T * hd),
        k("rope", "rotary.scale_cos", "pointwise", [_a("cos0", F32 * T * hd)], [_a("cos", F32 * T * hd)], T * hd),
        k("rope", "rotary.sin", "pointwise", [_a("emb", F32 * T * hd)], [_a("sin0", F32 * T * hd)], T * hd),
        k("rope", "rotary.scale_sin", "pointwise", [_a("sin0", F32 * T * hd)], [_a("sin", F32 * T * hd)], T * hd),
    ]

def head_fwd(d: LayerDims, x_in: str, opts: TrainOpts) -> List[Kernel]:
    T, h, V = d.tokens, d.hidden, d.vocab
    f, op = "forward", "linear_softmax"
    tv = float(T) * V
    ks = rmsnorm_fwd(op, x_in, "yf", d, "w_lnf")
    ks += linear_fwd(op, "yf", True, "w_embed" if d.tied_embeddings else "w_head", "logits", h, V, d)
    return ks + loss_fwd(d, opts)

def loss_fwd(d: LayerDims, opts: TrainOpts) -> List[Kernel]:
    T, V = d.tokens, d.vocab
    f, op, tv = "forward", "linear_softmax", float(T) * V
    ks = []
    loss_bytes = opts.loss_bytes
    logits = "logits32" if loss_bytes == F32 else "logits"
    if loss_bytes == F32:
        ks.append(_k(op, f, "logits_float", "pointwise", [_a("logits", BF16 * tv)], [_a("logits32", F32 * tv)]))
    lab = d.batch * (T // d.batch + 1)
    ks.append(_k(op, f, "label_pad_fill", "pointwise", [], [_a("labels_pad", I64 * lab)]))
    if opts.include_memcpy:
        ks.append(_k(op, f, "label_pad_copy", "memcpy", [_a("ids", I64 * T)], [_a("labels_pad", I64 * T)]))
    r = opts.softmax_reads

    ks.append(_k(op, f, "log_softmax", "softmax", [_a(logits, loss_bytes * tv * r)], [_a("logp", loss_bytes * tv)],
                 LOGSOFTMAX_FWD_FLOPS * tv))
    ks.append(_k(op, f, "nll_loss", "reduce", [_a("logp.at_target", loss_bytes * T), _a("labels_pad", I64 * T)],
                 [_a("loss", loss_bytes), _a("total_weight", loss_bytes)], T))
    return ks

def loss_bwd(d: LayerDims, opts: TrainOpts) -> List[Kernel]:
    T, h, V = d.tokens, d.hidden, d.vocab
    b, op = "backward", "linear_softmax"
    tv = float(T) * V
    r = opts.softmax_reads
    loss_bytes = opts.loss_bytes
    glogits = "glogits32" if loss_bytes == F32 else "glogits"
    ks = [
        _k(op, b, "loss_seed", "pointwise", [], [_a("gloss", loss_bytes)]),
        _k(op, b, "nll_zeros", "pointwise", [], [_a("glogp", loss_bytes * tv)]),
        _k(op, b, "nll_loss", "pointwise", [_a("labels_pad", I64 * T), _a("gloss", loss_bytes), _a("total_weight", loss_bytes)],
           [_a("glogp", loss_bytes * T)], T),

        _k(op, b, "log_softmax_bwd", "softmax",
           [_a("glogp", loss_bytes * tv * (r - 1 if r > 1 else 1)), _a("logp", loss_bytes * tv)], [_a(glogits, loss_bytes * tv)],
           LOGSOFTMAX_BWD_FLOPS * tv),
    ]
    if loss_bytes == F32:
        ks.append(_k(op, b, "logits_float_bwd", "pointwise", [_a("glogits32", F32 * tv)], [_a("glogits", BF16 * tv)]))
    return ks

def head_bwd(d: LayerDims, x_in: str, g_x: str, opts: TrainOpts) -> List[Kernel]:
    h, V, op = d.hidden, d.vocab, "linear_softmax"
    ks = loss_bwd(d, opts)
    ks += linear_bwd(op, "glogits", "yf", True, "w_embed" if d.tied_embeddings else "w_head", "gyf", h, V, d)
    ks += rmsnorm_bwd(op, "gyf", x_in, "yf", "w_lnf", g_x, d, False, opts)
    return ks

def embedding_bwd(d: LayerDims, g_x0: str) -> List[Kernel]:
    T, h, V = d.tokens, d.hidden, d.vocab
    b, ev = "backward", float(V) * h
    if T > 3072:
        return embedding_bwd_sort_path(d, g_x0)
    return [
        _k("embedding", b, "embedding.zeros", "pointwise", [], [_a("w_embed.grad2", F32 * ev)]),
        _k("embedding", b, "embedding.feature_kernel", "scatter",
           [_a(g_x0, F32 * T * h), _a("ids", I64 * T), _a("w_embed.grad2.rows", F32 * T * h)],
           [_a("w_embed.grad2.rows", F32 * T * h)], T * h),
        _k("embedding", b, "embedding.tied_accum", "pointwise", [_a("w_embed.grad", F32 * ev), _a("w_embed.grad2", F32 * ev)],
           [_a("w_embed.grad", F32 * ev)], ev),
    ]

def embedding_bwd_sort_path(d: LayerDims, g_x0: str) -> List[Kernel]:
    T, h, V = d.tokens, d.hidden, d.vocab
    b, ev = "backward", float(V) * h
    k = lambda r, kind, rd, wr, f=0.0: _k("embedding", b, f"embedding.{r}", kind, rd, wr, f)  # noqa: E731
    return [
        k("arange", "pointwise", [], [_a("orig_pos", I64 * T)]),
        k("radix_sort", "sort", [_a("ids", I64 * T), _a("orig_pos", I64 * T)],
          [_a("sorted_ids", I64 * T), _a("orig_idx", I64 * T)]),
        k("segment_offsets", "scan", [_a("sorted_ids", I64 * T)], [_a("seg_off", I64 * T)]),
        k("partials_per_segment", "pointwise", [_a("seg_off", I64 * T)], [_a("pps", I64 * T)]),
        k("partials_scan", "scan", [_a("pps", I64 * T)], [_a("pps_off", I64 * T)]),
        k("partial_segment_offset", "pointwise", [_a("pps_off", I64 * T), _a("seg_off", I64 * T)], [_a("pseg", I64 * T)]),
        k("zeros", "pointwise", [], [_a("w_embed.grad2", F32 * ev)]),
        k("compute_grad_weight", "reduce", [_a(g_x0, F32 * T * h), _a("orig_idx", I64 * T), _a("pseg", I64 * T)],
          [_a("partials", F32 * T * h)], T * h),
        k("sum_and_scatter", "reduce", [_a("partials", F32 * T * h), _a("sorted_ids", I64 * T)],
          [_a("w_embed.grad2.rows", F32 * T * h)], T * h),
        k("tied_accum", "pointwise", [_a("w_embed.grad", F32 * ev), _a("w_embed.grad2", F32 * ev)],
          [_a("w_embed.grad", F32 * ev)], ev),
    ]

def param_sizes(d: LayerDims, layers: int) -> List[Tuple[str, int]]:
    h, I, V = d.hidden, d.intermediate, d.vocab
    qn, kn = d.heads * d.head_dim, d.kv_heads * d.head_dim
    ps = [("w_embed", V * h)]
    for L in range(layers):
        for nm, n in (("q", qn), ("k", kn), ("v", kn)):
            ps.append((f"L{L}.w{nm}", h * n))
            if d.model_type == "qwen2":
                ps.append((f"L{L}.b{nm}", n))
        ps.append((f"L{L}.wo", qn * h))
        if d.model_type == "qwen3":
            ps += [(f"L{L}.w_qnorm", d.head_dim), (f"L{L}.w_knorm", d.head_dim)]
        ps += [(f"L{L}.wg", h * I), (f"L{L}.wu", h * I), (f"L{L}.wd", I * h), (f"L{L}.w_ln1", h), (f"L{L}.w_ln2", h)]
    ps.append(("w_lnf", h))
    if not d.tied_embeddings:
        ps.append(("w_head", V * h))
    return ps

def _mta_launches(sizes: Sequence[int], max_tensors: int, max_blocks: int = 320, chunk: int = 65536) -> List[float]:
    out, blocks, tensors, elems = [], 0, 0, 0.0
    for n in sizes:
        if n == 0:
            continue
        tensors += 1
        chunks = -(-n // chunk)
        for c in range(chunks):
            blocks += 1
            elems += min(chunk, n - c * chunk)
            if (tensors == max_tensors and c == chunks - 1) or blocks == max_blocks:
                out.append(elems)
                blocks, elems = 0, 0.0
                tensors = 0 if c == chunks - 1 else 1
    if blocks:
        out.append(elems)
    return out

def optimizer_step(d: LayerDims, layers: int) -> List[Kernel]:
    ps = param_sizes(d, layers)
    ks = []
    for i, n in enumerate(_mta_launches([1] * len(ps), 110)):
        ks.append(_k("optimizer", "optimizer", "adamw.step_count", "optimizer", [_a(f"steps{i}", F32 * n)],
                     [_a(f"steps{i}", F32 * n)], n))
    for i, n in enumerate(_mta_launches([s for _, s in ps], 36)):
        st = [_a(f"opt{i}.{t}", F32 * n, t == "param") for t in ("param", "grad", "exp_avg", "exp_avg_sq")]
        ks.append(_k("optimizer", "optimizer", "adamw.fused", "optimizer", st,
                     [st[0], st[2], st[3]], ADAMW_FLOPS * n))
    return ks

def train_step_trace(d: LayerDims, layers: int, opts: TrainOpts = None, checkpoint: bool = False) -> List[Kernel]:
    if opts is None:
        raise ValueError("opts with hardware sm_count is required")
    if d.model_type not in ("llama", "qwen2", "qwen3"):
        raise ValueError(f"training trace does not describe model_type {d.model_type!r}")
    if d.model_type == "qwen2":
        qwen2_bias_path(d.qwen2_bias_mode, 2, 2, 2)
    ks = setup_fwd(d)
    for L in range(layers):
        ks += layer_forward(L, d, f"x{L}", f"x{L + 1}", opts)
    ks += head_fwd(d, f"x{layers}", opts)
    ks += head_bwd(d, f"x{layers}", f"gx{layers}", opts)
    for L in reversed(range(layers)):
        ks += layer_backward(L, d, f"gx{L + 1}", f"gx{L}", f"x{L}", opts, recompute=checkpoint)
    emb = embedding_bwd(d, "gx0")
    if d.tied_embeddings:
        ks += emb
    else:
        from dataclasses import replace
        def untie(a):
            return replace(a, tensor=a.tensor.replace("w_embed.grad2", "w_embed.grad"))
        ks += [replace(k, reads=tuple(map(untie, k.reads)), writes=tuple(map(untie, k.writes))) for k in emb[:-1]]
    ks += optimizer_step(d, layers)
    return ks
