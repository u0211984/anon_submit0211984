from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import sim_settings  # noqa: F401  (registers the RAPID_* settings read below)
from opmodel import settings as _settings

FP32 = 4.0
IMPLEMENTATIONS = ("fused", "eager")

SILU_FWD_FLOPS = 4.0
GATE_MUL_FLOPS = 1.0
SILU_AND_MUL_FLOPS = SILU_FWD_FLOPS + GATE_MUL_FLOPS

RMSNORM_SQUARE_ACC_FLOPS = 2.0
RMSNORM_SCALE_WEIGHT_FLOPS = 2.0
RMSNORM_FLOPS = RMSNORM_SQUARE_ACC_FLOPS + RMSNORM_SCALE_WEIGHT_FLOPS
RESIDUAL_ADD_FLOPS = 1.0
FUSED_ADD_RMSNORM_FLOPS = RESIDUAL_ADD_FLOPS + RMSNORM_FLOPS

RMSNORM_FAMILIES = frozenset({"llama", "qwen2", "qwen3", "qwen3_moe", "deepseek_v3", "glm4_moe"})

def uses_rmsnorm(model_type) -> bool:
    return str(model_type or "").strip().lower() in RMSNORM_FAMILIES

@dataclass(frozen=True)
class Access:
    tensor: str
    nbytes: float
    weight: bool = False

@dataclass(frozen=True)
class Kernel:
    op: str

    name: str
    kind: str
    reads: Tuple[Access, ...]
    writes: Tuple[Access, ...]
    flops: float = 0.0
    gemm: Optional[Tuple[int, int, int]] = None
    activity_name: str = ""
    l2_reread: float = 0.0
    smallgrid_rows: int = field(default=0, repr=False)

    @property
    def read_bytes(self) -> float:
        return sum(a.nbytes for a in self.reads)

    @property
    def write_bytes(self) -> float:
        return sum(a.nbytes for a in self.writes)

@dataclass(frozen=True)
class LayerDims:
    tokens: int
    batch: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    act_bytes: float
    weight_bytes: float
    kv_bytes: float
    past: int = 0
    head_tokens: int = 0
    model_type: str = "llama"
    tied_embeddings: bool = True
    qwen2_bias_mode: Optional[str] = None

def implementation(model) -> str:
    impl = str(getattr(model, "implementation", "fused") or "fused").lower()
    if impl not in IMPLEMENTATIONS:
        raise ValueError(f"model_param.implementation must be one of {IMPLEMENTATIONS}, got {impl!r}")
    return impl

def l2_residency_enabled() -> bool:
    return bool(_settings.get("RAPID_L2_RESIDENCY"))

def qwen2_bias_path(mode: Optional[str], m: int, k: int, n: int, *, torch_version: str = "2.10") -> str:
    if mode not in ("cublaslt_epilogue", "addmm_copy_gemm"):
        raise ValueError("qwen2_bias_mode must be cublaslt_epilogue or addmm_copy_gemm")
    if torch_version not in ("2.8", "2.10"):
        raise ValueError(f"undescribed torch version {torch_version!r}")
    ineligible = min(k, n) <= 1 or (torch_version == "2.10" and m <= 1)
    return "addmm_copy_gemm" if ineligible else mode

def _a(t, n, w=False):
    return Access(t, float(n), w)

def rmsnorm_eager(op: str, x: str, y: str, rows: int, d: LayerDims, act_name: str, weight: str) -> List[Kernel]:
    e, a = float(rows) * d.hidden, d.act_bytes
    t = f"{y}.tmp"
    k = lambda n, kind, r, w, f: Kernel(op, f"rmsnorm.{n}", kind, tuple(r), tuple(w), f, None, act_name)  # noqa: E731
    return [
        k("to_fp32", "pointwise", [_a(x, a * e)], [_a(t + ".x32", FP32 * e)], 0.0),
        k("pow", "pointwise", [_a(t + ".x32", FP32 * e)], [_a(t + ".sq", FP32 * e)], e),
        k("mean", "reduce", [_a(t + ".sq", FP32 * e)], [_a(t + ".var", FP32 * rows)], e),
        k("add_eps", "pointwise", [_a(t + ".var", FP32 * rows)], [_a(t + ".ve", FP32 * rows)], rows),
        k("rsqrt", "pointwise", [_a(t + ".ve", FP32 * rows)], [_a(t + ".rs", FP32 * rows)], rows),
        k("mul", "pointwise", [_a(t + ".x32", FP32 * e), _a(t + ".rs", FP32 * rows)],
          [_a(t + ".n32", FP32 * e)], e),
        k("to_act", "pointwise", [_a(t + ".n32", FP32 * e)], [_a(t + ".n", a * e)], 0.0),
        k("mul_weight", "pointwise", [_a(weight, d.weight_bytes * d.hidden, True), _a(t + ".n", a * e)],
          [_a(y, a * e)], e),
    ]

def residual_add(op: str, x: str, r: str, out: str, rows: int, d: LayerDims, act_name: str) -> Kernel:
    e = float(rows) * d.hidden
    return Kernel(op, "residual_add", "pointwise", (_a(x, d.act_bytes * e), _a(r, d.act_bytes * e)),
                  (_a(out, d.act_bytes * e),), e, None, act_name)

def rope_eager(x: str, out: str, rows: int, heads: int, d: LayerDims, cos: str, sin: str) -> List[Kernel]:
    e = float(rows) * heads * d.head_dim
    a = d.act_bytes
    cs = float(rows) * d.head_dim * a
    t = f"{out}.tmp"
    k = lambda n, r, w, f, rr=0.0: Kernel("rope", f"rope.{n}", "pointwise", tuple(r), tuple(w), f, None, "rope", rr)  # noqa: E731

    return [
        k("mul_cos", [_a(x, a * e), _a(cos, cs)], [_a(t + ".c", a * e)], e),
        k("neg", [_a(x, a * e / 2)], [_a(t + ".neg", a * e / 2)], e / 2),
        k("cat", [_a(t + ".neg", a * e / 2), _a(x, a * e / 2)], [_a(t + ".rot", a * e)], 0.0),
        k("mul_sin", [_a(t + ".rot", a * e), _a(sin, cs)], [_a(t + ".s", a * e)], e, (heads - 1) * cs),
        k("add", [_a(t + ".c", a * e), _a(t + ".s", a * e)], [_a(out, a * e)], e),
    ]

def kv_update_eager(name: str, new: str, cache_in: str, cache_out: str, d: LayerDims) -> Kernel:
    per_tok = float(d.batch) * d.kv_heads * d.head_dim * d.kv_bytes
    new_b = per_tok * (d.tokens / max(1, d.batch))
    old_b = per_tok * d.past
    reads = [_a(new, new_b)] + ([_a(cache_in, old_b)] if old_b else [])
    return Kernel("attention", f"kv_update.{name}", "pointwise", tuple(reads), (_a(cache_out, old_b + new_b),),
                  0.0, None, "kv_update")

def swiglu_eager(g: str, u: str, out: str, rows: int, d: LayerDims, silu_flops_per_element: float) -> List[Kernel]:
    e = float(rows) * d.intermediate
    a = d.act_bytes
    return [
        Kernel("MLP", "silu", "pointwise", (_a(g, a * e),), (_a(g + ".silu", a * e),), silu_flops_per_element * e,
               None, "pointwise-swiglu-f"),
        Kernel("MLP", "mul", "pointwise", (_a(g + ".silu", a * e), _a(u, a * e)), (_a(out, a * e),), e, None,
               "pointwise-swiglu-f"),
    ]

def _gemm(op: str, name: str, x: str, w: str, y: str, m: int, k: int, n: int, d: LayerDims) -> Kernel:
    a = d.act_bytes
    return Kernel(op, name, "gemm", (_a(x, a * m * k), _a(w, d.weight_bytes * k * n, True)), (_a(y, a * m * n),),
                  0.0, (m, k, n), name)

def layer_trace(layer: int, d: LayerDims, impl: str, x_in: str, x_out: str, silu_flops_per_element: float,
                use_rope: bool = True) -> List[Kernel]:
    if d.model_type not in ("llama", "qwen2", "qwen3"):
        raise ValueError(f"inference trace does not describe model_type {d.model_type!r}")
    p = f"L{layer}"
    T, h, a = d.tokens, d.hidden, d.act_bytes
    qn, kn = d.heads * d.head_dim, d.kv_heads * d.head_dim
    ks: List[Kernel] = []

    if impl == "eager":
        ks += rmsnorm_eager("layernorm1", x_in, f"{p}.y1", T, d, "pointwise-layernorm-f", f"{p}.w_ln1")
    else:
        ks.append(Kernel("layernorm1", "rmsnorm", "pointwise", (_a(x_in, a * T * h), _a(f"{p}.w_ln1", d.weight_bytes * h, True)),
                         (_a(f"{p}.y1", a * T * h),), 0.0, None, "pointwise-layernorm-f"))

    if impl == "eager":
        for name, n, heads in (("q", qn, d.heads), ("k", kn, d.kv_heads), ("v", kn, d.kv_heads)):
            proj = _gemm("qkv_proj", f"{name}_proj", f"{p}.y1", f"{p}.w{name}", f"{p}.{name}", T, h, n, d)
            if d.model_type == "qwen2":
                mode = qwen2_bias_path(d.qwen2_bias_mode, T, h, n)
                if mode == "cublaslt_epilogue":
                    proj = replace(proj, reads=proj.reads + (_a(f"{p}.b{name}", d.weight_bytes * n, True),), flops=T * n)
                elif mode == "addmm_copy_gemm":
                    ks.append(Kernel("qkv_proj", f"bias_copy_{name}", "pointwise", (_a(f"{p}.b{name}", d.weight_bytes * n, True),),
                                     (_a(f"{p}.{name}", a * T * n),), 0.0, None, "qkv_proj"))
                    proj = replace(proj, reads=proj.reads + (_a(f"{p}.{name}", a * T * n),), flops=T * n)
                else:
                    raise ValueError("qwen2_bias_mode must be cublaslt_epilogue or addmm_copy_gemm")
            ks.append(proj)
            if d.model_type == "qwen3" and name != "v":
                ks += rmsnorm_eager("qkv_proj", f"{p}.{name}", f"{p}.{name}norm", T * heads,
                                    replace(d, hidden=d.head_dim), "pointwise-layernorm-f", f"{p}.w_{name}norm")
    else:
        proj = _gemm("qkv_proj", "qkv_proj", f"{p}.y1", f"{p}.wqkv", f"{p}.qkv", T, h, qn + 2 * kn, d)
        if d.model_type == "qwen2":
            n = qn + 2 * kn
            mode = qwen2_bias_path(d.qwen2_bias_mode, T, h, n)
            if mode == "cublaslt_epilogue":
                proj = replace(proj, reads=proj.reads + (_a(f"{p}.bqkv", d.weight_bytes * n, True),), flops=T * n)
            elif mode == "addmm_copy_gemm":
                ks.append(Kernel("qkv_proj", "bias_copy_qkv", "pointwise", (_a(f"{p}.bqkv", d.weight_bytes * n, True),),
                                 (_a(f"{p}.qkv", a * T * n),), 0.0, None, "qkv_proj"))
                proj = replace(proj, reads=proj.reads + (_a(f"{p}.qkv", a * T * n),), flops=T * n)
            else:
                raise ValueError("qwen2_bias_mode must be cublaslt_epilogue or addmm_copy_gemm")
        ks.append(proj)
        if d.model_type == "qwen3":
            for name, n, heads in (("q", qn, d.heads), ("k", kn, d.kv_heads)):
                ks.append(Kernel("qkv_proj", f"{name}_norm", "pointwise",
                                 (_a(f"{p}.qkv", a * T * n), _a(f"{p}.w_{name}norm", d.weight_bytes * d.head_dim, True)),
                                 (_a(f"{p}.{name}norm", a * T * n),), 4 * T * n + 2 * T * heads,
                                 None, "pointwise-layernorm-f"))

    q_src, k_src, v_src = (f"{p}.q", f"{p}.k", f"{p}.v") if impl == "eager" else (f"{p}.qkv",) * 3
    if d.model_type == "qwen3":
        q_src, k_src = f"{p}.qnorm", f"{p}.knorm"
    if use_rope:
        if impl == "eager":
            ks += rope_eager(q_src, f"{p}.qr", T, d.heads, d, "cos", "sin")
            ks += rope_eager(k_src, f"{p}.kr", T, d.kv_heads, d, "cos", "sin")
            q_src, k_src = f"{p}.qr", f"{p}.kr"
        else:
            e = float(T) * (d.heads + d.kv_heads) * d.head_dim
            if d.model_type == "qwen3":
                ks.append(Kernel("rope", "rope", "pointwise", (_a(q_src, a * T * qn), _a(k_src, a * T * kn), _a("cos", T * d.head_dim * a)),
                                 (_a(q_src, a * T * qn), _a(k_src, a * T * kn)), 0.0, None, "rope"))
            else:
                ks.append(Kernel("rope", "rope", "pointwise", (_a(f"{p}.qkv", a * e), _a("cos", T * d.head_dim * a)),
                                 (_a(f"{p}.qkv", a * e),), 0.0, None, "rope"))

    kv_b = float(d.batch) * d.kv_heads * d.head_dim * d.kv_bytes
    if impl == "eager":
        ks.append(kv_update_eager("k", k_src, f"{p}.kcache_old", f"{p}.kcache", d))
        ks.append(kv_update_eager("v", v_src, f"{p}.vcache_old", f"{p}.vcache", d))
        k_all, v_all = f"{p}.kcache", f"{p}.vcache"
        q_read = [_a(q_src, a * T * qn)]
        kv_reads = [_a(k_all, kv_b * (d.past + T / max(1, d.batch))), _a(v_all, kv_b * (d.past + T / max(1, d.batch)))]
    else:
        q_read = ([_a(q_src, a * T * qn), _a(k_src, a * T * kn), _a(v_src, a * T * kn)]
                  if d.model_type == "qwen3" else [_a(q_src, a * T * (qn + 2 * kn))])
        kv_reads = ([_a(f"{p}.kcache_old", kv_b * d.past), _a(f"{p}.vcache_old", kv_b * d.past)] if d.past else [])
    ks.append(Kernel("attention", "attention", "attention", tuple(q_read + kv_reads), (_a(f"{p}.o", a * T * qn),),
                     0.0, None, "attention"))
    ks.append(_gemm("output_proj", "o_proj", f"{p}.o", f"{p}.wo", f"{p}.ao", T, qn, h, d))
    ks.append(residual_add("layernorm1", x_in, f"{p}.ao", f"{p}.r1", T, d, "pointwise-residual-f"))

    if impl == "eager":
        ks += rmsnorm_eager("layernorm2", f"{p}.r1", f"{p}.y2", T, d, "pointwise-layernorm-f", f"{p}.w_ln2")
        ks.append(_gemm("MLP", "gate_proj", f"{p}.y2", f"{p}.wg", f"{p}.g", T, h, d.intermediate, d))
        silu, mul = swiglu_eager(f"{p}.g", f"{p}.u", f"{p}.m", T, d, silu_flops_per_element)
        ks.append(silu)
        ks.append(_gemm("MLP", "up_proj", f"{p}.y2", f"{p}.wu", f"{p}.u", T, h, d.intermediate, d))
        ks.append(mul)
    else:
        ks.append(Kernel("layernorm2", "rmsnorm", "pointwise", (_a(f"{p}.r1", a * T * h), _a(f"{p}.w_ln2", d.weight_bytes * h, True)),
                         (_a(f"{p}.y2", a * T * h),), 0.0, None, "pointwise-layernorm-f"))
        ks.append(_gemm("MLP", "gate_up_proj", f"{p}.y2", f"{p}.wgu", f"{p}.gu", T, h, 2 * d.intermediate, d))
        e = float(T) * d.intermediate
        ks.append(Kernel("MLP", "silu_and_mul", "pointwise", (_a(f"{p}.gu", 2 * a * e),), (_a(f"{p}.m", a * e),),
                         0.0, None, "pointwise-swiglu-f"))
    ks.append(_gemm("MLP", "down_proj", f"{p}.m", f"{p}.wd", f"{p}.dn", T, d.intermediate, h, d))
    ks.append(residual_add("layernorm2", f"{p}.r1", f"{p}.dn", x_out, T, d, "pointwise-residual-f"))
    return ks

def head_trace(d: LayerDims, impl: str, x_in: str) -> List[Kernel]:
    T, h, a = d.tokens, d.hidden, d.act_bytes
    if impl == "eager":
        ks = rmsnorm_eager("linear_softmax", x_in, "yf", T, d, "pointwise-final-norm-f", "w_lnf")
    else:
        ks = [Kernel("linear_softmax", "rmsnorm", "pointwise", (_a(x_in, a * T * h), _a("w_lnf", d.weight_bytes * h, True)),
                     (_a("yf", a * T * h),), 0.0, None, "pointwise-final-norm-f")]
    rows = d.head_tokens or T
    ks.append(Kernel("linear_softmax", "lm_head", "gemm",
                     (_a("yf", a * rows * h), _a("w_head", d.weight_bytes * h * d.vocab, True)),
                     (_a("logits", a * rows * d.vocab),), 0.0, (rows, h, d.vocab), "linear_softmax_f"))
    return ks

def step_trace(d: LayerDims, impl: str, layers: int, silu_flops_per_element: float, use_rope: bool = True) -> List[Kernel]:
    T, h, a = d.tokens, d.hidden, d.act_bytes
    ks = [Kernel("embedding", "embedding", "gather", (_a("w_embed.rows", d.weight_bytes * T * h, True),),
                 (_a("x0", a * T * h),), 0.0, None, "embedding_f")]
    for layer in range(layers):
        ks += layer_trace(layer, d, impl, f"x{layer}", f"x{layer + 1}", silu_flops_per_element, use_rope)
    ks += head_trace(d, impl, f"x{layers}")
    return ks
