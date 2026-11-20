from dataclasses import dataclass, replace
from functools import lru_cache
import math
from collections import Counter
from numbers import Integral
from typing import Callable, Optional, Tuple

from kernel_trace import Kernel, SILU_AND_MUL_FLOPS
from kernel_trace_vllm import _a, _k

def ceildiv(a, b):
    return (a + b - 1) // b

def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)

@dataclass(frozen=True)
class Tile:
    block_m: int
    block_n: int
    block_k: int
    group_m: int

def vllm_default_config(tokens: int, local_experts: int) -> Tile:
    _integer(tokens, "tokens")
    _integer(local_experts, "local_experts")
    return Tile(16, 32, 64, 1) if tokens <= local_experts else Tile(64, 64, 32, 8)

def balanced_counts(tokens: int, experts: int, top_k: int, offset: int = 0, ep: int = 1) -> Tuple[int, ...]:
    _integer(tokens, "tokens", minimum=0)
    _integer(experts, "experts")
    _integer(top_k, "top_k")
    _integer(offset, "offset", minimum=0)
    if top_k > experts:
        raise ValueError("routing requires tokens >= 0 and 0 < top_k <= experts")
    order = _expert_order(experts, ep)
    q, r = divmod(tokens * top_k, experts)
    counts = [q] * experts
    for slot in range(r):
        counts[order[(slot + offset) % experts]] += 1
    return tuple(counts)

@lru_cache(maxsize=128)
def _expert_order(experts, ep):
    _integer(ep, "ep")
    if ep > experts:
        raise ValueError("ep cannot exceed experts")
    width, remainder = divmod(experts, ep)
    starts = [rank * width + min(rank, remainder) for rank in range(ep)]

    return tuple(starts[rank] + local for local in range(width + bool(remainder))
                 for rank in range(ep) if local < width + (rank < remainder))

@lru_cache(maxsize=16)
def _local_routing_ranges(tokens, experts, top_k, routing_offset, first, last, ep):
    def runs(values):
        result = []
        for value in values:
            if result and result[-1][1] == value:
                result[-1] = (result[-1][0], value + 1)
            else:
                result.append((value, value + 1))
        return tuple(result)
    if first == 0 and last == experts and tokens * top_k:
        return ((0, tokens),), ((0, tokens * top_k),)
    order = _expert_order(experts, ep)
    slots = tuple(s for s in range(tokens * top_k)
                  if first <= order[(routing_offset + s) % experts] < last)
    return runs(dict.fromkeys(s // top_k for s in slots)), runs(slots)

def _runs(values):
    result = []
    for value in values:
        if result and result[-1][1] == value:
            result[-1] = (result[-1][0], value + 1)
        else:
            result.append((value, value + 1))
    return tuple(result)

@lru_cache(maxsize=64)
def uniform_occupancy(tokens, experts, top_k, ep=1):
    _integer(tokens, "tokens", minimum=0)
    _integer(experts, "experts")
    _integer(top_k, "top_k")
    _integer(ep, "ep")
    q, rem = divmod(experts, ep)
    sizes = tuple(q + (r < rem) for r in range(ep))
    inactive = experts * (1 - top_k / experts) ** tokens
    total = experts - inactive
    if inactive < 1e-9:
        return float(experts), float(max(sizes))
    if ep == 1:
        return total, total
    if ep != 2:
        return total, _uniform_occupancy_max_by_union(tokens, experts, top_k, sizes)
    if tokens > 4096:
        raise NotImplementedError("uniform_occupancy exact recursion is bounded to 4096 tokens")
    from math import comb
    q0, q1 = sizes
    norm = comb(experts, top_k)
    state = {(0, 0): 1.0}
    for _ in range(tokens):
        nxt = {}
        for (a, b), prob in state.items():
            for j0 in range(min(top_k, q0 - a) + 1):
                for j1 in range(min(top_k - j0, q1 - b) + 1):
                    w = comb(q0 - a, j0) * comb(q1 - b, j1) * comb(a + b, top_k - j0 - j1)
                    if w:
                        key = (a + j0, b + j1)
                        nxt[key] = nxt.get(key, 0.0) + prob * w / norm
        state = nxt
    mean_total = sum(p * (a + b) for (a, b), p in state.items())
    if abs(mean_total - total) > 1e-6 * max(1.0, total):
        raise AssertionError("uniform_occupancy recursion does not reproduce the occupancy mean")
    return total, sum(p * max(a, b) for (a, b), p in state.items())

def _uniform_occupancy_max_by_union(tokens, experts, top_k, sizes):
    if tokens > 4096:
        raise NotImplementedError("uniform_occupancy exact union law is bounded to 4096 tokens")
    from fractions import Fraction
    from math import comb
    e, k = experts, top_k
    power = [comb(t, k) ** tokens for t in range(e + 1)]
    exact = [sum((-1) ** j * comb(s, j) * power[s - j] for j in range(s + 1)) for s in range(e + 1)]
    norm = power[e]
    if sum(comb(e, s) * exact[s] for s in range(e + 1)) != norm:
        raise AssertionError("uniform_occupancy union law does not normalise")
    mean = Fraction(sum(s * comb(e, s) * exact[s] for s in range(e + 1)), norm)
    expected = e - e * (1 - Fraction(k, e)) ** tokens
    if abs(float(mean - expected)) > 1e-9 * max(1.0, float(expected)):
        raise AssertionError("uniform_occupancy union law does not reproduce the occupancy mean")
    top = max(sizes)
    numerator = 0
    for m in range(top):
        poly = [1]
        for n in sizes:
            factor = [comb(n, j) for j in range(min(m, n) + 1)]
            out = [0] * (len(poly) + len(factor) - 1)
            for i, a in enumerate(poly):
                if a:
                    for j, b in enumerate(factor):
                        out[i + j] += a * b
            poly = out
        numerator += sum(exact[s] * poly[s] for s in range(min(len(poly), e + 1)))
    return float(Fraction(top * norm - numerator, norm))

@lru_cache(maxsize=4096)
def multinomial_expected_tiles(tokens, experts, top_k, tile_m):
    _integer(tokens, "tokens", minimum=0)
    _integer(experts, "experts")
    _integer(top_k, "top_k")
    _integer(tile_m, "tile_m")
    if top_k > experts:
        raise ValueError("top_k cannot exceed experts")
    p = top_k / experts
    if tokens == 0:
        return 0.0, 0.0
    if p == 1.0:
        return float(ceildiv(tokens, tile_m)), 1.0
    lp, lq, lg = math.log(p), math.log1p(-p), math.lgamma(tokens + 1)
    logs = [lg - math.lgamma(c + 1) - math.lgamma(tokens - c + 1) + c * lp + (tokens - c) * lq
            for c in range(tokens + 1)]
    top = max(logs)
    weights = [math.exp(x - top) for x in logs]
    total = math.fsum(weights)
    tiles = math.fsum(w * ceildiv(c, tile_m) for c, w in enumerate(weights)) / total
    active = -math.expm1(tokens * lq)
    return tiles, active

def multinomial_rank_active(tokens, experts, top_k, ep, rank, offset):
    _integer(ep, "ep")
    if ep == 1:
        return multinomial_expected_tiles(tokens, experts, top_k, 1)[1]
    if experts % ep:
        raise ValueError("multinomial_rank_active requires ep to divide num_experts")
    local = experts // ep
    total, bottleneck = uniform_occupancy(tokens, experts, top_k, ep)
    if rank == offset % ep:
        return bottleneck / local
    return (total - bottleneck) / (experts - local)

def _round_half_up(x):
    return int(math.floor(x + 0.5))

@lru_cache(maxsize=64)
def uniform_occupancy_assignment(tokens, experts, top_k, offset=0, ep=1):
    if tokens == 0:
        return ()
    cycle = uniform_occupancy_cycle(tokens, experts, top_k, offset, ep)
    return tuple(cycle[s % len(cycle)] for s in range(tokens * top_k))

@lru_cache(maxsize=4096)
def uniform_occupancy_cycle(tokens, experts, top_k, offset=0, ep=1):
    total, bottleneck = uniform_occupancy(tokens, experts, top_k, ep)
    slots = tokens * top_k
    q, rem = divmod(experts, ep)
    sizes = [q + (r < rem) for r in range(ep)]
    starts = [r * q + min(r, rem) for r in range(ep)]
    active = min(slots, experts, max(top_k, _round_half_up(total)))
    hot = offset % ep
    counts = [0] * ep
    counts[hot] = min(sizes[hot], active, _round_half_up(bottleneck)) if ep > 1 else active
    rest = active - counts[hot]
    others = [r for r in range(ep) if r != hot]
    for i, r in enumerate(others):
        counts[r] = min(sizes[r], rest // len(others) + (i < rest % len(others)))
    if sum(counts) != active or (ep > 1 and max(counts) != counts[hot]):
        raise ValueError("uniform_occupancy cannot place the activated experts on the ranks")
    picks = []
    for r in range(ep):
        for j in range(counts[r]):
            expert = starts[r] + (offset + j) % sizes[r]
            picks.append(((j + 0.5) / counts[r], r, expert))
    return tuple(e for _, _, e in sorted(picks))

def _cycle_local_slots(cycle, base, length, first, last):
    period = len(cycle)
    offsets = sorted((p - base) % period for p, expert in enumerate(cycle) if first <= expert < last)
    slots = []
    for block in range(0, length, period):
        if block + period <= length:
            slots.extend(block + o for o in offsets)
        else:
            slots.extend(block + o for o in offsets if block + o < length)
    return tuple(slots)

def _cycle_counts(cycle, base, length, experts):
    full, rem = divmod(length, len(cycle))
    counts = [0] * experts
    for expert in cycle:
        counts[expert] += full
    for t in range(rem):
        counts[cycle[(base + t) % len(cycle)]] += 1
    return tuple(counts)

@lru_cache(maxsize=8)
def routed_experts(tokens, experts, top_k, offset=0, ep=1,
                   policy="multinomial_expected", imbalance=1.0):
    _integer(tokens, "tokens", minimum=0)
    _integer(experts, "experts")
    _integer(top_k, "top_k")
    _integer(offset, "offset", minimum=0)
    if top_k > experts:
        raise ValueError("top_k cannot exceed experts")
    if policy not in ("multinomial_expected", "integer_balanced_distinct", "one_hot_distinct", "uniform_occupancy"):
        raise ValueError("unsupported MoE routing_policy")
    if policy == "multinomial_expected":
        if imbalance != 1:
            raise ValueError("multinomial_expected requires expert_imbalance_factor 1")
        policy = "integer_balanced_distinct"
    if policy == "uniform_occupancy":
        if imbalance != 1:
            raise ValueError("uniform_occupancy fixes its own imbalance; expert_imbalance_factor must be 1")
        return uniform_occupancy_assignment(tokens, experts, top_k, offset, ep)
    if not math.isfinite(imbalance) or not 1 <= imbalance <= experts/top_k:
        raise ValueError("distinct top-k routing requires 1 <= expert_imbalance_factor <= num_experts/top_k")
    if policy == "integer_balanced_distinct" and imbalance != 1:
        raise ValueError("non-unit expert_imbalance_factor requires routing_policy=one_hot_distinct")
    order = _expert_order(experts, ep)
    if imbalance == 1 or not tokens:
        return tuple(order[(offset+s) % experts] for s in range(tokens*top_k))
    hot = min(tokens, int(math.floor(tokens*top_k*imbalance/experts + 0.5)))
    cold = tuple(e for e in order if e != 0)
    cold_slot = offset % len(cold)
    result = []
    for row in range(tokens):
        rotated_row = (row + offset) % tokens
        is_hot = ((rotated_row+1)*hot)//tokens > (rotated_row*hot)//tokens
        if is_hot:
            result.append(0)
        count = top_k - int(is_hot)
        result.extend(cold[(cold_slot+j) % len(cold)] for j in range(count))
        cold_slot += count
    return tuple(result)

@dataclass(frozen=True)
class MoeSpec:
    hidden: int
    intermediate: int
    experts: int
    top_k: int
    tp: int
    ep: int
    tile_rule: str
    chunk_size: int = 32768
    rapid_tile_selector: Optional[Callable] = None
    routing_policy: str = "multinomial_expected"
    backend: str = "naive"
    dtype: str = "bf16"
    quantization: Optional[str] = None

    weight_bytes: float = 2
    eplb: bool = False
    shared_experts: bool = False
    attention_dp: int = 1
    permute: bool = False
    expert_imbalance_factor: float = 1.0

    family: str = "qwen3_moe"
    shared_intermediate: int = 0
    routed_scaling_factor: float = 1.0
    scoring_func: str = "softmax"
    n_group: int = 0
    topk_group: int = 0
    renormalize: bool = True
    score_bias: bool = False
    fused_grouped_topk: bool = False

    @classmethod
    def from_config(cls, cfg, *, tp=1, ep=1, tile_rule=None, chunk_size=32768,
                    rapid_tile_selector=None):
        if is_deepseek_moe(cfg):
            return cls._deepseek(cfg, tp=tp, ep=ep, tile_rule=tile_rule, chunk_size=chunk_size,
                                 rapid_tile_selector=rapid_tile_selector)
        if cfg.get("model_type") != "qwen3_moe":
            raise ValueError("MoE adapter supports only qwen3_moe and DeepSeek V2/V3 MoE")
        if cfg.get("linear_weight_bytes") is not None:
            raise NotImplementedError("precision.linear_weights is described for DeepSeek MoE only")
        if tile_rule not in ("vllm_default_config", "rapid_grouped"):
            raise ValueError("moe_tile_rule must be set to vllm_default_config or rapid_grouped")
        tp = _integer(tp, "tp")
        ep = _integer(ep, "ep")
        chunk_size = _integer(chunk_size, "chunk_size")
        dp = _integer(cfg.get("attention_dp", cfg.get("moe_dp", 1)), "attention_dp")

        if ep > 1 and ep != tp * dp:
            raise ValueError("expert parallel world must equal attention TP * attention DP")
        for key in ("moe_backend", "expert_parallel_backend", "all2all_backend"):
            if cfg.get(key) not in (None, "naive"):
                raise ValueError(f"unsupported {key}={cfg[key]!r}: only naive MoE backend is supported")
        backend = _all2all_backend(cfg, tp=tp, ep=ep, dp=dp)
        if (cfg.get("hidden_act", "silu") != "silu" or not cfg.get("norm_topk_prob", True)
                or cfg.get("decoder_sparse_step", 1) != 1 or cfg.get("mlp_only_layers", [])
                or cfg.get("quantization_config") or cfg.get("attention_bias", False)
                or cfg.get("use_sliding_window", False) or cfg.get("shared_experts")
                or cfg.get("n_shared_experts") or cfg.get("num_shared_experts")
                or cfg.get("use_grouped_topk") or cfg.get("scoring_func", "softmax") != "softmax"
                or cfg.get("enable_eplb", False) or cfg.get("eplb", False)
                or cfg.get("torch_dtype", "bfloat16") not in ("bfloat16", "bf16")):
            raise ValueError("unsupported qwen3_moe variant: requires all-MoE, bf16, SiLU, normalized top-k, no shared experts, EPLB, bias, quantization or sliding window")
        h, i, e, k = (_integer(cfg[x], x) for x in ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok"))
        if min(h, i, e, k) <= 0 or k > e or (ep == 1 and i % (tp * dp)) or ep > e:
            raise ValueError("invalid MoE dimensions or TP must divide moe_intermediate_size")
        if e not in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            raise NotImplementedError("MoE trace describes the fused top-k path for power-of-two experts <= 256; other expert counts need a separate source sequence")
        if tile_rule == "rapid_grouped" and rapid_tile_selector is None:
            raise ValueError("rapid_grouped requires the operator model grouped tile selector")
        policy = cfg.get("moe_routing_policy", "multinomial_expected")
        imbalance = float(cfg.get("expert_imbalance_factor", 1.0))
        routed_experts(0, e, k, ep=ep, policy=policy, imbalance=imbalance)
        return cls(h, i if ep > 1 else i // (tp * dp), e, k, tp, ep, tile_rule, chunk_size,
                   rapid_tile_selector, routing_policy=policy, attention_dp=dp,
                   expert_imbalance_factor=imbalance, backend=backend)

    @classmethod
    def _deepseek(cls, cfg, *, tp, ep, tile_rule, chunk_size, rapid_tile_selector):
        if tile_rule not in ("vllm_default_config", "rapid_grouped"):
            raise ValueError("moe_tile_rule must be set to vllm_default_config or rapid_grouped")
        tp, ep, chunk_size = _integer(tp, "tp"), _integer(ep, "ep"), _integer(chunk_size, "chunk_size")
        dp = _integer(cfg.get("attention_dp", cfg.get("moe_dp", 1)), "attention_dp")
        if ep > 1 and ep != tp * dp:
            raise ValueError("expert parallel world must equal attention TP * attention DP")
        for key in ("moe_backend", "expert_parallel_backend", "all2all_backend"):
            if cfg.get(key) not in (None, "naive"):
                raise ValueError(f"unsupported {key}={cfg[key]!r}: only naive MoE backend is supported")
        backend = _all2all_backend(cfg, tp=tp, ep=ep, dp=dp)
        for key in ("scoring_func", "topk_method", "n_group", "topk_group", "norm_topk_prob", "routed_scaling_factor",
                    "n_shared_experts"):
            if cfg.get(key) is None and not (key == "n_shared_experts" and key in cfg):
                raise ValueError(f"DeepSeek MoE requires the published config field {key} (no default is assumed)")
        if (cfg.get("hidden_act", "silu") != "silu" or cfg.get("quantization_config") or cfg.get("enable_eplb", False)
                or cfg.get("torch_dtype", "bfloat16") not in ("bfloat16", "bf16")):
            raise ValueError("DeepSeek MoE description is bf16, SiLU, no quantization and no EPLB")
        scoring, method = str(cfg["scoring_func"]), str(cfg["topk_method"])
        if scoring not in ("softmax", "sigmoid"):
            raise ValueError("grouped_topk scoring_func must be softmax or sigmoid (fused_moe.py:971-976)")
        if method not in ("greedy", "group_limited_greedy", "noaux_tc"):
            raise ValueError("topk_method must be greedy, group_limited_greedy or noaux_tc")
        h, i, e, k = (_integer(cfg[x], x) for x in ("hidden_size", "moe_intermediate_size", "num_experts", "num_experts_per_tok"))
        g, tg = _integer(cfg["n_group"], "n_group"), _integer(cfg["topk_group"], "topk_group")
        if k > e or e % g or tg > g or ep > e or (ep == 1 and i % (tp * dp)):
            raise ValueError("invalid DeepSeek MoE dimensions (top_k <= experts, n_group | experts, topk_group <= n_group)")
        n_shared = int(cfg.get("n_shared_experts") or 0)
        if n_shared < 0 or (n_shared * i) % tp:
            raise ValueError("TP must divide the shared-expert width n_shared_experts * moe_intermediate_size")
        if tile_rule == "rapid_grouped" and rapid_tile_selector is None:
            raise ValueError("rapid_grouped requires the operator model grouped tile selector")
        policy = cfg.get("moe_routing_policy", "multinomial_expected")
        imbalance = float(cfg.get("expert_imbalance_factor", 1.0))
        routed_experts(0, e, k, ep=ep, policy=policy, imbalance=imbalance)
        bias = method == "noaux_tc"

        fused = bool(cfg.get("vllm_use_fused_moe_grouped_topk", True)) and bias and g <= 32 and k <= 32
        return cls(h, i if ep > 1 else i // (tp * dp), e, k, tp, ep, tile_rule, chunk_size,
                   rapid_tile_selector, routing_policy=policy, attention_dp=dp,
                   expert_imbalance_factor=imbalance, backend=backend, family="deepseek", shared_experts=n_shared > 0,
                   shared_intermediate=n_shared * i // tp, routed_scaling_factor=float(cfg["routed_scaling_factor"]),
                   scoring_func=scoring, n_group=g, topk_group=tg, renormalize=bool(cfg["norm_topk_prob"]),
                   score_bias=bias, fused_grouped_topk=fused,
                   **({"weight_bytes": float(cfg["linear_weight_bytes"])}
                      if cfg.get("linear_weight_bytes") is not None else {}))

def _all2all_backend(cfg, *, tp, ep, dp):
    from program.moe_rank import DEEPEP_BACKENDS
    backend = cfg.get("vllm_all2all_backend", "naive")
    if backend == "pplx":
        raise NotImplementedError("vllm_all2all_backend=pplx is not described")
    if backend != "naive" and backend not in DEEPEP_BACKENDS:
        raise ValueError(f"unknown vllm_all2all_backend {backend!r}")
    if backend == "naive" or not (ep > 1 and dp > 1):
        return "naive"
    if tp != 1:
        raise NotImplementedError("the DeepEP all-to-all description requires attention tp = 1")
    return backend

def is_deepseek_moe(cfg) -> bool:
    return (str(cfg.get("model_type", "")).lower() in ("deepseek_v2", "deepseek_v3")
            and int(cfg.get("num_experts") or 0) > 0)

def source_moe(cfg) -> bool:
    return str(cfg.get("model_type", "")).lower() == "qwen3_moe" or is_deepseek_moe(cfg)

def deepseek_moe_layers(cfg) -> Tuple[int, ...]:
    first, freq = int(cfg["first_k_dense_replace"]), int(cfg["moe_layer_freq"])
    return tuple(i for i in range(int(cfg["num_hidden_layers"])) if i >= first and i % freq == 0)

@dataclass(frozen=True)
class GroupedGemm:
    tile_rule: str
    tile: Tile
    tokens: int
    top_k: int
    expert_counts: Tuple[int, ...]
    padded_counts: Tuple[int, ...]
    local_experts: Tuple[int, ...]
    local_blocks: int
    remote_blocks: int
    allocated_sorted_ids: int
    allocated_expert_blocks: int
    logical_grid: int
    launched_grid: int
    useful_flops: int
    padded_flops: int
    routing_offset: int
    chunk_start: int
    input_width: int
    output_width: int

@dataclass(frozen=True)
class KernelDetails:
    source: str = ""
    arithmetic_ops: Tuple[Tuple[str, int], ...] = ()
    notes: Tuple[str, ...] = ()
    aliases: Tuple[Tuple[str, str], ...] = ()

@dataclass(frozen=True)
class MoeKernel(Kernel):
    grouped: Optional[GroupedGemm] = None
    details: KernelDetails = KernelDetails()
    spec: Optional[MoeSpec] = None

def _deepseek_router(spec, add, p, logits, tokens):
    e, k, g, tg, T = spec.experts, spec.top_k, spec.n_group, spec.topk_group, tokens
    def src(lines):
        return KernelDetails(source="vllm 0.10.2 " + lines,
                             notes=("one launch per torch op (vllm_grouped_topk_launches=eager_ops; grouped_topk is "
                                    "torch.compile'd, compiled launch grouping not in source)",))
    s_dt = 2
    if spec.scoring_func == "softmax":
        add("moe.gt_softmax", "reduce", [_a(logits, 2*T*e)], [_a(p+".gt_scores", s_dt*T*e)], T*(3*e-1),
            details=src("fused_moe.py:973-974, 1033-1034"))
    else:
        add("moe.gt_sigmoid", "pointwise", [_a(logits, 2*T*e)], [_a(p+".gt_scores", s_dt*T*e)], 3*T*e,
            details=src("fused_moe.py:975-976, 1035-1036"))
    if spec.fused_grouped_topk:
        add("moe.gt_add_bias", "pointwise", [_a(p+".gt_scores", s_dt*T*e), _a(p+".router_bias", 4*e, True)],
            [_a(p+".gt_biased32", 4*T*e)], T*e, details=src("fused_moe.py:1040"))
        add("moe.gt_bias_cast", "pointwise", [_a(p+".gt_biased32", 4*T*e)], [_a(p+".gt_biased", s_dt*T*e)],
            details=src("fused_moe.py:1041-1043 scores_with_bias.to(scores.dtype)"))
        add("moe.grouped_topk_k2", "reduce", [_a(p+".gt_biased", s_dt*T*e)], [_a(p+".gt_group_scores", s_dt*T*g)],
            T*e, details=KernelDetails(source="vllm 0.10.2 csrc/moe/grouped_topk_kernels.cu:459-484,645-656,708-709"))
        add("moe.grouped_topk_idx", "reduce",
            [_a(p+".gt_scores", s_dt*T*e), _a(p+".gt_group_scores", s_dt*T*g), _a(p+".gt_biased", s_dt*T*e)],
            [_a(p+".gt_values", s_dt*T*k), _a(p+".expert_ids", 4*T*k)], T*(2*k if spec.renormalize else k),
            details=KernelDetails(source="vllm 0.10.2 csrc/moe/grouped_topk_kernels.cu:486-630,662-673,710-713",
                                  notes=("renormalisation and routed_scaling_factor (1.0) inside the kernel",)))
        add("moe.gt_weights_fp32", "pointwise", [_a(p+".gt_values", s_dt*T*k)], [_a(p+".weights", 4*T*k)],
            details=src("fused_moe.py:1044 topk_values.to(float32); topk_indices are int32 already"))
        return
    scores = p+".gt_scores"
    if spec.score_bias:
        add("moe.gt_add_bias", "pointwise", [_a(scores, s_dt*T*e), _a(p+".router_bias", 4*e, True)],
            [_a(p+".gt_biased32", 4*T*e)], T*e, details=src("fused_moe.py:985"))
        scores, b_dt = p+".gt_biased32", 4
        add("moe.gt_group_top2", "reduce", [_a(scores, b_dt*T*e)],
            [_a(p+".gt_top2", b_dt*T*g*2), _a(p+".gt_top2_idx", 8*T*g*2)], details=src("fused_moe.py:986-987 topk(2)"))
        add("moe.gt_group_sum", "reduce", [_a(p+".gt_top2", b_dt*T*g*2)], [_a(p+".gt_group_scores", b_dt*T*g)], T*g,
            details=src("fused_moe.py:987 sum"))
    else:
        b_dt = s_dt
        add("moe.gt_group_max", "reduce", [_a(scores, s_dt*T*e)],
            [_a(p+".gt_group_scores", s_dt*T*g), _a(p+".gt_group_max_idx", 8*T*g)], T*e,
            details=src("fused_moe.py:989-990 max(dim=-1) (values and indices)"))
    add("moe.gt_group_topk", "reduce", [_a(p+".gt_group_scores", b_dt*T*g)],
        [_a(p+".gt_group_vals", b_dt*T*tg), _a(p+".gt_group_idx", 8*T*tg)], details=src("fused_moe.py:991-992"))
    add("moe.gt_group_mask_zeros", "pointwise", [], [_a(p+".gt_group_mask", b_dt*T*g)], details=src("fused_moe.py:993"))
    add("moe.gt_group_mask_scatter", "pointwise", [_a(p+".gt_group_idx", 8*T*tg)],
        [_a(p+".gt_group_mask", b_dt*T*tg)], details=src("fused_moe.py:994"))
    add("moe.gt_score_mask_expand", "pointwise", [_a(p+".gt_group_mask", b_dt*T*g)],
        [_a(p+".gt_score_mask", b_dt*T*e)], details=src("fused_moe.py:995-997 expand().reshape() copy"))
    add("moe.gt_score_mask_bool", "pointwise", [_a(p+".gt_score_mask", b_dt*T*e)], [_a(p+".gt_mask_bool", T*e)],
        details=src("fused_moe.py:998 .bool()"))
    add("moe.gt_score_mask_not", "pointwise", [_a(p+".gt_mask_bool", T*e)], [_a(p+".gt_mask_not", T*e)],
        details=src("fused_moe.py:998 ~"))
    add("moe.gt_masked_fill", "pointwise", [_a(scores, b_dt*T*e), _a(p+".gt_mask_not", T*e)],
        [_a(p+".gt_tmp_scores", b_dt*T*e)], details=src("fused_moe.py:998-999"))
    if spec.score_bias:
        add("moe.gt_topk", "reduce", [_a(p+".gt_tmp_scores", b_dt*T*e)],
            [_a(p+".gt_topk_vals", b_dt*T*k), _a(p+".gt_topk_ids", 8*T*k)], details=src("fused_moe.py:1002"))
        add("moe.gt_gather", "gather", [_a(p+".gt_scores", s_dt*T*e), _a(p+".gt_topk_ids", 8*T*k)],
            [_a(p+".gt_weights_raw", s_dt*T*k)], details=src("fused_moe.py:1004"))
    else:
        add("moe.gt_topk", "reduce", [_a(p+".gt_tmp_scores", b_dt*T*e)],
            [_a(p+".gt_weights_raw", s_dt*T*k), _a(p+".gt_topk_ids", 8*T*k)], details=src("fused_moe.py:1006-1008"))
    w = p+".gt_weights_raw"
    if spec.renormalize:
        add("moe.gt_renorm_sum", "reduce", [_a(w, s_dt*T*k)], [_a(p+".gt_weight_sums", s_dt*T)], T*(k-1),
            details=src("fused_moe.py:1011-1012 sum"))
        add("moe.gt_renorm_divide", "pointwise", [_a(w, s_dt*T*k), _a(p+".gt_weight_sums", s_dt*T)],
            [_a(p+".gt_weights_norm", s_dt*T*k)], T*k, details=src("fused_moe.py:1012 divide"))
        w = p+".gt_weights_norm"
    add("moe.gt_weights_fp32", "pointwise", [_a(w, s_dt*T*k)], [_a(p+".weights", 4*T*k)],
        details=src("fused_moe.py:1016 .to(float32)"))
    add("moe.gt_ids_int32", "pointwise", [_a(p+".gt_topk_ids", 8*T*k)], [_a(p+".expert_ids", 4*T*k)],
        details=src("fused_moe.py:1016 .to(int32)"))

VLLM_MOE_DP_CHUNK_SIZE = 256

def deepep_ll_received_rows(tokens, experts, top_k, local_experts):
    return tokens * top_k * local_experts / experts

def _deepep_ll_experts(spec, add, p, local_tokens, tokens, layer, step, start_expert, local_experts, expert_rank, sent,
                       input_tensor, ids_tensor):
    h, i, e, k = spec.hidden, spec.intermediate, spec.experts, spec.top_k
    if spec.routing_policy != "multinomial_expected":
        raise NotImplementedError("deepep_low_latency is described for moe_routing_policy=multinomial_expected")
    if spec.tile_rule != "vllm_default_config":
        raise NotImplementedError("deepep_low_latency is described with moe_tile_rule=vllm_default_config")
    if local_tokens > VLLM_MOE_DP_CHUNK_SIZE:
        raise NotImplementedError("deepep_low_latency describes one forward_impl_chunked round "
                                  f"(<= {VLLM_MOE_DP_CHUNK_SIZE} tokens per rank, layer.py:1644-1757)")
    el = len(local_experts)
    M = VLLM_MOE_DP_CHUNK_SIZE * spec.ep
    rows_e = tokens * k / e
    rows = deepep_ll_received_rows(tokens, e, k, el)

    received_only = spec.backend == "deepep_low_latency_received"
    span = rows if received_only else el * M
    add("moe.a2a_dispatch", "collective", [_a(input_tensor, 2*local_tokens*h), _a(ids_tensor, 8*local_tokens*k)],
        [_a(p+".ll_x", 2*rows*h)],
        details=KernelDetails(source="vllm 0.10.2 fused_moe/deepep_ll_prepare_finalize.py:151-158",
                              notes=(f"deepep dispatch bytes sent by this rank {sent!r}",)))
    tile = vllm_default_config(M, el)
    bm, bn, bk = tile.block_m, tile.block_n, tile.block_k
    tiles, active = multinomial_expected_tiles(tokens, e, k, bm)
    local_active = multinomial_rank_active(tokens, e, k, spec.ep, expert_rank, layer + step)
    local_tiles = tiles * local_active / active if active else 0.0
    m_tiles = ceildiv(M, bm)
    counts = tuple(rows_e if start_expert <= x < start_expert + el else 0.0 for x in range(e))
    padded = tuple(local_tiles * bm if start_expert <= x < start_expert + el else 0.0 for x in range(e))
    def desc(n, reduction):
        blocks = el * local_tiles
        launched = blocks if received_only else el*m_tiles
        return GroupedGemm(spec.tile_rule, tile, tokens, k, counts, padded, local_experts, blocks, launched - blocks,
                           0, 0, blocks*ceildiv(n, bn), launched*ceildiv(n, bn), 2*rows*n*reduction,
                           2*blocks*bm*ceildiv(n, bn)*bn*ceildiv(reduction, bk)*bk, 0, 0, reduction, n)
    def weights(tensor, size):
        return [_a(tensor, local_active*size, True, off=(x-start_expert)*size) for x in local_experts]
    def batched(name, reads, writes, d, lines):
        kernel = _k("MLP", name, "grouped_gemm", reads, writes)
        return MoeKernel(**kernel.__dict__, grouped=d, details=KernelDetails(
            source="vllm 0.10.2 fused_moe/fused_batched_moe.py:234-354,356-440," + lines))
    wb = spec.weight_bytes
    w13 = batched("moe.ll_w13", [_a(p+".ll_x", 2*rows*h)] + weights(p+".w13", 2*wb*h*i), [_a(p+".ll_cache1", 4*rows*i)],
                  desc(2*i, h), "1001-1015")
    return w13, [
        ("moe.ll_cache2_zero", "pointwise", [], [_a(p+".ll_cache2", 2*span*i)], 0,
         "fused_batched_moe.py:1017 intermediate_cache2.fill_(0)" +
         (" (shortcut: received rows only)" if received_only else "")),
        ("moe.ll_silu_and_mul", "pointwise", [_a(p+".ll_cache1", 4*span*i)], [_a(p+".ll_cache2", 2*span*i)],
         SILU_AND_MUL_FLOPS*span*i, "fused_batched_moe.py:1020-1021 " +
         ("(shortcut: received rows only)" if received_only else "activation over the whole workspace")),
    ], batched("moe.ll_w2", [_a(p+".ll_cache2", 2*rows*i)] + weights(p+".w2", wb*h*i), [_a(p+".ll_out", 2*rows*h)],
               desc(h, i), "1028-1042")

def _deepep_ht_receive(spec, add, p, input_tensor, local_tokens, tokens, start_expert, local_experts, expert_rank,
                       sent, ids_tensor):
    from program.moe_rank import deepep_ht_received_tokens
    h, e, k = spec.hidden, spec.experts, spec.top_k
    if spec.routing_policy != "multinomial_expected":
        raise NotImplementedError("deepep_high_throughput is described for moe_routing_policy=multinomial_expected")
    if spec.tile_rule != "vllm_default_config":
        raise NotImplementedError("deepep_high_throughput is described with moe_tile_rule=vllm_default_config")
    received = max(1, _round_half_up(deepep_ht_received_tokens(tokens, e, k, spec.ep, expert_rank)))
    add("moe.a2a_dispatch", "collective",
        [_a(input_tensor, 2*local_tokens*h), _a(ids_tensor, 8*local_tokens*k), _a(p+".weights", 4*local_tokens*k)],
        [_a(p+".ht_x", 2*received*h), _a(p+".ht_ids_raw", 8*received*k), _a(p+".ht_weights", 4*received*k)],
        details=KernelDetails(source="vllm 0.10.2 fused_moe/deepep_ht_prepare_finalize.py:61-117",
                              notes=(f"deepep dispatch bytes sent by this rank {sent!r}",)))
    add("moe.ht_ids_where", "pointwise", [_a(p+".ht_ids_raw", 8*received*k)], [_a(p+".ht_ids", 8*received*k)],
        received*k, details=KernelDetails(source="vllm 0.10.2 fused_moe/deepep_ht_prepare_finalize.py:149-153"))
    return p+".ht_x", received

def _grouped_kernel(name, reads, writes, descriptor):
    k = _k("MLP", name, "grouped_gemm", reads, writes)
    return MoeKernel(**k.__dict__, grouped=descriptor, details=KernelDetails(
        source="vllm 0.10.2 fused_moe.py:350-489,531-543,1646-1708"))

_GROUP = object()

def moe_layer_trace(spec: MoeSpec, tokens: int, layer: int, input_tensor: str,
                    output_tensor: str, *, rank: int = 0, token_counts=None, step: int = 0, cache=None):
    tokens = _integer(tokens, "tokens")
    layer = _integer(layer, "layer", minimum=0)
    step = _integer(step, "step", minimum=0)
    h, i, e, k = spec.hidden, spec.intermediate, spec.experts, spec.top_k

    flat = spec.ep == 1 and spec.attention_dp > 1
    if not 0 <= rank < (spec.tp * spec.attention_dp if flat else spec.ep):
        raise ValueError("rank outside expert parallel world")
    dp_rank = rank // spec.tp if spec.ep > 1 or flat else 0
    counts_by_rank = tuple(token_counts) if token_counts is not None else (tokens,) * spec.attention_dp
    if len(counts_by_rank) != spec.attention_dp or any(not isinstance(n, Integral) or n <= 0 for n in counts_by_rank) or counts_by_rank[dp_rank] != tokens:
        raise ValueError("token_counts must contain positive local token counts for every rank")
    local_tokens = tokens
    tokens = sum(counts_by_rank) if spec.attention_dp > 1 else tokens
    q, rem = divmod(e, spec.ep)
    expert_rank = rank if spec.ep > 1 else 0
    start_expert = expert_rank*q + min(expert_rank, rem)
    local_experts = tuple(range(start_expert, start_expert + q + (rank < rem)))

    args = (spec, layer, step, input_tensor, output_tensor, local_tokens, tokens, counts_by_rank, dp_rank,
            start_expert, local_experts, expert_rank)
    if cache is None:
        return _moe_layer_body(*args)
    try:
        hit = cache.get(args)
    except TypeError:
        return _moe_layer_body(*args)

    group = args[:1] + args[2:3] + args[5:]
    if cache.get(_GROUP) != group:
        cache.clear()
        cache[_GROUP] = group
    if hit is None:
        hit = cache[args] = tuple(_moe_layer_body(*args))
    return list(hit)

def _moe_layer_body(spec, layer, step, input_tensor, output_tensor, local_tokens, tokens, counts_by_rank, dp_rank,
                    start_expert, local_experts, expert_rank):
    h, i, e, k = spec.hidden, spec.intermediate, spec.experts, spec.top_k
    local_output = output_tensor
    p = f"L{layer}.moe"
    ks = []
    def add(name, kind, reads, writes, flops=0, gemm=None, details=KernelDetails()):
        kernel = replace(_k("MLP", name, kind, reads, writes, gemm=gemm), flops=flops)
        ks.append(MoeKernel(**kernel.__dict__, details=details))
    add("moe.router", "gemm", [_a(input_tensor, 2*local_tokens*h), _a(p+".router_weight", 2*e*h, True)],
        [_a(p+".logits", 2*local_tokens*e)], gemm=(local_tokens,h,e),
        details=KernelDetails(source="vllm 0.10.2 models/qwen3_moe.py:153-158,180 ReplicatedLinear gate (bias=False) "
                              "-> layers/utils.py F.linear: one cuBLAS GEMM [T, h] x [h, E]"))
    shared_out = None
    if spec.family == "deepseek" and spec.shared_experts:
        si = spec.shared_intermediate
        add("moe.shared_gate_up", "gemm", [_a(input_tensor, 2*local_tokens*h), _a(p+".shared_wgu", spec.weight_bytes*h*2*si, True)],
            [_a(p+".shared_gu", 2*local_tokens*2*si)], gemm=(local_tokens, h, 2*si),
            details=KernelDetails(source="vllm 0.10.2 deepseek_v2.py:88-93,106-110"))
        add("moe.shared_silu_and_mul", "pointwise", [_a(p+".shared_gu", 2*local_tokens*2*si)],
            [_a(p+".shared_m", 2*local_tokens*si)], SILU_AND_MUL_FLOPS*local_tokens*si,
            details=KernelDetails(source="vllm 0.10.2 deepseek_v2.py:104,108 SiluAndMul"))
        shared_out = p+".shared_out"
        add("moe.shared_down", "gemm", [_a(p+".shared_m", 2*local_tokens*si), _a(p+".shared_wd", spec.weight_bytes*si*h, True)],
            [_a(shared_out, 2*local_tokens*h)], gemm=(local_tokens, si, h),
            details=KernelDetails(source="vllm 0.10.2 deepseek_v2.py:94-100 (reduce_results=False)"))
    deepep = spec.backend != "naive" and spec.attention_dp > 1

    ll = deepep and spec.backend in ("deepep_low_latency", "deepep_low_latency_received")

    ht = deepep and spec.backend == "deepep_high_throughput"
    if deepep:
        from program.moe_rank import deepep_rank_bytes
        sent, returned = deepep_rank_bytes(spec.backend, local_tokens, h, e, k, spec.ep, expert_rank)
    if spec.attention_dp > 1 and not deepep:
        offset = sum(counts_by_rank[:dp_rank])
        for label, src, width in (("hidden", input_tensor, h), ("logits", p+".logits", e)):
            dst = p+".gathered_"+label
            add("moe.dispatch_copy_"+label, "memcpy", [_a(src, 2*local_tokens*width)], [_a(dst, 2*local_tokens*width, off=2*offset*width)])
            off = 0
            for source, n in enumerate(counts_by_rank):
                add(f"moe.broadcast_{label}_{source}", "collective", [_a(dst, 2*n*width, off=2*off*width)], [_a(dst, 2*n*width, off=2*off*width)])
                off += n
        input_tensor = p+".gathered_hidden"
        output_tensor = input_tensor
    logits = p+".gathered_logits" if spec.attention_dp > 1 and not deepep else p+".logits"
    route = local_tokens if deepep else tokens
    id_bytes = 8 if deepep else 4
    if spec.family == "deepseek":
        _deepseek_router(spec, add, p, logits, route)
        if deepep:
            add("moe.ids_int64", "pointwise", [_a(p+".expert_ids", 4*route*k)], [_a(p+".expert_ids64", 8*route*k)],
                details=KernelDetails(source="vllm 0.10.2 fused_moe/layer.py:1498-1499 topk_ids.to(indices_type)"))
    else:
        add("moe.logits_cast", "pointwise", [_a(logits, 2*route*e)], [_a(p+".logits32", 4*route*e)],
            details=KernelDetails(source="vllm 0.10.2 fused_moe.py:883-939", arithmetic_ops=(("bf16_to_fp32_conversions",route*e),)))
        add("moe.topk_softmax", "pointwise", [_a(p+".logits32", 4*route*e)],
            [_a(p+".weights_raw", 4*route*k), _a(p+".expert_ids", id_bytes*route*k), _a(p+".token_expert_indices", 4*route*k)],
            route*(3*e-1), details=KernelDetails(
                source="vllm 0.10.2 csrc/moe/topk_softmax_kernels.cu:275-315,318-390,395-423,449-487,531-547",
                arithmetic_ops=(("softmax_scalar_ops",route*(3*e-1)), ("exponentials",route*e), ("reciprocals",route)),
                notes=("top-k comparisons and warp-reduction instructions are uncounted", "token_expert_indices output is unused by fused experts")))
        add("moe.renorm_sum", "reduce", [_a(p+".weights_raw", 4*route*k)], [_a(p+".weight_sums", 4*route)], route*(k-1),
            details=KernelDetails(source="vllm 0.10.2 fused_moe.py:894 topk_weights.sum(dim=-1, keepdim=True) "
                                  "(renormalize = config.norm_topk_prob, qwen3_moe.py:147)"))
        add("moe.renorm_divide", "pointwise", [_a(p+".weights_raw", 4*route*k), _a(p+".weight_sums", 4*route)],
            [_a(p+".weights", 4*route*k)], route*k,
            details=KernelDetails(source="vllm 0.10.2 fused_moe.py:894 topk_weights / sum (new fp32 tensor)"))
    if ll:
        w13, pointwise, w2 = _deepep_ll_experts(spec, add, p, local_tokens, tokens, layer, step, start_expert,
                                                local_experts, expert_rank, sent, input_tensor,
                                                p+(".expert_ids64" if spec.family == "deepseek" else ".expert_ids"))
        ks.append(w13)
        for name, kind, reads, writes, flops, source in pointwise:
            add(name, kind, reads, writes, flops, details=KernelDetails(source="vllm 0.10.2 fused_moe/" + source))
        ks.append(w2)
    else:
        ids_name, ids_bytes, weights_name, global_tokens = p+".expert_ids", 4, p+".weights", tokens
        if ht:
            input_tensor, tokens = _deepep_ht_receive(spec, add, p, input_tensor, local_tokens, tokens, start_expert,
                                                      local_experts, expert_rank, sent,
                                                      p+(".expert_ids64" if spec.family == "deepseek" else ".expert_ids"))
            output_tensor = input_tensor
            ids_name, ids_bytes, weights_name = p+".ht_ids", 8, p+".ht_weights"
        assignments = cycle = None
        all_local = start_expert == 0 and len(local_experts) == e
        if spec.routing_policy == "uniform_occupancy":
            cycle = uniform_occupancy_cycle(tokens, e, k, layer + step, spec.ep)
        elif spec.routing_policy == "one_hot_distinct" and spec.expert_imbalance_factor != 1:
            assignments = routed_experts(tokens, e, k, layer + step, spec.ep,
                                         spec.routing_policy, spec.expert_imbalance_factor)
        for start in range(0, tokens, spec.chunk_size):
            m = min(spec.chunk_size, tokens-start)
            r = m*k
            offset = (start*k + layer + step) % e
            chunk_assignments = assignments[start*k:(start+m)*k] if assignments is not None else None
            if cycle is not None:
                counts = _cycle_counts(cycle, start*k, r, e)
            elif chunk_assignments is None:
                counts = balanced_counts(m, e, k, offset, spec.ep)
            else:
                histogram = Counter(chunk_assignments)
                counts = tuple(histogram[x] for x in range(e))
            if spec.tile_rule == "vllm_default_config":
                tile = vllm_default_config(m, len(local_experts))
            else:
                tile = spec.rapid_tile_selector(counts=counts, hidden_size=h, intermediate_size=i)
                if not isinstance(tile, Tile):
                    tile = Tile(*tile)
            for name in ("block_m", "block_n", "block_k", "group_m"):
                _integer(getattr(tile, name), name)
            bm,bn,bk = tile.block_m,tile.block_n,tile.block_k
            expected = None
            if spec.routing_policy == "multinomial_expected":
                routed = _round_half_up(global_tokens * m / tokens) if ht else m
                tiles, active = multinomial_expected_tiles(routed, e, k, bm)
                local_tiles, local_active = tiles, active
                if spec.ep > 1:
                    local_active = multinomial_rank_active(routed, e, k, spec.ep, expert_rank, layer + step)
                    local_tiles = tiles * local_active / active if active else 0.0
                expected = (local_tiles, local_active)
                local_set = set(local_experts)
                padded = tuple((local_tiles if x in local_set else tiles)*bm for x in range(e))
                if ht:
                    rows_c = routed * k * len(local_experts) / e
                    other = r - rows_c
                    sentinel = e - 1 if start_expert == 0 else 0
                    counts = tuple(rows_c / len(local_experts) if x in local_set else (other if x == sentinel else 0.0)
                                   for x in range(e))
                    padded = tuple(local_tiles*bm if x in local_set else
                                   (math.ceil(other/bm)*bm if x == sentinel else 0.0) for x in range(e))
            else:
                padded = tuple(ceildiv(n,bm)*bm for n in counts)
            q = sum(padded)
            a = r + e*(bm-1)
            j = ceildiv(a,bm)
            chunk = f"{p}.chunk{start}"
            ids = _a(ids_name, ids_bytes*r, off=ids_bytes*start*k)
            if r < 1024 and e <= 64:
                add("moe.align_small", "pointwise", [ids, ids],
                    [_a(chunk+".sorted_ids", 4*a), _a(chunk+".sorted_ids", 4*r),
                     _a(chunk+".block_experts", 4*j), _a(chunk+".padded_count", 4)],
                    details=KernelDetails(source="vllm 0.10.2 csrc/moe/moe_align_sum_kernels.cu:126-202,228-245", notes=("integer histogram, prefix, padding and sorting work uncounted",)))
            else:
                add("moe.align_count_prefix_init", "pointwise", [ids],
                    [_a(chunk+".sorted_ids", 4*a), _a(chunk+".block_experts", 4*j), _a(chunk+".padded_count", 4), _a(chunk+".scratch", 4*(e+1))],
                    details=KernelDetails(source="vllm 0.10.2 csrc/moe/moe_align_sum_kernels.cu:18-93,246-259", notes=("shared-memory integer histogram, atomics, prefix and padding work uncounted",)))
                add("moe.align_sort", "pointwise", [ids, _a(chunk+".scratch", 4*r)],
                    [_a(chunk+".scratch", 4*r), _a(chunk+".sorted_ids", 4*r)],
                    details=KernelDetails(source="vllm 0.10.2 csrc/moe/moe_align_sum_kernels.cu:95-107,261-273", arithmetic_ops=(("global_int32_atomic_increments",r),)))
            qb = q/bm if expected is not None else ceildiv(q,bm)
            meta = [_a(chunk+".sorted_ids", 4*q), _a(chunk+".block_experts", 4*qb), _a(chunk+".padded_count", 4)]
            if spec.ep > 1:
                add("moe.expert_remap", "pointwise", [_a(chunk+".block_experts", 4*j), _a(p+".expert_map", 4*e, True)], [_a(chunk+".local_block_experts", 4*j)])
                meta[1] = _a(chunk+".local_block_experts", 4*qb)
            local_rows = sum(counts[x] for x in local_experts)
            local_padded = sum(padded[x] for x in local_experts)
            if ht:
                token_ranges, slot_ranges = ((0, m),), ((0, local_rows),)
            elif cycle is not None and all_local:
                token_ranges, slot_ranges = ((0, m),), ((0, r),)
            elif cycle is not None:
                slots = _cycle_local_slots(cycle, start*k, r, start_expert, start_expert + len(local_experts))
                token_ranges = _runs(dict.fromkeys(slot // k for slot in slots))
                slot_ranges = _runs(slots)
            elif chunk_assignments is None:
                token_ranges, slot_ranges = _local_routing_ranges(
                    m, e, k, offset, start_expert, start_expert + len(local_experts), spec.ep)
            else:
                slots = tuple(slot for slot, expert in enumerate(chunk_assignments)
                              if start_expert <= expert < start_expert + len(local_experts))
                token_ranges = _runs(dict.fromkeys(slot // k for slot in slots))
                slot_ranges = _runs(slots)
            def accesses(tensor, ranges, width, base=0, weight=False):
                return [_a(tensor, (hi-lo)*width, weight, off=base+lo*width) for lo, hi in ranges]
            active_ranges = []
            for expert in local_experts:
                if expected is not None:
                    break
                if counts[expert]:
                    local_id = expert - start_expert
                    if active_ranges and active_ranges[-1][1] == local_id:
                        active_ranges[-1] = (active_ranges[-1][0], local_id + 1)
                    else:
                        active_ranges.append((local_id, local_id + 1))
            def desc(n, reduction, call_rows, call_topk):
                em = min(a, call_rows*call_topk*bm) if call_rows < bm else a
                if expected is not None:
                    return GroupedGemm(spec.tile_rule,tile,m,k,counts,padded,local_experts,local_padded/bm,(q-local_padded)/bm,a,j,
                                       (q/bm)*ceildiv(n,bn),ceildiv(em,bm)*ceildiv(n,bn),
                                       2*local_rows*n*reduction,2*local_padded*ceildiv(n,bn)*bn*ceildiv(reduction,bk)*bk,
                                       offset,start,reduction,n)
                return GroupedGemm(spec.tile_rule,tile,m,k,counts,padded,local_experts,local_padded//bm,(q-local_padded)//bm,a,j,
                                   (q//bm)*ceildiv(n,bn),ceildiv(em,bm)*ceildiv(n,bn),
                                   2*local_rows*n*reduction,2*local_padded*ceildiv(n,bn)*bn*ceildiv(reduction,bk)*bk,
                                   offset,start,reduction,n)
            def weights(tensor, size):
                if expected is None:
                    return accesses(tensor, active_ranges, size, weight=True)

                return [_a(tensor, expected[1]*size, True, off=(x-start_expert)*size) for x in local_experts]
            ks.append(_grouped_kernel("moe.w13", accesses(input_tensor, token_ranges, 2*h, 2*start*h)
                      + weights(p+".w13", 2*spec.weight_bytes*h*i) + meta,
                      [_a(chunk+".cache1", 4*r*i)], desc(2*i,h,m,k)))
            add("moe.silu_and_mul", "pointwise", [_a(chunk+".cache1", 4*r*i)], [_a(chunk+".cache2", 2*r*i)], SILU_AND_MUL_FLOPS*r*i,
                details=KernelDetails(source="vllm 0.10.2 fused_moe.py:1670-1671 torch.ops._C.silu_and_mul(cache2 [M k, N/2], "
                                      "cache1.view(-1, N)); csrc activation_kernels.cu act_and_mul_kernel"))
            ks.append(_grouped_kernel("moe.w2", accesses(chunk+".cache2", slot_ranges, 2*i)
                      + weights(p+".w2", spec.weight_bytes*h*i)
                      + accesses(weights_name, slot_ranges, 4, 4*start*k) + meta,
                      [_a(chunk+".cache3", 2*r*h)], desc(h,i,r,1)))

            ks[-1] = replace(ks[-1], flops=local_rows*h)
            add("moe.sum", "reduce", [_a(chunk+".cache3", 2*r*h)], [_a(output_tensor, 2*m*h, off=2*start*h)], m*h*(k-1),
                details=KernelDetails(source="vllm 0.10.2 fused_moe.py:1578-1603,1709-1719; csrc/moe/moe_align_sum_kernels.cu:277-317",
                                      aliases=((output_tensor,input_tensor), (chunk+".cache3",chunk+".cache1")),
                                      notes=(("implementation: moe_sum_kernel" if k in (2,3,4) else "implementation: ATen sum_out"),
                                             "aliases use l2_sim source-specific allocation and lifetime rules")))
    if deepep:
        add("moe.a2a_combine", "collective",
            [_a(p+".ll_out", 2*h*tokens*k*len(local_experts)/e)] if ll else [_a(output_tensor, 2*tokens*h)],
            [_a(p+".all_reduce_output", 2*local_tokens*h)],
            details=KernelDetails(source="vllm 0.10.2 fused_moe/" + ("deepep_ll_prepare_finalize.py:201-229"
                                  if ll else "deepep_ht_prepare_finalize.py:246-283"),
                                  aliases=((local_output, p+".all_reduce_output"),),
                                  notes=(f"deepep combine bytes sent by this rank {returned!r}",)))
    elif spec.attention_dp > 1:
        add("moe.dp_all_reduce", "collective", [_a(output_tensor, 2*tokens*h)], [_a(p+".all_reduce_output", 2*tokens*h)])

        ks[-1] = replace(ks[-1], details=KernelDetails(aliases=((local_output, p+".all_reduce_output"),), notes=(f"local view byte offset {2*sum(counts_by_rank[:dp_rank])*h}",)))
    if spec.family == "deepseek":
        add("moe.routed_scale", "pointwise", [_a(local_output, 2*local_tokens*h)], [_a(local_output, 2*local_tokens*h)],
            local_tokens*h, details=KernelDetails(source="vllm 0.10.2 deepseek_v2.py:302-303"))
        if shared_out is not None:
            add("moe.shared_add", "pointwise", [_a(local_output, 2*local_tokens*h), _a(shared_out, 2*local_tokens*h)],
                [_a(local_output, 2*local_tokens*h)], local_tokens*h,
                details=KernelDetails(source="vllm 0.10.2 deepseek_v2.py:308-310"))

    return [replace(kernel, spec=spec) for kernel in ks]
