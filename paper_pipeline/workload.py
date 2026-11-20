from __future__ import annotations

import copy
import json
from pathlib import Path

PHASES = ("prefill", "decode", "train")

def decode_points(rule: str, seq: int, row: dict):
    if rule == "fixed_seq":
        return [("", int(seq), 1.0)]
    if rule == "segment_simpson":
        def col(name, default):
            v = row.get(name)
            return int(float(v)) if v not in (None, "") else int(default)
        first, mid, last = col("ctx_keys_first", seq + 1), col("ctx_keys_mean", seq + 128), col("ctx_keys_last", seq + 255)
        if not first < mid < last or (mid - first) != (last - mid):
            raise ValueError(f"Simpson points must be equally spaced: {first}, {mid}, {last}")
        return [(f"k{first}", first - 1, 1 / 6), (f"k{mid}", mid - 1, 4 / 6), (f"k{last}", last - 1, 1 / 6)]
    raise ValueError(f"unknown decode_context rule {rule!r}")

def kv_tokens_on_entry(rule: str, seq: int, row: dict) -> int:
    if rule == "fixed_seq":
        return int(seq)
    if rule == "segment_mean":
        mean_keys = float(row["ctx_keys_mean"]) if row.get("ctx_keys_mean") not in (None, "") else seq + 128.0
        if mean_keys != int(mean_keys):
            raise ValueError(f"segment mean context {mean_keys} is not an integer key count")
        return int(mean_keys) - 1
    raise ValueError(f"unknown decode_context rule {rule!r}")

QWEN3_MOE_REQUIRED = {"decoder_sparse_step": 1, "mlp_only_layers": [], "hidden_act": "silu", "norm_topk_prob": True,
                      "torch_dtype": "bfloat16", "attention_bias": False, "use_sliding_window": False}

def qwen3_moe_block(cfg: dict, moe_rules: dict) -> dict:
    for key, value in QWEN3_MOE_REQUIRED.items():
        if cfg.get(key) != value:
            raise ValueError(f"qwen3_moe published config field {key}={cfg.get(key)!r} is not described (needs {value!r})")
    if cfg.get("shared_expert_intermediate_size") or cfg.get("n_shared_experts"):
        raise ValueError("qwen3_moe with shared experts is not described")
    missing = [k for k in ("moe_tile_rule", "routing_policy", "grouped_traffic", "collective_rule", "replay_steps")
               if moe_rules.get(k) is None]
    if missing:
        raise ValueError(f"qwen3_moe counts require the stated MoE rules {missing} (registry workload.rules.moe)")
    moe = {"num_experts": int(cfg["num_experts"]), "top_k": int(cfg["num_experts_per_tok"]),
           "moe_intermediate_size": int(cfg["moe_intermediate_size"]), "n_shared_experts": 0,
           "moe_layer_freq": 1, "first_k_dense_replace": 0, "routing_policy": moe_rules["routing_policy"],
           "expert_imbalance_factor": 1.0, "norm_topk_prob": True}
    fields = {"moe_tile_rule": moe_rules["moe_tile_rule"], "moe_source_collective_rule": moe_rules["collective_rule"],
              "moe_source_replay_steps": int(moe_rules["replay_steps"]),
              "moe_grouped_traffic": moe_rules["grouped_traffic"]}
    return moe, fields

def model_yaml(cfg: dict, phase: str, batch: int, seq: int, kv_on_entry: int | None, implementation: dict,
               family: str = "llama", *, published_cfg: dict | None = None,
               architecture: dict | None = None, qwen2_bias_mode: str | None = None,
               moe_rules: dict | None = None) -> dict:
    if phase not in PHASES:
        raise ValueError(phase)
    if family not in {"llama", "qwen2", "qwen3", "qwen3_moe"}:
        raise ValueError(f"model family {family!r} is not described by the simulator; the run must stop, not be relabelled")
    if family == "qwen3_moe":
        return _qwen3_moe_model_yaml(cfg, phase, batch, seq, kv_on_entry, implementation,
                                     published_cfg=published_cfg, architecture=architecture, moe_rules=moe_rules)
    if family in {"qwen2", "qwen3"}:
        if published_cfg is None or architecture is None:
            raise ValueError(f"{family} requires its published config and architecture record")
        for key in cfg.keys() & published_cfg.keys():
            if cfg[key] != published_cfg[key]:
                raise ValueError(f"short config disagrees with published config: {key}: {cfg[key]!r} != {published_cfg[key]!r}")
        cfg = published_cfg
        biases = {k: family == "qwen2" and k != "o" for k in ("q", "k", "v", "o")}
        if (cfg["model_type"] != family or architecture["family"] != family
                or architecture["projection_bias"] != biases or architecture["qk_norm"] != (family == "qwen3")
                or architecture.get("moe")):
            raise ValueError(f"{family} published config / architecture variant is not described")
        if family == "qwen2" and qwen2_bias_mode not in {"cublaslt_epilogue", "addmm_copy_gemm"}:
            raise ValueError("qwen2 requires the explicit paper model.qwen2_bias_mode setting")
    h, H, K = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    mp = {
        "mode": "LLM",
        "run_type": "training" if phase == "train" else "inference",
        "tied_embeddings": bool(cfg["tie_word_embeddings"]),
        "model_type": family,
        "global_batch_size": int(batch),
        "gradient_accumulation_steps": 1,
        "hidden_dim": h,
        "attention": {"attention_type": "gqa" if K < H else "mha", "use_rope": True, "num_heads": H, "kv_heads": K,
                      "use_flashattention": True, "attention_tile_size": 128},
        "moe": {"num_experts": 1, "top_k": 1, "moe_intermediate_size": cfg["intermediate_size"], "n_shared_experts": 0,
                "moe_layer_freq": 1, "first_k_dense_replace": cfg["num_hidden_layers"]},
        "intermediate_size": cfg["intermediate_size"],
        "vocab_size": cfg["vocab_size"],
        "num_layers": cfg["num_hidden_layers"],
    }
    if phase == "train":
        mp["seq_len"] = int(seq)
    elif phase == "prefill":
        mp["seq_len"], mp["decode_len"] = int(seq) + 1, 1
    else:
        mp["seq_len"], mp["decode_len"] = int(kv_on_entry) + 2, 2
    mp["implementation"] = copy.deepcopy(implementation)
    order = ["mode", "run_type", "tied_embeddings", "model_type", "global_batch_size", "gradient_accumulation_steps",
             "seq_len", "decode_len", "hidden_dim", "attention", "moe", "intermediate_size", "vocab_size", "num_layers",
             "implementation"]
    result = {"model_param": {k: mp[k] for k in order if k in mp}}
    if family in {"qwen2", "qwen3"}:
        result["model_param"].update(max_position_embeddings=cfg["max_position_embeddings"],
                                     use_sliding_window=cfg["use_sliding_window"])
        if family == "qwen2":
            result["model_param"]["qwen2_bias_mode"] = qwen2_bias_mode
        else:
            result["model_param"]["attention"]["head_dim"] = cfg["head_dim"]
            result["model_param"]["attention_bias"] = cfg["attention_bias"]
    return result

def _qwen3_moe_model_yaml(cfg, phase, batch, seq, kv_on_entry, implementation, *, published_cfg, architecture,
                          moe_rules):
    if phase == "train":
        raise ValueError("qwen3_moe training is not described on the vLLM source MoE path")
    if published_cfg is None or architecture is None:
        raise ValueError("qwen3_moe requires its published config and architecture record")
    for key in cfg.keys() & published_cfg.keys():
        if cfg[key] != published_cfg[key]:
            raise ValueError(f"short config disagrees with published config: {key}: {cfg[key]!r} != {published_cfg[key]!r}")
    cfg = published_cfg
    if (cfg["model_type"] != "qwen3_moe" or architecture["family"] != "qwen3_moe" or not architecture.get("moe")
            or any(architecture["projection_bias"].values()) or not architecture["qk_norm"]):
        raise ValueError("qwen3_moe published config / architecture variant is not described")
    moe, fields = qwen3_moe_block(cfg, moe_rules or {})
    h, H, K = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    mp = {
        "mode": "LLM", "run_type": "inference", "tied_embeddings": bool(cfg["tie_word_embeddings"]),
        "model_type": "qwen3_moe", "global_batch_size": int(batch), "gradient_accumulation_steps": 1,
        "hidden_dim": h,
        "attention": {"attention_type": "gqa" if K < H else "mha", "use_rope": True, "num_heads": H, "kv_heads": K,
                      "use_flashattention": True, "attention_tile_size": 128, "head_dim": cfg["head_dim"]},
        "moe": moe,
        "intermediate_size": cfg["intermediate_size"],
        "vocab_size": cfg["vocab_size"],
        "num_layers": cfg["num_hidden_layers"],
    }
    if phase == "prefill":
        mp["seq_len"], mp["decode_len"] = int(seq) + 1, 1
    else:
        mp["seq_len"], mp["decode_len"] = int(kv_on_entry) + 2, 2
    mp["implementation"] = copy.deepcopy(implementation)
    order = ["mode", "run_type", "tied_embeddings", "model_type", "global_batch_size", "gradient_accumulation_steps",
             "seq_len", "decode_len", "hidden_dim", "attention", "moe", "intermediate_size", "vocab_size", "num_layers",
             "implementation"]
    result = {"model_param": {k: mp[k] for k in order}}
    result["model_param"].update(max_position_embeddings=cfg["max_position_embeddings"],
                                 use_sliding_window=cfg["use_sliding_window"], attention_bias=cfg["attention_bias"],
                                 **fields)
    return result

def published_inputs(architecture: dict) -> dict:
    return {"published_cfg": json.loads((Path(__file__).resolve().parents[1] / architecture["hf_config"]).read_text()),
            "architecture": architecture}

def load_hf_config(model_configs: Path, model: str, *, architecture: dict | None = None) -> dict:
    path = Path(model_configs) / f"{model}.json"
    if not path.exists() and architecture and architecture["family"] in {"qwen2", "qwen3", "qwen3_moe"}:
        return published_inputs(architecture)["published_cfg"]
    return json.loads(path.read_text())
