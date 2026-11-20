from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, Mapping, Optional

from opmodel import settings as S

NCCL_LATENCY_RULES = ("astrasim_per_hop", "tuning_formula", "tuning_protocol")
MEMORY_TIME_RULES = ("cold_kernel",)
L2_STEADY_STATES = ("replay3", "periodic_prime")
L2_ADDRESS_PLANS = ("warmup_replay", "onepass")

def _l2_address_plan(raw):
    if raw not in L2_ADDRESS_PLANS:
        raise ValueError(f"address_plan must be one of {L2_ADDRESS_PLANS}, got {raw!r}")
    return raw

def _l2_steady_state(raw):
    if raw not in L2_STEADY_STATES:
        raise ValueError(f"steady_state must be one of {L2_STEADY_STATES}, got {raw!r}")
    return raw

L2_PARTITIONS_RULES = ("a100_literal", "full_capacity")

def _l2_partitions_rule(raw):
    if raw not in L2_PARTITIONS_RULES:
        raise ValueError(f"l2_model.partitions_rule must be one of {L2_PARTITIONS_RULES}, got {raw!r}")
    return raw

def _memory_time_rule(raw):
    if raw == "step_simulated":
        raise ValueError("l2_model.memory_time_rule='step_simulated' is OBSOLETE "
                         "; use 'cold_kernel'")
    if raw not in MEMORY_TIME_RULES:
        raise ValueError(f"memory_time_rule must be one of {MEMORY_TIME_RULES}, got {raw!r}")
    return raw

def parse_nccl_latency_rule(raw):
    if raw not in NCCL_LATENCY_RULES:
        raise ValueError(f"sw_param.nccl_latency_rule must be one of {NCCL_LATENCY_RULES}, got {raw!r}")
    return raw

S.register("RAPID_NCCL_LATENCY_RULE", "astrasim_per_hop", parse_nccl_latency_rule,
           "hw:sw_param.nccl_latency_rule",
           "NCCL v2.27.7 src/graph/tuning.cc:134-149,250-252,329-355 ncclTopoTuneModel; "
           "tuning_formula: single-node Ring/Simple only: baseLat + nsteps * hwLat; configured launch overhead and dimension latency. "
           "LL/LL128 protocol selection is NOT modelled; Simple only (8 MiB n=8: Simple 56 us versus LL128 40.6 us). "
           "NCCL collectives run as ring under tuning_formula (Ring/Simple constants). "
           "astrasim_per_hop preserves the legacy model; tuning_formula rejects mixed native traffic. "
           "tuning_protocol selects LL/LL128/Simple for homogeneous single-node A100 NVLink, "
           "requiring topology.links_per_gpu. Source: program.nccl_latency.PROTOCOL_SOURCE; "
           "tuning.cc:135,145 base/hw latency; :167,174 Ampere caps; :250-280 steps/bus/protocol caps; "
           ":320-355 algBw/latency; :457-473 LL128 enable; :561-587 time; topo.h:18 20 GB/s per SM80 link. "
           "Equivalent graph nChannels=links, bwIntra=20 GB/s, not a captured NCCL graph.")

def l2_bypass(hw) -> bool:
    l2 = getattr(getattr(hw, "tech_config", None), "SRAML2", None)
    return bool(getattr(l2, "l2_bypass", False))

def nccl_latency_rule(hw):
    if S.origin("RAPID_NCCL_LATENCY_RULE") in ("env", "override"):
        return S.get("RAPID_NCCL_LATENCY_RULE")
    return getattr(getattr(hw, "sw_config", None), "nccl_latency_rule", "astrasim_per_hop")

MODELLING_DEFAULTS_ON = frozenset({
    "RAPID_KV_CACHE_WRITE",
    "RAPID_PREFILL_LOGITS_LAST_TOKEN",
    "RAPID_INFER_VOCAB_PARALLEL_HEAD",
    "RAPID_EMBEDDING_GATHER",
    "RAPID_VLLM_POINTWISE",
    "RAPID_DECODE_BATCHED_ATTENTION",
    "RAPID_MOE_ACTIVE_EXPERTS",
    "RAPID_MOE_GROUPED_GEMM",
    "RAPID_MOE_PERMUTE_KERNELS",
    "RAPID_OPTIMIZER_LAUNCH_TRACE",
    "RAPID_VLLM_ATTN_OUTPUT_ZEROS",
    "RAPID_VLLM_SELECTED_SAMPLER",
})

def env_flag_value(raw: str) -> bool:
    return raw.strip().lower() not in {"", "0", "false", "no"}

def _flash_rebuild(raw: str) -> bool:
    if not env_flag_value(raw):
        raise ValueError("RAPID_FLASH_REBUILD=0 is obsolete; use the flash attention rebuild")
    return True

def _on(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _allreduce_choice(raw, choices):
    if raw not in choices:
        raise ValueError(f"allreduce setting must be one of {choices}, got {raw!r}")
    return raw

ALLREDUCE_SETTINGS = {
    "allreduce_model": ("astrasim", lambda raw: _allreduce_choice(raw, ("astrasim", "measured_curve"))),
    "allreduce_timing_basis": ("kernel_union", lambda raw: _allreduce_choice(
        raw, ("kernel_union", "cuda_event_large_message_proxy"))),
    "allreduce_curve": ("paper_pipeline/inputs/network/a100_sxm4_1065_nccl.json", str),
    "allreduce_include_unverified": (False, env_flag_value),
}
for _key, (_default, _parser) in ALLREDUCE_SETTINGS.items():
    S.register("RAPID_" + _key.upper(), _default, _parser, "hw:sw_param." + _key,
               "designation NCCL_CURVE")

ALLREDUCE_RSAG_SOURCE = ("designation NCCL_CURVE_SCOPE.extension_rsag; NVIDIA/nccl-tests "
                         "doc/PERFORMANCE.md (afd59aba) bus bandwidth: all-reduce algBw x 2(n-1)/n, reduce-scatter "
                         "and all-gather algBw x (n-1)/n, S = recvcount (RS) / sendcount (AG) x typesize x n")
S.register("RAPID_ALLREDUCE_RSAG_RULE", "network_model",
           lambda raw: _allreduce_choice(raw, ("network_model", "allreduce_bus_bandwidth")),
           "hw:sw_param.allreduce_rsag_rule", ALLREDUCE_RSAG_SOURCE)

def _private_copies(raw: str) -> Fraction:
    copies = raw.strip()
    if copies not in ("1.5", "1.0", "1"):
        raise ValueError(f"RAPID_L2_PRIVATE_COPIES must be 1.5 or 1.0, got {copies!r}")
    return Fraction(copies)

def _optional_float(raw: str) -> Optional[float]:
    return float(raw) if raw else None

def _link_energy_rule(raw: str) -> str:
    if raw not in ("per_payload_byte", "line_rate_while_active"):
        raise ValueError("RAPID_LINK_ENERGY_RULE must be per_payload_byte or line_rate_while_active")
    return raw

def parse_kernel_launch_mode(raw):
    value = str(raw).strip().lower()
    if value not in ("stream", "graph_when_captured"):
        raise ValueError("kernel_launch_mode must be stream or graph_when_captured")
    return value

def parse_graph_launch_overhead(raw):
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("graph_launch_overhead_s must be finite and >= 0") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("graph_launch_overhead_s must be finite and >= 0")
    return value

S.register("RAPID_KERNEL_LAUNCH_MODE", "stream", parse_kernel_launch_mode, "hw:sw_param.kernel_launch_mode",
           "TaxBreak arXiv:2603.12465 measures eager stream launch; graph replay is opt-in (time only)")
S.register("RAPID_GRAPH_LAUNCH_OVERHEAD_S", None, parse_graph_launch_overhead, "hw:sw_param.graph_launch_overhead_s",
           "Per-kernel graph replay overhead; no measured value or default is supplied (time only)")
S.register("RAPID_SMALLGRID_TIMING", False, env_flag_value, "model:implementation.kernel_rules.smallgrid_timing",
           "Experimental source-derived small-grid latency estimate; timing only")
S.register("RAPID_SMALLGRID_DECODE_ONLY", False, env_flag_value, "model:implementation.kernel_rules.smallgrid_decode_only",
           "Limit the small-grid experiment to decode template construction; phase scope, not a hardware assumption")

_KERNEL_RULE_SOURCES = {
    "RAPID_KV_CACHE_WRITE": "vLLM reshape_and_cache kernel (changelog 18-30)",
    "RAPID_PREFILL_LOGITS_LAST_TOKEN": "vLLM logits_indices / HF logits_to_keep=1 (captures)",
    "RAPID_INFER_VOCAB_PARALLEL_HEAD": "vLLM/Megatron vocab-parallel head + all-gather",
    "RAPID_EMBEDDING_GATHER": "embedding forward as a row gather",
    "RAPID_VLLM_POINTWISE": "vLLM fused pointwise kernels (fused_add_rms_norm, silu_and_mul)",
    "RAPID_DECODE_BATCHED_ATTENTION": "one flash-decoding kernel per decode step",
    "RAPID_MOE_ACTIVE_EXPERTS": "MoE expected active experts, uniform routing",
    "RAPID_MOE_GROUPED_GEMM": "MoE experts as one grouped GEMM",
    "RAPID_MOE_PERMUTE_KERNELS": "MoE permute / unpermute kernels",
    "RAPID_INFER_DP_ATTENTION": "case-study switch: DP attention groups (off)",
    "RAPID_MOE_TP_REPLICATED_COMBINE": "case-study switch: EP over TP replicated combine (off)",
    "RAPID_ROPE_FUSED_KV_WRITE": "runtime rule: RoPE fused into the KV write (off)",

    "RAPID_FLASH_REBUILD": "FA-2 rebuild, default off (flash_attention_rebuild/DESIGN.md sections 4 and 8)",
    "RAPID_OPTIMIZER_LAUNCH_TRACE": "kernel_trace_train.py:742-797 optimizer_step / _mta_launches; torch MultiTensorApply.cuh and fused_adam_utils.cuh",
    "RAPID_VLLM_ATTN_OUTPUT_ZEROS": "kernel_trace_vllm.py:330-335 attn_out_zeros before cache write / flash",
    "RAPID_VLLM_SELECTED_SAMPLER": "kernel_trace_vllm.py:376-435 selected-row head and sampler sequence",
    "RAPID_FLASH_SPILLS": "S1b pinned flash-only SASS spill accounting, default off; A100 whitepaper p.33 shared L1/SMEM service",
}
for _name, _src in _KERNEL_RULE_SOURCES.items():
    S.register(_name, _name in MODELLING_DEFAULTS_ON or _name == "RAPID_FLASH_REBUILD",
               _flash_rebuild if _name == "RAPID_FLASH_REBUILD" else env_flag_value,
               "model:implementation.kernel_rules." + _name[len("RAPID_"):].lower(), _src)

S.register("RAPID_FLASH_FAST_FORWARD", True, env_flag_value, "model:implementation.kernel_rules.flash_fast_forward",
           "APPROXIMATE flash timeline: once every CTA on an SM has a steady window period, its steady tiles pass as "
           "one delay; first tiles, the last FF_TAIL tiles (causal diagonal) and the last wave are simulated")
S.register("RAPID_FLASH_FAST_FORWARD_RTOL", 1e-6, float, "model:implementation.kernel_rules.flash_fast_forward_rtol",
           "flash_fast_forward: a CTA is steady when its last N window periods agree within this relative tolerance")
S.register("RAPID_FLASH_FAST_FORWARD_PERIODS", 4, int, "model:implementation.kernel_rules.flash_fast_forward_periods",
           "flash_fast_forward: N, the number of consecutive window periods that must agree (window = CTAs per SM)")
S.register("RAPID_FLASH_FAST_FORWARD_FWD", True, env_flag_value, "model:implementation.kernel_rules.flash_fast_forward_fwd",
           "flash_fast_forward sub-switch: also fast-forward the ordinary forward kernel (a steady SM pair skips whole "
           "periods together; flash_native.c, 'approximate fast-forward, forward'); false keeps its exact timeline")
S.register("RAPID_MOE_GROUPED_BW", 0.0, S.float_or_zero, "model:implementation.kernel_rules.moe_grouped_bw",
           "bytes/s, 0 = off; grouped MoE kernel as one roofline (time only) [no public source]")
S.register("RAPID_FA_PREFILL_SCALE", 1.0, lambda raw: float(raw or 1), "model:implementation.kernel_rules.fa_prefill_scale",
           "flash-attention prefill time scale, sensitivity only (time only) [no public source]")
S.register("RAPID_FA_TILE_UTIL_FWD", None, _optional_float, "model:implementation.kernel_rules.fa_tile_util_fwd",
           "None = 0.25 (Dao 2023 FA-1 range): flash-attention tiles are timed by the native tile model with this "
           "SM utilisation, no padding FLOPs and no OPMODEL_* switch (base_timing flash path); exposed unchanged (time only)")
S.register("RAPID_FA_TILE_UTIL_BWD", None, _optional_float, "model:implementation.kernel_rules.fa_tile_util_bwd",
           "None = 0.25 (Dao 2023 FA-1 range): flash-attention tiles are timed by the native tile model with this "
           "SM utilisation, no padding FLOPs and no OPMODEL_* switch (base_timing flash path); exposed unchanged (time only)")
S.register("RAPID_FLASH_REGISTERS_PER_SM", 65536, int, "model:implementation.kernel_rules.flash_registers_per_sm",
           "CUDA C++ Programming Guide compute capability 8.0 register capacity per SM")
S.register("RAPID_FLASH_REGISTER_ALLOCATION_PER_THREAD", 8, int, "model:implementation.kernel_rules.flash_register_allocation_per_thread",
           "CUDA occupancy register allocation granularity, 8 registers per thread")
S.register("RAPID_FLASH_SFU_OPS_PER_CYCLE", 16.0, float, "model:implementation.kernel_rules.flash_sfu_ops_per_cycle",
           "CUDA C++ Programming Guide arithmetic instruction throughput table, cc 8.0, EX2/RCP/LG2 per SM per clock")
S.register("RAPID_MOE_GEMM_MIN_TILE", 16, lambda raw: int(raw.strip() or 0), "model:implementation.gemm_library.moe_min_tile",
           "EnergyLens arXiv:2605.14249 Sec. 3.2; vLLM v0.10.0 fused_moe BLOCK_SIZE_M = 16")

S.register("RAPID_LINK_ENERGY_RULE", None,
           _link_energy_rule,
           "hw:network.link_energy_rule",
           "Stated link-active time composition rule: AstraSim NCCL = emitted rank launch duration "
           "(program meta collective_launch_overhead fallback) + existing completion_time_ns "
           "(max per-link busy_time_ns + transfers x configured latency); analytical NCCL/P2P = "
           "timing function return; custom AR = custom_seconds; logits gather = gather_seconds. "
           "Payload bits or line rate x active time; A100 low-power mechanism unconfirmed; "
           "no low-power entry time; tiny-message upper bound (network_link_power_design sections 1, 5a, 7).")
S.register("RAPID_L2_RESIDENCY", False, _on, "hw:l2_model.step_simulation",
           "step-level L2 simulation (l2_sim.py)")
S.register("RAPID_L2_PARTITIONS_RULE", None,
           _l2_partitions_rule,
           "hw:l2_model.partitions_rule",
           "H100 counts plan: a100_literal applies the A100 two-partition "
           "mechanism (25 MiB effective); full_capacity uses one partition (50 MiB effective).")
S.register("RAPID_MEMORY_TIME_RULE", "cold_kernel",
           _memory_time_rule, "hw:l2_model.memory_time_rule",
           "memory-bound time from the step-simulated HBM/L2 bytes (consistency with counts)")
S.register("RAPID_L2_STEADY_STATE", "replay3", _l2_steady_state, "hw:l2_model.steady_state",
           "L2 step initialization: legacy replay or periodic tail priming")
S.register("RAPID_L2_ADDRESS_PLAN", "warmup_replay",
           _l2_address_plan, "hw:l2_model.address_plan",
           "allocator address placement: legacy warm-up replay or one event traversal")
S.register("RAPID_L2_SHARING_CLASSES", False, _on, "hw:l2_model.sharing_classes",
           "shared lines cost P copies, private 1 + (P-1)/P (l2_sim 'Sharing classes')")
S.register("RAPID_L2_PRIVATE_COPIES", None, _private_copies, "hw:l2_model.private_copies",
           "1.0 = the PAPER value: single-reader lines cost one copy of the physical L2 (Luo: home "
           "copy only), shared lines two; energy within 0.5 points of the derived value on every preview set, and the "
           "derived value needs the reader-on-the-writer's-side assumption. None = derived 1 + (P-1)/P from "
           "cache.partitions (1.5 on A100), kept as the other value of this setting")
S.register("RAPID_L2_GEMM_MIN_TILE_M", 64, int, "hw:l2_model.gemm_min_tile_m",
           "smallest CTA tile height of the captured cuBLAS GEMMs (ampere_*_64x64_*); B private when m <= it")
S.register("RAPID_L2_COLLECTIVE_BUFFERS", None,
           lambda x: _flash_choice(x, ("occupy_l2", "leave_l2")), "hw:l2_model.collective_buffers",
           "VTP requires an explicit choice for tp > 1; tp = 1 does not consult it. occupy_l2 touches only "
           "user-visible local collective buffers: custom registered IPC and metadata, input and output; "
           "NCCL input and output. NCCL internal FIFOs are excluded: channel count is a runtime tuner result "
           "that cannot be confirmed without a capture. Peer-memory reads are assumed not cached in local L2. "
           "leave_l2 leaves the local cache untouched. Collective traffic is communication, never compute.")

_GEMM_TILE = {
    "order": ("OPMODEL_HBM_ORDER", ("legacy", "launch", "default", "rowmajor_general", "cutlass_identity_swizzle")),
    "single_miss": ("OPMODEL_HBM_SINGLE_MISS", bool),
    "miss_shape": ("OPMODEL_HBM_MISS_SHAPE", ("overflow", "recency")),
    "ws_order": ("OPMODEL_HBM_WS_ORDER", bool),
    "ef_stores": ("OPMODEL_HBM_EF_STORES", bool),
    "strip_in_wave": ("OPMODEL_HBM_STRIP_IN_WAVE", bool),
    "splitk": ("OPMODEL_HBM_SPLITK", bool),
    "splitk_writes": ("OPMODEL_HBM_SPLITK_WRITES", bool),
    "raster_weight_stationary": ("OPMODEL_RASTER_WEIGHT_STATIONARY", bool),
    "dram_read_ecc": ("OPMODEL_HBM_READ_ECC", bool),
    "default_launch_order": ("OPMODEL_HBM_DEFAULT_ORDER", "launch_order"),
}
_L2_MODEL = {
    "memory_time_rule": ("RAPID_MEMORY_TIME_RULE", "memory_time_rule"),
    "steady_state": ("RAPID_L2_STEADY_STATE", L2_STEADY_STATES),
    "partitions_rule": ("RAPID_L2_PARTITIONS_RULE", L2_PARTITIONS_RULES),
    "address_plan": ("RAPID_L2_ADDRESS_PLAN", L2_ADDRESS_PLANS),
    "collective_buffers": ("RAPID_L2_COLLECTIVE_BUFFERS", ("occupy_l2", "leave_l2")),
    "step_simulation": ("RAPID_L2_RESIDENCY", bool),
    "sharing_classes": ("RAPID_L2_SHARING_CLASSES", bool),
    "private_copies": ("RAPID_L2_PRIVATE_COPIES", "copies"),
    "capacity": ("OPMODEL_L2_EFFECTIVE_MIB", "capacity"),
    "gemm_min_tile_m": ("RAPID_L2_GEMM_MIN_TILE_M", int),
}
_GEMM_LIBRARY = {
    "catalog": ("OPMODEL_GEMM_CATALOG", "catalog"),
    "selection_rule": ("OPMODEL_GEMM_SELECTION_RULE", ("min_latency", "useful_fraction")),
    "sm90_min_two_stages": ("OPMODEL_SM90_MIN_TWO_STAGES", bool),
    "sm90_accumulator_limit": ("OPMODEL_SM90_ACCUMULATOR_LIMIT", bool),
    "sm90_cutlass_n_selector": ("OPMODEL_SM90_CUTLASS_N_SELECTOR", bool),
    "forced_templates": ("OPMODEL_GEMM_FORCED_TEMPLATES", "optional_path"),
    "min_tile": ("OPMODEL_MIN_GEMM_TILE", int),
    "moe_min_tile": ("RAPID_MOE_GEMM_MIN_TILE", int),
    "tile_padding": ("OPMODEL_TILE_PADDING", "tile_padding"),
    "epilogue_smem": ("OPMODEL_SMEM_EPILOGUE", ("none", "cutlass")),
    "clear_last_stage": ("OPMODEL_SMEM_CLEAR_STAGE", bool),
    "smem_staging": ("OPMODEL_SMEM_STAGING", bool),
    "smem_staging_latency": ("OPMODEL_SMEM_STAGING_LATENCY", bool),
}
_GEMV = {
    "rule": ("OPMODEL_GEMV_RULE", ("cublas", "trtllm")),
    "max_n": ("OPMODEL_GEMV_MAX_N", int),
    "max_m": ("OPMODEL_GEMV_MAX_M", int),
    "stream_bw": ("OPMODEL_GEMV_STREAM_BW", float),
    "stream_hbm": ("OPMODEL_GEMV_STREAM_HBM", bool),
    "ideal_roofline": ("OPMODEL_GEMV_IDEAL_ROOFLINE", bool),
}
_KERNEL_RULES = {name[len("RAPID_"):].lower(): (name, bool) for name in _KERNEL_RULE_SOURCES}
_KERNEL_RULES["smallgrid_timing"] = ("RAPID_SMALLGRID_TIMING", bool)
_KERNEL_RULES["smallgrid_decode_only"] = ("RAPID_SMALLGRID_DECODE_ONLY", bool)
_KERNEL_RULES.update({
    "moe_grouped_bw": ("RAPID_MOE_GROUPED_BW", float),
    "fa_prefill_scale": ("RAPID_FA_PREFILL_SCALE", float),
    "fa_tile_util_fwd": ("RAPID_FA_TILE_UTIL_FWD", "optional_float"),
    "fa_tile_util_bwd": ("RAPID_FA_TILE_UTIL_BWD", "optional_float"),
    "flash_registers_per_sm": ("RAPID_FLASH_REGISTERS_PER_SM", int),
    "flash_register_allocation_per_thread": ("RAPID_FLASH_REGISTER_ALLOCATION_PER_THREAD", int),
    "flash_fast_forward": ("RAPID_FLASH_FAST_FORWARD", bool),
    "flash_fast_forward_rtol": ("RAPID_FLASH_FAST_FORWARD_RTOL", float),
    "flash_fast_forward_periods": ("RAPID_FLASH_FAST_FORWARD_PERIODS", int),
    "flash_fast_forward_fwd": ("RAPID_FLASH_FAST_FORWARD_FWD", bool),
    "flash_sfu_ops_per_cycle": ("RAPID_FLASH_SFU_OPS_PER_CYCLE", "optional_float"),
})
IMPLEMENTATION_KEYS = ("kind", "framework", "allocator", "engine", "gemm_library", "kernel_rules", "options")
ALLOCATORS = ("default",)

IMPLEMENTATION_OPTIONS = {
    "gradient_accumulation_fusion": (False, bool,
        "Megatron-Core core_v0.12.1 model_parallel_config.py:129-134 gradient_accumulation_fusion; "
        "tensor_parallel/layers.py fused_weight_gradient_mlp_cuda path (true not described); "
        "Kareus conf/megatron_llama3.2_3b_config.yaml sets false; separate dW and main_grad add"),
    "megatron_param_copy": ("separate_copy", ("separate_copy",),
        "Megatron-Core 0.12.1 optimizer/optimizer.py Float16OptimizerWithFloat16Params and "
        "optimizer/distrib_optimizer.py DistributedOptimizer copy main parameters to model parameters"),
    "first_pipeline_stage_layers": (0, int,
                                    "Kareus conf/megatron_llama3.2_3b_config.yaml: num_layers_in_first_pipeline_stage=15; "
                                    "0 preserves remainder-first operator placement, positive selects an explicit first stage in 1F1B"),
    "tp_allreduce": ("vllm_size_rule", ("vllm_size_rule", "nccl"),
                     "program/vllm_allreduce.py: custom_all_reduce.py size/world/topology rule; nccl exposes the legacy path"),
    "vllm_custom_barrier_timing": ("legacy_data_hops", ("legacy_data_hops", "explicit_flag_barriers"),
                     "vLLM v0.10.2 csrc/custom_all_reduce.cuh:181-220,273-336 has two serial flag barriers; "
                     "explicit_flag_barriers is a counterfactual proxy using configured link latency per barrier, "
                     "not a measured flag latency; excludes rank skew and additional polling"),
    "vllm_custom_sync_energy": (False, bool,
                     "vLLM v0.10.2 csrc/custom_all_reduce.cuh:181-220,273-336,488-518: "
                     "price two barriers' 32-bit flag stores and one successful load per peer per block; "
                     "reuse existing communication HBM+L2 byte pricing assumption, not measured cache traffic; "
                     "excludes repeated polls, counter updates and instruction energy"),
    "kernel_min_duration": ("taxbreak_launch_floor", ("off", "taxbreak_launch_floor"),
                     "KERNEL_MIN_5US: every kernel launch lasts at least "
                     "the model's existing TaxBreak launch floor (a100_literature [TAX] LAUNCH_FLOOR_S = 5 us, "
                     "core.device_fixed_overhead_cycles / clock); time = max(model time incl. its launch term, 5 us) per "
                     "launch. Stream kernels already carry the 5 us launch term (no change, never charged twice); CUDA-graph "
                     "nodes keep their configured node term (paper: 0, GRAPH_NODE_CHARGE) inside the compared time. Not a new or tuned number; time only"),
    "vllm_input_broadcast": ("omitted", ("omitted", "source_ring"),
                     "vLLM v0.10.2 worker input GPU tensors broadcast once each; source_ring uses "
                     "NCCL Ring/Simple base, configured per-step latency and bandwidth; assumes "
                     "serial non-overlapping calls, not measured broadcast duration"),
    "flash_source": ("vllm", ("vllm", "dao", "dao-pip-unknown"),
                     "flash_attention_adapter.py source mapping; training defaults to Dao; pip resources remain an assumption"),
    "vocab_parallel_head_gather": (True, bool,
                                   "train_timing.py: padded vocab head and TP gather; false retains the independent infer_vocab_parallel_head kernel rule"),
    **{f"fuse_{op}": (True, bool, "Megatron port rule: one-pass fused forward/backward if true; "
                                  "otherwise kernel_trace_train eager operation sequence; kernel_trace_megatron first reader")
       for op in ("norm", "rope", "activation", "residual_add", "loss")},
    "vocab_parallel_head": (True, bool, "train_timing.py trainfix G1a: legacy per-rank head and CE collectives for every training run"),
    "vocab_parallel_embedding": (True, bool, "train_timing.py trainfix G1b: legacy masked embedding and TP reduction for every training run"),
    "async_dgrad_overlap": (True, bool, "train_timing.py trainfix G6: legacy column-parallel dgrad overlap for every training run"),
    "fp32_loss": (True, bool, "train_timing.py CROSS_ENTROPY_TRAIN_BYTES=4; precision.stats controls normalization, not loss"),

    "megatron_timing": ("operator_path", ("operator_path", "kernel_trace"),
        "program/megatron.py: operator_path keeps the placed operators' durations; kernel_trace sets each compute "
        "operator's duration to the sum of its described launches (GEMM: tile model; flash: structural record; every "
        "other kernel: cold streaming roofline plus one launch overhead, the rule of train_timing._apply_eager_training)"),
    "megatron_unfused_rope": ("hf_eager", ("hf_eager", "megatron_torch"),
        "fuse_rope=false sequence. megatron_torch: megatron/core/models/common/embeddings/rope_utils.py:70-81,107-123 "
        "(_apply_rotary_pos_emb_bshd: cos/sin of freqs, * mscale, .to(dtype) per call; t*cos_, _rotate_half chunk/neg/cat, "
        "*sin_, add, cat((t, t_pass))), all in the activation dtype; backward by autograd (7 launches)"),
    "megatron_layout_copies": (False, bool,
        "megatron/core/transformer/attention.py:838-841 (torch.split then query.reshape: query copy; backward cat of the "
        "three split gradients); transformer_engine/pytorch/dot_product_attention/utils.py:1687-1691 (get_qkv_layout makes "
        "the non-contiguous value slice contiguous); transformer_engine/pytorch/attention.py:4151-4154 and :4504,4512 "
        "(sbhd<->bshd transpose().contiguous(): copies of q, k, v and of the output when batch > 1; a size-1 batch "
        "dimension keeps them contiguous); transformer_engine/pytorch/ops/basic/rmsnorm.py:182 with ops/_common.py:80-93 "
        "(QK norm reshapes its input to contiguous: key copy); "
        "megatron/core/models/common/embeddings/language_model_embedding.py:114-116 (embedding transpose copy when batch > 1)"),
    "te_norm_backward": ("one_kernel", ("one_kernel", "bwd_finalize"),
        "TE RMSNorm backward launches rmsnorm_bwd_*_kernel then rmsnorm_bwd_finalize_*_kernel: "
        "TransformerEngine common/normalization/rmsnorm/rmsnorm_bwd_kernels.cuh (bwd reads x, dz, rs, gamma, writes dx and "
        "per-CTA dgamma_part; finalize sums dgamma_part); line numbers from the 2.4.0 copy on disk, megatron_env/downloads/"
        "transformer_engine_torch-2.4.0/common_headers/common/normalization/rmsnorm/rmsnorm_bwd_kernels.cuh:17,137,174,209; "
        "TE 2.2.0 launches the same two kernel names (trace)"),
    "megatron_loss": ("one_pass", ("one_pass", "native_fused"),
        "fuse_loss=true description. native_fused: cross_entropy_fusion_impl='native', "
        "megatron/core/fusions/fused_cross_entropy.py:13-82 and megatron/core/tensor_parallel/cross_entropy.py:28-30,45,65,80,"
        "114-117 under megatron/core/jit.py:10 (jit_fuser = torch.compile): fp32 logits + row max, exp(logits - max) + row "
        "sum, division by the sum (softmax kept for backward); backward softmax copy with the target entry, then "
        "* grad_output and cast to bf16"),
    "megatron_tied_grad": ("separate_main_grad", ("separate_main_grad", "autograd_sum"),
        "tied embedding/head weight. autograd_sum: autograd sums the head and embedding gradients of the one shared "
        "parameter before its hook runs once (megatron/core/distributed/distributed_data_parallel.py:319,395-416: "
        "param.main_grad.add_(param.grad.data) per parameter)"),
    "megatron_embedding_backward": ("dense_table", ("dense_table", "token_rows"),
        "token_rows: embedding_dense_backward's embedding_backward_feature_kernel adds each token's gradient row into "
        "the zero-filled table (aten/src/ATen/native/cuda/Embedding.cu, torch 2.8.0, not installed): it reads the "
        "tokens x hidden gradient and touches that many table elements, not the whole table; above 3072 indices torch "
        "takes the sort path (kernel_trace_megatron_te.embedding_backward_sorted, 19 launches besides the zero fill)"),
    "megatron_zero_grad": ("per_parameter", ("per_parameter", "grad_buffer"),
        "megatron/core/distributed/param_and_grad_buffer.py:776 grad_data.zero_() on the one contiguous main-grad buffer; "
        "torch splits a fill whose byte span exceeds int32 into halves (torch/include/ATen/TensorIterator.h:503-507, "
        "aten/src/ATen/TensorIterator.cpp can_use_32bit_indexing / SplitUntil32Bit, torch 2.8.0, not installed)"),
    "megatron_optimizer": ("per_parameter", ("per_parameter", "te_fused_adam"),
        "te_fused_adam: transformer_engine/pytorch/optimizers/fused_adam.py:467 step -> multi_tensor_adam over fp32 "
        "(grad, master, exp_avg, exp_avg_sq), multi_tensor_apply.py:24 chunk 2048*32 elements; TransformerEngine "
        "common/multi_tensor/multi_tensor_apply.cuh:65-66,126-137 (depth 4: 36 tensors or 320 blocks per launch; lines "
        "from the 2.4.0 copy under megatron_env/downloads; the TE 2.2.0 trace has the 59 x 320 + 9 launches they give); then "
        "megatron/core/optimizer/optimizer.py:77-92,362-363,651-656 one copy_ per parameter (bf16: no overflow buffer)"),

    "mla_inference": ("vllm_trace", ("vllm_trace", "legacy_operator"),
        "vllm_trace: kernel_trace_vllm_mla (vllm platforms/cuda.py:229-299 TRITON_MLA on sm80; model_executor/layers/"
        "mla.py:110-158; attention/backends/mla/common.py:1170-1307; triton_mla.py:70-110); every GEMM through the tile "
        "model, prefill attention through the FA2 rebuild, the Triton decode kernel by its source geometry; "
        "legacy_operator: the inherited MLA operator composite (train_timing _build_mla_*), not a kernel description"),
    "mla_kv_reads": ("per_head_group", ("per_head_group", "once_per_sequence"),
        "vllm attention/ops/triton_decode_attention.py:417-423 grid (B, cdiv(H, 16), 4) and :320-349: every head-group "
        "program loads its split's latent K, k_pe and the latent V again (per_head_group); once_per_sequence is the "
        "counterfactual of one load per cached row (ideal on-chip reuse)"),
    "mla_prefill_pad_v": (True, bool,
        "vllm attention/backends/mla/common.py:958-966: _pad_v unless FA3 on sm90; sm80 runs FA2, which needs equal "
        "head dims, so v is zero-padded to qk_nope + qk_rope (F.pad: fill + copy); false = the sm90 FA3 path"),
    "mla_decode_stage1_time": ("roofline_l2_hbm", ("roofline_l2_hbm",),
        "base_timing.roofline over (L2 = loads issued by the programs, HBM = first-touch bytes), tensor rate scaled to "
        "min(CTAs, SMs) / SMs, plus one launch overhead; the Triton kernel has no structural model (no compiled "
        "resource data); no fitted number"),
    "mla_decode_stage1_rows": ("local_heads", ("local_heads", "block_h"),
        "tensor work in the Triton decode kernel's time (branch mla-vtp). vllm attention/ops/"
        "triton_decode_attention.py:270-276 masks the heads past VALID_BLOCK_H = min(16, local heads) while the "
        "tl.dot tiles keep BLOCK_H = 16 rows (:417, :461), so with fewer than 16 heads per program (H/tp < 16 under "
        "TP) the tensor cores still issue 16 rows. local_heads: the roofline's tensor rate counts the FLOPs of the "
        "local heads (the single-GPU rule; identical whenever the local heads are a multiple of 16); block_h: it counts "
        "the issued 16-row tiles. Issued FLOPs are recorded in the activity either way"),
    "mla_training": ("megatron_trace", ("megatron_trace", "legacy_operator"),
        "megatron_trace: kernel_trace_megatron MLA attention block (mcore transformer/multi_latent_attention.py:"
        "370-500, gpt_layer_specs.py:106-140) forward and backward; legacy_operator: the inherited composite"),
    "megatron_mla_attention": ("te_unfused", ("te_unfused",),
        "te pytorch/dot_product_attention/utils.py:466-471: FlashAttention is disabled when head_dim_qk != head_dim_v; "
        "the FusedAttention (cuDNN) choice at utils.py:756-775 is decided in C++ (not in the wheel), taken as "
        "No_Backend on sm80 for head dim 192: UnfusedDotProductAttention (attention.py:3801-4017): mask build, "
        "baddbmm, scaled_masked_softmax, bmm, permute copy. NOT verified on a GPU (NVTE_DEBUG=1 would log it)"),
    "megatron_mla_qk_layernorm": (True, bool,
        "mcore models/gpt/gpt_layer_specs.py:118-128: qk_layernorm=True makes linear_q_up_proj / linear_kv_up_proj "
        "TELayerNormColumnParallelLinear (RMSNorm forward layernorm_linear.py:179, then the GEMM :296); DeepSeek-V2/V3 "
        "checkpoints carry q_a_layernorm / kv_a_layernorm, so a faithful port sets it (mcore default False: no norm)"),

    "megatron_dp_grad_sync": ("end_of_backward", ("end_of_backward", "per_work_item"),
        "end_of_backward: DistributedDataParallelConfig.overlap_grad_reduce=False (default, "
        "megatron/core/distributed/distributed_data_parallel_config.py:14) sets bucket_size=None "
        "(distributed_data_parallel.py:60-62): one bucket, reduced synchronously by finish_grad_sync after the "
        "whole backward (param_and_grad_buffer.py:316-320, finalize_model_grads.py); per_work_item: "
        "overlap_grad_reduce=True, asynchronous bucketed reduction during backward (param_and_grad_buffer.py:316-358), "
        "described with one bucket per work item (each layer, embedding, head) where Megatron uses buckets of "
        "max(40M, 1M x dp) parameters (distributed_data_parallel.py:56-59): a stated approximation. The parameter "
        "all-gather is always overlap_param_gather=False (default, distributed_data_parallel_config.py:17): "
        "synchronous after the step (distrib_optimizer.py:2088-2110); overlap_param_gather=True is not described"),
    "megatron_dp_buckets": ("per_reducer", ("per_reducer", "one"),
        "per_reducer: each data-parallel reducer of the program (each layer, embedding, head) carries its own grad-buffer "
        "span (dp_payload padded_grad_buffer). one: DistributedDataParallel(disable_bucketing=True) "
        "(distributed_data_parallel.py:73-79 sets bucket_size=None; :204 partition_buckets(force_single_bucket_group="
        "True)), so the whole grad buffer is one bucket and finish_grad_sync issues ONE all_reduce (DDP) or "
        "reduce_scatter (distributed optimizer) over bucket.grad_data (param_and_grad_buffer.py:340-358), and one "
        "parameter all_gather (:193-229): the first reducer of each kind carries the whole padded buffer, the others "
        "carry zero bytes and emit no collective. Requires megatron_dp_grad_sync=end_of_backward. Set per row "
        "(paper_pipeline/driver.megatron_layout, workload.rules.megatron_ddp_disable_bucketing)"),
    "megatron_grad_scaling": ("pre_scale", ("pre_scale", "average_in_collective"),
        "pre_scale: average_in_collective=False (distributed_data_parallel_config.py:52): grad_data *= 1/dp, one "
        "in-place launch over the reduced buffer before the collective (param_and_grad_buffer.py:298-302; "
        "factor distributed_data_parallel.py:236-275); average_in_collective: ReduceOp.AVG, no launch "
        "(param_and_grad_buffer.py:305-307)"),

    "megatron_moe_score": ("softmax_post_topk", ("softmax_post_topk", "softmax_pre_topk"),
        "moe_router_pre_softmax (default False): moe_utils.py:539-545 topk_softmax_with_capacity; post = topk on logits "
        "then softmax(fp32) over the top-k (normalized top-k probs); pre = softmax(fp32) over all experts, type_as, then "
        "topk (unnormalized, e.g. DeepSeekMoE norm_topk_prob=false)"),
    "megatron_moe_load_balancing": ("aux_loss_zero_coeff", ("aux_loss_zero_coeff", "aux_loss", "none"),
        "moe_router_load_balancing_type aux_loss (default) with moe_aux_loss_coeff 0.0 (default): router.py:217-221 still "
        "computes softmax(logits, fp32) and router.py:285-286 returns before the loss; aux_loss = coeff > 0: "
        "moe_utils.py:25-70 switch_load_balancing_loss_func, router.py:296-311 tracker and MoEAuxLossAutoScaler "
        "(moe_utils.py:177-227); none = routing_type none"),
    "megatron_moe_z_loss": (False, bool,
        "moe_z_loss_coeff (default None): router.py:317-349 and moe_utils.py:126-139 z_loss_func (logsumexp, square, "
        "mean, * coeff; tracker; autograd backward)"),
    "megatron_moe_experts": ("sequential", ("sequential", "te_grouped"),
        "moe_grouped_gemm (default False): experts.py:856-930 SequentialMLP, one MLP (fc1 GEMM, weighted swiglu, fc2 "
        "GEMM) per local expert and torch.cat of the outputs; te_grouped = experts.py:665-819 TEGroupedMLP: TE 2.2 "
        "GroupedLinear (general_grouped_gemm, one cuBLAS GEMM per local expert), one weighted swiglu over all rows"),
    "megatron_moe_count_allgather": (True, bool,
        "token_dispatcher.py:401-402 gather_from_sequence_parallel_region(num_local_tokens_per_expert, tp_ep_group): an "
        "int64 x num_experts all-gather per MoE layer forward, as a program network event (ep > 1, Megatron MoE trace only)"),
    "megatron_moe_probs_alltoall": (True, bool,
        "token_dispatcher.py:515 all_to_all(ep_group, permuted_probs, ...): routed-slot probs in the activation dtype, "
        "forward and its autograd backward (tensor_parallel/mappings.py:418-466), as program network events (ep > 1, "
        "Megatron MoE trace only); off = the received probs are described as the local permuted probs"),
    "megatron_moe_dispatcher": ("alltoall_unfused", ("alltoall_unfused",),
        "moe_token_dispatcher_type alltoall with moe_permute_fusion False (Megatron default dispatcher is allgather; the "
        "operator program's expert-parallel exchange is an all-to-all): token_dispatcher.py:248-656, moe_utils.py:230-417"),

    "vllm_enable_expert_parallel": (False, bool,
        "vLLM 0.10.2 config/parallel.py:99 enable_expert_parallel=False (default). "
        "model_executor/layers/fused_moe/config.py:268-302 FusedMoEParallelConfig.make: false = experts "
        "tensor-parallel over the flattened TP x DP world (every rank holds all experts, "
        "intermediate_size // (tp*dp), layer.py:859-867, no expert_map); true = expert parallel with "
        "ep_size = tp*dp, FusedMoE tp_size = 1, num_experts/ep local experts (layer.py:841-858 determine_expert_map). "
        "Both reduce after the experts with tensor_model_parallel_all_reduce when tp > 1 (layer.py:1820-1824, "
        "qwen3_moe.py:144-153 reduce_results=True)"),
    "vllm_all2all_backend": ("naive", ("naive", "deepep_low_latency", "deepep_high_throughput", "pplx",
                                    "deepep_low_latency_received"),
        "vLLM 0.10.2 envs.py:1059-1060 VLLM_ALL2ALL_BACKEND default 'naive'; device_communicators/"
        "cuda_communicator.py:84-103 creates the manager when data_parallel_size > 1 "
        "(base_device_communicator.py:99-109); all2all.py:17-70 NaiveAll2AllManager: dispatch = one "
        "dp_group.broadcast per DP rank of hidden states and router logits, combine = dp_group.all_reduce then "
        "the local row slice (layer.py:1768-1784,1819-1821). deepep_low_latency / deepep_high_throughput act only "
        "with expert parallel and dp > 1 (fused_moe/config.py:175-191 use_all2all_kernels = dp_size > 1 and use_ep): "
        "one dispatch and one combine all-to-all per MoE layer (deepep_ll_prepare_finalize.py:151-158,221-229; "
        "deepep_ht_prepare_finalize.py:74-117,258-283), bytes by program.moe_rank.deepep_rank_bytes; tp must be 1. "
        "Case-study shortcut for deepep_* only: one link latency per all-to-all (concurrent sends) + bytes/bandwidth "
        "(program.moe_rank.price_all_to_all). Rank-local: top-k on the local tokens; low_latency BatchedTritonExperts on the received rows, high_throughput TritonExperts on the received tokens (kernel_trace_vllm_moe). "
        "deepep_low_latency_received = CASE-STUDY SHORTCUT: deepep_low_latency with the fill, "
        "SiluAndMul and masked w13/w2 over the expected received rows only, not the E_loc x (256 x EP) workspace "
        "(DeepGEMM-masked style). pplx is refused (NotImplementedError); at dp = 1 no dispatch or combine runs"),

    "vllm_use_fused_moe_grouped_topk": (True, bool,
        "vLLM 0.10.2 envs.py:993-994 VLLM_USE_FUSED_MOE_GROUPED_TOPK default 1: fused_moe.py:955-967 sends grouped_topk "
        "to the CUDA kernels (csrc/moe/grouped_topk_kernels.cu topk_with_k2_kernel + group_idx_and_topk_idx_kernel) "
        "when a correction bias exists (topk_method noaux_tc), n_group <= 32 and top_k <= 32; false = the torch "
        "op sequence fused_moe.py:970-1016"),
    "vllm_grouped_topk_launches": ("eager_ops", ("eager_ops",),
        "grouped_topk is @torch.compile(dynamic=True) with the inductor backend (fused_moe.py:942, "
        "platforms/interface.py:134); its compiled kernel grouping is not in the source. eager_ops: one launch per "
        "tensor op with the op's own bytes (an upper bound on launches); a compiled grouping needs a capture"),
}

MLA_DESCRIPTION_OPTIONS = ("mla_inference", "mla_kv_reads", "mla_prefill_pad_v", "mla_decode_stage1_time",
                           "mla_decode_stage1_rows", "mla_training", "megatron_mla_attention",
                           "megatron_mla_qk_layernorm")

IMPLEMENTATION_OPTIONS["megatron_grad_bucket_params"] = (0, int,
    "With megatron_dp_grad_sync=per_work_item (overlap_grad_reduce=True): 0 = one bucket per work item, the "
    "MegatronDP description (Megatron's own buckets are max(40000000, 1000000 x dp) parameters, "
    "distributed_data_parallel.py:56-62, 75-76); N > 0 is not described (program.megatron.validate refuses it)")

MEGATRON_RECOMPUTE_MODULE_CHOICES = ("core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe")
IMPLEMENTATION_OPTIONS["megatron_recompute_granularity"] = ("none", ("none", "selective"),
    "transformer_config.py:210-218 recompute_granularity (default None = 'none': every activation is saved); "
    "'selective' recomputes the submodules of megatron_recompute_modules in every layer (:684-716). 'full' is "
    "sw_param.full_recomputation and is not combined with this option")
IMPLEMENTATION_OPTIONS["megatron_recompute_modules"] = ("core_attn", "modules",
    "transformer_config.py:238-251 recompute_modules, comma-separated (default ['core_attn'], :723-724); acts only "
    "with megatron_recompute_granularity=selective. Described: layernorm (transformer_layer.py:348-353, 440-466, "
    "503-540: input_layernorm and pre_mlp_layernorm when not IdentityOp) and mla_up_proj "
    "(multi_latent_attention.py:76-79, 214-217, 501-509), both output-discarding CheckpointWithoutOutput "
    "(tensor_parallel/random.py:494-574). core_attn, moe_act, mlp and moe are not described "
    "(program.megatron.validate refuses them)")

MEGATRON_RECOMPUTE_OPTIONS = ("megatron_recompute_granularity", "megatron_recompute_modules")

MEGATRON_MULTIGPU_OPTIONS = ("megatron_dp_grad_sync", "megatron_grad_scaling", "megatron_dp_buckets")

def option_setting(key):
    return "RAPID_IMPL_" + key.upper()

MEGATRON_DESCRIPTION_OPTIONS = (
    "megatron_timing", "megatron_unfused_rope", "megatron_layout_copies", "te_norm_backward",
    "megatron_loss", "megatron_tied_grad", "megatron_embedding_backward", "megatron_zero_grad",
    "megatron_optimizer",
)

VLLM_MOE_DESCRIPTION_OPTIONS = ("vllm_enable_expert_parallel", "vllm_all2all_backend",
                                "vllm_use_fused_moe_grouped_topk", "vllm_grouped_topk_launches")

MEGATRON_MOE_OPTIONS = ("megatron_moe_score", "megatron_moe_load_balancing", "megatron_moe_z_loss",
                        "megatron_moe_experts", "megatron_moe_dispatcher", "megatron_moe_count_allgather",
                        "megatron_moe_probs_alltoall", "megatron_grad_bucket_params")

for _key, (_default, _kind, _source) in IMPLEMENTATION_OPTIONS.items():
    S.register(option_setting(_key), _default, env_flag_value if _kind is bool else int if _kind is int else S.as_str,
               "model:implementation.options." + _key, _source)
S.register("RAPID_IMPLEMENTATION", "fused", S.as_str, "model:implementation.kind",
           "interim, until the HF-list acceptance test; fused generic kernel list or handwritten eager HF lists")

def default_implementation_options(run_type="training", kind="fused"):
    options = {key: value[0] for key, value in IMPLEMENTATION_OPTIONS.items()}
    if str(run_type).lower() == "training" or kind == "eager":
        options.update(tp_allreduce="nccl", flash_source="dao", vocab_parallel_head_gather=False)
    return options

def validate_implementation_options(options):
    if not isinstance(options, Mapping):
        raise ValueError("model_param.implementation.options must be a mapping")
    unknown = sorted(set(options) - set(IMPLEMENTATION_OPTIONS))
    if unknown:
        raise ValueError(f"model_param.implementation.options: unknown keys {unknown}")
    for key, value in options.items():
        kind = IMPLEMENTATION_OPTIONS[key][1]
        if kind == "modules":
            parts = value.split(",") if isinstance(value, str) else []
            valid = bool(parts) and all(p in MEGATRON_RECOMPUTE_MODULE_CHOICES for p in parts) \
                and len(set(parts)) == len(parts)
        elif kind is bool:
            valid = isinstance(value, bool)
        elif kind is int:
            valid = isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else:
            valid = isinstance(value, str) and value in kind
        if not valid:
            raise ValueError(f"model_param.implementation.options.{key}: invalid value {value!r}")

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _convert(context: str, value: Any, kind: Any, spec: "CacheSpec") -> Any:
    if kind == "memory_time_rule":
        return _memory_time_rule(value)
    if kind == "optional_path":
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{context} must be null or a nonempty path")
        return value
    if kind is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{context} must be true or false, got {value!r}")
        return value
    if kind is int:
        if not _is_number(value) or int(value) != value or value < 0:
            raise ValueError(f"{context} must be a non-negative integer, got {value!r}")
        return int(value)
    if kind is float:
        if not _is_number(value) or value < 0:
            raise ValueError(f"{context} must be a non-negative number, got {value!r}")
        return float(value)
    if kind == "optional_float":
        if value is None:
            return None
        if not _is_number(value) or value <= 0:
            raise ValueError(f"{context} must be null or a positive number, got {value!r}")
        return float(value)
    if isinstance(kind, tuple):
        if not isinstance(value, str) or value not in kind:
            raise ValueError(f"{context} must be one of {kind}, got {value!r}")
        return str(value)
    if kind == "tile_padding":
        if value != "cta":
            raise ValueError(f"{context} {value!r} is OBSOLETE: tile padding is always 'cta'")
        return "cta"
    if kind == "catalog":
        from opmodel.models.extended_roofline import gemm_catalog_names
        if not isinstance(value, str) or value not in gemm_catalog_names():
            raise ValueError(f"{context} must be one of {gemm_catalog_names()}, got {value!r}")
        return str(value)
    if kind == "capacity":
        if value == "physical":
            return None
        if value == "effective":
            return float(spec.shared_capacity_bytes)
        if isinstance(value, Mapping) and set(value) == {"mib"} and _is_number(value["mib"]) and value["mib"] > 0:
            return float(value["mib"]) * 1024 * 1024
        raise ValueError(f"{context} must be physical, effective or {{mib: <positive number>}}, got {value!r}")
    if kind == "copies":
        if value == "derived":
            return None
        if _is_number(value) and value in (1, 1.5):
            return Fraction(str(float(value)))
        raise ValueError(f"{context} must be derived, 1.0 or 1.5 (ablation files only), got {value!r}")
    if kind == "launch_order":
        if not isinstance(value, Mapping) or set(value) != {"sweep_axis", "strip_width", "serpentine"}:
            raise ValueError(f"{context} must be a mapping {{sweep_axis, strip_width, serpentine}}")
        width = value["strip_width"]
        if value["sweep_axis"] not in ("m", "n") or not isinstance(value["serpentine"], bool) \
                or not _is_number(width) or int(width) != width or width < 1:
            raise ValueError(f"{context}: sweep_axis m | n, strip_width a positive integer, serpentine true | false; "
                             f"got {dict(value)!r}")
        return (str(value["sweep_axis"]), int(width), bool(value["serpentine"]))
    raise AssertionError(kind)

def _apply(context: str, block: Any, table: Mapping[str, Any], out: Dict[str, Any], spec: "CacheSpec",
           nested: Mapping[str, Mapping[str, Any]] = ()) -> None:
    if block is None:
        return
    if not isinstance(block, Mapping):
        raise ValueError(f"{context} must be a mapping")
    nested = dict(nested)
    unknown = [k for k in block if k not in table and k not in nested]
    if unknown:
        raise ValueError(f"{context}: unknown settings {unknown}; known {sorted(list(table) + list(nested))}")
    for key, value in block.items():
        if key in nested:
            _apply(f"{context}.{key}", value, nested[key], out, spec)
            continue
        name, kind = table[key]
        out[name] = _convert(f"{context}.{key}", value, kind, spec)

_CACHE_FIELDS = ("line_bytes", "partitions", "associativity", "dirty_ways")

@dataclass(frozen=True)
class CacheSpec:
    physical_bytes: Optional[int]
    partitions: Optional[int] = None
    line_bytes: Optional[int] = None
    sector_bytes: Optional[int] = None
    associativity: Optional[int] = None
    dirty_ways: Optional[int] = None
    sources: Mapping[str, str] = field(default_factory=dict)
    declared: bool = False

    @property
    def shared_capacity_bytes(self) -> int:
        if self.physical_bytes is None:
            raise ValueError("the L2 capacity needs tech_param.SRAM-L2.size")
        if self.partitions is None:
            raise ValueError("the L2 capacity needs tech_param.SRAM-L2.cache.partitions")
        return int(self.physical_bytes) // int(self.partitions)

    @property
    def private_copies(self) -> Fraction:
        if self.partitions is None:
            raise ValueError("L2 sharing needs tech_param.SRAM-L2.cache.partitions")
        p = int(self.partitions)
        return Fraction(1) + Fraction(p - 1, p)

    @classmethod
    def from_hw(cls, hw_config: Any) -> "CacheSpec":
        tech = getattr(hw_config, "tech_config", None)
        l2 = getattr(tech, "SRAML2", None)
        size = getattr(l2, "size", None)
        block = getattr(l2, "cache", None)
        kwargs: Dict[str, Any] = {}
        sources: Dict[str, str] = {}
        if block is not None:
            if not isinstance(block, Mapping):
                raise ValueError("tech_param.SRAM-L2.cache must be a mapping")
            unknown = [k for k in block if k not in _CACHE_FIELDS + ("sources",)]
            if unknown:
                raise ValueError(f"tech_param.SRAM-L2.cache: unknown fields {unknown}; known {_CACHE_FIELDS + ('sources',)}")
            sources = dict(block.get("sources") or {})
            for key in _CACHE_FIELDS:
                if key not in block or block[key] is None:
                    continue
                value = block[key]
                if isinstance(value, bool) or int(value) != value or int(value) <= 0:
                    raise ValueError(f"tech_param.SRAM-L2.cache.{key} must be a positive integer, got {value!r}")
                if key not in sources:
                    raise ValueError(f"tech_param.SRAM-L2.cache.{key} needs a sources entry (design note S1.1)")
                kwargs[key] = int(value)
        policy = getattr(hw_config, "l2_model", None) or {}
        rule = policy.get("partitions_rule") if isinstance(policy, Mapping) else None
        if S.origin("RAPID_L2_PARTITIONS_RULE") in ("env", "override"):
            rule = S.get("RAPID_L2_PARTITIONS_RULE")
        if rule is not None:
            _l2_partitions_rule(rule)
            kwargs["partitions"] = 2 if rule == "a100_literal" else 1
            sources["partitions"] = "H100 counts plan: " + rule
        return cls(physical_bytes=None if size is None else int(size),
                   sector_bytes=getattr(l2, "sector_size_bytes", None), sources=sources,
                   declared=block is not None, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return {"physical_bytes": self.physical_bytes, "partitions": self.partitions, "line_bytes": self.line_bytes,
                "sector_bytes": self.sector_bytes, "associativity": self.associativity, "dirty_ways": self.dirty_ways,
                "sources": dict(self.sources), "declared": self.declared,
                "shared_capacity_bytes": None if self.physical_bytes is None or self.partitions is None else self.shared_capacity_bytes,
                "private_copies": None if self.partitions is None else str(self.private_copies)}

@dataclass(frozen=True)
class SimPolicy:
    step_simulation: bool
    sharing_classes: bool
    private_copies: Optional[Fraction]
    effective_capacity_bytes: Optional[float]
    gemm_min_tile_m: int
    steady_state: str
    address_plan: str

    @classmethod
    def current(cls, spec: CacheSpec) -> "SimPolicy":
        copies = S.get("RAPID_L2_PRIVATE_COPIES") if S.get("RAPID_L2_SHARING_CLASSES") else None
        return cls(step_simulation=bool(S.get("RAPID_L2_RESIDENCY")),
                   sharing_classes=bool(S.get("RAPID_L2_SHARING_CLASSES")),
                   private_copies=copies if copies is not None else (
                       spec.private_copies if spec.partitions is not None or S.get("RAPID_L2_SHARING_CLASSES") else None),
                   effective_capacity_bytes=S.get("OPMODEL_L2_EFFECTIVE_MIB"),
                   gemm_min_tile_m=int(S.get("RAPID_L2_GEMM_MIN_TILE_M")),
                   steady_state=S.get("RAPID_L2_STEADY_STATE"),
                   address_plan=S.get("RAPID_L2_ADDRESS_PLAN"))

def values_from_configs(hw_config: Any, model_config: Any = None) -> Dict[str, Any]:
    if "RAPID_MEMORY_TIME_RULE" in os.environ:
        _memory_time_rule(os.environ["RAPID_MEMORY_TIME_RULE"])
    spec = CacheSpec.from_hw(hw_config)
    out: Dict[str, Any] = {}
    sw = getattr(hw_config, "sw_config", None)
    if sw is not None and getattr(sw, "allreduce_model", "astrasim") == "measured_curve":
        for key, (default, parser) in ALLREDUCE_SETTINGS.items():
            value = getattr(sw, key, default)
            if key == "allreduce_include_unverified":
                if not isinstance(value, bool):
                    raise ValueError("sw_param.allreduce_include_unverified must be true or false")
            else:
                value = parser(value)
            out["RAPID_" + key.upper()] = value
    if sw is not None and getattr(sw, "allreduce_rsag_rule", "network_model") != "network_model":
        out["RAPID_ALLREDUCE_RSAG_RULE"] = getattr(sw, "allreduce_rsag_rule")
    if sw is not None:
        mode = parse_kernel_launch_mode(getattr(sw, "kernel_launch_mode", "stream"))
        graph_overhead = parse_graph_launch_overhead(getattr(sw, "graph_launch_overhead_s", None))
        if mode != "stream":
            out["RAPID_KERNEL_LAUNCH_MODE"] = mode
        if graph_overhead is not None:
            out["RAPID_GRAPH_LAUNCH_OVERHEAD_S"] = graph_overhead
    if sw is not None and getattr(sw, "nccl_latency_rule", "astrasim_per_hop") != "astrasim_per_hop":
        out["RAPID_NCCL_LATENCY_RULE"] = parse_nccl_latency_rule(
            getattr(sw, "nccl_latency_rule", "astrasim_per_hop"))
    link_rule = getattr(getattr(hw_config, "network_layout", None), "link_energy_rule", None)
    if link_rule is not None:
        if link_rule not in ("per_payload_byte", "line_rate_while_active"):
            raise ValueError("network.link_energy_rule must be per_payload_byte or line_rate_while_active")
        out["RAPID_LINK_ENERGY_RULE"] = link_rule
    _apply("l2_model", getattr(hw_config, "l2_model", None), _L2_MODEL, out, spec,
           nested={"gemm_tile": _GEMM_TILE})
    model = getattr(model_config, "model_config", model_config)
    _apply("model_param.implementation.gemm_library", getattr(model, "gemm_library", None), _GEMM_LIBRARY, out, spec,
           nested={"gemv": _GEMV})
    _apply("model_param.implementation.kernel_rules", getattr(model, "kernel_rules", None), _KERNEL_RULES, out, spec)
    if out.get("RAPID_FLASH_REBUILD") is False:
        raise ValueError("kernel_rules.flash_rebuild: false is obsolete; use the flash attention rebuild")
    if "RAPID_FLASH_REBUILD" in os.environ:
        _flash_rebuild(os.environ["RAPID_FLASH_REBUILD"])
    options = getattr(model, "implementation_options", None)
    if options is None and getattr(model, "implementation", None) is not None:
        options = default_implementation_options(getattr(model, "run_type", "training"), model.implementation)
    if options is not None:
        validate_implementation_options(options)
        out.update({option_setting(key): value for key, value in options.items()})
        out["RAPID_IMPLEMENTATION"] = model.implementation
    return out

def layer(hw_config: Any, model_config: Any = None) -> Dict[str, Any]:
    return S.check(values_from_configs(hw_config, model_config))

def resolution(hw_config: Any, model_config: Any = None) -> Dict[str, Any]:
    spec = CacheSpec.from_hw(hw_config)
    core = getattr(getattr(hw_config, "tech_config", None), "core", None)
    model = getattr(model_config, "model_config", model_config)
    policy = SimPolicy.current(spec)
    settings = S.resolved(exclude_prefixes=() if getattr(core, "architecture", None) == "sm90"
                          else ("OPMODEL_SM90_",))

    if settings["RAPID_L2_PARTITIONS_RULE"]["from"] == "default":
        del settings["RAPID_L2_PARTITIONS_RULE"]
    curve_enabled = S.get("RAPID_ALLREDUCE_MODEL") == "measured_curve"
    if not curve_enabled:
        for key in ALLREDUCE_SETTINGS:
            name = "RAPID_" + key.upper()
            if settings[name]["from"] == "default":
                del settings[name]
    if settings["RAPID_ALLREDUCE_RSAG_RULE"]["from"] == "default":
        del settings["RAPID_ALLREDUCE_RSAG_RULE"]
    for name in ("RAPID_FLASH_FAST_FORWARD", "RAPID_FLASH_FAST_FORWARD_RTOL", "RAPID_FLASH_FAST_FORWARD_PERIODS",
                 "RAPID_FLASH_FAST_FORWARD_FWD"):
        if settings[name]["from"] == "default":
            del settings[name]
    launch_mode = S.get("RAPID_KERNEL_LAUNCH_MODE")
    graph_overhead = S.get("RAPID_GRAPH_LAUNCH_OVERHEAD_S")
    if launch_mode == "graph_when_captured" and graph_overhead is None:
        raise ValueError("graph_when_captured requires sw_param.graph_launch_overhead_s")

    for key, origin in (getattr(model, "implementation_origins", None) or {}).items():
        name = "RAPID_IMPLEMENTATION" if key == "kind" else option_setting(key)
        if settings[name]["from"] == "yaml":
            settings[name]["from"] = origin

    omitted_options = set()
    if getattr(model, "framework", None) != "megatron-core-0.12.1":
        for key in MEGATRON_DESCRIPTION_OPTIONS:
            name = option_setting(key)
            if settings[name]["from"] == "default":
                del settings[name]
                omitted_options.add(key)
    attention = getattr(getattr(model, "attention", None), "attention_type", None)
    if str(attention or "").lower() != "mla":
        for key in MLA_DESCRIPTION_OPTIONS:
            name = option_setting(key)
            if settings[name]["from"] == "default":
                del settings[name]
                omitted_options.add(key)
    _moe_model = (str(getattr(model, "model_type", "") or "").lower() == "qwen3_moe"
                  or (str(getattr(model, "model_type", "") or "").lower() in ("deepseek_v2", "deepseek_v3")
                      and int(getattr(getattr(model, "moe", None), "num_experts", 1) or 1) > 1))
    if not _moe_model:
        for key in VLLM_MOE_DESCRIPTION_OPTIONS:
            name = option_setting(key)
            if settings[name]["from"] == "default":
                del settings[name]
                omitted_options.add(key)
    for key in MEGATRON_MULTIGPU_OPTIONS + MEGATRON_RECOMPUTE_OPTIONS:
        name = option_setting(key)
        if settings[name]["from"] == "default":
            del settings[name]
            omitted_options.add(key)
    if not bool(getattr(model, "use_moe", False)) or getattr(model, "framework", None) != "megatron-core-0.12.1":
        for key in MEGATRON_MOE_OPTIONS:
            name = option_setting(key)
            if name in settings and settings[name]["from"] == "default":
                del settings[name]
                omitted_options.add(key)
    record = {
        "settings": settings,
        "kernel_launch": {"mode": launch_mode,
                          "stream_overhead_s": getattr(getattr(hw_config, "sw_config", None), "kernel_launch_overhead", None),
                          "graph_overhead_s": graph_overhead, "decode_templates": []},
        "network_line_rate_inputs": [
            {"dimension": getattr(dim, "label", None),
             "links_per_gpu": getattr(dim, "links_per_gpu", None),
             "lanes_per_link": getattr(dim, "lanes_per_link", None),
             "lane_rate_bps": getattr(dim, "lane_rate_bps", None),
             "source": "A100 reference: NVIDIA A100 Tensor Core GPU Architecture whitepaper, "
                       "Third-generation NVLink (12 links x 4 lanes x 50e9 bps); "
                       "actual values are dimension configuration, not a default for other networks"}
            for dim in getattr(getattr(hw_config, "network_layout", None), "dimensions", ())
        ],
        "cache": spec.to_dict(),
        "l2_policy": {"step_simulation": policy.step_simulation, "sharing_classes": policy.sharing_classes,
                      "private_copies": None if policy.private_copies is None else str(policy.private_copies),
                      "shared_capacity_bytes": None if spec.physical_bytes is None or spec.partitions is None else spec.shared_capacity_bytes,
                      "effective_capacity_bytes": policy.effective_capacity_bytes,
                      "gemm_min_tile_m": policy.gemm_min_tile_m},
        "core": {"max_ctas_per_sm": getattr(core, "max_ctas_per_sm", None),
                 "max_warps_per_sm": getattr(core, "max_warps_per_sm", None)},
        "implementation": {"kind": S.get("RAPID_IMPLEMENTATION"), "framework": getattr(model, "framework", None),
                           "allocator": getattr(model, "allocator", None), "engine": getattr(model, "engine", None),
                           "options": {key: S.get(option_setting(key)) for key in IMPLEMENTATION_OPTIONS
                                       if key not in omitted_options},
                           "kernel_description_resolved": getattr(model, "kernel_description_resolved", False)},
    }
    if curve_enabled:
        record["allreduce_usage"] = {
            "status": "not_applied", "not_applied_reason": "unsupported_execution_path",
            "scope": "evaluated logical collectives including DP multiplicity; decode samples only",
            "curve_collectives": 0, "fallback_collectives": 0, "fallback_reasons": {},
        }
    return record

def settings_profile(resolution_record: Mapping[str, Any], reference: Mapping[str, Any],
                     reference_name: str = "paper") -> Dict[str, Any]:
    record = resolution_record.get("settings", {}) or {}
    registry = S.registry()
    differs: Dict[str, Any] = {}
    ignored = []
    compared = 0
    for name, entry in sorted(record.items()):
        setting = registry.get(name)
        if setting is not None and "time only" in str(setting.source).lower():
            ignored.append(name)
            continue
        ref = reference[name] if name in reference else (setting.default if setting is not None else None)
        compared += 1
        value = entry.get("value") if isinstance(entry, Mapping) else entry
        if str(value) != str(ref):
            differs[name] = {"value": value, "reference": ref,
                             "origin": entry.get("from") if isinstance(entry, Mapping) else None}
    for block in ("implementation", "l2_policy"):
        if block in reference:
            got = resolution_record.get(block, {}) or {}
            for key, ref in dict(reference[block]).items():
                compared += 1
                if str(got.get(key)) != str(ref):
                    differs[f"{block}.{key}"] = {"value": got.get(key), "reference": ref, "origin": "config"}
    return {"profile": reference_name if not differs else f"non-{reference_name}", "reference": reference_name,
            "compared": compared, "differs": differs, "time_only_ignored": ignored}
