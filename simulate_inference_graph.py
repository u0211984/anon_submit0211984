import os
import shutil
from typing import Any, Callable, Dict, List, Tuple, Optional
from dataclasses import dataclass

import llm_util
from train_timing import LLMExecutionDispatcher
from program.execution_energy import (
    ExecutionEnergyResult,
    apply_inference_gpu_accounting,
    combine_execution_energy,
)

@dataclass
class InferenceConfig:
    batch_size: int
    seq_len: int
    decode_len: int
    hidden_dim: int
    num_heads: int
    kv_heads: int
    intermediate_size: int
    vocab_size: int
    use_moe: bool
    num_experts: int
    top_k: int

    sample_every: int = 32

@dataclass
class DecodeSample:
    step_id: int
    current_seq_len: int
    execution_time: float
    execution_energy: float
    kv_cache_tokens: int
    energy_result: Optional[ExecutionEnergyResult] = None
    kernel_launch: Optional[dict] = None
    allreduce_usage: Optional[dict] = None

class DecodeGraph:
    def __init__(
        self,
        config: InferenceConfig,
        hw_config,
        model_config,
        time_calc_cls: Callable[..., Any],
    ):
        self.config = config
        self.hw_config = hw_config
        self.model_config = model_config
        self.time_calc_cls = time_calc_cls
        self._decode_sample_root: Optional[str] = None
        self.use_moe = config.use_moe
        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.moe_num_experts = int(getattr(config, "num_experts", 1) or 1)
        self.moe_top_k = int(getattr(config, "top_k", 1) or 1)
        model_cfg = getattr(model_config, "model_config", None)
        self.moe_intermediate_size = int(
            getattr(model_cfg, "moe_intermediate_size", getattr(config, "intermediate_size", 0))
        )
        self.n_shared_experts = int(getattr(model_cfg, "n_shared_experts", 0) or 0)
        self.head_dim = getattr(model_cfg, "head_dim", None)
        attention_cfg = getattr(model_cfg, "attention", None)
        self.attention_type = getattr(attention_cfg, "attention_type", "mha")
        self.kv_lora_rank = getattr(attention_cfg, "kv_lora_rank", None)
        self.q_lora_rank = getattr(attention_cfg, "q_lora_rank", None)
        self.qk_nope_head_dim = getattr(attention_cfg, "qk_nope_head_dim", None)
        self.qk_rope_head_dim = getattr(attention_cfg, "qk_rope_head_dim", None)
        self.v_head_dim = getattr(attention_cfg, "v_head_dim", None)
        self.run_type = str(getattr(model_cfg, "run_type", "inference")).lower()
        self.execution_energy_result: Optional[ExecutionEnergyResult] = None

    def build_decode_graph(self) -> Tuple[float, float, List[DecodeSample]]:
        sample_points = self._generate_sample_points()

        decode_samples = []
        for step_id in sample_points:
            generated_tokens = step_id + 1
            total_seq_len = self.config.seq_len + generated_tokens

            gemm_shapes = llm_util.process_decode_gemm_shapes(
                self,
                batch_size=self.config.batch_size,
                current_seq_len=total_seq_len,
                d_model=self.config.hidden_dim,
                num_heads=self.config.num_heads,
                kv_heads=self.config.kv_heads,
                intermediate_size=self.moe_intermediate_size if self.use_moe else self.config.intermediate_size,
                vocab_size=self.config.vocab_size,
                model_type=self.model_config.model_config.model_type,
            )

            sample_time, sample_energy_result = self._execute_decode_step(
                step_id=step_id,
                total_seq_len=total_seq_len,
                gemm_shapes=gemm_shapes,
            )

            decode_samples.append(
                DecodeSample(
                    step_id=step_id,
                    current_seq_len=total_seq_len,
                    execution_time=sample_time,
                    execution_energy=sample_energy_result.total_j,
                    kv_cache_tokens=total_seq_len,
                    energy_result=sample_energy_result,
                    kernel_launch=getattr(self, "_last_kernel_launch", None),
                    allreduce_usage=getattr(self, "_last_allreduce_usage", None),
                )
            )

        total_decode_time, total_decode_energy = self._integrate_decode_samples(decode_samples)

        return total_decode_time, total_decode_energy, decode_samples

    def _generate_sample_points(self) -> List[int]:
        sample_points = []

        sample_points.append(0)

        for step_id in range(self.config.sample_every - 1, self.config.decode_len, self.config.sample_every):
            sample_points.append(step_id)

        last_step = self.config.decode_len - 1
        if last_step not in sample_points:
            sample_points.append(last_step)

        import sim_settings
        from opmodel import settings
        graph_mode = False
        if getattr(self, "hw_config", None) is not None and getattr(self, "model_config", None) is not None:
            with settings.use(sim_settings.layer(self.hw_config, self.model_config)):
                graph_mode = (settings.get("RAPID_KERNEL_LAUNCH_MODE") == "graph_when_captured"
                              and settings.get("RAPID_IMPLEMENTATION") == "fused"
                              and settings.get(sim_settings.option_setting("flash_source")) == "vllm")
        if graph_mode:
            from kernel_trace_vllm import VllmEngine
            sm_count = self.hw_config.tech_config.core.num_bundles
            if sm_count is None:
                raise ValueError("decode graph sampling requires hardware tech_param.core.num_bundles")
            eng = VllmEngine(sm_count=int(sm_count),
                             **dict(getattr(self.model_config.model_config, "engine", None) or {}))
            first_eager = min(8192, int(eng.max_model_len)) - int(self.config.seq_len)
            if eng.cuda_graph and 0 < first_eager < self.config.decode_len:
                sample_points.extend((first_eager - 1, first_eager))

        return sorted(set(sample_points))

    def _execute_decode_step(
        self,
        *,
        step_id: int,
        total_seq_len: int,
        gemm_shapes: Dict[str, Tuple[int, ...]],
    ) -> Tuple[float, ExecutionEnergyResult]:
        if not self.hw_config or not self.model_config:
            raise RuntimeError("Hardware config and model config are required for decode step execution.")
        if not self.time_calc_cls:
            raise RuntimeError("time_calc_cls must be provided for decode execution.")

        temp_time_calc = self.time_calc_cls(
            hw_config=self.hw_config,
            model_config=self.model_config,
            mode="LLM",
            output_dir="./output/LLM/",
        )

        base_dir = temp_time_calc.output_dir.rstrip(os.sep)
        sample_dir = None
        if self._debug_graphs_enabled():
            sample_dir = os.path.join(base_dir, "decode_samples", f"step_{step_id:04d}")
            root_dir = os.path.dirname(sample_dir)
            if self._decode_sample_root is None:
                self._decode_sample_root = root_dir
                if os.path.isdir(root_dir):
                    shutil.rmtree(root_dir, ignore_errors=True)
            os.makedirs(sample_dir, exist_ok=True)
            gemm_shapes_file = os.path.join(sample_dir, "decode_gemm_shapes.txt")
            with open(gemm_shapes_file, "w", encoding="utf-8") as f:
                for key, shape in gemm_shapes.items():
                    f.write(f"{key}: {shape}\n")
            prev_output_dir = temp_time_calc.output_dir
            temp_time_calc.output_dir = sample_dir

        execution_graphs, _legacy_energy = temp_time_calc.prepare_decode_graphs(
            batch_size=self.config.batch_size,
            total_seq_len=total_seq_len,
            gemm_shapes=gemm_shapes,
        )
        workload, _ = execution_graphs
        self._last_kernel_launch = dict(temp_time_calc.resolved_settings["kernel_launch"]["decode_templates"][-1])

        dispatcher = LLMExecutionDispatcher(temp_time_calc, workload)

        with temp_time_calc.inference_phase_network_override("decode"):
            result = dispatcher.run(temp_time_calc.execution_mode)
        self._last_allreduce_usage = temp_time_calc.resolved_settings.get("allreduce_usage")
        if sample_dir:
            temp_time_calc.output_dir = prev_output_dir
            print(
                f"[decode] sample step {step_id}: seq_len={total_seq_len}, "
                f"time={result.total_time:.4f}s, artifacts={sample_dir}"
            )
        else:
            print(
                f"[decode] sample step {step_id}: seq_len={total_seq_len}, "
                f"time={result.total_time:.4f}s"
            )
        if result.energy is None:
            raise RuntimeError("decode execution did not return energy")
        energy_result = result.energy
        energy_result = apply_inference_gpu_accounting(
            energy_result,
            getattr(temp_time_calc, "_last_inference_kernel_energy_records", ()),
            layer_count=temp_time_calc.num_layers,
        )
        return result.total_time, energy_result

    def _integrate_decode_samples(self, samples: List[DecodeSample]) -> Tuple[float, float]:
        if not samples:
            raise ValueError("No decode samples available for integration")

        total_time = 0.0
        total_energy = 0.0
        energy_parts = []

        for idx, sample in enumerate(samples):
            if idx == 0:
                total_time += sample.execution_time
                total_energy += sample.execution_energy
                if sample.energy_result is None:
                    raise RuntimeError("decode sample is missing typed energy")
                energy_parts.append((sample.energy_result, 1.0))
                if self._debug_graphs_enabled():
                    print(
                        f"[decode] integration seed step {sample.step_id:02d}: "
                        f"time={sample.execution_time:.4f}s, energy={sample.execution_energy:.4f}J"
                    )
                continue

            prev_sample = samples[idx - 1]
            step_gap = sample.step_id - prev_sample.step_id
            if step_gap <= 0:
                raise ValueError("Sample points must be strictly increasing")

            transition = (sample.kernel_launch is not None and prev_sample.kernel_launch is not None
                          and sample.kernel_launch["mode"] == "graph_when_captured"
                          and sample.kernel_launch["captured"] != prev_sample.kernel_launch["captured"])
            if transition and step_gap != 1:
                raise ValueError("kernel launch capture transition must have adjacent decode samples")
            midpoint = (sample.execution_time if transition else
                        0.5 * (prev_sample.execution_time + sample.execution_time))
            segment_time = midpoint * step_gap
            total_time += segment_time

            midpoint_energy = (sample.execution_energy if transition else
                               0.5 * (prev_sample.execution_energy + sample.execution_energy))
            segment_energy = midpoint_energy * step_gap
            total_energy += segment_energy
            if prev_sample.energy_result is None or sample.energy_result is None:
                raise RuntimeError("decode sample is missing typed energy")
            if transition:
                energy_parts.append((sample.energy_result, 1.0))
            else:
                energy_parts.extend(((prev_sample.energy_result, 0.5 * step_gap),
                                     (sample.energy_result, 0.5 * step_gap)))

            print(
                f"[decode] integration segment {prev_sample.step_id}->{sample.step_id}: "
                f"width={step_gap}, midpoint={midpoint:.4f}s, contribution={segment_time:.4f}s"
            )

        last_sample = samples[-1]
        remaining_steps = self.config.decode_len - (last_sample.step_id + 1)
        if remaining_steps > 0:
            tail_time = remaining_steps * last_sample.execution_time
            total_time += tail_time
            total_energy += remaining_steps * last_sample.execution_energy
            if last_sample.energy_result is None:
                raise RuntimeError("decode sample is missing typed energy")
            energy_parts.append((last_sample.energy_result, float(remaining_steps)))
            print(
                f"[decode] integration tail from {last_sample.step_id} covering {remaining_steps} steps: "
                f"contribution={tail_time:.4f}s"
            )

        print(
            f"[decode] total interpolated decode time: {total_time:.4f}s, energy: {total_energy:.4f}J "
            f"from {len(samples)} samples"
        )

        self.execution_energy_result = combine_execution_energy(
            energy_parts,
            execution_mode="sampled_decode",
        )
        return total_time, total_energy

    def _debug_graphs_enabled(self) -> bool:
        flag = os.environ.get("RAPID_VISUALIZE_GRAPHS")
        if flag is None:
            return False
        return flag.strip().lower() not in {"", "0", "false", "no"}

class InferenceEngine:
    def __init__(
        self,
        config: InferenceConfig,
        hw_config=None,
        model_config=None,
        *,
        time_calc_cls: Optional[Callable[..., Any]] = None,
    ):
        self.config = config
        self.hw_config = hw_config
        self.model_config = model_config
        self.decode_graph: Optional[DecodeGraph] = None
        self.time_calc_cls = time_calc_cls

    def _build_decode_graph(self) -> Tuple[float, float, List[DecodeSample]]:
        if self.time_calc_cls is None:
            raise RuntimeError("InferenceEngine requires time_calc_cls for decode graph building.")

        self.decode_graph = DecodeGraph(
            config=self.config,
            hw_config=self.hw_config,
            model_config=self.model_config,
            time_calc_cls=self.time_calc_cls,
        )

        return self.decode_graph.build_decode_graph()
