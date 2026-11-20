from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from program.ir import ComputeOp, Direction, OpRole, Program

@dataclass(frozen=True)
class BlockTimings:
    dense: Optional[Any] = None
    moe: Optional[Any] = None
    stage_dense: Mapping[Tuple[int, int], Any] = field(default_factory=dict)
    stage_moe: Mapping[Tuple[int, int], Any] = field(default_factory=dict)

def apply_block_timings(
    pipeline_program: Program,
    timings: BlockTimings,
    dp_count: int,
) -> int:
    if not isinstance(pipeline_program, Program):
        raise TypeError(
            f"apply_block_timings expects a Program (got {type(pipeline_program).__name__})"
        )
    if pipeline_program.meta.misc.get("granularity") != "pipeline":
        raise RuntimeError("apply_block_timings requires a PIPELINE program")

    dp_count = max(1, int(dp_count))
    retimed = 0

    for op in pipeline_program.ops:
        if not isinstance(op, ComputeOp) or op.role is not OpRole.TRANSFORMER_LAYER:
            continue

        is_moe_layer = bool(op.is_moe_layer)
        timing_source = (
            timings.moe if is_moe_layer and timings.moe is not None else timings.dense
        )
        if timing_source is None:
            timing_source = timings.dense
        if timing_source is None:
            continue

        hw_stage = int(op.device)

        if pipeline_program.meta.misc.get("sharding_policy") == "zero_moe":
            hw_stage = int(pipeline_program.layout.coords_of(int(op.device)).get("pp", 0))
        is_forward = op.direction is Direction.FORWARD
        overrides = timings.stage_moe if is_moe_layer else timings.stage_dense

        values = []
        for dp_idx in range(dp_count):
            default = timing_source.forward if is_forward else timing_source.backward
            timing_override = overrides.get((dp_idx, hw_stage))
            if timing_override:
                values.append(
                    float(timing_override.forward if is_forward else timing_override.backward)
                )
            else:
                values.append(float(default))

        op.duration = tuple(values)
        retimed += 1

    return retimed
