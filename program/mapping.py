from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

from astrasim_lib import gmap
from timing_model import CollectiveType

from program.ir import CollectiveOp, Program, TransferOp

__all__ = ["apply_first_dim_mapping"]

_DOUBLE_WEIGHT = (CollectiveType.ALL_REDUCE, CollectiveType.ALL_TO_ALL)

def _axis_layout(program: Program) -> Tuple[Dict[str, int], Dict[int, Dict[str, int]]]:
    layout = program.layout
    axis_sizes = {
        str(axis): max(1, int(layout.axis_sizes.get(axis, 1)))
        for axis in layout.axis_order
    }
    coords = {
        int(device): {
            str(axis): int(value) for axis, value in layout.coords_of(int(device)).items()
        }
        for device in program.devices
    }
    return axis_sizes, coords

def apply_first_dim_mapping(
    program: Program,
    *,
    optimize_2dmap: Optional[Mapping[str, Any]],
    workdir: Optional[str] = None,
) -> Program:
    if not optimize_2dmap:
        return program
    axis_sizes, stage_axis_coords = _axis_layout(program)
    if not axis_sizes:
        return program

    tmpdir: Optional[str] = None
    if workdir is None:
        tmpdir = tempfile.mkdtemp(prefix="rapid_gmap_")
        workdir = tmpdir
    os.makedirs(workdir, exist_ok=True)
    try:
        collector = gmap.begin_collection(
            optimize_2dmap, axis_sizes, stage_axis_coords, workdir
        )
        if collector is None:  # pragma: no cover - begin_collection guards
            return program

        for op in program.ops:
            if isinstance(op, CollectiveOp):
                axis = str(op.interconnect or "").strip().lower()
                if not axis or axis not in collector.subset_axes:
                    continue
                collector.record_collective(
                    axis=axis,
                    stage_id=int(op.device),
                    size_bytes=int(op.size_bytes),
                    participant_count=int(op.participants),
                    double_weight=op.coll in _DOUBLE_WEIGHT,
                )
            elif isinstance(op, TransferOp):
                if not collector.include_pipeline:
                    continue
                if op.comm_type is not CollectiveType.PIPELINE:
                    continue
                if int(op.src_device) == int(op.dst_device):
                    continue
                collector.record_pipeline(
                    src_stage=int(op.src_device),
                    dst_stage=int(op.dst_device),
                    size_bytes=int(op.size_bytes),
                )

        result = gmap.finalize_collection(collector)
        if not result:
            return program
        permutation = result.permutation
        if len(permutation) != collector.vertex_count:
            raise ValueError(
                "Permutation length does not match first-dimension vertex count."
            )

        def _sort_key(device: int) -> Tuple[int, ...]:
            coords = collector.stage_axis_coords.get(device, {})
            higher = tuple(
                int(coords.get(axis, 0)) for axis in collector.replication_axes
            )
            return higher + (permutation[collector.local_index_for_stage(device)],)

        program.devices = tuple(sorted(program.devices, key=_sort_key))
        program.meta.misc["first_dim_mapping"] = tuple(int(v) for v in permutation)
        return program
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
