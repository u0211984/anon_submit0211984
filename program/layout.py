from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from program.axes import (  # noqa: F401 - re-exported for existing importers
    AXES,
    AXIS_BY_NAME,
    CANONICAL_AXES,
    CLUSTER_AXES,
    PIPELINE_AXES,
    REPLICA_AXES,
    AxisName,
    AxisRole,
    cluster_strides,
)

def cluster_coords(
    axis_order: Sequence[AxisName],
    tp_rank: int,
    stage_id: int,
    *,
    sizes: Mapping[AxisName, int],
) -> Dict[AxisName, int]:
    coords: Dict[AxisName, int] = {}

    for name, stride in cluster_strides(axis_order, sizes).items():
        coords[name] = (tp_rank // max(1, stride)) % max(1, int(sizes[name]))
    for name in PIPELINE_AXES:
        if name not in axis_order:
            continue
        extent = max(1, int(sizes.get(name, 1)))
        if stage_id < 0 or stage_id >= extent:
            raise ValueError(
                f"stage_id {stage_id} is out of range for {name}={extent}"
            )
        coords[name] = stage_id % extent
    return coords

@dataclass(frozen=True, slots=True)
class RankLayout:
    axis_order: Tuple[AxisName, ...]
    axis_sizes: Dict[AxisName, int]
    axis_strides: Dict[AxisName, int]

    def num_ranks(self) -> int:
        span = 1
        for axis in self.axis_order:
            span *= self.axis_sizes.get(axis, 1)
        return span

    def linearize(self, coords: Mapping[AxisName, int]) -> int:
        linear_rank = 0
        for axis in self.axis_order:
            coord = coords.get(axis, 0)
            size = self.axis_sizes.get(axis, 1)
            if coord < 0 or coord >= size:
                raise ValueError(f"Coordinate {coord} for axis '{axis}' is out of range <{size}")
            stride = self.axis_strides.get(axis)
            if stride is None:
                raise KeyError(f"Rank layout stride missing for axis '{axis}'")
            linear_rank += coord * stride
        return linear_rank

    def coords_of(self, rank: int) -> Dict[AxisName, int]:
        rank_int = int(rank)
        if rank_int < 0 or rank_int >= self.num_ranks():
            raise ValueError(f"Rank {rank_int} is out of range for layout with {self.num_ranks()} ranks")
        coords: Dict[AxisName, int] = {}
        for axis in self.axis_order:
            size = self.axis_sizes.get(axis, 1)
            stride = self.axis_strides.get(axis)
            if stride is None:
                raise KeyError(f"Rank layout stride missing for axis '{axis}'")
            coords[axis] = (rank_int // stride) % size
        return coords

    def subset(self, axes: Sequence[AxisName]) -> "RankLayout":
        selected = [axis for axis in self.axis_order if axis in axes and self.axis_sizes.get(axis, 1) >= 1]
        strides: Dict[AxisName, int] = {}
        span = 1
        for axis in selected:
            strides[axis] = span
            span *= self.axis_sizes[axis]
        return RankLayout(
            axis_order=tuple(selected),
            axis_sizes={axis: self.axis_sizes[axis] for axis in selected},
            axis_strides=strides,
        )

    def descriptor(self) -> Dict[str, Any]:
        return {
            "axis_order": list(self.axis_order),
            "axis_sizes": dict(self.axis_sizes),
            "axis_strides": dict(self.axis_strides),
            "stage_span": self.num_ranks(),
        }

    @classmethod
    def from_network_layout(
        cls,
        network_layout: Any,
        axis_sizes: Mapping[AxisName, int],
        execution_mode_enforce_cluster: bool,
        *,
        extract_optimize_2dmap: bool = True,
        unsupported_axis_context: str = "AstraSim integration",
        empty_axis_order_is_none: bool = False,
    ) -> Tuple[Optional["RankLayout"], Optional[Dict[str, Any]]]:
        dimensions = getattr(network_layout, "dimensions", None) if network_layout is not None else None
        if not dimensions:
            raise ValueError("Network layout with at least one dimension is required to build a RankLayout.")

        optimize_cfg: Optional[Dict[str, Any]] = None
        if extract_optimize_2dmap:
            for idx, dim in enumerate(dimensions):
                if getattr(dim, "optimize_2dmap", False):
                    if idx != 0:
                        raise ValueError("optimize_2dmap is only supported on the first network dimension.")
                    if optimize_cfg is not None:
                        raise ValueError("Multiple network dimensions requested optimize_2dmap; only one is supported.")
                    topo_type = getattr(dim, "topology_type", None)
                    if not topo_type:
                        raise ValueError("optimize_2dmap requires a topology type on the target dimension.")
                    size_value = getattr(dim, "size", None)
                    if size_value is None:
                        raise ValueError("optimize_2dmap requires an explicit dimension size.")
                    dims_value = getattr(dim, "size_2d", None)
                    if dims_value is not None:
                        dims_value = (int(dims_value[0]), int(dims_value[1]))
                    optimize_cfg = {
                        "dimension_index": idx,
                        "topology": str(topo_type),
                        "size": int(size_value),
                        "parallelisms": tuple(getattr(dim, "parallelisms", ()) or ()),
                    }
                    if dims_value:
                        optimize_cfg["dims"] = dims_value

        if execution_mode_enforce_cluster:
            cluster_axes = sorted(axis for axis in CLUSTER_AXES if axis_sizes[axis] > 1)
            covered: List[str] = []
            for dim in dimensions:
                if sorted(set(covered)) == cluster_axes:
                    break
                if int(getattr(dim, "size", 1)) <= 1:
                    continue
                dim_axes_l = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
                cluster_here = [
                    axis for axis in dim_axes_l if axis in CLUSTER_AXES and axis_sizes[axis] > 1
                ]
                sched_here = [
                    axis for axis in dim_axes_l if axis in PIPELINE_AXES + REPLICA_AXES and axis_sizes[axis] > 1
                ]
                if sched_here:
                    raise ValueError(
                        "For hierarchical/hybrid AstraSim modes, the active TP/CP/EP axes must "
                        "occupy the leading active network dimensions (before any dimension "
                        "carrying active PP/DP) to represent the transformer cluster. "
                        f"Dimension '{getattr(dim, 'label', getattr(dim, 'id', '<unnamed>'))}' carries "
                        f"{sched_here} while the cluster axes {cluster_axes} are not yet fully mapped "
                        f"(covered so far: {sorted(set(covered))})."
                    )
                covered.extend(cluster_here)
            if cluster_axes and sorted(set(covered)) != cluster_axes:
                raise ValueError(
                    "For hierarchical/hybrid AstraSim modes, the leading active network dimensions "
                    f"must jointly carry the active TP/CP/EP axes {cluster_axes} "
                    f"(found only {sorted(set(covered))})."
                )

        axis_order: List[str] = []
        for dim in dimensions:
            dim_axes = [str(axis).strip().lower() for axis in getattr(dim, "parallelisms", ())]
            declared = int(getattr(dim, "size", 1))

            for name in dim_axes:
                if name not in axis_sizes:
                    raise ValueError(
                        f"Unsupported parallelism axis '{name}' in network layout. "
                        f"Supported axes for {unsupported_axis_context} are: tp, cp, ep, pp, dp."
                    )
                if name not in axis_order:
                    axis_order.append(name)

            expected = 1
            for axis_name in dim_axes:
                expected *= axis_sizes.get(axis_name, 1)
            if expected != declared:
                raise ValueError(
                    f"Network dimension '{getattr(dim, 'label', getattr(dim, 'id', '<unnamed>'))}' "
                    f"size mismatch: declared {declared}, but parallelism factors imply {expected}."
                )

        if axis_order:
            ordered_axes = list(CANONICAL_AXES)
            axis_order = [axis for axis in ordered_axes if axis in axis_order]

        if empty_axis_order_is_none and not axis_order:
            return None, optimize_cfg

        for _name in CLUSTER_AXES + PIPELINE_AXES:
            if int(axis_sizes.get(_name, 1)) > 1 and _name not in axis_order:
                raise ValueError(
                    f"Network layout must include {_name!r} when "
                    f"{AXIS_BY_NAME[_name].label} > 1."
                )

        axis_strides: Dict[str, int] = {}
        span = 1
        for axis in axis_order:
            axis_strides[axis] = span
            span *= axis_sizes[axis]

        layout = cls(
            axis_order=tuple(axis_order),
            axis_sizes=dict(axis_sizes),
            axis_strides=axis_strides,
        )
        return layout, optimize_cfg
