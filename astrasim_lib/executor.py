import os
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.setrecursionlimit(100000)

from graphviz import Digraph
import util

from .config_generation import generate_astrasim_configs_from_hw

from . import gmap  # noqa: F401
from .et_utils import (
    chakra_decode,
    chakra_open,
    pb,
)
from .integration import run_cache_astrasim
from .layout_utils import derive_axes_filter, extract_axis_layout
from util import relpath_display

def _env_truthy(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}

def _clean_astrasim_artifacts(directory: str) -> None:
    try:
        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            if entry.startswith("llm_graph.") and entry.endswith(".et"):
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry == "manifest.json":
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry == "comm_groups.json" or entry.startswith("comm_groups_"):
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry.startswith("system_native_collectives"):
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif entry.startswith("network_analytical_") and entry.endswith(".yml"):
                try:
                    os.remove(path)
                except OSError:
                    pass
        workload_dir = os.path.join(directory, "workload")
        if os.path.isdir(workload_dir):
            shutil.rmtree(workload_dir)
    except FileNotFoundError:
        pass

def _attr_to_dict(node: pb.Node) -> Dict[str, Any]:
    attr_map: Dict[str, Any] = {}
    for attr in node.attr:
        name = attr.name or f"attr_{len(attr_map)}"
        which = attr.WhichOneof("value")
        if not which:
            continue
        raw = getattr(attr, which)
        if hasattr(raw, "values"):
            attr_map[name] = list(raw.values)
        else:
            attr_map[name] = raw
    return attr_map

def _visualize_et_files(et_paths: List[str]) -> None:
    if not et_paths:
        return

    et_paths = et_paths[:20]

    def _render_et_file(et_path: str) -> None:
        try:
            fh = chakra_open(et_path)
        except OSError as exc:
            print(f"[WARN] Failed to open {et_path} for visualization: {exc}")
            return

        meta = pb.GlobalMetadata()
        if not chakra_decode(fh, meta):
            print(f"[WARN] {et_path} does not contain GlobalMetadata; skipping graph output")
            fh.close()
            return

        nodes: List[pb.Node] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        fh.close()

        dot = Digraph(comment=os.path.basename(et_path), format="svg",graph_attr={"rankdir": "TB", "fontsize": "10"})

        id_to_node = {int(node.id): node for node in nodes}
        type_color = {
            pb.COMP_NODE: "lightblue",
            pb.COMM_COLL_NODE: "palegreen",
            pb.COMM_SEND_NODE: "khaki",
            pb.COMM_RECV_NODE: "lightsalmon",
        }

        for node in nodes:
            attr_map = _attr_to_dict(node)
            try:
                node_type = pb.NodeType.Name(node.type)
            except ValueError:
                node_type = str(node.type)
            label_lines = [node.name or f"node_{node.id}", f"id={node.id}", node_type]
            if node.duration_micros:
                label_lines.append(f"dur={node.duration_micros}us")
            if "comm_type" in attr_map:
                try:
                    comm_name = pb.CollectiveCommType.Name(int(attr_map["comm_type"]))
                except ValueError:
                    comm_name = str(attr_map["comm_type"])
                label_lines.append(f"comm={comm_name}")
            if "comm_dst" in attr_map:
                label_lines.append(f"dst={attr_map['comm_dst']}")
            if "comm_src" in attr_map:
                label_lines.append(f"src={attr_map['comm_src']}")
            if "comm_tag" in attr_map:
                label_lines.append(f"tag={attr_map['comm_tag']}")
            if "comm_size" in attr_map:
                label_lines.append(f"bytes={attr_map['comm_size']}")

            color = type_color.get(node.type, "white")
            node_id = str(node.id)
            dot.node(node_id, label="\n".join(label_lines), style="filled", fillcolor=color, shape="box")

        for node in nodes:
            for dep in node.ctrl_deps:
                if dep in id_to_node:
                    dot.edge(str(dep), str(node.id))

        dir_name, base_name = os.path.split(et_path)
        viz_base = base_name + ".viz"
        try:
            output_path = dot.render(viz_base, directory=dir_name or None, format="svg", cleanup=True)
            final_svg = et_path + ".svg"
            if output_path and output_path != final_svg:
                try:
                    os.replace(output_path, final_svg)
                except FileNotFoundError:
                    pass
        except Exception as exc:
            print(f"[WARN] Failed to render Graphviz graph for {et_path}: {exc}")

    if not et_paths:
        return

    if len(et_paths) == 1:
        et_path = et_paths[0]
        display_path = relpath_display(f"{et_path}.svg")
        message = f" | ET Graph saved to {display_path}"
        util.graphviz_submit(
            f"et:{os.path.basename(et_path)}",
            _render_et_file,
            et_path,
            print_message=message,
        )
        return

    first_path = et_paths[0]
    display_first = relpath_display(f"{first_path}.svg")
    summary_message = f" | {len(et_paths)} ET graphs saved to {display_first} (..)"
    util.graphviz_submit(
        f"et:{os.path.basename(first_path)}",
        _render_et_file,
        first_path,
        print_message=summary_message,
    )
    for et_path in et_paths[1:]:
        util.graphviz_submit(
            f"et:{os.path.basename(et_path)}",
            _render_et_file,
            et_path,
            print_message=None,
        )

def _dump_et_text(et_paths: List[str]) -> None:
    if not et_paths:
        return

    for et_path in et_paths:
        try:
            fh = chakra_open(et_path)
        except OSError as exc:
            print(f"[WARN] Failed to open {et_path} for text dump: {exc}")
            continue

        meta = pb.GlobalMetadata()
        has_meta = chakra_decode(fh, meta)

        nodes: List[pb.Node] = []
        while True:
            node = pb.Node()
            if not chakra_decode(fh, node):
                break
            nodes.append(node)
        fh.close()

        id_to_node = {int(node.id): node for node in nodes}

        lines: List[str] = []
        lines.append(f"ET file: {os.path.basename(et_path)}")
        if has_meta:
            lines.append(f"GlobalMetadata: version={meta.version}")
        lines.append(f"Nodes: {len(nodes)}")
        lines.append("")

        def fmt_attr_map(node: pb.Node) -> str:
            items: List[str] = []
            for attr in node.attr:
                name = attr.name or "unnamed"
                which = attr.WhichOneof("value")
                if not which:
                    continue
                raw = getattr(attr, which)
                if hasattr(raw, "values"):
                    val = list(raw.values)
                else:
                    val = raw
                items.append(f"{name}={val}")
            return ", ".join(items)

        lines.append("Nodes detail:")
        for node in nodes:
            try:
                node_type = pb.NodeType.Name(node.type)
            except ValueError:
                node_type = str(node.type)
            attr_str = fmt_attr_map(node)
            if node.type == pb.COMP_NODE:
                lines.append(
                    f"- id={node.id} name={node.name} type={node_type} dur_us={node.duration_micros} attrs=[{attr_str}]"
                )
            else:
                lines.append(
                    f"- id={node.id} name={node.name} type={node_type} attrs=[{attr_str}]"
                )
        lines.append("")

        lines.append("Edges (ctrl_deps):")
        for node in nodes:
            if not node.ctrl_deps:
                continue
            deps_str = ", ".join(str(int(d)) for d in node.ctrl_deps if int(d) in id_to_node)
            lines.append(f"- {node.id} <- [{deps_str}]")

        out_path = et_path + ".txt"
        try:
            with open(out_path, "w") as outf:
                outf.write("\n".join(lines) + "\n")

        except OSError as exc:
            print(f"[WARN] Failed to write ET text dump for {et_path}: {exc}")

def run_astra_simulation_only_onepath(
    fwdbwd_root,
    time_calc_obj,
    output_dir: str = "astra_comparison_output",
    persist_artifacts: Optional[bool] = None,
    faulty_links_override: Optional[Sequence[Tuple[int, int, float]]] = None,
    rank_layout: Optional[Dict[str, Any]] = None,
):
    print("\n" + "="*60)
    print("ASTRASIM SIMULATION RESULTS")
    print("="*60)

    persist = persist_artifacts if persist_artifacts is not None else _env_truthy(
        "RAPID_PERSIST_ASTRASIM_ARTIFACTS"
    )
    os.makedirs(output_dir, exist_ok=True)
    work_dir: str
    if persist:
        work_dir = output_dir
        _clean_astrasim_artifacts(work_dir)
    else:
        work_dir = tempfile.mkdtemp(prefix="astrasim_", dir=output_dir)

    try:
        astrasim_start = time.time()

        print(f"[AstraSim] Converting graph...")

        from program.ir import Program as _Program

        if not isinstance(fwdbwd_root, _Program):
            raise TypeError(
                "run_astra_simulation_only_onepath expects a program.ir.Program "
                f"(got {type(fwdbwd_root).__name__}); the legacy-graph entry "
                "died at M6 (docs/rewrite/DESIGN.md §5)."
            )
        from program.et_emit import emit_chakra
        from program.pipeline_collectives import prepare_pipeline_collectives

        dp_count = max(1, int(fwdbwd_root.dp_count))
        prepare_pipeline_collectives(fwdbwd_root, getattr(time_calc_obj, "network_model", None),
                                     time_calc_obj.hw_config)
        bundle = emit_chakra(fwdbwd_root, work_dir)
        fwd_et_prefix, rank_ids, fwd_manifest = (
            bundle.et_prefix,
            bundle.rank_ids,
            bundle.manifest_path,
        )
        print(
            f"[AstraSim] Generated ET files for ranks: {fwd_et_prefix}.{{0..{len(rank_ids)-1}}}.et"
        )
        print(f"[AstraSim] Wrote graph manifest to {fwd_manifest}")
        rank_count = len(rank_ids)

        synthetic_pair = False
        if rank_count == 1:
            src = os.path.join(work_dir, "llm_graph.0.et")
            dst = os.path.join(work_dir, "llm_graph.1.et")
            shutil.copy(src, dst)
            rank_ids = [rank_ids[0], rank_ids[0]+1]
            synthetic_pair = True
        rank_count = len(rank_ids)

        if persist and _env_truthy("RAPID_PERSIST_ARTIFACT_VIZ"):
            et_paths = []
            for rank in rank_ids:
                et_path = os.path.join(work_dir, f"llm_graph.{rank}.et")
                if os.path.exists(et_path):
                    et_paths.append(et_path)

            if et_paths:
                print(f"[AstraSim] Visualizing {len(et_paths)} persisted ET files...")
                _visualize_et_files(et_paths)
                _dump_et_text(et_paths)

        print(f"[AstraSim] Generating configuration files...")

        comm_groups_path = os.path.join(work_dir, "comm_groups.json")
        if not os.path.exists(comm_groups_path):
            comm_groups_path = None
        if os.environ.get("RAPID_ASTRA_SKIP_EXEC"):
            print("[AstraSim] RAPID_ASTRA_SKIP_EXEC set. Exiting after ET artifact generation.")
            exit()

        if rank_layout is None and fwdbwd_root.layout.axis_order:
            rank_layout = fwdbwd_root.layout.descriptor()
        axis_order, axis_sizes, _ = extract_axis_layout(rank_layout)
        preferred_axes_for_synthetic = tuple(axis_order) if axis_order else tuple()
        axes_filter = derive_axes_filter(axis_order, axis_sizes, dp_count)
        if not axes_filter:
            axes_filter = None
        if synthetic_pair:
            axes_filter = ["synthetic2"]
        astra_configs = generate_astrasim_configs_from_hw(
            time_calc_obj.hw_config,
            work_dir,
            rank_count,
            axes_filter=axes_filter,
            faulty_links_override=faulty_links_override,
            preferred_axes_for_synthetic=preferred_axes_for_synthetic,

            zero_moe_coordinate_routing=bundle.zero_moe_operation_groups is not None,
            nccl_latency_program=fwdbwd_root,
        )

        print(f"[AstraSim] Executing forward simulation with {rank_count} ranks...")
        cache_override_env = os.environ.pop("ASTRA_CACHE_DIR", None)
        local_cache_dir = work_dir
        local_cache_path = os.path.join(work_dir, "cache.json")

        result_record_path = os.path.join(work_dir, "astra_runs.json") if persist else None
        try:
            run_scope = str(getattr(fwdbwd_root.meta, "label", "") or "graph")
            fwd_result = run_cache_astrasim(
                time_calc_obj.hw_config,
                comm="graph",
                npus_count=rank_count,
                size_bytes=0,
                astra_config_dir=local_cache_dir,
                cache_path=local_cache_path,
                manifest_json_path=fwd_manifest,
                workload_prefix=fwd_et_prefix,
                comm_group_json=comm_groups_path,
                axes_filter=axes_filter,
                files=astra_configs,
                result_record_path=result_record_path,
                link_stats_run_scope=run_scope,
            )

            if bundle.zero_moe_operation_groups is not None and isinstance(fwd_result.link_stats, dict):
                fwd_result.link_stats["zero_moe_operation_groups"] = bundle.zero_moe_operation_groups
            fwd_times = fwd_result.per_node_sec
            fwd_total = fwd_result.max_sec
        finally:
            if cache_override_env is not None:
                os.environ["ASTRA_CACHE_DIR"] = cache_override_env

        conversion_and_sim_time = time.time() - astrasim_start

        if len(fwd_times) > 5:
            printstr = ""
            for i in range(5):
                rev_times = list(reversed(fwd_times))
                printstr += f" {round(rev_times[i],2)},"
            printstr += "...."
            print(f"[AstraSim] Times per node:{printstr}")
        else:
            print(f"[AstraSim] Times per node: {fwd_times}")
        print(f"[AstraSim] Total execution time: {fwd_total:.6f} seconds")
        print(f"[AstraSim] Simulation duration: {conversion_and_sim_time:.3f} seconds")

        print("="*60)

        return fwd_result

    except Exception as e:
        print(f"[AstraSim] ERROR: Failed to run simulation: {e}")
        print("="*60)
        raise
    finally:
        if not persist:
            shutil.rmtree(work_dir, ignore_errors=True)
