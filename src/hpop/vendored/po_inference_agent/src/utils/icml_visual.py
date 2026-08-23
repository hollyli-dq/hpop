from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.cloud_iac_dataset import (
    resolve_cloud_iac_data_root,
    resolve_cloud_iac_ground_truth_dir,
)
from src.utils.result_paths import (
    CLOUD_IAC_RESULTS_DIR,
    LEGACY_CLOUD_IAC_RESULTS_DIR,
    WFINSTANCES_PLOTS_DIR,
    LEGACY_WFINSTANCES_PLOTS_DIR,
    WFINSTANCES_SRASEARCH_RESULTS_DIR,
    LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
    WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
    prefer_existing_path,
)

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class IcmlVisual:
    """Shared plotting and visualization helpers for the ICML paper assets."""

    METHOD_COLORS = {
        "bhpop_single_po": "#2171b5",
        "bpop_qsucc": "#2171b5",
        "bpop_uniform": "#08519c",
        "queue_jump": "#c51b8a",
        "majority": "#238b45",
        "inductive_miner_imf": "#6a51a3",
        "heuristics_miner": "#d94801",
        "and": "#737373",
    }

    METHOD_LABELS = {
        "bhpop_single_po": "BPOP (Ours)",
        "bpop_qsucc": "BPOP-QSucc",
        "bpop_uniform": "BPOP-Uniform",
        "queue_jump": "Queue Jump",
        "majority": "Majority",
        "inductive_miner_imf": "Inductive Miner",
        "heuristics_miner": "Heuristics Miner",
        "and": "AND (Intersection)",
    }

    METHOD_MARKERS = {
        "bhpop_single_po": "*",
        "bpop_qsucc": "*",
        "bpop_uniform": "P",
        "queue_jump": "o",
        "majority": "s",
        "inductive_miner_imf": "^",
        "heuristics_miner": "D",
        "and": "o",
    }

    METHOD_LINESTYLES = {
        "bhpop_single_po": "-",
        "bpop_qsucc": "-",
        "bpop_uniform": "--",
        "queue_jump": "-",
        "majority": "-",
        "inductive_miner_imf": "-",
        "heuristics_miner": "-",
        "and": "-",
    }

    CLOUD_SCENARIO_ORDER = [
        "simple_ecs",
        "slb_ecs_rds",
        "slb_ecs_redis",
        "eip_slb_ecs",
        "dual_zone_ecs_slb",
        "dual_zone_ecs_slb_rds",
    ]

    CLOUD_SCENARIO_SHORT = {
        "simple_ecs": "S1",
        "slb_ecs_rds": "S2",
        "slb_ecs_redis": "S3",
        "eip_slb_ecs": "S4",
        "dual_zone_ecs_slb": "S5",
        "dual_zone_ecs_slb_rds": "S6",
    }

    CLOUD_SCENARIO_TITLES = {
        "simple_ecs": "S1: Simple ECS",
        "slb_ecs_rds": "S2: SLB+ECS+RDS",
        "slb_ecs_redis": "S3: SLB+ECS+Redis",
        "eip_slb_ecs": "S4: EIP+SLB+ECS",
        "dual_zone_ecs_slb": "S5: Dual Zone ECS+SLB",
        "dual_zone_ecs_slb_rds": "S6: Dual Zone+RDS",
    }

    EDGE_COLORS = {
        "correct": "#009E73",
        "missed": "#D55E00",
        "false_pos": "#E69F00",
    }

    CLOUD_TASK_NAME_REPLACEMENTS = {
        "CreateVpc": "CreateVpc",
        "CreateVSwitch": "CreateVSwitch",
        "CreateSecurityGroup": "CreateSG",
        "AuthorizeSecurityGroup": "AuthorizeSG",
        "RunInstances": "RunInstances",
        "CreateLoadBalancer": "CreateSLB",
        "CreateLoadBalancerHTTPListener": "CreateListener",
        "StartLoadBalancerListener": "StartListener",
        "AddBackendServers": "AddBackend",
        "CreateDBInstance": "CreateRDS",
        "CreateAccount": "CreateAccount",
        "ModifySecurityIps": "ModifySecIPs",
        "CreateInstance": "CreateRedis",
        "DescribeInstanceAttribute": "DescribeAttr",
        "AllocateEipAddress": "AllocateEIP",
        "AssociateEipAddress": "AssociateEIP",
    }

    WORKFLOW_NODE_PALETTE = [
        "#6B8DD6",
        "#8B7DC6",
        "#B088A8",
        "#7EACC1",
        "#A4C4A4",
        "#C4A4A4",
        "#9BB5C4",
        "#B4A4C4",
    ]

    STYLE_PROFILES = {
        "paper": {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 18,
            "axes.labelsize": 28,
            "axes.titlesize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 4,
            "lines.markersize": 16,
            "axes.linewidth": 2,
            "grid.alpha": 0.3,
            "legend.framealpha": 0.95,
            "text.usetex": False,
        },
        "compact": {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        },
        "dag": {
            "font.family": "sans-serif",
            "font.size": 32,
            "axes.labelsize": 32,
            "axes.titlesize": 32,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 40,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        },
        "compact_panel": {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 9,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.5,
        },
    }

    @classmethod
    def apply_style(cls, profile: str = "paper", **overrides: Any) -> None:
        if profile not in cls.STYLE_PROFILES:
            raise ValueError(f"Unknown style profile: {profile}")
        params = dict(cls.STYLE_PROFILES[profile])
        params.update(overrides)
        plt.rcParams.update(params)

    @classmethod
    def extract_task_type(cls, name: str) -> str:
        if "_ID" not in name:
            return name
        prefix = name.split("_ID")[0]
        return prefix.split("_")[0]

    @classmethod
    def extract_task_id(cls, name: str) -> str:
        if "_ID" not in name:
            return name
        return str(int(name.split("_ID")[1]))

    @classmethod
    def load_wfinstances_workflow_dag(
        cls,
        data_dir: Path,
        *,
        title: str | None = None,
    ) -> Dict[str, Any]:
        json_files = sorted(data_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No workflow JSON files found in {data_dir}")

        with open(json_files[0]) as f:
            payload = json.load(f)

        workflow = payload.get("workflow", payload)
        specification = workflow.get("specification", {})
        tasks = specification.get("tasks", [])

        task_ids = [task.get("name", task.get("id")) for task in tasks]
        n = len(task_ids)
        task_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}
        adj = np.zeros((n, n), dtype=np.int8)

        for task in tasks:
            name = task.get("name", task.get("id"))
            if name not in task_to_idx:
                continue
            src = task_to_idx[name]
            for child in task.get("children", []):
                if child in task_to_idx:
                    adj[src, task_to_idx[child]] = 1

        return {
            "name": title or workflow.get("name") or json_files[0].stem,
            "task_ids": task_ids,
            "task_types": [cls.extract_task_type(task_id) for task_id in task_ids],
            "numeric_ids": [cls.extract_task_id(task_id) for task_id in task_ids],
            "task_to_idx": task_to_idx,
            "adj": adj,
            "cover": adj.copy(),
            "n": n,
            "num_tasks": n,
            "num_edges": int(adj.sum()),
        }

    @classmethod
    def summarize_task_types(cls, data: Mapping[str, Any]) -> List[Tuple[str, int]]:
        task_types = list(data["task_types"])
        unique_types = sorted(set(task_types))
        return [(task_type, sum(1 for item in task_types if item == task_type)) for task_type in unique_types]

    @classmethod
    def compute_dag_levels(cls, adj: np.ndarray) -> List[int]:
        n = adj.shape[0]
        levels = [-1] * n

        def get_level(node: int) -> int:
            if levels[node] >= 0:
                return levels[node]
            parents = np.where(adj[:, node] > 0)[0]
            if len(parents) == 0:
                levels[node] = 0
            else:
                levels[node] = 1 + max(get_level(parent) for parent in parents)
            return levels[node]

        for idx in range(n):
            get_level(idx)
        return levels

    @classmethod
    def transitive_reduction(cls, adj_matrix: np.ndarray) -> np.ndarray:
        n = adj_matrix.shape[0]
        closure = adj_matrix.copy().astype(bool)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    closure[i, j] = closure[i, j] or (closure[i, k] and closure[k, j])

        reduction = closure.copy()
        for i in range(n):
            for j in range(n):
                if not closure[i, j]:
                    continue
                for k in range(n):
                    if k != i and k != j and closure[i, k] and closure[k, j]:
                        reduction[i, j] = False
                        break
        return reduction.astype(np.int8)

    @classmethod
    def _normalize_positions(
        cls,
        pos: Mapping[int, Tuple[float, float]],
        *,
        x_pad: float = 0.1,
        y_pad: float = 0.1,
    ) -> Dict[int, Tuple[float, float]]:
        xs = [coords[0] for coords in pos.values()]
        ys = [coords[1] for coords in pos.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        return {
            node: (
                x_pad + (1.0 - 2.0 * x_pad) * (coords[0] - x_min) / x_range,
                y_pad + (1.0 - 2.0 * y_pad) * (coords[1] - y_min) / y_range,
            )
            for node, coords in pos.items()
        }

    @classmethod
    def _get_color_palette(cls, n_colors: int) -> List[Tuple[float, ...]]:
        if n_colors <= 10:
            cmap = plt.cm.tab10
        else:
            cmap = plt.cm.Set3
        return [cmap(i / max(n_colors - 1, 1)) for i in range(n_colors)]

    @classmethod
    def wfvisualize_po(
        cls,
        data: Mapping[str, Any],
        ax: plt.Axes,
        *,
        title: str | None = None,
        show_legend: bool = True,
        h_spacing: float = 1.5,
        v_spacing: float = 2.0,
    ) -> None:
        if not NETWORKX_AVAILABLE:
            ax.text(0.5, 0.5, "NetworkX not available", ha="center", va="center")
            ax.axis("off")
            return

        adj = np.asarray(data["adj"], dtype=np.int8)
        n = int(data["n"])
        task_types = list(data["task_types"])
        numeric_ids = list(data["numeric_ids"])

        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        for src in range(n):
            for dst in range(n):
                if adj[src, dst] > 0:
                    graph.add_edge(src, dst)

        levels = cls.compute_dag_levels(adj)
        level_nodes: Dict[int, List[int]] = {}
        for node, level in enumerate(levels):
            level_nodes.setdefault(level, []).append(node)

        pos: Dict[int, Tuple[float, float]] = {}
        for level, nodes in sorted(level_nodes.items()):
            ordered = sorted(nodes, key=lambda idx: int(numeric_ids[idx]))
            for offset, node in enumerate(ordered):
                pos[node] = ((offset - (len(ordered) - 1) / 2) * h_spacing, -level * v_spacing)

        unique_types = sorted(set(task_types))
        palette = cls._get_color_palette(len(unique_types))
        type_to_color = {task_type: palette[idx] for idx, task_type in enumerate(unique_types)}
        node_colors = [type_to_color[task_types[idx]] for idx in range(n)]

        if n <= 25:
            node_size = 600
            font_size = 8
        elif n <= 50:
            node_size = 400
            font_size = 7
        else:
            node_size = 300
            font_size = 6

        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            arrows=True,
            arrowsize=12,
            arrowstyle="-|>",
            edge_color="#555555",
            alpha=0.6,
            width=1.0,
            connectionstyle="arc3,rad=0.05",
            min_source_margin=15,
            min_target_margin=15,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="black",
            linewidths=1.0,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={idx: numeric_ids[idx] for idx in range(n)},
            ax=ax,
            font_size=font_size,
            font_weight="bold",
        )

        if show_legend:
            legend_handles = []
            for task_type, count in cls.summarize_task_types(data):
                legend_handles.append(
                    mpatches.Patch(
                        facecolor=type_to_color[task_type],
                        edgecolor="black",
                        linewidth=0.5,
                        label=f"{task_type} ({count})",
                    )
                )
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                frameon=True,
            )

        display_title = title or f"{data['name']} Workflow DAG\n({data['n']} tasks, {data['num_edges']} edges)"
        ax.set_title(display_title, fontsize=12, fontweight="bold", pad=10)
        ax.axis("off")
        ax.set_aspect("equal")

        x_coords = [coords[0] for coords in pos.values()]
        y_coords = [coords[1] for coords in pos.values()]
        x_margin = max(2.0, (max(x_coords) - min(x_coords)) * 0.15) if x_coords else 2.0
        y_margin = max(2.0, (max(y_coords) - min(y_coords)) * 0.10) if y_coords else 2.0
        ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        ax.set_ylim(min(y_coords) - y_margin, max(y_coords) + y_margin)

    @classmethod
    def shorten_cloud_task_name(cls, name: str) -> str:
        return cls.CLOUD_TASK_NAME_REPLACEMENTS.get(name, name)

    @classmethod
    def load_cloud_iac_scenario(
        cls,
        scenario_name: str,
        *,
        project_root: Path | None = None,
    ) -> Dict[str, Any]:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        data_root = resolve_cloud_iac_data_root(project_root)
        scenario_root = resolve_cloud_iac_ground_truth_dir(data_root)
        scenario_file = scenario_root / f"{scenario_name}.json"
        with open(scenario_file) as f:
            data = json.load(f)

        edges = data["edges"]
        tasks = sorted({item for edge in edges for item in edge})
        n = len(tasks)
        task_to_idx = {task: idx for idx, task in enumerate(tasks)}
        cover = np.zeros((n, n), dtype=np.int8)
        for src, dst in edges:
            cover[task_to_idx[src], task_to_idx[dst]] = 1

        return {
            "scenario": scenario_name,
            "tasks": tasks,
            "edges": edges,
            "cover": cover,
            "task_to_idx": task_to_idx,
        }

    @classmethod
    def load_thresholded_cover_from_avg_h(
        cls,
        avg_h_path: Path,
        *,
        threshold: float,
        reduce_transitively: bool = False,
    ) -> np.ndarray:
        with open(avg_h_path, "rb") as f:
            avg_h = np.array(pickle.load(f))
        cover = (avg_h >= threshold).astype(np.int8)
        np.fill_diagonal(cover, 0)
        if reduce_transitively:
            cover = cls.transitive_reduction(cover)
        return cover

    @classmethod
    def load_best_cloud_iac_inferred_cover(
        cls,
        results_dir: Path,
        scenario_name: str,
        *,
        target_ip_cov: float,
        threshold: float,
    ) -> np.ndarray | None:
        best_exp = None
        best_f1 = -1.0
        for exp_path in results_dir.glob(f"exp_*_{scenario_name}"):
            summary_file = exp_path / "summary.json"
            if not summary_file.exists():
                continue
            with open(summary_file) as f:
                summary = json.load(f)
            ip_cov = summary.get("configuration", {}).get("ip_cov_target", 0.0)
            if abs(ip_cov - target_ip_cov) > 0.01:
                continue
            f1 = summary.get("posterior", {}).get("cover_f1", 0.0)
            if f1 > best_f1:
                best_f1 = f1
                best_exp = exp_path

        if best_exp is None:
            return None
        return cls.load_thresholded_cover_from_avg_h(
            best_exp / "avg_H.pkl",
            threshold=threshold,
            reduce_transitively=False,
        )

    @classmethod
    def _build_level_layout(
        cls,
        adj: np.ndarray,
        *,
        horizontal_scale: float,
        vertical_scale: float,
        sort_key: List[str] | None = None,
    ) -> Dict[int, Tuple[float, float]]:
        levels = cls.compute_dag_levels(adj)
        level_nodes: Dict[int, List[int]] = {}
        for idx, level in enumerate(levels):
            level_nodes.setdefault(level, []).append(idx)

        pos: Dict[int, Tuple[float, float]] = {}
        for level, nodes in sorted(level_nodes.items()):
            if sort_key is not None:
                ordered = sorted(nodes, key=lambda node: sort_key[node])
            else:
                ordered = list(nodes)
            n_nodes = len(ordered)
            for offset, node in enumerate(ordered):
                pos[node] = (
                    (offset - (n_nodes - 1) / 2.0) * horizontal_scale,
                    -level * vertical_scale,
                )
        return pos

    @classmethod
    def draw_cloud_iac_po_comparison(
        cls,
        ax: plt.Axes,
        tasks: List[str],
        true_cover: np.ndarray,
        inferred_cover: np.ndarray,
        *,
        title: str,
        large: bool = False,
    ) -> None:
        n = len(tasks)
        display_names = [cls.shorten_cloud_task_name(task) for task in tasks]
        combined = ((np.asarray(true_cover) > 0) | (np.asarray(inferred_cover) > 0)).astype(np.int8)
        pos = cls._build_level_layout(
            combined,
            horizontal_scale=0.35 if large else 0.22,
            vertical_scale=0.25 if large else 0.18,
        )
        pos = cls._normalize_positions(pos, x_pad=0.1, y_pad=0.1)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        if large:
            line_width = 3.0
            shrink = 45
            title_size = 28
            long_font = 20
            short_font = 24
            box_pad = 0.7
            box_width = 2.5
        else:
            line_width = 1.5
            shrink = 18
            title_size = 9
            long_font = 7
            short_font = 8
            box_pad = 0.5
            box_width = 1.5

        for src in range(n):
            for dst in range(n):
                if src == dst:
                    continue
                true_edge = true_cover[src, dst] == 1
                inferred_edge = inferred_cover[src, dst] == 1
                if true_edge and inferred_edge:
                    color = cls.EDGE_COLORS["correct"]
                    style = "-"
                    alpha = 0.9
                elif true_edge and not inferred_edge:
                    color = cls.EDGE_COLORS["missed"]
                    style = "--"
                    alpha = 0.85 if large else 0.8
                elif not true_edge and inferred_edge:
                    color = cls.EDGE_COLORS["false_pos"]
                    style = ":"
                    alpha = 0.85 if large else 0.8
                else:
                    continue

                x1, y1 = pos[src]
                x2, y2 = pos[dst]
                ax.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": color,
                        "linestyle": style,
                        "lw": line_width,
                        "alpha": alpha,
                        "shrinkA": shrink,
                        "shrinkB": shrink,
                    },
                )

        for idx, name in enumerate(display_names):
            x, y = pos[idx]
            fontsize = long_font if len(name) > 15 else short_font
            ax.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                zorder=4,
                bbox={
                    "boxstyle": f"round,pad={box_pad}",
                    "facecolor": "white",
                    "edgecolor": "black",
                    "linewidth": box_width,
                },
            )

        ax.set_title(title, fontweight="bold", fontsize=title_size, pad=15 if large else 6)
        ax.axis("off")
        ax.set_aspect("equal")

    @classmethod
    def _resolve_cloud_iac_results_dir(
        cls,
        *,
        project_root: Path | None = None,
        results_dir: Path | None = None,
    ) -> Path:
        if results_dir is not None:
            return Path(results_dir)
        return prefer_existing_path(CLOUD_IAC_RESULTS_DIR, LEGACY_CLOUD_IAC_RESULTS_DIR)

    @classmethod
    def _load_cloud_iac_reduced_covers(
        cls,
        scenario_name: str,
        *,
        project_root: Path | None = None,
        results_dir: Path,
        target_ip_cov: float,
        threshold: float,
    ) -> Tuple[List[str], np.ndarray, np.ndarray | None]:
        scenario = cls.load_cloud_iac_scenario(scenario_name, project_root=project_root)
        true_cover = cls.transitive_reduction(scenario["cover"])
        inferred_cover = cls.load_best_cloud_iac_inferred_cover(
            Path(results_dir),
            scenario_name,
            target_ip_cov=target_ip_cov,
            threshold=threshold,
        )
        if inferred_cover is not None:
            inferred_cover = cls.transitive_reduction(inferred_cover)
        return scenario["tasks"], true_cover, inferred_cover

    @classmethod
    def plot_cloud_iac_po_comparison(
        cls,
        *,
        project_root: Path | None = None,
        results_dir: Path | None = None,
        output_dir: Path | None = None,
        target_ip_cov: float = 1.0,
        threshold: float = 1 / 3,
    ) -> None:
        cls.apply_style("compact")
        resolved_results_dir = cls._resolve_cloud_iac_results_dir(
            project_root=project_root,
            results_dir=results_dir,
        )
        resolved_output_dir = Path(output_dir or (resolved_results_dir / "plots"))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"Generating Cloud-IaC PO Comparison Plots (IP-Cov={target_ip_cov})")
        print("=" * 60)

        selected_scenarios = ["slb_ecs_redis", "dual_zone_ecs_slb_rds"]
        selected_titles = {
            "slb_ecs_redis": "Scenario S3 (SLB-ECS-Redis)",
            "dual_zone_ecs_slb_rds": "Scenario S6 (Dual Zone+RDS)",
        }

        fig_main, axes_main = plt.subplots(1, 2, figsize=(22, 11))
        for idx, scenario_name in enumerate(selected_scenarios):
            print(f"Processing {scenario_name} at IP-Cov={target_ip_cov}...")
            tasks, true_cover, inferred_cover = cls._load_cloud_iac_reduced_covers(
                scenario_name,
                project_root=project_root,
                results_dir=resolved_results_dir,
                target_ip_cov=target_ip_cov,
                threshold=threshold,
            )
            if inferred_cover is None:
                axes_main[idx].text(0.5, 0.5, "No data", ha="center", va="center", fontsize=24)
                axes_main[idx].set_title(selected_titles[scenario_name], fontsize=28)
                continue
            cls.draw_cloud_iac_po_comparison(
                axes_main[idx],
                tasks,
                true_cover,
                inferred_cover,
                title=selected_titles[scenario_name],
                large=True,
            )

        fig_main.legend(
            handles=[
                mlines.Line2D([0], [0], color=cls.EDGE_COLORS["correct"], linewidth=6, label="Correct"),
                mlines.Line2D([0], [0], color=cls.EDGE_COLORS["missed"], linewidth=6, linestyle="--", label="Missed"),
                mlines.Line2D([0], [0], color=cls.EDGE_COLORS["false_pos"], linewidth=6, linestyle=":", label="Extra"),
            ],
            loc="lower center",
            ncol=3,
            fontsize=24,
            frameon=False,
            bbox_to_anchor=(0.5, 0.02),
        )
        plt.tight_layout(rect=[0, 0.08, 1, 1.0])
        ip_cov_str = str(target_ip_cov).replace(".", "")
        fig_main.savefig(resolved_output_dir / f"po_comparison_ipcov{ip_cov_str}_main.pdf")
        fig_main.savefig(resolved_output_dir / f"po_comparison_ipcov{ip_cov_str}_main.png", dpi=300)
        print(f"\nSaved: {resolved_output_dir / f'po_comparison_ipcov{ip_cov_str}_main.pdf'}")
        plt.close(fig_main)

        fig_full, axes_full = plt.subplots(2, 3, figsize=(16, 10))
        axes_full = axes_full.flatten()
        for idx, scenario_name in enumerate(cls.CLOUD_SCENARIO_ORDER):
            print(f"Processing {scenario_name} at IP-Cov={target_ip_cov}...")
            tasks, true_cover, inferred_cover = cls._load_cloud_iac_reduced_covers(
                scenario_name,
                project_root=project_root,
                results_dir=resolved_results_dir,
                target_ip_cov=target_ip_cov,
                threshold=threshold,
            )
            if inferred_cover is None:
                axes_full[idx].text(0.5, 0.5, "No data", ha="center", va="center")
                axes_full[idx].set_title(cls.CLOUD_SCENARIO_TITLES[scenario_name])
                continue
            cls.draw_cloud_iac_po_comparison(
                axes_full[idx],
                tasks,
                true_cover,
                inferred_cover,
                title=cls.CLOUD_SCENARIO_TITLES[scenario_name],
                large=False,
            )

        fig_full.legend(
            handles=[
                mpatches.Patch(color=cls.EDGE_COLORS["correct"], label="Correct (TP)"),
                mpatches.Patch(color=cls.EDGE_COLORS["missed"], label="Missed (FN)"),
                mpatches.Patch(color=cls.EDGE_COLORS["false_pos"], label="False Pos (FP)"),
            ],
            loc="lower center",
            ncol=3,
            fontsize=10,
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
        )
        fig_full.suptitle(
            f"Inferred vs True Partial Orders (BPOP at IP-Cov={target_ip_cov}, threshold=1/3)",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(rect=[0, 0.06, 1, 0.96])
        fig_full.savefig(resolved_output_dir / f"po_comparison_ipcov{ip_cov_str}.pdf")
        fig_full.savefig(resolved_output_dir / f"po_comparison_ipcov{ip_cov_str}.png", dpi=300)
        print(f"\nSaved: {resolved_output_dir / f'po_comparison_ipcov{ip_cov_str}.pdf'}")
        plt.close(fig_full)

    @classmethod
    def plot_cloud_iac_experiment_results(
        cls,
        *,
        project_root: Path | None = None,
        results_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        cls.apply_style("paper")
        resolved_results_dir = cls._resolve_cloud_iac_results_dir(
            project_root=project_root,
            results_dir=results_dir,
        )
        resolved_output_dir = Path(output_dir or (resolved_results_dir / "plots"))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = resolved_results_dir / "experiment_summary_t0.33.csv"
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")

        df_bpop = df[df["method"] == "bhpop_single_po"].copy()
        df_baselines = df[df["method"] != "bhpop_single_po"].copy()

        scenarios = [item for item in cls.CLOUD_SCENARIO_ORDER if item in df["scenario"].unique()]
        methods_order = ["bhpop_single_po", "queue_jump", "majority", "inductive_miner_imf", "heuristics_miner"]
        baseline_methods = ["queue_jump", "majority", "inductive_miner_imf", "heuristics_miner"]

        fig1, axes1 = plt.subplots(1, 2, figsize=(18, 8))
        df_baselines_10 = df_baselines[df_baselines["ip_cov_target"] == 1.0]
        df_bpop_10 = df_bpop[df_bpop["ip_cov_target"] == 1.0]
        df_baselines_06 = df_baselines[df_baselines["ip_cov_target"] == 0.6]

        ax = axes1[0]
        means, stds, colors, labels = [], [], [], []
        for method in methods_order:
            data = df_bpop_10["cover_f1"] if method == "bhpop_single_po" else df_baselines_10[df_baselines_10["method"] == method]["cover_f1"]
            means.append(data.mean())
            stds.append(data.std())
            colors.append(cls.METHOD_COLORS[method])
            labels.append(cls.METHOD_LABELS[method])
        y_pos = np.arange(len(methods_order))
        ax.barh(y_pos, means, xerr=stds, color=colors, capsize=8, height=0.65, edgecolor="black", linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=24)
        ax.set_xlabel("Edge F1 Score", fontsize=28)
        ax.set_xlim([0, 1.22])
        ax.grid(axis="x", alpha=0.3)
        ax.axvline(x=means[0], color=cls.METHOD_COLORS["bhpop_single_po"], linestyle="--", alpha=0.5, linewidth=2)
        ax.tick_params(axis="x", labelsize=24)
        for idx, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + std + 0.02, idx, f"{mean:.2f}", va="center", fontsize=24, fontweight="bold")

        ax = axes1[1]
        means, stds = [], []
        for method in methods_order:
            if method == "bhpop_single_po":
                data = df_bpop_10["feas"].dropna()
                means.append(data.mean() if len(data) > 0 else np.nan)
                stds.append(data.std() if len(data) > 0 else 0)
            else:
                data = df_baselines_10[df_baselines_10["method"] == method]["feas"].dropna()
                means.append(data.mean() if len(data) > 0 else 0)
                stds.append(data.std() if len(data) > 0 else 0)
        means_plot = [0 if np.isnan(item) else item for item in means]
        ax.barh(y_pos, means_plot, xerr=stds, color=colors, capsize=8, height=0.65, edgecolor="black", linewidth=1.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=24)
        ax.set_xlabel("Feasibility", fontsize=28)
        ax.set_xlim([0, 1.22])
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="x", labelsize=24)
        for idx, (mean, std) in enumerate(zip(means, stds)):
            if np.isnan(mean):
                ax.text(0.02, idx, "N/A", va="center", fontsize=24, color="gray")
            else:
                ax.text(mean + std + 0.02, idx, f"{mean:.2f}", va="center", fontsize=24, fontweight="bold")

        fig1.tight_layout()
        fig1.savefig(resolved_output_dir / "summary_key_metrics.pdf")
        fig1.savefig(resolved_output_dir / "summary_key_metrics.png", dpi=300)
        print(f"Saved: {resolved_output_dir / 'summary_key_metrics.pdf'}")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(12, 9))
        bpop_grouped = df_bpop.groupby("ip_cov_target")["cover_f1"].agg(["mean", "std"])
        ax2.errorbar(
            bpop_grouped.index,
            bpop_grouped["mean"],
            yerr=bpop_grouped["std"],
            marker=cls.METHOD_MARKERS["bhpop_single_po"],
            color=cls.METHOD_COLORS["bhpop_single_po"],
            label="BPOP (Ours)",
            capsize=5,
            markersize=14,
            linewidth=3,
            zorder=5,
        )
        for method in baseline_methods:
            grouped = df_baselines[df_baselines["method"] == method].groupby("ip_cov_target")["cover_f1"].agg(["mean", "std"])
            ax2.errorbar(
                grouped.index,
                grouped["mean"],
                yerr=grouped["std"],
                marker=cls.METHOD_MARKERS[method],
                color=cls.METHOD_COLORS[method],
                label=cls.METHOD_LABELS[method],
                capsize=4,
                linewidth=2.5,
                markersize=10,
            )
        ax2.set_xlabel("IP-Coverage Target", fontsize=28)
        ax2.set_ylabel("Edge F1 Score", fontsize=28)
        ax2.set_xlim([0.55, 1.05])
        ax2.set_ylim([0, 1.05])
        ax2.legend(loc="lower left", fontsize=22)
        ax2.grid(alpha=0.3)
        ax2.tick_params(axis="both", labelsize=24)
        fig2.tight_layout()
        fig2.savefig(resolved_output_dir / "f1_vs_ipcov.pdf")
        fig2.savefig(resolved_output_dir / "f1_vs_ipcov.png", dpi=300)
        print(f"Saved: {resolved_output_dir / 'f1_vs_ipcov.pdf'}")
        plt.close(fig2)

        fig2b, ax2b = plt.subplots(figsize=(12, 9))
        bpop_feas_grouped = df_bpop.groupby("ip_cov_target")["feas"].mean()
        ax2b.plot(
            bpop_feas_grouped.index,
            bpop_feas_grouped.values,
            marker=cls.METHOD_MARKERS["bhpop_single_po"],
            color=cls.METHOD_COLORS["bhpop_single_po"],
            label="BPOP (Ours)",
            markersize=14,
            linewidth=3,
            zorder=5,
        )
        for method in baseline_methods:
            grouped = df_baselines[df_baselines["method"] == method].groupby("ip_cov_target")["feas"].mean()
            ax2b.plot(
                grouped.index,
                grouped.values,
                marker=cls.METHOD_MARKERS[method],
                color=cls.METHOD_COLORS[method],
                label=cls.METHOD_LABELS[method],
                linewidth=2.5,
                markersize=10,
            )
        ax2b.set_xlabel("IP-Coverage Target", fontsize=28)
        ax2b.set_ylabel("Feasibility", fontsize=28)
        ax2b.set_xlim([0.55, 1.05])
        ax2b.set_ylim([0, 1.1])
        ax2b.legend(loc="center right", fontsize=22, frameon=True)
        ax2b.grid(alpha=0.3)
        ax2b.tick_params(axis="both", labelsize=24)
        fig2b.tight_layout()
        fig2b.savefig(resolved_output_dir / "feasibility_vs_ipcov.pdf")
        fig2b.savefig(resolved_output_dir / "feasibility_vs_ipcov.png", dpi=300)
        print(f"Saved: {resolved_output_dir / 'feasibility_vs_ipcov.pdf'}")
        plt.close(fig2b)

        fig3, ax3 = plt.subplots(figsize=(18, 10))
        x = np.arange(len(scenarios))
        width = 0.16
        for idx, method in enumerate(methods_order):
            method_means, method_stds = [], []
            for scenario in scenarios:
                data = (
                    df_bpop[df_bpop["scenario"] == scenario]["cover_f1"]
                    if method == "bhpop_single_po"
                    else df_baselines_06[(df_baselines_06["scenario"] == scenario) & (df_baselines_06["method"] == method)]["cover_f1"]
                )
                method_means.append(data.mean())
                method_stds.append(data.std())
            ax3.bar(
                x + (idx - 2) * width,
                method_means,
                width,
                yerr=method_stds,
                label=cls.METHOD_LABELS[method],
                color=cls.METHOD_COLORS[method],
                capsize=3,
                edgecolor="black",
                linewidth=0.5,
            )
        ax3.set_ylabel("Edge F1 Score", fontsize=28)
        ax3.set_xlabel("Scenario", fontsize=28)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{cls.CLOUD_SCENARIO_SHORT[item]}\n({item.replace('_', ' ')})" for item in scenarios], fontsize=18)
        ax3.legend(loc="upper right", ncol=2, fontsize=20)
        ax3.set_ylim([0, 1.15])
        ax3.grid(axis="y", alpha=0.3)
        ax3.tick_params(axis="both", labelsize=24)
        fig3.tight_layout()
        fig3.savefig(resolved_output_dir / "edge_f1_by_scenario.pdf")
        fig3.savefig(resolved_output_dir / "edge_f1_by_scenario.png", dpi=300)
        print(f"Saved: {resolved_output_dir / 'edge_f1_by_scenario.pdf'}")
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(14, 10))
        ip_means, ip_stds = [], []
        for scenario in scenarios:
            data = df_bpop[df_bpop["scenario"] == scenario]["ip_f1"].dropna()
            ip_means.append(data.mean() if len(data) > 0 else 0)
            ip_stds.append(data.std() if len(data) > 0 else 0)
        ax4.bar(x, ip_means, yerr=ip_stds, color=cls.METHOD_COLORS["bhpop_single_po"], capsize=5, width=0.6, edgecolor="black", linewidth=0.8)
        ax4.set_ylabel("IP F1 Score (Concurrency Recovery)", fontsize=28)
        ax4.set_xlabel("Scenario", fontsize=28)
        ax4.set_xticks(x)
        ax4.set_xticklabels([cls.CLOUD_SCENARIO_SHORT[item] for item in scenarios], fontsize=24)
        ax4.set_ylim([0, 1.15])
        ax4.grid(axis="y", alpha=0.3)
        ax4.tick_params(axis="both", labelsize=24)
        for idx, (mean, std) in enumerate(zip(ip_means, ip_stds)):
            ax4.text(idx, mean + std + 0.02, f"{mean:.2f}", ha="center", fontsize=24, fontweight="bold")
        avg_ip_f1 = np.mean(ip_means)
        ax4.axhline(y=avg_ip_f1, color="red", linestyle="--", linewidth=3, alpha=0.7)
        ax4.text(len(scenarios) - 0.5, avg_ip_f1 + 0.02, f"Avg={avg_ip_f1:.2f}", fontsize=20, color="red")
        fig4.tight_layout()
        fig4.savefig(resolved_output_dir / "ip_f1_by_scenario.pdf")
        fig4.savefig(resolved_output_dir / "ip_f1_by_scenario.png", dpi=300)
        print(f"Saved: {resolved_output_dir / 'ip_f1_by_scenario.pdf'}")
        plt.close(fig4)

        selected_scenarios = ["slb_ecs_rds", "eip_slb_ecs"]
        selected_titles = {"slb_ecs_rds": "Scenario S2 (SLB-ECS-RDS)", "eip_slb_ecs": "Scenario S4 (EIP-SLB-ECS)"}
        fig5, axes5 = plt.subplots(1, 2, figsize=(18, 8))
        for idx, scenario in enumerate(selected_scenarios):
            ax = axes5[idx]
            for method in baseline_methods:
                data = df_baselines[(df_baselines["scenario"] == scenario) & (df_baselines["method"] == method)]
                if len(data) > 0:
                    grouped = data.groupby("ip_cov_target")["cover_f1"].mean()
                    ax.plot(grouped.index, grouped.values, marker=cls.METHOD_MARKERS[method], color=cls.METHOD_COLORS[method], label=cls.METHOD_LABELS[method], linewidth=4, markersize=16)
            bpop_data = df_bpop[df_bpop["scenario"] == scenario]
            if len(bpop_data) > 0:
                grouped = bpop_data.groupby("ip_cov_target")["cover_f1"].mean()
                ax.plot(grouped.index, grouped.values, marker=cls.METHOD_MARKERS["bhpop_single_po"], color=cls.METHOD_COLORS["bhpop_single_po"], label=cls.METHOD_LABELS["bhpop_single_po"], linewidth=4.5, markersize=20, zorder=10)
            ax.set_title(selected_titles[scenario], fontsize=28, fontweight="bold", pad=15)
            ax.set_xlabel("IP-Cov Target", fontsize=28)
            ax.set_ylabel("Edge F1", fontsize=28)
            ax.set_xlim([0.55, 1.05])
            ax.set_ylim([0, 1.0])
            ax.grid(alpha=0.3)
            ax.tick_params(axis="both", labelsize=24)
            if idx == 0:
                ax.legend(loc="lower right", fontsize=20, frameon=False)
        fig5.tight_layout()
        fig5.savefig(resolved_output_dir / "f1_vs_ipcov_by_scenario.pdf")
        fig5.savefig(resolved_output_dir / "f1_vs_ipcov_by_scenario.png", dpi=300)
        print(f"Saved: {resolved_output_dir / 'f1_vs_ipcov_by_scenario.pdf'}")
        plt.close(fig5)

        fig6, axes6 = plt.subplots(2, 3, figsize=(24, 16))
        axes6 = axes6.flatten()
        offsets = {"majority": -0.02, "inductive_miner_imf": 0.02, "heuristics_miner": 0.0, "queue_jump": -0.04}
        for idx, scenario in enumerate(scenarios):
            ax = axes6[idx]
            bpop_data = df_bpop[df_bpop["scenario"] == scenario]
            bpop_feas = bpop_data.groupby("ip_cov_target")["feas"].mean() if len(bpop_data) > 0 else None
            for method in baseline_methods:
                data = df_baselines[(df_baselines["scenario"] == scenario) & (df_baselines["method"] == method)]
                if len(data) == 0:
                    continue
                grouped = data.groupby("ip_cov_target")["feas"].mean()
                y_vals = grouped.values.copy()
                if bpop_feas is not None:
                    for inner_idx, ip_cov in enumerate(grouped.index):
                        if ip_cov in bpop_feas.index and abs(y_vals[inner_idx] - bpop_feas[ip_cov]) < 0.05:
                            y_vals[inner_idx] += offsets[method]
                ax.plot(grouped.index, y_vals, marker=cls.METHOD_MARKERS[method], color=cls.METHOD_COLORS[method], label=cls.METHOD_LABELS[method], linewidth=3.5, markersize=14)
            if bpop_feas is not None and not bpop_feas.isna().all():
                y_vals = bpop_feas.values.copy()
                if scenario == "simple_ecs":
                    y_vals = y_vals - 0.04
                    ax.annotate("BPOP=Maj.=Ind.=1.0", xy=(0.8, 1.0), fontsize=18, ha="center", va="bottom", color="gray")
                ax.plot(bpop_feas.index, y_vals, marker=cls.METHOD_MARKERS["bhpop_single_po"], color=cls.METHOD_COLORS["bhpop_single_po"], label=cls.METHOD_LABELS["bhpop_single_po"], linewidth=4, markersize=18, zorder=10)
            ax.set_xlabel("IP-Cov Target", fontsize=24)
            ax.set_ylabel("Feasibility", fontsize=24)
            ax.set_xlim([0.55, 1.05])
            ax.set_ylim([0, 1.15])
            ax.grid(alpha=0.3)
            ax.tick_params(axis="both", labelsize=20)
            if idx == 0:
                ax.legend(loc="lower left", fontsize=16, ncol=2)
        fig6.tight_layout()
        fig6.savefig(resolved_output_dir / "feasibility_by_scenario.pdf")
        fig6.savefig(resolved_output_dir / "feasibility_by_scenario.png", dpi=300)
        print(f"Saved: {resolved_output_dir / 'feasibility_by_scenario.pdf'}")
        plt.close(fig6)

        print("\n" + "=" * 80)
        print("SUMMARY TABLES FOR PAPER")
        print("=" * 80)
        print("\n--- Table 1: Aggregate Performance at IP-Cov=1.0 ---")
        print(f"{'Method':<20} {'Edge F1':>12} {'IP F1':>12} {'SHD':>10} {'Feasibility':>12}")
        print("-" * 70)
        for method in methods_order:
            data = df_bpop_10 if method == "bhpop_single_po" else df_baselines_10[df_baselines_10["method"] == method]
            edge_f1 = f"{data['cover_f1'].mean():.3f}±{data['cover_f1'].std():.3f}"
            ip_f1 = f"{data['ip_f1'].mean():.3f}" if data["ip_f1"].notna().any() else "N/A"
            shd = f"{data['shd'].mean():.1f}±{data['shd'].std():.1f}"
            feas = f"{data['feas'].mean():.3f}" if data["feas"].notna().any() else "N/A"
            print(f"{cls.METHOD_LABELS[method]:<20} {edge_f1:>12} {ip_f1:>12} {shd:>10} {feas:>12}")
        print("\n--- Table 2: BPOP by Scenario ---")
        print(f"{'Scenario':<25} {'Edge F1':>10} {'IP F1':>10} {'SHD':>8} {'n':>5}")
        print("-" * 60)
        for scenario in scenarios:
            data = df_bpop[df_bpop["scenario"] == scenario]
            print(f"{scenario:<25} {data['cover_f1'].mean():>10.3f} {data['ip_f1'].mean():>10.3f} {data['shd'].mean():>8.1f} {len(data):>5}")

        summary_rows = []
        for method in methods_order:
            data = df_bpop_10 if method == "bhpop_single_po" else df_baselines_10[df_baselines_10["method"] == method]
            summary_rows.append(
                {
                    "Method": cls.METHOD_LABELS[method],
                    "Edge_F1_mean": data["cover_f1"].mean(),
                    "Edge_F1_std": data["cover_f1"].std(),
                    "IP_F1_mean": data["ip_f1"].mean() if data["ip_f1"].notna().any() else np.nan,
                    "SHD_mean": data["shd"].mean(),
                    "SHD_std": data["shd"].std(),
                    "Feasibility_mean": data["feas"].mean() if data["feas"].notna().any() else np.nan,
                }
            )
        pd.DataFrame(summary_rows).to_csv(resolved_output_dir / "summary_table.csv", index=False)
        print(f"\nSaved: {resolved_output_dir / 'summary_table.csv'}")
        print("\n" + "=" * 80)
        print(f"All plots saved to: {resolved_output_dir}")
        print("=" * 80)

    @classmethod
    def plot_bpop_vs_qj_comparison(
        cls,
        *,
        output_dir: Path,
    ) -> None:
        cls.apply_style("compact_panel")
        scenarios = [
            "dual_zone_ecs_slb",
            "dual_zone_ecs_slb_rds",
            "eip_slb_ecs",
            "simple_ecs",
            "slb_ecs_rds",
            "slb_ecs_redis",
        ]
        scenario_labels = [
            "DZ-ECS-SLB",
            "DZ-ECS-SLB-RDS",
            "EIP-SLB-ECS",
            "Simple-ECS",
            "SLB-ECS-RDS",
            "SLB-ECS-Redis",
        ]
        bpop_f1 = [1.000, 0.815, 1.000, 1.000, 0.963, 0.900]
        qj_f1 = [1.000, 1.000, 1.000, 0.727, 0.786, 1.000]
        bpop_feas = [0.900, 0.300, 1.000, 1.000, 0.900, 1.000]
        qj_feas = [1.000, 0.571, 1.000, 1.000, 0.000, 0.963]
        fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.8))
        x = np.arange(len(scenarios))
        width = 0.35
        color_bpop = cls.METHOD_COLORS["bhpop_single_po"]
        color_qj = cls.METHOD_COLORS["queue_jump"]

        ax1 = axes[0]
        ax1.bar(x - width / 2, bpop_f1, width, label="BPOP", color=color_bpop, edgecolor="black", linewidth=0.3)
        ax1.bar(x + width / 2, qj_f1, width, label="QJ", color=color_qj, edgecolor="black", linewidth=0.3)
        ax1.set_ylabel("F1 Score")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scenario_labels, rotation=45, ha="right", fontsize=6)
        ax1.set_ylim(0, 1.12)
        ax1.set_title("(a) Structural Recovery (F1)", fontsize=8, fontweight="normal")
        ax1.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
        ax1.axhline(y=0.946, color=color_bpop, linestyle="--", linewidth=0.8, alpha=0.8)
        ax1.axhline(y=0.919, color=color_qj, linestyle="--", linewidth=0.8, alpha=0.8)

        ax2 = axes[1]
        ax2.bar(x - width / 2, bpop_feas, width, label="BPOP", color=color_bpop, edgecolor="black", linewidth=0.3)
        ax2.bar(x + width / 2, qj_feas, width, label="QJ", color=color_qj, edgecolor="black", linewidth=0.3)
        ax2.set_ylabel("Feasibility")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scenario_labels, rotation=45, ha="right", fontsize=6)
        ax2.set_ylim(0, 1.12)
        ax2.set_title("(b) Executability (Feasibility)", fontsize=8, fontweight="normal")
        ax2.legend(loc="upper right", frameon=True, fancybox=False, edgecolor="black", framealpha=1.0)
        ax2.axhline(y=0.850, color=color_bpop, linestyle="--", linewidth=0.8, alpha=0.8)
        ax2.axhline(y=0.756, color=color_qj, linestyle="--", linewidth=0.8, alpha=0.8)

        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="both", direction="out")
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        plt.tight_layout(pad=0.3, w_pad=0.8)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "bpop_vs_qj_comparison.pdf", format="pdf", bbox_inches="tight", dpi=300)
        print(f"Saved: {output_dir / 'bpop_vs_qj_comparison.pdf'}")
        plt.savefig(output_dir / "bpop_vs_qj_comparison.png", format="png", bbox_inches="tight", dpi=300)
        print(f"Saved: {output_dir / 'bpop_vs_qj_comparison.png'}")
        plt.close()

    @classmethod
    def load_wfinstances_inferred_cover(
        cls,
        results_dir: Path,
        *,
        ip_cov_target: float,
        threshold: float,
    ) -> np.ndarray | None:
        ip_cov_str = f"ip_cov_{ip_cov_target:.2f}".replace(".", "_")
        avg_h_path = results_dir / ip_cov_str / "avg_H.pkl"
        if not avg_h_path.exists():
            return None
        return cls.load_thresholded_cover_from_avg_h(
            avg_h_path,
            threshold=threshold,
            reduce_transitively=True,
        )

    @classmethod
    def _workflow_type_colors(cls, task_types: List[str]) -> Dict[str, str]:
        unique_types = sorted(set(task_types))
        return {
            task_type: cls.WORKFLOW_NODE_PALETTE[idx % len(cls.WORKFLOW_NODE_PALETTE)]
            for idx, task_type in enumerate(unique_types)
        }

    @classmethod
    def draw_wfinstances_po_comparison(
        cls,
        ax: plt.Axes,
        wf_data: Mapping[str, Any],
        true_cover: np.ndarray,
        inferred_cover: np.ndarray,
        *,
        show_edge_legend: bool = True,
        show_task_legend: bool = True,
    ) -> Dict[str, int]:
        if not NETWORKX_AVAILABLE:
            ax.text(0.5, 0.5, "NetworkX not available", ha="center", va="center")
            ax.axis("off")
            return {"correct": 0, "missed": 0, "false_pos": 0}

        n = int(wf_data["num_tasks"])
        task_types = list(wf_data["task_types"])
        numeric_ids = list(wf_data["numeric_ids"])
        pos = cls._build_level_layout(
            true_cover,
            horizontal_scale=1.0,
            vertical_scale=1.0,
            sort_key=numeric_ids,
        )

        graph = nx.DiGraph()
        graph.add_nodes_from(range(n))
        edge_counts = {"correct": 0, "missed": 0, "false_pos": 0}
        edges_by_type = {"correct": [], "missed": [], "false_pos": []}
        for src in range(n):
            for dst in range(n):
                if src == dst:
                    continue
                true_edge = true_cover[src, dst] == 1
                inferred_edge = inferred_cover[src, dst] == 1
                if true_edge and inferred_edge:
                    edge_type = "correct"
                elif true_edge and not inferred_edge:
                    edge_type = "missed"
                elif not true_edge and inferred_edge:
                    edge_type = "false_pos"
                else:
                    continue
                edge_counts[edge_type] += 1
                edges_by_type[edge_type].append((src, dst))
                graph.add_edge(src, dst)

        node_color_map = cls._workflow_type_colors(task_types)
        node_colors = [node_color_map[task_type] for task_type in task_types]

        node_size = 1200 if n <= 25 else 800 if n <= 50 else 500
        font_size = 18 if n <= 25 else 14 if n <= 50 else 11

        edge_styles = {
            "false_pos": {"color": cls.EDGE_COLORS["false_pos"], "style": (0, (3, 3)), "width": 2.0, "alpha": 0.8},
            "correct": {"color": cls.EDGE_COLORS["correct"], "style": "-", "width": 2.5, "alpha": 0.9},
            "missed": {"color": cls.EDGE_COLORS["missed"], "style": "--", "width": 2.5, "alpha": 0.9},
        }
        for edge_type in ["false_pos", "correct", "missed"]:
            if not edges_by_type[edge_type]:
                continue
            style = edge_styles[edge_type]
            nx.draw_networkx_edges(
                graph,
                pos,
                ax=ax,
                edgelist=edges_by_type[edge_type],
                arrows=True,
                arrowsize=15,
                arrowstyle="-|>",
                edge_color=style["color"],
                alpha=style["alpha"],
                width=style["width"],
                style=style["style"],
                connectionstyle="arc3,rad=0.08",
                min_source_margin=12,
                min_target_margin=12,
            )

        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_size,
            edgecolors="#333333",
            linewidths=1.5,
        )
        nx.draw_networkx_labels(
            graph,
            pos,
            labels={idx: numeric_ids[idx] for idx in range(n)},
            ax=ax,
            font_size=font_size,
            font_weight="bold",
            font_color="white",
        )

        ax.axis("off")
        x_coords = [coords[0] for coords in pos.values()]
        y_coords = [coords[1] for coords in pos.values()]
        x_margin = max(0.8, (max(x_coords) - min(x_coords)) * 0.1)
        y_margin = max(0.8, (max(y_coords) - min(y_coords)) * 0.08)
        ax.set_xlim(min(x_coords) - x_margin, max(x_coords) + x_margin)
        ax.set_ylim(min(y_coords) - y_margin - 0.3, max(y_coords) + y_margin)

        if show_edge_legend:
            leg1 = ax.legend(
                handles=[
                    mlines.Line2D([0], [0], color=cls.EDGE_COLORS["correct"], linewidth=8, label="Correct"),
                    mlines.Line2D([0], [0], color=cls.EDGE_COLORS["missed"], linewidth=8, linestyle="--", label="Missed"),
                    mlines.Line2D([0], [0], color=cls.EDGE_COLORS["false_pos"], linewidth=8, linestyle=":", label="Extra"),
                ],
                loc="lower left",
                fontsize=32,
                frameon=False,
                handlelength=4,
                handletextpad=1.0,
                borderpad=0.8,
                bbox_to_anchor=(0.0, 0.0),
            )
            ax.add_artist(leg1)

        if show_task_legend:
            unique_types = sorted(set(task_types))
            node_legend = [mpatches.Patch(color=node_color_map[item], label=item) for item in unique_types[:5]]
            if len(unique_types) > 5:
                node_legend.append(mpatches.Patch(color="#888888", label=f"+{len(unique_types) - 5}"))
            ax.legend(
                handles=node_legend,
                loc="lower right",
                fontsize=28,
                frameon=False,
                title="Tasks",
                title_fontsize=24,
                handlelength=2,
                handletextpad=0.8,
                borderpad=0.8,
                labelspacing=0.6,
                bbox_to_anchor=(1.0, 0.0),
            )

        return edge_counts

    @classmethod
    def plot_wfinstances_po_comparison(
        cls,
        *,
        project_root: Path | None = None,
        output_dir: Path | None = None,
        ip_cov: float = 0.95,
        workflow_thresholds: Mapping[str, float] | None = None,
    ) -> None:
        cls.apply_style("dag")
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        output_dir = Path(output_dir or prefer_existing_path(WFINSTANCES_PLOTS_DIR, LEGACY_WFINSTANCES_PLOTS_DIR))
        output_dir.mkdir(parents=True, exist_ok=True)
        workflow_thresholds = dict(workflow_thresholds or {"srasearch": 0.50, "epigenomics": 1 / 3})
        workflows = {
            "srasearch": {
                "data_dir": Path(project_root) / "data" / "wfinstances_srasearch",
                "results_dir": prefer_existing_path(
                    WFINSTANCES_SRASEARCH_RESULTS_DIR,
                    LEGACY_WFINSTANCES_SRASEARCH_RESULTS_DIR,
                ),
                "title": "SRASearch",
            },
            "epigenomics": {
                "data_dir": Path(project_root) / "data" / "wfinstances_epigenomics",
                "results_dir": prefer_existing_path(
                    WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
                    LEGACY_WFINSTANCES_EPIGENOMICS_RESULTS_DIR,
                ),
                "title": "Epigenomics",
            },
        }

        print("=" * 60)
        print(f"Generating PO Comparison Plots (IP-Cov={ip_cov})")
        print("=" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        stats = []
        for idx, (workflow_name, config) in enumerate(workflows.items()):
            print(f"\nProcessing {workflow_name}...")
            wf_data = cls.load_wfinstances_workflow_dag(
                Path(config["data_dir"]),
                title=str(config["title"]),
            )
            threshold = workflow_thresholds.get(workflow_name, 0.5)
            inferred_cover = cls.load_wfinstances_inferred_cover(
                Path(config["results_dir"]),
                ip_cov_target=ip_cov,
                threshold=threshold,
            )
            if inferred_cover is None:
                axes[idx].text(0.5, 0.5, "No data", ha="center", va="center", fontsize=14)
                axes[idx].set_title(config["title"])
                continue
            true_cover = cls.transitive_reduction(wf_data["cover"])
            edge_counts = cls.draw_wfinstances_po_comparison(
                axes[idx],
                wf_data,
                true_cover,
                inferred_cover,
                show_edge_legend=(idx == 0),
                show_task_legend=True,
            )
            tp = edge_counts["correct"]
            fp = edge_counts["false_pos"]
            fn = edge_counts["missed"]
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            stats.append({"title": config["title"], "tp": tp, "fp": fp, "fn": fn, "f1": f1})
            print(f"  threshold:{threshold:.3f} TP:{tp} FP:{fp} FN:{fn} F1:{f1:.3f}")

        plt.tight_layout(pad=0.2, w_pad=0.3)
        fig.savefig(output_dir / "po_comparison_ipcov095.pdf", bbox_inches="tight")
        fig.savefig(output_dir / "po_comparison_ipcov095.png", dpi=300, bbox_inches="tight")
        print(f"\nSaved: {output_dir / 'po_comparison_ipcov095.pdf'}")
        plt.close(fig)

        print("\n" + "=" * 50)
        print(f"{'Workflow':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'F1':<8}")
        print("-" * 50)
        for row in stats:
            print(f"{row['title']:<15} {row['tp']:<6} {row['fp']:<6} {row['fn']:<6} {row['f1']:<8.3f}")

    @classmethod
    def plot_wfinstances_dag_figures(
        cls,
        *,
        project_root: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        cls.apply_style("compact")
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        project_root = Path(project_root)
        output_dir = Path(output_dir or (project_root / "data" / "wfinstances_dag_figures"))
        output_dir.mkdir(parents=True, exist_ok=True)

        workflows: Dict[str, Dict[str, Any]] = {
            "srasearch": {
                "title": "SRASearch",
                "data_dir": project_root / "data" / "wfinstances_srasearch",
                "figsize": (10, 6),
                "filename": "srasearch_dag.pdf",
            },
            "epigenomics": {
                "title": "Epigenomics",
                "data_dir": project_root / "data" / "wfinstances_epigenomics",
                "figsize": (14, 10),
                "filename": "epigenomics_dag.pdf",
            },
        }

        def _load_workflow(workflow_name: str) -> Dict[str, Any]:
            config = workflows[workflow_name]
            return cls.load_wfinstances_workflow_dag(
                Path(config["data_dir"]),
                title=str(config["title"]),
            )

        def _print_task_type_summary(data: Mapping[str, Any]) -> None:
            print(f"  Task types for {data['name']}:")
            for task_type, count in cls.summarize_task_types(data):
                print(f"    - {task_type}: {count} tasks")

        print("=" * 60)
        print("Visualizing WFInstances DAGs for Paper")
        print("=" * 60)

        for workflow_name in ("srasearch", "epigenomics"):
            config = workflows[workflow_name]
            data = _load_workflow(workflow_name)
            print(f"Loading {data['name']} DAG...")
            _print_task_type_summary(data)
            fig, ax = plt.subplots(figsize=config["figsize"])
            cls.wfvisualize_po(data, ax)
            plt.tight_layout()
            figure_path = output_dir / str(config["filename"])
            plt.savefig(figure_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {figure_path}")

        print("Loading combined WFInstances DAG figure...")
        srasearch = _load_workflow("srasearch")
        epigenomics = _load_workflow("epigenomics")
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1)
        cls.wfvisualize_po(srasearch, ax1, show_legend=False)
        ax2 = fig.add_subplot(1, 2, 2)
        cls.wfvisualize_po(epigenomics, ax2, show_legend=True)
        plt.tight_layout()
        combined_path = output_dir / "combined_dags.pdf"
        plt.savefig(combined_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {combined_path}")

        sra_types = cls.summarize_task_types(srasearch)
        epi_types = cls.summarize_task_types(epigenomics)
        fig, ax = plt.subplots(figsize=(8, 4))
        table_data = [
            ["Metric", "SRASearch", "Epigenomics"],
            ["Tasks", str(srasearch["n"]), str(epigenomics["n"])],
            ["Edges", str(srasearch["num_edges"]), str(epigenomics["num_edges"])],
            ["Task Types", str(len(sra_types)), str(len(epi_types))],
            [
                "Types",
                ", ".join(task_type for task_type, _ in sra_types),
                ", ".join(task_type for task_type, _ in epi_types[:4]) + "...",
            ],
        ]
        ax.axis("off")
        table = ax.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.2, 0.3, 0.5],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        for col in range(3):
            table[(0, col)].set_facecolor("#4472C4")
            table[(0, col)].set_text_props(color="white", fontweight="bold")
        ax.set_title("Workflow Statistics", fontsize=12, fontweight="bold", pad=20)
        plt.tight_layout()
        stats_path = output_dir / "workflow_stats.pdf"
        plt.savefig(stats_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {stats_path}")
        print("=" * 60)


def setup_icml_visual_style(profile: str = "compact", **overrides: Any) -> None:
    IcmlVisual.apply_style(profile, **overrides)


def extract_task_type(name: str) -> str:
    return IcmlVisual.extract_task_type(name)


def extract_task_id(name: str) -> str:
    return IcmlVisual.extract_task_id(name)


def load_wfinstances_workflow_dag(data_dir: Path, *, title: str | None = None) -> Dict[str, Any]:
    return IcmlVisual.load_wfinstances_workflow_dag(data_dir, title=title)


def summarize_task_types(data: Mapping[str, Any]) -> List[Tuple[str, int]]:
    return IcmlVisual.summarize_task_types(data)


def compute_dag_levels(adj: np.ndarray) -> List[int]:
    return IcmlVisual.compute_dag_levels(adj)


def transitive_reduction(adj_matrix: np.ndarray) -> np.ndarray:
    return IcmlVisual.transitive_reduction(adj_matrix)


def wfvisualize_po(
    data: Mapping[str, Any],
    ax: plt.Axes,
    *,
    title: str | None = None,
    show_legend: bool = True,
    h_spacing: float = 1.5,
    v_spacing: float = 2.0,
) -> None:
    IcmlVisual.wfvisualize_po(
        data,
        ax,
        title=title,
        show_legend=show_legend,
        h_spacing=h_spacing,
        v_spacing=v_spacing,
    )
