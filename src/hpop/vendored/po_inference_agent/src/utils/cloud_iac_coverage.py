from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from src.utils.cloud_iac_dataset import (
    resolve_cloud_iac_data_root,
    resolve_cloud_iac_ground_truth_dir,
)
from src.utils.po_fun import BasicUtils


SCENARIO_MAPPING = {
    "simple_ecs": "S1",
    "slb_ecs_rds": "S2",
    "slb_ecs_redis": "S3",
    "eip_slb_ecs": "S4",
    "dual_zone_ecs_slb": "S5",
    "dual_zone_ecs_slb_rds": "S6",
}


class CloudIacCriticalPairCoverageAnalyzer:
    """Analyze Cloud-IaC incomparable-pair coverage (IP-Cov) over normalized observed traces."""

    def __init__(
        self,
        project_root: Path | None = None,
        *,
        trace_source: str = "combined",
        only_success: bool = True,
        drop_unknown_actions: bool = True,
        dedup_actions: bool = True,
    ) -> None:
        self.project_root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.data_root = resolve_cloud_iac_data_root(self.project_root)
        self.ground_truth_dir = resolve_cloud_iac_ground_truth_dir(self.data_root)
        self.trace_source = trace_source
        self.only_success = only_success
        self.drop_unknown_actions = drop_unknown_actions
        self.dedup_actions = dedup_actions

    @staticmethod
    def load_traces(dir_path: Path) -> List[Dict[str, Any]]:
        traces: List[Dict[str, Any]] = []
        for path in sorted(dir_path.glob("*.json")):
            if "summary" in path.name:
                continue
            traces.append(json.loads(path.read_text()))
        return traces

    @staticmethod
    def load_scenarios(scenarios_dir: Path) -> Dict[str, Dict[str, Any]]:
        scenarios: Dict[str, Dict[str, Any]] = {}
        for path in sorted(scenarios_dir.glob("*.json")):
            scenarios[path.stem] = json.loads(path.read_text())
        return scenarios

    @staticmethod
    def scenario_to_adj(scenario: Dict[str, Any]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
        edges = scenario.get("edges", [])
        task_ids = sorted({task for edge in edges for task in edge})
        task_to_idx = {task: idx for idx, task in enumerate(task_ids)}
        adj = np.zeros((len(task_ids), len(task_ids)), dtype=np.int8)
        for src, dst in edges:
            if src in task_to_idx and dst in task_to_idx:
                adj[task_to_idx[src], task_to_idx[dst]] = 1
        return adj, task_ids, task_to_idx

    @staticmethod
    def incomparable_pairs_from_closure(closure: np.ndarray) -> List[Tuple[int, int]]:
        n = closure.shape[0]
        return [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if closure[i, j] == 0 and closure[j, i] == 0
        ]

    @classmethod
    def critical_pair_coverage(
        cls,
        orders_local: Sequence[Sequence[int]],
        true_closure: np.ndarray,
    ) -> Tuple[int, int, float]:
        pairs = cls.incomparable_pairs_from_closure(true_closure)
        if not pairs:
            return 0, 0, 1.0

        seen: Dict[Tuple[int, int], int] = {pair: 0 for pair in pairs}
        for order in orders_local:
            pos = {item: idx for idx, item in enumerate(order)}
            for i, j in pairs:
                if i in pos and j in pos:
                    seen[(i, j)] |= 1 if pos[i] < pos[j] else 2

        covered = sum(mask == 3 for mask in seen.values())
        total = len(pairs)
        return covered, total, covered / total

    @staticmethod
    def _extract_intent_type(trace: Dict[str, Any]) -> str | None:
        return trace.get("intent_type") or (trace.get("intent") or {}).get("intent_type")

    @staticmethod
    def _extract_sequence(trace: Dict[str, Any]) -> List[str]:
        sequence = trace.get("action_sequence") or trace.get("actions_executed") or []
        return [str(action) for action in sequence]

    def _select_traces(
        self,
        traces: Iterable[Dict[str, Any]],
        scenario_id: str,
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        for trace in traces:
            if self._extract_intent_type(trace) != scenario_id:
                continue
            if self.only_success and trace.get("status") not in (None, "success"):
                continue
            if not self._extract_sequence(trace):
                continue
            selected.append(trace)
        return selected

    def _normalize_sequence(self, seq: Sequence[str], task_set: set[str]) -> List[str]:
        out: List[str] = []
        seen: set[str] = set()
        for action in seq:
            if self.drop_unknown_actions and action not in task_set:
                continue
            if self.dedup_actions and action in seen:
                continue
            seen.add(action)
            out.append(action)
        return out

    def _selected_traces_for_scenario(
        self,
        scenario_id: str,
        expert_traces: Sequence[Dict[str, Any]],
        model_traces: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        selected: List[Dict[str, Any]] = []
        if self.trace_source in ("expert", "combined"):
            selected.extend(self._select_traces(expert_traces, scenario_id))
        if self.trace_source in ("model", "combined"):
            selected.extend(self._select_traces(model_traces, scenario_id))
        return selected

    def build_dataset(self) -> Tuple[Dict[str, Any], Dict[str, List[List[int]]]]:
        scenarios = self.load_scenarios(self.ground_truth_dir)
        expert_traces = self.load_traces(self.data_root / "expert_traces")
        model_traces = self.load_traces(self.data_root / "traces")

        scenario_data: Dict[str, Any] = {}
        orders_local_by_assessor: Dict[str, List[List[int]]] = {}

        for scenario_id in sorted(scenarios.keys()):
            true_adj, task_ids, task_to_idx = self.scenario_to_adj(scenarios[scenario_id])
            true_closure = BasicUtils.transitive_closure(true_adj.astype(np.int8))
            true_cover = BasicUtils.transitive_reduction_optimized(true_closure.astype(np.int8))
            scenario_data[scenario_id] = {
                "task_ids": task_ids,
                "task_to_idx": task_to_idx,
                "true_adj": true_adj,
                "true_closure": true_closure,
                "true_cover": true_cover,
                "num_nodes": len(task_ids),
                "num_edges": int(true_adj.sum()),
            }

            task_set = set(task_ids)
            orders_local: List[List[int]] = []
            for trace in self._selected_traces_for_scenario(scenario_id, expert_traces, model_traces):
                normalized = self._normalize_sequence(self._extract_sequence(trace), task_set)
                if not normalized:
                    continue
                orders_local.append([task_to_idx[action] for action in normalized])

            orders_local_by_assessor[scenario_id] = orders_local

        return scenario_data, orders_local_by_assessor

    def analyze(self) -> List[Dict[str, Any]]:
        scenario_data, orders_local_by_assessor = self.build_dataset()
        results: List[Dict[str, Any]] = []
        for scenario_id in sorted(SCENARIO_MAPPING.keys(), key=lambda sid: SCENARIO_MAPPING[sid]):
            info = scenario_data[scenario_id]
            orders_local = orders_local_by_assessor.get(scenario_id, [])
            covered_pairs, total_pairs, coverage_ratio = self.critical_pair_coverage(
                orders_local,
                info["true_closure"],
            )
            results.append(
                {
                    "scenario": SCENARIO_MAPPING[scenario_id],
                    "id": scenario_id,
                    "nodes": info["num_nodes"],
                    "edges": info["num_edges"],
                    "traces": len(orders_local),
                    "crit_pairs": total_pairs,
                    "covered_pairs": covered_pairs,
                    "cp_cov": coverage_ratio,
                }
            )
        return results

    def print_report(self, results: Sequence[Dict[str, Any]]) -> None:
        print("=== Cloud-IaC IP-Cov Analysis ===")
        print()
        for row in results:
            print(f"{row['scenario']} ({row['id']}):")
            print(f"  |V| = {row['nodes']}, |E| = {row['edges']}, N = {row['traces']}")
            print(f"  IP-Cov = {row['cp_cov']:.3f}")
            print(f"  Incomparable pairs = {row['crit_pairs']}")
            print()

        print("=== Summary Table ===")
        print("Scenario | |V| | |E| | N | IP-Cov | Covered/Total")
        print("---------|-----|-----|----|--------|--------------")
        for row in results:
            print(
                f"{row['scenario']:8s} | {row['nodes']:3d} | {row['edges']:3d} | "
                f"{row['traces']:2d} | {row['cp_cov']:.3f} | {row['covered_pairs']}/{row['crit_pairs']}"
            )


class CloudIacObservedCriticalPairCoverageAnalyzer(CloudIacCriticalPairCoverageAnalyzer):
    """Detailed Cloud-IaC observed IP-Cov report with counts."""

    def analyze(self) -> List[Dict[str, Any]]:
        scenario_data, orders_local_by_assessor = self.build_dataset()
        results: List[Dict[str, Any]] = []
        for scenario_id in sorted(SCENARIO_MAPPING.keys(), key=lambda sid: SCENARIO_MAPPING[sid]):
            info = scenario_data[scenario_id]
            orders_local = orders_local_by_assessor.get(scenario_id, [])
            covered_pairs, total_pairs, coverage_ratio = self.critical_pair_coverage(
                orders_local,
                info["true_closure"],
            )
            results.append(
                {
                    "scenario": SCENARIO_MAPPING[scenario_id],
                    "id": scenario_id,
                    "nodes": info["num_nodes"],
                    "edges": info["num_edges"],
                    "traces": len(orders_local),
                    "covered_pairs": covered_pairs,
                    "total_pairs": total_pairs,
                    "coverage_ratio": coverage_ratio,
                }
            )
        return results

    def print_report(self, results: Sequence[Dict[str, Any]]) -> None:
        print("=== Cloud-IaC Observed IP-Cov Analysis ===")
        print()
        for row in results:
            print(f"{row['scenario']} ({row['id']}):")
            print(f"  |V| = {row['nodes']}, |E| = {row['edges']}, N = {row['traces']}")
            print(
                f"  Incomparable pairs covered: {row['covered_pairs']}/{row['total_pairs']} "
                f"({row['coverage_ratio']:.3f})"
            )
            print()

        print("=== Summary Table ===")
        print("Scenario | |V| | |E| | N | Covered/Total IP | Coverage")
        print("---------|-----|-----|----|-----------------|----------")
        for row in results:
            print(
                f"{row['scenario']:8s} | {row['nodes']:3d} | {row['edges']:3d} | "
                f"{row['traces']:2d} | {row['covered_pairs']:3d}/{row['total_pairs']:<11d} | "
                f"{row['coverage_ratio']:.3f}"
            )
