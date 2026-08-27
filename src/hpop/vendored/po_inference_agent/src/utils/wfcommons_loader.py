from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Set, Tuple


class WFCommonsHPOInputs(NamedTuple):
    """
    Structured (but tuple-compatible) container for BHPOP/HPO inputs derived from WFCommons.

    Notes
    -----
    This is a `NamedTuple` so callers can use either attribute access (e.g. `inputs.M0`)
    or tuple-unpacking (backwards-compatible with older code).
    """

    M0: List[int]
    assessors: List[int]
    M_a_dict: Dict[int, List[int]]
    O_a_i_dict: Dict[int, List[List[int]]]
    observed_orders: Dict[int, List[List[int]]]
    task_to_idx: Dict[str, int]
    idx_to_task: Dict[int, str]
    parents_subset: Dict[str, List[str]]
    trace_metadata: List[Dict[str, Any]]


def load_workflow_instance(path: Path, *, use_finish_time: bool = False) -> Dict[str, Any]:
    """
    Load a single WFCommons WfFormat workflow instance.

    Returns
    -------
    {
        "task_ids": List[str],
        "parents": Dict[str, Set[str]],
        "executed_at": Dict[str, float],
        "execution_order": List[str],
        "execution_order_source": str,
        "execution_task_order": List[str],
        "metadata": Dict[str, Any],
    }
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return load_workflow_instance_data(
        data, use_finish_time=use_finish_time, source_path=path
    )


def load_workflow_instance_data(
    data: Dict[str, Any],
    *,
    use_finish_time: bool = False,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    workflow = _resolve_workflow(data)
    spec = workflow.get("specification", {})
    tasks = spec.get("tasks", [])
    if not isinstance(tasks, list):
        location = f" in {source_path}" if source_path else ""
        raise ValueError(f"workflow.specification.tasks not found{location}.")

    task_ids, parents = _parse_spec_tasks(tasks)
    task_id_set = set(task_ids)
    executed_at = _extract_execution_times(
        workflow, task_id_set, use_finish_time=use_finish_time
    )
    execution_task_order = _extract_execution_task_order(workflow)
    execution_order, execution_source = _derive_execution_order(
        task_ids,
        parents,
        executed_at,
        execution_task_order,
    )

    return {
        "task_ids": task_ids,
        "parents": parents,
        "executed_at": executed_at,
        "execution_order": execution_order,
        "execution_order_source": execution_source,
        "execution_task_order": execution_task_order,
        "metadata": _extract_metadata(data, workflow, source_path),
    }


def iter_workflow_instance_paths(root: Path) -> Iterator[Path]:
    root = Path(root)
    if root.is_dir():
        yield from sorted(root.rglob("*.json"))
    else:
        yield root


def iter_workflow_instances(
    root: Path, *, use_finish_time: bool = False, skip_invalid: bool = True
) -> Iterator[Dict[str, Any]]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() == ".zip":
        yield from iter_workflow_instances_from_zip(
            root, use_finish_time=use_finish_time, skip_invalid=skip_invalid
        )
        return
    for path in iter_workflow_instance_paths(root):
        try:
            yield load_workflow_instance(path, use_finish_time=use_finish_time)
        except ValueError:
            if not skip_invalid:
                raise


def load_workflow_instances(
    root: Path, *, use_finish_time: bool = False, skip_invalid: bool = True
) -> List[Dict[str, Any]]:
    return list(
        iter_workflow_instances(root, use_finish_time=use_finish_time, skip_invalid=skip_invalid)
    )


def iter_workflow_instances_from_zip(
    zip_path: Path, *, use_finish_time: bool = False, skip_invalid: bool = True
) -> Iterator[Dict[str, Any]]:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".json"):
                continue
            try:
                with zf.open(name) as handle:
                    data = json.load(handle)
            except json.JSONDecodeError:
                if not skip_invalid:
                    raise
                continue
            try:
                source_path = Path(f"{zip_path}::{name}")
                yield load_workflow_instance_data(
                    data, use_finish_time=use_finish_time, source_path=source_path
                )
            except ValueError:
                if not skip_invalid:
                    raise


def load_workflow_instances_from_zip(
    zip_path: Path, *, use_finish_time: bool = False, skip_invalid: bool = True
) -> List[Dict[str, Any]]:
    return list(
        iter_workflow_instances_from_zip(
            zip_path, use_finish_time=use_finish_time, skip_invalid=skip_invalid
        )
    )


def build_adjacency_matrix(
    task_ids: List[str], parents: Dict[str, Set[str]]
) -> List[List[int]]:
    index = {task_id: idx for idx, task_id in enumerate(task_ids)}
    n = len(task_ids)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for child_id, parent_ids in parents.items():
        if child_id not in index:
            continue
        child_idx = index[child_id]
        for parent_id in parent_ids:
            if parent_id in index:
                matrix[index[parent_id]][child_idx] = 1
    return matrix


def group_instances_by_name(
    instances: Iterable[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for instance in instances:
        metadata = instance.get("metadata", {})
        name = metadata.get("instance_name") or "unknown"
        grouped.setdefault(name, []).append(instance)
    return grouped


def build_hpo_inputs_from_instances(
    instances: Iterable[Dict[str, Any]],
    *,
    max_items: Optional[int] = None,
    prefer_execution_order: bool = True,
    validate_task_ids: bool = True,
) -> WFCommonsHPOInputs:
    """
    Convert WFCommons instances into BHPOP/HPO inputs.

    Returns
    -------
    WFCommonsHPOInputs:
        `M0, assessors, M_a_dict, O_a_i_dict, observed_orders,
        task_to_idx, idx_to_task, parents_subset, trace_metadata`
    """
    instance_list = list(instances)
    if not instance_list:
        raise ValueError("No workflow instances provided.")

    base = instance_list[0]
    task_ids = list(base.get("task_ids", []))
    parents = dict(base.get("parents", {}))
    if not task_ids:
        raise ValueError("Base instance has no task_ids.")

    if validate_task_ids:
        base_set = set(task_ids)
        for inst in instance_list[1:]:
            if set(inst.get("task_ids", [])) != base_set:
                raise ValueError("Task sets differ across instances; group them first.")

    if max_items is not None and len(task_ids) > max_items:
        order_hint = base.get("execution_order") if prefer_execution_order else None
        if not order_hint:
            order_hint = task_ids
        task_ids = list(order_hint[:max_items])

    task_set = set(task_ids)
    parents_subset: Dict[str, List[str]] = {}
    for task_id in task_ids:
        filtered = [p for p in parents.get(task_id, set()) if p in task_set]
        parents_subset[task_id] = sorted(filtered)

    task_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}
    idx_to_task = {idx: task_id for task_id, idx in task_to_idx.items()}

    M0 = list(range(len(task_ids)))
    assessors = [1]
    M_a_dict = {1: M0}
    O_a_i_dict: Dict[int, List[List[int]]] = {1: []}
    observed_orders: Dict[int, List[List[int]]] = {1: []}
    trace_metadata: List[Dict[str, Any]] = []

    for inst in instance_list:
        order = inst.get("execution_order") if prefer_execution_order else None
        if not order:
            order = inst.get("task_ids", [])
        if not order:
            continue
        order_indices = [task_to_idx[task_id] for task_id in order if task_id in task_to_idx]
        if not order_indices:
            continue
        choice_set = sorted(set(order_indices))
        observed_orders[1].append(order_indices)
        O_a_i_dict[1].append(choice_set)
        trace_metadata.append(inst.get("metadata", {}))

    if not observed_orders[1]:
        raise ValueError("No valid traces after filtering instances.")

    return WFCommonsHPOInputs(
        M0=M0,
        assessors=assessors,
        M_a_dict=M_a_dict,
        O_a_i_dict=O_a_i_dict,
        observed_orders=observed_orders,
        task_to_idx=task_to_idx,
        idx_to_task=idx_to_task,
        parents_subset=parents_subset,
        trace_metadata=trace_metadata,
    )


def order_by_execution_time(
    executed_at: Dict[str, float], *, task_ids: Optional[List[str]] = None
) -> List[str]:
    if not executed_at:
        return []

    if task_ids is None:
        return [
            task_id
            for task_id, _ in sorted(
                executed_at.items(), key=lambda item: (item[1], item[0])
            )
        ]

    tie_break = {task_id: idx for idx, task_id in enumerate(task_ids)}
    max_idx = len(task_ids)
    return [
        task_id
        for task_id, _ in sorted(
            executed_at.items(),
            key=lambda item: (item[1], tie_break.get(item[0], max_idx), item[0]),
        )
    ]


def _extract_metadata(
    data: Dict[str, Any], workflow: Dict[str, Any], source_path: Optional[Path]
) -> Dict[str, Any]:
    runtime_system = data.get("runtimeSystem")
    runtime_system_name = None
    runtime_system_version = None
    if isinstance(runtime_system, dict):
        runtime_system_name = runtime_system.get("name")
        runtime_system_version = runtime_system.get("version")
    elif isinstance(runtime_system, str):
        runtime_system_name = runtime_system
    return {
        "instance_name": data.get("name") or workflow.get("name"),
        "description": data.get("description"),
        "schema_version": data.get("schemaVersion"),
        "created_at": data.get("createdAt"),
        "runtime_system": runtime_system,
        "runtime_system_name": runtime_system_name,
        "runtime_system_version": runtime_system_version,
        "source_path": str(source_path) if source_path else None,
    }


def _extract_execution_task_order(workflow: Dict[str, Any]) -> List[str]:
    execution = workflow.get("execution", {})
    tasks = execution.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    order: List[str] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = _normalize_task_id(task)
        if task_id:
            order.append(task_id)
    return order


def _derive_execution_order(
    task_ids: List[str],
    parents: Dict[str, Set[str]],
    executed_at: Dict[str, float],
    execution_task_order: List[str],
) -> Tuple[List[str], str]:
    fallback_order = execution_task_order or task_ids
    fallback_idx = {task_id: idx for idx, task_id in enumerate(fallback_order)}
    max_idx = len(fallback_order)

    if executed_at:
        def priority(task_id: str) -> Tuple[float, int, str]:
            return (
                executed_at.get(task_id, float("inf")),
                fallback_idx.get(task_id, max_idx),
                task_id,
            )

        source = "executed_at"
    elif execution_task_order:
        def priority(task_id: str) -> Tuple[int, str]:
            return (fallback_idx.get(task_id, max_idx), task_id)

        source = "execution_list"
    else:
        def priority(task_id: str) -> Tuple[int, str]:
            return (fallback_idx.get(task_id, max_idx), task_id)

        source = "spec"

    return _toposort_with_priority(task_ids, parents, priority), source


def _toposort_with_priority(
    task_ids: List[str],
    parents: Dict[str, Set[str]],
    priority,
) -> List[str]:
    import heapq

    task_set = set(task_ids)
    normalized_parents = {
        task_id: {p for p in parents.get(task_id, set()) if p in task_set}
        for task_id in task_ids
    }
    successors: Dict[str, Set[str]] = {task_id: set() for task_id in task_ids}
    indegree: Dict[str, int] = {task_id: 0 for task_id in task_ids}

    for child, parent_set in normalized_parents.items():
        indegree[child] = len(parent_set)
        for parent in parent_set:
            successors[parent].add(child)

    heap: List[Tuple[Any, str]] = []
    for task_id, deg in indegree.items():
        if deg == 0:
            heapq.heappush(heap, (priority(task_id), task_id))

    order: List[str] = []
    while heap:
        _, task_id = heapq.heappop(heap)
        order.append(task_id)
        for child in successors[task_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(heap, (priority(child), child))

    if len(order) != len(task_ids):
        raise ValueError("Adjacency contains a cycle; cannot derive execution order.")
    return order


def _resolve_workflow(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict) and "workflow" in data:
        workflow = data["workflow"]
    else:
        workflow = data
    if not isinstance(workflow, dict):
        raise ValueError("Invalid workflow payload.")
    return workflow


def _parse_spec_tasks(tasks: Iterable[Any]) -> Tuple[List[str], Dict[str, Set[str]]]:
    task_ids: List[str] = []
    parents: Dict[str, Set[str]] = {}
    children_map: Dict[str, Set[str]] = {}
    seen: Set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = _normalize_task_id(task)
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        task_ids.append(task_id)
        parents[task_id] = _parse_parents(task.get("parents"))
        children_map[task_id] = _parse_children(task.get("children"))

    for parent_id, children in children_map.items():
        for child_id in sorted(children):
            if child_id not in parents:
                parents[child_id] = set()
                task_ids.append(child_id)
            parents[child_id].add(parent_id)
    return task_ids, parents


def _parse_parents(raw_parents: Any) -> Set[str]:
    return _parse_dependency_list(raw_parents)


def _parse_children(raw_children: Any) -> Set[str]:
    return _parse_dependency_list(raw_children)


def _parse_dependency_list(raw_list: Any) -> Set[str]:
    if not raw_list:
        return set()
    if not isinstance(raw_list, list):
        return set()
    ids: Set[str] = set()
    for entry in raw_list:
        entry_id = _normalize_parent_id(entry)
        if entry_id:
            ids.add(entry_id)
    return ids


def _normalize_task_id(task: Dict[str, Any]) -> Optional[str]:
    for key in ("id", "taskId", "task_id", "name"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_parent_id(parent: Any) -> Optional[str]:
    if isinstance(parent, str) and parent.strip():
        return parent.strip()
    if isinstance(parent, dict):
        for key in ("id", "taskId", "task_id", "name"):
            value = parent.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_execution_times(
    workflow: Dict[str, Any],
    task_ids: Set[str],
    *,
    use_finish_time: bool,
) -> Dict[str, float]:
    execution = workflow.get("execution", {})
    tasks = execution.get("tasks", [])
    if not isinstance(tasks, list):
        return {}

    executed_at: Dict[str, float] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = _normalize_task_id(task)
        if not task_id or (task_ids and task_id not in task_ids):
            continue
        start_time = _parse_timestamp(
            task.get("executedAt")
            or task.get("executed_at")
            or task.get("startTime")
            or task.get("start_time")
            or task.get("startedAt")
        )
        if start_time is None:
            continue
        if use_finish_time:
            runtime = _parse_duration(
                task.get("runtimeInSeconds")
                or task.get("runtime_in_seconds")
                or task.get("runtimeSeconds")
                or task.get("runtime")
            )
            if runtime is not None:
                start_time += runtime
        executed_at[task_id] = float(start_time)
    return executed_at


def _parse_duration(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _parse_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return None
    return None
