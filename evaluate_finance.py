#!/usr/bin/env python3
"""Evaluate finance benchmark results with DAG-level metrics.

This script compares generated DAGs against expected DAGs to measure
dependency generation and parameter generation precision.
"""

import argparse
import json
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


def string_similarity(s1: str, s2: str) -> float:
    """Calculate string similarity ratio (0-1)."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()


def normalize_args(args: Any) -> Tuple:
    """Normalize args for comparison."""
    if isinstance(args, str):
        return (args,)
    elif isinstance(args, list):
        return tuple(args)
    elif isinstance(args, tuple):
        return args
    return (str(args),)


def extract_task_signature(task: Dict) -> Tuple:
    """Extract a comparable signature from a task."""
    tool = task.get("name", task.get("tool", ""))
    args = task.get("args", [])
    return (tool, normalize_args(args))


def match_tasks(
    generated_tasks: List[Dict],
    expected_tasks: List[Dict],
    similarity_threshold: float = 0.8,
) -> Dict[int, Optional[int]]:
    """Match generated tasks to expected tasks using Hungarian-like matching.

    Returns:
        Dict mapping generated task idx -> expected task idx (or None if no match)
    """
    # Build similarity matrix
    gen_by_idx = {t["idx"]: t for t in generated_tasks}
    exp_by_idx = {t["idx"]: t for t in expected_tasks}

    gen_indices = list(gen_by_idx.keys())
    exp_indices = list(exp_by_idx.keys())

    # Simple greedy matching (can be improved with Hungarian algorithm)
    matches = {}
    used_expected = set()

    for gen_idx in gen_indices:
        gen_task = gen_by_idx[gen_idx]
        gen_tool = gen_task.get("name", gen_task.get("tool", ""))
        gen_args = gen_task.get("args", [])

        best_match = None
        best_score = similarity_threshold

        for exp_idx in exp_indices:
            if exp_idx in used_expected:
                continue

            exp_task = exp_by_idx[exp_idx]
            exp_tool = exp_task.get("tool", "")
            exp_args = exp_task.get("args", [])

            # Tool must match exactly
            if gen_tool != exp_tool:
                continue

            # Calculate arg similarity
            if gen_tool == "search":
                # For search, compare the query string
                if gen_args and exp_args:
                    score = string_similarity(str(gen_args[0]), str(exp_args[0]))
                    if score > best_score:
                        best_score = score
                        best_match = exp_idx
            elif gen_tool == "math":
                # For math, compare problem string and context references
                if gen_args and exp_args:
                    # Compare problem string
                    gen_problem = str(gen_args[0]) if gen_args else ""
                    exp_problem = str(exp_args[0]) if exp_args else ""
                    problem_score = string_similarity(gen_problem, exp_problem)

                    # Check context references match
                    gen_context = set(gen_args[1]) if len(gen_args) > 1 else set()
                    exp_context = set(exp_args[1]) if len(exp_args) > 1 else set()

                    # Context must match exactly for good match
                    if gen_context == exp_context and problem_score > best_score:
                        best_score = problem_score
                        best_match = exp_idx
                    elif problem_score > best_score + 0.1:  # Allow looser match
                        best_score = problem_score
                        best_match = exp_idx
            elif gen_tool == "join":
                best_match = exp_idx
                best_score = 1.0

        if best_match is not None:
            matches[gen_idx] = best_match
            used_expected.add(best_match)
        else:
            matches[gen_idx] = None

    return matches


def compare_dependencies(
    generated_deps: Dict[str, List[int]],
    expected_deps: Dict[str, List[int]],
    task_matches: Dict[int, Optional[int]],
) -> Tuple[int, int, int, int]:
    """Compare dependency edges.

    Returns:
        (correct_edges, missing_edges, extra_edges, total_expected)
    """
    # Build reverse mapping: expected idx -> generated idx
    reverse_matches = {v: k for k, v in task_matches.items() if v is not None}

    correct = 0
    missing = 0
    extra = 0
    total_expected = 0

    # Count expected edges
    for depender, dependees in expected_deps.items():
        total_expected += len(dependees)

    # Count generated edges vs expected
    for gen_depender, gen_dependees in generated_deps.items():
        gen_depender_int = int(gen_depender) if isinstance(gen_depender, str) else gen_depender

        for gen_dependee in gen_dependees:
            # Check if this edge exists in expected
            matched_depender = task_matches.get(gen_depender_int)
            matched_dependee = task_matches.get(gen_dependee)

            if matched_depender is None or matched_dependee is None:
                extra += 1
                continue

            # Check if expected has this edge
            expected_dependees = expected_deps.get(str(matched_depender), [])
            if matched_dependee in expected_dependees:
                correct += 1
            else:
                extra += 1

    # Count missing edges
    for exp_depender, exp_dependees in expected_deps.items():
        exp_depender_int = int(exp_depender)
        gen_depender = reverse_matches.get(exp_depender_int)

        if gen_depender is None:
            missing += len(exp_dependees)
            continue

        gen_dependees = generated_deps.get(str(gen_depender), [])

        for exp_dependee in exp_dependees:
            gen_dependee = reverse_matches.get(exp_dependee)
            if gen_dependee is None or gen_dependee not in gen_dependees:
                missing += 1

    return correct, missing, extra, total_expected


def check_dag_isomorphism(
    generated_tasks: List[Dict],
    generated_deps: Dict[str, List[int]],
    expected_tasks: List[Dict],
    expected_deps: Dict[str, List[int]],
    task_matches: Dict[int, Optional[int]],
) -> bool:
    """Check if generated DAG is isomorphic to expected DAG."""
    if len(generated_tasks) != len(expected_tasks):
        return False

    # Build NetworkX graphs
    gen_G = nx.DiGraph()
    exp_G = nx.DiGraph()

    # Add nodes with attributes
    for task in generated_tasks:
        gen_G.add_node(task["idx"], tool=task.get("name", task.get("tool", "")))

    for task in expected_tasks:
        exp_G.add_node(task["idx"], tool=task.get("tool", ""))

    # Add edges
    for depender, dependees in generated_deps.items():
        for dependee in dependees:
            gen_G.add_edge(dependee, int(depender))

    for depender, dependees in expected_deps.items():
        for dependee in dependees:
            exp_G.add_edge(dependee, int(depender))

    # Check isomorphism
    try:
        return nx.is_isomorphic(gen_G, exp_G, node_match=lambda n1, n2: n1.get("tool") == n2.get("tool"))
    except Exception:
        return False


def evaluate_arg_references(
    generated_tasks: List[Dict],
    expected_tasks: List[Dict],
    task_matches: Dict[int, Optional[int]],
) -> Tuple[int, int]:
    """Evaluate argument reference accuracy.

    Returns:
        (correct_refs, total_refs_needed)
    """
    correct = 0
    total = 0

    for gen_task in generated_tasks:
        gen_idx = gen_task["idx"]
        matched_exp_idx = task_matches.get(gen_idx)

        if matched_exp_idx is None:
            continue

        # Find expected task
        exp_task = None
        for t in expected_tasks:
            if t["idx"] == matched_exp_idx:
                exp_task = t
                break

        if exp_task is None:
            continue

        gen_args = gen_task.get("args", [])
        exp_args = exp_task.get("args", [])

        # Check if this task needs references
        if gen_task.get("name", gen_task.get("tool", "")) == "math" and len(exp_args) > 1:
            total += 1
            gen_context = set(gen_args[1]) if len(gen_args) > 1 else set()
            exp_context = set(exp_args[1]) if len(exp_args) > 1 else set()

            if gen_context == exp_context:
                correct += 1

    return correct, total


def build_dependency_dict(tasks: List[Dict]) -> Dict[str, List[int]]:
    """Build dependency dict from tasks list.

    Each task has a 'dependencies' list indicating which tasks it depends on.
    Returns dict mapping depender idx -> list of dependee idxs.
    """
    deps = {}
    for task in tasks:
        depender = str(task["idx"])
        dependees = task.get("dependencies", [])
        if dependees:
            deps[depender] = dependees
    return deps


def evaluate_sample(
    generated: Dict,
    expected: Dict,
) -> Dict[str, Any]:
    """Evaluate a single sample."""
    gen_plan = generated.get("plan", {})
    exp_dag = expected.get("expected_dag", {})

    gen_tasks = gen_plan.get("tasks", [])
    gen_deps = build_dependency_dict(gen_tasks)  # Build from task.dependencies
    exp_tasks = exp_dag.get("tasks", [])
    exp_deps = exp_dag.get("dependencies", {})

    # Match tasks
    task_matches = match_tasks(gen_tasks, exp_tasks)

    # Task metrics
    matched_count = sum(1 for m in task_matches.values() if m is not None)
    task_recall = matched_count / len(exp_tasks) if exp_tasks else 0
    task_precision = matched_count / len(gen_tasks) if gen_tasks else 0

    # Dependency metrics
    correct_deps, missing_deps, extra_deps, total_exp_deps = compare_dependencies(
        gen_deps, exp_deps, task_matches
    )
    dep_recall = correct_deps / total_exp_deps if total_exp_deps > 0 else 0
    dep_precision = correct_deps / (correct_deps + extra_deps) if (correct_deps + extra_deps) > 0 else 0

    # DAG isomorphism
    is_isomorphic = check_dag_isomorphism(gen_tasks, gen_deps, exp_tasks, exp_deps, task_matches)

    # Arg reference accuracy
    correct_args, total_args = evaluate_arg_references(gen_tasks, exp_tasks, task_matches)
    arg_ref_acc = correct_args / total_args if total_args > 0 else 1.0

    return {
        "task_recall": task_recall,
        "task_precision": task_precision,
        "dep_recall": dep_recall,
        "dep_precision": dep_precision,
        "dag_isomorphic": is_isomorphic,
        "arg_ref_acc": arg_ref_acc,
        "complexity": expected.get("complexity", "unknown"),
    }


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--file", type=str, required=True, help="Results JSON file")
    argparser.add_argument("--detail", action="store_true", help="Show per-sample details")
    argparser.add_argument("--dataset", type=str, default="datasets/finance_dataset.json", help="Dataset file")
    args = argparser.parse_args()

    # Load results
    with open(args.file, "r") as f:
        results = json.load(f)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    # Build expected lookup
    expected_by_id = {str(item["id"]): item for item in dataset}

    # Evaluate each sample
    all_metrics = []
    by_complexity = defaultdict(list)

    for sample_id, generated in results.items():
        expected = expected_by_id.get(str(sample_id))
        if expected is None:
            print(f"Warning: No expected data for sample {sample_id}")
            continue

        metrics = evaluate_sample(generated, expected)
        metrics["id"] = sample_id
        all_metrics.append(metrics)
        by_complexity[metrics["complexity"]].append(metrics)

        if args.detail:
            print(f"Sample {sample_id}:")
            print(f"  Task Recall: {metrics['task_recall']:.2%}")
            print(f"  Dep Recall: {metrics['dep_recall']:.2%}")
            print(f"  Arg Ref Acc: {metrics['arg_ref_acc']:.2%}")
            print(f"  DAG Match: {metrics['dag_isomorphic']}")
            print()

    # Aggregate metrics
    total_samples = len(all_metrics)

    avg_task_recall = sum(m["task_recall"] for m in all_metrics) / total_samples
    avg_task_precision = sum(m["task_precision"] for m in all_metrics) / total_samples
    avg_dep_recall = sum(m["dep_recall"] for m in all_metrics) / total_samples
    avg_dep_precision = sum(m["dep_precision"] for m in all_metrics) / total_samples
    avg_dag_match = sum(m["dag_isomorphic"] for m in all_metrics) / total_samples
    avg_arg_ref = sum(m["arg_ref_acc"] for m in all_metrics) / total_samples

    # Print report
    print("=" * 50)
    print("Finance Benchmark Evaluation Report")
    print("=" * 50)
    print(f"Total Samples: {total_samples}")
    print()

    # Complexity breakdown
    print("[Complexity Breakdown]")
    for complexity in ["shallow", "medium", "deep"]:
        metrics_list = by_complexity.get(complexity, [])
        if metrics_list:
            dag_match = sum(m["dag_isomorphic"] for m in metrics_list) / len(metrics_list)
            dep_recall = sum(m["dep_recall"] for m in metrics_list) / len(metrics_list)
            arg_acc = sum(m["arg_ref_acc"] for m in metrics_list) / len(metrics_list)
            print(f"  {complexity} ({len(metrics_list)}): DAG Match {dag_match:.0%}, Dep Recall {dep_recall:.0%}, Arg Acc {arg_acc:.0%}")
    print()

    # Overall metrics
    print("[Overall Metrics]")
    print(f"  Task Recall:    {avg_task_recall:.2f}")
    print(f"  Task Precision: {avg_task_precision:.2f}")
    print(f"  Dep Recall:     {avg_dep_recall:.2f}")
    print(f"  Dep Precision:  {avg_dep_precision:.2f}")
    print(f"  Arg Ref Acc:    {avg_arg_ref:.2f}")
    print(f"  DAG Isomorphism: {avg_dag_match:.2f}")


if __name__ == "__main__":
    main()
