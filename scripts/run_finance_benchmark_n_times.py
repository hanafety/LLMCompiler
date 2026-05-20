#!/usr/bin/env python3
"""Run finance benchmark N times in parallel and produce cross-run comparison report.

Usage:
    HTTP_PROXY=http://127.0.0.1:10808 HTTPS_PROXY=http://127.0.0.1:10808 \
    uv run python scripts/run_finance_benchmark_n_times.py \
        --n 10 \
        --model_name deepseek-v4-flash \
        --output_dir results/finance_runs \
        --workers 5
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluate_finance import evaluate_sample

# Thread-safe print lock
print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def run_single_benchmark(
    run_id: int,
    output_path: str,
    model_name: str,
    n_samples: int = None,
) -> Optional[Dict[str, Any]]:
    """Run a single benchmark and return results.

    Each run writes to its own output file to avoid conflicts.
    """
    cmd = [
        "uv", "run", "python", "run_llm_compiler.py",
        "--benchmark_name", "finance",
        "--store", output_path,  # Each run has unique output file
        "--stream",
        "--model_name", model_name,
    ]
    if n_samples:
        cmd.extend(["--N", str(n_samples)])

    safe_print(f"[Run {run_id}] Starting...")

    env = os.environ.copy()
    # Run in project root directory
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).parent.parent
    )

    if result.returncode != 0:
        safe_print(f"[Run {run_id}] FAILED!")
        safe_print(f"[Run {run_id}] STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
        return None

    # Load and return results from the unique output file
    try:
        with open(output_path, "r") as f:
            results = json.load(f)
        safe_print(f"[Run {run_id}] Completed with {len(results)} samples")
        return results
    except Exception as e:
        safe_print(f"[Run {run_id}] Error loading results: {e}")
        return None


def evaluate_single_run(results: Dict, dataset: List[Dict]) -> Dict[str, Any]:
    """Evaluate a single run's results."""
    expected_by_id = {str(item["id"]): item for item in dataset}

    metrics_list = []
    by_complexity = defaultdict(list)

    for sample_id, generated in results.items():
        expected = expected_by_id.get(str(sample_id))
        if expected is None:
            continue

        metrics = evaluate_sample(generated, expected)
        metrics["id"] = sample_id
        metrics_list.append(metrics)
        by_complexity[metrics["complexity"]].append(metrics)

    # Aggregate
    n_samples = len(metrics_list)
    if n_samples == 0:
        return {}

    agg = {
        "task_recall": float(np.mean([m["task_recall"] for m in metrics_list])),
        "task_precision": float(np.mean([m["task_precision"] for m in metrics_list])),
        "dep_recall": float(np.mean([m["dep_recall"] for m in metrics_list])),
        "dep_precision": float(np.mean([m["dep_precision"] for m in metrics_list])),
        "arg_ref_acc": float(np.mean([m["arg_ref_acc"] for m in metrics_list])),
        "dag_isomorphism": float(np.mean([m["dag_isomorphic"] for m in metrics_list])),
        "latency_mean": float(np.mean([r.get("time", 0) for r in results.values()])),
    }

    return {
        "aggregate": agg,
        "by_complexity": {
            k: {
                "dag_isomorphism": float(np.mean([m["dag_isomorphic"] for m in v])),
                "dep_recall": float(np.mean([m["dep_recall"] for m in v]))
            }
            for k, v in by_complexity.items()
        },
        "n_samples": n_samples,
    }


def compute_comparison_report(run_metrics: List[Optional[Dict]], dataset: List[Dict]) -> Dict[str, Any]:
    """Compute cross-run comparison statistics."""
    # Filter out failed runs
    valid_metrics = [m for m in run_metrics if m is not None]

    if not valid_metrics:
        return {"aggregate": {}, "by_complexity": {}, "per_run_summary": []}

    # Extract per-run aggregates
    task_recalls = [m["aggregate"]["task_recall"] for m in valid_metrics]
    dep_recalls = [m["aggregate"]["dep_recall"] for m in valid_metrics]
    dep_precisions = [m["aggregate"]["dep_precision"] for m in valid_metrics]
    arg_ref_accs = [m["aggregate"]["arg_ref_acc"] for m in valid_metrics]
    dag_isos = [m["aggregate"]["dag_isomorphism"] for m in valid_metrics]
    latencies = [m["aggregate"]["latency_mean"] for m in valid_metrics]

    def stats(arr):
        if not arr:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    report = {
        "aggregate": {
            "task_recall": stats(task_recalls),
            "dep_recall": stats(dep_recalls),
            "dep_precision": stats(dep_precisions),
            "arg_ref_acc": stats(arg_ref_accs),
            "dag_isomorphism": stats(dag_isos),
            "latency_mean": stats(latencies),
        },
    }

    # By complexity
    for complexity in ["shallow", "medium", "deep"]:
        dag_vals = []
        dep_vals = []
        for m in valid_metrics:
            if complexity in m.get("by_complexity", {}):
                dag_vals.append(m["by_complexity"][complexity]["dag_isomorphism"])
                dep_vals.append(m["by_complexity"][complexity]["dep_recall"])
        if dag_vals:
            report.setdefault("by_complexity", {})[complexity] = {
                "dag_isomorphism": stats(dag_vals),
                "dep_recall": stats(dep_vals),
            }

    # Per-run summary (including failed runs as None)
    report["per_run_summary"] = [
        {"run": i+1, "status": "success" if m else "failed", **(m.get("aggregate", {}) if m else {})}
        for i, m in enumerate(run_metrics)
    ]

    return report


def main():
    parser = argparse.ArgumentParser(description="Run finance benchmark N times in parallel")
    parser.add_argument("--n", type=int, default=10, help="Number of runs")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--output_dir", type=str, default="results/finance_runs", help="Output directory")
    parser.add_argument("--N", type=int, default=None, help="Limit samples per run")
    parser.add_argument("--dataset", type=str, default="datasets/finance_dataset.json", help="Dataset file")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    n_samples = args.N or len(dataset)

    print(f"Starting {args.n} runs of finance benchmark (parallel workers: {args.workers})")
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Samples per run: {n_samples}")
    print(f"{'='*60}")

    all_run_metrics: List[Optional[Dict]] = [None] * args.n  # Pre-allocate to preserve order

    # Run benchmarks in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_run_id = {}
        for run_id in range(1, args.n + 1):
            output_path = output_dir / f"run_{run_id:02d}.json"
            future = executor.submit(
                run_single_benchmark,
                run_id=run_id,
                output_path=str(output_path),
                model_name=args.model_name,
                n_samples=args.N,
            )
            future_to_run_id[future] = run_id

        # Collect results as they complete
        for future in as_completed(future_to_run_id):
            run_id = future_to_run_id[future]
            try:
                results = future.result()
                if results is not None:
                    metrics = evaluate_single_run(results, dataset)
                    all_run_metrics[run_id - 1] = metrics

                    agg = metrics.get("aggregate", {})
                    safe_print(f"[Run {run_id}] DAG={agg.get('dag_isomorphism', 0):.2f}, "
                              f"DepRecall={agg.get('dep_recall', 0):.2f}, "
                              f"Latency={agg.get('latency_mean', 0):.1f}s")
                else:
                    safe_print(f"[Run {run_id}] Failed - no results")
            except Exception as e:
                safe_print(f"[Run {run_id}] Exception: {e}")

    # Generate comparison report
    report = compute_comparison_report(all_run_metrics, dataset)
    report["config"] = {
        "n_runs": args.n,
        "workers": args.workers,
        "model": args.model_name,
        "n_samples": n_samples,
    }

    # Save report
    report_path = output_dir / "comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON REPORT")
    print("="*60)

    # Count successful runs
    successful = sum(1 for m in all_run_metrics if m is not None)
    print(f"\nSuccessful runs: {successful}/{args.n}")

    agg = report["aggregate"]
    print("\n[Overall Metrics (mean ± std)]")
    for metric in ["task_recall", "dep_recall", "dep_precision", "arg_ref_acc", "dag_isomorphism"]:
        m = agg.get(metric, {"mean": 0, "std": 0})
        print(f"  {metric}: {m['mean']:.2f} ± {m['std']:.2f} (min={m['min']:.2f}, max={m['max']:.2f})")

    if "by_complexity" in report:
        print("\n[By Complexity]")
        for complexity, metrics in report["by_complexity"].items():
            dag = metrics.get("dag_isomorphism", {"mean": 0, "std": 0})
            dep = metrics.get("dep_recall", {"mean": 0, "std": 0})
            print(f"  {complexity}: DAG {dag['mean']:.2f}±{dag['std']:.2f}, Dep {dep['mean']:.2f}±{dep['std']:.2f}")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
