# Finance Benchmark Multi-Run Comparison Design

## Goal

Run the finance benchmark N times and produce a cross-run comparison report to analyze stability of DAG generation metrics.

## Architecture

### New File: `scripts/run_finance_benchmark_n_times.py`

Automated runner that:
1. Loops N times, running `run_llm_compiler.py --benchmark_name finance` each iteration
2. Saves each run's results to `results/finance_runs/run_{i}.json`
3. Evaluates each run using `evaluate_finance.py` logic (imported directly)
4. Produces a comparison report with per-sample and aggregate statistics

### Output Structure

```
results/
├── finance_runs/
│   ├── run_01.json
│   ├── run_02.json
│   └── ...
└── finance_comparison_report.json
```

### CLI Interface

```bash
HTTP_PROXY=http://127.0.0.1:10808 HTTPS_PROXY=http://127.0.0.1:10808 \
uv run python scripts/run_finance_benchmark_n_times.py \
  --n 10 \
  --model_name deepseek-v4-flash \
  --output_dir results/finance_runs
```

Parameters:
- `--n`: Number of runs (default: 10)
- `--model_name`: Model to use (required)
- `--output_dir`: Output directory for run results (default: `results/finance_runs`)
- `--N`: Limit number of samples per run (optional, for quick testing)

### Comparison Report Format

```json
{
  "config": {
    "n_runs": 10,
    "model": "deepseek-v4-flash",
    "n_samples": 25
  },
  "per_run_summary": [
    {"run": 1, "task_recall": 0.87, "dep_recall": 0.68, "dag_isomorphism": 0.44, "latency_mean": 15.2},
    ...
  ],
  "aggregate": {
    "task_recall": {"mean": 0.87, "std": 0.05, "min": 0.80, "max": 0.95},
    "dep_recall": {"mean": 0.68, "std": 0.08, "min": 0.55, "max": 0.82},
    "dep_precision": {"mean": 0.59, "std": 0.07, "min": 0.48, "max": 0.70},
    "arg_ref_acc": {"mean": 0.69, "std": 0.06, "min": 0.60, "max": 0.80},
    "dag_isomorphism": {"mean": 0.44, "std": 0.06, "min": 0.35, "max": 0.55}
  },
  "by_complexity": {
    "shallow": {
      "dag_isomorphism": {"mean": 0.70, "std": 0.05},
      "dep_recall": {"mean": 0.90, "std": 0.03}
    },
    "medium": {
      "dag_isomorphism": {"mean": 0.0, "std": 0.0},
      "dep_recall": {"mean": 0.28, "std": 0.10}
    },
    "deep": {
      "dag_isomorphism": {"mean": 0.0, "std": 0.0},
      "dep_recall": {"mean": 0.43, "std": 0.12}
    }
  },
  "by_sample": {
    "1": {
      "dag_match_rate": 0.9,
      "dep_recall_mean": 0.85,
      "dep_recall_std": 0.05,
      "complexity": "shallow"
    }
  }
}
```

### Implementation Approach

- Import `evaluate_sample` and `build_dependency_dict` from `evaluate_finance.py`
- Use `subprocess.run()` to invoke `run_llm_compiler.py` for each iteration (same as manual execution)
- After each run, load results and evaluate using imported functions
- Accumulate metrics across runs, then compute statistics
- Print progress after each run (run number, current metrics)
- Print final comparison table to stdout

### Key Metrics for Comparison

| Metric | Description | Why It Matters |
|--------|-------------|----------------|
| DAG Isomorphism | Whether generated DAG matches expected | Core correctness measure |
| Dep Recall | Fraction of expected dependencies found | Dependency precision |
| Arg Ref Acc | Whether $id references match expected | Parameter passing accuracy |
| Task Recall | Fraction of expected tasks generated | Planning completeness |
| Latency | Average time per question | Performance stability |
