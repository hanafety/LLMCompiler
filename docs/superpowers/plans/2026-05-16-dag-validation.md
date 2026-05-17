# DAG Validation 测试数据收集 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 LLMCompiler 添加 DAG 生成准确性和参数传递正确性的验证功能，通过执行失败检测和自动验证规则收集测试数据。

**Architecture:** 两阶段方案 - 先扩展 `run_llm_compiler.py` 记录 plan 详情到 store JSON，再通过独立脚本 `validate_dag.py` 读取 JSON 执行验证并输出统计结果。

**Tech Stack:** Python 3.10+, asyncio, JSON

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/llm_compiler/llm_compiler.py` | 添加 `get_plan_info()` 方法返回 plan 详情 |
| `run_llm_compiler.py` | 在 store JSON 中新增 `plan` 字段 |
| `validate_dag.py` | 新建，独立验证脚本 |

---

### Task 1: 在 LLMCompiler 中添加 plan 信息收集方法

**Files:**
- Modify: `src/llm_compiler/llm_compiler.py`

- [ ] **Step 1: 添加 get_plan_info 方法**

在 `LLMCompiler` 类中添加方法，提取当前 plan 的详细信息：

```python
def get_plan_info(self) -> Dict[str, Any]:
    """Get detailed plan information for validation.

    Returns:
        Dict containing raw_llm_output, tasks, and execution_results.
    """
    return getattr(self, "_plan_info", {})
```

- [ ] **Step 2: 在 _acall 中收集 plan 信息**

在 `_acall` 方法中，在 `task_fetching_unit.schedule()` 之后收集 plan 信息。在第 276 行 `tasks = task_fetching_unit.tasks` 之后添加：

```python
            tasks = task_fetching_unit.tasks

            # Collect plan info for validation (only in first iteration)
            if is_first_iter:
                self._plan_info = {
                    "raw_llm_output": getattr(self.planner, "_last_raw_response", ""),
                    "tasks": [
                        {
                            "idx": task.idx,
                            "name": task.name,
                            "args": list(task.args) if isinstance(task.args, (list, tuple)) else task.args,
                            "dependencies": list(task.dependencies),
                        }
                        for task in tasks.values()
                    ],
                    "execution_results": {
                        str(task.idx): task.observation
                        for task in tasks.values()
                        if task.observation is not None
                    },
                }
```

- [ ] **Step 3: 在 Planner 中记录原始 LLM 输出**

修改 `src/llm_compiler/planner.py` 的 `plan` 方法，在第 262-263 行修改：

```python
    async def plan(
        self, inputs: dict, is_replan: bool, callbacks: Callbacks = None, **kwargs: Any
    ):
        llm_response = await self.run_llm(
            inputs=inputs, is_replan=is_replan, callbacks=callbacks
        )
        llm_response = llm_response + "\n"
        self._last_raw_response = llm_response  # Store for plan info collection
        return self.output_parser.parse(llm_response)
```

同样在 `aplan` 方法第 282 行之后添加：

```python
        await self.run_llm(inputs=inputs, is_replan=is_replan, callbacks=all_callbacks)
        self._last_raw_response = ""  # Streaming mode, raw output not captured
```

- [ ] **Step 4: Commit**

```bash
git add src/llm_compiler/llm_compiler.py src/llm_compiler/planner.py
git commit -m "$(cat <<'EOF'
feat: add plan info collection for DAG validation

- Add get_plan_info() method to LLMCompiler
- Store raw LLM output, task details, and execution results
- Record last raw response in Planner for non-streaming mode

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 在 run_llm_compiler.py 中记录 plan 信息

**Files:**
- Modify: `run_llm_compiler.py`

- [ ] **Step 1: 在结果中添加 plan 字段**

修改 `run_llm_compiler.py` 第 221-236 行，在 `all_results[id]` 赋值后添加 plan 信息：

找到这段代码（约第 221-226 行）：
```python
            all_results[id] = {
                "question": question,
                "label": _label,  # not normalized
                "answer": raw_answer,  # not normalized
                "time": e2e_time,
            }
```

修改为：
```python
            all_results[id] = {
                "question": question,
                "label": _label,  # not normalized
                "answer": raw_answer,  # not normalized
                "time": e2e_time,
            }
            # Record plan info for DAG validation
            if not args.react and hasattr(agent, "get_plan_info"):
                all_results[id]["plan"] = agent.get_plan_info()
```

- [ ] **Step 2: 测试数据收集**

运行一个小样本测试验证 plan 信息被正确记录：

```bash
python run_llm_compiler.py --benchmark_name hotpotqa --store test_plan.json --N 1 --do_benchmark
```

检查 `test_plan.json` 中是否包含 `plan` 字段。

- [ ] **Step 3: Commit**

```bash
git add run_llm_compiler.py
git commit -m "$(cat <<'EOF'
feat: record plan info in store JSON for validation

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: 创建 validate_dag.py 验证脚本

**Files:**
- Create: `validate_dag.py`

- [ ] **Step 1: 创建验证脚本骨架**

创建 `validate_dag.py`:

```python
#!/usr/bin/env python3
"""Validate DAG generation and parameter passing for LLMCompiler.

This script reads a store JSON file produced by run_llm_compiler.py
and validates:
1. Dependency existence: $n references a task that exists
2. Dependency ordering: $n references a task with idx < current task idx
3. Parameter substitution: execution completed without substitution errors

Usage:
    python validate_dag.py --file results.json [--output validation_failures.json]
"""

import argparse
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ValidationError:
    """Represents a validation error found in a sample."""
    sample_id: str
    task_idx: int
    error_type: str  # "missing_dependency", "invalid_order", "substitution_failed"
    message: str
    details: Optional[Dict[str, Any]] = None


def validate_dependencies(tasks: List[Dict[str, Any]]) -> List[ValidationError]:
    """Validate dependency existence and ordering.

    Args:
        tasks: List of task dicts with idx, name, args, dependencies

    Returns:
        List of validation errors found
    """
    errors = []
    task_ids = {task["idx"] for task in tasks}

    for task in tasks:
        idx = task["idx"]
        deps = task.get("dependencies", [])

        for dep in deps:
            # Check dependency existence
            if dep not in task_ids:
                errors.append(ValidationError(
                    sample_id="",  # Will be filled by caller
                    task_idx=idx,
                    error_type="missing_dependency",
                    message=f"Task {idx} depends on non-existent task {dep}",
                    details={"dependency": dep, "available_tasks": list(task_ids)},
                ))
            # Check dependency ordering
            elif dep >= idx:
                errors.append(ValidationError(
                    sample_id="",  # Will be filled by caller
                    task_idx=idx,
                    error_type="invalid_order",
                    message=f"Task {idx} depends on task {dep} which has idx >= current task",
                    details={"current_idx": idx, "dependency_idx": dep},
                ))

    return errors


def check_arg_references(args: Any, task_ids: set, current_idx: int) -> List[Dict[str, Any]]:
    """Check if $n references in args are valid.

    Args:
        args: Task arguments (can be str, list, tuple, or other)
        task_ids: Set of valid task IDs
        current_idx: Current task index

    Returns:
        List of error details
    """
    errors = []
    pattern = r"\$\{?(\d+)\}?"

    if isinstance(args, str):
        matches = re.findall(pattern, args)
        for match in matches:
            dep_idx = int(match)
            if dep_idx not in task_ids:
                errors.append({
                    "referenced_task": dep_idx,
                    "error": "missing",
                })
            elif dep_idx >= current_idx:
                errors.append({
                    "referenced_task": dep_idx,
                    "error": "invalid_order",
                })

    elif isinstance(args, (list, tuple)):
        for item in args:
            errors.extend(check_arg_references(item, task_ids, current_idx))

    return errors


def validate_sample(sample_id: str, sample: Dict[str, Any]) -> List[ValidationError]:
    """Validate a single sample.

    Args:
        sample_id: Sample ID
        sample: Sample dict containing plan info

    Returns:
        List of validation errors
    """
    errors = []

    # Check if plan info exists
    if "plan" not in sample:
        return errors  # Skip samples without plan info

    plan = sample["plan"]
    tasks = plan.get("tasks", [])

    if not tasks:
        return errors  # Skip empty plans

    # Validate dependencies
    dep_errors = validate_dependencies(tasks)
    for err in dep_errors:
        err.sample_id = sample_id
    errors.extend(dep_errors)

    # Validate $n references in args
    task_ids = {task["idx"] for task in tasks}
    for task in tasks:
        idx = task["idx"]
        args = task.get("args", [])
        arg_errors = check_arg_references(args, task_ids, idx)

        for err_detail in arg_errors:
            errors.append(ValidationError(
                sample_id=sample_id,
                task_idx=idx,
                error_type=f"arg_reference_{err_detail['error']}",
                message=f"Task {idx} has invalid $n reference to task {err_detail['referenced_task']}",
                details=err_detail,
            ))

    return errors


def main():
    argparser = argparse.ArgumentParser(description="Validate DAG generation and parameter passing")
    argparser.add_argument("--file", type=str, required=True, help="Path to store JSON file")
    argparser.add_argument("--output", type=str, default="validation_failures.json", help="Output file for failures")
    args = argparser.parse_args()

    # Load data
    with open(args.file, "r") as f:
        data = json.load(f)

    # Validate all samples
    all_errors = []
    samples_with_plan = 0
    samples_without_plan = 0

    for sample_id, sample in data.items():
        if "plan" not in sample:
            samples_without_plan += 1
            continue

        samples_with_plan += 1
        errors = validate_sample(sample_id, sample)
        all_errors.extend(errors)

    # Calculate statistics
    samples_with_errors = len(set(err.sample_id for err in all_errors))

    # Print summary
    print("=" * 60)
    print("DAG Validation Report")
    print("=" * 60)
    print(f"Total samples: {len(data)}")
    print(f"Samples with plan info: {samples_with_plan}")
    print(f"Samples without plan info: {samples_without_plan}")
    print()

    # Dependency validation stats
    dep_errors = [e for e in all_errors if e.error_type in ("missing_dependency", "invalid_order")]
    arg_ref_errors = [e for e in all_errors if e.error_type.startswith("arg_reference_")]

    print("Dependency Validation:")
    print(f"  - Total errors: {len(dep_errors)}")
    print(f"  - Missing dependencies: {len([e for e in dep_errors if e.error_type == 'missing_dependency'])}")
    print(f"  - Invalid ordering: {len([e for e in dep_errors if e.error_type == 'invalid_order'])}")
    print()

    print("Argument Reference Validation:")
    print(f"  - Total errors: {len(arg_ref_errors)}")
    print(f"  - Missing references: {len([e for e in arg_ref_errors if 'missing' in e.error_type])}")
    print(f"  - Invalid order references: {len([e for e in arg_ref_errors if 'invalid_order' in e.error_type])}")
    print()

    # Accuracy
    if samples_with_plan > 0:
        dep_accuracy = (samples_with_plan - len(set(e.sample_id for e in dep_errors))) / samples_with_plan * 100
        arg_accuracy = (samples_with_plan - len(set(e.sample_id for e in arg_ref_errors))) / samples_with_plan * 100
        overall_accuracy = (samples_with_plan - samples_with_errors) / samples_with_plan * 100

        print("Accuracy:")
        print(f"  - Dependency correctness: {dep_accuracy:.2f}%")
        print(f"  - Argument reference correctness: {arg_accuracy:.2f}%")
        print(f"  - Overall DAG correctness: {overall_accuracy:.2f}%")
    print()

    # Save failures
    if all_errors:
        failures = [
            {
                "sample_id": err.sample_id,
                "task_idx": err.task_idx,
                "error_type": err.error_type,
                "message": err.message,
                "details": err.details,
            }
            for err in all_errors
        ]
        with open(args.output, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"Failures saved to: {args.output}")
    else:
        print("No validation errors found!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证脚本可以运行**

```bash
python validate_dag.py --help
```

期望输出：脚本帮助信息

- [ ] **Step 3: Commit**

```bash
git add validate_dag.py
git commit -m "$(cat <<'EOF'
feat: add validate_dag.py for DAG validation

- Validate dependency existence and ordering
- Validate $n argument references
- Output accuracy statistics and failure details

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: 端到端测试

**Files:**
- None (testing only)

- [ ] **Step 1: 收集测试数据**

运行 hotpotqa benchmark 收集少量样本：

```bash
python run_llm_compiler.py --benchmark_name hotpotqa --store test_validation.json --N 5 --do_benchmark
```

- [ ] **Step 2: 运行验证**

```bash
python validate_dag.py --file test_validation.json --output test_failures.json
```

期望输出：包含统计信息的报告

- [ ] **Step 3: 验证输出格式**

检查 `test_failures.json` 格式是否正确（如果有错误的话）：

```bash
cat test_failures.json
```

- [ ] **Step 4: 清理测试文件**

```bash
rm -f test_validation.json test_failures.json test_plan.json
```

---

### Task 5: 运行完整 benchmark 收集数据

**Files:**
- None (execution only)

- [ ] **Step 1: 运行 hotpotqa benchmark**

```bash
python run_llm_compiler.py --benchmark_name hotpotqa --store results_hotpotqa.json --do_benchmark
```

- [ ] **Step 2: 验证 hotpotqa 结果**

```bash
python validate_dag.py --file results_hotpotqa.json --output failures_hotpotqa.json
```

- [ ] **Step 3: 运行 movie benchmark**

```bash
python run_llm_compiler.py --benchmark_name movie --store results_movie.json --do_benchmark
```

- [ ] **Step 4: 验证 movie 结果**

```bash
python validate_dag.py --file results_movie.json --output failures_movie.json
```

- [ ] **Step 5: 运行 parallelqa benchmark**

```bash
python run_llm_compiler.py --benchmark_name parallelqa --store results_parallelqa.json --do_benchmark
```

- [ ] **Step 6: 验证 parallelqa 结果**

```bash
python validate_dag.py --file results_parallelqa.json --output failures_parallelqa.json
```

---

## 预期产出

运行完成后，你将获得：

1. **三个 benchmark 的原始结果文件**：
   - `results_hotpotqa.json`
   - `results_movie.json`
   - `results_parallelqa.json`

2. **三个验证失败详情文件**：
   - `failures_hotpotqa.json`
   - `failures_movie.json`
   - `failures_parallelqa.json`

3. **终端输出的统计报告**，包含：
   - 依赖正确率
   - 参数引用正确率
   - 整体 DAG 正确率
