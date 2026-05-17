#!/usr/bin/env python3
"""验证LLMCompiler生成的DAG和参数传递正确性。

Usage:
    uv run python validate_test_data.py
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TaskInfo:
    """解析后的任务信息。"""
    idx: int
    name: str
    args: List[Any]
    dependencies: List[int]


@dataclass
class SampleInfo:
    """样本信息。"""
    sample_id: str
    benchmark: str
    question: str
    tasks: List[TaskInfo] = field(default_factory=list)
    raw_plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Issue:
    """问题记录。"""
    sample_id: str
    benchmark: str
    severity: str  # "error" or "warning"
    issue_type: str
    message: str
    location: Optional[Dict[str, Any]] = None


def detect_cycle(tasks: List[TaskInfo]) -> Tuple[bool, Optional[List[int]]]:
    """使用Kahn算法检测DAG中的循环依赖。

    Returns:
        Tuple[bool, Optional[List[int]]]: (是否存在循环, 循环路径或None)
    """
    if not tasks:
        return False, None

    # 构建邻接表和入度表
    task_indices = {t.idx for t in tasks}
    in_degree = {idx: 0 for idx in task_indices}
    graph = defaultdict(list)

    for task in tasks:
        for dep in task.dependencies:
            if dep in task_indices:
                graph[dep].append(task.idx)
                in_degree[task.idx] += 1

    # Kahn算法
    queue = [idx for idx in task_indices if in_degree[idx] == 0]
    sorted_tasks = []

    while queue:
        node = queue.pop(0)
        sorted_tasks.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # 如果排序后的任务数小于总任务数，说明存在循环
    if len(sorted_tasks) < len(task_indices):
        # 找到循环中的节点
        cycle_nodes = [idx for idx in task_indices if idx not in sorted_tasks]
        return True, cycle_nodes

    return False, None


def validate_dag(sample: SampleInfo) -> List[Issue]:
    """验证DAG结构。

    检查项:
    1. 无循环依赖
    2. 所有依赖引用的任务存在
    3. 依赖编号小于当前任务编号（避免前向引用）
    """
    issues = []
    task_indices = {t.idx for t in sample.tasks}

    # 检查循环依赖
    has_cycle, cycle_path = detect_cycle(sample.tasks)
    if has_cycle:
        issues.append(Issue(
            sample_id=sample.sample_id,
            benchmark=sample.benchmark,
            severity="error",
            issue_type="cycle_dependency",
            message=f"检测到循环依赖，涉及任务: {cycle_path}",
        ))

    # 检查依赖引用
    for task in sample.tasks:
        for dep in task.dependencies:
            if dep not in task_indices:
                issues.append(Issue(
                    sample_id=sample.sample_id,
                    benchmark=sample.benchmark,
                    severity="error",
                    issue_type="invalid_reference",
                    message=f"任务{task.idx}引用了不存在的依赖${dep}",
                    location={"task_idx": task.idx},
                ))
            elif dep >= task.idx:
                issues.append(Issue(
                    sample_id=sample.sample_id,
                    benchmark=sample.benchmark,
                    severity="warning",
                    issue_type="forward_reference",
                    message=f"任务{task.idx}前向引用了${dep}（依赖编号应小于当前编号）",
                    location={"task_idx": task.idx},
                ))

    return issues


def extract_references(args: List[Any]) -> Set[int]:
    """从参数列表中提取所有$id和${id}引用。"""
    references = set()

    def extract_from_value(value: Any) -> None:
        if isinstance(value, str):
            matches = ID_PATTERN.findall(value)
            references.update(int(m) for m in matches)
        elif isinstance(value, (list, tuple)):
            for item in value:
                extract_from_value(item)

    for arg in args:
        extract_from_value(arg)

    return references


def validate_parameter_passing(sample: SampleInfo) -> List[Issue]:
    """验证参数传递正确性。

    检查项:
    1. $id和${id}语法正确
    2. 引用的任务ID在有效范围内
    3. 参数引用与依赖列表一致
    """
    issues = []
    task_indices = {t.idx for t in sample.tasks}

    for task in sample.tasks:
        if task.name == "join":
            continue  # join任务不需要参数验证

        # 提取参数中的引用
        arg_refs = extract_references(task.args)

        # 检查引用是否有效
        for ref in arg_refs:
            if ref not in task_indices:
                issues.append(Issue(
                    sample_id=sample.sample_id,
                    benchmark=sample.benchmark,
                    severity="error",
                    issue_type="invalid_arg_reference",
                    message=f"任务{task.idx}的参数引用了不存在的${ref}",
                    location={"task_idx": task.idx},
                ))

        # 检查参数引用与依赖列表是否一致
        dep_set = set(task.dependencies)

        # 参数引用应该是依赖的子集
        missing_deps = arg_refs - dep_set
        if missing_deps:
            issues.append(Issue(
                sample_id=sample.sample_id,
                benchmark=sample.benchmark,
                severity="warning",
                issue_type="inconsistent_dependencies",
                message=f"任务{task.idx}的参数引用了{missing_deps}，但这些不在依赖列表中",
                location={"task_idx": task.idx},
            ))

    return issues


def calculate_max_parallelism(tasks: List[TaskInfo]) -> int:
    """计算DAG的最大并行度。

    使用拓扑排序，统计每层可并行执行的任务数。
    """
    if not tasks:
        return 0

    task_indices = {t.idx for t in tasks}
    in_degree = {idx: 0 for idx in task_indices}
    graph = defaultdict(list)

    for task in tasks:
        for dep in task.dependencies:
            if dep in task_indices:
                graph[dep].append(task.idx)
                in_degree[task.idx] += 1

    max_parallelism = 0
    current_level = [idx for idx in task_indices if in_degree[idx] == 0]
    visited = set()

    while current_level:
        max_parallelism = max(max_parallelism, len(current_level))
        visited.update(current_level)

        next_level = []
        for node in current_level:
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0 and neighbor not in visited:
                    next_level.append(neighbor)

        current_level = next_level

    return max_parallelism


def calculate_dependency_depth(tasks: List[TaskInfo]) -> int:
    """计算DAG的依赖深度（关键路径长度）。

    如果存在循环依赖，返回0。
    """
    if not tasks:
        return 0

    # 先检测循环依赖，如果有则无法计算深度
    has_cycle, _ = detect_cycle(tasks)
    if has_cycle:
        return 0

    task_dict = {t.idx: t for t in tasks}
    task_indices = set(task_dict.keys())

    # 使用动态规划计算每个任务的最大深度
    depth = {}

    def get_depth(idx: int) -> int:
        if idx in depth:
            return depth[idx]

        task = task_dict.get(idx)
        if not task:
            return 0

        if not task.dependencies:
            depth[idx] = 1
        else:
            valid_deps = [d for d in task.dependencies if d in task_indices]
            depth[idx] = 1 + max((get_depth(d) for d in valid_deps), default=0)

        return depth[idx]

    for task in tasks:
        get_depth(task.idx)

    return max(depth.values()) if depth else 0


@dataclass
class ValidationResult:
    """验证结果。"""
    summary: Dict[str, Any]
    issues: List[Issue]
    samples: List[Dict[str, Any]]


def generate_report(samples: List[SampleInfo]) -> ValidationResult:
    """生成验证报告。"""
    all_issues: List[Issue] = []
    sample_results = []

    # 按基准测试统计
    by_benchmark = defaultdict(lambda: {"samples": 0, "issues": 0})
    total_parallelism = 0
    total_depth = 0

    for sample in samples:
        # 验证DAG
        dag_issues = validate_dag(sample)
        # 验证参数传递
        param_issues = validate_parameter_passing(sample)

        all_issues.extend(dag_issues)
        all_issues.extend(param_issues)

        # 计算统计
        parallelism = calculate_max_parallelism(sample.tasks)
        depth = calculate_dependency_depth(sample.tasks)
        total_parallelism += parallelism
        total_depth += depth

        # 记录样本结果
        sample_results.append({
            "sample_id": sample.sample_id,
            "benchmark": sample.benchmark,
            "question": sample.question,
            "task_count": len(sample.tasks),
            "max_parallelism": parallelism,
            "dependency_depth": depth,
            "issues_count": len(dag_issues) + len(param_issues),
            "tasks": [
                {
                    "idx": t.idx,
                    "name": t.name,
                    "args": t.args,
                    "dependencies": t.dependencies,
                }
                for t in sample.tasks
            ],
        })

        # 更新统计
        by_benchmark[sample.benchmark]["samples"] += 1
        by_benchmark[sample.benchmark]["issues"] += len(dag_issues) + len(param_issues)

    # 汇总统计
    summary = {
        "total_samples": len(samples),
        "issues_found": len(all_issues),
        "by_benchmark": dict(by_benchmark),
        "avg_parallelism": round(total_parallelism / len(samples), 2) if samples else 0,
        "avg_dependency_depth": round(total_depth / len(samples), 2) if samples else 0,
    }

    return ValidationResult(
        summary=summary,
        issues=all_issues,
        samples=sample_results,
    )


def save_json_report(result: ValidationResult, output_path: Path) -> None:
    """保存JSON报告。"""
    report = {
        "summary": result.summary,
        "issues": [
            {
                "sample_id": i.sample_id,
                "benchmark": i.benchmark,
                "severity": i.severity,
                "type": i.issue_type,
                "message": i.message,
                "location": i.location,
            }
            for i in result.issues
        ],
        "samples": result.samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"JSON报告已保存到: {output_path}")


def save_markdown_report(result: ValidationResult, output_path: Path) -> None:
    """保存Markdown报告。"""
    lines = ["# DAG验证报告\n"]

    # 概览
    lines.append("## 概览\n")
    lines.append(f"- 总样本数: {result.summary['total_samples']}")
    lines.append(f"- 问题数: {result.summary['issues_found']}")
    lines.append(f"- 平均并行度: {result.summary['avg_parallelism']}")
    lines.append(f"- 平均依赖深度: {result.summary['avg_dependency_depth']}\n")

    # 按基准测试统计
    lines.append("### 按基准测试统计\n")
    lines.append("| 基准测试 | 样本数 | 问题数 |")
    lines.append("|----------|--------|--------|")
    for benchmark, stats in result.summary["by_benchmark"].items():
        lines.append(f"| {benchmark} | {stats['samples']} | {stats['issues']} |")
    lines.append("")

    # 问题列表
    if result.issues:
        lines.append("## 问题列表\n")

        errors = [i for i in result.issues if i.severity == "error"]
        warnings = [i for i in result.issues if i.severity == "warning"]

        if errors:
            lines.append(f"### 严重问题 ({len(errors)})\n")
            lines.append("| 样本ID | 基准测试 | 问题类型 | 描述 |")
            lines.append("|--------|----------|----------|------|")
            for issue in errors:
                lines.append(f"| {issue.sample_id} | {issue.benchmark} | {issue.issue_type} | {issue.message} |")
            lines.append("")

        if warnings:
            lines.append(f"### 警告 ({len(warnings)})\n")
            lines.append("| 样本ID | 基准测试 | 问题类型 | 描述 |")
            lines.append("|--------|----------|----------|------|")
            for issue in warnings:
                lines.append(f"| {issue.sample_id} | {issue.benchmark} | {issue.issue_type} | {issue.message} |")
            lines.append("")
    else:
        lines.append("## 问题列表\n\n所有样本验证通过，未发现问题。\n")

    # 样本详情（只显示有问题的样本）
    problem_samples = [s for s in result.samples if s["issues_count"] > 0]
    if problem_samples:
        lines.append("## 有问题的样本\n")
        for sample in problem_samples[:10]:  # 最多显示10个
            lines.append(f"### {sample['benchmark']} - 样本 {sample['sample_id']}\n")
            lines.append(f"**问题:** {sample['question'][:100]}...\n")
            lines.append("**DAG结构:**")
            lines.append("```")
            for task in sample["tasks"]:
                deps = f"依赖{task['dependencies']}" if task['dependencies'] else "无依赖"
                args_str = str(task['args'])[:50]
                lines.append(f"[{task['idx']}] {task['name']}({args_str}) → {deps}")
            lines.append("```\n")
            lines.append(f"**问题数:** {sample['issues_count']}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown报告已保存到: {output_path}")


# 正则表达式：匹配 $id 和 ${id}
ID_PATTERN = re.compile(r"\$\{?(\d+)\}?")


def parse_task_from_plan(task_data: Dict[str, Any]) -> TaskInfo:
    """从plan字段解析单个任务。"""
    return TaskInfo(
        idx=task_data["idx"],
        name=task_data["name"],
        args=list(task_data["args"]) if isinstance(task_data["args"], (list, tuple)) else [task_data["args"]],
        dependencies=list(task_data["dependencies"]),
    )


def parse_sample(sample_id: str, sample_data: Dict[str, Any], benchmark: str) -> SampleInfo:
    """解析单个样本的数据。"""
    plan = sample_data.get("plan", {})
    tasks = []

    for task_data in plan.get("tasks", []):
        tasks.append(parse_task_from_plan(task_data))

    return SampleInfo(
        sample_id=sample_id,
        benchmark=benchmark,
        question=sample_data.get("question", ""),
        tasks=tasks,
        raw_plan=plan,
    )


def load_results(result_files: Dict[str, Path]) -> List[SampleInfo]:
    """加载所有结果文件。"""
    samples = []

    for benchmark, file_path in result_files.items():
        if not file_path.exists():
            print(f"警告: 文件不存在 {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample_id, sample_data in data.items():
            samples.append(parse_sample(sample_id, sample_data, benchmark))

    return samples


def main():
    """主函数。"""
    # 定义结果文件路径
    result_files = {
        "hotpotqa": Path("results/test_hotpotqa.json"),
        "movie": Path("results/test_movie.json"),
        "parallelqa": Path("results/test_parallelqa.json"),
    }

    # 加载数据
    print("加载测试数据...")
    samples = load_results(result_files)
    print(f"共加载 {len(samples)} 个样本")

    if not samples:
        print("错误: 没有找到测试数据，请先运行 collect_test_data.sh")
        return

    # 验证并生成报告
    print("正在验证...")
    result = generate_report(samples)

    # 输出摘要
    print("\n=== 验证摘要 ===")
    print(f"总样本数: {result.summary['total_samples']}")
    print(f"问题数: {result.summary['issues_found']}")
    print(f"平均并行度: {result.summary['avg_parallelism']}")
    print(f"平均依赖深度: {result.summary['avg_dependency_depth']}")

    # 保存报告
    output_dir = Path("results")
    save_json_report(result, output_dir / "validation_report.json")
    save_markdown_report(result, output_dir / "validation_report.md")

    print("\n验证完成！")


if __name__ == "__main__":
    main()
