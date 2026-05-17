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

    # 后续任务会添加验证逻辑
    print("验证功能将在后续任务中实现")


if __name__ == "__main__":
    main()
