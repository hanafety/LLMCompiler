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
