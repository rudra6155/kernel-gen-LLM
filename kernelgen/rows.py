"""CSV row schema shared by the in-process evaluator and the sandbox.

status is one of:
    ok         passed every gate and was timed
    incorrect  ran, but its output failed the correctness gate
    error      failed to load, compile, or run (an exception was raised)
    rejected   violated the candidate policy (see kernelgen.policy)
    timeout    exceeded the wall-clock limit and was killed
    crash      the worker process died without reporting a result
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from kernelgen.reward import compute_reward
from kernelgen.tasks import Task

FIELDS = [
    "task",
    "family",
    "params",
    "dtype",
    "baseline_desc",
    "candidate_id",
    "status",
    "correct",
    "max_abs_err",
    "baseline_ms",
    "baseline_ms_p10",
    "baseline_ms_p90",
    "candidate_ms",
    "candidate_ms_p10",
    "candidate_ms_p90",
    "speedup",
    "reward",
    "warmup",
    "reps",
    "seeds",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "triton_version",
    "error",
]


def environment_info(device: str) -> dict[str, str]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = ""
    return {
        "gpu_name": torch.cuda.get_device_name(device) if device.startswith("cuda") else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "triton_version": triton_version,
    }


def one_line(text: str, limit: int = 300) -> str:
    return " ".join(text.split())[:limit]


def new_row(
    task: Task,
    candidate_id: str,
    warmup: int,
    reps: int,
    n_seeds: int,
    env: dict[str, str],
) -> dict[str, Any]:
    """A row for a candidate that has not produced any result yet."""
    row: dict[str, Any] = {name: None for name in FIELDS}
    row.update(
        task=task.name,
        family=task.family,
        params=json.dumps(task.params, sort_keys=True),
        dtype=str(task.dtype).removeprefix("torch."),
        baseline_desc=task.baseline_desc,
        candidate_id=candidate_id,
        status="error",
        correct=0,
        warmup=warmup,
        reps=reps,
        seeds=n_seeds,
        error="",
    )
    row.update(env)
    row["reward"] = 0.0
    return row


def fail_row(row: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    """Mark `row` as a failure with reward 0."""
    row["status"] = status
    row["correct"] = 0
    row["error"] = one_line(error)
    row["reward"] = compute_reward(None, False)
    return row


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
