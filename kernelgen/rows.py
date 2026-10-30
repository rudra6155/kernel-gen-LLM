"""CSV row schema shared by the in-process evaluator and the sandbox.

status is one of:
    ok         passed every gate and was timed
    incorrect  ran, but its output failed the correctness gate
    error      failed to load, compile, or run (an exception was raised)
    rejected   violated the candidate policy (see kernelgen.policy)
    timeout    exceeded the wall-clock limit and was killed
    crash      the worker process died without reporting a result
    no_code    a model reply that contained no complete code block defining solution()
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
    "library_desc",
    "library_ms",
    "speedup_vs_library",
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


PROFILE_FIELDS = [
    "task",
    "family",
    "params",
    "dtype",
    "impl",
    "candidate_id",
    "status",
    "bytes_moved",
    "iters",
    "event_ms",
    "kernel_ms_per_iter",
    "kernel_launches_per_iter",
    "effective_gbps",
    "copy_gbps",
    "pct_of_copy_bw",
    "peak_gbps",
    "pct_of_peak",
    "top_kernels",
    "trace_file",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "triton_version",
    "error",
]

# Profile row status is one of the statuses above, plus:
#     suspect    profiled, but kernel time exceeds the CUDA-event time, so the
#                bandwidth figure cannot be trusted (see kernelgen.profiling)


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
        library_desc=task.library_desc,
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


def new_profile_row(
    task: Task,
    impl: str,
    candidate_id: str,
    iters: int,
    env: dict[str, str],
    peak_gbps: float | None = None,
) -> dict[str, Any]:
    """A profile row that has not produced any measurement yet."""
    row: dict[str, Any] = {name: None for name in PROFILE_FIELDS}
    row.update(
        task=task.name,
        family=task.family,
        params=json.dumps(task.params, sort_keys=True),
        dtype=str(task.dtype).removeprefix("torch."),
        impl=impl,
        candidate_id=candidate_id,
        status="error",
        bytes_moved=task.bytes_moved,
        iters=iters,
        peak_gbps=peak_gbps,
        error="",
    )
    row.update(env)
    return row


def fail_profile_row(row: dict[str, Any], status: str, error: str) -> dict[str, Any]:
    row["status"] = status
    row["error"] = one_line(error)
    return row


def write_csv(
    rows: Sequence[dict[str, Any]], path: str | Path, fields: Sequence[str] = FIELDS
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
