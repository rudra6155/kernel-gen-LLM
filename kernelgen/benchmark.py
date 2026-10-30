"""Benchmark harness: gate candidates for correctness, time them, write a CSV.

Candidates are evaluated in-process. A CUDA error such as an illegal memory
access leaves the context unusable, so every later row in the same run will
also report an error.

Usage:
    python -m kernelgen.benchmark --out results/kernel_benchmarks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from kernelgen.candidate import load_candidate
from kernelgen.correctness import check_correctness
from kernelgen.reward import compute_reward
from kernelgen.tasks import Task, default_tasks
from kernelgen.timing import benchmark_pair, percentile

TIMING_SEED = 1234

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

KNOWN_DIR = Path(__file__).parent / "known_kernels"
# family -> (candidate_id, source file)
KNOWN_KERNELS = {
    "layernorm": ("known_layernorm_triton", KNOWN_DIR / "layernorm_triton.py"),
}


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


def _one_line(exc: BaseException, limit: int = 300) -> str:
    return " ".join(f"{type(exc).__name__}: {exc}".split())[:limit]


def evaluate(
    task: Task,
    source: str,
    candidate_id: str,
    device: str,
    warmup: int = 10,
    reps: int = 100,
    seeds: Sequence[int] = (0, 1, 2),
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate on one task and return one CSV row.

    status is "ok" (passed the gate and was timed), "incorrect" (ran but
    failed the gate), or "error" (failed to load, compile, or run).
    """
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
        seeds=len(seeds),
        error="",
    )
    row.update(env if env is not None else environment_info(device))

    try:
        with load_candidate(source) as fn:
            gate = check_correctness(task, fn, device, seeds)
            row["max_abs_err"] = gate.max_abs_err
            row["correct"] = int(gate.correct)
            if not gate.correct:
                row["status"] = "incorrect"
                row["error"] = gate.reason
            else:
                inputs = task.make_inputs(TIMING_SEED, device)
                timing = benchmark_pair(
                    lambda: task.reference(*inputs),
                    lambda: fn(*inputs),
                    device,
                    warmup=warmup,
                    reps=reps,
                )
                row.update(
                    status="ok",
                    baseline_ms=timing.baseline_median,
                    baseline_ms_p10=percentile(timing.baseline_ms, 0.1),
                    baseline_ms_p90=percentile(timing.baseline_ms, 0.9),
                    candidate_ms=timing.candidate_median,
                    candidate_ms_p10=percentile(timing.candidate_ms, 0.1),
                    candidate_ms_p90=percentile(timing.candidate_ms, 0.9),
                    speedup=timing.speedup,
                )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = _one_line(exc)

    row["reward"] = compute_reward(row["speedup"], bool(row["correct"]))
    return row


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def run_known_kernels(
    device: str, warmup: int, reps: int, seeds: Sequence[int]
) -> list[dict[str, Any]]:
    env = environment_info(device)
    rows = []
    for task in default_tasks():
        candidate_id, path = KNOWN_KERNELS[task.family]
        rows.append(
            evaluate(task, path.read_text(), candidate_id, device, warmup, reps, seeds, env)
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/kernel_benchmarks.csv")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3, help="number of correctness-gate inputs")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("error: no CUDA device available; refusing to write benchmark results", file=sys.stderr)
        return 2

    device = "cuda"
    rows = run_known_kernels(device, args.warmup, args.reps, range(args.seeds))
    write_csv(rows, args.out)

    for r in rows:
        speedup = "-" if r["speedup"] is None else f"{r['speedup']:.2f}x"
        print(f"{r['task']:<32} {r['candidate_id']:<24} {r['status']:<10} {speedup:>8}  {r['error']}")
    print(f"wrote {len(rows)} rows to {args.out} on {rows[0]['gpu_name']}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
