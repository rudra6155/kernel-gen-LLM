"""Benchmark harness: gate candidates for correctness, time them, write a CSV.

`evaluate` is the measurement core. It runs in the calling process and does not
enforce the candidate policy, so use it only for trusted code. Untrusted
(generated) code goes through kernelgen.sandbox.evaluate_isolated, which checks
the policy and runs `evaluate` in a worker process under a wall-clock limit.

Usage:
    python -m kernelgen.benchmark --out results/kernel_benchmarks.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from kernelgen.candidate import load_candidate
from kernelgen.correctness import check_correctness
from kernelgen.policy import PolicyViolation
from kernelgen.reward import compute_reward
from kernelgen.rows import FIELDS, environment_info, fail_row, new_row, one_line, write_csv
from kernelgen.sandbox import DEFAULT_TIMEOUT_S, evaluate_isolated
from kernelgen.tasks import TIMING_SEED, Task, default_tasks
from kernelgen.timing import benchmark_pair, percentile

__all__ = ["FIELDS", "KNOWN_KERNELS", "evaluate", "main", "run_known_kernels", "write_csv"]

KNOWN_DIR = Path(__file__).parent / "known_kernels"
# family -> (candidate_id, source file)
KNOWN_KERNELS = {
    "layernorm": ("known_layernorm_triton", KNOWN_DIR / "layernorm_triton.py"),
}


def _time_against_library(task, fn, inputs, device, warmup, reps, row) -> None:
    """Time the candidate against PyTorch's own optimized op (a second interleaved run).

    A failure here does not fail the candidate; it leaves the library columns
    empty and says why in the error column.
    """
    try:
        timing = benchmark_pair(
            lambda: task.library(*inputs), lambda: fn(*inputs), device, warmup=warmup, reps=reps
        )
        row["library_ms"] = timing.baseline_median
        row["speedup_vs_library"] = timing.speedup
    except Exception as exc:
        row["error"] = one_line(f"library timing failed: {type(exc).__name__}: {exc}")


def evaluate(
    task: Task,
    source: str,
    candidate_id: str,
    device: str,
    warmup: int = 10,
    reps: int = 100,
    seeds: Sequence[int] = (0, 1, 2),
    env: dict[str, str] | None = None,
    forbid_native_ops: bool = False,
) -> dict[str, Any]:
    """Evaluate one candidate on one task in this process and return one CSV row.

    status is "ok" (passed the gate and was timed), "incorrect" (ran but failed
    the gate), "rejected" (used native PyTorch ops while `forbid_native_ops` was
    set), or "error" (failed to load, compile, or run).
    """
    row = new_row(
        task, candidate_id, warmup, reps, len(seeds), env if env is not None else environment_info(device)
    )

    try:
        with load_candidate(source) as fn:
            gate = check_correctness(task, fn, device, seeds, forbid_native_ops=forbid_native_ops)
            row["max_abs_err"] = gate.max_abs_err
            row["correct"] = int(gate.correct)
            if not gate.correct:
                row["status"] = "incorrect"
                row["error"] = one_line(gate.reason)
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
                if task.library is not None:
                    _time_against_library(task, fn, inputs, device, warmup, reps, row)
    except PolicyViolation as exc:
        return fail_row(row, "rejected", str(exc))
    except Exception as exc:
        # correct is left as the gate set it: a candidate that passed the gate
        # and then failed while being timed still passed the gate.
        row["status"] = "error"
        row["error"] = one_line(f"{type(exc).__name__}: {exc}")

    row["reward"] = compute_reward(row["speedup"], bool(row["correct"]))
    return row


def run_known_kernels(
    device: str,
    warmup: int,
    reps: int,
    seeds: Sequence[int],
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> list[dict[str, Any]]:
    env = environment_info(device)
    rows = []
    for task in default_tasks():
        candidate_id, path = KNOWN_KERNELS[task.family]
        rows.append(
            evaluate_isolated(
                task,
                path.read_text(),
                candidate_id,
                device,
                warmup=warmup,
                reps=reps,
                seeds=seeds,
                env=env,
                timeout_s=timeout_s,
            )
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/kernel_benchmarks.csv")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3, help="number of correctness-gate inputs")
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help="wall-clock limit per candidate in seconds, including process start and kernel compilation",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        print("error: no CUDA device available; refusing to write benchmark results", file=sys.stderr)
        return 2

    device = "cuda"
    rows = run_known_kernels(device, args.warmup, args.reps, range(args.seeds), args.timeout)
    write_csv(rows, args.out)

    for r in rows:
        speedup = "-" if r["speedup"] is None else f"{r['speedup']:.2f}x"
        print(f"{r['task']:<32} {r['candidate_id']:<24} {r['status']:<10} {speedup:>8}  {r['error']}")
    print(f"wrote {len(rows)} rows to {args.out} on {rows[0]['gpu_name']}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
