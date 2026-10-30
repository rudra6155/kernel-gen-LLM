"""Benchmark harness: gate candidates for correctness, time them, write a CSV.

`evaluate` is the measurement core. It runs in the calling process and does not
enforce the candidate policy, so use it only for trusted code. Untrusted
(generated) code goes through kernelgen.sandbox.evaluate_isolated, which checks
the policy and runs `evaluate` in a worker process under a wall-clock limit.

`make benchmark` runs the template kernel for each of the 50 held-out tasks and
writes results/kernel_benchmarks.csv: one row per task with the baseline time,
the candidate time, the speedup, and the speedup over PyTorch's own op. These
rows are the model-independent reference. Kernels written by the model are
evaluated on the same tasks by kernelgen.evaluate_model.

Usage:
    python -m kernelgen.benchmark --out results/kernel_benchmarks.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

from kernelgen.candidate import load_candidate
from kernelgen.correctness import check_correctness, expected_output, output_problem
from kernelgen.policy import PolicyViolation
from kernelgen.reward import compute_reward
from kernelgen.rows import FIELDS, environment_info, fail_row, new_row, one_line, write_csv
from kernelgen.sandbox import DEFAULT_TIMEOUT_S, evaluate_isolated
from kernelgen.tasks import TIMING_SEED, Task
from kernelgen.timing import TimingVerificationError, benchmark_fresh, percentile

__all__ = ["FIELDS", "KNOWN_KERNELS", "evaluate", "main", "reference_candidates", "run_reference_kernels", "write_csv"]

KNOWN_DIR = Path(__file__).parent / "known_kernels"
# family -> (candidate_id, source file)
KNOWN_KERNELS = {
    "layernorm": ("known_layernorm_triton", KNOWN_DIR / "layernorm_triton.py"),
}


def _time_against_library(task, fn, make_args, device, warmup, reps, row) -> None:
    """Time the candidate against PyTorch's own optimized op (a second interleaved run).

    A failure here does not fail the candidate; it leaves the library columns
    empty and says why in the error column.
    """
    try:
        timing = benchmark_fresh(make_args, task.library, fn, device, warmup=warmup, reps=reps)
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
                def make_args(i: int) -> tuple:
                    return task.make_inputs(TIMING_SEED + i, device)

                def verify(out: object, args: tuple) -> str | None:
                    return output_problem(task, out, expected_output(task, args))[0]

                timing = benchmark_fresh(
                    make_args, task.reference, fn, device, warmup=warmup, reps=reps, verify=verify
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
                    _time_against_library(task, fn, make_args, device, warmup, reps, row)
    except PolicyViolation as exc:
        return fail_row(row, "rejected", str(exc))
    except TimingVerificationError as exc:
        # It passed the gate on its seeds but is wrong on the data used for timing.
        row["max_abs_err"] = None
        return fail_row(row, "incorrect", str(exc))
    except Exception as exc:
        # correct is left as the gate set it: a candidate that passed the gate
        # and then failed while being timed still passed the gate.
        row["status"] = "error"
        row["error"] = one_line(f"{type(exc).__name__}: {exc}")

    row["reward"] = compute_reward(row["speedup"], bool(row["correct"]))
    return row


def reference_candidates(
    family: str | None = None, sample_only: bool = False
) -> list[tuple[Task, str, str]]:
    """The template kernel (default config) for each held-out task: (task, candidate_id, source).

    These are the model-independent reference points of the benchmark: what a
    hand-written template achieves on the held-out tasks, with no tuning.
    `sample_only` keeps just the first held-out task of each family.
    """
    from kernelgen.families import all_families

    out = []
    for fam in all_families():
        if family and fam.name != family:
            continue
        for params in fam.heldout_shapes[:1] if sample_only else fam.heldout_shapes:
            config = fam.default_config(params)
            candidate_id = f"template:{fam.name}:" + json.dumps(config, sort_keys=True, separators=(",", ":"))
            out.append((fam.task(params), candidate_id, fam.render(params, config)))
    return out


def run_reference_kernels(
    device: str,
    warmup: int,
    reps: int,
    seeds: Sequence[int],
    timeout_s: float = DEFAULT_TIMEOUT_S,
    family: str | None = None,
    limit: int | None = None,
    progress=print,
) -> list[dict[str, Any]]:
    env = environment_info(device)
    rows = []
    candidates = reference_candidates(family)[:limit]
    for i, (task, candidate_id, source) in enumerate(candidates, 1):
        row = evaluate_isolated(
            task, source, candidate_id, device,
            warmup=warmup, reps=reps, seeds=seeds, env=env, timeout_s=timeout_s,
        )
        rows.append(row)
        speedup = "-" if row["speedup"] is None else f"{row['speedup']:.2f}x"
        progress(f"[{i}/{len(candidates)}] {row['task']:<44} {row['status']:<9} {speedup:>8}")
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/kernel_benchmarks.csv")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3, help="number of correctness-gate inputs")
    parser.add_argument("--family", default=None, help="only this family's held-out tasks")
    parser.add_argument("--limit", type=int, default=None, help="only the first N tasks (a quick trial run)")
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

    rows = run_reference_kernels(
        "cuda", args.warmup, args.reps, range(args.seeds), args.timeout, args.family, args.limit
    )
    write_csv(rows, args.out)

    ok = [r for r in rows if r["status"] == "ok"]
    print(f"wrote {len(rows)} rows to {args.out} on {rows[0]['gpu_name']}: {len(ok)} ok")
    for r in rows:
        if r["status"] != "ok":
            print(f"NOT OK  {r['task']}: {r['status']}: {r['error']}")
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
