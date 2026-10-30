"""Evaluate untrusted candidate code safely: policy check, then a worker process.

A candidate that hangs, segfaults, exits, or exhausts memory takes down only its
own worker. The parent records a row with reward 0 and the run continues.

POSIX only: the worker gets its own process group so that a timeout also kills
anything the candidate spawned.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from kernelgen.policy import check_source
from kernelgen.reward import compute_reward
from kernelgen.rows import FIELDS, environment_info, fail_row, new_row
from kernelgen.tasks import Task, task_spec

log = logging.getLogger(__name__)

# Covers interpreter start, importing torch, Triton compilation, correctness
# runs, and timing. A starting value, not a measured one: tune it with --timeout.
DEFAULT_TIMEOUT_S = 120.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TAIL_BYTES = 8192


def _tail(path: Path) -> str:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(f.tell() - TAIL_BYTES, 0))
        return f.read().decode("utf-8", errors="replace")


def _summarize_stderr(text: str) -> str:
    """The most informative line of a worker's stderr, for the CSV error column.

    faulthandler's "Fatal Python error" line says why the process died but is
    followed by a stack dump, so prefer it over simply taking the last line.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if ln.startswith("Fatal Python error"):
            return ln
    return " | ".join(lines[-2:])


def _describe_exit(returncode: int) -> str:
    if returncode < 0:
        try:
            return f"killed by signal {signal.Signals(-returncode).name}"
        except ValueError:
            return f"killed by signal {-returncode}"
    return f"exit code {returncode}"


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def evaluate_isolated(
    task: Task,
    source: str,
    candidate_id: str,
    device: str,
    warmup: int = 10,
    reps: int = 100,
    seeds: Sequence[int] = (0, 1, 2),
    env: dict[str, str] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    check_policy: bool = True,
) -> dict[str, Any]:
    """Evaluate one candidate in a worker process and return one CSV row.

    With `check_policy` the source must pass kernelgen.policy.check_source
    before anything runs, and the worker runs it under the native-op guard.
    Turn it off only to test the isolation itself with candidates that are not
    Triton kernels.

    Policy violations, timeouts, and worker crashes come back as rows with
    status "rejected", "timeout", or "crash" and reward 0.
    """
    seeds = list(seeds)
    row = new_row(
        task, candidate_id, warmup, reps, len(seeds), env if env is not None else environment_info(device)
    )

    if check_policy:
        violations = check_source(source)
        if violations:
            log.warning("candidate %s rejected: %s", candidate_id, "; ".join(violations))
            return fail_row(row, "rejected", "; ".join(violations))

    request = {
        "task": task_spec(task),
        "source": source,
        "candidate_id": candidate_id,
        "device": device,
        "warmup": warmup,
        "reps": reps,
        "seeds": seeds,
        "env": {k: row[k] for k in ("gpu_name", "torch_version", "cuda_version", "triton_version")},
        "forbid_native_ops": check_policy,
    }

    with tempfile.TemporaryDirectory(prefix="kernelgen_run_") as tmp:
        tmp = Path(tmp)
        request_path, result_path = tmp / "request.json", tmp / "result.json"
        stdout_path, stderr_path = tmp / "stdout.log", tmp / "stderr.log"
        request_path.write_text(json.dumps(request), encoding="utf-8")

        cmd = [sys.executable, "-X", "faulthandler", "-m", "kernelgen.worker", str(request_path), str(result_path)]
        timed_out = False
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=PROJECT_ROOT,
                start_new_session=True,
            )
            try:
                proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                if proc.poll() is None:
                    _kill_group(proc)

        stderr_tail = _tail(stderr_path)

        if timed_out:
            return _failure(row, "timeout", f"killed after exceeding {timeout_s:g}s wall-clock limit", stderr_tail)

        result = _read_result(result_path)
        if result is None:
            return _failure(row, "crash", f"worker died ({_describe_exit(proc.returncode)}) without a result", stderr_tail)

        if proc.returncode != 0:
            log.warning(
                "candidate %s: worker exited abnormally (%s) after reporting a result",
                candidate_id,
                _describe_exit(proc.returncode),
            )

    # Never trust a reward computed in the process that ran the candidate.
    result["reward"] = compute_reward(result.get("speedup"), bool(result.get("correct")))
    return result


def _read_result(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(result, dict) or set(result) != set(FIELDS):
        return None
    return result


def _failure(row: dict[str, Any], status: str, what: str, stderr_tail: str) -> dict[str, Any]:
    summary = _summarize_stderr(stderr_tail)
    log.warning("candidate %s: %s\n--- worker stderr (tail) ---\n%s", row["candidate_id"], what, stderr_tail.strip())
    return fail_row(row, status, f"{what}: {summary}" if summary else what)
