"""Worker process: run one job and write the resulting row as JSON.

Started by kernelgen.sandbox; not meant to be run by hand.

    python -m kernelgen.worker REQUEST_JSON RESULT_JSON

The result goes to a file rather than stdout so that anything the candidate
prints cannot corrupt it. If the process dies before writing, the parent
reports a crash.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from kernelgen.benchmark import evaluate
from kernelgen.profiling import profile_job
from kernelgen.tasks import task_from_spec


def run_job(request: dict) -> dict:
    job = request["job"]
    if job == "evaluate":
        return evaluate(
            task_from_spec(request["task"]),
            request["source"],
            request["candidate_id"],
            request["device"],
            warmup=request["warmup"],
            reps=request["reps"],
            seeds=request["seeds"],
            env=request["env"],
            forbid_native_ops=request["forbid_native_ops"],
        )
    if job == "profile":
        return profile_job(request)
    raise ValueError(f"unknown job: {job!r}")


def main(argv: list[str]) -> int:
    request_path, result_path = argv
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    row = run_job(request)

    partial = result_path + ".part"
    Path(partial).write_text(json.dumps(row), encoding="utf-8")
    os.replace(partial, result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
