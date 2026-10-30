import os
import sys
import time

import pytest

from kernelgen import sandbox
from kernelgen.benchmark import KNOWN_KERNELS
from kernelgen.tasks import layernorm_task

import sources

TASK = layernorm_task(8, 64)
ENV = {"gpu_name": "cpu", "torch_version": "t", "cuda_version": "", "triton_version": ""}

HANG = """
def solution(x, weight, bias, eps=1e-5):
    while True:
        pass
"""

SEGFAULT = """
import ctypes

def solution(x, weight, bias, eps=1e-5):
    ctypes.string_at(0)
"""

EXIT_AT_IMPORT = """
import sys
sys.exit(7)
"""

PRINTS_FORGED_RESULT = sources.NAIVE + """
print('{"status": "ok", "speedup": 1000.0}')
"""


def run(source, timeout_s=90.0, check_policy=False, **kw):
    return sandbox.evaluate_isolated(
        TASK, source, "c", "cpu", warmup=1, reps=3, seeds=(0, 1), env=ENV,
        timeout_s=timeout_s, check_policy=check_policy, **kw,
    )


def test_correct_candidate_round_trips_through_a_worker():
    row = run(sources.LIBRARY)
    assert row["status"] == "ok", row["error"]
    assert row["correct"] == 1
    assert row["speedup"] > 0
    assert row["reward"] == pytest.approx(row["speedup"])
    assert row["task"] == TASK.name and row["candidate_id"] == "c"


def test_incorrect_candidate_is_reported_by_the_worker():
    row = run(sources.MISSING_BIAS)
    assert row["status"] == "incorrect"
    assert row["reward"] == 0.0


def test_exception_in_candidate_is_an_error_row():
    row = run(sources.RAISES)
    assert row["status"] == "error"
    assert "boom" in row["error"]
    assert row["reward"] == 0.0


def test_hanging_candidate_times_out_and_gets_zero_reward():
    start = time.monotonic()
    row = run(HANG, timeout_s=8.0)
    elapsed = time.monotonic() - start
    assert row["status"] == "timeout"
    assert row["correct"] == 0 and row["reward"] == 0.0
    assert "8s" in row["error"]
    assert elapsed < 8.0 + 20.0, "the parent must return soon after the timeout"


def test_segfaulting_candidate_is_a_crash_row():
    row = run(SEGFAULT)
    assert row["status"] == "crash"
    assert row["reward"] == 0.0
    assert "SIGSEGV" in row["error"]
    assert "Fatal Python error: Segmentation fault" in row["error"]


def test_candidate_that_exits_is_a_crash_row():
    row = run(EXIT_AT_IMPORT)
    assert row["status"] == "crash"
    assert "exit code 7" in row["error"]


def test_candidate_output_cannot_forge_or_corrupt_the_result():
    row = run(PRINTS_FORGED_RESULT)
    assert row["status"] == "ok"
    assert row["speedup"] != 1000.0


def test_crash_does_not_stop_later_candidates():
    assert run(SEGFAULT)["status"] == "crash"
    assert run(sources.LIBRARY)["status"] == "ok"


@pytest.mark.skipif(not os.path.isdir("/proc"), reason="needs /proc to inspect process state")
def test_timeout_kills_processes_spawned_by_the_candidate(tmp_path):
    pid_file = tmp_path / "child.pid"
    source = f"""
import subprocess

child = subprocess.Popen(["sleep", "300"])
open({str(pid_file)!r}, "w").write(str(child.pid))

def solution(x, weight, bias, eps=1e-5):
    while True:
        pass
"""
    row = run(source, timeout_s=8.0)
    assert row["status"] == "timeout"
    pid = int(pid_file.read_text())
    try:
        state = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except FileNotFoundError:
        state = "gone"
    assert state in ("gone", "Z"), f"child process {pid} is still running (state {state})"


def test_policy_violation_is_rejected_without_starting_a_worker(monkeypatch):
    def no_spawn(*a, **k):
        raise AssertionError("a worker was started")

    monkeypatch.setattr(sandbox.subprocess, "Popen", no_spawn)
    row = run(sources.LIBRARY, check_policy=True)
    assert row["status"] == "rejected"
    assert row["correct"] == 0 and row["reward"] == 0.0
    assert "torch.nn" in row["error"] or "no @triton.jit kernel" in row["error"]


def test_dynamic_guard_rejects_native_math_that_passes_the_static_check():
    pytest.importorskip("triton")
    source = """
import torch
import triton
import triton.language as tl

@triton.jit
def _k(X, BLOCK: tl.constexpr):
    pass

def solution(x, weight, bias, eps=1e-5):
    y = x * weight + bias
    if False:
        _k[(1,)](x, BLOCK=1)
    return y
"""
    row = run(source, check_policy=True)
    assert row["status"] == "rejected", row
    assert "aten::mul" in row["error"]
    assert row["reward"] == 0.0


def test_known_kernel_passes_the_whole_pipeline_on_cpu(monkeypatch):
    # Triton's CPU interpreter runs the kernel logic without a GPU. This checks
    # the static policy, the dispatch guard, the worker, and the correctness
    # gate; it says nothing about GPU compilation or speed.
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    candidate_id, path = KNOWN_KERNELS["layernorm"]
    row = sandbox.evaluate_isolated(
        TASK, path.read_text(), candidate_id, "cpu", warmup=1, reps=2, seeds=(0, 1, 2),
        env=ENV, timeout_s=180.0,
    )
    assert row["status"] == "ok", row["error"]
    assert row["correct"] == 1
    assert row["max_abs_err"] < 1e-4
