"""Profiling tests that run without a GPU.

The kernel-event parsing is tested on synthetic traces and the plumbing on real
CPU profiler runs. What a CUDA trace actually contains is checked only by the
GPU run (make profile), not here.
"""

import json
import time

import pytest
import torch

from kernelgen import sandbox
from kernelgen.benchmark import KNOWN_KERNELS
from kernelgen.profiling import (
    classify,
    effective_gbps,
    kernel_events,
    make_flusher,
    measure_copy_gbps,
    run_profiler,
    summarize_kernels,
    time_events,
)
from kernelgen.rows import PROFILE_FIELDS, write_csv
from kernelgen.tasks import layernorm_task, task_from_spec, task_spec

import sources

TASK = layernorm_task(8, 64)
ENV = {"gpu_name": "cpu", "torch_version": "t", "cuda_version": "", "triton_version": ""}

HANG = "def solution(x, weight, bias, eps=1e-5):\n    while True:\n        pass\n"


def kernel(name, dur, **args):
    return {"ph": "X", "cat": "kernel", "name": name, "ts": 0, "dur": dur, "args": args}


# --- bandwidth arithmetic -----------------------------------------------------


def test_effective_gbps():
    assert effective_gbps(1_000_000_000, 1000.0) == pytest.approx(1.0)
    assert effective_gbps(2_000_000, 0.01) == pytest.approx(200.0)


def test_effective_gbps_without_time_is_none():
    assert effective_gbps(1000, 0.0) is None


def test_layernorm_bytes_moved_counts_x_in_y_out_and_the_two_vectors():
    task = layernorm_task(4096, 1024)
    assert task.bytes_moved == (2 * 4096 * 1024 + 2 * 1024) * 4
    assert layernorm_task(4, 8, torch.float16).bytes_moved == (2 * 4 * 8 + 2 * 8) * 2


def test_task_spec_round_trip_keeps_bytes_moved():
    assert task_from_spec(task_spec(TASK)).bytes_moved == TASK.bytes_moved


# --- trace parsing --------------------------------------------------------------


def test_kernel_events_keeps_only_gpu_kernels():
    trace = {
        "traceEvents": [
            kernel("k1", 10.0),
            {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "dur": 5.0},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy DtoD", "dur": 900.0},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "dur": 3.0},
            {"ph": "M", "cat": "kernel", "name": "metadata, not a duration event"},
            kernel("k2", 20.0),
        ]
    }
    assert [e["name"] for e in kernel_events(trace)] == ["k1", "k2"]


def test_kernel_events_of_an_empty_trace():
    assert kernel_events({}) == []


def test_summarize_groups_by_name_and_sorts_by_total_time():
    events = [
        kernel("small", 1.0),
        kernel("big", 10.0, grid=[8, 1, 1], block=[128, 1, 1]),
        kernel("big", 30.0, grid=[8, 1, 1], block=[128, 1, 1]),
        kernel("small", 3.0),
    ]
    top = summarize_kernels(events, iters=2)
    assert [k["name"] for k in top] == ["big", "small"]
    assert top[0]["per_iter"] == 1.0 and top[0]["avg_us"] == 20.0
    assert top[0]["grid"] == [8, 1, 1] and top[0]["block"] == [128, 1, 1]
    assert "grid" not in top[1]


def test_summarize_limits_to_the_top_n_and_truncates_long_names():
    events = [kernel(f"k{i}", float(i + 1)) for i in range(10)] + [kernel("x" * 500, 100.0)]
    top = summarize_kernels(events, iters=1, top=3)
    assert len(top) == 3
    assert len(top[0]["name"]) == 120


# --- consistency check ------------------------------------------------------------


def test_classify_cpu_run_is_ok_without_kernels():
    assert classify(False, 0, 0.0, 1.0) == ("ok", "")


def test_classify_cuda_run_without_kernel_events_is_an_error():
    status, message = classify(True, 0, 0.0, 1.0)
    assert status == "error" and "no GPU kernel events" in message


def test_classify_flags_kernel_time_above_event_time_as_suspect():
    status, message = classify(True, 5, 2.0, 1.0)
    assert status == "suspect" and "not trustworthy" in message


def test_classify_accepts_kernel_time_within_event_time():
    assert classify(True, 5, 0.9, 1.0) == ("ok", "")
    assert classify(True, 5, 1.04, 1.0) == ("ok", "")  # inside the noise margin


# --- real profiler on CPU -----------------------------------------------------------


def test_run_profiler_writes_a_valid_chrome_trace(tmp_path):
    x = torch.randn(64, 64)
    path = tmp_path / "nested" / "trace.json"
    trace = run_profiler(lambda: (x * x).sum(), "cpu", 3, make_flusher("cpu"), path)

    assert path.exists()
    assert json.loads(path.read_text()) == trace
    names = {e.get("name") for e in trace["traceEvents"]}
    assert "aten::mul" in names
    assert kernel_events(trace) == []  # no GPU kernels on CPU


def test_run_profiler_without_a_path_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    trace = run_profiler(lambda: None, "cpu", 1, make_flusher("cpu"), None)
    assert "traceEvents" in trace
    assert list(tmp_path.iterdir()) == []


def test_time_events_on_cpu_returns_median_milliseconds():
    ms = time_events(lambda: time.sleep(0.005), "cpu", 3, make_flusher("cpu"))
    assert 4.0 < ms < 100.0


def test_copy_bandwidth_is_not_measured_on_cpu():
    assert measure_copy_gbps("cpu") is None


# --- the CUDA branch of profile_callable, with the GPU-touching pieces stubbed ------------
#
# This checks the row arithmetic and status logic only. Whether a real CUDA
# trace has "kernel" events, and what they contain, is checked by make profile.


def stub_gpu(monkeypatch, events, event_ms, copy_gbps=100.0):
    from kernelgen import profiling

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(profiling, "make_flusher", lambda device: (lambda: None))
    monkeypatch.setattr(profiling, "time_events", lambda *a, **k: event_ms)
    monkeypatch.setattr(profiling, "run_profiler", lambda *a, **k: {"traceEvents": events})
    monkeypatch.setattr(profiling, "measure_copy_gbps", lambda device: copy_gbps)
    return profiling


def gpu_row(profiling, peak_gbps=None, iters=2):
    from kernelgen.rows import new_profile_row

    row = new_profile_row(TASK, "candidate", "c", iters, ENV, peak_gbps)
    return profiling.profile_callable(TASK, lambda: None, row, "cuda", warmup=0, iters=iters, trace_path=None)


def test_cuda_branch_derives_bandwidth_from_kernel_time(monkeypatch):
    # 2 iterations x 2 kernels x 50 us = 0.1 ms of kernel time per iteration.
    events = [kernel("a", 50.0, grid=[4, 1, 1]), kernel("b", 50.0)] * 2
    profiling = stub_gpu(monkeypatch, events, event_ms=0.2)
    row = gpu_row(profiling, peak_gbps=200.0)

    assert row["status"] == "ok", row["error"]
    assert row["kernel_ms_per_iter"] == pytest.approx(0.1)
    assert row["kernel_launches_per_iter"] == 2.0
    assert row["event_ms"] == 0.2
    expected = TASK.bytes_moved / 1e-4 / 1e9
    assert row["effective_gbps"] == pytest.approx(expected)
    assert row["copy_gbps"] == 100.0
    assert row["pct_of_copy_bw"] == pytest.approx(100.0 * expected / 100.0)
    assert row["pct_of_peak"] == pytest.approx(100.0 * expected / 200.0)
    top = json.loads(row["top_kernels"])
    assert {k["name"] for k in top} == {"a", "b"}


def test_cuda_branch_leaves_pct_of_peak_empty_without_a_peak(monkeypatch):
    profiling = stub_gpu(monkeypatch, [kernel("a", 50.0)] * 2, event_ms=0.2)
    row = gpu_row(profiling, peak_gbps=None)
    assert row["pct_of_peak"] is None and row["peak_gbps"] is None
    assert row["pct_of_copy_bw"] is not None


def test_cuda_branch_marks_inflated_kernel_time_suspect(monkeypatch):
    # A kernel that outlasts the whole timed call means something unrelated was counted.
    profiling = stub_gpu(monkeypatch, [kernel("flush_like", 3000.0)] * 2, event_ms=0.2)
    row = gpu_row(profiling)
    assert row["status"] == "suspect" and "not trustworthy" in row["error"]


def test_cuda_branch_without_kernel_events_is_an_error_not_zero_bandwidth(monkeypatch):
    profiling = stub_gpu(monkeypatch, [{"ph": "X", "cat": "cpu_op", "name": "aten::mul", "dur": 5.0}], event_ms=0.2)
    row = gpu_row(profiling)
    assert row["status"] == "error" and "no GPU kernel events" in row["error"]
    assert row["effective_gbps"] is None and row["kernel_ms_per_iter"] is None


# --- profile_isolated (real worker process, CPU) --------------------------------------


def profile(impl, source=None, timeout_s=90.0, check_policy=False, trace_path=None, **kw):
    return sandbox.profile_isolated(
        TASK, impl, source, "c" if impl == "candidate" else "pytorch_baseline", "cpu",
        warmup=1, iters=3, seeds=(0, 1), env=ENV, timeout_s=timeout_s,
        check_policy=check_policy, trace_path=trace_path, **kw,
    )


def test_baseline_profile_round_trips_through_a_worker(tmp_path):
    trace = tmp_path / "traces" / "baseline.json"
    row = profile("baseline", trace_path=trace)
    assert row["status"] == "ok", row["error"]
    assert row["impl"] == "baseline" and row["task"] == TASK.name
    assert row["bytes_moved"] == TASK.bytes_moved and row["iters"] == 3
    assert row["event_ms"] > 0
    assert row["kernel_ms_per_iter"] is None and row["effective_gbps"] is None  # CPU: no kernel figures
    assert row["trace_file"] == str(trace)  # the caller's spelling, not an absolute path from the worker
    assert "traceEvents" in json.loads(trace.read_text())


def test_candidate_profile_without_a_trace_path_writes_no_trace():
    row = profile("candidate", sources.LIBRARY)
    assert row["status"] == "ok", row["error"]
    assert row["trace_file"] is None


def test_incorrect_candidate_is_not_profiled(tmp_path):
    trace = tmp_path / "t.json"
    row = profile("candidate", sources.MISSING_BIAS, trace_path=trace)
    assert row["status"] == "incorrect"
    assert row["event_ms"] is None and row["trace_file"] is None
    assert not trace.exists()


def test_candidate_error_is_reported(tmp_path):
    row = profile("candidate", sources.RAISES)
    assert row["status"] == "error" and "boom" in row["error"]


def test_hanging_profile_times_out():
    row = profile("candidate", HANG, timeout_s=8.0)
    assert row["status"] == "timeout"
    assert "8s" in row["error"]


def test_policy_violation_is_rejected_before_profiling(monkeypatch):
    def no_spawn(*a, **k):
        raise AssertionError("a worker was started")

    monkeypatch.setattr(sandbox.subprocess, "Popen", no_spawn)
    row = profile("candidate", sources.LIBRARY, check_policy=True)
    assert row["status"] == "rejected"
    assert row["event_ms"] is None


def test_native_ops_are_rejected_by_the_guard_inside_the_profile_job():
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
    row = profile("candidate", source, check_policy=True)
    assert row["status"] == "rejected" and "aten::mul" in row["error"]


def test_known_kernel_is_profiled_on_cpu_under_the_interpreter(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    monkeypatch.setenv("TRITON_INTERPRET", "1")
    candidate_id, path = KNOWN_KERNELS["layernorm"]
    trace = tmp_path / "known.json"
    row = sandbox.profile_isolated(
        TASK, "candidate", path.read_text(), candidate_id, "cpu",
        warmup=1, iters=2, seeds=(0, 1, 2), env=ENV, timeout_s=180.0, trace_path=trace,
    )
    assert row["status"] == "ok", row["error"]
    assert trace.exists()


def test_invalid_impl_is_rejected():
    with pytest.raises(ValueError, match="impl"):
        sandbox.profile_isolated(TASK, "both", None, "c", "cpu", env=ENV)


def test_candidate_without_source_is_rejected():
    with pytest.raises(ValueError, match="source"):
        sandbox.profile_isolated(TASK, "candidate", None, "c", "cpu", env=ENV)


# --- malformed worker results ------------------------------------------------------------


def malformed(*a, **k):
    return sandbox.WorkerOutcome({"unexpected": 1}, None, "", "")


def test_malformed_profile_result_becomes_a_crash_row(monkeypatch):
    monkeypatch.setattr(sandbox, "run_worker", malformed)
    row = profile("baseline")
    assert row["status"] == "crash" and "malformed" in row["error"]


def test_malformed_evaluate_result_becomes_a_crash_row(monkeypatch):
    monkeypatch.setattr(sandbox, "run_worker", malformed)
    row = sandbox.evaluate_isolated(
        TASK, sources.LIBRARY, "c", "cpu", warmup=1, reps=2, seeds=(0,), env=ENV, check_policy=False
    )
    assert row["status"] == "crash" and "malformed" in row["error"] and row["reward"] == 0.0


# --- CSV ---------------------------------------------------------------------------------------


def test_profile_csv_round_trip(tmp_path):
    import csv

    row = profile("baseline")
    out = tmp_path / "profile_summary.csv"
    write_csv([row], out, PROFILE_FIELDS)
    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == PROFILE_FIELDS
        (read,) = list(reader)
    assert read["impl"] == "baseline" and read["status"] == "ok"
    assert read["effective_gbps"] == ""  # empty, not "None"
