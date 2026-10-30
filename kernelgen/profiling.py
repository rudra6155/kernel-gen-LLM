"""Profile kernels with torch.profiler: kernel time, launch counts, effective bandwidth.

Effective bandwidth is Task.bytes_moved (the minimum traffic any implementation
must generate) divided by GPU kernel time taken from the profiler trace. It is
not measured DRAM traffic, which needs Nsight Compute. The same byte count is
used for the PyTorch baseline and the candidate, so the figures compare "useful
bytes per second", and a baseline that makes many passes over memory scores low.

As an empirical reference the profile also measures the bandwidth of a plain
device-to-device copy on the same GPU (read + write, like layernorm). No peak
bandwidth is assumed; pass peak_gbps if you know the card's spec.

Each call is preceded by an L2-cache flush, as in the benchmark timing. The
flush is a device-to-device copy, which Kineto should record as a memcpy rather
than a kernel, so it is left out of the kernel time. That is not assumed: every
run also times the same call with CUDA events outside the profiler, and a row
whose profiler kernel time exceeds the event time is marked "suspect".

Only the CUDA path is meaningful. The CPU path exists so the plumbing can be
tested without a GPU: it records a trace but reports no kernel figures.

Usage:
    python -m kernelgen.profiling --out results/profile_summary.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from torch.profiler import ProfilerActivity, profile

from kernelgen.benchmark import KNOWN_KERNELS
from kernelgen.candidate import load_candidate
from kernelgen.correctness import check_correctness
from kernelgen.policy import PolicyViolation
from kernelgen.rows import PROFILE_FIELDS, environment_info, fail_profile_row, new_profile_row, write_csv
from kernelgen.sandbox import DEFAULT_TIMEOUT_S, profile_isolated
from kernelgen.tasks import TIMING_SEED, Task, default_tasks, task_from_spec

FLUSH_BYTES = 256 * 1024 * 1024
COPY_BYTES = 256 * 1024 * 1024
# Summed kernel time cannot exceed wall time on one stream; allow some noise.
SUSPECT_RATIO = 1.05
# Kernel-event args worth keeping for diagnosing a slow kernel (best effort:
# whichever of these the trace provides).
KERNEL_ARG_KEYS = ("grid", "block", "registers per thread", "shared memory", "est. achieved occupancy %")


def effective_gbps(bytes_moved: int, ms: float) -> float | None:
    """Decimal GB/s for `bytes_moved` bytes in `ms` milliseconds."""
    if ms <= 0:
        return None
    return bytes_moved / (ms * 1e-3) / 1e9


def make_flusher(device: str) -> Callable[[], None]:
    if not device.startswith("cuda"):
        return lambda: None
    src = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    return lambda: dst.copy_(src)


def time_events(fn: Callable[[], object], device: str, iters: int, flush: Callable[[], None]) -> float:
    """Median milliseconds per call, flushing L2 before each, outside the profiler."""
    cuda = device.startswith("cuda")
    samples = []
    for _ in range(iters):
        flush()
        if cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def measure_copy_gbps(device: str, iters: int = 20, warmup: int = 5) -> float | None:
    """Bandwidth of a large device-to-device copy (bytes read + bytes written)."""
    if not device.startswith("cuda"):
        return None
    src = torch.empty(COPY_BYTES, dtype=torch.uint8, device=device)
    dst = torch.empty_like(src)
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    ms = time_events(lambda: dst.copy_(src), device, iters, lambda: None)
    return effective_gbps(2 * COPY_BYTES, ms)


def run_profiler(
    fn: Callable[[], object],
    device: str,
    iters: int,
    flush: Callable[[], None],
    trace_path: str | Path | None,
) -> dict[str, Any]:
    """Profile `iters` flushed calls of `fn` and return the parsed Chrome trace."""
    cuda = device.startswith("cuda")
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=activities, record_shapes=False, with_stack=False) as prof:
        for _ in range(iters):
            flush()
            fn()
        if cuda:
            torch.cuda.synchronize()

    if trace_path is not None:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(path))
        return json.loads(path.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="kernelgen_trace_") as tmp:
        path = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(path))
        return json.loads(path.read_text(encoding="utf-8"))


def kernel_events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """GPU kernel events (durations in microseconds) from a Chrome trace."""
    return [e for e in trace.get("traceEvents", []) if e.get("ph") == "X" and e.get("cat") == "kernel"]


def summarize_kernels(events: Sequence[dict[str, Any]], iters: int, top: int = 5) -> list[dict[str, Any]]:
    """Per-kernel-name totals, largest first."""
    groups: dict[str, dict[str, Any]] = {}
    for e in events:
        name = str(e.get("name", "?"))
        args = e.get("args") or {}
        g = groups.setdefault(
            name, {"name": name, "count": 0, "total_us": 0.0, "args": {k: args.get(k) for k in KERNEL_ARG_KEYS}}
        )
        g["count"] += 1
        g["total_us"] += float(e.get("dur", 0.0))
    ordered = sorted(groups.values(), key=lambda g: g["total_us"], reverse=True)[:top]
    return [
        {
            "name": g["name"][:120],
            "per_iter": g["count"] / iters,
            "avg_us": g["total_us"] / g["count"],
            **{k: v for k, v in g["args"].items() if v is not None},
        }
        for g in ordered
    ]


def classify(cuda: bool, n_kernel_events: int, kernel_ms: float, event_ms: float) -> tuple[str, str]:
    """Row status and message for one profiled run."""
    if not cuda:
        return "ok", ""
    if n_kernel_events == 0:
        return "error", (
            "no GPU kernel events in the profiler trace: CUPTI may be unavailable, "
            "or kernels are not categorized as 'kernel' in this torch version"
        )
    if kernel_ms > SUSPECT_RATIO * event_ms:
        return "suspect", (
            f"profiler kernel time {kernel_ms:.4f} ms per call exceeds CUDA-event time "
            f"{event_ms:.4f} ms; the bandwidth figure is not trustworthy"
        )
    return "ok", ""


def profile_callable(
    task: Task,
    fn: Callable[[], object],
    row: dict[str, Any],
    device: str,
    warmup: int,
    iters: int,
    trace_path: str | Path | None,
) -> dict[str, Any]:
    cuda = device.startswith("cuda")
    flush = make_flusher(device)

    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()

    row["event_ms"] = time_events(fn, device, iters, flush)
    trace = run_profiler(fn, device, iters, flush, trace_path)
    if trace_path is not None:
        row["trace_file"] = str(trace_path)

    events = kernel_events(trace)
    if cuda and events:
        kernel_ms = sum(float(e.get("dur", 0.0)) for e in events) / 1e3 / iters
        row["kernel_ms_per_iter"] = kernel_ms
        row["kernel_launches_per_iter"] = len(events) / iters
        row["effective_gbps"] = effective_gbps(task.bytes_moved, kernel_ms)
        row["top_kernels"] = json.dumps(summarize_kernels(events, iters))
    else:
        kernel_ms = 0.0

    row["copy_gbps"] = measure_copy_gbps(device)
    if row["effective_gbps"] is not None:
        if row["copy_gbps"]:
            row["pct_of_copy_bw"] = 100.0 * row["effective_gbps"] / row["copy_gbps"]
        if row["peak_gbps"]:
            row["pct_of_peak"] = 100.0 * row["effective_gbps"] / row["peak_gbps"]

    row["status"], message = classify(cuda, len(events), kernel_ms, row["event_ms"])
    row["error"] = message
    return row


def profile_job(request: dict[str, Any]) -> dict[str, Any]:
    """Run one profile request (see kernelgen.sandbox.profile_isolated) in this process.

    A candidate is profiled only after it passes the correctness gate (and the
    native-op guard when requested), so a wrong or cheating kernel never gets a
    bandwidth figure.
    """
    task = task_from_spec(request["task"])
    device = request["device"]
    row = new_profile_row(
        task, request["impl"], request["candidate_id"], request["iters"], request["env"], request.get("peak_gbps")
    )
    try:
        if request["impl"] == "baseline":
            inputs = task.make_inputs(TIMING_SEED, device)
            return profile_callable(
                task, lambda: task.reference(*inputs), row, device,
                request["warmup"], request["iters"], request["trace_path"],
            )
        with load_candidate(request["source"]) as fn:
            gate = check_correctness(
                task, fn, device, request["seeds"], forbid_native_ops=request["forbid_native_ops"]
            )
            if not gate.correct:
                return fail_profile_row(row, "incorrect", gate.reason)
            inputs = task.make_inputs(TIMING_SEED, device)
            return profile_callable(
                task, lambda: fn(*inputs), row, device,
                request["warmup"], request["iters"], request["trace_path"],
            )
    except PolicyViolation as exc:
        return fail_profile_row(row, "rejected", str(exc))
    except Exception as exc:
        return fail_profile_row(row, "error", f"{type(exc).__name__}: {exc}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/profile_summary.csv")
    parser.add_argument("--trace-dir", default="results/traces", help="where Chrome traces are written")
    parser.add_argument("--no-trace", action="store_true", help="do not keep Chrome trace files")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20, help="profiled calls per run")
    parser.add_argument("--seeds", type=int, default=3, help="number of correctness-gate inputs")
    parser.add_argument("--peak-gbps", type=float, default=None, help="the card's spec peak bandwidth in GB/s, if known")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if not torch.cuda.is_available():
        print("error: no CUDA device available; refusing to write profile results", file=sys.stderr)
        return 2

    device = "cuda"
    env = environment_info(device)
    rows: list[dict[str, Any]] = []
    for task in default_tasks():
        candidate_id, path = KNOWN_KERNELS[task.family]
        runs = (("baseline", "pytorch_baseline", None), ("candidate", candidate_id, path.read_text()))
        for impl, cid, source in runs:
            trace_path = None if args.no_trace else Path(args.trace_dir) / f"{task.name}__{impl}.json"
            rows.append(
                profile_isolated(
                    task, impl, source, cid, device,
                    warmup=args.warmup, iters=args.iters, seeds=range(args.seeds), env=env,
                    timeout_s=args.timeout, trace_path=trace_path, peak_gbps=args.peak_gbps,
                )
            )
            write_csv(rows, args.out, PROFILE_FIELDS)  # keep partial results if a later run dies

    def fmt(v: Any, spec: str) -> str:
        return "-" if v is None else format(v, spec)

    for r in rows:
        print(
            f"{r['task']:<30} {r['impl']:<9} {r['status']:<9} "
            f"kernels/iter={fmt(r['kernel_launches_per_iter'], '.1f'):>5} "
            f"kernel_ms={fmt(r['kernel_ms_per_iter'], '.4f'):>8} "
            f"GB/s={fmt(r['effective_gbps'], '.1f'):>7} "
            f"%copy={fmt(r['pct_of_copy_bw'], '.0f'):>4}  {r['error']}"
        )
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
