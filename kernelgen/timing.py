"""Timing protocol shared by the baseline and every candidate.

On CUDA each call is timed with CUDA events after the L2 cache has been
flushed, and the baseline and candidate are interleaved rep by rep so clock or
thermal drift hits both equally. The reported time is the median.

The measured time includes the host-side launch latency between the start
event and the first kernel, which is the same for baseline and candidate.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch

FLUSH_BYTES = 256 * 1024 * 1024


@dataclass
class Timing:
    warmup: int
    reps: int
    baseline_ms: list[float]
    candidate_ms: list[float]

    @property
    def baseline_median(self) -> float:
        return statistics.median(self.baseline_ms)

    @property
    def candidate_median(self) -> float:
        return statistics.median(self.candidate_ms)

    @property
    def speedup(self) -> float:
        return self.baseline_median / self.candidate_median


def percentile(samples: list[float], p: float) -> float:
    """Nearest-rank percentile, p in [0, 1]."""
    ordered = sorted(samples)
    return ordered[int(round(p * (len(ordered) - 1)))]


class TimingVerificationError(Exception):
    """A candidate's output on the data used for timing was wrong."""


def benchmark_fresh(
    make_args: Callable[[int], tuple],
    baseline: Callable[..., object],
    candidate: Callable[..., object],
    device: str,
    warmup: int = 10,
    reps: int = 100,
    verify: Callable[[object, tuple], str | None] | None = None,
) -> Timing:
    """Like benchmark_pair, but every call gets new data and the candidate's output is checked.

    Call i runs both functions on `make_args(i)`, which must give different data
    for every i. Making the data and checking the output happen outside the timed
    region. `verify(output, args)` returns a reason string if the candidate's
    output for `args` is wrong, and then TimingVerificationError is raised.

    Timing the same tensors over and over would let a candidate cache its result
    (for example keyed on the input pointer) and look instantaneous. With new data
    every call, such a cache either misses and does the real work, or returns a
    stale result that verification rejects.
    """
    if warmup < 0 or reps < 1:
        raise ValueError("need warmup >= 0 and reps >= 1")

    cuda = device.startswith("cuda")
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=device) if cuda else None

    def measure(fn: Callable[..., object], args: tuple) -> tuple[float, object]:
        if cuda:
            flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn(*args)
            end.record()
            end.synchronize()
            return start.elapsed_time(end), out
        t0 = time.perf_counter()
        out = fn(*args)
        return (time.perf_counter() - t0) * 1e3, out

    base_ms: list[float] = []
    cand_ms: list[float] = []
    for i in range(warmup + reps):
        args = make_args(i)
        if i % 2 == 0:
            b, _ = measure(baseline, args)
            c, out = measure(candidate, args)
        else:
            c, out = measure(candidate, args)
            b, _ = measure(baseline, args)
        if verify is not None:
            reason = verify(out, args)
            if reason is not None:
                raise TimingVerificationError(f"call {i} of the timing run: {reason}")
        if i >= warmup:
            base_ms.append(b)
            cand_ms.append(c)

    return Timing(warmup=warmup, reps=reps, baseline_ms=base_ms, candidate_ms=cand_ms)


def benchmark_pair(
    baseline: Callable[[], object],
    candidate: Callable[[], object],
    device: str,
    warmup: int = 10,
    reps: int = 100,
) -> Timing:
    if warmup < 0 or reps < 1:
        raise ValueError("need warmup >= 0 and reps >= 1")

    cuda = device.startswith("cuda")
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device=device) if cuda else None

    def measure(fn: Callable[[], object]) -> float:
        if cuda:
            flush.zero_()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            return start.elapsed_time(end)
        t0 = time.perf_counter()
        fn()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(warmup):
        baseline()
        candidate()
    if cuda:
        torch.cuda.synchronize()

    base_ms: list[float] = []
    cand_ms: list[float] = []
    for i in range(reps):
        # Alternate who goes first so neither always follows the other.
        if i % 2 == 0:
            base_ms.append(measure(baseline))
            cand_ms.append(measure(candidate))
        else:
            cand_ms.append(measure(candidate))
            base_ms.append(measure(baseline))

    return Timing(warmup=warmup, reps=reps, baseline_ms=base_ms, candidate_ms=cand_ms)
