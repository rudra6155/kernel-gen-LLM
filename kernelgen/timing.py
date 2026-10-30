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
