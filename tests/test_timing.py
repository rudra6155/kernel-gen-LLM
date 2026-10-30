import time

import pytest

from kernelgen.timing import benchmark_pair, percentile


def test_percentile_nearest_rank():
    data = [5, 1, 4, 2, 3]
    assert percentile(data, 0.0) == 1
    assert percentile(data, 0.5) == 3
    assert percentile(data, 1.0) == 5


def test_faster_candidate_has_speedup_above_one():
    timing = benchmark_pair(
        lambda: time.sleep(0.004), lambda: time.sleep(0.001), "cpu", warmup=1, reps=5
    )
    assert timing.speedup > 2.0


def test_slower_candidate_has_speedup_below_one():
    timing = benchmark_pair(
        lambda: time.sleep(0.001), lambda: time.sleep(0.004), "cpu", warmup=1, reps=5
    )
    assert timing.speedup < 0.5


def test_both_functions_run_warmup_plus_reps_times():
    calls = {"baseline": 0, "candidate": 0}

    def baseline():
        calls["baseline"] += 1

    def candidate():
        calls["candidate"] += 1

    timing = benchmark_pair(baseline, candidate, "cpu", warmup=3, reps=7)
    assert calls == {"baseline": 10, "candidate": 10}
    assert len(timing.baseline_ms) == len(timing.candidate_ms) == 7


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        benchmark_pair(lambda: None, lambda: None, "cpu", warmup=0, reps=0)
