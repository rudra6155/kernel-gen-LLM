import time

import pytest

from kernelgen.timing import TimingVerificationError, benchmark_fresh, benchmark_pair, percentile


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


# --- benchmark_fresh: new data every call, output verified ---------------------------------------------


def test_fresh_gives_every_call_its_own_args_and_runs_both_functions_on_them():
    seen = []

    def make_args(i):
        return (i,)

    def baseline(i):
        seen.append(("b", i))
        return i

    def candidate(i):
        seen.append(("c", i))
        return i

    timing = benchmark_fresh(make_args, baseline, candidate, "cpu", warmup=2, reps=4)
    assert sorted(i for kind, i in seen if kind == "b") == list(range(6))
    assert sorted(i for kind, i in seen if kind == "c") == list(range(6))
    assert len(timing.baseline_ms) == len(timing.candidate_ms) == 4  # warmup calls are not timed
    assert timing.warmup == 2 and timing.reps == 4


def test_fresh_alternates_which_function_goes_first():
    order = []
    benchmark_fresh(lambda i: (i,), lambda i: order.append("b"), lambda i: order.append("c"), "cpu", warmup=0, reps=4)
    assert order == ["b", "c", "c", "b", "b", "c", "c", "b"]


def test_fresh_speedup_follows_the_timings():
    fast = benchmark_fresh(lambda i: (), lambda: time.sleep(0.004), lambda: time.sleep(0.001), "cpu", warmup=1, reps=4)
    assert fast.speedup > 2.0


def test_fresh_verifies_every_candidate_output_including_warmup():
    verified = []

    def verify(out, args):
        verified.append((out, args))
        return None

    benchmark_fresh(lambda i: (i,), lambda i: None, lambda i: i * 10, "cpu", warmup=2, reps=3, verify=verify)
    assert verified == [(i * 10, (i,)) for i in range(5)]


def test_fresh_raises_when_verification_fails_and_names_the_call():
    def verify(out, args):
        return "wrong values" if args[0] == 3 else None

    with pytest.raises(TimingVerificationError, match="call 3 of the timing run: wrong values"):
        benchmark_fresh(lambda i: (i,), lambda i: i, lambda i: i, "cpu", warmup=1, reps=5, verify=verify)


def test_fresh_rejects_invalid_arguments():
    with pytest.raises(ValueError):
        benchmark_fresh(lambda i: (), lambda: None, lambda: None, "cpu", warmup=0, reps=0)
