import csv

import pytest

from kernelgen.benchmark import FIELDS, evaluate, write_csv
from kernelgen.tasks import layernorm_task

import sources

TASK = layernorm_task(8, 64)
ENV = {"gpu_name": "cpu", "torch_version": "t", "cuda_version": "", "triton_version": ""}


def run(source, candidate_id="c"):
    return evaluate(TASK, source, candidate_id, "cpu", warmup=1, reps=5, seeds=(0, 1), env=ENV)


def test_correct_candidate_is_timed_and_rewarded():
    row = run(sources.LIBRARY)
    assert row["status"] == "ok"
    assert row["correct"] == 1
    assert row["speedup"] == pytest.approx(row["baseline_ms"] / row["candidate_ms"])
    assert row["reward"] == pytest.approx(row["speedup"])
    assert row["error"] == ""


def test_incorrect_candidate_gets_zero_reward_and_no_timing():
    row = run(sources.MISSING_BIAS)
    assert row["status"] == "incorrect"
    assert row["correct"] == 0
    assert row["speedup"] is None and row["candidate_ms"] is None
    assert row["reward"] == 0.0
    assert "tolerance" in row["error"]


@pytest.mark.parametrize(
    "source, message",
    [
        (sources.RAISES, "boom"),
        (sources.NO_ENTRY_POINT, "solution"),
        (sources.SYNTAX_ERROR, "SyntaxError"),
    ],
)
def test_failing_candidate_is_reported_as_error(source, message):
    row = run(source)
    assert row["status"] == "error"
    assert row["correct"] == 0
    assert row["reward"] == 0.0
    assert message in row["error"]
    assert "\n" not in row["error"]


def test_csv_round_trip(tmp_path):
    rows = [run(sources.LIBRARY, "good"), run(sources.MISSING_BIAS, "bad")]
    out = tmp_path / "nested" / "kernel_benchmarks.csv"
    write_csv(rows, out)

    with out.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        read = list(reader)

    assert [r["candidate_id"] for r in read] == ["good", "bad"]
    good, bad = read
    assert good["status"] == "ok"
    assert float(good["speedup"]) == pytest.approx(
        float(good["baseline_ms"]) / float(good["candidate_ms"]), rel=1e-12
    )
    assert float(good["reward"]) == pytest.approx(float(good["speedup"]), rel=1e-12)
    assert bad["status"] == "incorrect"
    assert bad["speedup"] == "" and bad["candidate_ms"] == ""
    assert float(bad["reward"]) == 0.0
    assert good["params"] == '{"M": 8, "N": 64}'
