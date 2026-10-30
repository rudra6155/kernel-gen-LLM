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


def test_candidate_is_also_timed_against_the_library_op():
    row = run(sources.LIBRARY)
    assert row["library_desc"] == "torch.nn.functional.layer_norm"
    assert row["library_ms"] > 0
    assert row["speedup_vs_library"] == pytest.approx(row["library_ms"] / row["candidate_ms"], rel=0.5)
    assert row["speedup_vs_library"] > 0
    assert row["error"] == ""


def test_task_without_a_library_op_leaves_the_library_columns_empty():
    from kernelgen.families import get_family

    task = get_family("gemm").task({"M": 8, "N": 8, "K": 8})
    assert task.library is None
    src = "import torch\ndef solution(a, b):\n    return torch.matmul(a, b)\n"
    row = evaluate(task, src, "c", "cpu", warmup=1, reps=3, seeds=(0,), env=ENV)
    assert row["status"] == "ok", row["error"]
    assert row["library_ms"] is None and row["speedup_vs_library"] is None and row["library_desc"] == ""


def test_library_failure_does_not_fail_the_candidate():
    from dataclasses import replace

    def broken(*args):
        raise RuntimeError("no kernel for this device")

    row = evaluate(replace(TASK, library=broken), sources.LIBRARY, "c", "cpu", warmup=1, reps=3, seeds=(0,), env=ENV)
    assert row["status"] == "ok" and row["correct"] == 1 and row["speedup"] > 0
    assert row["library_ms"] is None and row["speedup_vs_library"] is None
    assert "library timing failed" in row["error"] and "no kernel for this device" in row["error"]


def test_native_ops_are_rejected_when_forbidden():
    row = evaluate(
        TASK, sources.LIBRARY, "c", "cpu", warmup=1, reps=5, seeds=(0, 1), env=ENV, forbid_native_ops=True
    )
    assert row["status"] == "rejected"
    assert row["correct"] == 0
    assert row["speedup"] is None and row["candidate_ms"] is None
    assert row["reward"] == 0.0
    assert "native_layer_norm" in row["error"]


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
