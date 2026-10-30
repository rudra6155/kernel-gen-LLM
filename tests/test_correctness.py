import pytest

from kernelgen.candidate import load_candidate
from kernelgen.correctness import check_correctness
from kernelgen.tasks import layernorm_task

import sources

TASK = layernorm_task(8, 64)


def run_gate(source):
    with load_candidate(source) as fn:
        return check_correctness(TASK, fn, "cpu", seeds=(0, 1, 2))


def test_identical_implementation_passes():
    result = run_gate(sources.NAIVE)
    assert result.correct
    assert result.max_abs_err == 0.0


def test_different_but_correct_implementation_passes():
    result = run_gate(sources.LIBRARY)
    assert result.correct
    assert 0.0 <= result.max_abs_err < 1e-4


@pytest.mark.parametrize(
    "source, reason",
    [
        (sources.MISSING_BIAS, "exceeds tolerance"),
        (sources.WRONG_SHAPE, "shape"),
        (sources.WRONG_DTYPE, "dtype"),
        (sources.NOT_A_TENSOR, "not a Tensor"),
        (sources.HAS_NAN, "NaN or Inf"),
        (sources.MUTATES_INPUT, "modified its inputs"),
    ],
)
def test_wrong_candidates_are_rejected(source, reason):
    result = run_gate(source)
    assert not result.correct
    assert reason in result.reason


def test_exceptions_from_the_candidate_propagate():
    with pytest.raises(RuntimeError, match="boom"):
        run_gate(sources.RAISES)


def test_seeds_give_different_inputs():
    a = TASK.make_inputs(0, "cpu")[0]
    b = TASK.make_inputs(1, "cpu")[0]
    assert not (a == b).all()


def test_same_seed_gives_same_inputs():
    a = TASK.make_inputs(7, "cpu")
    b = TASK.make_inputs(7, "cpu")
    assert all((x == y).all() for x, y in zip(a, b))
