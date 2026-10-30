import sys

import pytest

from kernelgen.candidate import CandidateError, load_candidate

from sources import NAIVE, NO_ENTRY_POINT, SYNTAX_ERROR


def test_loads_entry_point():
    with load_candidate(NAIVE) as fn:
        assert callable(fn)


def test_module_is_removed_from_sys_modules_afterwards():
    with load_candidate(NAIVE):
        assert any(name.startswith("kernelgen_candidate_") for name in sys.modules)
    assert not any(name.startswith("kernelgen_candidate_") for name in sys.modules)


def test_missing_entry_point_raises():
    with pytest.raises(CandidateError, match="solution"):
        with load_candidate(NO_ENTRY_POINT):
            pass


def test_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        with load_candidate(SYNTAX_ERROR):
            pass


def test_decorators_that_read_source_work():
    # @triton.jit calls inspect.getsource on the decorated function, which
    # only works if the candidate is imported from a real file.
    source = (
        "import inspect\n"
        "def needs_source(fn):\n"
        "    inspect.getsource(fn)\n"
        "    return fn\n"
        "@needs_source\n"
        "def solution(x):\n"
        "    return x\n"
    )
    with load_candidate(source) as fn:
        assert fn(3) == 3
