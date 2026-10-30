import textwrap

import pytest

from kernelgen.benchmark import KNOWN_KERNELS
from kernelgen.candidate import load_candidate
from kernelgen.correctness import check_correctness
from kernelgen.policy import PolicyViolation, check_source
from kernelgen.tasks import layernorm_task

import sources

TASK = layernorm_task(8, 64)


def candidate(host: str, imports: str = "") -> str:
    """A well-formed Triton candidate whose host code is `host`."""
    body = textwrap.indent(textwrap.dedent(host).strip("\n"), "    ")
    return (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        f"{imports}\n"
        "@triton.jit\n"
        "def _k(X, BLOCK: tl.constexpr):\n"
        "    pass\n\n"
        "def solution(x, weight, bias, eps=1e-5):\n"
        f"{body}\n"
        "    _k[(1,)](x, BLOCK=1)\n"
        "    return x\n"
    )


KNOWN = KNOWN_KERNELS["layernorm"][1].read_text()

# --- static check: sources that must pass -----------------------------------


def test_known_kernel_passes():
    assert check_source(KNOWN) == []


@pytest.mark.parametrize(
    "host, imports",
    [
        ("y = torch.empty_like(x)", ""),
        ("m, n = x.shape\ngrid = (triton.cdiv(m, 4),)\ny = torch.empty((m, n), device=x.device, dtype=x.dtype)", ""),
        ("y = x.contiguous().view(-1)", ""),
        ("y = x.new_empty(4).zero_()", ""),
        ("n = torch.cuda.get_device_properties(x.device).multi_processor_count", ""),
        ("y = torch.empty_like(x, dtype=torch.float16)", ""),
        ("y = t.empty_like(x)", "import torch as t"),
        ("y = empty_like(x)", "from torch import empty_like"),
        ("y = x.to(torch.float32)", ""),
        ("grid = []\ngrid.append(triton.cdiv(8, 2))", ""),
    ],
)
def test_legitimate_host_code_passes(host, imports):
    assert check_source(candidate(host, imports)) == []


def test_import_forms_for_triton_pass():
    src = (
        "import torch\n"
        "from triton import jit, cdiv\n"
        "from triton import language as tl\n"
        "@jit\n"
        "def _k(X, BLOCK: tl.constexpr):\n"
        "    v = tl.load(X + tl.arange(0, BLOCK)).to(tl.float32)\n"
        "def solution(x):\n"
        "    _k[(cdiv(4, 2),)](x, BLOCK=4)\n"
        "    return x\n"
    )
    assert check_source(src) == []


def test_autotuned_kernel_passes():
    src = (
        "import torch\n"
        "import triton\n"
        "import triton.language as tl\n"
        "@triton.autotune(configs=[triton.Config({'BLOCK': 64}, num_warps=2)], key=['n'])\n"
        "@triton.jit\n"
        "def _k(X, n, BLOCK: tl.constexpr):\n"
        "    pass\n"
        "def solution(x):\n"
        "    _k[(1,)](x, 4)\n"
        "    return x\n"
    )
    assert check_source(src) == []


def test_methods_and_matmul_are_allowed_inside_a_kernel():
    src = candidate("y = x").replace("    pass\n", "    v = tl.zeros((2, 2), tl.float32).sum(0)\n    w = v @ v\n", 1)
    assert check_source(src) == []


# --- static check: sources that must be rejected ----------------------------


@pytest.mark.parametrize(
    "host, imports, expected",
    [
        ("y = torch.nn.functional.layer_norm(x, (4,))", "", "torch.nn.functional.layer_norm"),
        ("y = torch.matmul(x, x)", "", "torch.matmul"),
        ("y = torch.layer_norm(x, (4,))", "", "torch.layer_norm"),
        ("y = torch.sqrt(x)", "", "torch.sqrt"),
        ("y = t.matmul(x, x)", "import torch as t", "torch.matmul"),
        ("y = 1", "from torch.nn.functional import layer_norm", "torch.nn.functional"),
        ("y = 1", "from torch import matmul", "from torch import matmul"),
        ("y = 1", "from torch import nn", "from torch import nn"),
        ("y = 1", "from torch import *", "star import"),
        ("import torch.nn.functional as F\ny = F.layer_norm(x, (4,))", "", "torch.nn.functional"),
        ("t = torch\ny = t.matmul(x, x)", "", "bare reference"),
        ("y = x @ x", "", "'@'"),
        ("x @= x", "", "'@='"),
        ("y = x.mean(-1)", "", ".mean()"),
        ("y = torch.empty_like(x).sum()", "", ".sum()"),
        ("y = getattr(torch, 'matmul')(x, x)", "", "'getattr'"),
        ("y = x.__class__", "", "__class__"),
        ("exec('1')", "", "'exec'"),
        ("y = 1", "import os", "'os'"),
        ("y = 1", "import subprocess", "'subprocess'"),
        ("y = 1", "import triton.ops", "'triton.ops'"),
        ("y = triton.testing.do_bench(lambda: None)", "", "triton.testing"),
        ("y = torch.cuda.synchronize()", "", "torch.cuda.synchronize"),
        ("y = torch.cuda", "", "torch.cuda"),
    ],
)
def test_policy_violations_are_rejected(host, imports, expected):
    violations = check_source(candidate(host, imports))
    assert violations, "expected a violation"
    assert any(expected in v for v in violations), violations


def test_pure_pytorch_candidate_is_rejected():
    violations = check_source(sources.NAIVE)
    assert any("no @triton.jit kernel" in v for v in violations)


def test_kernel_that_is_never_launched_is_rejected():
    src = candidate("y = 1").replace("    _k[(1,)](x, BLOCK=1)\n", "")
    assert any("never launched" in v for v in check_source(src))


def test_missing_entry_point_is_rejected():
    src = candidate("y = 1").replace("def solution", "def other")
    assert any("'solution'" in v for v in check_source(src))


@pytest.mark.parametrize("src", ["def solution(:\n", "x = 1\x00", "x = " + "(" * 5000 + "1" + ")" * 5000])
def test_unparseable_source_is_rejected_not_raised(src):
    assert check_source(src)


def test_every_violation_carries_a_line_number():
    violations = check_source(candidate("y = torch.matmul(x, x)"))
    assert all(v.startswith("line ") for v in violations if "torch.matmul" in v)


def test_checking_does_not_execute_the_source(tmp_path):
    marker = tmp_path / "ran"
    src = candidate("y = 1") + f"\nopen({str(marker)!r}, 'w').write('x')\n"
    assert check_source(src)  # rejected for open()
    assert not marker.exists()


# --- dynamic guard: native ops that the AST cannot see -----------------------


def run_guarded(source):
    with load_candidate(source) as fn:
        return check_correctness(TASK, fn, "cpu", seeds=(0,), forbid_native_ops=True)


def test_guard_rejects_library_layernorm():
    with pytest.raises(PolicyViolation) as exc:
        run_guarded(sources.LIBRARY)
    assert "aten::native_layer_norm" in exc.value.ops


def test_guard_rejects_tensor_operators_the_ast_cannot_see():
    src = (
        "def solution(x, weight, bias, eps=1e-5):\n"
        "    return x * weight + bias\n"
    )
    with pytest.raises(PolicyViolation) as exc:
        run_guarded(src)
    assert {"aten::mul", "aten::add"} <= set(exc.value.ops)


def test_guard_rejects_matmul_operator():
    src = "def solution(x, weight, bias, eps=1e-5):\n    return x @ x.t()\n"
    with pytest.raises(PolicyViolation) as exc:
        run_guarded(src)
    assert "aten::mm" in exc.value.ops


def test_guard_cannot_be_hidden_behind_try_except():
    src = (
        "import torch\n"
        "def solution(x, weight, bias, eps=1e-5):\n"
        "    try:\n"
        "        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)\n"
        "    except BaseException:\n"
        "        return x\n"
    )
    with pytest.raises(PolicyViolation):
        run_guarded(src)


def test_guard_allows_allocation_views_and_copies():
    src = (
        "import torch\n"
        "def solution(x, weight, bias, eps=1e-5):\n"
        "    y = torch.empty_like(x)\n"
        "    y.copy_(x.t().contiguous().t())\n"
        "    return y.view(x.shape).to(torch.float64).to(torch.float32).clone()\n"
    )
    result = run_guarded(src)  # wrong values, but no PolicyViolation
    assert not result.correct and "tolerance" in result.reason


def test_guard_is_off_by_default():
    with load_candidate(sources.LIBRARY) as fn:
        assert check_correctness(TASK, fn, "cpu", seeds=(0,)).correct


def test_reference_runs_outside_the_guard():
    # The reference uses mean/sub/pow/...; it must not trip the guard.
    with load_candidate("def solution(x, weight, bias, eps=1e-5):\n    return x.new_empty(x.shape).zero_()\n") as fn:
        result = check_correctness(TASK, fn, "cpu", seeds=(0,), forbid_native_ops=True)
    assert not result.correct
