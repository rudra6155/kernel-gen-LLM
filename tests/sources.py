"""Candidate sources used by the CPU tests (layernorm task: x, weight, bias)."""

NAIVE = """
import torch

def solution(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias
"""

# Correct, but computed by a different implementation than the reference.
LIBRARY = """
import torch

def solution(x, weight, bias, eps=1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
"""

MISSING_BIAS = """
import torch

def solution(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight
"""

# Shape (1, N) would broadcast against (M, N) inside allclose.
WRONG_SHAPE = NAIVE + """
_inner = solution

def solution(x, weight, bias, eps=1e-5):
    return _inner(x, weight, bias, eps)[:1]
"""

WRONG_DTYPE = NAIVE + """
_inner = solution

def solution(x, weight, bias, eps=1e-5):
    return _inner(x, weight, bias, eps).double()
"""

NOT_A_TENSOR = """
def solution(x, weight, bias, eps=1e-5):
    return 1.0
"""

HAS_NAN = NAIVE + """
_inner = solution

def solution(x, weight, bias, eps=1e-5):
    out = _inner(x, weight, bias, eps)
    out[0, 0] = float("nan")
    return out
"""

# Output is right, but the inputs are clobbered afterwards.
MUTATES_INPUT = NAIVE + """
_inner = solution

def solution(x, weight, bias, eps=1e-5):
    out = _inner(x, weight, bias, eps)
    x.zero_()
    return out
"""

RAISES = """
def solution(x, weight, bias, eps=1e-5):
    raise RuntimeError("boom")
"""

NO_ENTRY_POINT = """
def something_else():
    pass
"""

SYNTAX_ERROR = "def solution(:\n"
