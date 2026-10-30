"""Row-wise softmax over the last dimension (float32)."""

from __future__ import annotations

import torch

from kernelgen.families import Family, register
from kernelgen.families.layernorm import configs, default_config
from kernelgen.families.util import fill
from kernelgen.tasks import Task

SINGLE = '''import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd(X, Y, stride, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=float("-inf")).to(tl.float32)
    e = tl.exp(x - tl.max(x, axis=0))
    y = e / tl.sum(e, axis=0)
    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)


def solution(x):
    x = x.contiguous()
    m, n = x.shape
    y = torch.empty_like(x)
    _softmax_fwd[(m,)](x, y, x.stride(0), n, BLOCK_N=__BLOCK_N__, num_warps=__NUM_WARPS__)
    return y
'''

LOOP = '''import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_fwd(X, Y, stride, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = X + row * stride
    mvec = tl.full([BLOCK_N], float("-inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = tl.load(base + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
        mvec = tl.maximum(mvec, x)
    row_max = tl.max(mvec, axis=0)
    svec = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = tl.load(base + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
        svec += tl.exp(x - row_max)
    denom = tl.sum(svec, axis=0)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(base + cols, mask=mask, other=float("-inf")).to(tl.float32)
        tl.store(Y + row * stride + cols, (tl.exp(x - row_max) / denom).to(Y.dtype.element_ty), mask=mask)


def solution(x):
    x = x.contiguous()
    m, n = x.shape
    y = torch.empty_like(x)
    _softmax_fwd[(m,)](x, y, x.stride(0), n, BLOCK_N=__BLOCK_N__, num_warps=__NUM_WARPS__)
    return y
'''


def _naive_softmax(x):
    e = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)


def _library_softmax(x):
    return torch.softmax(x, dim=-1)


def build_task(params, dtype) -> Task:
    m, n = params["M"], params["N"]

    def make_inputs(seed: int, device: str) -> tuple:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return (torch.randn(m, n, generator=g, device=device, dtype=dtype) * 3.0,)

    elem = torch.empty((), dtype=dtype).element_size()
    return Task(
        name=f"softmax_M{m}_N{n}_{str(dtype).removeprefix('torch.')}",
        family="softmax",
        params={"M": m, "N": n},
        dtype=dtype,
        bytes_moved=2 * m * n * elem,  # read x, write y
        make_inputs=make_inputs,
        reference=_naive_softmax,
        baseline_desc="eager PyTorch softmax from max/sub/exp/sum/div ops",
        rtol=1e-4,
        atol=1e-6,
        library=_library_softmax,
        library_desc="torch.softmax",
    )


def render(params, config):
    template = SINGLE if config["variant"] == "single" else LOOP
    return fill(template, BLOCK_N=config["block_n"], NUM_WARPS=config["num_warps"])


def describe(params):
    m, n = params["M"], params["N"]
    return (
        f"Row-wise softmax over the last dimension. The input is a float32 CUDA tensor x of shape "
        f"({m}, {n}). Return a float32 tensor of shape ({m}, {n})."
    )


FAMILY = register(
    Family(
        name="softmax",
        dtype="float32",
        signature="solution(x)",
        build_task=build_task,
        render=render,
        configs=configs,
        default_config=default_config,
        describe=describe,
        reference=_naive_softmax,
        train_shapes=[{"M": m, "N": n} for m in (1024, 2048, 4096) for n in (256, 512, 1024, 2048, 4096)],
        heldout_shapes=[
            {"M": 4096, "N": 384},
            {"M": 2048, "N": 768},
            {"M": 4096, "N": 1536},
            {"M": 1024, "N": 3072},
            {"M": 2048, "N": 5120},
            {"M": 1024, "N": 6144},
            {"M": 512, "N": 8192},
            {"M": 4096, "N": 1280},
        ],
    )
)
