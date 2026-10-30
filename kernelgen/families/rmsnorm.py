"""RMSNorm forward over the last dimension (float32)."""

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
def _rmsnorm_fwd(X, W, Y, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def solution(x, weight, eps=1e-5):
    x = x.contiguous()
    m, n = x.shape
    y = torch.empty_like(x)
    _rmsnorm_fwd[(m,)](x, weight, y, x.stride(0), n, eps, BLOCK_N=__BLOCK_N__, num_warps=__NUM_WARPS__)
    return y
'''

LOOP = '''import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd(X, W, Y, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = X + row * stride
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        x = tl.load(base + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        tl.store(Y + row * stride + cols, (x * rstd * w).to(Y.dtype.element_ty), mask=mask)


def solution(x, weight, eps=1e-5):
    x = x.contiguous()
    m, n = x.shape
    y = torch.empty_like(x)
    _rmsnorm_fwd[(m,)](x, weight, y, x.stride(0), n, eps, BLOCK_N=__BLOCK_N__, num_warps=__NUM_WARPS__)
    return y
'''


def _naive_rmsnorm(x, weight, eps: float = 1e-5):
    return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps) * weight


def _library_rmsnorm(x, weight, eps: float = 1e-5):
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, eps)


def build_task(params, dtype) -> Task:
    m, n = params["M"], params["N"]

    def make_inputs(seed: int, device: str) -> tuple:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x = torch.randn(m, n, generator=g, device=device, dtype=dtype) * 2.0 + 0.5
        weight = torch.randn(n, generator=g, device=device, dtype=dtype)
        return x, weight

    elem = torch.empty((), dtype=dtype).element_size()
    return Task(
        name=f"rmsnorm_M{m}_N{n}_{str(dtype).removeprefix('torch.')}",
        family="rmsnorm",
        params={"M": m, "N": n},
        dtype=dtype,
        bytes_moved=(2 * m * n + n) * elem,  # read x, write y, read weight
        make_inputs=make_inputs,
        reference=_naive_rmsnorm,
        baseline_desc="eager PyTorch RMSNorm from mul/mean/add/rsqrt/mul/mul ops",
        rtol=1e-4,
        atol=1e-4,
        library=_library_rmsnorm,
        library_desc="torch.nn.functional.rms_norm",
    )


def render(params, config):
    template = SINGLE if config["variant"] == "single" else LOOP
    return fill(template, BLOCK_N=config["block_n"], NUM_WARPS=config["num_warps"])


def describe(params):
    m, n = params["M"], params["N"]
    return (
        f"RMSNorm forward over the last dimension (no mean subtraction, no bias). Inputs are float32 "
        f"CUDA tensors: x of shape ({m}, {n}) and weight of shape ({n},). eps defaults to 1e-5. "
        f"Return a float32 tensor of shape ({m}, {n})."
    )


FAMILY = register(
    Family(
        name="rmsnorm",
        dtype="float32",
        signature="solution(x, weight, eps=1e-5)",
        build_task=build_task,
        render=render,
        configs=configs,
        default_config=default_config,
        describe=describe,
        reference=_naive_rmsnorm,
        train_shapes=[{"M": m, "N": n} for m in (1024, 2048, 4096) for n in (512, 768, 1024, 1536, 2048, 3072)],
        heldout_shapes=[
            {"M": 4096, "N": 640},
            {"M": 2048, "N": 1280},
            {"M": 4096, "N": 2560},
            {"M": 1024, "N": 4096},
            {"M": 2048, "N": 5120},
            {"M": 4096, "N": 6144},
            {"M": 1024, "N": 8192},
            {"M": 8192, "N": 3584},
        ],
    )
)
