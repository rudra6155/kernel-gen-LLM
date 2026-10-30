"""LayerNorm forward over the last dimension (float32)."""

from __future__ import annotations

import torch

from kernelgen.families import Family, register
from kernelgen.families.util import default_warps, fill, next_pow2
from kernelgen.tasks import _naive_layernorm, layernorm_task

SINGLE = '''import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd(X, W, B, Y, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)


def solution(x, weight, bias, eps=1e-5):
    x = x.contiguous()
    m, n = x.shape
    y = torch.empty_like(x)
    _layernorm_fwd[(m,)](x, weight, bias, y, x.stride(0), n, eps, BLOCK_N=__BLOCK_N__, num_warps=__NUM_WARPS__)
    return y
'''

LOOP = '''import torch
import triton
import triton.language as tl


@triton.jit
def _layernorm_fwd(X, W, B, Y, stride, N, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    base = X + row * stride
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        acc += tl.load(base + cols, mask=cols < N, other=0.0).to(tl.float32)
    mean = tl.sum(acc, axis=0) / N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        xc = tl.where(mask, x - mean, 0.0)
        acc += xc * xc
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(base + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y + row * stride + cols, y.to(Y.dtype.element_ty), mask=mask)


def solution(x, weight, bias, eps=1e-5):
    x = x.contiguous()
    m, n = x.shape
    y = torch.empty_like(x)
    _layernorm_fwd[(m,)](x, weight, bias, y, x.stride(0), n, eps, BLOCK_N=__BLOCK_N__, num_warps=__NUM_WARPS__)
    return y
'''


def render(params, config):
    template = SINGLE if config["variant"] == "single" else LOOP
    return fill(template, BLOCK_N=config["block_n"], NUM_WARPS=config["num_warps"])


def configs(params):
    full = next_pow2(params["N"])
    out = [
        {"variant": "single", "block_n": full, "num_warps": w}
        for w in (1, 2, 4, 8, 16)
        if 32 * w <= full
    ]
    out += [
        {"variant": "loop", "block_n": b, "num_warps": w}
        for b in (128, 256, 512, 1024, 2048, 4096)
        if b < full
        for w in (1, 2, 4, 8)
        if 32 * w <= b
    ]
    return out


def default_config(params):
    full = next_pow2(params["N"])
    return {"variant": "single", "block_n": full, "num_warps": default_warps(full)}


def describe(params):
    m, n = params["M"], params["N"]
    return (
        f"LayerNorm forward over the last dimension. Inputs are float32 CUDA tensors: "
        f"x of shape ({m}, {n}), weight and bias of shape ({n},). eps defaults to 1e-5. "
        f"Return a float32 tensor of shape ({m}, {n})."
    )


FAMILY = register(
    Family(
        name="layernorm",
        dtype="float32",
        signature="solution(x, weight, bias, eps=1e-5)",
        build_task=lambda params, dtype: layernorm_task(params["M"], params["N"], dtype),
        render=render,
        configs=configs,
        default_config=default_config,
        describe=describe,
        reference=_naive_layernorm,
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
