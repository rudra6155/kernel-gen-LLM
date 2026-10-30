"""Fused LayerNorm forward in Triton: one program per row, one read and one write of x."""

import torch
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
    block_n = triton.next_power_of_2(n)
    num_warps = min(max(block_n // 256, 1), 8)
    _layernorm_fwd[(m,)](x, weight, bias, y, x.stride(0), n, eps, BLOCK_N=block_n, num_warps=num_warps)
    return y
