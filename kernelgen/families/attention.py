"""Non-causal multi-head attention forward (float16 in and out, float32 softmax and accumulate)."""

from __future__ import annotations

import itertools

import torch

from kernelgen.families import Family, register
from kernelgen.families.util import SMEM_BUDGET_BYTES, fill
from kernelgen.tasks import Task

TEMPLATE = '''import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(Q, K, V, O, sm_scale, S, stride_bh, stride_s,
              HEAD_D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    base = bh * stride_bh
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_D)
    q = tl.load(Q + base + offs_m[:, None] * stride_s + offs_d[None, :], mask=offs_m[:, None] < S, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_D], dtype=tl.float32)
    for start_n in range(0, S, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(K + base + offs_n[None, :] * stride_s + offs_d[:, None], mask=offs_n[None, :] < S, other=0.0)
        qk = tl.dot(q, k) * sm_scale
        qk = tl.where(offs_n[None, :] < S, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(V + base + offs_n[:, None] * stride_s + offs_d[None, :], mask=offs_n[:, None] < S, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + base + offs_m[:, None] * stride_s + offs_d[None, :], acc.to(O.dtype.element_ty), mask=offs_m[:, None] < S)


def solution(q, k, v):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    b, h, s, d = q.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(s, __BLOCK_M__), b * h)
    _attn_fwd[grid](
        q, k, v, o, d ** -0.5, s, s * d, d,
        HEAD_D=d, BLOCK_M=__BLOCK_M__, BLOCK_N=__BLOCK_N__,
        num_warps=__NUM_WARPS__, num_stages=__NUM_STAGES__,
    )
    return o
'''


def render(params, config):
    return fill(
        TEMPLATE,
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        NUM_WARPS=config["num_warps"],
        NUM_STAGES=config["num_stages"],
    )


def configs(params):
    d = params["D"]
    out = []
    for bm, bn, w, s in itertools.product((32, 64, 128), (32, 64, 128), (2, 4, 8), (1, 2, 3)):
        if bm * bn < 128 * w:
            continue
        if (bm * d + 2 * bn * d) * 2 * s > SMEM_BUDGET_BYTES:  # Q tile plus K and V tiles, float16
            continue
        out.append({"block_m": bm, "block_n": bn, "num_warps": w, "num_stages": s})
    return out


def default_config(params):
    return {"block_m": 64, "block_n": 64, "num_warps": 4, "num_stages": 2}


def _naive_attention(q, k, v):
    scale = q.shape[-1] ** -0.5
    return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1), v)


def _library_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _truth_attention(q, k, v):
    q, k, v = q.float(), k.float(), v.float()
    scores = torch.matmul(q, k.transpose(-1, -2)) * q.shape[-1] ** -0.5
    return torch.matmul(torch.softmax(scores, dim=-1), v).to(torch.float16)


def build_task(params, dtype) -> Task:
    b, h, s, d = params["B"], params["H"], params["S"], params["D"]

    def make_inputs(seed: int, device: str) -> tuple:
        g = torch.Generator(device=device)
        g.manual_seed(seed)

        def rand(scale):
            return (torch.randn(b, h, s, d, generator=g, device=device, dtype=torch.float32) * scale).to(torch.float16)

        # q and k are scaled up so the softmax is not close to uniform.
        return rand(1.5), rand(1.5), rand(1.0)

    return Task(
        name=f"attention_B{b}_H{h}_S{s}_D{d}_{str(dtype).removeprefix('torch.')}",
        family="attention",
        params={"B": b, "H": h, "S": s, "D": d},
        dtype=dtype,
        bytes_moved=4 * b * h * s * d * 2,  # read q, k, v; write o (float16); the score matrix never needs to exist
        make_inputs=make_inputs,
        reference=_naive_attention,
        baseline_desc="eager PyTorch softmax(q @ k^T * scale) @ v, materializing the S x S score matrix, float16",
        rtol=1e-2,
        atol=1e-3,
        library=_library_attention,
        library_desc="torch.nn.functional.scaled_dot_product_attention",
        truth=_truth_attention,
    )


def describe(params):
    b, h, s, d = params["B"], params["H"], params["S"], params["D"]
    return (
        f"Non-causal multi-head attention forward: softmax(q @ k^T / sqrt(D)) @ v. Inputs are float16 "
        f"CUDA tensors q, k, v, each of shape (B={b}, H={h}, S={s}, D={d}), contiguous. Compute the softmax "
        f"in float32 and return a float16 tensor of shape ({b}, {h}, {s}, {d})."
    )


FAMILY = register(
    Family(
        name="attention",
        dtype="float16",
        signature="solution(q, k, v)",
        build_task=build_task,
        render=render,
        configs=configs,
        default_config=default_config,
        describe=describe,
        reference=_naive_attention,
        train_shapes=[
            {"B": b, "H": 8, "S": s, "D": d} for b in (2, 4) for s in (128, 256, 512, 1024) for d in (32, 64)
        ],
        heldout_shapes=[
            {"B": 2, "H": 8, "S": 192, "D": 64},
            {"B": 2, "H": 8, "S": 384, "D": 64},
            {"B": 4, "H": 8, "S": 768, "D": 64},
            {"B": 2, "H": 8, "S": 1536, "D": 64},
            {"B": 2, "H": 16, "S": 512, "D": 128},
            {"B": 1, "H": 8, "S": 2048, "D": 64},
            {"B": 4, "H": 8, "S": 256, "D": 128},
            {"B": 2, "H": 8, "S": 640, "D": 32},
            {"B": 2, "H": 8, "S": 1024, "D": 128},
            {"B": 1, "H": 16, "S": 1024, "D": 64},
        ],
    )
)
