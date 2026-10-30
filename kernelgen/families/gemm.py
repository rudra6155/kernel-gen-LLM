"""Matrix multiplication (float16 in and out, float32 accumulate), plain and with a fused bias + ReLU epilogue."""

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
def _matmul(A, B, C__KERNEL_EXTRA__, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
__EPILOGUE__
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


def solution(__SOLUTION_ARGS__):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, __BLOCK_M__) * triton.cdiv(N, __BLOCK_N__),)
    _matmul[grid](
        a, b, c__CALL_EXTRA__, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=__BLOCK_M__, BLOCK_N=__BLOCK_N__, BLOCK_K=__BLOCK_K__, GROUP_M=__GROUP_M__,
        num_warps=__NUM_WARPS__, num_stages=__NUM_STAGES__,
    )
    return c
'''

BIAS_EPILOGUE = """    bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = tl.maximum(acc + bias[None, :], 0.0)"""


def render_plain(params, config):
    return _render(config, plain=True)


def render_bias_relu(params, config):
    return _render(config, plain=False)


def _render(config, plain: bool) -> str:
    return fill(
        TEMPLATE,
        KERNEL_EXTRA="" if plain else ", BIAS",
        CALL_EXTRA="" if plain else ", bias",
        EPILOGUE="" if plain else BIAS_EPILOGUE,
        SOLUTION_ARGS="a, b" if plain else "a, b, bias",
        BLOCK_M=config["block_m"],
        BLOCK_N=config["block_n"],
        BLOCK_K=config["block_k"],
        GROUP_M=config["group_m"],
        NUM_WARPS=config["num_warps"],
        NUM_STAGES=config["num_stages"],
    )


def configs(params):
    out = []
    for bm, bn, bk, gm, w, s in itertools.product((32, 64, 128), (32, 64, 128), (32, 64), (4, 8), (2, 4, 8), (2, 3, 4)):
        if bm * bn < 128 * w:  # each warp needs at least a 16x8 accumulator tile
            continue
        if (bm * bk + bk * bn) * 2 * s > SMEM_BUDGET_BYTES:  # fp16 operands, s pipeline stages
            continue
        out.append(
            {"block_m": bm, "block_n": bn, "block_k": bk, "group_m": gm, "num_warps": w, "num_stages": s}
        )
    return out


def default_config(params):
    return {"block_m": 128, "block_n": 128, "block_k": 32, "group_m": 8, "num_warps": 4, "num_stages": 3}


def _inputs(m, n, k, bias: bool):
    def make_inputs(seed: int, device: str) -> tuple:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        scale = k**-0.25  # keeps the output near unit variance
        a = (torch.randn(m, k, generator=g, device=device, dtype=torch.float32) * scale).to(torch.float16)
        b = (torch.randn(k, n, generator=g, device=device, dtype=torch.float32) * scale).to(torch.float16)
        if not bias:
            return a, b
        return a, b, torch.randn(n, generator=g, device=device, dtype=torch.float32).to(torch.float16)

    return make_inputs


def _naive_gemm(a, b):
    return torch.matmul(a, b)


def _naive_gemm_bias_relu(a, b, bias):
    return torch.relu(torch.matmul(a, b) + bias)


def _library_gemm_bias_relu(a, b, bias):
    return torch.relu(torch.addmm(bias, a, b))


def _truth_gemm(a, b):
    return torch.matmul(a.float(), b.float()).to(torch.float16)


def _truth_gemm_bias_relu(a, b, bias):
    return torch.relu(torch.matmul(a.float(), b.float()) + bias.float()).to(torch.float16)


def build_gemm(params, dtype) -> Task:
    m, n, k = params["M"], params["N"], params["K"]
    return Task(
        name=f"gemm_M{m}_N{n}_K{k}_{str(dtype).removeprefix('torch.')}",
        family="gemm",
        params={"M": m, "N": n, "K": k},
        dtype=dtype,
        bytes_moved=(m * k + k * n + m * n) * 2,  # read A and B, write C (float16)
        make_inputs=_inputs(m, n, k, bias=False),
        reference=_naive_gemm,
        baseline_desc="torch.matmul (cuBLAS), float16",
        rtol=1e-2,
        atol=1e-3,
        truth=_truth_gemm,
    )


def build_gemm_bias_relu(params, dtype) -> Task:
    m, n, k = params["M"], params["N"], params["K"]
    return Task(
        name=f"gemm_bias_relu_M{m}_N{n}_K{k}_{str(dtype).removeprefix('torch.')}",
        family="gemm_bias_relu",
        params={"M": m, "N": n, "K": k},
        dtype=dtype,
        bytes_moved=(m * k + k * n + m * n + n) * 2,  # read A, B, bias; write C (float16)
        make_inputs=_inputs(m, n, k, bias=True),
        reference=_naive_gemm_bias_relu,
        baseline_desc="eager PyTorch relu(matmul(a, b) + bias): three kernels, float16",
        rtol=1e-2,
        atol=1e-3,
        library=_library_gemm_bias_relu,
        library_desc="torch.relu(torch.addmm(bias, a, b))",
        truth=_truth_gemm_bias_relu,
    )


def describe_gemm(params):
    m, n, k = params["M"], params["N"], params["K"]
    return (
        f"Matrix multiplication C = A @ B. Inputs are float16 CUDA tensors: a of shape ({m}, {k}) and "
        f"b of shape ({k}, {n}). Accumulate in float32 and return a float16 tensor of shape ({m}, {n})."
    )


def describe_gemm_bias_relu(params):
    m, n, k = params["M"], params["N"], params["K"]
    return (
        f"Fused matrix multiplication with bias and ReLU: relu(A @ B + bias). Inputs are float16 CUDA "
        f"tensors: a of shape ({m}, {k}), b of shape ({k}, {n}), bias of shape ({n},). Accumulate in "
        f"float32 and return a float16 tensor of shape ({m}, {n})."
    )


TRAIN_SHAPES = [{"M": m, "N": n, "K": k} for m in (512, 1024, 2048) for n in (512, 1024, 2048) for k in (256, 512, 1024)]
HELDOUT_SHAPES = [
    {"M": 768, "N": 768, "K": 768},
    {"M": 1536, "N": 1536, "K": 1536},
    {"M": 3072, "N": 1024, "K": 1024},
    {"M": 1024, "N": 3072, "K": 512},
    {"M": 1000, "N": 1000, "K": 1000},
    {"M": 4096, "N": 1024, "K": 256},
    {"M": 512, "N": 512, "K": 4096},
    {"M": 2048, "N": 1536, "K": 768},
]

GEMM = register(
    Family(
        name="gemm",
        dtype="float16",
        signature="solution(a, b)",
        build_task=build_gemm,
        render=render_plain,
        configs=configs,
        default_config=default_config,
        describe=describe_gemm,
        reference=_naive_gemm,
        train_shapes=TRAIN_SHAPES,
        heldout_shapes=HELDOUT_SHAPES,
    )
)

GEMM_BIAS_RELU = register(
    Family(
        name="gemm_bias_relu",
        dtype="float16",
        signature="solution(a, b, bias)",
        build_task=build_gemm_bias_relu,
        render=render_bias_relu,
        configs=configs,
        default_config=default_config,
        describe=describe_gemm_bias_relu,
        reference=_naive_gemm_bias_relu,
        train_shapes=TRAIN_SHAPES,
        heldout_shapes=HELDOUT_SHAPES,
    )
)
