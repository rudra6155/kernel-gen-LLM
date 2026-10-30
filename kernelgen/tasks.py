"""Kernel tasks: inputs, PyTorch baseline, and comparison tolerances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True, eq=False)
class Task:
    name: str
    family: str
    params: dict[str, Any]
    dtype: torch.dtype
    # (seed, device) -> positional arguments for both reference and candidate
    make_inputs: Callable[[int, str], tuple]
    # The PyTorch baseline. Speedup is measured against this function.
    reference: Callable[..., torch.Tensor]
    baseline_desc: str
    rtol: float
    atol: float


def _naive_layernorm(x, weight, bias, eps: float = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def layernorm_task(m: int, n: int, dtype: torch.dtype = torch.float32) -> Task:
    def make_inputs(seed: int, device: str) -> tuple:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x = torch.randn(m, n, generator=g, device=device, dtype=dtype) * 2.0 + 1.5
        weight = torch.randn(n, generator=g, device=device, dtype=dtype)
        bias = torch.randn(n, generator=g, device=device, dtype=dtype)
        return x, weight, bias

    dtype_tag = str(dtype).removeprefix("torch.")
    return Task(
        name=f"layernorm_M{m}_N{n}_{dtype_tag}",
        family="layernorm",
        params={"M": m, "N": n},
        dtype=dtype,
        make_inputs=make_inputs,
        reference=_naive_layernorm,
        baseline_desc="eager PyTorch LayerNorm from mean/sub/pow/sqrt/div/mul/add ops",
        rtol=1e-4,
        atol=1e-4,
    )


def default_tasks() -> list[Task]:
    return [layernorm_task(4096, n) for n in (1024, 2048, 4096)]


def task_spec(task: Task) -> dict[str, Any]:
    """JSON-serializable description from which `task_from_spec` rebuilds the task."""
    return {
        "family": task.family,
        "params": task.params,
        "dtype": str(task.dtype).removeprefix("torch."),
    }


def task_from_spec(spec: dict[str, Any]) -> Task:
    """Rebuild a task in another process. Tasks hold closures, so they are not picklable."""
    dtype = getattr(torch, spec["dtype"])
    params = spec["params"]
    if spec["family"] == "layernorm":
        return layernorm_task(params["M"], params["N"], dtype)
    raise ValueError(f"unknown task family: {spec['family']!r}")
