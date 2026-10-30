"""Kernel tasks: inputs, PyTorch baseline, and comparison tolerances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

# Seed for the inputs used when timing or profiling (the correctness gate uses others).
TIMING_SEED = 1234


@dataclass(frozen=True, eq=False)
class Task:
    name: str
    family: str
    params: dict[str, Any]
    dtype: torch.dtype
    # Minimum bytes any implementation must move through memory for one call:
    # every input read once and every output written once. Effective bandwidth
    # is this divided by kernel time; it is not measured DRAM traffic.
    bytes_moved: int
    # (seed, device) -> positional arguments for both reference and candidate
    make_inputs: Callable[[int, str], tuple]
    # The PyTorch baseline. Speedup is measured against this function.
    reference: Callable[..., torch.Tensor]
    baseline_desc: str
    rtol: float
    atol: float
    # PyTorch's own optimized op for the same computation, if there is one
    # (F.layer_norm, torch.softmax, ...). Candidates are also timed against it,
    # so a speedup over the naive baseline cannot be mistaken for a speedup over
    # the library. Same call signature as `reference`.
    library: Callable[..., torch.Tensor] | None = None
    library_desc: str = ""
    # If set, the correctness gate compares candidates to this instead of to
    # `reference`: a float32 computation rounded to the output dtype. A
    # low-precision baseline carries its own rounding error, which should not
    # decide what counts as correct. Same call signature as `reference`.
    truth: Callable[..., torch.Tensor] | None = None


def _naive_layernorm(x, weight, bias, eps: float = 1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) / torch.sqrt(var + eps) * weight + bias


def _library_layernorm(x, weight, bias, eps: float = 1e-5):
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def layernorm_task(m: int, n: int, dtype: torch.dtype = torch.float32) -> Task:
    def make_inputs(seed: int, device: str) -> tuple:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x = torch.randn(m, n, generator=g, device=device, dtype=dtype) * 2.0 + 1.5
        weight = torch.randn(n, generator=g, device=device, dtype=dtype)
        bias = torch.randn(n, generator=g, device=device, dtype=dtype)
        return x, weight, bias

    dtype_tag = str(dtype).removeprefix("torch.")
    elem = torch.empty((), dtype=dtype).element_size()
    return Task(
        name=f"layernorm_M{m}_N{n}_{dtype_tag}",
        family="layernorm",
        params={"M": m, "N": n},
        dtype=dtype,
        bytes_moved=(2 * m * n + 2 * n) * elem,  # read x, write y, read weight and bias
        make_inputs=make_inputs,
        reference=_naive_layernorm,
        baseline_desc="eager PyTorch LayerNorm from mean/sub/pow/sqrt/div/mul/add ops",
        rtol=1e-4,
        atol=1e-4,
        library=_library_layernorm,
        library_desc="torch.nn.functional.layer_norm",
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
    from kernelgen.families import get_family  # imported here: the families import this module

    return get_family(spec["family"]).build_task(spec["params"], getattr(torch, spec["dtype"]))
