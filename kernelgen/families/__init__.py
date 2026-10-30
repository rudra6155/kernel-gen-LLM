"""Kernel task families: baselines, inputs, Triton templates, and shape splits.

Each family module defines a `FAMILY`. A family knows how to
  - build a Task (PyTorch baseline, inputs, tolerances) for given shape parameters,
  - render a Triton kernel source from shape parameters and a tuning config,
  - list the valid configs for a shape,
  - describe itself for the prompt,
  - name its training shapes and its held-out shapes.
The training and held-out shapes of a family never overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from kernelgen.tasks import Task

Params = dict[str, Any]
Config = dict[str, Any]


@dataclass(frozen=True)
class Family:
    name: str
    dtype: str  # torch dtype name; one per family
    signature: str  # the entry point the candidate must define
    build_task: Callable[[Params, torch.dtype], Task]
    render: Callable[[Params, Config], str]
    configs: Callable[[Params], list[Config]]
    default_config: Callable[[Params], Config]
    describe: Callable[[Params], str]
    reference: Callable[..., torch.Tensor]  # the baseline, shown in the prompt
    train_shapes: list[Params]
    heldout_shapes: list[Params]

    def task(self, params: Params) -> Task:
        return self.build_task(params, getattr(torch, self.dtype))

    def spec(self, params: Params) -> dict[str, Any]:
        return {"family": self.name, "params": dict(params), "dtype": self.dtype}


_REGISTRY: dict[str, Family] = {}


def register(family: Family) -> Family:
    _REGISTRY[family.name] = family
    return family


def get_family(name: str) -> Family:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown task family: {name!r}") from None


def all_families() -> list[Family]:
    return [_REGISTRY[name] for name in FAMILY_ORDER]


FAMILY_ORDER = ["layernorm", "rmsnorm", "softmax", "gemm", "gemm_bias_relu", "attention"]

# Importing the modules registers the families.
from kernelgen.families import attention, gemm, layernorm, rmsnorm, softmax  # noqa: E402,F401
