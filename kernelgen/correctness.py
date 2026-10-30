"""Correctness gate: a candidate must match the PyTorch reference on every seed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from kernelgen.policy import NativeOpGuard, PolicyViolation
from kernelgen.tasks import Task


@dataclass
class CorrectnessResult:
    correct: bool
    max_abs_err: float | None = None
    reason: str = ""


def _clone(args: tuple) -> list:
    return [a.clone() if isinstance(a, torch.Tensor) else a for a in args]


def _unchanged(before: list, after: tuple) -> bool:
    for b, a in zip(before, after):
        if isinstance(b, torch.Tensor) and not torch.equal(b, a):
            return False
    return True


def check_correctness(
    task: Task,
    fn: Callable,
    device: str,
    seeds: Sequence[int] = (0, 1, 2),
    forbid_native_ops: bool = False,
) -> CorrectnessResult:
    """Run `fn` and the reference on a fresh input per seed and compare.

    Exceptions raised by `fn` are not caught here, so the caller can tell a
    crash from a wrong answer. With `forbid_native_ops`, only `fn` runs under
    NativeOpGuard (the reference does not), and PolicyViolation is raised if it
    triggered any aten op that computes.
    """
    worst = 0.0
    for seed in seeds:
        inputs = task.make_inputs(seed, device)
        snapshot = _clone(inputs)
        expected = (task.truth or task.reference)(*inputs)

        guard = NativeOpGuard()
        if forbid_native_ops:
            with guard:
                out = fn(*inputs)
            if guard.violations:
                raise PolicyViolation(guard.violations)
        else:
            out = fn(*inputs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        if not isinstance(out, torch.Tensor):
            return CorrectnessResult(False, None, f"seed {seed}: output is {type(out).__name__}, not a Tensor")
        if out.shape != expected.shape:
            return CorrectnessResult(False, None, f"seed {seed}: shape {tuple(out.shape)} != {tuple(expected.shape)}")
        if out.dtype != expected.dtype:
            return CorrectnessResult(False, None, f"seed {seed}: dtype {out.dtype} != {expected.dtype}")
        if out.device != expected.device:
            return CorrectnessResult(False, None, f"seed {seed}: device {out.device} != {expected.device}")
        if not _unchanged(snapshot, inputs):
            return CorrectnessResult(False, None, f"seed {seed}: candidate modified its inputs")
        if not bool(torch.isfinite(out).all()):
            return CorrectnessResult(False, None, f"seed {seed}: output has NaN or Inf")

        err = float((out.float() - expected.float()).abs().max())
        worst = max(worst, err)
        if not torch.allclose(out, expected, rtol=task.rtol, atol=task.atol):
            return CorrectnessResult(False, err, f"seed {seed}: max abs err {err:.3e} exceeds tolerance")

    return CorrectnessResult(True, worst)
