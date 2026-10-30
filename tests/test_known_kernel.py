"""Harness sanity check: a known-optimized kernel must beat the naive PyTorch baseline.

Runs through the same path as generated code: policy check, worker process,
correctness gate, timing.
"""

import pytest

pytest.importorskip("triton")

from kernelgen.benchmark import KNOWN_KERNELS
from kernelgen.sandbox import evaluate_isolated
from kernelgen.tasks import layernorm_task

pytestmark = pytest.mark.gpu


def test_fused_triton_layernorm_beats_naive_pytorch():
    candidate_id, path = KNOWN_KERNELS["layernorm"]
    task = layernorm_task(4096, 4096)

    row = evaluate_isolated(
        task, path.read_text(), candidate_id, "cuda", warmup=10, reps=50, timeout_s=300.0
    )

    assert row["status"] == "ok", row["error"]
    assert row["correct"] == 1
    assert row["speedup"] > 1.0, (
        f"fused kernel {row['candidate_ms']:.3f} ms vs naive baseline "
        f"{row['baseline_ms']:.3f} ms (speedup {row['speedup']:.2f}x)"
    )
    assert row["reward"] == pytest.approx(row["speedup"])
