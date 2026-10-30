# SPEC

Scope of this project. Anything not stated here is not claimed.

## Goal

Fine-tune a small code model to write optimized GPU kernels (CUDA and Triton),
then improve it with reinforcement learning where the reward is the measured
speedup over a PyTorch baseline, gated on correctness.

## Scope

| Item | Decision |
| --- | --- |
| Hardware | NVIDIA RTX 4060, 8 GB VRAM (Ada, compute capability 8.9), 24 GB system RAM |
| CUDA | 12.x |
| Host | WSL2, Ubuntu 22.04 |
| Model | Qwen2.5-Coder-3B-Instruct, 4-bit QLoRA |
| Reward | Speedup vs. PyTorch baseline, with a correctness gate |
| Benchmark | 50 held-out kernel tasks (GEMM, attention, layernorm, ...) |

## Reward

```
reward  = speedup * correct
speedup = t_baseline / t_candidate
correct = 1 if the candidate compiles, runs, and matches the PyTorch
          reference on every test input within tolerance, else 0
```

- A kernel that fails to compile, crashes, or returns wrong results gets reward 0.
- Timing uses CUDA events after warmup, with the same protocol for baseline and
  candidate. The exact protocol lives in the benchmark module.
- The baseline for each task is recorded in the task manifest. A speedup is only
  meaningful against the baseline it was measured against.
- No clipping, normalization, or baseline subtraction is part of the reward as
  specified. If any is added later, it is documented here and in `CLAIMS.md`.

## Deliverables

1. **Dataset.** 500-2000 kernels, each annotated with correctness and measured
   speedup, plus a script that re-measures and verifies the annotations.
   Annotations come from measurements on the target GPU, never entered by hand.
   The kernels are generated programmatically from parameterized templates
   (for example GEMM with varying matrix sizes and tilings, LayerNorm with
   varying block sizes), not written by hand one by one.
2. **Fine-tuning.** A LoRA/QLoRA script. Logs training loss and a few generated
   samples per epoch.
3. **RL loop.** GRPO (or a simplified PPO) using the reward above.
4. **Benchmark harness.** Runs generated kernels on the RTX 4060 and writes
   `results/kernel_benchmarks.csv`. Includes a test asserting that a
   known-optimized kernel beats a naive PyTorch baseline.
5. **Profiling.** `torch.profiler` traces are the primary profiling output.
   Nsight Compute (`ncu`) and Nsight Systems (`nsys`) are optional, because
   their support under WSL2 is not guaranteed.
6. **`CLAIMS.md`.** A ledger of every claim. Every speedup number points to a row
   in `results/kernel_benchmarks.csv`.

## Acceptance

```
make install
make test
make benchmark   # writes results/kernel_benchmarks.csv
```

## Working split

Code is written and reviewed on a machine without a GPU. Everything that needs the
GPU (dataset annotation, training, RL, benchmarks, profiling) is run on the target
machine, and the resulting artifacts are committed afterward.

Tests that need a GPU skip when CUDA is unavailable. A skipped test is not
evidence of anything.

## Claims and failure policy

- A claim without a matching CSV row is `UNVERIFIED`.
- If generated kernels are slower than baseline, the bottleneck is profiled and
  logged, and success is not claimed.
- If the RL loop diverges, the reward curve and the generated code are logged.

## Not yet decided

- Task families beyond GEMM, attention, and layernorm.
- Whether GRPO fits in 8 GB VRAM at the intended group size (not measured).
- Whether `ncu` / `nsys` work under this WSL2 setup (not checked).

## Planned layout

```
kernelgen/    package: dataset, train, rl, benchmark
tests/
data/         annotated kernels
results/      kernel_benchmarks.csv, profiler output
CLAIMS.md
```

## Definition of done

- [ ] `make test` green
- [ ] `results/kernel_benchmarks.csv` committed
- [ ] Profiler trace committed (Nsight profile if available under WSL2)
- [ ] `CLAIMS.md` entries all `VERIFIED`
- [ ] Git history shows incremental commits
