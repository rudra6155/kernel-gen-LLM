# SPEC

Scope of this project. Anything not stated here is not claimed.

## Goal

Fine-tune a small code model to write optimized GPU kernels in Triton, then improve it
with reinforcement learning where the reward is the measured speedup over a PyTorch
baseline, gated on correctness.

## Scope

| Item | Decision |
| --- | --- |
| Hardware | NVIDIA RTX 4060, 8 GB VRAM (Ada, compute capability 8.9), 24 GB system RAM |
| CUDA | 12.x |
| Host | WSL2, Ubuntu 22.04 |
| Model | Qwen2.5-Coder-3B-Instruct, 4-bit QLoRA |
| Reward | Speedup vs. PyTorch baseline, with a correctness gate |
| Benchmark | 50 held-out kernel tasks across six families (below) |
| Candidate language | Triton. CUDA C++ candidates are not supported. |

## Task families

Each family has a naive eager PyTorch baseline (the speedup denominator) and, where PyTorch
has one, its own optimized op (`speedup_vs_library`, a harder bar). Normalization families use
float32, matrix families float16 with float32 accumulation.

| Family | Baseline | PyTorch's own op | Held-out tasks |
| --- | --- | --- | --- |
| layernorm | mean/sub/pow/sqrt/div/mul/add | `F.layer_norm` | 8 |
| rmsnorm | mul/mean/add/rsqrt/mul/mul | `F.rms_norm` | 8 |
| softmax | max/sub/exp/sum/div | `torch.softmax` | 8 |
| gemm | `torch.matmul` (cuBLAS) | none: the baseline is the library | 8 |
| gemm_bias_relu | `relu(matmul + bias)`, three kernels | `relu(addmm(bias, a, b))` | 8 |
| attention (non-causal) | `softmax(q @ k^T * scale) @ v` | `scaled_dot_product_attention` | 10 |

Training uses other shapes of the same families; training and held-out shapes never overlap.

## Reward

```
reward  = speedup * correct
speedup = t_baseline / t_candidate
correct = 1 if the candidate passes the policy, compiles, runs, and matches the truth on
          every test input within tolerance, else 0
```

- A kernel that fails to compile, crashes, times out, breaks the policy, or returns wrong
  results gets reward 0.
- No clipping, normalization, or baseline subtraction is part of the reward. (GRPO standardizes
  rewards within a group to form advantages; that is part of the algorithm, not of the reward.)
- The truth is the baseline for float32 families. For float16 families it is a float32
  computation rounded to float16, because the naive float16 baseline carries rounding error of
  its own and fails a tight tolerance on some shapes.
- Timing uses CUDA events after warmup and an L2 flush, baseline and candidate interleaved,
  median of the repetitions. The exact protocol is in `kernelgen/timing.py`.

## Candidate rules

A candidate is one Python source file that defines `solution(...)` and at least one
`@triton.jit` kernel that `solution` launches. The harness enforces this in four layers, so
that speedup can only come from the generated kernel:

1. **Static check** (`kernelgen/policy.py`, runs before any candidate code executes). Imports
   are limited to `torch`, `triton`, `triton.language`, `math`, `typing`, and `functools`. Host
   code may use `torch` only for allocation, dtypes, and device queries, and a fixed set of
   tensor methods (views, casts, metadata). Everything else (`torch.matmul`, `torch.nn`,
   `x.mean()`, `x @ w`, `getattr`, `exec`, dunder access) is rejected.
2. **Native-op guard** (same module, runs during the correctness check). Records every aten op
   the candidate triggers and rejects any that computes. This catches what the static check
   cannot see, such as `x * w + b` on tensors.
3. **Process isolation** (`kernelgen/sandbox.py`). Each candidate runs in its own worker
   process with a wall-clock limit. A hang, crash, or exit becomes a row with status `timeout`
   or `crash` and reward 0; the run continues.
4. **Fresh-data timing** (`kernelgen/timing.py`, `benchmark_fresh`). Every timed call, for the
   baseline and the candidate alike, gets new input data, and the candidate's output on that
   data is checked against the truth outside the timer. A candidate that caches its result (for
   example keyed on the input pointer) gains nothing: the cache either misses, so the real work
   is done, or returns a stale result that is rejected. Without this, such a cache measured a
   53x speedup in a test.

Rows that fail these checks have status `rejected`, `incorrect`, `timeout`, or `crash` and
reward 0. They are written to the CSV like any other row.

Known limits: this is a policy filter, not a security sandbox. A candidate that behaves
differently while being timed is only caught if that makes its output wrong. The allowlists
were written without seeing real model output, and will be widened only when a legitimate
kernel is wrongly rejected. The 120 s default timeout is a starting value, not a measured one.

## Deliverables

1. **Dataset** (`kernelgen/dataset.py`, `annotate.py`, `verify_dataset.py`). 720 records, each a
   (task, config) pair rendered from a parameterized Triton template, on training shapes only.
   Many records share a kernel source (327 distinct sources), because a normalization kernel's
   source does not depend on the number of rows; each record still gets its own measured
   annotation. Annotations (correctness and speedup) come from GPU measurements, never from
   hand entry, and the verifier re-checks them for staleness, consistency, and by re-measuring
   a sample.
2. **Fine-tuning** (`kernelgen/finetune.py`). QLoRA on kernels the annotation found correct and
   faster than the baseline. Logs training and validation loss, and a few generated samples
   per epoch. Validation is split by task shape.
3. **RL loop** (`kernelgen/rl.py`). GRPO with reward = speedup * correct, a KL penalty to the
   starting policy, and divergence detection that writes the reward curve and the generated code.
4. **Benchmark harness** (`kernelgen/benchmark.py`, `evaluate_model.py`). `make benchmark`
   measures the template kernel on each of the 50 held-out tasks and writes
   `results/kernel_benchmarks.csv`; `make eval-model` does the same for kernels a model writes.
   A test asserts that a known-optimized kernel beats a naive PyTorch baseline (on the GPU).
5. **Profiling** (`kernelgen/profiling.py`, `diagnose.py`, `nsight.py`). `torch.profiler`
   traces are the primary profiling output. `make profile` writes `results/profile_summary.csv`
   and a Chrome trace per run, with kernel time, launches per call, and effective bandwidth.
   - Effective bandwidth is the task's minimum traffic (every input read once, every output
     written once) divided by GPU kernel time from the trace. It is not measured DRAM traffic;
     that needs `ncu`.
   - No peak bandwidth is assumed. Each run also measures a device-to-device copy on the same
     GPU as an empirical reference, and `PEAK_GBPS` takes the card's spec value if known.
   - Each run is also timed with CUDA events outside the profiler; a run whose kernel time
     exceeds that is marked `suspect`.
   - When a correct generated kernel is slower than the baseline, the slowest few are profiled
     and the measured facts are logged, with no cause asserted.
   - Nsight Compute and Nsight Systems are optional (`make nsight`); the runner never fails
     because the tools are missing or cannot collect, which is common under WSL2.
6. **`CLAIMS.md`** (`kernelgen/claims.py`). A ledger generated from claim definitions and the
   CSVs they point to. Statuses are computed: VERIFIED, REFUTED, or PENDING. Every measured
   number carries the CSV file and line it came from.

## Acceptance

```
make install
make test
make benchmark   # writes results/kernel_benchmarks.csv
```

## Working split

Code was written and tested on a machine without a GPU. Everything that needs the GPU (dataset
annotation, training, RL, benchmarks, profiling) is run on the target machine, and the
resulting artifacts are committed afterward. Tests that need a GPU skip when CUDA is
unavailable, and a skipped test is not evidence of anything. Without a GPU the other tests run
Triton kernels in its CPU interpreter, which checks kernel logic but not GPU compilation or
speed; with a GPU the same tests run the kernels for real.

## Claims and failure policy

- A claim without a matching CSV row is `PENDING`; a claim its CSV contradicts is `REFUTED`.
- If generated kernels are slower than baseline, the bottleneck is profiled and logged, and
  success is not claimed.
- If the RL loop diverges, the reward curve and the generated code are logged
  (`divergence_report.json`).

## Open questions

- Whether GRPO fits in 8 GB VRAM at the default group size (not measured).
- Whether the 3B model is capable enough for RL to find signal; early groups with equal rewards
  carry none, and are skipped.
- Whether `ncu` / `nsys` work under this WSL2 setup (not checked).
- Whether CUDA C++ candidates should be added.

## Layout

```
kernelgen/    the package (see Deliverables)
tests/
data/         kernels.jsonl, heldout_tasks.json, annotations.jsonl once measured
results/      CSVs, profiles, traces, training logs
runs/         training output including adapters (not committed)
CLAIMS.md
```

## Definition of done

`make status` computes this checklist from the repository.

- [ ] `make test` green (run it)
- [ ] dataset of 500-2000 kernels, verified
- [ ] dataset annotated with measured correctness and speedup (GPU)
- [ ] `results/kernel_benchmarks.csv` committed, 50 rows (GPU)
- [ ] profiler trace committed (GPU); Nsight profile if available under WSL2
- [ ] fine-tuning logs and RL reward curve committed (GPU)
- [ ] generated kernels of the base, SFT, and RL models evaluated on the held-out tasks (GPU)
- [ ] `CLAIMS.md` entries all `VERIFIED`
- [ ] git history shows incremental commits
