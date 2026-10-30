# kernel-gen-LLM

Fine-tune a 3B code model (Qwen2.5-Coder-3B-Instruct, 4-bit QLoRA) to write fast Triton
kernels, then improve it with reinforcement learning (GRPO) whose reward is the measured
speedup over a PyTorch baseline, gated on correctness. Everything is measured on one
RTX 4060 (8 GB, CUDA 12.x, WSL2).

[`SPEC.md`](SPEC.md) states the scope. [`CLAIMS.md`](CLAIMS.md) lists every claim and
the CSV rows behind it.

## What has been verified, and what has not

All the code and tests were written and run on a machine **without** the RTX 4060. The test
suite passes there using Triton's CPU interpreter, which checks kernel logic (indexing,
masking, reductions) but not GPU compilation or speed. So:

| Verified (CPU) | **Pending: needs the RTX 4060** |
| --- | --- |
| The harness, the sandbox, the candidate policy, the timing protocol | Kernel speedups: nothing here has been timed on a GPU |
| Each family's template (default config plus sampled configs) matches its PyTorch baseline | That Triton compiles every template config on an sm_89 GPU |
| The dataset builds, verifies, and has no held-out shapes | Annotating the dataset (correctness and speedup, measured) |
| The QLoRA loop and GRPO loop run end to end on a tiny model | That the 3B model trains in 8 GB, and that SFT or RL improve it |
| The sign and clipping of the GRPO update, on a tiny model | Nsight under WSL2 (often unavailable) |

`make status` prints the Definition of Done checklist computed from the repository, with
the pending items and the command that produces each. `make claims` regenerates
`CLAIMS.md`; a claim is VERIFIED only when its CSV evidence exists and holds.

## Quick start

```bash
sudo apt install build-essential        # Triton compiles a small C helper
python3 -m venv .venv && . .venv/bin/activate
make install
make test                               # GPU tests run for real when CUDA is available
make benchmark                          # writes results/kernel_benchmarks.csv
```

## The pipeline

Run in this order. Steps marked GPU need the RTX 4060.

| Step | Command | GPU | Produces |
| --- | --- | --- | --- |
| Build the dataset (already committed) | `make dataset` | no | `data/kernels.jsonl`, `data/heldout_tasks.json` |
| Measure every dataset kernel | `make annotate` | yes | `data/annotations.jsonl` (resumable) |
| Check the annotations | `make verify-dataset REMEASURE=20` | for `REMEASURE` | pass or the list of problems |
| Fine-tune (QLoRA) | `make sft` | yes | `runs/sft/` (loss per step, samples per epoch, adapters) |
| Reinforcement learning (GRPO) | `make rl` | yes | `runs/rl/` (reward curve, every generated kernel) |
| Keep the small logs | `make collect-logs` | no | `results/training/` |
| Benchmark the 50 held-out tasks | `make benchmark` | yes | `results/kernel_benchmarks.csv` |
| Profile (torch.profiler) | `make profile` | yes | `results/profile_summary.csv`, `results/traces/` |
| Nsight (optional) | `make nsight` | yes | `results/nsight/` (never fails if unavailable) |
| Evaluate the models | `make eval-model TAG=base`, `TAG=sft ADAPTER=runs/sft/final`, `TAG=rl ADAPTER=runs/rl/final` | yes | `results/generated_<tag>.csv` |
| The ledger and the checklist | `make claims`, `make status` | no | `CLAIMS.md`, the checklist |

**After producing or changing any result file** (`data/annotations.jsonl`, anything in
`results/`), run `make claims` and commit `CLAIMS.md`. `make test` fails while `CLAIMS.md` is out
of date, on purpose: a stale ledger would state claims the evidence no longer supports.

The annotation measures one kernel at a time (kernels timed together disturb each other),
so it takes a long time; how long was not measured. It can be interrupted and resumed.
`make sft` needs `data/annotations.jsonl`. To train on the unmeasured template kernels
instead, pass `SFT_ARGS="--include-unannotated"`.

## How it works

- **Tasks.** Six families, each with a naive eager PyTorch baseline and, where one exists,
  PyTorch's own optimized op: layernorm, rmsnorm, softmax, gemm, gemm with fused bias and
  ReLU, and attention. 720 training kernels are rendered from parameterized Triton templates
  (tile sizes, warps, stages) on training shapes. The benchmark uses 50 held-out tasks on
  shapes that no training example uses.
- **Reward.** `speedup * correct`, where speedup is baseline time over candidate time. A
  kernel that fails to compile, crashes, times out, breaks the policy, or is wrong earns 0.
  Speedup over PyTorch's own op is reported separately (`speedup_vs_library`), because a
  win over the naive baseline is not a win over the library.
- **The candidate must be a Triton kernel.** An AST check runs before any candidate code
  executes (import and `torch` allowlists, no `getattr`/`exec`, a `@triton.jit` kernel that
  is launched), and a dispatch guard records every aten op the candidate triggers, which
  catches tensor arithmetic the AST cannot see.
- **Timing cannot be gamed by remembering results.** Every timed call gets new input data
  and its output is checked against the truth outside the timer. A cache keyed on the
  input pointer measured a 53x speedup under the old protocol and earns nothing now.
- **Isolation.** Each candidate runs in its own process with a wall-clock limit; a hang or
  a crash costs one row, not the run.
- **Correctness** is judged against a float32 truth rounded to the output dtype where the
  baseline is low precision, so the baseline's own rounding error does not set the bar.
- **GRPO** standardizes the rewards within each task's group of samples, uses the clipped
  policy-gradient objective with a KL penalty to the starting policy, and stops with a
  report (the reward curve and the latest generated code) if it diverges.

## Layout

```
kernelgen/        the package: families/, policy, sandbox, benchmark, profiling, dataset,
                  annotate, verify_dataset, finetune, rl, evaluate_model, claims, status
tests/            the test suite (key checks were mutation-tested: breaking them fails the suite)
data/             kernels.jsonl, heldout_tasks.json, and annotations.jsonl once measured
results/          kernel_benchmarks.csv, profiles, traces, generated_<tag>.csv, training logs
runs/             training output including adapters (large, not committed)
SPEC.md  CLAIMS.md
```

## Known limits

- The candidate policy is a filter against reward hacking and accidents, not a security
  sandbox. A candidate that behaves differently while being timed is only caught if that
  makes its output wrong.
- The policy allowlists were written without seeing real model output. They will be widened
  only when a legitimate kernel is wrongly rejected.
- Some thresholds are starting values, not tuned ones: the 120 s per-candidate timeout, the
  divergence thresholds in the RL loop, the shared-memory estimate that prunes tile configs.
- Template configs the 4060 cannot run are not hidden: they become annotated rows with
  their error.
- Only Triton candidates are supported; CUDA C++ is not.
- The Nsight command lines could not be run where they were written.
