# CLAIMS

Every claim this project makes, with the evidence it rests on. This file is produced by
`make claims` from the claim definitions in `kernelgen/claims.py` and the CSV files they
point to; a test fails if it is out of date. A status is computed from the evidence, not
written by hand:

- **VERIFIED**: the evidence exists, covers every task the claim is about, and the claim holds.
- **REFUTED**: the evidence exists and the claim does not hold. A refuted claim stays here.
- **PENDING**: the CSV does not exist yet, or does not cover every task. Nothing is claimed.

Speedups are over the naive eager PyTorch baseline of each task unless the claim says
`speedup_vs_library`, which is over PyTorch's own optimized op. Aggregates are geometric means.
Each CSV records the GPU, driver stack, and library versions of its measurements in its
`gpu_name`, `torch_version`, `cuda_version`, and `triton_version` columns.

**1 VERIFIED, 0 REFUTED, 18 PENDING** of 19 claims.

| ID | Claim | Status | Measured | Evidence |
| --- | --- | --- | --- | --- |
| D1 | The dataset has between 500 and 2000 kernel records, built from templates, none on a held-out shape. | **VERIFIED** | 720 records, 327 distinct kernel sources (data/kernels.jsonl) | data/kernels.jsonl, data/heldout_tasks.json |
| D2 | Every dataset record has a current annotation: its correctness and speedup measured on the GPU. | **PENDING** | not measured: data/annotations.jsonl does not exist yet (run `make annotate` on the GPU) | data/annotations.jsonl |
| B1 | The benchmark measured the template kernel on all 50 held-out tasks, and every measurement is valid. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` |
| B2 | The layernorm template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out layernorm tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=layernorm) |
| B3 | The rmsnorm template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out rmsnorm tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=rmsnorm) |
| B4 | The softmax template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out softmax tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=softmax) |
| B5 | The gemm template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out gemm tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=gemm) |
| B6 | The gemm_bias_relu template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out gemm_bias_relu tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=gemm_bias_relu) |
| B7 | The attention template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out attention tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=attention) |
| B8 | The layernorm template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out layernorm tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=layernorm) |
| B9 | The rmsnorm template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out rmsnorm tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=rmsnorm) |
| B10 | The softmax template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out softmax tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=softmax) |
| B11 | The gemm_bias_relu template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out gemm_bias_relu tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=gemm_bias_relu) |
| B12 | The attention template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out attention tasks is above 1. | **PENDING** | not measured: results/kernel_benchmarks.csv does not exist yet | `results/kernel_benchmarks.csv` (family=attention) |
| G1 | After fine-tuning, the model writes a correct kernel for more held-out tasks than the untuned base model does (pass@k). | **PENDING** | not measured: results/generated_sft.csv does not exist yet | `results/generated_sft.csv`<br>`results/generated_base.csv` |
| G2 | After reinforcement learning, the model's best-of-k kernels are faster than the fine-tuned model's (geometric mean over held-out tasks with a correct sample). | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv`<br>`results/generated_sft.csv` |
| G3 | The reinforcement-learned model's best-of-k kernels beat the naive PyTorch baseline: geometric-mean speedup is above 1. | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv` |
| G4 | Reinforcement learning did not cost correctness: pass@k is at least the fine-tuned model's. | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv`<br>`results/generated_sft.csv` |
| G5 | The reinforcement-learned model writes a kernel that beats PyTorch's own optimized op on at least one held-out task. | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv` |
