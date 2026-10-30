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

**12 VERIFIED, 2 REFUTED, 5 PENDING** of 19 claims.

| ID | Claim | Status | Measured | Evidence |
| --- | --- | --- | --- | --- |
| D1 | The dataset has between 500 and 2000 kernel records, built from templates, none on a held-out shape. | **VERIFIED** | 720 records, 327 distinct kernel sources (data/kernels.jsonl) | data/kernels.jsonl, data/heldout_tasks.json |
| D2 | Every dataset record has a current annotation: its correctness and speedup measured on the GPU. | **VERIFIED** | 720 of 720 records annotated, 720 correct (data/annotations.jsonl) | data/annotations.jsonl |
| B1 | The benchmark measured the template kernel on all 50 held-out tasks, and every measurement is valid. | **VERIFIED** | 1.000 == 1.000 (50 of 50 rows are valid) | `results/kernel_benchmarks.csv` |
| B2 | The layernorm template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out layernorm tasks is above 1. | **VERIFIED** | 5.060 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=layernorm) |
| B3 | The rmsnorm template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out rmsnorm tasks is above 1. | **VERIFIED** | 2.692 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=rmsnorm) |
| B4 | The softmax template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out softmax tasks is above 1. | **VERIFIED** | 2.502 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=softmax) |
| B5 | The gemm template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out gemm tasks is above 1. | **REFUTED** | 0.898 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=gemm) |
| B6 | The gemm_bias_relu template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out gemm_bias_relu tasks is above 1. | **VERIFIED** | 1.215 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=gemm_bias_relu) |
| B7 | The attention template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out attention tasks is above 1. | **VERIFIED** | 3.316 > 1.000 (over 10 tasks) | `results/kernel_benchmarks.csv` (family=attention) |
| B8 | The layernorm template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out layernorm tasks is above 1. | **VERIFIED** | 1.016 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=layernorm) |
| B9 | The rmsnorm template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out rmsnorm tasks is above 1. | **VERIFIED** | 1.017 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=rmsnorm) |
| B10 | The softmax template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out softmax tasks is above 1. | **VERIFIED** | 1.015 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=softmax) |
| B11 | The gemm_bias_relu template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out gemm_bias_relu tasks is above 1. | **VERIFIED** | 1.103 > 1.000 (over 8 tasks) | `results/kernel_benchmarks.csv` (family=gemm_bias_relu) |
| B12 | The attention template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out attention tasks is above 1. | **REFUTED** | 0.953 > 1.000 (over 10 tasks) | `results/kernel_benchmarks.csv` (family=attention) |
| G1 | After fine-tuning, the model writes a correct kernel for more held-out tasks than the untuned base model does (pass@k). | **PENDING** | not measured: results/generated_sft.csv does not exist yet | `results/generated_sft.csv`<br>`results/generated_base.csv` |
| G2 | After reinforcement learning, the model's best-of-k kernels are faster than the fine-tuned model's (geometric mean over held-out tasks with a correct sample). | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv`<br>`results/generated_sft.csv` |
| G3 | The reinforcement-learned model's best-of-k kernels beat the naive PyTorch baseline: geometric-mean speedup is above 1. | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv` |
| G4 | Reinforcement learning did not cost correctness: pass@k is at least the fine-tuned model's. | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv`<br>`results/generated_sft.csv` |
| G5 | The reinforcement-learned model writes a kernel that beats PyTorch's own optimized op on at least one held-out task. | **PENDING** | not measured: results/generated_rl.csv does not exist yet | `results/generated_rl.csv` |

## Template kernels on the held-out tasks

Source: `results/kernel_benchmarks.csv`. The line is the row's line in that file.

| Line | Task | Candidate | Status | Speedup | Speedup vs library |
| --- | --- | --- | --- | --- | --- |
| results/kernel_benchmarks.csv:2 | layernorm_M4096_N640_float32 | `template:layernorm:{"block_n":1024,"n...` | ok | 2.957 | 1.043 |
| results/kernel_benchmarks.csv:3 | layernorm_M2048_N1280_float32 | `template:layernorm:{"block_n":2048,"n...` | ok | 2.649 | 1.054 |
| results/kernel_benchmarks.csv:4 | layernorm_M4096_N2560_float32 | `template:layernorm:{"block_n":4096,"n...` | ok | 6.701 | 1.003 |
| results/kernel_benchmarks.csv:5 | layernorm_M1024_N4096_float32 | `template:layernorm:{"block_n":4096,"n...` | ok | 3.931 | 1.042 |
| results/kernel_benchmarks.csv:6 | layernorm_M2048_N5120_float32 | `template:layernorm:{"block_n":8192,"n...` | ok | 6.608 | 0.994 |
| results/kernel_benchmarks.csv:7 | layernorm_M4096_N6144_float32 | `template:layernorm:{"block_n":8192,"n...` | ok | 6.903 | 0.984 |
| results/kernel_benchmarks.csv:8 | layernorm_M1024_N8192_float32 | `template:layernorm:{"block_n":8192,"n...` | ok | 6.474 | 1.024 |
| results/kernel_benchmarks.csv:9 | layernorm_M8192_N3584_float32 | `template:layernorm:{"block_n":4096,"n...` | ok | 7.050 | 0.987 |
| results/kernel_benchmarks.csv:10 | rmsnorm_M4096_N640_float32 | `template:rmsnorm:{"block_n":1024,"num...` | ok | 1.762 | 1.043 |
| results/kernel_benchmarks.csv:11 | rmsnorm_M2048_N1280_float32 | `template:rmsnorm:{"block_n":2048,"num...` | ok | 1.902 | 1.043 |
| results/kernel_benchmarks.csv:12 | rmsnorm_M4096_N2560_float32 | `template:rmsnorm:{"block_n":4096,"num...` | ok | 3.346 | 0.992 |
| results/kernel_benchmarks.csv:13 | rmsnorm_M1024_N4096_float32 | `template:rmsnorm:{"block_n":4096,"num...` | ok | 1.918 | 1.034 |
| results/kernel_benchmarks.csv:14 | rmsnorm_M2048_N5120_float32 | `template:rmsnorm:{"block_n":8192,"num...` | ok | 3.302 | 0.992 |
| results/kernel_benchmarks.csv:15 | rmsnorm_M4096_N6144_float32 | `template:rmsnorm:{"block_n":8192,"num...` | ok | 3.433 | 0.992 |
| results/kernel_benchmarks.csv:16 | rmsnorm_M1024_N8192_float32 | `template:rmsnorm:{"block_n":8192,"num...` | ok | 3.196 | 1.058 |
| results/kernel_benchmarks.csv:17 | rmsnorm_M8192_N3584_float32 | `template:rmsnorm:{"block_n":4096,"num...` | ok | 3.541 | 0.985 |
| results/kernel_benchmarks.csv:18 | softmax_M4096_N384_float32 | `template:softmax:{"block_n":512,"num_...` | ok | 1.982 | 1.018 |
| results/kernel_benchmarks.csv:19 | softmax_M2048_N768_float32 | `template:softmax:{"block_n":1024,"num...` | ok | 1.965 | 0.982 |
| results/kernel_benchmarks.csv:20 | softmax_M4096_N1536_float32 | `template:softmax:{"block_n":2048,"num...` | ok | 3.097 | 1.005 |
| results/kernel_benchmarks.csv:21 | softmax_M1024_N3072_float32 | `template:softmax:{"block_n":4096,"num...` | ok | 1.927 | 1.064 |
| results/kernel_benchmarks.csv:22 | softmax_M2048_N5120_float32 | `template:softmax:{"block_n":8192,"num...` | ok | 3.642 | 1.006 |
| results/kernel_benchmarks.csv:23 | softmax_M1024_N6144_float32 | `template:softmax:{"block_n":8192,"num...` | ok | 2.940 | 1.014 |
| results/kernel_benchmarks.csv:24 | softmax_M512_N8192_float32 | `template:softmax:{"block_n":8192,"num...` | ok | 2.276 | 1.014 |
| results/kernel_benchmarks.csv:25 | softmax_M4096_N1280_float32 | `template:softmax:{"block_n":2048,"num...` | ok | 2.710 | 1.022 |
| results/kernel_benchmarks.csv:26 | gemm_M768_N768_K768_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 0.900 | - |
| results/kernel_benchmarks.csv:27 | gemm_M1536_N1536_K1536_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 1.072 | - |
| results/kernel_benchmarks.csv:28 | gemm_M3072_N1024_K1024_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 1.029 | - |
| results/kernel_benchmarks.csv:29 | gemm_M1024_N3072_K512_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 1.044 | - |
| results/kernel_benchmarks.csv:30 | gemm_M1000_N1000_K1000_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 0.501 | - |
| results/kernel_benchmarks.csv:31 | gemm_M4096_N1024_K256_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 1.024 | - |
| results/kernel_benchmarks.csv:32 | gemm_M512_N512_K4096_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 0.761 | - |
| results/kernel_benchmarks.csv:33 | gemm_M2048_N1536_K768_float16 | `template:gemm:{"block_k":32,"block_m"...` | ok | 1.045 | - |
| results/kernel_benchmarks.csv:34 | gemm_bias_relu_M768_N768_K768_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 1.171 | 1.261 |
| results/kernel_benchmarks.csv:35 | gemm_bias_relu_M1536_N1536_K1536_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 1.311 | 1.188 |
| results/kernel_benchmarks.csv:36 | gemm_bias_relu_M3072_N1024_K1024_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 1.390 | 1.220 |
| results/kernel_benchmarks.csv:37 | gemm_bias_relu_M1024_N3072_K512_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 1.598 | 1.351 |
| results/kernel_benchmarks.csv:38 | gemm_bias_relu_M1000_N1000_K1000_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 0.573 | 0.502 |
| results/kernel_benchmarks.csv:39 | gemm_bias_relu_M4096_N1024_K256_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 2.159 | 1.706 |
| results/kernel_benchmarks.csv:40 | gemm_bias_relu_M512_N512_K4096_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 0.831 | 0.810 |
| results/kernel_benchmarks.csv:41 | gemm_bias_relu_M2048_N1536_K768_float16 | `template:gemm_bias_relu:{"block_k":32...` | ok | 1.352 | 1.280 |
| results/kernel_benchmarks.csv:42 | attention_B2_H8_S192_D64_float16 | `template:attention:{"block_m":64,"blo...` | ok | 2.529 | 1.805 |
| results/kernel_benchmarks.csv:43 | attention_B2_H8_S384_D64_float16 | `template:attention:{"block_m":64,"blo...` | ok | 2.564 | 1.000 |
| results/kernel_benchmarks.csv:44 | attention_B4_H8_S768_D64_float16 | `template:attention:{"block_m":64,"blo...` | ok | 5.517 | 0.973 |
| results/kernel_benchmarks.csv:45 | attention_B2_H8_S1536_D64_float16 | `template:attention:{"block_m":64,"blo...` | ok | 6.073 | 0.957 |
| results/kernel_benchmarks.csv:46 | attention_B2_H16_S512_D128_float16 | `template:attention:{"block_m":64,"blo...` | ok | 1.650 | 0.678 |
| results/kernel_benchmarks.csv:47 | attention_B1_H8_S2048_D64_float16 | `template:attention:{"block_m":64,"blo...` | ok | 6.000 | 1.040 |
| results/kernel_benchmarks.csv:48 | attention_B4_H8_S256_D128_float16 | `template:attention:{"block_m":64,"blo...` | ok | 1.148 | 0.716 |
| results/kernel_benchmarks.csv:49 | attention_B2_H8_S640_D32_float16 | `template:attention:{"block_m":64,"blo...` | ok | 5.164 | 0.957 |
| results/kernel_benchmarks.csv:50 | attention_B2_H8_S1024_D128_float16 | `template:attention:{"block_m":64,"blo...` | ok | 2.014 | 0.724 |
| results/kernel_benchmarks.csv:51 | attention_B1_H16_S1024_D64_float16 | `template:attention:{"block_m":64,"blo...` | ok | 6.251 | 1.054 |
