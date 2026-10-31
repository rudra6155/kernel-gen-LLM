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

**15 VERIFIED, 4 REFUTED, 0 PENDING** of 19 claims.

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
| G1 | After fine-tuning, the model writes a correct kernel for more held-out tasks than the untuned base model does (pass@k). | **VERIFIED** | 1.000 > 0.000 (50 of 50 tasks have a correct sample) | `results/generated_sft.csv`<br>`results/generated_base.csv` |
| G2 | After reinforcement learning, the model's best-of-k kernels are faster than the fine-tuned model's (geometric mean over held-out tasks with a correct sample). | **REFUTED** | 2.093 > 2.390 (over 39 tasks) | `results/generated_rl.csv`<br>`results/generated_sft.csv` |
| G3 | The reinforcement-learned model's best-of-k kernels beat the naive PyTorch baseline: geometric-mean speedup is above 1. | **VERIFIED** | 2.093 > 1.000 (over 39 tasks) | `results/generated_rl.csv` |
| G4 | Reinforcement learning did not cost correctness: pass@k is at least the fine-tuned model's. | **REFUTED** | 0.780 >= 1.000 (39 of 50 tasks have a correct sample) | `results/generated_rl.csv`<br>`results/generated_sft.csv` |
| G5 | The reinforcement-learned model writes a kernel that beats PyTorch's own optimized op on at least one held-out task. | **VERIFIED** | 20.000 >= 1.000 (of 31 tasks with a measurement) | `results/generated_rl.csv` |

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

## Model `base`: fastest correct sample per held-out task

Source: `results/generated_base.csv`. The line is the row's line in that file.

| Line | Task | Candidate | Status | Speedup | Speedup vs library |
| --- | --- | --- | --- | --- | --- |

## Model `sft`: fastest correct sample per held-out task

Source: `results/generated_sft.csv`. The line is the row's line in that file.

| Line | Task | Candidate | Status | Speedup | Speedup vs library |
| --- | --- | --- | --- | --- | --- |
| results/generated_sft.csv:2 | layernorm_M4096_N640_float32 | `gen:sft:layernorm_M4096_N640_float32:0` | ok | 3.016 | 1.021 |
| results/generated_sft.csv:7 | layernorm_M2048_N1280_float32 | `gen:sft:layernorm_M2048_N1280_float32:1` | ok | 2.906 | 1.011 |
| results/generated_sft.csv:13 | layernorm_M4096_N2560_float32 | `gen:sft:layernorm_M4096_N2560_float32:3` | ok | 6.810 | 0.997 |
| results/generated_sft.csv:16 | layernorm_M1024_N4096_float32 | `gen:sft:layernorm_M1024_N4096_float32:2` | ok | 3.979 | 1.042 |
| results/generated_sft.csv:21 | layernorm_M2048_N5120_float32 | `gen:sft:layernorm_M2048_N5120_float32:3` | ok | 6.822 | 1.006 |
| results/generated_sft.csv:22 | layernorm_M4096_N6144_float32 | `gen:sft:layernorm_M4096_N6144_float32:0` | ok | 7.319 | 1.019 |
| results/generated_sft.csv:26 | layernorm_M1024_N8192_float32 | `gen:sft:layernorm_M1024_N8192_float32:0` | ok | 6.680 | 1.021 |
| results/generated_sft.csv:31 | layernorm_M8192_N3584_float32 | `gen:sft:layernorm_M8192_N3584_float32:1` | ok | 6.993 | 1.006 |
| results/generated_sft.csv:36 | rmsnorm_M4096_N640_float32 | `gen:sft:rmsnorm_M4096_N640_float32:2` | ok | 1.872 | 1.022 |
| results/generated_sft.csv:40 | rmsnorm_M2048_N1280_float32 | `gen:sft:rmsnorm_M2048_N1280_float32:2` | ok | 1.592 | 0.979 |
| results/generated_sft.csv:42 | rmsnorm_M4096_N2560_float32 | `gen:sft:rmsnorm_M4096_N2560_float32:0` | ok | 3.434 | 0.997 |
| results/generated_sft.csv:48 | rmsnorm_M1024_N4096_float32 | `gen:sft:rmsnorm_M1024_N4096_float32:2` | ok | 3.228 | 1.010 |
| results/generated_sft.csv:50 | rmsnorm_M2048_N5120_float32 | `gen:sft:rmsnorm_M2048_N5120_float32:0` | ok | 3.323 | 0.986 |
| results/generated_sft.csv:56 | rmsnorm_M4096_N6144_float32 | `gen:sft:rmsnorm_M4096_N6144_float32:2` | ok | 3.528 | 1.008 |
| results/generated_sft.csv:58 | rmsnorm_M1024_N8192_float32 | `gen:sft:rmsnorm_M1024_N8192_float32:0` | ok | 3.189 | 1.053 |
| results/generated_sft.csv:65 | rmsnorm_M8192_N3584_float32 | `gen:sft:rmsnorm_M8192_N3584_float32:3` | ok | 3.561 | 1.005 |
| results/generated_sft.csv:66 | softmax_M4096_N384_float32 | `gen:sft:softmax_M4096_N384_float32:0` | ok | 1.797 | 0.983 |
| results/generated_sft.csv:70 | softmax_M2048_N768_float32 | `gen:sft:softmax_M2048_N768_float32:0` | ok | 2.036 | 1.035 |
| results/generated_sft.csv:74 | softmax_M4096_N1536_float32 | `gen:sft:softmax_M4096_N1536_float32:0` | ok | 5.559 | 0.995 |
| results/generated_sft.csv:79 | softmax_M1024_N3072_float32 | `gen:sft:softmax_M1024_N3072_float32:1` | ok | 1.904 | 1.017 |
| results/generated_sft.csv:84 | softmax_M2048_N5120_float32 | `gen:sft:softmax_M2048_N5120_float32:2` | ok | 3.762 | 1.028 |
| results/generated_sft.csv:86 | softmax_M1024_N6144_float32 | `gen:sft:softmax_M1024_N6144_float32:0` | ok | 2.929 | 1.005 |
| results/generated_sft.csv:91 | softmax_M512_N8192_float32 | `gen:sft:softmax_M512_N8192_float32:1` | ok | 2.295 | 1.034 |
| results/generated_sft.csv:95 | softmax_M4096_N1280_float32 | `gen:sft:softmax_M4096_N1280_float32:1` | ok | 2.622 | 0.989 |
| results/generated_sft.csv:99 | gemm_M768_N768_K768_float16 | `gen:sft:gemm_M768_N768_K768_float16:1` | ok | 1.089 | - |
| results/generated_sft.csv:103 | gemm_M1536_N1536_K1536_float16 | `gen:sft:gemm_M1536_N1536_K1536_float16:1` | ok | 1.052 | - |
| results/generated_sft.csv:106 | gemm_M3072_N1024_K1024_float16 | `gen:sft:gemm_M3072_N1024_K1024_float16:0` | ok | 1.033 | - |
| results/generated_sft.csv:112 | gemm_M1024_N3072_K512_float16 | `gen:sft:gemm_M1024_N3072_K512_float16:2` | ok | 1.029 | - |
| results/generated_sft.csv:115 | gemm_M1000_N1000_K1000_float16 | `gen:sft:gemm_M1000_N1000_K1000_float16:1` | ok | 0.532 | - |
| results/generated_sft.csv:121 | gemm_M4096_N1024_K256_float16 | `gen:sft:gemm_M4096_N1024_K256_float16:3` | ok | 1.037 | - |
| results/generated_sft.csv:123 | gemm_M512_N512_K4096_float16 | `gen:sft:gemm_M512_N512_K4096_float16:1` | ok | 0.768 | - |
| results/generated_sft.csv:128 | gemm_M2048_N1536_K768_float16 | `gen:sft:gemm_M2048_N1536_K768_float16:2` | ok | 1.047 | - |
| results/generated_sft.csv:131 | gemm_bias_relu_M768_N768_K768_float16 | `gen:sft:gemm_bias_relu_M768_N768_K768...` | ok | 1.340 | 1.327 |
| results/generated_sft.csv:134 | gemm_bias_relu_M1536_N1536_K1536_float16 | `gen:sft:gemm_bias_relu_M1536_N1536_K1...` | ok | 1.274 | 1.189 |
| results/generated_sft.csv:140 | gemm_bias_relu_M3072_N1024_K1024_float16 | `gen:sft:gemm_bias_relu_M3072_N1024_K1...` | ok | 1.405 | 1.220 |
| results/generated_sft.csv:145 | gemm_bias_relu_M1024_N3072_K512_float16 | `gen:sft:gemm_bias_relu_M1024_N3072_K5...` | ok | 1.894 | 1.372 |
| results/generated_sft.csv:149 | gemm_bias_relu_M1000_N1000_K1000_float16 | `gen:sft:gemm_bias_relu_M1000_N1000_K1...` | ok | 0.740 | 0.686 |
| results/generated_sft.csv:150 | gemm_bias_relu_M4096_N1024_K256_float16 | `gen:sft:gemm_bias_relu_M4096_N1024_K2...` | ok | 2.146 | 1.683 |
| results/generated_sft.csv:156 | gemm_bias_relu_M512_N512_K4096_float16 | `gen:sft:gemm_bias_relu_M512_N512_K409...` | ok | 1.113 | 1.080 |
| results/generated_sft.csv:160 | gemm_bias_relu_M2048_N1536_K768_float16 | `gen:sft:gemm_bias_relu_M2048_N1536_K7...` | ok | 1.464 | 1.294 |
| results/generated_sft.csv:164 | attention_B2_H8_S192_D64_float16 | `gen:sft:attention_B2_H8_S192_D64_floa...` | ok | 2.500 | 1.750 |
| results/generated_sft.csv:169 | attention_B2_H8_S384_D64_float16 | `gen:sft:attention_B2_H8_S384_D64_floa...` | ok | 2.733 | 1.139 |
| results/generated_sft.csv:171 | attention_B4_H8_S768_D64_float16 | `gen:sft:attention_B4_H8_S768_D64_floa...` | ok | 6.174 | 0.942 |
| results/generated_sft.csv:176 | attention_B2_H8_S1536_D64_float16 | `gen:sft:attention_B2_H8_S1536_D64_flo...` | ok | 6.028 | 0.994 |
| results/generated_sft.csv:181 | attention_B2_H16_S512_D128_float16 | `gen:sft:attention_B2_H16_S512_D128_fl...` | ok | 1.741 | 0.757 |
| results/generated_sft.csv:183 | attention_B1_H8_S2048_D64_float16 | `gen:sft:attention_B1_H8_S2048_D64_flo...` | ok | 5.758 | 1.070 |
| results/generated_sft.csv:188 | attention_B4_H8_S256_D128_float16 | `gen:sft:attention_B4_H8_S256_D128_flo...` | ok | 1.457 | 0.788 |
| results/generated_sft.csv:192 | attention_B2_H8_S640_D32_float16 | `gen:sft:attention_B2_H8_S640_D32_floa...` | ok | 4.879 | 1.000 |
| results/generated_sft.csv:197 | attention_B2_H8_S1024_D128_float16 | `gen:sft:attention_B2_H8_S1024_D128_fl...` | ok | 2.156 | 0.773 |
| results/generated_sft.csv:198 | attention_B1_H16_S1024_D64_float16 | `gen:sft:attention_B1_H16_S1024_D64_fl...` | ok | 5.421 | 1.054 |

## Model `rl`: fastest correct sample per held-out task

Source: `results/generated_rl.csv`. The line is the row's line in that file.

| Line | Task | Candidate | Status | Speedup | Speedup vs library |
| --- | --- | --- | --- | --- | --- |
| results/generated_rl.csv:5 | layernorm_M4096_N640_float32 | `gen:rl:layernorm_M4096_N640_float32:3` | ok | 3.033 | 1.043 |
| results/generated_rl.csv:15 | layernorm_M1024_N4096_float32 | `gen:rl:layernorm_M1024_N4096_float32:1` | ok | 3.965 | 1.042 |
| results/generated_rl.csv:26 | layernorm_M1024_N8192_float32 | `gen:rl:layernorm_M1024_N8192_float32:0` | ok | 5.909 | 0.946 |
| results/generated_rl.csv:32 | layernorm_M8192_N3584_float32 | `gen:rl:layernorm_M8192_N3584_float32:2` | ok | 6.875 | 0.984 |
| results/generated_rl.csv:35 | rmsnorm_M4096_N640_float32 | `gen:rl:rmsnorm_M4096_N640_float32:1` | ok | 1.913 | 1.042 |
| results/generated_rl.csv:48 | rmsnorm_M1024_N4096_float32 | `gen:rl:rmsnorm_M1024_N4096_float32:2` | ok | 1.945 | 1.028 |
| results/generated_rl.csv:61 | rmsnorm_M1024_N8192_float32 | `gen:rl:rmsnorm_M1024_N8192_float32:3` | ok | 3.202 | 1.056 |
| results/generated_rl.csv:62 | rmsnorm_M8192_N3584_float32 | `gen:rl:rmsnorm_M8192_N3584_float32:0` | ok | 3.478 | 0.985 |
| results/generated_rl.csv:67 | softmax_M4096_N384_float32 | `gen:rl:softmax_M4096_N384_float32:1` | ok | 2.036 | 1.000 |
| results/generated_rl.csv:72 | softmax_M2048_N768_float32 | `gen:rl:softmax_M2048_N768_float32:2` | ok | 2.036 | 1.045 |
| results/generated_rl.csv:77 | softmax_M4096_N1536_float32 | `gen:rl:softmax_M4096_N1536_float32:3` | ok | 3.177 | 1.009 |
| results/generated_rl.csv:81 | softmax_M1024_N3072_float32 | `gen:rl:softmax_M1024_N3072_float32:3` | ok | 2.009 | 1.074 |
| results/generated_rl.csv:92 | softmax_M512_N8192_float32 | `gen:rl:softmax_M512_N8192_float32:2` | ok | 2.310 | 1.035 |
| results/generated_rl.csv:99 | gemm_M768_N768_K768_float16 | `gen:rl:gemm_M768_N768_K768_float16:1` | ok | 0.945 | - |
| results/generated_rl.csv:103 | gemm_M1536_N1536_K1536_float16 | `gen:rl:gemm_M1536_N1536_K1536_float16:1` | ok | 1.054 | - |
| results/generated_rl.csv:107 | gemm_M3072_N1024_K1024_float16 | `gen:rl:gemm_M3072_N1024_K1024_float16:1` | ok | 1.009 | - |
| results/generated_rl.csv:111 | gemm_M1024_N3072_K512_float16 | `gen:rl:gemm_M1024_N3072_K512_float16:1` | ok | 1.027 | - |
| results/generated_rl.csv:117 | gemm_M1000_N1000_K1000_float16 | `gen:rl:gemm_M1000_N1000_K1000_float16:3` | ok | 0.622 | - |
| results/generated_rl.csv:119 | gemm_M4096_N1024_K256_float16 | `gen:rl:gemm_M4096_N1024_K256_float16:1` | ok | 1.072 | - |
| results/generated_rl.csv:122 | gemm_M512_N512_K4096_float16 | `gen:rl:gemm_M512_N512_K4096_float16:0` | ok | 0.941 | - |
| results/generated_rl.csv:126 | gemm_M2048_N1536_K768_float16 | `gen:rl:gemm_M2048_N1536_K768_float16:0` | ok | 1.041 | - |
| results/generated_rl.csv:130 | gemm_bias_relu_M768_N768_K768_float16 | `gen:rl:gemm_bias_relu_M768_N768_K768_...` | ok | 1.381 | 1.239 |
| results/generated_rl.csv:137 | gemm_bias_relu_M1536_N1536_K1536_float16 | `gen:rl:gemm_bias_relu_M1536_N1536_K15...` | ok | 1.247 | 1.159 |
| results/generated_rl.csv:140 | gemm_bias_relu_M3072_N1024_K1024_float16 | `gen:rl:gemm_bias_relu_M3072_N1024_K10...` | ok | 1.304 | 1.203 |
| results/generated_rl.csv:144 | gemm_bias_relu_M1024_N3072_K512_float16 | `gen:rl:gemm_bias_relu_M1024_N3072_K51...` | ok | 1.567 | 1.322 |
| results/generated_rl.csv:148 | gemm_bias_relu_M1000_N1000_K1000_float16 | `gen:rl:gemm_bias_relu_M1000_N1000_K10...` | ok | 0.778 | 0.694 |
| results/generated_rl.csv:151 | gemm_bias_relu_M4096_N1024_K256_float16 | `gen:rl:gemm_bias_relu_M4096_N1024_K25...` | ok | 2.171 | 1.695 |
| results/generated_rl.csv:154 | gemm_bias_relu_M512_N512_K4096_float16 | `gen:rl:gemm_bias_relu_M512_N512_K4096...` | ok | 1.042 | 0.972 |
| results/generated_rl.csv:159 | gemm_bias_relu_M2048_N1536_K768_float16 | `gen:rl:gemm_bias_relu_M2048_N1536_K76...` | ok | 1.471 | 1.272 |
| results/generated_rl.csv:164 | attention_B2_H8_S192_D64_float16 | `gen:rl:attention_B2_H8_S192_D64_float...` | ok | 2.706 | 1.688 |
| results/generated_rl.csv:168 | attention_B2_H8_S384_D64_float16 | `gen:rl:attention_B2_H8_S384_D64_float...` | ok | 3.216 | 1.206 |
| results/generated_rl.csv:171 | attention_B4_H8_S768_D64_float16 | `gen:rl:attention_B4_H8_S768_D64_float...` | ok | 5.573 | 0.968 |
| results/generated_rl.csv:174 | attention_B2_H8_S1536_D64_float16 | `gen:rl:attention_B2_H8_S1536_D64_floa...` | ok | 5.634 | 0.955 |
| results/generated_rl.csv:181 | attention_B2_H16_S512_D128_float16 | `gen:rl:attention_B2_H16_S512_D128_flo...` | ok | 1.932 | 0.780 |
| results/generated_rl.csv:185 | attention_B1_H8_S2048_D64_float16 | `gen:rl:attention_B1_H8_S2048_D64_floa...` | ok | 5.460 | 1.043 |
| results/generated_rl.csv:188 | attention_B4_H8_S256_D128_float16 | `gen:rl:attention_B4_H8_S256_D128_floa...` | ok | 1.544 | 0.808 |
| results/generated_rl.csv:193 | attention_B2_H8_S640_D32_float16 | `gen:rl:attention_B2_H8_S640_D32_float...` | ok | 6.328 | 1.071 |
| results/generated_rl.csv:196 | attention_B2_H8_S1024_D128_float16 | `gen:rl:attention_B2_H8_S1024_D128_flo...` | ok | 2.522 | 0.794 |
| results/generated_rl.csv:200 | attention_B1_H16_S1024_D64_float16 | `gen:rl:attention_B1_H16_S1024_D64_flo...` | ok | 6.006 | 1.006 |
