PYTHON ?= python3
RESULTS := results

.PHONY: install test benchmark profile dataset annotate verify-dataset sft rl eval-model nsight collect-logs claims claims-check status

# Run inside an activated virtualenv, e.g.:
#   python3 -m venv .venv && . .venv/bin/activate
install:
	$(PYTHON) -m pip install -r requirements.txt

# -ra prints skip reasons so skipped GPU tests are visible.
test:
	$(PYTHON) -m pytest tests -v -ra

# The template kernel for each of the 50 held-out tasks, measured on the GPU. Writes
# results/kernel_benchmarks.csv. Quick trial: make benchmark BENCH_ARGS="--limit 3"
benchmark:
	$(PYTHON) -m kernelgen.benchmark --out $(RESULTS)/kernel_benchmarks.csv $(BENCH_ARGS)

# Rebuild data/kernels.jsonl and data/heldout_tasks.json from the family templates (no GPU needed).
dataset:
	$(PYTHON) -m kernelgen.dataset --out data/kernels.jsonl --heldout-out data/heldout_tasks.json

# Measure every dataset kernel on the GPU; resumable. Writes data/annotations.jsonl.
annotate:
	$(PYTHON) -m kernelgen.annotate --kernels data/kernels.jsonl --out data/annotations.jsonl

# Structural checks always; annotation checks when data/annotations.jsonl exists.
# Add REMEASURE=<n> to re-measure n annotated kernels on the GPU.
verify-dataset:
	$(PYTHON) -m kernelgen.verify_dataset $(if $(REMEASURE),--remeasure $(REMEASURE))

# QLoRA fine-tuning on the annotated kernels (needs data/annotations.jsonl and the GPU).
# Logs loss and generated samples per epoch under runs/sft/. Extra flags: make sft SFT_ARGS="--epochs 2"
sft:
	$(PYTHON) -m kernelgen.finetune --out runs/sft $(SFT_ARGS)

# GRPO with reward = speedup * correctness, starting from the SFT adapter. Logs the reward
# curve and every generated kernel under runs/rl/; writes divergence_report.json if it diverges.
# Extra flags: make rl RL_ARGS="--steps 50 --group-size 4"
rl:
	$(PYTHON) -m kernelgen.rl --init-adapter runs/sft/final --out runs/rl $(RL_ARGS)

# Kernels written by a model, measured on the 50 held-out tasks. Writes results/generated_<TAG>.csv,
# the replies, and a summary. The untuned base model: make eval-model TAG=base
#   make eval-model TAG=sft ADAPTER=runs/sft/final
#   make eval-model TAG=rl ADAPTER=runs/rl/final
TAG ?= model
eval-model:
	$(PYTHON) -m kernelgen.evaluate_model --tag $(TAG) $(if $(ADAPTER),--adapter $(ADAPTER)) $(EVAL_ARGS)

# Optional Nsight Compute / Nsight Systems profiles. Never fails if the tools are missing or
# unusable (common under WSL2); results/nsight/STATUS.json records what happened.
nsight:
	$(PYTHON) -m kernelgen.nsight --out $(RESULTS)/nsight

# Kernel time and effective memory bandwidth via torch.profiler. Pass the
# card's spec bandwidth if known: make profile PEAK_GBPS=<GB/s>
# All 50 held-out tasks instead of one per family: make profile PROFILE_ARGS="--scope all"
profile:
	$(PYTHON) -m kernelgen.profiling --out $(RESULTS)/profile_summary.csv --trace-dir $(RESULTS)/traces $(if $(PEAK_GBPS),--peak-gbps $(PEAK_GBPS)) $(PROFILE_ARGS)

# Copy the small training logs from runs/ to results/training/ so they can be committed (adapters stay local).
collect-logs:
	$(PYTHON) -m kernelgen.collect_logs

# Regenerate CLAIMS.md from the claim definitions and the CSVs they point to.
claims:
	$(PYTHON) -m kernelgen.claims --write

# Fail if CLAIMS.md is stale. make claims-check STRICT=1 also fails unless every claim is VERIFIED.
claims-check:
	$(PYTHON) -m kernelgen.claims --check $(if $(STRICT),--require-verified)

# The Definition of Done checklist, computed from the repository. Add --run-tests via STATUS_ARGS.
status:
	$(PYTHON) -m kernelgen.status $(STATUS_ARGS)
