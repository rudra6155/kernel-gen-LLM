PYTHON ?= python3
RESULTS := results

.PHONY: install test benchmark profile dataset annotate verify-dataset

# Run inside an activated virtualenv, e.g.:
#   python3 -m venv .venv && . .venv/bin/activate
install:
	$(PYTHON) -m pip install -r requirements.txt

# -ra prints skip reasons so skipped GPU tests are visible.
test:
	$(PYTHON) -m pytest tests -v -ra

benchmark:
	$(PYTHON) -m kernelgen.benchmark --out $(RESULTS)/kernel_benchmarks.csv

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

# Kernel time and effective memory bandwidth via torch.profiler. Pass the
# card's spec bandwidth if known: make profile PEAK_GBPS=<GB/s>
profile:
	$(PYTHON) -m kernelgen.profiling --out $(RESULTS)/profile_summary.csv --trace-dir $(RESULTS)/traces $(if $(PEAK_GBPS),--peak-gbps $(PEAK_GBPS))
