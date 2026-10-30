PYTHON ?= python3
RESULTS := results

.PHONY: install test benchmark profile

# Run inside an activated virtualenv, e.g.:
#   python3 -m venv .venv && . .venv/bin/activate
install:
	$(PYTHON) -m pip install -r requirements.txt

# -ra prints skip reasons so skipped GPU tests are visible.
test:
	$(PYTHON) -m pytest tests -v -ra

benchmark:
	$(PYTHON) -m kernelgen.benchmark --out $(RESULTS)/kernel_benchmarks.csv

# Kernel time and effective memory bandwidth via torch.profiler. Pass the
# card's spec bandwidth if known: make profile PEAK_GBPS=<GB/s>
profile:
	$(PYTHON) -m kernelgen.profiling --out $(RESULTS)/profile_summary.csv --trace-dir $(RESULTS)/traces $(if $(PEAK_GBPS),--peak-gbps $(PEAK_GBPS))
