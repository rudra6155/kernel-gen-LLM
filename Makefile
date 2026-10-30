PYTHON ?= python3
RESULTS := results

.PHONY: install test benchmark

# Run inside an activated virtualenv, e.g.:
#   python3 -m venv .venv && . .venv/bin/activate
install:
	$(PYTHON) -m pip install -r requirements.txt

# -ra prints skip reasons so skipped GPU tests are visible.
test:
	$(PYTHON) -m pytest tests -v -ra

benchmark:
	$(PYTHON) -m kernelgen.benchmark --out $(RESULTS)/kernel_benchmarks.csv
