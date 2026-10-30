import pytest
import torch


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU; skipped otherwise")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available: GPU test was not run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
