import os

import pytest
import torch

# Without a GPU, run Triton kernels in its CPU interpreter. This has to be set
# before triton is first imported. It checks kernel logic, not GPU compilation
# or speed; with a GPU the same tests compile and run the kernels for real.
if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


@pytest.fixture(autouse=True)
def _cuda_context_stays_healthy():
    """Fail the test that leaves a CUDA error pending, not whichever test happens to run next.

    A device-side assert is reported asynchronously, at the next CUDA call that synchronizes, and it
    leaves the context unusable. Without this, the failure appears in an unrelated later test.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU; skipped otherwise")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available: GPU test was not run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
