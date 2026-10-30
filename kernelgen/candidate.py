"""Load candidate kernel source as a callable."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

ENTRY_POINT = "solution"


class CandidateError(Exception):
    pass


@contextmanager
def load_candidate(source: str, entry: str = ENTRY_POINT) -> Iterator[Callable]:
    """Import `source` as a module and yield its `entry` function.

    The source is written to a real file because @triton.jit reads the source
    of the decorated function with inspect, which fails for exec'd strings.
    The file exists only inside the with-block, so JIT compilation that happens
    on the first call must also happen inside it.
    """
    digest = hashlib.sha256(source.encode()).hexdigest()[:12]
    mod_name = f"kernelgen_candidate_{digest}"
    with tempfile.TemporaryDirectory(prefix="kernelgen_") as tmp:
        path = Path(tmp) / f"{mod_name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
            fn = getattr(module, entry, None)
            if not callable(fn):
                raise CandidateError(f"candidate does not define a callable '{entry}'")
            yield fn
        finally:
            sys.modules.pop(mod_name, None)
