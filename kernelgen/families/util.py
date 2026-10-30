"""Helpers shared by the family modules."""

from __future__ import annotations

import re

# Estimated shared-memory budget for a tile config. A pruning heuristic only:
# configs just over it may still compile, and configs under it may still fail.
# Failures are not hidden; they become annotated rows with their error.
SMEM_BUDGET_BYTES = 96 * 1024


def next_pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def fill(template: str, **values: object) -> str:
    """Replace __NAME__ placeholders. Plain replacement, so code braces need no escaping."""
    out = template
    for key, value in values.items():
        out = out.replace(f"__{key}__", str(value))
    left = re.findall(r"__[A-Z_]+__", out)
    if left:
        raise ValueError(f"unfilled placeholders: {sorted(set(left))}")
    return out


def default_warps(block: int) -> int:
    """One warp per 256 elements of a row, between 1 and 8."""
    return min(max(block // 256, 1), 8)
