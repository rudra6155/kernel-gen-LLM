"""Facts about why a correct kernel is slower than the baseline, from two profile rows.

This states what the profile shows (kernel time, launches, bandwidth, launch
configuration). It does not name a cause: a profile of this kind can show where
the time goes, not why. Reading the Chrome trace, or running Nsight Compute, is
the next step.
"""

from __future__ import annotations

import json
from typing import Any


def _num(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def describe_slowdown(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Observations from the candidate's and the baseline's profile rows (kernelgen.rows.PROFILE_FIELDS)."""
    for name, row in (("candidate", candidate), ("baseline", baseline)):
        if row.get("status") not in ("ok", "suspect"):
            return [f"no usable {name} profile (status {row.get('status')}): {row.get('error') or 'no message'}"]

    notes: list[str] = []
    if candidate.get("status") == "suspect" or baseline.get("status") == "suspect":
        notes.append("a profile row is marked suspect: its kernel time exceeded the CUDA-event time, so treat the figures below with care")

    c_ms, b_ms = _num(candidate, "kernel_ms_per_iter"), _num(baseline, "kernel_ms_per_iter")
    if c_ms is None or b_ms is None:
        notes.append("the profile has no GPU kernel times (a CPU run, or the trace had no kernel events)")
        return notes

    notes.append(f"GPU kernel time per call: candidate {c_ms:.4f} ms, baseline {b_ms:.4f} ms ({c_ms / b_ms:.2f}x the baseline's)")

    c_n, b_n = _num(candidate, "kernel_launches_per_iter"), _num(baseline, "kernel_launches_per_iter")
    if c_n is not None and b_n is not None:
        notes.append(f"kernel launches per call: candidate {c_n:g}, baseline {b_n:g}")

    c_bw, b_bw = _num(candidate, "effective_gbps"), _num(baseline, "effective_gbps")
    c_pct, b_pct = _num(candidate, "pct_of_copy_bw"), _num(baseline, "pct_of_copy_bw")
    if c_bw is not None and b_bw is not None:
        text = f"effective bandwidth: candidate {c_bw:.1f} GB/s, baseline {b_bw:.1f} GB/s"
        if c_pct is not None and b_pct is not None:
            text += f" ({c_pct:.0f}% and {b_pct:.0f}% of the {candidate.get('copy_gbps') or 0:.0f} GB/s measured for a plain copy)"
        notes.append(text)
        if c_pct is not None and c_pct < 50:
            notes.append(f"the candidate reaches {c_pct:.0f}% of the measured copy bandwidth, so it is not limited by memory bandwidth")

    c_event = _num(candidate, "event_ms")
    if c_event is not None and c_ms > 0 and c_event > 1.3 * c_ms:
        notes.append(
            f"wall time per call is {c_event:.4f} ms but its kernels take {c_ms:.4f} ms: the remainder is launch overhead or gaps between kernels"
        )

    try:
        top = json.loads(candidate.get("top_kernels") or "[]")
    except ValueError:
        top = []
    if top:
        k = top[0]
        details = ", ".join(f"{key} {k[key]}" for key in ("grid", "block", "registers per thread", "shared memory", "est. achieved occupancy %") if key in k)
        notes.append(f"longest candidate kernel: {k['name'][:80]} averaging {k['avg_us']:.1f} us" + (f" ({details})" if details else ""))
    return notes
