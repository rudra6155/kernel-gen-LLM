"""The claims ledger: every claim, its evidence in a CSV, and a status computed from it.

A claim is a definition below, not a sentence someone typed into CLAIMS.md. Its
status comes from the CSV rows it points to:
  VERIFIED   the evidence exists, is complete, and the claim holds
  REFUTED    the evidence exists and the claim does not hold (these are reported)
  PENDING    the CSV is missing or does not yet cover every task the claim is about

CLAIMS.md is generated from this module (`make claims`) and a test fails if it is
stale. Every measured number in it carries the CSV file and line it came from.

Usage:
    python -m kernelgen.claims --write
    python -m kernelgen.claims --check [--require-verified]
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from kernelgen.annotate import load_annotations
from kernelgen.dataset import MAX_RECORDS, MIN_RECORDS, read_jsonl
from kernelgen.families import FAMILY_ORDER, get_family

VERIFIED, REFUTED, PENDING = "VERIFIED", "REFUTED", "PENDING"
NUMERIC = {"speedup", "speedup_vs_library", "baseline_ms", "candidate_ms", "library_ms", "reward", "max_abs_err"}
BENCHMARK_CSV = "results/kernel_benchmarks.csv"
OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
}


@dataclass(frozen=True)
class Evidence:
    """A number computed from the rows of one CSV: `agg` of `metric` over the rows matching `where`."""

    csv: str
    agg: str  # geomean | ok_fraction | best_of_k_geomean | pass_at_k | tasks_beating
    metric: str = "speedup"
    where: dict[str, str] = field(default_factory=dict)
    tasks: int = 0  # how many tasks the evidence must cover
    require_all_ok: bool = False  # every task must have a valid measurement
    threshold: float = 1.0  # for tasks_beating: a task beats it when its best metric is above this


@dataclass(frozen=True)
class Claim:
    id: str
    statement: str
    evidence: Evidence | None  # None for the dataset claims, which are computed from the data files
    op: str = ">"
    threshold: float | None = None  # compare the evidence to this number...
    versus: Evidence | None = None  # ...or to this other evidence
    kind: str = "csv"  # csv | dataset_size | annotation_coverage


# --- definitions ----------------------------------------------------------------------------------------------


def _heldout_count(family: str) -> int:
    return len(get_family(family).heldout_shapes)


def definitions() -> list[Claim]:
    claims = [
        Claim(
            "D1",
            f"The dataset has between {MIN_RECORDS} and {MAX_RECORDS} kernel records, built from templates, none on a held-out shape.",
            None,
            kind="dataset_size",
        ),
        Claim(
            "D2",
            "Every dataset record has a current annotation: its correctness and speedup measured on the GPU.",
            None,
            kind="annotation_coverage",
        ),
        Claim(
            "B1",
            "The benchmark measured the template kernel on all 50 held-out tasks, and every measurement is valid.",
            Evidence(BENCHMARK_CSV, "ok_fraction", tasks=50, require_all_ok=True),
            "==",
            1.0,
        ),
    ]
    for i, fam in enumerate(FAMILY_ORDER, start=2):
        claims.append(
            Claim(
                f"B{i}",
                f"The {fam} template kernels beat the naive PyTorch baseline: geometric-mean speedup over the held-out {fam} tasks is above 1.",
                Evidence(BENCHMARK_CSV, "geomean", "speedup", {"family": fam}, _heldout_count(fam), True),
                ">",
                1.0,
            )
        )
    n = len(FAMILY_ORDER) + 2
    for fam in FAMILY_ORDER:
        if get_family(fam).task(get_family(fam).train_shapes[0]).library is None:
            continue
        claims.append(
            Claim(
                f"B{n}",
                f"The {fam} template kernels beat PyTorch's own optimized op, not just the naive baseline: geometric-mean speedup_vs_library over the held-out {fam} tasks is above 1.",
                Evidence(BENCHMARK_CSV, "geomean", "speedup_vs_library", {"family": fam}, _heldout_count(fam), True),
                ">",
                1.0,
            )
        )
        n += 1

    def gen(tag: str, agg: str, metric: str = "speedup", **kw) -> Evidence:
        return Evidence(f"results/generated_{tag}.csv", agg, metric, tasks=50, **kw)

    claims += [
        Claim(
            "G1",
            "After fine-tuning, the model writes a correct kernel for more held-out tasks than the untuned base model does (pass@k).",
            gen("sft", "pass_at_k"),
            ">",
            versus=gen("base", "pass_at_k"),
        ),
        Claim(
            "G2",
            "After reinforcement learning, the model's best-of-k kernels are faster than the fine-tuned model's (geometric mean over held-out tasks with a correct sample).",
            gen("rl", "best_of_k_geomean"),
            ">",
            versus=gen("sft", "best_of_k_geomean"),
        ),
        Claim(
            "G3",
            "The reinforcement-learned model's best-of-k kernels beat the naive PyTorch baseline: geometric-mean speedup is above 1.",
            gen("rl", "best_of_k_geomean"),
            ">",
            1.0,
        ),
        Claim(
            "G4",
            "Reinforcement learning did not cost correctness: pass@k is at least the fine-tuned model's.",
            gen("rl", "pass_at_k"),
            ">=",
            versus=gen("sft", "pass_at_k"),
        ),
        Claim(
            "G5",
            "The reinforcement-learned model writes a kernel that beats PyTorch's own optimized op on at least one held-out task.",
            gen("rl", "tasks_beating", "speedup_vs_library", threshold=1.0),
            ">=",
            1.0,
        ),
    ]
    return claims


# --- reading evidence -------------------------------------------------------------------------------------------


def load_csv(path: Path) -> list[dict[str, Any]] | None:
    """Rows of a benchmark CSV with numbers parsed and empty cells as None; each row has its file line in `_line`."""
    if not path.exists():
        return None
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for i, raw in enumerate(csv.DictReader(f), start=2):
            row: dict[str, Any] = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in NUMERIC:
                    row[key] = float(value)
                else:
                    row[key] = value
            row["_line"] = i
            rows.append(row)
    return rows


def _geomean(values: Sequence[float]) -> float | None:
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def _best_per_task(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        v = r.get(metric)
        if r["status"] == "ok" and v is not None and v > 0 and (r["task"] not in best or v > best[r["task"]][metric]):
            best[r["task"]] = r
    return best


@dataclass
class Measured:
    status: str  # PENDING or "ok" (the value could be computed)
    value: float | None = None
    detail: str = ""
    rows: list[dict[str, Any]] = field(default_factory=list)  # the rows the value came from


def measure(ev: Evidence, root: Path) -> Measured:
    rows = load_csv(root / ev.csv)
    if rows is None:
        return Measured(PENDING, detail=f"{ev.csv} does not exist yet")
    scope = [r for r in rows if all(str(r.get(k)) == v for k, v in ev.where.items())]
    tasks = {r["task"] for r in scope}
    if len(tasks) < ev.tasks:
        return Measured(PENDING, detail=f"{ev.csv} covers {len(tasks)} of the {ev.tasks} tasks this claim is about")

    if ev.agg == "ok_fraction":
        ok = [r for r in scope if r["status"] == "ok"]
        return Measured("ok", len(ok) / len(scope), f"{len(ok)} of {len(scope)} rows are valid", scope)

    best = _best_per_task(scope, ev.metric)
    if ev.require_all_ok and len(best) < len(tasks):
        missing = sorted(tasks - set(best))
        return Measured("ok", None, f"{len(missing)} of {len(tasks)} tasks have no valid {ev.metric} (e.g. {missing[0]})", scope)

    picked = list(best.values())
    if ev.agg in ("geomean", "best_of_k_geomean"):
        return Measured("ok", _geomean([r[ev.metric] for r in picked]), f"over {len(picked)} tasks", picked)
    if ev.agg == "pass_at_k":
        return Measured("ok", len(picked) / ev.tasks, f"{len(picked)} of {ev.tasks} tasks have a correct sample", picked)
    if ev.agg == "tasks_beating":
        beating = [r for r in picked if r[ev.metric] > ev.threshold]
        return Measured("ok", float(len(beating)), f"of {len(picked)} tasks with a measurement", beating)
    raise ValueError(f"unknown aggregation: {ev.agg!r}")


@dataclass
class Result:
    claim: Claim
    status: str
    value: float | None
    other: float | None
    detail: str
    rows: list[dict[str, Any]]


def evaluate_claim(claim: Claim, root: Path) -> Result:
    if claim.kind == "dataset_size":
        return _dataset_size(claim, root)
    if claim.kind == "annotation_coverage":
        return _annotation_coverage(claim, root)

    m = measure(claim.evidence, root)
    if m.status == PENDING:
        return Result(claim, PENDING, None, None, m.detail, [])
    other = None
    if claim.versus is not None:
        o = measure(claim.versus, root)
        if o.status == PENDING:
            return Result(claim, PENDING, m.value, None, o.detail, m.rows)
        other, rows = o.value, m.rows + o.rows
        bound = other
    else:
        rows, bound = m.rows, claim.threshold

    if m.value is None or (claim.versus is not None and other is None):
        return Result(claim, REFUTED, m.value, other, "not shown: " + (m.detail if m.value is None else "no valid measurement to compare with"), rows)
    status = VERIFIED if OPS[claim.op](m.value, bound) else REFUTED
    return Result(claim, status, m.value, other, m.detail, rows)


def _dataset_size(claim: Claim, root: Path) -> Result:
    path = root / "data" / "kernels.jsonl"
    if not path.exists():
        return Result(claim, PENDING, None, None, "data/kernels.jsonl does not exist yet", [])
    from kernelgen.verify_dataset import check_heldout, check_records

    records = read_jsonl(path)
    problems = check_records(records, MIN_RECORDS, MAX_RECORDS) + check_heldout(root / "data" / "heldout_tasks.json")
    distinct = len({r["source_sha256"] for r in records})
    detail = f"{len(records)} records, {distinct} distinct kernel sources (data/kernels.jsonl)"
    if problems:
        return Result(claim, REFUTED, float(len(records)), None, f"{detail}; {len(problems)} problem(s), first: {problems[0]}", [])
    return Result(claim, VERIFIED, float(len(records)), None, detail, [])


def _annotation_coverage(claim: Claim, root: Path) -> Result:
    kernels, annotations = root / "data" / "kernels.jsonl", root / "data" / "annotations.jsonl"
    ann = load_annotations(annotations)
    if not ann:
        return Result(claim, PENDING, None, None, "data/annotations.jsonl does not exist yet (run `make annotate` on the GPU)", [])
    from kernelgen.verify_dataset import check_annotations

    records = read_jsonl(kernels)
    current = {r["id"]: r["source_sha256"] for r in records}
    fresh = [a for a in ann.values() if current.get(a["id"]) == a["source_sha256"]]
    problems = check_annotations(records, ann)
    ok = sum(a["status"] == "ok" for a in fresh)
    detail = f"{len(fresh)} of {len(records)} records annotated, {ok} correct (data/annotations.jsonl)"
    if problems:
        return Result(claim, REFUTED, len(fresh) / len(records), None, f"{detail}; {len(problems)} problem(s), first: {problems[0]}", [])
    if len(fresh) < len(records):
        return Result(claim, PENDING, len(fresh) / len(records), None, detail, [])
    return Result(claim, VERIFIED, 1.0, None, detail, [])


# --- CLAIMS.md ----------------------------------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _pointer(result: Result) -> str:
    claim = result.claim
    if claim.kind == "dataset_size":
        return "data/kernels.jsonl, data/heldout_tasks.json"
    if claim.kind == "annotation_coverage":
        return "data/annotations.jsonl"
    parts = []
    for ev in filter(None, (claim.evidence, claim.versus)):
        where = ", ".join(f"{k}={v}" for k, v in ev.where.items())
        parts.append(f"`{ev.csv}`" + (f" ({where})" if where else ""))
    return "<br>".join(parts)


def _measured_text(result: Result) -> str:
    claim = result.claim
    if result.status == PENDING:
        return f"not measured: {result.detail}"
    if claim.kind in ("dataset_size", "annotation_coverage"):
        return result.detail
    if claim.versus is not None:
        text = f"{_fmt(result.value)} {claim.op} {_fmt(result.other)}"
    else:
        text = f"{_fmt(result.value)} {claim.op} {_fmt(claim.threshold)}"
    return f"{text} ({result.detail})"


def render(root: Path) -> str:
    results = [evaluate_claim(c, root) for c in definitions()]
    counts = {s: sum(r.status == s for r in results) for s in (VERIFIED, REFUTED, PENDING)}
    out = [
        "# CLAIMS",
        "",
        "Every claim this project makes, with the evidence it rests on. This file is produced by",
        "`make claims` from the claim definitions in `kernelgen/claims.py` and the CSV files they",
        "point to; a test fails if it is out of date. A status is computed from the evidence, not",
        "written by hand:",
        "",
        "- **VERIFIED**: the evidence exists, covers every task the claim is about, and the claim holds.",
        "- **REFUTED**: the evidence exists and the claim does not hold. A refuted claim stays here.",
        "- **PENDING**: the CSV does not exist yet, or does not cover every task. Nothing is claimed.",
        "",
        "Speedups are over the naive eager PyTorch baseline of each task unless the claim says",
        "`speedup_vs_library`, which is over PyTorch's own optimized op. Aggregates are geometric means.",
        "Each CSV records the GPU, driver stack, and library versions of its measurements in its",
        "`gpu_name`, `torch_version`, `cuda_version`, and `triton_version` columns.",
        "",
        f"**{counts[VERIFIED]} VERIFIED, {counts[REFUTED]} REFUTED, {counts[PENDING]} PENDING** of {len(results)} claims.",
        "",
        "| ID | Claim | Status | Measured | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        out.append(f"| {r.claim.id} | {r.claim.statement} | **{r.status}** | {_measured_text(r)} | {_pointer(r)} |")
    out.append("")
    out += _appendix(root)
    return "\n".join(out).rstrip() + "\n"


def _appendix(root: Path) -> list[str]:
    """The measured per-task numbers, each with the CSV line it came from."""
    out: list[str] = []
    for rel, title, best_only in [(BENCHMARK_CSV, "Template kernels on the held-out tasks", False)] + [
        (f"results/generated_{t}.csv", f"Model `{t}`: fastest correct sample per held-out task", True) for t in ("base", "sft", "rl")
    ]:
        rows = load_csv(root / rel)
        if not rows:
            continue
        if best_only:
            rows = sorted(_best_per_task(rows, "speedup").values(), key=lambda r: r["_line"])
        out += [f"## {title}", "", f"Source: `{rel}`. The line is the row's line in that file.", "",
                "| Line | Task | Candidate | Status | Speedup | Speedup vs library |", "| --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            cand = r["candidate_id"] if len(r["candidate_id"]) <= 40 else r["candidate_id"][:37] + "..."
            out.append(f"| {rel}:{r['_line']} | {r['task']} | `{cand}` | {r['status']} | {_fmt(r.get('speedup'))} | {_fmt(r.get('speedup_vs_library'))} |")
        out.append("")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--write", action="store_true", help="regenerate CLAIMS.md")
    parser.add_argument("--check", action="store_true", help="fail if CLAIMS.md is stale")
    parser.add_argument("--require-verified", action="store_true", help="with --check, also fail unless every claim is VERIFIED")
    args = parser.parse_args(argv)

    root = Path(args.root)
    text = render(root)
    path = root / "CLAIMS.md"
    if args.write:
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    results = [evaluate_claim(c, root) for c in definitions()]
    counts = {s: sum(r.status == s for r in results) for s in (VERIFIED, REFUTED, PENDING)}
    print(f"{counts[VERIFIED]} VERIFIED, {counts[REFUTED]} REFUTED, {counts[PENDING]} PENDING of {len(results)} claims")
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            print("CLAIMS.md is out of date: run `make claims`", file=sys.stderr)
            return 1
        if args.require_verified and counts[VERIFIED] != len(results):
            print("not every claim is VERIFIED", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
