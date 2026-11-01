"""The Definition of Done checklist, computed from the state of the repository.

Each line is [x] when the evidence is in the repository, and [ ] with the reason when
it is not. Items that need the RTX 4060 are PENDING until their results are committed.
Nothing here is typed in by hand: the dataset is checked, the CSVs are read, CLAIMS.md
is recomputed, and the git history is inspected.

`make test` is only run with --run-tests, because it takes minutes.

Usage:
    python -m kernelgen.status [--run-tests]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

DONE, PENDING, FAILED, NOT_CHECKED = "DONE", "PENDING", "FAILED", "NOT CHECKED"


@dataclass
class Item:
    name: str
    state: str
    detail: str


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def _tracked(root: Path, rel: str) -> bool:
    return _git(root, "ls-files", "--error-unmatch", rel) is not None


def _committed(root: Path, rel: str, what: str, gpu_cmd: str) -> Item:
    """DONE if `rel` exists and is tracked by git, PENDING if it is missing, and a warning if it is uncommitted."""
    path = root / rel
    if not path.exists():
        return Item(what, PENDING, f"{rel} does not exist yet (needs the GPU: {gpu_cmd})")
    if not _tracked(root, rel):
        return Item(what, PENDING, f"{rel} exists but is not committed")
    return Item(what, DONE, rel)


def check_tests(root: Path, run: bool) -> Item:
    name = "make test is green"
    if not run:
        return Item(name, NOT_CHECKED, "run `make test` (or `python -m kernelgen.status --run-tests`)")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=root, capture_output=True, text=True)
    summary = (proc.stdout.strip().splitlines() or [""])[-1]
    return Item(name, DONE if proc.returncode == 0 else FAILED, summary)


def check_dataset(root: Path) -> list[Item]:
    from kernelgen.annotate import load_annotations
    from kernelgen.dataset import MAX_RECORDS, MIN_RECORDS, read_jsonl
    from kernelgen.verify_dataset import check_annotations, check_heldout, check_records

    kernels = root / "data" / "kernels.jsonl"
    if not kernels.exists():
        return [Item("dataset of 500-2000 template kernels", FAILED, "data/kernels.jsonl is missing: run `make dataset`")]
    records = read_jsonl(kernels)
    problems = check_records(records, MIN_RECORDS, MAX_RECORDS) + check_heldout(root / "data" / "heldout_tasks.json")
    distinct = len({r["source_sha256"] for r in records})
    items = [
        Item(
            "dataset of 500-2000 template kernels, no held-out shapes, all policy-clean",
            FAILED if problems else DONE,
            problems[0] if problems else f"{len(records)} records, {distinct} distinct sources, 50 held-out tasks",
        )
    ]

    ann = load_annotations(root / "data" / "annotations.jsonl")
    name = "dataset annotated with correctness and speedup, and the annotations verified"
    if not ann:
        items.append(Item(name, PENDING, "data/annotations.jsonl does not exist yet (needs the GPU: make annotate)"))
    else:
        current = {r["id"]: r["source_sha256"] for r in records}
        fresh = sum(current.get(a["id"]) == a["source_sha256"] for a in ann.values())
        bad = check_annotations(records, ann)
        if bad:
            items.append(Item(name, FAILED, bad[0]))
        elif fresh < len(records):
            items.append(Item(name, PENDING, f"{fresh} of {len(records)} records annotated"))
        else:
            tracked = _tracked(root, "data/annotations.jsonl")
            items.append(Item(name, DONE if tracked else PENDING, f"{fresh} annotated" + ("" if tracked else ", but not committed")))
    return items


def check_benchmark(root: Path) -> Item:
    name = "results/kernel_benchmarks.csv committed, one row per held-out task"
    item = _committed(root, "results/kernel_benchmarks.csv", name, "make benchmark")
    if item.state != DONE:
        return item
    rows = (root / "results" / "kernel_benchmarks.csv").read_text(encoding="utf-8").splitlines()[1:]
    return Item(name, DONE if len(rows) == 50 else FAILED, f"{len(rows)} rows" + ("" if len(rows) == 50 else ", expected 50"))


def check_profile(root: Path) -> list[Item]:
    name = "torch.profiler summary and Chrome traces committed"
    summary = _committed(root, "results/profile_summary.csv", name, "make profile")
    traces = sorted((root / "results" / "traces").glob("*.json")) if (root / "results" / "traces").exists() else []
    if summary.state == DONE and not traces:
        summary = Item(name, PENDING, "results/profile_summary.csv is committed but results/traces/ has no traces")
    elif summary.state == DONE:
        summary = Item(name, DONE, f"results/profile_summary.csv and {len(traces)} traces")

    status = root / "results" / "nsight" / "STATUS.json"
    if not status.exists():
        nsight = Item("Nsight profile (optional)", PENDING, "not run: make nsight (often unavailable under WSL2; torch.profiler is the primary profile)")
    else:
        import json

        data = json.loads(status.read_text(encoding="utf-8"))
        produced = [r for r in data.get("runs", []) if r.get("produced_profile")]
        nsight = Item(
            "Nsight profile (optional)",
            DONE if produced else PENDING,
            f"{len(produced)} profile(s) produced" if produced else "ran, but no tool produced a profile (see results/nsight/STATUS.json)",
        )
    return [summary, nsight]


def check_training(root: Path) -> list[Item]:
    sft = _committed(root, "results/training/sft/train_log.csv", "QLoRA fine-tuning logs (loss, samples per epoch) committed", "make sft, then make collect-logs")
    rl = _committed(root, "results/training/rl/reward_curve.csv", "GRPO reward curve and generated kernels committed", "make rl, then make collect-logs")
    if rl.state == DONE and (root / "results" / "training" / "rl" / "divergence_report.json").exists():
        rl = Item(rl.name, DONE, rl.detail + "; the run DIVERGED, see results/training/rl/divergence_report.json")
    gen = []
    for tag in ("base", "sft", "rl"):
        gen.append(_committed(root, f"results/generated_{tag}.csv", f"generated kernels of the {tag} model evaluated on the held-out tasks", f"make eval-model TAG={tag}"))
    return [sft, rl, *gen]


def check_claims(root: Path) -> Item:
    from kernelgen.claims import PENDING as C_PENDING, REFUTED as C_REFUTED, VERIFIED as C_VERIFIED, definitions, evaluate_claim, render

    name = "CLAIMS.md: every claim VERIFIED"
    results = [evaluate_claim(c, root) for c in definitions()]
    counts = {s: sum(r.status == s for r in results) for s in (C_VERIFIED, C_REFUTED, C_PENDING)}
    text = f"{counts[C_VERIFIED]} VERIFIED, {counts[C_REFUTED]} REFUTED, {counts[C_PENDING]} PENDING of {len(results)}"
    path = root / "CLAIMS.md"
    if not path.exists() or path.read_text(encoding="utf-8") != render(root):
        return Item(name, FAILED, f"CLAIMS.md is out of date (run `make claims`); {text}")
    if counts[C_REFUTED]:
        refuted = ", ".join(r.claim.id for r in results if r.status == C_REFUTED)
        return Item(name, FAILED, f"{text}; refuted: {refuted}. They stay in the ledger and are not hidden.")
    return Item(name, DONE if counts[C_PENDING] == 0 else PENDING, text)


def check_history(root: Path) -> Item:
    name = "git history: incremental commits by one author, no co-author trailers"
    log = _git(root, "log", "--format=%an <%ae>%x1f%B%x1e")
    if log is None:
        return Item(name, NOT_CHECKED, "not a git repository")
    commits = [c.strip() for c in log.split("\x1e") if c.strip()]
    authors = {c.split("\x1f")[0] for c in commits}
    attribution = re.compile(r"co-authored-by|generated with", re.IGNORECASE)
    flagged = [c.split("\x1f")[1].splitlines()[0] for c in commits if attribution.search(c.split("\x1f")[1])]
    if len(authors) != 1:
        return Item(name, FAILED, f"{len(authors)} different authors: {sorted(authors)}")
    if flagged:
        return Item(name, FAILED, f"{len(flagged)} commit message(s) mention attribution, e.g. {flagged[0]!r}")
    return Item(name, DONE if len(commits) >= 5 else FAILED, f"{len(commits)} commits by {next(iter(authors))}")


def collect(root: Path, run_tests: bool = False) -> list[Item]:
    return [
        check_tests(root, run_tests),
        *check_dataset(root),
        check_benchmark(root),
        *check_profile(root),
        *check_training(root),
        check_claims(root),
        check_history(root),
    ]


def render(items: Sequence[Item]) -> str:
    lines = []
    for it in items:
        mark = "x" if it.state == DONE else " "
        suffix = "" if it.state == DONE else f"  [{it.state}]"
        lines.append(f"[{mark}] {it.name}{suffix}")
        lines.append(f"      {it.detail}")
    done = sum(it.state == DONE for it in items)
    lines.append("")
    lines.append(f"{done} of {len(items)} items done; the rest are PENDING (they need the GPU run) or not checked.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args(argv)
    items = collect(Path(args.root), args.run_tests)
    print(render(items))
    return 1 if any(it.state == FAILED for it in items) else 0


if __name__ == "__main__":
    raise SystemExit(main())
