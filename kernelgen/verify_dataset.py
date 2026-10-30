"""Verify the dataset and its annotations.

Structural checks need no GPU and always run:
  - record schema, unique ids, unique (task, config), record count in range
  - sources re-render exactly from (family, task, config) and their hash matches
  - every source passes the candidate policy
  - training shapes never overlap the held-out shapes; there are 50 held-out tasks
Annotation checks run when an annotations file exists:
  - annotations match a record and the record's current source (not stale)
  - each annotation is internally consistent (speedup == baseline_ms / candidate_ms,
    correct only when the status is ok, ...)
With --remeasure N, N annotated records are measured again on the GPU and compared
with their stored annotation.

Missing annotations are reported as PENDING and fail the run only with
--require-annotations.

Usage:
    python -m kernelgen.verify_dataset [--require-annotations] [--remeasure N]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

from kernelgen.annotate import ANNOTATION_FIELDS, load_annotations
from kernelgen.dataset import MAX_RECORDS, MIN_RECORDS, heldout_tasks, read_jsonl
from kernelgen.families import get_family
from kernelgen.policy import check_source

RECORD_FIELDS = {"id", "family", "task", "task_name", "config", "source", "source_sha256"}
STATUSES = {"ok", "incorrect", "error", "rejected", "timeout", "crash"}


def key(params: dict[str, Any]) -> tuple:
    return tuple(sorted(params.items()))


def check_records(records: list[dict[str, Any]], min_records: int, max_records: int) -> list[str]:
    from kernelgen.dataset import sha256

    problems: list[str] = []
    if not min_records <= len(records) <= max_records:
        problems.append(f"{len(records)} records, expected between {min_records} and {max_records}")

    seen_ids: set[str] = set()
    seen_pairs: set[str] = set()
    for rec in records:
        rid = rec.get("id", "<no id>")
        missing = RECORD_FIELDS - set(rec)
        if missing:
            problems.append(f"{rid}: missing fields {sorted(missing)}")
            continue
        if rid in seen_ids:
            problems.append(f"{rid}: duplicate id")
        seen_ids.add(rid)
        pair = json.dumps({"task": rec["task"], "config": rec["config"]}, sort_keys=True)
        if pair in seen_pairs:
            problems.append(f"{rid}: duplicate (task, config)")
        seen_pairs.add(pair)

        try:
            fam = get_family(rec["family"])
        except ValueError as exc:
            problems.append(f"{rid}: {exc}")
            continue
        params = rec["task"]["params"]
        if rec["task"]["family"] != rec["family"] or rec["task"]["dtype"] != fam.dtype:
            problems.append(f"{rid}: task spec does not match its family")
        if key(params) not in {key(p) for p in fam.train_shapes}:
            problems.append(f"{rid}: shape {params} is not a training shape")
        if key(params) in {key(p) for p in fam.heldout_shapes}:
            problems.append(f"{rid}: shape {params} is held out (leakage)")
        try:
            rendered = fam.render(params, rec["config"])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{rid}: cannot re-render source: {exc}")
            continue
        if rendered != rec["source"]:
            problems.append(f"{rid}: stored source differs from the current template (rebuild the dataset)")
        if sha256(rec["source"]) != rec["source_sha256"]:
            problems.append(f"{rid}: source_sha256 does not match source")
        violations = check_source(rec["source"])
        if violations:
            problems.append(f"{rid}: fails the candidate policy: {violations[0]}")
    return problems


def check_heldout(path: Path) -> list[str]:
    if not path.exists():
        return [f"{path} is missing"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if stored != heldout_tasks():
        problems.append(f"{path} differs from the held-out shapes in the family definitions (rebuild the dataset)")
    if len(stored) != 50:
        problems.append(f"{len(stored)} held-out tasks, expected 50")
    ids = [t["id"] for t in stored]
    if len(set(ids)) != len(ids):
        problems.append("duplicate held-out task ids")
    return problems


def check_annotation(ann: dict[str, Any], record: dict[str, Any] | None) -> list[str]:
    aid = ann.get("id", "<no id>")
    problems: list[str] = []
    if set(ann) != set(ANNOTATION_FIELDS):
        return [f"{aid}: fields are not the annotation schema"]
    if record is None:
        return [f"{aid}: no such record"]
    if ann["source_sha256"] != record["source_sha256"]:
        problems.append(f"{aid}: stale, annotated for a different source")
    if ann["status"] not in STATUSES:
        problems.append(f"{aid}: unknown status {ann['status']!r}")
        return problems

    if ann["status"] == "ok":
        if ann["correct"] != 1:
            problems.append(f"{aid}: status ok but correct={ann['correct']}")
        for f in ("baseline_ms", "candidate_ms", "speedup"):
            if not isinstance(ann[f], (int, float)) or not math.isfinite(ann[f]) or ann[f] <= 0:
                problems.append(f"{aid}: {f}={ann[f]!r} is not a positive finite number")
        if not problems and not math.isclose(ann["speedup"], ann["baseline_ms"] / ann["candidate_ms"], rel_tol=1e-6):
            problems.append(f"{aid}: speedup {ann['speedup']} != baseline_ms / candidate_ms")
        lib, lib_speedup = ann["library_ms"], ann["speedup_vs_library"]
        if (lib is None) != (lib_speedup is None):
            problems.append(f"{aid}: library_ms and speedup_vs_library must both be set or both empty")
    else:
        if ann["correct"] != 0:
            problems.append(f"{aid}: status {ann['status']} but correct={ann['correct']}")
        if ann["speedup"] is not None:
            problems.append(f"{aid}: status {ann['status']} but has a speedup")
        if not ann["error"]:
            problems.append(f"{aid}: status {ann['status']} without an error message")
    return problems


def check_annotations(records: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]) -> list[str]:
    by_id = {r["id"]: r for r in records}
    problems: list[str] = []
    for ann in annotations.values():
        problems += check_annotation(ann, by_id.get(ann.get("id")))
    return problems


def remeasure(
    records: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    n: int,
    tol: float,
    seed: int,
    evaluate_fn,
) -> list[str]:
    """Measure n annotated records again and compare with their stored annotation."""
    by_id = {r["id"]: r for r in records}
    pool = sorted(a["id"] for a in annotations.values() if a["id"] in by_id)
    problems: list[str] = []
    for rid in random.Random(seed).sample(pool, min(n, len(pool))):
        stored, row = annotations[rid], evaluate_fn(by_id[rid])
        if int(row["correct"]) != int(stored["correct"]):
            problems.append(f"{rid}: correct was {stored['correct']}, now {row['correct']}")
        elif stored["status"] == "ok" and row["status"] == "ok":
            rel = abs(row["speedup"] - stored["speedup"]) / stored["speedup"]
            if rel > tol:
                problems.append(f"{rid}: speedup was {stored['speedup']:.3f}, now {row['speedup']:.3f} ({rel:.0%} apart, tolerance {tol:.0%})")
        elif stored["status"] != row["status"]:
            problems.append(f"{rid}: status was {stored['status']}, now {row['status']}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kernels", default="data/kernels.jsonl")
    parser.add_argument("--annotations", default="data/annotations.jsonl")
    parser.add_argument("--heldout", default="data/heldout_tasks.json")
    parser.add_argument("--min-records", type=int, default=MIN_RECORDS)
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--require-annotations", action="store_true")
    parser.add_argument("--remeasure", type=int, default=0, help="re-measure N annotated records on the GPU")
    parser.add_argument("--tolerance", type=float, default=0.25, help="allowed relative speedup difference when re-measuring")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    problems: list[str] = []
    records = read_jsonl(args.kernels)
    problems += check_records(records, args.min_records, args.max_records)
    problems += check_heldout(Path(args.heldout))
    distinct = len({r["source_sha256"] for r in records})
    print(f"records: {len(records)} ({distinct} distinct kernel sources)")

    annotations = load_annotations(args.annotations)
    if not annotations:
        print(f"annotations: PENDING ({args.annotations} not found or empty; run `make annotate` on the GPU)")
        if args.require_annotations:
            problems.append("annotations are required but missing")
    else:
        problems += check_annotations(records, annotations)
        current = {r["id"]: r["source_sha256"] for r in records}
        fresh = [a for a in annotations.values() if current.get(a["id"]) == a["source_sha256"]]
        ok = [a for a in fresh if a["status"] == "ok"]
        beat = [a for a in ok if a["speedup"] > 1.0]
        print(f"annotations: {len(fresh)}/{len(records)} records annotated, {len(ok)} correct, {len(beat)} faster than baseline")
        if args.require_annotations and len(fresh) < len(records):
            problems.append(f"only {len(fresh)}/{len(records)} records are annotated")

    if args.remeasure:
        import torch

        if not torch.cuda.is_available():
            print("error: --remeasure needs a CUDA device", file=sys.stderr)
            return 2
        from kernelgen.annotate import evaluate_record
        from kernelgen.rows import environment_info

        env = environment_info("cuda")
        problems += remeasure(
            records, annotations, args.remeasure, args.tolerance, args.seed,
            lambda r: evaluate_record(r, "cuda", 5, 30, range(3), 120.0, env),
        )

    for p in problems:
        print("PROBLEM:", p)
    print("dataset OK" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
