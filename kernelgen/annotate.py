"""Measure correctness and speedup for every dataset record on the GPU.

Each record runs through the sandbox (policy check, worker process, timeout), so
a bad kernel costs one annotation row, not the run. Annotations are appended to
the output file as they finish, and a rerun skips records that are already
annotated for the current kernel source, so an interrupted run can be resumed.

Usage:
    python -m kernelgen.annotate --kernels data/kernels.jsonl --out data/annotations.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch

from kernelgen.dataset import read_jsonl
from kernelgen.rows import environment_info
from kernelgen.sandbox import DEFAULT_TIMEOUT_S, evaluate_isolated
from kernelgen.tasks import task_from_spec

ANNOTATION_FIELDS = [
    "id",
    "source_sha256",
    "task_name",
    "family",
    "status",
    "correct",
    "max_abs_err",
    "baseline_ms",
    "candidate_ms",
    "speedup",
    "library_ms",
    "speedup_vs_library",
    "error",
    "warmup",
    "reps",
    "seeds",
    "gpu_name",
    "torch_version",
    "cuda_version",
    "triton_version",
]

log = logging.getLogger(__name__)


def to_annotation(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    ann = {k: row.get(k) for k in ANNOTATION_FIELDS}
    ann.update(id=record["id"], source_sha256=record["source_sha256"], task_name=record["task_name"], family=record["family"])
    return ann


def load_annotations(path: str | Path) -> dict[str, dict[str, Any]]:
    """Annotations by record id. If an id appears twice, the later line wins."""
    path = Path(path)
    if not path.exists():
        return {}
    return {a["id"]: a for a in read_jsonl(path)}


def evaluate_record(
    record: dict[str, Any],
    device: str,
    warmup: int,
    reps: int,
    seeds: Sequence[int],
    timeout_s: float,
    env: dict[str, str] | None,
) -> dict[str, Any]:
    task = task_from_spec(record["task"])
    return evaluate_isolated(
        task, record["source"], record["id"], device,
        warmup=warmup, reps=reps, seeds=seeds, env=env, timeout_s=timeout_s,
    )


def annotate(
    records: Iterable[dict[str, Any]],
    out_path: str | Path,
    evaluate_fn: Callable[[dict[str, Any]], dict[str, Any]],
    progress: Callable[[str], None] = print,
) -> dict[str, int]:
    """Annotate `records` that are not yet annotated for their current source.

    Returns counts: done (annotated now), skipped (already annotated).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = list(records)
    existing = load_annotations(out_path)
    todo = [r for r in records if existing.get(r["id"], {}).get("source_sha256") != r["source_sha256"]]
    counts = {"done": 0, "skipped": len(records) - len(todo)}

    start = time.monotonic()
    with out_path.open("a", encoding="utf-8") as f:
        for i, record in enumerate(todo, 1):
            row = evaluate_fn(record)
            f.write(json.dumps(to_annotation(record, row), sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
            counts["done"] += 1
            speedup = "-" if row.get("speedup") is None else f"{row['speedup']:.2f}x"
            elapsed = time.monotonic() - start
            eta = elapsed / i * (len(todo) - i)
            progress(
                f"[{i}/{len(todo)}] {record['id']:<28} {row['status']:<9} {speedup:>8}  "
                f"elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min"
            )
    return counts


def select(records: list[dict[str, Any]], family: str | None, shard: str | None, limit: int | None) -> list[dict[str, Any]]:
    if family:
        records = [r for r in records if r["family"] == family]
    if shard:
        index, total = (int(x) for x in shard.split("/"))
        if not (total >= 1 and 0 <= index < total):
            raise ValueError(f"--shard must be i/n with 0 <= i < n, got {shard!r}")
        records = [r for k, r in enumerate(records) if k % total == index]
    return records[:limit] if limit else records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kernels", default="data/kernels.jsonl")
    parser.add_argument("--out", default="data/annotations.jsonl")
    parser.add_argument("--family", default=None)
    parser.add_argument("--shard", default=None, help="i/n: annotate every n-th record starting at i")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30, help="fewer than make benchmark uses, to keep a long run short")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        print("error: no CUDA device available; refusing to write annotations", file=sys.stderr)
        return 2

    records = select(read_jsonl(args.kernels), args.family, args.shard, args.limit)
    env = environment_info("cuda")
    counts = annotate(
        records,
        args.out,
        lambda r: evaluate_record(r, "cuda", args.warmup, args.reps, range(args.seeds), args.timeout, env),
    )
    print(f"annotated {counts['done']} records, skipped {counts['skipped']} already annotated, -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
