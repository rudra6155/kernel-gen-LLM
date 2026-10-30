"""Build the kernel dataset from the family templates.

A record is one (task, config) pair rendered to Triton source. Its annotation
(correctness and speedup) depends on the task's shape, so two shapes that share
the same source are two records. Kernel sources for the normalization families
depend only on the config, so the number of distinct sources is lower than the
number of records; kernelgen.verify_dataset reports both.

Records contain no measurements. Annotations come from running the kernels on
the GPU (kernelgen.annotate) and live in a separate file.

Usage:
    python -m kernelgen.dataset --out data/kernels.jsonl --heldout-out data/heldout_tasks.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from kernelgen.families import Family, all_families

DEFAULT_TARGET = 720
MIN_RECORDS, MAX_RECORDS = 500, 2000

# Share of the dataset per family.
SHARES = {
    "layernorm": 0.18,
    "rmsnorm": 0.14,
    "softmax": 0.18,
    "gemm": 0.21,
    "gemm_bias_relu": 0.14,
    "attention": 0.15,
}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def quotas(target: int) -> dict[str, int]:
    q = {name: int(target * share) for name, share in SHARES.items()}
    names = list(SHARES)
    i = 0
    while sum(q.values()) < target:  # hand out the rounding remainder
        q[names[i % len(names)]] += 1
        i += 1
    return q


def make_record(fam: Family, params: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    spec = fam.spec(params)
    source = fam.render(params, config)
    ident = sha256(json.dumps({"task": spec, "config": config}, sort_keys=True))[:12]
    return {
        "id": f"{fam.name}-{ident}",
        "family": fam.name,
        "task": spec,
        "task_name": fam.task(params).name,
        "config": config,
        "source": source,
        "source_sha256": sha256(source),
    }


def build_records(target: int = DEFAULT_TARGET, seed: int = 0) -> list[dict[str, Any]]:
    """Deterministic: the same target and seed always give the same records.

    For each family, shapes are visited round-robin. A shape's default config
    comes first and its other configs follow in a seeded random order.
    """
    want = quotas(target)
    records: list[dict[str, Any]] = []
    for fam in all_families():
        queues = []
        for params in fam.train_shapes:
            default = fam.default_config(params)
            others = [c for c in fam.configs(params) if c != default]
            random.Random(f"{seed}:{fam.name}:{sorted(params.items())}").shuffle(others)
            queues.append((params, [default] + others))

        count, depth = 0, 0
        while count < want[fam.name]:
            progressed = False
            for params, queue in queues:
                if count >= want[fam.name]:
                    break
                if depth < len(queue):
                    records.append(make_record(fam, params, queue[depth]))
                    count += 1
                    progressed = True
            if not progressed:
                raise ValueError(f"family {fam.name} cannot supply {want[fam.name]} records")
            depth += 1
    return records


def heldout_tasks() -> list[dict[str, Any]]:
    """The held-out benchmark tasks: shapes that no training record uses."""
    out = []
    for fam in all_families():
        for params in fam.heldout_shapes:
            out.append({"id": fam.task(params).name, **fam.spec(params)})
    return out


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/kernels.jsonl")
    parser.add_argument("--heldout-out", default="data/heldout_tasks.json")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    if not MIN_RECORDS <= args.target <= MAX_RECORDS:
        parser.error(f"--target must be between {MIN_RECORDS} and {MAX_RECORDS}")

    records = build_records(args.target, args.seed)
    write_jsonl(records, args.out)
    heldout = heldout_tasks()
    Path(args.heldout_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.heldout_out).write_text(json.dumps(heldout, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    by_family: dict[str, int] = {}
    for r in records:
        by_family[r["family"]] = by_family.get(r["family"], 0) + 1
    print(f"wrote {len(records)} records ({len({r['source_sha256'] for r in records})} distinct sources) to {args.out}")
    print("per family:", ", ".join(f"{k}={v}" for k, v in by_family.items()))
    print(f"wrote {len(heldout)} held-out tasks to {args.heldout_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
