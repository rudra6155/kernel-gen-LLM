"""Copy the small training logs from runs/ into results/training/ so they can be committed.

The LoRA adapters in runs/ are large and stay out of git (see .gitignore). What backs
the claims about training is small: the loss curve and per-epoch samples of fine-tuning,
and the reward curve, every generated kernel, and any divergence report of RL.

Usage:
    python -m kernelgen.collect_logs
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Sequence

LOGS = {
    "sft": ["config.json", "data_stats.json", "train_log.csv", "epoch_summary.json", "samples"],
    "rl": ["config.json", "reward_curve.csv", "summary.json", "divergence_report.json", "samples"],
}


def collect(root: Path = Path("."), dest: str = "results/training") -> list[str]:
    """Copy whichever log files exist; returns the destination paths copied."""
    copied = []
    for run, items in LOGS.items():
        for item in items:
            src = root / "runs" / run / item
            if not src.exists():
                continue
            dst = root / dest / run / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(root)))
    return copied


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--dest", default="results/training")
    args = parser.parse_args(argv)
    copied = collect(Path(args.root), args.dest)
    for path in copied:
        print("copied", path)
    if not copied:
        print("nothing to copy: no logs under runs/sft or runs/rl yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
