"""Figures and the README results tables, built from the committed result files.

The numbers in the README tables come from this module, not from typing: `make figures`
rewrites the block between the `results` markers in README.md and the images in docs/img/,
and a test fails if the README block differs from what the CSVs give. So the README cannot
drift from the data.

Usage:
    python -m kernelgen.figures            # rewrite README block and images
    python -m kernelgen.figures --check    # fail if the README block is stale
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

FAMILIES = ("layernorm", "rmsnorm", "softmax", "gemm", "gemm_bias_relu", "attention")
MODELS = ("base", "sft", "rl")
OUTCOMES = ("ok", "incorrect", "error", "rejected", "no_code")
OUTCOME_LABELS = {"ok": "correct", "incorrect": "wrong result", "error": "error", "rejected": "rejected by policy", "no_code": "no code"}
START, END = "<!-- results:start -->", "<!-- results:end -->"

# Validated with the data-viz palette checker: blue, orange, aqua, yellow, magenta in this order.
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SFT_C, RL_C, REF_C = "#2a78d6", "#eb6834", "#8f8e89"
OUTCOME_COLORS = {"ok": "#2a78d6", "incorrect": "#eb6834", "error": "#1baf7a", "rejected": "#eda100", "no_code": "#e87ba4"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def geomean(values: Sequence[float]) -> float | None:
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def best_per_task(rows: Sequence[dict[str, str]], key: str) -> dict[str, float]:
    """The largest value of `key` among each task's correct samples."""
    best: dict[str, float] = {}
    for r in rows:
        if r["status"] == "ok" and r.get(key):
            best[r["task"]] = max(best.get(r["task"], 0.0), float(r[key]))
    return best


def family_of(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    return {r["task"]: r["family"] for r in rows}


def load(root: Path) -> dict[str, Any]:
    results = root / "results"
    data: dict[str, Any] = {"template": read_csv(results / "kernel_benchmarks.csv")}
    for tag in MODELS:
        data[tag] = read_csv(results / f"generated_{tag}.csv")
    return data


def fmt_x(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}x"


# --- the README tables -----------------------------------------------------------------------------------------


def outcome_table(data: dict[str, Any]) -> list[str]:
    lines = [
        "| Model | Correct | Wrong result | Error | Rejected by policy | No code | Tasks with a correct kernel (of 50) | First sample correct |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {"base": "Base model", "sft": "After SFT", "rl": "After SFT + RL"}
    for tag in MODELS:
        rows = data[tag]
        counts = {s: sum(r["status"] == s for r in rows) for s in OUTCOMES}
        tasks = {r["task"] for r in rows}
        first = {}
        for r in rows:
            first.setdefault(r["task"], r["status"])
        solved = len(best_per_task(rows, "speedup"))
        lines.append(
            f"| {names[tag]} | {counts['ok']} | {counts['incorrect']} | {counts['error']} | {counts['rejected']} | "
            f"{counts['no_code']} | {solved} | {sum(s == 'ok' for s in first.values())}/{len(tasks)} |"
        )
    return lines


def speedup_table(data: dict[str, Any]) -> list[str]:
    template = {r["task"]: float(r["speedup"]) for r in data["template"]}
    fam = family_of(data["template"])
    sft, rl = best_per_task(data["sft"], "speedup"), best_per_task(data["rl"], "speedup")
    lines = [
        "| Family | Held-out tasks | Template, no model | SFT, best of 4 | SFT + RL, best of 4 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for f in FAMILIES:
        tasks = [t for t in template if fam[t] == f]
        cells = []
        for d in (template, sft, rl):
            got = [d[t] for t in tasks if t in d]
            cells.append(f"{fmt_x(geomean(got))} ({len(got)}/{len(tasks)})" if d is not template else fmt_x(geomean(got)))
        lines.append(f"| {f} | {len(tasks)} | " + " | ".join(cells) + " |")
    both = [t for t in sft if t in rl]
    lines.append(
        f"| **Tasks where SFT and RL both have a correct kernel** | {len(both)} | n/a | "
        f"{fmt_x(geomean([sft[t] for t in both]))} | {fmt_x(geomean([rl[t] for t in both]))} |"
    )
    return lines


def library_table(data: dict[str, Any]) -> list[str]:
    lines = [
        "| Kernels | Tasks compared | Above 1.00x | Above 1.05x | Geometric mean |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    sources = {
        "Template, no model (one measurement per task)": {r["task"]: float(r["speedup_vs_library"]) for r in data["template"] if r["speedup_vs_library"]},
        "SFT, best of 4": best_per_task(data["sft"], "speedup_vs_library"),
        "SFT + RL, best of 4": best_per_task(data["rl"], "speedup_vs_library"),
    }
    for name, d in sources.items():
        v = list(d.values())
        lines.append(f"| {name} | {len(v)} | {sum(x > 1.0 for x in v)} | {sum(x > 1.05 for x in v)} | {fmt_x(geomean(v))} |")
    return lines


def results_markdown(root: Path) -> str:
    data = load(root)
    parts = [
        "**What the 200 sampled kernels per model turned out to be** (50 held-out tasks, 4 samples each):",
        "",
        *outcome_table(data),
        "",
        "**Speedup over the naive eager PyTorch baseline** (geometric mean over tasks; the fraction is how many "
        "of the family's tasks have a correct kernel):",
        "",
        *speedup_table(data),
        "",
        "**Speedup over PyTorch's own optimized op** (layernorm, rmsnorm, softmax, gemm+bias+ReLU and attention; "
        "gemm has none because its baseline is already cuBLAS):",
        "",
        *library_table(data),
    ]
    return "\n".join(parts)


def update_readme(root: Path, write: bool) -> bool:
    """True if the README block matches the data. With `write`, rewrite it first."""
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise ValueError(f"README.md needs the {START} and {END} markers")
    head, rest = text.split(START, 1)
    old, tail = rest.split(END, 1)
    new = "\n\n" + results_markdown(root) + "\n\n"
    if write and old != new:
        path.write_text(head + START + new + END + tail, encoding="utf-8")
        return True
    return old == new


# --- figures ---------------------------------------------------------------------------------------------------


def _style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def plot_speedups(data: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    template = {r["task"]: float(r["speedup"]) for r in data["template"]}
    fam = family_of(data["template"])
    series = [("Template, no model", REF_C, template),
              ("SFT, best of 4", SFT_C, best_per_task(data["sft"], "speedup")),
              ("SFT + RL, best of 4", RL_C, best_per_task(data["rl"], "speedup"))]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    width, x = 0.26, range(len(FAMILIES))
    for i, (name, color, d) in enumerate(series):
        vals = [geomean([d[t] for t in template if fam[t] == f and t in d]) or 0 for f in FAMILIES]
        bars = ax.bar([k + (i - 1) * (width + 0.02) for k in x], vals, width, label=name, color=color, edgecolor=SURFACE, linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(1.0, color=INK_2, linewidth=1, linestyle=(0, (4, 3)), label="same speed as the baseline")
    ax.set_xticks(list(x), FAMILIES)
    ax.set_ylabel("speedup, geometric mean over tasks")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Speedup over the naive PyTorch baseline, held-out tasks")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outcomes(data: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    names = {"base": "Base model", "sft": "After SFT", "rl": "After SFT + RL"}
    fig, ax = plt.subplots(figsize=(9, 3.0))
    for y, tag in enumerate(MODELS):
        left = 0
        for s in OUTCOMES:
            n = sum(r["status"] == s for r in data[tag])
            if n:
                ax.barh(y, n, left=left, color=OUTCOME_COLORS[s], edgecolor=SURFACE, linewidth=2, height=0.62)
                if n >= 5:
                    ax.text(left + n / 2, y, str(n), ha="center", va="center", fontsize=9, color=INK)
            left += n
    ax.set_yticks(range(len(MODELS)), [names[m] for m in MODELS])
    ax.invert_yaxis()
    ax.set_xlim(0, 200)
    ax.set_xlabel("sampled kernels (50 held-out tasks x 4 samples)")
    from matplotlib.patches import Patch

    handles = [Patch(facecolor=OUTCOME_COLORS[s], label=OUTCOME_LABELS[s]) for s in OUTCOMES]
    ax.legend(handles=handles, frameon=False, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.22), fontsize=8)
    ax.set_title("What the sampled kernels turned out to be", pad=28)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _rolling(values: Sequence[float], window: int) -> list[float]:
    return [statistics.fmean(values[max(0, i - window + 1): i + 1]) for i in range(len(values))]


def plot_training(root: Path, path: Path) -> None:
    import matplotlib.pyplot as plt

    _style(plt)
    sft = read_csv(root / "results" / "training" / "sft" / "train_log.csv")
    rl = read_csv(root / "results" / "training" / "rl" / "reward_curve.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    ax = axes[0]
    ax.plot([int(r["step"]) for r in sft], [float(r["loss"]) for r in sft], color=SFT_C, linewidth=2)
    ax.set_title("SFT: training loss")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss")

    steps = [int(r["step"]) for r in rl]
    reward = [float(r["mean_reward"]) for r in rl]
    ax = axes[1]
    ax.plot(steps, reward, color=RL_C, linewidth=1, alpha=0.35, label="each step")
    ax.plot(steps, _rolling(reward, 10), color=RL_C, linewidth=2, label="10-step average")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title("RL: mean reward per step")
    ax.set_xlabel("RL step")
    ax.set_ylabel("speedup x correct")

    ax = axes[2]
    ax.plot(steps, [float(r["kl"]) if r["kl"] != "" else math.nan for r in rl], color=RL_C, linewidth=1.5)
    ax.set_ylim(bottom=0)
    ax.set_title("RL: distance from the SFT policy")
    ax.set_xlabel("RL step")
    ax.set_ylabel("KL per token")
    for ax in axes:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_figures(root: Path) -> list[Path]:
    out = root / "docs" / "img"
    out.mkdir(parents=True, exist_ok=True)
    data = load(root)
    paths = [out / "speedup_by_family.png", out / "sample_outcomes.png", out / "training.png"]
    plot_speedups(data, paths[0])
    plot_outcomes(data, paths[1])
    plot_training(root, paths[2])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--check", action="store_true", help="fail if the README results block is stale; write nothing")
    args = parser.parse_args(argv)
    root = Path(args.root)
    if args.check:
        ok = update_readme(root, write=False)
        print("README results block is up to date" if ok else "README results block is stale: run `make figures`", file=sys.stdout if ok else sys.stderr)
        return 0 if ok else 1
    changed = update_readme(root, write=True)
    for p in make_figures(root):
        print("wrote", p)
    print("updated the README results block" if changed else "README results block was already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
