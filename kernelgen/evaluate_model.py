"""Evaluate kernels written by a model on the 50 held-out tasks.

For every held-out task the model writes `--samples-per-task` kernels. Each one
goes through the full sandboxed measurement (policy check, correctness gate, timing
on fresh data) and gets a row in results/generated_<tag>.csv, with the same columns
as results/kernel_benchmarks.csv. Replies with no complete code block get a row too,
with status no_code.

The summary distinguishes what it measures:
  pass_at_1       fraction of tasks whose first sample is correct
  pass_at_k       fraction of tasks with at least one correct sample
  best-of-k       per task, the fastest correct sample; the model had k tries and
                  the measurement picked the winner, so it is not a single-shot number
Speedups are over the naive PyTorch baseline; the "library" ones are over PyTorch's
own optimized op, which is a harder bar.

If a correct kernel is slower than the baseline, the slowest few are profiled and
the profile facts are logged (results/slow_kernels_<tag>.json), and nothing is
claimed from them.

Usage:
    python -m kernelgen.evaluate_model --tag rl --adapter runs/rl/final
    python -m kernelgen.evaluate_model --tag base            # the untuned base model
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from kernelgen.diagnose import describe_slowdown
from kernelgen.prompts import build_prompt, extract_code
from kernelgen.rows import FIELDS, PROFILE_FIELDS, environment_info, fail_row, new_row, write_csv
from kernelgen.sandbox import DEFAULT_TIMEOUT_S, evaluate_isolated, profile_isolated
from kernelgen.tasks import task_from_spec

log = logging.getLogger(__name__)

GenerateFn = Callable[[list[str], int], list[list[str]]]  # (prompts, k) -> k replies per prompt


@dataclass
class EvalConfig:
    tag: str = "model"
    adapter: str | None = None
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    tokenizer_name: str | None = None
    tasks_file: str = "data/heldout_tasks.json"
    out_dir: str = "results"
    quantize: bool = True
    samples_per_task: int = 4
    temperature: float = 0.6
    top_p: float = 0.95
    max_new_tokens: int = 1024
    warmup: int = 10
    reps: int = 100
    seeds: int = 3
    timeout: float = DEFAULT_TIMEOUT_S
    limit_tasks: int | None = None
    profile_slow: int = 3


def load_heldout(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    return tasks[:limit] if limit else tasks


def make_model_generator(cfg: EvalConfig) -> GenerateFn:
    from kernelgen.modeling import add_lora, load_adapter, load_base_model, load_tokenizer, sample

    tokenizer = load_tokenizer(cfg.tokenizer_name or cfg.model_name)
    model = load_base_model(cfg.model_name, quantize=cfg.quantize)
    if cfg.adapter:
        model = load_adapter(model, cfg.adapter, trainable=False)
    model.eval()

    def generate(prompts: list[str], k: int) -> list[list[str]]:
        replies = []
        for prompt in prompts:  # one prompt at a time keeps the key-value cache small
            (group,) = sample(
                model, tokenizer, [prompt], cfg.max_new_tokens, cfg.temperature, cfg.top_p, num_return_sequences=k
            )
            replies.append([s.text for s in group])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return replies

    return generate


def evaluate_task_replies(
    spec: dict[str, Any], replies: Sequence[str], cfg: EvalConfig, env: dict[str, str], device: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows and reply records for one task's replies."""
    task = task_from_spec(spec)
    rows, records = [], []
    for k, reply in enumerate(replies):
        candidate_id = f"gen:{cfg.tag}:{task.name}:{k}"
        code = extract_code(reply)
        if code is None:
            row = new_row(task, candidate_id, cfg.warmup, cfg.reps, cfg.seeds, env)
            fail_row(row, "no_code", "no complete python code block defining solution()")
        else:
            row = evaluate_isolated(
                task, code, candidate_id, device,
                warmup=cfg.warmup, reps=cfg.reps, seeds=range(cfg.seeds), env=env, timeout_s=cfg.timeout,
            )
        rows.append(row)
        records.append({"candidate_id": candidate_id, "task": task.name, "sample": k, "reply": reply, "code": code})
    return rows, records


def _geomean(values: Sequence[float]) -> float | None:
    return math.exp(statistics.fmean(math.log(v) for v in values)) if values else None


def summarize(rows: Sequence[dict[str, Any]], n_tasks: int) -> dict[str, Any]:
    """Aggregate metrics over the rows of one evaluation. See the module docstring for definitions."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    def best(task_rows: list[dict[str, Any]], key: str) -> float | None:
        values = [r[key] for r in task_rows if r["status"] == "ok" and r[key] is not None]
        return max(values) if values else None

    best_speedup = {t: best(rs, "speedup") for t, rs in by_task.items()}
    best_library = {t: best(rs, "speedup_vs_library") for t, rs in by_task.items()}
    with_correct = [v for v in best_speedup.values() if v is not None]
    with_library = [v for v in best_library.values() if v is not None]
    correct_rows = [r for r in rows if r["status"] == "ok"]

    per_family = {}
    for fam in sorted({r["family"] for r in rows}):
        tasks = [t for t, rs in by_task.items() if rs[0]["family"] == fam]
        got = [best_speedup[t] for t in tasks if best_speedup[t] is not None]
        per_family[fam] = {
            "tasks": len(tasks),
            "pass_at_k": len(got) / len(tasks),
            "best_of_k_speedup_geomean": _geomean(got),
        }

    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    return {
        "n_tasks": n_tasks,
        "n_tasks_evaluated": len(by_task),
        "n_samples": len(rows),
        "status_counts": status_counts,
        "frac_samples_with_code": sum(r["status"] != "no_code" for r in rows) / len(rows) if rows else None,
        "frac_samples_correct": len(correct_rows) / len(rows) if rows else None,
        "pass_at_1": sum(rs[0]["status"] == "ok" for rs in by_task.values()) / n_tasks,
        "pass_at_k": len(with_correct) / n_tasks,
        "tasks_with_a_correct_sample": len(with_correct),
        "tasks_where_best_beats_baseline": sum(v > 1.0 for v in with_correct),
        "best_of_k_speedup_geomean": _geomean(with_correct),
        "mean_speedup_of_correct_samples": statistics.fmean(r["speedup"] for r in correct_rows) if correct_rows else None,
        "tasks_where_best_beats_library": sum(v > 1.0 for v in with_library),
        "tasks_with_a_library_comparison": len(with_library),
        "best_of_k_speedup_vs_library_geomean": _geomean(with_library),
        "per_family": per_family,
    }


def diagnose_slow(
    rows: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    cfg: EvalConfig,
    env: dict[str, str],
    device: str,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Profile the slowest correct kernels that lost to the baseline and log the facts."""
    code_by_id = {r["candidate_id"]: r["code"] for r in records}
    slow = sorted(
        (r for r in rows if r["status"] == "ok" and r["speedup"] is not None and r["speedup"] < 1.0),
        key=lambda r: r["speedup"],
    )[: cfg.profile_slow]
    reports, profile_rows = [], []
    for row in slow:
        task = task_from_spec(_spec_from_row(row))
        trace_dir = out_dir / "traces"
        common = dict(warmup=5, iters=20, seeds=range(cfg.seeds), env=env, timeout_s=cfg.timeout)
        base = profile_isolated(task, "baseline", None, "pytorch_baseline", device, trace_path=trace_dir / f"{row['candidate_id']}__baseline.json".replace(":", "_"), **common)
        cand = profile_isolated(task, "candidate", code_by_id[row["candidate_id"]], row["candidate_id"], device, trace_path=trace_dir / f"{row['candidate_id']}__candidate.json".replace(":", "_"), **common)
        profile_rows += [base, cand]
        facts = describe_slowdown(cand, base)
        log.warning("slower than the baseline (%.2fx): %s", row["speedup"], row["candidate_id"])
        for note in facts:
            log.warning("  %s", note)
        reports.append({"candidate_id": row["candidate_id"], "task": row["task"], "speedup": row["speedup"], "observations": facts})
    if profile_rows:
        write_csv(profile_rows, out_dir / f"slow_kernel_profiles_{cfg.tag}.csv", PROFILE_FIELDS)
    return reports


def _spec_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """The task spec of an evaluation row (family, params, dtype)."""
    return {"family": row["family"], "params": json.loads(row["params"]), "dtype": row["dtype"]}


def run_evaluation(
    cfg: EvalConfig,
    generate: GenerateFn | None = None,
    tasks: list[dict[str, Any]] | None = None,
    device: str = "cuda",
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate, measure, and write results/generated_<tag>.csv, the replies, and a summary."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = tasks if tasks is not None else load_heldout(cfg.tasks_file, cfg.limit_tasks)
    env = env if env is not None else environment_info(device)
    generate = generate or make_model_generator(cfg)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    csv_path = out_dir / f"generated_{cfg.tag}.csv"
    for i, spec in enumerate(tasks, 1):
        (replies,) = generate([build_prompt(spec)], cfg.samples_per_task)
        task_rows, task_records = evaluate_task_replies(spec, replies, cfg, env, device)
        rows += task_rows
        records += task_records
        write_csv(rows, csv_path)  # keep the partial results if a later task dies
        ok = [r["speedup"] for r in task_rows if r["status"] == "ok"]
        log.info(
            "[%d/%d] %s: %d/%d correct%s", i, len(tasks), task_rows[0]["task"], len(ok), len(task_rows),
            f", best {max(ok):.2f}x" if ok else "",
        )

    (out_dir / f"generated_{cfg.tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    summary = summarize(rows, len(tasks))
    summary["config"] = asdict(cfg)
    summary["slow_kernels"] = diagnose_slow(rows, records, cfg, env, device, out_dir) if cfg.profile_slow else []
    (out_dir / f"generated_{cfg.tag}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = EvalConfig()
    parser.add_argument("--tag", default=d.tag, help="names the output files, e.g. base, sft, rl")
    parser.add_argument("--adapter", default=None, help="LoRA adapter to evaluate; omit for the untuned base model")
    parser.add_argument("--model", dest="model_name", default=d.model_name)
    parser.add_argument("--tasks-file", default=d.tasks_file)
    parser.add_argument("--out-dir", default=d.out_dir)
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.add_argument("--samples-per-task", type=int, default=d.samples_per_task)
    parser.add_argument("--temperature", type=float, default=d.temperature)
    parser.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    parser.add_argument("--warmup", type=int, default=d.warmup)
    parser.add_argument("--reps", type=int, default=d.reps)
    parser.add_argument("--timeout", type=float, default=d.timeout)
    parser.add_argument("--limit-tasks", type=int, default=None, help="only the first N held-out tasks (a quick trial)")
    parser.add_argument("--profile-slow", type=int, default=d.profile_slow, help="profile this many correct-but-slower kernels")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        parser.error("evaluation needs a CUDA GPU to generate and to measure kernels; none found")
    summary = run_evaluation(EvalConfig(**vars(args)))
    keys = ("n_tasks", "n_samples", "status_counts", "pass_at_1", "pass_at_k", "tasks_where_best_beats_baseline", "best_of_k_speedup_geomean")
    print(json.dumps({k: summary[k] for k in keys}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
