"""QLoRA supervised fine-tuning of Qwen2.5-Coder-3B-Instruct on annotated kernels.

The model sees the task prompt and learns to write the kernel; the loss covers
only the reply. Training uses kernels that the GPU annotation found correct and
at least `--min-speedup` faster than the baseline, so the model is shown what
works. Examples are split by task shape, so validation loss and the per-epoch
samples measure generalization to shapes not seen in training.

After every epoch the script logs the training and validation loss, saves the
LoRA adapter, and writes a few generated samples (checked for code extraction,
Python syntax, and the candidate policy) to runs/sft/samples/.

Usage:
    python -m kernelgen.finetune --out runs/sft
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from kernelgen.annotate import load_annotations
from kernelgen.dataset import read_jsonl
from kernelgen.modeling import DEFAULT_MODEL, add_lora, input_device, load_base_model, load_tokenizer, sample
from kernelgen.policy import check_source
from kernelgen.prompts import build_prompt, extract_code, format_response, to_messages

log = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    model_name: str = DEFAULT_MODEL
    tokenizer_name: str | None = None
    kernels: str = "data/kernels.jsonl"
    annotations: str = "data/annotations.jsonl"
    out_dir: str = "runs/sft"
    quantize: bool = True
    epochs: int = 3
    micro_batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    max_seq_len: int = 1536  # the longest example is about 1,030 tokens with the Qwen tokenizer
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    min_speedup: float = 1.0
    include_unannotated: bool = False
    val_fraction: float = 0.1
    n_samples: int = 3
    sample_max_new_tokens: int = 1024
    log_every: int = 5
    seed: int = 0


# --- data ----------------------------------------------------------------------------------------------


def load_sft_examples(
    kernels: str | Path,
    annotations: str | Path,
    min_speedup: float = 1.0,
    include_unannotated: bool = False,
) -> list[dict[str, Any]]:
    """(prompt, response) examples from kernels the annotations show to be correct and fast enough.

    Annotations for an older version of a kernel's source are ignored. With
    `include_unannotated`, kernels without a current annotation are used too;
    their correctness on the GPU has then not been measured.
    """
    records = read_jsonl(kernels)
    ann = load_annotations(annotations)
    if not ann and not include_unannotated:
        raise FileNotFoundError(
            f"no annotations at {annotations}. Run `make annotate` on the GPU first, "
            "or pass --include-unannotated to train on unmeasured template kernels."
        )

    prompts: dict[str, str] = {}
    examples = []
    for rec in records:
        a = ann.get(rec["id"])
        current = a is not None and a["source_sha256"] == rec["source_sha256"]
        if current:
            if a["status"] != "ok" or not a["correct"] or a["speedup"] < min_speedup:
                continue
        elif not include_unannotated:
            continue
        key = json.dumps(rec["task"], sort_keys=True)
        if key not in prompts:
            prompts[key] = build_prompt(rec["task"])
        examples.append(
            {
                "id": rec["id"],
                "family": rec["family"],
                "task_name": rec["task_name"],
                "prompt": prompts[key],
                "response": format_response(rec["source"]),
                "speedup": a["speedup"] if current else None,
            }
        )
    return examples


def split_by_task(examples: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list, list]:
    """Hold out whole tasks (shapes) for validation, so no validation prompt appears in training."""
    tasks = sorted({e["task_name"] for e in examples})
    random.Random(seed).shuffle(tasks)
    n_val = max(1, round(len(tasks) * val_fraction)) if val_fraction > 0 and len(tasks) > 1 else 0
    val_tasks = set(tasks[:n_val])
    return (
        [e for e in examples if e["task_name"] not in val_tasks],
        [e for e in examples if e["task_name"] in val_tasks],
    )


def build_features(tokenizer, example: dict[str, Any], max_len: int) -> dict[str, list[int]] | None:
    """Token ids and labels for one example; labels are -100 on the prompt.

    Returns None if the example is longer than `max_len`. Truncating would cut
    the kernel off, which would teach the model to emit broken code.
    """
    prompt_text = tokenizer.apply_chat_template(to_messages(example["prompt"]), tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(to_messages(example["prompt"], example["response"]), tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("the chat template does not extend the prompt; cannot mask the prompt tokens")
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt tokens are not a prefix of the full example's tokens")
    if len(full_ids) > max_len:
        return None
    return {"input_ids": full_ids, "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]}


def collate(features: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(f["input_ids"]) for f in features)
    ids = torch.full((len(features), width), pad_id, dtype=torch.long)
    labels = torch.full((len(features), width), -100, dtype=torch.long)
    mask = torch.zeros((len(features), width), dtype=torch.long)
    for i, f in enumerate(features):
        n = len(f["input_ids"])
        ids[i, :n] = torch.tensor(f["input_ids"])
        labels[i, :n] = torch.tensor(f["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def lr_at(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup, then cosine decay to 10% of the base rate."""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def prepare_data(cfg: SFTConfig, tokenizer) -> dict[str, Any]:
    """Select, split, and tokenize the training data, with statistics about what was kept."""
    examples = load_sft_examples(cfg.kernels, cfg.annotations, cfg.min_speedup, cfg.include_unannotated)
    train_ex, val_ex = split_by_task(examples, cfg.val_fraction, cfg.seed)
    if not train_ex:
        raise ValueError(
            f"no training examples left from {len(examples)} candidates; lower --min-speedup "
            "or annotate more kernels"
        )
    train_feats = [f for f in (build_features(tokenizer, e, cfg.max_seq_len) for e in train_ex) if f]
    val_feats = [f for f in (build_features(tokenizer, e, cfg.max_seq_len) for e in val_ex) if f]
    speedups = sorted(e["speedup"] for e in examples if e["speedup"] is not None)
    lengths = [len(f["input_ids"]) for f in train_feats]
    stats = {
        "examples": len(examples),
        "train": len(train_feats),
        "val": len(val_feats),
        "dropped_too_long": len(train_ex) + len(val_ex) - len(train_feats) - len(val_feats),
        "per_family": {fam: sum(e["family"] == fam for e in examples) for fam in sorted({e["family"] for e in examples})},
        "tasks": len({e["task_name"] for e in examples}),
        "unmeasured_examples": sum(e["speedup"] is None for e in examples),
        "speedup_min_median_max": (
            [speedups[0], speedups[len(speedups) // 2], speedups[-1]] if speedups else None
        ),
        "train_tokens": sum(lengths),
        "longest_example_tokens": max(lengths) if lengths else 0,
    }
    return {"train_ex": train_ex, "val_ex": val_ex, "train_feats": train_feats, "val_feats": val_feats, "stats": stats}


def dry_run(cfg: SFTConfig, tokenizer=None) -> dict[str, Any]:
    """What training would use, without loading the model or touching the GPU."""
    stats = prepare_data(cfg, tokenizer or load_tokenizer(cfg.tokenizer_name or cfg.model_name))["stats"]
    steps_per_epoch = math.ceil(stats["train"] / (cfg.micro_batch_size * cfg.grad_accum))
    stats["optimizer_steps_per_epoch"] = steps_per_epoch
    stats["optimizer_steps_total"] = steps_per_epoch * cfg.epochs
    return stats


# --- training ----------------------------------------------------------------------------------------------


def _autocast(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


@torch.no_grad()
def validation_loss(model, features: list[dict[str, list[int]]], pad_id: int, device: torch.device) -> float | None:
    if not features:
        return None
    model.eval()
    total, count = 0.0, 0
    for f in features:
        batch = {k: v.to(device) for k, v in collate([f], pad_id).items()}
        with _autocast(device):
            loss = model(**batch).loss
        n = int((batch["labels"][:, 1:] != -100).sum())
        total += float(loss) * n
        count += n
    model.train()
    return total / max(count, 1)


def generate_samples(model, tokenizer, examples: list[dict[str, Any]], n: int, max_new_tokens: int) -> list[dict[str, Any]]:
    """Greedy samples for the first `n` distinct validation tasks, checked but not executed."""
    chosen, seen = [], set()
    for e in examples:
        if e["task_name"] not in seen:
            seen.add(e["task_name"])
            chosen.append(e)
        if len(chosen) == n:
            break
    if not chosen:
        return []
    replies = sample(model, tokenizer, [e["prompt"] for e in chosen], max_new_tokens=max_new_tokens, greedy=True)
    out = []
    for e, (s,) in zip(chosen, replies):
        code = extract_code(s.text)
        parses, violations = False, []
        if code is not None:
            try:
                ast.parse(code)
                parses = True
            except (SyntaxError, ValueError):
                pass
            violations = check_source(code)
        out.append(
            {
                "id": e["id"],
                "task_name": e["task_name"],
                "finished": s.finished,
                "code_extracted": code is not None,
                "parses": parses,
                "policy_violations": violations,
                "generated": s.text,
            }
        )
    return out


def train(cfg: SFTConfig, model=None, tokenizer=None) -> dict[str, Any]:
    """Run fine-tuning and return the per-epoch summary. `model` and `tokenizer` may be injected for tests."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")

    tokenizer = tokenizer or load_tokenizer(cfg.tokenizer_name or cfg.model_name)
    data = prepare_data(cfg, tokenizer)
    train_ex, val_ex, train_feats, val_feats, stats = (
        data["train_ex"], data["val_ex"], data["train_feats"], data["val_feats"], data["stats"],
    )
    (out / "data_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    log.info("data: %s", stats)

    if model is None:
        model = add_lora(
            load_base_model(cfg.model_name, quantize=cfg.quantize), cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout
        )
    model.train()
    device = input_device(model)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)

    steps_per_epoch = math.ceil(len(train_feats) / (cfg.micro_batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    rng = random.Random(cfg.seed)

    log_path = out / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "step", "loss", "lr", "grad_norm", "tokens_per_s", "elapsed_s"])

    summaries: list[dict[str, Any]] = []
    step, start = 0, time.monotonic()
    for epoch in range(1, cfg.epochs + 1):
        order = list(range(len(train_feats)))
        rng.shuffle(order)
        epoch_loss, epoch_tokens, window_loss, window_tokens, window_start = 0.0, 0, 0.0, 0, time.monotonic()
        chunk = cfg.micro_batch_size * cfg.grad_accum
        for first in range(0, len(order), chunk):
            group = order[first : first + chunk]
            micro = [group[i : i + cfg.micro_batch_size] for i in range(0, len(group), cfg.micro_batch_size)]
            lr = lr_at(step, total_steps, cfg.lr, cfg.warmup_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr
            step_loss, step_tokens = 0.0, 0
            for idx in micro:
                batch = {k: v.to(device) for k, v in collate([train_feats[i] for i in idx], tokenizer.pad_token_id).items()}
                with _autocast(device):
                    loss = model(**batch).loss
                (loss / len(micro)).backward()
                n = int((batch["labels"][:, 1:] != -100).sum())
                step_loss += float(loss.detach()) * n
                step_tokens += n
            if not math.isfinite(step_loss):
                raise FloatingPointError(f"non-finite loss at step {step}; see {log_path}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            epoch_loss += step_loss
            epoch_tokens += step_tokens
            window_loss += step_loss
            window_tokens += step_tokens
            if step % cfg.log_every == 0 or first + chunk >= len(order):
                now = time.monotonic()
                row = [
                    epoch, step, window_loss / max(window_tokens, 1), lr, grad_norm,
                    window_tokens / max(now - window_start, 1e-9), now - start,
                ]
                with log_path.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(row)
                log.info("epoch %d step %d/%d loss %.4f lr %.2e grad_norm %.2f", epoch, step, total_steps, row[2], lr, grad_norm)
                window_loss, window_tokens, window_start = 0.0, 0, now

        val = validation_loss(model, val_feats, tokenizer.pad_token_id, device)
        samples = generate_samples(model, tokenizer, val_ex, cfg.n_samples, cfg.sample_max_new_tokens)
        (out / "samples" / f"epoch_{epoch}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in samples), encoding="utf-8"
        )
        model.save_pretrained(out / f"epoch_{epoch}")
        summary = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(epoch_tokens, 1),
            "val_loss": val,
            "samples": len(samples),
            "samples_with_code": sum(s["code_extracted"] for s in samples),
            "samples_parse": sum(s["parses"] for s in samples),
            "samples_policy_clean": sum(s["code_extracted"] and not s["policy_violations"] for s in samples),
        }
        summaries.append(summary)
        log.info("epoch %d: %s", epoch, summary)
        for s in samples:
            head = " ".join(s["generated"].split())[:160]
            log.info("  sample %s (code=%s parses=%s policy_ok=%s): %s", s["task_name"], s["code_extracted"], s["parses"], not s["policy_violations"], head)

    model.save_pretrained(out / "final")
    tokenizer.save_pretrained(out / "final")
    (out / "epoch_summary.json").write_text(json.dumps(summaries, indent=1), encoding="utf-8")
    return {"data": stats, "epochs": summaries}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    defaults = SFTConfig()
    parser.add_argument("--model", dest="model_name", default=defaults.model_name)
    parser.add_argument("--tokenizer", dest="tokenizer_name", default=None)
    parser.add_argument("--kernels", default=defaults.kernels)
    parser.add_argument("--annotations", default=defaults.annotations)
    parser.add_argument("--out", dest="out_dir", default=defaults.out_dir)
    parser.add_argument("--no-quantize", dest="quantize", action="store_false", help="skip 4-bit quantization (needs far more memory)")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--micro-batch-size", type=int, default=defaults.micro_batch_size)
    parser.add_argument("--grad-accum", type=int, default=defaults.grad_accum)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)
    parser.add_argument("--lora-r", type=int, default=defaults.lora_r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora_alpha)
    parser.add_argument("--min-speedup", type=float, default=defaults.min_speedup)
    parser.add_argument("--include-unannotated", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--n-samples", type=int, default=defaults.n_samples)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--dry-run", action="store_true", help="show what training would use, without loading the model")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dry = args.dry_run
    cfg = SFTConfig(**{k: v for k, v in vars(args).items() if k != "dry_run"})
    if dry:
        print(json.dumps(dry_run(cfg), indent=1))
        return 0
    if args.quantize and not torch.cuda.is_available():
        parser.error("4-bit QLoRA needs a CUDA GPU; none found")
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
