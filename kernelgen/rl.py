"""GRPO reinforcement learning with the measured speedup as the reward.

Each step samples a few training tasks, generates a group of kernels per task,
and scores every kernel in the sandbox: reward = speedup * correct, so a kernel
that does not compile, crashes, times out, breaks the candidate policy, or is
wrong earns 0, and a correct kernel earns its speedup over the PyTorch baseline.
Within each task's group the rewards are standardized into advantages (Group
Relative Policy Optimization), and the policy takes a gradient step that makes
above-average kernels more likely and below-average ones less likely, with a KL
penalty to the starting policy.

Only training-set tasks are used; the held-out tasks are never seen here.

Reward evaluation is sequential on purpose: kernels timed at the same time on
one GPU disturb each other's timings.

Logged under --out:
  reward_curve.csv       one row per step (rewards, correct and faster fractions, loss, KL, ...)
  samples/step_N.jsonl   every generated kernel with its reward, status, and error
  step_N/, final/        LoRA checkpoints
  divergence_report.json written, and training stopped, if the run diverges: the
                         reason, the whole reward curve, and the most recent generated code

Usage:
    python -m kernelgen.rl --init-adapter runs/sft/final --out runs/rl
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import math
import random
import statistics
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from kernelgen.dataset import read_jsonl, sha256
from kernelgen.modeling import DEFAULT_MODEL, add_lora, load_adapter, load_base_model, load_tokenizer, sample
from kernelgen.prompts import build_prompt, extract_code
from kernelgen.rows import environment_info
from kernelgen.sandbox import evaluate_isolated
from kernelgen.tasks import task_from_spec

log = logging.getLogger(__name__)

CURVE_FIELDS = [
    "step", "mean_reward", "max_reward", "frac_correct", "frac_faster", "mean_speedup_correct",
    "frac_no_code", "frac_rejected", "frac_failed", "dead_groups", "active_samples",
    "loss", "kl", "grad_norm", "mean_completion_tokens", "frac_unfinished", "step_seconds",
]


@dataclass
class RLConfig:
    model_name: str = DEFAULT_MODEL
    tokenizer_name: str | None = None
    init_adapter: str | None = None  # the SFT adapter to start from; None starts from the base model
    kernels: str = "data/kernels.jsonl"  # where the training tasks come from
    out_dir: str = "runs/rl"
    quantize: bool = True
    steps: int = 100
    prompts_per_step: int = 2
    group_size: int = 8
    max_new_tokens: int = 1024
    temperature: float = 0.9
    top_p: float = 0.95
    lr: float = 1e-5
    beta: float = 0.04  # weight of the KL penalty to the starting policy
    clip_eps: float = 0.2
    num_updates: int = 1  # gradient updates per batch of samples
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    reward_warmup: int = 5
    reward_reps: int = 20  # fewer than make benchmark uses: the reward is noisier, but 8 hours shorter
    reward_seeds: int = 3
    reward_timeout: float = 120.0
    save_every: int = 10
    # Divergence detection. These thresholds are starting values, not tuned ones.
    kl_max: float = 5.0
    collapse_window: int = 10
    collapse_ratio: float = 0.25
    collapse_min_best: float = 0.5
    seed: int = 0


# --- reward -------------------------------------------------------------------------------------------------


@dataclass
class RewardResult:
    reward: float
    status: str  # ok, incorrect, error, rejected, timeout, crash, or no_code
    correct: bool
    speedup: float | None
    error: str
    code: str | None


RewardFn = Callable[[dict[str, Any], str], RewardResult]


def make_sandbox_reward(
    device: str = "cuda",
    warmup: int = 5,
    reps: int = 20,
    seeds: int = 3,
    timeout_s: float = 120.0,
    env: dict[str, str] | None = None,
) -> RewardFn:
    """The real reward: extract the kernel from the reply and measure it in the sandbox.

    Results are cached by (task, code), so a kernel generated twice is measured once.
    """
    cache: dict[tuple[str, str], RewardResult] = {}

    def reward_fn(task_spec: dict[str, Any], text: str) -> RewardResult:
        code = extract_code(text)
        if code is None:
            return RewardResult(0.0, "no_code", False, None, "no complete python code block defining solution()", None)
        key = (json.dumps(task_spec, sort_keys=True), sha256(code))
        if key not in cache:
            row = evaluate_isolated(
                task_from_spec(task_spec), code, f"rl-{key[1][:8]}", device,
                warmup=warmup, reps=reps, seeds=range(seeds), env=env, timeout_s=timeout_s,
            )
            cache[key] = RewardResult(
                float(row["reward"]), row["status"], bool(row["correct"]), row["speedup"], row["error"] or "", code
            )
        return cache[key]

    return reward_fn


# --- the GRPO objective -----------------------------------------------------------------------------------------


def group_advantages(rewards: Sequence[float]) -> list[float]:
    """Rewards standardized within a group. All zeros if every reward is the same (nothing to learn from)."""
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std == 0.0:
        return [0.0] * len(rewards)
    return [(r - mean) / (std + 1e-4) for r in rewards]


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Per-token estimate of KL(policy || reference) that is never negative (Schulman's k3)."""
    d = ref_logp - logp
    return d.exp() - d - 1.0


def grpo_sample_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor | None,
    advantage: float,
    clip_eps: float,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss for one completion, averaged over its tokens, and its mean KL to the reference."""
    ratio = (logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
    kl = kl_k3(logp, ref_logp) if ref_logp is not None else torch.zeros_like(logp)
    return (-surrogate + beta * kl).mean(), kl.detach().mean()


def token_logprobs(model, prompt_ids: Sequence[int], completion_ids: Sequence[int]) -> torch.Tensor:
    """Log-probability the model gives each completion token, shape [len(completion_ids)]."""
    device = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], device=device)
    logits = model(input_ids=ids).logits[0, len(prompt_ids) - 1 : -1].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, ids[0, len(prompt_ids) :].unsqueeze(-1)).squeeze(-1)


# --- reference policy by swapping LoRA weights --------------------------------------------------------------------


def lora_state(model) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@contextlib.contextmanager
def use_lora_state(model, state: dict[str, torch.Tensor]):
    """Temporarily load `state` into the trainable weights, then restore the current ones."""
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    current = {n: p.detach().clone() for n, p in params.items()}
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n])
    try:
        yield
    finally:
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(current[n])


def disable_dropout(model) -> None:
    """Sampling runs without dropout, so training must too, or old and new log-probs would not match."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0


# --- divergence ------------------------------------------------------------------------------------------------------


class DivergenceMonitor:
    """Flags a run that has gone bad: non-finite values, KL blow-up, or collapsed rewards."""

    def __init__(self, kl_max: float, window: int, ratio: float, min_best: float):
        self.kl_max, self.window, self.ratio, self.min_best = kl_max, window, ratio, min_best
        self.rewards: list[float] = []
        self.best_rolling = -math.inf

    def update(self, metrics: dict[str, Any]) -> str | None:
        for name in ("loss", "kl", "grad_norm", "mean_reward"):
            value = metrics.get(name)
            if value is not None and not math.isfinite(value):
                return f"non-finite {name}"
        kl = metrics.get("kl")
        if kl is not None and kl > self.kl_max:
            return f"KL to the starting policy ({kl:.3g}) exceeded {self.kl_max:g}"

        self.rewards.append(metrics["mean_reward"])
        if len(self.rewards) >= self.window:
            recent = statistics.fmean(self.rewards[-self.window :])
            self.best_rolling = max(self.best_rolling, recent)
            if self.best_rolling >= self.min_best and recent < self.ratio * self.best_rolling:
                return (
                    f"reward collapsed: mean of the last {self.window} steps is {recent:.3g}, "
                    f"down from a best of {self.best_rolling:.3g}"
                )
        return None


# --- the loop --------------------------------------------------------------------------------------------------------


def load_training_tasks(kernels: str | Path) -> list[dict[str, Any]]:
    """Unique task specs of the training set, in dataset order. Never includes held-out tasks."""
    seen, tasks = set(), []
    for rec in read_jsonl(kernels):
        key = json.dumps(rec["task"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            tasks.append(rec["task"])
    return tasks


def _curve_row(step: int, results: list[list[RewardResult]], samples, dead: int, active: int, loss, kl, grad_norm, seconds) -> dict[str, Any]:
    flat = [r for group in results for r in group]
    correct = [r for r in flat if r.correct]
    return {
        "step": step,
        "mean_reward": statistics.fmean(r.reward for r in flat),
        "max_reward": max(r.reward for r in flat),
        "frac_correct": len(correct) / len(flat),
        "frac_faster": sum(r.correct and (r.speedup or 0) > 1.0 for r in flat) / len(flat),
        "mean_speedup_correct": statistics.fmean(r.speedup for r in correct) if correct else None,
        "frac_no_code": sum(r.status == "no_code" for r in flat) / len(flat),
        "frac_rejected": sum(r.status == "rejected" for r in flat) / len(flat),
        "frac_failed": sum(r.status in ("error", "timeout", "crash") for r in flat) / len(flat),
        "dead_groups": dead,
        "active_samples": active,
        "loss": loss,
        "kl": kl,
        "grad_norm": grad_norm,
        "mean_completion_tokens": statistics.fmean(len(s.completion_ids) for g in samples for s in g),
        "frac_unfinished": sum(not s.finished for g in samples for s in g) / sum(len(g) for g in samples),
        "step_seconds": seconds,
    }


def _report_divergence(out: Path, cfg: RLConfig, model, step: int, reason: str, curve: list, recent_samples) -> None:
    """Write the reward curve and the latest generated code, and save the policy as it was."""
    log.error("training diverged at step %d: %s", step, reason)
    model.save_pretrained(out / "diverged")
    report = {
        "reason": reason,
        "step": step,
        "config": asdict(cfg),
        "reward_curve": curve,
        "recent_samples": list(recent_samples),
    }
    (out / "divergence_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def train_rl(cfg: RLConfig, model=None, tokenizer=None, reward_fn: RewardFn | None = None) -> dict[str, Any]:
    """Run GRPO. `model`, `tokenizer`, and `reward_fn` may be injected for tests."""
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=1), encoding="utf-8")

    tokenizer = tokenizer or load_tokenizer(cfg.tokenizer_name or cfg.model_name)
    if model is None:
        base = load_base_model(cfg.model_name, quantize=cfg.quantize)
        if cfg.init_adapter:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.config.use_cache = False  # sample() asks for the cache explicitly when generating
            model = load_adapter(base, cfg.init_adapter, trainable=True)
        else:
            model = add_lora(base, cfg.lora_r, cfg.lora_alpha, dropout=0.0)
    disable_dropout(model)
    model.train()
    if reward_fn is None:
        reward_fn = make_sandbox_reward(
            "cuda", cfg.reward_warmup, cfg.reward_reps, cfg.reward_seeds, cfg.reward_timeout, environment_info("cuda")
        )

    tasks = load_training_tasks(cfg.kernels)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)
    ref_state = lora_state(model) if cfg.beta > 0 else None  # the starting policy
    monitor = DivergenceMonitor(cfg.kl_max, cfg.collapse_window, cfg.collapse_ratio, cfg.collapse_min_best)
    rng = random.Random(cfg.seed)

    curve_path = out / "reward_curve.csv"
    with curve_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CURVE_FIELDS).writeheader()

    curve: list[dict[str, Any]] = []
    recent_samples: deque = deque(maxlen=48)
    order: list[int] = []
    status, reason = "completed", None

    for step in range(1, cfg.steps + 1):
        started = time.monotonic()
        if len(order) < cfg.prompts_per_step:
            order = rng.sample(range(len(tasks)), len(tasks))
        batch = [tasks[order.pop()] for _ in range(cfg.prompts_per_step)]

        # 1. generate a group of kernels per task
        prompts = [build_prompt(t) for t in batch]
        try:
            samples = sample(
                model, tokenizer, prompts, cfg.max_new_tokens, cfg.temperature, cfg.top_p, num_return_sequences=cfg.group_size
            )
        except RuntimeError as exc:
            if "probability tensor" not in str(exc):
                raise
            # The policy's weights, or the logits they produce, are no longer finite.
            status, reason = "diverged", "sampling failed: the policy produces non-finite probabilities"
            _report_divergence(out, cfg, model, step, reason, curve, recent_samples)
            break

        # 2. score every kernel
        results = [[reward_fn(task, s.text) for s in group] for task, group in zip(batch, samples)]
        advantages = [group_advantages([r.reward for r in group]) for group in results]
        items = [
            (s.prompt_ids, s.completion_ids, adv)
            for group, advs in zip(samples, advantages)
            for s, adv in zip(group, advs)
            if adv != 0.0
        ]
        dead = sum(all(a == 0.0 for a in advs) for advs in advantages)

        # 3. one or more policy-gradient updates on the samples that carry a signal
        loss_value = kl_value = grad_norm = None
        if items:
            ref_logps = old_logps = None
            if ref_state is not None:
                with use_lora_state(model, ref_state), torch.no_grad():
                    ref_logps = [token_logprobs(model, p, c) for p, c, _ in items]
            if cfg.num_updates > 1:
                with torch.no_grad():
                    old_logps = [token_logprobs(model, p, c) for p, c, _ in items]
            for _ in range(cfg.num_updates):
                optimizer.zero_grad(set_to_none=True)
                total_loss, total_kl = 0.0, 0.0
                for i, (p, c, adv) in enumerate(items):
                    logp = token_logprobs(model, p, c)
                    old = old_logps[i] if old_logps is not None else logp.detach()
                    loss, kl = grpo_sample_loss(logp, old, ref_logps[i] if ref_logps else None, adv, cfg.clip_eps, cfg.beta)
                    (loss / len(items)).backward()
                    total_loss += float(loss.detach())
                    total_kl += float(kl)
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
                if math.isfinite(total_loss) and math.isfinite(grad_norm):
                    optimizer.step()
                loss_value, kl_value = total_loss / len(items), total_kl / len(items)

        row = _curve_row(step, results, samples, dead, len(items), loss_value, kl_value, grad_norm, time.monotonic() - started)
        curve.append(row)
        with curve_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CURVE_FIELDS).writerow(row)

        records = [
            {
                "step": step, "task": task["params"], "family": task["family"], "group": g,
                "reward": r.reward, "status": r.status, "correct": r.correct, "speedup": r.speedup,
                "error": r.error, "finished": s.finished, "completion_tokens": len(s.completion_ids), "code": r.code,
            }
            for task, group_r, group_s in zip(batch, results, samples)
            for g, (r, s) in enumerate(zip(group_r, group_s))
        ]
        (out / "samples" / f"step_{step}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
        recent_samples.extend(records)

        log.info(
            "step %d/%d reward %.3f (max %.3f) correct %.0f%% faster %.0f%% dead groups %d/%d kl %s loss %s (%.0fs)",
            step, cfg.steps, row["mean_reward"], row["max_reward"], 100 * row["frac_correct"], 100 * row["frac_faster"],
            dead, len(batch), "-" if kl_value is None else f"{kl_value:.4f}", "-" if loss_value is None else f"{loss_value:.4f}",
            row["step_seconds"],
        )

        # 4. stop and report if the run has gone bad
        reason = monitor.update(row)
        if reason:
            status = "diverged"
            _report_divergence(out, cfg, model, step, reason, curve, recent_samples)
            break

        if step % cfg.save_every == 0:
            model.save_pretrained(out / f"step_{step}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if status == "completed":
        model.save_pretrained(out / "final")
    summary = {"status": status, "reason": reason, "steps_run": len(curve), "reward_curve": str(curve_path)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = RLConfig()
    parser.add_argument("--model", dest="model_name", default=d.model_name)
    parser.add_argument("--tokenizer", dest="tokenizer_name", default=None)
    parser.add_argument("--init-adapter", default=None, help="LoRA adapter to start from, e.g. runs/sft/final")
    parser.add_argument("--kernels", default=d.kernels)
    parser.add_argument("--out", dest="out_dir", default=d.out_dir)
    parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.add_argument("--steps", type=int, default=d.steps)
    parser.add_argument("--prompts-per-step", type=int, default=d.prompts_per_step)
    parser.add_argument("--group-size", type=int, default=d.group_size)
    parser.add_argument("--max-new-tokens", type=int, default=d.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=d.temperature)
    parser.add_argument("--lr", type=float, default=d.lr)
    parser.add_argument("--beta", type=float, default=d.beta)
    parser.add_argument("--reward-reps", type=int, default=d.reward_reps)
    parser.add_argument("--reward-timeout", type=float, default=d.reward_timeout)
    parser.add_argument("--save-every", type=int, default=d.save_every)
    parser.add_argument("--kl-max", type=float, default=d.kl_max)
    parser.add_argument("--seed", type=int, default=d.seed)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if not torch.cuda.is_available():
        parser.error("GRPO needs a CUDA GPU to generate and to measure kernels; none found")
    summary = train_rl(RLConfig(**vars(args)))
    print(json.dumps(summary, indent=1))
    return 0 if summary["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
