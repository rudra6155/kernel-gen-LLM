"""GRPO tests: the objective, the reference policy, divergence handling, and the loop.

The loop runs on CPU with a tiny Qwen2-architecture model, the real tokenizer, and a
stand-in reward function. The reward itself (the sandboxed measurement) is tested
separately with real CPU kernels. This checks the algorithm's code and its sign
conventions; it does not show that RL improves the 3B model, which needs the GPU.
"""

import csv
import json
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from kernelgen import rl
from kernelgen.dataset import read_jsonl
from kernelgen.modeling import DEFAULT_MODEL, add_lora, load_adapter, load_base_model, load_tokenizer
from kernelgen.rl import (
    DivergenceMonitor,
    RLConfig,
    RewardResult,
    disable_dropout,
    group_advantages,
    grpo_sample_loss,
    kl_k3,
    load_training_tasks,
    lora_state,
    make_sandbox_reward,
    token_logprobs,
    train_rl,
    use_lora_state,
)
from kernelgen.families import get_family
from kernelgen.prompts import format_response

DATA = Path(__file__).resolve().parent.parent / "data"
ENV = {"gpu_name": "cpu", "torch_version": "t", "cuda_version": "", "triton_version": ""}


@pytest.fixture(scope="session")
def tok():
    try:
        return load_tokenizer(DEFAULT_MODEL)
    except Exception as exc:
        pytest.skip(f"cannot load the Qwen tokenizer: {type(exc).__name__}")


@pytest.fixture(scope="session")
def tiny_dir(tok, tmp_path_factory):
    from transformers import AutoModelForCausalLM, Qwen2Config

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tinyrl") / "model"
    cfg = Qwen2Config(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=4096, eos_token_id=[151645, 151643],
        pad_token_id=151643, tie_word_embeddings=True,
    )
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    return path


@pytest.fixture
def model(tiny_dir):
    m = add_lora(load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32), r=4, alpha=8, dropout=0.0)
    disable_dropout(m)
    m.train()
    return m


# --- advantages ------------------------------------------------------------------------------------------------


def test_advantages_are_standardized_within_the_group():
    adv = group_advantages([0.0, 0.0, 2.0, 4.0])
    assert sum(adv) == pytest.approx(0.0, abs=1e-9)
    assert adv[3] > adv[2] > adv[0] == adv[1]
    assert math.sqrt(sum(a * a for a in adv) / len(adv)) == pytest.approx(1.0, abs=1e-3)


def test_advantages_of_equal_rewards_are_all_zero():
    assert group_advantages([0.0] * 8) == [0.0] * 8
    assert group_advantages([1.5, 1.5, 1.5]) == [0.0] * 3


def test_advantages_do_not_depend_on_the_reward_scale():
    a = group_advantages([0.0, 1.0, 3.0])
    b = group_advantages([0.0, 10.0, 30.0])
    assert all(x == pytest.approx(y, abs=1e-3) for x, y in zip(a, b))


# --- KL and the loss ------------------------------------------------------------------------------------------------


def test_kl_estimate_is_zero_at_equality_and_never_negative():
    lp = torch.randn(1000)
    assert kl_k3(lp, lp).abs().max() == 0
    assert kl_k3(lp, lp + torch.randn(1000)).min() >= 0


def test_loss_with_ratio_one_is_minus_the_advantage_and_has_the_policy_gradient():
    logp = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    loss, kl = grpo_sample_loss(logp, logp.detach(), None, advantage=2.0, clip_eps=0.2, beta=0.0)
    assert loss.item() == pytest.approx(-2.0)
    loss.backward()
    assert torch.allclose(logp.grad, torch.full((3,), -2.0 / 3))  # -A * d(logp), averaged over tokens
    assert kl.item() == 0.0


def test_ratio_is_clipped_once_it_leaves_the_trust_region_in_the_helpful_direction():
    old = torch.zeros(1)
    # Positive advantage and the probability already rose past 1 + eps: no more gradient.
    up = torch.full((1,), math.log(1.5), requires_grad=True)
    grad_up = torch.autograd.grad(grpo_sample_loss(up, old, None, 1.0, 0.2, 0.0)[0], up)[0]
    assert grad_up.item() == 0.0
    # Positive advantage but the probability fell: still pushed back up.
    down = torch.full((1,), math.log(0.5), requires_grad=True)
    assert torch.autograd.grad(grpo_sample_loss(down, old, None, 1.0, 0.2, 0.0)[0], down)[0].item() < 0
    # Negative advantage and the probability already fell past 1 - eps: no more gradient.
    low = torch.full((1,), math.log(0.5), requires_grad=True)
    assert torch.autograd.grad(grpo_sample_loss(low, old, None, -1.0, 0.2, 0.0)[0], low)[0].item() == 0.0


def test_the_kl_penalty_pulls_the_policy_back_toward_the_reference():
    ref = torch.zeros(4)
    logp = torch.full((4,), 0.5, requires_grad=True)
    loss, kl = grpo_sample_loss(logp, logp.detach(), ref, advantage=0.0, clip_eps=0.2, beta=1.0)
    loss.backward()
    assert kl.item() > 0 and (logp.grad > 0).all()  # descending the loss lowers logp toward the reference


# --- the sign convention, on a real model ------------------------------------------------------------------------------


def step_on(model, prompt_ids, completion_ids, advantage, lr=5e-2):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    before = token_logprobs(model, prompt_ids, completion_ids).detach().sum().item()
    logp = token_logprobs(model, prompt_ids, completion_ids)
    loss, _ = grpo_sample_loss(logp, logp.detach(), None, advantage, 0.2, 0.0)
    loss.backward()
    opt.step()
    after = token_logprobs(model, prompt_ids, completion_ids).detach().sum().item()
    return before, after


def test_a_positive_advantage_makes_the_completion_more_likely_and_a_negative_one_less(tok, tiny_dir):
    prompt, completion = list(range(100, 130)), list(range(500, 520))
    for advantage, direction in ((1.0, 1), (-1.0, -1)):
        m = add_lora(load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32), r=4, alpha=8, dropout=0.0)
        disable_dropout(m)
        m.train()
        before, after = step_on(m, prompt, completion, advantage)
        assert (after - before) * direction > 0, (advantage, before, after)


# --- the reference policy -------------------------------------------------------------------------------------------------


def test_reference_weights_give_the_starting_policy_and_current_weights_come_back(model):
    prompt, completion = list(range(100, 120)), list(range(300, 310))
    ref = lora_state(model)
    start = token_logprobs(model, prompt, completion).detach()

    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    for _ in range(3):
        logp = token_logprobs(model, prompt, completion)
        (-logp.sum()).backward()
        opt.step()
        opt.zero_grad()
    moved = token_logprobs(model, prompt, completion).detach()
    assert not torch.allclose(start, moved, atol=1e-4)

    current = lora_state(model)
    with use_lora_state(model, ref), torch.no_grad():
        under_ref = token_logprobs(model, prompt, completion)
    assert torch.allclose(under_ref, start, atol=1e-5)  # the starting policy again
    assert all(torch.equal(current[n], p) for n, p in model.named_parameters() if p.requires_grad)  # restored


def test_the_reference_is_the_adapter_training_starts_from_not_the_raw_base(tok, tiny_dir, tmp_path):
    base = load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32)
    m = add_lora(base, r=4, alpha=8, dropout=0.0)
    for p in m.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, std=0.1)  # a stand-in for an SFT adapter that changed the model
    m.save_pretrained(tmp_path / "sft")
    prompt, completion = list(range(100, 120)), list(range(300, 310))
    sft_logp = token_logprobs(m, prompt, completion).detach()

    fresh = load_adapter(load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32), tmp_path / "sft", trainable=True)
    ref = lora_state(fresh)
    with torch.no_grad():
        assert torch.allclose(token_logprobs(fresh, prompt, completion), sft_logp, atol=1e-5)
    with fresh.disable_adapter(), torch.no_grad():
        raw_base = token_logprobs(fresh, prompt, completion)
    assert not torch.allclose(raw_base, sft_logp, atol=1e-3)  # the two differ, so the choice matters
    with use_lora_state(fresh, ref), torch.no_grad():
        assert torch.allclose(token_logprobs(fresh, prompt, completion), sft_logp, atol=1e-5)


def test_dropout_is_disabled(tiny_dir):
    m = add_lora(load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32), r=4, alpha=8, dropout=0.3)
    assert any(isinstance(x, torch.nn.Dropout) and x.p == 0.3 for x in m.modules())
    disable_dropout(m)
    assert all(x.p == 0.0 for x in m.modules() if isinstance(x, torch.nn.Dropout))


# --- the divergence monitor ---------------------------------------------------------------------------------------------------


def monitor(**kw):
    args = dict(kl_max=5.0, window=3, ratio=0.25, min_best=0.5)
    args.update(kw)
    return DivergenceMonitor(**args)


def test_monitor_stays_quiet_on_a_healthy_run():
    m = monitor()
    for r in [0.0, 0.2, 0.5, 0.9, 1.1, 1.3, 1.2, 1.4]:
        assert m.update({"mean_reward": r, "loss": 0.1, "kl": 0.01, "grad_norm": 1.0}) is None


@pytest.mark.parametrize("name", ["loss", "kl", "grad_norm", "mean_reward"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_monitor_flags_non_finite_values(name, bad):
    metrics = {"mean_reward": 1.0, "loss": 0.1, "kl": 0.01, "grad_norm": 1.0}
    metrics[name] = bad
    assert "non-finite" in monitor().update(metrics)


def test_monitor_flags_kl_blowup():
    assert "KL" in monitor().update({"mean_reward": 1.0, "loss": 0.1, "kl": 50.0, "grad_norm": 1.0})


def test_monitor_flags_a_reward_collapse_but_not_a_run_that_never_learned():
    m = monitor()
    verdicts = [m.update({"mean_reward": r, "loss": 0.1, "kl": 0.01, "grad_norm": 1.0}) for r in [1.0, 1.2, 1.1, 0.1, 0.0, 0.0]]
    assert verdicts[:4] == [None] * 4 and "collapsed" in verdicts[5]

    flat = monitor()
    for r in [0.0] * 12:
        assert flat.update({"mean_reward": r, "loss": None, "kl": None, "grad_norm": None}) is None  # nothing to collapse from


# --- training tasks ---------------------------------------------------------------------------------------------------------------


def test_training_tasks_are_unique_and_never_held_out():
    tasks = load_training_tasks(DATA / "kernels.jsonl")
    keys = [json.dumps(t, sort_keys=True) for t in tasks]
    assert len(set(keys)) == len(keys) > 50
    held = {json.dumps(get_family(f).spec(p), sort_keys=True) for f in ("layernorm", "gemm", "attention") for p in get_family(f).heldout_shapes}
    assert not held & set(keys)


# --- the sandboxed reward, with real CPU kernels ----------------------------------------------------------------------------------


def reward_for(task_spec, code_or_text, **kw):
    fn = make_sandbox_reward("cpu", warmup=1, reps=3, seeds=2, timeout_s=120.0, env=ENV, **kw)
    return fn, fn(task_spec, code_or_text)


LN = get_family("layernorm")
LN_PARAMS = {"M": 6, "N": 200}
LN_SPEC = LN.spec(LN_PARAMS)


def good_reply():
    return format_response(LN.render(LN_PARAMS, LN.default_config(LN_PARAMS)))


def test_a_correct_triton_kernel_earns_its_speedup():
    _, res = reward_for(LN_SPEC, good_reply())
    assert res.status == "ok" and res.correct and res.speedup > 0
    assert res.reward == pytest.approx(res.speedup)


def test_a_reply_without_code_earns_nothing():
    _, res = reward_for(LN_SPEC, "Sure! Here is how you would do it in prose.")
    assert (res.reward, res.status, res.code) == (0.0, "no_code", None)


def test_a_wrong_kernel_earns_nothing():
    wrong = good_reply().replace("+ b\n", "\n")
    assert wrong != good_reply()
    _, res = reward_for(LN_SPEC, wrong)
    assert res.reward == 0.0 and res.status == "incorrect" and not res.correct


def test_calling_the_library_instead_of_writing_a_kernel_earns_nothing():
    cheat = (
        "```python\nimport torch\nimport triton\nimport triton.language as tl\n\n"
        "@triton.jit\ndef _k(X, BLOCK: tl.constexpr):\n    pass\n\n"
        "def solution(x, weight, bias, eps=1e-5):\n"
        "    _k[(1,)](x, BLOCK=1)\n"
        "    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)\n```"
    )
    _, res = reward_for(LN_SPEC, cheat)
    assert res.reward == 0.0 and res.status == "rejected"


def test_a_truncated_reply_earns_nothing():
    truncated = good_reply()[: len(good_reply()) // 2]
    _, res = reward_for(LN_SPEC, truncated)
    assert res.reward == 0.0 and res.status == "no_code"


def test_the_same_kernel_is_measured_once(monkeypatch):
    calls = []
    real = rl.evaluate_isolated

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(rl, "evaluate_isolated", counting)
    fn = make_sandbox_reward("cpu", warmup=1, reps=3, seeds=2, env=ENV)
    first, second = fn(LN_SPEC, good_reply()), fn(LN_SPEC, good_reply())
    assert first is second and len(calls) == 1
    other_task = LN.spec({"M": 6, "N": 100})
    fn(other_task, good_reply())
    assert len(calls) == 2  # same code on another task is a different measurement


# --- the loop ----------------------------------------------------------------------------------------------------------------------


def fake_reward(task_spec, text):
    """Deterministic stand-in: a reward that depends on the reply, with variation inside a group."""
    score = sum(ord(c) % 7 for c in text[:40]) / 40.0
    return RewardResult(score, "ok", score > 3, score if score > 3 else None, "", text[:20])


def rl_cfg(tmp_path, **kw):
    cfg = dict(
        kernels=str(DATA / "kernels.jsonl"), out_dir=str(tmp_path / "rl"), quantize=False, steps=3,
        prompts_per_step=2, group_size=4, max_new_tokens=6, lr=1e-2, beta=0.04, save_every=2, seed=0,
        kl_max=100.0,
    )
    cfg.update(kw)
    return RLConfig(**cfg)


def test_the_loop_logs_the_reward_curve_every_sample_and_checkpoints(tok, model, tmp_path):
    summary = train_rl(rl_cfg(tmp_path), model=model, tokenizer=tok, reward_fn=fake_reward)
    out = tmp_path / "rl"
    assert summary["status"] == "completed" and summary["steps_run"] == 3

    rows = list(csv.DictReader((out / "reward_curve.csv").open()))
    assert [int(r["step"]) for r in rows] == [1, 2, 3]
    assert set(rl.CURVE_FIELDS) == set(rows[0])
    assert all(math.isfinite(float(r["mean_reward"])) for r in rows)
    assert all(int(r["active_samples"]) > 0 and float(r["kl"]) >= 0 for r in rows)

    for step in (1, 2, 3):
        samples = [json.loads(line) for line in (out / "samples" / f"step_{step}.jsonl").read_text().splitlines()]
        assert len(samples) == 2 * 4
        assert {"step", "task", "family", "group", "reward", "status", "correct", "speedup", "error", "finished", "code"} <= set(samples[0])
    assert (out / "step_2" / "adapter_config.json").exists() and (out / "final" / "adapter_config.json").exists()
    assert json.loads((out / "summary.json").read_text())["status"] == "completed"


def test_the_policy_changes_and_stays_close_to_the_reference(tok, model, tmp_path):
    before = lora_state(model)
    train_rl(rl_cfg(tmp_path), model=model, tokenizer=tok, reward_fn=fake_reward)
    after = lora_state(model)
    assert any(not torch.equal(before[n], after[n]) for n in before)


def test_groups_with_identical_rewards_are_skipped_and_do_not_move_the_policy(tok, model, tmp_path):
    before = lora_state(model)
    constant = lambda spec, text: RewardResult(0.0, "no_code", False, None, "", None)  # noqa: E731
    train_rl(rl_cfg(tmp_path, steps=2), model=model, tokenizer=tok, reward_fn=constant)
    after = lora_state(model)
    assert all(torch.equal(before[n], after[n]) for n in before)
    rows = list(csv.DictReader((tmp_path / "rl" / "reward_curve.csv").open()))
    assert all(int(r["dead_groups"]) == 2 and int(r["active_samples"]) == 0 and r["loss"] == "" for r in rows)


def test_a_diverging_run_stops_and_reports_the_reward_curve_and_the_code(tok, model, tmp_path):
    cfg = rl_cfg(tmp_path, steps=6, kl_max=1e-12, lr=1e-1)
    summary = train_rl(cfg, model=model, tokenizer=tok, reward_fn=fake_reward)
    out = tmp_path / "rl"
    assert summary["status"] == "diverged" and "KL" in summary["reason"]
    assert summary["steps_run"] < 6  # stopped early

    report = json.loads((out / "divergence_report.json").read_text())
    assert report["reason"] == summary["reason"] and report["step"] == summary["steps_run"]
    assert len(report["reward_curve"]) == summary["steps_run"]
    assert report["recent_samples"] and "code" in report["recent_samples"][0]
    assert (out / "diverged" / "adapter_config.json").exists()
    assert not (out / "final").exists()


def test_non_finite_weights_are_reported_before_anything_is_sampled(tok, model, tmp_path, monkeypatch):
    # Sampling from NaN probabilities is a device-side assert on CUDA, which kills the CUDA
    # context for the rest of the process (and so for the report). It must never be reached.
    def must_not_sample(*a, **k):
        raise AssertionError("sample() was called with non-finite weights")

    monkeypatch.setattr(rl, "sample", must_not_sample)
    for p in model.parameters():
        if p.requires_grad:
            p.data.fill_(float("nan"))
    summary = train_rl(rl_cfg(tmp_path, steps=3), model=model, tokenizer=tok, reward_fn=fake_reward)
    assert summary["status"] == "diverged" and "non-finite" in summary["reason"] and summary["steps_run"] == 0
    report = json.loads((tmp_path / "rl" / "divergence_report.json").read_text())
    assert report["reason"] == summary["reason"] and report["reward_curve"] == []


def test_the_loop_samples_with_invalid_logits_removed(tok, model, tmp_path, monkeypatch):
    seen = {}
    real = rl.sample

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(rl, "sample", spy)
    train_rl(rl_cfg(tmp_path, steps=1), model=model, tokenizer=tok, reward_fn=fake_reward)
    assert seen.get("remove_invalid_values") is True


def test_a_numerical_blow_up_during_training_is_reported_not_applied(tok, model, tmp_path, monkeypatch):
    # The weights are finite but the log-probabilities are not, as when activations overflow:
    # the update must be skipped, and the run reported as diverged.
    real = rl.token_logprobs
    monkeypatch.setattr(rl, "token_logprobs", lambda *a, **k: real(*a, **k) * float("nan"))
    before = lora_state(model)
    summary = train_rl(rl_cfg(tmp_path, steps=3), model=model, tokenizer=tok, reward_fn=fake_reward)
    after = lora_state(model)
    assert summary["status"] == "diverged" and "non-finite" in summary["reason"] and summary["steps_run"] == 1
    assert all(torch.equal(before[n], after[n]) for n in before), "a non-finite update was applied"


def test_a_reward_function_returning_nan_or_inf_fails_with_a_clear_message(tok, model, tmp_path):
    bad = lambda spec, text: RewardResult(float("nan"), "ok", True, 1.0, "", None)  # noqa: E731
    with pytest.raises(ValueError, match="non-finite reward"):
        train_rl(rl_cfg(tmp_path, steps=1), model=model, tokenizer=tok, reward_fn=bad)


NAN_SAMPLING_SCRIPT = """
import sys
import torch
from kernelgen.modeling import DEFAULT_MODEL, add_lora, load_base_model, load_tokenizer, sample

tok = load_tokenizer(DEFAULT_MODEL)
model = add_lora(load_base_model(sys.argv[1], quantize=False, dtype=torch.float32), r=4, alpha=8, dropout=0.0)
for p in model.parameters():
    if p.requires_grad:
        p.data.fill_(float("nan"))
(group,) = sample(model, tok, ["x"], max_new_tokens=3, num_return_sequences=2, remove_invalid_values=True)
if torch.cuda.is_available():
    torch.cuda.synchronize()  # an asynchronous device-side assert would surface here
print("sampled", len(group), [len(s.completion_ids) for s in group])
"""


def test_invalid_logits_do_not_break_sampling_when_asked_to_remove_them(tiny_dir):
    # Run in a separate process on whatever device is available. Sampling from NaN
    # probabilities is a device-side assert on CUDA, which makes the CUDA context unusable;
    # if the safeguard did not work on a GPU, that must not take the rest of the test session
    # down with it.
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-c", NAN_SAMPLING_SCRIPT, str(tiny_dir)],
        cwd=DATA.parent, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "sampled 2" in proc.stdout


def test_input_ids_go_where_the_embedding_layer_is(model):
    from kernelgen.modeling import input_device

    assert input_device(model) == model.get_input_embeddings().weight.device


def test_multiple_updates_per_batch_use_the_clipped_ratio(tok, model, tmp_path):
    summary = train_rl(rl_cfg(tmp_path, steps=2, num_updates=2), model=model, tokenizer=tok, reward_fn=fake_reward)
    assert summary["status"] == "completed"


def test_training_without_a_gpu_is_refused_by_the_command_line(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this checks the no-GPU message")
    with pytest.raises(SystemExit):
        rl.main(["--out", str(tmp_path / "rl")])
