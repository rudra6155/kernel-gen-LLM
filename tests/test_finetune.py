"""Fine-tuning tests: data selection and masking, the training loop, and sampling.

The loop runs on CPU with a tiny Qwen2-architecture model and the real Qwen
tokenizer. This checks the code path (masking, accumulation, logging, saving,
sampling, 4-bit loading and LoRA). It does not check that the 3B model fits in
8 GB of VRAM or that training improves it; that needs the GPU run.
"""

import csv
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from kernelgen import finetune
from kernelgen.annotate import ANNOTATION_FIELDS, to_annotation
from kernelgen.dataset import read_jsonl, write_jsonl
from kernelgen.finetune import (
    SFTConfig,
    build_features,
    collate,
    load_sft_examples,
    lr_at,
    split_by_task,
    train,
)
from kernelgen.modeling import DEFAULT_MODEL, add_lora, load_adapter, load_base_model, load_tokenizer, sample

DATA = Path(__file__).resolve().parent.parent / "data"
RECORDS = read_jsonl(DATA / "kernels.jsonl")
ENV = {"gpu_name": "cpu", "torch_version": "t", "cuda_version": "", "triton_version": ""}


@pytest.fixture(scope="session")
def tok():
    try:
        return load_tokenizer(DEFAULT_MODEL)
    except Exception as exc:  # no network and no cached tokenizer
        pytest.skip(f"cannot load the Qwen tokenizer: {type(exc).__name__}")


def tiny_config(tok):
    from transformers import Qwen2Config

    return Qwen2Config(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
        eos_token_id=[151645, 151643], pad_token_id=151643, tie_word_embeddings=True,
    )


@pytest.fixture(scope="session")
def tiny_dir(tok, tmp_path_factory):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("tiny") / "model"
    AutoModelForCausalLM.from_config(tiny_config(tok)).save_pretrained(path)
    return path


def annotation(record, status="ok", speedup=2.0):
    row = {k: None for k in ANNOTATION_FIELDS}
    if status == "ok":
        row.update(status="ok", correct=1, baseline_ms=2.0 * speedup, candidate_ms=2.0, speedup=speedup, error="")
    else:
        row.update(status=status, correct=0, error="wrong")
    row.update(warmup=1, reps=1, seeds=1, **ENV)
    return to_annotation(record, row)


# --- example selection ------------------------------------------------------------------------------------


def test_only_correct_and_fast_enough_kernels_become_examples(tmp_path):
    r = RECORDS[:6]
    anns = [
        annotation(r[0], speedup=3.0),
        annotation(r[1], speedup=0.8),  # slower than the baseline
        annotation(r[2], status="incorrect"),
        annotation(r[3], status="timeout"),
        dict(annotation(r[4]), source_sha256="0" * 64),  # annotation for an older source
    ]  # r[5] has no annotation
    write_jsonl(anns, tmp_path / "a.jsonl")
    ex = load_sft_examples(DATA / "kernels.jsonl", tmp_path / "a.jsonl", min_speedup=1.0)
    assert [e["id"] for e in ex] == [r[0]["id"]]
    assert ex[0]["speedup"] == 3.0
    assert ex[0]["response"].startswith("```python\n") and ex[0]["response"].endswith("\n```")


def test_min_speedup_is_respected(tmp_path):
    r = RECORDS[:2]
    write_jsonl([annotation(r[0], speedup=1.5), annotation(r[1], speedup=2.5)], tmp_path / "a.jsonl")
    assert len(load_sft_examples(DATA / "kernels.jsonl", tmp_path / "a.jsonl", min_speedup=2.0)) == 1
    assert len(load_sft_examples(DATA / "kernels.jsonl", tmp_path / "a.jsonl", min_speedup=1.0)) == 2


def test_unannotated_kernels_are_used_only_when_asked(tmp_path):
    write_jsonl([annotation(RECORDS[0])], tmp_path / "a.jsonl")
    n = len(load_sft_examples(DATA / "kernels.jsonl", tmp_path / "a.jsonl", include_unannotated=True))
    assert n == len(RECORDS)
    ex = load_sft_examples(DATA / "kernels.jsonl", tmp_path / "a.jsonl", include_unannotated=True)
    assert ex[0]["speedup"] == 2.0 and ex[1]["speedup"] is None


def test_missing_annotations_explain_how_to_get_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="make annotate"):
        load_sft_examples(DATA / "kernels.jsonl", tmp_path / "none.jsonl")


def test_same_task_shares_one_prompt():
    ex = load_sft_examples(DATA / "kernels.jsonl", "none", include_unannotated=True)
    by_task = {}
    for e in ex:
        by_task.setdefault(e["task_name"], set()).add(e["prompt"])
    assert all(len(prompts) == 1 for prompts in by_task.values())


def test_split_holds_out_whole_tasks_deterministically():
    ex = load_sft_examples(DATA / "kernels.jsonl", "none", include_unannotated=True)
    train_ex, val_ex = split_by_task(ex, 0.1, seed=0)
    assert train_ex and val_ex and len(train_ex) + len(val_ex) == len(ex)
    assert {e["task_name"] for e in train_ex}.isdisjoint({e["task_name"] for e in val_ex})
    assert split_by_task(ex, 0.1, seed=0) == (train_ex, val_ex)
    assert split_by_task(ex, 0.1, seed=1) != (train_ex, val_ex)
    assert split_by_task(ex, 0.0, seed=0) == (ex, [])


# --- tokenization ---------------------------------------------------------------------------------------------


def one_per_family():
    seen, out = set(), []
    for r in RECORDS:
        if r["family"] not in seen:
            seen.add(r["family"])
            out.append(r)
    return out


def example_for(record):
    ex = load_sft_examples(DATA / "kernels.jsonl", "none", include_unannotated=True)
    return next(e for e in ex if e["id"] == record["id"])


@pytest.mark.parametrize("record", one_per_family(), ids=lambda r: r["family"])
def test_prompt_is_masked_and_the_reply_is_the_target(tok, record):
    ex = example_for(record)
    feats = build_features(tok, ex, max_len=4096)
    assert len(feats["input_ids"]) == len(feats["labels"])
    target_ids = [t for t, lab in zip(feats["input_ids"], feats["labels"]) if lab != -100]
    assert all(lab == t for lab, t in zip(feats["labels"], feats["input_ids"]) if lab != -100)
    assert tok.decode(target_ids) == ex["response"] + "<|im_end|>\n"
    first_target = next(i for i, lab in enumerate(feats["labels"]) if lab != -100)
    assert all(lab == -100 for lab in feats["labels"][:first_target])
    assert "Write a Triton kernel" in tok.decode(feats["input_ids"][:first_target])


def test_prompt_tokens_are_a_prefix_for_every_5th_record(tok):
    ex = load_sft_examples(DATA / "kernels.jsonl", "none", include_unannotated=True)
    for e in ex[::5]:
        assert build_features(tok, e, max_len=4096) is not None  # raises if the prefix property fails


def test_examples_longer_than_the_window_are_dropped_not_truncated(tok):
    ex = example_for(RECORDS[0])
    assert build_features(tok, ex, max_len=100) is None


def test_every_example_fits_the_default_window(tok):
    ex = load_sft_examples(DATA / "kernels.jsonl", "none", include_unannotated=True)
    lengths = [len(build_features(tok, e, max_len=10**6)["input_ids"]) for e in ex[::6]]
    assert max(lengths) <= SFTConfig().max_seq_len


def test_collate_pads_and_masks():
    a = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    b = {"input_ids": [4, 5], "labels": [-100, 5]}
    batch = collate([a, b], pad_id=9)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 9]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]


def test_learning_rate_schedule():
    lrs = [lr_at(s, 100, 1.0, 10) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)  # linear warmup to the base rate
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:]))  # then decays
    assert lrs[-1] == pytest.approx(0.1, abs=0.01)  # to 10% of the base rate
    assert lr_at(1000, 100, 1.0, 10) == pytest.approx(0.1)  # and stays there


# --- the training loop ------------------------------------------------------------------------------------------------


def small_kernels(tmp_path, n_tasks=5, per_task=2):
    seen, picked = {}, []
    for r in RECORDS:
        if len(seen) == n_tasks and r["task_name"] not in seen:
            continue
        if seen.get(r["task_name"], 0) < per_task:
            seen[r["task_name"]] = seen.get(r["task_name"], 0) + 1
            picked.append(r)
    write_jsonl(picked, tmp_path / "k.jsonl")
    return tmp_path / "k.jsonl", picked


def tiny_cfg(tmp_path, kernels, **kw):
    cfg = dict(
        kernels=str(kernels), annotations=str(tmp_path / "none.jsonl"), include_unannotated=True,
        out_dir=str(tmp_path / "run"), quantize=False, epochs=2, micro_batch_size=1, grad_accum=2,
        lr=5e-3, warmup_steps=1, max_seq_len=2048, lora_r=4, lora_alpha=8, lora_dropout=0.0,
        val_fraction=0.4, n_samples=2, sample_max_new_tokens=6, log_every=1, seed=0,
    )
    cfg.update(kw)
    return SFTConfig(**cfg)


@pytest.fixture
def tiny_model(tiny_dir):
    return add_lora(load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32), r=4, alpha=8, dropout=0.0)


def test_training_loop_logs_samples_and_saves_every_epoch(tok, tiny_model, tmp_path):
    kernels, picked = small_kernels(tmp_path)
    before = {n: p.detach().clone() for n, p in tiny_model.named_parameters() if p.requires_grad}
    result = train(tiny_cfg(tmp_path, kernels), model=tiny_model, tokenizer=tok)
    run = tmp_path / "run"

    # every epoch is logged, with finite losses
    assert [e["epoch"] for e in result["epochs"]] == [1, 2]
    assert all(torch.isfinite(torch.tensor(e["train_loss"])) for e in result["epochs"])
    assert all(e["val_loss"] is not None and e["val_loss"] > 0 for e in result["epochs"])
    rows = list(csv.DictReader((run / "train_log.csv").open()))
    assert rows and {"epoch", "step", "loss", "lr", "grad_norm", "tokens_per_s", "elapsed_s"} <= set(rows[0])
    assert {int(r["epoch"]) for r in rows} == {1, 2}
    assert all(float(r["loss"]) > 0 for r in rows)

    # a few generated samples per epoch, each checked
    for epoch in (1, 2):
        samples = [json.loads(line) for line in (run / "samples" / f"epoch_{epoch}.jsonl").read_text().splitlines()]
        assert len(samples) == 2 and len({s["task_name"] for s in samples}) == 2
        assert {"id", "task_name", "finished", "code_extracted", "parses", "policy_violations", "generated"} <= set(samples[0])
        assert (run / f"epoch_{epoch}" / "adapter_config.json").exists()
    assert (run / "final" / "adapter_config.json").exists() and (run / "final" / "tokenizer_config.json").exists()
    assert json.loads((run / "data_stats.json").read_text())["dropped_too_long"] == 0
    assert json.loads((run / "config.json").read_text())["lora_r"] == 4

    # only the LoRA weights changed
    after = {n: p for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert any(not torch.equal(before[n], after[n]) for n in before)
    assert all("lora_" in n for n in before)


def test_training_reduces_the_loss_on_a_tiny_dataset(tok, tiny_model, tmp_path):
    kernels, _ = small_kernels(tmp_path, n_tasks=2, per_task=2)
    cfg = tiny_cfg(tmp_path, kernels, epochs=6, grad_accum=1, val_fraction=0.0, n_samples=0, lr=1e-2)
    result = train(cfg, model=tiny_model, tokenizer=tok)
    losses = [e["train_loss"] for e in result["epochs"]]
    # A random 32-dimensional model with frozen embeddings can only fit so much through
    # rank-4 adapters (about 11.91 -> 11.82 here), so this checks a steady, clear decrease.
    assert losses[-1] < losses[0] - 0.05, losses
    assert all(b <= a + 0.02 for a, b in zip(losses, losses[1:])), losses


def test_saved_adapter_reproduces_the_trained_model(tok, tiny_model, tiny_dir, tmp_path):
    kernels, _ = small_kernels(tmp_path)
    train(tiny_cfg(tmp_path, kernels, epochs=1), model=tiny_model, tokenizer=tok)
    reloaded = load_adapter(load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32), tmp_path / "run" / "final")
    ids = torch.randint(0, 1000, (1, 24))
    tiny_model.eval()
    reloaded.eval()
    assert torch.allclose(tiny_model(input_ids=ids).logits, reloaded(input_ids=ids).logits, atol=1e-5)


def test_training_needs_at_least_one_example(tok, tiny_model, tmp_path):
    write_jsonl([annotation(RECORDS[0], speedup=0.5)], tmp_path / "a.jsonl")
    cfg = tiny_cfg(tmp_path, DATA / "kernels.jsonl", annotations=str(tmp_path / "a.jsonl"), include_unannotated=False)
    with pytest.raises(ValueError, match="no training examples"):
        train(cfg, model=tiny_model, tokenizer=tok)


def test_training_without_annotations_says_what_to_do(tok, tiny_model, tmp_path):
    cfg = tiny_cfg(tmp_path, DATA / "kernels.jsonl", include_unannotated=False)
    with pytest.raises(FileNotFoundError, match="make annotate"):
        train(cfg, model=tiny_model, tokenizer=tok)


def test_a_non_finite_loss_stops_training(tok, tiny_model, tmp_path, monkeypatch):
    kernels, _ = small_kernels(tmp_path)
    for p in tiny_model.parameters():
        if p.requires_grad:
            p.data.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        train(tiny_cfg(tmp_path, kernels), model=tiny_model, tokenizer=tok)


def test_cli_refuses_4bit_without_a_gpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this checks the no-GPU message")
    with pytest.raises(SystemExit):
        finetune.main(["--out", str(tmp_path / "run")])


# --- QLoRA loading and sampling -----------------------------------------------------------------------------------------


def test_4bit_model_gets_lora_and_trains_one_step(tiny_dir, tok):
    try:
        model = load_base_model(str(tiny_dir), quantize=True, dtype=torch.float32)
    except Exception as exc:  # 4-bit loading depends on the bitsandbytes backend
        pytest.skip(f"4-bit loading unavailable here: {type(exc).__name__}")
    kinds = {type(m).__name__ for m in model.modules()}
    assert "Linear4bit" in kinds
    peft_model = add_lora(model, r=4, alpha=8, dropout=0.0)
    ids = torch.randint(0, 1000, (1, 16))
    loss = peft_model(input_ids=ids, labels=ids).loss
    loss.backward()
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n, _ in trainable)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in trainable)
    assert torch.isfinite(loss)


def test_sample_returns_the_requested_number_per_prompt(tok, tiny_dir):
    model = load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32)
    out = sample(model, tok, ["write a kernel", "hello"], max_new_tokens=5, num_return_sequences=3)
    assert [len(group) for group in out] == [3, 3]
    for group in out:
        for s in group:
            assert 1 <= len(s.completion_ids) <= 5 and s.prompt_ids and isinstance(s.text, str)
            assert s.finished == (s.completion_ids[-1] in (151645, 151643))


def test_greedy_sampling_is_unaffected_by_batching_with_other_prompts(tok, tiny_dir):
    model = load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32)
    short, long_ = "hi", "a much longer prompt " * 12
    alone = sample(model, tok, [short], max_new_tokens=6, greedy=True)[0][0]
    batched = sample(model, tok, [short, long_], max_new_tokens=6, greedy=True)[0][0]
    assert alone.prompt_ids == batched.prompt_ids  # padding is not part of the prompt
    assert alone.completion_ids == batched.completion_ids  # and left padding does not change the output


def test_sample_restores_the_training_mode(tok, tiny_dir):
    model = load_base_model(str(tiny_dir), quantize=False, dtype=torch.float32)
    model.train()
    sample(model, tok, ["x"], max_new_tokens=2, greedy=True)
    assert model.training
    model.eval()
    sample(model, tok, ["x"], max_new_tokens=2, greedy=True)
    assert not model.training
