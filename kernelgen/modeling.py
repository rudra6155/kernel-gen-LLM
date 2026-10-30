"""Model loading, LoRA wrapping, and sampling, shared by fine-tuning, RL, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from kernelgen.prompts import to_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_tokenizer(name: str = DEFAULT_MODEL):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _from_pretrained(name: str, **kwargs):
    """from_pretrained across transformers versions that spell the dtype argument differently."""
    from transformers import AutoModelForCausalLM

    dtype = kwargs.pop("dtype", None)
    if dtype is None:
        return AutoModelForCausalLM.from_pretrained(name, **kwargs)
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kwargs)
    except (TypeError, ValueError):
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, **kwargs)


def load_base_model(name: str = DEFAULT_MODEL, quantize: bool = True, dtype: torch.dtype = torch.bfloat16):
    """The base model, 4-bit NF4 quantized (QLoRA) if `quantize`, placed on GPU 0 when there is one."""
    kwargs: dict[str, Any] = {"dtype": dtype}
    if quantize:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    if torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}
    return _from_pretrained(name, **kwargs)


def add_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05, gradient_checkpointing: bool = True):
    """Wrap `model` with a trainable LoRA adapter on every attention and MLP projection."""
    from peft import LoraConfig, get_peft_model

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False  # sample() asks for the cache explicitly when generating
    config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def load_adapter(model, adapter_dir: str | Path, trainable: bool = False):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=trainable)


def input_device(model) -> torch.device:
    """Where a model's input ids must live: the device of its embedding layer."""
    return model.get_input_embeddings().weight.device


@dataclass
class Sample:
    prompt_ids: list[int]
    completion_ids: list[int]  # up to and including the first end-of-turn token, if any
    text: str
    finished: bool  # False if generation stopped at max_new_tokens without an end token


def _eos_ids(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    for value in ([eos] if isinstance(eos, int) else list(eos or [])):
        ids.add(int(value))
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    return ids


@torch.no_grad()
def sample(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    greedy: bool = False,
    remove_invalid_values: bool = False,
) -> list[list[Sample]]:
    """Chat-format `prompts`, generate, and return `num_return_sequences` samples per prompt.

    With `remove_invalid_values`, NaN and Inf in the logits are replaced before sampling. Without
    it, sampling from NaN probabilities raises on CPU but is a device-side assert on CUDA, which
    leaves the CUDA context unusable. A training loop wants the former behaviour for its own
    sake: it detects the blow-up from the loss and reports it, which needs a live context.
    """
    texts = [
        tokenizer.apply_chat_template(to_messages(p), tokenize=False, add_generation_prompt=True) for p in prompts
    ]
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tokenizer.padding_side = old_side
    batch = {k: v.to(input_device(model)) for k, v in batch.items()}

    was_training = model.training
    model.eval()
    try:
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, use_cache=True,
        )
        if remove_invalid_values:
            gen_kwargs["remove_invalid_values"] = True
        if greedy:
            gen_kwargs.update(do_sample=False, num_return_sequences=1)
        else:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=num_return_sequences)
        out = model.generate(**batch, **gen_kwargs)
    finally:
        model.train(was_training)

    n = 1 if greedy else num_return_sequences
    in_len = batch["input_ids"].shape[1]
    eos = _eos_ids(model, tokenizer)
    results: list[list[Sample]] = [[] for _ in prompts]
    for j in range(out.shape[0]):
        i = j // n
        prompt_ids = batch["input_ids"][i][batch["attention_mask"][i].bool()].tolist()
        completion: list[int] = []
        finished = False
        for tok in out[j, in_len:].tolist():
            completion.append(tok)
            if tok in eos:
                finished = True
                break
        text = tokenizer.decode(completion, skip_special_tokens=True)
        results[i].append(Sample(prompt_ids, completion, text, finished))
    return results
