"""Prompts shown to the model, and extraction of kernel code from its reply."""

from __future__ import annotations

import inspect
import re
from typing import Any

from kernelgen.families import get_family

SYSTEM_PROMPT = "You are an expert GPU kernel engineer. You write correct, fast Triton kernels."

_USER_TEMPLATE = """Write a Triton kernel for this task.

Task: {description}

The baseline to beat is this PyTorch code:
```python
{reference}
```

Requirements:
- Define `{signature}` and return the result tensor. It is called with CUDA tensors of the shapes and dtypes above.
- Do the computation in a `@triton.jit` kernel that `solution` launches. Do not call torch.nn, torch.matmul, or other PyTorch compute ops; use PyTorch only to allocate outputs and read shapes and strides.
- Use only these imports: torch, triton, triton.language (as tl), math.
- Reply with a single ```python code block and nothing else."""

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)


def build_prompt(spec: dict[str, Any]) -> str:
    """The user message for the task described by `spec` (family, params, dtype)."""
    fam = get_family(spec["family"])
    return _USER_TEMPLATE.format(
        description=fam.describe(spec["params"]),
        reference=inspect.getsource(fam.reference).rstrip(),
        signature=fam.signature,
    )


def to_messages(prompt: str, response: str | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def format_response(source: str) -> str:
    """The assistant reply that a kernel source corresponds to."""
    return "```python\n" + source.rstrip("\n") + "\n```"


def extract_code(text: str) -> str | None:
    """The kernel source in a model reply, or None.

    Takes the last complete python code block that defines `solution`. A block
    that was cut off before its closing fence (the reply hit the token limit)
    does not count, because truncated code is not a candidate.
    """
    blocks = [
        body
        for lang, body in _FENCE.findall(text)
        if lang.lower() in ("", "python", "py") and "def solution" in body
    ]
    return blocks[-1].strip("\n") + "\n" if blocks else None
