import pytest

from kernelgen.families import all_families, get_family
from kernelgen.prompts import SYSTEM_PROMPT, build_prompt, extract_code, format_response, to_messages

SRC = "import torch\nimport triton\n\n\ndef solution(x):\n    return x\n"


@pytest.mark.parametrize("fam", all_families(), ids=lambda f: f.name)
def test_prompt_states_the_task_signature_baseline_and_rules(fam):
    params = fam.train_shapes[0]
    prompt = build_prompt(fam.spec(params))
    assert fam.describe(params) in prompt
    assert f"`{fam.signature}`" in prompt
    assert fam.reference.__name__ in prompt  # the baseline's source is shown
    assert "@triton.jit" in prompt and "single ```python code block" in prompt


def test_prompt_depends_on_the_shape_and_is_deterministic():
    fam = get_family("gemm")
    a, b = fam.train_shapes[0], fam.train_shapes[1]
    assert build_prompt(fam.spec(a)) == build_prompt(fam.spec(a))
    assert build_prompt(fam.spec(a)) != build_prompt(fam.spec(b))


def test_messages():
    assert to_messages("hi") == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "hi"},
    ]
    assert to_messages("hi", "yo")[-1] == {"role": "assistant", "content": "yo"}


def test_format_then_extract_round_trips():
    assert extract_code(format_response(SRC)) == SRC


@pytest.mark.parametrize("lang", ["python", "py", "Python", ""])
def test_extract_accepts_python_fences(lang):
    assert extract_code(f"```{lang}\n{SRC}```") == SRC


def test_extract_ignores_surrounding_prose():
    assert extract_code(f"Here you go:\n```python\n{SRC}```\nHope that helps.") == SRC


def test_extract_takes_the_last_block_that_defines_solution():
    text = "```python\ndef helper():\n    pass\n```\n```python\ndef solution(x):\n    return 1\n```\n```python\ndef solution(x):\n    return 2\n```"
    assert extract_code(text) == "def solution(x):\n    return 2\n"


def test_extract_ignores_other_languages():
    assert extract_code("```bash\ndef solution(x): pass\n```") is None


def test_extract_rejects_a_block_cut_off_before_its_closing_fence():
    assert extract_code("```python\ndef solution(x):\n    return x\n") is None


def test_extract_rejects_replies_without_a_solution():
    assert extract_code("```python\nprint('hi')\n```") is None
    assert extract_code("no code at all") is None
    assert extract_code("") is None
