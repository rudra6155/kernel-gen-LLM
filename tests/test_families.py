"""Task families: shape splits, template rendering, and template correctness.

Template correctness runs under Triton's CPU interpreter, which checks kernel
logic (indexing, masking, reductions, epilogues) against the PyTorch baseline.
It does not check GPU compilation, tuning parameters, or speed.
"""

import random

import pytest
import torch

from kernelgen.candidate import load_candidate
from kernelgen.correctness import check_correctness
from kernelgen.families import FAMILY_ORDER, all_families, get_family
from kernelgen.policy import check_source
from kernelgen.tasks import task_from_spec

FAMILIES = [f.name for f in all_families()]

# Small shapes with tiles larger than the problem, so masking is exercised.
SMALL = {
    "layernorm": [{"M": 6, "N": 200}, {"M": 4, "N": 1000}],
    "rmsnorm": [{"M": 6, "N": 200}, {"M": 4, "N": 1000}],
    "softmax": [{"M": 6, "N": 200}, {"M": 4, "N": 1000}],
    "gemm": [{"M": 40, "N": 50, "K": 70}, {"M": 64, "N": 64, "K": 64}],
    "gemm_bias_relu": [{"M": 40, "N": 50, "K": 70}, {"M": 64, "N": 64, "K": 64}],
    "attention": [{"B": 1, "H": 2, "S": 40, "D": 16}, {"B": 1, "H": 1, "S": 70, "D": 32}],
}


def key(params):
    return tuple(sorted(params.items()))


# --- registry and splits ------------------------------------------------------------------------


def test_registry_has_the_expected_families_in_order():
    assert FAMILY_ORDER == ["layernorm", "rmsnorm", "softmax", "gemm", "gemm_bias_relu", "attention"]
    assert [f.name for f in all_families()] == FAMILY_ORDER


def test_unknown_family_is_an_error():
    with pytest.raises(ValueError, match="unknown task family"):
        get_family("conv2d")


def test_there_are_exactly_50_heldout_tasks():
    assert sum(len(f.heldout_shapes) for f in all_families()) == 50


@pytest.mark.parametrize("name", FAMILIES)
def test_train_and_heldout_shapes_are_disjoint_and_unique(name):
    fam = get_family(name)
    train = [key(p) for p in fam.train_shapes]
    heldout = [key(p) for p in fam.heldout_shapes]
    assert len(set(train)) == len(train) and len(set(heldout)) == len(heldout)
    assert not set(train) & set(heldout)
    assert train and heldout


@pytest.mark.parametrize("name", FAMILIES)
def test_task_spec_round_trip_for_every_shape(name):
    fam = get_family(name)
    for params in fam.train_shapes[:3] + fam.heldout_shapes[:3]:
        task = fam.task(params)
        rebuilt = task_from_spec(fam.spec(params))
        assert rebuilt.name == task.name and rebuilt.bytes_moved == task.bytes_moved
        assert task.family == name and task.params == params


def test_task_names_are_unique_across_all_shapes():
    names = [f.task(p).name for f in all_families() for p in f.train_shapes + f.heldout_shapes]
    assert len(names) == len(set(names))


# --- rendering ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_every_config_of_a_shape_renders_to_policy_clean_unique_source(name):
    fam = get_family(name)
    params = fam.train_shapes[0]
    cfgs = fam.configs(params)
    assert cfgs
    sources = [fam.render(params, c) for c in cfgs]
    assert len(set(sources)) == len(sources), "two configs rendered identical source"
    for cfg, src in zip(cfgs, sources):
        assert check_source(src) == [], (cfg, check_source(src))
        assert "__" not in src.replace("__init__", ""), "unfilled placeholder"


@pytest.mark.parametrize("name", FAMILIES)
def test_default_config_renders_for_every_shape(name):
    fam = get_family(name)
    for params in fam.train_shapes + fam.heldout_shapes:
        assert check_source(fam.render(params, fam.default_config(params))) == []


def test_gemm_configs_respect_the_shared_memory_estimate_and_warp_tile_rule():
    from kernelgen.families.util import SMEM_BUDGET_BYTES

    for c in get_family("gemm").configs({"M": 512, "N": 512, "K": 512}):
        assert (c["block_m"] * c["block_k"] + c["block_k"] * c["block_n"]) * 2 * c["num_stages"] <= SMEM_BUDGET_BYTES
        assert c["block_m"] * c["block_n"] >= 128 * c["num_warps"]


def test_norm_configs_cover_the_row_and_have_enough_elements_per_warp():
    for name in ("layernorm", "rmsnorm", "softmax"):
        fam = get_family(name)
        for n in (200, 1000, 4096, 8192):
            for c in fam.configs({"M": 4, "N": n}):
                if c["variant"] == "single":
                    assert c["block_n"] >= n
                assert c["block_n"] >= 32 * c["num_warps"]


def test_fill_rejects_unfilled_placeholders():
    from kernelgen.families.util import fill

    with pytest.raises(ValueError, match="unfilled"):
        fill("x = __A__ + __B__", A=1)


# --- tasks ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", FAMILIES)
def test_inputs_are_deterministic_and_seeded(name):
    fam = get_family(name)
    task = fam.task(SMALL[name][0])
    a, b, c = task.make_inputs(5, "cpu"), task.make_inputs(5, "cpu"), task.make_inputs(6, "cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))
    assert all(x.dtype == getattr(torch, fam.dtype) for x in a)


@pytest.mark.parametrize("name", FAMILIES)
def test_baseline_and_truth_produce_the_declared_output(name):
    fam = get_family(name)
    task = fam.task(SMALL[name][0])
    inputs = task.make_inputs(0, "cpu")
    ref = task.reference(*inputs)
    assert ref.dtype == task.dtype
    if task.truth is not None:
        truth = task.truth(*inputs)
        assert truth.shape == ref.shape and truth.dtype == ref.dtype
    if task.library is not None:
        assert task.library(*inputs).shape == ref.shape


@pytest.mark.parametrize("name", FAMILIES)
def test_library_op_passes_the_correctness_gate_where_there_is_one(name):
    task = get_family(name).task(SMALL[name][0])
    if task.library is None:
        pytest.skip(f"{name} has no library op: the baseline is the library")
    assert check_correctness(task, lambda *a: task.library(*a), "cpu", seeds=(0, 1)).correct


def test_bytes_moved_formulas():
    assert get_family("layernorm").task({"M": 4, "N": 8}).bytes_moved == (2 * 4 * 8 + 2 * 8) * 4
    assert get_family("rmsnorm").task({"M": 4, "N": 8}).bytes_moved == (2 * 4 * 8 + 8) * 4
    assert get_family("softmax").task({"M": 4, "N": 8}).bytes_moved == 2 * 4 * 8 * 4
    assert get_family("gemm").task({"M": 4, "N": 8, "K": 2}).bytes_moved == (4 * 2 + 2 * 8 + 4 * 8) * 2
    assert get_family("gemm_bias_relu").task({"M": 4, "N": 8, "K": 2}).bytes_moved == (4 * 2 + 2 * 8 + 4 * 8 + 8) * 2
    assert get_family("attention").task({"B": 1, "H": 2, "S": 3, "D": 4}).bytes_moved == 4 * 1 * 2 * 3 * 4 * 2


# --- template correctness: Triton interpreter without a GPU, real kernels with one --------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cases():
    rng = random.Random(0)
    out = []
    for name in FAMILIES:
        fam = get_family(name)
        for params in SMALL[name]:
            picks = [fam.default_config(params)] + rng.sample(fam.configs(params), 2)
            out += [pytest.param(name, params, cfg, id=f"{name}-{i}-{j}") for j, cfg in enumerate(picks) for i in [SMALL[name].index(params)]]
    return out


@pytest.mark.parametrize("name, params, config", cases())
def test_template_is_correct_and_passes_the_native_op_guard(name, params, config):
    pytest.importorskip("triton")
    fam = get_family(name)
    task = fam.task(params)
    with load_candidate(fam.render(params, config)) as fn:
        result = check_correctness(task, fn, DEVICE, seeds=(0, 1), forbid_native_ops=True)
    assert result.correct, result.reason
