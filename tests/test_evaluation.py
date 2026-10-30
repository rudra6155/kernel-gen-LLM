"""The 50-task benchmark, model evaluation, slow-kernel diagnosis, and the Nsight runner.

Everything here runs without a GPU, with stand-ins for the model's replies. What it
cannot show is how fast any kernel is on the RTX 4060, or what Nsight does under WSL2.
"""

import csv
import json
import shutil
from pathlib import Path

import pytest
import torch

from kernelgen import benchmark, nsight, run_once
from kernelgen.benchmark import reference_candidates
from kernelgen.diagnose import describe_slowdown
from kernelgen.evaluate_model import EvalConfig, evaluate_task_replies, load_heldout, run_evaluation, summarize
from kernelgen.families import FAMILY_ORDER, get_family
from kernelgen.policy import check_source
from kernelgen.prompts import format_response
from kernelgen.rows import FIELDS, PROFILE_FIELDS

DATA = Path(__file__).resolve().parent.parent / "data"
ENV = {"gpu_name": "cpu", "torch_version": "t", "cuda_version": "", "triton_version": ""}


# --- the 50-task benchmark ----------------------------------------------------------------------------------


def test_the_benchmark_covers_the_50_heldout_tasks_once_each():
    cands = reference_candidates()
    assert len(cands) == 50
    names = [task.name for task, _, _ in cands]
    assert len(set(names)) == 50
    assert set(names) == {t["id"] for t in json.loads((DATA / "heldout_tasks.json").read_text())}


def test_reference_candidates_are_policy_clean_template_kernels():
    for task, candidate_id, source in reference_candidates():
        assert candidate_id.startswith(f"template:{task.family}:")
        assert check_source(source) == [], task.name


def test_reference_candidates_can_be_narrowed():
    assert {t.family for t, _, _ in reference_candidates("softmax")} == {"softmax"}
    sample = reference_candidates(sample_only=True)
    assert [t.family for t, _, _ in sample] == FAMILY_ORDER


def test_benchmark_refuses_to_run_without_a_gpu(tmp_path):
    if torch.cuda.is_available():
        pytest.skip("this checks the no-GPU message")
    assert benchmark.main(["--out", str(tmp_path / "k.csv")]) == 2
    assert not (tmp_path / "k.csv").exists()


def test_run_reference_kernels_writes_a_row_per_task_through_the_sandbox(monkeypatch, tmp_path):
    # Small stand-in tasks: the real held-out shapes are too big for the CPU interpreter.
    fam = get_family("softmax")
    small = [(fam.task({"M": 4, "N": 100}), "template:softmax:small", fam.render({"M": 4, "N": 100}, fam.default_config({"M": 4, "N": 100})))]
    monkeypatch.setattr(benchmark, "reference_candidates", lambda family=None, sample_only=False: small)
    lines = []
    rows = benchmark.run_reference_kernels("cpu", 1, 3, range(2), 120.0, progress=lines.append)
    assert len(rows) == 1 and rows[0]["status"] == "ok", rows[0]["error"]
    assert rows[0]["library_desc"] == "torch.softmax" and rows[0]["library_ms"] is not None
    benchmark.write_csv(rows, tmp_path / "k.csv")
    assert list(csv.DictReader((tmp_path / "k.csv").open()))[0]["task"] == "softmax_M4_N100_float32"
    assert lines and "1/1" in lines[0]


# --- evaluating model replies --------------------------------------------------------------------------------------


LN = get_family("layernorm")
SM = get_family("softmax")
SPECS = [LN.spec({"M": 6, "N": 200}) | {"id": "ln"}, SM.spec({"M": 4, "N": 100}) | {"id": "sm"}]


def reply_for(fam, params):
    return format_response(fam.render(params, fam.default_config(params)))


def cfg(tmp_path, **kw):
    base = dict(tag="t", out_dir=str(tmp_path / "res"), samples_per_task=3, warmup=1, reps=3, seeds=2, timeout=120.0, profile_slow=0)
    base.update(kw)
    return EvalConfig(**base)


def fake_generate(prompts, k):
    """First task: a correct kernel, a wrong one, prose. Second task: k correct kernels."""
    wrong = reply_for(LN, {"M": 6, "N": 200}).replace("+ b\n", "\n")
    out = []
    for prompt in prompts:
        if "LayerNorm" in prompt:
            out.append([reply_for(LN, {"M": 6, "N": 200}), wrong, "I would write a kernel but here is only prose."][:k])
        else:
            out.append([reply_for(SM, {"M": 4, "N": 100})] * k)
    return out


def test_every_sample_gets_a_row_and_the_outputs_are_written(tmp_path):
    summary = run_evaluation(cfg(tmp_path), generate=fake_generate, tasks=SPECS, device="cpu", env=ENV)
    res = tmp_path / "res"

    rows = list(csv.DictReader((res / "generated_t.csv").open()))
    assert list(rows[0]) == FIELDS and len(rows) == 6
    by_id = {r["candidate_id"]: r for r in rows}
    assert by_id["gen:t:layernorm_M6_N200_float32:0"]["status"] == "ok"
    assert by_id["gen:t:layernorm_M6_N200_float32:1"]["status"] == "incorrect"
    prose = by_id["gen:t:layernorm_M6_N200_float32:2"]
    assert prose["status"] == "no_code" and prose["reward"] == "0.0" and prose["speedup"] == ""

    records = [json.loads(line) for line in (res / "generated_t.jsonl").read_text().splitlines()]
    assert len(records) == 6 and records[2]["code"] is None and records[0]["code"] is not None
    assert (res / "generated_t_summary.json").exists()
    assert summary["n_samples"] == 6 and summary["status_counts"] == {"ok": 4, "incorrect": 1, "no_code": 1}


def test_summary_metrics(tmp_path):
    s = run_evaluation(cfg(tmp_path), generate=fake_generate, tasks=SPECS, device="cpu", env=ENV)
    assert s["n_tasks"] == 2 and s["pass_at_1"] == 1.0 and s["pass_at_k"] == 1.0
    assert s["tasks_with_a_correct_sample"] == 2
    assert s["frac_samples_with_code"] == pytest.approx(5 / 6)
    assert s["frac_samples_correct"] == pytest.approx(4 / 6)
    assert set(s["per_family"]) == {"layernorm", "softmax"}


def test_partial_results_survive_a_task_that_dies(tmp_path):
    def dying(prompts, k):
        if "Row-wise softmax" in prompts[0]:
            raise RuntimeError("generation crashed")
        return fake_generate(prompts, k)

    with pytest.raises(RuntimeError, match="generation crashed"):
        run_evaluation(cfg(tmp_path), generate=dying, tasks=SPECS, device="cpu", env=ENV)
    assert len(list(csv.DictReader((tmp_path / "res" / "generated_t.csv").open()))) == 3


def test_load_heldout_and_limit():
    assert len(load_heldout(DATA / "heldout_tasks.json")) == 50
    assert len(load_heldout(DATA / "heldout_tasks.json", limit=4)) == 4


# --- the summary arithmetic ------------------------------------------------------------------------------------------


def row(task, family, status="ok", speedup=None, library=None):
    return {"task": task, "family": family, "status": status, "speedup": speedup, "speedup_vs_library": library}


def test_summary_arithmetic_on_known_rows():
    rows = [
        row("a", "gemm", "ok", 4.0, 2.0), row("a", "gemm", "incorrect"), row("a", "gemm", "ok", 1.0, 0.5),
        row("b", "gemm", "incorrect"), row("b", "gemm", "no_code"), row("b", "gemm", "timeout"),
        row("c", "softmax", "ok", 0.25, None), row("c", "softmax", "ok", 0.5, None), row("c", "softmax", "error"),
        row("d", "softmax", "no_code"), row("d", "softmax", "no_code"), row("d", "softmax", "no_code"),
        row("e", "gemm", "incorrect"), row("e", "gemm", "ok", 2.0, None),  # first sample wrong, second right
    ]
    s = summarize(rows, n_tasks=5)
    assert s["pass_at_1"] == 0.4  # only the first samples of a and c are correct (c's is slower than the baseline)
    assert s["pass_at_k"] == 0.6  # a, c and e have some correct sample
    assert s["tasks_with_a_correct_sample"] == 3
    assert s["tasks_where_best_beats_baseline"] == 2  # a: best 4.0, e: best 2.0; c: best 0.5 does not
    assert s["best_of_k_speedup_geomean"] == pytest.approx((4.0 * 0.5 * 2.0) ** (1 / 3))
    assert s["mean_speedup_of_correct_samples"] == pytest.approx((4.0 + 1.0 + 0.25 + 0.5 + 2.0) / 5)
    assert s["tasks_where_best_beats_library"] == 1 and s["tasks_with_a_library_comparison"] == 1
    assert s["best_of_k_speedup_vs_library_geomean"] == pytest.approx(2.0)
    assert s["per_family"]["gemm"]["tasks"] == 3 and s["per_family"]["gemm"]["pass_at_k"] == pytest.approx(2 / 3)
    assert s["per_family"]["softmax"]["tasks"] == 2 and s["per_family"]["softmax"]["pass_at_k"] == 0.5
    assert s["status_counts"]["no_code"] == 4 and s["n_samples"] == 14


def test_summary_of_a_run_with_nothing_correct_has_no_speedup_to_report():
    s = summarize([row("a", "gemm", "incorrect"), row("a", "gemm", "no_code")], n_tasks=1)
    assert s["pass_at_k"] == 0.0 and s["best_of_k_speedup_geomean"] is None and s["mean_speedup_of_correct_samples"] is None


# --- slow-kernel diagnosis --------------------------------------------------------------------------------------------------


def profile_row(status="ok", kernel_ms=1.0, launches=1, gbps=100.0, pct=40.0, event_ms=1.1, top=None):
    r = {k: None for k in PROFILE_FIELDS}
    r.update(status=status, kernel_ms_per_iter=kernel_ms, kernel_launches_per_iter=launches, effective_gbps=gbps,
             pct_of_copy_bw=pct, copy_gbps=250.0, event_ms=event_ms, error="" if status == "ok" else "boom")
    r["top_kernels"] = json.dumps(top if top is not None else [])
    return r


def test_diagnosis_states_the_measured_facts():
    cand = profile_row(kernel_ms=3.0, launches=1, gbps=30.0, pct=12.0, event_ms=4.5,
                       top=[{"name": "_k", "per_iter": 1.0, "avg_us": 3000.0, "grid": [8, 1, 1], "block": [32, 1, 1]}])
    base = profile_row(kernel_ms=1.5, launches=9, gbps=60.0, pct=24.0)
    notes = describe_slowdown(cand, base)
    text = "\n".join(notes)
    assert "2.00x the baseline's" in text
    assert "candidate 1, baseline 9" in text
    assert "12%" in text and "not limited by memory bandwidth" in text
    assert "launch overhead or gaps" in text
    assert "grid [8, 1, 1]" in text and "block [32, 1, 1]" in text


def test_diagnosis_does_not_invent_a_cause():
    text = "\n".join(describe_slowdown(profile_row(kernel_ms=3.0), profile_row(kernel_ms=1.0)))
    assert "because" not in text and "likely" not in text and "caused" not in text


def test_diagnosis_says_so_when_a_profile_is_unusable():
    notes = describe_slowdown(profile_row(status="error"), profile_row())
    assert len(notes) == 1 and "no usable candidate profile" in notes[0] and "boom" in notes[0]


def test_diagnosis_without_gpu_kernel_times():
    cpu = profile_row(kernel_ms=None, launches=None, gbps=None, pct=None)
    assert "no GPU kernel times" in "\n".join(describe_slowdown(cpu, cpu))


def test_diagnosis_flags_suspect_profiles():
    assert "suspect" in describe_slowdown(profile_row(status="suspect"), profile_row())[0]


def test_slow_correct_kernels_are_profiled_and_logged(tmp_path):
    pytest.importorskip("triton")  # on CPU every interpreted Triton kernel loses to the baseline
    tasks = [LN.spec({"M": 6, "N": 200}) | {"id": "ln"}]
    only_correct = lambda prompts, k: [[reply_for(LN, {"M": 6, "N": 200})] * k for _ in prompts]  # noqa: E731
    summary = run_evaluation(cfg(tmp_path, samples_per_task=2, profile_slow=1), generate=only_correct, tasks=tasks, device="cpu", env=ENV)
    slow = summary["slow_kernels"]
    assert len(slow) == 1 and slow[0]["speedup"] < 1.0 and slow[0]["observations"]
    assert (tmp_path / "res" / "slow_kernel_profiles_t.csv").exists()
    profiles = list(csv.DictReader((tmp_path / "res" / "slow_kernel_profiles_t.csv").open()))
    assert [p["impl"] for p in profiles] == ["baseline", "candidate"]


# --- Nsight (optional) -------------------------------------------------------------------------------------------------------------


def test_available_tools_reports_what_is_on_the_path(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/opt/nsight/ncu" if name == "ncu" else None)
    assert nsight.available_tools() == {"ncu": "/opt/nsight/ncu", "nsys": None}


def test_kernel_names_are_the_jit_functions():
    src = get_family("gemm").render({"M": 64, "N": 64, "K": 64}, get_family("gemm").default_config({}))
    assert nsight.kernel_names(src) == ["_matmul"]
    assert nsight.kernel_names(get_family("attention").render({"B": 1, "H": 1, "S": 8, "D": 16}, get_family("attention").default_config({}))) == ["_attn_fwd"]
    assert nsight.kernel_names("def helper():\n    pass\n") == []


def test_command_lines(tmp_path):
    run = ["python", "-m", "kernelgen.run_once"]
    ncu = nsight.build_command("ncu", "/bin/ncu", tmp_path / "x.ncu", run, "_matmul")
    assert ncu[0] == "/bin/ncu" and "--kernel-name=regex:_matmul" in ncu and ncu[-3:] == run and f"--export={tmp_path / 'x.ncu'}" in ncu
    nsys = nsight.build_command("nsys", "/bin/nsys", tmp_path / "x.nsys", run, None)
    assert nsys[:2] == ["/bin/nsys", "profile"] and nsys[-3:] == run
    with pytest.raises(ValueError):
        nsight.build_command("vtune", "/bin/vtune", tmp_path / "x", run, None)


def test_nsight_without_tools_records_that_nothing_was_produced(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(nsight, "available_tools", lambda: {"ncu": None, "nsys": None})
    assert nsight.main(["--out", str(tmp_path / "ns")]) == 0
    status = json.loads((tmp_path / "ns" / "STATUS.json").read_text())
    assert status["runs"] == [] and "no Nsight profile was produced" in status["note"]
    assert "neither ncu nor nsys" in capsys.readouterr().out


def test_a_tool_that_fails_is_recorded_not_raised(monkeypatch, tmp_path):
    fake = tmp_path / "ncu"
    fake.write_text("#!/bin/sh\necho 'ERR_NVGPUCTRPERM: no permission to access performance counters' >&2\nexit 1\n")
    fake.chmod(0o755)
    monkeypatch.setattr(nsight, "available_tools", lambda: {"ncu": str(fake), "nsys": None})
    assert nsight.main(["--out", str(tmp_path / "ns")]) == 0
    (run,) = json.loads((tmp_path / "ns" / "STATUS.json").read_text())["runs"]
    assert run["tool"] == "ncu" and run["returncode"] == 1 and not run["produced_profile"]
    assert "ERR_NVGPUCTRPERM" in run["output_tail"]


def test_a_tool_that_succeeds_has_its_files_listed(monkeypatch, tmp_path):
    fake = tmp_path / "nsys"
    fake.write_text('#!/bin/sh\nfor a in "$@"; do case "$a" in --output=*) touch "${a#--output=}.nsys-rep";; esac; done\nexit 0\n')
    fake.chmod(0o755)
    monkeypatch.setattr(nsight, "available_tools", lambda: {"ncu": None, "nsys": str(fake)})
    nsight.main(["--out", str(tmp_path / "ns")])
    (run,) = json.loads((tmp_path / "ns" / "STATUS.json").read_text())["runs"]
    assert run["produced_profile"] and run["profile_files"][0].endswith(".nsys-rep")


def test_run_once_runs_the_template_and_refuses_a_cheating_source(tmp_path, capsys):
    assert run_once.main(["--family", "softmax", "--params", '{"M": 4, "N": 100}', "--warmup", "1", "--iters", "1"]) == 0
    assert "ran softmax_M4_N100_float32" in capsys.readouterr().out
    cheat = tmp_path / "cheat.py"
    cheat.write_text("import torch\n\ndef solution(x):\n    return torch.softmax(x, -1)\n")
    assert run_once.main(["--family", "softmax", "--params", '{"M": 4, "N": 100}', "--source", str(cheat)]) == 1
    assert "rejected by the policy" in capsys.readouterr().err
