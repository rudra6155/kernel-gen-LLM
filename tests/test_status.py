"""The Definition of Done checklist and the log collector."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kernelgen import collect_logs, status
from kernelgen.claims import render
from kernelgen.status import DONE, FAILED, NOT_CHECKED, PENDING

REPO = Path(__file__).resolve().parent.parent


def by_name(items):
    return {i.name: i for i in items}


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A small git repository with the dataset and a current CLAIMS.md, and nothing else."""
    (tmp_path / "data").mkdir()
    for name in ("kernels.jsonl", "heldout_tasks.json"):
        shutil.copy(REPO / "data" / name, tmp_path / "data" / name)
    (tmp_path / "CLAIMS.md").write_text(render(tmp_path), encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test Author")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "commit.gpgsign", "false")
    return tmp_path


def commit(root, message="chore: step"):
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", message)


# --- the checklist ------------------------------------------------------------------------------------------------


def test_a_fresh_repository_has_the_dataset_done_and_every_gpu_item_pending(repo):
    for i in range(5):
        commit(repo, f"chore: step {i}")
    items = by_name(status.collect(repo))
    assert items["dataset of 500-2000 template kernels, no held-out shapes, all policy-clean"].state == DONE
    for name, item in items.items():
        if "GPU" in item.detail or item.name.startswith(("results/", "dataset annotated", "torch.profiler", "QLoRA", "GRPO", "generated")):
            assert item.state == PENDING, name
    assert items["make test is green"].state == NOT_CHECKED
    assert items["CLAIMS.md: every claim VERIFIED"].state == PENDING
    assert items["git history: incremental commits by one author, no co-author trailers"].state == DONE


def test_pending_items_say_which_command_produces_them(repo):
    items = status.collect(repo)
    text = " ".join(i.detail for i in items)
    for command in ("make annotate", "make benchmark", "make profile", "make sft", "make rl", "make eval-model TAG=base"):
        assert command in text


def test_a_result_file_that_is_not_committed_does_not_count(repo):
    (repo / "results").mkdir()
    (repo / "results" / "kernel_benchmarks.csv").write_text("task\n" + "x\n" * 50)
    item = status.check_benchmark(repo)
    assert item.state == PENDING and "not committed" in item.detail
    commit(repo)
    assert status.check_benchmark(repo).state == DONE


def test_the_benchmark_must_have_one_row_per_held_out_task(repo):
    (repo / "results").mkdir()
    (repo / "results" / "kernel_benchmarks.csv").write_text("task\n" + "x\n" * 49)
    commit(repo)
    item = status.check_benchmark(repo)
    assert item.state == FAILED and "expected 50" in item.detail


def test_profile_needs_both_the_summary_and_traces(repo):
    (repo / "results").mkdir()
    (repo / "results" / "profile_summary.csv").write_text("task\n")
    commit(repo)
    profile, nsight = status.check_profile(repo)
    assert profile.state == PENDING and "no traces" in profile.detail
    (repo / "results" / "traces").mkdir()
    (repo / "results" / "traces" / "a.json").write_text("{}")
    commit(repo)
    assert status.check_profile(repo)[0].state == DONE
    assert nsight.state == PENDING and "optional" in nsight.name


def test_nsight_that_ran_without_producing_a_profile_is_not_done(repo):
    (repo / "results" / "nsight").mkdir(parents=True)
    (repo / "results" / "nsight" / "STATUS.json").write_text(json.dumps({"runs": [{"tool": "ncu", "produced_profile": False}]}))
    nsight = status.check_profile(repo)[1]
    assert nsight.state == PENDING and "no tool produced a profile" in nsight.detail
    (repo / "results" / "nsight" / "STATUS.json").write_text(json.dumps({"runs": [{"tool": "nsys", "produced_profile": True}]}))
    assert status.check_profile(repo)[1].state == DONE


def test_a_diverged_rl_run_is_reported_as_such(repo):
    base = repo / "results" / "training" / "rl"
    base.mkdir(parents=True)
    (base / "reward_curve.csv").write_text("step\n")
    (base / "divergence_report.json").write_text("{}")
    commit(repo)
    rl = next(i for i in status.check_training(repo) if i.name.startswith("GRPO"))
    assert rl.state == DONE and "DIVERGED" in rl.detail


def test_a_stale_claims_file_fails_the_checklist(repo):
    (repo / "CLAIMS.md").write_text("# edited by hand\n")
    item = status.check_claims(repo)
    assert item.state == FAILED and "out of date" in item.detail


def test_a_refuted_claim_fails_the_checklist_and_is_named(repo, monkeypatch):
    from kernelgen import claims
    from kernelgen.claims import REFUTED, Result

    real = claims.evaluate_claim
    monkeypatch.setattr(
        claims, "evaluate_claim",
        lambda c, root: Result(c, REFUTED, 0.5, None, "", []) if c.id == "B5" else real(c, root),
    )
    (repo / "CLAIMS.md").write_text(claims.render(repo), encoding="utf-8")
    item = status.check_claims(repo)
    assert item.state == FAILED and "refuted: B5" in item.detail


# --- git history ------------------------------------------------------------------------------------------------------------------


def test_history_needs_a_handful_of_commits(repo):
    commit(repo)
    assert status.check_history(repo).state == FAILED


@pytest.mark.parametrize(
    "message",
    ["feat: add x\n\nCo-Authored-By: Someone <a@b.c>", "feat: add x\n\nGenerated with a tool"],
)
def test_history_flags_attribution_in_commit_messages(repo, message):
    for i in range(5):
        commit(repo, f"chore: step {i}")
    commit(repo, message)
    item = status.check_history(repo)
    assert item.state == FAILED and "attribution" in item.detail


def test_history_flags_a_second_author(repo):
    for i in range(5):
        commit(repo, f"chore: step {i}")
    git(repo, "config", "user.name", "Someone Else")
    commit(repo, "chore: another author")
    assert "2 different authors" in status.check_history(repo).detail


def test_the_real_history_is_clean():
    item = status.check_history(REPO)
    assert item.state == DONE, item.detail


def test_outside_a_repository_the_history_is_not_checked(tmp_path):
    assert status.check_history(tmp_path).state == NOT_CHECKED


# --- rendering and the command line --------------------------------------------------------------------------------------------------


def test_rendering_marks_only_done_items(repo):
    text = status.render(status.collect(repo))
    assert "[x] dataset of 500-2000 template kernels" in text
    assert "[ ] results/kernel_benchmarks.csv committed, one row per held-out task  [PENDING]" in text
    assert text.splitlines()[-1].endswith("or not checked.")


def test_main_exit_code_reflects_failures_not_pending_items(repo, capsys):
    assert status.main(["--root", str(repo)]) == 0  # pending is not a failure
    (repo / "CLAIMS.md").write_text("stale")
    assert status.main(["--root", str(repo)]) == 1
    assert "[FAILED]" in capsys.readouterr().out


def test_run_tests_option_reports_the_test_result(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n")
    item = status.check_tests(tmp_path, run=True)
    assert item.state == DONE and "passed" in item.detail
    (tmp_path / "tests" / "test_bad.py").write_text("def test_bad():\n    assert False\n")
    assert status.check_tests(tmp_path, run=True).state == FAILED


# --- the log collector ---------------------------------------------------------------------------------------------------------------------


def test_collect_copies_small_logs_and_leaves_adapters(tmp_path):
    sft, rl = tmp_path / "runs" / "sft", tmp_path / "runs" / "rl"
    (sft / "samples").mkdir(parents=True)
    (sft / "final").mkdir()
    (sft / "train_log.csv").write_text("epoch\n1\n")
    (sft / "samples" / "epoch_1.jsonl").write_text("{}\n")
    (sft / "final" / "adapter_model.safetensors").write_bytes(b"x" * 1000)
    rl.mkdir(parents=True)
    (rl / "reward_curve.csv").write_text("step\n1\n")

    copied = collect_logs.collect(tmp_path)

    dest = tmp_path / "results" / "training"
    assert (dest / "sft" / "train_log.csv").read_text() == "epoch\n1\n"
    assert (dest / "sft" / "samples" / "epoch_1.jsonl").exists() and (dest / "rl" / "reward_curve.csv").exists()
    assert not (dest / "sft" / "final").exists(), "adapters are large and stay local"
    assert "results/training/sft/train_log.csv" in copied and not any("final" in c for c in copied)


def test_collect_is_repeatable_and_refreshes_the_samples(tmp_path):
    samples = tmp_path / "runs" / "rl" / "samples"
    samples.mkdir(parents=True)
    (samples / "step_1.jsonl").write_text("a")
    collect_logs.collect(tmp_path)
    (samples / "step_1.jsonl").unlink()
    (samples / "step_2.jsonl").write_text("b")
    collect_logs.collect(tmp_path)
    assert sorted(p.name for p in (tmp_path / "results" / "training" / "rl" / "samples").iterdir()) == ["step_2.jsonl"]


def test_collect_with_no_runs_says_so(tmp_path, capsys):
    assert collect_logs.main(["--root", str(tmp_path)]) == 0
    assert "nothing to copy" in capsys.readouterr().out
