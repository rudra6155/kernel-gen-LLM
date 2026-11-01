"""The README tables and figures come from the result files, so they cannot drift from them."""

import shutil
from pathlib import Path

import pytest

from kernelgen import figures
from kernelgen.figures import best_per_task, geomean, results_markdown, update_readme

REPO = Path(__file__).resolve().parent.parent


def row(task, status="ok", speedup="2.0", library="1.5", family="softmax"):
    return {"task": task, "status": status, "speedup": speedup, "speedup_vs_library": library, "family": family}


def test_geomean_is_not_the_arithmetic_mean():
    assert geomean([1.0, 64.0]) == pytest.approx(8.0)
    assert geomean([]) is None


def test_best_per_task_takes_the_largest_correct_sample_and_ignores_failures():
    rows = [row("a", speedup="1.5"), row("a", speedup="3.0"), row("a", status="incorrect", speedup="99"), row("b", status="error", speedup="")]
    assert best_per_task(rows, "speedup") == {"a": 3.0}


def test_readme_results_block_matches_the_committed_results():
    # If this fails, run `make figures` and commit README.md and docs/img/.
    assert update_readme(REPO, write=False), "README.md results tables are stale: run `make figures`"


def test_results_tables_say_what_the_csvs_say():
    text = results_markdown(REPO)
    assert "| After SFT | 185 | 9 | 6 | 0 | 0 | 50 | 48/50 |" in text
    assert "| Base model | 0 | 0 | 74 | 121 | 5 | 0 | 0/50 |" in text
    assert "SFT, best of 4" in text and "Template, no model" in text


def test_readme_tables_are_rewritten_when_the_data_changes(tmp_path):
    for rel in ("README.md", "results/kernel_benchmarks.csv", "results/generated_base.csv", "results/generated_sft.csv", "results/generated_rl.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    assert update_readme(tmp_path, write=False)
    rl = tmp_path / "results" / "generated_rl.csv"
    rl.write_text(rl.read_text().replace(",ok,", ",error,", 3))
    assert not update_readme(tmp_path, write=False), "a changed CSV must make the README block stale"
    update_readme(tmp_path, write=True)
    assert update_readme(tmp_path, write=False), "rewriting the block must make it current"


def test_readme_needs_its_markers(tmp_path):
    (tmp_path / "README.md").write_text("no markers here\n")
    with pytest.raises(ValueError, match="markers"):
        update_readme(tmp_path, write=False)


def test_check_command_reports_stale_and_current(tmp_path, capsys):
    assert figures.main(["--root", str(REPO), "--check"]) == 0
    (tmp_path / "results").mkdir()
    for rel in ("README.md", "results/kernel_benchmarks.csv", "results/generated_base.csv", "results/generated_sft.csv", "results/generated_rl.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text().replace("| After SFT | 185 |", "| After SFT | 999 |"))
    assert figures.main(["--root", str(tmp_path), "--check"]) == 1
    assert "stale" in capsys.readouterr().err


def test_figures_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    for rel in ("results/kernel_benchmarks.csv", "results/generated_base.csv", "results/generated_sft.csv", "results/generated_rl.csv",
                "results/training/sft/train_log.csv", "results/training/rl/reward_curve.csv"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    paths = figures.make_figures(tmp_path)
    assert [p.name for p in paths] == ["speedup_by_family.png", "sample_outcomes.png", "training.png"]
    for p in paths:
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and p.stat().st_size > 10_000


def test_committed_images_exist_and_the_readme_shows_them():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    for name in ("speedup_by_family.png", "sample_outcomes.png", "training.png"):
        assert (REPO / "docs" / "img" / name).exists()
        assert f"docs/img/{name}" in readme
