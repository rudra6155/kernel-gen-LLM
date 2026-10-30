"""The claims ledger: statuses come from CSV evidence, and CLAIMS.md cannot be edited by hand."""

import math
from pathlib import Path

import pytest

from kernelgen import claims
from kernelgen.claims import PENDING, REFUTED, VERIFIED, definitions, evaluate_claim, load_csv, render
from kernelgen.dataset import heldout_tasks
from kernelgen.rows import FIELDS, write_csv

REPO = Path(__file__).resolve().parent.parent
HELDOUT = heldout_tasks()


def row(task, family, status="ok", speedup=2.0, library=1.5, candidate="template:x"):
    r = {k: None for k in FIELDS}
    r.update(task=task, family=family, candidate_id=candidate, status=status, correct=int(status == "ok"),
             speedup=speedup if status == "ok" else None, speedup_vs_library=library if status == "ok" else None,
             reward=speedup if status == "ok" else 0.0, gpu_name="test gpu")
    return r


def benchmark_rows(speedup=lambda fam: 2.0, library=lambda fam: 1.5):
    return [row(t["id"], t["family"], speedup=speedup(t["family"]), library=library(t["family"])) for t in HELDOUT]


def put(root, name, rows):
    write_csv(rows, root / "results" / name)


def by_id(root):
    return {r.claim.id: r for r in (evaluate_claim(c, root) for c in definitions())}


# --- the definitions --------------------------------------------------------------------------------------------


def test_claim_ids_are_unique_and_every_claim_points_at_evidence():
    ds = definitions()
    assert len({c.id for c in ds}) == len(ds) == 19
    for c in ds:
        if c.kind == "csv":
            assert c.evidence.csv.startswith("results/") and c.evidence.csv.endswith(".csv")
            assert c.threshold is not None or c.versus is not None


def test_library_claims_cover_exactly_the_families_that_have_a_library_op():
    text = " ".join(c.statement for c in definitions() if "own optimized op" in c.statement and c.id.startswith("B"))
    for fam in ("layernorm", "rmsnorm", "softmax", "gemm_bias_relu", "attention"):
        assert f"The {fam} template" in text
    assert "The gemm template kernels beat PyTorch's own" not in text


# --- reading CSVs --------------------------------------------------------------------------------------------------


def test_load_csv_parses_numbers_and_records_the_file_line(tmp_path):
    put(tmp_path, "k.csv", [row("a", "gemm", speedup=3.5), row("b", "gemm", status="error")])
    rows = load_csv(tmp_path / "results" / "k.csv")
    assert [r["_line"] for r in rows] == [2, 3]
    assert rows[0]["speedup"] == 3.5 and rows[1]["speedup"] is None
    assert load_csv(tmp_path / "missing.csv") is None


# --- statuses -------------------------------------------------------------------------------------------------------------


def test_with_no_evidence_everything_is_pending(tmp_path):
    results = by_id(tmp_path)
    assert {r.status for r in results.values()} == {PENDING}
    assert all(r.value is None for r in results.values())


def test_the_real_repositorys_dataset_claim_is_verified():
    # Only this claim is asserted: the others depend on which result files have been committed.
    results = by_id(REPO)
    assert results["D1"].status == VERIFIED and "720 records" in results["D1"].detail
    assert all(r.status in (VERIFIED, REFUTED, PENDING) for r in results.values())


def test_a_complete_benchmark_that_beats_the_baseline_verifies_every_benchmark_claim(tmp_path):
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows())
    results = by_id(tmp_path)
    for cid in ["B1"] + [f"B{i}" for i in range(2, 13)]:
        assert results[cid].status == VERIFIED, (cid, results[cid].detail)
    assert results["B2"].value == pytest.approx(2.0) and results["B8"].value == pytest.approx(1.5)


def test_a_family_that_loses_to_the_baseline_is_refuted_not_hidden(tmp_path):
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows(speedup=lambda f: 0.5 if f == "gemm" else 2.0))
    results = by_id(tmp_path)
    assert results["B5"].status == REFUTED and results["B5"].value == pytest.approx(0.5)
    assert results["B2"].status == VERIFIED and results["B1"].status == VERIFIED  # the measurements themselves are fine


def test_beating_the_baseline_but_not_the_library_is_reported_separately(tmp_path):
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows(library=lambda f: 0.8))
    results = by_id(tmp_path)
    assert results["B2"].status == VERIFIED  # vs the naive baseline
    assert results["B8"].status == REFUTED and results["B8"].value == pytest.approx(0.8)  # but not vs F.layer_norm


def test_geometric_mean_not_arithmetic(tmp_path):
    rows = benchmark_rows()
    sm = [r for r in rows if r["family"] == "softmax"]
    for i, r in enumerate(sm):
        r["speedup"] = 1.0 if i < len(sm) - 1 else 64.0
    put(tmp_path, "kernel_benchmarks.csv", rows)
    assert by_id(tmp_path)["B4"].value == pytest.approx(64.0 ** (1 / len(sm)))


def test_a_partial_benchmark_is_pending_not_verified(tmp_path):
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows()[:49])
    results = by_id(tmp_path)
    assert results["B1"].status == PENDING and "49 of the 50 tasks" in results["B1"].detail
    assert results["B2"].status == VERIFIED  # layernorm's rows are all there


def test_a_failed_measurement_means_the_claim_is_not_shown(tmp_path):
    rows = benchmark_rows()
    broken = next(r for r in rows if r["family"] == "softmax")
    broken.update(status="error", correct=0, speedup=None, speedup_vs_library=None, reward=0.0)
    put(tmp_path, "kernel_benchmarks.csv", rows)
    results = by_id(tmp_path)
    assert results["B1"].status == REFUTED and "49 of 50" in results["B1"].detail
    assert results["B4"].status == REFUTED and "not shown" in results["B4"].detail and results["B4"].value is None


def test_a_missing_library_measurement_is_not_counted_as_a_win(tmp_path):
    rows = benchmark_rows()
    next(r for r in rows if r["family"] == "attention")["speedup_vs_library"] = None
    put(tmp_path, "kernel_benchmarks.csv", rows)
    assert by_id(tmp_path)["B12"].status == REFUTED


# --- model claims -------------------------------------------------------------------------------------------------------------


def generated(correct_tasks, speedup=2.0, library=1.2, samples=2):
    """`samples` rows per held-out task; the first `correct_tasks` tasks have a correct sample."""
    rows = []
    for i, t in enumerate(HELDOUT):
        for k in range(samples):
            if i < correct_tasks and k == samples - 1:
                rows.append(row(t["id"], t["family"], speedup=speedup, library=library, candidate=f"gen:x:{t['id']}:{k}"))
            else:
                rows.append(row(t["id"], t["family"], status="incorrect", candidate=f"gen:x:{t['id']}:{k}"))
    return rows


def test_model_claims_compare_two_evidence_files(tmp_path):
    put(tmp_path, "generated_base.csv", generated(10, speedup=1.0))
    put(tmp_path, "generated_sft.csv", generated(30, speedup=1.5))
    put(tmp_path, "generated_rl.csv", generated(30, speedup=3.0, library=1.2))
    r = by_id(tmp_path)
    assert r["G1"].status == VERIFIED and (r["G1"].value, r["G1"].other) == (0.6, 0.2)  # pass@k 30/50 vs 10/50
    assert r["G2"].status == VERIFIED and r["G2"].value == pytest.approx(3.0) and r["G2"].other == pytest.approx(1.5)
    assert r["G3"].status == VERIFIED
    assert r["G4"].status == VERIFIED  # equal pass@k satisfies >=
    assert r["G5"].status == VERIFIED and r["G5"].value == 30.0


def test_a_model_that_got_worse_is_refuted(tmp_path):
    put(tmp_path, "generated_base.csv", generated(10))
    put(tmp_path, "generated_sft.csv", generated(30, speedup=2.0))
    put(tmp_path, "generated_rl.csv", generated(20, speedup=0.8, library=0.5))
    r = by_id(tmp_path)
    assert r["G2"].status == REFUTED and r["G3"].status == REFUTED
    assert r["G4"].status == REFUTED  # 20/50 < 30/50: RL lost correctness
    assert r["G5"].status == REFUTED and r["G5"].value == 0.0


def test_a_comparison_with_one_side_missing_is_pending(tmp_path):
    put(tmp_path, "generated_sft.csv", generated(30))
    r = by_id(tmp_path)
    assert r["G1"].status == PENDING and "generated_base.csv" in r["G1"].detail


def test_a_model_with_no_correct_kernel_cannot_win_a_comparison(tmp_path):
    put(tmp_path, "generated_sft.csv", generated(30))
    put(tmp_path, "generated_rl.csv", generated(0))
    r = by_id(tmp_path)
    assert r["G2"].status == REFUTED and "not shown" in r["G2"].detail
    assert r["G3"].status == REFUTED


def test_pass_at_k_counts_tasks_not_samples(tmp_path):
    put(tmp_path, "generated_sft.csv", generated(25, samples=4))
    put(tmp_path, "generated_base.csv", generated(5, samples=1))
    assert by_id(tmp_path)["G1"].value == 0.5


def test_a_model_csv_covering_few_tasks_is_pending(tmp_path):
    put(tmp_path, "generated_rl.csv", [r for r in generated(50) if r["task"] in {t["id"] for t in HELDOUT[:10]}])
    assert by_id(tmp_path)["G3"].status == PENDING


# --- dataset claims -------------------------------------------------------------------------------------------------------------


def test_annotation_claim_follows_the_annotation_file(tmp_path):
    import shutil

    from kernelgen.annotate import ANNOTATION_FIELDS, to_annotation
    from kernelgen.dataset import read_jsonl, write_jsonl

    (tmp_path / "data").mkdir()
    for name in ("kernels.jsonl", "heldout_tasks.json"):
        shutil.copy(REPO / "data" / name, tmp_path / "data" / name)
    records = read_jsonl(tmp_path / "data" / "kernels.jsonl")
    d2 = next(c for c in definitions() if c.id == "D2")

    def annotation(rec):
        measured = {k: None for k in ANNOTATION_FIELDS}
        measured.update(status="ok", correct=1, baseline_ms=2.0, candidate_ms=1.0, speedup=2.0, error="", warmup=1, reps=1,
                        seeds=1, gpu_name="g", torch_version="t", cuda_version="c", triton_version="tr")
        return to_annotation(rec, measured)

    write_jsonl([annotation(r) for r in records[:100]], tmp_path / "data" / "annotations.jsonl")
    partial = evaluate_claim(d2, tmp_path)
    assert partial.status == PENDING and "100 of 720" in partial.detail

    write_jsonl([annotation(r) for r in records], tmp_path / "data" / "annotations.jsonl")
    assert evaluate_claim(d2, tmp_path).status == VERIFIED

    inconsistent = [annotation(r) for r in records]
    inconsistent[0]["speedup"] = 99.0
    write_jsonl(inconsistent, tmp_path / "data" / "annotations.jsonl")
    assert evaluate_claim(d2, tmp_path).status == REFUTED


def test_dataset_size_claim_is_refuted_for_a_bad_dataset(tmp_path):
    from kernelgen.dataset import build_records, write_jsonl

    (tmp_path / "data").mkdir()
    write_jsonl(build_records(500)[:100], tmp_path / "data" / "kernels.jsonl")
    (tmp_path / "data" / "heldout_tasks.json").write_text("[]")
    r = evaluate_claim(next(c for c in definitions() if c.id == "D1"), tmp_path)
    assert r.status == REFUTED and "problem" in r.detail


# --- CLAIMS.md ---------------------------------------------------------------------------------------------------------------------


def test_committed_claims_md_is_up_to_date():
    # This fails on purpose whenever a result file changes without the ledger being updated,
    # because a stale ledger would state claims the evidence no longer supports.
    assert (REPO / "CLAIMS.md").read_text(encoding="utf-8") == render(REPO), (
        "CLAIMS.md is out of date with the CSVs and data files it points to: run `make claims` and commit it"
    )


def test_every_measured_number_in_the_appendix_points_to_the_right_csv_line(tmp_path):
    rows = benchmark_rows()
    for i, r in enumerate(rows):
        r["speedup"] = 1.0 + i / 100  # a distinct speedup per row
    put(tmp_path, "kernel_benchmarks.csv", rows)
    text = render(tmp_path)
    csv_lines = (tmp_path / "results" / "kernel_benchmarks.csv").read_text().splitlines()
    pointers = [l for l in text.splitlines() if l.startswith("| results/kernel_benchmarks.csv:")]
    assert len(pointers) == 50
    for l in pointers:
        cells = [c.strip() for c in l.strip("|").split("|")]
        line_no = int(cells[0].split(":")[1])
        assert csv_lines[line_no - 1].startswith(cells[1] + ",")  # that line is this task's row
        assert f"{float(csv_lines[line_no - 1].split(',')[csv_lines[0].split(',').index('speedup')]):.3f}" == cells[4]


def test_the_appendix_for_a_model_lists_its_fastest_correct_sample_per_task(tmp_path):
    rows = generated(3, samples=3)
    for r in rows:
        if r["status"] == "ok":
            r["speedup"] = 2.0
    put(tmp_path, "generated_rl.csv", rows)
    text = render(tmp_path)
    assert "Model `rl`: fastest correct sample per held-out task" in text
    assert sum(l.startswith("| results/generated_rl.csv:") for l in text.splitlines()) == 3


def test_check_fails_on_a_hand_edited_status(tmp_path):
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows(speedup=lambda f: 0.5 if f == "gemm" else 2.0))
    assert claims.main(["--root", str(tmp_path), "--write"]) == 0
    assert claims.main(["--root", str(tmp_path), "--check"]) == 0
    path = tmp_path / "CLAIMS.md"
    text = path.read_text()
    assert "| B5 |" in text and "**REFUTED**" in text
    path.write_text(text.replace("**REFUTED**", "**VERIFIED**"))  # someone tidies up the ledger
    assert claims.main(["--root", str(tmp_path), "--check"]) == 1


def test_check_fails_when_the_csv_changed_after_the_ledger_was_written(tmp_path):
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows())
    claims.main(["--root", str(tmp_path), "--write"])
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows(speedup=lambda f: 3.0))
    assert claims.main(["--root", str(tmp_path), "--check"]) == 1


def test_require_verified_fails_while_anything_is_pending(tmp_path, capsys):
    claims.main(["--root", str(tmp_path), "--write"])
    assert claims.main(["--root", str(tmp_path), "--check"]) == 0
    assert claims.main(["--root", str(tmp_path), "--check", "--require-verified"]) == 1
    assert "not every claim is VERIFIED" in capsys.readouterr().err


def test_unknown_aggregation_is_an_error(tmp_path):
    put(tmp_path, "kernel_benchmarks.csv", benchmark_rows())
    bad = claims.Claim("X", "x", claims.Evidence(claims.BENCHMARK_CSV, "median", tasks=50), ">", 1.0)
    with pytest.raises(ValueError, match="unknown aggregation"):
        evaluate_claim(bad, tmp_path)
