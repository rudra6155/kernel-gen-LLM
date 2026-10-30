import copy
import json

import pytest

from kernelgen import annotate as ann_mod
from kernelgen.annotate import ANNOTATION_FIELDS, annotate, evaluate_record, load_annotations, select, to_annotation
from kernelgen.dataset import MAX_RECORDS, MIN_RECORDS, build_records, make_record, read_jsonl, sha256, write_jsonl
from kernelgen.families import get_family
from kernelgen.verify_dataset import check_annotation, check_heldout, check_records, main, remeasure

ENV = {"gpu_name": "cpu", "torch_version": "t", "cuda_version": "", "triton_version": ""}
RECORDS = build_records()


def problems_for(mutate):
    records = copy.deepcopy(RECORDS)
    mutate(records)
    return check_records(records, MIN_RECORDS, MAX_RECORDS)


# --- record checks ----------------------------------------------------------------------------------


def test_clean_dataset_has_no_problems():
    assert check_records(RECORDS, MIN_RECORDS, MAX_RECORDS) == []


def test_too_few_records_is_a_problem():
    assert any("expected between" in p for p in check_records(RECORDS[:10], MIN_RECORDS, MAX_RECORDS))


def test_missing_field_is_a_problem():
    assert any("missing fields" in p for p in problems_for(lambda rs: rs[0].pop("config")))


def test_duplicate_id_and_duplicate_pair_are_problems():
    def dup(rs):
        rs[1] = copy.deepcopy(rs[0])

    found = problems_for(dup)
    assert any("duplicate id" in p for p in found) and any("duplicate (task, config)" in p for p in found)


def test_edited_source_is_caught_even_if_its_hash_is_fixed_up():
    def edit(rs):
        rs[0]["source"] += "# edited\n"
        rs[0]["source_sha256"] = sha256(rs[0]["source"])

    assert any("differs from the current template" in p for p in problems_for(edit))


def test_stale_hash_is_a_problem():
    def stale(rs):
        rs[0]["source_sha256"] = "0" * 64

    assert any("source_sha256 does not match" in p for p in problems_for(stale))


def test_a_heldout_shape_in_the_training_set_is_leakage():
    fam = get_family("softmax")
    leaked = make_record(fam, fam.heldout_shapes[0], fam.default_config(fam.heldout_shapes[0]))

    def leak(rs):
        rs[0] = leaked

    found = problems_for(leak)
    assert any("held out (leakage)" in p for p in found) and any("not a training shape" in p for p in found)


def test_unknown_family_is_a_problem():
    def rename(rs):
        rs[0]["family"] = "conv2d"

    assert any("unknown task family" in p for p in problems_for(rename))


def test_source_that_breaks_the_policy_is_a_problem(monkeypatch):
    from kernelgen import verify_dataset

    monkeypatch.setattr(verify_dataset, "check_source", lambda s: ["line 1: torch.matmul is not allowed"])
    assert any("fails the candidate policy" in p for p in check_records(RECORDS[:3], 1, 10))


def test_heldout_file_checks(tmp_path):
    from kernelgen.dataset import heldout_tasks

    good = tmp_path / "h.json"
    good.write_text(json.dumps(heldout_tasks()))
    assert check_heldout(good) == []
    assert any("missing" in p for p in check_heldout(tmp_path / "nope.json"))
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(heldout_tasks()[:-1]))
    assert any("differs" in p for p in check_heldout(bad)) and any("expected 50" in p for p in check_heldout(bad))


# --- annotation checks ----------------------------------------------------------------------------------


def good_annotation(record):
    row = {
        "status": "ok", "correct": 1, "max_abs_err": 1e-6, "baseline_ms": 2.0, "candidate_ms": 0.5, "speedup": 4.0,
        "library_ms": 1.0, "speedup_vs_library": 2.0, "error": "", "warmup": 5, "reps": 30, "seeds": 3, **ENV,
    }
    return to_annotation(record, row)


def failed_annotation(record, status="incorrect"):
    row = {k: None for k in ANNOTATION_FIELDS}
    row.update(status=status, correct=0, error="seed 0: max abs err 1.0 exceeds tolerance", warmup=5, reps=30, seeds=3, **ENV)
    return to_annotation(record, row)


def test_annotation_has_exactly_the_schema_fields():
    assert set(good_annotation(RECORDS[0])) == set(ANNOTATION_FIELDS)


def test_good_and_failed_annotations_pass():
    assert check_annotation(good_annotation(RECORDS[0]), RECORDS[0]) == []
    for status in ("incorrect", "error", "rejected", "timeout", "crash"):
        assert check_annotation(failed_annotation(RECORDS[0], status), RECORDS[0]) == []


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda a: a.update(speedup=9.0), "speedup 9.0 != baseline_ms / candidate_ms"),
        (lambda a: a.update(correct=0), "status ok but correct=0"),
        (lambda a: a.update(candidate_ms=-1.0, speedup=-2.0), "not a positive finite number"),
        (lambda a: a.update(baseline_ms=float("nan")), "not a positive finite number"),
        (lambda a: a.update(speedup_vs_library=None), "both be set or both empty"),
        (lambda a: a.update(status="winning"), "unknown status"),
        (lambda a: a.update(source_sha256="f" * 64), "stale"),
        (lambda a: a.pop("error"), "not the annotation schema"),
    ],
)
def test_inconsistent_annotations_are_caught(mutate, expected):
    ann = good_annotation(RECORDS[0])
    mutate(ann)
    assert any(expected in p for p in check_annotation(ann, RECORDS[0]))


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda a: a.update(correct=1), "correct=1"),
        (lambda a: a.update(speedup=2.0), "has a speedup"),
        (lambda a: a.update(error=""), "without an error message"),
    ],
)
def test_failed_annotations_must_not_claim_success(mutate, expected):
    ann = failed_annotation(RECORDS[0])
    mutate(ann)
    assert any(expected in p for p in check_annotation(ann, RECORDS[0]))


def test_annotation_without_a_record_is_a_problem():
    assert any("no such record" in p for p in check_annotation(good_annotation(RECORDS[0]), None))


# --- the annotation runner ---------------------------------------------------------------------------------


def fake_eval(counter):
    def evaluate_fn(record):
        counter.append(record["id"])
        row = {k: None for k in ANNOTATION_FIELDS}
        row.update(status="ok", correct=1, baseline_ms=2.0, candidate_ms=1.0, speedup=2.0, error="", warmup=1, reps=1, seeds=1, **ENV)
        return row

    return evaluate_fn


def test_annotate_writes_one_line_per_record_and_prints_progress(tmp_path):
    out, seen, lines = tmp_path / "a.jsonl", [], []
    counts = annotate(RECORDS[:3], out, fake_eval(seen), progress=lines.append)
    assert counts == {"done": 3, "skipped": 0} and len(lines) == 3 and "1/3" in lines[0]
    assert sorted(load_annotations(out)) == sorted(r["id"] for r in RECORDS[:3])
    assert all(set(a) == set(ANNOTATION_FIELDS) for a in load_annotations(out).values())


def test_annotate_resumes_without_redoing_finished_records(tmp_path):
    out, seen = tmp_path / "a.jsonl", []
    annotate(RECORDS[:2], out, fake_eval(seen), progress=lambda s: None)
    seen.clear()
    counts = annotate(RECORDS[:4], out, fake_eval(seen), progress=lambda s: None)
    assert counts == {"done": 2, "skipped": 2}
    assert seen == [RECORDS[2]["id"], RECORDS[3]["id"]]


def test_annotate_redoes_a_record_whose_source_changed(tmp_path):
    out, seen = tmp_path / "a.jsonl", []
    annotate(RECORDS[:1], out, fake_eval(seen), progress=lambda s: None)
    changed = dict(RECORDS[0], source_sha256="0" * 64)
    counts = annotate([changed], out, fake_eval(seen), progress=lambda s: None)
    assert counts["done"] == 1
    assert load_annotations(out)[changed["id"]]["source_sha256"] == "0" * 64  # the later line wins


def test_annotate_accepts_a_generator(tmp_path):
    assert annotate((r for r in RECORDS[:2]), tmp_path / "a.jsonl", fake_eval([]), progress=lambda s: None)["done"] == 2


def test_select_by_family_shard_and_limit():
    assert {r["family"] for r in select(RECORDS, "softmax", None, None)} == {"softmax"}
    shards = [select(RECORDS, None, f"{i}/3", None) for i in range(3)]
    assert sum(len(s) for s in shards) == len(RECORDS)
    assert {r["id"] for r in shards[0]}.isdisjoint({r["id"] for r in shards[1]})
    assert len(select(RECORDS, None, None, 7)) == 7
    with pytest.raises(ValueError, match="--shard"):
        select(RECORDS, None, "3/3", None)


def test_real_annotation_of_a_small_record_passes_verification(tmp_path):
    # Runs the real sandbox: policy, worker process, correctness gate, timing.
    fam = get_family("layernorm")
    params = {"M": 6, "N": 200}
    record = make_record(fam, params, fam.default_config(params))
    row = evaluate_record(record, "cpu", 1, 3, range(2), 120.0, ENV)
    assert row["status"] == "ok", row["error"]
    ann = to_annotation(record, row)
    assert check_annotation(ann, record) == []


# --- re-measurement -------------------------------------------------------------------------------------------


def remeasure_problems(stored_speedup, new_row):
    record = RECORDS[0]
    stored = good_annotation(record)
    stored.update(speedup=stored_speedup, baseline_ms=stored_speedup, candidate_ms=1.0)
    return remeasure([record], {record["id"]: stored}, 1, 0.25, 0, lambda r: new_row)


def test_remeasure_accepts_a_close_match():
    assert remeasure_problems(4.0, {"correct": 1, "status": "ok", "speedup": 4.4}) == []


def test_remeasure_flags_a_speedup_that_drifted():
    found = remeasure_problems(4.0, {"correct": 1, "status": "ok", "speedup": 2.0})
    assert len(found) == 1 and "apart" in found[0]


def test_remeasure_flags_a_flipped_correctness():
    found = remeasure_problems(4.0, {"correct": 0, "status": "incorrect", "speedup": None})
    assert len(found) == 1 and "correct was 1, now 0" in found[0]


# --- the command line --------------------------------------------------------------------------------------------------


def test_main_reports_pending_annotations_and_passes(tmp_path, capsys):
    code = main(["--annotations", str(tmp_path / "none.jsonl")])
    out = capsys.readouterr().out
    assert code == 0
    assert "PENDING" in out and "dataset OK" in out and "720 (" in out and "distinct kernel sources" in out


def test_main_fails_without_annotations_when_they_are_required(tmp_path, capsys):
    assert main(["--annotations", str(tmp_path / "none.jsonl"), "--require-annotations"]) == 1
    assert "annotations are required" in capsys.readouterr().out


def test_main_checks_annotations_when_present(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    good, bad = good_annotation(RECORDS[0]), good_annotation(RECORDS[1])
    bad["speedup"] = 99.0
    write_jsonl([good, bad], path)
    assert main(["--annotations", str(path)]) == 1
    out = capsys.readouterr().out
    assert "2/720 records annotated" in out and "speedup 99.0" in out


def test_main_counts_only_current_annotations(tmp_path, capsys):
    path = tmp_path / "a.jsonl"
    write_jsonl([good_annotation(r) for r in RECORDS[:5]], path)
    assert main(["--annotations", str(path), "--require-annotations"]) == 1
    out = capsys.readouterr().out
    assert "5/720 records annotated, 5 correct, 5 faster than baseline" in out and "only 5/720" in out


def test_remeasure_needs_a_gpu(tmp_path, capsys):
    import torch

    if torch.cuda.is_available():
        pytest.skip("this checks the no-GPU message")
    assert main(["--annotations", str(tmp_path / "x.jsonl"), "--remeasure", "3"]) == 2
