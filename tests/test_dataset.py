import json
from pathlib import Path

import pytest

from kernelgen import dataset
from kernelgen.dataset import (
    DEFAULT_TARGET,
    MAX_RECORDS,
    MIN_RECORDS,
    build_records,
    heldout_tasks,
    quotas,
    read_jsonl,
    write_jsonl,
)
from kernelgen.families import FAMILY_ORDER
from kernelgen.verify_dataset import check_records

DATA = Path(__file__).resolve().parent.parent / "data"


def test_default_target_is_inside_the_required_range():
    assert MIN_RECORDS <= DEFAULT_TARGET <= MAX_RECORDS


@pytest.mark.parametrize("target", [500, 720, 999, 2000])
def test_quotas_sum_to_the_target_and_cover_every_family(target):
    q = quotas(target)
    assert sum(q.values()) == target
    assert list(q) == FAMILY_ORDER or set(q) == set(FAMILY_ORDER)
    assert all(v > 0 for v in q.values())


def test_records_are_deterministic_and_seeded():
    assert build_records(DEFAULT_TARGET, 0) == build_records(DEFAULT_TARGET, 0)
    assert build_records(DEFAULT_TARGET, 0) != build_records(DEFAULT_TARGET, 1)


def test_default_dataset_has_the_quota_per_family_and_unique_ids():
    records = build_records()
    assert len(records) == DEFAULT_TARGET
    assert len({r["id"] for r in records}) == len(records)
    q = quotas(DEFAULT_TARGET)
    for fam in FAMILY_ORDER:
        assert sum(r["family"] == fam for r in records) == q[fam]


def test_every_shape_gets_its_default_config_first():
    records = build_records()
    first_per_task = {}
    for r in records:
        first_per_task.setdefault(r["task_name"], r["config"])
    from kernelgen.families import get_family

    for r in records:
        fam = get_family(r["family"])
        assert first_per_task[r["task_name"]] == fam.default_config(r["task"]["params"])


def test_generated_records_pass_every_structural_check():
    assert check_records(build_records(), MIN_RECORDS, MAX_RECORDS) == []


def test_heldout_tasks_are_50_unique_and_disjoint_from_training():
    held = heldout_tasks()
    assert len(held) == 50 and len({t["id"] for t in held}) == 50
    train_names = {r["task_name"] for r in build_records()}
    assert not train_names & {t["id"] for t in held}


def test_committed_dataset_is_up_to_date_with_the_templates():
    # If this fails, run `make dataset` and commit the result.
    assert read_jsonl(DATA / "kernels.jsonl") == build_records()
    assert json.loads((DATA / "heldout_tasks.json").read_text()) == heldout_tasks()


def test_jsonl_round_trip(tmp_path):
    records = build_records(500)[:5]
    write_jsonl(records, tmp_path / "nested" / "k.jsonl")
    assert read_jsonl(tmp_path / "nested" / "k.jsonl") == records


def test_cli_writes_both_files(tmp_path, capsys):
    out, held = tmp_path / "k.jsonl", tmp_path / "h.json"
    assert dataset.main(["--out", str(out), "--heldout-out", str(held), "--target", "500"]) == 0
    assert len(read_jsonl(out)) == 500 and len(json.loads(held.read_text())) == 50
    assert "distinct sources" in capsys.readouterr().out


@pytest.mark.parametrize("target", ["499", "2001"])
def test_cli_rejects_a_target_outside_the_required_range(target):
    with pytest.raises(SystemExit):
        dataset.main(["--target", target])
