import json

import pytest

from judge_calibrate.data import DataError, load_items, write_items


def _write(tmp_path, rows):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def test_load_full_and_shorthand(tmp_path):
    p = _write(tmp_path, [
        {"id": "a", "prompt": "q", "candidates": [
            {"text": "x", "source": "claude", "human_score": 7, "extra": 1}, "y"],
         "human_preference": "B"},
        {"prompt": "q2", "answer_a": "u", "answer_b": "v"},
    ])
    items = load_items(p)
    assert items[0].candidates[0].meta == {"extra": 1}
    assert items[0].candidates[1].source is None
    assert items[0].human_outcome(0, 1) == 0.0
    assert items[0].human_outcome(1, 0) == 1.0
    assert items[1].id == "1"
    assert len(items[1].candidates) == 2


def test_human_outcome_from_scores():
    from judge_calibrate.data import Candidate, Item

    it = Item("i", "p", [Candidate("a", human_score=3), Candidate("b", human_score=5),
                         Candidate("c", human_score=5)])
    assert it.human_outcome(0, 1) == 0.0
    assert it.human_outcome(1, 2) == 0.5
    assert Item("j", "p", [Candidate("a"), Candidate("b")]).human_outcome(0, 1) is None


def test_roundtrip(tmp_path):
    p = _write(tmp_path, [{"id": "a", "prompt": "q", "candidates": ["x", "y"],
                           "human_preference": "tie"}])
    items = load_items(p)
    out = tmp_path / "o.jsonl"
    write_items(items, out)
    again = load_items(out)
    assert again[0].to_dict() == items[0].to_dict()


@pytest.mark.parametrize("row", [
    {"candidates": ["x"]},
    {"prompt": "q", "candidates": []},
    {"prompt": "q", "candidates": [{"txt": "x"}]},
    {"prompt": "q", "candidates": ["x", "y"], "human_preference": 3},
    {"prompt": "q", "candidates": [{"text": "x", "human_score": "high"}]},
])
def test_invalid_rows(tmp_path, row):
    with pytest.raises(DataError):
        load_items(_write(tmp_path, [row]))


def test_duplicate_ids(tmp_path):
    with pytest.raises(DataError):
        load_items(_write(tmp_path, [{"id": 1, "prompt": "q", "candidates": ["x"]},
                                     {"id": 1, "prompt": "q", "candidates": ["x"]}]))
