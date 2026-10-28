import json

import pytest

from judge_calibrate.cli import main
from judge_calibrate.data import load_items
from judge_calibrate.report import summarize_probes, to_jsonable


@pytest.fixture()
def data(tmp_path):
    path = tmp_path / "data.jsonl"
    assert main(["synth", "--n", "80", "--out", str(path)]) == 0
    return path


def test_probe_writes_report_and_summary(data, tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = main(["probe", str(data), "--fake", "position_bias=0.9", "--family", "claude",
               "--n-boot", "100", "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "position bias" in text and "flip rate" in text and "differs from" in text
    report = json.loads(out.read_text())
    pos = report["probes"]["position"]
    assert pos["logit_shift"]["value"] == pytest.approx(0.9)
    assert report["judge"]["family"] == "claude"


def test_probe_subset_and_mode(data, capsys):
    rc = main(["probe", str(data), "--probes", "length,scale", "--mode", "pointwise",
               "--n-boot", "50"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "scale use" in text and "position bias" not in text


def test_calibrate_save_and_apply(data, tmp_path, capsys):
    model = tmp_path / "m.json"
    rep = tmp_path / "cal.json"
    fake = ["--fake", "compression=0.5", "--fake", "anchor=7", "--fake",
            "point_length_bias=1"]
    rc = main(["calibrate", str(data), *fake, "--method", "isotonic", "--length-control",
               "--save", str(model), "--out", str(rep)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "cross-validation" in text and "length control" in text
    cv = json.loads(rep.read_text())["cross_validation"]
    assert cv["calibrated"]["mae"] < cv["raw"]["mae"]

    scored = tmp_path / "s.jsonl"
    assert main(["apply", str(model), str(data), *fake, "--out", str(scored)]) == 0
    items = load_items(scored)
    c = items[0].candidates[0]
    assert c.judge_score is not None and "calibrated_score" in c.meta

    # re-calibrating from the scores now stored in the file needs no judge
    assert main(["calibrate", str(scored), "--use-data-scores", "--method", "platt",
                 "--quiet"]) == 0


def test_errors_return_code_2(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "q"}\n')
    assert main(["probe", str(bad)]) == 2
    assert "error" in capsys.readouterr().err
    unlabeled = tmp_path / "u.jsonl"
    unlabeled.write_text('{"prompt": "q", "candidates": ["a", "b"]}\n')
    assert main(["calibrate", str(unlabeled)]) == 2
    assert main(["probe", str(unlabeled), "--fake", "nope=1"]) == 2


def test_summary_handles_skipped_and_nan():
    report = {"judge": {"name": "j", "family": None, "scale": [1, 10]}, "n_items": 1,
              "config": {"n_boot": 10, "alpha": 0.1},
              "probes": {"self_preference": {"skipped": "judge has no model family set"}}}
    text = summarize_probes(report)
    assert "skipped: judge has no model family set" in text and "90%" in text
    assert to_jsonable({"x": float("nan")}) == {"x": None}
