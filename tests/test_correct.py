import numpy as np
import pytest

from judge_calibrate.correct import (
    CalibrationModel,
    Calibrator,
    LengthControl,
    PositionDebiased,
    cross_validate,
    debiased_preference,
    isotonic_fit,
    length_controlled_preference,
)
from judge_calibrate.padding import word_count
from judge_calibrate.probes import ProbeConfig, probe_position
from judge_calibrate.providers import BiasConfig, FakeJudge
from judge_calibrate.synth import make_synthetic


@pytest.fixture(scope="module")
def items():
    return make_synthetic(n_items=200, human_noise=0.3, seed=4)


def _pointwise(judge, items):
    rows = [(it, c) for it in items for c in it.candidates]
    s = np.array([judge.judge_pointwise(it.prompt, c) for it, c in rows])
    h = np.array([c.human_score for _, c in rows])
    n = np.array([word_count(c.text) for _, c in rows])
    g = [it.id for it, _ in rows]
    return s, h, n, g


def test_debiased_preference_cancels_position_bias(items):
    biased = FakeJudge(BiasConfig(position_bias=1.5))
    clean = FakeJudge()
    a, b = items[0].candidates
    p = debiased_preference(biased, "p", a, b)
    assert p + debiased_preference(biased, "p", b, a) == pytest.approx(1.0)
    assert (p > 0.5) == (clean.judge_pairwise("p", a, b) > 0.5)
    rep = probe_position(PositionDebiased(biased), items, ProbeConfig(n_boot=50))
    assert rep["flip_rate"]["value"] == 0.0
    assert rep["first_position_win_rate"]["value"] == pytest.approx(0.5)


def test_isotonic_fit_pav():
    kx, ky = isotonic_fit([1, 2, 3, 4, 5], [1, 3, 2, 4, 3.5])
    assert np.all(np.diff(ky) >= 0)
    cal = Calibrator.fit([1, 2, 3, 4, 5], [1, 3, 2, 4, 3.5], "isotonic", (1, 10))
    assert cal.predict([2, 3]).tolist() == pytest.approx([2.5, 2.5])
    assert cal.predict([4, 5]).tolist() == pytest.approx([3.75, 3.75])
    assert cal.predict([0, 99]).tolist() == pytest.approx([1.0, 3.75])
    # ties in x are pooled before PAV
    kx, ky = isotonic_fit([1, 1, 2], [0, 2, 5])
    assert ky.tolist() == [1.0, 5.0]


def test_platt_and_linear_undo_compression():
    rng = np.random.default_rng(0)
    h = rng.uniform(1, 10, 500)
    s = 7 + 0.3 * (h - 7)
    for method in ("linear", "platt"):
        cal = Calibrator.fit(s, h, method, (1, 10))
        mae = np.abs(cal.predict(s) - h).mean()
        assert mae < (0.05 if method == "linear" else 0.5), method
    with pytest.raises(ValueError):
        Calibrator.fit(s, h, "magic")


def test_length_control_removes_only_spurious_length_effect():
    rng = np.random.default_rng(1)
    n = 2000
    loglen = rng.uniform(2, 5, n)
    human = 3 + 1.0 * loglen + rng.normal(0, 1, n)  # longer answers really are better
    score = human + 0.8 * loglen
    lc = LengthControl.fit(score, np.exp(loglen), human)
    assert lc.coef == pytest.approx(0.8, abs=1e-6)
    naive = LengthControl.fit(score, np.exp(loglen))
    assert naive.coef > 1.5  # without the human control real quality is removed too
    adj = lc.transform(score, np.exp(loglen))
    assert np.std(adj - human) < 1e-6  # remaining gap is a constant offset


def test_length_controlled_pairwise():
    rng = np.random.default_rng(2)
    la, lb = rng.integers(10, 200, 400), rng.integers(10, 200, 400)
    dq = rng.normal(0, 1, 400)
    z = 2 * dq + 0.7 * np.log(la / lb)
    p_adj, coef = length_controlled_preference(1 / (1 + np.exp(-z)), la, lb, dq)
    assert coef == pytest.approx(0.7, abs=1e-3)
    assert np.allclose(p_adj, 1 / (1 + np.exp(-2 * dq)), atol=1e-3)


def test_calibration_model_fixes_biased_judge(items, tmp_path):
    judge = FakeJudge(BiasConfig(compression=0.4, anchor=7.5, point_length_bias=1.2))
    s, h, n, g = _pointwise(judge, items)
    cv = cross_validate(s, h, n, g, method="isotonic", length_control=True)
    assert cv["calibrated"]["mae"] < 0.5 * cv["raw"]["mae"]
    assert cv["calibrated"]["kappa_quadratic"] > cv["raw"]["kappa_quadratic"]
    assert abs(cv["calibrated"]["residual_length_spearman"]) < abs(
        cv["raw"]["residual_length_spearman"])
    no_lc = cross_validate(s, h, n, g, method="isotonic")
    assert cv["calibrated"]["mae"] < no_lc["calibrated"]["mae"]

    model = CalibrationModel.fit(s, h, n, "platt", length_control=True, judge="fake")
    path = tmp_path / "m.json"
    model.save(path)
    again = CalibrationModel.load(path)
    assert np.allclose(again.predict(s, n), model.predict(s, n))
    assert again.length is not None and again.judge == "fake"
    with pytest.raises(ValueError):
        again.predict(s)


def test_cross_validate_folds_by_group():
    s = np.arange(20.0)
    cv = cross_validate(s, s / 2.2 + 1, groups=np.repeat(np.arange(4), 5), k=10,
                        method="linear")
    assert cv["k"] == 4
    assert cv["calibrated"]["mae"] == pytest.approx(0.0, abs=1e-6)
