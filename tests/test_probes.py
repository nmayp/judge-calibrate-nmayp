import pytest

from judge_calibrate.data import Candidate, Item
from judge_calibrate.probes import (
    ProbeConfig,
    probe_agreement,
    probe_length,
    probe_position,
    probe_scale,
    probe_self_preference,
    run_probes,
)
from judge_calibrate.providers import BiasConfig, FakeJudge
from judge_calibrate.synth import make_synthetic

CFG = ProbeConfig(n_boot=200)


@pytest.fixture(scope="module")
def items():
    return make_synthetic(n_items=150, human_noise=0.0, integer_human=False, seed=1)


def judge(**biases):
    family = biases.pop("family", "claude")
    return FakeJudge(BiasConfig(**biases), family=family)


def test_unbiased_judge_shows_no_position_or_length_bias(items):
    j = judge()
    pos = probe_position(j, items, CFG)
    assert pos["first_position_win_rate"]["value"] == pytest.approx(0.5)
    assert pos["flip_rate"]["value"] == 0.0
    assert pos["logit_shift"]["value"] == pytest.approx(0.0, abs=1e-9)
    ln = probe_length(j, items, CFG)
    assert ln["pairwise"]["padded_win_rate"]["value"] == pytest.approx(0.5)
    assert ln["pointwise"]["score_diff"]["value"] == pytest.approx(0.0)


@pytest.mark.parametrize("bias", [0.4, 1.2, -0.7])
def test_position_probe_recovers_injected_shift(items, bias):
    pos = probe_position(judge(position_bias=bias), items, CFG)
    assert pos["logit_shift"]["value"] == pytest.approx(bias, abs=1e-6)
    assert pos["first_position_win_rate"]["significant"]
    assert (pos["first_position_win_rate"]["value"] > 0.5) == (bias > 0)
    assert pos["flip_rate"]["value"] > 0


def test_position_probe_with_noise_and_hard_verdicts(items):
    pos = probe_position(judge(position_bias=1.0, noise=0.3), items, CFG)
    assert pos["logit_shift"]["value"] == pytest.approx(1.0, abs=0.1)
    lo, hi = pos["logit_shift"]["ci"]
    assert lo < 1.0 < hi or abs(pos["logit_shift"]["value"] - 1.0) < 0.05
    hard = probe_position(judge(position_bias=1.5, decisive=True), items, CFG)
    assert not hard["soft_probabilities"] and "logit_shift" not in hard
    assert hard["first_position_win_rate"]["value"] > 0.55
    assert hard["flip_rate"]["value"] > 0.2


def test_flip_rate_grows_with_position_bias(items):
    rates = [probe_position(judge(position_bias=b), items, CFG)["flip_rate"]["value"]
             for b in (0.0, 0.5, 1.5, 3.0)]
    assert rates == sorted(rates) and rates[-1] > rates[1]


def test_length_probe_recovers_pairwise_and_pointwise_slopes(items):
    ln = probe_length(judge(length_bias=0.6, point_length_bias=0.8), items, CFG)
    assert ln["pairwise"]["logit_per_log_length"]["value"] == pytest.approx(0.6, abs=1e-6)
    assert ln["pairwise"]["padded_win_rate"]["value"] > 0.5
    # pointwise scores clip at the top of the scale, so allow some attenuation
    assert ln["pointwise"]["score_per_log_length"]["value"] == pytest.approx(0.8, abs=0.08)
    assert ln["pointwise"]["observational_slope_given_human"]["value"] == pytest.approx(
        0.8, abs=0.15)


def test_length_probe_is_not_fooled_by_position_bias(items):
    ln = probe_length(judge(position_bias=1.0), items, CFG)
    assert ln["pairwise"]["padded_win_rate"]["value"] == pytest.approx(0.5)


def test_self_preference_recovered_given_labels(items):
    sp = probe_self_preference(judge(self_bias=0.8, point_self_bias=1.0), items, CFG)
    assert sp["pairwise"]["logit_shift_given_human"]["value"] == pytest.approx(0.8, abs=0.1)
    assert sp["pairwise"]["excess_win_rate"]["significant"]
    assert sp["pointwise"]["score_shift_given_human"]["value"] == pytest.approx(1.0, abs=0.15)


def test_self_preference_null_when_absent(items):
    sp = probe_self_preference(judge(position_bias=0.5), items, CFG)
    assert sp["pairwise"]["logit_shift_given_human"]["value"] == pytest.approx(0.0, abs=0.1)
    assert abs(sp["pointwise"]["score_shift_given_human"]["value"]) < 0.05


def test_self_preference_skips_without_family_or_sources(items):
    assert "skipped" in probe_self_preference(judge(family=None), items, CFG)
    unlabeled = make_synthetic(n_items=20, sources=(), seed=2)
    sp = probe_self_preference(judge(), unlabeled, CFG)
    assert "skipped" in sp["pairwise"] and "skipped" in sp["pointwise"]


def test_scale_probe_recovers_compression_and_anchor(items):
    sc = probe_scale(judge(compression=0.4, anchor=7.0), items, CFG)
    assert sc["compression_slope"]["value"] == pytest.approx(0.4, abs=0.02)
    assert sc["anchor"]["value"] == pytest.approx(7.0, abs=0.2)
    assert sc["compression_slope"]["significant"]
    full = probe_scale(judge(), items, CFG)
    assert full["compression_slope"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert full["normalized_entropy"] > sc["normalized_entropy"]
    assert full["range_used"] > sc["range_used"]


def test_agreement_metrics(items):
    ag = probe_agreement(judge(), items, CFG)
    assert ag["pointwise"]["spearman"]["value"] == pytest.approx(1.0)
    assert ag["pointwise"]["mae"]["value"] == pytest.approx(0.0, abs=1e-9)
    biased = probe_agreement(judge(position_bias=2.0), items, CFG)["pairwise"]
    assert biased["accuracy_both_orders"]["value"] > biased["accuracy_single_order"]["value"]
    assert biased["kappa_both_orders"]["value"] > biased["kappa_single_order"]["value"]


def test_agreement_skips_without_labels():
    unlabeled = [Item(str(k), "p", [Candidate(f"answer {k}."), Candidate(f"other {k}.")])
                 for k in range(5)]
    ag = probe_agreement(judge(), unlabeled, CFG)
    assert "skipped" in ag["pointwise"] and "skipped" in ag["pairwise"]


def test_run_probes_report_and_modes(items):
    rep = run_probes(judge(position_bias=0.5), items[:30], CFG)
    assert set(rep["probes"]) == {"position", "length", "self_preference", "scale",
                                  "agreement"}
    assert rep["judge"]["family"] == "claude"
    only_point = run_probes(judge(), items[:30], ProbeConfig(n_boot=50, modes=("pointwise",)))
    assert "skipped" in only_point["probes"]["position"]
    assert "pairwise" not in only_point["probes"]["length"]
    with pytest.raises(ValueError):
        run_probes(judge(), items, CFG, probes=["bogus"])


def test_run_probes_reuses_judgments(items):
    j = judge()
    run_probes(j, items[:20], ProbeConfig(n_boot=20))
    # per item: 2 ordered pairs, 4 padded pairs, 2 original + 2 padded pointwise scores
    assert j.calls == 20 * 10
