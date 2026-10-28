import math

import numpy as np
import pytest

from judge_calibrate import stats


def test_rankdata_ties():
    assert stats.rankdata([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]


def test_spearman_monotone_and_known_value():
    x = np.arange(20.0)
    assert stats.spearman(x, x**3) == pytest.approx(1.0)
    assert stats.spearman(x, -x) == pytest.approx(-1.0)
    # textbook example: d^2 sum = 2 for n=5 -> 1 - 6*2/(5*24) = 0.9
    assert stats.spearman([1, 2, 3, 4, 5], [2, 1, 3, 4, 5]) == pytest.approx(0.9)


def test_cohen_kappa_known_values():
    a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa 0.4
    assert stats.cohen_kappa(a, b) == pytest.approx(0.4)
    assert stats.cohen_kappa([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_weighted_kappa_penalises_far_disagreement_more():
    truth = [1, 2, 3, 4, 5] * 10
    near = [min(5, t + 1) for t in truth]
    far = [6 - t for t in truth]
    lab = [1, 2, 3, 4, 5]
    k_near = stats.cohen_kappa(truth, near, "quadratic", labels=lab)
    k_far = stats.cohen_kappa(truth, far, "quadratic", labels=lab)
    assert k_near > 0.7
    assert k_far == pytest.approx(-1.0)


def test_linear_fit_and_ols():
    x = np.linspace(0, 1, 50)
    slope, intercept = stats.linear_fit(x, 3 * x - 2)
    assert slope == pytest.approx(3.0)
    assert intercept == pytest.approx(-2.0)


def test_normalized_entropy():
    assert stats.normalized_entropy([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert stats.normalized_entropy([3, 3, 3], [1, 2, 3, 4]) == pytest.approx(0.0)


def test_bootstrap_covers_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 1.0, size=400)
    est = stats.bootstrap(lambda idx: x[idx].mean(), len(x), n_boot=500, null=0.0)
    assert est.lo < 2.0 < est.hi
    assert est.significant is True
    assert est.hi - est.lo == pytest.approx(2 * 1.96 / math.sqrt(400), rel=0.25)


def test_cluster_bootstrap_is_wider_for_correlated_groups():
    rng = np.random.default_rng(2)
    group_effect = np.repeat(rng.normal(0, 1, 40), 10)
    x = group_effect + rng.normal(0, 0.1, 400)
    groups = np.repeat(np.arange(40), 10)
    naive = stats.bootstrap(lambda i: x[i].mean(), 400, n_boot=400)
    clustered = stats.bootstrap(lambda i: x[i].mean(), 400, groups=groups, n_boot=400)
    assert (clustered.hi - clustered.lo) > 2 * (naive.hi - naive.lo)


def test_estimate_to_dict_handles_nan():
    d = stats.Estimate(float("nan"), float("nan"), 1.0, 0, null=0.0).to_dict()
    assert d["value"] is None
    assert d["significant"] is None
