"""Statistics: bootstrap CIs, rank correlation, kappa, regression helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def logit(p, eps: float = 1e-3):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def rankdata(x: ArrayLike) -> np.ndarray:
    """Ranks starting at 1 with ties assigned their average rank."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(x: ArrayLike, y: ArrayLike) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = math.sqrt(float(xc @ xc) * float(yc @ yc))
    return float(xc @ yc / denom) if denom > 0 else float("nan")


def spearman(x: ArrayLike, y: ArrayLike) -> float:
    return pearson(rankdata(x), rankdata(y))


def cohen_kappa(a: Sequence, b: Sequence, weights: str | None = None,
                labels: Sequence | None = None) -> float:
    """Cohen's kappa; weights in {None, 'linear', 'quadratic'} for ordinal labels."""
    if len(a) != len(b):
        raise ValueError("rating sequences differ in length")
    if not len(a):
        return float("nan")
    cats = sorted(set(a) | set(b)) if labels is None else list(labels)
    index = {c: k for k, c in enumerate(cats)}
    k = len(cats)
    if k < 2:
        return float("nan")
    obs = np.zeros((k, k))
    for u, v in zip(a, b, strict=True):
        obs[index[u], index[v]] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    grid = np.arange(k)
    dist = np.abs(grid[:, None] - grid[None, :]).astype(float)
    if weights is None:
        w = (dist > 0).astype(float)
    elif weights == "linear":
        w = dist / (k - 1)
    elif weights == "quadratic":
        w = (dist / (k - 1)) ** 2
    else:
        raise ValueError(f"unknown weights {weights!r}")
    denom = float((w * exp).sum())
    if denom == 0:
        return float("nan")
    return 1.0 - float((w * obs).sum()) / denom


def ols(X: np.ndarray, y: ArrayLike, ridge: float = 0.0) -> np.ndarray:
    """Least-squares coefficients; X should already contain an intercept column if wanted."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if ridge:
        A = X.T @ X + ridge * np.eye(X.shape[1])
        return np.linalg.solve(A, X.T @ y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def linear_fit(x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
    """Return (slope, intercept) of y on x."""
    x = np.asarray(x, dtype=float)
    X = np.column_stack([x, np.ones_like(x)])
    slope, intercept = ols(X, y)
    return float(slope), float(intercept)


def normalized_entropy(values: ArrayLike, bins: Sequence[float]) -> float:
    """Shannon entropy of the histogram over ``bins`` (bin centres), divided by log(len(bins))."""
    values = np.asarray(values, dtype=float)
    bins = np.asarray(bins, dtype=float)
    if len(bins) < 2 or not len(values):
        return float("nan")
    idx = np.abs(values[:, None] - bins[None, :]).argmin(axis=1)
    p = np.bincount(idx, minlength=len(bins)) / len(values)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(bins)))


def cohens_dz(diffs: ArrayLike) -> float:
    """Standardised mean of paired differences."""
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd == 0:
        return float("inf") * np.sign(d.mean()) if d.mean() else 0.0
    return float(d.mean() / sd)


@dataclass
class Estimate:
    value: float
    lo: float
    hi: float
    n: int
    null: float | None = None

    @property
    def significant(self) -> bool | None:
        if self.null is None or any(math.isnan(v) for v in (self.lo, self.hi)):
            return None
        return not (self.lo <= self.null <= self.hi)

    def to_dict(self) -> dict:
        out = {"value": _clean(self.value), "ci": [_clean(self.lo), _clean(self.hi)],
               "n": self.n}
        if self.null is not None:
            out["null"] = self.null
            out["significant"] = self.significant
        return out


def _clean(v: float) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), 6)


def bootstrap(stat: Callable[[np.ndarray], float], n: int, groups: Sequence | None = None,
              n_boot: int = 1000, alpha: float = 0.05, seed: int = 0,
              null: float | None = None) -> Estimate:
    """Percentile bootstrap of ``stat(indices)``; resamples whole groups when given (cluster)."""
    all_idx = np.arange(n)
    value = float(stat(all_idx)) if n else float("nan")
    if n < 2 or n_boot <= 0:
        return Estimate(value, float("nan"), float("nan"), n, null)
    rng = np.random.default_rng(seed)
    if groups is None:
        clusters = [np.array([k]) for k in range(n)]
    else:
        keys: dict = {}
        for k, g in enumerate(groups):
            keys.setdefault(g, []).append(k)
        clusters = [np.array(v) for v in keys.values()]
    m = len(clusters)
    samples = []
    for _ in range(n_boot):
        pick = rng.integers(0, m, size=m)
        idx = np.concatenate([clusters[p] for p in pick])
        with np.errstate(all="ignore"):
            v = stat(idx)
        if v is not None and math.isfinite(v):
            samples.append(v)
    if len(samples) < max(10, n_boot // 10):
        return Estimate(value, float("nan"), float("nan"), n, null)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return Estimate(value, float(lo), float(hi), n, null)
