"""Corrections: order averaging, length residualization, isotonic/Platt/linear calibration."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import stats
from .data import Candidate
from .providers import Provider


def debiased_preference(provider: Provider, prompt: str, a: Candidate, b: Candidate) -> float:
    """P(a beats b) averaged over both presentation orders; cancels an additive slot bias."""
    return 0.5 * (provider.judge_pairwise(prompt, a, b)
                  + 1.0 - provider.judge_pairwise(prompt, b, a))


class PositionDebiased:
    """Provider wrapper whose pairwise judgments are order-averaged."""

    def __init__(self, inner: Provider):
        self.inner = inner
        self.name = f"{inner.name}+swap"
        self.family = inner.family
        self.scale = inner.scale

    def judge_pairwise(self, prompt: str, first: Candidate, second: Candidate) -> float:
        return debiased_preference(self.inner, prompt, first, second)

    def judge_pointwise(self, prompt: str, candidate: Candidate) -> float:
        return self.inner.judge_pointwise(prompt, candidate)


@dataclass
class LengthControl:
    """Remove the linear effect of log word count from judge scores.

    Fit ``score = b0 + b1*log(len) [+ b2*human]``; the adjusted score is
    ``score - b1*(log(len) - ref)``, keeping the mean at the reference length. Including the
    human score as a control means only length effects *beyond* true quality are removed.
    """

    coef: float = 0.0
    ref: float = 0.0
    controls_human: bool = False

    @classmethod
    def fit(cls, scores: Sequence[float], lengths: Sequence[float],
            human: Sequence[float] | None = None) -> LengthControl:
        s = np.asarray(scores, float)
        loglen = np.log(np.maximum(np.asarray(lengths, float), 1.0))
        cols = [np.ones_like(s), loglen]
        if human is not None:
            cols.append(np.asarray(human, float))
        X = np.column_stack(cols)
        beta = stats.ols(X, s, ridge=1e-9)
        return cls(coef=float(beta[1]), ref=float(loglen.mean()),
                   controls_human=human is not None)

    def transform(self, scores: Sequence[float], lengths: Sequence[float]) -> np.ndarray:
        loglen = np.log(np.maximum(np.asarray(lengths, float), 1.0))
        return np.asarray(scores, float) - self.coef * (loglen - self.ref)

    def to_dict(self) -> dict:
        return {"coef": self.coef, "ref": self.ref, "controls_human": self.controls_human}


def length_controlled_preference(p: Sequence[float], len_a: Sequence[float],
                                 len_b: Sequence[float],
                                 human_diff: Sequence[float] | None = None
                                 ) -> tuple[np.ndarray, float]:
    """Pairwise analogue: regress logit(p) on log-length ratio and subtract that term.

    Returns (adjusted probabilities, fitted logit-per-log-length coefficient).
    """
    z = stats.logit(p)
    dl = np.log(np.asarray(len_a, float) / np.asarray(len_b, float))
    cols = [np.ones_like(z), dl]
    if human_diff is not None:
        cols.append(np.asarray(human_diff, float))
    beta = stats.ols(np.column_stack(cols), z, ridge=1e-9)
    return stats.sigmoid(z - beta[1] * dl), float(beta[1])


def isotonic_fit(x: Sequence[float], y: Sequence[float],
                 w: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators. Returns (knots_x, knots_y) of the non-decreasing fit."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    order = np.argsort(x, kind="mergesort")
    x, y, w = x[order], y[order], w[order]
    ux, start = np.unique(x, return_index=True)
    wsum = np.add.reduceat(w, start)
    ymean = np.add.reduceat(w * y, start) / wsum
    # blocks: [weighted mean, weight, first x index, last x index]
    blocks: list[list[float]] = []
    for k in range(len(ux)):
        blocks.append([ymean[k], wsum[k], k, k])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, s2, _ = blocks[-2]
            m1, w1, _, e1 = blocks.pop()
            blocks[-1] = [(m2 * w2 + m1 * w1) / (w1 + w2), w1 + w2, s2, e1]
    kx, ky = [], []
    for m, _, s, e in blocks:
        kx.append(ux[int(s)])
        ky.append(m)
        if e != s:
            kx.append(ux[int(e)])
            ky.append(m)
    return np.asarray(kx), np.asarray(ky)


@dataclass
class Calibrator:
    """Monotone map from (possibly length-controlled) judge score to the human scale."""

    method: str
    scale: tuple[float, float]
    params: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, x: Sequence[float], y: Sequence[float], method: str = "isotonic",
            scale: tuple[float, float] = (1.0, 10.0)) -> Calibrator:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) < 2:
            raise ValueError("need at least two labelled scores to calibrate")
        if method == "isotonic":
            kx, ky = isotonic_fit(x, y)
            params = {"x": kx.tolist(), "y": ky.tolist()}
        elif method == "platt":
            params = _fit_platt(x, y, scale)
        elif method == "linear":
            slope, intercept = stats.linear_fit(x, y) if np.ptp(x) > 0 else (0.0, y.mean())
            params = {"slope": slope, "intercept": intercept}
        else:
            raise ValueError(f"unknown calibration method {method!r}")
        return cls(method, (float(scale[0]), float(scale[1])), params)

    def predict(self, x: Sequence[float]) -> np.ndarray:
        x = np.asarray(x, float)
        lo, hi = self.scale
        p = self.params
        if self.method == "isotonic":
            out = np.interp(x, p["x"], p["y"])
        elif self.method == "platt":
            z = (x - p["mu"]) / p["sigma"]
            out = lo + (hi - lo) * stats.sigmoid(p["a"] * z + p["b"])
        else:
            out = p["slope"] * x + p["intercept"]
        return np.clip(out, lo, hi)

    def to_dict(self) -> dict:
        return {"method": self.method, "scale": list(self.scale), "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> Calibrator:
        return cls(d["method"], tuple(d["scale"]), dict(d["params"]))


def _fit_platt(x: np.ndarray, y: np.ndarray, scale: tuple[float, float],
               iters: int = 100, ridge: float = 1e-6) -> dict:
    """Logistic fit with fractional targets t=(y-lo)/(hi-lo) by Newton's method (IRLS)."""
    lo, hi = scale
    t = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    mu, sigma = float(x.mean()), float(x.std()) or 1.0
    z = (x - mu) / sigma
    X = np.column_stack([z, np.ones_like(z)])
    beta = np.array([1.0, float(stats.logit(t.mean(), eps=1e-3))])
    for _ in range(iters):
        p = stats.sigmoid(X @ beta)
        grad = X.T @ (p - t) + ridge * beta
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(2)
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.abs(step).max() < 1e-10:
            break
    return {"a": float(beta[0]), "b": float(beta[1]), "mu": mu, "sigma": sigma}


@dataclass
class CalibrationModel:
    """Optional length control followed by a calibrator; JSON-serialisable."""

    calibrator: Calibrator
    length: LengthControl | None = None
    judge: str | None = None

    @classmethod
    def fit(cls, scores: Sequence[float], human: Sequence[float],
            lengths: Sequence[float] | None = None, method: str = "isotonic",
            scale: tuple[float, float] = (1.0, 10.0), length_control: bool = False,
            judge: str | None = None) -> CalibrationModel:
        s = np.asarray(scores, float)
        lc = None
        if length_control:
            if lengths is None:
                raise ValueError("length control needs answer lengths")
            lc = LengthControl.fit(s, lengths, human)
            s = lc.transform(s, lengths)
        return cls(Calibrator.fit(s, human, method, scale), lc, judge)

    def predict(self, scores: Sequence[float],
                lengths: Sequence[float] | None = None) -> np.ndarray:
        s = np.asarray(scores, float)
        if self.length is not None:
            if lengths is None:
                raise ValueError("this model applies length control; pass lengths")
            s = self.length.transform(s, lengths)
        return self.calibrator.predict(s)

    def to_dict(self) -> dict:
        return {"calibrator": self.calibrator.to_dict(),
                "length_control": None if self.length is None else self.length.to_dict(),
                "judge": self.judge}

    @classmethod
    def from_dict(cls, d: dict) -> CalibrationModel:
        lc = d.get("length_control")
        return cls(Calibrator.from_dict(d["calibrator"]),
                   None if lc is None else LengthControl(**lc), d.get("judge"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def score_metrics(pred: Sequence[float], human: Sequence[float],
                  lengths: Sequence[float] | None = None,
                  scale: tuple[float, float] = (1.0, 10.0)) -> dict:
    pred = np.asarray(pred, float)
    human = np.asarray(human, float)
    lo, hi = scale
    labels = list(range(math.ceil(lo), math.floor(hi) + 1))
    pr = np.clip(np.round(pred), labels[0], labels[-1]).astype(int)
    hr = np.clip(np.round(human), labels[0], labels[-1]).astype(int)
    err = pred - human
    out = {
        "mae": float(np.abs(err).mean()),
        "rmse": float(math.sqrt((err**2).mean())),
        "mean_bias": float(err.mean()),
        "spearman": stats.spearman(pred, human),
        "pearson": stats.pearson(pred, human),
        "kappa_quadratic": stats.cohen_kappa(list(hr), list(pr), "quadratic", labels),
    }
    if lengths is not None:
        out["residual_length_spearman"] = stats.spearman(err, np.log(lengths))
    return {k: (None if not math.isfinite(v) else round(v, 6)) for k, v in out.items()}


def cross_validate(scores: Sequence[float], human: Sequence[float],
                   lengths: Sequence[float] | None = None, groups: Sequence | None = None,
                   method: str = "isotonic", scale: tuple[float, float] = (1.0, 10.0),
                   length_control: bool = False, k: int = 5, seed: int = 0) -> dict:
    """Out-of-fold metrics for raw vs calibrated scores, folding by group (item)."""
    s = np.asarray(scores, float)
    h = np.asarray(human, float)
    L = None if lengths is None else np.asarray(lengths, float)
    g = np.arange(len(s)) if groups is None else np.asarray(groups)
    uniq = np.unique(g)
    k = max(2, min(k, len(uniq)))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq.tolist(), rng.permutation(len(uniq)) % k, strict=True))
    folds = np.array([fold_of[v] for v in g.tolist()])
    pred = np.empty_like(s)
    for f in range(k):
        test = folds == f
        train = ~test
        model = CalibrationModel.fit(s[train], h[train], None if L is None else L[train],
                                     method, scale, length_control)
        pred[test] = model.predict(s[test], None if L is None else L[test])
    return {"k": k, "raw": score_metrics(s, h, L, scale),
            "calibrated": score_metrics(pred, h, L, scale)}
