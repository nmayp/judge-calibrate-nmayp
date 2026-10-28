"""Bias probes: position, length/verbosity, self-preference, scale use, human agreement."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from . import stats
from .data import Candidate, Item
from .padding import pad, word_count
from .providers import CachedProvider, Provider, same_family

ALL_PROBES = ("position", "length", "self_preference", "scale", "agreement")
MODES = ("pairwise", "pointwise")


@dataclass
class ProbeConfig:
    n_boot: int = 1000
    alpha: float = 0.05
    seed: int = 0
    pad_ratio: float = 2.0
    max_pairs_per_item: int = 1
    max_candidates_per_item: int = 2
    tie_tol: float = 1e-9
    modes: tuple[str, ...] = MODES

    def boot(self, stat, n, groups, null=None) -> stats.Estimate:
        return stats.bootstrap(stat, n, groups=groups, n_boot=self.n_boot, alpha=self.alpha,
                               seed=self.seed, null=null)


def _pairs(items: Sequence[Item], limit: int) -> Iterator[tuple[Item, int, int]]:
    for item in items:
        k = 0
        n = len(item.candidates)
        for i in range(n):
            for j in range(i + 1, n):
                if k >= limit:
                    break
                yield item, i, j
                k += 1


def _is_soft(ps: np.ndarray) -> bool:
    """True when the judge returns graded probabilities rather than hard verdicts."""
    if not len(ps):
        return False
    hard = np.isin(np.round(ps, 9), (0.0, 0.5, 1.0))
    return float(hard.mean()) < 0.5


def _verdict(p: np.ndarray, tol: float) -> np.ndarray:
    """+1 if the first-shown answer wins, -1 if the second wins, 0 for a tie."""
    return np.where(p > 0.5 + tol, 1, np.where(p < 0.5 - tol, -1, 0))


def _mean_stat(x: np.ndarray):
    return lambda idx: float(x[idx].mean())


def _dz_stat(x: np.ndarray):
    return lambda idx: stats.cohens_dz(x[idx])


def _ols_coef_stat(X: np.ndarray, y: np.ndarray, col: int):
    def stat(idx):
        Xi = X[idx]
        if np.linalg.matrix_rank(Xi) < Xi.shape[1]:
            return float("nan")
        return float(stats.ols(Xi, y[idx])[col])
    return stat


def _skipped(reason: str) -> dict:
    return {"skipped": reason}


def probe_position(provider: Provider, items: Sequence[Item], cfg: ProbeConfig) -> dict:
    """Judge every pair in both orders; any systematic preference for slot A is position bias."""
    ab, ba, groups = [], [], []
    for item, i, j in _pairs(items, cfg.max_pairs_per_item):
        a, b = item.candidates[i], item.candidates[j]
        ab.append(provider.judge_pairwise(item.prompt, a, b))
        ba.append(provider.judge_pairwise(item.prompt, b, a))
        groups.append(item.id)
    if not ab:
        return _skipped("no items with two or more candidates")
    p_ab, p_ba = np.asarray(ab), np.asarray(ba)
    n = len(p_ab)
    first_rate = 0.5 * (p_ab + p_ba)
    d = p_ab + p_ba - 1.0
    # A consistent judge picks the same candidate in both orders: v_ab == -v_ba.
    flips = (_verdict(p_ab, cfg.tie_tol) != -_verdict(p_ba, cfg.tie_tol)).astype(float)
    out = {
        "n_pairs": n,
        "first_position_win_rate": cfg.boot(_mean_stat(first_rate), n, groups, 0.5).to_dict(),
        "flip_rate": cfg.boot(_mean_stat(flips), n, groups, 0.0).to_dict(),
        "effect_size_dz": cfg.boot(_dz_stat(d), n, groups, 0.0).to_dict(),
        "soft_probabilities": _is_soft(np.concatenate([p_ab, p_ba])),
    }
    if out["soft_probabilities"]:
        shift = 0.5 * (stats.logit(p_ab) + stats.logit(p_ba))
        out["logit_shift"] = cfg.boot(_mean_stat(shift), n, groups, 0.0).to_dict()
    return out


def probe_length(provider: Provider, items: Sequence[Item], cfg: ProbeConfig) -> dict:
    """Pad answers with content-free filler; any score change is verbosity bias."""
    rows = []
    for item in items:
        for c in item.candidates[: cfg.max_candidates_per_item]:
            padded = Candidate(pad(c.text, cfg.pad_ratio), source=c.source)
            log_ratio = math.log(word_count(padded.text) / word_count(c.text))
            if log_ratio <= 0:
                continue
            rows.append((item, c, padded, log_ratio))
    if not rows:
        return _skipped("no candidates could be padded")
    groups = [r[0].id for r in rows]
    log_ratio = np.array([r[3] for r in rows])
    n = len(rows)
    out: dict = {"n_candidates": n, "pad_ratio": cfg.pad_ratio,
                 "mean_log_length_ratio": round(float(log_ratio.mean()), 6)}

    if "pairwise" in cfg.modes:
        p1 = np.array([provider.judge_pairwise(it.prompt, pd, c) for it, c, pd, _ in rows])
        p2 = np.array([provider.judge_pairwise(it.prompt, c, pd) for it, c, pd, _ in rows])
        pref = 0.5 * (p1 + 1.0 - p2)
        pw = {"padded_win_rate": cfg.boot(_mean_stat(pref), n, groups, 0.5).to_dict(),
              "effect_size_dz": cfg.boot(_dz_stat(pref - 0.5), n, groups, 0.0).to_dict(),
              "soft_probabilities": _is_soft(np.concatenate([p1, p2]))}
        if pw["soft_probabilities"]:
            slope = 0.5 * (stats.logit(p1) - stats.logit(p2)) / log_ratio
            pw["logit_per_log_length"] = cfg.boot(_mean_stat(slope), n, groups, 0.0).to_dict()
        out["pairwise"] = pw

    if "pointwise" in cfg.modes:
        s_orig = np.array([provider.judge_pointwise(it.prompt, c) for it, c, _, _ in rows])
        s_pad = np.array([provider.judge_pointwise(it.prompt, pd) for it, _, pd, _ in rows])
        diff = s_pad - s_orig
        pt = {"score_diff": cfg.boot(_mean_stat(diff), n, groups, 0.0).to_dict(),
              "score_per_log_length": cfg.boot(_mean_stat(diff / log_ratio), n, groups,
                                               0.0).to_dict(),
              "effect_size_dz": cfg.boot(_dz_stat(diff), n, groups, 0.0).to_dict()}
        # Observational check on the unpadded answers: does score track length beyond quality?
        loglen = np.log([word_count(c.text) for _, c, _, _ in rows])
        human = [c.human_score for _, c, _, _ in rows]
        pt["spearman_score_vs_length"] = cfg.boot(
            lambda idx: stats.spearman(s_orig[idx], loglen[idx]), n, groups, 0.0).to_dict()
        if all(h is not None for h in human):
            X = np.column_stack([np.ones(n), np.asarray(human, float), loglen])
            pt["observational_slope_given_human"] = cfg.boot(
                _ols_coef_stat(X, s_orig, 2), n, groups, 0.0).to_dict()
        out["pointwise"] = pt
    return out


def probe_self_preference(provider: Provider, items: Sequence[Item],
                          cfg: ProbeConfig) -> dict:
    """Compare own-family answers with others, controlling for human labels when present."""
    family = provider.family
    if not family:
        return _skipped("judge has no model family set")
    out: dict = {"family": family}

    if "pairwise" in cfg.modes:
        rows = []
        for item, i, j in _pairs(items, cfg.max_pairs_per_item):
            ci, cj = item.candidates[i], item.candidates[j]
            own_i, own_j = same_family(ci.source, family), same_family(cj.source, family)
            if own_i == own_j:
                continue
            own, other = (ci, cj) if own_i else (cj, ci)
            h = item.human_outcome(i, j) if own_i else item.human_outcome(j, i)
            rows.append((item, own, other, h))
        if rows:
            groups = [r[0].id for r in rows]
            n = len(rows)
            p1 = np.array([provider.judge_pairwise(it.prompt, o, t) for it, o, t, _ in rows])
            p2 = np.array([provider.judge_pairwise(it.prompt, t, o) for it, o, t, _ in rows])
            pref = 0.5 * (p1 + 1.0 - p2)
            pw: dict = {"n_pairs": n,
                        "own_win_rate": cfg.boot(_mean_stat(pref), n, groups, 0.5).to_dict(),
                        "soft_probabilities": _is_soft(np.concatenate([p1, p2]))}
            hs = [r[3] for r in rows]
            if all(h is not None for h in hs):
                h = np.asarray(hs, float)
                pw["human_own_win_rate"] = round(float(h.mean()), 6)
                pw["excess_win_rate"] = cfg.boot(_mean_stat(pref - h), n, groups,
                                                 0.0).to_dict()
                pw["effect_size_dz"] = cfg.boot(_dz_stat(pref - h), n, groups, 0.0).to_dict()
            if pw["soft_probabilities"]:
                L = 0.5 * (stats.logit(p1) - stats.logit(p2))
                dh = [None if o.human_score is None or t.human_score is None
                      else o.human_score - t.human_score for _, o, t, _ in rows]
                if all(v is not None for v in dh):
                    X = np.column_stack([np.ones(n), np.asarray(dh, float)])
                    pw["logit_shift_given_human"] = cfg.boot(
                        _ols_coef_stat(X, L, 0), n, groups, 0.0).to_dict()
                else:
                    pw["logit_shift_unadjusted"] = cfg.boot(_mean_stat(L), n, groups,
                                                            0.0).to_dict()
            out["pairwise"] = pw
        else:
            out["pairwise"] = _skipped("no pairs with exactly one own-family candidate")

    if "pointwise" in cfg.modes:
        rows = [(item, c) for item in items for c in item.candidates if c.source]
        own = np.array([same_family(c.source, family) for _, c in rows], dtype=float)
        if rows and 0 < own.sum() < len(own):
            groups = [it.id for it, _ in rows]
            n = len(rows)
            s = np.array([provider.judge_pointwise(it.prompt, c) for it, c in rows])
            human = [c.human_score for _, c in rows]
            if all(h is not None for h in human):
                X = np.column_stack([np.ones(n), np.asarray(human, float), own])
                key = "score_shift_given_human"
            else:
                X = np.column_stack([np.ones(n), own])
                key = "score_shift_unadjusted"
            col = X.shape[1] - 1
            out["pointwise"] = {"n_candidates": n, "n_own": int(own.sum()),
                                key: cfg.boot(_ols_coef_stat(X, s, col), n, groups,
                                              0.0).to_dict()}
        else:
            out["pointwise"] = _skipped("need both own-family and other labelled sources")
    return out


def _pointwise_rows(provider: Provider, items: Sequence[Item]):
    rows = [(item, c) for item in items for c in item.candidates]
    scores = np.array([provider.judge_pointwise(it.prompt, c) for it, c in rows])
    return rows, scores


def probe_scale(provider: Provider, items: Sequence[Item], cfg: ProbeConfig) -> dict:
    """How the judge uses its scale: spread, entropy, compression and anchor vs humans."""
    rows, s = _pointwise_rows(provider, items)
    if len(s) < 2:
        return _skipped("need at least two candidates")
    lo, hi = provider.scale
    groups = [it.id for it, _ in rows]
    n = len(s)
    bins = np.arange(math.ceil(lo), math.floor(hi) + 1)
    rounded = np.clip(np.round(s), lo, hi)
    _, counts = np.unique(rounded, return_counts=True)
    out: dict = {
        "n_candidates": n, "scale": [lo, hi],
        "mean": round(float(s.mean()), 6), "sd": round(float(s.std(ddof=1)), 6),
        "min": float(s.min()), "max": float(s.max()),
        "range_used": round(float((s.max() - s.min()) / (hi - lo)), 6),
        "normalized_entropy": round(stats.normalized_entropy(s, bins), 6),
        "mode_share": round(float(counts.max() / n), 6),
    }
    human = [c.human_score for _, c in rows]
    if all(h is not None for h in human):
        h = np.asarray(human, float)

        def slope(idx):
            if np.ptp(h[idx]) == 0:
                return float("nan")
            return stats.linear_fit(h[idx], s[idx])[0]

        def anchor(idx):
            if np.ptp(h[idx]) == 0:
                return float("nan")
            b, a = stats.linear_fit(h[idx], s[idx])
            # the fixed point is ill-determined when there is little compression
            return a / (1.0 - b) if abs(1.0 - b) > 0.1 else float("nan")

        out["compression_slope"] = cfg.boot(slope, n, groups, 1.0).to_dict()
        out["anchor"] = cfg.boot(anchor, n, groups).to_dict()
        out["sd_ratio"] = cfg.boot(
            lambda idx: float(s[idx].std() / h[idx].std()) if h[idx].std() > 0
            else float("nan"), n, groups, 1.0).to_dict()
        out["mean_shift"] = cfg.boot(_mean_stat(s - h), n, groups, 0.0).to_dict()
    return out


def pointwise_agreement(judge: np.ndarray, human: np.ndarray, scale: tuple[float, float],
                        groups: Sequence, cfg: ProbeConfig) -> dict:
    lo, hi = scale
    n = len(judge)
    labels = list(range(math.ceil(lo), math.floor(hi) + 1))
    jr = np.clip(np.round(judge), labels[0], labels[-1]).astype(int)
    hr = np.clip(np.round(human), labels[0], labels[-1]).astype(int)
    return {
        "n": n,
        "spearman": cfg.boot(lambda i: stats.spearman(judge[i], human[i]), n, groups).to_dict(),
        "pearson": cfg.boot(lambda i: stats.pearson(judge[i], human[i]), n, groups).to_dict(),
        "kappa_quadratic": cfg.boot(
            lambda i: stats.cohen_kappa(list(hr[i]), list(jr[i]), "quadratic", labels),
            n, groups).to_dict(),
        "mae": cfg.boot(_mean_stat(np.abs(judge - human)), n, groups).to_dict(),
    }


def probe_agreement(provider: Provider, items: Sequence[Item], cfg: ProbeConfig) -> dict:
    """Agreement with human labels: rank correlation and kappa (pointwise), kappa (pairwise)."""
    out: dict = {}
    if "pointwise" in cfg.modes:
        rows = [(it, c) for it in items for c in it.candidates if c.human_score is not None]
        if len(rows) >= 3:
            s = np.array([provider.judge_pointwise(it.prompt, c) for it, c in rows])
            h = np.array([c.human_score for _, c in rows])
            out["pointwise"] = pointwise_agreement(s, h, provider.scale,
                                                   [it.id for it, _ in rows], cfg)
        else:
            out["pointwise"] = _skipped("fewer than three human-scored candidates")
    if "pairwise" in cfg.modes:
        rows = [(it, i, j, it.human_outcome(i, j))
                for it, i, j in _pairs(items, cfg.max_pairs_per_item)]
        rows = [r for r in rows if r[3] is not None]
        if len(rows) >= 3:
            n = len(rows)
            groups = [r[0].id for r in rows]
            p_ab = np.array([provider.judge_pairwise(it.prompt, it.candidates[i],
                                                     it.candidates[j]) for it, i, j, _ in rows])
            p_ba = np.array([provider.judge_pairwise(it.prompt, it.candidates[j],
                                                     it.candidates[i]) for it, i, j, _ in rows])
            h = np.array([r[3] for r in rows])
            hv = np.sign(h - 0.5).astype(int)
            single = _verdict(p_ab, cfg.tie_tol)
            debiased = _verdict(0.5 * (p_ab + 1.0 - p_ba), cfg.tie_tol)
            decided = hv != 0
            labels = [-1, 0, 1]

            def acc(v):
                return lambda i: (float((v[i] == hv[i])[decided[i]].mean())
                                  if decided[i].any() else float("nan"))

            def kap(v):
                return lambda i: stats.cohen_kappa(list(hv[i]), list(v[i]), labels=labels)

            out["pairwise"] = {
                "n": n, "n_human_ties": int((~decided).sum()),
                "accuracy_single_order": cfg.boot(acc(single), n, groups).to_dict(),
                "accuracy_both_orders": cfg.boot(acc(debiased), n, groups).to_dict(),
                "kappa_single_order": cfg.boot(kap(single), n, groups).to_dict(),
                "kappa_both_orders": cfg.boot(kap(debiased), n, groups).to_dict(),
            }
        else:
            out["pairwise"] = _skipped("fewer than three pairs with human preferences")
    return out


_PROBE_FUNCS = {
    "position": probe_position,
    "length": probe_length,
    "self_preference": probe_self_preference,
    "scale": probe_scale,
    "agreement": probe_agreement,
}


def run_probes(provider: Provider, items: Sequence[Item], cfg: ProbeConfig | None = None,
               probes: Sequence[str] = ALL_PROBES) -> dict:
    cfg = cfg or ProbeConfig()
    unknown = set(probes) - set(_PROBE_FUNCS)
    if unknown:
        raise ValueError(f"unknown probes: {', '.join(sorted(unknown))}")
    cached = provider if isinstance(provider, CachedProvider) else CachedProvider(provider)
    results = {}
    for name in probes:
        if name == "position" and "pairwise" not in cfg.modes:
            results[name] = _skipped("pairwise mode disabled")
        elif name == "scale" and "pointwise" not in cfg.modes:
            results[name] = _skipped("pointwise mode disabled")
        else:
            results[name] = _PROBE_FUNCS[name](cached, items, cfg)
    return {
        "judge": {"name": provider.name, "family": provider.family,
                  "scale": list(provider.scale)},
        "n_items": len(items),
        "config": {"n_boot": cfg.n_boot, "alpha": cfg.alpha, "seed": cfg.seed,
                   "pad_ratio": cfg.pad_ratio, "modes": list(cfg.modes)},
        "probes": results,
    }
