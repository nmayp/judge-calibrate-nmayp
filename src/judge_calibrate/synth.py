"""Synthetic datasets whose human labels follow the fake judge's latent quality."""

from __future__ import annotations

import numpy as np

from .data import Candidate, Item
from .providers import latent_quality

_WORD_TEXT = (
    "system cache latency index query model gradient token buffer thread kernel schema "
    "network packet sensor signal budget forecast policy contract revenue margin protein "
    "enzyme orbit planet climate river glacier sentence grammar melody rhythm canvas "
    "texture recipe oven flour harvest soil engine torque circuit voltage"
)
_WORDS = _WORD_TEXT.split()

_TEMPLATES = (
    "The {a} depends on the {b}, so adjusting the {c} first is sensible.",
    "A common mistake is to ignore the {a} when reasoning about the {b}.",
    "Consider measuring the {a} directly before changing the {b}.",
    "In practice the {a} and the {b} interact through the {c}.",
    "If the {a} is unstable, the {b} will drift over time.",
    "One concrete step is to compare the {a} against a known {b}.",
)


def _answer(rng: np.random.Generator, n_sentences: int) -> str:
    out = []
    for _ in range(n_sentences):
        t = _TEMPLATES[rng.integers(len(_TEMPLATES))]
        a, b, c = rng.choice(_WORDS, size=3, replace=False)
        out.append(t.format(a=a, b=b, c=c))
    return " ".join(out)


def make_synthetic(n_items: int = 200, n_candidates: int = 2,
                   sources: tuple[str, ...] = ("claude-sim", "gpt-sim"),
                   scale: tuple[float, float] = (1.0, 10.0), human_noise: float = 0.5,
                   integer_human: bool = True, sentence_range: tuple[int, int] = (2, 12),
                   seed: int = 0) -> list[Item]:
    """Items with human scores ``lo + q*(hi-lo) + N(0, human_noise)``.

    Sources are assigned round-robin so every family is equally represented and quality is
    independent of source and of length; any judge-human gap is then attributable to bias.
    """
    rng = np.random.default_rng(seed)
    lo, hi = scale
    items = []
    for k in range(n_items):
        cands = []
        for j in range(n_candidates):
            text = _answer(rng, int(rng.integers(sentence_range[0], sentence_range[1] + 1)))
            q = latent_quality(text)
            h = lo + q * (hi - lo) + rng.normal(0.0, human_noise)
            h = float(np.clip(round(h) if integer_human else h, lo, hi))
            src = sources[(k + j) % len(sources)] if sources else None
            cands.append(Candidate(text=text, source=src, id=f"{k}-{j}", human_score=h))
        pref: int | str | None = None
        if n_candidates >= 2:
            h0, h1 = cands[0].human_score, cands[1].human_score
            pref = "tie" if h0 == h1 else (0 if h0 > h1 else 1)
        items.append(Item(id=f"item-{k}", prompt=f"Question {k}: how should one approach "
                          f"the {_WORDS[k % len(_WORDS)]} problem?", candidates=cands,
                          human_preference=pref))
    return items
