"""Judge providers: the protocol, a deterministic biased fake, a cache and an HTTP client."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .data import Candidate
from .padding import strip_filler, word_count


@runtime_checkable
class Provider(Protocol):
    """A judge. ``family`` names the model family it belongs to (for self-preference)."""

    name: str
    family: str | None
    scale: tuple[float, float]

    def judge_pairwise(self, prompt: str, first: Candidate, second: Candidate) -> float:
        """Probability-like preference that ``first`` (shown in position A) is better."""
        ...

    def judge_pointwise(self, prompt: str, candidate: Candidate) -> float:
        """Absolute score on ``scale``."""
        ...


def same_family(source: str | None, family: str | None) -> bool:
    if not source or not family:
        return False
    return family.lower() in source.lower()


def _hash_unit(*parts: Any) -> float:
    h = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).hexdigest()
    return int(h, 16) / float(1 << 64)


def latent_quality(text: str) -> float:
    """Deterministic 'true' quality in [0, 1) of the non-filler content of ``text``."""
    core = " ".join(strip_filler(text).split()).lower()
    return _hash_unit("quality", core)


@dataclass
class BiasConfig:
    """Biases injected into :class:`FakeJudge`.

    Pairwise quantities are on the logit scale; pointwise ones are in score points.
    """

    quality_weight: float = 4.0
    position_bias: float = 0.0
    length_bias: float = 0.0
    self_bias: float = 0.0
    noise: float = 0.0
    point_length_bias: float = 0.0
    point_self_bias: float = 0.0
    compression: float = 1.0
    anchor: float | None = None
    point_noise: float = 0.0
    ref_length: float = 40.0
    integer_scores: bool = False
    decisive: bool = False

    @classmethod
    def from_pairs(cls, pairs: list[str]) -> BiasConfig:
        """Build from ``key=value`` strings (the CLI ``--fake`` flag)."""
        types = {f.name: f.type for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for pair in pairs:
            key, sep, raw = pair.partition("=")
            key = key.strip()
            if not sep or key not in types:
                raise ValueError(f"bad fake option {pair!r}; known: {', '.join(types)}")
            raw = raw.strip()
            if "bool" in str(types[key]):
                kwargs[key] = raw.lower() in {"1", "true", "yes", "on"}
            elif key == "anchor" and raw.lower() in {"none", ""}:
                kwargs[key] = None
            else:
                kwargs[key] = float(raw)
        return cls(**kwargs)


@dataclass
class FakeJudge:
    """Deterministic simulated judge with configurable, known biases.

    Pairwise:  logit P(first wins) = w*(q1-q2) + position + length*log(len1/len2)
                                     + self*(own1-own2) + noise
    Pointwise: s = anchor + compression*(lo + q*(hi-lo) - anchor)
                 + point_length*log(len/ref) + point_self*own + noise

    ``q`` is :func:`latent_quality`, which ignores filler text, so padding changes only
    the length term. Source labels stand in for the stylistic cues a real judge would use
    to recognise its own family's outputs.
    """

    biases: BiasConfig = field(default_factory=BiasConfig)
    family: str | None = "fake"
    scale: tuple[float, float] = (1.0, 10.0)
    seed: int = 0
    quality_fn: Callable[[str], float] = latent_quality
    name: str = "fake-judge"
    calls: int = 0

    def _noise(self, sd: float, *key: Any) -> float:
        if sd <= 0:
            return 0.0
        u = _hash_unit(self.seed, *key)
        rng = np.random.default_rng(int(u * (1 << 62)))
        return float(rng.normal(0.0, sd))

    def pairwise_logit(self, prompt: str, first: Candidate, second: Candidate) -> float:
        b = self.biases
        z = b.quality_weight * (self.quality_fn(first.text) - self.quality_fn(second.text))
        z += b.position_bias
        z += b.length_bias * math.log(word_count(first.text) / word_count(second.text))
        z += b.self_bias * (same_family(first.source, self.family)
                            - same_family(second.source, self.family))
        z += self._noise(b.noise, "pair", prompt, first.text, second.text)
        return z

    def judge_pairwise(self, prompt: str, first: Candidate, second: Candidate) -> float:
        self.calls += 1
        p = 1.0 / (1.0 + math.exp(-self.pairwise_logit(prompt, first, second)))
        if self.biases.decisive:
            u = _hash_unit(self.seed, "draw", prompt, first.text, second.text)
            return 1.0 if u < p else 0.0
        return p

    def judge_pointwise(self, prompt: str, candidate: Candidate) -> float:
        self.calls += 1
        b = self.biases
        lo, hi = self.scale
        anchor = (lo + hi) / 2 if b.anchor is None else b.anchor
        true = lo + self.quality_fn(candidate.text) * (hi - lo)
        s = anchor + b.compression * (true - anchor)
        s += b.point_length_bias * math.log(word_count(candidate.text) / b.ref_length)
        s += b.point_self_bias * same_family(candidate.source, self.family)
        s += self._noise(b.point_noise, "point", prompt, candidate.text)
        s = min(hi, max(lo, s))
        return float(round(s)) if b.integer_scores else s


class CachedProvider:
    """Memoises judgments so probes sharing comparisons do not re-query the judge."""

    def __init__(self, inner: Provider):
        self.inner = inner
        self.name = inner.name
        self.family = inner.family
        self.scale = inner.scale
        self._pair: dict[tuple, float] = {}
        self._point: dict[tuple, float] = {}

    @staticmethod
    def _key(c: Candidate) -> tuple:
        return (c.text, c.source)

    def judge_pairwise(self, prompt: str, first: Candidate, second: Candidate) -> float:
        key = (prompt, self._key(first), self._key(second))
        if key not in self._pair:
            self._pair[key] = float(self.inner.judge_pairwise(prompt, first, second))
        return self._pair[key]

    def judge_pointwise(self, prompt: str, candidate: Candidate) -> float:
        key = (prompt, self._key(candidate))
        if key not in self._point:
            self._point[key] = float(self.inner.judge_pointwise(prompt, candidate))
        return self._point[key]


PAIRWISE_TEMPLATE = """Please act as an impartial judge and evaluate the quality of the two \
responses to the user question below. Do not let the order of the responses, their length, \
or the identity of their authors influence your decision.

[User Question]
{prompt}

[Response A]
{a}

[Response B]
{b}

Explain briefly, then output your verdict strictly as "[[A]]" if A is better, "[[B]]" if B \
is better, or "[[C]]" for a tie."""

POINTWISE_TEMPLATE = """Please act as an impartial judge and rate the quality of the response \
to the user question below on a scale from {lo} to {hi}.

[User Question]
{prompt}

[Response]
{answer}

Explain briefly, then output your rating strictly in the form "Rating: [[N]]"."""

_VERDICT = re.compile(r"\[\[\s*([ABC])\s*\]\]")
_RATING = re.compile(r"\[\[\s*(-?\d+(?:\.\d+)?)\s*\]\]")


def parse_verdict(text: str) -> float:
    found = _VERDICT.findall(text)
    if not found:
        raise ValueError("no [[A]]/[[B]]/[[C]] verdict in judge output")
    return {"A": 1.0, "B": 0.0, "C": 0.5}[found[-1]]


def parse_rating(text: str, scale: tuple[float, float]) -> float:
    found = _RATING.findall(text)
    if not found:
        raise ValueError("no [[N]] rating in judge output")
    lo, hi = scale
    return min(hi, max(lo, float(found[-1])))


Transport = Callable[[str, dict[str, str], dict[str, Any]], dict[str, Any]]


def _urllib_transport(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers,
                                 method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


class HTTPJudge:
    """Judge backed by an OpenAI-compatible or Anthropic Messages HTTP endpoint.

    Uses only the standard library; the network is touched only when this class is used.
    """

    def __init__(self, model: str, api_style: str = "openai", base_url: str | None = None,
                 api_key_env: str | None = None, family: str | None = None,
                 scale: tuple[float, float] = (1.0, 10.0), max_tokens: int = 512,
                 transport: Transport | None = None):
        if api_style not in {"openai", "anthropic"}:
            raise ValueError("api_style must be 'openai' or 'anthropic'")
        self.model = model
        self.api_style = api_style
        default_url = ("https://api.openai.com/v1" if api_style == "openai"
                       else "https://api.anthropic.com/v1")
        self.base_url = (base_url or default_url).rstrip("/")
        self.api_key_env = api_key_env or (
            "OPENAI_API_KEY" if api_style == "openai" else "ANTHROPIC_API_KEY")
        self.name = model
        self.family = family
        self.scale = scale
        self.max_tokens = max_tokens
        self.transport = transport or _urllib_transport

    def _complete(self, content: str) -> str:
        key = os.environ.get(self.api_key_env, "")
        messages = [{"role": "user", "content": content}]
        if self.api_style == "openai":
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
            body = {"model": self.model, "messages": messages, "temperature": 0,
                    "max_tokens": self.max_tokens}
            data = self.transport(f"{self.base_url}/chat/completions", headers, body)
            return data["choices"][0]["message"]["content"]
        headers = {"Content-Type": "application/json", "x-api-key": key,
                   "anthropic-version": "2023-06-01"}
        body = {"model": self.model, "messages": messages, "max_tokens": self.max_tokens}
        data = self.transport(f"{self.base_url}/messages", headers, body)
        return "".join(block.get("text", "") for block in data["content"])

    def judge_pairwise(self, prompt: str, first: Candidate, second: Candidate) -> float:
        text = self._complete(PAIRWISE_TEMPLATE.format(prompt=prompt, a=first.text,
                                                       b=second.text))
        return parse_verdict(text)

    def judge_pointwise(self, prompt: str, candidate: Candidate) -> float:
        lo, hi = self.scale
        text = self._complete(POINTWISE_TEMPLATE.format(prompt=prompt, answer=candidate.text,
                                                        lo=_fmt(lo), hi=_fmt(hi)))
        return parse_rating(text, self.scale)


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)
