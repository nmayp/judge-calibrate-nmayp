import math

import pytest

from judge_calibrate.data import Candidate
from judge_calibrate.padding import pad, strip_filler, word_count
from judge_calibrate.providers import (
    BiasConfig,
    CachedProvider,
    FakeJudge,
    HTTPJudge,
    Provider,
    latent_quality,
    parse_rating,
    parse_verdict,
)
from judge_calibrate.synth import make_synthetic

A = Candidate("The cache depends on the index, so tune the query first.", source="claude-x")
B = Candidate("Measure the latency directly before changing the schema.", source="gpt-y")


def test_padding_is_content_free_for_quality():
    padded = pad(A.text, ratio=3.0)
    assert word_count(padded) >= 3 * word_count(A.text)
    assert strip_filler(padded) == A.text
    assert latent_quality(padded) == latent_quality(A.text)
    assert pad(A.text, ratio=1.0) == A.text


def test_fake_is_protocol_and_deterministic():
    j = FakeJudge(BiasConfig(noise=0.5, point_noise=0.5), seed=3)
    assert isinstance(j, Provider)
    assert j.judge_pairwise("p", A, B) == FakeJudge(BiasConfig(noise=0.5), seed=3)\
        .judge_pairwise("p", A, B)
    assert j.judge_pointwise("p", A) == j.judge_pointwise("p", A)


def test_unbiased_fake_is_order_consistent():
    j = FakeJudge()
    assert j.judge_pairwise("p", A, B) + j.judge_pairwise("p", B, A) == pytest.approx(1.0)


def test_position_bias_injected_on_logit_scale():
    j = FakeJudge(BiasConfig(position_bias=1.0))
    z1 = j.pairwise_logit("p", A, B)
    z2 = j.pairwise_logit("p", B, A)
    assert (z1 + z2) / 2 == pytest.approx(1.0)


def test_self_and_length_terms():
    j = FakeJudge(BiasConfig(self_bias=0.7, length_bias=0.4), family="claude")
    base = FakeJudge(family="claude")
    long_a = Candidate(pad(A.text, 2.0), source=A.source)
    delta = j.pairwise_logit("p", long_a, B) - base.pairwise_logit("p", long_a, B)
    expect = 0.7 + 0.4 * math.log(word_count(long_a.text) / word_count(B.text))
    assert delta == pytest.approx(expect)


def test_pointwise_compression_and_bounds():
    j = FakeJudge(BiasConfig(compression=0.0, anchor=7.0))
    assert j.judge_pointwise("p", A) == pytest.approx(7.0)
    j2 = FakeJudge(BiasConfig(point_length_bias=50.0, integer_scores=True))
    assert j2.judge_pointwise("p", Candidate(pad(A.text, 4.0))) == 10.0


def test_decisive_mode_outputs_hard_verdicts():
    j = FakeJudge(BiasConfig(decisive=True))
    assert j.judge_pairwise("p", A, B) in (0.0, 1.0)


def test_cache_avoids_repeat_calls():
    inner = FakeJudge()
    c = CachedProvider(inner)
    for _ in range(3):
        c.judge_pairwise("p", A, B)
        c.judge_pointwise("p", A)
    assert inner.calls == 2


def test_bias_config_from_pairs():
    cfg = BiasConfig.from_pairs(["position_bias=0.8", "integer_scores=true", "anchor=none"])
    assert cfg.position_bias == 0.8 and cfg.integer_scores and cfg.anchor is None
    with pytest.raises(ValueError):
        BiasConfig.from_pairs(["nope=1"])


def test_parsers():
    assert parse_verdict("reasoning... [[B]]") == 0.0
    assert parse_verdict("[[A]] then actually [[C]]") == 0.5
    assert parse_rating("Rating: [[8]]", (1, 10)) == 8.0
    assert parse_rating("Rating: [[14]]", (1, 10)) == 10.0
    with pytest.raises(ValueError):
        parse_verdict("no verdict")


@pytest.mark.parametrize("style,reply", [
    ("openai", {"choices": [{"message": {"content": "ok [[A]]"}}]}),
    ("anthropic", {"content": [{"type": "text", "text": "ok [[A]]"}]}),
])
def test_http_judge_with_fake_transport(style, reply):
    seen = {}

    def transport(url, headers, body):
        seen["url"], seen["body"] = url, body
        return reply

    j = HTTPJudge("m", api_style=style, transport=transport)
    assert j.judge_pairwise("q", A, B) == 1.0
    assert seen["body"]["model"] == "m"
    assert seen["url"].endswith("chat/completions" if style == "openai" else "messages")


def test_synthetic_labels_track_latent_quality():
    items = make_synthetic(n_items=100, human_noise=0.0, integer_human=False)
    for it in items:
        for c in it.candidates:
            assert c.human_score == pytest.approx(1 + 9 * latent_quality(c.text))
    sources = [c.source for it in items for c in it.candidates]
    assert sources.count("claude-sim") == sources.count("gpt-sim")
