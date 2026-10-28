"""Content-free padding used by the verbosity probe."""

from __future__ import annotations

import hashlib
import re

FILLER_SENTENCES = (
    "It is worth noting that this is an important consideration.",
    "In other words, the points above address the question as asked.",
    "This perspective can be helpful when thinking about the topic.",
    "To put it another way, the answer remains the same as stated.",
    "There are many ways to look at this, and this is one of them.",
    "As mentioned, the reasoning above applies in this situation.",
    "Overall, this summarizes the main idea in a clear manner.",
    "It should be emphasized that context matters in cases like this.",
    "With that in mind, the explanation above should be sufficient.",
    "Taking everything into account, the conclusion stays unchanged.",
    "This is a common question and the answer above covers it.",
    "Ultimately, the key takeaway is the one already described.",
)

_FILLER_SET = frozenset(FILLER_SENTENCES)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def word_count(text: str) -> int:
    return max(1, len(text.split()))


def pad(text: str, ratio: float = 2.0) -> str:
    """Append filler sentences until the word count reaches ``ratio`` times the original."""
    if ratio <= 1.0:
        return text
    target = ratio * word_count(text)
    start = int(hashlib.blake2b(text.encode(), digest_size=4).hexdigest(), 16)
    parts = [text.rstrip()]
    words = word_count(text)
    k = 0
    while words < target:
        sentence = FILLER_SENTENCES[(start + k) % len(FILLER_SENTENCES)]
        parts.append(sentence)
        words += len(sentence.split())
        k += 1
    return " ".join(parts)


def strip_filler(text: str) -> str:
    """Remove known filler sentences; used by the fake judge to see only real content."""
    kept = [s for s in _SENT_SPLIT.split(text.strip()) if s.strip() not in _FILLER_SET]
    return " ".join(kept)
