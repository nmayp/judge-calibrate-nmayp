"""Item/candidate data model and JSONL input/output."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_KNOWN_CANDIDATE_KEYS = {"text", "source", "id", "human_score", "judge_score"}


@dataclass
class Candidate:
    text: str
    source: str | None = None
    id: str | None = None
    human_score: float | None = None
    judge_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return max(1, len(self.text.split()))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"text": self.text}
        for key in ("source", "id", "human_score", "judge_score"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        out.update(self.meta)
        return out


@dataclass
class Item:
    id: str
    prompt: str
    candidates: list[Candidate]
    # Explicit human verdict between candidates 0 and 1: 0, 1, or "tie".
    human_preference: int | str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def human_outcome(self, i: int, j: int, tie_tol: float = 0.0) -> float | None:
        """Human probability-style outcome for candidate i beating j (1, 0, 0.5) or None."""
        if {i, j} == {0, 1} and self.human_preference is not None:
            pref = self.human_preference
            if pref == "tie":
                return 0.5
            return 1.0 if int(pref) == i else 0.0
        hi, hj = self.candidates[i].human_score, self.candidates[j].human_score
        if hi is None or hj is None:
            return None
        if abs(hi - hj) <= tie_tol:
            return 0.5
        return 1.0 if hi > hj else 0.0

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "candidates": [c.to_dict() for c in self.candidates],
        }
        if self.human_preference is not None:
            out["human_preference"] = self.human_preference
        out.update(self.meta)
        return out


class DataError(ValueError):
    pass


def _parse_preference(raw: Any, where: str) -> int | str | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in {"a", "0"}:
            return 0
        if low in {"b", "1"}:
            return 1
        if low in {"tie", "draw", "c"}:
            return "tie"
        raise DataError(f"{where}: unrecognised human_preference {raw!r}")
    if isinstance(raw, bool) or raw not in (0, 1):
        raise DataError(f"{where}: human_preference must be 0, 1, 'A', 'B' or 'tie'")
    return int(raw)


def _parse_candidate(raw: Any, where: str) -> Candidate:
    if isinstance(raw, str):
        return Candidate(text=raw)
    if not isinstance(raw, dict) or "text" not in raw:
        raise DataError(f"{where}: candidate must be a string or an object with 'text'")
    meta = {k: v for k, v in raw.items() if k not in _KNOWN_CANDIDATE_KEYS}

    def num(key: str) -> float | None:
        v = raw.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError) as exc:
            raise DataError(f"{where}: {key} must be numeric") from exc

    return Candidate(
        text=str(raw["text"]),
        source=raw.get("source"),
        id=None if raw.get("id") is None else str(raw["id"]),
        human_score=num("human_score"),
        judge_score=num("judge_score"),
        meta=meta,
    )


def parse_item(raw: dict[str, Any], index: int) -> Item:
    where = f"line {index + 1}"
    if not isinstance(raw, dict):
        raise DataError(f"{where}: expected a JSON object")
    if "prompt" not in raw:
        raise DataError(f"{where}: missing 'prompt'")
    cands_raw = raw.get("candidates", raw.get("answers"))
    if cands_raw is None and "answer_a" in raw and "answer_b" in raw:
        cands_raw = [raw["answer_a"], raw["answer_b"]]
    if not isinstance(cands_raw, list) or not cands_raw:
        raise DataError(f"{where}: 'candidates' must be a non-empty list")
    candidates = [_parse_candidate(c, f"{where} candidate {k}") for k, c in enumerate(cands_raw)]
    reserved = {"id", "prompt", "candidates", "answers", "answer_a", "answer_b",
                "human_preference"}
    return Item(
        id=str(raw.get("id", index)),
        prompt=str(raw["prompt"]),
        candidates=candidates,
        human_preference=_parse_preference(raw.get("human_preference"), where),
        meta={k: v for k, v in raw.items() if k not in reserved},
    )


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataError(f"line {lineno + 1}: invalid JSON ({exc.msg})") from exc


def load_items(path: str | Path) -> list[Item]:
    items = [parse_item(raw, k) for k, raw in enumerate(iter_jsonl(path))]
    if not items:
        raise DataError(f"{path}: no items found")
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        raise DataError(f"{path}: duplicate item ids")
    return items


def write_items(items: Iterable[Item], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
