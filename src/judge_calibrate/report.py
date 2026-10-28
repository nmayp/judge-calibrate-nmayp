"""JSON report writing and plain-text summaries."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

_TITLES = {
    "position": "position bias",
    "length": "length / verbosity bias",
    "self_preference": "self-preference",
    "scale": "scale use / anchoring",
    "agreement": "agreement with human labels",
    "raw": "raw judge scores",
    "calibrated": "calibrated (out-of-fold)",
}


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def write_json(obj: Any, path: str | Path) -> None:
    Path(path).write_text(json.dumps(to_jsonable(obj), indent=2) + "\n", encoding="utf-8")


def _num(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, list):
        return "[" + ", ".join(_num(x) for x in v) + "]"
    return str(v)


def _is_estimate(v: Any) -> bool:
    return isinstance(v, dict) and "value" in v and "ci" in v


def _lines(node: dict, depth: int) -> list[str]:
    pad = "  " * depth
    width = 36 - len(pad)
    out = []
    if "skipped" in node:
        return [f"{pad}skipped: {node['skipped']}"]
    for key, v in node.items():
        label = key.replace("_", " ")
        if _is_estimate(v):
            lo, hi = v["ci"]
            flag = ""
            if v.get("significant"):
                flag = f"  * differs from {_num(v['null'])}"
            out.append(f"{pad}{label:<{width}} {_num(v['value']):>8}  "
                       f"[{_num(lo)}, {_num(hi)}]{flag}")
        elif isinstance(v, dict):
            out.append(f"{pad}{_TITLES.get(key, label)}")
            out.extend(_lines(v, depth + 1))
        else:
            out.append(f"{pad}{label:<{width}} {_num(v):>8}")
    return out


def summarize_probes(report: dict) -> str:
    j = report["judge"]
    cfg = report["config"]
    ci = int(round(100 * (1 - cfg["alpha"])))
    head = [f"judge: {j['name']}  family: {j['family'] or '-'}  scale: "
            f"{_num(j['scale'])}  items: {report['n_items']}",
            f"bootstrap: {cfg['n_boot']} resamples by item, {ci}% percentile intervals; "
            "* marks intervals excluding the no-bias value", ""]
    body = []
    for name, res in report["probes"].items():
        body.append(_TITLES.get(name, name))
        body.extend(_lines(res, 1))
        body.append("")
    return "\n".join(head + body).rstrip() + "\n"


def summarize_calibration(report: dict) -> str:
    m = report["model"]
    cal = m["calibrator"]
    lines = [f"method: {cal['method']}  scale: {_num(cal['scale'])}  "
             f"labelled scores: {report['n']}  score source: {report['score_source']}"]
    lc = m.get("length_control")
    if lc:
        extra = " (controlling for human score)" if lc["controls_human"] else ""
        lines.append(f"length control: {lc['coef']:+.3f} points per log word count "
                     f"removed{extra}")
    cv = report.get("cross_validation")
    if cv:
        lines.append(f"{cv['k']}-fold cross-validation by item")
        keys = list(cv["raw"])
        lines.append(f"  {'metric':<26} {'raw':>8} {'calibrated':>11}")
        for k in keys:
            lines.append(f"  {k.replace('_', ' '):<26} {_num(cv['raw'][k]):>8} "
                         f"{_num(cv['calibrated'][k]):>11}")
    return "\n".join(lines) + "\n"
