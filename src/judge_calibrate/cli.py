"""Command-line interface: synth, probe, calibrate, apply."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from . import __version__
from .correct import CalibrationModel, cross_validate
from .data import DataError, Item, load_items, write_items
from .padding import word_count
from .probes import ALL_PROBES, MODES, ProbeConfig, run_probes
from .providers import BiasConfig, CachedProvider, FakeJudge, HTTPJudge, Provider
from .report import summarize_calibration, summarize_probes, to_jsonable, write_json
from .synth import make_synthetic


def _add_judge_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("judge")
    g.add_argument("--fake", action="append", default=[], metavar="KEY=VALUE",
                   help="bias injected into the built-in fake judge (repeatable)")
    g.add_argument("--family", default="fake",
                   help="model family of the judge, matched against candidate sources")
    g.add_argument("--scale", nargs=2, type=float, default=(1.0, 10.0), metavar=("LO", "HI"))
    g.add_argument("--judge-seed", type=int, default=0)
    g.add_argument("--http-model", help="use a real judge over HTTP with this model name")
    g.add_argument("--api-style", choices=("openai", "anthropic"), default="openai")
    g.add_argument("--base-url")
    g.add_argument("--api-key-env")


def _make_judge(args: argparse.Namespace) -> Provider:
    scale = (float(args.scale[0]), float(args.scale[1]))
    if args.http_model:
        if args.fake:
            raise SystemExit("--fake cannot be combined with --http-model")
        return HTTPJudge(args.http_model, api_style=args.api_style, base_url=args.base_url,
                         api_key_env=args.api_key_env, family=args.family, scale=scale)
    return FakeJudge(BiasConfig.from_pairs(args.fake), family=args.family, scale=scale,
                     seed=args.judge_seed)


def _emit(text: str, json_obj: dict, out: str | None, quiet: bool) -> None:
    if out:
        write_json(json_obj, out)
    if not quiet:
        sys.stdout.write(text)
        if out:
            sys.stdout.write(f"\nreport written to {out}\n")


def cmd_synth(args: argparse.Namespace) -> int:
    items = make_synthetic(n_items=args.n, n_candidates=args.candidates,
                           sources=tuple(args.sources), human_noise=args.human_noise,
                           seed=args.seed)
    write_items(items, args.out)
    print(f"wrote {len(items)} items to {args.out}")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    items = load_items(args.data)
    modes = MODES if args.mode == "both" else (args.mode,)
    cfg = ProbeConfig(n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
                      pad_ratio=args.pad_ratio, max_pairs_per_item=args.max_pairs,
                      max_candidates_per_item=args.max_candidates, modes=modes)
    probes = args.probes.split(",") if args.probes else ALL_PROBES
    report = run_probes(_make_judge(args), items, cfg, probes)
    report["data"] = args.data
    _emit(summarize_probes(report), report, args.out, args.quiet)
    return 0


def _scored_rows(items: Sequence[Item], judge: Provider | None, labelled_only: bool):
    rows = [(it, c) for it in items for c in it.candidates
            if not labelled_only or c.human_score is not None]
    have = all(c.judge_score is not None for _, c in rows)
    if have:
        scores = np.array([c.judge_score for _, c in rows])
        source = "data"
    else:
        if judge is None:
            raise DataError("candidates lack judge_score and no judge is configured")
        scores = np.array([judge.judge_pointwise(it.prompt, c) for it, c in rows])
        source = judge.name
    return rows, scores, source


def cmd_calibrate(args: argparse.Namespace) -> int:
    items = load_items(args.data)
    judge = None if args.use_data_scores else CachedProvider(_make_judge(args))
    rows, scores, source = _scored_rows(items, judge, labelled_only=True)
    if args.use_data_scores and source != "data":
        raise DataError("--use-data-scores given but some candidates lack judge_score")
    if len(rows) < 4:
        raise DataError("need at least four candidates with human_score to calibrate")
    human = np.array([c.human_score for _, c in rows])
    lengths = np.array([word_count(c.text) for _, c in rows])
    groups = [it.id for it, _ in rows]
    scale = (float(args.scale[0]), float(args.scale[1]))
    model = CalibrationModel.fit(scores, human, lengths, args.method, scale,
                                 args.length_control, judge=source)
    report = {"data": args.data, "n": len(rows), "score_source": source,
              "model": model.to_dict()}
    if args.folds > 1:
        report["cross_validation"] = cross_validate(
            scores, human, lengths, groups, args.method, scale, args.length_control,
            k=args.folds, seed=args.seed)
    if args.save:
        model.save(args.save)
    text = summarize_calibration(report)
    if args.save and not args.quiet:
        text += f"model saved to {args.save}\n"
    _emit(text, to_jsonable(report), args.out, args.quiet)
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    model = CalibrationModel.load(args.model)
    items = load_items(args.data)
    judge = None if args.use_data_scores else CachedProvider(_make_judge(args))
    rows, scores, _ = _scored_rows(items, judge, labelled_only=False)
    lengths = np.array([word_count(c.text) for _, c in rows])
    calibrated = model.predict(scores, lengths)
    for (_, c), raw, cal in zip(rows, scores, calibrated, strict=True):
        c.judge_score = float(raw)
        c.meta["calibrated_score"] = round(float(cal), 6)
    write_items(items, args.out)
    print(f"wrote {len(rows)} calibrated scores to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="judge-calibrate",
                                 description="Measure and correct bias in LLM-as-judge scores.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write a synthetic labelled dataset")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--candidates", type=int, default=2)
    p.add_argument("--sources", nargs="*", default=["claude-sim", "gpt-sim"])
    p.add_argument("--human-noise", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("probe", help="run bias probes and report effect sizes")
    p.add_argument("data", help="JSONL of items")
    _add_judge_args(p)
    p.add_argument("--probes", help=f"comma-separated subset of {','.join(ALL_PROBES)}")
    p.add_argument("--mode", choices=("both", *MODES), default="both")
    p.add_argument("--pad-ratio", type=float, default=2.0)
    p.add_argument("--max-pairs", type=int, default=1, help="pairs per item")
    p.add_argument("--max-candidates", type=int, default=2,
                   help="candidates per item padded by the length probe")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the JSON report here")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("calibrate", help="fit a score calibration against human labels")
    p.add_argument("data")
    _add_judge_args(p)
    p.add_argument("--method", choices=("isotonic", "platt", "linear"), default="isotonic")
    p.add_argument("--length-control", action="store_true")
    p.add_argument("--use-data-scores", action="store_true",
                   help="use judge_score from the data instead of querying a judge")
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds (0 to skip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", help="write the fitted model JSON here")
    p.add_argument("--out", help="write the JSON report here")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("apply", help="apply a saved calibration model to items")
    p.add_argument("model")
    p.add_argument("data")
    _add_judge_args(p)
    p.add_argument("--use-data-scores", action="store_true")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_apply)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataError, ValueError, OSError) as exc:
        print(f"judge-calibrate: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
