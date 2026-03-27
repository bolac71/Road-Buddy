from __future__ import annotations

import argparse
import json
from pathlib import Path

from road_buddy.config import as_dict, load_config
from road_buddy.dataio import extract_answer_letter, load_answer_map, load_submission
from road_buddy.pipeline import evaluate_predictions, run_inference


def _cmd_show_config(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(json.dumps(as_dict(cfg), ensure_ascii=False, indent=2))
    return 0


def _cmd_infer(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    summary = run_inference(cfg)

    print(f"total_samples={summary.total_samples}")
    print(f"output_csv={summary.output_csv}")
    print(f"avg_latency_sec={summary.avg_latency_sec:.3f}")
    print(f"p95_latency_sec={summary.p95_latency_sec:.3f}")
    print(f"over_time_limit_count={summary.over_time_limit_count}")

    if summary.eval_stats is not None:
        print(f"eval_total={summary.eval_stats.total}")
        print(f"eval_correct={summary.eval_stats.correct}")
        print(f"eval_accuracy={summary.eval_stats.accuracy:.6f}")

    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    submission_map = load_submission(args.submission)
    answer_map = load_answer_map(args.answer_json)

    rows = [{"id": k, "answer": v} for k, v in submission_map.items()]
    stats = evaluate_predictions(rows, answer_map)

    print(f"eval_total={stats.total}")
    print(f"eval_correct={stats.correct}")
    print(f"eval_accuracy={stats.accuracy:.6f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RoadBuddy baseline pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    show_cfg = sub.add_parser("show-config", help="Print resolved configuration")
    show_cfg.add_argument("--config", type=str, required=True)
    show_cfg.set_defaults(func=_cmd_show_config)

    infer = sub.add_parser("infer", help="Run inference and optional local eval")
    infer.add_argument("--config", type=str, required=True)
    infer.set_defaults(func=_cmd_infer)

    evaluate = sub.add_parser("eval", help="Evaluate existing submission.csv")
    evaluate.add_argument("--submission", type=str, required=True)
    evaluate.add_argument("--answer-json", type=str, required=True)
    evaluate.set_defaults(func=_cmd_eval)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
