from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from road_buddy_agentic.orchestration.agentic_run import run_agentic_inference


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg.setdefault("dataset", {})
    cfg.setdefault("support_agent", {})
    cfg.setdefault("runtime", {})

    if args.dataset_json:
        cfg["dataset"]["dataset_json"] = args.dataset_json
    if args.video_root:
        cfg["dataset"]["video_root"] = args.video_root
    if args.answer_json:
        cfg["dataset"]["answer_json"] = args.answer_json
    if args.max_samples is not None:
        cfg["dataset"]["take_first_n"] = int(args.max_samples)
        cfg["runtime"]["max_samples"] = int(args.max_samples)
    if args.support_provider:
        cfg["support_agent"]["provider"] = args.support_provider
        cfg["support_agent"]["enabled"] = args.support_provider != "none"
    if args.support_model:
        cfg["support_agent"]["model_name"] = args.support_model
    if args.support_api_env_path:
        cfg["support_agent"]["api_env_path"] = args.support_api_env_path
    if args.support_api_key_env_name:
        cfg["support_agent"]["api_key_env_name"] = args.support_api_key_env_name
    if args.output_root:
        cfg["runtime"]["output_root"] = args.output_root
    return cfg


def cmd_infer_agentic(args: argparse.Namespace) -> int:
    cfg = _apply_overrides(_load_yaml(args.config), args)

    print("[INFO] infer-agentic")
    print(f"[DATASET] {cfg['dataset']['dataset_json']}")
    print(f"[VIDEO_ROOT] {cfg['dataset']['video_root']}")
    print(f"[MAX_SAMPLES] {cfg['dataset'].get('take_first_n', 100)}")
    print(f"[SUPPORT_PROVIDER] {cfg['support_agent'].get('provider', 'none')}")
    print(f"[SUPPORT_MODEL] {cfg['support_agent'].get('model_name', 'none')}")
    print(f"[SUPPORT_ENV] {cfg['support_agent'].get('api_env_path')}")

    summary = run_agentic_inference(cfg)

    print(f"[DONE] saved to {cfg['runtime']['output_root']}")
    print(f"eval_total={summary['eval_total']}")
    print(f"eval_correct={summary['eval_correct']}")
    print(f"eval_accuracy={summary['eval_accuracy']:.6f}")
    print(f"avg_latency_sec={summary['avg_latency_sec']:.3f}")
    print(f"p95_latency_sec={summary['p95_latency_sec']:.3f}")
    print(f"over_time_limit_count={summary['over_time_limit_count']}")
    return 0


def cmd_bench_support(args: argparse.Namespace) -> int:
    cfg = _load_yaml(args.config)
    runs = cfg["benchmark"]["support_runs"]
    base_dataset = cfg["dataset"]
    base_runtime = cfg["runtime"]
    base_qwen = cfg["qwen_reasoner"]
    base_support = cfg.get("support_agent", {})

    root = Path(base_runtime["output_root"])
    root.mkdir(parents=True, exist_ok=True)

    leaderboard = []
    for run in runs:
        run_cfg = {
            "dataset": dict(base_dataset),
            "qwen_reasoner": dict(base_qwen),
            "support_agent": {
                **dict(base_support),
                "enabled": run.get("enabled", False),
                "provider": run.get("provider", "none"),
                "model_name": run.get("model_name", "none"),
                "api_env_path": run.get("api_env_path", base_support.get("api_env_path")),
                "api_key_env_name": run.get("api_key_env_name", base_support.get("api_key_env_name")),
            },
            "runtime": {
                **dict(base_runtime),
                "output_root": str(root / run["run_name"]),
            },
        }
        summary = run_agentic_inference(run_cfg)
        leaderboard.append(
            {
                "run_name": run["run_name"],
                "support_provider": run.get("provider", "none"),
                "support_model": run.get("model_name", "none"),
                "total_samples": summary["max_samples"],
                "eval_total": summary["eval_total"],
                "eval_correct": summary["eval_correct"],
                "eval_accuracy": summary["eval_accuracy"],
                "avg_latency_sec": summary["avg_latency_sec"],
                "p95_latency_sec": summary["p95_latency_sec"],
            }
        )

    with (root / "leaderboard.json").open("w", encoding="utf-8") as f:
        json.dump(leaderboard, f, ensure_ascii=False, indent=2)

    print(f"[DONE] bench saved to {root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Road Buddy Agentic Support CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("infer-agentic", help="Run one support configuration on first N samples")
    p1.add_argument("--config", required=True)
    p1.add_argument("--dataset-json")
    p1.add_argument("--video-root")
    p1.add_argument("--answer-json")
    p1.add_argument("--max-samples", type=int)
    p1.add_argument("--support-provider", choices=["none", "gemini", "groq"])
    p1.add_argument("--support-model")
    p1.add_argument("--support-api-env-path")
    p1.add_argument("--support-api-key-env-name")
    p1.add_argument("--output-root")
    p1.set_defaults(func=cmd_infer_agentic)

    p2 = sub.add_parser("bench-support", help="Run multiple support configurations")
    p2.add_argument("--config", required=True)
    p2.set_defaults(func=cmd_bench_support)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
