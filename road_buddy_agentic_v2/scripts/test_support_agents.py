from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from road_buddy_agentic.analyzers.dataset_reader import load_first_n_samples
from road_buddy_agentic.orchestration.support_pipeline import build_question_profile
from road_buddy_agentic.builders.support_request_builder import build_support_request
from road_buddy_agentic.support_agents.manager import build_support_agent
from road_buddy_agentic.validators.answer_leak_validator import detect_answer_leak
from road_buddy_agentic.validators.support_brief_validator import validate_support_brief


def load_yaml(path: str) -> dict:
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    p = argparse.ArgumentParser(description='Support-agent system test (no Qwen, no GPU required).')
    p.add_argument('--config', required=True)
    p.add_argument('--provider', choices=['gemini','groq','none'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--api-env-path', default=None)
    p.add_argument('--api-key-env-name', default=None)
    p.add_argument('--max-samples', type=int, default=5)
    p.add_argument('--output-root', required=True)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    dataset_cfg = cfg.get('dataset', {})
    support_cfg = cfg.get('support_agent', {})
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    samples = load_first_n_samples(
        dataset_json=dataset_cfg['dataset_json'],
        take_first_n=int(args.max_samples),
        require_type_field=bool(dataset_cfg.get('require_type_field', True)),
    )

    support_kwargs = {
        'api_env_path': args.api_env_path or support_cfg.get('api_env_path'),
        'api_key_env_name': args.api_key_env_name or support_cfg.get('api_key_env_name'),
        'max_retries_per_sample': int(support_cfg.get('max_retries_per_sample', 5)),
        'key_cooldown_sec': int(support_cfg.get('key_cooldown_sec', 60)),
        'transient_cooldown_sec': int(support_cfg.get('transient_cooldown_sec', 20)),
        'invalid_key_cooldown_sec': int(support_cfg.get('invalid_key_cooldown_sec', 1800)),
        'max_output_tokens': int(support_cfg.get('max_output_tokens', 512)),
        'temperature': float(support_cfg.get('temperature', 0.0)),
    }

    agent = build_support_agent(args.provider, args.model, **support_kwargs)

    rows = []
    status_counts = {}
    answer_leak_count = 0

    for sample in samples:
        profile = build_question_profile(sample)
        request = build_support_request(sample, profile)
        brief = agent.generate_support_brief(request)
        leak = detect_answer_leak(brief.raw_text)
        brief.answer_leak_detected = leak
        validated = validate_support_brief(brief)
        status_counts[validated.status] = status_counts.get(validated.status, 0) + 1
        if leak:
            answer_leak_count += 1

        row = {
            'qid': sample.id,
            'dataset_type': sample.dataset_type,
            'resolved_dataset_type': profile.dataset_type,
            'intent': profile.intent,
            'provider': args.provider,
            'model': args.model,
            'status': validated.status,
            'answer_leak_detected': leak,
            'legal_topics': validated.legal_topics,
            'evidence_checklist': validated.evidence_checklist,
            'option_hints_count': len(validated.option_hints),
            'raw_text': validated.raw_text,
            'normalized_payload': validated.normalized_payload,
        }
        rows.append(row)

    jsonl_path = output_root / 'support_only_test.jsonl'
    with jsonl_path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    summary = {
        'provider': args.provider,
        'model': args.model,
        'max_samples': len(rows),
        'status_counts': status_counts,
        'answer_leak_count': answer_leak_count,
        'jsonl_path': str(jsonl_path),
    }
    with (output_root / 'support_only_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] support-only test saved to {output_root}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
