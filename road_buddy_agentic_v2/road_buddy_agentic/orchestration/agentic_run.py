from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from road_buddy_agentic.analyzers.dataset_reader import load_first_n_samples
from road_buddy_agentic.orchestration.support_pipeline import build_question_profile
from road_buddy_agentic.builders.support_request_builder import build_support_request
from road_buddy_agentic.builders.qwen_context_builder import build_qwen_context
from road_buddy_agentic.support_agents.manager import build_support_agent
from road_buddy_agentic.validators.support_brief_validator import validate_support_brief
from road_buddy_agentic.reasoners import QwenReasonerAdapter
from road_buddy_agentic.utils.metrics import compute_accuracy, compute_accuracy_by_type


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_system_hint(cfg: dict, support_context: str = "") -> str:
    base_hint = str(
        cfg.get("qwen_reasoner", {}).get(
            "system_hint",
            "Bạn là trợ lý AI cho bài toán giao thông Việt Nam. Chỉ chọn một đáp án đúng nhất theo video và câu hỏi.",
        )
    )
    if not support_context.strip():
        return base_hint
    return f"{base_hint}\n\n{support_context.strip()}"


def run_agentic_inference(config: dict) -> dict:
    from road_buddy.dataio import (
        extract_answer_letter,
        load_answer_map,
        resolve_video_path,
        save_submission,
    )
    from road_buddy.query_aware import analyze_question, select_query_aware_frames
    from road_buddy.video import sample_video_frames

    dataset_cfg = config["dataset"]
    runtime_cfg = config.get("runtime", {})
    qwen_cfg = config.get("qwen_reasoner", {})
    support_cfg = config.get("support_agent", {})
    output_root = Path(runtime_cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    _seed_everything(int(runtime_cfg.get("seed", 42)))

    samples = load_first_n_samples(
        dataset_json=dataset_cfg["dataset_json"],
        take_first_n=int(dataset_cfg.get("take_first_n", 100)),
        require_type_field=bool(dataset_cfg.get("require_type_field", True)),
    )

    answer_map = load_answer_map(dataset_cfg.get("answer_json")) if dataset_cfg.get("answer_json") else {}

    support_agent = build_support_agent(
        provider=support_cfg.get("provider", "none"),
        model_name=support_cfg.get("model_name", "none"),
        api_env_path=support_cfg.get("api_env_path"),
        api_key_env_name=support_cfg.get("api_key_env_name"),
        max_retries_per_sample=support_cfg.get("max_retries_per_sample", 5),
        key_cooldown_sec=support_cfg.get("key_cooldown_sec", 60),
        transient_cooldown_sec=support_cfg.get("transient_cooldown_sec", 20),
        invalid_key_cooldown_sec=support_cfg.get("invalid_key_cooldown_sec", 1800),
        jpeg_quality=support_cfg.get("jpeg_quality", 85),
        max_output_tokens=support_cfg.get("max_output_tokens", 512),
        temperature=support_cfg.get("temperature", 0.0),
    )

    reasoner = QwenReasonerAdapter(qwen_cfg)
    reasoner.load()

    submission_rows: list[dict[str, str]] = []
    support_rows: list[dict] = []
    audit_rows: list[dict] = []
    latencies: list[float] = []
    support_latencies: list[float] = []
    qwen_latencies: list[float] = []
    over_time_limit_count = 0
    time_limit_sec = float(runtime_cfg.get("time_limit_sec", 30.0))
    print_every_n = max(1, int(runtime_cfg.get("print_every_n", 1)))

    for i, sample in enumerate(samples, start=1):
        profile = build_question_profile(sample)
        request = build_support_request(sample, profile)

        support_t0 = time.perf_counter()
        brief = validate_support_brief(support_agent.generate_support_brief(request))
        support_latency = time.perf_counter() - support_t0
        support_latencies.append(support_latency)

        video_abs = resolve_video_path(dataset_cfg["video_root"], sample.video_path)
        selected_images = sample_video_frames(
            video_abs,
            num_frames=max(
                int(qwen_cfg.get("num_frames", 10)),
                int(qwen_cfg.get("num_frames", 10)) * int(qwen_cfg.get("candidate_frame_multiplier", 3)),
            ),
            max_side=int(qwen_cfg.get("max_side", 960)),
        )
        if not selected_images:
            answer = ""
            raw_text = ""
            pred_source = "no_result_frame_extract_error"
            reason = "no_frames_from_video"
            selected_indices: list[int] = []
            selected_scores: list[float] = []
            qwen_latency = 0.0
        else:
            selected_indices = list(range(len(selected_images)))
            selected_scores = [1.0] * len(selected_images)
            if bool(qwen_cfg.get("query_aware_enabled", True)):
                qa = analyze_question(sample.question)
                selected_images, selected_indices, selected_scores = select_query_aware_frames(
                    selected_images,
                    qa,
                    max_frames=int(qwen_cfg.get("num_frames", 10)),
                )
            else:
                selected_images = selected_images[: int(qwen_cfg.get("num_frames", 10))]
                selected_indices = selected_indices[: len(selected_images)]
                selected_scores = selected_scores[: len(selected_images)]

            support_context = build_qwen_context(sample, brief) if support_cfg.get("provider", "none") != "none" else ""
            system_hint = _build_system_hint(config, support_context)

            qwen_t0 = time.perf_counter()
            result = reasoner.predict(
                question=sample.question,
                choices=sample.choices,
                images=selected_images,
                system_hint=system_hint,
                target_objects=profile.target_objects,
                temporal_hints=profile.temporal_hints,
            )
            qwen_latency = getattr(result, "latency_sec", None) or (time.perf_counter() - qwen_t0)
            answer = getattr(result, "answer", "") or ""
            raw_text = getattr(result, "raw_text", "")
            pred_source = getattr(result, "source", "unknown")
            reason = getattr(result, "reason", "")

        qwen_latencies.append(qwen_latency)
        total_latency = support_latency + qwen_latency
        latencies.append(total_latency)
        if total_latency > time_limit_sec:
            over_time_limit_count += 1

        gt_label = extract_answer_letter(answer_map.get(sample.id, "")) if answer_map else None
        pred_label = extract_answer_letter(answer)
        is_correct = None if gt_label is None or pred_label is None else (pred_label == gt_label)

        submission_rows.append({"id": sample.id, "answer": answer})
        support_rows.append(
            {
                "qid": sample.id,
                "question": sample.question,
                "dataset_type": sample.dataset_type,
                "intent": profile.intent,
                "support_provider": support_cfg.get("provider", "none"),
                "support_model": support_cfg.get("model_name", "none"),
                "support_enabled": support_cfg.get("provider", "none") != "none",
                "support_status": brief.status,
                "support_raw_output": brief.raw_text,
                "support_brief": brief.normalized_payload,
                "support_answer_leak_detected": brief.answer_leak_detected,
                "support_latency_sec": round(float(support_latency), 6),
            }
        )
        audit_rows.append(
            {
                "id": sample.id,
                "video_path": sample.video_path,
                "resolved_video_path": video_abs,
                "question": sample.question,
                "choices": sample.choices,
                "dataset_type": sample.dataset_type,
                "query_intent": profile.intent,
                "query_target_objects": profile.target_objects,
                "query_temporal_hints": profile.temporal_hints,
                "selected_frame_indices": selected_indices,
                "selected_frame_scores": [float(x) for x in selected_scores],
                "answer": answer,
                "raw_text": raw_text,
                "pred_source": pred_source,
                "reason": reason,
                "gt_label": gt_label or "",
                "is_correct": is_correct,
                "support_provider": support_cfg.get("provider", "none"),
                "support_model": support_cfg.get("model_name", "none"),
                "support_status": brief.status,
                "support_answer_leak_detected": brief.answer_leak_detected,
                "support_latency_sec": round(float(support_latency), 6),
                "qwen_latency_sec": round(float(qwen_latency), 6),
                "latency_sec": round(float(total_latency), 6),
            }
        )

        if i % print_every_n == 0:
            eval_suffix = f" gt={gt_label} ok={int(bool(is_correct))}" if is_correct is not None else " gt=SKIP ok=SKIP"
            print(
                f"[Infer] {i}/{len(samples)} id={sample.id} type={sample.dataset_type} pred={answer or 'KHONG_CO_KET_QUA'}"
                f" src={pred_source} support={brief.status}{eval_suffix} latency={total_latency:.2f}s",
                flush=True,
            )

    submission_path = output_root / "submission.csv"
    save_submission(submission_rows, str(submission_path))

    support_jsonl_path = output_root / "support_outputs.jsonl"
    with support_jsonl_path.open("w", encoding="utf-8") as f:
        for row in support_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    accuracy = compute_accuracy(submission_rows, answer_map, extract_answer_letter) if answer_map else {"total": 0, "correct": 0, "accuracy": 0.0}
    accuracy_by_type = compute_accuracy_by_type(audit_rows)

    audit_payload = {
        "meta": {
            "dataset_json": dataset_cfg["dataset_json"],
            "video_root": dataset_cfg["video_root"],
            "answer_json": dataset_cfg.get("answer_json"),
            "total_samples": len(samples),
            "support_provider": support_cfg.get("provider", "none"),
            "support_model": support_cfg.get("model_name", "none"),
            "qwen_model": qwen_cfg.get("model_name", "Qwen/Qwen3-VL"),
            "eval_total": accuracy["total"],
            "eval_correct": accuracy["correct"],
            "eval_accuracy": round(float(accuracy["accuracy"]), 6),
        },
        "records": audit_rows,
    }
    audit_path = output_root / "audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_payload, f, ensure_ascii=False, indent=2)

    mean_total = statistics.fmean(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p95_total = sorted_lat[int(0.95 * (len(sorted_lat) - 1))] if sorted_lat else 0.0

    run_summary = {
        "dataset_json": dataset_cfg["dataset_json"],
        "video_root": dataset_cfg["video_root"],
        "answer_json": dataset_cfg.get("answer_json"),
        "max_samples": len(samples),
        "support_provider": support_cfg.get("provider", "none"),
        "support_model": support_cfg.get("model_name", "none"),
        "qwen_model": qwen_cfg.get("model_name", "Qwen/Qwen3-VL"),
        "submission_csv": str(submission_path),
        "audit_json": str(audit_path),
        "support_outputs_jsonl": str(support_jsonl_path),
        "eval_total": accuracy["total"],
        "eval_correct": accuracy["correct"],
        "eval_accuracy": round(float(accuracy["accuracy"]), 6),
        "accuracy_by_type": accuracy_by_type,
        "avg_latency_sec": round(float(mean_total), 6),
        "p95_latency_sec": round(float(p95_total), 6),
        "avg_support_latency_sec": round(float(statistics.fmean(support_latencies) if support_latencies else 0.0), 6),
        "avg_qwen_latency_sec": round(float(statistics.fmean(qwen_latencies) if qwen_latencies else 0.0), 6),
        "over_time_limit_count": over_time_limit_count,
    }
    with (output_root / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    return run_summary
