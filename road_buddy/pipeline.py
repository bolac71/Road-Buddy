from __future__ import annotations

import gc
import json
import random
import signal
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from road_buddy.config import AppConfig
from road_buddy.dataio import (
    extract_answer_letter,
    load_answer_map,
    load_dataset,
    resolve_video_path,
    save_submission,
)
from road_buddy.model.qwen_vl import QwenVLRunner
from road_buddy.prompting import extract_choice_letters
from road_buddy.heuristic_video import sample_frames_heuristic
from road_buddy.video import sample_video_frames


@dataclass
class EvalStats:
    total: int
    correct: int
    accuracy: float


@dataclass
class InferenceSummary:
    total_samples: int
    output_csv: str
    avg_latency_sec: float
    p95_latency_sec: float
    over_time_limit_count: int
    eval_stats: EvalStats | None = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_predictions(pred_rows: list[dict[str, str]], answer_map: dict[str, str]) -> EvalStats:
    correct = 0
    total = 0

    pred_map = {row["id"]: row["answer"] for row in pred_rows}
    for qid, gt_raw in answer_map.items():
        gt = extract_answer_letter(gt_raw)
        if gt is None:
            continue
        if qid not in pred_map:
            continue
        total += 1
        if pred_map[qid].strip().upper() == gt:
            correct += 1

    acc = float(correct) / float(total) if total else 0.0
    return EvalStats(total=total, correct=correct, accuracy=acc)


def _save_audit_payload(
    config: AppConfig,
    rows: list[dict[str, str]],
    audit_rows: list[dict],
    interrupted: bool,
    eval_stats: EvalStats | None,
) -> None:
    if not config.paths.audit_json:
        return

    payload = {
        "meta": {
            "total_samples": len(rows),
            "dataset_json": config.paths.dataset_json,
            "video_root": config.paths.video_root,
            "output_csv": config.paths.output_csv,
            "model_name_or_path": config.model.model_name_or_path,
            "interrupted": interrupted,
            "eval_total": eval_stats.total if eval_stats else 0,
            "eval_correct": eval_stats.correct if eval_stats else 0,
            "eval_accuracy": round(float(eval_stats.accuracy), 6) if eval_stats else 0.0,
        },
        "records": audit_rows,
    }

    with Path(config.paths.audit_json).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_inference(config: AppConfig) -> InferenceSummary:
    seed_everything(config.runtime.seed)

    dataset = load_dataset(config.paths.dataset_json)
    if config.runtime.max_samples is not None:
        dataset = dataset[: config.runtime.max_samples]
        print(f"[INFO] max_samples={config.runtime.max_samples}: chay tren {len(dataset)} mau dau tien.", flush=True)
    answer_map = load_answer_map(config.paths.answer_json)

    model_runner = QwenVLRunner(config.model)
    model_runner.load()

    rows: list[dict[str, str]] = []
    audit_rows: list[dict] = []
    latencies: list[float] = []
    over_time_limit_count = 0

    checkpoint_every = max(1, int(config.runtime.checkpoint_every_n))
    print_every = max(1, int(config.runtime.print_every_n))
    batch_size = max(1, int(config.runtime.batch_size))

    stop_requested = False
    interrupted = False
    processed = 0
    pending: list[dict] = []

    def _request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"[WARN] Nhan tin hieu {signum}. Dang dung xu ly va luu ket qua.", flush=True)

    old_sigterm = signal.getsignal(signal.SIGTERM)
    old_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    def _eval_fields(qid: str, pred_answer: str) -> tuple[str, bool | None]:
        gt_raw = answer_map.get(qid)
        gt = extract_answer_letter(gt_raw) if gt_raw else None
        if gt is None:
            return "", None
        return gt, pred_answer.strip().upper() == gt

    def _current_eval() -> EvalStats | None:
        if not answer_map:
            return None
        return evaluate_predictions(rows, answer_map)

    def _checkpoint_if_needed() -> None:
        if not rows:
            return
        if len(rows) % checkpoint_every != 0:
            return
        save_submission(rows, config.paths.output_csv)
        _save_audit_payload(config, rows, audit_rows, interrupted=False, eval_stats=_current_eval())

    def _process_pending() -> None:
        nonlocal over_time_limit_count
        if not pending:
            return

        for item in pending:
            t0 = time.perf_counter()
            answer = item["choice_letters"][0]
            raw_text = ""
            probs: dict[str, float] = {}
            pred_source = ""
            status = "ok"
            error_message = ""

            try:
                result = model_runner.predict(
                    question=item["question"],
                    choices=item["choices"],
                    images=item["selected_images"],
                    system_hint=config.prompt.system_hint,
                )
                answer = result.answer if result.answer in item["choice_letters"] else item["choice_letters"][0]
                raw_text = result.raw_text
                probs = result.probs
                pred_source = result.source or "predict"
            except torch.cuda.OutOfMemoryError as e:
                status = "predict_exception"
                pred_source = "default_on_oom"
                error_message = f"{type(e).__name__}: {e}"
                print(f"[WARN] OOM at id={item['qid']}: {error_message}", flush=True)
            except Exception as e:
                status = "predict_exception"
                pred_source = "default_on_exception"
                error_message = f"{type(e).__name__}: {e}"
                print(f"[WARN] Predict exception at id={item['qid']}: {error_message}", flush=True)

            latency = time.perf_counter() - t0
            latencies.append(latency)
            if latency > config.runtime.time_limit_sec:
                over_time_limit_count += 1

            gt_label, is_correct = _eval_fields(item["qid"], answer)
            rows.append({"id": item["qid"], "answer": answer})
            audit_rows.append(
                {
                    "id": item["qid"],
                    "video_path": item["video_rel"],
                    "resolved_video_path": item["video_abs"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "choice_letters": item["choice_letters"],
                    "answer": answer,
                    "raw_text": raw_text,
                    "probs": probs,
                    "pred_source": pred_source,
                    "selected_frame_indices": item["selected_frame_indices"],
                    "gt_label": gt_label,
                    "is_correct": is_correct,
                    "latency_sec": round(float(latency), 6),
                    "status": status,
                    "error": error_message,
                }
            )

            if item["processed"] % print_every == 0:
                eval_suffix = f" gt={gt_label} ok={int(bool(is_correct))}" if is_correct is not None else ""
                print(
                    f"[Infer] {item['processed']}/{len(dataset)} id={item['qid']} pred={answer} src={pred_source}{eval_suffix} latency={latency:.2f}s",
                    flush=True,
                )

            if torch.cuda.is_available() and config.runtime.clear_cuda_cache_on_each_sample:
                torch.cuda.empty_cache()
            gc.collect()
            _checkpoint_if_needed()

        pending.clear()

    try:
        for sample in tqdm(dataset, desc="Infer"):
            if stop_requested:
                interrupted = True
                break

            processed += 1
            qid = str(sample["id"])
            question = str(sample["question"])
            choices = [str(c) for c in sample["choices"]]
            video_rel = str(sample.get("video_path", ""))

            choice_letters = extract_choice_letters(choices)
            if not choice_letters:
                answer = config.runtime.default_answer
                gt_label, is_correct = _eval_fields(qid, answer)
                rows.append({"id": qid, "answer": answer})
                audit_rows.append(
                    {
                        "id": qid,
                        "video_path": video_rel,
                        "resolved_video_path": "",
                        "question": question,
                        "choices": choices,
                        "choice_letters": [],
                        "answer": answer,
                        "raw_text": "",
                        "probs": {},
                        "pred_source": "default_no_choice",
                        "selected_frame_indices": [],
                        "gt_label": gt_label,
                        "is_correct": is_correct,
                        "latency_sec": 0.0,
                        "status": "no_choice_letters",
                        "error": "",
                    }
                )
                _checkpoint_if_needed()
                continue

            video_abs = resolve_video_path(config.paths.video_root, video_rel)
            try:
                if config.sampling.use_heuristic:
                    selected_images = sample_frames_heuristic(
                        video_abs,
                        question=question,
                        num_frames=config.sampling.num_frames,
                        max_side=config.sampling.max_side,
                    )
                else:
                    selected_images = sample_video_frames(
                        video_abs,
                        num_frames=config.sampling.num_frames,
                        max_side=config.sampling.max_side,
                    )
                if not selected_images:
                    raise ValueError("no_frames_from_video")
                selected_indices = list(range(len(selected_images)))
            except Exception as e:
                rows.append({"id": qid, "answer": choice_letters[0]})
                gt_label, is_correct = _eval_fields(qid, choice_letters[0])
                audit_rows.append(
                    {
                        "id": qid,
                        "video_path": video_rel,
                        "resolved_video_path": video_abs,
                        "question": question,
                        "choices": choices,
                        "choice_letters": choice_letters,
                        "answer": choice_letters[0],
                        "raw_text": "",
                        "probs": {},
                        "pred_source": "default_on_frame_extract_error",
                        "selected_frame_indices": [],
                        "gt_label": gt_label,
                        "is_correct": is_correct,
                        "latency_sec": 0.0,
                        "status": "frame_extract_exception",
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                _checkpoint_if_needed()
                continue

            pending.append(
                {
                    "processed": processed,
                    "qid": qid,
                    "question": question,
                    "choices": choices,
                    "choice_letters": choice_letters,
                    "video_rel": video_rel,
                    "video_abs": video_abs,
                    "selected_images": selected_images,
                    "selected_frame_indices": selected_indices,
                }
            )

            if len(pending) >= batch_size:
                _process_pending()

    finally:
        if pending:
            _process_pending()
        signal.signal(signal.SIGTERM, old_sigterm)
        signal.signal(signal.SIGINT, old_sigint)

    save_submission(rows, config.paths.output_csv)
    eval_stats = _current_eval()
    _save_audit_payload(config, rows, audit_rows, interrupted=interrupted, eval_stats=eval_stats)

    avg_latency = statistics.fmean(latencies) if latencies else 0.0
    sorted_lat = sorted(latencies)
    p95_latency = sorted_lat[int(0.95 * (len(sorted_lat) - 1))] if sorted_lat else 0.0

    if interrupted:
        print(f"[WARN] Inference stop sau khi xu ly {len(rows)}/{len(dataset)} samples.", flush=True)
        if eval_stats is not None:
            print(
                f"[WARN] Partial eval: total={eval_stats.total} correct={eval_stats.correct} acc={eval_stats.accuracy:.6f}",
                flush=True,
            )

    return InferenceSummary(
        total_samples=len(rows),
        output_csv=config.paths.output_csv,
        avg_latency_sec=avg_latency,
        p95_latency_sec=p95_latency,
        over_time_limit_count=over_time_limit_count,
        eval_stats=eval_stats,
    )
