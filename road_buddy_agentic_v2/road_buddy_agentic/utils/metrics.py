from __future__ import annotations

from collections import defaultdict


def compute_accuracy(pred_rows: list[dict], answer_map: dict[str, str], extract_answer_letter) -> dict:
    total = 0
    correct = 0
    for row in pred_rows:
        qid = row["id"]
        gt_raw = answer_map.get(qid)
        gt = extract_answer_letter(gt_raw) if gt_raw else None
        if gt is None:
            continue
        pred = extract_answer_letter(row.get("answer", ""))
        if pred is None:
            continue
        total += 1
        if pred == gt:
            correct += 1
    acc = correct / total if total else 0.0
    return {"total": total, "correct": correct, "accuracy": acc}


def compute_accuracy_by_type(records: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        if r.get("is_correct") is None:
            continue
        buckets[r.get("dataset_type", "other")].append(bool(r["is_correct"]))

    out: dict[str, dict] = {}
    for k, vals in sorted(buckets.items()):
        total = len(vals)
        correct = sum(vals)
        out[k] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / total if total else 0.0,
        }
    return out
