from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATASET_JSON = "/home/cndt_khanhnd/data/road-buddy/datasets/public_test_with_type.json"
VIDEO_ROOT = "/home/cndt_khanhnd/data/road-buddy/datasets"
ANSWER_JSON = "/home/cndt_khanhnd/data/road-buddy/datasets/public_test/public_test_with_answers.json"


def load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Khong tim thay file: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dataset(raw: Any) -> list[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if isinstance(raw.get("data"), list):
            return raw["data"]
        if isinstance(raw.get("records"), list):
            return raw["records"]
    raise ValueError("Khong nhan dien duoc format dataset_json")


def extract_answer_map(raw: Any) -> dict[str, str]:
    out: dict[str, str] = {}

    def try_put(obj: dict):
        qid = None
        for k in ["id", "qid", "question_id"]:
            if k in obj and str(obj[k]).strip():
                qid = str(obj[k]).strip()
                break
        if not qid:
            return

        ans = None
        for k in ["answer", "label", "gt", "ground_truth", "correct_answer"]:
            if k in obj and str(obj[k]).strip():
                ans = str(obj[k]).strip()
                break
        if ans is not None:
            out[qid] = ans

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                try_put(item)
        return out

    if isinstance(raw, dict):
        # case 1: {"data": [...]}
        if isinstance(raw.get("data"), list):
            for item in raw["data"]:
                if isinstance(item, dict):
                    try_put(item)
            return out

        # case 2: {"answers": [...]}
        if isinstance(raw.get("answers"), list):
            for item in raw["answers"]:
                if isinstance(item, dict):
                    try_put(item)
            return out

        # case 3: {"testa_0001": "A", ...}
        simple_ok = True
        simple_map: dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(v, (str, int, float)):
                simple_map[str(k)] = str(v)
            else:
                simple_ok = False
                break
        if simple_ok and simple_map:
            return simple_map

    return out


def main() -> None:
    print("=== CHECK DATA READ ===")
    print(f"dataset_json = {DATASET_JSON}")
    print(f"video_root   = {VIDEO_ROOT}")
    print(f"answer_json  = {ANSWER_JSON}")
    print()

    dataset_raw = load_json(DATASET_JSON)
    dataset = normalize_dataset(dataset_raw)
    print(f"[OK] Dataset load duoc: {len(dataset)} mau")

    required_fields = ["id", "question", "choices", "video_path", "type"]
    missing_field_rows = []

    for i, row in enumerate(dataset):
        miss = [k for k in required_fields if k not in row]
        if miss:
            missing_field_rows.append((i, row.get("id", f"idx_{i}"), miss))

    if missing_field_rows:
        print(f"[WARN] Co {len(missing_field_rows)} mau thieu field bat buoc")
        for item in missing_field_rows[:10]:
            print("   ", item)
    else:
        print("[OK] Tat ca mau deu co id/question/choices/video_path/type")

    print()
    print("=== CHECK VIDEO PATHS ===")
    existing_videos = 0
    missing_videos = []

    for row in dataset[:50]:
        video_rel = str(row.get("video_path", "")).strip()
        video_abs = Path(VIDEO_ROOT) / video_rel
        if video_abs.exists():
            existing_videos += 1
        else:
            missing_videos.append((row.get("id", ""), str(video_abs)))

    print(f"[INFO] Check thu 50 mau dau")
    print(f"[OK] Video ton tai: {existing_videos}/50")
    print(f"[MISS] Video thieu: {len(missing_videos)}/50")
    for item in missing_videos[:10]:
        print("   ", item)

    print()
    print("=== CHECK ANSWER MAP ===")
    answer_raw = load_json(ANSWER_JSON)
    answer_map = extract_answer_map(answer_raw)
    print(f"[INFO] So answer doc duoc: {len(answer_map)}")

    dataset_ids = [str(r.get("id", "")).strip() for r in dataset]
    overlap = [qid for qid in dataset_ids if qid in answer_map]
    no_answer = [qid for qid in dataset_ids if qid and qid not in answer_map]

    print(f"[OK] So id match giua dataset va answer_map: {len(overlap)}")
    print(f"[MISS] So id khong co trong answer_map: {len(no_answer)}")

    print("5 id dau co trong dataset:")
    print(dataset_ids[:5])

    print("5 answer dau tim duoc:")
    shown = 0
    for k, v in answer_map.items():
        print(f"   {k}: {v}")
        shown += 1
        if shown >= 5:
            break

    if no_answer:
        print("10 id dataset dau tien khong match answer_map:")
        print(no_answer[:10])

    print()
    print("=== CHECK 5 MAU DAU ===")
    for row in dataset[:5]:
        qid = str(row.get("id", "")).strip()
        video_rel = str(row.get("video_path", "")).strip()
        video_abs = Path(VIDEO_ROOT) / video_rel
        gt = answer_map.get(qid, None)
        print(
            json.dumps(
                {
                    "id": qid,
                    "type": row.get("type"),
                    "video_rel": video_rel,
                    "video_exists": video_abs.exists(),
                    "gt_found": gt is not None,
                    "gt_value": gt,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
