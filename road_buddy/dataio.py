from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, Optional


def load_dataset(dataset_json: str) -> list[dict]:
    with Path(dataset_json).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], list):
        data = raw["data"]
    elif isinstance(raw, list):
        data = raw
    else:
        raise ValueError(f"Unsupported dataset format in {dataset_json}")

    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if "id" not in item or "question" not in item or "choices" not in item or "video_path" not in item:
            continue
        out.append(item)
    return out


def resolve_video_path(video_root: str, sample_video_path: str) -> str:
    path = Path(sample_video_path)
    if path.is_absolute():
        return str(path)

    root = Path(video_root)
    primary = (root / path).resolve()
    if primary.exists():
        return str(primary)

    # Backward-compatible aliases seen in some metadata dumps.
    as_posix = path.as_posix()
    alias_candidates: list[Path] = []
    if as_posix.startswith("dataset/"):
        alias_candidates.append((root / as_posix.replace("dataset/", "train/", 1)).resolve())
    if as_posix.startswith("train/"):
        alias_candidates.append((root / as_posix.replace("train/", "dataset/", 1)).resolve())
    # Fallback: try under datasets/ subdirectory (e.g. when video_root points to
    # the repo root but videos live under datasets/public_test/... or datasets/train/...).
    alias_candidates.append((root / "datasets" / path).resolve())

    for candidate in alias_candidates:
        if candidate.exists():
            return str(candidate)

    return str(primary)


def extract_answer_letter(answer_text: str) -> Optional[str]:
    if not answer_text:
        return None
    match = re.search(r"([A-D])", answer_text.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).upper()


def load_answer_map(answer_json: Optional[str]) -> dict[str, str]:
    if not answer_json:
        return {}

    p = Path(answer_json)
    if not p.exists():
        return {}

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        if "data" in raw and isinstance(raw["data"], list):
            answer_map: dict[str, str] = {}
            for item in raw["data"]:
                if not isinstance(item, dict):
                    continue
                qid = str(item.get("id", "")).strip()
                ans = str(item.get("answer", "")).strip()
                if qid and ans:
                    answer_map[qid] = ans
            return answer_map
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, list):
        answer_map: dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            qid = str(item.get("id", "")).strip()
            ans = str(item.get("answer", "")).strip()
            if qid and ans:
                answer_map[qid] = ans
        return answer_map
    return {}


def save_submission(rows: Iterable[dict[str, str]], output_csv: str) -> None:
    with Path(output_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "answer"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"id": row["id"], "answer": row["answer"]})


def load_submission(output_csv: str) -> dict[str, str]:
    p = Path(output_csv)
    if not p.exists():
        raise FileNotFoundError(output_csv)

    rows: dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            qid = str(row.get("id", "")).strip()
            ans = str(row.get("answer", "")).strip()
            if not qid or not ans:
                continue
            rows[qid] = ans
    return rows
