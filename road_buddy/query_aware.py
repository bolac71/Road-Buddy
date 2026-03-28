from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


class QuestionIntent(str, Enum):
    TEMPORAL = "temporal"
    VALUE = "value"
    DIRECTION = "direction"
    IDENTIFICATION = "identification"
    EXISTENCE = "existence"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    question: str
    intent: QuestionIntent
    target_objects: list[str] = field(default_factory=list)
    temporal_hints: list[str] = field(default_factory=list)
    keywords_found: list[str] = field(default_factory=list)


_KEYWORD_OBJECTS: dict[str, list[str]] = {
    "biển": ["traffic_sign"],
    "biển báo": ["traffic_sign"],
    "đèn": ["traffic_light"],
    "đèn giao thông": ["traffic_light"],
    "làn": ["lane"],
    "vạch": ["road_marking"],
    "xe": ["vehicle"],
    "tốc độ": ["speed_limit_sign"],
    "rẽ trái": ["left_turn"],
    "rẽ phải": ["right_turn"],
}

_INTENT_PATTERNS: list[tuple[QuestionIntent, list[str]]] = [
    (QuestionIntent.TEMPORAL, ["đầu tiên", "cuối cùng", "trước", "sau", "hiện tại", "đang"]),
    (QuestionIntent.VALUE, ["bao nhiêu", "mấy", "tốc độ"]),
    (QuestionIntent.DIRECTION, ["hướng", "rẽ", "đi thẳng", "quay đầu"]),
    (QuestionIntent.IDENTIFICATION, ["biển gì", "loại nào", "là gì"]),
    (QuestionIntent.EXISTENCE, ["có", "không"]),
]

_TEMPORAL_MAP: dict[str, str] = {
    "đầu tiên": "first",
    "trước": "first",
    "cuối cùng": "last",
    "sau": "last",
    "hiện tại": "current",
    "đang": "current",
}


def analyze_question(question: str) -> QueryAnalysis:
    q = (question or "").strip()
    q_lower = q.lower()

    target_objects: list[str] = []
    keywords_found: list[str] = []
    for keyword, objects in _KEYWORD_OBJECTS.items():
        if keyword in q_lower:
            keywords_found.append(keyword)
            for obj in objects:
                if obj not in target_objects:
                    target_objects.append(obj)

    temporal_hints: list[str] = []
    for vn, hint in _TEMPORAL_MAP.items():
        if vn in q_lower and hint not in temporal_hints:
            temporal_hints.append(hint)

    intent = QuestionIntent.UNKNOWN
    for candidate_intent, patterns in _INTENT_PATTERNS:
        if any(p in q_lower for p in patterns):
            intent = candidate_intent
            break

    return QueryAnalysis(
        question=q,
        intent=intent,
        target_objects=target_objects,
        temporal_hints=temporal_hints,
        keywords_found=keywords_found,
    )


def _compute_sharpness_scores(frames: Iterable[Image.Image]) -> np.ndarray:
    vals: list[float] = []
    for img in frames:
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    if not vals:
        return np.array([], dtype=np.float32)

    scores = np.array(vals, dtype=np.float32)
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-8:
        return np.ones_like(scores, dtype=np.float32)
    return (scores - lo) / (hi - lo)


def _temporal_prior(n: int, temporal_hints: list[str]) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float32)

    idx = np.arange(n, dtype=np.float32)
    if "first" in temporal_hints:
        return 1.0 - (idx / max(1.0, float(n - 1)))
    if "last" in temporal_hints:
        return idx / max(1.0, float(n - 1))

    center = (n - 1) / 2.0
    dist = np.abs(idx - center)
    max_dist = max(1.0, center)
    return 1.0 - (dist / max_dist)


def select_query_aware_frames(
    frames: list[Image.Image],
    analysis: QueryAnalysis,
    max_frames: int,
) -> tuple[list[Image.Image], list[int], list[float]]:
    if not frames:
        return [], [], []

    n = len(frames)
    k = max(1, min(max_frames, n))

    sharpness = _compute_sharpness_scores(frames)
    temporal = _temporal_prior(n, analysis.temporal_hints)

    # Query-aware blend: temporal intent drives when it matters, otherwise visual quality dominates.
    if analysis.intent == QuestionIntent.TEMPORAL or analysis.temporal_hints:
        final_scores = 0.65 * temporal + 0.35 * sharpness
    else:
        final_scores = 0.40 * temporal + 0.60 * sharpness

    top_idx = np.argsort(-final_scores)[:k].tolist()
    top_idx = sorted(top_idx)

    selected = [frames[i] for i in top_idx]
    selected_scores = [float(final_scores[i]) for i in top_idx]
    return selected, top_idx, selected_scores
