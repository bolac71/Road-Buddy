from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# YOLO lazy singleton (DGS — Detection-Guided Scoring)
# ---------------------------------------------------------------------------
_yolo_model = None
_yolo_model_path: Optional[str] = None

# Road Lane v2 class names (6 classes — model hiện tại)
# Khi có unified model (Road Lane + BDD100K = 17 classes), cập nhật list này.
_YOLO_CLASS_NAMES = [
    "divider-line", "dotted-line", "double-line",
    "random-line", "road-sign-line", "solid-line",
]

# Map target_objects (từ analyze_question) → YOLO class name substrings
_OBJECT_TO_YOLO_CLASSES: dict[str, list[str]] = {
    "traffic_sign":    ["traffic sign", "road-sign-line"],
    "traffic_light":   ["traffic light"],
    "speed_limit_sign":["traffic sign"],
    "lane":            ["divider-line", "dotted-line", "double-line", "solid-line"],
    "road_marking":    ["road-sign-line", "dotted-line"],
    "vehicle":         ["car", "bus", "truck", "motor", "bike"],
    "left_turn":       ["traffic sign"],
    "right_turn":      ["traffic sign"],
}


def _get_yolo(model_path: str):
    global _yolo_model, _yolo_model_path

    if _yolo_model is not None and _yolo_model_path == model_path:
        return _yolo_model

    try:
        from ultralytics import YOLO
        logger.info(f"Loading YOLO model: {model_path}")
        _yolo_model = YOLO(model_path)
        _yolo_model_path = model_path
        logger.info("YOLO ready.")
    except Exception as e:
        logger.warning(f"YOLO load failed: {e} — DGS disabled.")
        _yolo_model = None

    return _yolo_model


def unload_yolo() -> None:
    global _yolo_model, _yolo_model_path
    if _yolo_model is not None:
        del _yolo_model
        _yolo_model = None
        _yolo_model_path = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("YOLO unloaded.")


# ---------------------------------------------------------------------------
# CLIP lazy singleton
# ---------------------------------------------------------------------------
_clip_model = None
_clip_processor = None


def _get_clip(device: str = "cuda") -> tuple:
    global _clip_model, _clip_processor

    if _clip_model is not None:
        return _clip_model, _clip_processor

    try:
        from transformers import CLIPModel, CLIPProcessor
        model_id = "openai/clip-vit-base-patch32"
        logger.info(f"Loading CLIP {model_id} on {device}")
        _clip_processor = CLIPProcessor.from_pretrained(model_id)
        _clip_model = CLIPModel.from_pretrained(model_id).to(device)
        _clip_model.eval()
        logger.info("CLIP ready.")
    except Exception as e:
        logger.warning(f"CLIP load failed: {e} — QFS disabled.")
        _clip_model = None
        _clip_processor = None

    return _clip_model, _clip_processor


def unload_clip() -> None:
    global _clip_model, _clip_processor
    if _clip_model is not None:
        _clip_model.cpu()
        del _clip_model
        _clip_model = None
        _clip_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CLIP unloaded.")


# ---------------------------------------------------------------------------
# Question analysis
# ---------------------------------------------------------------------------

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
    "biển báo": ["traffic_sign"],
    "biển": ["traffic_sign"],
    "đèn giao thông": ["traffic_light"],
    "đèn": ["traffic_light"],
    "làn đường": ["lane"],
    "làn": ["lane"],
    "vạch kẻ": ["road_marking"],
    "vạch": ["road_marking"],
    "tốc độ": ["speed_limit_sign"],
    "rẽ trái": ["left_turn"],
    "rẽ phải": ["right_turn"],
    "xe": ["vehicle"],
}

_INTENT_PATTERNS: list[tuple[QuestionIntent, list[str]]] = [
    (QuestionIntent.TEMPORAL,       ["đầu tiên", "cuối cùng", "trước", "sau", "hiện tại", "đang"]),
    (QuestionIntent.VALUE,          ["bao nhiêu", "mấy", "tốc độ"]),
    (QuestionIntent.DIRECTION,      ["hướng", "rẽ", "đi thẳng", "quay đầu"]),
    (QuestionIntent.IDENTIFICATION, ["biển gì", "loại nào", "là gì"]),
    (QuestionIntent.EXISTENCE,      ["có", "không"]),
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


# ---------------------------------------------------------------------------
# English CLIP query builder
# CLIP được train trên English — dùng keyword mapping thay vì raw Vietnamese text
# ---------------------------------------------------------------------------

_OBJECT_TO_EN: dict[str, str] = {
    "traffic_sign":    "traffic road sign warning",
    "traffic_light":   "traffic light signal red green yellow",
    "lane":            "road lane marking white line",
    "road_marking":    "road marking painted line pavement",
    "speed_limit_sign":"speed limit number sign",
    "left_turn":       "left turn arrow direction",
    "right_turn":      "right turn arrow direction",
    "vehicle":         "car vehicle motorcycle truck",
}

_INTENT_TO_EN: dict[str, str] = {
    "temporal":       "first last sequence moment",
    "value":          "number count amount",
    "direction":      "direction left right straight ahead",
    "identification": "identify type kind label",
    "existence":      "present visible appearing",
    "unknown":        "",
}


def _build_clip_query(analysis: QueryAnalysis) -> str:
    """Build English query cho CLIP từ kết quả analyze_question.

    Ví dụ:
      - "Có biển báo tốc độ không?" → "speed limit number sign present visible appearing"
      - "Đèn tín hiệu màu gì?"     → "traffic light signal red green yellow identify type kind label"
    """
    parts: list[str] = []

    for obj in analysis.target_objects:
        en = _OBJECT_TO_EN.get(obj, "")
        if en:
            parts.append(en)

    intent_en = _INTENT_TO_EN.get(analysis.intent.value, "")
    if intent_en:
        parts.append(intent_en)

    # Fallback nếu không extract được gì
    if not parts:
        parts.append("traffic road dashcam scene")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Frame scoring
# ---------------------------------------------------------------------------

def _compute_qfs_scores(
    frames: list[Image.Image],
    clip_query: str,
    device: str = "cuda",
) -> np.ndarray:
    """Question-Frame Similarity dùng CLIP cosine similarity.

    clip_query phải là English (dùng _build_clip_query để build từ QueryAnalysis).
    Trả về float32 [0,1]. Fallback uniform nếu CLIP không available.
    """
    n = len(frames)
    if n == 0:
        return np.array([], dtype=np.float32)

    model, processor = _get_clip(device)
    if model is None or processor is None:
        return np.ones(n, dtype=np.float32)

    try:
        dev = next(model.parameters()).device

        text_inputs = processor(
            text=[clip_query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        text_inputs = {k: v.to(dev) for k, v in text_inputs.items()}

        def _to_tensor(out) -> torch.Tensor:
            """Normalize output: tensor thẳng hoặc dataclass (transformers 5.x)."""
            if isinstance(out, torch.Tensor):
                return out
            if hasattr(out, "image_embeds"):
                return out.image_embeds
            if hasattr(out, "text_embeds"):
                return out.text_embeds
            if hasattr(out, "pooler_output"):
                return out.pooler_output
            # Fallback: lấy phần tử đầu tiên
            return out[0]

        img_features_list: list[torch.Tensor] = []
        batch_size = 16
        for i in range(0, n, batch_size):
            batch = frames[i : i + batch_size]
            img_inputs = processor(images=batch, return_tensors="pt")
            img_inputs = {k: v.to(dev) for k, v in img_inputs.items()}
            with torch.no_grad():
                feats = _to_tensor(model.get_image_features(**img_inputs))
            img_features_list.append(feats)

        with torch.no_grad():
            text_feat = _to_tensor(model.get_text_features(**text_inputs))
            img_feat = torch.cat(img_features_list, dim=0)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ text_feat.T).squeeze(-1).float().cpu().numpy()

        lo, hi = float(sims.min()), float(sims.max())
        if hi - lo < 1e-6:
            return np.ones(n, dtype=np.float32)
        return ((sims - lo) / (hi - lo)).astype(np.float32)

    except Exception as e:
        logger.warning(f"QFS failed: {e}")
        return np.ones(n, dtype=np.float32)


def _compute_frame_histograms(frames: list[Image.Image]) -> list[np.ndarray]:
    hists: list[np.ndarray] = []
    for img in frames:
        arr = np.array(img)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        hists.append(h.flatten().astype(np.float32))
    return hists


def _compute_ifd_scores(frames: list[Image.Image], window: int = 2) -> np.ndarray:
    """Inter-Frame Distinctiveness: frame càng khác hàng xóm → score cao."""
    n = len(frames)
    if n == 0:
        return np.array([], dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    hists = _compute_frame_histograms(frames)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        dists = []
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j == i:
                continue
            sim = float(cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL))
            dists.append(1.0 - sim)
        scores[i] = float(np.mean(dists)) if dists else 1.0

    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-8:
        return np.ones(n, dtype=np.float32)
    return (scores - lo) / (hi - lo)


def _compute_dgs_scores(
    frames: list[Image.Image],
    target_objects: list[str],
    yolo_model_path: str,
    confidence: float = 0.25,
) -> np.ndarray:
    """Detection-Guided Scoring (DGS) dùng YOLO.

    Frame chứa target object → score × 2.0 (double boost).
    Frame chứa object khác  → score × 0.5 (penalty).
    Công thức từ repo gốc: frame_scorer.py DetectionScoringStrategy.
    """
    n = len(frames)
    if n == 0:
        return np.array([], dtype=np.float32)

    model = _get_yolo(yolo_model_path)
    if model is None:
        return np.ones(n, dtype=np.float32)

    # Build set of YOLO class name substrings cho target objects
    target_yolo: set[str] = set()
    for obj in target_objects:
        for cls_name in _OBJECT_TO_YOLO_CLASSES.get(obj, []):
            target_yolo.add(cls_name.lower())

    try:
        # Chạy YOLO trên tất cả frames (batch)
        frame_arrays = [np.array(img) for img in frames]
        results = model(frame_arrays, conf=confidence, verbose=False, imgsz=640)

        scores = np.zeros(n, dtype=np.float32)
        for i, res in enumerate(results):
            if res.boxes is None or len(res.boxes) == 0:
                continue

            total = 0.0
            for j in range(len(res.boxes)):
                conf_val = float(res.boxes.conf[j])
                cls_id = int(res.boxes.cls[j])
                cls_name = (res.names.get(cls_id, "") or "").lower()

                if target_yolo:
                    # Check xem detection có match với target không
                    matched = any(
                        t in cls_name or cls_name in t
                        for t in target_yolo
                    )
                    total += conf_val * 2.0 if matched else conf_val * 0.5
                else:
                    total += conf_val  # không có target → dùng raw confidence

            scores[i] = total

        # Normalize to [0, 1]
        mx = scores.max()
        if mx > 0:
            scores /= mx
        return scores

    except Exception as e:
        logger.warning(f"DGS failed: {e}")
        return np.ones(n, dtype=np.float32)


def _temporal_prior(n: int, temporal_hints: list[str]) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float32)
    idx = np.arange(n, dtype=np.float32)
    if "first" in temporal_hints:
        return 1.0 - (idx / max(1.0, float(n - 1)))
    if "last" in temporal_hints:
        return idx / max(1.0, float(n - 1))
    return np.ones(n, dtype=np.float32)


# ---------------------------------------------------------------------------
# Diverse Top-K (giống repo gốc: _select_diverse_top_k)
# ---------------------------------------------------------------------------

def _diverse_top_k(
    scores: np.ndarray,
    hists: list[np.ndarray],
    k: int,
    diversity_threshold: float = 0.35,
) -> list[int]:
    n = len(scores)
    if k >= n:
        return list(range(n))

    ordered = [int(i) for i in np.argsort(-scores)]
    selected: list[int] = []
    rejected: list[int] = []

    for candidate in ordered:
        if len(selected) >= k:
            break
        if not selected:
            selected.append(candidate)
            continue
        min_dist = min(
            1.0 - float(cv2.compareHist(hists[candidate], hists[sel], cv2.HISTCMP_CORREL))
            for sel in selected
        )
        if min_dist >= diversity_threshold:
            selected.append(candidate)
        else:
            rejected.append(candidate)

    # Fallback: fill remaining slots từ rejected (theo thứ tự score)
    for candidate in rejected:
        if len(selected) >= k:
            break
        selected.append(candidate)

    return sorted(selected)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# Weights theo repo gốc: QFS=0.5, IFD=0.2, Temporal=0.3
_W_QFS      = 0.5
_W_IFD      = 0.2
_W_TEMPORAL = 0.3


def select_query_aware_frames(
    frames: list[Image.Image],
    analysis: QueryAnalysis,
    max_frames: int,
    clip_enabled: bool = True,
    clip_device: str = "cuda",
    yolo_model_path: Optional[str] = None,
    yolo_confidence: float = 0.25,
    diversity_threshold: float = 0.35,
) -> tuple[list[Image.Image], list[int], list[float]]:
    """Chọn keyframes từ candidate pool theo query-aware scoring.

    Full pipeline (giống repo gốc):
      Score = QFS(0.5)·CLIP + DGS(0.3)·YOLO + IFD(0.2)

    Fallback khi thiếu model:
      - Không YOLO: QFS(0.55) + IFD(0.2) + Temporal(0.25)
      - Không CLIP:           DGS(0.5) + IFD(0.2) + Temporal(0.3)
      - Không cả 2:                       IFD(0.5) + Temporal(0.5)
    """
    if not frames:
        return [], [], []

    n = len(frames)
    k = max(1, min(max_frames, n))

    temporal = _temporal_prior(n, analysis.temporal_hints)
    has_temporal = bool(analysis.temporal_hints) or analysis.intent == QuestionIntent.TEMPORAL
    ifd = _compute_ifd_scores(frames)

    # --- QFS (CLIP) ---
    qfs = np.ones(n, dtype=np.float32)
    clip_active = False
    if clip_enabled:
        clip_query = _build_clip_query(analysis)
        logger.debug(f"CLIP query: '{clip_query}'")
        qfs_raw = _compute_qfs_scores(frames, clip_query, device=clip_device)
        clip_active = not np.allclose(qfs_raw, 1.0)
        if clip_active:
            qfs = qfs_raw

    # --- DGS (YOLO) ---
    dgs = np.ones(n, dtype=np.float32)
    yolo_active = False
    if yolo_model_path:
        dgs_raw = _compute_dgs_scores(
            frames, analysis.target_objects, yolo_model_path, yolo_confidence
        )
        # yolo_active = True chỉ khi có detection thực sự (max > 0).
        # Nếu YOLO load thất bại → trả về ones → allclose(1.0) → False.
        # Nếu YOLO load thành công nhưng không detect được gì → trả về zeros
        # → max = 0 → yolo_active = False → tránh giảm weight IFD không cần thiết.
        yolo_active = float(dgs_raw.max()) > 1e-6
        if yolo_active:
            dgs = dgs_raw

    # --- Blend (weights theo repo gốc: QFS=0.5, DGS=0.3, IFD=0.2) ---
    if clip_active and yolo_active:
        if has_temporal:
            final_scores = 0.35 * qfs + 0.25 * dgs + 0.10 * ifd + 0.30 * temporal
        else:
            final_scores = 0.50 * qfs + 0.30 * dgs + 0.20 * ifd
    elif clip_active:
        if has_temporal:
            final_scores = 0.40 * qfs + 0.15 * ifd + 0.45 * temporal
        else:
            final_scores = 0.55 * qfs + 0.20 * ifd + 0.25 * temporal
    elif yolo_active:
        if has_temporal:
            final_scores = 0.50 * dgs + 0.10 * ifd + 0.40 * temporal
        else:
            final_scores = 0.50 * dgs + 0.20 * ifd + 0.30 * temporal
    else:
        if has_temporal:
            final_scores = 0.25 * ifd + 0.75 * temporal
        else:
            final_scores = 0.55 * ifd + 0.45 * temporal

    # --- Diverse Top-K ---
    hists = _compute_frame_histograms(frames)
    top_idx = _diverse_top_k(final_scores, hists, k, diversity_threshold)

    selected = [frames[i] for i in top_idx]
    selected_scores = [float(final_scores[i]) for i in top_idx]
    return selected, top_idx, selected_scores
