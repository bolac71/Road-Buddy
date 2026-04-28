from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


# ─── Feature extraction ────────────────────────────────────────────────────────

def _laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _canny_edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).mean())


def _brightness_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean() / 255.0)


def _novelty_score(image: np.ndarray, prev_image: np.ndarray | None) -> float:
    if prev_image is None:
        return 1.0
    gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    diff = cv2.absdiff(gray1, gray2)
    return float(diff.mean() / 255.0)


def _crop_roi(image: np.ndarray, roi: list[float]) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, int(roi[0] * w)))
    y1 = max(0, min(h - 1, int(roi[1] * h)))
    x2 = max(x1 + 1, min(w, int(roi[2] * w)))
    y2 = max(y1 + 1, min(h, int(roi[3] * h)))
    return image[y1:y2, x1:x2]


def _roi_feature_max(image: np.ndarray, rois: list[list[float]], feature_fn) -> float:
    if not rois:
        return float(feature_fn(image))
    crops = [_crop_roi(image, roi) for roi in rois]
    vals = [float(feature_fn(c)) for c in crops if c.size > 0]
    return float(max(vals)) if vals else float(feature_fn(image))


# ─── Normalizer & scoring utils ────────────────────────────────────────────────

def _minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin, vmax = min(values), max(values)
    if math.isclose(vmin, vmax):
        return [0.5] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def _center_bias(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    x = index / (total - 1)
    return 1.0 - abs(x - 0.5) * 2.0


# ─── Temporal NMS ──────────────────────────────────────────────────────────────

def _temporal_nms(
    times_sec: Sequence[float],
    scores: Sequence[float],
    top_k: int,
    min_gap_sec: float,
) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for idx in order:
        t = times_sec[idx]
        if all(abs(times_sec[k] - t) >= min_gap_sec for k in keep):
            keep.append(idx)
        if len(keep) >= top_k:
            break
    return sorted(keep, key=lambda i: times_sec[i])


# ─── Question-type inference ───────────────────────────────────────────────────

def infer_question_type(question: str) -> str:
    q = question.lower()
    if "đúng hay sai" in q or "phải không" in q or "đúng không" in q:
        return "verification"
    if "có xuất hiện" in q or "có đèn" in q or "có biển" in q:
        return "object_presence"
    if "bao nhiêu" in q or "tốc độ" in q or "khoảng cách" in q:
        return "information_reading"
    if "hướng nào" in q or "đi theo hướng nào" in q or "muốn đi" in q:
        return "navigation"
    if "có mấy" in q or "bao nhiêu biển" in q or "đếm" in q:
        return "counting"
    if "được phép" in q or "có được" in q:
        return "rule_compliance"
    return "sign_identification"


# ─── Per-type policies (tuned weights) ─────────────────────

_SIGN_POLICY: dict = {
    "top_k": 4,
    "temporal_nms_gap_sec": 0.40,
    "refine_window_sec": 0.85,
    "fps_refine": 7.0,
    "roi_regions": [
        [0.30, 0.00, 0.72, 0.34],
        [0.52, 0.00, 1.00, 0.42],
        [0.22, 0.14, 0.82, 0.56],
    ],
    "weights": {
        "sharpness": 0.06, "edge_density": 0.06, "brightness": 0.02,
        "novelty": 0.00, "center_bias": 0.00,
        "roi_sharpness": 0.38, "roi_edge_density": 0.40, "roi_brightness": 0.08,
    },
}

_LANE_POLICY: dict = {
    "top_k": 4,
    "temporal_nms_gap_sec": 0.45,
    "refine_window_sec": 1.0,
    "fps_refine": 6.0,
    "roi_regions": [
        [0.28, 0.00, 0.70, 0.28],
        [0.18, 0.24, 0.82, 0.72],
        [0.00, 0.45, 1.00, 1.00],
    ],
    "weights": {
        "sharpness": 0.12, "edge_density": 0.16, "brightness": 0.04,
        "novelty": 0.00, "center_bias": 0.02,
        "roi_sharpness": 0.24, "roi_edge_density": 0.28, "roi_brightness": 0.04,
    },
}

_COVERAGE_POLICY: dict = {
    "top_k": 5,
    "temporal_nms_gap_sec": 0.35,
    "refine_window_sec": 1.1,
    "fps_refine": 6.0,
    "roi_regions": [
        [0.18, 0.00, 0.82, 0.42],
        [0.00, 0.18, 1.00, 0.82],
    ],
    "weights": {
        "sharpness": 0.10, "edge_density": 0.12, "brightness": 0.04,
        "novelty": 0.00, "center_bias": 0.00,
        "roi_sharpness": 0.26, "roi_edge_density": 0.32, "roi_brightness": 0.06,
    },
}

_TYPE_TO_POLICY: dict[str, dict] = {
    "sign_identification": _SIGN_POLICY,
    "information_reading": _SIGN_POLICY,
    "navigation": _SIGN_POLICY,
    "other": _SIGN_POLICY,
    "rule_compliance": _LANE_POLICY,
    "verification": _LANE_POLICY,
    "object_presence": _COVERAGE_POLICY,
    "counting": _COVERAGE_POLICY,
}


def _get_policy(question_type: str) -> dict:
    return _TYPE_TO_POLICY.get(question_type, _SIGN_POLICY)


# ─── Internal frame types ──────────────────────────────────────────────────────

@dataclass
class _Frame:
    frame_idx: int
    time_sec: float
    image: np.ndarray


@dataclass
class _ScoredFrame:
    frame_idx: int
    time_sec: float
    image: np.ndarray
    score: float


# ─── Video sampling ────────────────────────────────────────────────────────────

def _sample_uniform(
    video_path: str,
    fps_sample: float,
    max_frames: int | None = None,
) -> list[_Frame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(int(round(native_fps / fps_sample)), 1)
    sampled: list[_Frame] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            sampled.append(_Frame(frame_idx=frame_idx, time_sec=frame_idx / native_fps, image=frame))
            if max_frames is not None and len(sampled) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return sampled


def _sample_window(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps_sample: float,
    max_frames: int | None = None,
) -> list[_Frame]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps if total_frames > 0 else 0.0

    start_sec = max(0.0, start_sec)
    end_sec = min(duration_sec, end_sec)
    if end_sec <= start_sec:
        cap.release()
        return []

    start_frame = int(start_sec * native_fps)
    end_frame = min(int(end_sec * native_fps), total_frames - 1 if total_frames > 0 else int(end_sec * native_fps))
    frame_step = max(int(round(native_fps / fps_sample)), 1)

    sampled: list[_Frame] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_step == 0:
            sampled.append(_Frame(frame_idx=frame_idx, time_sec=frame_idx / native_fps, image=frame))
            if max_frames is not None and len(sampled) >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return sampled


# ─── Frame scoring ─────────────────────────────────────────────────────────────

def _score_frames(frames: list[_Frame], policy: dict) -> list[_ScoredFrame]:
    if not frames:
        return []

    weights = policy.get("weights", {})
    rois = policy.get("roi_regions", [])

    raw_sharpness, raw_edges, raw_brightness, raw_novelty = [], [], [], []
    raw_roi_sharpness, raw_roi_edges, raw_roi_brightness = [], [], []

    prev_image = None
    for fr in frames:
        img = fr.image
        raw_sharpness.append(_laplacian_variance(img))
        raw_edges.append(_canny_edge_density(img))
        raw_brightness.append(_brightness_score(img))
        raw_novelty.append(_novelty_score(img, prev_image))
        raw_roi_sharpness.append(_roi_feature_max(img, rois, _laplacian_variance))
        raw_roi_edges.append(_roi_feature_max(img, rois, _canny_edge_density))
        raw_roi_brightness.append(_roi_feature_max(img, rois, _brightness_score))
        prev_image = img

    norm_sharp = _minmax_normalize(raw_sharpness)
    norm_edges = _minmax_normalize(raw_edges)
    norm_bright = _minmax_normalize(raw_brightness)
    norm_novelty = _minmax_normalize(raw_novelty)
    norm_roi_sharp = _minmax_normalize(raw_roi_sharpness)
    norm_roi_edges = _minmax_normalize(raw_roi_edges)
    norm_roi_bright = _minmax_normalize(raw_roi_brightness)

    n = len(frames)
    scored: list[_ScoredFrame] = []
    for i, fr in enumerate(frames):
        score = (
            weights.get("sharpness", 0.0) * norm_sharp[i]
            + weights.get("edge_density", 0.0) * norm_edges[i]
            + weights.get("brightness", 0.0) * norm_bright[i]
            + weights.get("novelty", 0.0) * norm_novelty[i]
            + weights.get("center_bias", 0.0) * _center_bias(i, n)
            + weights.get("roi_sharpness", 0.0) * norm_roi_sharp[i]
            + weights.get("roi_edge_density", 0.0) * norm_roi_edges[i]
            + weights.get("roi_brightness", 0.0) * norm_roi_bright[i]
        )
        scored.append(_ScoredFrame(frame_idx=fr.frame_idx, time_sec=fr.time_sec, image=fr.image, score=float(score)))
    return scored


def _deduplicate(frames: list[_ScoredFrame], tol: float = 1e-6) -> list[_ScoredFrame]:
    if not frames:
        return []
    frames = sorted(frames, key=lambda x: (x.time_sec, -x.score))
    deduped = [frames[0]]
    for fr in frames[1:]:
        if abs(fr.time_sec - deduped[-1].time_sec) > tol:
            deduped.append(fr)
        elif fr.score > deduped[-1].score:
            deduped[-1] = fr
    return deduped


# ─── Public interface ──────────────────────────────────────────────────────────

def sample_frames_heuristic(
    video_path: str,
    question: str,
    num_frames: int,
    max_side: int,
) -> list[Image.Image]:
    """
    Heuristic keyframe selection using visual quality scoring + temporal NMS.

    Selects up to num_frames frames (may return fewer if video is very short).
    Uses question type to pick ROI regions and scoring weights.
    Falls back to uniform sample if heuristic produces nothing.
    """
    qtype = infer_question_type(question)
    policy = _get_policy(qtype)

    fps_coarse = 2.0
    fps_refine = float(policy.get("fps_refine", 6.0))
    max_frames_total = 96
    coarse_top_m = 2
    trim_ratio = 0.10

    top_k = max(num_frames, int(policy.get("top_k", 4)))
    min_gap_sec = float(policy.get("temporal_nms_gap_sec", 0.5))
    refine_window_sec = float(policy.get("refine_window_sec", 1.0))

    # Step 1: coarse uniform sample
    coarse_frames = _sample_uniform(video_path, fps_coarse, max_frames_total)
    if not coarse_frames:
        return []

    # Trim first/last ~10% to avoid logo frames and fade-outs
    n_total = len(coarse_frames)
    if n_total >= 10:
        s = int(n_total * trim_ratio)
        e = int(n_total * (1.0 - trim_ratio))
        coarse_frames = coarse_frames[s:e] or coarse_frames

    # Step 2: score coarse frames
    coarse_scored = _score_frames(coarse_frames, policy)
    coarse_times = [x.time_sec for x in coarse_scored]
    coarse_scores = [x.score for x in coarse_scored]

    # Step 3: pick coarse peaks via temporal NMS
    coarse_keep = _temporal_nms(
        coarse_times, coarse_scores,
        top_k=coarse_top_m,
        min_gap_sec=max(min_gap_sec, refine_window_sec * 0.75),
    )
    coarse_peaks = [coarse_scored[i] for i in coarse_keep]

    # Step 4: refine around each peak at higher fps
    fine_candidates: list[_ScoredFrame] = []
    for peak in coarse_peaks:
        local_frames = _sample_window(
            video_path,
            start_sec=peak.time_sec - refine_window_sec,
            end_sec=peak.time_sec + refine_window_sec,
            fps_sample=fps_refine,
            max_frames=max_frames_total,
        )
        if local_frames:
            fine_candidates.extend(_score_frames(local_frames, policy))

    if not fine_candidates:
        fine_candidates = coarse_scored[:]

    # Step 5: merge coarse + fine, dedup, final NMS
    merged = _deduplicate(coarse_scored + fine_candidates)
    merged_times = [x.time_sec for x in merged]
    merged_scores = [x.score for x in merged]

    final_keep = _temporal_nms(merged_times, merged_scores, top_k=top_k, min_gap_sec=min_gap_sec)
    selected = [merged[i] for i in final_keep]

    if not selected:
        selected = coarse_scored[:num_frames]

    # Convert BGR numpy → PIL, resize if needed
    images: list[Image.Image] = []
    for fr in selected:
        frame_rgb = cv2.cvtColor(fr.image, cv2.COLOR_BGR2RGB)
        if max_side > 0:
            h, w = frame_rgb.shape[:2]
            longest = max(h, w)
            if longest > max_side:
                scale = max_side / float(longest)
                new_w = max(1, int(round(w * scale)))
                new_h = max(1, int(round(h * scale)))
                frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(frame_rgb))

    return images
