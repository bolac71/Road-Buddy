from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract_frames_1fps_from_video(video_path: str, max_frames: int = 10) -> list[np.ndarray]:
    frames_bgr: list[np.ndarray] = []

    if max_frames <= 0:
        return frames_bgr

    p = Path(video_path)
    if not p.exists():
        return frames_bgr

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        return frames_bgr

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if total_frames <= 0:
        cap.release()
        return frames_bgr

    if fps <= 0:
        fps = 25.0

    duration_sec = total_frames / fps
    max_seconds = int(math.floor(duration_sec))

    frame_indices: list[int] = []
    for sec in range(max_seconds):
        idx = int(sec * fps)
        if idx < total_frames:
            frame_indices.append(idx)

    if not frame_indices:
        frame_indices = [0]

    frame_indices = sorted(set(frame_indices))[:max_frames]
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames_bgr.append(frame)

    cap.release()
    return frames_bgr


def select_topk_frames_multiframe(
    annotated_pil_frames: list[Image.Image],
    num_boxes_list: list[int],
    top_k: int = 10,
) -> list[Image.Image]:
    if not annotated_pil_frames:
        return []

    n = len(annotated_pil_frames)
    top_k = max(1, min(int(top_k), n))

    if len(num_boxes_list) != n:
        num_boxes_list = [0] * n

    if sum(num_boxes_list) == 0:
        indices = np.linspace(0, n - 1, top_k, dtype=int)
        selected_indices = sorted(set(indices.tolist()))
    else:
        indices = list(range(n))
        selected_indices = sorted(sorted(indices, key=lambda i: num_boxes_list[i], reverse=True)[:top_k])

    return [annotated_pil_frames[i] for i in selected_indices]


def sample_video_frames(video_path: str, num_frames: int, max_side: int) -> list[Image.Image]:
    # Backward-compatible helper: now built on 1fps extraction.
    frames = extract_frames_1fps_from_video(video_path, max_frames=num_frames)
    if not frames:
        return []

    images: list[Image.Image] = []
    for frame_bgr in frames:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
