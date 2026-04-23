from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def extract_frames_from_video(
    video_path: str,
    max_frames: int = 10,
) -> list[np.ndarray]:
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

    if total_frames <= 0:
        cap.release()
        return frames_bgr

    n = min(max_frames, total_frames)
    if n == 1:
        frame_indices = [0]
    else:
        frame_indices = [
            int(round(i * (total_frames - 1) / (n - 1))) for i in range(n)
        ]
    frame_indices = sorted(set(frame_indices))

    if not frame_indices:
        frame_indices = [0]

    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames_bgr.append(frame)

    cap.release()
    return frames_bgr


def sample_video_frames(
    video_path: str,
    num_frames: int,
    max_side: int,
    sample_fps: float = 2.0,
) -> list[Image.Image]:
    frames = extract_frames_from_video(video_path, max_frames=num_frames)
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
