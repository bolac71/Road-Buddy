from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetSample:
    id: str
    question: str
    choices: list[str]
    video_path: str
    dataset_type: str = "other"


@dataclass
class SelectedFrames:
    indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    count: int = 0
