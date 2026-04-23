from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuestionProfile:
    dataset_type: str
    intent: str
    keywords: list[str] = field(default_factory=list)
    target_objects: list[str] = field(default_factory=list)
    temporal_hints: list[str] = field(default_factory=list)
    option_semantics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SupportRequest:
    sample_id: str
    question: str
    choices: list[str]
    dataset_type: str
    intent: str
    legal_pack: dict[str, Any]
    option_semantics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SupportBrief:
    status: str
    legal_topics: list[str] = field(default_factory=list)
    option_hints: list[dict[str, Any]] = field(default_factory=list)
    evidence_checklist: list[str] = field(default_factory=list)
    raw_text: str = ""
    normalized_payload: dict[str, Any] = field(default_factory=dict)
    answer_leak_detected: bool = False


@dataclass
class RunRecord:
    qid: str
    dataset_type: str
    intent: str
    support_provider: str
    support_model: str
    support_enabled: bool
    support_status: str
    qwen_answer: str = ""
    gt_label: str = ""
    is_correct: bool | None = None
