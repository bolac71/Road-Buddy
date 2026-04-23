from __future__ import annotations
from road_buddy_agentic.schemas.support import SupportBrief
from road_buddy_agentic.validators.answer_leak_validator import detect_answer_leak


def validate_support_brief(brief: SupportBrief) -> SupportBrief:
    brief.answer_leak_detected = detect_answer_leak(brief.raw_text)
    if brief.answer_leak_detected:
        brief.status = "invalid_answer_leak"
        brief.normalized_payload["answer_forbidden"] = True
        if "final_note" not in brief.normalized_payload:
            brief.normalized_payload["final_note"] = "Không được suy ra đáp án cuối cùng."
    return brief
