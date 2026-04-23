from __future__ import annotations
from road_buddy_agentic.schemas.support import SupportBrief, SupportRequest
from road_buddy_agentic.support_agents.base import SupportAgent


class NoopSupportAgent(SupportAgent):
    def generate_support_brief(self, request: SupportRequest) -> SupportBrief:
        payload = {
            "task_type": request.dataset_type,
            "intent": request.intent,
            "legal_topics": [],
            "option_hints": [],
            "evidence_checklist": [],
            "answer_forbidden": True,
        }
        return SupportBrief(
            status="disabled",
            legal_topics=[],
            option_hints=[],
            evidence_checklist=[],
            raw_text="",
            normalized_payload=payload,
            answer_leak_detected=False,
        )
