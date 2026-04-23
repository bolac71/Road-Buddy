from __future__ import annotations
from road_buddy_agentic.schemas.support import SupportRequest, QuestionProfile
from road_buddy_agentic.schemas.sample import DatasetSample
from road_buddy_agentic.legal_knowledge.loader import load_task_pack


def build_support_request(sample: DatasetSample, profile: QuestionProfile) -> SupportRequest:
    legal_pack = load_task_pack(profile.dataset_type)
    return SupportRequest(
        sample_id=sample.id,
        question=sample.question,
        choices=sample.choices,
        dataset_type=profile.dataset_type,
        intent=profile.intent,
        legal_pack=legal_pack,
        option_semantics=profile.option_semantics,
    )
