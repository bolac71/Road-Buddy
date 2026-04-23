from __future__ import annotations
from road_buddy_agentic.schemas.support import QuestionProfile
from road_buddy_agentic.schemas.sample import DatasetSample
from road_buddy_agentic.analyzers.task_type_router import resolve_dataset_type
from road_buddy_agentic.analyzers.intent_router import analyze_intent
from road_buddy_agentic.analyzers.choice_semantic_parser import parse_choice_semantics


def build_question_profile(sample: DatasetSample) -> QuestionProfile:
    dataset_type = resolve_dataset_type(sample.dataset_type)
    intent_info = analyze_intent(sample.question)
    option_semantics = parse_choice_semantics(sample.choices)
    return QuestionProfile(
        dataset_type=dataset_type,
        intent=intent_info["intent"],
        keywords=intent_info.get("keywords", []),
        target_objects=intent_info.get("target_objects", []),
        temporal_hints=intent_info.get("temporal_hints", []),
        option_semantics=option_semantics,
    )
