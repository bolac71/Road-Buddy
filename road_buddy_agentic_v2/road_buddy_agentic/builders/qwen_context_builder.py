from __future__ import annotations
from road_buddy_agentic.schemas.sample import DatasetSample
from road_buddy_agentic.schemas.support import SupportBrief


def build_qwen_context(sample: DatasetSample, support_brief: SupportBrief) -> str:
    choices_text = "\n".join(sample.choices)
    checklist = "\n".join(f"- {x}" for x in support_brief.evidence_checklist)
    topics = ", ".join(support_brief.legal_topics)
    option_lines = []
    for item in support_brief.option_hints:
        if not isinstance(item, dict):
            continue
        option_lines.append(
            f"- {item.get('option_letter', '')}: {item.get('option_text', '')} | "
            f"meaning={item.get('legal_meaning', '')} | "
            f"visual={item.get('visual_cues', [])} | "
            f"exclude={item.get('exclusion_cues', [])}"
        )
    option_text = "\n".join(option_lines)

    return (
        "THÔNG TIN HỖ TRỢ PHÁP LÝ / NHẬN DIỆN (không phải đáp án cuối cùng):\n"
        f"- Legal topics: {topics}\n"
        f"- Checklist:\n{checklist}\n"
        f"- Option hints:\n{option_text}\n\n"
        f"Câu hỏi gốc: {sample.question}\n\n"
        f"Lựa chọn:\n{choices_text}\n"
    )
