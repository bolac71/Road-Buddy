from __future__ import annotations


def parse_choice_semantics(choices: list[str]) -> list[dict]:
    out: list[dict] = []
    for choice in choices:
        text = str(choice).strip()
        upper = text.upper()
        letter = upper[0] if upper and upper[0] in "ABCD" else ""
        lowered = text.lower()

        tags: list[str] = []
        if "biển cấm" in lowered:
            tags.append("prohibitory_sign")
        if "biển nguy hiểm" in lowered or "cảnh báo" in lowered:
            tags.append("warning_sign")
        if "biển chỉ dẫn" in lowered:
            tags.append("guide_sign")
        if "biển hiệu lệnh" in lowered:
            tags.append("mandatory_sign")
        if "tốc độ" in lowered or "km/h" in lowered:
            tags.append("speed_related")
        if any(x in lowered for x in ["rẽ trái", "rẽ phải", "đi thẳng", "quay đầu"]):
            tags.append("direction_related")

        out.append({
            "option_letter": letter,
            "option_text": text,
            "semantic_tags": tags,
        })
    return out
