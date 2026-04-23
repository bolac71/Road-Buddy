from __future__ import annotations


def analyze_intent(question: str) -> dict:
    q = (question or "").lower()

    if any(x in q for x in ["đầu tiên", "cuối cùng", "trước", "sau", "hiện tại", "đang"]):
        intent = "temporal"
    elif any(x in q for x in ["bao nhiêu", "mấy", "tốc độ", "khung giờ", "khoảng cách"]):
        intent = "value"
    elif any(x in q for x in ["hướng", "rẽ", "đi thẳng", "quay đầu"]):
        intent = "direction"
    elif any(x in q for x in ["biển gì", "loại nào", "là gì", "ý nghĩa gì"]):
        intent = "identification"
    elif any(x in q for x in ["có", "không", "xuất hiện"]):
        intent = "existence"
    else:
        intent = "unknown"

    return {
        "intent": intent,
        "keywords": [],
        "target_objects": [],
        "temporal_hints": [],
    }
