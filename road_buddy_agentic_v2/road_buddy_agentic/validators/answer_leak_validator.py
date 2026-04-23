from __future__ import annotations

import json
import re
from typing import Any


LEAK_PATTERNS = [
    r"\bđáp án\s*(đúng|cuối cùng|là)?\s*[:\-]?\s*[ABCD]\b",
    r"\bchọn\s*[ABCD]\b",
    r"\bfinal answer\s*[:\-]?\s*[ABCD]\b",
    r"\bthe answer is\s*[ABCD]\b",
    r"\bcorrect answer\s*[:\-]?\s*[ABCD]\b",
]


def _stringify(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return str(payload)


def detect_answer_leak(raw_text: str, normalized_payload: dict | None = None) -> bool:
    text = (raw_text or "").strip()

    for pattern in LEAK_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    if normalized_payload:
        flat = _stringify(normalized_payload)
        for pattern in LEAK_PATTERNS:
            if re.search(pattern, flat, flags=re.IGNORECASE):
                return True

    return False
