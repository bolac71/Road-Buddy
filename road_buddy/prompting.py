from __future__ import annotations

import re
from typing import Iterable


_PROMPT_TEMPLATE = (
    "{system_hint}\n\n"
    "Dựa trên các khung hình trích từ video dashcam, hãy trả lời câu hỏi trắc nghiệm sau.\n"
    "Chỉ được chọn DUY NHẤT một đáp án đúng nhất.\n\n"
    "Câu hỏi: {question}\n\n"
    "Lựa chọn:\n{choices}\n\n"
    "Bắt buộc: dòng đầu tiên chỉ được ghi DUY NHẤT một ký tự in hoa A/B/C/D, không giải thích.\n"
    "Ví dụ hợp lệ: A"
)


def build_prompt(question: str, choices: Iterable[str], system_hint: str) -> str:
    choices_text = "\n".join(str(c) for c in choices)
    return _PROMPT_TEMPLATE.format(system_hint=system_hint, question=question, choices=choices_text)


def extract_choice_letters(choices: Iterable[str]) -> list[str]:
    out: list[str] = []
    for choice in choices:
        match = re.match(r"^\s*([A-D])", str(choice), flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter not in out:
                out.append(letter)
    return out


def extract_final_letter(text: str, allowed_letters: list[str]) -> str | None:
    if not text:
        return None

    allowed = [x.upper() for x in allowed_letters if x]
    if not allowed:
        return None

    # Prefer explicit forms such as "dap an: C" or "answer = B".
    explicit_pat = re.compile(
        r"(?:dap\s*an|đáp\s*án|answer)\s*[:\-]?\s*([A-D])\b",
        flags=re.IGNORECASE,
    )
    explicit_matches = explicit_pat.findall(text)
    if explicit_matches:
        letter = explicit_matches[-1].upper()
        if letter in allowed:
            return letter

    # Fallback: pick the last standalone choice letter in model output.
    standalone = re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE)
    for token in reversed(standalone):
        letter = token.upper()
        if letter in allowed:
            return letter

    return None
