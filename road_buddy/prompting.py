from __future__ import annotations

import re
from typing import Iterable


_PROMPT_TEMPLATE = (
    "{system_hint}\n\n"
    "Dựa trên các khung hình trích từ video dashcam, hãy trả lời câu hỏi trắc nghiệm sau.\n"
    "Chỉ được chọn DUY NHẤT một đáp án đúng nhất.\n\n"
    "{query_context}"
    "Câu hỏi: {question}\n\n"
    "Lựa chọn:\n{choices}\n\n"
    "Bắt buộc: dòng đầu tiên chỉ được ghi DUY NHẤT một ký tự in hoa A/B/C/D, không giải thích.\n"
    "Ví dụ hợp lệ: A"
)


def build_prompt(
    question: str,
    choices: Iterable[str],
    system_hint: str,
    target_objects: list[str] | None = None,
    temporal_hints: list[str] | None = None,
) -> str:
    choices_text = "\n".join(str(c) for c in choices)
    context_parts: list[str] = []
    if target_objects:
        context_parts.append(f"Đối tượng cần chú ý: {', '.join(target_objects)}")
    if temporal_hints:
        context_parts.append(f"Gợi ý thời điểm quan sát: {', '.join(temporal_hints)}")

    query_context = ""
    if context_parts:
        query_context = "Thông tin bổ sung:\n" + "\n".join(context_parts) + "\n\n"

    return _PROMPT_TEMPLATE.format(
        system_hint=system_hint,
        query_context=query_context,
        question=question,
        choices=choices_text,
    )


def extract_choice_letters(choices: Iterable[str]) -> list[str]:
    out: list[str] = []
    for choice in choices:
        match = re.match(r"^\s*([A-D])", str(choice), flags=re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter not in out:
                out.append(letter)
    return out


def _strip_thinking_blocks(text: str) -> str:
    """Tách phần answer thực sự từ output có thinking tags.

    - Nếu có </think> → chỉ lấy text sau tag đó (phần answer).
    - Nếu có <think> mà không có </think> → thinking bị truncate, trả về chuỗi rỗng.
    - Nếu không có tag nào → trả về nguyên text.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    if "<think>" in text:
        return ""
    return text


def extract_final_letter(text: str, allowed_letters: list[str]) -> str | None:
    if not text:
        return None

    allowed = [x.upper() for x in allowed_letters if x]
    if not allowed:
        return None

    # Strip thinking blocks nếu model dùng thinking mode (Qwen3.5, DeepSeek, ...).
    answer_text = _strip_thinking_blocks(text)

    # Prefer explicit forms such as "dap an: C" or "answer = B".
    explicit_pat = re.compile(
        r"(?:dap\s*an|đáp\s*án|answer)\s*[:\-]?\s*([A-D])\b",
        flags=re.IGNORECASE,
    )
    explicit_matches = explicit_pat.findall(answer_text)
    if explicit_matches:
        letter = explicit_matches[-1].upper()
        if letter in allowed:
            return letter

    # Fallback: pick the last standalone choice letter in answer output.
    standalone = re.findall(r"\b([A-D])\b", answer_text, flags=re.IGNORECASE)
    for token in reversed(standalone):
        letter = token.upper()
        if letter in allowed:
            return letter

    return None
