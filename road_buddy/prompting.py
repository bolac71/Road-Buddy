from __future__ import annotations

import re
from typing import Iterable


_PROMPT_TEMPLATE = (
    "{system_hint}\n\n"
    "Dựa trên các khung hình trích từ video dashcam, hãy trả lời câu hỏi trắc nghiệm sau.\n"
    "Chỉ được chọn DUY NHẤT một đáp án đúng nhất. "
    "Hãy phân tích kỹ nội dung video trước khi chọn, không thiên vị về vị trí đáp án.\n\n"
    "{query_context}"
    "{thinking_hint}"
    "Câu hỏi: {question}\n\n"
    "Lựa chọn:\n{choices}\n\n"
    "Bắt buộc: dòng đầu tiên chỉ được ghi DUY NHẤT một ký tự in hoa A/B/C/D, không giải thích.\n"
    "Ví dụ hợp lệ: A"
)

_THINKING_HINT = (
    "Trong phần suy nghĩ, hãy kết thúc bằng dòng 'Đáp án: X' "
    "(X là chữ cái A/B/C/D bạn chọn) trước khi đóng thẻ </think>.\n\n"
)


def build_prompt(
    question: str,
    choices: Iterable[str],
    system_hint: str,
    thinking_mode: bool = False,
) -> str:
    choices_text = "\n".join(str(c) for c in choices)
    return _PROMPT_TEMPLATE.format(
        system_hint=system_hint,
        query_context="",
        thinking_hint=_THINKING_HINT if thinking_mode else "",
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


_EXPLICIT_PAT = re.compile(
    r"(?:dap\s*an|đáp\s*án|answer|chọn|chon)\s*[:\-=]?\s*([A-D])\b",
    flags=re.IGNORECASE,
)


def _search_letter(text: str, allowed: list[str]) -> str | None:
    """Tìm đáp án trong đoạn text: ưu tiên explicit pattern, fallback standalone letter."""
    if not text:
        return None
    for m in reversed(_EXPLICIT_PAT.findall(text)):
        if m.upper() in allowed:
            return m.upper()
    for tok in reversed(re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE)):
        if tok.upper() in allowed:
            return tok.upper()
    return None


def extract_final_letter(text: str, allowed_letters: list[str]) -> str | None:
    if not text:
        return None

    allowed = [x.upper() for x in allowed_letters if x]
    if not allowed:
        return None

    # Tách thinking block khỏi phần answer.
    think_content = ""
    if "</think>" in text:
        before, after = text.rsplit("</think>", 1)
        answer_text = after.strip()
        # Lấy nội dung bên trong <think>...</think> để dùng làm fallback.
        think_content = before.split("<think>", 1)[-1].strip() if "<think>" in before else before.strip()
    elif "<think>" in text:
        # Thinking bị truncate — không có </think>
        answer_text = ""
        think_content = text.split("<think>", 1)[1].strip()
    else:
        answer_text = text
        think_content = ""

    # Priority 1: tìm trong phần answer (sau </think>).
    result = _search_letter(answer_text, allowed)
    if result:
        return result

    # Priority 2: tìm "Đáp án: X" trong thinking (dòng cam kết mà prompt yêu cầu).
    # Dùng explicit pattern trước để tránh false positive từ letters trong lúc phân tích.
    if think_content:
        for m in reversed(_EXPLICIT_PAT.findall(think_content)):
            if m.upper() in allowed:
                return m.upper()

    # Priority 3: last standalone letter trong thinking (kém tin cậy nhất, hơn fallback A).
    if think_content:
        for tok in reversed(re.findall(r"\b([A-D])\b", think_content, flags=re.IGNORECASE)):
            if tok.upper() in allowed:
                return tok.upper()

    # Priority 4: binary heuristic cho câu Đúng/Sai (chỉ 2 lựa chọn).
    # Model có thể trả về "Đúng"/"Sai" thay vì "A"/"B".
    if len(allowed) == 2:
        full_text = (answer_text + " " + think_content).lower()
        _YES = ("đúng", "có", "phải", "yes", "true", "correct", "right")
        _NO = ("sai", "không", "không phải", "no", "false", "wrong", "incorrect")
        yes_hits = sum(1 for w in _YES if w in full_text)
        no_hits = sum(1 for w in _NO if w in full_text)
        if yes_hits != no_hits:
            # Map về đúng letter: lựa chọn đầu tiên thường là "Đúng", thứ hai là "Sai"
            return allowed[0] if yes_hits > no_hits else allowed[1]

    return None
