from __future__ import annotations

import json
import re
from typing import Any

CHOICE_PREFIX_RE = re.compile(r"^\s*[A-Da-d][\.\)\:\-]\s*")


def strip_choice_prefix(text: str) -> str:
    return CHOICE_PREFIX_RE.sub("", (text or "").strip()).strip()


def _obj_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _choice_to_text(choice: Any) -> str:
    if isinstance(choice, dict):
        raw = str(
            choice.get(
                "text",
                choice.get(
                    "option_text",
                    choice.get(
                        "label",
                        choice.get("name", ""),
                    ),
                ),
            )
        ).strip()
        return strip_choice_prefix(raw)

    raw = str(choice).strip()
    return strip_choice_prefix(raw)


def _extract_balanced_json(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue

        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return ""


def extract_json_block(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}

    fenced = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
        balanced = _extract_balanced_json(candidate)
        if balanced:
            try:
                return json.loads(balanced)
            except Exception:
                pass

    balanced = _extract_balanced_json(text)
    if balanced:
        try:
            return json.loads(balanced)
        except Exception:
            pass

    return {}


def _clean_str_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for v in values:
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def normalize_support_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    normalized: dict[str, Any] = {
        "task_type": str(payload.get("task_type", "")).strip(),
        "intent": str(payload.get("intent", "")).strip(),
        "legal_topics": _clean_str_list(payload.get("legal_topics", [])),
        "evidence_checklist": _clean_str_list(payload.get("evidence_checklist", [])),
        "candidate_concepts": [],
        "final_note": str(payload.get("final_note", "")).strip(),
        "answer_forbidden": bool(payload.get("answer_forbidden", True)),
    }

    raw_concepts = payload.get("candidate_concepts", [])
    if not isinstance(raw_concepts, list):
        raw_concepts = payload.get("option_hints", [])

    if isinstance(raw_concepts, list):
        concepts: list[dict[str, Any]] = []
        for item in raw_concepts:
            if not isinstance(item, dict):
                continue

            concept_text = str(item.get("concept", item.get("option_text", ""))).strip()
            concept_text = strip_choice_prefix(concept_text)

            concept = {
                "concept": concept_text,
                "legal_meaning": str(item.get("legal_meaning", "")).strip(),
                "visual_cues": _clean_str_list(item.get("visual_cues", [])),
                "exclusion_cues": _clean_str_list(item.get("exclusion_cues", [])),
            }
            if (
                concept["concept"]
                or concept["legal_meaning"]
                or concept["visual_cues"]
                or concept["exclusion_cues"]
            ):
                concepts.append(concept)

        normalized["candidate_concepts"] = concepts

    return normalized


def _task_guidance(task_type: str) -> str:
    mapping = {
        "navigation": (
            "Tập trung vào mũi tên chỉ hướng, tên đường, địa danh, hướng rẽ, làn đường, "
            "điểm đến và quan hệ giữa các hướng đi."
        ),
        "sign_identification": (
            "Tập trung vào hình dạng biển báo, màu sắc, biểu tượng, chữ trên biển, "
            "nhóm biển cấm/nguy hiểm/chỉ dẫn/hiệu lệnh."
        ),
        "verification": (
            "Tập trung vào mệnh đề cần kiểm chứng, điều kiện đúng/sai, dấu hiệu phủ định, "
            "và các bằng chứng trực quan xác nhận hoặc bác bỏ mệnh đề."
        ),
        "object_presence": (
            "Tập trung vào dấu hiệu có/không có đối tượng, vị trí xuất hiện, khả năng bị che khuất, "
            "và điều kiện quan sát trong khung hình."
        ),
        "rule_compliance": (
            "Tập trung vào hành vi giao thông, tín hiệu đèn, biển báo, làn đường, vạch kẻ đường, "
            "quy tắc ưu tiên, và việc tuân thủ hay vi phạm."
        ),
    }
    return mapping.get(
        task_type,
        "Tập trung vào các dấu hiệu trực quan có thể giúp đối chiếu nội dung câu hỏi với video.",
    )



def _few_shot_examples(task_type: str) -> str:
    examples = {
        "navigation": """
Ví dụ tốt:
{
  "task_type": "navigation",
  "intent": "direction",
  "legal_topics": ["biển chỉ dẫn", "hướng đi", "địa danh"],
  "candidate_concepts": [
    {
      "concept": "rẽ phải vào đường có tên trên biển",
      "legal_meaning": "Đối chiếu lựa chọn với mũi tên chỉ hướng và tên đường/địa danh xuất hiện trên biển chỉ dẫn.",
      "visual_cues": ["mũi tên rẽ phải", "tên đường", "địa danh"],
      "exclusion_cues": ["không có mũi tên tương ứng", "tên đường không xuất hiện"]
    },
    {
      "concept": "đi thẳng theo hướng biển chỉ dẫn",
      "legal_meaning": "Đối chiếu lựa chọn với hướng mũi tên đi thẳng và thông tin điểm đến trên biển.",
      "visual_cues": ["mũi tên đi thẳng", "biển chỉ dẫn nền xanh"],
      "exclusion_cues": ["chỉ có mũi tên rẽ", "không có tên điểm đến phù hợp"]
    }
  ],
  "evidence_checklist": [
    "xác định mũi tên chỉ hướng",
    "đọc tên đường hoặc địa danh",
    "đối chiếu hướng rẽ với lựa chọn"
  ],
  "final_note": "Không được suy ra đáp án cuối cùng.",
  "answer_forbidden": true
}
""",
        "sign_identification": """
Ví dụ tốt:
{
  "task_type": "sign_identification",
  "intent": "identification",
  "legal_topics": ["nhóm biển báo", "hình dạng", "màu sắc", "biểu tượng"],
  "candidate_concepts": [
    {
      "concept": "biển cấm xe máy",
      "legal_meaning": "Đối chiếu lựa chọn với biển tròn viền đỏ, nền trắng và biểu tượng xe máy bị cấm.",
      "visual_cues": ["biển tròn", "viền đỏ", "biểu tượng xe máy"],
      "exclusion_cues": ["biển nền xanh", "biểu tượng xe đạp", "biển tam giác"]
    },
    {
      "concept": "biển cấm xe đạp",
      "legal_meaning": "Đối chiếu lựa chọn với biển tròn viền đỏ và biểu tượng xe đạp.",
      "visual_cues": ["biểu tượng xe đạp", "biển tròn viền đỏ"],
      "exclusion_cues": ["biểu tượng xe máy", "biểu tượng xe tải"]
    }
  ],
  "evidence_checklist": [
    "quan sát hình dạng biển",
    "quan sát màu nền và viền",
    "đọc biểu tượng trên biển"
  ],
  "final_note": "Không được suy ra đáp án cuối cùng.",
  "answer_forbidden": true
}
""",
        "verification": """
Ví dụ tốt:
{
  "task_type": "verification",
  "intent": "direction",
  "legal_topics": ["kiểm chứng mệnh đề", "điều kiện áp dụng"],
  "candidate_concepts": [
    {
      "concept": "mệnh đề đúng khi dấu hiệu trong video trùng khớp",
      "legal_meaning": "Kiểm tra mệnh đề bằng cách đối chiếu trực tiếp hướng đi, làn đường hoặc biển báo trong video.",
      "visual_cues": ["hướng di chuyển", "vị trí xe", "biển báo liên quan"],
      "exclusion_cues": ["không thấy dấu hiệu xác nhận", "dấu hiệu mâu thuẫn với mệnh đề"]
    },
    {
      "concept": "mệnh đề sai khi điều kiện không thỏa",
      "legal_meaning": "Nếu video không cho thấy điều kiện như mô tả trong mệnh đề thì không nên xác nhận mệnh đề.",
      "visual_cues": ["thiếu dấu hiệu xác nhận", "dấu hiệu phủ định"],
      "exclusion_cues": ["mọi điều kiện đều khớp"]
    }
  ],
  "evidence_checklist": [
    "tách mệnh đề cần kiểm chứng",
    "kiểm tra điều kiện áp dụng",
    "tìm dấu hiệu phủ định"
  ],
  "final_note": "Không được suy ra đáp án cuối cùng.",
  "answer_forbidden": true
}
""",
        "object_presence": """
Ví dụ tốt:
{
  "task_type": "object_presence",
  "intent": "existence",
  "legal_topics": ["dấu hiệu nhận diện đối tượng", "phạm vi quan sát"],
  "candidate_concepts": [
    {
      "concept": "đối tượng có xuất hiện trong khung hình",
      "legal_meaning": "Kiểm tra trực tiếp sự hiện diện của đối tượng được hỏi trong video.",
      "visual_cues": ["hình dạng đối tượng", "vị trí xuất hiện", "khung hình rõ"],
      "exclusion_cues": ["đối tượng không xuất hiện", "bị che khuất hoàn toàn"]
    },
    {
      "concept": "đối tượng không xuất hiện rõ ràng",
      "legal_meaning": "Nếu không quan sát thấy đối tượng hoặc bằng chứng không đủ rõ, không nên khẳng định có xuất hiện.",
      "visual_cues": ["khung hình trống", "không có dấu hiệu nhận diện"],
      "exclusion_cues": ["đối tượng nhìn thấy rõ"]
    }
  ],
  "evidence_checklist": [
    "xác định đối tượng cần tìm",
    "quan sát toàn khung hình",
    "kiểm tra khả năng bị che khuất"
  ],
  "final_note": "Không được suy ra đáp án cuối cùng.",
  "answer_forbidden": true
}
""",
        "rule_compliance": """
Ví dụ tốt:
{
  "task_type": "rule_compliance",
  "intent": "existence",
  "legal_topics": ["hành vi giao thông", "tín hiệu giao thông", "biển báo điều khiển"],
  "candidate_concepts": [
    {
      "concept": "hành vi tuân thủ tín hiệu và biển báo",
      "legal_meaning": "Đối chiếu hành vi của phương tiện với đèn tín hiệu, biển báo và làn đường điều khiển.",
      "visual_cues": ["đèn giao thông", "biển báo", "vị trí xe trong làn"],
      "exclusion_cues": ["vượt đèn", "đi sai làn", "bỏ qua biển báo"]
    },
    {
      "concept": "hành vi có dấu hiệu vi phạm",
      "legal_meaning": "Nếu hành vi đi ngược tín hiệu hoặc sai làn thì có dấu hiệu không tuân thủ.",
      "visual_cues": ["xe lấn làn", "vượt đèn đỏ", "không chấp hành biển báo"],
      "exclusion_cues": ["đi đúng làn", "dừng đúng tín hiệu"]
    }
  ],
  "evidence_checklist": [
    "xác định hành vi",
    "kiểm tra biển báo và đèn tín hiệu",
    "đối chiếu làn đường"
  ],
  "final_note": "Không được suy ra đáp án cuối cùng.",
  "answer_forbidden": true
}
"""
    }
    return examples.get(task_type, "")

def build_support_prompt(request: dict[str, Any] | None = None, **kwargs: Any) -> str:
    req: dict[str, Any] = {}
    if request is not None:
        req = {
            "task_type": _obj_get(request, "task_type", ""),
            "intent": _obj_get(request, "intent", ""),
            "question": _obj_get(request, "question", ""),
            "choices": _obj_get(request, "choices", []),
        }
    req.update(kwargs)

    task_type = resolve_task_type(request if request is not None else req)
    intent = str(req.get("intent", "")).strip()
    question = str(req.get("question", "")).strip()
    choices = req.get("choices", [])

    choice_lines: list[str] = []
    if isinstance(choices, list):
        for c in choices:
            text = _choice_to_text(c)
            if text:
                choice_lines.append(f"- {text}")
    else:
        text = _choice_to_text(choices)
        if text:
            choice_lines.append(f"- {text}")

    choices_text = "\n".join(choice_lines)
    task_guidance = _task_guidance(task_type)
    few_shot = _few_shot_examples(task_type)

    return f"""Bạn là agent hỗ trợ pháp luật giao thông Việt Nam cho bài toán VQA.

QUY TẮC BẮT BUỘC:
- Chỉ trả lời bằng TIẾNG VIỆT.
- Tuyệt đối không dùng tiếng Anh, tiếng Trung, hoặc ngôn ngữ khác.
- Không được chọn đáp án cuối cùng.
- Không được viết: "đáp án là", "chọn A/B/C/D", "final answer".
- Không được dùng A/B/C/D làm nhãn trong output.
- Chỉ tạo tri thức hỗ trợ để model khác suy luận.

GIỮ NGUYÊN:
- task_type phải giữ đúng giá trị đầu vào: "{task_type}"
- intent phải giữ đúng giá trị đầu vào: "{intent}"

HƯỚNG DẪN CHUYÊN BIỆT:
{task_guidance}

HÃY TRẢ VỀ JSON HỢP LỆ THEO ĐÚNG SCHEMA NÀY:
{{
  "task_type": "{task_type}",
  "intent": "{intent}",
  "legal_topics": ["..."],
  "candidate_concepts": [
    {{
      "concept": "...",
      "legal_meaning": "...",
      "visual_cues": ["..."],
      "exclusion_cues": ["..."]
    }}
  ],
  "evidence_checklist": ["..."],
  "final_note": "Không được suy ra đáp án cuối cùng.",
  "answer_forbidden": true
}}

RÀNG BUỘC CHẤT LƯỢNG:
- legal_topics: ít nhất 1 mục.
- candidate_concepts: từ 2 đến 4 mục.
- evidence_checklist: ít nhất 2 mục.
- candidate_concepts phải viết theo NGHĨA của lựa chọn, không nhắc A/B/C/D.
- Chỉ trả JSON, không thêm văn bản ngoài JSON.

VÍ DỤ THAM KHẢO:
{few_shot}

question:
{question}

choices:
{choices_text}
"""


def brief_from_text(text: str) -> dict[str, Any]:
    parsed = extract_json_block(text)
    return normalize_support_payload(parsed)


def resolve_task_type(request: Any) -> str:
    for key in ["task_type", "resolved_dataset_type", "dataset_type", "type"]:
        value = _obj_get(request, key, "")
        value = str(value).strip()
        if value:
            return value
    return ""