from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._common import build_support_prompt, brief_from_text, strip_choice_prefix


@dataclass
class SupportResult:
    status: str
    raw_text: str
    normalized_payload: dict[str, Any]

    @property
    def legal_topics(self) -> list[str]:
        return self.normalized_payload.get("legal_topics", [])

    @property
    def evidence_checklist(self) -> list[str]:
        return self.normalized_payload.get("evidence_checklist", [])

    @property
    def option_hints(self) -> list[dict[str, Any]]:
        return self.normalized_payload.get(
            "candidate_concepts",
            self.normalized_payload.get("option_hints", []),
        )


def _req_get(request: Any, key: str, default: Any = None) -> Any:
    if isinstance(request, dict):
        return request.get(key, default)
    return getattr(request, key, default)


def _fallback_topics(task_type: str) -> list[str]:
    mapping = {
        "navigation": ["biển chỉ dẫn", "hướng đi", "làn đường", "điểm đến"],
        "verification": ["kiểm chứng mệnh đề", "điều kiện áp dụng", "dấu hiệu loại trừ"],
        "object_presence": ["dấu hiệu nhận diện đối tượng", "khả năng bị che khuất", "phạm vi quan sát"],
        "sign_identification": ["nhóm biển báo", "hình dạng biển", "màu sắc", "biểu tượng"],
        "rule_compliance": ["hành vi giao thông", "tín hiệu giao thông", "biển báo điều khiển", "làn đường"],
    }
    return mapping.get(task_type, ["quy tắc giao thông đường bộ"])


def _fallback_checklist(task_type: str) -> list[str]:
    mapping = {
        "navigation": ["xác định mũi tên chỉ hướng", "đọc tên đường hoặc địa danh", "đối chiếu hướng rẽ"],
        "verification": ["tách mệnh đề cần kiểm chứng", "kiểm tra điều kiện áp dụng", "tìm dấu hiệu phủ định"],
        "object_presence": ["xác định đối tượng cần tìm", "quan sát toàn khung hình", "kiểm tra khả năng bị che khuất"],
        "sign_identification": ["quan sát hình dạng biển", "quan sát màu nền và viền", "đọc biểu tượng hoặc chữ trên biển"],
        "rule_compliance": ["xác định hành vi", "kiểm tra biển báo và đèn tín hiệu", "đối chiếu làn đường"],
    }
    return mapping.get(task_type, ["xác định đối tượng pháp lý", "đối chiếu dấu hiệu quan sát"])


def _fallback_legal_meaning(task_type: str) -> str:
    mapping = {
        "navigation": "Đối chiếu lựa chọn này với mũi tên chỉ hướng, tên đường, địa danh và hướng rẽ trong video.",
        "verification": "Kiểm tra xem lựa chọn này có phù hợp với mệnh đề cần xác minh trong video hay không.",
        "object_presence": "Kiểm tra xem đối tượng tương ứng với lựa chọn này có thực sự xuất hiện trong video hay không.",
        "sign_identification": "Đối chiếu lựa chọn này với hình dạng, màu sắc, biểu tượng và nhóm biển báo xuất hiện trong video.",
        "rule_compliance": "Đối chiếu lựa chọn này với hành vi giao thông, làn đường, tín hiệu và biển báo điều khiển.",
    }
    return mapping.get(task_type, "Đối chiếu nội dung lựa chọn với dấu hiệu trực quan trong video.")


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


def _fallback_concepts_from_choices(request: Any) -> list[dict[str, Any]]:
    choices = _req_get(request, "choices", [])
    task_type = _resolve_task_type(request)

    concepts: list[dict[str, Any]] = []
    if isinstance(choices, list):
        for c in choices[:4]:
            text = _choice_to_text(c)
            if not text:
                continue

            concepts.append(
                {
                    "concept": text,
                    "legal_meaning": _fallback_legal_meaning(task_type),
                    "visual_cues": [],
                    "exclusion_cues": [],
                }
            )

    return concepts


def _merge_with_fallback(request: Any, normalized: dict[str, Any]) -> dict[str, Any]:
    task_type = _resolve_task_type(request)
    intent = str(_req_get(request, "intent", "")).strip()

    if not normalized.get("task_type"):
        normalized["task_type"] = task_type
    if not normalized.get("intent"):
        normalized["intent"] = intent

    if not normalized.get("legal_topics"):
        normalized["legal_topics"] = _fallback_topics(task_type)

    if not normalized.get("evidence_checklist"):
        normalized["evidence_checklist"] = _fallback_checklist(task_type)

    if not normalized.get("candidate_concepts"):
        normalized["candidate_concepts"] = _fallback_concepts_from_choices(request)

    if not normalized.get("final_note"):
        normalized["final_note"] = "Không được suy ra đáp án cuối cùng."

    normalized["answer_forbidden"] = True
    return normalized


def _resolve_task_type(request: Any) -> str:
    for key in ["task_type", "resolved_dataset_type", "dataset_type", "type"]:
        value = _req_get(request, key, "")
        value = str(value).strip()
        if value:
            return value
    return ""

class GeminiSupportAgent:
    def __init__(
        self,
        model_name: str,
        api_env_path: str,
        api_key_env_name: str = "API_KEYS",
        max_output_tokens: int = 800,
        temperature: float = 0.0,
        **_: Any,
    ):
        try:
            from google import genai
            from road_buddy.model.gemini_key_pool import GeminiKeyPool
        except Exception as e:
            raise RuntimeError(
                "Khong import duoc google.genai hoac road_buddy.model.gemini_key_pool"
            ) from e

        self.genai = genai
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.key_pool = GeminiKeyPool(api_env_path)
        self._clients: dict[int, Any] = {}

    def _get_client(self, key_index: int, api_key: str):
        if key_index not in self._clients:
            self._clients[key_index] = self.genai.Client(api_key=api_key)
        return self._clients[key_index]

    def generate_support_brief(self, request: Any) -> SupportResult:
        last_error: Exception | None = None

        for _ in range(max(1, self.key_pool.size)):
            ks = self.key_pool.acquire()
            try:
                client = self._get_client(ks.index, ks.key)
                prompt = build_support_prompt(request)

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_output_tokens,
                        "response_mime_type": "application/json",
                    },
                )

                try:
                    raw_text = response.text or ""
                except Exception:
                    raw_text = str(response)

                normalized = brief_from_text(raw_text)
                normalized = _merge_with_fallback(request, normalized)

                self.key_pool.report_success(ks.index)
                return SupportResult(
                    status="ok",
                    raw_text=raw_text,
                    normalized_payload=normalized,
                )

            except Exception as e:
                last_error = e
                self.key_pool.report_failure(
                    key_index=ks.index,
                    reason=str(e),
                    cooldown_sec=20,
                    disable_forever=False,
                )

        fallback_payload = _merge_with_fallback(request, {})
        return SupportResult(
            status="error",
            raw_text=str(last_error) if last_error else "",
            normalized_payload=fallback_payload,
        )