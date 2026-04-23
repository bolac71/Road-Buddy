from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ._common import build_support_prompt, brief_from_text


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


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
        # giữ tương thích với script test hiện tại
        return self.normalized_payload.get("candidate_concepts", self.normalized_payload.get("option_hints", []))


class LlamaSupportAgent:
    def __init__(
        self,
        model_name: str,
        api_env_path: str,
        api_key_env_name: str = "GROQ_API_KEY",
        max_output_tokens: int = 800,
        temperature: float = 0.0,
        **_: Any,
    ):
        try:
            from road_buddy.model.groq_key_pool import GroqKeyPool
        except Exception as e:
            raise RuntimeError("Khong import duoc road_buddy.model.groq_key_pool") from e

        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.key_pool = GroqKeyPool(api_env_path, api_key_env_name)

    def generate_support_brief(self, request: dict[str, Any]) -> SupportResult:
        last_error: Exception | None = None

        for _ in range(max(1, self.key_pool.size)):
            ks = self.key_pool.acquire()
            try:
                client = OpenAI(api_key=ks.key, base_url=GROQ_BASE_URL)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": build_support_prompt(request)},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                )
                raw_text = response.choices[0].message.content or ""
                normalized = brief_from_text(raw_text)

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

        return SupportResult(
            status="error",
            raw_text=str(last_error) if last_error else "",
            normalized_payload={},
        )
