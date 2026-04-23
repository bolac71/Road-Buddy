from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _CompatModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    enable_thinking: bool | None = None
    # Generation params — must match road_buddy.config.ModelConfig
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int = 512
    thinking_budget_bias: float | None = None


class QwenReasonerAdapter:
    """Adapter around the user's existing road_buddy.model.qwen_vl.QwenVLRunner.

    This keeps the new agentic package thin and reuses the working Qwen baseline
    implementation from the original repo.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.runner = None

    def load(self) -> None:
        try:
            from road_buddy.model.qwen_vl import QwenVLRunner
        except Exception as e:  # pragma: no cover - import path depends on user repo
            raise RuntimeError(
                "Khong import duoc road_buddy.model.qwen_vl.QwenVLRunner. "
                "Hay dam bao repo goc van co Qwen runner va dang chay tu repo root."
            ) from e

        model_cfg = _CompatModelConfig(
            model_name_or_path=str(self.cfg.get("model_name", "Qwen/Qwen3-VL")),
            trust_remote_code=bool(self.cfg.get("trust_remote_code", True)),
            device_map=str(self.cfg.get("device_map", "auto")),
            torch_dtype=str(self.cfg.get("torch_dtype", "bfloat16")),
            attn_implementation=self.cfg.get("attn_implementation"),
            enable_thinking=self.cfg.get("enable_thinking"),
            do_sample=bool(self.cfg.get("do_sample", False)),
            temperature=float(self.cfg["temperature"]) if self.cfg.get("temperature") is not None else None,
            top_p=float(self.cfg["top_p"]) if self.cfg.get("top_p") is not None else None,
            top_k=int(self.cfg["top_k"]) if self.cfg.get("top_k") is not None else None,
            max_new_tokens=int(self.cfg.get("max_new_tokens", 512)),
            thinking_budget_bias=float(self.cfg["thinking_budget_bias"]) if self.cfg.get("thinking_budget_bias") is not None else None,
        )
        self.runner = QwenVLRunner(model_cfg)
        self.runner.load()

    def predict(
        self,
        *,
        question: str,
        choices: list[str],
        images: list,
        system_hint: str,
        target_objects: list[str] | None = None,
        temporal_hints: list[str] | None = None,
    ):
        if self.runner is None:
            raise RuntimeError("QwenReasonerAdapter chua load")
        return self.runner.predict(
            question=question,
            choices=choices,
            images=images,
            system_hint=system_hint,
            target_objects=target_objects,
            temporal_hints=temporal_hints,
        )
