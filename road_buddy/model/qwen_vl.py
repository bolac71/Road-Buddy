from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from road_buddy.config import ModelConfig
from road_buddy.prompting import build_prompt, extract_final_letter

class _SuppressNoiseFilter(logging.Filter):
    _SUPPRESSED = (
        "Kwargs passed to `processor.__call__`",
        "The following generation flags are not valid",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SUPPRESSED)


_noise_filter = _SuppressNoiseFilter()
logging.getLogger("transformers").addFilter(_noise_filter)


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class PredictResult:
    answer: str
    probs: dict[str, float]
    raw_text: str = ""
    source: str = ""


class QwenVLRunner:
    def __init__(self, model_cfg: ModelConfig):
        self.model_cfg = model_cfg
        self.model = None
        self.processor = None

    def load(self) -> None:
        dtype = _DTYPE_MAP.get(self.model_cfg.torch_dtype.lower(), torch.bfloat16)

        model_kwargs = {
            "dtype": dtype,
            "device_map": self.model_cfg.device_map,
            "trust_remote_code": self.model_cfg.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.model_cfg.attn_implementation:
            model_kwargs["attn_implementation"] = self.model_cfg.attn_implementation

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_cfg.model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_cfg.model_name_or_path,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )

        if getattr(self.processor, "image_processor", None) is None:
            raise RuntimeError(
                "Model/processor hien tai khong ho tro dau vao anh-video. "
            )

    @property
    def device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Model chua duoc load, khong the truy cap device")
        return next(self.model.parameters()).device

    def _prepare_inputs(self, prompt: str, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        template_kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.model_cfg.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.model_cfg.enable_thinking

        text = self.processor.apply_chat_template(
            messages,
            **template_kwargs,
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
        )

        prepared: dict[str, torch.Tensor] = {}
        model_dtype = next(self.model.parameters()).dtype
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                prepared[k] = v
                continue
            if v.dtype.is_floating_point:
                prepared[k] = v.to(self.device, dtype=model_dtype)
            else:
                prepared[k] = v.to(self.device)
        return prepared

    def _predict_generate(
        self,
        prompt: str,
        images: list[Image.Image],
        choice_letters: list[str],
        max_new_tokens: int = 512,
    ) -> PredictResult:
        inputs = self._prepare_inputs(prompt, images)
        pad_token_id = self.processor.tokenizer.eos_token_id
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=pad_token_id,
            )

        input_len = inputs["input_ids"].shape[-1]
        gen_ids = output_ids[:, input_len:]
        decoded = self.processor.batch_decode(gen_ids, skip_special_tokens=True)
        raw_text = decoded[0].strip() if decoded else ""

        answer = extract_final_letter(raw_text, choice_letters)
        source = "generate_regex"
        if answer is None:
            answer = choice_letters[0]
            source = "generate_default"
        return PredictResult(answer=answer, probs={}, raw_text=raw_text, source=source)

    def predict(
        self,
        question: str,
        choices: list[str],
        images: list[Image.Image],
        system_hint: str,
        target_objects: list[str] | None = None,
        temporal_hints: list[str] | None = None,
    ) -> PredictResult:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model chua duoc load nen khong the predict")

        prompt = build_prompt(
            question=question,
            choices=choices,
            system_hint=system_hint,
            target_objects=target_objects,
            temporal_hints=temporal_hints,
        )
        choice_letters = [c[0] for c in choices if c]
        return self._predict_generate(prompt, images, choice_letters)
