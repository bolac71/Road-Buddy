from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from road_buddy.config import ModelConfig
from road_buddy.prompting import build_prompt, extract_final_letter


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
                "Model/processor hiện tại không hỗ trợ đầu vào ảnh-video. "
            )

    @property
    def device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        return next(self.model.parameters()).device

    def _encode_choice_tokens(self, choice_letters: list[str]) -> dict[str, int]:
        token_map: dict[str, int] = {}
        for letter in choice_letters:
            ids = self.processor.tokenizer(letter, add_special_tokens=False).input_ids
            if ids and len(ids) == 1:
                token_map[letter] = ids[0]
                continue

            ids = self.processor.tokenizer(f" {letter}", add_special_tokens=False).input_ids
            if ids and len(ids) == 1:
                token_map[letter] = ids[0]

        if not token_map:
            raise ValueError("Cannot build token map for choices")
        return token_map

    def _prepare_inputs(self, prompt: str, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            truncation=False,
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

    def _predict_logits_only(
        self,
        prompt: str,
        images: list[Image.Image],
        choice_letters: list[str],
    ) -> PredictResult:
        if not choice_letters:
            raise ValueError("choice_letters is empty")

        token_map = self._encode_choice_tokens(choice_letters)
        token_ids = torch.tensor([token_map[c] for c in token_map], device=self.device)
        id_to_letter = {v: k for k, v in token_map.items()}

        inputs = self._prepare_inputs(prompt, images)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )

        if not outputs.scores:
            raise RuntimeError("Model did not return generation scores")

        next_token_logits = outputs.scores[0][0]
        option_logits = torch.index_select(next_token_logits, 0, token_ids)
        option_probs = F.softmax(option_logits, dim=-1)

        best_idx = int(torch.argmax(option_probs).item())
        best_token = int(token_ids[best_idx].item())
        answer = id_to_letter[best_token]

        probs: dict[str, float] = {}
        for i, token_id in enumerate(token_ids.tolist()):
            probs[id_to_letter[int(token_id)]] = float(option_probs[i].item())

        return PredictResult(answer=answer, probs=probs, raw_text="", source="logits_only")

    def _predict_generate(
        self,
        prompt: str,
        images: list[Image.Image],
        choice_letters: list[str],
        max_new_tokens: int = 128,
    ) -> PredictResult:
        inputs = self._prepare_inputs(prompt, images)
        pad_token_id = self.processor.tokenizer.eos_token_id
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
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
        use_logits_only: bool,
    ) -> PredictResult:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model must be loaded before predict")

        prompt = build_prompt(question=question, choices=choices, system_hint=system_hint)
        choice_letters = [c[0] for c in choices if c]

        if use_logits_only:
            try:
                return self._predict_logits_only(prompt, images, choice_letters)
            except Exception as e:
                result = self._predict_generate(prompt, images, choice_letters)
                if result.source:
                    result.source = f"{result.source}_after_logits_fallback"
                else:
                    result.source = "generate_after_logits_fallback"
                if result.raw_text:
                    result.raw_text = f"[logits_error:{type(e).__name__}] {result.raw_text}"
                return result

        return self._predict_generate(prompt, images, choice_letters)
