from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    dataset_json: str
    video_root: str
    answer_json: Optional[str] = None
    take_first_n: int = 100
    require_type_field: bool = True


@dataclass
class QwenReasonerConfig:
    enabled: bool = True
    provider: str = "local_qwen"
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    enable_thinking: Optional[bool] = None
    num_frames: int = 10
    max_side: int = 960
    query_aware_enabled: bool = True
    candidate_frame_multiplier: int = 3
    system_hint: str = (
        "Bạn là trợ lý AI cho bài toán giao thông Việt Nam. "
        "Chỉ chọn một đáp án đúng nhất theo video và câu hỏi."
    )


@dataclass
class SupportAgentConfig:
    enabled: bool = True
    provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    api_env_path: Optional[str] = None
    api_key_env_name: Optional[str] = None
    mode: str = "legal_brief_only"
    log_mode: str = "full"
    max_retries_per_sample: int = 5
    key_cooldown_sec: int = 60
    transient_cooldown_sec: int = 20
    invalid_key_cooldown_sec: int = 1800
    jpeg_quality: int = 85
    max_output_tokens: int = 512
    temperature: float = 0.0


@dataclass
class RuntimeConfig:
    seed: int = 42
    max_samples: int = 100
    checkpoint_every_n: int = 10
    print_every_n: int = 1
    time_limit_sec: float = 30.0
    output_root: str = "outputs/agentic"


@dataclass
class OutputConfig:
    save_support_prompt: bool = True
    save_support_raw_output: bool = True
    save_support_brief: bool = True
    save_support_jsonl: bool = True
    save_run_summary: bool = True
    save_accuracy_by_type: bool = True
