from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class PathsConfig:
    dataset_json: str
    video_root: str
    output_csv: str
    answer_json: Optional[str] = None
    audit_json: Optional[str] = None


@dataclass
class ModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = None
    enable_thinking: Optional[bool] = None
    # Generation params — tuning theo từng model
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 512
    thinking_budget_bias: Optional[float] = None  # logit bias cho </think> token (11.8/12.5/13.3)


@dataclass
class SamplingConfig:
    num_frames: int = 10
    max_side: int = 960
    sample_fps: float = 2.0


@dataclass
class RuntimeConfig:
    seed: int = 42
    default_answer: str = "A"
    time_limit_sec: float = 30.0
    max_samples: Optional[int] = 50
    checkpoint_every_n: int = 10
    print_every_n: int = 1
    batch_size: int = 1
    clear_cuda_cache_on_each_sample: bool = True


@dataclass
class PromptConfig:
    system_hint: str = (
        "Bạn là trợ lý AI cho bài toán giao thông Việt Nam. "
        "Chỉ chọn một đáp án đúng nhất theo video và câu hỏi."
    )


@dataclass
class AppConfig:
    paths: PathsConfig
    model: ModelConfig
    sampling: SamplingConfig
    runtime: RuntimeConfig
    prompt: PromptConfig


_REQUIRED_TOP_LEVEL_KEYS = {"paths", "model", "sampling", "runtime", "prompt"}


def _resolve_path(path_value: Optional[str], config_dir: Path) -> Optional[str]:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((config_dir / path).resolve())


def load_config(config_path: str) -> AppConfig:
    cfg_path = Path(config_path).resolve()
    config_dir = cfg_path.parent

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config YAML must be a mapping.")

    missing_keys = sorted(_REQUIRED_TOP_LEVEL_KEYS - set(raw.keys()))
    if missing_keys:
        raise ValueError(f"Missing required config sections: {missing_keys}")

    paths = raw["paths"]
    model = raw["model"]
    sampling = raw["sampling"]
    runtime = raw["runtime"]
    prompt = raw["prompt"]

    app_cfg = AppConfig(
        paths=PathsConfig(
            dataset_json=_resolve_path(paths["dataset_json"], config_dir),
            video_root=_resolve_path(paths["video_root"], config_dir),
            output_csv=_resolve_path(paths["output_csv"], config_dir),
            answer_json=_resolve_path(paths.get("answer_json"), config_dir),
            audit_json=_resolve_path(paths.get("audit_json"), config_dir),
        ),
        model=ModelConfig(
            model_name_or_path=model["model_name_or_path"],
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            device_map=str(model.get("device_map", "auto")),
            torch_dtype=str(model.get("torch_dtype", "bfloat16")),
            attn_implementation=model.get("attn_implementation"),
            enable_thinking=model.get("enable_thinking"),
            do_sample=bool(model.get("do_sample", False)),
            temperature=float(model["temperature"]) if model.get("temperature") is not None else None,
            top_p=float(model["top_p"]) if model.get("top_p") is not None else None,
            top_k=int(model["top_k"]) if model.get("top_k") is not None else None,
            max_new_tokens=int(model.get("max_new_tokens", 512)),
            thinking_budget_bias=float(model["thinking_budget_bias"]) if model.get("thinking_budget_bias") is not None else None,
        ),
        sampling=SamplingConfig(
            num_frames=int(sampling.get("num_frames", 10)),
            max_side=int(sampling.get("max_side", 960)),
            sample_fps=float(sampling.get("sample_fps", 2.0)),
        ),
        runtime=RuntimeConfig(
            seed=int(runtime.get("seed", 42)),
            default_answer=str(runtime.get("default_answer", "A")),
            time_limit_sec=float(runtime.get("time_limit_sec", 30.0)),
            max_samples=int(runtime["max_samples"]) if runtime.get("max_samples") is not None else None,
            checkpoint_every_n=int(runtime.get("checkpoint_every_n", 10)),
            print_every_n=int(runtime.get("print_every_n", 1)),
            batch_size=int(runtime.get("batch_size", 1)),
            clear_cuda_cache_on_each_sample=bool(runtime.get("clear_cuda_cache_on_each_sample", True)),
        ),
        prompt=PromptConfig(
            system_hint=str(prompt.get("system_hint", PromptConfig.system_hint)),
        ),
    )

    output_dir = Path(app_cfg.paths.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if app_cfg.paths.audit_json:
        audit_dir = Path(app_cfg.paths.audit_json).parent
        audit_dir.mkdir(parents=True, exist_ok=True)

    return app_cfg


def as_dict(config: AppConfig) -> dict[str, Any]:
    return {
        "paths": config.paths.__dict__,
        "model": config.model.__dict__,
        "sampling": config.sampling.__dict__,
        "runtime": config.runtime.__dict__,
        "prompt": config.prompt.__dict__,
    }
