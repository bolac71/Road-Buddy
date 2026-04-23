from __future__ import annotations
import json
from pathlib import Path
from road_buddy_agentic.schemas.sample import DatasetSample


def load_first_n_samples(dataset_json: str, take_first_n: int = 100, require_type_field: bool = True) -> list[DatasetSample]:
    p = Path(dataset_json)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    data = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    out: list[DatasetSample] = []
    for item in data[:take_first_n]:
        dataset_type = str(item.get("type", "other")).strip() or "other"
        if require_type_field and "type" not in item:
            raise ValueError(f"Sample {item.get('id')} khong co field 'type'")
        out.append(
            DatasetSample(
                id=str(item["id"]),
                question=str(item["question"]),
                choices=[str(x) for x in item["choices"]],
                video_path=str(item["video_path"]),
                dataset_type=dataset_type,
            )
        )
    return out
