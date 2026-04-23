# Road Buddy Agentic Support v2

Bản này nâng `infer-agentic` từ scaffold sang **Qwen-only chạy thật**:

- đọc **100 sample đầu tiên** từ `public_test_with_type.json`
- resolve video path từ `video_root`
- cắt frame bằng module hiện có của `road_buddy`
- chọn frame query-aware bằng module hiện có của `road_buddy`
- gọi **QwenVLRunner** hiện có từ `road_buddy.model.qwen_vl`
- sinh các file output thật:
  - `submission.csv`
  - `audit.json`
  - `support_outputs.jsonl`
  - `run_summary.json`

## Trạng thái

- `support_provider=none` → chạy thật hoàn chỉnh
- `support_provider=gemini/groq` → chưa phải implementation cuối, hiện vẫn dùng support stub

## Cách chạy baseline Qwen-only

```bash
cd /path/to/road-buddy
python -m road_buddy_agentic.cli.main infer-agentic \
  --config /path/to/road-buddy/road_buddy_agentic/configs/agentic_qwen_base.yaml \
  --dataset-json /path/to/road-buddy/public_test_with_type.json \
  --video-root /path/to/road-buddy/datasets \
  --answer-json /path/to/road-buddy/public_test_answers.json \
  --max-samples 100 \
  --support-provider none \
  --support-model none \
  --output-root /path/to/road-buddy/outputs/qwen_only_first100
```


## System test for support models only (VSCode / terminal)

Run Gemini support only:

```bash
cd /datastore/cndt_khanhnd/road-buddy
PYTHONPATH=/datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2 python /datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2/scripts/test_support_agents.py   --config /datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2/configs/agentic_qwen_base.yaml   --provider gemini   --model gemini-2.5-flash   --api-env-path /datastore/cndt_khanhnd/road-buddy/.env   --api-key-env-name API_KEYS   --max-samples 5   --output-root /datastore/cndt_khanhnd/road-buddy/outputs/support_test_gemini
```

Run Llama support only via Groq:

```bash
cd /datastore/cndt_khanhnd/road-buddy
PYTHONPATH=/datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2 python /datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2/scripts/test_support_agents.py   --config /datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v2/configs/agentic_qwen_base.yaml   --provider groq   --model meta-llama/llama-4-scout-17b-16e-instruct   --api-env-path /datastore/cndt_khanhnd/road-buddy/.env   --api-key-env-name GROQ_API_KEY   --max-samples 5   --output-root /datastore/cndt_khanhnd/road-buddy/outputs/support_test_groq
```
