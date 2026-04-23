# Patch hỗ trợ Gemini/Groq support agents

Các thay đổi chính:
- Bổ sung `road_buddy_agentic/support_agents/providers/`
- Support agent Gemini dùng `road_buddy.model.gemini_key_pool.GeminiKeyPool`
- Support agent Groq/Llama dùng `road_buddy.model.groq_key_pool.GroqKeyPool`
- CLI hỗ trợ override:
  - `--support-api-env-path`
  - `--support-api-key-env-name`

Ví dụ Gemini:
```bash
PYTHONPATH=/datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v3 python -m road_buddy_agentic.cli.main infer-agentic   --config /datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v3/configs/agentic_qwen_base.yaml   --dataset-json /datastore/cndt_khanhnd/road-buddy/public_test_with_type.json   --video-root /datastore/cndt_khanhnd/road-buddy   --answer-json /datastore/cndt_khanhnd/road-buddy/public_test_answers.json   --max-samples 100   --support-provider gemini   --support-model gemini-2.5-flash   --support-api-env-path /datastore/cndt_khanhnd/road-buddy/.env   --support-api-key-env-name API_KEYS   --output-root /datastore/cndt_khanhnd/road-buddy/outputs/gemini_support_first100
```

Ví dụ Groq/Llama:
```bash
PYTHONPATH=/datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v3 python -m road_buddy_agentic.cli.main infer-agentic   --config /datastore/cndt_khanhnd/road-buddy/road_buddy_agentic_v3/configs/agentic_qwen_base.yaml   --dataset-json /datastore/cndt_khanhnd/road-buddy/public_test_with_type.json   --video-root /datastore/cndt_khanhnd/road-buddy   --answer-json /datastore/cndt_khanhnd/road-buddy/public_test_answers.json   --max-samples 100   --support-provider groq   --support-model meta-llama/llama-4-scout-17b-16e-instruct   --support-api-env-path /datastore/cndt_khanhnd/road-buddy/.env   --support-api-key-env-name GROQ_API_KEY   --output-root /datastore/cndt_khanhnd/road-buddy/outputs/llama_support_first100
```
