# Road-Buddy — Hướng dẫn Source Code

> Baseline cho bài toán **Zalo AI Challenge – Road Buddy**:
> Nhận video dashcam + câu hỏi tiếng Việt → trả lời trắc nghiệm A/B/C/D về giao thông.

---

## 1. Cấu trúc thư mục

```
road-buddy/
├── road_buddy/              # Package Python chính – chạy được ngay
│   ├── config.py            # Đọc và validate file YAML config
│   ├── dataio.py            # Đọc dataset, ghi submission CSV
│   ├── video.py             # Trích xuất frame từ video (OpenCV)
│   ├── prompting.py         # Xây dựng prompt + parse đáp án
│   ├── query_aware.py       # Chọn frame thông minh dựa theo câu hỏi
│   ├── pipeline.py          # Orchestrate toàn bộ quá trình inference
│   ├── cli.py               # Giao diện dòng lệnh (CLI)
│   └── model/
│       └── qwen_vl.py       # Wrapper cho model Qwen2.5-VL
├── configs/
│   └── baseline_qwen.yaml   # File config mặc định
├── datasets/
│   ├── train/
│   │   ├── videos/          # File .mp4 (train)
│   │   └── train.json       # Dataset train (id, question, choices, video_path)
│   └── public_test/
│       ├── videos/          # File .mp4 (test)
│       ├── public_test.json
│       └── public_test_with_answers.json   # Đáp án để eval local
├── outputs/                 # submission.csv + audit.json (kết quả)
├── logs/                    # Log từng SLURM job
└── scripts/
    └── run_baseline.slurm   # Script chạy trên HPC cluster
```

---

## 2. Luồng xử lý (End-to-End)

```
Video (.mp4) + Question (tiếng Việt)
           │
           ▼
  [1] Đọc dataset  ──────────── dataio.load_dataset()
           │
           ▼
  [2] Phân tích câu hỏi ─────── query_aware.analyze_question()
      → intent (TEMPORAL / VALUE / DIRECTION / ...)
      → target_objects (["traffic_sign", "speed_limit_sign", ...])
      → temporal_hints (["first", "last", "current"])
           │
           ▼
  [3] Trích xuất frames ─────── video.sample_video_frames()
      (lấy nhiều hơn cần, vd: 30 frames nếu num_frames=10 và multiplier=3)
           │
           ▼
  [4] Chọn top-K frames tốt nhất ── query_aware.select_query_aware_frames()
      (kết hợp: độ nét Laplacian + temporal prior theo câu hỏi)
           │
           ▼
  [5] Xây prompt tiếng Việt ─── prompting.build_prompt()
      (MCQ format + thông tin context từ bước 2)
           │
           ▼
  [6] Chạy VLM ──────────────── model.qwen_vl.QwenVLRunner.predict()
      → Ưu tiên: logits mode (nhanh, chính xác)
      → Fallback:  generate mode (sinh text → regex)
           │
           ▼
  [7] Parse đáp án A/B/C/D ─── prompting.extract_final_letter()
           │
           ▼
  [8] Lưu kết quả ──────────── dataio.save_submission()  → submission.csv
                                pipeline._save_audit_payload() → audit.json
```

---

## 3. Giải thích từng file

### `road_buddy/config.py` — Quản lý cấu hình

Đọc file YAML và ánh xạ vào các dataclass Python.

| Dataclass | Ý nghĩa |
|-----------|---------|
| `PathsConfig` | Đường dẫn: dataset_json, video_root, output_csv, answer_json, audit_json |
| `ModelConfig` | Tên model, torch_dtype (bfloat16), device_map (auto) |
| `SamplingConfig` | num_frames=10, max_side=960, query_aware_enabled, candidate_frame_multiplier=3 |
| `RuntimeConfig` | seed, time_limit_sec=30, checkpoint_every_n=5, batch_size=4 |
| `PromptConfig` | system_hint: đoạn mô tả vai trò của AI (tiếng Việt) |
| `AppConfig` | Container chứa tất cả config trên |

**Hàm quan trọng:**
- `load_config(path)` — đọc YAML, resolve relative paths thành absolute, tạo thư mục output nếu chưa có
- `_resolve_path(value, config_dir)` — chuyển path tương đối → tuyệt đối
- `as_dict(config)` — serialize config ra dict để in/debug

---

### `road_buddy/dataio.py` — Đọc/ghi dữ liệu

**Hàm quan trọng:**

```python
load_dataset(dataset_json)
```
Đọc `train.json` / `public_test.json`. Hỗ trợ 2 format: `{"data": [...]}` hoặc `[...]` trực tiếp.
Validate các field bắt buộc: `id`, `question`, `choices`, `video_path`.

```python
resolve_video_path(video_root, sample_video_path)
```
Tìm file video theo thứ tự ưu tiên:
1. Đường dẫn tuyệt đối → dùng luôn
2. `video_root / sample_video_path` → kiểm tra tồn tại
3. Thử alias: `dataset/` ↔ `train/` (backward compat)

```python
load_answer_map(answer_json)   # → dict {id: "A"/"B"/"C"/"D"}
save_submission(rows, output_csv)  # → ghi file CSV (id, answer)
load_submission(output_csv)        # → đọc file CSV về dict
extract_answer_letter(text)        # → tách chữ A/B/C/D từ chuỗi bất kỳ
```

**Format dữ liệu:**
```json
// dataset JSON
{"data": [{"id": "abc123", "video_path": "train/videos/xxx.mp4",
           "question": "Biển báo nào...", "choices": ["A. ...", "B. ...", "C. ...", "D. ..."]}]}

// submission CSV (nộp bài)
id,answer
abc123,C
```

---

### `road_buddy/video.py` — Trích xuất frame từ video

```python
extract_frames_1fps_from_video(video_path, max_frames=10)
```
Lấy 1 frame mỗi giây (uniform sampling) bằng OpenCV.
Trả về list `np.ndarray` (BGR format).

```python
sample_video_frames(video_path, num_frames, max_side)
```
Wrapper tiện lợi: gọi hàm trên + convert BGR→RGB + resize giữ tỉ lệ nếu cạnh dài > `max_side`.
Trả về `list[PIL.Image]`.

```python
select_topk_frames_multiframe(frames, num_boxes_list, top_k)
```
Chọn top-k frames dựa trên số bounding box được detect (frames có nhiều object → ưu tiên hơn).
Nếu không có detect nào → dùng uniform sampling thay thế.

---

### `road_buddy/query_aware.py` — Chọn frame thông minh

**Phân tích câu hỏi:**

```python
analyze_question(question) → QueryAnalysis
```
Dùng rule-based (keyword matching) để nhận dạng:
- **intent**: loại câu hỏi (xem bảng bên dưới)
- **target_objects**: đối tượng cần tìm trong video
- **temporal_hints**: gợi ý về vị trí thời gian trong video

| `QuestionIntent` | Từ khóa tiếng Việt |
|-----------------|-------------------|
| `TEMPORAL` | đầu tiên, cuối cùng, trước, sau, hiện tại, đang |
| `VALUE` | bao nhiêu, mấy, tốc độ |
| `DIRECTION` | hướng, rẽ, đi thẳng, quay đầu |
| `IDENTIFICATION` | biển gì, loại nào, là gì |
| `EXISTENCE` | có, không |

**Chọn frame:**

```python
select_query_aware_frames(frames, analysis, max_frames) → (selected, indices, scores)
```
Kết hợp 2 điểm số:
- `sharpness_score` — đo độ nét bằng phương sai Laplacian (frame mờ → điểm thấp)
- `temporal_prior` — trọng số vị trí thời gian:
  - "first" → ưu tiên frames đầu
  - "last" → ưu tiên frames cuối
  - không có gợi ý → ưu tiên frames giữa

**Trọng số blend:**
- Câu hỏi về thời gian: `score = 0.65 × temporal + 0.35 × sharpness`
- Câu hỏi khác: `score = 0.40 × temporal + 0.60 × sharpness`

---

### `road_buddy/prompting.py` — Xây dựng prompt

```python
build_prompt(question, choices, system_hint, target_objects, temporal_hints) → str
```
Tạo prompt tiếng Việt dạng MCQ:
```
{system_hint}

Dựa trên các khung hình trích từ video dashcam, hãy trả lời câu hỏi trắc nghiệm sau.
Chỉ được chọn DUY NHẤT một đáp án đúng nhất.

Thông tin bổ sung:
Đối tượng cần chú ý: traffic_sign, speed_limit_sign
Gợi ý thời điểm quan sát: first

Câu hỏi: Biển báo nào xuất hiện đầu tiên?

Lựa chọn:
A. Biển cấm dừng xe
B. Biển tốc độ 60
C. Biển nhường đường
D. Biển cấm rẽ trái

Bắt buộc: dòng đầu tiên chỉ được ghi DUY NHẤT một ký tự in hoa A/B/C/D, không giải thích.
Ví dụ hợp lệ: A
```

```python
extract_final_letter(text, allowed_letters) → str | None
```
Parse đáp án từ output của model theo 2 bước:
1. Tìm pattern rõ ràng: `"đáp án: C"`, `"answer: B"`, `"dap an = A"`
2. Fallback: tìm chữ cái cuối cùng đứng độc lập trong text

---

### `road_buddy/model/qwen_vl.py` — Wrapper Qwen2.5-VL

**Class `QwenVLRunner`:**

```python
load()
```
Load model `Qwen/Qwen3-VL-8B-Instruct` từ HuggingFace với `AutoModelForImageTextToText`.
Cài dtype (bfloat16), device_map="auto" (tự phân bổ GPU/CPU).

```python
predict(question, choices, images, system_hint, use_logits_only, ...) → PredictResult
```
Hàm predict chính, có 2 chế độ:

| Chế độ | Hàm | Cách hoạt động | Ưu điểm |
|--------|-----|----------------|---------|
| **Logits** | `_predict_logits_only()` | Generate max 1 token, lấy logit của token A/B/C/D, softmax → argmax | Nhanh, deterministic, trả về xác suất |
| **Generate** | `_predict_generate()` | Generate text đầy đủ (128 tokens), parse bằng regex | Dùng khi logits fail |

**Cơ chế fallback:** nếu `use_logits_only=True` nhưng logits bị lỗi → tự động chạy generate mode, ghi `source = "generate_after_logits_fallback"`.

**`PredictResult`:**
```python
@dataclass
class PredictResult:
    answer: str          # "A" / "B" / "C" / "D"
    probs: dict          # {"A": 0.7, "B": 0.1, "C": 0.1, "D": 0.1}
    raw_text: str        # Text output thô từ model (nếu generate mode)
    source: str          # Nguồn gốc: "logits_only", "generate_regex", ...
```

---

### `road_buddy/pipeline.py` — Orchestration chính

**`run_inference(config) → InferenceSummary`** — hàm điều phối toàn bộ:

1. **Setup**: `seed_everything()`, load dataset, load answer map, khởi tạo model
2. **Signal handling**: bắt SIGTERM/SIGINT → graceful shutdown (lưu partial results)
3. **Vòng lặp inference** (mỗi sample):
   - Phân tích câu hỏi (`analyze_question`)
   - Lấy frames (`sample_video_frames`) với `candidate_multiplier × num_frames` candidates
   - Chọn frames tốt nhất (`select_query_aware_frames`)
   - Thêm vào `pending` buffer
   - Khi đủ `batch_size` → xử lý (`_process_pending`)
4. **`_process_pending()`**: chạy `model_runner.predict()`, xử lý OOM exception, tính latency
5. **Checkpoint**: cứ mỗi N mẫu → ghi intermediate `submission.csv` + `audit.json`
6. **Kết thúc**: lưu final results, tính stats

**Tracking per-sample trong `audit.json`:**
- Question, choices, answer dự đoán
- Raw text từ model, probability distribution
- pred_source (biết model dùng cách nào để trả lời)
- Query intent, target objects, temporal hints
- Indices và scores của frames được chọn
- Ground truth và is_correct (nếu có answer_json)
- Latency, status, error message

**`InferenceSummary`** (kết quả trả về):
```python
total_samples, output_csv, avg_latency_sec, p95_latency_sec,
over_time_limit_count, eval_stats (accuracy nếu có ground truth)
```

---

### `road_buddy/cli.py` — Giao diện dòng lệnh

3 subcommand:

```bash
# In config đã được resolve (debug paths)
python -m road_buddy.cli show-config --config configs/baseline_qwen.yaml

# Chạy inference (nộp bài)
python -m road_buddy.cli infer --config configs/baseline_qwen.yaml

# Đánh giá file submission đã có
python -m road_buddy.cli eval --submission outputs/submission.csv --answer-json datasets/public_test/public_test_with_answers.json
```

---

## 4. File cấu hình (`configs/baseline_qwen.yaml`)

```yaml
paths:
  dataset_json: ../datasets/public_test/public_test.json
  video_root: ../datasets
  output_csv: ../outputs/submission_baseline_qwen.csv
  answer_json: ../datasets/public_test/public_test_with_answers.json  # optional, để eval local
  audit_json: ../outputs/audit_baseline_qwen.json

model:
  model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
  torch_dtype: bfloat16
  device_map: auto

sampling:
  num_frames: 10                    # Số frames đưa vào model
  max_side: 960                     # Resize nếu cạnh dài > 960px
  query_aware_enabled: true         # Bật chọn frame thông minh
  candidate_frame_multiplier: 3     # Lấy 10×3=30 candidates rồi chọn 10 tốt nhất

runtime:
  seed: 42
  default_answer: A                 # Trả lời mặc định nếu mọi thứ fail
  time_limit_sec: 30                # Giới hạn thời gian/câu (theo đề bài)
  use_logits_only: false            # false = dùng generate mode
  checkpoint_every_n: 5            # Lưu checkpoint mỗi 5 mẫu
  batch_size: 4                     # Số mẫu xử lý mỗi batch
  clear_cuda_cache_on_each_sample: true

prompt:
  system_hint: "Bạn là một trợ lý thông minh..."  # Vai trò của AI
```

---

## 5. SLURM Script (`scripts/run_baseline.slurm`)

Script để submit job lên HPC cluster:

| Bước | Mô tả |
|------|-------|
| `#SBATCH` headers | 1 node, 4 CPU, 24GB RAM, 1 GPU (L40 qua MPS) |
| Bước 1 | Load modules: CUDA 12.8, Slurm |
| Bước 2 | Khai báo biến paths (REPO, PYTHON, CONFIG_PATH, ...) |
| Bước 3 | Kiểm tra GPU VRAM đủ 24GB qua `gpu_check.sh` |
| Bước 4 | Khởi tạo CUDA MPS (Multi-Process Service) |
| Bước 5 | Setup cache dirs HuggingFace, tạo thư mục log/output |
| Bước 6 | Load HF token từ `/datastore/cndt_khanhnd/.hf_token` |
| Bước 7 | **Tạo config riêng cho job này** (inline Python) — resolve paths tuyệt đối, đặt output vào `outputs/$SLURM_JOB_ID/` để nhiều job chạy song song không ghi đè nhau |
| Bước 8 | Chạy inference: `python -m road_buddy.cli infer --config <job_config>` |

**Mỗi job tạo output riêng:**
```
outputs/<SLURM_JOB_ID>/
├── submission.csv
├── audit.json
└── config_job_<SLURM_JOB_ID>.yaml

logs/<SLURM_JOB_ID>/
├── job.out
└── job.err
```

---

## 6. Xử lý lỗi và edge cases

| Tình huống | Xử lý |
|-----------|-------|
| Không parse được choices | Dùng `default_answer` (A), ghi `pred_source="default_no_choice"` |
| Không đọc được video / không có frame | Dùng đáp án đầu tiên trong choices, ghi `status="frame_extract_exception"` |
| OOM (CUDA out of memory) | Catch exception, dùng default, ghi `pred_source="default_on_oom"` |
| Logits mode lỗi | Fallback sang generate mode, ghi `source="generate_after_logits_fallback"` |
| Không regex ra đáp án | Dùng `choice_letters[0]`, ghi `source="generate_default"` |
| SIGTERM / SIGINT | Dừng vòng lặp, lưu partial results, restore signal handlers |
| Vượt time_limit_sec | Vẫn lưu kết quả nhưng tăng `over_time_limit_count` |

---

## 7. Cách chạy nhanh

```bash
# Cài đặt
cd road-buddy
pip install -e .

# Chạy inference
python -m road_buddy.cli infer --config configs/baseline_qwen.yaml

# Đánh giá kết quả
python -m road_buddy.cli eval \
  --submission outputs/submission_baseline_qwen.csv \
  --answer-json datasets/public_test/public_test_with_answers.json

# Submit SLURM job
sbatch scripts/run_baseline.slurm
```

---

## 8. Sơ đồ dependency giữa các module

```
cli.py
  └── pipeline.run_inference()
        ├── config.load_config()
        ├── dataio.load_dataset()
        ├── dataio.load_answer_map()
        ├── video.sample_video_frames()
        │     └── video.extract_frames_1fps_from_video()  [OpenCV]
        ├── query_aware.analyze_question()
        ├── query_aware.select_query_aware_frames()
        │     ├── _compute_sharpness_scores()  [Laplacian]
        │     └── _temporal_prior()
        ├── model.qwen_vl.QwenVLRunner.predict()
        │     ├── prompting.build_prompt()
        │     ├── _predict_logits_only()  [softmax trên A/B/C/D tokens]
        │     └── _predict_generate()    [generate + regex]
        │           └── prompting.extract_final_letter()
        ├── dataio.save_submission()       → submission.csv
        └── pipeline._save_audit_payload() → audit.json
```
