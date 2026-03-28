# RoadBuddy Baseline (Qwen-VL)

Đây là baseline có kiến trúc rõ ràng cho bài toán RoadBuddy, tách thành các khối độc lập:
- Cấu hình và đường dẫn
- Đọc dữ liệu
- Trích khung hình từ video
- Tạo prompt và tách đáp án
- Wrapper model Qwen-VL
- Pipeline suy luận và đánh giá cục bộ

## Cấu trúc thư mục

- configs/: Cấu hình YAML
- datasets/: Dữ liệu train/public_test (json + videos)
- road_buddy/config.py: Đọc, kiểm tra và resolve config
- road_buddy/dataio.py: Load dataset/answer, ghi submission CSV
- road_buddy/video.py: Sample frame trực tiếp từ video mp4
- road_buddy/prompting.py: Prompt template và regex tách đáp án
- road_buddy/model/qwen_vl.py: Wrapper model Qwen2.5-VL
- road_buddy/pipeline.py: Chạy infer + đánh giá local
- road_buddy/cli.py: Điểm vào CLI
- scripts/run_baseline.sh: Script chạy nhanh local
- scripts/run_baseline.slurm: Script chạy trên cụm SLURM

## Cài đặt

1. Di chuyển vào thư mục dự án.
2. Cài phụ thuộc Python.

Ví dụ:
python -m pip install -r requirements.txt

## Tạo môi trường Conda riêng cho dự án

Nên tạo env mới tên road_buddy để tách biệt với SeCap.

1. Tạo env:
conda create -n road_buddy python=3.12 -y

2. Kích hoạt env:
conda activate road_buddy

3. Nâng cấp pip:
python -m pip install --upgrade pip

4. Cài dependencies:
python -m pip install -r requirements.txt

5. Kiểm tra nhanh:
python -c "import torch, transformers, cv2, PIL, yaml, tqdm, numpy, pandas; print('OK')"## Cấu hình

Config mặc định: configs/baseline_qwen.yaml

Các trường quan trọng cần kiểm tra:
- paths.dataset_json
- paths.video_root
- paths.output_csv
- paths.answer_json
- paths.audit_json
- model.model_name_or_path

Ghi chú:
- Nếu máy chủ không có Internet, hãy đặt model.model_name_or_path về thư mục model cục bộ đã tải sẵn.
- Mặc định hiện tại dùng output text của model rồi regex để tách đáp án (không ưu tiên logits token).

## Chạy suy luận (infer)

Cách 1, chạy trực tiếp:
python -m road_buddy.cli infer --config configs/baseline_qwen.yaml

Cách 2, dùng script:
bash scripts/run_baseline.sh

CLI sẽ in:
- total_samples
- output_csv
- avg_latency_sec
- p95_latency_sec
- over_time_limit_count
- eval_accuracy (nếu có answer_json)

Ngoài ra, hệ thống sẽ ghi file audit JSON tại đường dẫn paths.audit_json.
File này chứa từng mẫu với các thông tin để kiểm tra lại như: question, choices, raw_text từ model, answer cuối, latency, trạng thái lỗi/fallback.

Để tránh mất kết quả khi bị time limit:
- runtime.checkpoint_every_n: cứ N câu sẽ ghi tạm submission và audit ra đĩa.
- runtime.print_every_n: cứ N câu sẽ in log tiến trình trực tiếp ra stdout.
- runtime.batch_size: số mẫu infer theo batch (hiệu quả nhất khi mỗi mẫu dùng ít frame).
- Khi nhận SIGTERM/SIGINT, pipeline sẽ dừng mềm, lưu kết quả tạm đã có và có thể dùng ngay để eval trên số câu đã infer.

Chiến lược lấy frame hiện tại:
- Cắt 1 frame mỗi giây từ video.
- Lấy tối đa sampling.num_frames frame đầu theo nhịp 1fps.

## Chạy đánh giá riêng (eval)

python -m road_buddy.cli eval --submission outputs/submission_baseline_qwen.csv --answer-json datasets/public_test/public_test_with_answers.json

## Chạy bằng SLURM

Đã có sẵn script: scripts/run_baseline.slurm

Bạn chỉ cần:
1. Tạo thư mục log trước khi submit:
mkdir -p logs
2. Mở script và chỉnh các biến theo máy của bạn: CONDA_SH, CONDA_ENV_NAME, REPO_DIR, CONFIG_PATH, REQUIRED_VRAM.
3. Submit job:
sbatch scripts/run_baseline.slurm

## Ghi chú baseline

- Baseline hiện dùng generate output rồi trích đáp án A/B/C/D bằng regex.
- Frame sampling đang là uniform theo tổng số frame (sampling.num_frames).
- Chưa fine-tune, đang dùng zero-shot Qwen-VL.
- Đây là nền để mở rộng thêm các hướng như frame scoring theo motion, caching, ensembling và hậu xử lý theo luật.

## Analize audit
python scripts/analyze_audit.py \
    --audit  outputs/11231/audit.json \
    --output outputs/11231/analysis