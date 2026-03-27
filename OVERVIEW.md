# Tài liệu tổng hợp RoadBuddy

## 1. Mục tiêu dự án

RoadBuddy baseline giải bài toán trắc nghiệm từ video giao thông:
- Input: video + câu hỏi + các lựa chọn.
- Output: một đáp án chữ cái (A/B/C/D) cho mỗi câu.
- Kết quả cuối: file CSV gồm hai cột id, answer.

## 2. Luồng chạy end-to-end

1. CLI đọc file config YAML.
2. Nạp dataset JSON và map đáp án (nếu có).
3. Với từng sample:
- Resolve đường dẫn video.
- Trích frame đều theo thời gian.
- Tạo prompt từ câu hỏi + choices.
- Chạy model Qwen-VL để sinh text trả lời.
- Dùng regex để rút đáp án A/B/C/D từ text model.
4. Ghi submission CSV.
5. Ghi audit JSON để phục vụ kiểm tra chất lượng dự đoán.
5. Nếu có answer_json, tính accuracy local.

## 3. Thành phần chính

- road_buddy/config.py
- Nhiệm vụ: parse YAML, validate key bắt buộc, resolve path tương đối thành tuyệt đối.

- road_buddy/dataio.py
- Nhiệm vụ: load dataset, resolve video path, load answer map, ghi/đọc submission CSV.
- Lưu ý: đã hỗ trợ format answer dạng map hoặc dạng data list có id/answer.
- Lưu ý: có fallback alias cho train path dataset/videos -> train/videos.

- road_buddy/video.py
- Nhiệm vụ: dùng OpenCV đọc video và lấy num_frames khung hình phân bố đều.
- Có resize giữ tỉ lệ theo max_side để giảm tải tính toán.

- road_buddy/prompting.py
- Nhiệm vụ: build prompt và regex trích đáp án.
- Cơ chế trích đáp án:
  - Ưu tiên cụm rõ nghĩa như "Đáp án: C" hoặc "answer: B".
  - Nếu không có, lấy chữ cái A-D cuối cùng xuất hiện độc lập trong output.

- road_buddy/model/qwen_vl.py
- Nhiệm vụ: load model/processor, chuẩn bị input multimodal, sinh output.
- Mặc định hiện tại ưu tiên generate text và regex extract.
- Vẫn có nhánh logits-only để thử nghiệm khi cần.

- road_buddy/pipeline.py
- Nhiệm vụ: điều phối infer toàn bộ dataset, đo latency, thống kê quá hạn, tính accuracy local.

- road_buddy/cli.py
- Lệnh:
  - show-config
  - infer
  - eval

## 4. Dữ liệu và cấu hình hiện tại

Config mặc định nằm ở configs/baseline_qwen.yaml, đang trỏ về:
- Dataset: datasets/public_test/public_test.json
- Video root: datasets
- Answer: datasets/public_test/public_test_with_answers.json
- Output: outputs/submission_baseline_qwen.csv
- Audit: outputs/audit_baseline_qwen.json

## 5. Cách chạy

Chạy local:
- python -m road_buddy.cli infer --config configs/baseline_qwen.yaml

Chạy eval riêng:
- python -m road_buddy.cli eval --submission outputs/submission_baseline_qwen.csv --answer-json datasets/public_test/public_test_with_answers.json

Chạy qua SLURM:
- sbatch scripts/run_baseline.slurm

## 6. Các điểm cần lưu ý khi mở rộng

- Nếu dùng model lớn hơn, cân nhắc giảm num_frames hoặc max_side để giữ latency.
- Nếu output model có xu hướng dài dòng, regex vẫn hoạt động nhưng nên chuẩn hóa prompt theo một format đáp án cố định.
- Nếu thay đổi schema dataset, cần cập nhật load_dataset và load_answer_map tương ứng.
- Nếu chạy production, nên bổ sung logging chi tiết cho raw_text để phục vụ phân tích lỗi.

## 7. Checklist rà soát nhanh

- Config path có trỏ đúng datasets mới chưa.
- Video path có tồn tại thật trong thư mục videos.
- model_name_or_path có truy cập được (local hoặc hub).
- CUDA/driver tương thích với torch + transformers.
- Có đủ dung lượng output và cache model.
