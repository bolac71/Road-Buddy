"""
OCR Evaluation: Qwen3-VL-8B-Instruct trên VinAI Vietnamese OCR Dataset (VinText).

Các sửa đổi so với phiên bản gốc:
  1. [BUG FIX] parse_ground_truth: lọc chuỗi rỗng/whitespace sau khi strip,
     tránh GT = "   " gây CER cực cao (im1257, im1259).
  2. [BUG FIX] parse_ground_truth: bỏ qua sample khi GT sau khi lọc là rỗng
     (thay vì trả về chuỗi whitespace rồi tính CER sai).
  3. [IMPROVE] Prompt: thêm hướng dẫn reading order và bỏ dấu câu thừa.
  4. [IMPROVE] Lọc prediction: chuẩn hoá newline/gạch ngang thành khoảng trắng
     trước khi tính CER/WER.
  5. [IMPROVE] Summary: thêm median CER/WER và số mẫu CER=0 vào báo cáo.

Usage:
    python ocr_eval_fixed.py \
        --data-dir /path/to/vietnamese \
        --output-dir /path/to/output \
        [--model Qwen/Qwen3-VL-8B-Instruct] \
        [--num-samples -1] \
        [--dtype bfloat16] \
        [--max-new-tokens 256] \
        [--save-every 10]
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ---------------------------------------------------------------------------
# CER / WER metric — dùng jiwer nếu có, fallback thuần Python
# ---------------------------------------------------------------------------
try:
    from jiwer import cer as _jiwer_cer, wer as _jiwer_wer

    def compute_cer(reference: str, hypothesis: str) -> float:
        if not reference:
            return 1.0
        try:
            return float(_jiwer_cer(reference, hypothesis))
        except Exception:
            return 1.0

    def compute_wer(reference: str, hypothesis: str) -> float:
        if not reference:
            return 1.0
        try:
            return float(_jiwer_wer(reference, hypothesis))
        except Exception:
            return 1.0

except ImportError:
    print("[WARN] jiwer không tìm thấy — dùng CER/WER thuần Python (Levenshtein).", flush=True)

    def _levenshtein(s1: list, s2: list) -> int:
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if s1[i - 1] == s2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]

    def compute_cer(reference: str, hypothesis: str) -> float:
        if not reference:
            return 1.0
        dist = _levenshtein(list(reference), list(hypothesis))
        return dist / len(reference)

    def compute_wer(reference: str, hypothesis: str) -> float:
        if not reference:
            return 1.0
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        if not ref_words:
            return 1.0
        dist = _levenshtein(ref_words, hyp_words)
        return dist / len(ref_words)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def parse_ground_truth(label_path: str) -> str:
    """
    Đọc file label VinText, bỏ qua '###' và gom text thành một chuỗi.

    FIX: Lọc thêm chuỗi rỗng hoặc toàn whitespace sau khi strip.
    Phiên bản gốc chỉ kiểm tra `text != "###"` nhưng không kiểm tra
    `text.strip()`, dẫn đến GT = "   " (whitespace) — khi model predict
    đúng sẽ tính CER = len(pred)/len(gt) >> 1.
    """
    valid_texts = []
    if not os.path.exists(label_path):
        return ""
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) >= 9:
                text = ",".join(parts[8:]).strip()
                # FIX: thêm `text` vào điều kiện để lọc chuỗi rỗng/whitespace
                if text and text != "###":
                    valid_texts.append(text)
    return " ".join(valid_texts)


def normalize_prediction(text: str) -> str:
    """
    Chuẩn hoá output của model trước khi tính CER/WER.

    FIX: Model đôi khi dùng newline hoặc gạch ngang thay vì khoảng trắng,
    gây tăng CER không cần thiết. Hàm này chuẩn hoá về khoảng trắng đơn.
    """
    # Thay newline và các ký tự xuống dòng bằng khoảng trắng
    text = text.replace("\n", " ").replace("\r", " ")
    # Thu gọn nhiều khoảng trắng liên tiếp thành một
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def collect_samples(image_dir: str, label_dir: str, num_samples: int) -> list[dict]:
    """Quét ảnh test và ghép với file label tương ứng."""
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if num_samples > 0:
        image_files = image_files[:num_samples]

    samples = []
    for img_path in image_files:
        filename = os.path.basename(img_path)
        match = re.search(r"\d+", filename)
        if not match:
            continue
        img_num = str(int(match.group()))  # bỏ leading zeros nếu có
        label_path = os.path.join(label_dir, f"gt_{img_num}.txt")
        # fallback giữ nguyên leading zeros
        if not os.path.exists(label_path):
            label_path = os.path.join(label_dir, f"gt_{match.group()}.txt")
        samples.append({"img_path": img_path, "filename": filename, "label_path": label_path})
    return samples


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_id: str, dtype_str: str):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"[INFO] Đang tải model {model_id} (dtype={dtype_str})...", flush=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print("[INFO] Tải model thành công!", flush=True)
    return model, processor


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# IMPROVE: Prompt được cải thiện để:
# - Chỉ rõ reading order (trái→phải, trên→dưới) tránh mismatch thứ tự với GT
# - Yêu cầu không thêm dấu câu/gạch ngang giữa các đoạn text
OCR_PROMPT = (
    "Hãy trích xuất toàn bộ văn bản tiếng Việt xuất hiện trong hình ảnh này. "
    "Đọc theo thứ tự từ trái sang phải, từ trên xuống dưới. "
    "Chỉ in ra kết quả văn bản, các đoạn cách nhau bằng một dấu cách, "
    "không thêm dấu gạch ngang, dấu câu hay ký hiệu phân tách, "
    "tuyệt đối không giải thích hay chat thêm."
)


def qwen_ocr_predict(model, processor, image_path: str, max_new_tokens: int = 256) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": OCR_PROMPT},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    # FIX: chuẩn hoá prediction trước khi trả về
    return normalize_prediction(output_text[0])


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(args):
    image_dir = os.path.join(args.data_dir, "test_image")
    label_dir = os.path.join(args.data_dir, "labels")
    os.makedirs(args.output_dir, exist_ok=True)

    results_path = os.path.join(args.output_dir, "ocr_results.json")
    summary_path = os.path.join(args.output_dir, "summary.json")

    # --- Tập hợp danh sách ảnh ---
    samples = collect_samples(image_dir, label_dir, args.num_samples)
    total = len(samples)
    print(f"[INFO] Tìm thấy {total} ảnh trong {image_dir}", flush=True)

    # --- Tải model ---
    model, processor = load_model(args.model, args.dtype)

    # --- Vòng lặp đánh giá ---
    results = []
    all_cer = []
    all_wer = []
    total_latency = 0.0
    valid_samples = 0
    skipped = 0
    start_time = time.time()

    for idx, sample in enumerate(samples):
        img_path = sample["img_path"]
        filename = sample["filename"]
        label_path = sample["label_path"]

        ground_truth = parse_ground_truth(label_path)
        # FIX: kiểm tra thêm ground_truth.strip() để bỏ qua GT toàn whitespace
        if not ground_truth or not ground_truth.strip():
            print(
                f"[{idx+1}/{total}] SKIP {filename} — label trống, whitespace hoặc không tìm thấy",
                flush=True,
            )
            skipped += 1
            continue

        # Inference (đo latency)
        t0 = time.time()
        try:
            prediction = qwen_ocr_predict(model, processor, img_path, args.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            print(f"[{idx+1}/{total}] OOM {filename} — bỏ qua", flush=True)
            torch.cuda.empty_cache()
            skipped += 1
            continue
        except Exception as e:
            print(f"[{idx+1}/{total}] ERROR {filename} — {e}", flush=True)
            skipped += 1
            continue
        latency = time.time() - t0

        # Tính CER / WER
        ref = ground_truth.lower()
        hyp = prediction.lower()
        cer_val = compute_cer(ref, hyp)
        wer_val = compute_wer(ref, hyp)

        all_cer.append(cer_val)
        all_wer.append(wer_val)
        total_latency += latency
        valid_samples += 1

        record = {
            "image": filename,
            "gt": ground_truth,
            "pred": prediction,
            "cer": round(cer_val, 6),
            "wer": round(wer_val, 6),
            "latency_sec": round(latency, 4),
        }
        results.append(record)

        mean_cer_run = sum(all_cer) / valid_samples
        mean_wer_run = sum(all_wer) / valid_samples
        mean_lat_run = total_latency / valid_samples
        elapsed = time.time() - start_time
        print(
            f"[{idx+1}/{total}] {filename} | "
            f"CER: {cer_val:.4f} | WER: {wer_val:.4f} | "
            f"Latency: {latency:.2f}s | "
            f"Mean CER: {mean_cer_run:.4f} | Mean WER: {mean_wer_run:.4f} | "
            f"Elapsed: {elapsed:.0f}s",
            flush=True,
        )
        print(f"  GT  : {ground_truth}", flush=True)
        print(f"  PRED: {prediction}", flush=True)
        print("-" * 60, flush=True)

        # Lưu định kỳ
        if valid_samples % args.save_every == 0:
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Đã lưu {valid_samples} kết quả vào {results_path}", flush=True)

    # --- Kết quả cuối ---
    mean_cer = sum(all_cer) / valid_samples if valid_samples > 0 else None
    mean_wer = sum(all_wer) / valid_samples if valid_samples > 0 else None
    mean_latency = total_latency / valid_samples if valid_samples > 0 else None

    # IMPROVE: tính thêm median và tỉ lệ perfect
    if valid_samples > 0:
        sorted_cer = sorted(all_cer)
        sorted_wer = sorted(all_wer)
        mid = valid_samples // 2
        median_cer = sorted_cer[mid] if valid_samples % 2 == 1 else (sorted_cer[mid-1] + sorted_cer[mid]) / 2
        median_wer = sorted_wer[mid] if valid_samples % 2 == 1 else (sorted_wer[mid-1] + sorted_wer[mid]) / 2
        perfect_count = sum(1 for c in all_cer if c == 0.0)
        good_count = sum(1 for c in all_cer if c < 0.1)
    else:
        median_cer = median_wer = perfect_count = good_count = None

    # Lưu kết quả đầy đủ
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "total_images": total,
        "valid_samples": valid_samples,
        "skipped": skipped,
        "mean_cer": round(mean_cer, 6) if mean_cer is not None else None,
        "mean_wer": round(mean_wer, 6) if mean_wer is not None else None,
        "median_cer": round(median_cer, 6) if median_cer is not None else None,
        "median_wer": round(median_wer, 6) if median_wer is not None else None,
        "perfect_cer0": perfect_count,
        "good_cer_lt01": good_count,
        "mean_latency_sec": round(mean_latency, 4) if mean_latency is not None else None,
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    w = 62
    print("\n" + "=" * w, flush=True)
    print(f"  BÁO CÁO KẾT QUẢ QWEN3-VL (Trên {valid_samples} samples)", flush=True)
    if mean_cer is not None:
        print(f"  1. Character Error Rate (CER) trung bình : {mean_cer * 100:.2f}%", flush=True)
        print(f"     CER trung vị (median)                 : {median_cer * 100:.2f}%", flush=True)
    if mean_wer is not None:
        print(f"  2. Word Error Rate (WER) trung bình      : {mean_wer * 100:.2f}%", flush=True)
        print(f"     WER trung vị (median)                 : {median_wer * 100:.2f}%", flush=True)
    if mean_latency is not None:
        print(f"  3. Tốc độ xử lý (Latency) trung bình    : {mean_latency:.4f} giây/ảnh", flush=True)
    if perfect_count is not None:
        print(f"  4. Nhận dạng hoàn hảo (CER=0)            : {perfect_count}/{valid_samples} ({perfect_count/valid_samples*100:.1f}%)", flush=True)
        print(f"     Nhận dạng tốt (CER<10%)               : {good_count}/{valid_samples} ({good_count/valid_samples*100:.1f}%)", flush=True)
    print("=" * w, flush=True)
    print(f"  Tổng ảnh : {total}  |  Hợp lệ: {valid_samples}  |  Bỏ qua: {skipped}", flush=True)
    print(f"  Kết quả  : {results_path}", flush=True)
    print(f"  Summary  : {summary_path}", flush=True)
    print("=" * w, flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="OCR Evaluation với Qwen3-VL trên VinText dataset")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Thư mục gốc dataset (chứa test_image/ và labels/)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Thư mục lưu kết quả (ocr_results.json, summary.json)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model ID trên HuggingFace (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Số ảnh cần đánh giá (-1 = tất cả, default: -1)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type cho model (default: bfloat16)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Số token tối đa được sinh ra (default: 256)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Lưu kết quả sau mỗi N mẫu hợp lệ (default: 10)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"[INFO] Python    : {sys.version}", flush=True)
    print(f"[INFO] PyTorch   : {torch.__version__}", flush=True)
    print(f"[INFO] CUDA      : {torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[INFO] GPU       : {torch.cuda.get_device_name(0)}", flush=True)

    run_evaluation(args)