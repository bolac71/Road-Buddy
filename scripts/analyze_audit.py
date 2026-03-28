#!/usr/bin/env python3
"""
analyze_audit.py — Phân tích và báo cáo kết quả từ audit.json của Road-Buddy.

Tạo ra:
  <output_dir>/
    figures/
      01_accuracy_by_intent.png
      02_confusion_matrix.png
      03_latency_distribution.png
      04_prediction_source.png
      05_frame_score_by_intent.png
    summary.json          ← tất cả số liệu dạng machine-readable
    report.md             ← báo cáo hoàn chỉnh bằng tiếng Việt, nhúng hình

Usage:
    python scripts/analyze_audit.py \\
        --audit outputs/11231/audit.json \\
        --output outputs/11231/analysis
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Optional heavy deps — degrade gracefully so users know what to install.
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ============================================================
# 1. LOAD & FLATTEN DATA
# ============================================================

def load_audit(path: str) -> tuple[dict, list[dict]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload.get("meta", {})
    records = payload.get("records", [])
    return meta, records


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ============================================================
# 2. OVERALL METRICS
# ============================================================

def compute_overall(meta: dict, records: list[dict]) -> dict:
    total = len(records)
    graded = [r for r in records if r.get("is_correct") is not None]
    correct = sum(1 for r in graded if r["is_correct"])
    accuracy = correct / len(graded) if graded else 0.0

    latencies = [r["latency_sec"] for r in records if r.get("latency_sec", 0) > 0]
    over_limit = sum(1 for r in records if r.get("latency_sec", 0) > 30.0)

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        sorted_d = sorted(data)
        idx = p / 100 * (len(sorted_d) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_d) - 1)
        frac = idx - lo
        return sorted_d[lo] * (1 - frac) + sorted_d[hi] * frac

    latency_stats: dict = {}
    if latencies:
        latency_stats = {
            "mean":   round(sum(latencies) / len(latencies), 4),
            "median": round(percentile(latencies, 50), 4),
            "p75":    round(percentile(latencies, 75), 4),
            "p95":    round(percentile(latencies, 95), 4),
            "p99":    round(percentile(latencies, 99), 4),
            "max":    round(max(latencies), 4),
            "min":    round(min(latencies), 4),
        }

    return {
        "total_samples":    total,
        "graded_samples":   len(graded),
        "correct":          correct,
        "accuracy":         round(accuracy, 6),
        "over_time_limit":  over_limit,
        "over_limit_rate":  round(over_limit / total, 6) if total else 0.0,
        "model":            meta.get("model_name_or_path", "unknown"),
        "interrupted":      meta.get("interrupted", False),
        "latency":          latency_stats,
    }


# ============================================================
# 3. PER-INTENT ACCURACY
# ============================================================

INTENT_LABEL = {
    "temporal":       "Temporal\n(trước/sau/hiện tại)",
    "value":          "Value\n(bao nhiêu/tốc độ)",
    "direction":      "Direction\n(hướng/rẽ)",
    "identification": "Identification\n(biển gì/là gì)",
    "existence":      "Existence\n(có/không)",
    "unknown":        "Unknown",
}

def compute_by_intent(records: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        intent = r.get("query_intent", "unknown") or "unknown"
        if r.get("is_correct") is not None:
            buckets[intent].append(r["is_correct"])

    result = {}
    for intent, vals in sorted(buckets.items()):
        n = len(vals)
        c = sum(vals)
        result[intent] = {"total": n, "correct": c, "accuracy": round(c / n, 6) if n else 0.0}
    return result


# ============================================================
# 4. CONFUSION MATRIX
# ============================================================

LABELS = ["A", "B", "C", "D"]

def compute_confusion(records: list[dict]) -> list[list[int]]:
    """4×4 matrix: cm[true_idx][pred_idx]"""
    cm = [[0] * 4 for _ in range(4)]
    for r in records:
        gt = (r.get("gt_label") or "").strip().upper()
        pred = (r.get("answer") or "").strip().upper()
        if gt in LABELS and pred in LABELS:
            cm[LABELS.index(gt)][LABELS.index(pred)] += 1
    return cm


# ============================================================
# 5. PREDICTION SOURCE
# ============================================================

def compute_by_source(records: list[dict]) -> dict[str, dict]:
    buckets: dict[str, list] = defaultdict(list)
    for r in records:
        src = r.get("pred_source") or "unknown"
        buckets[src].append(r.get("is_correct"))

    result = {}
    for src, vals in sorted(buckets.items()):
        graded = [v for v in vals if v is not None]
        n_total = len(vals)
        n_graded = len(graded)
        c = sum(graded)
        result[src] = {
            "total":    n_total,
            "graded":   n_graded,
            "correct":  c,
            "accuracy": round(c / n_graded, 6) if n_graded else None,
        }
    return result


# ============================================================
# 6. FRAME SELECTION STATS
# ============================================================

def compute_frame_stats(records: list[dict]) -> dict:
    n_frames = [len(r.get("selected_frame_indices") or []) for r in records]
    all_scores = []
    intent_scores: dict[str, list] = defaultdict(list)
    for r in records:
        scores = r.get("selected_frame_scores") or []
        intent = r.get("query_intent", "unknown") or "unknown"
        all_scores.extend(scores)
        intent_scores[intent].extend(scores)

    def _stats(lst: list[float]) -> dict:
        if not lst:
            return {}
        s = sorted(lst)
        n = len(s)
        return {
            "mean":   round(sum(s) / n, 4),
            "median": round(s[n // 2], 4),
            "min":    round(s[0], 4),
            "max":    round(s[-1], 4),
        }

    return {
        "mean_frames_selected": round(sum(n_frames) / len(n_frames), 2) if n_frames else 0,
        "overall_score":        _stats(all_scores),
        "score_by_intent":      {k: _stats(v) for k, v in sorted(intent_scores.items())},
        "n_frames_distribution": {
            str(k): n_frames.count(k)
            for k in sorted(set(n_frames))
        },
    }


# ============================================================
# 7. FIGURES
# ============================================================

FIG_DPI  = 150
FIG_DIR  = "figures"
PALETTE  = "#2196F3"
GREEN    = "#4CAF50"
RED      = "#F44336"
GRAY     = "#9E9E9E"


def _save(fig: "plt.Figure", path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path}")


def fig_accuracy_by_intent(by_intent: dict[str, dict], out: Path) -> None:
    if not HAS_MPL:
        return
    intents = list(by_intent.keys())
    accs    = [by_intent[i]["accuracy"] * 100 for i in intents]
    totals  = [by_intent[i]["total"] for i in intents]
    labels  = [INTENT_LABEL.get(i, i) for i in intents]

    fig, ax = plt.subplots(figsize=(9, max(4, len(intents) * 0.9)))
    bars = ax.barh(labels, accs, color=PALETTE, edgecolor="white", height=0.6)
    ax.axvline(x=100 / 4, color=GRAY, linestyle="--", linewidth=1, label="Random baseline (25%)")
    for bar, acc, total in zip(bars, accs, totals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%  (n={total})", va="center", ha="left", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Accuracy theo loại câu hỏi (Question Intent)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, out)


def fig_confusion_matrix(cm: list[list[int]], out: Path) -> None:
    if not HAS_MPL or not HAS_NP:
        return
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    # Normalize by true class (row)
    with np.errstate(invalid="ignore"):
        cm_norm = np.where(row_sums > 0, cm_arr / row_sums, 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- raw counts ---
    ax = axes[0]
    if HAS_SNS:
        sns.heatmap(cm_arr.astype(int), annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax, linewidths=0.5)
    else:
        im = ax.imshow(cm_arr, cmap="Blues", aspect="auto")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(int(cm_arr[i, j])), ha="center", va="center", fontsize=11)
        ax.set_xticks(range(4)); ax.set_xticklabels(LABELS)
        ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
        fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (số lượng)")

    # --- normalized ---
    ax2 = axes[1]
    if HAS_SNS:
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=LABELS, yticklabels=LABELS, ax=ax2,
                    vmin=0, vmax=1, linewidths=0.5)
    else:
        im2 = ax2.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax2.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=10)
        ax2.set_xticks(range(4)); ax2.set_xticklabels(LABELS)
        ax2.set_yticks(range(4)); ax2.set_yticklabels(LABELS)
        fig.colorbar(im2, ax=ax2)
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("Ground Truth")
    ax2.set_title("Confusion Matrix (tỷ lệ theo hàng / True class)")

    fig.suptitle("Confusion Matrix — Dự đoán vs Đáp án đúng", fontsize=13, fontweight="bold")
    _save(fig, out)


def fig_latency_distribution(records: list[dict], out: Path) -> None:
    if not HAS_MPL or not HAS_NP:
        return
    lats = [r["latency_sec"] for r in records if r.get("latency_sec", 0) > 0]
    if not lats:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram
    ax = axes[0]
    n_bins = min(40, max(10, len(lats) // 5))
    ax.hist(lats, bins=n_bins, color=PALETTE, edgecolor="white", alpha=0.85)
    for p, label, color in [(50, "p50", GREEN), (95, "p95", "#FF9800"), (99, "p99", RED)]:
        idx  = int(p / 100 * (len(sorted(lats)) - 1))
        val  = sorted(lats)[idx]
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5, label=f"{label}={val:.1f}s")
    ax.axvline(30, color=RED, linestyle="-", linewidth=2, alpha=0.5, label="Time limit (30s)")
    ax.set_xlabel("Latency (giây)"); ax.set_ylabel("Số mẫu")
    ax.set_title("Phân phối latency (histogram)")
    ax.legend(fontsize=8)

    # Box plot + swarm by correct/incorrect
    ax2 = axes[1]
    correct_lats   = [r["latency_sec"] for r in records if r.get("is_correct") is True  and r.get("latency_sec", 0) > 0]
    incorrect_lats = [r["latency_sec"] for r in records if r.get("is_correct") is False and r.get("latency_sec", 0) > 0]
    data_to_plot   = [x for x in [correct_lats, incorrect_lats] if x]
    labels_bp      = []
    if correct_lats:   labels_bp.append("Đúng")
    if incorrect_lats: labels_bp.append("Sai")
    if data_to_plot:
        bp = ax2.boxplot(data_to_plot, tick_labels=labels_bp, patch_artist=True,
                         medianprops=dict(color="black", linewidth=2))
        colors = [GREEN, RED]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.6)
    ax2.axhline(30, color=RED, linestyle="--", linewidth=1.5, alpha=0.6, label="Time limit (30s)")
    ax2.set_ylabel("Latency (giây)"); ax2.set_title("Latency: Đúng vs Sai")
    ax2.legend(fontsize=8)

    fig.suptitle("Phân tích Latency", fontsize=13, fontweight="bold")
    _save(fig, out)


def fig_prediction_source(by_source: dict[str, dict], out: Path) -> None:
    if not HAS_MPL:
        return
    sources = list(by_source.keys())
    totals  = [by_source[s]["total"] for s in sources]
    correct_vals = [by_source[s]["correct"] for s in sources]
    wrong_vals   = [by_source[s]["total"] - by_source[s]["correct"] for s in sources]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(sources) * 0.7 + 2)))

    # Stacked bar: correct/incorrect
    ax = axes[0]
    y_pos = range(len(sources))
    ax.barh(list(y_pos), correct_vals, color=GREEN, alpha=0.85, label="Đúng")
    ax.barh(list(y_pos), wrong_vals, left=correct_vals, color=RED, alpha=0.75, label="Sai")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(sources, fontsize=8)
    ax.set_xlabel("Số mẫu")
    ax.set_title("Phân phối Prediction Source\n(đúng/sai)")
    ax.legend(fontsize=9)
    for i, (c, t) in enumerate(zip(correct_vals, totals)):
        ax.text(t + 0.3, i, f"n={t}", va="center", fontsize=8)

    # Accuracy per source
    ax2 = axes[1]
    accs = [by_source[s]["accuracy"] * 100 if by_source[s]["accuracy"] is not None else 0
            for s in sources]
    bar_colors = [GREEN if a >= 50 else RED for a in accs]
    ax2.barh(list(y_pos), accs, color=bar_colors, alpha=0.8)
    ax2.axvline(25, color=GRAY, linestyle="--", linewidth=1, label="Random (25%)")
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(sources, fontsize=8)
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Accuracy (%)")
    ax2.set_title("Accuracy theo Prediction Source")
    ax2.legend(fontsize=9)
    for i, (a, s) in enumerate(zip(accs, sources)):
        g = by_source[s]["graded"]
        if g:
            ax2.text(a + 0.5, i, f"{a:.1f}%", va="center", fontsize=8)

    fig.suptitle("Phân tích Prediction Source", fontsize=13, fontweight="bold")
    _save(fig, out)


def fig_frame_scores_by_intent(records: list[dict], out: Path) -> None:
    if not HAS_MPL or not HAS_NP:
        return
    intent_scores: dict[str, list] = defaultdict(list)
    for r in records:
        intent = r.get("query_intent", "unknown") or "unknown"
        scores = r.get("selected_frame_scores") or []
        if scores:
            intent_scores[intent].extend(scores)

    intents = sorted(intent_scores.keys())
    if not intents:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    data_bp  = [intent_scores[i] for i in intents]
    labels_i = [INTENT_LABEL.get(i, i) for i in intents]
    bp = ax.boxplot(data_bp, tick_labels=labels_i, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE); patch.set_alpha(0.65)
    ax.set_ylabel("Frame Score (0–1)")
    ax.set_title("Phân phối Frame Score theo Question Intent", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _save(fig, out)


# ============================================================
# 8. MARKDOWN REPORT
# ============================================================

def _table(headers: list[str], rows: list[list[str]]) -> str:
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in col_widths) + " |"
    head = "| " + " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return f"{head}\n{sep}\n{body}"


def write_report(
    out_dir: Path,
    overall: dict,
    by_intent: dict[str, dict],
    cm: list[list[int]],
    by_source: dict[str, dict],
    frame_stats: dict,
    has_figures: bool,
) -> None:
    fig_prefix = "figures/" if has_figures else ""
    lines: list[str] = []

    lines += [
        "# Road-Buddy — Báo cáo Kết quả Inference",
        "",
        "> Tài liệu này được tạo tự động bởi `scripts/analyze_audit.py`.",
        "",
        "---",
        "",
        "## 1. Tổng quan",
        "",
        f"| Chỉ số | Giá trị |",
        f"|--------|---------|",
        f"| Model | `{overall['model']}` |",
        f"| Tổng số mẫu | {overall['total_samples']} |",
        f"| Mẫu có ground truth | {overall['graded_samples']} |",
        f"| Dự đoán đúng | {overall['correct']} |",
        f"| **Accuracy** | **{_pct(overall['accuracy'])}** |",
        f"| Vượt time limit (>30s) | {overall['over_time_limit']} ({_pct(overall['over_limit_rate'])}) |",
        f"| Bị ngắt giữa chừng | {'Có' if overall['interrupted'] else 'Không'} |",
        "",
    ]

    # Latency table
    lat = overall.get("latency", {})
    if lat:
        lines += [
            "### Latency (giây)",
            "",
            "| Mean | Median | p75 | p95 | p99 | Max |",
            "|------|--------|-----|-----|-----|-----|",
            f"| {lat.get('mean','—')} | {lat.get('median','—')} | {lat.get('p75','—')} | {lat.get('p95','—')} | {lat.get('p99','—')} | {lat.get('max','—')} |",
            "",
        ]

    lines += ["---", "", "## 2. Accuracy theo loại câu hỏi (Question Intent)", ""]
    if has_figures:
        lines.append(f"![Accuracy by Intent]({fig_prefix}01_accuracy_by_intent.png)")
        lines.append("")

    intent_rows = []
    for intent, stats in sorted(by_intent.items()):
        label = INTENT_LABEL.get(intent, intent).replace("\n", " ")
        intent_rows.append([
            label,
            str(stats["total"]),
            str(stats["correct"]),
            _pct(stats["accuracy"]),
        ])
    lines.append(_table(["Intent", "Tổng", "Đúng", "Accuracy"], intent_rows))
    lines += ["", "---", "", "## 3. Confusion Matrix", ""]
    if has_figures:
        lines.append(f"![Confusion Matrix]({fig_prefix}02_confusion_matrix.png)")
        lines.append("")

    # Raw counts table
    header = ["GT \\ Pred"] + LABELS
    cm_rows = [[LABELS[i]] + [str(cm[i][j]) for j in range(4)] for i in range(4)]
    lines.append("**Số lượng (raw counts):**")
    lines.append("")
    lines.append(_table(header, cm_rows))
    lines.append("")

    # Per-class metrics
    lines.append("**Precision / Recall / F1 per class:**")
    lines.append("")
    pr_rows = []
    for i, lbl in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(4)) - tp
        fn = sum(cm[i][j] for j in range(4)) - tp
        prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        pr_rows.append([lbl, str(tp), f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}"])
    lines.append(_table(["Class", "TP", "Precision", "Recall", "F1"], pr_rows))
    lines += ["", "---", "", "## 4. Phân tích Latency", ""]
    if has_figures:
        lines.append(f"![Latency]({fig_prefix}03_latency_distribution.png)")
        lines.append("")

    lines += ["---", "", "## 5. Prediction Source", ""]
    if has_figures:
        lines.append(f"![Prediction Source]({fig_prefix}04_prediction_source.png)")
        lines.append("")

    src_rows = []
    for src, s in sorted(by_source.items()):
        acc_str = _pct(s["accuracy"]) if s["accuracy"] is not None else "—"
        src_rows.append([src, str(s["total"]), str(s["correct"]), acc_str])
    lines.append(_table(["Source", "Tổng", "Đúng", "Accuracy"], src_rows))
    lines += [
        "",
        "> **generate_regex**: model sinh text → regex tìm A/B/C/D thành công.",
        "> **generate_default**: regex không tìm được → dùng đáp án đầu tiên (A).",
        "> **default_on_exception / default_on_oom**: lỗi runtime / OOM.",
        "> **default_on_frame_extract_error**: không đọc được video.",
        "",
    ]

    lines += ["---", "", "## 6. Frame Selection", ""]
    if has_figures:
        lines.append(f"![Frame Scores]({fig_prefix}05_frame_score_by_intent.png)")
        lines.append("")

    lines.append(f"- Trung bình số frame được chọn mỗi mẫu: **{frame_stats.get('mean_frames_selected', '—')}**")
    lines.append("")

    score_by_intent = frame_stats.get("score_by_intent", {})
    if score_by_intent:
        fs_rows = []
        for intent, s in sorted(score_by_intent.items()):
            label = INTENT_LABEL.get(intent, intent).replace("\n", " ")
            fs_rows.append([label,
                            str(s.get("mean","—")), str(s.get("median","—")),
                            str(s.get("min","—")),  str(s.get("max","—"))])
        lines.append(_table(["Intent", "Mean Score", "Median", "Min", "Max"], fs_rows))
        lines.append("")

    lines += [
        "---",
        "",
        "## 7. Ghi chú Phương pháp",
        "",
        "- **Frame sampling**: trích xuất 1 frame/giây (uniform), lấy `num_frames × candidate_multiplier` candidate frames.",
        "- **Query-aware selection**: kết hợp *sharpness score* (Laplacian variance) và *temporal prior*",
        "  (ưu tiên đầu/cuối/giữa video theo intent câu hỏi).",
        "- **Prediction**: Qwen2.5-VL generate text → regex parse A/B/C/D.",
        "  Nếu regex fail → fallback về đáp án đầu tiên trong choices.",
        "- **Scoring**: Accuracy = số đúng / tổng số mẫu có ground truth.",
        "",
    ]

    report_path = out_dir / "report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [report] {report_path}")


# ============================================================
# 9. MAIN
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Road-Buddy audit.json")
    parser.add_argument("--audit",  required=True, help="Đường dẫn tới audit.json")
    parser.add_argument("--output", required=True, help="Thư mục output cho báo cáo")
    args = parser.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"[ERROR] Không tìm thấy: {audit_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    fig_dir = out_dir / FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {audit_path}")
    meta, records = load_audit(str(audit_path))
    print(f"  → {len(records)} records")

    # ---- compute ----
    print("[compute] metrics...")
    overall     = compute_overall(meta, records)
    by_intent   = compute_by_intent(records)
    cm          = compute_confusion(records)
    by_source   = compute_by_source(records)
    frame_stats = compute_frame_stats(records)

    # ---- figures ----
    has_figures = HAS_MPL and HAS_NP
    if not HAS_MPL:
        print("[warn] matplotlib không có — bỏ qua vẽ chart. Cài: pip install matplotlib")
    if not HAS_SNS:
        print("[warn] seaborn không có — confusion matrix dùng fallback. Cài: pip install seaborn")

    if has_figures:
        print("[figures] đang vẽ...")
        if HAS_SNS:
            sns.set_theme(style="whitegrid", font_scale=1.0)

        fig_accuracy_by_intent(by_intent, fig_dir / "01_accuracy_by_intent.png")
        fig_confusion_matrix(cm, fig_dir / "02_confusion_matrix.png")
        fig_latency_distribution(records, fig_dir / "03_latency_distribution.png")
        fig_prediction_source(by_source, fig_dir / "04_prediction_source.png")
        fig_frame_scores_by_intent(records, fig_dir / "05_frame_score_by_intent.png")

    # ---- summary.json ----
    summary = {
        "overall":     overall,
        "by_intent":   by_intent,
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
        "by_source":   by_source,
        "frame_stats": frame_stats,
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  [summary] {summary_path}")

    # ---- report.md ----
    print("[report] generating...")
    write_report(out_dir, overall, by_intent, cm, by_source, frame_stats, has_figures)

    # ---- print summary to stdout ----
    print()
    print("=" * 50)
    print(f"  Accuracy : {_pct(overall['accuracy'])}  ({overall['correct']}/{overall['graded_samples']})")
    if overall.get("latency"):
        lat = overall["latency"]
        print(f"  Latency  : mean={lat['mean']}s  p95={lat['p95']}s  max={lat['max']}s")
    print(f"  Over limit: {overall['over_time_limit']} mẫu ({_pct(overall['over_limit_rate'])})")
    print("=" * 50)
    print(f"\n→ Xem báo cáo: {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
