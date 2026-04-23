#!/usr/bin/env python3
"""
analyze_audit.py — Phân tích cơ bản từ audit.json của Road-Buddy.

Tạo ra:
  <output_dir>/
    figures/
      01_accuracy_by_intent.png
      02_confusion_matrix.png
    summary.json
    report.md

Usage:
    python scripts/analyze_audit.py --audit outputs/11231/audit.json --output outputs/11231/analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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


LABELS = ["A", "B", "C", "D"]
FIG_DPI = 150

INTENT_LABEL = {
    "temporal":       "Temporal (trước/sau)",
    "value":          "Value (bao nhiêu)",
    "direction":      "Direction (rẽ/hướng)",
    "identification": "Identification (biển gì)",
    "existence":      "Existence (có/không)",
    "unknown":        "Unknown",
}


# ── helpers ──────────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    idx = p / 100 * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] * (1 - (idx - lo)) + s[hi] * (idx - lo)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return f"{head}\n{sep}\n{body}"


# ── compute ───────────────────────────────────────────────────────────────────

def compute_all(meta: dict, records: list[dict]) -> dict:
    graded  = [r for r in records if r.get("is_correct") is not None]
    correct = sum(1 for r in graded if r["is_correct"])
    lats    = [r["latency_sec"] for r in records if r.get("latency_sec", 0) > 0]

    # accuracy by intent
    intent_buckets: dict[str, list] = defaultdict(list)
    for r in graded:
        intent_buckets[r.get("query_intent") or "unknown"].append(r["is_correct"])
    by_intent = {
        k: {"total": len(v), "correct": sum(v), "accuracy": sum(v) / len(v) if v else 0.0}
        for k, v in sorted(intent_buckets.items())
    }

    # confusion matrix
    cm = [[0] * 4 for _ in range(4)]
    for r in graded:
        gt   = (r.get("gt_label") or "").strip().upper()
        pred = (r.get("answer")   or "").strip().upper()
        if gt in LABELS and pred in LABELS:
            cm[LABELS.index(gt)][LABELS.index(pred)] += 1

    return {
        "overall": {
            "model":          meta.get("model_name_or_path", "?"),
            "total":          len(records),
            "graded":         len(graded),
            "correct":        correct,
            "accuracy":       round(correct / len(graded), 6) if graded else 0.0,
            "over_limit":     sum(1 for r in records if r.get("latency_sec", 0) > 30),
            "interrupted":    meta.get("interrupted", False),
            "latency": {
                "mean":   round(sum(lats) / len(lats), 3) if lats else 0,
                "median": round(_percentile(lats, 50), 3),
                "p95":    round(_percentile(lats, 95), 3),
                "max":    round(max(lats), 3) if lats else 0,
            },
        },
        "by_intent": by_intent,
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
    }


# ── figures ───────────────────────────────────────────────────────────────────

def fig_accuracy_by_intent(by_intent: dict, out: Path) -> None:
    if not HAS_MPL:
        return
    intents = list(by_intent)
    accs    = [by_intent[i]["accuracy"] * 100 for i in intents]
    totals  = [by_intent[i]["total"] for i in intents]
    labels  = [INTENT_LABEL.get(i, i) for i in intents]

    fig, ax = plt.subplots(figsize=(8, max(3, len(intents) * 0.75 + 1)))
    bars = ax.barh(labels, accs, color="#2196F3", edgecolor="white", height=0.55)
    ax.axvline(25, color="#9E9E9E", linestyle="--", linewidth=1, label="Random (25%)")
    for bar, acc, n in zip(bars, accs, totals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%  (n={n})", va="center", fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Accuracy theo loại câu hỏi", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def fig_confusion_matrix(cm: list[list[int]], out: Path) -> None:
    if not HAS_MPL or not HAS_NP:
        return
    arr      = np.array(cm, dtype=float)
    row_sums = arr.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        norm = np.where(row_sums > 0, arr / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(5, 4))
    if HAS_SNS:
        sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS,
                    ax=ax, vmin=0, vmax=1, linewidths=0.5)
    else:
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center", fontsize=10)
        ax.set_xticks(range(4)); ax.set_xticklabels(LABELS)
        ax.set_yticks(range(4)); ax.set_yticklabels(LABELS)
        fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("Confusion Matrix (tỷ lệ theo hàng)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


# ── report ────────────────────────────────────────────────────────────────────

def write_report(out_dir: Path, stats: dict, has_figs: bool) -> None:
    ov  = stats["overall"]
    lat = ov["latency"]
    lines: list[str] = [
        "# Road-Buddy — Kết quả Inference",
        "",
        f"> Model: `{ov['model']}`",
        "",
        "## 1. Tổng quan",
        "",
        "| Chỉ số | Giá trị |",
        "|--------|---------|",
        f"| Tổng mẫu | {ov['total']} |",
        f"| Mẫu có ground truth | {ov['graded']} |",
        f"| **Accuracy** | **{_pct(ov['accuracy'])}** ({ov['correct']}/{ov['graded']}) |",
        f"| Latency trung bình | {lat['mean']}s |",
        f"| Latency p95 | {lat['p95']}s |",
        f"| Latency max | {lat['max']}s |",
        f"| Vượt time limit (>30s) | {ov['over_limit']} mẫu |",
        "",
        "## 2. Accuracy theo loại câu hỏi",
        "",
    ]

    if has_figs:
        lines += ["![Accuracy by Intent](figures/01_accuracy_by_intent.png)", ""]

    rows = []
    for intent, s in stats["by_intent"].items():
        rows.append([INTENT_LABEL.get(intent, intent), str(s["total"]),
                     str(s["correct"]), _pct(s["accuracy"])])
    lines += [_table(["Intent", "Tổng", "Đúng", "Accuracy"], rows), ""]

    lines += ["## 3. Confusion Matrix", ""]
    if has_figs:
        lines += ["![Confusion Matrix](figures/02_confusion_matrix.png)", ""]

    cm = stats["confusion_matrix"]["matrix"]
    cm_rows = [[LABELS[i]] + [str(cm[i][j]) for j in range(4)] for i in range(4)]
    lines += [_table(["GT \\ Pred"] + LABELS, cm_rows), ""]

    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [report] {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"[ERROR] Không tìm thấy: {audit_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output)
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {audit_path}")
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    meta, records = payload.get("meta", {}), payload.get("records", [])
    print(f"  → {len(records)} records")

    stats = compute_all(meta, records)
    has_figs = HAS_MPL and HAS_NP

    if not HAS_MPL:
        print("[warn] matplotlib không có → bỏ qua chart. pip install matplotlib")

    if has_figs:
        if HAS_SNS:
            sns.set_theme(style="whitegrid", font_scale=1.0)
        fig_accuracy_by_intent(stats["by_intent"], fig_dir / "01_accuracy_by_intent.png")
        fig_confusion_matrix(stats["confusion_matrix"]["matrix"], fig_dir / "02_confusion_matrix.png")

    (out_dir / "summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  [summary] {out_dir / 'summary.json'}")

    write_report(out_dir, stats, has_figs)

    ov = stats["overall"]
    print(f"\n{'='*45}")
    print(f"  Accuracy : {_pct(ov['accuracy'])}  ({ov['correct']}/{ov['graded']})")
    print(f"  Latency  : mean={ov['latency']['mean']}s  p95={ov['latency']['p95']}s")
    print(f"{'='*45}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
