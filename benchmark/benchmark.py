# benchmark.py
# ------------------------------------------------------------
# Local CPU benchmark for 3 small SST-2 sentiment models.
# - Measures latency (avg/min/max/median/p95)
# - Simple accuracy on a tiny labeled set
# - Prints a Markdown table + per-sample predictions
#
# Run:
#   python benchmark.py
#
# Optional: save output to a file:
#   python benchmark.py > benchmark_results.md
# ------------------------------------------------------------

import time
from statistics import mean, median
from typing import List, Tuple, Dict, Any

from transformers import pipeline

# ---------- Config ----------

# Three small, CPU-friendly SST-2 models (binary sentiment)
MODELS: List[Tuple[str, str]] = [
    ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT SST-2 (HF)"),
    ("textattack/distilbert-base-uncased-SST-2", "TextAttack DistilBERT SST-2"),
    ("textattack/albert-base-v2-SST-2", "TextAttack ALBERT v2 SST-2"),
]

# Tiny labeled set for a quick, simple accuracy estimate.
# Labels use {"positive","negative"} for consistency.
SAMPLES: List[Tuple[str, str]] = [
    ("I absolutely love this!", "positive"),
    ("This is the best thing ever.", "positive"),
    ("Not bad at all, pretty good.", "positive"),
    ("I hated the experience.", "negative"),
    ("This is terrible and disappointing.", "negative"),
    ("I wouldn't recommend it.", "negative"),
    ("It's okay, nothing special.", "negative"),  # treat mild as negative for simplicity
    ("I am pleasantly surprised.", "positive"),
]

# For more stable latency numbers, do multiple runs per sample
RUNS_PER_SAMPLE = 3  # increase if you want smoother numbers


# ---------- Helpers ----------

def normalize_label(raw_label: str) -> str:
    """
    Normalize model output labels to {'positive','negative'}.
    Handles common variants like 'POSITIVE'/'NEGATIVE' and 'LABEL_1'/'LABEL_0'.
    """
    if raw_label is None:
        return "unknown"
    t = raw_label.strip().lower()
    if t in {"positive", "pos", "label_1", "1"}:
        return "positive"
    if t in {"negative", "neg", "label_0", "0"}:
        return "negative"
    # Some models might return 'neutral' on other tasks (not SST-2) — map to negative here.
    if t in {"neutral"}:
        return "negative"
    return t


def p95(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(0.95 * (len(vals) - 1))
    return vals[idx]


def run_one(model_name: str) -> Dict[str, Any]:
    """
    Load one model pipeline on CPU and benchmark on the SAMPLES set.
    Returns metrics + per-sample predictions.
    """
    clf = pipeline("sentiment-analysis", model=model_name, device=-1)  # device=-1 => CPU

    # Warmup (fills caches, avoids first-call overhead in measurements)
    _ = clf("warmup")

    latencies_ms: List[float] = []
    correct = 0
    preds: List[Tuple[str, str, str, float]] = []  # (text, expected, predicted, confidence)

    for text, expected in SAMPLES:
        for _ in range(RUNS_PER_SAMPLE):
            t0 = time.perf_counter()
            out = clf(text)[0]  # {'label': 'POSITIVE'|'NEGATIVE', 'score': float}
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

        pred_label = normalize_label(out["label"])
        conf = float(out["score"])
        preds.append((text, expected, pred_label, conf))
        if pred_label == expected:
            correct += 1

    n = len(SAMPLES)
    metrics = {
        "avg_ms": mean(latencies_ms),
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "med_ms": median(latencies_ms),
        "p95_ms": p95(latencies_ms),
        "acc": correct / n,
        "preds": preds,
        "num_samples": n,
        "runs_per_sample": RUNS_PER_SAMPLE,
    }
    return metrics


def main():
    all_results: List[Tuple[str, str, Dict[str, Any]]] = []

    for model_id, pretty in MODELS:
        print(f"\nLoading model: {model_id} …")
        try:
            res = run_one(model_id)
            all_results.append((model_id, pretty, res))
        except Exception as e:
            print(f"❌ Failed on {model_id}: {e}")
            # Keep going with the other models
            continue

    # Summary as Markdown
    print("\n## Mini-Benchmark (CPU)")
    print(f"*Samples:* {len(SAMPLES)}  ·  *Runs/sample:* {RUNS_PER_SAMPLE}\n")
    print("| Model | Avg ms | Median | p95 | Min | Max | Accuracy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model_id, pretty, r in all_results:
        print(f"| {pretty} | {r['avg_ms']:.1f} | {r['med_ms']:.1f} | {r['p95_ms']:.1f} | "
              f"{r['min_ms']:.1f} | {r['max_ms']:.1f} | {r['acc']*100:.0f}% |")

    # Per-sample predictions (handy for README appendix)
    print("\n<details><summary>Per-sample predictions</summary>\n")
    print("| Model | Text | Expected | Predicted | Confidence |")
    print("|---|---|---|---|---:|")
    for model_id, pretty, r in all_results:
        for text, expected, pred, conf in r["preds"]:
            safe_text = text.replace("|", "\\|")  # escape pipes for Markdown
            print(f"| {pretty} | {safe_text} | {expected} | {pred} | {conf:.4f} |")
    print("\n</details>")


if __name__ == "__main__":
    main()
