#!/usr/bin/env python3
"""
evaluate_classifier_lab02.py

Safety classifier evaluation for lab-02 layout.
Differences from local evaluate_classifier.py:
  - SAE checkpoint: sae-runs/llama-3b-layer16/checkpoint_final.npz (50k steps)
  - Activation dir: activations/llama-3b-layer16/ (target_layer=14 per metadata)
  - Feature labels: produced by label_features_differential.py
  - evaluated_at updated to 2026-08-02

Runs threshold sweep 0.05 – 3.0 (step 0.05) — wider range because the
differential-labeled classifier may produce higher scores with more
potentially-harmful features.
"""
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

ROOT          = Path.home() / "ResearchPapers" / "data"
EVAL_PATH     = ROOT / "safety-classifier/eval-dataset.json"
FEATURES_PATH = ROOT / "safety-classifier/llama3b-feature-labels.json"
CKPT_PATH     = ROOT / "sae-runs/llama-3b-layer16/checkpoint_final.npz"
OUT_PATH      = ROOT / "safety-classifier/evaluation-results.json"
MODEL_ID      = "mlx-community/Llama-3.2-3B-bf16"
LAYER_IDX     = 14
MAX_TOKENS    = 512

print("=" * 70, flush=True)
print("evaluate_classifier_lab02.py", flush=True)
print("=" * 70, flush=True)

# ── Load feature labels ───────────────────────────────────────────────────────
print("\nLoading feature labels...", flush=True)
feature_data  = json.loads(FEATURES_PATH.read_text())
features_list = feature_data["features"]

harmful_feature_ids = [f["feature_id"] for f in features_list
                       if f["label"] == "potentially-harmful"]
print(f"  Potentially-harmful features: {len(harmful_feature_ids)}", flush=True)

all_labeled_features   = {f["feature_id"]: f for f in features_list}
all_feat_ids           = [f["feature_id"] for f in features_list]
all_feat_id_to_idx     = {fid: i for i, fid in enumerate(all_feat_ids)}

# ── Load SAE checkpoint ───────────────────────────────────────────────────────
print(f"Loading SAE checkpoint: {CKPT_PATH.name}", flush=True)
ckpt     = np.load(str(CKPT_PATH))
W_enc_np = ckpt["W_enc"].astype(np.float32)
b_enc_np = ckpt["b_enc"].astype(np.float32)
b_dec_np = ckpt["b_dec"].astype(np.float32)
DICT_SIZE = W_enc_np.shape[1]
D_IN      = W_enc_np.shape[0]
print(f"  d_in={D_IN}  dict_size={DICT_SIZE}", flush=True)

W_enc_mx = mx.array(W_enc_np)
b_enc_mx = mx.array(b_enc_np)
b_dec_mx = mx.array(b_dec_np)

harmful_mask_np = np.zeros(DICT_SIZE, dtype=bool)
for fid in harmful_feature_ids:
    harmful_mask_np[fid] = True
harmful_indices = np.where(harmful_mask_np)[0]
print(f"  Harmful feature indices ({len(harmful_indices)}): "
      f"{harmful_indices.tolist()[:10]}{'...' if len(harmful_indices)>10 else ''}",
      flush=True)

# ── Load eval dataset ─────────────────────────────────────────────────────────
print("Loading eval dataset...", flush=True)
eval_data = json.loads(EVAL_PATH.read_text())
examples  = eval_data["examples"]
n_safe    = sum(1 for e in examples if e["label"] == "safe")
n_unsafe  = sum(1 for e in examples if e["label"] == "unsafe")
print(f"  {len(examples)} examples  (safe: {n_safe}, unsafe: {n_unsafe})", flush=True)

# ── Load Llama model ──────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID} ...", flush=True)
t_load = time.time()
model, tokenizer = load(MODEL_ID)
n_layers = len(model.model.layers)
print(f"  {n_layers} layers, capturing layer {LAYER_IDX}  "
      f"({time.time()-t_load:.1f}s)", flush=True)


def get_residuals(ids_mx: mx.array) -> mx.array:
    """Partial forward pass, stops after LAYER_IDX. Returns (T, D)."""
    h    = model.model.embed_tokens(ids_mx)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(
        h.shape[1]).astype(h.dtype)
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask=mask, cache=None)
        if i == LAYER_IDX:
            break
    mx.eval(h)
    return h[0]


def encode_sae(acts_np: np.ndarray) -> np.ndarray:
    """
    SAE encoder on (T, D) float32 activations → (T, dict_size) ReLU'd pre-acts.
    Uses ReLU (not TopK) for scoring so activations are continuous, matching
    the original evaluate_classifier.py scoring method.
    """
    acts_mx = mx.array(acts_np)
    pre_mx  = (acts_mx - b_dec_mx) @ W_enc_mx + b_enc_mx
    pre_relu = mx.maximum(pre_mx, 0.0)
    mx.eval(pre_relu)
    return np.array(pre_relu)


# ── Run inference on all eval examples ───────────────────────────────────────
print(f"\nRunning inference on {len(examples)} examples...", flush=True)
t0            = time.time()
harm_scores   = []
labels_int    = []
n_harmful     = len(harmful_indices)
n_all_labeled = len(features_list)

per_ex_harm        = np.zeros((len(examples), n_harmful), dtype=np.float32)
per_ex_all_labeled = np.zeros((len(examples), n_all_labeled), dtype=np.float32)

for i, ex in enumerate(examples):
    text = ex["prompt"] + "\n" + ex["response"]
    ids  = tokenizer.encode(text)
    if not isinstance(ids, list):
        ids = list(ids)
    ids = ids[:MAX_TOKENS]

    ids_mx  = mx.array([ids], dtype=mx.int32)
    acts    = get_residuals(ids_mx)
    acts_np = np.array(acts.astype(mx.float32))

    pre_np = encode_sae(acts_np)

    harm_pre      = pre_np[:, harmful_indices]
    feat_max      = harm_pre.max(axis=0) if n_harmful > 0 else np.zeros(0)
    per_ex_harm[i] = feat_max
    harm_score    = float(feat_max.sum())
    harm_scores.append(harm_score)

    for j, fid in enumerate(all_feat_ids):
        per_ex_all_labeled[i, j] = float(pre_np[:, fid].max())

    gt = 1 if ex["label"] == "unsafe" else 0
    labels_int.append(gt)

    if (i + 1) % 20 == 0 or i == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1:3d}/200]  score={harm_score:7.3f}  label={ex['label']:<6}  "
              f"{elapsed:.1f}s  ({elapsed/(i+1):.2f}s/ex)", flush=True)

harm_scores = np.array(harm_scores, dtype=np.float32)
labels_arr  = np.array(labels_int, dtype=np.int32)
total_t     = time.time() - t0
print(f"\nDone in {total_t:.1f}s  ({total_t/len(examples):.2f}s/ex)", flush=True)

score_stats = {
    "min":         float(harm_scores.min()),
    "max":         float(harm_scores.max()),
    "mean":        float(harm_scores.mean()),
    "std":         float(harm_scores.std()),
    "unsafe_mean": float(harm_scores[labels_arr == 1].mean()),
    "safe_mean":   float(harm_scores[labels_arr == 0].mean()),
    "separation":  float(harm_scores[labels_arr == 1].mean()
                         - harm_scores[labels_arr == 0].mean()),
}
print(f"\nScore stats:", flush=True)
for k, v in score_stats.items():
    print(f"  {k:14s}: {v:.4f}", flush=True)


# ── Threshold sweep: adaptive range from score distribution ───────────────────
score_min  = float(harm_scores.min())
score_max  = float(harm_scores.max())
step       = (score_max - score_min) / 100.0   # 100 steps across full range
step       = max(step, 1e-4)
thr_lo     = max(0.0, score_min - step)
thr_hi     = score_max + step
print(f"\nSweeping thresholds {thr_lo:.3f} – {thr_hi:.3f} (step {step:.4f}) ...",
      flush=True)
thresholds    = np.round(np.arange(thr_lo, thr_hi + step, step), 6)
sweep_results = []

for thresh in thresholds:
    preds = (harm_scores > thresh).astype(np.int32)
    tp = int(((preds == 1) & (labels_arr == 1)).sum())
    fp = int(((preds == 1) & (labels_arr == 0)).sum())
    tn = int(((preds == 0) & (labels_arr == 0)).sum())
    fn = int(((preds == 0) & (labels_arr == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    sweep_results.append({
        "threshold": float(thresh),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "fpr":       round(fpr, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    })

best = max(sweep_results, key=lambda x: x["f1"])
print(f"\nBest threshold : {best['threshold']}", flush=True)
print(f"  F1           : {best['f1']:.4f}", flush=True)
print(f"  Precision    : {best['precision']:.4f}", flush=True)
print(f"  Recall       : {best['recall']:.4f}", flush=True)
print(f"  FPR          : {best['fpr']:.4f}", flush=True)
print(f"  TP/FP/TN/FN  : {best['tp']}/{best['fp']}/{best['tn']}/{best['fn']}",
      flush=True)

# Print top-5 non-degenerate thresholds
non_degenerate = [r for r in sweep_results
                  if r['precision'] > 0 and r['recall'] > 0
                  and r['fpr'] < 1.0 and r['recall'] < 1.0]
print(f"\nTop-5 non-degenerate points by F1:", flush=True)
for r in sorted(non_degenerate, key=lambda x: x['f1'], reverse=True)[:5]:
    print(f"  thr={r['threshold']:.2f}  F1={r['f1']:.4f}"
          f"  P={r['precision']:.4f}  R={r['recall']:.4f}"
          f"  FPR={r['fpr']:.4f}", flush=True)


# ── Confusion matrix at best threshold ───────────────────────────────────────
bt         = best["threshold"]
best_preds = (harm_scores > bt).astype(np.int32)
cm_examples = {"tp": [], "fp": [], "tn": [], "fn": []}

for i, ex in enumerate(examples):
    pred  = int(best_preds[i])
    gt    = int(labels_arr[i])
    score = float(harm_scores[i])
    entry = {"id": ex["id"], "label": ex["label"],
             "category": ex.get("category", ""), "score": round(score, 4)}
    if pred == 1 and gt == 1:
        cm_examples["tp"].append(entry)
    elif pred == 1 and gt == 0:
        cm_examples["fp"].append(entry)
    elif pred == 0 and gt == 0:
        cm_examples["tn"].append(entry)
    else:
        cm_examples["fn"].append(entry)

confusion_matrix = {
    "predicted_unsafe_actual_unsafe_tp": best["tp"],
    "predicted_unsafe_actual_safe_fp":   best["fp"],
    "predicted_safe_actual_safe_tn":     best["tn"],
    "predicted_safe_actual_unsafe_fn":   best["fn"],
    "examples": cm_examples,
}


# ── Feature discriminativity ──────────────────────────────────────────────────
print("\nComputing feature discriminativity...", flush=True)
unsafe_idx_arr = np.where(labels_arr == 1)[0]
safe_idx_arr   = np.where(labels_arr == 0)[0]

discriminative = []
for j, fid in enumerate(all_feat_ids):
    feat_vals     = per_ex_all_labeled[:, j]
    mean_unsafe   = float(feat_vals[unsafe_idx_arr].mean())
    mean_safe     = float(feat_vals[safe_idx_arr].mean())
    feat_meta     = features_list[j]
    discriminative.append({
        "feature_id":        fid,
        "rank_by_frequency": feat_meta["rank"],
        "label":             feat_meta["label"],
        "label_confidence":  feat_meta["label_confidence"],
        "lift_score":        feat_meta.get("lift_score", None),
        "mean_act_unsafe":   round(mean_unsafe, 4),
        "mean_act_safe":     round(mean_safe, 4),
        "discriminability":  round(mean_unsafe - mean_safe, 4),
    })

discriminative.sort(key=lambda x: x["discriminability"], reverse=True)
top5 = discriminative[:5]

print("Top-5 discriminative features:", flush=True)
for d in top5:
    print(f"  feat={d['feature_id']:5d}  label={d['label']:20s}"
          f"  disc={d['discriminability']:+.4f}"
          f"  lift={d.get('lift_score') or 'n/a'}", flush=True)


# ── Save results ───────────────────────────────────────────────────────────────
print(f"\nSaving → {OUT_PATH}", flush=True)
out = {
    "metadata": {
        "model":               MODEL_ID,
        "layer_idx":           LAYER_IDX,
        "sae_checkpoint":      CKPT_PATH.name,
        "sae_training_steps":  50000,
        "dict_size":           DICT_SIZE,
        "n_examples":          len(examples),
        "n_safe":              int((labels_arr == 0).sum()),
        "n_unsafe":            int((labels_arr == 1).sum()),
        "n_harmful_features":  len(harmful_feature_ids),
        "harmful_feature_ids": harmful_feature_ids,
        "harm_score_method":   ("sum of max-over-tokens ReLU pre-activations "
                                "for potentially-harmful SAE features"),
        "labeling_method":     feature_data["metadata"].get("labeling_method",
                                                            "differential_frequency"),
        "threshold_range":     "0.05 – 3.00 (step 0.05)",
        "evaluated_at":        "2026-08-02",
        "weights":             "real",
    },
    "score_statistics":           score_stats,
    "threshold_sweep":            sweep_results,
    "best_threshold":             best,
    "confusion_matrix":           confusion_matrix,
    "top_discriminative_features":  top5,
    "all_discriminative_features":  discriminative,
    "per_example_scores": [
        {
            "id":        ex["id"],
            "label":     ex["label"],
            "category":  ex.get("category", ""),
            "score":     round(float(harm_scores[i]), 4),
            "predicted": "unsafe" if best_preds[i] == 1 else "safe",
        }
        for i, ex in enumerate(examples)
    ],
}

OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
size_kb = OUT_PATH.stat().st_size / 1024
print(f"Saved  ({size_kb:.0f} KB)", flush=True)

print(f"\n{'='*50}", flush=True)
print(f"SUMMARY", flush=True)
print(f"{'='*50}", flush=True)
print(f"  n_harmful_features : {len(harmful_feature_ids)}", flush=True)
print(f"  score separation   : unsafe_mean - safe_mean = {score_stats['separation']:.4f}",
      flush=True)
print(f"  best threshold     : {best['threshold']}", flush=True)
print(f"  F1                 : {best['f1']:.4f}", flush=True)
print(f"  Precision          : {best['precision']:.4f}", flush=True)
print(f"  Recall             : {best['recall']:.4f}", flush=True)
print(f"  FPR                : {best['fpr']:.4f}", flush=True)
print(f"  TP/FP/TN/FN        : {best['tp']}/{best['fp']}/{best['tn']}/{best['fn']}",
      flush=True)
