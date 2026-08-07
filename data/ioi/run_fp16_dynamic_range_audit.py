"""
Float16 dynamic range audit — Llama-3.2-3B on the IOI dataset.

For each transformer layer (0-27), captures the post-layer residual stream
over all IOI examples and logs:
  p1, p25, p50, p75, p99, max_abs, n_fp16_overflow (|x| > 65504)

Checks CORRECTIONS.md item #2: whether the float16 losslessness precondition
holds at the split boundary (layer 14, the SAE-adjacent layer).

Output: data/ioi/fp16_dynamic_range_audit.json
"""
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

FP16_MAX = 65504.0
MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"
DATA_DIR = Path(__file__).parent
OUT_FILE = DATA_DIR / "fp16_dynamic_range_audit.json"

SAE_LAYER = 14          # where the Llama SAE sits
SPLIT_BOUNDARY = 14     # same — the distributed-inference split point


def causal_mask(seq_len: int, dtype) -> mx.array:
    return nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(dtype)


def quantiles_and_overflow(arr: np.ndarray) -> dict:
    """Compute distribution stats and float16 overflow count for a flat array."""
    abs_arr = np.abs(arr)
    n_overflow = int(np.sum(abs_arr > FP16_MAX))
    n_total = arr.size
    return {
        "p1":    float(np.percentile(abs_arr, 1)),
        "p25":   float(np.percentile(abs_arr, 25)),
        "p50":   float(np.percentile(abs_arr, 50)),
        "p75":   float(np.percentile(abs_arr, 75)),
        "p99":   float(np.percentile(abs_arr, 99)),
        "max_abs":         float(abs_arr.max()),
        "mean_abs":        float(abs_arr.mean()),
        "n_fp16_overflow": n_overflow,
        "n_total":         n_total,
        "overflow_frac":   n_overflow / n_total if n_total > 0 else 0.0,
    }


def run_full_layerwise(model, input_ids: mx.array) -> list[np.ndarray]:
    """
    Full forward pass through all layers; returns list of residual streams
    (one numpy array per layer), each shape (seq_len, hidden_dim).
    Processes a single sequence (no batch dim).
    """
    h = model.model.embed_tokens(input_ids)          # (1, T, D)
    mask = causal_mask(h.shape[1], h.dtype)
    snapshots = []
    for layer in model.model.layers:
        h = layer(h, mask=mask, cache=None)
        mx.eval(h)
        # Squeeze batch dim; cast to float32 first (bf16 has no numpy dtype)
        h32 = h[0].astype(mx.float32)
        mx.eval(h32)
        snapshots.append(np.array(h32))
    return snapshots


def main():
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  loaded in {time.time()-t0:.1f}s")

    dataset = json.loads((DATA_DIR / "dataset.json").read_text())
    examples = dataset["examples"]
    print(f"  {len(examples)} IOI examples")

    n_layers = len(model.model.layers)
    print(f"  {n_layers} transformer layers  (SAE/split at layer {SAE_LAYER})")

    # Accumulate per-layer flat arrays (all positions, all examples)
    # To avoid OOM, compute running stats via reservoir: collect max_abs per example
    # and do one full numpy accumulation in a list (28 layers × 100 examples × ~16 tokens
    # × 3072 dims ≈ 130M float32 values ≈ 500MB — manageable)
    layer_buckets: list[list[np.ndarray]] = [[] for _ in range(n_layers)]

    t_start = time.time()
    for ex_idx, ex in enumerate(examples):
        prompt = ex["prompt"]
        input_ids = mx.array(tokenizer.encode(prompt, add_special_tokens=True))[None]
        snapshots = run_full_layerwise(model, input_ids)
        for layer_idx, snap in enumerate(snapshots):
            layer_buckets[layer_idx].append(snap.ravel())
        if (ex_idx + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  processed {ex_idx+1}/{len(examples)} examples  ({elapsed:.1f}s)")

    print(f"\nComputing per-layer statistics...")
    layer_stats = []
    flagged_layers = []

    for layer_idx in range(n_layers):
        all_vals = np.concatenate(layer_buckets[layer_idx])
        stats = quantiles_and_overflow(all_vals)
        stats["layer"] = layer_idx
        stats["is_sae_layer"] = (layer_idx == SAE_LAYER)
        stats["is_split_boundary"] = (layer_idx == SPLIT_BOUNDARY)

        # Flag saturation risk: p99 > 10% of FP16_MAX or any overflow
        stats["saturation_risk"] = (
            stats["p99"] > 0.1 * FP16_MAX or stats["n_fp16_overflow"] > 0
        )
        layer_stats.append(stats)

        flag = ""
        if stats["n_fp16_overflow"] > 0:
            flag = " *** FP16 OVERFLOW ***"
            flagged_layers.append(layer_idx)
        elif stats["saturation_risk"]:
            flag = " (saturation risk)"

        sae_marker = " <-- SAE/split" if stats["is_sae_layer"] else ""
        print(
            f"  L{layer_idx:2d}  p99={stats['p99']:8.2f}  max={stats['max_abs']:10.2f}"
            f"  overflow={stats['n_fp16_overflow']:6d}/{stats['n_total']}{flag}{sae_marker}"
        )

    # Summary
    sae_stats = layer_stats[SAE_LAYER]
    summary = {
        "model": MODEL_ID,
        "n_examples": len(examples),
        "n_layers": n_layers,
        "sae_layer": SAE_LAYER,
        "fp16_max": FP16_MAX,
        "flagged_layers": flagged_layers,
        "sae_layer_stats": sae_stats,
        "all_layers": layer_stats,
        "verdict": (
            "OVERFLOW_DETECTED" if flagged_layers else
            "SATURATION_RISK" if any(s["saturation_risk"] for s in layer_stats) else
            "CLEAN"
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    OUT_FILE.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {OUT_FILE}")

    print(f"\n--- SUMMARY ---")
    print(f"Verdict: {summary['verdict']}")
    print(f"SAE layer {SAE_LAYER}: p99={sae_stats['p99']:.2f}, max_abs={sae_stats['max_abs']:.2f}, "
          f"overflow={sae_stats['n_fp16_overflow']}/{sae_stats['n_total']} ({sae_stats['overflow_frac']*100:.4f}%)")
    if flagged_layers:
        print(f"FP16 OVERFLOW at layers: {flagged_layers}")
    else:
        print("No FP16 overflow detected.")
        fp16_headroom = FP16_MAX / max(s["max_abs"] for s in layer_stats)
        print(f"Worst-case headroom: {fp16_headroom:.1f}x below FP16_MAX across all layers")


if __name__ == "__main__":
    main()
