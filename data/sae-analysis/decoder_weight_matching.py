#!/usr/bin/env python3
"""
Decoder-weight cosine similarity: Llama-3.2-3B vs Mistral-7B.

Llama has d_model=3072, Mistral has d_model=4096. We align the two spaces
using Procrustes analysis (SVD of the cross-covariance of matched decoder
vectors), anchored on the 12k activation-based pairs in llama_mistral_matches.json.

Output: data/sae-analysis/llama-mistral-matching.json
"""
import json, time
import numpy as np
from pathlib import Path

WORKSPACE    = Path("/Users/melton/ResearchPapers")
LLAMA_CKPT   = WORKSPACE / "data/sae-runs/llama-3b-layer14/checkpoint_step_010000.npz"
MISTRAL_CKPT = WORKSPACE / "data/sae-runs/mistral-7b-layer16/checkpoint_final.npz"
ANCHOR_FILE  = WORKSPACE / "data/sae-analysis/llama_mistral_matches.json"
OUT_PATH     = WORKSPACE / "data/sae-analysis/llama-mistral-matching.json"

TOP_N = 1000
BATCH = 1024   # Llama features per matmul; peak working memory ~64 MB per batch


def l2_norm(W: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    return W / np.maximum(norms, 1e-12)


def main():
    t0 = time.time()

    # 1. Load and L2-normalise decoder matrices
    print("Loading decoder weights...", flush=True)
    W_llama   = l2_norm(np.load(str(LLAMA_CKPT))  ["W_dec"].astype(np.float32))
    W_mistral = l2_norm(np.load(str(MISTRAL_CKPT)) ["W_dec"].astype(np.float32))
    n_llama,   d_llama   = W_llama.shape
    n_mistral, d_mistral = W_mistral.shape
    print(f"  Llama:   {W_llama.shape}  (n={n_llama}, d={d_llama})", flush=True)
    print(f"  Mistral: {W_mistral.shape}  (n={n_mistral}, d={d_mistral})", flush=True)

    # 2. Load activation-based anchor pairs and compute Procrustes alignment
    #    The alignment maps Llama's R^3072 → Mistral's R^4096.
    print("Computing Procrustes alignment from activation anchors...", flush=True)
    with open(ANCHOR_FILE) as f:
        anchor_data = json.load(f)
    anchors = anchor_data["matches"]

    li_idx = np.array([m["feat_a"] for m in anchors], dtype=np.int32)
    mi_idx = np.array([m["feat_b"] for m in anchors], dtype=np.int32)
    A = W_llama  [li_idx].astype(np.float64)   # (n_anchors, 3072) row-normed
    B = W_mistral[mi_idx].astype(np.float64)   # (n_anchors, 4096) row-normed

    M = A.T @ B                                # (3072, 4096) cross-covariance
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    # U: (3072, 3072)  Vt: (3072, 4096)
    R = (U @ Vt).astype(np.float32)           # (3072, 4096) alignment matrix

    print(f"  {len(anchors)} anchors (act-pattern threshold={anchor_data['threshold']})", flush=True)
    print(f"  Alignment R: {R.shape}  "
          f"top singular values: {np.round(S[:6], 3)}", flush=True)

    # 3. Project all Llama decoder vectors into Mistral's space
    print("Projecting Llama W_dec into Mistral space...", flush=True)
    W_llama_a = l2_norm(W_llama @ R)           # (16384, 4096)
    print(f"  Aligned Llama W_dec: {W_llama_a.shape}  ({time.time()-t0:.1f}s)", flush=True)

    # 4. Batched cosine similarity — for each Llama feature find nearest Mistral
    print(f"Finding nearest-Mistral-neighbor for each Llama feature "
          f"({n_llama}×{n_mistral})...", flush=True)
    best_j = np.zeros(n_llama, dtype=np.int32)
    best_s = np.zeros(n_llama, dtype=np.float32)

    n_batches = (n_llama + BATCH - 1) // BATCH
    for bi, i in enumerate(range(0, n_llama, BATCH)):
        batch = W_llama_a[i : i + BATCH]       # (B, 4096)
        sims  = batch @ W_mistral.T             # (B, n_mistral)
        idx   = np.argmax(sims, axis=1)
        vals  = sims[np.arange(len(batch)), idx]
        best_j[i : i + BATCH] = idx
        best_s[i : i + BATCH] = vals
        if bi % 4 == 0 or bi == n_batches - 1:
            print(f"  batch {bi+1}/{n_batches}  "
                  f"({i + len(batch)}/{n_llama} features)  "
                  f"{time.time()-t0:.1f}s", flush=True)

    # 5. Top-1000 pairs sorted by cosine similarity descending
    order = np.argsort(-best_s)
    top_matches = [
        {
            "llama_feat":   int(order[k]),
            "mistral_feat": int(best_j[order[k]]),
            "cosim":        round(float(best_s[order[k]]), 5),
        }
        for k in range(min(TOP_N, n_llama))
    ]

    # 6. Full similarity distribution (per-feature nearest-neighbor cosim)
    hist, edges = np.histogram(best_s, bins=50, range=(-1.0, 1.0))
    distribution = {
        "note":      "per-llama-feature nearest-mistral-neighbor cosine similarity",
        "bin_edges": [round(float(e), 4) for e in edges],
        "counts":    [int(c) for c in hist],
        "mean":      round(float(np.mean(best_s)),          5),
        "median":    round(float(np.median(best_s)),        5),
        "p90":       round(float(np.percentile(best_s, 90)), 5),
        "p95":       round(float(np.percentile(best_s, 95)), 5),
        "p99":       round(float(np.percentile(best_s, 99)), 5),
        "max":       round(float(np.max(best_s)),            5),
    }

    # 7. Save
    output = {
        "method":             "decoder_weight_cosine_similarity",
        "alignment":          "procrustes_svd",
        "alignment_anchors":  "data/sae-analysis/llama_mistral_matches.json",
        "n_anchor_pairs":     len(anchors),
        "anchor_threshold":   anchor_data["threshold"],
        "llama_checkpoint":   str(LLAMA_CKPT),
        "mistral_checkpoint": str(MISTRAL_CKPT),
        "n_llama_features":   n_llama,
        "n_mistral_features": n_mistral,
        "d_llama":            d_llama,
        "d_mistral":          d_mistral,
        "elapsed_seconds":    round(time.time() - t0, 1),
        "top_1000_matches":   top_matches,
        "similarity_distribution": distribution,
        "per_feature_max_cosim": [round(float(s), 5) for s in best_s],
    }

    OUT_PATH.write_text(json.dumps(output, indent=2))
    elapsed = time.time() - t0
    print(f"\nSaved → {OUT_PATH}  ({elapsed:.1f}s total)", flush=True)
    print(f"Top match: Llama[{top_matches[0]['llama_feat']}] "
          f"↔ Mistral[{top_matches[0]['mistral_feat']}]  "
          f"cosim={top_matches[0]['cosim']}", flush=True)
    print(f"Distribution: mean={distribution['mean']}  "
          f"p95={distribution['p95']}  max={distribution['max']}", flush=True)


if __name__ == "__main__":
    main()
