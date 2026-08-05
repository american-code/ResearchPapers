#!/usr/bin/env python3
"""
Power analysis for the universality null in cross-architecture SAE feature matching.

Loads real Mistral-7B layer-16 SAE fingerprints (activations_50k.npy + checkpoint_final.npz)
and simulates the reciprocal-best-match detection method from cross_arch_matching_v2.py
with planted synthetic universal features at varying Pearson correlation strengths α.

Key algorithmic choice: for the null-threshold calibration and planted-pair detection,
only K×D inner products are computed per trial (not D×D), making the simulation fast:
  - For each planted pair (ai, bi), check that fp_a[ai]'s GLOBAL best match in fp_b is bi
    (and vice versa). This requires K query features × D database features, not D×D.

Outputs:
  data/power-analysis/power_table.json   — machine-readable
  data/power-analysis/power_table.txt    — human-readable table
"""
import json
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path("/Users/melton/ResearchPapers")
SAE_CKPT  = WORKSPACE / "data/sae-runs/mistral-7b-layer16/checkpoint_final.npz"
ACTS_50K  = WORKSPACE / "data/activations/mistral-7b-layer16/activations_50k.npy"
OUT_DIR   = WORKSPACE / "data/power-analysis"

D_IN       = 4096
CHUNK_TOKS = 500     # tokens per chunk (matches real analysis)
N_CHUNKS   = 100     # 50 000 / 500 = 100 chunks
TOPK       = 128
MIN_ACTIVE = 5       # minimum active chunks for a feature to be "valid"

# Power analysis settings
DICT_SIZES  = [256, 1024, 4096]   # test at multiple dictionary sizes
ALPHA_GRID  = np.round(np.arange(0.50, 1.01, 0.05), 2)
K_PLANTED   = 50     # synthetic universal pairs per trial
N_TRIALS    = 300    # Monte Carlo trials per (alpha, dict_size)

# Null calibration: sample-based (fast)
N_CALIB_QUERIES = 300   # random query features per null permutation
N_CALIB_PERMS   = 30    # permutations for calibration
NULL_PCT         = 99.0 # threshold percentile (matches real analysis)

SEED = 42


# ── Fingerprint computation ───────────────────────────────────────────────────

def load_sae(ckpt: Path) -> dict:
    z = np.load(str(ckpt))
    return {k: z[k].astype(np.float32) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}


def compute_fingerprints(acts_path: Path, sae: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Process activations chunk-by-chunk.
    Returns (fp_pearson [D, N_CHUNKS], n_active [D]).
    """
    # Load all 50k tokens at once (391MB float16 → ~800MB float32) for fast batch encoding
    acts_raw = np.load(str(acts_path))             # [50k, D_in] float16
    acts_f32 = acts_raw.astype(np.float32)
    del acts_raw
    D     = sae["W_enc"].shape[1]
    N_TOK = N_CHUNKS * CHUNK_TOKS                  # 50 000

    # Encode in batches of 10 chunks (5000 tokens) to cap the intermediate matrix size
    ENCODE_BATCH = CHUNK_TOKS * 10                 # 5000 tokens → [5000, D] = 330 MB
    fp   = np.zeros((N_CHUNKS, D), dtype=np.float32)
    chunk_off = 0
    for s in range(0, N_TOK, ENCODE_BATCH):
        e   = min(s + ENCODE_BATCH, N_TOK)
        blk = acts_f32[s:e]                        # [B, D_in]
        pre = (blk - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]   # [B, D]
        kth = np.partition(pre, -TOPK, axis=1)[:, -TOPK][:, None]
        enc = np.where(pre >= kth, pre, 0.0)       # [B, D] sparse
        n_chunks_this = (e - s) // CHUNK_TOKS
        fp[chunk_off : chunk_off + n_chunks_this] = (
            enc.reshape(n_chunks_this, CHUNK_TOKS, D).mean(axis=1)
        )
        chunk_off += n_chunks_this
    del acts_f32

    fp = fp.T.copy()  # [D, N_CHUNKS]
    n_active = (fp > 0).sum(axis=1).astype(np.int32)

    fp -= fp.mean(axis=1, keepdims=True)   # Pearson centering
    norms = np.linalg.norm(fp, axis=1, keepdims=True)
    fp /= np.maximum(norms, 1e-12)

    return fp.astype(np.float32), n_active


# ── Null calibration ─────────────────────────────────────────────────────────

def calibrate_tau(fp: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Estimate the null best-match threshold using random query sampling.

    For N_CALIB_QUERIES random query features in fp, find the best-match similarity
    against a chunk-permuted version of fp (destroys cross-model correspondence while
    preserving each feature's marginal distribution — same null as the real analysis).
    Returns (tau_p99, null_mean, null_p99) from the empirical null distribution.
    """
    D, N   = fp.shape
    n_q    = min(N_CALIB_QUERIES, D)
    q_idx  = rng.choice(D, n_q, replace=False)
    fa_q   = fp[q_idx].copy()   # [n_q, N] — contiguous for fast matmul
    null_b = []

    for _ in range(N_CALIB_PERMS):
        perm   = rng.permutation(N)
        fp_p   = fp[:, perm]          # chunk-shuffled, not a copy (view → copy for @ )
        sims   = fa_q @ fp_p.T        # [n_q, D]
        null_b.append(sims.max(axis=1))  # best match per query

    allv = np.concatenate(null_b)
    tau  = float(np.percentile(allv, NULL_PCT))
    return tau, float(allv.mean()), float(np.percentile(allv, 99))


# ── Signal planting (vectorised over K pairs) ────────────────────────────────

def plant_signals_batch(templates: np.ndarray, alpha: float,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised: return [K, N] fingerprints each with Pearson correlation α
    with the corresponding row of templates [K, N] (unit-norm, zero-mean).

    Steps per batch of K templates:
      1. Draw K×N noise
      2. Center each noise vector (keep in Pearson subspace)
      3. Project out each template component (Gram-Schmidt)
      4. Normalise noise vectors
      5. Blend: alpha * template + sqrt(1-alpha²) * noise
      6. Renormalise (float safety)
    """
    K, N   = templates.shape
    noise  = rng.standard_normal((K, N)).astype(np.float32)
    noise -= noise.mean(axis=1, keepdims=True)                        # step 2
    dots   = (noise * templates).sum(axis=1, keepdims=True)          # K×1
    noise -= dots * templates                                          # step 3
    norms  = np.linalg.norm(noise, axis=1, keepdims=True)
    norms  = np.maximum(norms, 1e-10)
    noise /= norms                                                     # step 4
    scale  = float(np.sqrt(max(0.0, 1.0 - alpha ** 2)))
    fps    = alpha * templates + scale * noise                         # step 5
    fps   /= np.maximum(np.linalg.norm(fps, axis=1, keepdims=True), 1e-10)
    return fps


# ── Detection (fast: only K×D, not D×D) ──────────────────────────────────────

def detection_rate(fp_a: np.ndarray, fp_b: np.ndarray,
                   planted_a: np.ndarray, planted_b: np.ndarray,
                   tau: float) -> float:
    """
    Fraction of planted pairs (ai, bi) recovered by reciprocal best match.

    For each planted query ai in A:
      - Compute fp_a[ai] · fp_b[j] for ALL j (checking global best match in B)
      - Compute fp_b[bi] · fp_a[j] for ALL j (checking global best match in A)
    Only K×D inner products, not D×D.
    """
    K  = len(planted_a)

    # A→B: planted A features vs. full fp_b
    sims_a   = fp_a[planted_a] @ fp_b.T          # [K, D]
    best_b   = sims_a.argmax(axis=1)              # [K] — argmax in B for each planted A
    sim_tgt  = sims_a[np.arange(K), planted_b]   # [K] — similarity at intended B target

    # B→A: planted B features vs. full fp_a
    sims_b   = fp_b[planted_b] @ fp_a.T          # [K, D]
    best_a   = sims_b.argmax(axis=1)              # [K] — argmax in A for each planted B

    # Reciprocal match + threshold
    mutual = (best_b == planted_b) & (best_a == planted_a) & (sim_tgt >= tau)
    return float(mutual.mean())


# ── Power simulation ──────────────────────────────────────────────────────────

def run_power(fp_sub: np.ndarray, tau: float,
              rng: np.random.Generator) -> list[dict]:
    """
    Power curve over ALPHA_GRID, for a given subsampled fingerprint dictionary fp_sub [D, N].
    """
    D, N   = fp_sub.shape
    results = []

    for alpha in ALPHA_GRID:
        rates = []
        for _ in range(N_TRIALS):
            # Null model-B: chunk-shuffle of fp_sub
            perm  = rng.permutation(N)
            fp_b  = fp_sub[:, perm].copy()

            # Plant K disjoint signal pairs (vectorised)
            k       = min(K_PLANTED, D // 2)
            chosen  = rng.choice(D, 2 * k, replace=False)
            p_a     = chosen[:k]
            p_b     = chosen[k:]

            templates   = fp_sub[p_a]            # [K, N]
            fp_b[p_b]   = plant_signals_batch(templates, float(alpha), rng)

            rates.append(detection_rate(fp_sub, fp_b, p_a, p_b, tau))

        mean_r = float(np.mean(rates))
        se_r   = float(np.std(rates) / np.sqrt(N_TRIALS))
        results.append({
            "alpha":    float(alpha),
            "mean":     round(mean_r, 4),
            "se":       round(se_r, 4),
            "power_80": mean_r >= 0.80,
        })
        print(f"    α={alpha:.2f}  power={mean_r:.3f} ± {se_r:.3f}", flush=True)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0  = time.time()
    rng = np.random.default_rng(SEED)

    # 1. Load real fingerprints
    print("Loading Mistral-7B SAE checkpoint …", flush=True)
    sae  = load_sae(SAE_CKPT)
    D_full = sae["W_enc"].shape[1]
    print(f"  D={D_full}  d_in={D_IN}")

    print(f"Computing fingerprints ({N_CHUNKS} chunks × {CHUNK_TOKS} tok) …", flush=True)
    fp_all, n_active = compute_fingerprints(ACTS_50K, sae)
    del sae

    dead        = int((n_active == 0).sum())
    low_support = int(((n_active > 0) & (n_active < MIN_ACTIVE)).sum())
    valid_mask  = n_active >= MIN_ACTIVE
    D_valid     = int(valid_mask.sum())
    fp_valid    = fp_all[valid_mask]
    del fp_all
    print(f"  dead={dead}  support<{MIN_ACTIVE}={low_support}  valid={D_valid}")

    # 2. Run for each dictionary size
    all_results: dict = {}
    for D_test in DICT_SIZES + [D_valid]:
        D_use   = min(D_test, D_valid)
        desc    = "D_valid" if D_use == D_valid else f"D={D_use}"
        print(f"\n=== {desc} ===", flush=True)

        # Subsample if needed
        if D_use < D_valid:
            idx    = rng.choice(D_valid, D_use, replace=False)
            fp_sub = fp_valid[idx].copy()
        else:
            fp_sub = fp_valid.copy()

        # Calibrate threshold on this dictionary size
        print(f"  Calibrating null threshold (sample-based, D={D_use}) …", flush=True)
        tau, null_mean, null_p99 = calibrate_tau(fp_sub, rng)
        print(f"  τ={tau:.4f}  null_mean={null_mean:.4f}", flush=True)

        # Power simulation
        print(f"  Power simulation: {len(ALPHA_GRID)} α levels × {N_TRIALS} trials …", flush=True)
        power_rows = run_power(fp_sub, tau, rng)

        above_80   = [r for r in power_rows if r["power_80"]]
        mdes       = float(above_80[0]["alpha"]) if above_80 else None

        all_results[str(D_use)] = {
            "D": D_use,
            "tau_calibrated":       round(tau, 4),
            "null_best_match_mean": round(null_mean, 4),
            "null_best_match_p99":  round(null_p99, 4),
            "power_by_alpha":       power_rows,
            "mdes_80pct":          mdes,
        }
        print(f"  MDES at 80% power: α={mdes}", flush=True)

    elapsed = round(time.time() - t0, 1)

    # 3. Save outputs
    output = {
        "description": (
            "Power analysis for cross-architecture SAE universality detection. "
            "Real Mistral-7B layer-16 SAE fingerprints (N_CHUNKS=100) form the null. "
            "Synthetic cross-model pairs with Pearson correlation α are planted. "
            "Detection uses the reciprocal-best-match method from cross_arch_matching_v2.py."
        ),
        "real_data_reference": {
            "source": "cross_arch_matching_v2.py (Pearson metric, N_CHUNKS=1000)",
            "observed_universal_features": 1,
            "triangle_null_mean_pearson": 0.15,
            "interpretation": "1 found ≈ chance → no reliable evidence for universality",
            "pairwise_null_means_pearson": {
                "llama_qwen":    0.5945,
                "mistral_qwen":  0.5943,
                "llama_mistral": 0.7436,
            },
            "pairwise_thresholds_pearson": {
                "llama_qwen":    0.9497,
                "mistral_qwen":  0.9435,
                "llama_mistral": 0.9803,
            },
        },
        "simulation": {
            "model":             "Mistral-7B-v0.3-4bit layer 16",
            "sae_checkpoint":    "data/sae-runs/mistral-7b-layer16/checkpoint_final.npz",
            "activations_file":  "data/activations/mistral-7b-layer16/activations_50k.npy",
            "n_tokens":          N_CHUNKS * CHUNK_TOKS,
            "N_chunks":          N_CHUNKS,
            "D_full":            D_full,
            "D_valid":           D_valid,
            "dead_features":     dead,
            "low_support":       low_support,
            "min_active_chunks": MIN_ACTIVE,
            "null_method":       "chunk-order permutation (same as real analysis)",
            "null_calib_queries": N_CALIB_QUERIES,
            "null_calib_perms":  N_CALIB_PERMS,
            "null_percentile":   NULL_PCT,
            "K_planted":         K_PLANTED,
            "n_trials":          N_TRIALS,
            "seed":              SEED,
            "conservatism_note": (
                f"N_CHUNKS=100 here vs 1000 in the full analysis. "
                f"Fewer chunks → noisier fingerprints → higher τ → MDES is an upper bound. "
                f"True MDES at N_CHUNKS=1000 (500k tokens) may be lower (easier detection)."
            ),
        },
        "results_by_dict_size": all_results,
        "elapsed_s": elapsed,
    }
    (OUT_DIR / "power_table.json").write_text(json.dumps(output, indent=2))

    # Human-readable table
    lines = [
        "Power Analysis: Minimum Detectable Effect Size for SAE Feature Universality",
        "=" * 75,
        "",
        "Context",
        "-------",
        "  Three-model cross-architecture SAE feature universality test.",
        "  Method: reciprocal best match, Pearson fingerprints, permutation-calibrated τ.",
        "  Real result (N_CHUNKS=1000, 500k tokens per model):",
        "    Closed-triangle universal features found: 1",
        "    Triangle null mean: 0.15  →  1 found ≈ chance",
        "",
        "Simulation Setup",
        "----------------",
        f"  Null: real Mistral-7B SAE fingerprints (D={D_full}, N_CHUNKS={N_CHUNKS})",
        f"  Valid features: {D_valid} / {D_full}  (support ≥ {MIN_ACTIVE} chunks)",
        f"  Null model: chunk-order permutation (same as real analysis calibration)",
        f"  {K_PLANTED} planted pairs × {N_TRIALS} trials × {len(ALPHA_GRID)} α levels per dict size",
        "",
        "  Note: N_CHUNKS=100 vs 1000 in the full analysis. MDES estimates are",
        "  conservative upper bounds (noisier fingerprints → harder detection).",
        "",
    ]

    for D_str, res in all_results.items():
        D_use = res["D"]
        mdes  = res["mdes_80pct"]
        lines += [
            f"Dictionary size D = {D_use}",
            f"  τ = {res['tau_calibrated']:.4f}  "
            f"(null best-match mean = {res['null_best_match_mean']:.4f})",
            f"",
            f"  {'α':>6}  {'Detection rate':>16}  {'SE':>8}  {'≥80%?':>7}",
            f"  {'─'*6}  {'─'*16}  {'─'*8}  {'─'*7}",
        ]
        for r in res["power_by_alpha"]:
            flag = "YES" if r["power_80"] else ""
            lines.append(
                f"  {r['alpha']:>6.2f}  {r['mean']:>16.3f}  {r['se']:>8.4f}  {flag:>7}"
            )
        lines += [
            f"",
            f"  MDES at 80% power: α = {mdes}",
            f"",
        ]

    # MDES summary table
    lines += [
        "Summary: MDES at 80% Power by Dictionary Size",
        "----------------------------------------------",
        f"  {'D':>8}  {'τ':>8}  {'MDES (α)':>10}",
        f"  {'─'*8}  {'─'*8}  {'─'*10}",
    ]
    for D_str, res in all_results.items():
        lines.append(
            f"  {res['D']:>8}  {res['tau_calibrated']:>8.4f}  {str(res['mdes_80pct']):>10}"
        )

    lines += [
        "",
        "Interpretation",
        "--------------",
        "  Universal SAE features (if present) that have cross-model Pearson",
        "  correlation α below the MDES cannot be reliably detected by this method.",
        "  Since only 1 universal feature was found (≈ chance), any true universal",
        "  features must have α below the MDES for the relevant dictionary size.",
        "  This converts the null result into a quantified upper bound.",
        "",
        "  Caveat: all MDES values are conservative (N_CHUNKS=100 here vs 1000",
        "  in the real analysis). True MDES at full scale may be lower.",
        f"",
        f"Elapsed: {elapsed}s",
    ]
    txt = "\n".join(lines)
    (OUT_DIR / "power_table.txt").write_text(txt)

    print("\n" + txt)
    print(f"\nSaved to {OUT_DIR}/power_table.{{json,txt}}")


if __name__ == "__main__":
    main()
