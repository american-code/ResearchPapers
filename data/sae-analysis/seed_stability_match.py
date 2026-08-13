#!/usr/bin/env python3
"""
Seed-stability control for the cross-model SAE feature-matching method.

THE QUESTION
------------
sae-comparison reports that exactly one feature triple is shared across
Llama/Mistral/Qwen, against a null that produces at least one 14% of the time.
That number cannot be interpreted without knowing what the same detector returns
on a case where sharing is guaranteed. This script supplies that case.

Two TopK-SAEs, same source model (Llama-3.2-3B layer 14), same activations, same
hyperparameters, differing ONLY in seed (123 vs 456; the paper used 42). Any
feature that is a real property of the model's representation should appear in
both dictionaries. The match rate here is therefore the CEILING of the matching
procedure -- its true-positive rate under realistic retraining variation.

Reading the result:
  high match rate  -> the method can detect shared features; the near-zero
                      cross-model count is a fact about the models.
  low match rate   -> the method cannot detect sharing even when it is
                      guaranteed; the cross-model null is a fact about the
                      METHOD, and sae-comparison's claim must be reframed.

All matching/null/fingerprint functions below are copied VERBATIM from
cross_arch_matching_v3.py so the procedure is bit-for-bit the published one.
Only the model pair and the reporting differ.
"""

import json
import time
from pathlib import Path

import numpy as np

# lab-02 home differs from the workstation the v2/v3 scripts were written on
WORKSPACE = Path.home() / "ResearchPapers"
SAE_DIR   = WORKSPACE / "data/sae-runs"
ACTS_DIR  = WORKSPACE / "data/activations"
OUT_DIR   = WORKSPACE / "data/seed-stability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_TOKENS       = 500_000
N_CHUNKS          = 1_000
CHUNK_TOKS        = EVAL_TOKENS // N_CHUNKS
ENCODE_BATCH      = 10_000
TOPK              = 128
MIN_ACTIVE_CHUNKS = max(5, N_CHUNKS // 20)
NULL_PERMUTATIONS = 20
NULL_PERCENTILE   = 99.0
FIXED_TAU         = 0.80
SEED              = 42

# Both arms share one activation dump: the directory is named "layer16" but its
# metadata records target_layer 14 (Llama-3.2-3B has 28 layers, not 32).
ACTS = ACTS_DIR / "llama-3b-layer16"
D_IN = 3072

ARMS = {
    "seed123": {"ckpt": SAE_DIR / "llama-3b-layer14-seed123/checkpoint_final.npz"},
    "seed456": {"ckpt": SAE_DIR / "llama-3b-layer14-seed456/checkpoint_final.npz"},
    # third arm: the paper's own checkpoint, seed 42, same data and config
    "seed42":  {"ckpt": SAE_DIR / "llama-3b-layer14/checkpoint_final.npz"},
}
PAIRS = (("seed123", "seed456"), ("seed42", "seed123"), ("seed42", "seed456"))


# ── loading / encoding (verbatim from v3) ────────────────────────────────────

def open_acts(acts_dir: Path, d_in: int, override_file=None):
    meta = json.loads((acts_dir / "metadata.json").read_text())
    f = acts_dir / (override_file or meta.get("activations_file", "activations.npy"))
    with open(f, "rb") as fh:
        is_npy = fh.read(6) == b"\x93NUMPY"
    if is_npy:
        raw = np.load(str(f), mmap_mode="r")
    else:
        n_avail = f.stat().st_size // (d_in * 2)
        raw = np.memmap(str(f), dtype=np.float16, mode="r", shape=(n_avail, d_in))
    assert raw.shape[1] == d_in, f"{f}: d_in {raw.shape[1]} != {d_in}"
    return raw


def load_sae(ckpt: Path) -> dict:
    z = np.load(str(ckpt))
    return {k: z[k].astype(np.float32) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}


def topk_encode(acts: np.ndarray, sae: dict, k: int = TOPK) -> np.ndarray:
    pre = (acts - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    kth = np.partition(pre, -k, axis=1)[:, -k][:, None]
    return np.where(pre >= kth, pre, 0.0)


def fingerprints(acts, sae: dict):
    D       = sae["W_enc"].shape[1]
    n_total = N_CHUNKS * CHUNK_TOKS
    assert acts.shape[0] >= n_total, f"need {n_total} tokens, have {acts.shape[0]}"
    fp  = np.zeros((N_CHUNKS, D), dtype=np.float32)
    off = 0
    for s in range(0, n_total, ENCODE_BATCH):
        e   = min(s + ENCODE_BATCH, n_total)
        blk = topk_encode(np.asarray(acts[s:e], dtype=np.float32), sae)
        n   = (e - s) // CHUNK_TOKS
        fp[off:off + n] = blk.reshape(n, CHUNK_TOKS, D).mean(axis=1)
        off += n
        del blk
    fp       = fp.T.copy()
    n_active = (fp > 0).sum(axis=1).astype(np.int32)

    unit = fp / np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
    cen  = fp - fp.mean(axis=1, keepdims=True)
    cen  = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    return unit, cen, n_active


# ── matching (verbatim from v3) ──────────────────────────────────────────────

def best_match(fp_a: np.ndarray, fp_b: np.ndarray, batch: int = 2048):
    n   = fp_a.shape[0]
    idx = np.empty(n, dtype=np.int32)
    val = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        sims = fp_a[i:i + batch] @ fp_b.T
        j    = np.argmax(sims, axis=1)
        idx[i:i + batch] = j
        val[i:i + batch] = sims[np.arange(len(j)), j]
    return idx, val


def reciprocal_matches(fp_a, fp_b, valid_a, valid_b, tau):
    ia = np.flatnonzero(valid_a)
    ib = np.flatnonzero(valid_b)
    A, B = fp_a[ia], fp_b[ib]
    a2b, a2b_s = best_match(A, B)
    b2a, _     = best_match(B, A)
    out = []
    for local_a, local_b in enumerate(a2b):
        if b2a[local_b] != local_a:
            continue
        s = float(a2b_s[local_a])
        if s < tau:
            continue
        out.append((int(ia[local_a]), int(ib[local_b]), round(s, 4)))
    return out


def null_threshold(fp_a, fp_b, valid_a, valid_b, rng, n_perm=NULL_PERMUTATIONS):
    ia = np.flatnonzero(valid_a)
    ib = np.flatnonzero(valid_b)
    A  = fp_a[ia]
    sims_null, per_rep = [], []
    for _ in range(n_perm):
        perm = rng.permutation(fp_b.shape[1])
        Bp   = fp_b[ib][:, perm]
        Bp   = Bp / np.maximum(np.linalg.norm(Bp, axis=1, keepdims=True), 1e-12)
        a2b, a2b_s = best_match(A, Bp)
        b2a, _     = best_match(Bp, A)
        mutual = b2a[a2b] == np.arange(len(a2b))
        sims_null.append(a2b_s[mutual])
        per_rep.append(int(mutual.sum()))
    allv = np.concatenate(sims_null)
    return float(np.percentile(allv, NULL_PERCENTILE)), allv, float(np.mean(per_rep))


# ── pipeline ─────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    missing = [n for n, c in ARMS.items() if not c["ckpt"].exists()]
    if missing:
        raise SystemExit(f"checkpoints not found: {missing}\n"
                         f"run data/sae-runs/run_seed_stability.sh first")

    print(f"Seed-stability control — {len(ARMS)} arms, same model, same activations")
    print(f"  activations : {ACTS}")

    acts = open_acts(ACTS, D_IN)
    UNIT, CEN, ACTIVE, META = {}, {}, {}, {}
    for name, cfg in ARMS.items():
        sae = load_sae(cfg["ckpt"])
        u, c, n_act = fingerprints(acts, sae)
        UNIT[name], CEN[name], ACTIVE[name] = u, c, n_act
        D = sae["W_enc"].shape[1]
        dead = int((n_act == 0).sum())
        low  = int(((n_act > 0) & (n_act < MIN_ACTIVE_CHUNKS)).sum())
        META[name] = {"checkpoint": str(cfg["ckpt"]), "dict_size": D,
                      "dead_on_eval": dead, "low_support": low,
                      "matchable": int(D - dead - low)}
        print(f"  {name:8s} dict={D} dead={dead} low_support={low} "
              f"matchable={D - dead - low}")
        del sae

    valid = {n: ACTIVE[n] >= MIN_ACTIVE_CHUNKS for n in ARMS}
    rng   = np.random.default_rng(SEED)
    out   = {"experiment": "seed-stability control",
             "question": ("what does the cross-model matching procedure return when "
                          "feature sharing is guaranteed by construction?"),
             "source_model": "Llama-3.2-3B layer 14",
             "activations": str(ACTS),
             "parameters": {"eval_tokens": EVAL_TOKENS, "n_chunks": N_CHUNKS,
                            "topk": TOPK, "min_active_chunks": MIN_ACTIVE_CHUNKS,
                            "null_permutations": NULL_PERMUTATIONS,
                            "null_percentile": NULL_PERCENTILE,
                            "fixed_tau": FIXED_TAU, "rng_seed": SEED},
             "arms": META, "metrics": {}}

    for metric, FP in (("cosine", UNIT), ("pearson", CEN)):
        out["metrics"][metric] = {}
        for a, b in PAIRS:
            tau_null, null_vals, null_per_rep = null_threshold(
                FP[a], FP[b], valid[a], valid[b], rng)
            tau_used = max(FIXED_TAU, tau_null) if metric == "cosine" else tau_null
            m_cal   = reciprocal_matches(FP[a], FP[b], valid[a], valid[b], tau_used)
            m_fixed = reciprocal_matches(FP[a], FP[b], valid[a], valid[b], FIXED_TAU)
            exp_fp  = null_per_rep * (1.0 - NULL_PERCENTILE / 100.0)
            denom   = min(META[a]["matchable"], META[b]["matchable"])
            rec = {
                "null_mean": round(float(null_vals.mean()), 4),
                "null_p99_tau": round(tau_null, 4),
                "null_matches_per_replicate": round(null_per_rep, 1),
                "expected_false_positives": round(exp_fp, 2),
                "matches_calibrated": len(m_cal),
                "matches_at_fixed_0.80": len(m_fixed),
                "enrichment_over_null": (round(len(m_cal) / exp_fp, 1)
                                         if exp_fp > 0 else None),
                "pct_of_matchable": round(100.0 * len(m_cal) / denom, 2),
            }
            out["metrics"][metric][f"{a}_vs_{b}"] = rec
            print(f"  [{metric}] {a} vs {b}: null_mean={rec['null_mean']:.4f} "
                  f"tau={rec['null_p99_tau']:.4f} matched={rec['matches_calibrated']} "
                  f"({rec['pct_of_matchable']}% of matchable) "
                  f"enrichment={rec['enrichment_over_null']}x")
            json.dump(m_cal, (OUT_DIR / f"{metric}_{a}_{b}_matches.json").open("w"))

    out["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "seed-stability-report.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_DIR/'seed-stability-report.json'}  ({out['elapsed_s']}s)")

    print("\n─ headline ─")
    for a, b in PAIRS:
        r = out["metrics"]["pearson"][f"{a}_vs_{b}"]
        print(f"  {a} vs {b}: {r['matches_calibrated']} matched "
              f"({r['pct_of_matchable']}% of matchable), "
              f"{r['enrichment_over_null']}x over null")
    print("  cross-model reference (paper, pearson): 15-23 matches, "
          "0.1% of matchable, 1.7-3.2x over null")


if __name__ == "__main__":
    main()
