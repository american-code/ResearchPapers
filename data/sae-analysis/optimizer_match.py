#!/usr/bin/env python3
"""
Quantization x interpretability: do SAE features survive 4-bit quantization?

Two TopK-SAEs, identical configuration and seed, trained on activations from the
SAME model at the SAME layer over the SAME corpus. One variable: whether the
activations came from bf16 weights or from 4-bit weights converted locally from
those same bf16 weights.

Read against two reference points produced by this identical procedure:
  same model, different seed .... ~8% of matchable agree      (the ceiling)
  different architectures ....... 0.18-0.33%                  (the floor)

Near the ceiling means quantization perturbs the learned dictionary about as much
as reseeding does, and interpretability results obtained on quantized weights are
defensible. Near the floor means a 4-bit model's dictionary is close to a
different model's, and a large body of consumer-hardware interpretability work
rests on an unexamined assumption.

Matching/fingerprint functions are copied VERBATIM from cross_arch_matching_v3.py
via seed_stability_match.py, so the procedure is the published one.
"""

import json
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path.home() / "ResearchPapers"
SAE_DIR   = WORKSPACE / "data/sae-runs"
ACTS_DIR  = WORKSPACE / "data/activations"
OUT_DIR   = WORKSPACE / "data/optimizer-rung"
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
D_IN              = 3072

# Each arm decomposes its OWN model's activations. The chunk axis is shared --
# the same 1,000 text chunks in the same order -- which is what makes the
# fingerprints comparable, exactly as in the cross-architecture case.
ARMS = {
    "adam": {"ckpt": SAE_DIR / "llama3b-l14-bf16-sae/checkpoint_final.npz",
             "acts": ACTS_DIR / "llama3b-l14-bf16"},
    "muon": {"ckpt": SAE_DIR / "llama3b-l14-muon-sae/checkpoint_final.npz",
             "acts": ACTS_DIR / "llama3b-l14-bf16"},
}
PAIRS = (("adam", "muon"),)

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
                         f"run data/sae-runs/run_quant_interp.sh first")

    print("Optimizer rung, 50k steps, same corpus -- Adam vs Muon")
    UNIT, CEN, ACTIVE, META = {}, {}, {}, {}
    for name, cfg in ARMS.items():
        # the current collector writes activations.bin; the older one wrote .npy
        acts = open_acts(cfg["acts"], D_IN, override_file="activations.bin")
        sae  = load_sae(cfg["ckpt"])
        u, c, n_act = fingerprints(acts, sae)
        UNIT[name], CEN[name], ACTIVE[name] = u, c, n_act
        D = sae["W_enc"].shape[1]
        dead = int((n_act == 0).sum())
        low  = int(((n_act > 0) & (n_act < MIN_ACTIVE_CHUNKS)).sum())
        META[name] = {"checkpoint": str(cfg["ckpt"]), "activations": str(cfg["acts"]),
                      "dict_size": D, "dead_on_eval": dead, "low_support": low,
                      "matchable": int(D - dead - low)}
        print(f"  {name:5s} dead={dead} low_support={low} matchable={D - dead - low}")
        del sae, acts

    valid = {n: ACTIVE[n] >= MIN_ACTIVE_CHUNKS for n in ARMS}
    rng   = np.random.default_rng(SEED)
    out   = {"experiment": "quantization x interpretability",
             "question": "do SAE features survive 4-bit quantization of the source model?",
             "source_model": "Llama-3.2-3B layer 14; 4-bit arm converted locally with mlx_lm.convert --q-bits 4",
             "reference_points": {"same_model_different_seed_pct": 8.0,
                                  "cross_architecture_pct": [0.18, 0.33]},
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
            sims    = sorted(x[2] for x in m_cal)
            rec = {"null_mean": round(float(null_vals.mean()), 4),
                   "null_p99_tau": round(tau_null, 4),
                   "null_matches_per_replicate": round(null_per_rep, 1),
                   "expected_false_positives": round(exp_fp, 2),
                   "matches_calibrated": len(m_cal),
                   "matches_at_fixed_0.80": len(m_fixed),
                   "enrichment_over_null": (round(len(m_cal)/exp_fp, 1) if exp_fp > 0 else None),
                   "pct_of_matchable": round(100.0*len(m_cal)/denom, 2),
                   "median_similarity": (round(sims[len(sims)//2], 4) if sims else None)}
            out["metrics"][metric][f"{a}_vs_{b}"] = rec
            print(f"  [{metric}] {a} vs {b}: matched={rec['matches_calibrated']} "
                  f"({rec['pct_of_matchable']}% of matchable) "
                  f"enrichment={rec['enrichment_over_null']}x "
                  f"median_sim={rec['median_similarity']}")
            json.dump(m_cal, (OUT_DIR / f"{metric}_{a}_{b}_matches.json").open("w"))

    out["elapsed_s"] = round(time.time() - t0, 1)
    (OUT_DIR / "optimizer-rung-report.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT_DIR/'optimizer-rung-report.json'}  ({out['elapsed_s']}s)")

    # Key the headline off the configured pair rather than a hardcoded arm name --
    # the previous literal survived a rename and crashed the summary after a
    # four-minute matching run had already succeeded.
    a, b = PAIRS[0]
    r = out["metrics"]["pearson"][f"{a}_vs_{b}"]
    print("\n\u2500 headline \u2500")
    print(f"  {a} vs {b}: {r['matches_calibrated']} matched, "
          f"{r['pct_of_matchable']}% of matchable, {r['enrichment_over_null']}x over null")
    print( "  ceiling (same corpus, 50k): 9.07% pearson / 10.99% cosine")
    print( "  precision rung:             7.70% pearson / 10.30% cosine")


if __name__ == "__main__":
    main()
