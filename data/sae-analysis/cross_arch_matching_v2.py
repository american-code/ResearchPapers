#!/usr/bin/env python3
"""
Cross-architecture SAE feature matching, corrected method (v2).

Supersedes cross_arch_matching.py, which had three defects that inflated the
reported universality rate by roughly an order of magnitude:

  (1) MANY-TO-ONE MATCHING.  v1's nearest_neighbor_match() returned the *union*
      of the A->B and B->A nearest-neighbour sets.  A single feature in B could
      therefore be claimed as the match for arbitrarily many features in A (one
      Mistral feature was the recorded match for 137 distinct Llama features),
      and the reported "match count" could exceed the dictionary size.  v2 uses
      RECIPROCAL BEST MATCH (mutual nearest neighbour), which is one-to-one by
      construction.

  (2) UNCLOSED TRIANGLES.  v1's find_universal_features() chained
      llama -> qwen -> mistral and only *recorded* the llama-mistral cosine
      "if available"; it never required that edge to exist above threshold.
      67% of v1's "universal" triples had no verified llama-mistral edge.
      v2 requires a CLOSED TRIANGLE: all three reciprocal edges present.

  (3) DEGENERATE FINGERPRINTS.  A 100-dimensional chunk fingerprint built from
      50k tokens is low-dimensional relative to a 16,384-feature dictionary.
      Features active in only one or two chunks yield cosine similarity of
      exactly 1.0 with any other feature active in the same chunk, purely by
      construction.  v2 excludes features whose fingerprint support is below
      MIN_ACTIVE_CHUNKS, and reports how many were removed.

v2 additionally calibrates the similarity threshold against a PERMUTATION NULL
(chunk order shuffled independently per model, destroying cross-model
correspondence while preserving each feature's marginal activation
distribution) instead of using an unjustified fixed tau = 0.8.

Outputs (data/sae-analysis/):
  matching-v2/{llama_qwen,mistral_qwen,llama_mistral}_matches.json
  matching-v2/universal-features.json
  matching-v2/matching-report.json
"""

import json
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path("/Users/melton/ResearchPapers")
SAE_DIR   = WORKSPACE / "data/sae-runs"
ACTS_DIR  = WORKSPACE / "data/activations"
OUT_DIR   = WORKSPACE / "data/sae-analysis/matching-v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Fingerprint resolution. v1 used 50k tokens / 100 chunks, giving a
# 100-dimensional fingerprint for a 16,384-feature dictionary. At that
# resolution the best-of-16k match similarity reaches ~0.98 under a chunk-shuffle
# null, i.e. the statistic has no discriminative power. All three models have
# 500k stored activations (Mistral's SAE was *trained* on only 50k, but it can
# be *evaluated* on the full dump), so v2 uses 500k tokens / 1000 chunks.
EVAL_TOKENS       = 500_000
N_CHUNKS          = 1_000
CHUNK_TOKS        = EVAL_TOKENS // N_CHUNKS   # 500 tokens per chunk, as in v1
ENCODE_BATCH      = 10_000
TOPK              = 128
MIN_ACTIVE_CHUNKS = max(5, N_CHUNKS // 20)   # 5% support floor (see defect 3)
NULL_PERMUTATIONS = 20
NULL_PERCENTILE   = 99.0
FIXED_TAU         = 0.80     # v1's threshold, retained for comparability
SEED              = 42

MODELS = {
    "llama":   {"acts": ACTS_DIR / "llama-3b-layer14",
                "ckpt": SAE_DIR / "llama-3b-layer14/checkpoint_final.npz",
                "d_in": 3072},
    "qwen":    {"acts": ACTS_DIR / "qwen-3b-layer18",
                "ckpt": SAE_DIR / "qwen-3b-layer18/checkpoint_final.npz",
                "d_in": 2048},
    # Mistral's metadata points at the 50k training subset; for evaluation we
    # read the full 500k dump so all three fingerprints share a corpus.
    "mistral": {"acts": ACTS_DIR / "mistral-7b-layer16",
                "ckpt": SAE_DIR / "mistral-7b-layer16/checkpoint_final.npz",
                "d_in": 4096, "acts_file": "activations.npy"},
}


# ── loading / encoding ───────────────────────────────────────────────────────

def open_acts(acts_dir: Path, d_in: int, override_file: str | None = None):
    """Return a read-only memmap/ndarray view of the activation dump."""
    """
    Load activations [offset_tokens : offset_tokens+n_tokens].

    Note: despite the .npy extension, the Llama, Qwen and Mistral full-corpus
    dumps are headerless raw float16; only Mistral's 50k subset carries a real
    NPY header. Detect by magic rather than trusting the extension.
    """
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
    """
    Return (unit_fp, centered_unit_fp, n_active_chunks), each [D, N_CHUNKS].

    unit_fp     -> cosine similarity (v1's metric, retained for comparability)
    centered_fp -> Pearson correlation across chunks

    Cosine on raw fingerprints is the wrong similarity for this data: SAE
    activations are non-negative, so every fingerprint lies in the positive
    orthant and the cosine between two *unrelated* features has a large
    positive baseline (empirically ~0.75-0.82 here). Centering each fingerprint
    before normalising removes that baseline, making the statistic a
    correlation over chunks and the null distribution centred near zero.
    """
    D = sae["W_enc"].shape[1]
    n_total = N_CHUNKS * CHUNK_TOKS
    assert acts.shape[0] >= n_total, f"need {n_total} tokens, have {acts.shape[0]}"
    fp = np.zeros((N_CHUNKS, D), dtype=np.float32)
    off = 0
    for s in range(0, n_total, ENCODE_BATCH):
        e = min(s + ENCODE_BATCH, n_total)
        # materialise only this block as float32 (the full dump is 4-8 GB)
        blk = topk_encode(np.asarray(acts[s:e], dtype=np.float32), sae)
        n = (e - s) // CHUNK_TOKS
        fp[off:off + n] = blk.reshape(n, CHUNK_TOKS, D).mean(axis=1)
        off += n
        del blk
    fp = fp.T.copy()                                   # [D, N_CHUNKS]
    n_active = (fp > 0).sum(axis=1).astype(np.int32)

    unit = fp / np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
    cen = fp - fp.mean(axis=1, keepdims=True)
    cen = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    return unit, cen, n_active


# ── matching ─────────────────────────────────────────────────────────────────

def best_match(fp_a: np.ndarray, fp_b: np.ndarray, batch: int = 2048):
    """argmax_b cos(a,b) and its value, for every a. Fingerprints must be unit-norm."""
    n = fp_a.shape[0]
    idx = np.empty(n, dtype=np.int32)
    val = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        sims = fp_a[i:i + batch] @ fp_b.T
        j = np.argmax(sims, axis=1)
        idx[i:i + batch] = j
        val[i:i + batch] = sims[np.arange(len(j)), j]
    return idx, val


def reciprocal_matches(fp_a, fp_b, valid_a, valid_b, tau):
    """
    Mutual nearest neighbours above tau, restricted to valid features.
    One-to-one by construction. Returns list of (a, b, cosim).
    """
    # Restrict to valid features, keeping index maps back to full dictionaries.
    ia = np.flatnonzero(valid_a)
    ib = np.flatnonzero(valid_b)
    A = fp_a[ia]
    B = fp_b[ib]

    a2b, a2b_s = best_match(A, B)
    b2a, _     = best_match(B, A)

    out = []
    for local_a, local_b in enumerate(a2b):
        if b2a[local_b] != local_a:
            continue                      # not mutual
        s = float(a2b_s[local_a])
        if s < tau:
            continue
        out.append((int(ia[local_a]), int(ib[local_b]), round(s, 4)))
    return out


def null_threshold(fp_a, fp_b, valid_a, valid_b, rng, n_perm=NULL_PERMUTATIONS):
    """
    Permutation null: shuffle chunk order of B independently each replicate.
    This destroys cross-model chunk correspondence while preserving each
    feature's marginal activation profile. Returns the NULL_PERCENTILE of the
    reciprocal-match similarity distribution under the null.
    """
    ia = np.flatnonzero(valid_a)
    ib = np.flatnonzero(valid_b)
    A = fp_a[ia]
    sims_null = []
    for _ in range(n_perm):
        perm = rng.permutation(fp_b.shape[1])
        Bp = fp_b[ib][:, perm]
        Bp = Bp / np.maximum(np.linalg.norm(Bp, axis=1, keepdims=True), 1e-12)
        a2b, a2b_s = best_match(A, Bp)
        b2a, _ = best_match(Bp, A)
        mutual = b2a[a2b] == np.arange(len(a2b))
        sims_null.append(a2b_s[mutual])
    allv = np.concatenate(sims_null)
    return float(np.percentile(allv, NULL_PERCENTILE)), allv




# ── main ─────────────────────────────────────────────────────────────────────

METRICS = ("cosine", "pearson")
PAIRS = (("llama", "qwen"), ("mistral", "qwen"), ("llama", "mistral"))


def run_metric(metric, FP, valid, meta, rng):
    """Full matching pipeline for one similarity metric."""
    results, nulls = {}, {}
    for a, b in PAIRS:
        key = f"{a}_{b}"
        tau_null, null_vals = null_threshold(FP[a], FP[b], valid[a], valid[b], rng)
        nulls[key] = {
            "null_p99": round(tau_null, 4),
            "null_mean": round(float(null_vals.mean()), 4),
            "null_max": round(float(null_vals.max()), 4),
            "n_null_samples": int(null_vals.size),
        }
        tau_used = max(FIXED_TAU, tau_null) if metric == "cosine" else tau_null
        m_fixed = reciprocal_matches(FP[a], FP[b], valid[a], valid[b], FIXED_TAU)
        m_cal = reciprocal_matches(FP[a], FP[b], valid[a], valid[b], tau_used)
        results[key] = {"tau_used": float(tau_used), "fixed": m_fixed, "calibrated": m_cal}
        print(f"  [{metric}] {key:16s} null mean={null_vals.mean():.4f} "
              f"p99={tau_null:.4f} | matches @0.80={len(m_fixed):5d} "
              f"@tau={tau_used:.4f}={len(m_cal):5d}", flush=True)

    lq_by_l = {a: (b, s) for a, b, s in results["llama_qwen"]["calibrated"]}
    lm_by_l = {a: (b, s) for a, b, s in results["llama_mistral"]["calibrated"]}
    mq_by_m = {a: (b, s) for a, b, s in results["mistral_qwen"]["calibrated"]}

    universal = []
    for lf, (qf, s_lq) in lq_by_l.items():
        if lf not in lm_by_l:
            continue
        mf, s_lm = lm_by_l[lf]
        if mq_by_m.get(mf, (None, None))[0] != qf:
            continue
        universal.append({
            "llama_feat": lf, "qwen_feat": qf, "mistral_feat": mf,
            "cosim_llama_qwen": s_lq, "cosim_llama_mistral": s_lm,
            "cosim_mistral_qwen": mq_by_m[mf][1],
        })
    universal.sort(key=lambda e: -(e["cosim_llama_qwen"] + e["cosim_llama_mistral"]
                                   + e["cosim_mistral_qwen"]) / 3)
    for k in ("llama_feat", "qwen_feat", "mistral_feat"):
        assert len({u[k] for u in universal}) == len(universal), f"{k} not one-to-one"

    # triangle null
    tri_null = []
    for _ in range(NULL_PERMUTATIONS):
        perm = rng.permutation(FP["qwen"].shape[1])
        Qp = FP["qwen"][:, perm]
        m1 = reciprocal_matches(FP["llama"], Qp, valid["llama"], valid["qwen"],
                                results["llama_qwen"]["tau_used"])
        m2 = reciprocal_matches(FP["mistral"], Qp, valid["mistral"], valid["qwen"],
                                results["mistral_qwen"]["tau_used"])
        d1 = {a: b for a, b, _ in m1}
        d2 = {b: a for a, b, _ in m2}
        tri_null.append(sum(1 for lf, (mf, _) in lm_by_l.items()
                            if lf in d1 and d2.get(d1[lf]) == mf))
    tri_null = np.array(tri_null)

    # Venn, per model, partitioning the dictionary
    uni = {"llama": {u["llama_feat"] for u in universal},
           "qwen": {u["qwen_feat"] for u in universal},
           "mistral": {u["mistral_feat"] for u in universal}}
    side = {
        ("llama", "qwen"): {a for a, _, _ in results["llama_qwen"]["calibrated"]},
        ("qwen", "llama"): {b for _, b, _ in results["llama_qwen"]["calibrated"]},
        ("mistral", "qwen"): {a for a, _, _ in results["mistral_qwen"]["calibrated"]},
        ("qwen", "mistral"): {b for _, b, _ in results["mistral_qwen"]["calibrated"]},
        ("llama", "mistral"): {a for a, _, _ in results["llama_mistral"]["calibrated"]},
        ("mistral", "llama"): {b for _, b, _ in results["llama_mistral"]["calibrated"]},
    }

    def regions(m, o1, o2, n1, n2):
        D = meta[m]["dict_size"]
        u = uni[m]
        s1, s2 = side[(m, o1)], side[(m, o2)]
        both = (s1 & s2) - u
        r1 = s1 - s2 - u
        r2 = s2 - s1 - u
        none = D - len(r1) - len(r2) - len(both) - len(u)
        out = {f"matched_{n1}_only": len(r1), f"matched_{n2}_only": len(r2),
               "matched_both_not_universal": len(both), "universal": len(u),
               "unmatched": none}
        assert sum(out.values()) == D, f"venn {m} does not partition ({sum(out.values())} vs {D})"
        return out

    venn = {"llama": regions("llama", "qwen", "mistral", "qwen", "mistral"),
            "qwen": regions("qwen", "llama", "mistral", "llama", "mistral"),
            "mistral": regions("mistral", "llama", "qwen", "llama", "qwen")}

    return {"results": results, "nulls": nulls, "universal": universal,
            "tri_null": tri_null, "venn": venn}


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    COS, PEA, actives, meta = {}, {}, {}, {}
    for name, cfg in MODELS.items():
        print(f"[{name}] loading and encoding …", flush=True)
        acts = open_acts(cfg["acts"], cfg["d_in"], cfg.get("acts_file"))
        sae = load_sae(cfg["ckpt"])
        unit, cen, nact = fingerprints(acts, sae)
        COS[name], PEA[name], actives[name] = unit, cen, nact
        D = unit.shape[0]
        meta[name] = {
            "checkpoint": str(cfg["ckpt"].relative_to(WORKSPACE)),
            "dict_size": int(D),
            "dead_on_eval_corpus": int((nact == 0).sum()),
            "support_lt_min_chunks": int(((nact > 0) & (nact < MIN_ACTIVE_CHUNKS)).sum()),
            "valid_features": int((nact >= MIN_ACTIVE_CHUNKS).sum()),
            "median_active_chunks": int(np.median(nact)),
            "eval_tokens": EVAL_TOKENS,
            "n_chunks": N_CHUNKS,
        }
        print(f"  dict={D}  dead_on_eval={meta[name]['dead_on_eval_corpus']}  "
              f"support<{MIN_ACTIVE_CHUNKS}={meta[name]['support_lt_min_chunks']}  "
              f"valid={meta[name]['valid_features']}", flush=True)
        del acts, sae

    valid = {n: actives[n] >= MIN_ACTIVE_CHUNKS for n in MODELS}

    out = {}
    for metric, FP in (("cosine", COS), ("pearson", PEA)):
        print(f"\n=== metric: {metric} ===", flush=True)
        out[metric] = run_metric(metric, FP, valid, meta, rng)
        print(f"  [{metric}] universal (closed triangle) = "
              f"{len(out[metric]['universal'])}   "
              f"null mean = {out[metric]['tri_null'].mean():.2f}", flush=True)

    for metric in METRICS:
        o = out[metric]
        for key, r in o["results"].items():
            (OUT_DIR / f"{metric}_{key}_matches.json").write_text(json.dumps({
                "metric": metric,
                "method": "reciprocal best match (mutual nearest neighbour), one-to-one",
                "fingerprint": f"chunk-averaged TopK-SAE activations, {N_CHUNKS} chunks "
                               f"x {CHUNK_TOKS} tokens = {EVAL_TOKENS} tokens",
                "corpus": "Salesforce/wikitext wikitext-103-raw-v1 (train)",
                "min_active_chunks": MIN_ACTIVE_CHUNKS,
                "tau_calibrated": round(r["tau_used"], 4),
                "permutation_null": o["nulls"][key],
                "n_matches_tau_0.80": len(r["fixed"]),
                "n_matches_tau_calibrated": len(r["calibrated"]),
                "matches": [{"feat_a": a, "feat_b": b, "sim": s} for a, b, s in r["calibrated"]],
            }, indent=1))

        (OUT_DIR / f"{metric}_universal-features.json").write_text(json.dumps({
            "metric": metric,
            "method": "closed-triangle reciprocal best match across three SAEs",
            "definition": "triple (l,q,m) with l<->q, l<->m and m<->q all mutual "
                          "nearest neighbours above the permutation-calibrated threshold",
            "min_active_chunks": MIN_ACTIVE_CHUNKS,
            "eval_tokens": EVAL_TOKENS, "n_chunks": N_CHUNKS,
            "thresholds": {k: round(v["tau_used"], 4) for k, v in o["results"].items()},
            "permutation_nulls": o["nulls"],
            "n_universal_features": len(o["universal"]),
            "triangle_null": {"n_permutations": int(NULL_PERMUTATIONS),
                              "mean": float(o["tri_null"].mean()),
                              "max": int(o["tri_null"].max())},
            "venn_counts": o["venn"],
            "model_metadata": meta,
            "universal_features": o["universal"],
        }, indent=1))

    report = {
        "generated_by": "data/sae-analysis/cross_arch_matching_v2.py",
        "supersedes": "data/sae-analysis/cross_arch_matching.py (v1)",
        "v1_defects_corrected": [
            "many-to-one matching (v1 unioned A->B and B->A nearest neighbours)",
            "unclosed triangles (v1 never required the llama-mistral edge)",
            "degenerate low-support fingerprints inflating cosine to exactly 1.0",
            "uncalibrated fixed threshold tau=0.80, which lies BELOW the "
            "permutation-null mean for the cosine metric",
        ],
        "model_metadata": meta,
        "metrics": {
            m: {
                "pairwise": {k: {"n_matches": len(v["calibrated"]),
                                 "n_matches_at_tau_0.80": len(v["fixed"]),
                                 "tau": round(v["tau_used"], 4),
                                 "null": out[m]["nulls"][k]}
                             for k, v in out[m]["results"].items()},
                "n_universal_features": len(out[m]["universal"]),
                "triangle_null_mean": float(out[m]["tri_null"].mean()),
                "triangle_null_max": int(out[m]["tri_null"].max()),
                "venn_counts": out[m]["venn"],
            } for m in METRICS
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "matching-report.json").write_text(json.dumps(report, indent=1))
    print("\n" + json.dumps({m: {"pairwise": {k: v["n_matches"] for k, v in
                                              report["metrics"][m]["pairwise"].items()},
                                 "universal": report["metrics"][m]["n_universal_features"],
                                 "triangle_null_mean": report["metrics"][m]["triangle_null_mean"]}
                             for m in METRICS}, indent=1))
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
