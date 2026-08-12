#!/usr/bin/env python3
"""
Cross-architecture SAE feature matching — v3 (retrained Qwen SAE with aux revival loss).

Identical logic to cross_arch_matching_v2.py.  The only change is that the
Qwen checkpoint now comes from data/qwen-sae-v2/ (retrained with lambda_aux=0.1)
rather than the baseline data/sae-runs/qwen-3b-layer18/.

Purpose: check whether the 57% dead-latent rate in the baseline Qwen SAE was
suppressing universality recall (false negatives).  If v3 shows higher pairwise
match counts and more closed universal triangles, the dead-latent problem was
the cause.

Outputs (data/qwen-sae-v2/matching/):
  {cosine,pearson}_{llama_qwen,mistral_qwen,llama_mistral}_matches.json
  {cosine,pearson}_universal-features.json
  matching-report.json
  comparison-v2-v3.json          <- delta vs v2 results
"""

import json
import time
from pathlib import Path

import numpy as np

WORKSPACE = Path("/Users/melton/ResearchPapers")
SAE_DIR   = WORKSPACE / "data/sae-runs"
ACTS_DIR  = WORKSPACE / "data/activations"
# New Qwen SAE trained with aux revival loss
QWEN_V2_DIR = WORKSPACE / "data/qwen-sae-v2"
OUT_DIR     = QWEN_V2_DIR / "matching"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# V2 matching results for delta comparison
V2_REPORT = WORKSPACE / "data/sae-analysis/matching-v2/matching-report.json"

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

MODELS = {
    "llama": {
        "acts": ACTS_DIR / "llama-3b-layer16",
        "ckpt": SAE_DIR  / "llama-3b-layer14/checkpoint_final.npz",
        "d_in": 3072,
    },
    "qwen": {
        "acts": ACTS_DIR / "qwen-3b-layer18",
        "ckpt": QWEN_V2_DIR / "checkpoint_final.npz",   # aux-revival retrain
        "d_in": 2048,
    },
    "mistral": {
        "acts":      ACTS_DIR / "mistral-7b-layer16",
        "ckpt":      SAE_DIR  / "mistral-7b-layer16/checkpoint_final.npz",
        "d_in":      4096,
        "acts_file": "activations.npy",
    },
}


# ── loading / encoding ───────────────────────────────────────────────────────

def open_acts(acts_dir: Path, d_in: int, override_file: str | None = None):
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
    fp      = fp.T.copy()
    n_active = (fp > 0).sum(axis=1).astype(np.int32)

    unit = fp / np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
    cen  = fp - fp.mean(axis=1, keepdims=True)
    cen  = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    return unit, cen, n_active


# ── matching ─────────────────────────────────────────────────────────────────

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
    A  = fp_a[ia]
    B  = fp_b[ib]

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
    sims_null = []
    for _ in range(n_perm):
        perm = rng.permutation(fp_b.shape[1])
        Bp   = fp_b[ib][:, perm]
        Bp   = Bp / np.maximum(np.linalg.norm(Bp, axis=1, keepdims=True), 1e-12)
        a2b, a2b_s = best_match(A, Bp)
        b2a, _     = best_match(Bp, A)
        mutual = b2a[a2b] == np.arange(len(a2b))
        sims_null.append(a2b_s[mutual])
    allv = np.concatenate(sims_null)
    return float(np.percentile(allv, NULL_PERCENTILE)), allv


# ── main pipeline ─────────────────────────────────────────────────────────────

METRICS = ("cosine", "pearson")
PAIRS   = (("llama", "qwen"), ("mistral", "qwen"), ("llama", "mistral"))


def run_metric(metric, FP, valid, meta, rng):
    results, nulls = {}, {}
    for a, b in PAIRS:
        key = f"{a}_{b}"
        tau_null, null_vals = null_threshold(FP[a], FP[b], valid[a], valid[b], rng)
        nulls[key] = {
            "null_p99":      round(tau_null, 4),
            "null_mean":     round(float(null_vals.mean()), 4),
            "null_max":      round(float(null_vals.max()), 4),
            "n_null_samples": int(null_vals.size),
        }
        tau_used = max(FIXED_TAU, tau_null) if metric == "cosine" else tau_null
        m_fixed  = reciprocal_matches(FP[a], FP[b], valid[a], valid[b], FIXED_TAU)
        m_cal    = reciprocal_matches(FP[a], FP[b], valid[a], valid[b], tau_used)
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

    tri_null = []
    for _ in range(NULL_PERMUTATIONS):
        perm = rng.permutation(FP["qwen"].shape[1])
        Qp   = FP["qwen"][:, perm]
        m1   = reciprocal_matches(FP["llama"],   Qp, valid["llama"],   valid["qwen"],
                                  results["llama_qwen"]["tau_used"])
        m2   = reciprocal_matches(FP["mistral"], Qp, valid["mistral"], valid["qwen"],
                                  results["mistral_qwen"]["tau_used"])
        d1   = {a: b for a, b, _ in m1}
        d2   = {b: a for a, b, _ in m2}
        tri_null.append(sum(1 for lf, (mf, _) in lm_by_l.items()
                            if lf in d1 and d2.get(d1[lf]) == mf))
    tri_null = np.array(tri_null)

    uni  = {"llama": {u["llama_feat"] for u in universal},
            "qwen":  {u["qwen_feat"]  for u in universal},
            "mistral": {u["mistral_feat"] for u in universal}}
    side = {
        ("llama",   "qwen"):    {a for a, _, _ in results["llama_qwen"]["calibrated"]},
        ("qwen",    "llama"):   {b for _, b, _ in results["llama_qwen"]["calibrated"]},
        ("mistral", "qwen"):    {a for a, _, _ in results["mistral_qwen"]["calibrated"]},
        ("qwen",    "mistral"): {b for _, b, _ in results["mistral_qwen"]["calibrated"]},
        ("llama",   "mistral"): {a for a, _, _ in results["llama_mistral"]["calibrated"]},
        ("mistral", "llama"):   {b for _, b, _ in results["llama_mistral"]["calibrated"]},
    }

    def regions(m, o1, o2, n1, n2):
        D    = meta[m]["dict_size"]
        u    = uni[m]
        s1, s2 = side[(m, o1)], side[(m, o2)]
        both = (s1 & s2) - u
        r1   = s1 - s2 - u
        r2   = s2 - s1 - u
        none = D - len(r1) - len(r2) - len(both) - len(u)
        out  = {f"matched_{n1}_only": len(r1), f"matched_{n2}_only": len(r2),
                "matched_both_not_universal": len(both), "universal": len(u),
                "unmatched": none}
        assert sum(out.values()) == D, f"venn {m} does not partition"
        return out

    venn = {"llama":   regions("llama",   "qwen",   "mistral", "qwen", "mistral"),
            "qwen":    regions("qwen",    "llama",  "mistral", "llama", "mistral"),
            "mistral": regions("mistral", "llama",  "qwen",    "llama", "qwen")}

    return {"results": results, "nulls": nulls, "universal": universal,
            "tri_null": tri_null, "venn": venn}


def main() -> None:
    rng = np.random.default_rng(SEED)
    t0  = time.time()

    # Verify new Qwen checkpoint exists before starting expensive computation
    qwen_ckpt = MODELS["qwen"]["ckpt"]
    if not qwen_ckpt.exists():
        raise FileNotFoundError(
            f"Qwen v2 checkpoint not found: {qwen_ckpt}\n"
            f"Run train_sae_aux.py first."
        )

    COS, PEA, actives, meta = {}, {}, {}, {}
    for name, cfg in MODELS.items():
        print(f"[{name}] loading and encoding …", flush=True)
        acts = open_acts(cfg["acts"], cfg["d_in"], cfg.get("acts_file"))
        sae  = load_sae(cfg["ckpt"])
        unit, cen, nact = fingerprints(acts, sae)
        COS[name], PEA[name], actives[name] = unit, cen, nact
        D = unit.shape[0]
        meta[name] = {
            "checkpoint":            str(cfg["ckpt"].relative_to(WORKSPACE)),
            "dict_size":             int(D),
            "dead_on_eval_corpus":   int((nact == 0).sum()),
            "support_lt_min_chunks": int(((nact > 0) & (nact < MIN_ACTIVE_CHUNKS)).sum()),
            "valid_features":        int((nact >= MIN_ACTIVE_CHUNKS).sum()),
            "median_active_chunks":  int(np.median(nact)),
            "eval_tokens":           EVAL_TOKENS,
            "n_chunks":              N_CHUNKS,
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
                "metric":                  metric,
                "qwen_checkpoint":         "data/qwen-sae-v2 (aux revival loss lambda=0.1)",
                "method":                  "reciprocal best match (mutual nearest neighbour)",
                "fingerprint":             f"chunk-averaged TopK-SAE activations, "
                                           f"{N_CHUNKS} chunks x {CHUNK_TOKS} tokens",
                "corpus":                  "Salesforce/wikitext wikitext-103-raw-v1 (train)",
                "min_active_chunks":       MIN_ACTIVE_CHUNKS,
                "tau_calibrated":          round(r["tau_used"], 4),
                "permutation_null":        o["nulls"][key],
                "n_matches_tau_0.80":      len(r["fixed"]),
                "n_matches_tau_calibrated": len(r["calibrated"]),
                "matches":                 [{"feat_a": a, "feat_b": b, "sim": s}
                                            for a, b, s in r["calibrated"]],
            }, indent=1))

        (OUT_DIR / f"{metric}_universal-features.json").write_text(json.dumps({
            "metric":              metric,
            "qwen_checkpoint":     "data/qwen-sae-v2 (aux revival loss lambda=0.1)",
            "method":              "closed-triangle reciprocal best match across three SAEs",
            "definition":          "triple (l,q,m) with l<->q, l<->m and m<->q all mutual "
                                   "nearest neighbours above permutation-calibrated threshold",
            "min_active_chunks":   MIN_ACTIVE_CHUNKS,
            "eval_tokens":         EVAL_TOKENS,
            "n_chunks":            N_CHUNKS,
            "thresholds":          {k: round(v["tau_used"], 4)
                                    for k, v in o["results"].items()},
            "permutation_nulls":   o["nulls"],
            "n_universal_features": len(o["universal"]),
            "triangle_null":       {"n_permutations": int(NULL_PERMUTATIONS),
                                    "mean": float(o["tri_null"].mean()),
                                    "max":  int(o["tri_null"].max())},
            "venn_counts":         o["venn"],
            "model_metadata":      meta,
            "universal_features":  o["universal"],
        }, indent=1))

    report = {
        "generated_by":     "data/sae-analysis/cross_arch_matching_v3.py",
        "supersedes":       "cross_arch_matching_v2.py (baseline Qwen SAE, 57% dead latents)",
        "qwen_checkpoint":  "data/qwen-sae-v2 (aux revival loss lambda_aux=0.1, k_aux=512)",
        "model_metadata":   meta,
        "metrics": {
            m: {
                "pairwise": {k: {"n_matches":           len(v["calibrated"]),
                                 "n_matches_at_tau_0.80": len(v["fixed"]),
                                 "tau":                  round(v["tau_used"], 4),
                                 "null":                 out[m]["nulls"][k]}
                             for k, v in out[m]["results"].items()},
                "n_universal_features":  len(out[m]["universal"]),
                "triangle_null_mean":    float(out[m]["tri_null"].mean()),
                "triangle_null_max":     int(out[m]["tri_null"].max()),
                "venn_counts":           out[m]["venn"],
            } for m in METRICS
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "matching-report.json").write_text(json.dumps(report, indent=1))

    # Delta comparison vs v2 baseline
    comparison: dict = {
        "v2_report": str(V2_REPORT.relative_to(WORKSPACE)) if V2_REPORT.exists() else None,
        "note": "positive delta = more matches in v3 (aux-loss Qwen) vs v2 (baseline Qwen)",
    }
    if V2_REPORT.exists():
        v2 = json.loads(V2_REPORT.read_text())
        for metric in METRICS:
            v2m = v2.get("metrics", {}).get(metric, {})
            v3m = report["metrics"][metric]
            comparison[metric] = {
                "universal_v2":    v2m.get("n_universal_features"),
                "universal_v3":    v3m["n_universal_features"],
                "universal_delta": (v3m["n_universal_features"]
                                    - (v2m.get("n_universal_features") or 0)),
                "pairwise_deltas": {
                    k: {
                        "v2": v2m.get("pairwise", {}).get(k, {}).get("n_matches"),
                        "v3": v3m["pairwise"][k]["n_matches"],
                        "delta": (v3m["pairwise"][k]["n_matches"]
                                  - (v2m.get("pairwise", {}).get(k, {}).get("n_matches") or 0)),
                    }
                    for k in ("llama_qwen", "mistral_qwen", "llama_mistral")
                },
            }
    (OUT_DIR / "comparison-v2-v3.json").write_text(json.dumps(comparison, indent=1))

    print("\n=== Summary ===")
    print(json.dumps({m: {"pairwise": {k: v["n_matches"] for k, v in
                                       report["metrics"][m]["pairwise"].items()},
                          "universal": report["metrics"][m]["n_universal_features"],
                          "triangle_null_mean": report["metrics"][m]["triangle_null_mean"]}
                      for m in METRICS}, indent=1))
    if V2_REPORT.exists():
        print("\n=== Delta vs v2 baseline ===")
        for metric in METRICS:
            c = comparison.get(metric, {})
            print(f"  [{metric}] universal: {c.get('universal_v2')} -> "
                  f"{c.get('universal_v3')} (Δ{c.get('universal_delta'):+d})")
            for k, d in c.get("pairwise_deltas", {}).items():
                print(f"    {k}: {d['v2']} -> {d['v3']} (Δ{d['delta']:+d})")
    print(f"\nelapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
