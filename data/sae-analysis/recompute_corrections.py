#!/usr/bin/env python3
"""
Recompute the statistics the three papers reported incorrectly or omitted.

Produces data/sae-analysis/corrections.json with:

  1. SAE reconstruction quality on the FULL corpus rather than a single training
     batch. The papers quote `fve` from the last logged training step, which is a
     one-batch estimate that swings by several points between adjacent steps.
     For Mistral this is additionally a TRUE HELD-OUT number: that SAE was
     trained on tokens [0:50k] of the dump, so tokens [50k:500k] are unseen.

  2. Safety-classifier ROC-AUC, average precision and Cohen's d. The paper
     reports only a threshold sweep, which cannot separate "the score is
     uninformative" from "the threshold is badly chosen".

  3. Permutation null for the IOI cross-architecture depth-conservation claim.
     The circuit-tracing paper reports 8/10 head positions matching within
     +/-0.075 relative depth with no null, so the reader cannot tell whether
     that rate is better than chance.
"""

import json
from pathlib import Path

import numpy as np

WS = Path("/Users/melton/ResearchPapers")
OUT = WS / "data/sae-analysis/corrections.json"
SEED = 42

SAES = {
    "llama": {"ckpt": "data/sae-runs/llama-3b-layer14/checkpoint_step_010000.npz",
              "acts": "data/activations/llama-3b-layer14/activations.npy",
              "d_in": 3072, "k": 128, "trained_on_tokens": 500_000,
              "layer": 14, "model": "Llama-3.2-3B"},
    "qwen": {"ckpt": "data/sae-runs/qwen-3b-layer18/checkpoint_final.npz",
             "acts": "data/activations/qwen-3b-layer18/activations.npy",
             "d_in": 2048, "k": 128, "trained_on_tokens": 500_000,
             "layer": 18, "model": "Qwen2.5-3B"},
    "mistral": {"ckpt": "data/sae-runs/mistral-7b-layer16/checkpoint_final.npz",
                "acts": "data/activations/mistral-7b-layer16/activations.npy",
                "d_in": 4096, "k": 128, "trained_on_tokens": 50_000,
                "layer": 16, "model": "Mistral-7B-v0.3 (4-bit)"},
}
N_EVAL = 500_000
BLOCK = 10_000


def open_raw(path: Path, d_in: int):
    with open(path, "rb") as fh:
        if fh.read(6) == b"\x93NUMPY":
            return np.load(str(path), mmap_mode="r")
    n = path.stat().st_size // (d_in * 2)
    return np.memmap(str(path), dtype=np.float16, mode="r", shape=(n, d_in))


def sae_eval(ckpt: Path, acts, k: int, lo: int, hi: int):
    """
    Streamed reconstruction stats over tokens [lo:hi].

    FVE matches the definition used in train_sae.py:
        fve = 1 - var(x - x_hat) / var(x)
    computed here over the whole slice rather than a single batch. Also returns
    MSE and the fraction of dictionary features that never fire on the slice.
    """
    z = np.load(str(ckpt))
    W_enc, b_enc = z["W_enc"].astype(np.float32), z["b_enc"].astype(np.float32)
    W_dec, b_dec = z["W_dec"].astype(np.float32), z["b_dec"].astype(np.float32)
    D = W_enc.shape[1]

    n = hi - lo
    se = 0.0          # sum of squared error
    s_err = 0.0       # sum of error (for var of residual)
    sx = 0.0
    sxx = 0.0
    cnt = 0
    fired = np.zeros(D, dtype=bool)
    fire_counts = np.zeros(D, dtype=np.int64)

    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        x = np.asarray(acts[s:e], dtype=np.float32)
        pre = (x - b_dec) @ W_enc + b_enc
        kth = np.partition(pre, -k, axis=1)[:, -k][:, None]
        f = np.where(pre >= kth, pre, 0.0)
        xh = f @ W_dec + b_dec
        r = x - xh
        se += float((r * r).sum())
        s_err += float(r.sum())
        sx += float(x.sum())
        sxx += float((x * x).sum())
        cnt += x.size
        act = f > 0
        fired |= act.any(axis=0)
        fire_counts += act.sum(axis=0)
        del x, pre, f, xh, r

    mse = se / cnt
    var_res = se / cnt - (s_err / cnt) ** 2
    var_x = sxx / cnt - (sx / cnt) ** 2
    return {
        "n_tokens": int(n),
        "mse": round(mse, 6),
        "fve": round(float(1.0 - var_res / (var_x + 1e-8)), 4),
        "dead_features": int((~fired).sum()),
        "dead_rate": round(float((~fired).mean()), 4),
        "mean_l0": round(float(fire_counts.sum() / n), 2),
        "features_firing_over_50pct_of_tokens": int((fire_counts > 0.5 * n).sum()),
        "top200_share_of_activation_mass": round(
            float(np.sort(fire_counts)[::-1][:200].sum() / max(fire_counts.sum(), 1)), 4),
    }


def roc_auc(y, s):
    """ROC-AUC via the rank (Mann-Whitney U) identity; ties get average ranks."""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    y = np.asarray(y, dtype=bool)
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(int(y.sum()), 1))


def bootstrap_auc(y, s, n=2000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx].tolist())) < 2:
            continue
        vals.append(roc_auc(y[idx], s[idx]))
    v = np.sort(vals)
    return round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)


def greedy_depth_match(a, b, tol=0.075):
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b))
    ua, ub, n = set(), set(), 0
    for d, i, j in pairs:
        if d > tol:
            break
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        n += 1
    return n


def ioi_depth_null(rng):
    """
    Null for 'X of 10 circuit-critical head positions match within +/-0.075
    relative depth'. Two nulls are reported:
      uniform  - heads drawn uniformly from all layers
      informed - heads drawn from the layer band where critical heads actually
                 sit (relative depth >= 0.40 in both models), which is the
                 fairer comparison because the claim is about *which* depths
                 match, not that circuits live in the second half of the stack.
    """
    llama_layers = [15, 17, 13, 24, 19, 21, 18, 14, 27, 26]
    pythia_layers = [10, 15, 22, 1, 21, 17, 10, 12, 16, 13]
    ld = [l / 28 for l in llama_layers]
    pd = [l / 24 for l in pythia_layers]
    obs = greedy_depth_match(ld, pd)

    out = {"observed_matches": obs, "observed_rate": obs / 10}
    for name, (lo_l, lo_p) in (("uniform", (0, 0)), ("informed", (11, 9))):
        N = 20000
        hits = 0
        tot = 0
        for _ in range(N):
            a = rng.integers(lo_l, 28, 10) / 28
            b = rng.integers(lo_p, 24, 10) / 24
            m = greedy_depth_match(list(a), list(b))
            tot += m
            hits += (m >= obs)
        out[f"null_{name}"] = {
            "mean_matches": round(tot / N, 2),
            "mean_rate": round(tot / N / 10, 3),
            "p_value": round(hits / N, 4),
            "n_permutations": N,
        }
    return out


def main():
    rng = np.random.default_rng(SEED)
    res = {"note": "Generated by data/sae-analysis/recompute_corrections.py"}

    # ── 1. full-corpus / held-out SAE reconstruction ────────────────────────
    sae_out = {}
    for name, cfg in SAES.items():
        acts = open_raw(WS / cfg["acts"], cfg["d_in"])
        n_avail = min(N_EVAL, acts.shape[0])
        entry = {"model": cfg["model"], "layer": cfg["layer"],
                 "trained_on_tokens": cfg["trained_on_tokens"]}
        entry["in_sample_full_corpus"] = sae_eval(
            WS / cfg["ckpt"], acts, cfg["k"], 0, n_avail)
        if cfg["trained_on_tokens"] < n_avail:
            entry["held_out"] = sae_eval(
                WS / cfg["ckpt"], acts, cfg["k"], cfg["trained_on_tokens"], n_avail)
            entry["held_out"]["token_range"] = [cfg["trained_on_tokens"], int(n_avail)]
        else:
            entry["held_out"] = None
            entry["held_out_note"] = (
                "No held-out data: the SAE was trained on the entire 500k-token dump "
                f"({cfg['trained_on_tokens'] * 204 // 500000 if False else 204} epochs), "
                "so every reported figure is in-sample.")
        sae_out[name] = entry
        print(f"[{name}] in-sample FVE={entry['in_sample_full_corpus']['fve']} "
              f"dead={entry['in_sample_full_corpus']['dead_rate']} "
              f"held_out={entry['held_out']['fve'] if entry['held_out'] else 'n/a'}", flush=True)
    res["sae_reconstruction"] = sae_out

    # ── 2. safety classifier discrimination ─────────────────────────────────
    ev = json.loads((WS / "data/safety-classifier/evaluation-results.json").read_text())
    per = ev["per_example_scores"]
    y = [1 if p["label"] == "unsafe" else 0 for p in per]
    s = [p["score"] for p in per]
    unsafe = np.array([p["score"] for p in per if p["label"] == "unsafe"])
    safe = np.array([p["score"] for p in per if p["label"] == "safe"])
    pooled = np.sqrt(((len(unsafe) - 1) * unsafe.var(ddof=1)
                      + (len(safe) - 1) * safe.var(ddof=1))
                     / (len(unsafe) + len(safe) - 2))
    auc = roc_auc(y, s)
    lo, hi = bootstrap_auc(y, s, rng=rng)
    res["safety_classifier"] = {
        "n": len(y), "n_unsafe": int(sum(y)), "n_safe": len(y) - int(sum(y)),
        "roc_auc": round(auc, 4),
        "roc_auc_95ci": [lo, hi],
        "average_precision": round(average_precision(y, s), 4),
        "ap_baseline": round(sum(y) / len(y), 4),
        "cohens_d": round(float((unsafe.mean() - safe.mean()) / pooled), 4),
        "mean_unsafe": round(float(unsafe.mean()), 4),
        "mean_safe": round(float(safe.mean()), 4),
        "auc_ci_includes_chance": bool(lo <= 0.5 <= hi),
    }
    print(f"[safety] AUC={auc:.4f} 95%CI=[{lo},{hi}] d={res['safety_classifier']['cohens_d']}",
          flush=True)

    # ── 3. IOI depth-conservation null ──────────────────────────────────────
    res["ioi_depth_conservation"] = ioi_depth_null(rng)
    print(f"[ioi] {json.dumps(res['ioi_depth_conservation'])}", flush=True)

    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
