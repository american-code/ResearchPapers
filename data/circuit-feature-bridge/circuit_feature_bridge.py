#!/usr/bin/env python3
"""
Circuit-Feature Bridge: L15H20 on IOI examples, Llama-3.2-3B

For each IOI example, identifies which layer-14 SAE features are:
  - READ by L15H20 (active in residual stream that feeds into the head)
  - WRITTEN by L15H20 (aligned with the head's W_O output subspace)

Reports "universal" features: those appearing consistently across >40% of examples.
"""

import json
import sys
import numpy as np
from pathlib import Path
import mlx.core as mx

SAE_PATH   = Path("~/ResearchPapers/data/sae-runs/llama-3b-layer14/checkpoint_final.npz").expanduser()
IOI_PATH   = Path("~/ResearchPapers/data/ioi/dataset-v2.json").expanduser()
OUT_DIR    = Path("~/ResearchPapers/data/circuit-feature-bridge").expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH         = "mlx-community/Llama-3.2-3B-bf16"
SAE_LAYER          = 14
HEAD_LAYER         = 15
HEAD_IDX           = 20
N_EXAMPLES         = 100
TOP_K_DISPLAY      = 20
UNIVERSAL_THRESH   = 0.40   # feature in top-k for >40% of examples

def log(msg=""):
    print(msg, flush=True)

log("=" * 60)
log("Circuit-Feature Bridge: L15H20 x IOI, Llama-3.2-3B layer-14 SAE")
log("=" * 60)

# ── Load SAE ─────────────────────────────────────────────────────────────────
log("\n[1] Loading SAE...")
sae_data = np.load(SAE_PATH)
W_enc = sae_data["W_enc"].astype(np.float32)   # [d_in=3072, dict_size=16384]
b_enc = sae_data["b_enc"].astype(np.float32)   # [dict_size]
W_dec = sae_data["W_dec"].astype(np.float32)   # [dict_size, d_in]
b_dec = sae_data["b_dec"].astype(np.float32)   # [d_in]
k_sae   = 128
d_in, dict_size = W_enc.shape
log(f"   d_in={d_in}  dict_size={dict_size}  k={k_sae}")


def sae_encode(x_np: np.ndarray) -> np.ndarray:
    """TopK SAE encode. x_np: [seq, d_in] → acts: [seq, dict_size] (sparse)."""
    pre = x_np @ W_enc + b_enc                               # [seq, dict_size]
    pre = np.maximum(pre, 0.0)
    # Zero all but top-k per position
    thresh = np.partition(pre, -k_sae, axis=-1)[:, -k_sae:].min(axis=-1, keepdims=True)
    return np.where(pre >= thresh, pre, 0.0)


# ── Load model ───────────────────────────────────────────────────────────────
log("\n[2] Loading model (may take ~30s)...")
from mlx_lm import load
model, tokenizer = load(MODEL_PATH)
mx.eval(model.parameters())
log("   Model loaded.")

# Extract attention geometry for head 20
attn15       = model.model.layers[HEAD_LAYER].self_attn
n_heads      = attn15.n_heads        # 24
n_kv_heads   = attn15.n_kv_heads     # 8
head_dim     = attn15.head_dim       # 128
kv_group     = n_heads // n_kv_heads # 3
kv_head_20   = HEAD_IDX // kv_group  # 6

# W_V for head 20's kv group: v_proj.weight[kv_head*head_dim : (kv_head+1)*head_dim, :]
# Shape (head_dim, d_in) — mlx-lm stores weights as (out, in)
# Cast to float32 first: bf16 arrays can't be directly converted via np.array()
def to_np(mlx_arr):
    return np.array(mlx_arr.astype(mx.float32))

W_V_20 = to_np(attn15.v_proj.weight[kv_head_20 * head_dim : (kv_head_20 + 1) * head_dim, :])
# W_O slice for head 20: o_proj.weight[:, HEAD_IDX*head_dim : (HEAD_IDX+1)*head_dim]
W_O_20 = to_np(attn15.o_proj.weight[:, HEAD_IDX * head_dim : (HEAD_IDX + 1) * head_dim])
log(f"   W_V_20 {W_V_20.shape}  W_O_20 {W_O_20.shape}")

# OV circuit: [d_in, d_in] mapping attended-token residual → written residual
# OV = W_V_20.T @ W_O_20.T   (d_in → head_dim → d_in)
OV = W_V_20.T @ W_O_20.T   # [d_in, d_in]

# ── Static write analysis: project SAE features through OV ───────────────────
# For each SAE feature j, the write vector when attending to a token with feat j:
#   write_j = W_dec[j] @ OV   ← direction that head 20 adds to query residual
# Project write_j onto every other SAE feature k:
#   proj_jk = (write_j @ W_dec[k]) / ||W_dec[k]||
log("\n[3] Computing OV-projected write features (static weight analysis)...")
W_dec_norms = np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8  # [dict_size, 1]
W_dec_unit  = W_dec / W_dec_norms                                   # [dict_size, d_in]

# write_vecs[j] = W_dec[j] @ OV  (shape: [dict_size, d_in])
# For top-K write features per source feature this would be dict_size^2 — too large.
# Instead compute: for each feature j, the L2 norm of write_vecs[j] (how strongly it writes)
# and the fraction of that write captured in the SAE basis.
write_vecs = W_dec @ OV                 # [dict_size, d_in]
write_norms = np.linalg.norm(write_vecs, axis=1)  # [dict_size]

# "Write alignment" = for each feat j, max cosine sim of write_vecs[j] with any W_dec[k]
# Expensive in full, so compute coarsely: what fraction of write_vecs is in W_dec span?
# Use: ||proj_W_dec(write_j)||_2 / ||write_j||  via random projection (approximate)
# Quick approximation: cosine sims of write_vecs with W_dec rows, take top-1 per row.
# For speed: batch in chunks of 512
CHUNK = 512
max_write_cosine = np.zeros(dict_size)
for start in range(0, dict_size, CHUNK):
    end = min(start + CHUNK, dict_size)
    wv_chunk = write_vecs[start:end]  # [chunk, d_in]
    wv_norm  = write_norms[start:end, None] + 1e-8
    # cosine with all W_dec rows: [chunk, dict_size]
    cos = (wv_chunk @ W_dec.T) / (wv_norm * W_dec_norms.T)
    max_write_cosine[start:end] = cos.max(axis=1)

top_write_by_norm = np.argsort(write_norms)[::-1][:TOP_K_DISPLAY]
log(f"   Top-{TOP_K_DISPLAY} features by |write_vec| (OV norm):")
write_summary = []
for fi in top_write_by_norm:
    write_summary.append({
        "feat_idx":      int(fi),
        "ov_write_norm": float(write_norms[fi]),
        "max_write_cos": float(max_write_cosine[fi]),
    })
    log(f"     feat_{fi:05d}: ov_norm={write_norms[fi]:.4f}  max_cos={max_write_cosine[fi]:.4f}")

# ── Hook layer 14 ────────────────────────────────────────────────────────────
log("\n[4] Installing hook on layer 14...")
_resid14 = {}

class _HookedLayer:
    """Thin wrapper to capture residual stream output after a layer."""
    def __init__(self, layer): self._layer = layer
    def __call__(self, x, mask=None, cache=None):
        out = self._layer(x, mask=mask, cache=cache)
        raw = out[0] if isinstance(out, tuple) else out
        _resid14["v"] = np.array(raw.astype(mx.float32))
        return out
    def __getattr__(self, a): return getattr(self._layer, a)

model.model.layers[SAE_LAYER] = _HookedLayer(model.model.layers[SAE_LAYER])
log("   Hook installed.")

# ── Load IOI dataset ──────────────────────────────────────────────────────────
log("\n[5] Loading IOI dataset...")
with open(IOI_PATH) as f:
    ioi_data = json.load(f)
examples = ioi_data["examples"][:N_EXAMPLES]
log(f"   Using {len(examples)} examples.")


def find_name_positions(token_strs, name):
    """Return sorted list of token-index positions where 'name' appears."""
    positions = []
    for i, tok in enumerate(token_strs):
        # match if this token starts or contains the name (handles sub-word splits)
        combined = tok.strip()
        if combined == name or combined.endswith(name) or combined.startswith(name):
            positions.append(i)
    return positions


# ── Main loop ─────────────────────────────────────────────────────────────────
log("\n[6] Running forward passes...")

feat_counts = {k: np.zeros(dict_size) for k in ("io_pos", "subj_pos", "query_pos")}
feat_vals   = {k: np.zeros(dict_size) for k in ("io_pos", "subj_pos", "query_pos")}
per_example = []
n_ok = 0

for ex in examples:
    prompt    = ex["prompt"]
    io_name   = ex["io_name"]
    subj_name = ex["subject_name"]

    enc       = tokenizer.encode(prompt)
    tok_strs  = [tokenizer.decode([t]) for t in enc]
    input_ids = mx.array([enc])

    io_pos    = find_name_positions(tok_strs, io_name)
    subj_pos  = find_name_positions(tok_strs, subj_name)
    # Fallback: scan for any token that contains the name
    if not io_pos:
        io_pos = [i for i, t in enumerate(tok_strs) if io_name in t]
    if not subj_pos:
        subj_pos = [i for i, t in enumerate(tok_strs) if subj_name in t]
    query_pos = [len(enc) - 1]

    try:
        out = model(input_ids)
        mx.eval(out)
    except Exception as e:
        log(f"   [!] Example {ex['id']} failed: {e}")
        continue

    if "v" not in _resid14:
        log(f"   [!] No capture for example {ex['id']}")
        continue

    resid = _resid14["v"][0]  # [seq_len, d_in]
    seq_len = resid.shape[0]

    ex_result = {
        "id": ex["id"], "prompt": prompt,
        "io_name": io_name, "subj_name": subj_name,
        "io_positions": io_pos, "subj_positions": subj_pos, "query_position": query_pos[0],
    }

    for key, positions in [("io_pos", io_pos), ("subj_pos", subj_pos), ("query_pos", query_pos)]:
        valid = [p for p in positions if p < seq_len]
        if not valid:
            ex_result[f"{key}_features"] = []
            continue
        mean_resid = np.mean(resid[valid], axis=0, keepdims=True)  # [1, d_in]
        acts = sae_encode(mean_resid)[0]  # [dict_size]
        top_idx = np.argsort(acts)[::-1][:TOP_K_DISPLAY]
        top_idx = top_idx[acts[top_idx] > 0]
        feat_counts[key][top_idx] += 1
        feat_vals[key][top_idx]   += acts[top_idx]
        ex_result[f"{key}_features"] = [
            {"feat_idx": int(fi), "act": float(acts[fi])}
            for fi in top_idx
        ]

    per_example.append(ex_result)
    n_ok += 1
    if n_ok % 10 == 0:
        log(f"   {n_ok}/{len(examples)} done...")

log(f"\n   Processed {n_ok} examples successfully.")

# ── Universal read features ───────────────────────────────────────────────────
log("\n[7] Computing universal features...")
universal_read = {}
for key in ("io_pos", "subj_pos", "query_pos"):
    freq     = feat_counts[key] / max(n_ok, 1)
    mean_act = feat_vals[key] / (feat_counts[key] + 1e-8)
    mask     = freq >= UNIVERSAL_THRESH
    idx      = np.where(mask)[0]
    idx      = idx[np.argsort(freq[idx])[::-1]][:TOP_K_DISPLAY]
    universal_read[key] = [
        {"feat_idx": int(fi), "freq": float(freq[fi]), "mean_act": float(mean_act[fi])}
        for fi in idx
    ]
    log(f"\n  [{key}] {len(idx)} universal features (freq > {UNIVERSAL_THRESH}):")
    for r in universal_read[key][:10]:
        log(f"    feat_{r['feat_idx']:05d}  freq={r['freq']:.2f}  mean_act={r['mean_act']:.3f}")

# ── Cross-position universality ────────────────────────────────────────────────
io_set    = {r["feat_idx"] for r in universal_read["io_pos"]}
subj_set  = {r["feat_idx"] for r in universal_read["subj_pos"]}
query_set = {r["feat_idx"] for r in universal_read["query_pos"]}

cross_io_query    = io_set   & query_set
cross_subj_query  = subj_set & query_set
truly_universal   = io_set   & subj_set & query_set

log(f"\n  IO ∩ query:           {sorted(cross_io_query)}")
log(f"  Subject ∩ query:      {sorted(cross_subj_query)}")
log(f"  IO ∩ Subject ∩ query: {sorted(truly_universal)}")

write_set = {w["feat_idx"] for w in write_summary}
bridge_features = truly_universal & write_set
log(f"  READ(all) ∩ WRITE:    {sorted(bridge_features)}  ← bridge candidates")

# ── Save ───────────────────────────────────────────────────────────────────────
log("\n[8] Saving results...")
summary = {
    "meta": {
        "model":              MODEL_PATH,
        "sae_layer":          SAE_LAYER,
        "head_layer":         HEAD_LAYER,
        "head_idx":           HEAD_IDX,
        "n_examples":         n_ok,
        "universal_thresh":   UNIVERSAL_THRESH,
        "top_k":              TOP_K_DISPLAY,
        "kv_head_for_h20":    int(kv_head_20),
        "kv_group_size":      int(kv_group),
        "description": (
            "Read features: layer-14 SAE features active in top-20 at IO/Subject/query "
            "token positions, averaged over examples. Universal = appears for ≥40% of examples. "
            "Write features: SAE-feature write-vectors obtained via OV circuit "
            "(W_dec[j] @ W_V_20.T @ W_O_20.T), ranked by L2 norm."
        ),
    },
    "universal_read": universal_read,
    "cross_position": {
        "io_and_query":          sorted(cross_io_query),
        "subj_and_query":        sorted(cross_subj_query),
        "io_and_subj_and_query": sorted(truly_universal),
    },
    "top_write_features": write_summary,
    "bridge_features": sorted(bridge_features),
}

with open(OUT_DIR / "bridge_results.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(OUT_DIR / "per_example.json", "w") as f:
    json.dump(per_example, f, indent=2)

log(f"   Saved bridge_results.json and per_example.json to {OUT_DIR}")
log("\n=== Done ===")
