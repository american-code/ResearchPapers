#!/usr/bin/env python3
"""
Find top-20 max-activating token sequences for top-200 SAE features by activation frequency.

Two-pass algorithm (all computation stays in MLX GPU; only tiny results copied to numpy):
  Pass 1 — for each batch, compute TopK-gated frequency update [dict_size] (~64 KB).
  Pass 2 — for top-200 features only, collect gated values [batch, 200] (~1.6 MB).

Avoids materializing the full [batch, 16384] dense acts matrix in numpy (was 524 MB/batch).
Uses @mx.compile to fuse ops and prevent lazy-graph pile-up across batches.
"""

import gc
import heapq
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

WORKSPACE = Path("/Users/melton/ResearchPapers")

CONFIGS = {
    "llama3b": {
        "model_id":   "mlx-community/Llama-3.2-3B-bf16",
        "acts_dir":   WORKSPACE / "data/activations/llama-3b-layer14",
        "sae_ckpt":   WORKSPACE / "data/sae-runs/llama-3b-layer14/checkpoint_step_010000.npz",
        "output":     WORKSPACE / "data/sae-analysis/llama3b-feature-examples.json",
        "k":          128,
    },
    "qwen3b": {
        "model_id":   "mlx-community/Qwen2.5-3B-bf16",
        "acts_dir":   WORKSPACE / "data/activations/qwen-3b-layer18",
        "sae_ckpt":   WORKSPACE / "data/sae-runs/qwen-3b-layer18/checkpoint_step_020000.npz",
        "output":     WORKSPACE / "data/sae-analysis/qwen3b-feature-examples.json",
        "k":          128,
    },
}

MISTRAL_OUTPUT = WORKSPACE / "data/sae-analysis/mistral7b-feature-examples.json"

TOP_FEATURES   = 200
TOP_EXAMPLES   = 20
CONTEXT_TOKENS = 20   # tokens on each side of the max-activating position
BATCH          = 2000  # smaller batch = less peak memory, avoids MLX pile-up


# ── MLX compiled kernels ─────────────────────────────────────────────────────

def make_freq_fn(k: int):
    """Returns a compiled function: x → freq_update [dict_size] int32."""
    @mx.compile
    def freq_fn(x, W_enc, b_enc, b_dec):
        pre    = (x - b_dec) @ W_enc + b_enc          # [B, dict_size]
        top_v  = mx.topk(pre, k=k, axis=-1)           # [B, k] values
        thresh = mx.min(top_v, axis=-1, keepdims=True) # [B, 1]
        active = (pre >= thresh).astype(mx.int32)      # [B, dict_size] 0/1
        return active.sum(axis=0)                      # [dict_size]
    return freq_fn


def make_top200_fn(k: int):
    """Returns a compiled function: x, top_feats → gated values [B, 200]."""
    @mx.compile
    def top200_fn(x, W_enc, b_enc, b_dec, top_feats):
        pre    = (x - b_dec) @ W_enc + b_enc
        top_v  = mx.topk(pre, k=k, axis=-1)
        thresh = mx.min(top_v, axis=-1, keepdims=True)
        # Select only top-200 features
        pre200 = pre[:, top_feats]                     # [B, 200]
        gate   = (pre200 >= thresh).astype(mx.float32) # [B, 200]
        return pre200 * gate                           # [B, 200] gated values
    return top200_fn


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_raw_acts(acts_dir: Path):
    meta  = json.loads((acts_dir / "metadata.json").read_text())
    n_tok = meta["n_tokens_written"]
    d_hid = meta["hidden_size"]
    arr   = np.fromfile(str(acts_dir / "activations.npy"), dtype=np.float16)
    return arr.reshape(n_tok, d_hid), meta


def get_token_ids(model_id: str, n_tokens: int) -> tuple:
    """Re-tokenize wikitext-103-raw-v1 in the exact same order as during collection."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"  Loading tokenizer: {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)

    print("  Streaming wikitext-103-raw-v1 (train)…")
    ds = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1",
        split="train", trust_remote_code=False,
    )

    CHUNK    = 512
    ids_arr  = np.zeros(n_tokens, dtype=np.int32)
    written  = 0
    buf: list[int] = []

    for row in ds:
        if written >= n_tokens:
            break
        text = row["text"].strip()
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= CHUNK and written < n_tokens:
            rem    = n_tokens - written
            clen   = min(CHUNK, rem)
            ids_arr[written : written + clen] = buf[:clen]
            buf    = buf[clen:]
            written += clen

    if written < n_tokens and buf:
        rem = n_tokens - written
        ids_arr[written : written + rem] = buf[:rem]

    print(f"  Token IDs reconstructed: {written:,}")
    return ids_arr, tok


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(name: str, cfg: dict) -> None:
    print(f"\n{'='*62}\n  {name}\n{'='*62}")
    t0 = time.time()

    # Load SAE weights into MLX (stays on GPU throughout)
    ckpt      = np.load(str(cfg["sae_ckpt"]))
    W_enc_mx  = mx.array(ckpt["W_enc"].astype(np.float32))  # [d_in, dict_size]
    b_enc_mx  = mx.array(ckpt["b_enc"].astype(np.float32))  # [dict_size]
    b_dec_mx  = mx.array(ckpt["b_dec"].astype(np.float32))  # [d_in]
    dict_size = int(ckpt["W_enc"].shape[1])
    k         = cfg["k"]
    ckpt_step = int(cfg["sae_ckpt"].stem.split("_")[-1])
    mx.eval(W_enc_mx, b_enc_mx, b_dec_mx)
    print(f"  SAE d_in={ckpt['W_enc'].shape[0]}  dict_size={dict_size}  k={k}  step={ckpt_step:,}")

    # Pre-compile kernels
    freq_fn   = make_freq_fn(k)
    top200_fn = make_top200_fn(k)

    # Load activations into RAM (float16, stays as numpy)
    acts_np, meta = load_raw_acts(cfg["acts_dir"])
    n_tokens      = acts_np.shape[0]
    n_batches     = (n_tokens + BATCH - 1) // BATCH
    print(f"  Activations: {acts_np.shape}  {acts_np.dtype}  ({acts_np.nbytes/1e9:.2f} GB)")
    print(f"  Batches: {n_batches} × {BATCH}")

    # ── Pass 1: activation frequency ─────────────────────────────────────────
    print(f"\n  Pass 1 — frequency ({n_batches} batches)…")
    freq  = np.zeros(dict_size, dtype=np.int64)
    t_p1  = time.time()

    for b_idx in range(n_batches):
        start = b_idx * BATCH
        end   = min(start + BATCH, n_tokens)
        x     = mx.array(acts_np[start:end].astype(np.float32))
        fu    = freq_fn(x, W_enc_mx, b_enc_mx, b_dec_mx)
        mx.eval(fu)
        freq += np.array(fu).astype(np.int64)   # only 64 KB copied per batch
        del x, fu
        if b_idx % 50 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t_p1
            rate    = (end) / elapsed if elapsed > 0 else 0
            print(f"    batch {b_idx+1}/{n_batches}  {end:,}/{n_tokens:,} tok"
                  f"  ({rate:.0f} tok/s)", flush=True)

    top_feats    = np.argsort(freq)[-TOP_FEATURES:][::-1]  # descending by freq
    top_feats_mx = mx.array(top_feats.astype(np.int32))
    mx.eval(top_feats_mx)
    print(f"  Top-200 freq: {freq[top_feats[0]]:,} … {freq[top_feats[-1]]:,}")
    print(f"  Pass 1 time: {(time.time()-t_p1)/60:.2f} min")

    # ── Pass 2: top-20 examples per feature ──────────────────────────────────
    print(f"\n  Pass 2 — top examples ({n_batches} batches)…")
    # Min-heap per feature: (val, global_token_idx) — smallest val evicted first
    heaps: list[list] = [[] for _ in range(TOP_FEATURES)]
    t_p2  = time.time()

    for b_idx in range(n_batches):
        start    = b_idx * BATCH
        end      = min(start + BATCH, n_tokens)
        x        = mx.array(acts_np[start:end].astype(np.float32))
        vals200  = top200_fn(x, W_enc_mx, b_enc_mx, b_dec_mx, top_feats_mx)
        mx.eval(vals200)
        batch_v  = np.array(vals200)   # [batch_size, 200] ~1.6 MB
        del x, vals200

        for fi in range(TOP_FEATURES):
            col     = batch_v[:, fi]
            nz_mask = col > 0.0
            if not nz_mask.any():
                continue
            hp = heaps[fi]
            for v, gp in zip(col[nz_mask].tolist(),
                              (np.where(nz_mask)[0] + start).tolist()):
                if len(hp) < TOP_EXAMPLES:
                    heapq.heappush(hp, (v, gp))
                elif v > hp[0][0]:
                    heapq.heapreplace(hp, (v, gp))

        if b_idx % 50 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t_p2
            rate    = end / elapsed if elapsed > 0 else 0
            print(f"    batch {b_idx+1}/{n_batches}  {end:,}/{n_tokens:,} tok"
                  f"  ({rate:.0f} tok/s)", flush=True)

    print(f"  Pass 2 time: {(time.time()-t_p2)/60:.2f} min")

    # ── Token reconstruction ──────────────────────────────────────────────────
    print("\n  Reconstructing token sequences…")
    token_ids, tok = get_token_ids(cfg["model_id"], n_tokens)

    # ── Build output ──────────────────────────────────────────────────────────
    out = {
        "model":                    cfg["model_id"],
        "sae_checkpoint":           str(cfg["sae_ckpt"].relative_to(WORKSPACE)),
        "sae_training_step":        ckpt_step,
        "n_tokens_analyzed":        int(n_tokens),
        "dict_size":                int(dict_size),
        "k":                        int(k),
        "top_features_n":           TOP_FEATURES,
        "top_examples_per_feature": TOP_EXAMPLES,
        "context_window_tokens":    CONTEXT_TOKENS,
        "features":                 [],
    }

    for hi, fid in enumerate(top_feats):
        fid      = int(fid)
        hp       = sorted(heaps[hi], reverse=True)
        examples = []
        for val, gpos in hp:
            ctx_s = max(0, gpos - CONTEXT_TOKENS)
            ctx_e = min(n_tokens, gpos + CONTEXT_TOKENS + 1)
            ctx   = token_ids[ctx_s:ctx_e].tolist()
            try:
                text = tok.decode(ctx, skip_special_tokens=False)
            except Exception:
                text = ""
            examples.append({
                "token_pos":     gpos,
                "activation":    round(float(val), 4),
                "context_text":  text,
                "target_offset": gpos - ctx_s,
            })

        out["features"].append({
            "feature_id":               fid,
            "activation_frequency":     int(freq[fid]),
            "activation_frequency_pct": round(float(freq[fid]) / n_tokens * 100, 4),
            "max_activating_examples":  examples,
        })

    cfg["output"].parent.mkdir(parents=True, exist_ok=True)
    cfg["output"].write_text(json.dumps(out, indent=2))
    size_mb = cfg["output"].stat().st_size / 1e6
    total   = (time.time() - t0) / 60
    print(f"\n  Saved → {cfg['output'].name}  ({size_mb:.1f} MB)")
    print(f"  Total time: {total:.1f} min")

    # Clean up MLX state between models
    del W_enc_mx, b_enc_mx, b_dec_mx, top_feats_mx
    gc.collect()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    MISTRAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MISTRAL_OUTPUT.write_text(json.dumps({
        "error": "no_sae_checkpoint",
        "model": "mlx-community/Mistral-7B-v0.3-4bit",
        "activations_available": True,
        "activations_path": "data/activations/mistral-7b-layer16",
        "note": (
            "Mistral-7B activations were collected (500k tokens, layer 16, 4096 hidden). "
            "No SAE has been trained on these activations — data/sae-runs/ contains only "
            "llama-3b-layer14 and qwen-3b-layer18 checkpoints. "
            "To train: python data/sae-runs/train_sae.py "
            "--activations data/activations/mistral-7b-layer16 "
            "--output data/sae-runs/mistral-7b-layer16 "
            "(~2 hours for 10k steps), then re-run this script."
        ),
    }, indent=2))
    print("Wrote error stub → mistral7b-feature-examples.json")

    for name, cfg in CONFIGS.items():
        analyze(name, cfg)

    print("\n\nAll done.")
