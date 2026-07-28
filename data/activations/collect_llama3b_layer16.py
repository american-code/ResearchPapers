"""
Collect residual stream activations from Llama-3.2-3B at 50% layer depth.

NOTE: Llama-3.2-3B has 28 layers (not 32). 50% depth = layer 14 (0-indexed).
The output directory is named 'llama-3b-layer16' per task spec, but layer 14
is the actual 50% midpoint.

Output: data/activations/llama-3b-layer16/
  activations.npy    -- float16 memmap, shape [n_tokens, 3072]
  metadata.json      -- run config and stats
"""

import json
import time
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask
from datasets import load_dataset

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID      = "mlx-community/Llama-3.2-3B-bf16"
TARGET_LAYER  = 14          # layer 14 of 28 = 50% depth
TARGET_TOKENS = 500_000
CHUNK_TOKENS  = 512         # tokens per forward-pass chunk
HIDDEN_SIZE   = 3072
OUT_DIR       = Path(__file__).parent / "llama-3b-layer16"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID}...")
t0 = time.time()
model, tokenizer = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")

num_layers = len(model.model.layers)
print(f"Model has {num_layers} layers; collecting at layer {TARGET_LAYER} "
      f"({TARGET_LAYER/num_layers*100:.0f}% depth)")
assert TARGET_LAYER < num_layers

# ── Corpus: stream wikitext-103-raw-v1 row by row ────────────────────────────
print("\nStreaming wikitext-103-raw-v1 (train split)...")
ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                  split="train", trust_remote_code=False)

# ── Allocate output memmap ────────────────────────────────────────────────────
act_path = OUT_DIR / "activations.npy"
size_gb  = TARGET_TOKENS * HIDDEN_SIZE * 2 / 1e9
print(f"\nAllocating memmap {act_path}  [{TARGET_TOKENS}, {HIDDEN_SIZE}] float16  (~{size_gb:.2f} GB)")
acts = np.memmap(act_path, dtype="float16", mode="w+",
                 shape=(TARGET_TOKENS, HIDDEN_SIZE))

# ── Activation extraction ─────────────────────────────────────────────────────
def extract_activations(chunk_ids: list) -> np.ndarray:
    """Forward pass through layers 0..TARGET_LAYER, return residual stream."""
    llama_model = model.model
    inp = mx.array([chunk_ids])          # [1, seq_len]
    h   = llama_model.embed_tokens(inp)

    fa_mask  = create_attention_mask(h, None)
    swa_mask = None
    if llama_model.swa_idx is not None:
        swa_mask = create_attention_mask(h, None, window_size=llama_model.sliding_window)

    for i, layer in enumerate(llama_model.layers):
        mask = swa_mask if layer.use_sliding else fa_mask
        h = layer(h, mask, cache=None)
        if i == TARGET_LAYER:
            # bfloat16 has no numpy equivalent; cast to float32 in MLX first
            h32 = h[0].astype(mx.float32)
            mx.eval(h32)
            return np.array(h32).astype(np.float16)   # [seq_len, hidden]

    raise RuntimeError("Target layer not reached")

# ── Main collection loop (streaming tokenization) ─────────────────────────────
print(f"\nCollecting {TARGET_TOKENS:,} tokens in {CHUNK_TOKENS}-token chunks ...")

written     = 0
chunk_count = 0
token_buf   = []   # rolling buffer of token ids not yet processed
t_start     = time.time()

for row in ds:
    if written >= TARGET_TOKENS:
        break

    text = row["text"].strip()
    if not text:
        continue

    ids = tokenizer.encode(text, add_special_tokens=False)
    token_buf.extend(ids)

    # Drain buffer in CHUNK_TOKENS blocks
    while len(token_buf) >= CHUNK_TOKENS and written < TARGET_TOKENS:
        remaining  = TARGET_TOKENS - written
        chunk_len  = min(CHUNK_TOKENS, remaining)
        chunk_ids  = token_buf[:chunk_len]
        token_buf  = token_buf[chunk_len:]

        chunk_acts = extract_activations(chunk_ids)   # [chunk_len, hidden]
        acts[written : written + chunk_len] = chunk_acts
        written     += chunk_len
        chunk_count += 1

        if chunk_count % 50 == 0 or written >= TARGET_TOKENS:
            elapsed = time.time() - t_start
            rate    = written / elapsed if elapsed > 0 else 0
            eta     = (TARGET_TOKENS - written) / rate if rate > 0 else 0
            print(f"  {written:>7,}/{TARGET_TOKENS:,}  "
                  f"({written/TARGET_TOKENS*100:.1f}%)  "
                  f"{rate:.0f} tok/s  ETA {eta:.0f}s")

# Flush any leftover tokens (< CHUNK_TOKENS) to fill the target
if written < TARGET_TOKENS and token_buf:
    remaining = TARGET_TOKENS - written
    chunk_ids = token_buf[:remaining]
    chunk_acts = extract_activations(chunk_ids)
    acts[written : written + len(chunk_ids)] = chunk_acts
    written += len(chunk_ids)

acts.flush()
elapsed_total = time.time() - t_start

# ── Save metadata ─────────────────────────────────────────────────────────────
meta = {
    "model":             MODEL_ID,
    "num_layers":        num_layers,
    "target_layer":      TARGET_LAYER,
    "layer_depth_pct":   round(TARGET_LAYER / num_layers * 100, 1),
    "hidden_size":       HIDDEN_SIZE,
    "n_tokens_written":  int(written),
    "chunk_size":        CHUNK_TOKENS,
    "corpus":            "Salesforce/wikitext wikitext-103-raw-v1 (train)",
    "activations_file":  "activations.npy",
    "activations_dtype": "float16",
    "activations_shape": [int(written), HIDDEN_SIZE],
    "elapsed_seconds":   round(elapsed_total, 1),
    "tokens_per_second": round(written / elapsed_total, 0) if elapsed_total > 0 else 0,
    "note": (
        "Layer index 14 is the 50% midpoint of 28 layers. "
        "Directory named 'llama-3b-layer16' per task spec; "
        "Llama-3.2-3B has 28 layers (not 32)."
    ),
}

meta_path = OUT_DIR / "metadata.json"
meta_path.write_text(json.dumps(meta, indent=2))

print(f"\n=== Done ===")
print(f"  Tokens collected : {written:,}")
print(f"  Output shape     : [{written}, {HIDDEN_SIZE}] float16")
print(f"  Activations file : {act_path}  ({act_path.stat().st_size/1e9:.2f} GB)")
print(f"  Metadata file    : {meta_path}")
print(f"  Elapsed          : {elapsed_total:.1f}s  ({written/elapsed_total:.0f} tok/s)")
