#!/usr/bin/env python3
"""
Single-process reference capture for distributed PoC validation.

Runs all layers of Llama-3.2-3B and saves:
  - handoff activations (output of layer LAYER_SPLIT-1)
  - final activations   (output of last layer)

These are the ground-truth tensors the distributed run must match.

Args (JSON, sys.argv[1]):
  token_ids_path  .npy file of int32 token IDs
  handoff_out     .npy path for layer-(layer_split-1) output
  final_out       .npy path for last-layer output
  layer_split     the split point used by the distributed run
"""
import json, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"


def main():
    args = json.loads(sys.argv[1])
    token_ids_path = args["token_ids_path"]
    handoff_out = args["handoff_out"]
    final_out = args["final_out"]
    layer_split = args["layer_split"]

    t0 = time.time()
    token_ids = np.load(token_ids_path)
    n_tokens = len(token_ids)

    print(f"REFERENCE: {n_tokens} tokens, layer_split={layer_split}", flush=True)
    print(f"REFERENCE: loading {MODEL_ID} ...", flush=True)
    model, _ = load(MODEL_ID)
    num_layers = len(model.model.layers)
    print(f"REFERENCE: model loaded ({num_layers} layers)", flush=True)

    llama = model.model
    inp = mx.array([token_ids.tolist()])   # [1, n_tokens]
    h = llama.embed_tokens(inp)

    fa_mask = create_attention_mask(h, None)
    swa_mask = None
    if hasattr(llama, "swa_idx") and llama.swa_idx is not None:
        swa_mask = create_attention_mask(h, None, window_size=llama.sliding_window)

    for i, layer in enumerate(llama.layers):
        mask = swa_mask if getattr(layer, "use_sliding", False) else fa_mask
        h = layer(h, mask, cache=None)

        if i == layer_split - 1:
            h32 = h[0].astype(mx.float32)
            mx.eval(h32)
            np.save(handoff_out, np.array(h32, dtype=np.float16))
            print(f"REFERENCE: handoff saved at layer {i}", flush=True)

    h32 = h[0].astype(mx.float32)
    mx.eval(h32)
    np.save(final_out, np.array(h32, dtype=np.float16))

    elapsed = time.time() - t0
    print(
        f"REFERENCE_DONE elapsed={elapsed:.1f}s n_tokens={n_tokens} num_layers={num_layers}",
        flush=True,
    )


main()
