"""
Steering application harness for Llama-3.2-3B — per-layer isolation.

For each concept × layer × alpha, applies alpha × steering_vector at
EXACTLY ONE layer and generates 20 completions from neutral prompts.
This enables genuine per-layer comparison (previously all three layers
were patched simultaneously, making layer-level analysis impossible).

Layers: 8, 16, 24   (from vectors-llama3b.json)
Alphas: [-2, -1, -0.5, 0, 0.5, 1, 2]
Completions per cell: 20

Output structure:
  results[concept][str(layer)][str(alpha)] = list of completion dicts

Output: data/steering/samples-llama3b.json
"""

import json
import time
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"
VECTORS_PATH = Path(__file__).parent / "vectors-llama3b.json"
OUT_PATH = Path(__file__).parent / "samples-llama3b.json"
CKPT_PATH = Path(__file__).parent / "samples-llama3b.ckpt.json"

ALPHAS = [-2, -1, -0.5, 0, 0.5, 1, 2]
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8

NEUTRAL_PROMPTS = [
    "The",
    "Today",
    "In the",
    "There",
    "It was",
    "When",
    "A person",
    "The world",
    "Once upon a time,",
    "Scientists have found",
    "The restaurant",
    "After work,",
    "The meeting",
    "She walked into",
    "He decided to",
    "They noticed",
    "The weather",
    "At the park,",
    "Looking at the stars,",
    "The old building",
]

assert len(NEUTRAL_PROMPTS) == 20, "Must have exactly 20 neutral prompts"


class _SteeringLayer:
    """Wraps a transformer layer, adding alpha × steering_vec to its output."""

    def __init__(self, layer, vec_mx, alpha):
        self._layer = layer
        self._vec = vec_mx      # single mx.array [hidden_size]
        self._alpha = float(alpha)

    def __call__(self, x, mask=None, cache=None):
        h = self._layer(x, mask, cache=cache)
        if self._alpha != 0.0:
            h = h + self._alpha * self._vec.astype(h.dtype)
        return h

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _patch_one(llama_model, layer_idx, vec_mx, alpha):
    """Patch exactly one layer; return the original."""
    orig = llama_model.layers[layer_idx]
    llama_model.layers[layer_idx] = _SteeringLayer(orig, vec_mx, alpha)
    return orig


def _restore_one(llama_model, layer_idx, orig):
    llama_model.layers[layer_idx] = orig


def main():
    mx.random.seed(42)

    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"Loaded in {time.time() - t0:.1f}s")

    llama_model = model.model

    vdata = json.loads(VECTORS_PATH.read_text())
    concepts = list(vdata["vectors"].keys())
    steering_layers = vdata["target_layers"]  # [8, 16, 24]

    print(f"Concepts:        {concepts}")
    print(f"Steering layers: {steering_layers}")
    print(f"Alphas:          {ALPHAS}")
    print(f"Per-layer isolation: ONE layer patched at a time")

    sampler = make_sampler(temp=TEMPERATURE)

    # Load checkpoint if resuming
    if CKPT_PATH.exists():
        print(f"\nResuming from checkpoint {CKPT_PATH}")
        output = json.loads(CKPT_PATH.read_text())
    else:
        output = {
            "model": MODEL_ID,
            "per_layer_isolation": True,
            "steering_layers": steering_layers,
            "alphas": ALPHAS,
            "n_completions": len(NEUTRAL_PROMPTS),
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "results": {},
        }

    total = len(concepts) * len(steering_layers) * len(ALPHAS) * len(NEUTRAL_PROMPTS)
    t_start = time.time()

    for concept in concepts:
        if concept not in output["results"]:
            output["results"][concept] = {}

        # Pre-convert vectors for all layers of this concept
        layer_vecs = {}
        for layer_idx in steering_layers:
            vec_np = np.array(
                vdata["vectors"][concept][f"layer_{layer_idx}"], dtype=np.float32
            )
            layer_vecs[layer_idx] = mx.array(vec_np)

        for layer_idx in steering_layers:
            layer_key = str(layer_idx)
            if layer_key not in output["results"][concept]:
                output["results"][concept][layer_key] = {}

            vec_mx = layer_vecs[layer_idx]
            print(f"\n=== {concept}  layer={layer_idx} ===")

            for alpha in ALPHAS:
                alpha_key = str(alpha)
                if alpha_key in output["results"][concept][layer_key]:
                    already = len(output["results"][concept][layer_key][alpha_key])
                    print(f"  alpha={alpha:+.1f}  [SKIP — {already} completions already]")
                    continue

                print(f"  alpha={alpha:+.1f} ", end="", flush=True)
                completions = []

                if alpha != 0:
                    orig = _patch_one(llama_model, layer_idx, vec_mx, alpha)

                for prompt in NEUTRAL_PROMPTS:
                    try:
                        text = generate(
                            model,
                            tokenizer,
                            prompt=prompt,
                            max_tokens=MAX_NEW_TOKENS,
                            sampler=sampler,
                            verbose=False,
                        )
                        completion = text
                        full_text = prompt + text
                    except Exception as exc:
                        completion = f"[ERROR: {exc}]"
                        full_text = prompt + completion

                    completions.append(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "full_text": full_text,
                        }
                    )
                    print(".", end="", flush=True)

                if alpha != 0:
                    _restore_one(llama_model, layer_idx, orig)

                output["results"][concept][layer_key][alpha_key] = completions

                done_cells = sum(
                    1
                    for c in concepts
                    for l in [str(li) for li in steering_layers]
                    for a in [str(x) for x in ALPHAS]
                    if c in output["results"]
                    and l in output["results"].get(c, {})
                    and a in output["results"][c].get(l, {})
                )
                done_prompts = done_cells * len(NEUTRAL_PROMPTS)
                elapsed = time.time() - t_start
                rate = done_prompts / elapsed if elapsed > 0 else 0
                remaining = (total - done_prompts) / rate if rate > 0 else 0
                print(f" [{done_prompts}/{total} prompts, ~{remaining:.0f}s left]")

            # Checkpoint after each (concept, layer) pair
            CKPT_PATH.write_text(json.dumps(output, indent=2))
            print(f"  [checkpoint saved]")

    output["elapsed_seconds"] = round(time.time() - t_start, 1)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    if CKPT_PATH.exists():
        CKPT_PATH.unlink()
    print(f"\nSaved to {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"Total time: {output['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
