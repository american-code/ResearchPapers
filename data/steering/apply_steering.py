"""
Steering application harness for Llama-3.2-3B.

Adds alpha × steering_vector to the residual stream at layers 8, 16, 24.
For each concept and alpha in [-2, -1, -0.5, 0, 0.5, 1, 2], generates 20
completions from neutral prompts.

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

    def __init__(self, layer, vecs_mx, alpha):
        # vecs_mx: list of mx.array each [hidden_size]
        self._layer = layer
        self._vecs = vecs_mx
        self._alpha = float(alpha)

    def __call__(self, x, mask=None, cache=None):
        h = self._layer(x, mask, cache=cache)
        if self._alpha != 0.0:
            for vec in self._vecs:
                h = h + self._alpha * vec.astype(h.dtype)
        return h

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _patch(llama_model, layer_to_vecs, alpha):
    """Replace target layers with steering wrappers; return originals."""
    originals = {}
    for idx, vecs in layer_to_vecs.items():
        originals[idx] = llama_model.layers[idx]
        llama_model.layers[idx] = _SteeringLayer(llama_model.layers[idx], vecs, alpha)
    return originals


def _restore(llama_model, originals):
    for idx, orig in originals.items():
        llama_model.layers[idx] = orig


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

    print(f"Concepts: {concepts}")
    print(f"Steering layers: {steering_layers}")
    print(f"Alphas: {ALPHAS}")

    sampler = make_sampler(temp=TEMPERATURE)

    output = {
        "model": MODEL_ID,
        "steering_layers": steering_layers,
        "alphas": ALPHAS,
        "n_completions": len(NEUTRAL_PROMPTS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "results": {},
    }

    total = len(concepts) * len(ALPHAS) * len(NEUTRAL_PROMPTS)
    done = 0
    t_start = time.time()

    for concept in concepts:
        print(f"\n=== {concept} ===")
        output["results"][concept] = {}

        # Pre-convert vectors to mx.arrays (float32 → converted per call)
        layer_to_vecs = {}
        for layer_idx in steering_layers:
            vec_np = np.array(vdata["vectors"][concept][f"layer_{layer_idx}"],
                              dtype=np.float32)
            layer_to_vecs[layer_idx] = [mx.array(vec_np)]

        for alpha in ALPHAS:
            print(f"  alpha={alpha:+.1f} ", end="", flush=True)
            completions = []

            originals = _patch(llama_model, layer_to_vecs, alpha)

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
                    # mlx_lm.generate returns generated tokens only (no prompt)
                    completion = text
                    full_text = prompt + text
                except Exception as exc:
                    completion = f"[ERROR: {exc}]"
                    full_text = prompt + completion

                completions.append({
                    "prompt": prompt,
                    "completion": completion,
                    "full_text": full_text,
                })
                done += 1
                print(".", end="", flush=True)

            _restore(llama_model, originals)

            output["results"][concept][str(alpha)] = completions
            elapsed = time.time() - t_start
            remaining = (total - done) / (done / elapsed) if done else 0
            print(f" [{done}/{total}, ~{remaining:.0f}s left]")

    output["elapsed_seconds"] = round(time.time() - t_start, 1)
    OUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {OUT_PATH}  ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(f"Total time: {output['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
