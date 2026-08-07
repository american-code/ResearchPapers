"""
Steering behavior and side-effect scoring for Llama-3.2-3B — per-layer isolation.

Reads samples-llama3b.json (per-layer structure from apply_steering.py),
scores each (concept, layer, alpha) cell on:
  - keyword_hit_rate     : fraction of completions with ≥1 target keyword
  - keyword_density      : keyword occurrences per 100 words
  - side_effect_ppl      : PPL on math/reasoning probes under steered model
  - fluency_ppl          : PPL on neutral sentences under steered model

Then writes:
  data/steering/scores-llama3b.json   — per-cell scores
  data/steering/benchmark-summary.json — aggregated table with real dominant-layer
"""

import json
import math
import time
import numpy as np
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

MODEL_ID     = "mlx-community/Llama-3.2-3B-bf16"
VECTORS_PATH = Path(__file__).parent / "vectors-llama3b.json"
SAMPLES_PATH = Path(__file__).parent / "samples-llama3b.json"
SCORES_PATH  = Path(__file__).parent / "scores-llama3b.json"
SUMMARY_PATH = Path(__file__).parent / "benchmark-summary.json"

# ── Keyword sets ──────────────────────────────────────────────────────────────

TARGET_KEYWORDS = {
    "positive_sentiment": [
        "great", "wonderful", "amazing", "excellent", "fantastic",
        "perfect", "beautiful", "lovely", "incredible", "outstanding",
        "happy", "joyful", "delightful", "splendid", "superb",
        "love", "joy", "best", "awesome", "brilliant",
    ],
    "negative_sentiment": [
        "terrible", "awful", "horrible", "dreadful", "atrocious",
        "disappointing", "frustrating", "miserable", "disgusting", "appalling",
        "bad", "poor", "worst", "pathetic", "ugly",
        "hate", "fear", "pain", "suffering", "failure",
    ],
    "french_language": [
        "vous", "nous", "est", "très", "bien", "que", "pas",
        "qui", "pour", "avec", "bonjour", "merci", "oui", "non",
        "je", "la", "le", "les", "une", "dans",
    ],
    "python_coding": [
        "def ", "import ", "class ", "return ", "print(",
        "if ", "for ", "while ", "try:", "lambda",
        "```python", "numpy", "torch", ".append(", "isinstance(",
        "True", "False", "None", "self.", "elif ",
    ],
    "refusal_behavior": [
        "cannot", "can't", "unable", "decline", "sorry",
        "apologize", "refuse", "won't", "inappropriate", "harmful",
        "outside", "guidelines", "policy", "safety", "ethical",
        "I'm not", "I am not", "I will not", "I cannot", "I can't",
    ],
}

# ── Probe sentences ───────────────────────────────────────────────────────────

MATH_PROBES = [
    "What is 12 plus 15? The answer is 27.",
    "If there are 8 apples and you eat 3, you have 5 left.",
    "A rectangle with width 4 and height 6 has area 24.",
    "7 times 8 equals 56.",
    "The square root of 144 is 12.",
    "If all birds have wings and a robin is a bird, then a robin has wings.",
    "100 divided by 4 equals 25.",
    "2 plus 2 equals 4, and 3 plus 3 equals 6.",
    "If today is Monday, then tomorrow is Tuesday.",
    "A triangle has 3 sides and its angles sum to 180 degrees.",
]

FLUENCY_SENTENCES = [
    "The sun rises in the east and sets in the west every day.",
    "She walked to the store to buy some bread and milk.",
    "The weather forecast predicts rain for most of the afternoon.",
    "He opened the book and began reading the first chapter slowly.",
    "The children played outside in the park until evening came.",
    "Scientists study the natural world to understand how things work.",
    "The train arrived at the station exactly on schedule this morning.",
    "She prepared a simple meal of pasta with vegetables and sauce.",
    "The library was quiet except for the soft sound of turning pages.",
    "He sent an email to his colleague about the meeting tomorrow.",
]


# ── Steering layer wrapper ────────────────────────────────────────────────────

class _SteeringLayer:
    def __init__(self, layer, vec_mx, alpha):
        self._layer = layer
        self._vec   = vec_mx
        self._alpha = float(alpha)

    def __call__(self, x, mask=None, cache=None):
        h = self._layer(x, mask, cache=cache)
        if self._alpha != 0.0:
            h = h + self._alpha * self._vec.astype(h.dtype)
        return h

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _patch_one(llama_model, layer_idx, vec_mx, alpha):
    orig = llama_model.layers[layer_idx]
    llama_model.layers[layer_idx] = _SteeringLayer(orig, vec_mx, alpha)
    return orig


def _restore_one(llama_model, layer_idx, orig):
    llama_model.layers[layer_idx] = orig


# ── Scoring helpers ───────────────────────────────────────────────────────────

def keyword_hit_rate(texts, keywords):
    lo_kws = [k.lower() for k in keywords]
    hits = [1.0 if any(kw in t.lower() for kw in lo_kws) else 0.0 for t in texts]
    return float(np.mean(hits))


def keyword_density(texts, keywords):
    lo_kws = [k.lower() for k in keywords]
    densities = []
    for t in texts:
        words = t.split()
        if not words:
            densities.append(0.0)
            continue
        lo = t.lower()
        hits = sum(lo.count(kw) for kw in lo_kws)
        densities.append(hits / len(words) * 100)
    return float(np.mean(densities))


def compute_perplexity(sentences, model, tokenizer):
    """Average per-token PPL on sentences under current model state."""
    total_nll  = 0.0
    total_toks = 0

    for sent in sentences:
        ids = tokenizer.encode(sent, add_special_tokens=True)
        if len(ids) < 2:
            continue
        inp     = mx.array([ids])
        logits  = model(inp)           # [1, seq_len, vocab]
        T       = len(ids) - 1
        targets = mx.array(ids[1:])
        lgt     = logits[0, :-1, :]
        log_z   = mx.logsumexp(lgt, axis=-1)
        target_l = lgt[mx.arange(T), targets]
        nlls    = (log_z - target_l).astype(mx.float32)
        mx.eval(nlls)
        vals = np.clip(np.array(nlls), 0, 100)
        total_nll  += float(vals.sum())
        total_toks += T

    if total_toks == 0:
        return float("inf")
    return float(math.exp(total_nll / total_toks))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"{SAMPLES_PATH} not found — run apply_steering.py first")

    print(f"Loading samples from {SAMPLES_PATH} ...")
    samples = json.loads(SAMPLES_PATH.read_text())

    if not samples.get("per_layer_isolation"):
        raise ValueError(
            "samples-llama3b.json does not have per_layer_isolation=true. "
            "Re-run apply_steering.py to regenerate with the updated harness."
        )

    concepts       = list(samples["results"].keys())
    alphas         = samples["alphas"]
    steering_layers = samples["steering_layers"]  # [8, 16, 24]
    n_completions  = samples["n_completions"]
    layer_keys     = [str(l) for l in steering_layers]

    print(f"  Concepts: {concepts}")
    print(f"  Layers:   {steering_layers}")
    print(f"  Alphas:   {alphas}")

    # ── Pass 1: keyword scoring (no model) ───────────────────────────────────
    print("\n── Pass 1: keyword scoring ──")
    kw_scores = {}
    for concept in concepts:
        kw_scores[concept] = {}
        kws = TARGET_KEYWORDS[concept]
        for layer_key in layer_keys:
            kw_scores[concept][layer_key] = {}
            for alpha in alphas:
                a_key  = str(alpha)
                comps  = samples["results"][concept][layer_key][a_key]
                texts  = [c["completion"] for c in comps]
                hit    = keyword_hit_rate(texts, kws)
                dens   = keyword_density(texts, kws)
                kw_scores[concept][layer_key][a_key] = {
                    "hit_rate":        round(hit,  4),
                    "density_per_100w": round(dens, 4),
                }
                print(
                    f"  {concept:<22} L{layer_key}  α={alpha:+.1f}"
                    f"  hit={hit:.3f}  dens={dens:.3f}"
                )

    # ── Pass 2: model-based PPL scoring ──────────────────────────────────────
    print(f"\nLoading {MODEL_ID} ...")
    t0 = time.time()
    model_obj, tokenizer = load(MODEL_ID)
    print(f"Loaded in {time.time() - t0:.1f}s")

    llama_model = model_obj.model

    vdata = json.loads(VECTORS_PATH.read_text())
    layer_vecs = {}
    for concept in concepts:
        layer_vecs[concept] = {}
        for li in steering_layers:
            vec_np = np.array(
                vdata["vectors"][concept][f"layer_{li}"], dtype=np.float32
            )
            layer_vecs[concept][li] = mx.array(vec_np)

    # JIT warmup
    print("JIT warmup ...")
    w_ids = tokenizer.encode("Warmup sentence.", add_special_tokens=True)
    _ = model_obj(mx.array([w_ids]))
    mx.eval(_)
    print("  Done.")

    # Baseline (alpha=0, no steering)
    print("\nBaseline PPL (alpha=0, no steering) ...")
    t1 = time.time()
    baseline_math_ppl    = compute_perplexity(MATH_PROBES, model_obj, tokenizer)
    baseline_fluency_ppl = compute_perplexity(FLUENCY_SENTENCES, model_obj, tokenizer)
    print(f"  math_ppl    = {baseline_math_ppl:.4f}")
    print(f"  fluency_ppl = {baseline_fluency_ppl:.4f}  ({time.time()-t1:.1f}s)")

    ppl_scores = {}
    total = len(concepts) * len(steering_layers) * len(alphas)
    done  = 0
    t_start = time.time()

    print("\n── Pass 2: PPL scoring ──")
    for concept in concepts:
        ppl_scores[concept] = {}
        for li in steering_layers:
            layer_key = str(li)
            ppl_scores[concept][layer_key] = {}
            vec_mx = layer_vecs[concept][li]

            for alpha in alphas:
                if alpha == 0:
                    math_ppl    = baseline_math_ppl
                    fluency_ppl = baseline_fluency_ppl
                else:
                    orig = _patch_one(llama_model, li, vec_mx, alpha)
                    math_ppl    = compute_perplexity(MATH_PROBES, model_obj, tokenizer)
                    fluency_ppl = compute_perplexity(FLUENCY_SENTENCES, model_obj, tokenizer)
                    _restore_one(llama_model, li, orig)

                ppl_scores[concept][layer_key][str(alpha)] = {
                    "side_effect_ppl":   round(math_ppl, 4),
                    "side_effect_ratio": round(math_ppl / baseline_math_ppl, 4),
                    "fluency_ppl":       round(fluency_ppl, 4),
                    "fluency_ratio":     round(fluency_ppl / baseline_fluency_ppl, 4),
                }
                done += 1
                elapsed = time.time() - t_start
                eta = (elapsed / done) * (total - done) if done > 0 else 0
                print(
                    f"  [{done:3d}/{total}]  {concept:<22}  L{layer_key}  α={alpha:+.1f}"
                    f"  math_ppl={math_ppl:.2f}  flu_ppl={fluency_ppl:.2f}  ETA {eta:.0f}s"
                )

    # ── Assemble scores-llama3b.json ─────────────────────────────────────────
    scores_out = {
        "model":          MODEL_ID,
        "scoring_date":   "2026-07-29",
        "per_layer_isolation": True,
        "samples_source": SAMPLES_PATH.name,
        "baselines": {
            "alpha_0_math_ppl":    round(baseline_math_ppl, 4),
            "alpha_0_fluency_ppl": round(baseline_fluency_ppl, 4),
        },
        "results": {},
    }

    for concept in concepts:
        scores_out["results"][concept] = {}
        for layer_key in layer_keys:
            scores_out["results"][concept][layer_key] = {}
            for alpha in alphas:
                a_key = str(alpha)
                kw    = kw_scores[concept][layer_key][a_key]
                pp    = ppl_scores[concept][layer_key][a_key]
                scores_out["results"][concept][layer_key][a_key] = {
                    "alpha":                   alpha,
                    "n_samples":               n_completions,
                    "keyword_hit_rate":        kw["hit_rate"],
                    "keyword_density_per_100w": kw["density_per_100w"],
                    **pp,
                }

    SCORES_PATH.write_text(json.dumps(scores_out, indent=2))
    print(f"\nSaved {SCORES_PATH}  ({SCORES_PATH.stat().st_size / 1024:.1f} KB)")

    # ── Build benchmark-summary.json ─────────────────────────────────────────
    print("\nBuilding benchmark-summary.json ...")

    # Compute L2 norms from vector data
    vector_norms = {}
    for concept in concepts:
        vector_norms[concept] = {}
        for li in steering_layers:
            vec = np.array(
                vdata["vectors"][concept][f"layer_{li}"], dtype=np.float32
            )
            vector_norms[concept][li] = float(np.linalg.norm(vec))

    # Per-(concept, layer): find best_alpha and scores at that alpha
    # best_alpha = alpha with highest keyword_hit_rate (positive alphas only,
    # since positive alpha steers toward the concept; break ties by largest alpha)
    concept_layer_stats = {}
    for concept in concepts:
        concept_layer_stats[concept] = {}
        for li in steering_layers:
            layer_key = str(li)
            best_alpha = None
            best_hit   = -1.0
            for alpha in alphas:
                if alpha <= 0:
                    continue  # only positive alphas represent steering toward concept
                hit = kw_scores[concept][layer_key][str(alpha)]["hit_rate"]
                if hit > best_hit or (hit == best_hit and (best_alpha is None or alpha > best_alpha)):
                    best_hit   = hit
                    best_alpha = alpha

            # If no positive alpha improved hit rate, use alpha=0 as fallback
            if best_alpha is None:
                best_alpha = 0

            a_key = str(best_alpha)
            pp    = ppl_scores[concept][layer_key][a_key]
            kw    = kw_scores[concept][layer_key][a_key]
            concept_layer_stats[concept][li] = {
                "best_alpha":      best_alpha,
                "behavior_score":  kw["hit_rate"],
                "keyword_density": kw["density_per_100w"],
                "side_effect_score": pp["side_effect_ratio"],
                "fluency_score":     pp["fluency_ratio"],
                "vector_norm":       round(vector_norms[concept][li], 4),
            }

    # Determine dominant layer: layer with highest behavior_score at its best_alpha
    # (ties broken by earliest layer, then by lowest side_effect_score)
    optimal_layers = {}
    for concept in concepts:
        best_layer = None
        best_score = -1.0
        best_side  = float("inf")
        for li in steering_layers:
            s = concept_layer_stats[concept][li]
            if (
                s["behavior_score"] > best_score
                or (s["behavior_score"] == best_score and s["side_effect_score"] < best_side)
            ):
                best_score = s["behavior_score"]
                best_side  = s["side_effect_score"]
                best_layer = li
        optimal_layers[concept] = best_layer

    # Build summary_table rows
    summary_rows = []
    NORM_SIG_THRESHOLD = 0.15  # layer norm fraction >= this → "significant"

    for concept in concepts:
        norms = {li: vector_norms[concept][li] for li in steering_layers}
        total_norm = sum(norms.values())
        norm_fracs = {li: norms[li] / total_norm if total_norm > 0 else 0.0
                      for li in steering_layers}

        dominant_li = optimal_layers[concept]

        for li in steering_layers:
            s = concept_layer_stats[concept][li]
            summary_rows.append({
                "concept":           concept,
                "layer":             li,
                "best_alpha":        s["best_alpha"],
                "behavior_score":    s["behavior_score"],
                "side_effect_score": round(s["side_effect_score"], 4),
                "fluency_score":     round(s["fluency_score"], 4),
                "vector_norm":       s["vector_norm"],
                "norm_fraction":     round(norm_fracs[li], 4),
                "is_dominant_layer": li == dominant_li,
                "is_earliest_significant_layer": (
                    li == min(
                        l for l in steering_layers if norm_fracs[l] >= NORM_SIG_THRESHOLD
                    )
                ),
            })

    # Per-concept summary entries (best layer's stats)
    concept_summary = {}
    for concept in concepts:
        li = optimal_layers[concept]
        s  = concept_layer_stats[concept][li]
        norms = {l: vector_norms[concept][l] for l in steering_layers}
        total_norm = sum(norms.values())
        concept_summary[concept] = {
            "best_alpha":                s["best_alpha"],
            "behavior_score":            s["behavior_score"],
            "side_effect_score":         round(s["side_effect_score"], 4),
            "fluency_score":             round(s["fluency_score"], 4),
            "keyword_density_per_100w":  round(s["keyword_density"], 4),
            "optimal_layer":             li,
            "vector_norms_by_layer": {
                str(l): round(norms[l], 4) for l in steering_layers
            },
            "norm_fractions_by_layer": {
                str(l): round(norms[l] / total_norm, 4) if total_norm > 0 else 0.0
                for l in steering_layers
            },
        }

    summary = {
        "model":          MODEL_ID,
        "generated_date": "2026-07-29",
        "per_layer_isolation": True,
        "baselines": {
            "alpha_0_math_ppl":    round(baseline_math_ppl, 4),
            "alpha_0_fluency_ppl": round(baseline_fluency_ppl, 4),
        },
        "method_note": (
            "Steering vectors applied at EXACTLY ONE layer at a time. "
            "behavior_score = keyword hit-rate at best_alpha (best positive alpha); "
            "side_effect_score = math-probe perplexity ratio vs unsteered baseline; "
            "fluency_score = neutral-sentence perplexity ratio vs unsteered baseline. "
            "optimal_layer = layer with highest behavior_score at its individual best_alpha."
        ),
        "concept_summary":  concept_summary,
        "summary_table":    summary_rows,
        "optimal_layers":   optimal_layers,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Saved {SUMMARY_PATH}  ({SUMMARY_PATH.stat().st_size / 1024:.1f} KB)")

    print("\n── Dominant layer results ──")
    for concept, li in optimal_layers.items():
        s = concept_layer_stats[concept][li]
        print(
            f"  {concept:<22}  dominant=L{li}"
            f"  behavior={s['behavior_score']:.3f}  best_α={s['best_alpha']}"
        )


if __name__ == "__main__":
    main()
