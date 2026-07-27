"""
IOI Statistical Validation — Bootstrap Confidence Intervals
============================================================
For each circuit-critical head (top-10 per model by mean patching score),
collect per-example normalised logit-diff recovery scores, then bootstrap
resample the 100-example dataset 1000 times to produce 95% CIs.

Reads existing patching-{model}.json to identify critical heads.
Runs targeted patching (only those heads, not the full sweep).
Saves: data/ioi/statistical-validation.json
"""
import json
import random
import time
from pathlib import Path
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

DATA_DIR = Path(__file__).parent
N_BOOTSTRAP = 1000
TOP_K = 10
THRESHOLD = 0.03  # also include any head with mean score >= this


# ── Shared helpers ────────────────────────────────────────────────────────────

def logit_diffs(logits, last_positions, io_tids, s_tids):
    out = []
    for i in range(len(last_positions)):
        pos = last_positions[i]
        l_io = float(logits[i, pos, io_tids[i]])
        l_s  = float(logits[i, pos,  s_tids[i]])
        out.append(l_io - l_s)
    return out


def name_token_id(tokenizer, name):
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    return ids[0]


def pad_batch(seqs, max_len):
    padded = [s + [0] * (max_len - len(s)) for s in seqs]
    return mx.array(padded, dtype=mx.uint32)


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05):
    n = len(values)
    boot_means = []
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def select_critical_heads(patching_json_path, top_k=TOP_K, threshold=THRESHOLD):
    with open(patching_json_path) as f:
        d = json.load(f)
    scores = d["patching_scores"]
    n_layers = len(scores)
    n_heads = len(scores[0])
    all_vals = [(scores[l][h], l, h) for l in range(n_layers) for h in range(n_heads)]
    all_vals.sort(reverse=True)
    top = all_vals[:top_k]
    # Also include anything above threshold not already in top-k
    extra = [(s, l, h) for s, l, h in all_vals[top_k:] if s >= threshold]
    combined = top + extra
    return combined, d["meta"]


def prepare_dataset(tokenizer):
    dataset = json.loads((DATA_DIR / "dataset.json").read_text())
    examples = dataset["examples"]
    template = dataset["meta"]["template"]

    io_tids, s_tids = [], []
    clean_encs, corrupt_encs = [], []

    for ex in examples:
        io_name = ex["io_name"]
        s_name  = ex["subject_name"]
        clean_encs.append(tokenizer.encode(ex["prompt"], add_special_tokens=True))
        corrupt_prompt = template.format(S=io_name, IO=s_name)
        corrupt_encs.append(tokenizer.encode(corrupt_prompt, add_special_tokens=True))
        io_tids.append(name_token_id(tokenizer, io_name))
        s_tids.append(name_token_id(tokenizer, s_name))

    clean_lens   = [len(e) for e in clean_encs]
    corrupt_lens = [len(e) for e in corrupt_encs]
    max_clean    = max(clean_lens)
    max_corrupt  = max(corrupt_lens)

    clean_ids   = pad_batch(clean_encs,   max_clean)
    corrupt_ids = pad_batch(corrupt_encs, max_corrupt)
    clean_last  = [l - 1 for l in clean_lens]
    corrupt_last = [l - 1 for l in corrupt_lens]

    return examples, io_tids, s_tids, clean_ids, corrupt_ids, clean_last, corrupt_last


# ── Llama PatchableAttention ──────────────────────────────────────────────────

class LlamaPatchableAttention(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.q_proj    = orig.q_proj
        self.k_proj    = orig.k_proj
        self.v_proj    = orig.v_proj
        self.o_proj    = orig.o_proj
        self.rope      = orig.rope
        self.n_heads   = orig.n_heads
        self.n_kv_heads = orig.n_kv_heads
        self.head_dim  = orig.head_dim
        self.scale     = orig.scale
        self.mode      = "normal"
        self.clean_sdpa   = None
        self.corrupt_sdpa = None
        self.patch_head   = None

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        queries = self.q_proj(x).reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        keys    = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values  = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries = self.rope(queries)
        keys    = self.rope(keys)
        sdpa = scaled_dot_product_attention(queries, keys, values, cache=None, scale=self.scale, mask=mask)
        if self.mode == "cache_clean":
            self.clean_sdpa = sdpa
        elif self.mode == "cache_corrupt":
            self.corrupt_sdpa = sdpa
        elif self.mode == "patch" and self.patch_head is not None:
            h = self.patch_head
            sdpa = mx.concatenate([sdpa[:, :h], self.clean_sdpa[:, h:h+1], sdpa[:, h+1:]], axis=1)
        out = sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


# ── Pythia PatchableAttention ─────────────────────────────────────────────────

class PythiaPatchableAttention(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.query_key_value     = orig.query_key_value
        self.dense               = orig.dense
        self.rope                = orig.rope
        self.num_attention_heads = orig.num_attention_heads
        self.head_dim            = orig.head_dim
        self.hidden_size         = orig.hidden_size
        self.scale               = orig.scale
        self.mode      = "normal"
        self.clean_sdpa   = None
        self.corrupt_sdpa = None
        self.patch_head   = None

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        n = self.num_attention_heads
        qkv = self.query_key_value(x).reshape(B, L, n, 3 * self.head_dim)
        queries, keys, values = [t.transpose(0, 2, 1, 3) for t in qkv.split(3, -1)]
        queries = self.rope(queries)
        keys    = self.rope(keys)
        sdpa = scaled_dot_product_attention(queries, keys, values, cache=None, scale=self.scale, mask=mask)
        if self.mode == "cache_clean":
            self.clean_sdpa = sdpa
        elif self.mode == "cache_corrupt":
            self.corrupt_sdpa = sdpa
        elif self.mode == "patch" and self.patch_head is not None:
            h = self.patch_head
            sdpa = mx.concatenate([sdpa[:, :h], self.clean_sdpa[:, h:h+1], sdpa[:, h+1:]], axis=1)
        out = sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(out)


def set_all_modes(attns, mode, patch_head=None):
    for a in attns:
        a.mode = mode
        a.patch_head = patch_head


# ── Per-model runner ──────────────────────────────────────────────────────────

def run_model(model_id, patching_json, arch):
    print(f"\n{'='*60}")
    print(f"Model: {model_id}  arch={arch}")
    print(f"{'='*60}")

    critical_heads, meta = select_critical_heads(patching_json)
    print(f"Circuit-critical heads ({len(critical_heads)}):")
    for score, l, h in critical_heads:
        print(f"  L{l:02d}.H{h:02d}: mean={score:.4f}")

    print(f"\nLoading {model_id}...")
    t0 = time.time()
    model, tokenizer = load(model_id)
    print(f"  loaded in {time.time()-t0:.1f}s")

    # Replace attention modules
    attns = []
    if arch == "llama":
        for layer in model.model.layers:
            pa = LlamaPatchableAttention(layer.self_attn)
            layer.self_attn = pa
            attns.append(pa)
    else:  # pythia / gpt-neox — mlx_lm exposes layers at model.layers
        for layer in model.layers:
            pa = PythiaPatchableAttention(layer.attention)
            layer.attention = pa
            attns.append(pa)

    examples, io_tids, s_tids, clean_ids, corrupt_ids, clean_last, corrupt_last = prepare_dataset(tokenizer)
    n_examples = len(examples)

    # Clean pass
    print("Clean forward pass...")
    t1 = time.time()
    set_all_modes(attns, "cache_clean")
    clean_logits = model(clean_ids)
    mx.eval(clean_logits, *[a.clean_sdpa for a in attns])
    clean_diffs = logit_diffs(clean_logits, clean_last, io_tids, s_tids)
    print(f"  {time.time()-t1:.1f}s  mean logit diff = {sum(clean_diffs)/len(clean_diffs):.4f}")

    # Corrupt pass
    print("Corrupt forward pass...")
    t1 = time.time()
    set_all_modes(attns, "cache_corrupt")
    corrupt_logits = model(corrupt_ids)
    mx.eval(corrupt_logits, *[a.corrupt_sdpa for a in attns])
    corrupt_diffs = logit_diffs(corrupt_logits, corrupt_last, io_tids, s_tids)
    print(f"  {time.time()-t1:.1f}s  mean logit diff = {sum(corrupt_diffs)/len(corrupt_diffs):.4f}")

    # Targeted patching for critical heads only
    head_results = []
    for mean_score, layer_idx, head_idx in critical_heads:
        set_all_modes(attns, "normal")
        attns[layer_idx].mode       = "patch"
        attns[layer_idx].patch_head = head_idx

        patched_logits = model(corrupt_ids)
        mx.eval(patched_logits)
        patched_diffs = logit_diffs(patched_logits, corrupt_last, io_tids, s_tids)

        per_example_recovery = []
        for i in range(n_examples):
            denom = clean_diffs[i] - corrupt_diffs[i]
            if abs(denom) < 1e-6:
                per_example_recovery.append(0.0)
            else:
                per_example_recovery.append((patched_diffs[i] - corrupt_diffs[i]) / denom)

        computed_mean = sum(per_example_recovery) / n_examples
        ci_lo, ci_hi = bootstrap_ci(per_example_recovery, n_boot=N_BOOTSTRAP)

        print(
            f"  L{layer_idx:02d}.H{head_idx:02d}: "
            f"mean={computed_mean:.4f}  "
            f"95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]  "
            f"(stored mean={mean_score:.4f})"
        )

        head_results.append({
            "layer": layer_idx,
            "head": head_idx,
            "label": f"L{layer_idx:02d}.H{head_idx:02d}",
            "mean_score": computed_mean,
            "stored_mean": mean_score,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "ci_width": ci_hi - ci_lo,
            "n_examples": n_examples,
            "n_bootstrap": N_BOOTSTRAP,
        })

    return {
        "model": model_id,
        "n_layers": meta["n_layers"],
        "n_heads": meta["n_heads"],
        "metric": meta["metric"],
        "heads": head_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    results = []

    results.append(run_model(
        model_id="mlx-community/Llama-3.2-3B-bf16",
        patching_json=DATA_DIR / "patching-llama3b.json",
        arch="llama",
    ))

    results.append(run_model(
        model_id="EleutherAI/pythia-1.4b",
        patching_json=DATA_DIR / "patching-pythia1b.json",
        arch="pythia",
    ))

    output = {
        "meta": {
            "description": (
                "Bootstrap confidence intervals for per-head activation patching scores "
                "on the IOI circuit. For each circuit-critical head (top-10 by mean score "
                "plus any head with mean >= 0.03), the 100-example dataset is resampled "
                f"{N_BOOTSTRAP} times with replacement. 95% CI = [2.5th, 97.5th] percentile "
                "of bootstrap distribution of the mean."
            ),
            "n_bootstrap": N_BOOTSTRAP,
            "ci_level": 0.95,
            "random_seed": 42,
            "selection_criteria": f"top-{TOP_K} heads by mean patching score, plus any head >= {THRESHOLD}",
        },
        "models": results,
    }

    out_path = DATA_DIR / "statistical-validation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n── Summary ──────────────────────────────────────────────")
    for model_result in results:
        print(f"\n{model_result['model']}")
        for h in model_result["heads"]:
            sig = "✓" if h["ci_95_lo"] > 0 else "~"
            print(
                f"  {sig} {h['label']}: "
                f"{h['mean_score']:.4f} "
                f"[{h['ci_95_lo']:.4f}, {h['ci_95_hi']:.4f}]"
            )


if __name__ == "__main__":
    main()
