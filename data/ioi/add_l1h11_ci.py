#!/usr/bin/env python3
"""
Add L1H11 bootstrap CI to statistical-validation.json.

L1H11 is excluded by the >= 0.03 mean-score threshold in run_statistical_validation.py
but appears as the #4 head in the paper's Pythia-1.4b circuit table (Table 4). Its CI
was reported in the paper without backing bootstrap data. This script computes the CI
for L1H11 on both models and merges it into the existing statistical-validation.json.

Run on lab-02:
  HF_HOME=/Users/lab-02/.cache/huggingface \
    python3 data/ioi/add_l1h11_ci.py
"""
import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

DATA_DIR = Path(__file__).parent
N_BOOTSTRAP = 1000
RANDOM_SEED = 99  # distinct from run_statistical_validation.py (42) to avoid confusion

MODELS = [
    {"model_id": "mlx-community/Llama-3.2-3B-bf16", "arch": "llama"},
    {"model_id": "EleutherAI/pythia-1.4b",           "arch": "pythia"},
]
FORCE_HEAD = (1, 11)  # L01.H11


# ── Shared helpers (copied from run_statistical_validation.py) ─────────────────

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
    rng = random.Random(RANDOM_SEED)
    boot_means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


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


# ── Patchable attention: Llama ─────────────────────────────────────────────────

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


# ── Patchable attention: Pythia ────────────────────────────────────────────────

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


# ── Per-model runner ───────────────────────────────────────────────────────────

def compute_l1h11(model_id, arch):
    layer_idx, head_idx = FORCE_HEAD
    label = f"L{layer_idx:02d}.H{head_idx:02d}"
    print(f"\n{'='*60}")
    print(f"Model: {model_id}  arch={arch}  target={label}")
    print(f"{'='*60}")

    print(f"Loading {model_id}...")
    t0 = time.time()
    model, tokenizer = load(model_id)
    print(f"  loaded in {time.time()-t0:.1f}s")

    attns = []
    if arch == "llama":
        for layer in model.model.layers:
            pa = LlamaPatchableAttention(layer.self_attn)
            layer.self_attn = pa
            attns.append(pa)
    else:
        for layer in model.layers:
            pa = PythiaPatchableAttention(layer.attention)
            layer.attention = pa
            attns.append(pa)

    examples, io_tids, s_tids, clean_ids, corrupt_ids, clean_last, corrupt_last = \
        prepare_dataset(tokenizer)
    n_examples = len(examples)

    print("Clean forward pass...")
    set_all_modes(attns, "cache_clean")
    clean_logits = model(clean_ids)
    mx.eval(clean_logits, *[a.clean_sdpa for a in attns])
    clean_diffs = logit_diffs(clean_logits, clean_last, io_tids, s_tids)
    print(f"  mean LD = {sum(clean_diffs)/len(clean_diffs):.4f}")

    print("Corrupt forward pass...")
    set_all_modes(attns, "cache_corrupt")
    corrupt_logits = model(corrupt_ids)
    mx.eval(corrupt_logits, *[a.corrupt_sdpa for a in attns])
    corrupt_diffs = logit_diffs(corrupt_logits, corrupt_last, io_tids, s_tids)
    print(f"  mean LD = {sum(corrupt_diffs)/len(corrupt_diffs):.4f}")

    print(f"Patching {label}...")
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
    ci_lo, ci_hi = bootstrap_ci(per_example_recovery)

    print(f"  {label}: mean={computed_mean:.4f}  95% CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        "layer": layer_idx,
        "head": head_idx,
        "label": label,
        "mean_score": computed_mean,
        "stored_mean": None,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "ci_width": ci_hi - ci_lo,
        "n_examples": n_examples,
        "n_bootstrap": N_BOOTSTRAP,
        "note": "force-included: excluded by >= 0.03 threshold but cited in paper Table 4 (Pythia) as ranked #4",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    out_path = DATA_DIR / "statistical-validation.json"
    existing = json.loads(out_path.read_text())

    label = f"L{FORCE_HEAD[0]:02d}.H{FORCE_HEAD[1]:02d}"

    for cfg in MODELS:
        model_id = cfg["model_id"]
        arch     = cfg["arch"]

        model_result = next(
            (m for m in existing["models"] if m["model"] == model_id), None
        )
        if model_result is None:
            print(f"WARNING: {model_id} not found in existing JSON — skipping")
            continue

        already = any(h["label"] == label for h in model_result["heads"])
        if already:
            print(f"{model_id}: {label} already present — skipping")
            continue

        entry = compute_l1h11(model_id, arch)
        model_result["heads"].append(entry)
        print(f"  Appended {label} to {model_id}")

    existing["meta"]["selection_criteria"] += (
        f"; plus force-included {label} (cited in paper table, excluded by threshold)"
    )

    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nSaved → {out_path}")

    print("\n── L01.H11 Summary ──────────────────────────────────")
    for m in existing["models"]:
        for h in m["heads"]:
            if h["label"] == label:
                sig = "✓" if h["ci_95_lo"] > 0 else "~"
                print(
                    f"  {sig} {m['model']}  {h['label']}: "
                    f"{h['mean_score']:.4f} [{h['ci_95_lo']:.4f}, {h['ci_95_hi']:.4f}]"
                )


if __name__ == "__main__":
    main()
