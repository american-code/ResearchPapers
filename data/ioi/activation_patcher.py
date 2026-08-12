#!/usr/bin/env python3
"""
activation_patcher.py — IOI circuit re-derivation on dataset-v2.

Full pipeline:
  1. Activation-patching sweep [L×H] → derive top-K circuit
  2. Faithfulness, minimality, completeness on the derived circuit
  3. Joint ablation + leave-one-in

Models:
  llama8b  mlx-community/Meta-Llama-3.1-8B-Instruct-4bit  (32L × 32H)
  pythia   EleutherAI/pythia-1.4b                          (24L × 16H)

Usage:
  python3 activation_patcher.py --model llama8b
  python3 activation_patcher.py --model pythia

Outputs (data/circuit-v2/):
  patching-{model}-v2.json
  faithfulness-{model}-v2.json
  joint-ablation-{model}-v2.json

Patching sweep correctness: all examples run in one batch so that the
cached clean SDPA covers the full dataset during all patch passes.
Faithfulness uses mini-batches for memory efficiency.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

DATA_DIR    = Path(__file__).parent
CIRCUIT_DIR = DATA_DIR.parent / "circuit-v2"
DATASET     = DATA_DIR / "dataset-v2.json"

CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "llama8b": {
        "id":       "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "kind":     "llama",
        "n_layers": 32,
        "n_heads":  32,
    },
    "pythia": {
        "id":       "EleutherAI/pythia-1.4b",
        "kind":     "neox",
        "n_layers": 24,
        "n_heads":  16,
    },
}

SEED                  = 42
TOP_K                 = 10
N_COMPLETENESS_SUBSETS = 12
FAITH_BATCH           = 25


# ─────────────────────────────────────────────────────────────────────────────
# Patchable attention: clean/corrupt SDPA caching + per-head replacement
# ─────────────────────────────────────────────────────────────────────────────

class LlamaPatchable(nn.Module):
    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        self.q_proj     = orig.q_proj
        self.k_proj     = orig.k_proj
        self.v_proj     = orig.v_proj
        self.o_proj     = orig.o_proj
        self.rope       = orig.rope
        self.n_heads    = orig.n_heads
        self.n_kv_heads = orig.n_kv_heads
        self.head_dim   = orig.head_dim
        self.scale      = orig.scale
        self.mode: str            = "normal"
        self.clean_sdpa:   Optional[mx.array] = None
        self.corrupt_sdpa: Optional[mx.array] = None
        self.patch_head:   Optional[int]       = None

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        sdpa = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
        mode = self.mode
        if mode == "cache_clean":
            self.clean_sdpa = sdpa
        elif mode == "cache_corrupt":
            self.corrupt_sdpa = sdpa
        elif mode == "patch" and self.patch_head is not None:
            h = self.patch_head
            sdpa = mx.concatenate(
                [sdpa[:, :h], self.clean_sdpa[:, h:h+1], sdpa[:, h+1:]], axis=1
            )
        return self.o_proj(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))


class NeoXPatchable(nn.Module):
    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        self.query_key_value     = orig.query_key_value
        self.dense               = orig.dense
        self.rope                = orig.rope
        self.num_attention_heads = orig.num_attention_heads
        self.head_dim            = orig.head_dim
        self.hidden_size         = orig.hidden_size
        self.scale               = orig.scale
        self.mode: str            = "normal"
        self.clean_sdpa:   Optional[mx.array] = None
        self.corrupt_sdpa: Optional[mx.array] = None
        self.patch_head:   Optional[int]       = None

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        n = self.num_attention_heads
        qkv = self.query_key_value(x).reshape(B, L, n, 3 * self.head_dim)
        q, k, v = [t.transpose(0, 2, 1, 3) for t in qkv.split(3, -1)]
        q, k = self.rope(q), self.rope(k)
        sdpa = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
        mode = self.mode
        if mode == "cache_clean":
            self.clean_sdpa = sdpa
        elif mode == "cache_corrupt":
            self.corrupt_sdpa = sdpa
        elif mode == "patch" and self.patch_head is not None:
            h = self.patch_head
            sdpa = mx.concatenate(
                [sdpa[:, :h], self.clean_sdpa[:, h:h+1], sdpa[:, h+1:]], axis=1
            )
        return self.dense(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))


# ─────────────────────────────────────────────────────────────────────────────
# Maskable attention: mean-ablation for faithfulness / ablation passes
# ─────────────────────────────────────────────────────────────────────────────

class LlamaMaskable(nn.Module):
    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        self.q_proj     = orig.q_proj
        self.k_proj     = orig.k_proj
        self.v_proj     = orig.v_proj
        self.o_proj     = orig.o_proj
        self.rope       = orig.rope
        self.n_heads    = orig.n_heads
        self.n_kv_heads = orig.n_kv_heads
        self.head_dim   = orig.head_dim
        self.scale      = orig.scale
        self.mode: str              = "normal"
        self.mean_sdpa: Optional[mx.array] = None
        self.ablate_heads: frozenset       = frozenset()

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        sdpa = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
        if self.mode == "cache_mean":
            self.mean_sdpa = mx.mean(sdpa, axis=0, keepdims=True)
        elif self.mode == "ablate" and self.ablate_heads:
            cols = []
            for h in range(self.n_heads):
                if h in self.ablate_heads:
                    cols.append(mx.broadcast_to(self.mean_sdpa[:, h:h+1], (B, 1, L, self.head_dim)))
                else:
                    cols.append(sdpa[:, h:h+1])
            sdpa = mx.concatenate(cols, axis=1)
        return self.o_proj(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))


class NeoXMaskable(nn.Module):
    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        self.query_key_value     = orig.query_key_value
        self.dense               = orig.dense
        self.rope                = orig.rope
        self.num_attention_heads = orig.num_attention_heads
        self.head_dim            = orig.head_dim
        self.hidden_size         = orig.hidden_size
        self.scale               = orig.scale
        self.mode: str              = "normal"
        self.mean_sdpa: Optional[mx.array] = None
        self.ablate_heads: frozenset       = frozenset()

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        n = self.num_attention_heads
        qkv = self.query_key_value(x).reshape(B, L, n, 3 * self.head_dim)
        q, k, v = [t.transpose(0, 2, 1, 3) for t in qkv.split(3, -1)]
        q, k = self.rope(q), self.rope(k)
        sdpa = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
        if self.mode == "cache_mean":
            self.mean_sdpa = mx.mean(sdpa, axis=0, keepdims=True)
        elif self.mode == "ablate" and self.ablate_heads:
            cols = []
            for h in range(n):
                if h in self.ablate_heads:
                    cols.append(mx.broadcast_to(self.mean_sdpa[:, h:h+1], (B, 1, L, self.head_dim)))
                else:
                    cols.append(sdpa[:, h:h+1])
            sdpa = mx.concatenate(cols, axis=1)
        return self.dense(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def set_patch_all(attns: list, mode: str, patch_head: Optional[int] = None) -> None:
    for a in attns:
        a.mode = mode
        a.patch_head = patch_head


def set_mask_mode(attns: list, mode: str) -> None:
    for a in attns:
        a.mode = mode
        a.ablate_heads = frozenset()


def set_ablation(attns: list, pairs: set) -> None:
    by_layer: dict = {}
    for (l, h) in pairs:
        by_layer.setdefault(l, set()).add(h)
    for i, a in enumerate(attns):
        a.ablate_heads = frozenset(by_layer.get(i, ()))
        a.mode = "ablate" if a.ablate_heads else "normal"


def logit_diffs(logits, last_pos, io_tids, s_tids):
    return [
        float(logits[i, last_pos[i], io_tids[i]]) - float(logits[i, last_pos[i], s_tids[i]])
        for i in range(len(last_pos))
    ]


def pad_batch(seqs: list, max_len: int) -> mx.array:
    return mx.array([s + [0] * (max_len - len(s)) for s in seqs], dtype=mx.uint32)


def name_token_id(tokenizer, name: str) -> int:
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) > 1:
        print(f"  WARNING: ' {name}' → {len(ids)} tokens; using first")
    return ids[0]


def run_batches_ablated(model, attns, batches, pairs):
    """Mean LD with `pairs` set of (layer, head) mean-ablated. None = clean run."""
    out = []
    for toks, last, io_t, s_t in batches:
        if pairs is None:
            set_mask_mode(attns, "normal")
        else:
            set_ablation(attns, pairs)
        logits = model(toks)
        mx.eval(logits)
        out.extend(logit_diffs(logits, last, io_t, s_t))
    return sum(out) / len(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--n-completeness", type=int, default=N_COMPLETENESS_SUBSETS)
    ap.add_argument("--n-examples", type=int, default=0,
                    help="cap examples for patching sweep (0 = all)")
    args = ap.parse_args()
    random.seed(SEED)

    cfg       = MODELS[args.model]
    model_tag = args.model
    top_k     = args.top_k
    n_layers  = cfg["n_layers"]
    n_heads   = cfg["n_heads"]
    kind      = cfg["kind"]

    t_start = time.time()

    # ── Dataset ───────────────────────────────────────────────────────────────
    ds = json.loads(DATASET.read_text())
    examples = ds["examples"]
    n_templates = ds["meta"].get("n_templates", "?")
    if args.n_examples > 0:
        examples = examples[: args.n_examples]
    print(f"Dataset: {len(examples)} examples, {n_templates} templates", flush=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading {cfg['id']} …", flush=True)
    t0 = time.time()
    model, tokenizer = load(cfg["id"])
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    # ── Tokenise all examples (for patching sweep) ────────────────────────────
    io_tids, s_tids, clean_encs, corrupt_encs = [], [], [], []
    for ex in examples:
        clean_encs.append(tokenizer.encode(ex["prompt"], add_special_tokens=True))
        corrupt_encs.append(tokenizer.encode(ex["corrupt_prompt"], add_special_tokens=True))
        io_tids.append(name_token_id(tokenizer, ex["io_name"]))
        s_tids.append(name_token_id(tokenizer, ex["subject_name"]))

    clean_lens   = [len(e) for e in clean_encs]
    corrupt_lens = [len(e) for e in corrupt_encs]
    mismatches   = sum(1 for c, k in zip(clean_lens, corrupt_lens) if c != k)
    if mismatches:
        print(f"  WARNING: {mismatches}/{len(examples)} examples have clean/corrupt length mismatch",
              flush=True)
    else:
        print(f"  lengths all match ({min(clean_lens)}–{max(clean_lens)} tokens)", flush=True)

    max_clean   = max(clean_lens)
    max_corrupt = max(corrupt_lens)
    clean_ids   = pad_batch(clean_encs,   max_clean)
    corrupt_ids = pad_batch(corrupt_encs, max_corrupt)
    clean_last  = [l - 1 for l in clean_lens]
    corrupt_last = [l - 1 for l in corrupt_lens]

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1: Activation patching sweep
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}", flush=True)
    print(f"PHASE 1: Patching sweep  {n_layers}L × {n_heads}H = {n_layers*n_heads} passes",
          flush=True)
    print(f"{'='*64}", flush=True)

    patch_attns: list = []
    if kind == "llama":
        for layer in model.model.layers:
            pa = LlamaPatchable(layer.self_attn)
            layer.self_attn = pa
            patch_attns.append(pa)
    else:
        for layer in model.layers:
            pa = NeoXPatchable(layer.attention)
            layer.attention = pa
            patch_attns.append(pa)
    print(f"  installed {len(patch_attns)} patchable layers", flush=True)

    # Clean pass — cache SDPA for all examples simultaneously
    print("Clean forward pass …", flush=True)
    t1 = time.time()
    set_patch_all(patch_attns, "cache_clean")
    clean_logits = model(clean_ids)
    mx.eval(clean_logits, *[a.clean_sdpa for a in patch_attns if a.clean_sdpa is not None])
    clean_diffs = logit_diffs(clean_logits, clean_last, io_tids, s_tids)
    mean_clean  = sum(clean_diffs) / len(clean_diffs)
    print(f"  {time.time()-t1:.1f}s  mean LD = {mean_clean:.4f}", flush=True)

    # Corrupt pass — cache SDPA
    print("Corrupt forward pass …", flush=True)
    t1 = time.time()
    set_patch_all(patch_attns, "cache_corrupt")
    corrupt_logits = model(corrupt_ids)
    mx.eval(corrupt_logits,
            *[a.corrupt_sdpa for a in patch_attns if a.corrupt_sdpa is not None])
    corrupt_diffs  = logit_diffs(corrupt_logits, corrupt_last, io_tids, s_tids)
    mean_corrupt   = sum(corrupt_diffs) / len(corrupt_diffs)
    print(f"  {time.time()-t1:.1f}s  mean LD = {mean_corrupt:.4f}", flush=True)
    if mean_corrupt >= mean_clean:
        print("  WARNING: corrupt LD ≥ clean LD — IOI effect may be weak or dataset is wrong",
              flush=True)

    # Patching sweep
    total_pairs = n_layers * n_heads
    print(f"\nSweeping {total_pairs} (layer, head) pairs …", flush=True)
    patching_scores: list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]
    done = 0
    t_sweep = time.time()

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            set_patch_all(patch_attns, "normal")
            patch_attns[layer_idx].mode       = "patch"
            patch_attns[layer_idx].patch_head = head_idx

            patched_logits = model(corrupt_ids)
            mx.eval(patched_logits)
            patched_diffs = logit_diffs(patched_logits, corrupt_last, io_tids, s_tids)

            recoveries = []
            for i in range(len(examples)):
                denom = clean_diffs[i] - corrupt_diffs[i]
                recoveries.append(
                    (patched_diffs[i] - corrupt_diffs[i]) / denom
                    if abs(denom) > 1e-6 else 0.0
                )
            patching_scores[layer_idx][head_idx] = sum(recoveries) / len(recoveries)
            done += 1

        elapsed = time.time() - t_sweep
        eta     = elapsed / done * (total_pairs - done) if done < total_pairs else 0
        top_h   = max(range(n_heads), key=lambda h: patching_scores[layer_idx][h])
        print(
            f"  L{layer_idx:02d}  top=H{top_h:02d}({patching_scores[layer_idx][top_h]:.3f})"
            f"  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
            flush=True,
        )

    sweep_time = time.time() - t_sweep
    print(f"\nSweep done in {sweep_time:.0f}s ({sweep_time/60:.1f} min)", flush=True)

    # Derive circuit: top-K by patching score
    all_vals   = [(patching_scores[l][h], l, h) for l in range(n_layers) for h in range(n_heads)]
    top_sorted = sorted(all_vals, reverse=True)[:top_k]
    circuit    = frozenset((l, h) for _, l, h in top_sorted)
    circuit_str = sorted([f"L{l}H{h}" for _, l, h in top_sorted])

    patching_out = {
        "meta": {
            "model":       cfg["id"],
            "dataset":     DATASET.name,
            "n_layers":    n_layers,
            "n_heads":     n_heads,
            "n_examples":  len(examples),
            "n_templates": n_templates,
            "metric":      "normalized_logit_diff_recovery",
            "top_k":       top_k,
            "top_k_circuit": circuit_str,
            "synthetic":   False,
        },
        "patching_scores": patching_scores,
        "top10": [
            {"head": f"L{l}H{h}", "score": round(s, 4)}
            for s, l, h in top_sorted
        ],
    }
    patching_path = CIRCUIT_DIR / f"patching-{model_tag}-v2.json"
    patching_path.write_text(json.dumps(patching_out, indent=2))
    print(f"\nCircuit (top-{top_k}): {circuit_str}", flush=True)
    print(f"Saved → {patching_path}", flush=True)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2: Faithfulness, minimality, completeness
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}", flush=True)
    print("PHASE 2: Faithfulness / minimality / completeness", flush=True)
    print(f"{'='*64}", flush=True)

    # Swap to maskable attention (copies weight refs from the patchable)
    mask_attns: list = []
    if kind == "llama":
        for layer in model.model.layers:
            ma = LlamaMaskable(layer.self_attn)
            layer.self_attn = ma
            mask_attns.append(ma)
    else:
        for layer in model.layers:
            ma = NeoXMaskable(layer.attention)
            layer.attention = ma
            mask_attns.append(ma)

    # Tokenise into mini-batches for faithfulness
    faith_batches = []
    for i in range(0, len(examples), FAITH_BATCH):
        chunk = examples[i : i + FAITH_BATCH]
        seqs, last, io_t, s_t = [], [], [], []
        for e in chunk:
            ids = tokenizer.encode(e["prompt"], add_special_tokens=True)
            seqs.append(ids)
            last.append(len(ids) - 1)
            io_t.append(name_token_id(tokenizer, e["io_name"]))
            s_t.append(name_token_id(tokenizer, e["subject_name"]))
        m = max(len(s) for s in seqs)
        faith_batches.append((pad_batch(seqs, m), last, io_t, s_t))

    # Cache mean SDPA across all batches (each pass approximates the global mean)
    print("Caching mean SDPA …", flush=True)
    set_mask_mode(mask_attns, "cache_mean")
    for toks, *_ in faith_batches:
        mx.eval(model(toks))

    all_heads = frozenset((l, h) for l in range(n_layers) for h in range(n_heads))

    ld_clean = run_batches_ablated(model, mask_attns, faith_batches, None)
    ld_floor = run_batches_ablated(model, mask_attns, faith_batches, all_heads)
    print(f"clean LD = {ld_clean:.4f}  floor LD = {ld_floor:.4f}", flush=True)

    def norm(ld: float) -> float:
        return (ld - ld_floor) / (ld_clean - ld_floor)

    # Faithfulness: ablate everything outside circuit
    ld_circ = run_batches_ablated(model, mask_attns, faith_batches, all_heads - circuit)
    faith   = norm(ld_circ)
    print(f"faithfulness: LD={ld_circ:.4f}  F={faith:.4f}", flush=True)

    # Minimality: remove each head from circuit in turn
    minimality = []
    for v in sorted(circuit):
        ld_v = run_batches_ablated(model, mask_attns, faith_batches,
                                    (all_heads - circuit) | {v})
        minimality.append({
            "head":      f"L{v[0]}H{v[1]}",
            "layer":     v[0],
            "head_idx":  v[1],
            "F_without": round(norm(ld_v), 4),
            "delta_F":   round(faith - norm(ld_v), 4),
        })
        print(f"  minimality L{v[0]}H{v[1]}: dF={faith - norm(ld_v):+.4f}", flush=True)
    minimality.sort(key=lambda r: -r["delta_F"])

    # Completeness: random subsets K
    circ_list  = sorted(circuit)
    completeness = []
    for _ in range(args.n_completeness):
        k = random.randint(1, max(1, len(circ_list) // 2))
        K = set(random.sample(circ_list, k))
        ld_ck = run_batches_ablated(model, mask_attns, faith_batches,
                                     (all_heads - circuit) | K)
        ld_mk = run_batches_ablated(model, mask_attns, faith_batches, K)
        completeness.append({
            "K":                  [f"L{l}H{h}" for l, h in sorted(K)],
            "F_circuit_minus_K":  round(norm(ld_ck), 4),
            "F_model_minus_K":    round(norm(ld_mk), 4),
            "gap":                round(abs(norm(ld_ck) - norm(ld_mk)), 4),
        })
    gaps = [c["gap"] for c in completeness]

    faith_result = {
        "meta": {
            "model":             cfg["id"],
            "dataset":           DATASET.name,
            "n_examples":        len(examples),
            "n_templates":       n_templates,
            "n_layers":          n_layers,
            "n_heads":           n_heads,
            "circuit":           circuit_str,
            "circuit_size":      len(circuit),
            "circuit_selection": f"top-{top_k} by activation patching (normalized LD recovery)",
            "seed":              SEED,
            "synthetic":         False,
            "normalization":     "F(X) = (LD(X) - LD_all_ablated) / (LD_clean - LD_all_ablated)",
        },
        "logit_diffs": {
            "clean":             round(ld_clean, 4),
            "all_heads_ablated": round(ld_floor, 4),
            "circuit_only":      round(ld_circ, 4),
        },
        "faithfulness": round(faith, 4),
        "minimality":   minimality,
        "completeness": {
            "n_subsets": args.n_completeness,
            "mean_gap":  round(sum(gaps) / len(gaps), 4),
            "max_gap":   round(max(gaps), 4),
            "subsets":   completeness,
        },
    }
    faith_path = CIRCUIT_DIR / f"faithfulness-{model_tag}-v2.json"
    faith_path.write_text(json.dumps(faith_result, indent=1))
    print(f"Faithfulness saved → {faith_path}", flush=True)

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3: Joint ablation + leave-one-in
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}", flush=True)
    print("PHASE 3: Joint ablation + leave-one-in", flush=True)
    print(f"{'='*64}", flush=True)

    # Joint ablation: remove all circuit heads from full model
    ld_residual = run_batches_ablated(model, mask_attns, faith_batches, circuit)
    F_complete  = norm(ld_residual)
    print(f"joint ablation: LD={ld_residual:.4f}  F_complete={F_complete:.4f}", flush=True)

    # Leave-one-in: ablate n-1 circuit heads, leave one active
    leave_one_in = []
    for (l, h) in sorted(circuit):
        nine = circuit - {(l, h)}
        ld_loi = run_batches_ablated(model, mask_attns, faith_batches, nine)
        F_loi   = norm(ld_loi)
        delta   = round(F_loi - F_complete, 4)
        leave_one_in.append({
            "head":        f"L{l}H{h}",
            "layer":       l,
            "head_idx":    h,
            "LD_leave_in": round(ld_loi, 4),
            "F_leave_in":  round(F_loi, 4),
            "delta_F":     delta,
        })
        print(f"  leave L{l}H{h}: F={F_loi:.4f}  delta={delta:+.4f}", flush=True)
    leave_one_in.sort(key=lambda r: -r["delta_F"])

    total_time = time.time() - t_start
    ablation_result = {
        "meta": {
            "model":         cfg["id"],
            "dataset":       DATASET.name,
            "n_examples":    len(examples),
            "n_templates":   n_templates,
            "n_layers":      n_layers,
            "n_heads":       n_heads,
            "circuit":       circuit_str,
            "circuit_size":  len(circuit),
            "synthetic":     False,
            "normalization": "F(X) = (LD(X) - LD_all_ablated) / (LD_clean - LD_all_ablated)",
        },
        "baselines": {
            "LD_clean":          round(ld_clean, 4),
            "LD_all_ablated":    round(ld_floor, 4),
            "LD_circuit_only":   round(ld_circ, 4),
            "F_faithfulness":    round(faith, 4),
        },
        "joint_ablation": {
            "LD_model_minus_circuit": round(ld_residual, 4),
            "F_complete":             round(F_complete, 4),
        },
        "leave_one_in": {
            "F_complete_reference": round(F_complete, 4),
            "heads":                leave_one_in,
        },
        "runtime_seconds": round(total_time, 1),
        "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    ablation_path = CIRCUIT_DIR / f"joint-ablation-{model_tag}-v2.json"
    ablation_path.write_text(json.dumps(ablation_result, indent=2))
    print(f"Joint ablation saved → {ablation_path}", flush=True)

    # ═════════════════════════════════════════════════════════════════════════
    # Summary
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*64}", flush=True)
    print(f"DONE — {model_tag} / {cfg['id']}", flush=True)
    print(f"  Dataset:        {len(examples)} examples, {n_templates} templates", flush=True)
    print(f"  Circuit:        {circuit_str}", flush=True)
    print(f"  Faithfulness:   {faith:.4f}", flush=True)
    print(f"  F_complete:     {F_complete:.4f}  (joint ablation)", flush=True)
    print(f"  Total runtime:  {total_time:.0f}s ({total_time/3600:.1f}h)", flush=True)
    print(f"  Outputs:        {CIRCUIT_DIR}/", flush=True)
    if faith > 0.5:
        print("  VERDICT: faithfulness holds — circuit is real", flush=True)
    elif faith > 0.15:
        print("  VERDICT: partial faithfulness — circuit is noisy", flush=True)
    else:
        print("  VERDICT: faithfulness collapsed — positional artifact likely", flush=True)
    print(f"{'='*64}", flush=True)


if __name__ == "__main__":
    main()
