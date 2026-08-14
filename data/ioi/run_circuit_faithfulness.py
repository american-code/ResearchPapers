#!/usr/bin/env python3
"""
Circuit-level faithfulness, completeness and minimality for the IOI circuit.

This is the experiment the circuit-tracing paper is missing. Per-head patching and
ablation scores measure each head's *marginal* effect; they are not additive (our
Llama top-10 patching scores sum to 1.013), so they cannot say what the circuit
explains jointly. Wang et al. (2022) validate their GPT-2 Small circuit as a set,
reporting 87% faithfulness. This script computes the same three quantities.

Definitions (following Wang et al. 2022, §3):

  FAITHFULNESS  Mean-ablate every head NOT in the circuit C, leave C intact, and
                measure the recovered logit difference:
                    F(C) = LD(only C active) / LD(clean)
                A faithful circuit recovers most of the behaviour on its own.

  MINIMALITY    For each head v in C, mean-ablate v while leaving the rest of C
                intact (everything outside C already ablated), and measure
                    F(C) - F(C \\ {v})
                A minimal circuit has no head whose removal is free.

  COMPLETENESS  For random subsets K of C, compare F(C \\ K) against F(M \\ K),
                where M is the full model. If C is complete, ablating K should hurt
                the circuit and the full model comparably; a large gap means C is
                missing components that compensate in the full model.

Usage:
  python3 run_circuit_faithfulness.py --model llama
  python3 run_circuit_faithfulness.py --model pythia

Output: data/ioi/faithfulness-{llama3b,pythia1b}.json
"""

import argparse
import json
import pathlib
import random
import time
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

DATA_DIR = Path(__file__).parent

MODELS = {
    "llama": {
        "id": "mlx-community/Llama-3.2-3B-bf16",
        "n_layers": 28, "n_heads": 24,
        "out": "faithfulness-llama3b.json",
        # top-10 by combined normalized rank, from the paper's Table 3
        "circuit": [(15, 20), (17, 17), (13, 14), (24, 15), (19, 1),
                    (21, 20), (18, 10), (14, 0), (27, 17), (26, 23)],
    },
    "pythia": {
        "id": "EleutherAI/pythia-1.4b",
        "n_layers": 24, "n_heads": 16,
        "out": "faithfulness-pythia1b.json",
        # top-10 from the paper's Table 4
        "circuit": [(10, 7), (15, 15), (22, 2), (1, 11), (21, 3),
                    (17, 7), (10, 0), (12, 15), (16, 13), (13, 6)],
    },
}

BATCH = 25
SEED = 42
N_COMPLETENESS_SUBSETS = 12


# ── Patchable attention: mean-ablate an arbitrary SET of heads ───────────────

class MaskedAttentionBase(nn.Module):
    """
    Shared ablation logic. `ablate_heads` is a set of head indices in this layer to
    replace with their per-head mean SDPA output (mean taken over clean examples).
    """

    def _init_state(self) -> None:
        self.mode: str = "normal"                    # normal | cache_mean | ablate
        self.mean_sdpa: Optional[mx.array] = None    # [1, n_heads, L, head_dim]
        self.ablate_heads: frozenset = frozenset()

    def _apply(self, sdpa: mx.array, B: int, L: int, n_heads: int) -> mx.array:
        if self.mode == "cache_mean":
            self.mean_sdpa = mx.mean(sdpa, axis=0, keepdims=True)
            return sdpa
        if self.mode == "ablate" and self.ablate_heads:
            if self.mean_sdpa is None:
                raise RuntimeError("ablate before cache_mean")
            cols = []
            for h in range(n_heads):
                if h in self.ablate_heads:
                    m = self.mean_sdpa[:, h:h + 1, :, :]
                    cols.append(mx.broadcast_to(m, (B, 1, L, self.head_dim)))
                else:
                    cols.append(sdpa[:, h:h + 1, :, :])
            return mx.concatenate(cols, axis=1)
        return sdpa


class LlamaMaskedAttention(MaskedAttentionBase):
    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        self.q_proj, self.k_proj = orig.q_proj, orig.k_proj
        self.v_proj, self.o_proj = orig.v_proj, orig.o_proj
        self.rope = orig.rope
        self.n_heads, self.n_kv_heads = orig.n_heads, orig.n_kv_heads
        self.head_dim, self.scale = orig.head_dim, orig.scale
        self._init_state()

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        sdpa = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
        sdpa = self._apply(sdpa, B, L, self.n_heads)
        return self.o_proj(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))


class NeoXMaskedAttention(MaskedAttentionBase):
    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        self.query_key_value, self.dense = orig.query_key_value, orig.dense
        self.rope = orig.rope
        self.num_attention_heads = orig.num_attention_heads
        self.head_dim, self.hidden_size = orig.head_dim, orig.hidden_size
        self.scale = orig.scale
        self._init_state()

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        n = self.num_attention_heads
        qkv = self.query_key_value(x).reshape(B, L, n, 3 * self.head_dim)
        q, k, v = [t.transpose(0, 2, 1, 3) for t in qkv.split(3, -1)]
        q, k = self.rope(q), self.rope(k)
        sdpa = scaled_dot_product_attention(q, k, v, cache=None, scale=self.scale, mask=mask)
        sdpa = self._apply(sdpa, B, L, n)
        return self.dense(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))


# ── helpers ─────────────────────────────────────────────────────────────────

def install(model, kind: str) -> list:
    # Llama exposes blocks at model.model.layers; GPT-NeoX exposes them at
    # model.layers (a property shorthand over model.model.h). hasattr(model,
    # "model") is True for both, so it cannot be used to discriminate.
    layers = model.model.layers if kind == "llama" else model.layers
    attns = []
    for layer in layers:
        if kind == "llama":
            new = LlamaMaskedAttention(layer.self_attn)
            layer.self_attn = new
        else:
            new = NeoXMaskedAttention(layer.attention)
            layer.attention = new
        attns.append(new)
    return attns


def set_ablation(attns: list, pairs: set) -> None:
    """pairs: set of (layer, head) to mean-ablate. Everything else runs normally."""
    by_layer: dict = {}
    for (l, h) in pairs:
        by_layer.setdefault(l, set()).add(h)
    for i, a in enumerate(attns):
        a.ablate_heads = frozenset(by_layer.get(i, ()))
        a.mode = "ablate" if a.ablate_heads else "normal"


def set_mode(attns: list, mode: str) -> None:
    for a in attns:
        a.mode = mode
        a.ablate_heads = frozenset()


def pad(seqs, max_len):
    return mx.array([s + [0] * (max_len - len(s)) for s in seqs])


def logit_diff(logits, last_pos, io_tids, s_tids):
    return [float(logits[i, last_pos[i], io_tids[i]]) - float(logits[i, last_pos[i], s_tids[i]])
            for i in range(len(last_pos))]


def run_batches(model, attns, batches, pairs: Optional[set]):
    """Mean LD over all examples with `pairs` mean-ablated (None = clean)."""
    out = []
    for toks, last, io_t, s_t in batches:
        if pairs is None:
            set_mode(attns, "normal")
        else:
            set_ablation(attns, pairs)
        logits = model(toks)
        mx.eval(logits)
        out.extend(logit_diff(logits, last, io_t, s_t))
    return sum(out) / len(out), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--dataset", default="dataset.json",
                    help="path relative to data/ioi/, or an absolute path")
    ap.add_argument("--circuit-file", default=None,
                    help='JSON with {"circuit": [[layer, head], ...]}. Overrides the '
                         "built-in per-model circuit. Required for robustness testing, "
                         "where the circuit is discovered on D and scored on D-prime "
                         "and so cannot be a constant baked into this file.")
    ap.add_argument("--out", default=None, help="override output filename")
    args = ap.parse_args()
    cfg = MODELS[args.model]
    random.seed(SEED)

    ds_path = pathlib.Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = DATA_DIR / args.dataset
    ds = json.loads(ds_path.read_text())
    examples = ds["examples"]
    print(f"Loading {cfg['id']} …", flush=True)
    model, tokenizer = load(cfg["id"])
    attns = install(model, args.model)
    print(f"  installed {len(attns)} maskable attention layers", flush=True)

    # tokenize
    batches = []
    for i in range(0, len(examples), BATCH):
        chunk = examples[i:i + BATCH]
        seqs, last, io_t, s_t = [], [], [], []
        for e in chunk:
            # BOS matters for Llama; the patching/ablation harness in this
            # directory uses add_special_tokens=True for prompts and False for
            # the name tokens. Match it, or clean LD does not line up with the
            # rest of the paper (4.77 vs 5.64 on the v1 dataset).
            ids = tokenizer.encode(e["prompt"], add_special_tokens=True)
            seqs.append(ids)
            last.append(len(ids) - 1)
            io_t.append(tokenizer.encode(" " + e["io_name"], add_special_tokens=False)[0])
            s_t.append(tokenizer.encode(" " + e["subject_name"], add_special_tokens=False)[0])
        m = max(len(s) for s in seqs)
        batches.append((pad(seqs, m), last, io_t, s_t))

    # 1. clean baseline, and cache per-layer mean SDPA
    set_mode(attns, "cache_mean")
    for toks, *_ in batches:
        mx.eval(model(toks))
    ld_clean, _ = run_batches(model, attns, batches, None)
    print(f"clean LD = {ld_clean:.4f}", flush=True)

    # 2. full ablation floor: every head mean-ablated
    all_heads = {(l, h) for l in range(cfg["n_layers"]) for h in range(cfg["n_heads"])}
    ld_floor, _ = run_batches(model, attns, batches, all_heads)
    print(f"all-heads-ablated LD = {ld_floor:.4f}", flush=True)

    if args.circuit_file:
        cf = pathlib.Path(args.circuit_file)
        circuit = {tuple(x) for x in json.loads(cf.read_text())["circuit"]}
        print(f"  circuit from {cf}: {len(circuit)} heads")
    else:
        circuit = {tuple(x) for x in cfg["circuit"]}

    def norm(ld: float) -> float:
        """Fraction of clean LD recovered above the all-ablated floor."""
        return (ld - ld_floor) / (ld_clean - ld_floor)

    # 3. FAITHFULNESS: ablate everything outside the circuit
    t0 = time.time()
    ld_circ, _ = run_batches(model, attns, batches, all_heads - circuit)
    faith = norm(ld_circ)
    print(f"faithfulness: LD={ld_circ:.4f}  F(C)={faith:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # 4. MINIMALITY: drop each head from the circuit
    minimality = []
    for v in sorted(circuit):
        ld_v, _ = run_batches(model, attns, batches, (all_heads - circuit) | {v})
        minimality.append({
            "head": f"L{v[0]}H{v[1]}",
            "layer": v[0], "head_idx": v[1],
            "F_without": round(norm(ld_v), 4),
            "delta_F": round(faith - norm(ld_v), 4),
        })
        print(f"  minimality L{v[0]}H{v[1]}: dF={faith - norm(ld_v):+.4f}", flush=True)
    minimality.sort(key=lambda r: -r["delta_F"])

    # 5. COMPLETENESS: random subsets K, compare circuit vs full model
    completeness = []
    circ_list = sorted(circuit)
    for _ in range(N_COMPLETENESS_SUBSETS):
        k = random.randint(1, max(1, len(circ_list) // 2))
        K = set(random.sample(circ_list, k))
        ld_ck, _ = run_batches(model, attns, batches, (all_heads - circuit) | K)
        ld_mk, _ = run_batches(model, attns, batches, K)
        completeness.append({
            "K": [f"L{l}H{h}" for l, h in sorted(K)],
            "F_circuit_minus_K": round(norm(ld_ck), 4),
            "F_model_minus_K": round(norm(ld_mk), 4),
            "gap": round(abs(norm(ld_ck) - norm(ld_mk)), 4),
        })
    gaps = [c["gap"] for c in completeness]

    result = {
        "meta": {
            "model": cfg["id"], "dataset": str(ds_path),
            "circuit_source": args.circuit_file or "built-in",
            "n_examples": len(examples),
            "n_layers": cfg["n_layers"], "n_heads": cfg["n_heads"],
            "circuit": [f"L{l}H{h}" for l, h in sorted(circuit)],
            "circuit_size": len(circuit),
            "seed": SEED, "synthetic": False,
            "normalization": "F(X) = (LD(X) - LD_all_ablated) / (LD_clean - LD_all_ablated)",
        },
        "logit_diffs": {
            "clean": round(ld_clean, 4),
            "all_heads_ablated": round(ld_floor, 4),
            "circuit_only": round(ld_circ, 4),
        },
        "faithfulness": round(faith, 4),
        "minimality": minimality,
        "completeness": {
            "n_subsets": N_COMPLETENESS_SUBSETS,
            "mean_gap": round(sum(gaps) / len(gaps), 4),
            "max_gap": round(max(gaps), 4),
            "subsets": completeness,
        },
    }
    out = pathlib.Path(args.out) if args.out else DATA_DIR / cfg["out"]
    if not out.is_absolute():
        out = DATA_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"\nfaithfulness={faith:.4f}  mean completeness gap={result['completeness']['mean_gap']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
