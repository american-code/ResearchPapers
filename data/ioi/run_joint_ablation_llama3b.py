#!/usr/bin/env python3
"""
Joint circuit ablation for Llama-3.2-3B on the IOI task.

faithfulness-llama3b-v2.json already has per-head marginal scores and the
circuit-only faithfulness (ablating all non-circuit heads). This script adds
the complementary measurement:

  JOINT ABLATION  Ablate all 10 circuit heads simultaneously from the FULL
                  model (all non-circuit heads left active) and measure the
                  residual logit difference:
                      LD_residual = LD(model minus circuit)
                      F_complete  = norm(LD_residual)
                  A complete circuit drives F_complete toward 0 (removing it
                  collapses performance to near the all-ablated floor).

  LEAVE-ONE-IN    For each circuit head h, ablate the other 9 while leaving h
                  active (all non-circuit heads also active). Measures whether
                  any single head can restore completeness on its own:
                      F_model_minus_circuit_minus_h = norm(LD_{-9 heads, h active})
                      delta = F_model_minus_circuit_minus_h - F_complete
                  Large delta means h alone carries a disproportionate share of
                  the circuit's work.

Inputs:
  data/ioi/dataset-v2.json
  data/ioi/faithfulness-llama3b-v2.json  (circuit definition and stored baselines)

Output:
  data/ioi/joint-ablation-llama3b.json

Usage:
  python3 run_joint_ablation_llama3b.py
"""

import json
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

DATA_DIR = Path(__file__).parent
MODEL_ID  = "mlx-community/Llama-3.2-3B-bf16"
FAITH_FILE = DATA_DIR / "faithfulness-llama3b-v2.json"
DATASET    = DATA_DIR / "dataset-v2.json"
OUT_FILE   = DATA_DIR / "joint-ablation-llama3b.json"

BATCH = 25


# ── Maskable attention (identical to run_circuit_faithfulness.py) ─────────────

class MaskedAttentionBase(nn.Module):
    def _init_state(self):
        self.mode: str = "normal"
        self.mean_sdpa: Optional[mx.array] = None
        self.ablate_heads: frozenset = frozenset()

    def _apply(self, sdpa, B, L, n_heads):
        if self.mode == "cache_mean":
            self.mean_sdpa = mx.mean(sdpa, axis=0, keepdims=True)
            return sdpa
        if self.mode == "ablate" and self.ablate_heads:
            if self.mean_sdpa is None:
                raise RuntimeError("must cache_mean before ablating")
            cols = []
            for h in range(n_heads):
                if h in self.ablate_heads:
                    m = self.mean_sdpa[:, h:h+1, :, :]
                    cols.append(mx.broadcast_to(m, (B, 1, L, self.head_dim)))
                else:
                    cols.append(sdpa[:, h:h+1, :, :])
            return mx.concatenate(cols, axis=1)
        return sdpa


class LlamaMaskedAttention(MaskedAttentionBase):
    def __init__(self, orig):
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


def install(model):
    attns = []
    for layer in model.model.layers:
        new = LlamaMaskedAttention(layer.self_attn)
        layer.self_attn = new
        attns.append(new)
    return attns


def set_ablation(attns, pairs):
    by_layer = {}
    for (l, h) in pairs:
        by_layer.setdefault(l, set()).add(h)
    for i, a in enumerate(attns):
        a.ablate_heads = frozenset(by_layer.get(i, ()))
        a.mode = "ablate" if a.ablate_heads else "normal"


def set_mode(attns, mode):
    for a in attns:
        a.mode = mode
        a.ablate_heads = frozenset()


def pad(seqs, max_len):
    return mx.array([s + [0] * (max_len - len(s)) for s in seqs])


def logit_diff(logits, last_pos, io_tids, s_tids):
    return [
        float(logits[i, last_pos[i], io_tids[i]]) - float(logits[i, last_pos[i], s_tids[i]])
        for i in range(len(last_pos))
    ]


def run_batches(model, attns, batches, pairs):
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
    return sum(out) / len(out)


def main():
    t0 = time.time()

    faith = json.loads(FAITH_FILE.read_text())
    circuit_str = faith["meta"]["circuit"]
    circuit = frozenset(
        (int(s[1:s.index("H")]), int(s[s.index("H")+1:]))
        for s in circuit_str
    )
    n_layers = faith["meta"]["n_layers"]
    n_heads  = faith["meta"]["n_heads"]
    all_heads = frozenset((l, h) for l in range(n_layers) for h in range(n_heads))

    # Stored baselines from the faithfulness run (no need to recompute)
    ld_clean  = faith["logit_diffs"]["clean"]
    ld_floor  = faith["logit_diffs"]["all_heads_ablated"]
    ld_faith  = faith["logit_diffs"]["circuit_only"]
    F_faith   = faith["faithfulness"]

    def norm(ld):
        return (ld - ld_floor) / (ld_clean - ld_floor)

    print(f"Circuit: {sorted(circuit_str)}")
    print(f"Stored baselines  clean={ld_clean:.4f}  floor={ld_floor:.4f}  "
          f"circuit_only={ld_faith:.4f}  F={F_faith:.4f}", flush=True)

    print(f"\nLoading {MODEL_ID} …", flush=True)
    model, tokenizer = load(MODEL_ID)
    attns = install(model)
    print(f"  {len(attns)} maskable layers installed", flush=True)

    ds = json.loads(DATASET.read_text())
    examples = ds["examples"]
    print(f"  {len(examples)} examples from {DATASET.name}", flush=True)

    # Tokenise into batches
    batches = []
    for i in range(0, len(examples), BATCH):
        chunk = examples[i:i+BATCH]
        seqs, last, io_t, s_t = [], [], [], []
        for e in chunk:
            ids = tokenizer.encode(e["prompt"], add_special_tokens=True)
            seqs.append(ids)
            last.append(len(ids) - 1)
            io_t.append(tokenizer.encode(" " + e["io_name"], add_special_tokens=False)[0])
            s_t.append(tokenizer.encode(" " + e["subject_name"], add_special_tokens=False)[0])
        m = max(len(s) for s in seqs)
        batches.append((pad(seqs, m), last, io_t, s_t))

    # Cache per-layer mean SDPA (needed for ablation)
    print("\nCaching clean mean SDPA …", flush=True)
    set_mode(attns, "cache_mean")
    for toks, *_ in batches:
        mx.eval(model(toks))

    # Verify clean LD matches stored value (sanity check)
    ld_verify = run_batches(model, attns, batches, None)
    print(f"Verified clean LD = {ld_verify:.4f}  (stored {ld_clean:.4f}  "
          f"delta={abs(ld_verify - ld_clean):.4f})", flush=True)

    # ── 1. JOINT ABLATION: all 10 circuit heads ablated, rest active ──────────
    print("\nRunning joint ablation (all 10 circuit heads) …", flush=True)
    t1 = time.time()
    ld_residual = run_batches(model, attns, batches, circuit)
    F_complete  = norm(ld_residual)
    print(f"  LD_model_minus_circuit = {ld_residual:.4f}  "
          f"F_complete = {F_complete:.4f}  ({time.time()-t1:.0f}s)", flush=True)

    # ── 2. LEAVE-ONE-IN: ablate 9 circuit heads, leave h active ──────────────
    print("\nLeave-one-in sweep …", flush=True)
    leave_one_in = []
    circ_sorted = sorted(circuit)
    for (l, h) in circ_sorted:
        nine_heads = circuit - {(l, h)}
        t2 = time.time()
        ld_loi = run_batches(model, attns, batches, nine_heads)
        F_loi   = norm(ld_loi)
        delta   = round(F_loi - F_complete, 4)
        leave_one_in.append({
            "head":         f"L{l}H{h}",
            "layer":        l,
            "head_idx":     h,
            "LD_leave_in":  round(ld_loi, 4),
            "F_leave_in":   round(F_loi, 4),
            "delta_F":      delta,
        })
        print(f"  leave L{l}H{h} active: LD={ld_loi:.4f}  F={F_loi:.4f}  "
              f"delta={delta:+.4f}  ({time.time()-t2:.0f}s)", flush=True)

    leave_one_in.sort(key=lambda r: -r["delta_F"])

    # ── Write result ──────────────────────────────────────────────────────────
    result = {
        "meta": {
            "model":        MODEL_ID,
            "dataset":      DATASET.name,
            "n_examples":   len(examples),
            "n_layers":     n_layers,
            "n_heads":      n_heads,
            "circuit":      circuit_str,
            "circuit_size": len(circuit),
            "synthetic":    False,
            "normalization": "F(X) = (LD(X) - LD_all_ablated) / (LD_clean - LD_all_ablated)",
            "description": (
                "Joint ablation: all 10 circuit heads ablated simultaneously from the "
                "full model (all non-circuit heads active). Leave-one-in: ablate 9 of "
                "10 circuit heads while leaving each head active in turn."
            ),
        },
        "baselines": {
            "LD_clean":           round(ld_verify, 4),
            "LD_all_ablated":     ld_floor,
            "LD_circuit_only":    ld_faith,
            "F_faithfulness":     F_faith,
            "source": "clean/floor/circuit_only from faithfulness-llama3b-v2.json; "
                      "LD_clean re-verified this run",
        },
        "joint_ablation": {
            "LD_model_minus_circuit": round(ld_residual, 4),
            "F_complete":             round(F_complete, 4),
            "interpretation": (
                "Fraction of clean LD that survives removing all 10 circuit heads "
                "simultaneously. Low value (→0) means the circuit accounts for most "
                "of the model's IOI behaviour (high completeness). "
                "Compare with F_faithfulness=%.4f (circuit-only direction)." % F_faith
            ),
        },
        "leave_one_in": {
            "description": (
                "For each circuit head h: ablate the other 9 circuit heads while "
                "leaving h active (all non-circuit heads also active). delta_F = "
                "F_leave_in - F_complete. Large positive delta_F means h alone can "
                "restore significant performance — h is the dominant contributor."
            ),
            "F_complete_reference": round(F_complete, 4),
            "heads": leave_one_in,
        },
        "runtime_seconds": round(time.time() - t0, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    OUT_FILE.write_text(json.dumps(result, indent=2))
    print(f"\n--- RESULTS ---")
    print(f"F_faithfulness  (circuit only):   {F_faith:.4f}  (LD={ld_faith:.4f})")
    print(f"F_complete      (joint ablation): {F_complete:.4f}  (LD={ld_residual:.4f})")
    print(f"\nLeave-one-in ranking (by delta_F, descending):")
    for r in leave_one_in:
        bar = "#" * max(0, int(r["delta_F"] * 100))
        print(f"  {r['head']:8s}  F={r['F_leave_in']:.4f}  delta={r['delta_F']:+.4f}  {bar}")
    print(f"\nWrote {OUT_FILE}  ({result['runtime_seconds']:.0f}s total)")


if __name__ == "__main__":
    main()
