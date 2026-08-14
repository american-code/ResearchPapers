#!/usr/bin/env python3
"""Shared attention surgery for circuit discovery and evaluation, dataset-agnostic.

The four scripts in data/ioi/ (run_patching_{llama3b,pythia1b}.py and
run_ablation_{llama3b,pythia1b}.py) are two experiments crossed with two architectures,
each with its dataset baked in. They produced the published results and are left
untouched. This module is the generalised replacement: one attention wrapper covering
both architectures and all four intervention modes, over any dataset.

MODES
  normal       run unmodified
  cache_clean  store per-head SDPA output for this batch (patching source)
  cache_mean   accumulate the per-head mean SDPA over clean examples (ablation value)
  patch        splice one head's cached clean SDPA into the current pass
  ablate       replace a set of heads with their cached mean

TWO DEVIATIONS FROM THE data/ioi/ HARNESS, both deliberate:

1. The mean for mean-ablation is accumulated over ALL batches here. In
   run_circuit_faithfulness.py, `cache_mean` assigns rather than accumulates
   (`self.mean_sdpa = mx.mean(sdpa, axis=0, keepdims=True)`), so after looping over
   batches only the LAST batch survives -- the published mean is over 25 examples,
   not the 100 the docstring describes. It is an estimate of the same quantity either
   way, so it does not overturn the published result, but it is not what was claimed
   and it is not what a D-vs-D-prime comparison should rest on.

2. Batches are padded to a GLOBAL maximum length, not per batch. The cached mean has
   a sequence dimension, so a mean cached at one batch's padded length cannot be
   broadcast onto a batch padded to a different length. The IOI datasets happen to be
   near-uniform in length, which is why this never surfaced; the factual and
   agreement families here are not.

Both changes mean numbers from this module will not reproduce the published IOI
figures to the last decimal. That is expected and should be stated wherever the two
are shown together.
"""

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention

MODELS = {
    "llama": {"id": "mlx-community/Llama-3.2-3B-bf16", "n_layers": 28, "n_heads": 24},
    "pythia": {"id": "EleutherAI/pythia-1.4b", "n_layers": 24, "n_heads": 16},
}


class InterventionAttentionBase(nn.Module):
    def _init_state(self) -> None:
        self.mode: str = "normal"
        self.clean_sdpa: Optional[mx.array] = None
        self.mean_sum: Optional[mx.array] = None
        self.mean_n: int = 0
        self.mean_sdpa: Optional[mx.array] = None
        self.patch_head: Optional[int] = None
        self.ablate_heads: frozenset = frozenset()

    def reset_mean(self) -> None:
        self.mean_sum, self.mean_n, self.mean_sdpa = None, 0, None

    def finalize_mean(self) -> None:
        if self.mean_sum is None:
            raise RuntimeError("finalize_mean before any cache_mean pass")
        self.mean_sdpa = self.mean_sum / self.mean_n

    def _apply(self, sdpa: mx.array, B: int, L: int, n_heads: int) -> mx.array:
        if self.mode == "cache_clean":
            self.clean_sdpa = sdpa
            return sdpa
        if self.mode == "cache_mean":
            s = mx.sum(sdpa, axis=0, keepdims=True)
            self.mean_sum = s if self.mean_sum is None else self.mean_sum + s
            self.mean_n += B
            return sdpa
        if self.mode == "patch" and self.patch_head is not None:
            h = self.patch_head
            if self.clean_sdpa is None:
                raise RuntimeError("patch before cache_clean")
            return mx.concatenate(
                [sdpa[:, :h], self.clean_sdpa[:, h:h + 1], sdpa[:, h + 1:]], axis=1)
        if self.mode == "ablate" and self.ablate_heads:
            if self.mean_sdpa is None:
                raise RuntimeError("ablate before finalize_mean")
            cols = []
            for h in range(n_heads):
                if h in self.ablate_heads:
                    cols.append(mx.broadcast_to(self.mean_sdpa[:, h:h + 1, :, :],
                                                (B, 1, L, self.head_dim)))
                else:
                    cols.append(sdpa[:, h:h + 1, :, :])
            return mx.concatenate(cols, axis=1)
        return sdpa


class LlamaInterventionAttention(InterventionAttentionBase):
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


class NeoXInterventionAttention(InterventionAttentionBase):
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


def install(model, kind: str) -> list:
    # Llama exposes blocks at model.model.layers; GPT-NeoX at model.layers (a property
    # over model.model.h). hasattr(model, "model") is True for both and cannot
    # discriminate -- this is the same trap the data/ioi/ harness documents.
    layers = model.model.layers if kind == "llama" else model.layers
    attns = []
    for layer in layers:
        if kind == "llama":
            new = LlamaInterventionAttention(layer.self_attn)
            layer.self_attn = new
        else:
            new = NeoXInterventionAttention(layer.attention)
            layer.attention = new
        attns.append(new)
    return attns


def set_mode(attns, mode, patch_head=None, ablate=None):
    by_layer = {}
    for (l, h) in (ablate or ()):
        by_layer.setdefault(l, set()).add(h)
    for i, a in enumerate(attns):
        a.patch_head = patch_head if mode == "patch" else None
        a.ablate_heads = frozenset(by_layer.get(i, ())) if mode == "ablate" else frozenset()
        if mode == "ablate":
            a.mode = "ablate" if a.ablate_heads else "normal"
        elif mode == "patch":
            a.mode = "normal"
        else:
            a.mode = mode


def set_patch_target(attns, layer, head):
    """Patch exactly one head at one layer; every other layer runs normally."""
    for i, a in enumerate(attns):
        a.mode = "patch" if i == layer else "normal"
        a.patch_head = head if i == layer else None
        a.ablate_heads = frozenset()


def tokenize(examples, tokenizer, batch_size=25):
    """Tokenize clean and corrupt prompts, padded to ONE global length.

    Returns [(clean_toks, corrupt_toks, last_positions, answer_ids, distractor_ids)].
    The answer and distractor are encoded WITHOUT special tokens and with a leading
    space, matching the data/ioi/ convention; prompts keep special tokens because BOS
    materially changes Llama's logit difference.
    """
    enc = [(tokenizer.encode(e["prompt"], add_special_tokens=True),
            tokenizer.encode(e["corrupt_prompt"], add_special_tokens=True),
            tokenizer.encode(" " + e["answer_token"], add_special_tokens=False)[0],
            tokenizer.encode(" " + e["distractor_token"], add_special_tokens=False)[0])
           for e in examples]
    bad = [i for i, (c, k, _, _) in enumerate(enc) if len(c) != len(k)]
    if bad:
        raise SystemExit(
            f"{len(bad)} example(s) have clean/corrupt length mismatch (e.g. index "
            f"{bad[0]}). Regenerate with generate_shift_pairs.py, which filters these: "
            f"patching splices cached activations positionally and a mismatch would "
            f"silently score the wrong positions.")
    gmax = max(len(c) for c, _, _, _ in enc)

    out = []
    for i in range(0, len(enc), batch_size):
        chunk = enc[i:i + batch_size]
        clean = mx.array([c + [0] * (gmax - len(c)) for c, _, _, _ in chunk])
        corr = mx.array([k + [0] * (gmax - len(k)) for _, k, _, _ in chunk])
        last = [len(c) - 1 for c, _, _, _ in chunk]
        a_ids = [a for _, _, a, _ in chunk]
        d_ids = [d for _, _, _, d in chunk]
        out.append((clean, corr, last, a_ids, d_ids))
    return out


def logit_diff(logits, last, a_ids, d_ids):
    return [float(logits[i, last[i], a_ids[i]]) - float(logits[i, last[i], d_ids[i]])
            for i in range(len(last))]


def run_clean_and_cache_mean(model, attns, batches):
    """Mean-SDPA over every clean example, then the clean logit difference."""
    for a in attns:
        a.reset_mean()
    set_mode(attns, "cache_mean")
    for clean, *_ in batches:
        mx.eval(model(clean))
    for a in attns:
        a.finalize_mean()
    set_mode(attns, "normal")
    lds = []
    for clean, _, last, a_ids, d_ids in batches:
        logits = model(clean)
        mx.eval(logits)
        lds += logit_diff(logits, last, a_ids, d_ids)
    return sum(lds) / len(lds), lds


def run_ablated(model, attns, batches, heads, on_corrupt=False):
    set_mode(attns, "ablate", ablate=heads)
    lds = []
    for clean, corr, last, a_ids, d_ids in batches:
        logits = model(corr if on_corrupt else clean)
        mx.eval(logits)
        lds += logit_diff(logits, last, a_ids, d_ids)
    set_mode(attns, "normal")
    return sum(lds) / len(lds), lds
