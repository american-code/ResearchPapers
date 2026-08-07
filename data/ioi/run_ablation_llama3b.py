"""
IOI Mean Ablation: Llama-3.2-3B
For every (layer, head), replace that head's SDPA output with the per-head mean
computed over the 100 IOI clean examples. Measure the drop in IOI logit difference.

Metric
------
For each (layer l, head h):
  ablated_diff = mean_over_examples[logit_io - logit_s  when head (l,h) is mean-ablated]
  drop = clean_diff_mean - ablated_diff_mean
  (positive drop → head was boosting the IO logit relative to S)

Strategy
--------
1. Replace every attention module with AblationAttention.
2. Run one batched clean pass (100 examples) → store per-head SDPA outputs.
3. Compute per-head mean SDPA over the batch: [1, n_heads, L, head_dim] per layer.
4. For each (l, h): run one batched clean pass with head h at layer l replaced
   by its mean → compute ablated logit diffs.
Total: 1 + 28*24 = 673 forward passes over the batch.
"""
import argparse
import json, time
from pathlib import Path
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

DATA_DIR = Path(__file__).parent
MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"


# ── AblationAttention ─────────────────────────────────────────────────────────

class AblationAttention(nn.Module):
    """Drop-in for Llama Attention supporting mean-ablation of individual heads."""

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

        # Mutable state
        self.mode: str = "normal"           # normal | cache_mean | ablate
        self.mean_sdpa: Optional[mx.array] = None   # [1, n_heads, L, head_dim]
        self.ablate_head: Optional[int]    = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys    = self.k_proj(x)
        values  = self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        keys    = keys.reshape(   B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values  = values.reshape( B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys    = self.rope(keys)

        sdpa = scaled_dot_product_attention(
            queries, keys, values, cache=None, scale=self.scale, mask=mask
        )  # [B, n_heads, L, head_dim]

        mode = self.mode
        if mode == "cache_mean":
            # Accumulate; we'll average after mx.eval in main
            self.mean_sdpa = sdpa
        elif mode == "ablate" and self.ablate_head is not None:
            h = self.ablate_head
            # Replace head h with its mean across examples (broadcast over B)
            mean_h = self.mean_sdpa[:, h : h + 1, :, :]  # [1, 1, L, head_dim]
            ablated = mx.concatenate([
                sdpa[:, :h],
                mx.broadcast_to(mean_h, (B, 1, L, self.head_dim)),
                sdpa[:, h + 1 :],
            ], axis=1)
            sdpa = ablated

        out = sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_all_modes(attns: list, mode: str, ablate_head: Optional[int] = None) -> None:
    for a in attns:
        a.mode = mode
        a.ablate_head = ablate_head


def logit_diffs(
    logits: mx.array,
    last_positions: list[int],
    io_tids: list[int],
    s_tids: list[int],
) -> list[float]:
    out = []
    for i in range(len(last_positions)):
        pos = last_positions[i]
        l_io = float(logits[i, pos, io_tids[i]])
        l_s  = float(logits[i, pos,  s_tids[i]])
        out.append(l_io - l_s)
    return out


def name_token_id(tokenizer, name: str) -> int:
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) > 1:
        print(f"  WARNING: ' {name}' → {len(ids)} tokens; using first")
    return ids[0]


def pad_batch(seqs: list[list[int]], max_len: int) -> mx.array:
    padded = [s + [0] * (max_len - len(s)) for s in seqs]
    return mx.array(padded, dtype=mx.uint32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.model.layers)
    n_heads  = model.model.layers[0].self_attn.n_heads
    head_dim = model.model.layers[0].self_attn.head_dim
    print(f"  n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")

    # ── Replace attention modules ─────────────────────────────────────────────
    attns: list[AblationAttention] = []
    for layer in model.model.layers:
        aa = AblationAttention(layer.self_attn)
        layer.self_attn = aa
        attns.append(aa)
    print(f"Replaced {len(attns)} attention modules with AblationAttention")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset  = json.loads((DATA_DIR / ARGS.dataset).read_text())
    examples = dataset["examples"]
    print(f"Dataset: {len(examples)} examples")

    # ── Tokenise ──────────────────────────────────────────────────────────────
    io_tids: list[int] = []
    s_tids:  list[int] = []
    clean_encs: list[list[int]] = []

    for ex in examples:
        clean_encs.append(tokenizer.encode(ex["prompt"], add_special_tokens=True))
        io_tids.append(name_token_id(tokenizer, ex["io_name"]))
        s_tids.append(name_token_id(tokenizer, ex["subject_name"]))

    clean_lens = [len(e) for e in clean_encs]
    seq_len    = clean_lens[0]
    print(f"  sequence lengths: min={min(clean_lens)}  max={max(clean_lens)}")
    if len(set(clean_lens)) > 1:
        print("  WARNING: variable-length sequences; mean computed over padded positions")

    max_len   = max(clean_lens)
    clean_ids = pad_batch(clean_encs, max_len)          # [100, L]
    last_pos  = [l - 1 for l in clean_lens]

    # ── Clean forward pass → cache per-head SDPA, compute mean over batch ─────
    print("\nClean forward pass (batch=100) → computing per-head means...")
    t1 = time.time()
    set_all_modes(attns, "cache_mean")
    clean_logits = model(clean_ids)
    mx.eval(clean_logits, *[a.mean_sdpa for a in attns])

    clean_diffs = logit_diffs(clean_logits, last_pos, io_tids, s_tids)
    mean_clean  = sum(clean_diffs) / len(clean_diffs)
    print(f"  {time.time()-t1:.1f}s  mean logit diff (clean) = {mean_clean:.4f}")

    # Average the cached SDPA over the batch dimension → [1, n_heads, L, head_dim]
    for a in attns:
        # mean_sdpa is [B, n_heads, L, head_dim]; reduce over B
        a.mean_sdpa = mx.mean(a.mean_sdpa, axis=0, keepdims=True)  # [1, n_heads, L, head_dim]
    mx.eval(*[a.mean_sdpa for a in attns])
    print("  per-head means computed and cached")

    # ── Ablation sweep ────────────────────────────────────────────────────────
    total_pairs = n_layers * n_heads
    print(f"\nAblation sweep: {n_layers} layers × {n_heads} heads = {total_pairs} passes")

    # ablation_results[l][h] = mean logit diff when head (l,h) is mean-ablated
    ablation_diffs: list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]
    drop_scores:    list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]

    done    = 0
    t_sweep = time.time()

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            set_all_modes(attns, "normal")
            attns[layer_idx].mode       = "ablate"
            attns[layer_idx].ablate_head = head_idx

            ablated_logits = model(clean_ids)
            mx.eval(ablated_logits)

            abl_diffs = logit_diffs(ablated_logits, last_pos, io_tids, s_tids)
            mean_abl  = sum(abl_diffs) / len(abl_diffs)

            ablation_diffs[layer_idx][head_idx] = mean_abl
            drop_scores[layer_idx][head_idx]    = mean_clean - mean_abl
            done += 1

            if head_idx == n_heads - 1:
                elapsed = time.time() - t_sweep
                eta     = elapsed / done * (total_pairs - done)
                top_h   = max(range(n_heads), key=lambda h: drop_scores[layer_idx][h])
                print(
                    f"  L{layer_idx:02d} done  "
                    f"top_head=H{top_h:02d} (drop={drop_scores[layer_idx][top_h]:.3f})  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                )

    elapsed_total = time.time() - t_sweep
    print(f"\nSweep completed in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "meta": {
            "model": MODEL_ID,
            "n_layers": n_layers,
            "n_heads":  n_heads,
            "n_examples": len(examples),
            "metric": "logit_diff_drop",
            "description": (
                "Per-head mean ablation. For each (layer, head), the head's SDPA output "
                "is replaced with the mean over all 100 IOI clean examples (averaged over "
                "the batch dimension, preserving the sequence-position structure). "
                "drop = clean_mean_logit_diff - ablated_mean_logit_diff. "
                "Positive drop means the head contributes positively to IO prediction."
            ),
            "clean_logit_diff_mean": round(mean_clean, 6),
            "synthetic": False,
        },
        "ablation_diffs": ablation_diffs,   # mean logit diff per (layer, head)
        "drop_scores":    drop_scores,       # clean_mean - ablated_mean per (layer, head)
    }

    out_path = DATA_DIR / ARGS.out
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_vals = [
        (drop_scores[l][h], l, h)
        for l in range(n_layers)
        for h in range(n_heads)
    ]
    top10 = sorted(all_vals, reverse=True)[:10]
    print(f"\nTop-10 heads by logit-diff drop (ablation sensitivity):")
    for drop, l, h in top10:
        print(f"  L{l:02d}·H{h:02d}: drop={drop:.4f}  (ablated_mean={ablation_diffs[l][h]:.4f})")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.json",
                    help="dataset filename inside data/ioi/")
    ap.add_argument("--out", default="ablation-llama3b.json",
                    help="output filename inside data/ioi/")
    return ap.parse_args()


ARGS = _parse_args() if __name__ == "__main__" else argparse.Namespace(
    dataset="dataset.json", out="ablation-llama3b.json")


if __name__ == "__main__":
    main()
