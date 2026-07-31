"""
IOI Activation Patching: Llama-3.2-3B
Full [n_layers × n_heads] sweep measuring normalised logit-diff recovery.

Metric
------
For each (layer l, head h):
  score = mean_over_examples[
      (logit_diff_patched - logit_diff_corrupted) /
      (logit_diff_clean   - logit_diff_corrupted)
  ]
where logit_diff = logit(IO name) - logit(S name) at the last token position,
and "corrupted" means the prompt with IO and S names swapped.

Strategy
--------
1. Replace every attention module with PatchableAttention (same parameters, adds
   SDPA interception for caching / head replacement).
2. Run one batched clean pass (100 examples) → cache per-head SDPA outputs.
3. Run one batched corrupt pass → cache per-head SDPA outputs.
4. For each (l, h): run one batched pass on corrupt input but replace head h at
   layer l with the cached clean SDPA output → compute recovery.
Total: 2 + 28*24 = 674 forward passes over the batch.
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
ROOT_DIR = DATA_DIR.parent.parent
MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"


# ── PatchableAttention ────────────────────────────────────────────────────────

class PatchableAttention(nn.Module):
    """Drop-in for Llama Attention with an SDPA interception point."""

    def __init__(self, orig: nn.Module) -> None:
        super().__init__()
        # Share weight tensors — no copies
        self.q_proj    = orig.q_proj
        self.k_proj    = orig.k_proj
        self.v_proj    = orig.v_proj
        self.o_proj    = orig.o_proj
        self.rope      = orig.rope
        self.n_heads   = orig.n_heads
        self.n_kv_heads = orig.n_kv_heads
        self.head_dim  = orig.head_dim
        self.scale     = orig.scale

        # Patching state (mutated between passes, not part of model parameters)
        self.mode: str = "normal"           # normal | cache_clean | cache_corrupt | patch
        self.clean_sdpa:   Optional[mx.array] = None
        self.corrupt_sdpa: Optional[mx.array] = None
        self.patch_head:   Optional[int]       = None

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
        if mode == "cache_clean":
            self.clean_sdpa = sdpa
        elif mode == "cache_corrupt":
            self.corrupt_sdpa = sdpa
        elif mode == "patch" and self.patch_head is not None:
            h = self.patch_head
            sdpa = mx.concatenate([
                sdpa[:, :h],
                self.clean_sdpa[:, h : h + 1],
                sdpa[:, h + 1 :],
            ], axis=1)

        out = sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_all_modes(attns: list, mode: str, patch_head: Optional[int] = None) -> None:
    for a in attns:
        a.mode = mode
        a.patch_head = patch_head


def logit_diffs(
    logits: mx.array,
    last_positions: list[int],
    io_tids: list[int],
    s_tids: list[int],
) -> list[float]:
    """Return (logit_io - logit_s) at the last real token for each example."""
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
    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.model.layers)
    n_heads  = model.model.layers[0].self_attn.n_heads
    print(f"  n_layers={n_layers}  n_heads={n_heads}")

    # ── Replace attention modules ─────────────────────────────────────────────
    attns: list[PatchableAttention] = []
    for layer in model.model.layers:
        pa = PatchableAttention(layer.self_attn)
        layer.self_attn = pa
        attns.append(pa)
    print(f"Replaced {len(attns)} attention modules with PatchableAttention")

    # ── Validate that replacement is transparent ──────────────────────────────
    test_ids = mx.array(
        [tokenizer.encode("Hello world", add_special_tokens=True)], dtype=mx.uint32
    )
    set_all_modes(attns, "normal")
    test_logits = model(test_ids)
    mx.eval(test_logits)
    print(f"  validation pass OK — output shape {test_logits.shape}")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset  = json.loads((DATA_DIR / ARGS.dataset).read_text())
    examples = dataset["examples"]
    # v1 datasets carry a single meta.template and the corrupt prompt is
    # reconstructed from it. v2 has 15 templates and stores a per-example
    # corrupt_prompt instead, so prefer that when present.
    template = dataset.get("meta", {}).get("template")
    print(f"Dataset: {len(examples)} examples")

    # ── Tokenise ──────────────────────────────────────────────────────────────
    io_tids: list[int] = []
    s_tids:  list[int] = []
    clean_encs:   list[list[int]] = []
    corrupt_encs: list[list[int]] = []

    for ex in examples:
        io_name = ex["io_name"]
        s_name  = ex["subject_name"]
        clean_encs.append(tokenizer.encode(ex["prompt"], add_special_tokens=True))
        if "corrupt_prompt" in ex:
            corrupt_prompt = ex["corrupt_prompt"]
        elif template is not None:
            corrupt_prompt = template.format(S=io_name, IO=s_name)
        else:
            raise KeyError("example lacks 'corrupt_prompt' and dataset lacks meta.template")
        corrupt_encs.append(tokenizer.encode(corrupt_prompt, add_special_tokens=True))
        io_tids.append(name_token_id(tokenizer, io_name))
        s_tids.append(name_token_id(tokenizer, s_name))

    clean_lens   = [len(e) for e in clean_encs]
    corrupt_lens = [len(e) for e in corrupt_encs]
    print(f"  clean  lengths : min={min(clean_lens)}  max={max(clean_lens)}")
    print(f"  corrupt lengths: min={min(corrupt_lens)}  max={max(corrupt_lens)}")

    # Sanity: per-example length match (names must tokenise to same length)
    mismatches = [(i, c, k) for i, (c, k) in enumerate(zip(clean_lens, corrupt_lens)) if c != k]
    if mismatches:
        print(f"WARNING: {len(mismatches)} examples have clean/corrupt length mismatch:")
        for i, c, k in mismatches[:5]:
            print(f"  ex {i}: clean={c} corrupt={k}")
        # Use corrupt_last positions as proxy; SDPA shapes will still match within each pass
    else:
        print("  all clean/corrupt lengths match ✓")

    max_clean   = max(clean_lens)
    max_corrupt = max(corrupt_lens)
    clean_ids   = pad_batch(clean_encs,   max_clean)    # [100, L_clean]
    corrupt_ids = pad_batch(corrupt_encs, max_corrupt)  # [100, L_corrupt]

    clean_last  = [l - 1 for l in clean_lens]
    corrupt_last = [l - 1 for l in corrupt_lens]

    # ── Clean forward pass ────────────────────────────────────────────────────
    print("\nClean forward pass (batch=100)...")
    t1 = time.time()
    set_all_modes(attns, "cache_clean")
    clean_logits = model(clean_ids)
    mx.eval(clean_logits, *[a.clean_sdpa for a in attns])
    clean_diffs = logit_diffs(clean_logits, clean_last, io_tids, s_tids)
    mean_clean = sum(clean_diffs) / len(clean_diffs)
    print(f"  {time.time()-t1:.1f}s  mean logit diff = {mean_clean:.4f}")

    # ── Corrupt forward pass ──────────────────────────────────────────────────
    print("Corrupt forward pass (batch=100)...")
    t1 = time.time()
    set_all_modes(attns, "cache_corrupt")
    corrupt_logits = model(corrupt_ids)
    mx.eval(corrupt_logits, *[a.corrupt_sdpa for a in attns])
    corrupt_diffs = logit_diffs(corrupt_logits, corrupt_last, io_tids, s_tids)
    mean_corrupt = sum(corrupt_diffs) / len(corrupt_diffs)
    print(f"  {time.time()-t1:.1f}s  mean logit diff = {mean_corrupt:.4f}")

    if mean_corrupt >= mean_clean:
        print("WARNING: corrupt mean >= clean mean — IOI effect may be weak or corruption is incorrect")

    # ── Patching sweep ────────────────────────────────────────────────────────
    total_pairs = n_layers * n_heads
    print(f"\nPatching sweep: {n_layers} layers × {n_heads} heads = {total_pairs} passes")
    print("(all 100 examples batched per pass)")

    patching_scores: list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]
    done = 0
    t_sweep = time.time()

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            # All layers run normally; only layer_idx patches head head_idx
            set_all_modes(attns, "normal")
            attns[layer_idx].mode       = "patch"
            attns[layer_idx].patch_head = head_idx

            patched_logits = model(corrupt_ids)
            mx.eval(patched_logits)

            patched_diffs = logit_diffs(patched_logits, corrupt_last, io_tids, s_tids)

            recoveries: list[float] = []
            for i in range(len(examples)):
                denom = clean_diffs[i] - corrupt_diffs[i]
                if abs(denom) < 1e-6:
                    recoveries.append(0.0)
                else:
                    recoveries.append((patched_diffs[i] - corrupt_diffs[i]) / denom)

            patching_scores[layer_idx][head_idx] = sum(recoveries) / len(recoveries)
            done += 1

            if head_idx == n_heads - 1:  # end of each layer
                elapsed = time.time() - t_sweep
                eta     = elapsed / done * (total_pairs - done)
                top_h   = max(range(n_heads), key=lambda h: patching_scores[layer_idx][h])
                print(
                    f"  L{layer_idx:02d} done  "
                    f"top_head=H{top_h:02d} ({patching_scores[layer_idx][top_h]:.3f})  "
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
            "metric": "normalized_logit_diff_recovery",
            "description": (
                "Per-head activation patching. Score = mean over examples of "
                "(logit_diff_patched - logit_diff_corrupted) / "
                "(logit_diff_clean - logit_diff_corrupted). "
                "Corrupted = IO and S names swapped."
            ),
            "synthetic": False,
        },
        "patching_scores": patching_scores,
    }

    out_path = DATA_DIR / ARGS.out
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_vals = [
        (patching_scores[l][h], l, h)
        for l in range(n_layers)
        for h in range(n_heads)
    ]
    top10 = sorted(all_vals, reverse=True)[:10]
    print("\nTop-10 heads by patching score:")
    for score, l, h in top10:
        print(f"  L{l:02d}·H{h:02d}: {score:.4f}")


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset.json",
                    help="dataset filename inside data/ioi/")
    ap.add_argument("--out", default="patching-llama3b.json",
                    help="output filename inside data/ioi/")
    return ap.parse_args()


ARGS = _parse_args() if __name__ == "__main__" else argparse.Namespace(
    dataset="dataset.json", out="patching-llama3b.json")


if __name__ == "__main__":
    main()
