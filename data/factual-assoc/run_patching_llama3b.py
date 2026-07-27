"""
Factual Association Activation Patching: Llama-3.2-3B
Full [n_layers × n_heads] sweep measuring normalised logit-diff recovery.

Metric
------
For each (layer l, head h):
  score = mean_over_valid_examples[
      (logit_diff_patched - logit_diff_corrupted) /
      (logit_diff_clean   - logit_diff_corrupted)
  ]
where logit_diff = logit(correct_object_token) - logit(corrupt_object_token)
at the last token position of the prompt.

Corruption strategy: swap the subject for a different entity (corrupt_subject)
whose associated fact answer (corrupt_object) is different from the clean object.

Grouping
--------
Examples are batched by sequence length. Only examples where clean and corrupt
prompts have the same length (i.e. subject and corrupt_subject tokenise to the
same number of tokens) are included. Mismatched examples are reported and skipped.

Total passes ≈ n_groups * (2 + n_layers * n_heads), batched per group.
"""
import json, time
from collections import defaultdict
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
    """Drop-in for Llama Attention with SDPA interception for head patching."""

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

        self.mode:       str              = "normal"
        self.clean_sdpa: Optional[mx.array] = None
        self.corrupt_sdpa: Optional[mx.array] = None
        self.patch_head: Optional[int]    = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads,    self.head_dim).transpose(0, 2, 1, 3)
        keys    = self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values  = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

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

        return self.o_proj(sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1))


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_all_modes(attns: list, mode: str, patch_head: Optional[int] = None) -> None:
    for a in attns:
        a.mode = mode
        a.patch_head = patch_head


def logit_diffs(
    logits: mx.array,
    last_positions: list[int],
    obj_tids: list[int],
    corrupt_obj_tids: list[int],
) -> list[float]:
    out = []
    for i in range(len(last_positions)):
        pos   = last_positions[i]
        l_obj = float(logits[i, pos, obj_tids[i]])
        l_cor = float(logits[i, pos, corrupt_obj_tids[i]])
        out.append(l_obj - l_cor)
    return out


def first_token_id(tokenizer, word: str) -> int:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(ids) > 1:
        print(f"  WARNING: ' {word}' → {len(ids)} tokens; using first")
    return ids[0]


def pad_batch(seqs: list[list[int]], length: int) -> mx.array:
    padded = [s + [0] * (length - len(s)) for s in seqs]
    return mx.array(padded, dtype=mx.uint32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_layers = len(model.model.layers)
    n_heads  = model.model.layers[0].self_attn.n_heads
    print(f"  n_layers={n_layers}  n_heads={n_heads}")

    attns: list[PatchableAttention] = []
    for layer in model.model.layers:
        pa = PatchableAttention(layer.self_attn)
        layer.self_attn = pa
        attns.append(pa)
    print(f"Replaced {len(attns)} attention modules with PatchableAttention")

    # Sanity check
    test_ids = mx.array(
        [tokenizer.encode("Hello world", add_special_tokens=True)], dtype=mx.uint32
    )
    set_all_modes(attns, "normal")
    mx.eval(model(test_ids))
    print("  validation pass OK")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset  = json.loads((DATA_DIR / "dataset.json").read_text())
    examples = dataset["examples"]
    print(f"\nDataset: {len(examples)} examples")

    # ── Tokenise ──────────────────────────────────────────────────────────────
    clean_encs:       list[list[int]] = []
    corrupt_encs:     list[list[int]] = []
    obj_tids:         list[int]       = []
    corrupt_obj_tids: list[int]       = []

    for ex in examples:
        clean_encs.append(tokenizer.encode(ex["prompt"], add_special_tokens=True))
        corrupt_encs.append(tokenizer.encode(ex["corrupt_prompt"], add_special_tokens=True))
        obj_tids.append(first_token_id(tokenizer, ex["object"]))
        corrupt_obj_tids.append(first_token_id(tokenizer, ex["corrupt_object"]))

    clean_lens   = [len(e) for e in clean_encs]
    corrupt_lens = [len(e) for e in corrupt_encs]

    # ── Separate valid examples (matched lengths) into groups ─────────────────
    # Patching requires matching sequence lengths between clean and corrupt so
    # cached SDPA tensors have the same shape.
    length_groups: dict[int, list[int]] = defaultdict(list)
    n_skipped = 0
    for i in range(len(examples)):
        cl, crl = clean_lens[i], corrupt_lens[i]
        if cl == crl:
            length_groups[cl].append(i)
        else:
            print(
                f"  SKIP ex {i:2d} ({examples[i]['subject']} / {examples[i]['corrupt_subject']}): "
                f"clean={cl} corrupt={crl}"
            )
            n_skipped += 1

    n_valid = len(examples) - n_skipped
    print(f"\nValid examples: {n_valid}/{len(examples)}  |  groups by length: {sorted(length_groups.keys())}")

    # per-example patching scores: [n_examples][n_layers][n_heads]
    example_scores: list[Optional[list[list[float]]]] = [None] * len(examples)
    group_clean_diffs:   dict[int, list[float]] = {}  # group_len -> clean diffs
    group_corrupt_diffs: dict[int, list[float]] = {}

    total_pairs   = n_layers * n_heads
    t_sweep_start = time.time()
    passes_done   = 0
    total_passes  = len(length_groups) * (2 + total_pairs)

    # ── Process each length group ─────────────────────────────────────────────
    for seq_len, indices in sorted(length_groups.items()):
        B = len(indices)
        print(f"\n── Length {seq_len} tokens │ {B} examples ──")

        g_clean_ids   = pad_batch([clean_encs[i]   for i in indices], seq_len)
        g_corrupt_ids = pad_batch([corrupt_encs[i] for i in indices], seq_len)
        g_last        = [clean_lens[i] - 1 for i in indices]
        g_obj_tids    = [obj_tids[i]         for i in indices]
        g_corr_tids   = [corrupt_obj_tids[i] for i in indices]

        # Clean pass
        t1 = time.time()
        set_all_modes(attns, "cache_clean")
        clean_logits = model(g_clean_ids)
        mx.eval(clean_logits, *[a.clean_sdpa for a in attns])
        c_diffs = logit_diffs(clean_logits, g_last, g_obj_tids, g_corr_tids)
        mean_c  = sum(c_diffs) / B
        passes_done += 1
        print(f"  clean pass   {time.time()-t1:.1f}s   mean logit_diff={mean_c:+.4f}")

        # Corrupt pass
        t1 = time.time()
        set_all_modes(attns, "cache_corrupt")
        corrupt_logits = model(g_corrupt_ids)
        mx.eval(corrupt_logits, *[a.corrupt_sdpa for a in attns])
        k_diffs = logit_diffs(corrupt_logits, g_last, g_obj_tids, g_corr_tids)
        mean_k  = sum(k_diffs) / B
        passes_done += 1
        print(f"  corrupt pass {time.time()-t1:.1f}s   mean logit_diff={mean_k:+.4f}")

        if mean_k >= mean_c:
            print("  WARNING: corrupt mean >= clean mean — factual signal may be weak")

        group_clean_diffs[seq_len]   = c_diffs
        group_corrupt_diffs[seq_len] = k_diffs

        # Initialise per-example score matrices
        for i in indices:
            example_scores[i] = [[0.0] * n_heads for _ in range(n_layers)]

        # Patching sweep
        print(f"  Patching sweep: {n_layers} × {n_heads} = {total_pairs} passes (batch={B})")
        t_group = time.time()
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                set_all_modes(attns, "normal")
                attns[layer_idx].mode       = "patch"
                attns[layer_idx].patch_head = head_idx

                patched_logits = model(g_corrupt_ids)
                mx.eval(patched_logits)

                p_diffs = logit_diffs(patched_logits, g_last, g_obj_tids, g_corr_tids)

                for k, i in enumerate(indices):
                    denom = c_diffs[k] - k_diffs[k]
                    rec   = (p_diffs[k] - k_diffs[k]) / denom if abs(denom) > 1e-6 else 0.0
                    example_scores[i][layer_idx][head_idx] = rec

                passes_done += 1

            elapsed = time.time() - t_group
            done_in_group = (layer_idx + 1) * n_heads
            eta_group = elapsed / done_in_group * (total_pairs - done_in_group)
            top_h = max(range(n_heads), key=lambda h: sum(
                example_scores[i][layer_idx][h] for i in indices
            ) / B)
            top_score = sum(example_scores[i][layer_idx][top_h] for i in indices) / B
            print(
                f"    L{layer_idx:02d} done  "
                f"top_head=H{top_h:02d} ({top_score:.3f})  "
                f"eta={eta_group:.0f}s"
            )

    elapsed_total = time.time() - t_sweep_start
    print(f"\nTotal elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    # ── Aggregate patching scores across valid examples ───────────────────────
    valid_indices = [i for i in range(len(examples)) if example_scores[i] is not None]
    print(f"\nAggregating over {len(valid_indices)} valid examples...")

    patching_scores: list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]
    if valid_indices:
        for layer_idx in range(n_layers):
            for head_idx in range(n_heads):
                patching_scores[layer_idx][head_idx] = sum(
                    example_scores[i][layer_idx][head_idx] for i in valid_indices
                ) / len(valid_indices)

    # ── Save results ──────────────────────────────────────────────────────────
    skipped_examples = [
        {"id": i, "subject": examples[i]["subject"],
         "corrupt_subject": examples[i]["corrupt_subject"],
         "reason": f"clean_len={clean_lens[i]} != corrupt_len={corrupt_lens[i]}"}
        for i in range(len(examples)) if example_scores[i] is None
    ]

    output = {
        "meta": {
            "model":       MODEL_ID,
            "n_layers":    n_layers,
            "n_heads":     n_heads,
            "n_examples":  len(examples),
            "n_valid":     len(valid_indices),
            "n_skipped":   n_skipped,
            "metric":      "normalized_logit_diff_recovery",
            "description": (
                "Per-head activation patching on factual association prompts. "
                "Score = mean over valid examples of "
                "(logit_diff_patched - logit_diff_corrupted) / "
                "(logit_diff_clean - logit_diff_corrupted). "
                "logit_diff = logit(object_token) - logit(corrupt_object_token). "
                "Corrupted = subject swapped for a different entity with different answer."
            ),
            "synthetic": False,
            "elapsed_seconds": elapsed_total,
            "skipped_examples": skipped_examples,
        },
        "patching_scores": patching_scores,
        "per_example_scores": {
            str(i): example_scores[i]
            for i in valid_indices
        },
    }

    out_path = DATA_DIR / "patching-llama3b.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    all_vals = [
        (patching_scores[l][h], l, h)
        for l in range(n_layers)
        for h in range(n_heads)
    ]
    top10 = sorted(all_vals, reverse=True)[:10]
    print("\nTop-10 heads by mean patching score:")
    for score, l, h in top10:
        print(f"  L{l:02d}·H{h:02d}: {score:.4f}")


if __name__ == "__main__":
    main()
