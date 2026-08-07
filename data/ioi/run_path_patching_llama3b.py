"""
IOI Edge-Level Path Patching: Llama-3.2-3B

Implements the 4-pass protocol from SwiftSci/Interp/PathPatching.swift,
translated to Python/MLX, testing directed edges between the top-5
circuit-critical heads identified in results-summary.md.

Protocol (per edge sender → receiver):
  Pass 1: clean baseline → clean_LD
  Pass 2: corrupted → capture sender head SDPA output → corrupted_LD
  Pass 3: clean with sender patched to corrupted value → capture receiver SDPA
  Pass 4: clean with receiver patched to pass-3 value → patched_LD
  Score  = (clean_LD - patched_LD) / (clean_LD - corrupted_LD)

Score ≈ 1 → edge fully transmits corrupted signal (in circuit)
Score ≈ 0 → edge carries no corrupted signal (not in circuit)

Optimisation (from PathPatching.swift sweep): edges sharing a sender share
one pass 2 + one pass 3.  Total passes: 2 + U_senders + N_edges = 16.

Top-5 circuit-critical heads (activation patching rank, results-summary.md):
  L15H20, L24H15, L14H0, L19H1, L17H17

Directed edges tested (10 total, forward-only):
  L14H0 → {L15H20, L17H17, L19H1, L24H15}
  L15H20 → {L17H17, L19H1, L24H15}
  L17H17 → {L19H1, L24H15}
  L19H1  → {L24H15}
"""
import json, time
from pathlib import Path
from typing import Optional, Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.base import scaled_dot_product_attention

DATA_DIR  = Path(__file__).parent
ROOT_DIR  = DATA_DIR.parent.parent
MODEL_ID  = "mlx-community/Llama-3.2-3B-bf16"

# Top-5 heads ranked by activation patching score (results-summary.md §2)
TOP5_HEADS = [(15, 20), (24, 15), (14, 0), (19, 1), (17, 17)]

# All directed forward edges between top-5 heads
EDGES: list[tuple[int, int, int, int]] = []  # (sl, sh, rl, rh)
for i, (sl, sh) in enumerate(TOP5_HEADS):
    for (rl, rh) in TOP5_HEADS:
        if rl > sl:
            EDGES.append((sl, sh, rl, rh))

EDGES.sort()  # deterministic order


# ── PathPatchableAttention ─────────────────────────────────────────────────────

class PathPatchableAttention(nn.Module):
    """Drop-in for Llama Attention supporting capture and injection modes."""

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

        # Mutated between passes; not model parameters
        self.mode: str               = "normal"  # normal | capture | inject | inject_and_capture
        self.capture_head: Optional[int]         = None
        self.inject_head:  Optional[int]         = None
        self.inject_value: Optional[mx.array]    = None  # [B, 1, L, head_dim]
        self.captured:     Optional[mx.array]    = None  # [B, 1, L, head_dim]

    def __call__(
        self,
        x:     mx.array,
        mask:  Optional[mx.array] = None,
        cache: Optional[Any]      = None,
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

        # Inject stored value for inject_head
        if mode in ("inject", "inject_and_capture") and self.inject_head is not None and self.inject_value is not None:
            h = self.inject_head
            sdpa = mx.concatenate([sdpa[:, :h], self.inject_value, sdpa[:, h + 1:]], axis=1)

        # Capture output for capture_head
        if mode in ("capture", "inject_and_capture") and self.capture_head is not None:
            self.captured = sdpa[:, self.capture_head: self.capture_head + 1]

        out = sdpa.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


# ── Helpers ───────────────────────────────────────────────────────────────────

def reset_all(attns: list[PathPatchableAttention]) -> None:
    for a in attns:
        a.mode         = "normal"
        a.capture_head = None
        a.inject_head  = None
        a.inject_value = None
        a.captured     = None


def logit_diffs(
    logits:         mx.array,
    last_positions: list[int],
    io_tids:        list[int],
    s_tids:         list[int],
) -> list[float]:
    out = []
    for i in range(len(last_positions)):
        pos  = last_positions[i]
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
    print(f"  n_layers={n_layers}  n_heads={n_heads}")

    # Replace attention modules
    attns: list[PathPatchableAttention] = []
    for layer in model.model.layers:
        pa = PathPatchableAttention(layer.self_attn)
        layer.self_attn = pa
        attns.append(pa)
    print(f"Replaced {len(attns)} attention modules")

    # Validate
    test_ids = mx.array(
        [tokenizer.encode("Hello world", add_special_tokens=True)], dtype=mx.uint32
    )
    test_logits = model(test_ids)
    mx.eval(test_logits)
    print(f"  validation pass OK — shape {test_logits.shape}")

    # Load dataset
    dataset  = json.loads((DATA_DIR / "dataset.json").read_text())
    examples = dataset["examples"]
    template = dataset["meta"]["template"]
    print(f"Dataset: {len(examples)} examples")

    # Tokenise
    io_tids: list[int] = []
    s_tids:  list[int] = []
    clean_encs:   list[list[int]] = []
    corrupt_encs: list[list[int]] = []

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

    mismatches = [(i, c, k) for i, (c, k) in enumerate(zip(clean_lens, corrupt_lens)) if c != k]
    if mismatches:
        print(f"WARNING: {len(mismatches)} clean/corrupt length mismatches (first 3): {mismatches[:3]}")
    else:
        print("  all clean/corrupt lengths match ✓")

    max_clean   = max(clean_lens)
    max_corrupt = max(corrupt_lens)
    clean_ids   = pad_batch(clean_encs,   max_clean)
    corrupt_ids = pad_batch(corrupt_encs, max_corrupt)
    clean_last  = [l - 1 for l in clean_lens]
    corrupt_last = [l - 1 for l in corrupt_lens]

    print(f"\nEdges to test ({len(EDGES)} total):")
    for sl, sh, rl, rh in EDGES:
        print(f"  L{sl}H{sh} → L{rl}H{rh}")

    # ── Pass 1: clean baseline ────────────────────────────────────────────────
    print("\nPass 1: clean baseline...")
    t1 = time.time()
    reset_all(attns)
    clean_logits = model(clean_ids)
    mx.eval(clean_logits)
    clean_diffs = logit_diffs(clean_logits, clean_last, io_tids, s_tids)
    mean_clean = sum(clean_diffs) / len(clean_diffs)
    print(f"  {time.time()-t1:.1f}s  mean clean LD = {mean_clean:.4f}")

    # ── Pass 2: corrupted baseline + capture all unique sender heads ──────────
    print("Pass 2: corrupted + capture sender heads...")
    t1 = time.time()
    unique_senders = sorted({(sl, sh) for sl, sh, rl, rh in EDGES})

    reset_all(attns)
    for sl, sh in unique_senders:
        attns[sl].mode         = "capture"
        attns[sl].capture_head = sh

    corrupt_logits = model(corrupt_ids)
    # Force eval of logits and all captured tensors
    captures_to_eval = [attns[sl].captured for sl, sh in unique_senders if attns[sl].captured is not None]
    mx.eval(corrupt_logits, *captures_to_eval)

    corrupt_diffs = logit_diffs(corrupt_logits, corrupt_last, io_tids, s_tids)
    mean_corrupt = sum(corrupt_diffs) / len(corrupt_diffs)
    print(f"  {time.time()-t1:.1f}s  mean corrupt LD = {mean_corrupt:.4f}")

    if mean_corrupt >= mean_clean:
        print("  WARNING: corrupt mean >= clean mean — check corruption")

    # Store sender corrupt activations
    sender_corrupt: dict[tuple[int, int], mx.array] = {}
    for sl, sh in unique_senders:
        assert attns[sl].captured is not None, f"Capture failed for L{sl}H{sh}"
        sender_corrupt[(sl, sh)] = attns[sl].captured

    # ── Pass 3: intermediate per unique sender ────────────────────────────────
    # For each sender: run clean with sender patched to corrupted value;
    # capture all receiver heads simultaneously.
    receiver_intermediate: dict[tuple[int, int, int, int], mx.array] = {}

    for idx, (sl, sh) in enumerate(unique_senders):
        receivers = [(rl, rh) for s_l, s_h, rl, rh in EDGES if s_l == sl and s_h == sh]
        print(f"Pass 3.{idx+1}: intermediate — L{sl}H{sh} patched → capture {['L'+str(rl)+'H'+str(rh) for rl,rh in receivers]}...")
        t1 = time.time()

        reset_all(attns)
        attns[sl].mode         = "inject"
        attns[sl].inject_head  = sh
        attns[sl].inject_value = sender_corrupt[(sl, sh)]

        for rl, rh in receivers:
            attns[rl].mode         = "capture"
            attns[rl].capture_head = rh

        model(clean_ids)
        rec_tensors = [attns[rl].captured for rl, rh in receivers if attns[rl].captured is not None]
        mx.eval(*rec_tensors)
        print(f"  {time.time()-t1:.1f}s")

        for rl, rh in receivers:
            assert attns[rl].captured is not None, f"Capture failed for receiver L{rl}H{rh}"
            receiver_intermediate[(sl, sh, rl, rh)] = attns[rl].captured

    # ── Pass 4: one patched pass per edge ─────────────────────────────────────
    results = []
    for edge_idx, (sl, sh, rl, rh) in enumerate(EDGES):
        print(f"Pass 4.{edge_idx+1}: patched — L{rl}H{rh} replaced → measure LD...")
        t1 = time.time()

        reset_all(attns)
        attns[rl].mode         = "inject"
        attns[rl].inject_head  = rh
        attns[rl].inject_value = receiver_intermediate[(sl, sh, rl, rh)]

        patched_logits = model(clean_ids)
        mx.eval(patched_logits)
        patched_diffs = logit_diffs(patched_logits, clean_last, io_tids, s_tids)

        # Score = (clean_LD - patched_LD) / (clean_LD - corrupted_LD)
        scores = []
        for i in range(len(examples)):
            denom = clean_diffs[i] - corrupt_diffs[i]
            if abs(denom) < 1e-6:
                scores.append(0.0)
            else:
                scores.append((clean_diffs[i] - patched_diffs[i]) / denom)

        mean_score      = sum(scores)                     / len(scores)
        mean_patched_ld = sum(patched_diffs)              / len(patched_diffs)
        abs_change      = mean_patched_ld - mean_clean

        results.append({
            "sender":   f"L{sl}H{sh}",
            "receiver": f"L{rl}H{rh}",
            "sender_layer":   sl,
            "sender_head":    sh,
            "receiver_layer": rl,
            "receiver_head":  rh,
            "normalized_patching_score": mean_score,
            "mean_patched_logit_diff":   mean_patched_ld,
            "absolute_change":           abs_change,
            "n_examples": len(examples),
        })
        print(f"  {time.time()-t1:.1f}s  score={mean_score:.4f}  patched_LD={mean_patched_ld:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "meta": {
            "model": MODEL_ID,
            "n_layers": n_layers,
            "n_heads":  n_heads,
            "n_examples": len(examples),
            "metric": "normalized_path_patching_score",
            "description": (
                "Edge-level path patching. Protocol: 4-pass per edge "
                "(clean baseline; corrupted + capture sender; intermediate "
                "clean with sender patched + capture receiver; patched clean "
                "with receiver replaced). Score = (clean_LD - patched_LD) / "
                "(clean_LD - corrupted_LD). Implements PathPatching.swift from "
                "SwiftSci/Interp, translated to Python/MLX."
            ),
            "heads_tested": [f"L{l}H{h}" for l, h in TOP5_HEADS],
            "edges_tested": len(EDGES),
            "mean_clean_logit_diff":   mean_clean,
            "mean_corrupt_logit_diff": mean_corrupt,
            "synthetic": False,
        },
        "edges": results,
    }

    out_path = DATA_DIR / "path-patching-llama3b-real.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    sorted_results = sorted(results, key=lambda r: r["normalized_patching_score"], reverse=True)
    print("\nEdge path patching scores (sorted):")
    print(f"  {'Edge':<20}  {'Score':>8}  {'|ΔLD|':>8}")
    for r in sorted_results:
        edge = f"{r['sender']} → {r['receiver']}"
        print(f"  {edge:<20}  {r['normalized_patching_score']:>8.4f}  {abs(r['absolute_change']):>8.4f}")


if __name__ == "__main__":
    main()
