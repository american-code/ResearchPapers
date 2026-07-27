"""
IOI Activation Patching: Pythia-1.4B (EleutherAI/pythia-1.4b)
Full [n_layers × n_heads] sweep measuring normalised logit-diff recovery.

Metric
------
For each (layer l, head h):
  score = mean_over_examples[
      (logit_diff_patched - logit_diff_corrupted) /
      (logit_diff_clean   - logit_diff_corrupted)
  ]
where logit_diff = logit(IO name) - logit(S name) at the last token position.
Corrupted = IO and S names swapped.

Architecture notes
------------------
Pythia-1.4B is GPT-NeoX with standard MHA (n_heads=16, no GQA).
We register forward hooks to intercept SDPA outputs per head.
"""
import json, time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_DIR  = Path(__file__).parent
MODEL_ID  = "EleutherAI/pythia-1.4b"
DEVICE    = "cpu"   # will be overridden to mps if available


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Head-level hook infrastructure ───────────────────────────────────────────

class HeadCache:
    """Holds per-head caches and the current patching state."""

    def __init__(self, n_layers: int, n_heads: int) -> None:
        self.n_heads   = n_heads
        self.clean:    list[torch.Tensor | None] = [None] * n_layers
        self.corrupt:  list[torch.Tensor | None] = [None] * n_layers
        self.mode      = "normal"       # normal | cache_clean | cache_corrupt | patch
        self.patch_layer: int | None = None
        self.patch_head:  int | None = None


def make_attn_hook(layer_idx: int, cache: HeadCache, head_dim: int):
    """
    Pre-hook on dense (o_proj). Input is [B, seq, n_heads * head_dim] —
    the concatenated per-head SDPA outputs before projection.
    """

    def hook(module, args):
        # args[0]: [B, seq, n_heads * head_dim]
        x = args[0]
        B, S, D = x.shape
        n_heads = cache.n_heads

        # Reshape to [B, S, n_heads, head_dim] then [B, n_heads, S, head_dim]
        per_head = x.reshape(B, S, n_heads, head_dim).permute(0, 2, 1, 3)

        mode = cache.mode
        if mode == "cache_clean":
            cache.clean[layer_idx] = per_head.detach().clone()
        elif mode == "cache_corrupt":
            cache.corrupt[layer_idx] = per_head.detach().clone()
        elif mode == "patch" and layer_idx == cache.patch_layer and cache.patch_head is not None:
            h = cache.patch_head
            clean_h = cache.clean[layer_idx][:, h : h + 1, :, :]  # [B,1,S,head_dim]
            per_head = torch.cat([per_head[:, :h], clean_h, per_head[:, h + 1:]], dim=1)
            x_patched = per_head.permute(0, 2, 1, 3).reshape(B, S, D)
            return (x_patched,)  # pre-hook returns modified args tuple

        return None  # no change

    return hook


# ── Dataset helpers ───────────────────────────────────────────────────────────

def name_token_id(tokenizer, name: str) -> int:
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) > 1:
        print(f"  WARNING: ' {name}' → {len(ids)} tokens; using first")
    return ids[0]


def pad_batch(seqs: list[list[int]], pad_id: int, device: str) -> torch.Tensor:
    max_len = max(len(s) for s in seqs)
    padded  = [s + [pad_id] * (max_len - len(s)) for s in seqs]
    return torch.tensor(padded, dtype=torch.long, device=device)


def logit_diffs(
    logits: torch.Tensor,
    last_positions: list[int],
    io_tids: list[int],
    s_tids:  list[int],
) -> list[float]:
    out = []
    for i in range(len(last_positions)):
        pos  = last_positions[i]
        l_io = float(logits[i, pos, io_tids[i]])
        l_s  = float(logits[i, pos, s_tids[i]])
        out.append(l_io - l_s)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = get_device()
    print(f"Device: {device}")

    print(f"Loading {MODEL_ID}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model     = model.to(device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    n_layers = model.config.num_hidden_layers
    n_heads  = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    print(f"  n_layers={n_layers}  n_heads={n_heads}  head_dim={head_dim}")

    # ── Register hooks on o_proj (dense projection after SDPA) ───────────────
    cache = HeadCache(n_layers, n_heads)
    hooks = []
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        h = layer.attention.dense.register_forward_pre_hook(
            make_attn_hook(layer_idx, cache, head_dim)
        )
        hooks.append(h)
    print(f"Registered {len(hooks)} hooks (one per layer, on o_proj pre-hook)")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset  = json.loads((DATA_DIR / "dataset.json").read_text())
    examples = dataset["examples"]
    template = dataset["meta"]["template"]
    print(f"Dataset: {len(examples)} examples")

    pad_id = tokenizer.eos_token_id or 0
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
    mismatches   = [i for i, (c, k) in enumerate(zip(clean_lens, corrupt_lens)) if c != k]
    if mismatches:
        print(f"WARNING: {len(mismatches)} examples have clean/corrupt length mismatch")
    else:
        print("  all clean/corrupt lengths match ✓")

    clean_ids   = pad_batch(clean_encs,   pad_id, device)
    corrupt_ids = pad_batch(corrupt_encs, pad_id, device)
    clean_last  = [l - 1 for l in clean_lens]
    corrupt_last = [l - 1 for l in corrupt_lens]

    # ── Clean forward pass ────────────────────────────────────────────────────
    print("\nClean forward pass (batch=100)...")
    t1 = time.time()
    cache.mode = "cache_clean"
    with torch.no_grad():
        clean_logits = model(clean_ids).logits
    clean_diffs = logit_diffs(clean_logits, clean_last, io_tids, s_tids)
    mean_clean  = sum(clean_diffs) / len(clean_diffs)
    print(f"  {time.time()-t1:.1f}s  mean logit diff = {mean_clean:.4f}")

    # ── Corrupt forward pass ──────────────────────────────────────────────────
    print("Corrupt forward pass (batch=100)...")
    t1 = time.time()
    cache.mode = "cache_corrupt"
    with torch.no_grad():
        corrupt_logits = model(corrupt_ids).logits
    corrupt_diffs = logit_diffs(corrupt_logits, corrupt_last, io_tids, s_tids)
    mean_corrupt  = sum(corrupt_diffs) / len(corrupt_diffs)
    print(f"  {time.time()-t1:.1f}s  mean logit diff = {mean_corrupt:.4f}")

    if mean_corrupt >= mean_clean:
        print("WARNING: corrupt mean >= clean mean — IOI effect may be weak")

    # ── Patching sweep ────────────────────────────────────────────────────────
    total_pairs = n_layers * n_heads
    print(f"\nPatching sweep: {n_layers} layers × {n_heads} heads = {total_pairs} passes")

    patching_scores: list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]
    done    = 0
    t_sweep = time.time()

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            cache.mode        = "patch"
            cache.patch_layer = layer_idx
            cache.patch_head  = head_idx

            with torch.no_grad():
                patched_logits = model(corrupt_ids).logits
            patched_diffs = logit_diffs(patched_logits, corrupt_last, io_tids, s_tids)

            recoveries = []
            for i in range(len(examples)):
                denom = clean_diffs[i] - corrupt_diffs[i]
                if abs(denom) < 1e-6:
                    recoveries.append(0.0)
                else:
                    recoveries.append((patched_diffs[i] - corrupt_diffs[i]) / denom)

            patching_scores[layer_idx][head_idx] = sum(recoveries) / len(recoveries)
            done += 1

            if head_idx == n_heads - 1:
                elapsed = time.time() - t_sweep
                eta     = elapsed / done * (total_pairs - done)
                top_h   = max(range(n_heads), key=lambda h: patching_scores[layer_idx][h])
                print(
                    f"  L{layer_idx:02d} done  "
                    f"top_head=H{top_h:02d} ({patching_scores[layer_idx][top_h]:.3f})  "
                    f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
                )

    for h in hooks:
        h.remove()

    elapsed_total = time.time() - t_sweep
    print(f"\nSweep completed in {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

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

    out_path = DATA_DIR / "patching-pythia1.4b.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")

    all_vals = [(patching_scores[l][h], l, h) for l in range(n_layers) for h in range(n_heads)]
    top10    = sorted(all_vals, reverse=True)[:10]
    print("\nTop-10 heads by patching score:")
    for score, l, h in top10:
        print(f"  L{l:02d}·H{h:02d}: {score:.4f}")


if __name__ == "__main__":
    main()
