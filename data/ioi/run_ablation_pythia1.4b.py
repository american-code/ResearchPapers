"""
IOI Mean Ablation: Pythia-1.4B (EleutherAI/pythia-1.4b)
For every (layer, head), replace that head's SDPA output with the per-head mean
computed over the 100 IOI clean examples. Measure the drop in logit difference.

Metric
------
  drop = clean_mean_logit_diff - ablated_mean_logit_diff
  Positive drop → head contributes positively to IO prediction.
"""
import json, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_DIR = Path(__file__).parent
MODEL_ID = "EleutherAI/pythia-1.4b"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Hook infrastructure ───────────────────────────────────────────────────────

class AblationCache:
    def __init__(self, n_layers: int, n_heads: int) -> None:
        self.n_heads      = n_heads
        self.mean_sdpa:   list[torch.Tensor | None] = [None] * n_layers
        self.raw_sdpa:    list[torch.Tensor | None] = [None] * n_layers  # for mean computation
        self.mode         = "normal"   # normal | cache_mean | ablate
        self.ablate_layer: int | None = None
        self.ablate_head:  int | None = None


def make_ablation_hook(layer_idx: int, cache: AblationCache, head_dim: int):
    def hook(module, args):
        x = args[0]    # [B, S, n_heads * head_dim]
        B, S, D = x.shape
        n_heads = cache.n_heads

        per_head = x.reshape(B, S, n_heads, head_dim).permute(0, 2, 1, 3)  # [B, n_heads, S, head_dim]

        mode = cache.mode
        if mode == "cache_mean":
            cache.raw_sdpa[layer_idx] = per_head.detach().clone()

        elif mode == "ablate" and layer_idx == cache.ablate_layer and cache.ablate_head is not None:
            h      = cache.ablate_head
            mean_h = cache.mean_sdpa[layer_idx][:, h : h + 1, :, :]   # [1, 1, S, head_dim]
            # Replace head h with its mean (broadcast over batch)
            ablated = torch.cat([
                per_head[:, :h],
                mean_h.expand(B, 1, S, head_dim),
                per_head[:, h + 1:],
            ], dim=1)
            x_new = ablated.permute(0, 2, 1, 3).reshape(B, S, D)
            return (x_new,)  # pre-hook returns modified args tuple

        return None

    return hook


# ── Helpers ───────────────────────────────────────────────────────────────────

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

    cache = AblationCache(n_layers, n_heads)
    hooks = []
    for layer_idx, layer in enumerate(model.gpt_neox.layers):
        h = layer.attention.dense.register_forward_pre_hook(
            make_ablation_hook(layer_idx, cache, head_dim)
        )
        hooks.append(h)
    print(f"Registered {len(hooks)} hooks")

    dataset  = json.loads((DATA_DIR / "dataset.json").read_text())
    examples = dataset["examples"]
    print(f"Dataset: {len(examples)} examples")

    pad_id = tokenizer.eos_token_id or 0
    io_tids, s_tids, clean_encs = [], [], []
    for ex in examples:
        clean_encs.append(tokenizer.encode(ex["prompt"], add_special_tokens=True))
        io_tids.append(name_token_id(tokenizer, ex["io_name"]))
        s_tids.append(name_token_id(tokenizer, ex["subject_name"]))

    clean_lens = [len(e) for e in clean_encs]
    clean_ids  = pad_batch(clean_encs, pad_id, device)
    last_pos   = [l - 1 for l in clean_lens]

    # ── Clean pass → compute per-head means ───────────────────────────────────
    print("\nClean forward pass (batch=100) → computing per-head means...")
    t1 = time.time()
    cache.mode = "cache_mean"
    with torch.no_grad():
        clean_logits = model(clean_ids).logits
    clean_diffs = logit_diffs(clean_logits, last_pos, io_tids, s_tids)
    mean_clean  = sum(clean_diffs) / len(clean_diffs)
    print(f"  {time.time()-t1:.1f}s  mean logit diff = {mean_clean:.4f}")

    # Average over batch
    for l_idx in range(n_layers):
        raw = cache.raw_sdpa[l_idx]   # [B, n_heads, S, head_dim]
        cache.mean_sdpa[l_idx] = raw.mean(dim=0, keepdim=True)  # [1, n_heads, S, head_dim]
        cache.raw_sdpa[l_idx]  = None  # free memory
    print("  per-head means computed")

    # ── Ablation sweep ────────────────────────────────────────────────────────
    total_pairs = n_layers * n_heads
    print(f"\nAblation sweep: {n_layers} layers × {n_heads} heads = {total_pairs} passes")

    ablation_diffs: list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]
    drop_scores:    list[list[float]] = [[0.0] * n_heads for _ in range(n_layers)]
    done    = 0
    t_sweep = time.time()

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            cache.mode         = "ablate"
            cache.ablate_layer = layer_idx
            cache.ablate_head  = head_idx

            with torch.no_grad():
                abl_logits = model(clean_ids).logits
            abl_diffs = logit_diffs(abl_logits, last_pos, io_tids, s_tids)
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
            "metric": "logit_diff_drop",
            "description": (
                "Per-head mean ablation. For each (layer, head), the head's output "
                "is replaced with the per-head mean over 100 IOI clean examples. "
                "drop = clean_mean_logit_diff - ablated_mean_logit_diff. "
                "Positive drop means the head contributes positively to IO prediction."
            ),
            "clean_logit_diff_mean": round(mean_clean, 6),
            "synthetic": False,
        },
        "ablation_diffs": ablation_diffs,
        "drop_scores":    drop_scores,
    }

    out_path = DATA_DIR / "ablation-pythia1b.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")

    all_vals = [(drop_scores[l][h], l, h) for l in range(n_layers) for h in range(n_heads)]
    top10    = sorted(all_vals, reverse=True)[:10]
    print(f"\nTop-10 heads by ablation drop:")
    for drop, l, h in top10:
        print(f"  L{l:02d}·H{h:02d}: drop={drop:.4f}  (ablated={ablation_diffs[l][h]:.4f})")


if __name__ == "__main__":
    main()
