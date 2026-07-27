"""
IOI baseline: Llama-3.2-3B
Runs a forward pass on all 100 IOI examples and records:
  - logit_io: logit for the IO name token at final position
  - logit_s:  logit for the subject name token at final position
  - logit_diff: logit_io - logit_s (positive = model prefers correct answer)
"""
import json, statistics, time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

DATA_DIR = Path(__file__).parent
MODEL_ID = "mlx-community/Llama-3.2-3B-bf16"

print(f"Loading {MODEL_ID}...")
t0 = time.time()
model, tokenizer = load(MODEL_ID)
print(f"Loaded in {time.time()-t0:.1f}s")

dataset = json.loads((DATA_DIR / "dataset.json").read_text())
examples = dataset["examples"]
names = dataset["meta"]["names"]

# Pre-compute token IDs for each name (space-prefixed, as they'd appear mid-sentence)
def get_name_token_id(name: str) -> int:
    """Return token ID for ' Name' (leading space). If multi-token, return first."""
    ids = tokenizer.encode(" " + name, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Name '{name}' encodes to empty sequence")
    if len(ids) > 1:
        print(f"  WARNING: ' {name}' -> {len(ids)} tokens {ids}, using first")
    return ids[0]

print("Resolving name token IDs...")
name_to_tid = {n: get_name_token_id(n) for n in names}
print({n: (tokenizer.decode([tid]), tid) for n, tid in name_to_tid.items()})

results = []
logit_diffs = []

print(f"\nRunning {len(examples)} forward passes...")
for i, ex in enumerate(examples):
    prompt = ex["prompt"]
    io_name = ex["io_name"]
    s_name  = ex["subject_name"]

    # Tokenize
    enc = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = mx.array([enc])           # [1, seq_len]

    # Forward pass — mlx-lm models return logits of shape [batch, seq_len, vocab]
    logits = model(input_ids)             # [1, seq_len, vocab_size]
    mx.eval(logits)

    last_logits = logits[0, -1, :]        # [vocab_size]

    tid_io = name_to_tid[io_name]
    tid_s  = name_to_tid[s_name]

    l_io   = float(last_logits[tid_io].item())
    l_s    = float(last_logits[tid_s].item())
    l_diff = l_io - l_s

    results.append({
        "id":         ex["id"],
        "prompt":     prompt,
        "io_name":    io_name,
        "s_name":     s_name,
        "token_id_io": tid_io,
        "token_id_s":  tid_s,
        "logit_io":   round(l_io,   4),
        "logit_s":    round(l_s,    4),
        "logit_diff": round(l_diff, 4),
    })
    logit_diffs.append(l_diff)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(examples)}  running mean diff: {statistics.mean(logit_diffs):.3f}")

mean_diff = statistics.mean(logit_diffs)
std_diff  = statistics.stdev(logit_diffs)

summary = {
    "model": MODEL_ID,
    "n_examples": len(results),
    "logit_diff_mean": round(mean_diff, 4),
    "logit_diff_std":  round(std_diff,  4),
    "positive_mean": mean_diff > 0,
}

output = {
    "meta": summary,
    "examples": results,
}

out_path = DATA_DIR / "baseline-llama3b.json"
out_path.write_text(json.dumps(output, indent=2))

print(f"\n=== IOI Baseline: {MODEL_ID} ===")
print(f"  mean logit diff (IO - S): {mean_diff:.4f} ± {std_diff:.4f}")
print(f"  positive mean: {mean_diff > 0}")
print(f"  saved to {out_path}")
