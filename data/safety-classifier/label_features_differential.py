#!/usr/bin/env python3
"""
label_features_differential.py  (v2 — 2026-08-02)

Re-labels Llama-3B SAE features using differential activation frequency:
  harm corpus  = PKU-SafeRLHF unsafe {prompt + response} pairs
                 + LibrAI/do-not-answer questions
                 + lmsys/toxic-chat toxic user_input
  neutral corpus = pre-computed WikiText-103 activations (500k tokens)

v1 bug: harm corpus contained prompts only; eval dataset uses prompt+response.
  Features selected on prompt text had negative eval discriminability.
v1 bug: only searched top-200 features by WikiText frequency; truly
  harm-discriminating features are RARE in WikiText (low wiki_rate) and
  won't appear in the top-200.

Algorithm (v2):
  1. Pass W  — recompute wiki_counts for ALL 16384 features from activation file.
  2. Pass H  — run harm texts (prompt+response) through Llama-3B SAE.
               Accumulate harm_counts for ALL 16384 features.
               Track max-activating positions for the TOP-200 by wiki freq
               (for context examples / interpretability output).
  3. Compute lift[f] = harm_rate[f] / (wiki_rate[f] + 1e-8) for all features.
  4. Candidate selection for 'potentially-harmful' label:
       wiki_rate[f] < WIKI_CAP        (rare on neutral text → low baseline)
       harm_rate[f] >= MIN_HARM_RATE  (fires meaningfully on harm text)
       lift[f]      >= LIFT_THRESH    (fires more on harm than wiki)
     Take the top MAX_HARMFUL by score = lift × harm_rate among candidates.
  5. Top-200 by wiki_rate remains the "interpretability feature set" in
     the output JSON, but selected harmful features are ADDED to it
     (or relabeled if they overlap).

Usage (lab-02): python3 label_features_differential.py
"""

import heapq
import json
import random
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import load_dataset
from mlx_lm import load
from transformers import AutoTokenizer

# ── Paths (lab-02 layout) ──────────────────────────────────────────────────────
ROOT      = Path.home() / "ResearchPapers" / "data"
CKPT_PATH = ROOT / "sae-runs/llama-3b-layer16/checkpoint_final.npz"
ACTS_PATH = ROOT / "activations/llama-3b-layer16/activations.npy"
META_PATH = ROOT / "activations/llama-3b-layer16/metadata.json"
OUT_PATH  = ROOT / "safety-classifier/llama3b-feature-labels.json"

MODEL_ID   = "mlx-community/Llama-3.2-3B-bf16"
LAYER_IDX  = 14      # target residual layer
K          = 128     # TopK sparsity matches SAE training
TOP_N      = 200     # features for interpretability output (top by wiki freq)
TOP_EX     = 10      # max-activating positions to track
CONTEXT    = 25      # context tokens around each position
BATCH_W    = 4096    # batch size for Pass W

# Feature selection thresholds for 'potentially-harmful' label
WIKI_CAP         = 0.05    # must fire on < 5% of wiki tokens (rare on neutral text)
MIN_HARM_RATE    = 0.005   # must fire on >= 0.5% of harm tokens
LIFT_THRESH      = 3.0     # must fire >= 3x more on harm than wiki
MAX_HARMFUL      = 40      # max features labeled 'potentially-harmful'

MAX_HARM_ITEMS   = 2000    # cap on harm texts run through model
MAX_TOK          = 256     # truncate each text (prompt+response) for speed

print("=" * 70, flush=True)
print("label_features_differential.py  v2", flush=True)
print("=" * 70, flush=True)

# ── Load SAE weights ──────────────────────────────────────────────────────────
print(f"\nLoading SAE checkpoint: {CKPT_PATH.name}", flush=True)
ckpt          = np.load(str(CKPT_PATH))
W_enc         = ckpt['W_enc'].astype(np.float32)   # (d_in, dict_size)
b_enc         = ckpt['b_enc'].astype(np.float32)
b_dec         = ckpt['b_dec'].astype(np.float32)
D_IN, DICT_SIZE = W_enc.shape
print(f"  d_in={D_IN}  dict_size={DICT_SIZE}  k={K}", flush=True)

W_enc_mx = mx.array(W_enc)
b_enc_mx = mx.array(b_enc)
b_dec_mx = mx.array(b_dec)


def sae_encode(acts_np: np.ndarray):
    """TopK SAE encoder. Returns (pre_np, fired_np), both (B, dict_size)."""
    batch_mx  = mx.array(acts_np.astype(np.float32))
    pre_mx    = (batch_mx - b_dec_mx) @ W_enc_mx + b_enc_mx
    top_vals  = mx.topk(pre_mx, k=K, axis=-1)
    thresh_mx = mx.min(top_vals, axis=-1, keepdims=True)
    fired_mx  = pre_mx >= thresh_mx
    mx.eval(pre_mx, fired_mx)
    return np.array(pre_mx), np.array(fired_mx)


# ── Pass W: feature frequencies from pre-computed WikiText activations ─────────
print(f"\nPass W: WikiText feature frequencies (all {DICT_SIZE} features)", flush=True)
meta       = json.loads(META_PATH.read_text())
acts_shape = tuple(meta['activations_shape'])
acts_mm    = np.memmap(str(ACTS_PATH), dtype='float16', mode='r', shape=acts_shape)
N_WIKI     = acts_mm.shape[0]
print(f"  activation file: {acts_mm.shape}  dtype={acts_mm.dtype}", flush=True)

t0 = time.time()
wiki_counts = np.zeros(DICT_SIZE, dtype=np.int64)
n_batches   = (N_WIKI + BATCH_W - 1) // BATCH_W

for b in range(n_batches):
    s, e = b * BATCH_W, min((b + 1) * BATCH_W, N_WIKI)
    _, fired = sae_encode(acts_mm[s:e])
    wiki_counts += fired.sum(axis=0).astype(np.int64)
    if (b + 1) % 30 == 0 or b == n_batches - 1:
        print(f"  {(b+1)/n_batches*100:5.1f}%  ({e:,}/{N_WIKI:,})  {time.time()-t0:.0f}s",
              flush=True)

wiki_rate = wiki_counts / N_WIKI

# Top-200 by wiki frequency (for interpretability output)
top_idx         = np.argsort(wiki_counts)[-TOP_N:][::-1]
top_wiki_counts = wiki_counts[top_idx]
top_set         = set(top_idx.tolist())
print(f"\nTop-{TOP_N} by wiki freq: {top_wiki_counts[0]:,} – {top_wiki_counts[-1]:,}",
      flush=True)


# ── Load Llama-3.2-3B ─────────────────────────────────────────────────────────
print(f"\nLoading model: {MODEL_ID}", flush=True)
t_load = time.time()
model, tokenizer = load(MODEL_ID)
n_layers = len(model.model.layers)
print(f"  {n_layers} transformer layers  ({time.time()-t_load:.1f}s)", flush=True)


def get_layer_acts(text: str):
    """
    Forward pass to LAYER_IDX. Returns (acts_np, token_ids).
    acts_np is (T, D) float32; token_ids is list[int].
    Returns (None, []) on empty input.
    """
    ids = tokenizer.encode(text)
    if not isinstance(ids, list):
        ids = list(ids)
    ids = ids[:MAX_TOK]
    if not ids:
        return None, []
    ids_mx = mx.array([ids], dtype=mx.int32)
    h      = model.model.embed_tokens(ids_mx)
    mask   = nn.MultiHeadAttention.create_additive_causal_mask(
                 h.shape[1]).astype(h.dtype)
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask=mask, cache=None)
        if i == LAYER_IDX:
            break
    mx.eval(h)
    return np.array(h[0].astype(mx.float32)), ids


# ── Build harm corpus (prompt + UNSAFE RESPONSE for format match) ──────────────
print("\nLoading harm corpus (prompt + unsafe response)...", flush=True)
harm_texts: list[str] = []

# PKU-SafeRLHF: use prompt + unsafe response_0 (response where is_response_0_safe=False)
try:
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train",
                      trust_remote_code=False)
    pairs = []
    for r in ds:
        if r.get("is_response_0_safe") is False:
            pairs.append(r["prompt"] + "\n" + r["response_0"])
        elif r.get("is_response_1_safe") is False:
            pairs.append(r["prompt"] + "\n" + r["response_1"])
    random.seed(42)
    random.shuffle(pairs)
    sample = pairs[:1500]
    harm_texts.extend(sample)
    print(f"  PKU-SafeRLHF (prompt+unsafe_resp): {len(pairs)} pairs → using {len(sample)}",
          flush=True)
except Exception as e:
    print(f"  PKU-SafeRLHF failed: {e}", flush=True)

# do-not-answer: questions only (no model response available in public version)
try:
    ds = load_dataset("LibrAI/do-not-answer", split="train", trust_remote_code=False)
    qs = [str(r["question"]) for r in ds]
    harm_texts.extend(qs)
    print(f"  do-not-answer: {len(qs)}", flush=True)
except Exception as e:
    print(f"  do-not-answer failed: {e}", flush=True)

# lmsys/toxic-chat: user_input only (toxic=1)
try:
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train",
                      trust_remote_code=False)
    toxic = [r["user_input"] for r in ds if r.get("toxicity", 0) == 1]
    harm_texts.extend(toxic)
    print(f"  toxic-chat (toxic=1): {len(toxic)}", flush=True)
except Exception as e:
    print(f"  toxic-chat failed: {e}", flush=True)

if not harm_texts:
    print("ERROR: no harm corpus loaded", file=sys.stderr)
    sys.exit(1)

random.seed(0)
random.shuffle(harm_texts)
harm_texts = harm_texts[:MAX_HARM_ITEMS]
print(f"Total harm texts for inference: {len(harm_texts)}", flush=True)


# ── Pass H: harm corpus → SAE feature frequencies (ALL features) ──────────────
print(f"\nPass H: harm corpus inference ({len(harm_texts)} texts)", flush=True)
t1 = time.time()

harm_counts      = np.zeros(DICT_SIZE, dtype=np.int64)
N_HARM           = 0
harm_token_stream: list[int] = []
# Track max-activating positions only for TOP-200 (for context examples)
harm_heaps: dict[int, list] = {int(f): [] for f in top_idx}

for i, text in enumerate(harm_texts):
    acts_np, ids = get_layer_acts(text)
    if acts_np is None:
        continue

    offset = len(harm_token_stream)
    harm_token_stream.extend(ids)

    pre_np, fired_np = sae_encode(acts_np)
    harm_counts += fired_np.sum(axis=0).astype(np.int64)
    T = acts_np.shape[0]
    N_HARM += T

    # Update heaps only for top-200-by-wiki features
    sub_pre   = pre_np[:, top_idx]
    sub_fired = fired_np[:, top_idx]
    for j, f in enumerate(top_idx.tolist()):
        col = sub_fired[:, j]
        if not col.any():
            continue
        rows = np.where(col)[0]
        vals = sub_pre[rows, j]
        heap = harm_heaps[f]
        for val, tok_pos in zip(vals.tolist(), (rows + offset).tolist()):
            if len(heap) < TOP_EX:
                heapq.heappush(heap, (val, tok_pos))
            elif val > heap[0][0]:
                heapq.heapreplace(heap, (val, tok_pos))

    if (i + 1) % 200 == 0 or i == len(harm_texts) - 1:
        print(f"  [{i+1:4d}/{len(harm_texts)}]  tokens={N_HARM:,}  "
              f"{time.time()-t1:.0f}s", flush=True)

for f in harm_heaps:
    harm_heaps[f] = sorted(harm_heaps[f], reverse=True)

print(f"Harm corpus: {N_HARM:,} tokens from {len(harm_texts)} texts", flush=True)


# ── Compute differential lift for ALL 16384 features ─────────────────────────
harm_rate = harm_counts / max(N_HARM, 1)
lift      = harm_rate / (wiki_rate + 1e-8)

# Score for selecting potentially-harmful features:
# reward high lift and high harm rate; penalize high wiki rate
harm_score_metric = lift * harm_rate * (1.0 / (wiki_rate + 0.01))

# Candidates: rare on wiki, meaningful on harm, high lift
candidates_mask = (
    (wiki_rate < WIKI_CAP)   &
    (harm_rate >= MIN_HARM_RATE) &
    (lift >= LIFT_THRESH)
)
n_candidates = int(candidates_mask.sum())
print(f"\nCandidate features (wiki<{WIKI_CAP}, harm>={MIN_HARM_RATE},"
      f" lift>={LIFT_THRESH}): {n_candidates}", flush=True)

# Sort candidates by harm_score_metric, take top MAX_HARMFUL
candidate_ids = np.where(candidates_mask)[0]
if len(candidate_ids) == 0:
    print("WARNING: no candidates — relaxing constraints", flush=True)
    # Fall back: just require lift >= 2.0 and wiki_rate < 0.10
    candidates_mask = (wiki_rate < 0.10) & (harm_rate >= MIN_HARM_RATE) & (lift >= 2.0)
    candidate_ids   = np.where(candidates_mask)[0]
    print(f"  Relaxed candidates: {len(candidate_ids)}", flush=True)

cand_scores   = harm_score_metric[candidate_ids]
cand_sorted   = candidate_ids[np.argsort(cand_scores)[::-1]]
selected_harmful = cand_sorted[:MAX_HARMFUL].tolist()

print(f"\nSelected 'potentially-harmful' features ({len(selected_harmful)}):", flush=True)
for rk, fid in enumerate(selected_harmful[:20]):
    print(f"  rank={rk+1:2d}  feat={fid:5d}"
          f"  lift={lift[fid]:.3f}"
          f"  harm={harm_rate[fid]:.4f}"
          f"  wiki={wiki_rate[fid]:.4f}"
          f"  in_top200={'yes' if fid in top_set else 'no'}",
          flush=True)
if len(selected_harmful) > 20:
    print(f"  ... ({len(selected_harmful)-20} more)", flush=True)

# Show distribution of lift for top-200-by-wiki features (informational)
top_lift = lift[top_idx]
top_harm = harm_rate[top_idx]
print(f"\nTop-10 by lift among the top-200-by-wiki features:", flush=True)
sorted_by_lift = np.argsort(top_lift)[::-1][:10]
for rk, j in enumerate(sorted_by_lift):
    f = int(top_idx[j])
    print(f"  rank={rk+1:2d}  feat={f:5d}"
          f"  lift={top_lift[j]:.3f}"
          f"  harm={top_harm[j]:.4f}"
          f"  wiki={wiki_rate[f]:.4f}"
          f"  selected={'yes' if f in selected_harmful else 'no'}", flush=True)


# ── Context decoding ──────────────────────────────────────────────────────────
T_harm = len(harm_token_stream)

def get_harm_ctx(pos: int) -> str:
    s = max(0, pos - CONTEXT)
    e = min(T_harm, pos + CONTEXT + 1)
    return tokenizer.decode(harm_token_stream[s:e], skip_special_tokens=True)


# ── Text-context labeling patterns (for non-harmful features) ─────────────────
PATTERNS: dict[str, list[tuple[str, float]]] = {
    'code-related': [
        (r'\b(def |class |import |return |void |bool |float |int )\b', 5.0),
        (r'[{}()\[\];]', 1.5),
        (r'\b(python|javascript|java|c\+\+|algorithm|compiler|runtime'
         r'|programming|software|function|method|variable|parameter'
         r'|syntax|script)\b', 4.0),
        (r'#include|System\.out|console\.log|printf|scanf|public static', 8.0),
        (r'\b(http|https|url|api|database|sql|html|css|xml|json|git|bash'
         r'|shell|linux|unix)\b', 3.0),
    ],
    'potentially-harmful': [
        (r'\b(weapon|gun|rifle|pistol|bomb|explosive|firearm|ammunition'
         r'|grenade|artillery)\b', 6.0),
        (r'\b(cocaine|heroin|meth|fentanyl|narcotic|overdose|illicit drug'
         r'|drug trafficking)\b', 6.0),
        (r'\b(murder|assassin|massacre|genocide|terrorist|radicali|extremist'
         r'|war crime)\b', 6.0),
        (r'\b(suicide|self.harm|sex trafficking|torture|abuse|violence|kill'
         r'|threat)\b', 6.0),
        (r'\b(lethal dose|toxic substance|chemical weapon|biological weapon'
         r'|nerve agent)\b', 8.0),
        (r'\b(hack|exploit|malware|ransomware|phishing|bypass|crack|steal'
         r'|fraud|scam)\b', 5.0),
    ],
    'emotional-positive': [
        (r'\b(success|celebrat|renown|famous|beloved|admired|popular|excellent'
         r'|acclaim)\b', 3.0),
        (r'\b(award|achiev|triumph|outstanding|prais|legendary|pioneer'
         r'|influential|honored)\b', 3.0),
        (r'\b(gold|champion|won|victor|greatest|distinguished|prestigious'
         r'|flourish|thriving)\b', 3.0),
        (r'\b(innovative|groundbreaking|exemplary|inspiring|remarkable)\b', 4.0),
    ],
    'emotional-negative': [
        (r'\b(fail|criticiz|condemn|defeat|loss|decline|destroy|disast'
         r'|devastat)\b', 3.0),
        (r'\b(protest|controver|blame|disput|scandal|tragedy|tragic'
         r'|unfortunate)\b', 3.0),
        (r'\b(casualt|victim|suffer|ruin|collapse|bankrupt|poverty'
         r'|famine)\b', 3.0),
    ],
    'refusal-related': [
        (r"\bI('m| am) (sorry|afraid|unable|not able)\b", 5.0),
        (r'\bcannot\b', 4.0),
        (r'\b(refuse|inappropriate|forbidden|prohibited|not allowed'
         r'|restricted|censored)\b', 5.0),
        (r'\b(must not|should not|do not engage|dangerous to)\b', 3.0),
        (r'\b(against (my|our|the) (guidelines|policy|ethics|values|terms))\b', 6.0),
    ],
    'stylistic': [
        (r'\b(however|furthermore|moreover|therefore|thus|consequently'
         r'|nevertheless|meanwhile|although|despite)\b', 4.0),
        (r'\b(also|including|such as|according to|noted|described|known as'
         r'|referred to as)\b', 2.0),
        (r'[,;:—\(\)\[\]"\'"]', 0.5),
    ],
    'factual': [
        (r'\b(born|died|founded|established|discovered|located|composed'
         r'|written|directed|invented|designed|created)\b', 4.0),
        (r'\b(1[456789][0-9][0-9]|1[0-9][0-9][0-9]|20[0-2][0-9])\b', 3.0),
        (r'\b(university|museum|government|country|city|town|region|century'
         r'|decade|period|dynasty|empire)\b', 3.0),
        (r'\b(war|battle|treaty|election|constitution|parliament|president'
         r'|minister|emperor|king|queen|general)\b', 3.0),
        (r'\b(album|film|series|novel|published|released|premiered|broadcast'
         r'|record label)\b', 3.0),
        (r'\b(is a|was a|are a|were a|is an|was an)\b', 2.0),
        (r'\b(named after|known for|based on|derived from|native to)\b', 3.0),
    ],
}

ALL_LABELS = ['factual', 'emotional-positive', 'emotional-negative',
              'potentially-harmful', 'refusal-related', 'stylistic',
              'code-related', 'other']


def score_text(text: str) -> dict[str, float]:
    t = text.lower()
    return {lbl: sum(len(re.findall(p, t)) * w for p, w in pats)
            for lbl, pats in PATTERNS.items()}


def assign_label(examples: list[str]) -> tuple[str, float]:
    total: dict[str, float] = {lbl: 0.0 for lbl in PATTERNS}
    for ex in examples:
        for k, v in score_text(ex).items():
            total[k] += v
    tot = sum(total.values())
    if tot < 3.0:
        return 'other', 0.5
    winner = max(total, key=lambda k: total[k])
    return winner, round(total[winner] / tot, 3)


# ── Build feature records ──────────────────────────────────────────────────────
# Start with the top-200 by wiki frequency as the interpretability features.
# Then add any selected_harmful features not already in top-200.
harmful_set = set(selected_harmful)

# Gather all feature IDs to include in output: top-200 ∪ selected_harmful
all_output_ids = list(top_idx)
extra_harmful  = [f for f in selected_harmful if f not in top_set]
all_output_ids.extend(extra_harmful)

print(f"\nBuilding feature records...", flush=True)
print(f"  Top-200 by wiki freq: {TOP_N}", flush=True)
print(f"  Extra harmful features not in top-200: {len(extra_harmful)}", flush=True)
print(f"  Total output features: {len(all_output_ids)}", flush=True)

features   = []
label_dist: dict[str, int] = {lbl: 0 for lbl in ALL_LABELS}

for rank, f in enumerate(all_output_ids):
    f = int(f)

    harm_examples: list[str] = []
    if f in harm_heaps:
        harm_examples = [get_harm_ctx(pos)
                         for _, pos in harm_heaps[f]
                         if pos < T_harm]

    if f in harmful_set:
        label    = 'potentially-harmful'
        conf     = min(0.99, round(float(lift[f]) / (LIFT_THRESH * 2), 3))
        examples = harm_examples
    elif harm_examples:
        label, conf = assign_label(harm_examples)
        # Never let text-context matching promote a feature to 'potentially-harmful'
        # unless it also passes the lift criterion — avoids refusal-language false labels
        if label == 'potentially-harmful':
            label, conf = 'other', 0.5
        examples = harm_examples
    else:
        label, conf = 'other', 0.5
        examples = []

    label_dist[label] = label_dist.get(label, 0) + 1

    # wiki_count for extra harmful features (not in top-200) from wiki_counts array
    wc    = int(wiki_counts[f])
    wc_pct = round(wc / N_WIKI * 100, 3)

    features.append({
        'rank':                     rank + 1,
        'feature_id':               f,
        'activation_frequency':     wc,
        'activation_frequency_pct': wc_pct,
        'lift_score':               round(float(lift[f]), 4),
        'harm_rate':                round(float(harm_rate[f]), 6),
        'wiki_rate':                round(float(wiki_rate[f]), 6),
        'label':                    label,
        'label_confidence':         conf,
        'max_activating_examples':  examples[:5],
    })

# ── Summary ────────────────────────────────────────────────────────────────────
total_n       = len(features)
non_other_pct = (total_n - label_dist.get('other', 0)) / total_n * 100

print(f"\nLabel distribution ({total_n} features):", flush=True)
for lbl in ALL_LABELS:
    n   = label_dist.get(lbl, 0)
    bar = '█' * n
    print(f"  {lbl:25s}: {n:3d}  {bar[:60]}", flush=True)
print(f"\nNon-other    : {non_other_pct:.1f}%", flush=True)
print(f"Potentially-harmful: {label_dist.get('potentially-harmful', 0)}", flush=True)

if label_dist.get('potentially-harmful', 0) == 0:
    print("WARNING: no potentially-harmful features labeled!", flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'metadata': {
        'model':                   MODEL_ID,
        'layer':                   LAYER_IDX,
        'layer_depth_pct':         50.0,
        'checkpoint':              CKPT_PATH.name,
        'training_steps':          50000,
        'dict_size':               DICT_SIZE,
        'k':                       K,
        'n_tokens_wiki':           N_WIKI,
        'n_tokens_harm':           N_HARM,
        'n_harm_texts':            len(harm_texts),
        'corpus_wiki':             'Salesforce/wikitext wikitext-103-raw-v1 (train)',
        'corpus_harm':             ('PKU-Alignment/PKU-SafeRLHF (unsafe prompt+response) + '
                                    'LibrAI/do-not-answer + '
                                    'lmsys/toxic-chat (toxicchat0124, toxic=1)'),
        'labeling_method':         'differential_frequency_v2',
        'wiki_cap':                WIKI_CAP,
        'min_harm_rate':           MIN_HARM_RATE,
        'lift_threshold':          LIFT_THRESH,
        'max_harmful_features':    MAX_HARMFUL,
        'n_harmful_features':      label_dist.get('potentially-harmful', 0),
        'total_output_features':   total_n,
        'generated_at':            '2026-08-02',
    },
    'features':           features,
    'label_distribution': label_dist,
    'non_other_pct':      round(non_other_pct, 1),
}

OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
size_kb = OUT_PATH.stat().st_size / 1024
print(f"\nSaved → {OUT_PATH}  ({size_kb:.0f} KB)", flush=True)
print(f"Total wall time: {time.time()-t0:.0f}s", flush=True)
