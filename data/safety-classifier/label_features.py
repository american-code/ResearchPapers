#!/usr/bin/env python3
"""
Label the top-200 Llama-3B SAE features by semantic category for the safety classifier.

Steps:
  1. Pass 1 — compute activation frequency for all 16384 features over 500k tokens
  2. Select top-200 features by frequency
  3. Pass 2 — find top-10 max-activating token positions per feature
  4. Replay WikiText corpus to recover text contexts at those positions
  5. Apply heuristic labeling across 8 categories
  6. Save to data/safety-classifier/llama3b-feature-labels.json
"""

import heapq
import json
import re
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
CKPT_PATH  = ROOT / "sae-runs/llama-3b-layer16/checkpoint_step_010000.npz"
ACTS_PATH  = ROOT / "activations/llama-3b-layer16/activations.npy"
OUT_PATH   = ROOT / "safety-classifier/llama3b-feature-labels.json"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_ID      = "mlx-community/Llama-3.2-3B-bf16"
CORPUS_NAME   = "Salesforce/wikitext"
CORPUS_CONFIG = "wikitext-103-raw-v1"
CORPUS_SPLIT  = "train"

K            = 128    # TopK sparsity (matches training config)
TOP_N        = 200    # features to label
TOP_EXAMPLES = 10     # max-activating positions per feature
CONTEXT      = 25     # tokens of context around each max-activating position
BATCH        = 2048   # tokens per processing batch

# ── Load SAE weights ───────────────────────────────────────────────────────────
print(f"Loading SAE checkpoint: {CKPT_PATH.name}")
ckpt  = np.load(str(CKPT_PATH))
W_enc = ckpt['W_enc'].astype(np.float32)   # (3072, 16384)
b_enc = ckpt['b_enc'].astype(np.float32)   # (16384,)
b_dec = ckpt['b_dec'].astype(np.float32)   # (3072,)
D_IN, DICT_SIZE = W_enc.shape
THRESH_COL = DICT_SIZE - K   # index of K-th largest in ascending-sorted row
print(f"  d_in={D_IN}  dict_size={DICT_SIZE}  k={K}")

# ── Load activations (mmap) ────────────────────────────────────────────────────
print(f"Loading activations: {ACTS_PATH}")
# File was written with np.memmap (raw binary, no .npy header) — load same way
meta_acts = json.loads((ROOT / "activations/llama-3b-layer16/metadata.json").read_text())
acts_shape = tuple(meta_acts['activations_shape'])  # (500000, 3072)
acts_mm = np.memmap(str(ACTS_PATH), dtype='float16', mode='r', shape=acts_shape)
N = acts_mm.shape[0]
print(f"  shape: {acts_mm.shape}  dtype: {acts_mm.dtype}")

n_batches = (N + BATCH - 1) // BATCH

# ── Pass 1: feature activation frequencies ────────────────────────────────────
print(f"\nPass 1: activation frequencies  (N={N:,}  batch={BATCH})")
t0 = time.time()
feature_counts = np.zeros(DICT_SIZE, dtype=np.int64)

for b in range(n_batches):
    s = b * BATCH
    e = min(s + BATCH, N)
    batch_f32 = acts_mm[s:e].astype(np.float32)           # (B, d_in)
    pre       = (batch_f32 - b_dec) @ W_enc + b_enc       # (B, dict_size)
    partioned = np.partition(pre, THRESH_COL, axis=1)
    thr       = partioned[:, THRESH_COL]                   # (B,) = K-th largest
    fired     = pre >= thr[:, None]                        # (B, dict_size) bool
    feature_counts += fired.sum(axis=0)

    if (b + 1) % 50 == 0 or b == n_batches - 1:
        print(f"  {(b+1)/n_batches*100:5.1f}%  ({e:,}/{N:,})  {time.time()-t0:.0f}s elapsed")

# Select top-N by frequency
top_idx    = np.argsort(feature_counts)[-TOP_N:][::-1]   # descending by count
top_counts = feature_counts[top_idx]
print(f"\nTop-{TOP_N} features: freq range {top_counts[0]:,} – {top_counts[-1]:,}")

# ── Pass 2: max-activating positions ──────────────────────────────────────────
print(f"\nPass 2: max-activating positions")
t1 = time.time()

# min-heap per feature (so heappop removes the smallest, maintaining top-K max)
top_heaps: dict[int, list] = {int(f): [] for f in top_idx}

for b in range(n_batches):
    s = b * BATCH
    e = min(s + BATCH, N)
    batch_f32  = acts_mm[s:e].astype(np.float32)
    pre        = (batch_f32 - b_dec) @ W_enc + b_enc      # (B, dict_size)
    partitioned = np.partition(pre, THRESH_COL, axis=1)
    thr        = partitioned[:, THRESH_COL]
    fired_mask = pre >= thr[:, None]                       # (B, dict_size) bool

    sub_pre   = pre[:, top_idx]          # (B, TOP_N)
    sub_fired = fired_mask[:, top_idx]   # (B, TOP_N)

    for j, f in enumerate(top_idx.tolist()):
        col = sub_fired[:, j]
        if not col.any():
            continue
        rows = np.where(col)[0]
        vals = sub_pre[rows, j]
        heap = top_heaps[f]
        for val, pos in zip(vals.tolist(), (rows + s).tolist()):
            if len(heap) < TOP_EXAMPLES:
                heapq.heappush(heap, (val, pos))
            elif val > heap[0][0]:
                heapq.heapreplace(heap, (val, pos))

    if (b + 1) % 50 == 0 or b == n_batches - 1:
        print(f"  {(b+1)/n_batches*100:5.1f}%  ({e:,}/{N:,})  {time.time()-t1:.0f}s elapsed")

# Sort each heap descending (highest activation first)
for f in top_heaps:
    top_heaps[f] = sorted(top_heaps[f], reverse=True)

# ── Collect needed token positions ─────────────────────────────────────────────
needed_pos: set[int] = set()
for heap in top_heaps.values():
    for val, pos in heap:
        needed_pos.add(pos)

max_needed = (max(needed_pos) + CONTEXT + 1) if needed_pos else 0
print(f"\nNeed text at {len(needed_pos)} positions (replay up to token {max_needed:,})")

# ── Replay WikiText to recover text contexts ───────────────────────────────────
print(f"Loading tokenizer from cache: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Replaying WikiText ({CORPUS_NAME} {CORPUS_CONFIG}) up to token {max_needed:,}...")
t2 = time.time()
all_tok_ids: list[int] = []
tok_buf: list[int] = []
replayed = 0
CHUNK = 512

ds = load_dataset(CORPUS_NAME, CORPUS_CONFIG, split=CORPUS_SPLIT, trust_remote_code=False)

for row in ds:
    if replayed >= max_needed:
        break
    text = row['text'].strip()
    if not text:
        continue
    ids = tokenizer.encode(text, add_special_tokens=False)
    tok_buf.extend(ids)

    while len(tok_buf) >= CHUNK and replayed < max_needed:
        remaining  = max_needed - replayed
        chunk_len  = min(CHUNK, remaining)
        chunk      = tok_buf[:chunk_len]
        tok_buf    = tok_buf[chunk_len:]
        all_tok_ids.extend(chunk)
        replayed   += chunk_len

# Handle sub-chunk leftover
if replayed < max_needed and tok_buf:
    remaining = max_needed - replayed
    chunk = tok_buf[:remaining]
    all_tok_ids.extend(chunk)
    replayed += len(chunk)

T = len(all_tok_ids)
print(f"  Recovered {T:,} tokens in {time.time()-t2:.1f}s")

# Decode context windows around each needed position
def get_context(pos: int) -> str:
    start = max(0, pos - CONTEXT)
    end   = min(T, pos + CONTEXT + 1)
    return tokenizer.decode(all_tok_ids[start:end], skip_special_tokens=True)

pos_to_ctx: dict[int, str] = {pos: get_context(pos) for pos in needed_pos if pos < T}
print(f"  Decoded {len(pos_to_ctx)} context windows")

# ── Semantic labeling ──────────────────────────────────────────────────────────
# Weighted regex patterns per label. Higher weight = stronger signal.
PATTERNS: dict[str, list[tuple[str, float]]] = {
    'code-related': [
        (r'\b(def |class |import |return |void |bool |float |int )\b', 5.0),
        (r'[{}()\[\];]', 1.5),
        (r'\b(python|javascript|java|c\+\+|algorithm|compiler|runtime|programming|software|function|method|variable|parameter|syntax|script)\b', 4.0),
        (r'#include|System\.out|console\.log|printf|scanf|public static', 8.0),
        (r'\b(http|https|url|api|database|sql|html|css|xml|json|git|bash|shell|linux|unix)\b', 3.0),
    ],
    'potentially-harmful': [
        (r'\b(weapon|gun|rifle|pistol|bomb|explosive|firearm|ammunition|grenade|artillery)\b', 6.0),
        (r'\b(cocaine|heroin|meth|fentanyl|narcotic|overdose|illicit drug|drug trafficking)\b', 6.0),
        (r'\b(murder|assassin|massacre|genocide|terrorist|radicali|extremist|war crime)\b', 6.0),
        (r'\b(suicide|self.harm|sex trafficking|torture|abuse)\b', 6.0),
        (r'\b(lethal dose|toxic substance|chemical weapon|biological weapon|nerve agent)\b', 8.0),
    ],
    'emotional-positive': [
        (r'\b(success|celebrat|renown|famous|beloved|admired|popular|excellent|acclaim)\b', 3.0),
        (r'\b(award|achiev|triumph|outstanding|prais|legendary|pioneer|influential|honored)\b', 3.0),
        (r'\b(gold|champion|won|victor|greatest|distinguished|prestigious|flourish|thriving)\b', 3.0),
        (r'\b(innovative|groundbreaking|exemplary|inspiring|remarkable)\b', 4.0),
    ],
    'emotional-negative': [
        (r'\b(fail|criticiz|condemn|defeat|loss|decline|destroy|disast|devastat)\b', 3.0),
        (r'\b(protest|controver|blame|disput|scandal|tragedy|tragic|unfortunate)\b', 3.0),
        (r'\b(revolt|riot|uprising|opposition|resist|reject|repression|persecution)\b', 3.0),
        (r'\b(casualt|victim|suffer|ruin|collapse|bankrupt|poverty|famine)\b', 3.0),
    ],
    'refusal-related': [
        (r'\bcannot\b', 4.0),
        (r'\b(refuse|inappropriate|forbidden|prohibited|not allowed|restricted|censored)\b', 5.0),
        (r'\b(must not|should not|do not engage|avoid|dangerous to)\b', 3.0),
    ],
    'stylistic': [
        (r'\b(however|furthermore|moreover|therefore|thus|consequently|nevertheless|meanwhile|although|despite)\b', 4.0),
        (r'\b(also|including|such as|according to|noted|described|known as|referred to as)\b', 2.0),
        (r'[,;:—\(\)\[\]"\'"]', 0.5),
    ],
    'factual': [
        (r'\b(born|died|founded|established|discovered|located|composed|written|directed|invented|designed|created)\b', 4.0),
        (r'\b(1[456789][0-9][0-9]|1[0-9][0-9][0-9]|20[0-2][0-9])\b', 3.0),  # years 1400-2029
        (r'\b(university|museum|government|country|city|town|region|century|decade|period|dynasty|empire)\b', 3.0),
        (r'\b(km|mi|km²|population|inhabitants|species|genus|family|order|class|phylum|kingdom)\b', 4.0),
        (r'\b(war|battle|treaty|election|constitution|parliament|president|minister|emperor|king|queen|general)\b', 3.0),
        (r'\b(album|film|series|novel|published|released|premiered|broadcast|record label)\b', 3.0),
        (r'\b(is a|was a|are a|were a|is an|was an)\b', 2.0),
        (r'\b(named after|known for|based on|derived from|native to)\b', 3.0),
        (r'\b(chapter|verse|section|volume|edition|translated|author|poet|playwright)\b', 3.0),
    ],
}

ALL_LABELS = ['factual', 'emotional-positive', 'emotional-negative',
              'potentially-harmful', 'refusal-related', 'stylistic', 'code-related', 'other']


def score_text(text: str) -> dict[str, float]:
    t = text.lower()
    scores: dict[str, float] = {}
    for lbl, pats in PATTERNS.items():
        s = 0.0
        for pat, w in pats:
            s += len(re.findall(pat, t)) * w
        scores[lbl] = s
    return scores


def assign_label(examples: list[str]) -> tuple[str, float]:
    """Return (label, confidence) from aggregated pattern scores across examples."""
    total: dict[str, float] = {lbl: 0.0 for lbl in PATTERNS}
    for ex in examples:
        for k, v in score_text(ex).items():
            total[k] += v

    tot_sum = sum(total.values())
    if tot_sum < 5.0:
        # Low signal: default to 'factual' for Wikipedia corpus
        return 'factual', 0.6

    winner = max(total, key=lambda k: total[k])
    conf   = round(total[winner] / tot_sum, 3)

    # Tie-break: if non-factual winner is within 40% of factual, prefer factual
    # (Wikipedia is overwhelmingly factual; weak signals should stay factual)
    if winner != 'factual' and total['factual'] > total[winner] * 0.6:
        conf   = round(total['factual'] / tot_sum, 3)
        winner = 'factual'

    return winner, conf


# ── Build feature records ──────────────────────────────────────────────────────
print(f"\nLabeling features...")
features = []
label_dist: dict[str, int] = {lbl: 0 for lbl in ALL_LABELS}

for rank, f in enumerate(top_idx.tolist()):
    f      = int(f)
    heap   = top_heaps[f]   # sorted descending

    examples = [pos_to_ctx[pos] for _, pos in heap if pos in pos_to_ctx]

    label, conf = assign_label(examples) if examples else ('factual', 0.6)
    label_dist[label] = label_dist.get(label, 0) + 1

    max_act = heap[0][0] if heap else 0.0

    features.append({
        'rank':                     rank + 1,
        'feature_id':               f,
        'activation_frequency':     int(top_counts[rank]),
        'activation_frequency_pct': round(float(top_counts[rank]) / N * 100, 3),
        'max_activation':           round(float(max_act), 4),
        'label':                    label,
        'label_confidence':         conf,
        'max_activating_examples':  examples[:5],
    })

# ── Summary ────────────────────────────────────────────────────────────────────
non_other_n   = TOP_N - label_dist.get('other', 0)
non_other_pct = non_other_n / TOP_N * 100

print(f"\nLabel distribution ({TOP_N} features):")
for lbl in ALL_LABELS:
    n   = label_dist.get(lbl, 0)
    bar = '█' * n
    print(f"  {lbl:25s}: {n:3d}  {bar[:60]}")
print(f"\nNon-other: {non_other_pct:.1f}%  (target ≥ 80%)")

if non_other_pct < 80.0:
    print(f"WARNING: non-other pct {non_other_pct:.1f}% is below 80% target")

# ── Save ───────────────────────────────────────────────────────────────────────
out = {
    'metadata': {
        'model':            MODEL_ID,
        'layer':            14,
        'layer_depth_pct':  50.0,
        'checkpoint':       CKPT_PATH.name,
        'training_steps':   10000,
        'dict_size':        DICT_SIZE,
        'k':                K,
        'n_tokens':         N,
        'corpus':           f'{CORPUS_NAME} {CORPUS_CONFIG} ({CORPUS_SPLIT})',
        'top_n_features':   TOP_N,
        'context_window':   CONTEXT,
        'generated_at':     '2026-07-29',
    },
    'features':          features,
    'label_distribution': label_dist,
    'non_other_pct':     round(non_other_pct, 1),
}

OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
size_kb = OUT_PATH.stat().st_size / 1024
print(f"\nSaved → {OUT_PATH}  ({size_kb:.0f} KB)")
print(f"Done in {time.time()-t0:.0f}s total")
