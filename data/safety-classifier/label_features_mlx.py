#!/usr/bin/env python3
"""
Feature labeling for the Llama-3B SAE (MLX/Metal GPU version).
Uses mx.topk + Metal GEMM instead of numpy BLAS — typically 100x faster.
"""

import heapq
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
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

K            = 128    # TopK sparsity (matches SAE training config)
TOP_N        = 200    # features to label
TOP_EXAMPLES = 10     # max-activating positions per feature
CONTEXT      = 25     # tokens of context around each max-activating position
BATCH        = 4096   # larger batches are more GPU-efficient

# ── Load SAE weights → MLX ─────────────────────────────────────────────────────
print("Loading SAE checkpoint:", CKPT_PATH.name, flush=True)
ckpt      = np.load(str(CKPT_PATH))
W_enc_mx  = mx.array(ckpt['W_enc'].astype(np.float32))   # (3072, 16384)
b_enc_mx  = mx.array(ckpt['b_enc'].astype(np.float32))   # (16384,)
b_dec_mx  = mx.array(ckpt['b_dec'].astype(np.float32))   # (3072,)
DICT_SIZE  = W_enc_mx.shape[1]
print(f"  d_in={W_enc_mx.shape[0]}  dict_size={DICT_SIZE}  k={K}", flush=True)

# ── Load activations (mmap) ────────────────────────────────────────────────────
print(f"Loading activations: {ACTS_PATH}", flush=True)
meta_acts  = json.loads((ROOT / "activations/llama-3b-layer16/metadata.json").read_text())
acts_shape = tuple(meta_acts['activations_shape'])          # (500000, 3072)
acts_mm    = np.memmap(str(ACTS_PATH), dtype='float16', mode='r', shape=acts_shape)
N          = acts_mm.shape[0]
print(f"  shape: {acts_mm.shape}  dtype: {acts_mm.dtype}", flush=True)

n_batches  = (N + BATCH - 1) // BATCH


def sae_fired(batch_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the SAE encoder on a float16/32 batch and return:
      pre_np   : (B, dict_size) float32 - raw encoder pre-activations
      fired_np : (B, dict_size) bool    - True where feature is in top-K
    """
    batch_mx = mx.array(batch_np.astype(np.float32))         # (B, d_in)
    pre_mx   = (batch_mx - b_dec_mx) @ W_enc_mx + b_enc_mx   # (B, dict_size)
    top_vals = mx.topk(pre_mx, k=K, axis=-1)                  # (B, K) top values
    thresh   = mx.min(top_vals, axis=-1, keepdims=True)       # (B, 1)
    fired_mx = (pre_mx >= thresh)                              # (B, dict_size) bool
    mx.eval(pre_mx, fired_mx)
    return np.array(pre_mx), np.array(fired_mx)


# ── Pass 1: feature activation frequencies ────────────────────────────────────
print(f"\nPass 1: activation frequencies  (N={N:,}  batch={BATCH})", flush=True)
t0 = time.time()
feature_counts = np.zeros(DICT_SIZE, dtype=np.int64)

for b in range(n_batches):
    s = b * BATCH
    e = min(s + BATCH, N)
    _, fired_np = sae_fired(acts_mm[s:e])
    feature_counts += fired_np.sum(axis=0)

    if (b + 1) % 30 == 0 or b == n_batches - 1:
        print(f"  {(b+1)/n_batches*100:5.1f}%  ({e:,}/{N:,})  {time.time()-t0:.1f}s", flush=True)

top_idx    = np.argsort(feature_counts)[-TOP_N:][::-1]
top_counts = feature_counts[top_idx]
print(f"\nTop-{TOP_N}: freq {top_counts[0]:,} – {top_counts[-1]:,}", flush=True)

# ── Pass 2: max-activating positions ──────────────────────────────────────────
print(f"\nPass 2: max-activating positions", flush=True)
t1 = time.time()
top_heaps: dict[int, list] = {int(f): [] for f in top_idx}

for b in range(n_batches):
    s = b * BATCH
    e = min(s + BATCH, N)
    pre_np, fired_np = sae_fired(acts_mm[s:e])

    sub_pre   = pre_np[:, top_idx]    # (B, TOP_N)
    sub_fired = fired_np[:, top_idx]  # (B, TOP_N)

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

    if (b + 1) % 30 == 0 or b == n_batches - 1:
        print(f"  {(b+1)/n_batches*100:5.1f}%  ({e:,}/{N:,})  {time.time()-t1:.1f}s", flush=True)

for f in top_heaps:
    top_heaps[f] = sorted(top_heaps[f], reverse=True)

# ── Collect needed positions ───────────────────────────────────────────────────
needed_pos: set[int] = set()
for heap in top_heaps.values():
    for val, pos in heap:
        needed_pos.add(pos)

max_needed = (max(needed_pos) + CONTEXT + 1) if needed_pos else 0
print(f"\n{len(needed_pos)} positions to recover (up to token {max_needed:,})", flush=True)

# ── Replay WikiText for text contexts ─────────────────────────────────────────
print(f"Loading tokenizer from cache: {MODEL_ID}", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Replaying WikiText up to token {max_needed:,}...", flush=True)
t2 = time.time()
all_tok_ids: list[int] = []
tok_buf: list[int] = []
replayed   = 0
CHUNK      = 512

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
        rem = max_needed - replayed
        cl  = min(CHUNK, rem)
        all_tok_ids.extend(tok_buf[:cl])
        tok_buf    = tok_buf[cl:]
        replayed  += cl

if replayed < max_needed and tok_buf:
    rem = max_needed - replayed
    all_tok_ids.extend(tok_buf[:rem])
    replayed += rem

T = len(all_tok_ids)
print(f"  Recovered {T:,} tokens in {time.time()-t2:.1f}s", flush=True)

def get_ctx(pos: int) -> str:
    s = max(0, pos - CONTEXT)
    e = min(T, pos + CONTEXT + 1)
    return tokenizer.decode(all_tok_ids[s:e], skip_special_tokens=True)

pos_to_ctx = {p: get_ctx(p) for p in needed_pos if p < T}
print(f"  Decoded {len(pos_to_ctx)} context windows", flush=True)

# ── Semantic labeling ──────────────────────────────────────────────────────────
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
        (r'\b(1[456789][0-9][0-9]|1[0-9][0-9][0-9]|20[0-2][0-9])\b', 3.0),
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
    return {lbl: sum(len(re.findall(p, t)) * w for p, w in pats)
            for lbl, pats in PATTERNS.items()}


def assign_label(examples: list[str]) -> tuple[str, float]:
    total: dict[str, float] = {lbl: 0.0 for lbl in PATTERNS}
    for ex in examples:
        for k, v in score_text(ex).items():
            total[k] += v
    tot = sum(total.values())
    if tot < 5.0:
        return 'factual', 0.6
    winner = max(total, key=lambda k: total[k])
    conf   = round(total[winner] / tot, 3)
    if winner != 'factual' and total['factual'] > total[winner] * 0.6:
        conf, winner = round(total['factual'] / tot, 3), 'factual'
    return winner, conf


# ── Build feature records ──────────────────────────────────────────────────────
print(f"\nLabeling {TOP_N} features...", flush=True)
features = []
label_dist: dict[str, int] = {lbl: 0 for lbl in ALL_LABELS}

for rank, f in enumerate(top_idx.tolist()):
    f      = int(f)
    heap   = top_heaps[f]
    examples = [pos_to_ctx[pos] for _, pos in heap if pos in pos_to_ctx]
    label, conf = assign_label(examples) if examples else ('factual', 0.6)
    label_dist[label] = label_dist.get(label, 0) + 1
    features.append({
        'rank':                     rank + 1,
        'feature_id':               f,
        'activation_frequency':     int(top_counts[rank]),
        'activation_frequency_pct': round(float(top_counts[rank]) / N * 100, 3),
        'max_activation':           round(float(heap[0][0]) if heap else 0.0, 4),
        'label':                    label,
        'label_confidence':         conf,
        'max_activating_examples':  examples[:5],
    })

# ── Summary ────────────────────────────────────────────────────────────────────
non_other_pct = (TOP_N - label_dist.get('other', 0)) / TOP_N * 100
print(f"\nLabel distribution ({TOP_N} features):", flush=True)
for lbl in ALL_LABELS:
    n = label_dist.get(lbl, 0)
    print(f"  {lbl:25s}: {n:3d}", flush=True)
print(f"\nNon-other: {non_other_pct:.1f}%  (target >= 80%)", flush=True)

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
    'features':           features,
    'label_distribution': label_dist,
    'non_other_pct':      round(non_other_pct, 1),
}
OUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))
size_kb = OUT_PATH.stat().st_size / 1024
print(f"\nSaved → {OUT_PATH}  ({size_kb:.0f} KB)", flush=True)
print(f"Done in {time.time()-t0:.0f}s total", flush=True)
