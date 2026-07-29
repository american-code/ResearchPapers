#!/usr/bin/env python3
"""
Cross-architecture SAE feature matching: three-way comparison.

Method: chunk-averaged activation pattern matching.
  All three models streamed wikitext-103-raw-v1 train in the same order.
  We use the first EVAL_TOKENS tokens from each model's activation file,
  split into N_CHUNKS chunks of CHUNK_TOKS tokens each.  This provides
  identical text coverage across models despite different tokenizer speeds.

  For each chunk we compute the mean SAE feature activation → [N_CHUNKS, dict_size].
  Each feature is described by an N_CHUNKS-dimensional "context fingerprint".
  Nearest-neighbour cosine similarity > SIM_THRESH is the match criterion.

  Performance: activations are processed in ENCODE_BATCH-token blocks (instead
  of per-chunk) so each step is a single large BLAS matmul.

Outputs (written to data/sae-analysis/):
  llama_qwen_matches.json    – feature pairs with cosim
  mistral_qwen_matches.json  – feature pairs
  llama_mistral_matches.json – feature pairs
  universal-features.json    – features replicated across all three models
  venn_diagram.svg            – three-circle Venn showing overlap counts
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE    = Path("/Users/melton/ResearchPapers")
SAE_DIR      = WORKSPACE / "data/sae-runs"
ACTS_DIR     = WORKSPACE / "data/activations"
OUT_DIR      = WORKSPACE / "data/sae-analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_TOKENS  = 50_000     # tokens from each model to use (Mistral only has 50k available)
N_CHUNKS     = 100        # fingerprint dimensionality
CHUNK_TOKS   = EVAL_TOKENS // N_CHUNKS   # 500 tokens per chunk
ENCODE_BATCH = 5_000      # tokens per matmul; must be multiple of CHUNK_TOKS
assert ENCODE_BATCH % CHUNK_TOKS == 0, "ENCODE_BATCH must be a multiple of CHUNK_TOKS"
SIM_THRESH   = 0.80       # nearest-neighbour cosine similarity threshold
BATCH_FEAT   = 2048       # features per matmul during matching

MODELS = {
    "llama": {
        "acts":  ACTS_DIR / "llama-3b-layer16",
        "ckpt":  SAE_DIR  / "llama-3b-layer16/checkpoint_step_010000.npz",
        "d_in":  3072,
    },
    "qwen": {
        "acts":  ACTS_DIR / "qwen-3b-layer18",
        "ckpt":  SAE_DIR  / "qwen-3b-layer18/checkpoint_final.npz",
        "d_in":  2048,
    },
    "mistral": {
        "acts":  ACTS_DIR / "mistral-7b-layer16",
        "ckpt":  SAE_DIR  / "mistral-7b-layer16/checkpoint_final.npz",
        "d_in":  4096,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_acts_slice(acts_dir: Path, d_in: int, n_tokens: int) -> np.ndarray:
    """Load only the first n_tokens activations as float32."""
    meta      = json.loads((acts_dir / "metadata.json").read_text())
    n_avail   = meta["n_tokens_written"]
    acts_file = acts_dir / meta.get("activations_file", "activations.npy")

    n_use = min(n_tokens, n_avail)
    print(f"  Loading {n_use:,} tokens from {acts_file.name} (avail={n_avail:,}, d={d_in})", flush=True)
    t0 = time.time()
    try:
        raw = np.load(str(acts_file), mmap_mode="r")[:n_use]
    except (ValueError, OSError):
        # raw binary float16
        raw = np.memmap(str(acts_file), dtype=np.float16, mode="r",
                        shape=(n_avail, d_in))[:n_use]
    arr = np.array(raw, dtype=np.float32)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    return arr


def load_sae(ckpt: Path) -> dict:
    data = np.load(str(ckpt))
    return {k: data[k].astype(np.float32) for k in ("W_enc", "b_enc", "W_dec", "b_dec")}


def encode_batch(acts: np.ndarray, sae: dict, k: int = 128) -> np.ndarray:
    """
    TopK-SAE encode a [T, d_in] chunk → [T, dict_size] sparse float32.
    Uses a single BLAS matmul for the projection.
    """
    pre  = (acts - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]  # [T, dict_size]
    kth  = np.partition(pre, -k, axis=1)[:, -k][:, None]         # [T, 1]  k-th largest
    return np.where(pre >= kth, pre, 0.0)                         # [T, dict_size]


def build_chunk_fingerprints(acts: np.ndarray, sae: dict,
                              n_chunks: int = N_CHUNKS,
                              chunk_toks: int = CHUNK_TOKS,
                              encode_batch_sz: int = ENCODE_BATCH,
                              ) -> np.ndarray:
    """
    Returns L2-normalised float32 fingerprint matrix [dict_size, n_chunks].
    Encodes in encode_batch_sz-token blocks for BLAS efficiency.
    """
    dict_size = sae["W_enc"].shape[1]
    fp = np.zeros((n_chunks, dict_size), dtype=np.float32)

    # chunk_sum[ci] accumulates SAE-acts sums for chunk ci; count tracks tokens
    chunk_counts = np.zeros(n_chunks, dtype=np.int32)

    n_total = n_chunks * chunk_toks
    print(f"  Encoding {n_total:,} tokens in blocks of {encode_batch_sz:,} …", flush=True)
    t0 = time.time()

    for block_start in range(0, n_total, encode_batch_sz):
        block_end  = min(block_start + encode_batch_sz, n_total)
        block_acts = encode_batch(acts[block_start:block_end], sae)   # [B, dict_size]

        # map each token in the block to its chunk index
        for local_off, tok_i in enumerate(range(block_start, block_end)):
            ci = tok_i // chunk_toks
            fp[ci] += block_acts[local_off]
            chunk_counts[ci] += 1

        if (block_end // encode_batch_sz) % 2 == 0 or block_end >= n_total:
            pct = block_end / n_total * 100
            print(f"    {block_end:>6,}/{n_total:,}  ({pct:.0f}%)  "
                  f"{(time.time()-t0):.0f}s", flush=True)

    # divide by token count to get means
    fp /= np.maximum(chunk_counts[:, None], 1)

    # transpose → [dict_size, n_chunks]; L2-normalise rows
    fp = fp.T.copy()
    norms = np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
    fp /= norms
    return fp


def build_chunk_fingerprints_fast(acts: np.ndarray, sae: dict,
                                   n_chunks: int = N_CHUNKS,
                                   chunk_toks: int = CHUNK_TOKS,
                                   encode_batch_sz: int = ENCODE_BATCH,
                                   ) -> np.ndarray:
    """
    Fast fingerprint build: encode in encode_batch_sz-token blocks, reshape
    [B, dict_size] → [chunks_per_block, chunk_toks, dict_size], mean over axis=1.
    Avoids slow scatter/np.add.at and uses BLAS for both the matmul and mean.
    ENCODE_BATCH must be a multiple of CHUNK_TOKS.
    """
    dict_size       = sae["W_enc"].shape[1]
    n_total         = n_chunks * chunk_toks
    chunks_per_blk  = encode_batch_sz // chunk_toks
    total_blocks    = math.ceil(n_total / encode_batch_sz)
    fp              = np.zeros((n_chunks, dict_size), dtype=np.float32)

    print(f"  Encoding {n_total:,} tokens in {total_blocks} blocks "
          f"({encode_batch_sz} tok/block, {chunks_per_blk} chunks/block) …", flush=True)
    t0 = time.time()

    chunk_offset = 0
    for block_start in range(0, n_total, encode_batch_sz):
        block_end  = min(block_start + encode_batch_sz, n_total)
        B          = block_end - block_start

        block_acts = encode_batch(acts[block_start:block_end], sae)  # [B, dict_size]

        # reshape to [n_chunks_in_block, chunk_toks, dict_size] → mean over axis=1
        n_cib = B // chunk_toks
        fp[chunk_offset : chunk_offset + n_cib] = (
            block_acts.reshape(n_cib, chunk_toks, dict_size).mean(axis=1)
        )
        chunk_offset += n_cib

        blk_i = block_start // encode_batch_sz + 1
        if blk_i % max(1, total_blocks // 5) == 0 or blk_i == total_blocks:
            elapsed = time.time() - t0
            pct = block_end / n_total * 100
            print(f"    block {blk_i}/{total_blocks}  ({pct:.0f}%)  {elapsed:.0f}s", flush=True)

    fp = fp.T.copy()   # [dict_size, n_chunks]
    norms = np.maximum(np.linalg.norm(fp, axis=1, keepdims=True), 1e-12)
    fp /= norms
    return fp


# ── pairwise matching ─────────────────────────────────────────────────────────

def nearest_neighbor_match(fp_a: np.ndarray, fp_b: np.ndarray,
                            thresh: float = SIM_THRESH,
                            batch: int = BATCH_FEAT,
                            label_a: str = "A", label_b: str = "B",
                            ) -> list[dict]:
    """
    For every feature in A find its nearest neighbour in B.
    Both directions searched (A→B and B→A) to catch all matches.
    Returns unique pairs with cosim > thresh.
    """
    dict_size = fp_a.shape[0]
    print(f"  {label_a}→{label_b}:", flush=True)

    # A → B
    best_b = np.full(dict_size, -1, dtype=np.int32)
    best_s = np.full(dict_size, -1.0, dtype=np.float32)
    for i in range(0, dict_size, batch):
        sims = fp_a[i:i+batch] @ fp_b.T       # [batch, dict_size]
        idx  = np.argmax(sims, axis=1)
        vals = sims[np.arange(len(idx)), idx]
        best_b[i:i+batch] = idx
        best_s[i:i+batch] = vals

    a_to_b_set   = set()
    matches_a2b  = []
    for ai, (bi, s) in enumerate(zip(best_b, best_s)):
        if s >= thresh:
            a_to_b_set.add((ai, int(bi)))
            matches_a2b.append({"feat_a": ai, "feat_b": int(bi), "cosim": round(float(s), 4)})

    # B → A (catch remaining)
    best_a = np.full(dict_size, -1, dtype=np.int32)
    best_s2 = np.full(dict_size, -1.0, dtype=np.float32)
    for i in range(0, dict_size, batch):
        sims = fp_b[i:i+batch] @ fp_a.T
        idx  = np.argmax(sims, axis=1)
        vals = sims[np.arange(len(idx)), idx]
        best_a[i:i+batch] = idx
        best_s2[i:i+batch] = vals

    matches_b2a = []
    for bi, (ai, s) in enumerate(zip(best_a, best_s2)):
        if s >= thresh and (int(ai), bi) not in a_to_b_set:
            matches_b2a.append({"feat_a": int(ai), "feat_b": bi, "cosim": round(float(s), 4)})

    all_matches = matches_a2b + matches_b2a
    print(f"    {label_a}→{label_b}: {len(matches_a2b)}  "
          f"{label_b}→{label_a} (new): {len(matches_b2a)}  "
          f"total: {len(all_matches)}", flush=True)
    return all_matches


# ── universal feature identification ─────────────────────────────────────────

def find_universal_features(lq_matches: list[dict], mq_matches: list[dict],
                             lm_matches: list[dict],
                             ) -> list[dict]:
    """
    Chain: llama_feat → qwen_feat → mistral_feat
    A feature triple is universal when:
      - llama feat has a Qwen match (lq)
      - that Qwen feat also has a Mistral match (mq)
    Also records the direct Llama-Mistral cosim if available.
    """
    # Llama → Qwen best match
    lq: dict[int, dict] = {}
    for m in lq_matches:
        lf, qf, s = m["feat_a"], m["feat_b"], m["cosim"]
        if lf not in lq or s > lq[lf]["cosim_lq"]:
            lq[lf] = {"qwen": qf, "cosim_lq": s}

    # Qwen → Mistral best match (indexed by qwen_feat)
    mq: dict[int, dict] = {}
    for m in mq_matches:
        mf, qf, s = m["feat_a"], m["feat_b"], m["cosim"]
        if qf not in mq or s > mq[qf]["cosim_mq"]:
            mq[qf] = {"mistral": mf, "cosim_mq": s}

    # Direct Llama → Mistral map
    lm_map: dict[int, float] = {}
    for m in lm_matches:
        lf, s = m["feat_a"], m["cosim"]
        if lf not in lm_map or s > lm_map[lf]:
            lm_map[lf] = s

    universal = []
    for lf, lq_info in lq.items():
        qf = lq_info["qwen"]
        if qf in mq:
            entry: dict = {
                "llama_feat":             lf,
                "qwen_feat":              qf,
                "mistral_feat":           mq[qf]["mistral"],
                "cosim_llama_qwen":       lq_info["cosim_lq"],
                "cosim_mistral_qwen":     mq[qf]["cosim_mq"],
            }
            if lf in lm_map:
                entry["cosim_llama_mistral"] = lm_map[lf]
            universal.append(entry)

    universal.sort(
        key=lambda e: -(e["cosim_llama_qwen"] + e["cosim_mistral_qwen"])
    )
    return universal


# ── Venn diagram SVG ─────────────────────────────────────────────────────────

def venn_svg(counts: dict, out_path: Path) -> None:
    """Three-circle Venn diagram saved as SVG."""
    import math as _math
    W, H = 700, 560
    spread = 95
    r      = 170
    cx, cy = 280, 240

    angles  = [-90 + i * 120 for i in range(3)]
    centres = [
        (cx + spread * _math.cos(_math.radians(a)),
         cy + spread * _math.sin(_math.radians(a)))
        for a in angles
    ]
    Lx, Ly = centres[0]   # Llama  (top)
    Qx, Qy = centres[1]   # Qwen   (bottom-right)
    Mx, My = centres[2]   # Mistral (bottom-left)

    def mid(a, b, f=0.50):
        return (a[0] * (1-f) + b[0] * f, a[1] * (1-f) + b[1] * f)

    lonly  = (Lx, Ly - r * 0.50)
    qonly  = (Qx + r * 0.50, Qy + r * 0.08)
    monly  = (Mx - r * 0.50, My + r * 0.08)
    lq_pt  = mid(centres[0], centres[1], 0.58)
    lm_pt  = mid(centres[0], centres[2], 0.58)
    qm_pt  = mid(centres[1], centres[2], 0.58)
    all_pt = (cx, cy + 14)

    def c(k): return str(counts.get(k, 0))

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">\n'
        f'  <title>SAE Universal Feature Overlap — Llama / Qwen / Mistral</title>\n'
        f'  <rect width="{W}" height="{H}" fill="#f8f9fa"/>\n'
        f'\n  <!-- circles -->\n'
        f'  <circle cx="{Lx:.1f}" cy="{Ly:.1f}" r="{r}" fill="#4e79a7" fill-opacity="0.28" stroke="#4e79a7" stroke-width="2"/>\n'
        f'  <circle cx="{Qx:.1f}" cy="{Qy:.1f}" r="{r}" fill="#f28e2b" fill-opacity="0.28" stroke="#f28e2b" stroke-width="2"/>\n'
        f'  <circle cx="{Mx:.1f}" cy="{My:.1f}" r="{r}" fill="#59a14f" fill-opacity="0.28" stroke="#59a14f" stroke-width="2"/>\n'
        f'\n  <!-- model labels -->\n'
        f'  <text x="{Lx:.1f}" y="{Ly - r - 14:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" font-weight="bold" fill="#2a4f78">Llama-3.2-3B</text>\n'
        f'  <text x="{Qx + r + 10:.1f}" y="{Qy - 6:.1f}" text-anchor="start" font-family="Arial,sans-serif" font-size="14" font-weight="bold" fill="#8a5200">Qwen2.5-3B</text>\n'
        f'  <text x="{Mx - r - 10:.1f}" y="{My - 6:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" font-weight="bold" fill="#2d5c27">Mistral-7B</text>\n'
        f'\n  <!-- region counts -->\n'
        f'  <text x="{lonly[0]:.1f}" y="{lonly[1]:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#2a4f78">{c("llama_only")}</text>\n'
        f'  <text x="{lonly[0]:.1f}" y="{lonly[1]+15:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#555">Llama only</text>\n'
        f'  <text x="{qonly[0]:.1f}" y="{qonly[1]:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#8a5200">{c("qwen_only")}</text>\n'
        f'  <text x="{qonly[0]:.1f}" y="{qonly[1]+15:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#555">Qwen only</text>\n'
        f'  <text x="{monly[0]:.1f}" y="{monly[1]:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="20" font-weight="bold" fill="#2d5c27">{c("mistral_only")}</text>\n'
        f'  <text x="{monly[0]:.1f}" y="{monly[1]+15:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#555">Mistral only</text>\n'
        f'  <text x="{lq_pt[0]:.1f}" y="{lq_pt[1]:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="#333">{c("llama_qwen")}</text>\n'
        f'  <text x="{lm_pt[0]:.1f}" y="{lm_pt[1]:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="#333">{c("llama_mistral")}</text>\n'
        f'  <text x="{qm_pt[0]:.1f}" y="{qm_pt[1]:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="18" font-weight="bold" fill="#333">{c("qwen_mistral")}</text>\n'
        f'  <text x="{all_pt[0]:.1f}" y="{all_pt[1]:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="26" font-weight="bold" fill="#111">{c("all_three")}</text>\n'
        f'  <text x="{all_pt[0]:.1f}" y="{all_pt[1]+16:.1f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#444">universal</text>\n'
        f'\n  <!-- footnote -->\n'
        f'  <text x="{W//2}" y="505" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#444">Chunk-averaged activation-pattern matching  ·  cosim &gt; 0.80</text>\n'
        f'  <text x="{W//2}" y="522" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#666">First 50 k tokens · wikitext-103-raw-v1 · 100 chunks · dict_size = 16 384</text>\n'
        f'  <text x="{W//2}" y="538" text-anchor="middle" font-family="Arial,sans-serif" font-size="10" fill="#888">Llama: 10 k steps  ·  Qwen: 50 k steps  ·  Mistral: 1 k steps (50 k token subset)</text>\n'
        f'</svg>\n'
    )
    out_path.write_text(svg)
    print(f"  Saved Venn → {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Cross-architecture SAE feature matching", flush=True)
    print(f"  EVAL_TOKENS={EVAL_TOKENS:,}  N_CHUNKS={N_CHUNKS}  "
          f"CHUNK_TOKS={CHUNK_TOKS}  SIM_THRESH={SIM_THRESH}", flush=True)

    # ── 1. Build fingerprints for all three models ────────────────────────────
    fingerprints: dict[str, np.ndarray] = {}
    for name, cfg in MODELS.items():
        ckpt_path = cfg["ckpt"]
        if not ckpt_path.exists():
            print(f"\n[ERROR] {name}: checkpoint not found: {ckpt_path}", flush=True)
            sys.exit(1)
        print(f"\n=== {name.upper()} ===", flush=True)
        sae = load_sae(ckpt_path)
        acts = load_acts_slice(cfg["acts"], cfg["d_in"], EVAL_TOKENS)
        fp   = build_chunk_fingerprints_fast(acts, sae)
        del acts
        fingerprints[name] = fp
        print(f"  fingerprint: {fp.shape}  "
              f"({(time.time()-t0)/60:.1f} min elapsed)", flush=True)

    # ── 2. Pairwise matching ──────────────────────────────────────────────────
    print("\n=== PAIRWISE MATCHING ===", flush=True)
    pairs = [
        ("llama",   "qwen",    "llama_qwen_matches.json"),
        ("mistral", "qwen",    "mistral_qwen_matches.json"),
        ("llama",   "mistral", "llama_mistral_matches.json"),
    ]
    pair_matches: dict[str, list] = {}
    for a, b, fname in pairs:
        print(f"\n  {a} ↔ {b}", flush=True)
        m = nearest_neighbor_match(
            fingerprints[a], fingerprints[b],
            label_a=a.capitalize(), label_b=b.capitalize(),
        )
        pair_matches[f"{a}_{b}"] = m
        (OUT_DIR / fname).write_text(
            json.dumps({"n_matches": len(m), "threshold": SIM_THRESH,
                        "eval_tokens": EVAL_TOKENS, "n_chunks": N_CHUNKS,
                        "matches": m}, indent=2)
        )

    lq_matches = pair_matches["llama_qwen"]
    mq_matches = pair_matches["mistral_qwen"]
    lm_matches = pair_matches["llama_mistral"]

    # ── 3. Universal features ─────────────────────────────────────────────────
    print("\n=== UNIVERSAL FEATURES ===", flush=True)
    universal = find_universal_features(lq_matches, mq_matches, lm_matches)
    print(f"  Universally replicated features: {len(universal)}", flush=True)

    # ── 4. Venn counts ────────────────────────────────────────────────────────
    dict_size = fingerprints["llama"].shape[0]

    llama_qwen_set    = {m["feat_a"] for m in lq_matches}
    qwen_llama_set    = {m["feat_b"] for m in lq_matches}
    mistral_qwen_set  = {m["feat_a"] for m in mq_matches}
    qwen_mistral_set  = {m["feat_b"] for m in mq_matches}
    llama_mistral_set = {m["feat_a"] for m in lm_matches}
    mistral_llama_set = {m["feat_b"] for m in lm_matches}

    llama_any    = llama_qwen_set | llama_mistral_set
    qwen_any     = qwen_llama_set | qwen_mistral_set
    mistral_any  = mistral_qwen_set | mistral_llama_set

    venn_counts = {
        "llama_only":    dict_size - len(llama_any),
        "qwen_only":     dict_size - len(qwen_any),
        "mistral_only":  dict_size - len(mistral_any),
        "llama_qwen":    len(qwen_llama_set - qwen_mistral_set),
        "llama_mistral": len(llama_mistral_set - llama_qwen_set),
        "qwen_mistral":  len(qwen_mistral_set - qwen_llama_set),
        "all_three":     len(universal),
    }
    print(f"  Venn: {venn_counts}", flush=True)

    # ── 5. Save universal-features.json ──────────────────────────────────────
    output = {
        "method":          "chunk-averaged activation pattern matching",
        "corpus":          "Salesforce/wikitext wikitext-103-raw-v1 (train)",
        "eval_tokens":     EVAL_TOKENS,
        "n_chunks":        N_CHUNKS,
        "chunk_tokens":    CHUNK_TOKS,
        "cosim_threshold": SIM_THRESH,
        "dict_size":       dict_size,
        "k":               128,
        "checkpoints": {
            "llama":   str(MODELS["llama"]["ckpt"]),
            "qwen":    str(MODELS["qwen"]["ckpt"]),
            "mistral": str(MODELS["mistral"]["ckpt"]),
        },
        "notes": {
            "llama":   "SAE trained 10k steps on 500k tokens (partial, 50% FVE target)",
            "qwen":    "SAE trained 50k steps on 500k tokens (fully converged, FVE=0.98)",
            "mistral": "SAE trained 1k steps on 50k-token subset; 4-bit weight precision",
        },
        "pair_match_counts": {
            "llama_qwen":    len(lq_matches),
            "mistral_qwen":  len(mq_matches),
            "llama_mistral": len(lm_matches),
        },
        "n_universal_features": len(universal),
        "venn_counts":           venn_counts,
        "universal_features":    universal,
    }
    out_json = OUT_DIR / "universal-features.json"
    out_json.write_text(json.dumps(output, indent=2))
    print(f"\n  Saved → {out_json}", flush=True)

    # ── 6. Venn diagram ───────────────────────────────────────────────────────
    venn_path = OUT_DIR / "venn_diagram.svg"
    venn_svg(venn_counts, venn_path)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min\n", flush=True)


if __name__ == "__main__":
    main()
