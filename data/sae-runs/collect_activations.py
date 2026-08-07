#!/usr/bin/env python3
"""
Collect residual-stream activations from an MLX language model for SAE training.

Reads text/JSONL files from a corpus directory, tokenises them, runs inference in
batches, captures the residual stream at a specified fractional layer depth, and
streams float16 activations to a binary file with a JSON metadata sidecar.

Usage
-----
python collect_activations.py \\
    --model  mlx-community/Llama-3.2-3B-bf16 \\
    --layer  0.5 \\
    --corpus /path/to/corpus \\
    --output /path/to/output \\
    --token-target 10_000_000 \\
    [--seq-len 512] \\
    [--batch-size 32] \\
    [--skip-bos]

Output
------
  <output>/activations.bin   – row-major float16, shape [N_tokens, hidden_dim]
  <output>/metadata.json     – shape, model info, collection params
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load


# ── Model utilities ───────────────────────────────────────────────────────────

def _causal_mask(seq_len: int, dtype) -> mx.array:
    return nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(dtype)


def _residuals_at_layer(model, input_ids: mx.array, layer_idx: int) -> mx.array:
    """
    Partial forward pass that stops after layer `layer_idx`.
    Returns the residual stream: shape (batch, seq_len, hidden_dim).
    """
    h = model.model.embed_tokens(input_ids)          # (B, T, D)
    mask = _causal_mask(h.shape[1], h.dtype)
    for i, layer in enumerate(model.model.layers):
        h = layer(h, mask=mask, cache=None)
        if i == layer_idx:
            break
    mx.eval(h)
    return h


# ── Corpus iteration ──────────────────────────────────────────────────────────

_TEXT_EXTS = {'.txt', '.md', '.jsonl', '.json'}


def _iter_chunks(
    corpus_dir: Path,
    tokenizer,
    seq_len: int,
) -> Iterator[list[int]]:
    """Yield non-overlapping fixed-length token-id lists from all text files."""
    files = sorted(p for p in corpus_dir.rglob('*')
                   if p.suffix in _TEXT_EXTS and p.is_file())
    if not files:
        raise ValueError(f'No text files ({_TEXT_EXTS}) found in {corpus_dir}')

    for path in files:
        text = path.read_text(errors='ignore')

        if path.suffix == '.jsonl':
            parts = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    parts.append(
                        obj.get('text') or obj.get('content') or obj.get('story') or ''
                    )
                except json.JSONDecodeError:
                    parts.append(line)
            text = '\n'.join(parts)

        ids: list[int] = tokenizer.encode(text)
        # mlx_lm tokenizers return list[int]; HF fast tokenizers do too.
        if not isinstance(ids, list):
            ids = list(ids)

        for start in range(0, len(ids) - seq_len + 1, seq_len):
            yield ids[start : start + seq_len]


# ── Verification ──────────────────────────────────────────────────────────────

def verify(bin_path: Path, meta_path: Path) -> None:
    """Load a random 1 k-token sample, assert shape and absence of NaN/Inf."""
    meta = json.loads(meta_path.read_text())
    n_tokens: int = meta['n_tokens']
    hidden_dim: int = meta['hidden_dim']
    bpt = hidden_dim * 2  # bytes per token (float16)

    check_n = min(1024, n_tokens)
    rng = np.random.default_rng(42)
    start_tok = int(rng.integers(0, max(1, n_tokens - check_n)))

    with open(bin_path, 'rb') as f:
        f.seek(start_tok * bpt)
        raw = f.read(check_n * bpt)

    arr = np.frombuffer(raw, dtype=np.float16).reshape(-1, hidden_dim)

    if arr.shape != (check_n, hidden_dim):
        sys.exit(f'VERIFY FAILED: shape {arr.shape} != ({check_n}, {hidden_dim})')

    arr32 = arr.astype(np.float32)
    n_nan = int(np.isnan(arr32).sum())
    n_inf = int(np.isinf(arr32).sum())

    if n_nan or n_inf:
        sys.exit(f'VERIFY FAILED: {n_nan} NaN, {n_inf} Inf in sampled batch '
                 f'(offset {start_tok:,})')

    print(f'  shape     : {arr.shape}')
    print(f'  sample    : offset={start_tok:,}, n={check_n:,}')
    print(f'  NaN/Inf   : 0 / 0')
    print(f'  range     : [{arr32.min():.4f}, {arr32.max():.4f}]  '
          f'mean={arr32.mean():.4f}')
    print('Verification PASSED.')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Collect residual-stream activations for SAE training'
    )
    p.add_argument('--model',        required=True, help='HF model ID or local path')
    p.add_argument('--layer',        type=float, required=True,
                   metavar='FLOAT',  help='Layer position 0.0 (first) – 1.0 (last)')
    p.add_argument('--corpus',       type=Path, required=True,
                   help='Directory of text / JSONL files')
    p.add_argument('--output',       type=Path, required=True,
                   help='Output directory (created if absent)')
    p.add_argument('--token-target', type=int, required=True,
                   help='Stop after collecting this many tokens')
    p.add_argument('--seq-len',      type=int, default=512,
                   help='Tokens per sequence chunk (default: 512)')
    p.add_argument('--batch-size',   type=int, default=32,
                   help='Sequences per forward pass (default: 32)')
    p.add_argument('--skip-bos',     action='store_true',
                   help='Exclude the first (BOS) token position from output')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.0 <= args.layer <= 1.0:
        sys.exit('--layer must be in [0.0, 1.0]')

    args.output.mkdir(parents=True, exist_ok=True)
    bin_path  = args.output / 'activations.bin'
    meta_path = args.output / 'metadata.json'

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading {args.model} ...', flush=True)
    model, tokenizer = load(args.model)

    n_layers   = len(model.model.layers)
    layer_idx  = round(args.layer * (n_layers - 1))
    # Prefer model.args.hidden_size: quantized embedding weights are stored in
    # compressed form so .weight.shape[1] == hidden_size // group_size (not hidden_size).
    if hasattr(model, 'args') and hasattr(model.args, 'hidden_size'):
        hidden_dim = model.args.hidden_size
    elif hasattr(model.model, 'args') and hasattr(model.model.args, 'hidden_size'):
        hidden_dim = model.model.args.hidden_size
    else:
        hidden_dim = model.model.embed_tokens.weight.shape[1]

    print(f'  layers    : {n_layers}')
    print(f'  capture   : layer {layer_idx} (pos={args.layer:.2f})')
    print(f'  hidden_dim: {hidden_dim}')
    print(f'  seq_len   : {args.seq_len}  batch_size={args.batch_size}')
    print(f'  target    : {args.token_target:,} tokens', flush=True)

    # ── Collect ───────────────────────────────────────────────────────────────
    t0           = time.time()
    total_tokens = 0
    total_seqs   = 0
    batch: list[list[int]] = []

    with open(bin_path, 'wb') as f_bin:

        def flush(batch: list[list[int]]) -> None:
            nonlocal total_tokens, total_seqs
            ids  = mx.array(batch, dtype=mx.int32)                    # (B, T)
            acts = _residuals_at_layer(model, ids, layer_idx)         # (B, T, D)
            if args.skip_bos:
                acts = acts[:, 1:, :]
            acts_np = np.array(acts.astype(mx.float32)).astype(np.float16)  # bf16→f32→f16
            f_bin.write(acts_np.tobytes())

            n = acts_np.shape[0] * acts_np.shape[1]
            total_tokens += n
            total_seqs   += len(batch)

            elapsed = time.time() - t0
            tok_s   = total_tokens / elapsed if elapsed else 0
            pct     = 100.0 * total_tokens / args.token_target
            print(
                f'  seqs={total_seqs:>7,}  '
                f'tokens={total_tokens:>12,} / {args.token_target:,}'
                f'  ({pct:5.1f}%)  {tok_s:>8,.0f} tok/s',
                flush=True,
            )

        for chunk in _iter_chunks(args.corpus, tokenizer, args.seq_len):
            batch.append(chunk)
            if len(batch) == args.batch_size:
                flush(batch)
                batch = []
                if total_tokens >= args.token_target:
                    break

        if batch and total_tokens < args.token_target:
            flush(batch)

    elapsed = time.time() - t0
    tok_s   = total_tokens / elapsed if elapsed else 0
    print(f'\nCollected {total_tokens:,} tokens in {elapsed:.1f}s ({tok_s:,.0f} tok/s)')

    # ── Metadata sidecar ──────────────────────────────────────────────────────
    seq_tokens = args.seq_len - (1 if args.skip_bos else 0)
    metadata = {
        'model':            args.model,
        'layer_idx':        layer_idx,
        'layer_pos':        args.layer,
        'n_layers':         n_layers,
        'hidden_dim':       hidden_dim,
        'n_tokens':         total_tokens,
        'n_seqs':           total_seqs,
        'seq_len':          seq_tokens,
        'batch_size':       args.batch_size,
        'skip_bos':         args.skip_bos,
        'dtype':            'float16',
        'shape':            [total_tokens, hidden_dim],
        'activations_file': bin_path.name,
        'corpus':           str(args.corpus.resolve()),
        'elapsed_s':        round(elapsed, 1),
        'tok_per_s':        round(tok_s, 1),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
    bin_size_gb = bin_path.stat().st_size / 1e9
    print(f'Metadata   : {meta_path}')
    print(f'Activations: {bin_path}  ({bin_size_gb:.2f} GB)')

    # ── Verification ─────────────────────────────────────────────────────────
    print('\nVerifying ...', flush=True)
    verify(bin_path, meta_path)


if __name__ == '__main__':
    main()
