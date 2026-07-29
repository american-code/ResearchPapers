#!/usr/bin/env python3
"""
Train a TopK Sparse Autoencoder on pre-collected residual-stream activations (MLX).

Architecture: TopK-SAE
  Encoder : Linear(d_in → dict_size) → TopK(k)  [no ReLU, keep raw pre-acts at top-k positions]
  Decoder : Linear(dict_size → d_in), columns norm-clamped to ≤ 1 after each step
Loss     : MSE reconstruction
LR       : linear warmup → cosine decay to 5% of peak

Usage
-----
python data/sae-runs/train_sae.py \\
    --activations data/activations/llama-3b-layer16/ \\
    --output      data/sae-runs/llama-3b-layer16/ \\
    [--dict-size 16384] [--k 128] [--lr 1e-4] [--steps 50000] [--batch 2048]
"""

import argparse
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


# ── Model ─────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_in: int, dict_size: int, k: int, seed: int = 42):
        super().__init__()
        mx.random.seed(seed)
        self.k = k

        scale = (2.0 / (d_in + dict_size)) ** 0.5  # Xavier
        self.W_enc = mx.random.normal((d_in, dict_size)) * scale
        self.b_enc = mx.zeros((dict_size,))
        W_dec      = mx.random.normal((dict_size, d_in)) * scale
        norms      = mx.sqrt((W_dec ** 2).sum(axis=-1, keepdims=True) + 1e-8)
        self.W_dec = W_dec / norms  # start with unit-norm columns
        self.b_dec = mx.zeros((d_in,))

    def __call__(self, x: mx.array):
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc  # (B, dict_size)
        # mx.topk returns k largest values in arbitrary order; take min as threshold
        top_vals = mx.topk(pre_acts, k=self.k, axis=-1)         # (B, k) unordered
        thresh   = mx.min(top_vals, axis=-1, keepdims=True)      # (B, 1) kth-largest
        acts     = mx.where(pre_acts >= thresh, pre_acts, mx.zeros_like(pre_acts))
        recon    = acts @ self.W_dec + self.b_dec
        return recon, acts

    def normalize_decoder(self) -> None:
        """Clamp decoder column norms to ≤ 1 (no-op when already ≤ 1)."""
        norms      = mx.sqrt((self.W_dec ** 2).sum(axis=-1, keepdims=True) + 1e-12)
        self.W_dec = self.W_dec / mx.maximum(norms, 1.0)
        mx.eval(self.W_dec)


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_lr(step: int, peak: float, warmup: int, total: int, min_frac: float = 0.05) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return peak * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t)))


# ── Dead latent tracker ───────────────────────────────────────────────────────

class DeadTracker:
    """Track dead latents over a sliding window of logging steps."""

    def __init__(self, dict_size: int, window: int = 25):
        self.window   = window
        self._counts  = np.zeros(dict_size, dtype=np.int32)
        self._history: list[np.ndarray] = []

    def update(self, acts_batch: np.ndarray) -> None:
        """acts_batch: (B, dict_size) float32"""
        fired = (acts_batch != 0).any(axis=0).astype(np.int8)  # (dict_size,)
        self._history.append(fired)
        self._counts += fired
        if len(self._history) > self.window:
            self._counts -= self._history.pop(0)

    @property
    def dead_count(self) -> int:
        return int((self._counts == 0).sum())


# ── Data loader ───────────────────────────────────────────────────────────────

def infinite_batches(acts: np.ndarray, batch: int, seed: int = 0):
    """Yields raw numpy float32 batches; caller wraps with mx.array to avoid lazy-graph pile-up during advance."""
    rng = np.random.default_rng(seed)
    n   = acts.shape[0]
    idx = np.arange(n)
    while True:
        rng.shuffle(idx)
        for s in range(0, n - batch + 1, batch):
            yield acts[idx[s : s + batch]].astype(np.float32)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(model: TopKSAE, path: Path) -> None:
    data = np.load(str(path))
    model.W_enc = mx.array(data['W_enc'])
    model.b_enc = mx.array(data['b_enc'])
    model.W_dec = mx.array(data['W_dec'])
    model.b_dec = mx.array(data['b_dec'])
    mx.eval(model.parameters())
    print(f'  [loaded checkpoint: {path.name}]', flush=True)


def infer_start_step(path: Path) -> int:
    """Extract step number from filename like checkpoint_step_010000.npz."""
    import re
    m = re.search(r'step_(\d+)', path.stem)
    return int(m.group(1)) if m else 0


def save_checkpoint(model: TopKSAE, out_dir: Path, step: int) -> None:
    mx.eval(model.parameters())
    tag    = 'final' if step < 0 else f'step_{step:06d}'
    path   = out_dir / f'checkpoint_{tag}.npz'
    params = {
        'W_enc': np.array(model.W_enc),
        'b_enc': np.array(model.b_enc),
        'W_dec': np.array(model.W_dec),
        'b_dec': np.array(model.b_dec),
    }
    np.savez(str(path), **params)
    print(f'  [saved {path.name}]', flush=True)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train TopK-SAE on residual-stream activations')
    p.add_argument('--activations',   type=Path,  required=True,
                   help='Directory containing activations.npy + metadata.json')
    p.add_argument('--output',        type=Path,  required=True,
                   help='Output directory for checkpoints and logs')
    p.add_argument('--dict-size',     type=int,   default=16384)
    p.add_argument('--k',             type=int,   default=128)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--steps',         type=int,   default=50_000)
    p.add_argument('--batch',         type=int,   default=2048)
    p.add_argument('--warmup',        type=int,   default=500,
                   help='Linear warmup steps')
    p.add_argument('--log-interval',  type=int,   default=200,
                   help='Steps between JSONL log entries')
    p.add_argument('--ckpt-interval', type=int,   default=10_000,
                   help='Steps between checkpoint saves')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--resume-from',   type=Path,  default=None,
                   help='Path to a .npz checkpoint to resume from')
    p.add_argument('--start-step',    type=int,   default=None,
                   help='Step to resume at (inferred from checkpoint name if omitted)')
    p.add_argument('--skip-advance',  action='store_true',
                   help='Skip data-iterator fast-forward when resuming (safe for SAE; saves hours on large datasets)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load activations ──────────────────────────────────────────────────────
    meta_path = args.activations / 'metadata.json'
    meta      = json.loads(meta_path.read_text())
    acts_file = args.activations / meta.get('activations_file', 'activations.npy')

    print(f'Loading activations: {acts_file}', flush=True)
    try:
        acts_np = np.load(str(acts_file))
    except (ValueError, OSError):
        # Raw float16 binary (no numpy header — written with .tobytes())
        n_tok = meta.get('n_tokens_written') or meta.get('n_tokens')
        d_hid = meta.get('hidden_size') or meta.get('hidden_dim')
        acts_np = np.fromfile(str(acts_file), dtype=np.float16).reshape(n_tok, d_hid)
    n_tokens, d_in   = acts_np.shape
    print(f'  shape     : {acts_np.shape}  dtype={acts_np.dtype}')
    print(f'  model     : {meta.get("model", "?")}')
    print(f'  layer     : {meta.get("target_layer", meta.get("layer_idx", "?"))}')

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = {
        'source_model':   meta.get('model'),
        'source_layer':   meta.get('target_layer', meta.get('layer_idx')),
        'd_in':           d_in,
        'dict_size':      args.dict_size,
        'k':              args.k,
        'expansion':      round(args.dict_size / d_in, 2),
        'peak_lr':        args.lr,
        'steps':          args.steps,
        'batch':          args.batch,
        'warmup':         args.warmup,
        'seed':           args.seed,
        'n_source_tokens': n_tokens,
        'total_tokens':   args.steps * args.batch,
        'epochs_equiv':   round(args.steps * args.batch / n_tokens, 1),
    }
    print('\nConfig:')
    print(json.dumps(cfg, indent=2))
    (args.output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume_from is not None:
        start_step = (args.start_step if args.start_step is not None
                      else infer_start_step(args.resume_from))

    # ── Model & optimizer ─────────────────────────────────────────────────────
    model     = TopKSAE(d_in, args.dict_size, args.k, seed=args.seed)
    mx.eval(model.parameters())

    if args.resume_from is not None:
        print(f'Resuming from {args.resume_from}  (start_step={start_step:,})', flush=True)
        load_checkpoint(model, args.resume_from)

    optimizer = optim.Adam(learning_rate=args.lr)

    def forward_loss(model: TopKSAE, x: mx.array) -> mx.array:
        recon, _ = model(x)
        return ((x - recon) ** 2).mean()

    grad_fn = nn.value_and_grad(model, forward_loss)

    tracker = DeadTracker(args.dict_size, window=25)  # ≈ 25×log_interval steps window
    data_it = infinite_batches(acts_np, args.batch, seed=args.seed)
    log_path = args.output / 'training.jsonl'

    # Advance data iterator to reproduce the same data order from start_step
    if start_step > 0 and not args.skip_advance:
        print(f'Advancing data iterator by {start_step:,} steps...', flush=True)
        for _ in range(start_step):
            next(data_it)
        print('  done.', flush=True)
    elif start_step > 0:
        print(f'Skipping data iterator advance (--skip-advance); data order will differ from a continuous run.', flush=True)

    # Truncate log to entries at or before start_step, then switch to append mode
    if start_step > 0 and log_path.exists():
        kept = [ln for ln in log_path.read_text().splitlines()
                if json.loads(ln)['step'] <= start_step]
        log_path.write_text('\n'.join(kept) + ('\n' if kept else ''))
        print(f'  [log truncated to {len(kept)} entries (step ≤ {start_step:,})]', flush=True)

    t_start   = time.time()
    loss_accum = 0.0
    loss_count = 0

    print(f'\nTraining  steps={args.steps:,}  dict={args.dict_size}  '
          f'K={args.k}  lr={args.lr}  batch={args.batch}'
          + (f'  resuming_from={start_step:,}' if start_step > 0 else '') + '\n', flush=True)

    log_mode = 'a' if start_step > 0 else 'w'
    with open(log_path, log_mode) as log_f:

        for step in range(start_step, args.steps):
            lr                      = cosine_lr(step, args.lr, args.warmup, args.steps)
            optimizer.learning_rate = lr

            x              = mx.array(next(data_it))
            loss, grads    = grad_fn(model, x)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            model.normalize_decoder()

            loss_v      = float(loss)
            loss_accum += loss_v
            loss_count += 1

            # ── Log every log_interval steps ──────────────────────────────────
            if (step + 1) % args.log_interval == 0 or step == 0:
                recon, acts = model(x)
                mx.eval(recon, acts)
                acts_np_b  = np.array(acts)
                recon_np   = np.array(recon)
                x_np       = np.array(x)

                l0_actual  = float((acts_np_b != 0).sum(axis=-1).mean())
                l1_mean    = float(np.abs(acts_np_b[acts_np_b != 0]).mean()
                                   if (acts_np_b != 0).any() else 0.0)
                var_x      = float(np.var(x_np))
                fve        = float(1.0 - np.var(x_np - recon_np) / (var_x + 1e-8))
                avg_loss   = loss_accum / loss_count

                tracker.update(acts_np_b)

                elapsed = time.time() - t_start
                tok_s   = (step - start_step + 1) * args.batch / elapsed

                rec = {
                    'step':      step + 1,
                    'loss':      round(avg_loss, 6),
                    'l0':        round(l0_actual, 2),
                    'l1':        round(l1_mean, 6),
                    'fve':       round(fve, 4),
                    'dead_5k':   tracker.dead_count,
                    'lr':        round(lr, 8),
                    'elapsed_s': round(elapsed, 1),
                    'tok_s':     round(tok_s, 0),
                }
                log_f.write(json.dumps(rec) + '\n')
                log_f.flush()

                print(
                    f'step {step+1:>6,}/{args.steps:,}  '
                    f'loss={avg_loss:.4f}  '
                    f'L0={l0_actual:.1f}  '
                    f'FVE={fve:.3f}  '
                    f'dead={tracker.dead_count:>5,}  '
                    f'lr={lr:.2e}  '
                    f'{tok_s:>8,.0f} tok/s',
                    flush=True,
                )
                loss_accum = 0.0
                loss_count = 0

            # ── Checkpoint ────────────────────────────────────────────────────
            if (step + 1) % args.ckpt_interval == 0:
                save_checkpoint(model, args.output, step + 1)

    # ── Final save & metrics ──────────────────────────────────────────────────
    save_checkpoint(model, args.output, -1)

    total_elapsed = time.time() - t_start
    final_metrics = {
        **cfg,
        'training_elapsed_s': round(total_elapsed, 1),
        'training_elapsed_min': round(total_elapsed / 60, 1),
    }
    (args.output / 'metrics_final.json').write_text(
        json.dumps(final_metrics, indent=2) + '\n'
    )
    print(f'\nDone — {args.steps:,} steps in {total_elapsed/60:.1f} min')
    print(f'Output : {args.output}')


if __name__ == '__main__':
    main()
