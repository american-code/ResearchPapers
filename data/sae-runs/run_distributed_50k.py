#!/usr/bin/env python3
"""
Full distributed SAE training to 50k steps using the validated two-process setup.

Extends distributed_sae_poc.py (which validated convergence at 5k steps) to the
full 50k-step training run. Same two-worker gradient-averaging architecture.

Config: dict_size=4096, k=64, batch_per_worker=256 (same as POC)
Steps: 50,000 (10x POC)

Metrics logged per interval:
  - reconstruction MSE  (eval batch)
  - L0                  (mean active features per token on eval batch)
  - variance explained  (1 - MSE / var(x), eval batch)
  - dead feature rate   (features with 0 activations in last 5k-step window)

Checkpoints saved every 10k steps.
Runs single-node baseline at same config for comparison.

Output: data/sae-runs/distributed-llama3b/
"""

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


OUT_DIR = Path('data/sae-runs/distributed-llama3b')

CFG = {
    'd_in':             3072,
    'dict_size':        4096,
    'k':                64,
    'lr':               1e-4,
    'steps':            50_000,
    'batch_per_worker': 256,
    'warmup':           2000,   # 4% of total steps (same ratio as POC)
    'seed':             42,
    'log_interval':     500,
    'ckpt_interval':    10_000,
    'dead_window':      5000,
}


# ── Model (identical to distributed_sae_poc.py and train_sae.py) ─────────────

class TopKSAE(nn.Module):
    def __init__(self, d_in: int, dict_size: int, k: int, seed: int = 42):
        super().__init__()
        mx.random.seed(seed)
        self.k = k
        scale = (2.0 / (d_in + dict_size)) ** 0.5
        self.W_enc = mx.random.normal((d_in, dict_size)) * scale
        self.b_enc = mx.zeros((dict_size,))
        W_dec = mx.random.normal((dict_size, d_in)) * scale
        norms = mx.sqrt((W_dec ** 2).sum(axis=-1, keepdims=True) + 1e-8)
        self.W_dec = W_dec / norms
        self.b_dec = mx.zeros((d_in,))

    def __call__(self, x: mx.array):
        pre_acts = (x - self.b_dec) @ self.W_enc + self.b_enc
        top_vals = mx.topk(pre_acts, k=self.k, axis=-1)
        thresh   = mx.min(top_vals, axis=-1, keepdims=True)
        acts     = mx.where(pre_acts >= thresh, pre_acts, mx.zeros_like(pre_acts))
        recon    = acts @ self.W_dec + self.b_dec
        return recon, acts

    def normalize_decoder(self) -> None:
        norms = mx.sqrt((self.W_dec ** 2).sum(axis=-1, keepdims=True) + 1e-12)
        self.W_dec = self.W_dec / mx.maximum(norms, 1.0)
        mx.eval(self.W_dec)


# ── Utilities ─────────────────────────────────────────────────────────────────

def cosine_lr(step: int, peak: float, warmup: int, total: int, min_frac: float = 0.05) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return peak * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t)))


def mse_loss(model: TopKSAE, x: mx.array) -> mx.array:
    recon, _ = model(x)
    return ((x - recon) ** 2).mean()


def avg_grads(g1, g2):
    if isinstance(g1, mx.array):
        return (g1 + g2) * 0.5
    if isinstance(g1, dict):
        return {k: avg_grads(g1[k], g2[k]) for k in g1}
    if isinstance(g1, list):
        return [avg_grads(a, b) for a, b in zip(g1, g2)]
    return g1


def infinite_batches(acts: np.ndarray, batch: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    n, idx = acts.shape[0], np.arange(acts.shape[0])
    while True:
        rng.shuffle(idx)
        for s in range(0, n - batch + 1, batch):
            yield acts[idx[s:s + batch]].astype(np.float32)


def save_checkpoint(model: TopKSAE, path: Path) -> None:
    p = model.parameters()
    np.savez(str(path),
             W_enc=np.array(p['W_enc']),
             b_enc=np.array(p['b_enc']),
             W_dec=np.array(p['W_dec']),
             b_dec=np.array(p['b_dec']))


# ── Training loop (handles both distributed and single-node modes) ────────────

def run_training(acts_np: np.ndarray, cfg: dict, out_dir: Path, mode: str) -> tuple:
    """
    mode='distributed' : two workers, gradient averaging (workers simulated sequentially)
    mode='baseline'    : single worker, full dataset, same total batch size
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f'training_{mode}.jsonl'
    log_path.unlink(missing_ok=True)  # fresh start

    dict_size   = cfg['dict_size']
    dead_window = cfg['dead_window']
    steps       = cfg['steps']

    # Fixed eval batch for consistent metrics (same set every measurement)
    rng_eval = np.random.default_rng(cfg['seed'] + 999)
    eval_idx  = rng_eval.choice(len(acts_np), size=2048, replace=False)
    eval_np   = acts_np[eval_idx].astype(np.float32)
    eval_mx   = mx.array(eval_np)
    var_x     = float(np.var(eval_np))

    # Dead-feature tracker: last step at which each feature fired on the eval batch
    last_active = np.full(dict_size, -dead_window - 1, dtype=np.int32)

    model     = TopKSAE(cfg['d_in'], dict_size, cfg['k'], seed=cfg['seed'])
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=cfg['lr'])
    grad_fn   = nn.value_and_grad(model, mse_loss)

    if mode == 'distributed':
        half       = len(acts_np) // 2
        partition  = [acts_np[:half], acts_np[half:]]
        iterators  = [
            infinite_batches(partition[0], cfg['batch_per_worker'], seed=cfg['seed']),
            infinite_batches(partition[1], cfg['batch_per_worker'], seed=cfg['seed'] + 1),
        ]
    else:
        data_it = infinite_batches(acts_np, cfg['batch_per_worker'] * 2, seed=cfg['seed'])

    t_start             = time.time()
    loss_accum          = 0.0
    loss_count          = 0
    log_entries: list   = []

    for step in range(steps):
        lr                    = cosine_lr(step, cfg['lr'], cfg['warmup'], steps)
        optimizer.learning_rate = lr

        if mode == 'distributed':
            x0            = mx.array(next(iterators[0]))
            loss0, grads0 = grad_fn(model, x0)
            mx.eval(loss0, grads0)

            x1            = mx.array(next(iterators[1]))
            loss1, grads1 = grad_fn(model, x1)
            mx.eval(loss1, grads1)

            agg_grads = avg_grads(grads0, grads1)
            step_loss = (float(loss0) + float(loss1)) * 0.5
            optimizer.update(model, agg_grads)
            mx.eval(model.parameters(), optimizer.state)
        else:
            x = mx.array(next(data_it))
            loss, grads = grad_fn(model, x)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            step_loss = float(loss)

        model.normalize_decoder()
        loss_accum += step_loss
        loss_count += 1

        # Checkpoint
        if (step + 1) % cfg['ckpt_interval'] == 0:
            ckpt = out_dir / f'checkpoint_{mode}_step_{step+1:06d}.npz'
            save_checkpoint(model, ckpt)
            print(f'  [{mode}] checkpoint → {ckpt.name}', flush=True)

        # Log + metrics
        if (step + 1) % cfg['log_interval'] == 0 or step == 0:
            elapsed = time.time() - t_start

            # Forward on fixed eval batch (no gradient)
            recon_e, acts_e = model(eval_mx)
            mx.eval(recon_e, acts_e)
            recon_np = np.array(recon_e)
            acts_np_e = np.array(acts_e)

            mse_v     = float(np.mean((eval_np - recon_np) ** 2))
            var_exp   = 1.0 - mse_v / (var_x + 1e-12)
            l0        = float(np.mean((acts_np_e > 0).sum(axis=-1)))

            # Update dead-feature tracker
            fired = (acts_np_e > 0).any(axis=0)
            last_active[fired] = step + 1
            dead_count = int(((step + 1) - last_active > dead_window).sum())
            dead_rate  = dead_count / dict_size

            avg_loss = loss_accum / loss_count
            entry = {
                'step':          step + 1,
                'loss':          round(avg_loss,  6),
                'mse':           round(mse_v,     6),
                'l0':            round(l0,         2),
                'var_explained': round(var_exp,   4),
                'dead_features': dead_count,
                'dead_rate':     round(dead_rate, 4),
                'lr':            round(lr,         8),
                'elapsed_s':     round(elapsed,   1),
            }
            log_entries.append(entry)
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

            print(
                f'  [{mode}] step {step+1:>6,}/{steps}  '
                f'loss={avg_loss:.5f}  mse={mse_v:.5f}  '
                f'l0={l0:.1f}  fve={var_exp:.4f}  '
                f'dead={dead_count}({dead_rate:.1%})  '
                f'lr={lr:.2e}  {elapsed/60:.1f}min',
                flush=True,
            )
            loss_accum, loss_count = 0.0, 0

    return log_entries, time.time() - t_start, var_x


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / 'config.json').write_text(json.dumps(CFG, indent=2) + '\n')

    acts_dir = Path('data/activations/llama-3b-layer14')
    meta     = json.loads((acts_dir / 'metadata.json').read_text())
    n_tok    = meta['n_tokens_written']
    d_in     = meta['hidden_size']

    print(f'Loading {n_tok:,} × {d_in} float16 activations...', flush=True)
    t0      = time.time()
    acts_np = np.fromfile(str(acts_dir / 'activations.npy'), dtype=np.float16).reshape(n_tok, d_in)
    print(f'  loaded in {time.time()-t0:.1f}s', flush=True)

    # ── Distributed run ──────────────────────────────────────────────────────
    print('\n=== Distributed Training (2 workers, gradient averaging, 50k steps) ===',
          flush=True)
    dist_log, dist_elapsed, var_x = run_training(acts_np, CFG, OUT_DIR, mode='distributed')

    # ── Single-node baseline (same config) ───────────────────────────────────
    print('\n=== Single-Node Baseline (same config, 50k steps) ===', flush=True)
    base_log, base_elapsed, _    = run_training(acts_np, CFG, OUT_DIR, mode='baseline')

    # ── Final metrics & comparison ───────────────────────────────────────────
    df = dist_log[-1]
    bf = base_log[-1]
    mse_ratio  = df['mse'] / (bf['mse'] + 1e-12)
    within_5pct = abs(df['mse'] - bf['mse']) / (bf['mse'] + 1e-12) < 0.05

    result = {
        'metadata': {
            'source_model':    meta['model'],
            'source_layer':    meta['target_layer'],
            'n_source_tokens': n_tok,
            **CFG,
            'note': (
                'Full 50k-step distributed run. Two-process setup validated in '
                'data/distributed-sae-training-validation.json (5k steps, loss within 0.52% of baseline).'
            ),
        },
        'distributed': {
            'n_workers':      2,
            'data_partition': [
                f'worker_0: tokens [0:{n_tok//2}]',
                f'worker_1: tokens [{n_tok//2}:{n_tok}]',
            ],
            'aggregation':        'weighted_average (equal weights, alpha=0.5)',
            'elapsed_s':          round(dist_elapsed, 1),
            'elapsed_min':        round(dist_elapsed / 60, 1),
            'final_step':         df['step'],
            'final_mse':          df['mse'],
            'final_l0':           df['l0'],
            'final_var_explained': df['var_explained'],
            'final_dead_features': df['dead_features'],
            'final_dead_rate':    df['dead_rate'],
        },
        'baseline_same_config': {
            'n_workers':      1,
            'batch':          CFG['batch_per_worker'] * 2,
            'elapsed_s':      round(base_elapsed, 1),
            'elapsed_min':    round(base_elapsed / 60, 1),
            'final_step':     bf['step'],
            'final_mse':      bf['mse'],
            'final_l0':       bf['l0'],
            'final_var_explained': bf['var_explained'],
            'final_dead_features': bf['dead_features'],
            'final_dead_rate': bf['dead_rate'],
        },
        'comparison': {
            'final_mse_dist':          df['mse'],
            'final_mse_base':          bf['mse'],
            'mse_ratio_dist_over_base': round(mse_ratio, 4),
            'final_l0_dist':           df['l0'],
            'final_l0_base':           bf['l0'],
            'final_var_explained_dist': df['var_explained'],
            'final_var_explained_base': bf['var_explained'],
            'final_dead_rate_dist':    df['dead_rate'],
            'final_dead_rate_base':    bf['dead_rate'],
            'within_5pct_mse':         within_5pct,
            'verdict': (
                'PASS: distributed MSE within 5% of single-node baseline at step 50000'
                if within_5pct
                else f'INFO: distributed MSE {mse_ratio:.3f}x baseline (>{5}% gap)'
            ),
        },
        'comparison_to_earlier_single_node': {
            'note': (
                'Earlier single-node run at data/sae-runs/llama-3b-layer14 used '
                'dict_size=16384, k=128, batch=2048 — 4x larger dict and batch. '
                'Not directly comparable. Reference metrics at step 13k of that run: '
                'loss=0.007271, l0=128.0, fve=0.9835, dead_5k=5278.'
            ),
            'config_diff': {
                'distributed_run':  {'dict_size': CFG['dict_size'], 'k': CFG['k'], 'batch': CFG['batch_per_worker'] * 2},
                'earlier_run':      {'dict_size': 16384, 'k': 128, 'batch': 2048},
            },
        },
    }

    out_file = OUT_DIR / 'metrics_final.json'
    out_file.write_text(json.dumps(result, indent=2) + '\n')

    print(f'\n{"="*60}')
    print(f'Results → {OUT_DIR}/')
    print(f'  Distributed: MSE={df["mse"]:.5f}  L0={df["l0"]:.1f}  '
          f'FVE={df["var_explained"]:.4f}  dead={df["dead_features"]} ({df["dead_rate"]:.1%})')
    print(f'  Baseline   : MSE={bf["mse"]:.5f}  L0={bf["l0"]:.1f}  '
          f'FVE={bf["var_explained"]:.4f}  dead={bf["dead_features"]} ({bf["dead_rate"]:.1%})')
    print(f'  MSE ratio  : {mse_ratio:.4f}')
    print(f'  {result["comparison"]["verdict"]}')


if __name__ == '__main__':
    main()
