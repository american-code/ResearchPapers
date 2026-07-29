#!/usr/bin/env python3
"""
Distributed SAE Training Proof of Concept

Partitions Llama-3.2-3B layer-16 activations into two halves.
Simulates two worker processes computing local gradients; coordinator
aggregates via weighted averaging and broadcasts weights.
Runs 5k steps, compares loss curve to single-process baseline on full dataset.

Architecture note: workers are simulated sequentially in Python (no actual IPC),
which is algebraically identical to real distributed processes — the gradient
averaging math is unchanged. This PoC validates convergence before a Swift/IPC
implementation.

Output: data/distributed-sae-training-validation.json
"""

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


# ── SAE (matches train_sae.py) ────────────────────────────────────────────────

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
        thresh = mx.min(top_vals, axis=-1, keepdims=True)
        acts = mx.where(pre_acts >= thresh, pre_acts, mx.zeros_like(pre_acts))
        recon = acts @ self.W_dec + self.b_dec
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


def avg_grads(g1, g2):
    """Recursively average two gradient pytrees (weighted equally)."""
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


def mse_loss(model: TopKSAE, x: mx.array) -> mx.array:
    recon, _ = model(x)
    return ((x - recon) ** 2).mean()


# ── Distributed training run ──────────────────────────────────────────────────

def run_distributed(acts: np.ndarray, cfg: dict) -> list[dict]:
    """
    Two workers each hold half of `acts`.
    Per step:
      worker_0 → (loss0, grads0) from its mini-batch
      worker_1 → (loss1, grads1) from its mini-batch
      coordinator → avg_grads, single optimizer.update, normalize decoder, broadcast
    """
    half = len(acts) // 2
    partition = [acts[:half], acts[half:]]
    print(f'  Partition sizes: {len(partition[0]):,} / {len(partition[1]):,} tokens', flush=True)

    model = TopKSAE(cfg['d_in'], cfg['dict_size'], cfg['k'], seed=cfg['seed'])
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=cfg['lr'])
    grad_fn = nn.value_and_grad(model, mse_loss)

    iterators = [
        infinite_batches(partition[0], cfg['batch_per_worker'], seed=cfg['seed']),
        infinite_batches(partition[1], cfg['batch_per_worker'], seed=cfg['seed'] + 1),
    ]

    log: list[dict] = []
    t_start = time.time()
    loss_accum, loss_count = 0.0, 0

    for step in range(cfg['steps']):
        lr = cosine_lr(step, cfg['lr'], cfg['warmup'], cfg['steps'])
        optimizer.learning_rate = lr

        # Worker 0 — local gradient on partition 0
        x0 = mx.array(next(iterators[0]))
        loss0, grads0 = grad_fn(model, x0)
        mx.eval(loss0, grads0)

        # Worker 1 — local gradient on partition 1
        x1 = mx.array(next(iterators[1]))
        loss1, grads1 = grad_fn(model, x1)
        mx.eval(loss1, grads1)

        # Coordinator — weighted average (equal weights) and parameter update
        agg_grads = avg_grads(grads0, grads1)
        avg_loss = (float(loss0) + float(loss1)) * 0.5

        optimizer.update(model, agg_grads)
        mx.eval(model.parameters(), optimizer.state)
        model.normalize_decoder()

        loss_accum += avg_loss
        loss_count += 1

        if (step + 1) % cfg['log_interval'] == 0 or step == 0:
            elapsed = time.time() - t_start
            entry = {
                'step': step + 1,
                'loss': round(loss_accum / loss_count, 6),
                'lr': round(lr, 8),
                'elapsed_s': round(elapsed, 1),
            }
            log.append(entry)
            print(
                f'  [dist] step {step+1:>5,}/{cfg["steps"]}  '
                f'loss={entry["loss"]:.4f}  lr={lr:.2e}  '
                f'{elapsed:.0f}s elapsed',
                flush=True,
            )
            loss_accum, loss_count = 0.0, 0

    return log


# ── Baseline single-process training ─────────────────────────────────────────

def run_baseline(acts: np.ndarray, cfg: dict) -> list[dict]:
    """Single process, same batch size as combined distributed workers, full dataset."""
    model = TopKSAE(cfg['d_in'], cfg['dict_size'], cfg['k'], seed=cfg['seed'])
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=cfg['lr'])
    grad_fn = nn.value_and_grad(model, mse_loss)

    data_it = infinite_batches(acts, cfg['batch_per_worker'] * 2, seed=cfg['seed'])

    log: list[dict] = []
    t_start = time.time()
    loss_accum, loss_count = 0.0, 0

    for step in range(cfg['steps']):
        lr = cosine_lr(step, cfg['lr'], cfg['warmup'], cfg['steps'])
        optimizer.learning_rate = lr

        x = mx.array(next(data_it))
        loss, grads = grad_fn(model, x)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        model.normalize_decoder()

        loss_v = float(loss)
        loss_accum += loss_v
        loss_count += 1

        if (step + 1) % cfg['log_interval'] == 0 or step == 0:
            elapsed = time.time() - t_start
            entry = {
                'step': step + 1,
                'loss': round(loss_accum / loss_count, 6),
                'lr': round(lr, 8),
                'elapsed_s': round(elapsed, 1),
            }
            log.append(entry)
            print(
                f'  [base] step {step+1:>5,}/{cfg["steps"]}  '
                f'loss={entry["loss"]:.4f}  lr={lr:.2e}  '
                f'{elapsed:.0f}s elapsed',
                flush=True,
            )
            loss_accum, loss_count = 0.0, 0

    return log


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ACTS_DIR = Path('data/activations/llama-3b-layer16')
    OUT_FILE = Path('data/distributed-sae-training-validation.json')

    meta = json.loads((ACTS_DIR / 'metadata.json').read_text())
    n_tok = meta['n_tokens_written']
    d_in = meta['hidden_size']

    print(f'Loading activations: {n_tok:,} × {d_in} float16 ...', flush=True)
    t_load = time.time()
    acts_np = np.fromfile(
        str(ACTS_DIR / meta.get('activations_file', 'activations.npy')),
        dtype=np.float16,
    ).reshape(n_tok, d_in)
    print(f'  loaded in {time.time()-t_load:.1f}s', flush=True)

    cfg = {
        'd_in':            d_in,
        'dict_size':       4096,   # smaller than prod (16384) for PoC speed
        'k':               64,
        'lr':              1e-4,
        'steps':           5000,
        'batch_per_worker': 256,   # 256×2=512 tokens/step (matches baseline)
        'warmup':          200,
        'seed':            42,
        'log_interval':    100,
    }

    print(f'\nConfig: dict_size={cfg["dict_size"]}  k={cfg["k"]}  '
          f'batch_per_worker={cfg["batch_per_worker"]}  steps={cfg["steps"]}', flush=True)

    print('\n=== Distributed Training (2 workers, gradient averaging) ===', flush=True)
    t0 = time.time()
    dist_log = run_distributed(acts_np, cfg)
    dist_elapsed = time.time() - t0
    print(f'Distributed done: {dist_elapsed/60:.1f} min', flush=True)

    print('\n=== Baseline Training (single process, full dataset) ===', flush=True)
    t0 = time.time()
    base_log = run_baseline(acts_np, cfg)
    base_elapsed = time.time() - t0
    print(f'Baseline done: {base_elapsed/60:.1f} min', flush=True)

    # ── Comparison ────────────────────────────────────────────────────────────
    dist_losses = [e['loss'] for e in dist_log]
    base_losses = [e['loss'] for e in base_log]
    dist_init, dist_final = dist_losses[0], dist_losses[-1]
    base_init, base_final = base_losses[0], base_losses[-1]
    auc_dist = sum(dist_losses) / len(dist_losses)
    auc_base = sum(base_losses) / len(base_losses)
    ratio = dist_final / (base_final + 1e-12)
    within_10pct = abs(dist_final - base_final) / (base_final + 1e-12) < 0.10

    verdict = (
        'PASS: distributed loss within 10% of baseline at step 5000'
        if within_10pct
        else f'INFO: distributed final loss {ratio:.3f}x baseline (>{10}% gap)'
    )

    result = {
        'metadata': {
            'source_model': meta['model'],
            'source_layer': meta['target_layer'],
            'n_source_tokens': n_tok,
            'd_in': d_in,
            'dict_size': cfg['dict_size'],
            'k': cfg['k'],
            'lr': cfg['lr'],
            'steps': cfg['steps'],
            'warmup': cfg['warmup'],
            'batch_per_worker': cfg['batch_per_worker'],
            'total_batch_per_step': cfg['batch_per_worker'] * 2,
            'seed': cfg['seed'],
            'log_interval': cfg['log_interval'],
            'implementation_note': (
                'Workers simulated sequentially in Python; gradient averaging is '
                'algebraically identical to real distributed Swift/IPC processes. '
                'This PoC validates convergence before implementing actual IPC.'
            ),
        },
        'distributed': {
            'n_workers': 2,
            'data_partition': [
                f'worker_0: tokens [0:{n_tok//2}]',
                f'worker_1: tokens [{n_tok//2}:{n_tok}]',
            ],
            'aggregation': 'weighted_average (equal weights, alpha=0.5)',
            'elapsed_s': round(dist_elapsed, 1),
            'initial_loss': round(dist_init, 6),
            'final_loss': round(dist_final, 6),
            'loss_reduction_pct': round((1 - dist_final / dist_init) * 100, 2),
            'mean_loss': round(auc_dist, 6),
            'curve': dist_log,
        },
        'baseline': {
            'n_workers': 1,
            'data': 'full dataset (same total tokens/step as distributed)',
            'batch': cfg['batch_per_worker'] * 2,
            'elapsed_s': round(base_elapsed, 1),
            'initial_loss': round(base_init, 6),
            'final_loss': round(base_final, 6),
            'loss_reduction_pct': round((1 - base_final / base_init) * 100, 2),
            'mean_loss': round(auc_base, 6),
            'curve': base_log,
        },
        'comparison': {
            'final_loss_dist': round(dist_final, 6),
            'final_loss_base': round(base_final, 6),
            'final_loss_ratio_dist_over_base': round(ratio, 4),
            'mean_loss_ratio_dist_over_base': round(auc_dist / auc_base, 4),
            'dist_converged': dist_final < dist_init,
            'base_converged': base_final < base_init,
            'within_10pct': within_10pct,
            'verdict': verdict,
        },
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(result, indent=2) + '\n')

    print(f'\nResults → {OUT_FILE}')
    print(f'  Distributed : init={dist_init:.4f}  final={dist_final:.4f}  '
          f'reduction={result["distributed"]["loss_reduction_pct"]:.1f}%')
    print(f'  Baseline    : init={base_init:.4f}  final={base_final:.4f}  '
          f'reduction={result["baseline"]["loss_reduction_pct"]:.1f}%')
    print(f'  Loss ratio  : {ratio:.4f}')
    print(f'  {verdict}')


if __name__ == '__main__':
    main()
