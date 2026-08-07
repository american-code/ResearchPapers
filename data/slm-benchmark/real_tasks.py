"""
real_tasks.py — an uncontaminated, difficulty-graded code-generation benchmark
derived from real research code in /Users/melton/ResearchPapers.

Every task is grounded in a specific function we actually wrote and shipped.
Constraints held throughout:
  * numpy + Python stdlib ONLY (no torch/transformers/mlx/scipy/sklearn/network)
  * no task requires the model to emit a markdown fence delimiter
  * fixtures are built in-memory (no disk I/O)
  * every task passes with its own reference solution (run validate())
"""

import json
import os
import subprocess
import sys
import tempfile

TIERS = ("EASY", "MEDIUM", "HARD")

TRAP_FAMILIES = (
    "tie-handling",
    "index-space",
    "order-of-operations",
    "streaming-boundary",
    "comment-vs-code",
)

# ---------------------------------------------------------------------------
# Shared in-memory fixture: a synthetic 32 x 256 TopK-SAE checkpoint.
# Injected ahead of any task whose dict has "fixture": True.
# ---------------------------------------------------------------------------

SAE_FIXTURE_SRC = r'''
import numpy as np

def make_sae_fixture(seed=0, d_in=32, dict_size=256):
    """Synthetic TopK-SAE checkpoint, same key names as our real .npz files."""
    rng = np.random.default_rng(seed)
    W_enc = (rng.standard_normal((d_in, dict_size)) * 0.1).astype(np.float32)
    b_enc = (rng.standard_normal(dict_size) * 0.01).astype(np.float32)
    W_dec = rng.standard_normal((dict_size, d_in)).astype(np.float32)
    W_dec = W_dec / np.linalg.norm(W_dec, axis=1, keepdims=True)
    b_dec = (rng.standard_normal(d_in) * 0.05).astype(np.float32)
    return {"W_enc": W_enc, "b_enc": b_enc,
            "W_dec": W_dec.astype(np.float32), "b_dec": b_dec}

def make_acts(n=64, d_in=32, seed=1):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d_in)).astype(np.float32)
'''

exec(SAE_FIXTURE_SRC, globals())


REAL_TASKS = [

    # =======================================================================
    # EASY
    # =======================================================================

    {
        "id": "rt_e01",
        "name": "cosine_lr",
        "tier": "EASY",
        "trap_family": "streaming-boundary",
        "source": "data/sae-runs/train_sae.py:65",
        "prompt": r'''
import math

def cosine_lr(step: int, peak: float, warmup: int, total: int, min_frac: float = 0.05) -> float:
    """
    Learning-rate schedule used to train our TopK sparse autoencoders:
    a linear warmup followed by a cosine decay down to a floor.

    During warmup the ramp is 1-indexed in the step: the very first step
    already receives a non-zero fraction of `peak`, and the last warmup step
    receives exactly `peak`.

    After warmup the schedule decays on a half-cosine over the REMAINING
    steps (total - warmup), from `peak` down to `min_frac * peak`.

    `warmup` may be 0 and `total` may equal `warmup`; neither may cause a
    division by zero.
    """
''',
        "reference": r'''
import math

def cosine_lr(step: int, peak: float, warmup: int, total: int, min_frac: float = 0.05) -> float:
    if step < warmup:
        return peak * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(total - warmup, 1)
    return peak * (min_frac + (1.0 - min_frac) * 0.5 * (1.0 + math.cos(math.pi * t)))
''',
        "tests": [
            "# TRAP: warmup is 1-indexed -- step 0 is NOT zero",
            "assert cosine_lr(0, 1.0, 500, 50000) > 0.0",
            "assert abs(cosine_lr(0, 1.0, 500, 50000) - 1.0 / 500) < 1e-12",
            "# last warmup step is exactly peak",
            "assert abs(cosine_lr(499, 1.0, 500, 50000) - 1.0) < 1e-12",
            "# first post-warmup step is also exactly peak (cos(0) = 1)",
            "assert abs(cosine_lr(500, 1.0, 500, 50000) - 1.0) < 1e-12",
            "# decay floor is min_frac * peak, not 0",
            "assert abs(cosine_lr(49999, 1.0, 500, 50000) - 0.05) < 1e-4",
            "assert cosine_lr(49999, 1.0, 500, 50000) > 0.049",
            "# decay normalises over (total - warmup), not total",
            "mid = cosine_lr(500 + (50000 - 500) // 2, 1.0, 500, 50000)",
            "assert abs(mid - (0.05 + 0.95 * 0.5)) < 1e-3",
            "# monotone non-increasing after warmup",
            "vals = [cosine_lr(s, 1.0, 500, 50000) for s in range(500, 50000, 977)]",
            "assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(len(vals) - 1))",
            "# degenerate params must not divide by zero",
            "assert abs(cosine_lr(0, 1.0, 0, 10) - 1.0) < 1e-12",
            "assert abs(cosine_lr(5, 1.0, 5, 5) - 1.0) < 1e-12",
            "assert abs(cosine_lr(3, 2.0, 4, 100, min_frac=0.0) - 2.0 * 4 / 4) < 1e-12",
        ],
    },

    {
        "id": "rt_e02",
        "name": "bootstrap_ci",
        "tier": "EASY",
        "trap_family": "index-space",
        "source": "data/ioi/run_statistical_validation.py:51",
        "prompt": r'''
import random

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    """
    Percentile bootstrap confidence interval for the mean, as used for the
    per-head activation-patching CIs in our IOI circuit work.

    Draw `n_boot` resamples of size len(values) with replacement (use
    random.randrange so the module-level `random` seed controls the result),
    take the mean of each, and sort them.

    The interval endpoints are taken by direct indexing into that sorted list
    of length n_boot -- no interpolation:
      * the lower endpoint is the element at position int(alpha / 2 * n_boot)
      * the upper endpoint is the LAST element strictly below the
        (1 - alpha/2) quantile position, i.e. one before
        int((1 - alpha / 2) * n_boot)

    This must not raise for alpha = 0.0.

    Returns (lo, hi).
    """
''',
        "reference": r'''
import random

def bootstrap_ci(values, n_boot=1000, alpha=0.05):
    n = len(values)
    boot_means = []
    for _ in range(n_boot):
        sample = [values[random.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int(alpha / 2 * n_boot)]
    hi = boot_means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi
''',
        "tests": [
            "random.seed(0)",
            "lo, hi = bootstrap_ci([3.5] * 40, n_boot=200)",
            "assert abs(lo - 3.5) < 1e-12 and abs(hi - 3.5) < 1e-12",
            "# TRAP: the -1 on the upper index is what keeps alpha=0 in range.",
            "# int((1 - 0.0) * n_boot) would be an off-the-end IndexError.",
            "random.seed(1)",
            "lo, hi = bootstrap_ci([0.0, 10.0], n_boot=200, alpha=0.0)",
            "assert lo == 0.0 and hi == 10.0",
            "random.seed(2)",
            "vals = [float(i) for i in range(100)]",
            "lo, hi = bootstrap_ci(vals, n_boot=500)",
            "assert lo < 49.5 < hi",
            "assert 0.0 < (hi - lo) < 25.0",
            "# interval must widen as alpha shrinks",
            "random.seed(3)",
            "l1, h1 = bootstrap_ci(vals, n_boot=500, alpha=0.50)",
            "random.seed(3)",
            "l2, h2 = bootstrap_ci(vals, n_boot=500, alpha=0.01)",
            "assert (h2 - l2) > (h1 - l1)",
            "# resampling is with replacement: a 1-element input is degenerate",
            "random.seed(4)",
            "lo, hi = bootstrap_ci([7.0], n_boot=25)",
            "assert lo == 7.0 and hi == 7.0",
        ],
    },

    {
        "id": "rt_e03",
        "name": "quantiles_and_overflow",
        "tier": "EASY",
        "trap_family": "order-of-operations",
        "source": "data/ioi/run_fp16_dynamic_range_audit.py:36",
        "prompt": r'''
import numpy as np

FP16_MAX = 65504.0

def quantiles_and_overflow(arr: np.ndarray) -> dict:
    """
    Distribution audit of a residual-stream tensor, used to check whether a
    float16 cast at a distributed-inference split boundary is lossless.

    Every reported percentile / max / mean is computed over the ABSOLUTE
    VALUES of the (flattened) input, not the raw values.

    An element counts as an overflow only when its absolute value is strictly
    greater than FP16_MAX.

    Returns a dict with keys:
      "p1", "p25", "p50", "p75", "p99"  -- percentiles of abs(arr), as floats
      "max_abs", "mean_abs"             -- floats
      "n_fp16_overflow"                 -- int
      "n_total"                         -- int, the number of ELEMENTS in arr
      "overflow_frac"                   -- n_fp16_overflow / n_total, or 0.0
                                           when n_total is 0
    """
''',
        "reference": r'''
import numpy as np

FP16_MAX = 65504.0

def quantiles_and_overflow(arr: np.ndarray) -> dict:
    abs_arr = np.abs(arr)
    n_overflow = int(np.sum(abs_arr > FP16_MAX))
    n_total = arr.size
    return {
        "p1":    float(np.percentile(abs_arr, 1)),
        "p25":   float(np.percentile(abs_arr, 25)),
        "p50":   float(np.percentile(abs_arr, 50)),
        "p75":   float(np.percentile(abs_arr, 75)),
        "p99":   float(np.percentile(abs_arr, 99)),
        "max_abs":         float(abs_arr.max()),
        "mean_abs":        float(abs_arr.mean()),
        "n_fp16_overflow": n_overflow,
        "n_total":         n_total,
        "overflow_frac":   n_overflow / n_total if n_total > 0 else 0.0,
    }
''',
        "tests": [
            "a = np.array([[-3.0, 1.0], [2.0, -70000.0]], dtype=np.float32)",
            "s = quantiles_and_overflow(a)",
            "assert set(s) == {'p1','p25','p50','p75','p99','max_abs','mean_abs','n_fp16_overflow','n_total','overflow_frac'}",
            "# TRAP: n_total is elements, not rows",
            "assert s['n_total'] == 4",
            "assert s['n_fp16_overflow'] == 1",
            "assert abs(s['overflow_frac'] - 0.25) < 1e-12",
            "# TRAP: percentiles are of abs(arr) -- sorted abs is [1, 2, 3, 70000]",
            "assert abs(s['p50'] - 2.5) < 1e-4",
            "assert abs(s['p25'] - 1.75) < 1e-4",
            "assert abs(s['p1'] - 1.03) < 1e-4",
            "assert abs(s['max_abs'] - 70000.0) < 1e-2",
            "assert abs(s['mean_abs'] - 17501.5) < 1e-2",
            "assert s['p1'] >= 0.0 and s['p1'] <= s['p25'] <= s['p50'] <= s['p75'] <= s['p99']",
            "# TRAP: the overflow test is strict '>' -- exactly FP16_MAX is fine",
            "b = np.array([65504.0], dtype=np.float32)",
            "assert quantiles_and_overflow(b)['n_fp16_overflow'] == 0",
            "c = np.array([-65504.5], dtype=np.float32)",
            "assert quantiles_and_overflow(c)['n_fp16_overflow'] == 1",
            "assert isinstance(s['n_fp16_overflow'], int) and isinstance(s['n_total'], int)",
            "assert isinstance(s['p50'], float) and isinstance(s['max_abs'], float)",
        ],
    },

    {
        "id": "rt_e04",
        "name": "rel_depth_bucket",
        "tier": "EASY",
        "trap_family": "order-of-operations",
        "source": "data/ioi/gen_cross_model_comparison.py:55",
        "prompt": r'''
def rel_depth_bucket(rel: float, tol: float = 0.08) -> str:
    """
    Map a head's relative layer depth (layer / n_layers) to a named thirds
    bucket for cross-model circuit comparison.

    The tolerance is subtracted from BOTH thirds boundaries, shifting them
    downward, so the buckets are:
        "early"  when rel <  1/3 - tol
        "middle" when rel <  2/3 - tol
        "late"   otherwise

    There is no upper bound on "late" and no lower bound on "early".
    """
''',
        "reference": r'''
def rel_depth_bucket(rel: float, tol: float = 0.08) -> str:
    if rel < 1/3 - tol:
        return "early"
    elif rel < 2/3 - tol:
        return "middle"
    else:
        return "late"
''',
        "tests": [
            "assert rel_depth_bucket(0.10) == 'early'",
            "assert rel_depth_bucket(0.50) == 'middle'",
            "assert rel_depth_bucket(0.90) == 'late'",
            "# TRAP: the tolerance shifts the boundaries DOWN, so values that",
            "# naive thirds would call 'early'/'middle' fall one bucket later.",
            "assert rel_depth_bucket(0.26) == 'middle'",
            "assert rel_depth_bucket(0.60) == 'late'",
            "assert rel_depth_bucket(0.25) == 'early'",
            "assert rel_depth_bucket(0.58) == 'middle'",
            "# unbounded ends",
            "assert rel_depth_bucket(1.5) == 'late'",
            "assert rel_depth_bucket(-0.5) == 'early'",
            "# tol is a real parameter",
            "assert rel_depth_bucket(0.30, tol=0.0) == 'early'",
            "assert rel_depth_bucket(0.30, tol=0.08) == 'middle'",
            "assert rel_depth_bucket(0.60, tol=0.0) == 'middle'",
            "assert rel_depth_bucket(0.0, tol=0.5) == 'middle'",
            "assert rel_depth_bucket(0.0, tol=0.7) == 'late'",
        ],
    },

    {
        "id": "rt_e05",
        "name": "val_to_hex",
        "tier": "EASY",
        "trap_family": "tie-handling",
        "source": "data/ioi/gen_patching_heatmap.py:113",
        "prompt": r'''
_CMAP = [
    (0.000, 255, 245, 240),
    (0.125, 254, 224, 210),
    (0.250, 252, 187, 161),
    (0.375, 252, 146, 114),
    (0.500, 251, 106,  74),
    (0.625, 239,  59,  44),
    (0.750, 203,  24,  29),
    (0.875, 165,  15,  21),
    (1.000, 103,   0,  13),
]

def val_to_hex(v: float) -> str:
    """
    Piecewise-linear ColorBrewer "Reds" colour lookup for the IOI activation
    patching heatmap.

    Clamp v into [0, 1], then walk _CMAP in order looking for the first
    segment (i, i+1) whose upper stop satisfies  v <= v1 + 1e-9  and
    linearly interpolate each channel across that segment with
    t = (v - v0) / (v1 - v0).

    Channel values are converted to integers by TRUNCATION, not rounding.

    Return a lowercase "#rrggbb" string with two hex digits per channel.
    If no segment matches, return the colour of the final stop.
    """
''',
        "reference": r'''
_CMAP = [
    (0.000, 255, 245, 240),
    (0.125, 254, 224, 210),
    (0.250, 252, 187, 161),
    (0.375, 252, 146, 114),
    (0.500, 251, 106,  74),
    (0.625, 239,  59,  44),
    (0.750, 203,  24,  29),
    (0.875, 165,  15,  21),
    (1.000, 103,   0,  13),
]

def val_to_hex(v: float) -> str:
    v = max(0.0, min(1.0, v))
    for i in range(len(_CMAP) - 1):
        v0, r0, g0, b0 = _CMAP[i]
        v1, r1, g1, b1 = _CMAP[i + 1]
        if v <= v1 + 1e-9:
            t = (v - v0) / (v1 - v0)
            r = int(r0 + t * (r1 - r0))
            g = int(g0 + t * (g1 - g0))
            b = int(b0 + t * (b1 - b0))
            return "#%02x%02x%02x" % (r, g, b)
    last = _CMAP[-1]
    return "#%02x%02x%02x" % (last[1], last[2], last[3])
''',
        "tests": [
            "assert val_to_hex(0.0) == '#fff5f0'",
            "assert val_to_hex(1.0) == '#67000d'",
            "# clamping happens before the lookup",
            "assert val_to_hex(-3.0) == '#fff5f0'",
            "assert val_to_hex(7.5) == '#67000d'",
            "# TRAP: 'v <= v1 + 1e-9' means a value sitting exactly on an interior",
            "# stop resolves in the LOWER segment with t == 1.0, not the upper one.",
            "assert val_to_hex(0.125) == '#fee0d2'",
            "assert val_to_hex(0.250) == '#fcbba1'",
            "# TRAP: channels are truncated, not rounded (g would be 0xab if rounded)",
            "assert val_to_hex(0.30) == '#fcaa8e'",
            "# well-formedness across the whole range",
            "import re as _re",
            "for i in range(0, 101):",
            "    h = val_to_hex(i / 100.0)",
            "    assert _re.fullmatch(r'#[0-9a-f]{6}', h), h",
            "# monotone darkening in the red channel over the top half",
            "reds = [int(val_to_hex(x)[1:3], 16) for x in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]",
            "assert all(reds[i] >= reds[i + 1] for i in range(len(reds) - 1))",
        ],
    },

    {
        "id": "rt_e06",
        "name": "average_precision",
        "tier": "EASY",
        "trap_family": "tie-handling",
        "source": "data/sae-analysis/recompute_corrections.py:134",
        "prompt": r'''
import numpy as np

def average_precision(y, s):
    """
    Step-wise average precision for the safety-classifier audit.

    Sort the examples by DESCENDING score using a STABLE sort (so that tied
    scores keep their original input order), then walk the ranking computing
    precision@k at every position. Average precision is the sum of
    precision@k over the positions holding a positive label, divided by the
    number of positives.

    This is deliberately NOT the interpolated / tie-grouped definition: tied
    scores are broken by input order, so the answer depends on that order.

    y : sequence of 0/1 (or bool) labels
    s : sequence of scores
    Returns a float. With zero positives, return 0.0 rather than NaN.
    """
''',
        "reference": r'''
import numpy as np

def average_precision(y, s):
    y = np.asarray(y, dtype=bool)
    order = np.argsort(-np.asarray(s, dtype=float), kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(int(y.sum()), 1))
''',
        "tests": [
            "assert abs(average_precision([1, 1, 0, 0], [4, 3, 2, 1]) - 1.0) < 1e-12",
            "assert abs(average_precision([1, 1, 0, 0], [1, 2, 3, 4]) - (1/3 + 0.5) / 2) < 1e-12",
            "# no positives -> 0.0, not NaN and not a ZeroDivisionError",
            "v = average_precision([0, 0, 0], [1.0, 2.0, 3.0])",
            "assert v == 0.0 and v == v",
            "# all positives -> 1.0",
            "assert abs(average_precision([1, 1, 1], [0.1, 0.2, 0.3]) - 1.0) < 1e-12",
            "# TRAP: every score tied -> stable sort keeps the INPUT order, so the",
            "# result is 0.8333..., not the 0.5 a tie-grouped definition gives.",
            "assert abs(average_precision([1, 0, 1, 0], [1.0, 1.0, 1.0, 1.0]) - (1.0 + 2/3) / 2) < 1e-12",
            "# and reversing the input order with the same tied scores changes it",
            "assert abs(average_precision([0, 1, 0, 1], [1.0, 1.0, 1.0, 1.0]) - (0.5 + 0.5) / 2) < 1e-12",
            "# accepts bools and numpy arrays alike",
            "assert abs(average_precision(np.array([True, False]), np.array([2.0, 1.0])) - 1.0) < 1e-12",
            "assert isinstance(average_precision([1, 0], [1.0, 0.0]), float)",
        ],
    },

    {
        "id": "rt_e07",
        "name": "avg_grads",
        "tier": "EASY",
        "trap_family": "order-of-operations",
        "source": "data/sae-runs/distributed_sae_poc.py:67",
        "prompt": r'''
import numpy as np

def avg_grads(g1, g2):
    """
    Coordinator-side gradient aggregation for our two-worker distributed SAE
    training proof of concept: recursively average two gradient pytrees with
    equal weights.

    Dispatch, in this order:
      * numpy array -> elementwise (g1 + g2) * 0.5
      * dict        -> recurse over the KEYS OF g1 only
      * list        -> recurse pairwise over zip(g1, g2)
      * anything else (scalars, tuples, None, ...) -> return g1 unchanged

    That final fallback is intentional: a leaf the aggregator does not
    recognise is passed through from worker 0 rather than averaged or
    rejected.
    """
''',
        "reference": r'''
import numpy as np

def avg_grads(g1, g2):
    if isinstance(g1, np.ndarray):
        return (g1 + g2) * 0.5
    if isinstance(g1, dict):
        return {k: avg_grads(g1[k], g2[k]) for k in g1}
    if isinstance(g1, list):
        return [avg_grads(a, b) for a, b in zip(g1, g2)]
    return g1
''',
        "tests": [
            "a = np.array([0.0, 2.0]); b = np.array([4.0, 6.0])",
            "out = avg_grads({'W': a, 'layers': [a, b]}, {'W': b, 'layers': [b, a]})",
            "assert np.allclose(out['W'], [2.0, 4.0])",
            "assert np.allclose(out['layers'][0], [2.0, 4.0])",
            "assert np.allclose(out['layers'][1], [2.0, 4.0])",
            "# TRAP: an unrecognised leaf type silently DROPS g2 and returns g1",
            "assert avg_grads({'lr': 1.0}, {'lr': 3.0}) == {'lr': 1.0}",
            "assert avg_grads((1.0, 2.0), (5.0, 6.0)) == (1.0, 2.0)",
            "assert avg_grads(None, np.array([1.0])) is None",
            "assert avg_grads(7, 9) == 7",
            "# TRAP: dict recursion iterates g1's keys, so extra keys in g2 vanish",
            "r = avg_grads({'a': a}, {'a': b, 'b': b})",
            "assert set(r.keys()) == {'a'}",
            "# TRAP: list recursion uses zip, so it truncates to the shorter list",
            "r = avg_grads([a, a], [b, b, b])",
            "assert isinstance(r, list) and len(r) == 2",
            "# nesting depth is unbounded",
            "deep = avg_grads({'x': [{'y': [a]}]}, {'x': [{'y': [b]}]})",
            "assert np.allclose(deep['x'][0]['y'][0], [2.0, 4.0])",
            "# inputs must not be mutated",
            "src = np.array([0.0, 2.0]); _ = avg_grads(src, np.array([4.0, 6.0]))",
            "assert np.allclose(src, [0.0, 2.0])",
        ],
    },

    {
        "id": "rt_e08",
        "name": "best_match",
        "tier": "EASY",
        "trap_family": "streaming-boundary",
        "source": "data/sae-analysis/cross_arch_matching_v3.py:120",
        "prompt": r'''
import numpy as np

def best_match(fp_a, fp_b, batch=2048):
    """
    Nearest-neighbour search between two SAE feature-fingerprint matrices,
    evaluated in row batches so the full similarity matrix is never
    materialised.

    fp_a : (n_a, N) float array
    fp_b : (n_b, N) float array

    For every row of fp_a, find the row of fp_b maximising the dot product.
    Ties resolve to the LOWEST column index.

    Returns (idx, val):
      idx : (n_a,) array of np.int32, the chosen fp_b row indices
      val : (n_a,) array of np.float32, the corresponding similarities

    The result must be identical for any `batch` value, including one that
    does not divide n_a evenly.
    """
''',
        "reference": r'''
import numpy as np

def best_match(fp_a, fp_b, batch=2048):
    n = fp_a.shape[0]
    idx = np.empty(n, dtype=np.int32)
    val = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch):
        sims = fp_a[i:i + batch] @ fp_b.T
        j = np.argmax(sims, axis=1)
        idx[i:i + batch] = j
        val[i:i + batch] = sims[np.arange(len(j)), j]
    return idx, val
''',
        "tests": [
            "rng = np.random.default_rng(0)",
            "A = rng.standard_normal((7, 5)).astype(np.float32)",
            "B = rng.standard_normal((11, 5)).astype(np.float32)",
            "i1, v1 = best_match(A, B, batch=1000)",
            "assert i1.shape == (7,) and v1.shape == (7,)",
            "assert i1.dtype == np.int32 and v1.dtype == np.float32",
            "# TRAP: the final short batch must not be mis-sized or dropped",
            "for bs in (1, 2, 3, 4, 6, 7, 8, 1000):",
            "    i2, v2 = best_match(A, B, batch=bs)",
            "    assert np.array_equal(i1, i2), bs",
            "    assert np.allclose(v1, v2, atol=1e-6), bs",
            "# matches a brute-force reference",
            "full = A @ B.T",
            "assert np.array_equal(i1, np.argmax(full, axis=1).astype(np.int32))",
            "assert np.allclose(v1, full[np.arange(7), np.argmax(full, axis=1)], atol=1e-6)",
            "# TRAP: ties resolve to the lowest index",
            "Bt = np.ones((4, 3), dtype=np.float32)",
            "At = np.ones((3, 3), dtype=np.float32)",
            "it, vt = best_match(At, Bt, batch=2)",
            "assert np.array_equal(it, np.zeros(3, dtype=np.int32))",
            "assert np.allclose(vt, 3.0)",
            "# single row, single candidate",
            "i3, v3 = best_match(np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32))",
            "assert i3.tolist() == [0] and abs(float(v3[0]) - 2.0) < 1e-6",
        ],
    },

    # =======================================================================
    # MEDIUM
    # =======================================================================

    {
        "id": "rt_m01",
        "name": "roc_auc",
        "tier": "MEDIUM",
        "trap_family": "tie-handling",
        "source": "data/sae-analysis/recompute_corrections.py:115",
        "prompt": r'''
import numpy as np

def roc_auc(y, s):
    """
    ROC-AUC computed through the Mann-Whitney U rank identity, with proper
    handling of tied scores. Used to give the safety classifier a
    threshold-free metric.

    Rank the scores in ASCENDING order using a stable sort. Ranks are
    1-BASED. Every group of exactly-equal scores receives the AVERAGE of the
    1-based ranks that group spans.

    With n1 positives and n0 negatives,
        AUC = (sum of ranks of the positives - n1 * (n1 + 1) / 2) / (n1 * n0)

    y : sequence of 0/1 (or bool) labels
    s : sequence of scores
    Returns a float.
    """
''',
        "reference": r'''
import numpy as np

def roc_auc(y, s):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
''',
        "tests": [
            "assert abs(roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) - 1.0) < 1e-12",
            "assert abs(roc_auc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) - 0.0) < 1e-12",
            "# TRAP: all scores tied -> every rank is the group average -> exactly 0.5",
            "assert abs(roc_auc([1, 0, 1, 0], [1.0, 1.0, 1.0, 1.0]) - 0.5) < 1e-12",
            "# TRAP: partial tie. Ranks must be 1-BASED averages. The correct value",
            "# is 0.875; 0-based averaged ranks give 0.375, un-averaged give 1.0.",
            "v = roc_auc([0, 1, 1, 0], [1.0, 1.0, 2.0, 0.5])",
            "assert abs(v - 0.875) < 1e-12, v",
            "# cross-check against the pairwise definition on random data",
            "rng = np.random.default_rng(5)",
            "for trial in range(6):",
            "    lab = rng.integers(0, 2, 40)",
            "    sc = np.round(rng.standard_normal(40), 1)",
            "    if lab.sum() == 0 or lab.sum() == 40:",
            "        continue",
            "    pos = sc[lab == 1]; neg = sc[lab == 0]",
            "    brute = float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))",
            "    assert abs(roc_auc(lab, sc) - brute) < 1e-10, (trial, roc_auc(lab, sc), brute)",
            "# a coarse-grained score column (lots of ties) must still work",
            "lab = [1, 1, 0, 0, 1, 0]",
            "sc = [1, 1, 1, 0, 0, 0]",
            "pos = np.array([1.0, 1.0, 0.0]); neg = np.array([1.0, 0.0, 0.0])",
            "brute = float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / 9)",
            "assert abs(roc_auc(lab, sc) - brute) < 1e-12",
            "assert isinstance(roc_auc([0, 1], [0.0, 1.0]), float)",
        ],
    },

    {
        "id": "rt_m02",
        "name": "topk_encode",
        "tier": "MEDIUM",
        "trap_family": "tie-handling",
        "source": "data/sae-analysis/cross_arch_matching_v3.py:90",
        "fixture": True,
        "prompt": r'''
import numpy as np

def topk_encode(acts: np.ndarray, sae: dict, k: int = 128) -> np.ndarray:
    """
    Forward pass of the TopK sparse autoencoders we trained on residual-stream
    activations.

    `sae` is a dict with keys "W_enc" (d_in, dict_size), "b_enc" (dict_size,),
    "W_dec" (dict_size, d_in) and "b_dec" (d_in,).

    This architecture uses a TIED pre-encoder bias: the DECODER bias is
    subtracted from the input before the encoder projection.

        pre = (acts - b_dec) @ W_enc + b_enc

    Sparsify each row by keeping the entries greater than or equal to that
    row's k-th largest pre-activation and zeroing the rest. There is NO
    ReLU -- a negative pre-activation that lands in the top k is kept with
    its sign.

    Because the gate is a >= comparison against the k-th largest value, a row
    with tied pre-activations at the threshold keeps MORE than k entries.

    acts : (n, d_in). Returns (n, dict_size).
    """
''',
        "reference": r'''
import numpy as np

def topk_encode(acts: np.ndarray, sae: dict, k: int = 128) -> np.ndarray:
    pre = (acts - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    kth = np.partition(pre, -k, axis=1)[:, -k][:, None]
    return np.where(pre >= kth, pre, 0.0)
''',
        "tests": [
            "sae = make_sae_fixture(0)",
            "x = make_acts(64, 32, seed=1)",
            "out = topk_encode(x, sae, k=8)",
            "assert out.shape == (64, 256)",
            "# generic float data has no ties -> exactly k survive",
            "assert np.array_equal((out != 0).sum(axis=1), np.full(64, 8))",
            "# TRAP: the decoder bias is subtracted from the input first.",
            "pre_wrong = x @ sae['W_enc'] + sae['b_enc']",
            "kth_w = np.partition(pre_wrong, -8, axis=1)[:, -8][:, None]",
            "wrong = np.where(pre_wrong >= kth_w, pre_wrong, 0.0)",
            "assert not np.allclose(out, wrong)",
            "# surviving values are the raw pre-activations, unscaled",
            "pre = (x - sae['b_dec']) @ sae['W_enc'] + sae['b_enc']",
            "m = out != 0",
            "assert np.allclose(out[m], pre[m], atol=1e-6)",
            "# TRAP: no ReLU -- with a large k, negative pre-acts must survive",
            "big = topk_encode(x, sae, k=200)",
            "assert (big < 0).any()",
            "assert np.array_equal((big != 0).sum(axis=1), np.full(64, 200))",
            "# TRAP: exact ties at the threshold keep MORE than k entries",
            "tie_sae = {'W_enc': np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),",
            "           'b_enc': np.zeros(4), 'b_dec': np.zeros(2),",
            "           'W_dec': np.zeros((4, 2))}",
            "t = topk_encode(np.array([[1.0, 0.0]]), tie_sae, k=1)",
            "assert int((t != 0).sum()) == 2, t",
            "assert np.allclose(t, [[1.0, 1.0, 0.0, 0.0]])",
            "# k == dict_size keeps everything (including negatives)",
            "allk = topk_encode(x[:3], sae, k=256)",
            "assert np.allclose(allk, pre[:3], atol=1e-6)",
        ],
    },

    {
        "id": "rt_m03",
        "name": "reciprocal_matches",
        "tier": "MEDIUM",
        "trap_family": "index-space",
        "source": "data/sae-analysis/cross_arch_matching_v3.py:132",
        "prompt": r'''
import numpy as np

def reciprocal_matches(fp_a, fp_b, valid_a, valid_b, tau):
    """
    Mutual-nearest-neighbour (reciprocal best match) feature pairing between
    two SAE fingerprint matrices, restricted to features with enough support.

    fp_a    : (D_a, N) rows are per-feature fingerprints
    fp_b    : (D_b, N)
    valid_a : (D_a,) boolean support mask
    valid_b : (D_b,) boolean support mask
    tau     : float similarity threshold

    Matching runs over the COMPACTED submatrices fp_a[valid_a] and
    fp_b[valid_b] using the dot product as the similarity. A pair qualifies
    when each side is the other's argmax (ties resolve to the lowest index).
    Reciprocity is checked BEFORE the threshold is applied.

    The returned indices must be remapped back into the ORIGINAL (global)
    row numbering of fp_a and fp_b -- not the compacted numbering.

    Returns a list of (feat_a, feat_b, sim) tuples where feat_a and feat_b are
    plain Python ints and sim is a plain Python float rounded to 4 decimals.
    Order follows ascending compacted fp_a index.
    """
''',
        "reference": r'''
import numpy as np

def reciprocal_matches(fp_a, fp_b, valid_a, valid_b, tau):
    ia = np.flatnonzero(valid_a)
    ib = np.flatnonzero(valid_b)
    A = fp_a[ia]
    B = fp_b[ib]
    sims = A @ B.T
    a2b = np.argmax(sims, axis=1)
    a2b_s = sims[np.arange(len(a2b)), a2b]
    b2a = np.argmax(sims.T, axis=1)
    out = []
    for local_a, local_b in enumerate(a2b):
        if b2a[local_b] != local_a:
            continue
        s = float(a2b_s[local_a])
        if s < tau:
            continue
        out.append((int(ia[local_a]), int(ib[local_b]), round(s, 4)))
    return out
''',
        "tests": [
            "e0 = np.array([1.0, 0.0, 0.0]); e1 = np.array([0.0, 1.0, 0.0]); e2 = np.array([0.0, 0.0, 1.0])",
            "fp_a = np.stack([e0, e1, e2, e0])",
            "fp_b = np.stack([e1, e0, e2, e1])",
            "va = np.array([False, True, True, False])",
            "vb = np.array([False, False, True, True])",
            "res = reciprocal_matches(fp_a, fp_b, va, vb, 0.5)",
            "# TRAP: results must be in GLOBAL index space. Without the remap you",
            "# would get [(0, 1, 1.0), (1, 0, 1.0)] -- same shape, wrong ids.",
            "assert res == [(1, 3, 1.0), (2, 2, 1.0)], res",
            "assert all(isinstance(a, int) and isinstance(b, int) and isinstance(s, float) for a, b, s in res)",
            "# threshold filter",
            "u = np.array([[1.0, 0.0]]); w = np.array([[0.6, 0.8]])",
            "on = np.array([True])",
            "assert reciprocal_matches(u, w, on, on, 0.5) == [(0, 0, 0.6)]",
            "assert reciprocal_matches(u, w, on, on, 0.8) == []",
            "# TRAP: a high-similarity pair that is NOT mutual is dropped, even",
            "# though its similarity comfortably clears tau.",
            "a0 = np.array([1.0, 0.0])",
            "a1 = np.array([0.98, 0.19899]); a1 = a1 / np.linalg.norm(a1)",
            "b0 = np.array([0.99, 0.141067]); b0 = b0 / np.linalg.norm(b0)",
            "b1 = np.array([0.0, 1.0])",
            "FA = np.stack([a0, a1]); FB = np.stack([b0, b1])",
            "on2 = np.array([True, True])",
            "r2 = reciprocal_matches(FA, FB, on2, on2, 0.5)",
            "# a0's best match is b0 with sim 0.99, comfortably above tau, but b0",
            "# prefers a1, so the a0 pair is dropped and only a1 survives.",
            "assert len(r2) == 1, r2",
            "assert [p[0] for p in r2] == [1], r2",
            "assert r2[0][1] == 0",
            "assert float(FA[0] @ FB[0]) > 0.5",
            "# rounding to 4 dp",
            "p = np.array([[1.0, 0.0, 0.0]]); q = np.array([[1/3, 0.0, 0.0]])",
            "assert reciprocal_matches(p, q, np.array([True]), np.array([True]), 0.0) == [(0, 0, 0.3333)]",
            "# narrowing the support masks changes WHICH global ids come back",
            "va2 = np.array([True, False, True, False])",
            "vb2 = np.array([False, True, True, False])",
            "res2 = reciprocal_matches(fp_a, fp_b, va2, vb2, 0.5)",
            "assert res2 == [(0, 1, 1.0), (2, 2, 1.0)], res2",
            "# a single valid row on each side is always trivially reciprocal",
            "one_a = np.array([True, False, False, False])",
            "one_b = np.array([False, False, True, False])",
            "assert reciprocal_matches(fp_a, fp_b, one_a, one_b, -1.0) == [(0, 2, 0.0)]",
            "assert reciprocal_matches(fp_a, fp_b, one_a, one_b, 0.5) == []",
            "# results are ordered by ascending compacted fp_a index",
            "assert [r[0] for r in res] == sorted(r[0] for r in res)",
        ],
    },

    {
        "id": "rt_m04",
        "name": "heuristic_pos",
        "tier": "MEDIUM",
        "trap_family": "order-of-operations",
        "source": "scripts/extract_rules.py:78",
        "prompt": r'''
_FUNC = {
    "the","a","an","of","to","in","and","or","is","are","was","were","be",
    "for","on","at","by","with","from","that","this","it","he","she","they",
    "we","you","i","as","which","who","what","when","where","its","their",
    "our","your","my","his","her","also","been","into","out","up","about",
    "after","before","over","under","between","through","not","no","so",
    "if","then","than",
}
_VERB_SUF = ("ing", "ed", "ify", "ize", "ise", "ate", "ify")
_ADJ_SUF  = ("al", "ful", "less", "ous", "ive", "ish", "able", "ible", "ic", "ary")
_NEG      = {"not", "no", "never", "neither", "nor", "without", "nothing", "nobody"}

def heuristic_pos(tok: str) -> str:
    """
    Cheap part-of-speech guess for a subword token, used to summarise which
    grammatical categories an SAE feature fires on.

    Let t be tok STRIPPED first and lowercased second. The strip set is
    exactly the six characters space, tab, newline, U+2581 (the SentencePiece
    marker), U+0120 (the GPT-2 marker) and underscore, removed from both ends.
    Stripping must happen before lowercasing, because U+0120 is an uppercase
    letter whose lowercase form is not in the strip set.

    Apply these rules IN ORDER, returning at the first match:
      1. t is empty                      -> "OTHER"
      2. t[0] is not alphabetic          -> "OTHER"
      3. t in _NEG                       -> "FUNC"
      4. t in _FUNC                      -> "FUNC"
      5. the ORIGINAL tok is non-empty, its first character is uppercase, and
         tok does not start with the GPT-2 or SentencePiece marker
                                         -> "NOUN"   (likely proper noun)
      6. t ends with any suffix in _VERB_SUF -> "VERB"
      7. t ends with any suffix in _ADJ_SUF  -> "ADJ"
      8. otherwise                       -> "NOUN"

    Note that rule 5 inspects the raw token while every other rule inspects
    the normalised one, and that rule 6 is tried before rule 7.
    """
''',
        "reference": r'''
_FUNC = {
    "the","a","an","of","to","in","and","or","is","are","was","were","be",
    "for","on","at","by","with","from","that","this","it","he","she","they",
    "we","you","i","as","which","who","what","when","where","its","their",
    "our","your","my","his","her","also","been","into","out","up","about",
    "after","before","over","under","between","through","not","no","so",
    "if","then","than",
}
_VERB_SUF = ("ing", "ed", "ify", "ize", "ise", "ate", "ify")
_ADJ_SUF  = ("al", "ful", "less", "ous", "ive", "ish", "able", "ible", "ic", "ary")
_NEG      = {"not", "no", "never", "neither", "nor", "without", "nothing", "nobody"}

_STRIP = " \t\n▁Ġ_"

def heuristic_pos(tok: str) -> str:
    t = tok.strip(_STRIP).lower()
    if not t:
        return "OTHER"
    if not t[0].isalpha():
        return "OTHER"
    if t in _NEG:
        return "FUNC"
    if t in _FUNC:
        return "FUNC"
    if tok and tok[0].isupper() and not tok.startswith(("Ġ", "▁")):
        return "NOUN"
    if any(t.endswith(s) for s in _VERB_SUF):
        return "VERB"
    if any(t.endswith(s) for s in _ADJ_SUF):
        return "ADJ"
    return "NOUN"
''',
        "tests": [
            "G = '\\u0120'; SP = '\\u2581'",
            "assert heuristic_pos(G + 'the') == 'FUNC'",
            "assert heuristic_pos(SP + 'the') == 'FUNC'",
            "assert heuristic_pos('never') == 'FUNC'",
            "assert heuristic_pos('nobody') == 'FUNC'",
            "assert heuristic_pos('cat') == 'NOUN'",
            "assert heuristic_pos('running') == 'VERB'",
            "assert heuristic_pos('beautiful') == 'ADJ'",
            "# TRAP: rule 5 reads the RAW token, so a marker prefix disables the",
            "# proper-noun branch and the word falls through to the suffix rules.",
            "assert heuristic_pos('Created') == 'NOUN'",
            "assert heuristic_pos(G + 'Created') == 'VERB'",
            "assert heuristic_pos(SP + 'Created') == 'VERB'",
            "# TRAP: proper-noun (rule 5) beats the suffix rules",
            "assert heuristic_pos('National') == 'NOUN'",
            "assert heuristic_pos(G + 'national') == 'ADJ'",
            "# TRAP: VERB suffixes are tested before ADJ suffixes",
            "assert heuristic_pos(G + 'automate') == 'VERB'",
            "assert heuristic_pos(G + 'radical') == 'ADJ'",
            "# OTHER cases",
            "assert heuristic_pos('123') == 'OTHER'",
            "assert heuristic_pos('   ') == 'OTHER'",
            "assert heuristic_pos('_') == 'OTHER'",
            "assert heuristic_pos(G) == 'OTHER'",
            "assert heuristic_pos('') == 'OTHER'",
            "assert heuristic_pos('!!') == 'OTHER'",
            "# _NEG entries that are not in _FUNC still map to FUNC",
            "assert heuristic_pos(SP + 'without') == 'FUNC'",
            "assert heuristic_pos(SP + 'neither') == 'FUNC'",
            "# every result is one of the five labels",
            "for w in ['Xyz', G + 'jumped', SP + 'happiness', '42a', 'a', G + 'ary']:",
            "    assert heuristic_pos(w) in {'FUNC', 'NOUN', 'VERB', 'ADJ', 'OTHER'}, w",
        ],
    },

    {
        "id": "rt_m05",
        "name": "drain_acks",
        "tier": "MEDIUM",
        "trap_family": "streaming-boundary",
        "source": "benchmarks/streaming_sender.py:43",
        "prompt": r'''
import struct

def drain_acks(chunks, credit):
    """
    Credit-window bookkeeping for our activation-streaming sender.

    `chunks` is the sequence of byte strings returned by successive
    non-blocking recv() calls. `credit` is the current window credit.

    Each ACK is a 4-byte big-endian record: an unsigned 16-bit type followed
    by an unsigned 16-bit credit amount. Only records whose type field equals
    0x0001 add their amount to the credit; other record types are consumed
    and ignored.

    A chunk is scanned in strict 4-byte steps starting at offset 0. Any
    trailing bytes that cannot form a complete record are DISCARDED -- they
    are not buffered or carried into the next chunk. Scanning of a chunk must
    never read past its end.

    An empty chunk means the peer stopped sending: stop immediately and
    ignore every remaining chunk.

    Returns the updated credit.
    """
''',
        "reference": r'''
import struct

def drain_acks(chunks, credit):
    for raw in chunks:
        if not raw:
            break
        for i in range(0, len(raw) - 3, 4):
            ack_type, c = struct.unpack(">HH", raw[i:i + 4])
            if ack_type == 0x0001:
                credit += c
    return credit
''',
        "tests": [
            "ack = lambda t, c: struct.pack('>HH', t, c)",
            "assert drain_acks([ack(1, 5) + ack(1, 7)], 0) == 12",
            "assert drain_acks([ack(1, 5), ack(1, 7)], 100) == 112",
            "# non-0x0001 records are consumed but contribute nothing",
            "assert drain_acks([ack(2, 99) + ack(1, 3) + ack(0, 50)], 0) == 3",
            "assert drain_acks([], 42) == 42",
            "assert drain_acks([b''], 42) == 42",
            "# TRAP: an empty chunk stops the drain entirely",
            "assert drain_acks([ack(1, 5), b'', ack(1, 5)], 0) == 5",
            "# TRAP: a split record is DISCARDED, not buffered across chunks.",
            "# Buffering would recover an extra ACK worth 4 and return 9.",
            "split = ack(1, 5) + b'\\x00\\x01\\x00'",
            "assert drain_acks([split, b'\\x04' + ack(1, 9)], 0) == 0 + 5",
            "# a short chunk must not raise and must not read past the end",
            "assert drain_acks([b'\\x00\\x01\\x00'], 7) == 7",
            "assert drain_acks([b''.join([b'\\x00'])], 7) == 7",
            "assert drain_acks([ack(1, 2) + b'\\x00'], 0) == 2",
            "# big-endian is load-bearing: little-endian would read type 256",
            "assert drain_acks([struct.pack('<HH', 1, 5)], 0) == 0",
            "# 16-bit fields",
            "assert drain_acks([ack(1, 65535)], 0) == 65535",
            "# many records in one chunk",
            "assert drain_acks([b''.join(ack(1, 1) for _ in range(64))], 0) == 64",
        ],
    },

    {
        "id": "rt_m06",
        "name": "find_shared_positions",
        "tier": "MEDIUM",
        "trap_family": "tie-handling",
        "source": "data/ioi/gen_cross_model_comparison.py:65",
        "prompt": r'''
def find_shared_positions(heads_a, heads_b, tol=0.08):
    """
    Pair up two models' top-10 circuit heads by relative layer depth, for the
    cross-model depth-conservation comparison.

    heads_a / heads_b are lists of 4-tuples (score, layer, head, rel_depth);
    only element 3 (rel_depth) is used for matching.

    Walk heads_a IN ORDER. For each entry, choose the still-unused entry of
    heads_b whose rel_depth is closest, provided the absolute difference is
    <= tol. This is a GREEDY, first-come-first-served assignment, not a
    globally optimal one: an earlier entry of heads_a may consume the partner
    a later entry needed. Exact distance ties keep the EARLIER heads_b entry.

    Returns (shared, only_a, only_b):
      shared : list of (entry_a, entry_b) pairs, in heads_a order
      only_a : heads_a entries that found no partner, in order
      only_b : heads_b entries never consumed, in heads_b order
    """
''',
        "reference": r'''
def find_shared_positions(heads_a, heads_b, tol=0.08):
    used_b = set()
    shared, only_a = [], []
    for entry_a in heads_a:
        match = None
        best_d = tol + 1
        for j, entry_b in enumerate(heads_b):
            if j in used_b:
                continue
            d = abs(entry_a[3] - entry_b[3])
            if d <= tol and d < best_d:
                best_d = d
                match = j
        if match is not None:
            shared.append((entry_a, heads_b[match]))
            used_b.add(match)
        else:
            only_a.append(entry_a)
    only_b = [heads_b[j] for j in range(len(heads_b)) if j not in used_b]
    return shared, only_a, only_b
''',
        "tests": [
            "E = lambda rel, tag: (0.5, tag, 0, rel)",
            "# TRAP: greedy in heads_a order. a0 is farther from b0 than a1 is, but",
            "# a0 goes first and takes it, leaving a1 unmatched. A globally optimal",
            "# assignment would pair a1 with b0 instead.",
            "a = [E(0.55, 'a0'), E(0.51, 'a1')]",
            "b = [E(0.51, 'b0')]",
            "sh, oa, ob = find_shared_positions(a, b, tol=0.08)",
            "assert len(sh) == 1 and sh[0][0][1] == 'a0' and sh[0][1][1] == 'b0'",
            "assert [e[1] for e in oa] == ['a1']",
            "assert ob == []",
            "# TRAP: greedy loses a pairing an optimal matcher would keep. a0 takes",
            "# its NEAREST partner b0, but a0 could have used b1 while a1 cannot.",
            "a2 = [E(0.50, 'a0'), E(0.49, 'a1')]",
            "b2 = [E(0.51, 'b0'), E(0.57, 'b1')]",
            "sh2, oa2, ob2 = find_shared_positions(a2, b2, tol=0.075)",
            "assert len(sh2) == 1, sh2",
            "assert sh2[0][0][1] == 'a0' and sh2[0][1][1] == 'b0'",
            "assert [e[1] for e in oa2] == ['a1'] and [e[1] for e in ob2] == ['b1']",
            "# TRAP: exact distance tie keeps the EARLIER heads_b entry",
            "a3 = [E(0.5, 'a0')]",
            "b3 = [E(0.5625, 'b0'), E(0.4375, 'b1')]",
            "sh3, _, ob3 = find_shared_positions(a3, b3, tol=0.08)",
            "assert sh3[0][1][1] == 'b0'",
            "assert [e[1] for e in ob3] == ['b1']",
            "# tol boundary is inclusive",
            "assert len(find_shared_positions([E(0.5, 'x')], [E(0.625, 'y')], tol=0.125)[0]) == 1",
            "assert len(find_shared_positions([E(0.5, 'x')], [E(0.626, 'y')], tol=0.125)[0]) == 0",
            "# each heads_b entry is consumed at most once",
            "a4 = [E(0.50, 'a0'), E(0.50, 'a1'), E(0.50, 'a2')]",
            "b4 = [E(0.50, 'b0'), E(0.50, 'b1')]",
            "sh4, oa4, ob4 = find_shared_positions(a4, b4)",
            "assert len(sh4) == 2 and [e[1] for e in oa4] == ['a2'] and ob4 == []",
            "assert [p[1][1] for p in sh4] == ['b0', 'b1']",
            "# everything is accounted for",
            "assert len(sh4) + len(oa4) == len(a4)",
            "assert len(sh4) + len(ob4) == len(b4)",
            "# empty inputs",
            "assert find_shared_positions([], [E(0.5, 'b')]) == ([], [], [E(0.5, 'b')])",
            "assert find_shared_positions([E(0.5, 'a')], []) == ([], [E(0.5, 'a')], [])",
        ],
    },

    {
        "id": "rt_m07",
        "name": "detection_rate",
        "tier": "MEDIUM",
        "trap_family": "index-space",
        "source": "scripts/power_analysis_universality.py:155",
        "prompt": r'''
import numpy as np

def detection_rate(fp_a, fp_b, planted_a, planted_b, tau):
    """
    Statistical-power statistic: what fraction of K planted cross-model
    feature pairs would our reciprocal-best-match pipeline actually recover?

    fp_a      : (D_a, N) fingerprints, rows unit-norm
    fp_b      : (D_b, N)
    planted_a : (K,) integer indices into fp_a
    planted_b : (K,) integer indices into fp_b -- planted_a[i] was built to
                correspond to planted_b[i]
    tau       : similarity threshold

    A planted pair i counts as recovered only when ALL THREE hold:
      * the argmax over ALL rows of fp_b for query fp_a[planted_a[i]]
        is exactly planted_b[i]
      * the argmax over ALL rows of fp_a for query fp_b[planted_b[i]]
        is exactly planted_a[i]
      * the similarity measured AT THE INTENDED TARGET, i.e.
        fp_a[planted_a[i]] . fp_b[planted_b[i]], is >= tau

    Note the similarity is indexed at planted_b[i], not at position i of the
    query block, and both directions must be checked.

    Returns the fraction recovered, as a float.
    """
''',
        "reference": r'''
import numpy as np

def detection_rate(fp_a, fp_b, planted_a, planted_b, tau):
    planted_a = np.asarray(planted_a)
    planted_b = np.asarray(planted_b)
    K = len(planted_a)
    sims_a = fp_a[planted_a] @ fp_b.T
    best_b = sims_a.argmax(axis=1)
    sim_tgt = sims_a[np.arange(K), planted_b]
    sims_b = fp_b[planted_b] @ fp_a.T
    best_a = sims_b.argmax(axis=1)
    mutual = (best_b == planted_b) & (best_a == planted_a) & (sim_tgt >= tau)
    return float(mutual.mean())
''',
        "tests": [
            "I = np.eye(4)",
            "fp_a = I.copy()",
            "fp_b = np.stack([I[2], I[3], I[0], I[1]])",
            "pa = np.array([0, 1]); pb = np.array([2, 3])",
            "# TRAP: the similarity must be read at planted_b, not on the diagonal.",
            "# sims_a[arange(K), arange(K)] here is [0, 0] and would score 0.0.",
            "assert abs(detection_rate(fp_a, fp_b, pa, pb, 0.9) - 1.0) < 1e-12",
            "assert abs(detection_rate(fp_a, fp_b, pa, pb, 1.0001) - 0.0) < 1e-12",
            "# argmax must be taken over ALL rows, so a wrong correspondence fails",
            "assert abs(detection_rate(fp_a, I.copy(), pa, pb, 0.5) - 0.0) < 1e-12",
            "# TRAP: both directions are required. Here A->B picks the intended",
            "# target with sim 0.99, but B->A prefers a different row, so the pair",
            "# is NOT recovered. Checking only A->B would return 1.0.",
            "a0 = np.array([1.0, 0.0])",
            "a1 = np.array([0.98, 0.19899]); a1 = a1 / np.linalg.norm(a1)",
            "b0 = np.array([0.99, 0.141067]); b0 = b0 / np.linalg.norm(b0)",
            "b1 = np.array([0.0, 1.0])",
            "FA = np.stack([a0, a1]); FB = np.stack([b0, b1])",
            "sa = FA[[0]] @ FB.T",
            "assert int(sa.argmax(axis=1)[0]) == 0",
            "assert abs(detection_rate(FA, FB, np.array([0]), np.array([0]), 0.5) - 0.0) < 1e-12",
            "# partial recovery averages over K",
            "fp_b2 = np.stack([I[2], I[1], I[0], I[1]])",
            "r = detection_rate(fp_a, fp_b2, np.array([0, 1]), np.array([2, 3]), 0.5)",
            "assert abs(r - 0.5) < 1e-12, r",
            "assert isinstance(detection_rate(fp_a, fp_b, pa, pb, 0.9), float)",
            "# lists are acceptable index inputs",
            "assert abs(detection_rate(fp_a, fp_b, [0, 1], [2, 3], 0.9) - 1.0) < 1e-12",
        ],
    },

    {
        "id": "rt_m08",
        "name": "DeadTracker",
        "tier": "MEDIUM",
        "trap_family": "streaming-boundary",
        "source": "data/sae-runs/train_sae.py:74",
        "prompt": r'''
import numpy as np

class DeadTracker:
    """
    Sliding-window dead-latent counter for SAE training. A latent is DEAD when
    it has not fired in any of the last `window` logged batches.
    """

    def __init__(self, dict_size: int, window: int = 25):
        """Store the window size and initialise the counters and history."""

    def update(self, acts_batch: np.ndarray) -> None:
        """
        acts_batch : (B, dict_size) float array of sparse SAE activations.

        A latent counts as having fired in this batch when ANY row of the
        batch is NON-ZERO for it. The autoencoder has no ReLU, so a negative
        activation counts as firing just as much as a positive one.

        Maintain the counts incrementally: append this batch's fired flags to
        the history and add them to the running counts; once the history is
        LONGER than `window`, pop the oldest entry and subtract it back out.
        """

    @property
    def dead_count(self) -> int:
        """Number of latents whose running count is zero, as a Python int."""

    def dead_mask_f32(self) -> np.ndarray:
        """
        (dict_size,) float32 array: 1.0 where the latent is dead, 0.0 where
        it is alive.
        """
''',
        "reference": r'''
import numpy as np

class DeadTracker:
    def __init__(self, dict_size: int, window: int = 25):
        self.window = window
        self._counts = np.zeros(dict_size, dtype=np.int32)
        self._history = []

    def update(self, acts_batch: np.ndarray) -> None:
        fired = (acts_batch != 0).any(axis=0).astype(np.int8)
        self._history.append(fired)
        self._counts += fired
        if len(self._history) > self.window:
            self._counts -= self._history.pop(0)

    @property
    def dead_count(self) -> int:
        return int((self._counts == 0).sum())

    def dead_mask_f32(self) -> np.ndarray:
        return (self._counts == 0).astype(np.float32)
''',
        "tests": [
            "t = DeadTracker(4, window=2)",
            "assert t.dead_count == 4",
            "# TRAP: no ReLU upstream -- a NEGATIVE activation still counts as fired",
            "t.update(np.array([[-1.0, 0.0, 0.0, 0.0]]))",
            "assert t.dead_count == 3, t.dead_count",
            "assert t.dead_mask_f32().tolist() == [0.0, 1.0, 1.0, 1.0]",
            "t.update(np.zeros((1, 4)))",
            "assert t.dead_count == 3",
            "# the third update evicts the first, so latent 0 goes dead again",
            "t.update(np.zeros((1, 4)))",
            "assert t.dead_count == 4",
            "m = t.dead_mask_f32()",
            "assert m.dtype == np.float32 and m.tolist() == [1.0, 1.0, 1.0, 1.0]",
            "assert isinstance(t.dead_count, int) and not isinstance(t.dead_count, bool)",
            "# firing is an OR across the whole batch, not a per-row test",
            "u = DeadTracker(3, window=5)",
            "u.update(np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]]))",
            "assert u.dead_count == 2",
            "assert u.dead_mask_f32().tolist() == [1.0, 0.0, 1.0]",
            "# window of 1: only the most recent batch matters",
            "w = DeadTracker(2, window=1)",
            "w.update(np.array([[1.0, 0.0]]))",
            "assert w.dead_count == 1",
            "w.update(np.array([[0.0, 1.0]]))",
            "assert w.dead_count == 1 and w.dead_mask_f32().tolist() == [1.0, 0.0]",
            "# long run: counts must stay bounded by the window, not grow forever",
            "z = DeadTracker(2, window=3)",
            "for _ in range(50):",
            "    z.update(np.array([[1.0, 0.0]]))",
            "assert z.dead_count == 1",
            "for _ in range(3):",
            "    z.update(np.zeros((1, 2)))",
            "assert z.dead_count == 2",
        ],
    },

    # =======================================================================
    # HARD
    # =======================================================================

    {
        "id": "rt_h01",
        "name": "sae_eval",
        "tier": "HARD",
        "trap_family": "streaming-boundary",
        "source": "data/sae-analysis/recompute_corrections.py:58",
        "fixture": True,
        "prompt": r'''
import numpy as np

def sae_eval(sae, acts, k, lo, hi, block=10000):
    """
    Streamed reconstruction statistics for a TopK sparse autoencoder over the
    token slice acts[lo:hi]. The real corpus does not fit in memory, so every
    statistic must be accumulated from running sums over blocks of at most
    `block` tokens and must NOT depend on the block size.

    `sae` is a dict with "W_enc" (d_in, D), "b_enc" (D,), "W_dec" (D, d_in)
    and "b_dec" (d_in,). Encoding uses the tied pre-encoder bias and a >= gate
    against the row's k-th largest pre-activation, with no ReLU:

        pre  = (x - b_dec) @ W_enc + b_enc
        f    = pre where pre >= kth_largest(pre, k) per row, else 0
        xhat = f @ W_dec + b_dec

    Accumulate, over the slice:
        se    = sum of squared residuals      (residual r = x - xhat)
        s_err = sum of residuals
        sx    = sum of x
        sxx   = sum of x squared
        cnt   = number of ELEMENTS seen (tokens * d_in)

    Then, per ELEMENT (not per token):
        mse     = se / cnt
        var_res = se / cnt - (s_err / cnt) ** 2
        var_x   = sxx / cnt - (sx / cnt) ** 2
        fve     = 1 - var_res / (var_x + 1e-8)

    A dictionary feature counts as having fired when its gated activation is
    strictly greater than zero.

    Returns a dict:
      "n_tokens"   int, hi - lo
      "mse"        float, rounded to 6
      "fve"        float, rounded to 4
      "dead_features"  int, features that never fired on the slice
      "dead_rate"  float, rounded to 4
      "mean_l0"    float, rounded to 2 -- total firings divided by the number
                   of TOKENS in the slice (note the different denominator)
      "features_firing_over_50pct_of_tokens"  int
      "top200_share_of_activation_mass"  float, rounded to 4 -- the 200
                   most-frequently-firing features' share of all firings
    """
''',
        "reference": r'''
import numpy as np

def sae_eval(sae, acts, k, lo, hi, block=10000):
    W_enc = sae["W_enc"].astype(np.float32)
    b_enc = sae["b_enc"].astype(np.float32)
    W_dec = sae["W_dec"].astype(np.float32)
    b_dec = sae["b_dec"].astype(np.float32)
    D = W_enc.shape[1]

    n = hi - lo
    se = 0.0
    s_err = 0.0
    sx = 0.0
    sxx = 0.0
    cnt = 0
    fired = np.zeros(D, dtype=bool)
    fire_counts = np.zeros(D, dtype=np.int64)

    for s in range(lo, hi, block):
        e = min(s + block, hi)
        x = np.asarray(acts[s:e], dtype=np.float32)
        pre = (x - b_dec) @ W_enc + b_enc
        kth = np.partition(pre, -k, axis=1)[:, -k][:, None]
        f = np.where(pre >= kth, pre, 0.0)
        xh = f @ W_dec + b_dec
        r = x - xh
        se += float((r * r).sum())
        s_err += float(r.sum())
        sx += float(x.sum())
        sxx += float((x * x).sum())
        cnt += x.size
        act = f > 0
        fired |= act.any(axis=0)
        fire_counts += act.sum(axis=0)

    mse = se / cnt
    var_res = se / cnt - (s_err / cnt) ** 2
    var_x = sxx / cnt - (sx / cnt) ** 2
    return {
        "n_tokens": int(n),
        "mse": round(mse, 6),
        "fve": round(float(1.0 - var_res / (var_x + 1e-8)), 4),
        "dead_features": int((~fired).sum()),
        "dead_rate": round(float((~fired).mean()), 4),
        "mean_l0": round(float(fire_counts.sum() / n), 2),
        "features_firing_over_50pct_of_tokens": int((fire_counts > 0.5 * n).sum()),
        "top200_share_of_activation_mass": round(
            float(np.sort(fire_counts)[::-1][:200].sum() / max(fire_counts.sum(), 1)), 4),
    }
''',
        "tests": [
            "sae = make_sae_fixture(0)",
            "acts = make_acts(200, 32, seed=2)",
            "K = 8",
            "res = sae_eval(sae, acts, K, 0, 200, block=10000)",
            "assert set(res) == {'n_tokens','mse','fve','dead_features','dead_rate','mean_l0','features_firing_over_50pct_of_tokens','top200_share_of_activation_mass'}",
            "assert res['n_tokens'] == 200",
            "# independent whole-slice reference",
            "x = acts.astype(np.float32)",
            "pre = (x - sae['b_dec']) @ sae['W_enc'] + sae['b_enc']",
            "kth = np.partition(pre, -K, axis=1)[:, -K][:, None]",
            "f = np.where(pre >= kth, pre, 0.0)",
            "xh = f @ sae['W_dec'] + sae['b_dec']",
            "r = (x - xh).astype(np.float64)",
            "x64 = x.astype(np.float64)",
            "exp_mse = float((r * r).sum()) / x.size",
            "exp_var_res = float((r * r).sum()) / x.size - (float(r.sum()) / x.size) ** 2",
            "exp_var_x = float((x64 * x64).sum()) / x.size - (float(x64.sum()) / x.size) ** 2",
            "exp_fve = 1.0 - exp_var_res / (exp_var_x + 1e-8)",
            "assert abs(res['mse'] - exp_mse) < 1e-5, (res['mse'], exp_mse)",
            "assert abs(res['fve'] - exp_fve) < 1e-3, (res['fve'], exp_fve)",
            "# TRAP: mean_l0 is divided by TOKENS while the variances are per",
            "# ELEMENT. Using the element count here would give roughly 8/32.",
            "counts = (f > 0).sum(axis=0)",
            "assert abs(res['mean_l0'] - round(float(counts.sum()) / 200, 2)) < 1e-9",
            "assert res['mean_l0'] > 1.0",
            "assert res['mean_l0'] <= float(K) + 1e-9",
            "assert res['dead_features'] == int((counts == 0).sum())",
            "assert 0 < res['dead_features'] < 256",
            "assert abs(res['dead_rate'] - round(float((counts == 0).mean()), 4)) < 1e-9",
            "assert res['features_firing_over_50pct_of_tokens'] == int((counts > 0.5 * 200).sum())",
            "share = float(np.sort(counts)[::-1][:200].sum()) / max(int(counts.sum()), 1)",
            "assert abs(res['top200_share_of_activation_mass'] - round(share, 4)) < 1e-9",
            "assert 0.0 < res['top200_share_of_activation_mass'] <= 1.0",
            "# TRAP: block size must not change the answer. A per-block average of",
            "# the statistics would drift once the final block is short (200 = 28*7 + 4).",
            "r7 = sae_eval(sae, acts, K, 0, 200, block=7)",
            "r1 = sae_eval(sae, acts, K, 0, 200, block=1)",
            "for key in ('mse', 'fve', 'mean_l0', 'dead_features', 'n_tokens'):",
            "    assert abs(r7[key] - res[key]) < 1e-4, (key, r7[key], res[key])",
            "    assert abs(r1[key] - res[key]) < 1e-4, (key, r1[key], res[key])",
            "# held-out style slice: lo is respected, not silently treated as 0",
            "half = sae_eval(sae, acts, K, 100, 200, block=13)",
            "assert half['n_tokens'] == 100",
            "assert half['dead_features'] >= res['dead_features']",
            "assert abs(half['mse'] - res['mse']) > 1e-12",
            "assert isinstance(res['dead_features'], int) and isinstance(res['n_tokens'], int)",
        ],
    },

    {
        "id": "rt_h02",
        "name": "plant_signals_batch",
        "tier": "HARD",
        "trap_family": "order-of-operations",
        "source": "scripts/power_analysis_universality.py:125",
        "prompt": r'''
import numpy as np

def plant_signals_batch(templates: np.ndarray, alpha: float,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Synthesise planted "universal" SAE feature fingerprints for a statistical
    power analysis.

    `templates` is (K, N) and every row is already ZERO-MEAN and UNIT-NORM,
    so the dot product of two such rows is their Pearson correlation.

    Return a (K, N) array in which row i has Pearson correlation exactly
    `alpha` with templates[i], built as follows:

      1. draw a (K, N) standard normal noise block from `rng`
      2. centre each noise row (subtract its own mean) so it stays inside the
         zero-mean subspace that Pearson correlation lives in
      3. remove the template component from each noise row by subtracting
         (noise_i . template_i) * template_i
      4. normalise each noise row to unit length, guarding against a zero norm
      5. blend:  alpha * template + sqrt(1 - alpha**2) * noise
      6. renormalise each output row to unit length for floating-point safety

    Steps 3 and 4 must happen in that order. Normalising the noise before
    removing the template component leaves a vector that is no longer unit
    length once the component is subtracted, the blend is then renormalised to
    compensate, and the realised correlation comes out above `alpha` instead
    of equal to it.

    Work in float32.
    """
''',
        "reference": r'''
import numpy as np

def plant_signals_batch(templates: np.ndarray, alpha: float,
                        rng: np.random.Generator) -> np.ndarray:
    K, N = templates.shape
    noise = rng.standard_normal((K, N)).astype(np.float32)
    noise -= noise.mean(axis=1, keepdims=True)
    dots = (noise * templates).sum(axis=1, keepdims=True)
    noise -= dots * templates
    norms = np.linalg.norm(noise, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    noise /= norms
    scale = float(np.sqrt(max(0.0, 1.0 - alpha ** 2)))
    fps = alpha * templates + scale * noise
    fps /= np.maximum(np.linalg.norm(fps, axis=1, keepdims=True), 1e-10)
    return fps
''',
        "tests": [
            "def mk_templates(K, N, seed):",
            "    r = np.random.default_rng(seed)",
            "    T = r.standard_normal((K, N)).astype(np.float32)",
            "    T -= T.mean(axis=1, keepdims=True)",
            "    T /= np.linalg.norm(T, axis=1, keepdims=True)",
            "    return T",
            "T = mk_templates(6, 300, 11)",
            "out = plant_signals_batch(T, 0.62, np.random.default_rng(7))",
            "assert out.shape == (6, 300)",
            "# rows are unit-norm and zero-mean",
            "assert np.abs(np.linalg.norm(out, axis=1) - 1.0).max() < 1e-4",
            "assert np.abs(out.mean(axis=1)).max() < 1e-5",
            "# THE oracle: realised Pearson correlation must equal alpha, and it is",
            "# EXACT for a correct implementation -- float32 noise still lands",
            "# within ~3e-8, so the tolerance here is deliberately tight.",
            "for i in range(6):",
            "    rr = float(np.corrcoef(out[i].astype(np.float64), T[i].astype(np.float64))[0, 1])",
            "    assert abs(rr - 0.62) < 1e-5, (i, rr)",
            "# TRAP: orthogonalise BEFORE normalising, and blend with sqrt(1-a**2)",
            "# rather than (1-a). Normalising first leaves the noise slightly short",
            "# once the template component is removed, and the renormalised blend",
            "# then correlates ABOVE alpha by roughly alpha*(1-alpha**2)/(2N).",
            "for a in (0.1, 0.35, 0.9, 0.99):",
            "    o = plant_signals_batch(T, a, np.random.default_rng(3))",
            "    for i in range(6):",
            "        rr = float(np.corrcoef(o[i].astype(np.float64), T[i].astype(np.float64))[0, 1])",
            "        assert abs(rr - a) < 1e-5, (a, i, rr)",
            "# a short fingerprint makes any residual template leakage far larger",
            "Tn = mk_templates(6, 60, 21)",
            "for a in (0.35, 0.62, 0.9):",
            "    o = plant_signals_batch(Tn, a, np.random.default_rng(13))",
            "    assert np.abs(np.linalg.norm(o, axis=1) - 1.0).max() < 1e-4",
            "    for i in range(6):",
            "        rr = float(np.corrcoef(o[i].astype(np.float64), Tn[i].astype(np.float64))[0, 1])",
            "        assert abs(rr - a) < 1e-5, (a, i, rr)",
            "# alpha = 1 collapses onto the template exactly",
            "o1 = plant_signals_batch(T, 1.0, np.random.default_rng(1))",
            "assert np.allclose(o1, T, atol=1e-4)",
            "# alpha = 0 is orthogonal to the template",
            "o0 = plant_signals_batch(T, 0.0, np.random.default_rng(2))",
            "for i in range(6):",
            "    rr = float(np.corrcoef(o0[i].astype(np.float64), T[i].astype(np.float64))[0, 1])",
            "    assert abs(rr) < 1e-5, (i, rr)",
            "# blend weight is sqrt(1 - alpha**2), not (1 - alpha): the two differ",
            "# in how much of the template survives, which the corr checks pin down.",
            "# deterministic given the generator",
            "x1 = plant_signals_batch(T, 0.5, np.random.default_rng(99))",
            "x2 = plant_signals_batch(T, 0.5, np.random.default_rng(99))",
            "assert np.array_equal(x1, x2)",
            "x3 = plant_signals_batch(T, 0.5, np.random.default_rng(100))",
            "assert not np.array_equal(x1, x3)",
            "# distinct rows get distinct noise (vectorised over K, not broadcast)",
            "assert not np.allclose(out[0], out[1])",
        ],
    },

    {
        "id": "rt_h03",
        "name": "procrustes_align",
        "tier": "HARD",
        "trap_family": "order-of-operations",
        "source": "data/sae-analysis/decoder_weight_matching.py:30",
        "prompt": r'''
import numpy as np

def procrustes_align(W_a, W_b, anchors_a, anchors_b):
    """
    Align two SAE decoder spaces of DIFFERENT dimensionality (our Llama SAE
    lives in R^3072, our Mistral SAE in R^4096) using orthogonal Procrustes on
    a set of already-matched anchor features.

    W_a : (n_a, d_a) decoder matrix, rows are feature directions
    W_b : (n_b, d_b) decoder matrix, d_b >= d_a
    anchors_a, anchors_b : equal-length index sequences; anchors_a[i] in W_a
        corresponds to anchors_b[i] in W_b

    Procedure:
      1. L2-normalise the ROWS of both matrices first, so that anchors with
         large raw norms cannot dominate the fit
      2. form the cross-covariance M = A.T @ B of the normalised anchor rows,
         computing in float64
      3. take the thin SVD M = U S Vt (full_matrices=False) and DISCARD the
         singular values: R = U @ Vt, cast back to float32
      4. project every normalised row of W_a through R and L2-normalise again

    R is (d_a, d_b) and semi-orthogonal: R @ R.T is the (d_a, d_a) identity,
    while R.T @ R is not the (d_b, d_b) identity.

    Returns (R, W_a_aligned) where W_a_aligned is (n_a, d_b) with unit rows.
    """
''',
        "reference": r'''
import numpy as np

def _l2(W):
    return W / np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-12)

def procrustes_align(W_a, W_b, anchors_a, anchors_b):
    Wa = _l2(np.asarray(W_a, dtype=np.float32))
    Wb = _l2(np.asarray(W_b, dtype=np.float32))
    ia = np.asarray(anchors_a, dtype=np.int64)
    ib = np.asarray(anchors_b, dtype=np.int64)
    A = Wa[ia].astype(np.float64)
    B = Wb[ib].astype(np.float64)
    M = A.T @ B
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    R = (U @ Vt).astype(np.float32)
    W_a_aligned = _l2(Wa @ R)
    return R, W_a_aligned
''',
        "tests": [
            "rng = np.random.default_rng(3)",
            "d_a, d_b, n_a, n_b = 6, 10, 40, 30",
            "Qf, _ = np.linalg.qr(rng.standard_normal((d_b, d_a)))",
            "Q = Qf.T.astype(np.float32)",
            "assert np.allclose(Q @ Q.T, np.eye(d_a), atol=1e-6)",
            "# TRAP: wildly different row norms. Skipping the pre-normalisation lets",
            "# the largest anchors dominate M and R comes out wrong.",
            "raw = rng.standard_normal((n_a, d_a)).astype(np.float32)",
            "raw = raw * rng.uniform(0.05, 200.0, (n_a, 1)).astype(np.float32)",
            "Wa_unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)",
            "W_b = rng.standard_normal((n_b, d_b)).astype(np.float32)",
            "anchors_a = list(range(20))",
            "anchors_b = list(range(5, 25))",
            "W_b[5:25] = (Wa_unit[:20] @ Q).astype(np.float32)",
            "R, Wal = procrustes_align(raw, W_b, anchors_a, anchors_b)",
            "assert R.shape == (d_a, d_b) and Wal.shape == (n_a, d_b)",
            "assert R.dtype == np.float32",
            "# the planted semi-orthogonal map must be recovered",
            "assert np.allclose(R, Q, atol=2e-4), float(np.abs(R - Q).max())",
            "# semi-orthogonality holds one way only",
            "assert np.allclose(R @ R.T, np.eye(d_a), atol=1e-4)",
            "assert not np.allclose(R.T @ R, np.eye(d_b), atol=1e-2)",
            "assert abs(float(np.trace(R.T @ R)) - d_a) < 1e-2",
            "# output rows are unit norm",
            "assert np.abs(np.linalg.norm(Wal, axis=1) - 1.0).max() < 1e-4",
            "# anchor rows land on their partners after alignment",
            "Wb_unit = W_b / np.linalg.norm(W_b, axis=1, keepdims=True)",
            "cos = (Wal[:20] * Wb_unit[5:25]).sum(axis=1)",
            "assert cos.min() > 0.999, float(cos.min())",
            "# TRAP: discarding the singular values is what makes R orthogonal.",
            "# Keeping S (R = U @ diag(S) @ Vt) breaks R @ R.T == I.",
            "A64 = Wa_unit[:20].astype(np.float64)",
            "B64 = Wb_unit[5:25].astype(np.float64)",
            "U, S, Vt = np.linalg.svd(A64.T @ B64, full_matrices=False)",
            "R_withS = (U @ np.diag(S) @ Vt).astype(np.float32)",
            "assert not np.allclose(R_withS @ R_withS.T, np.eye(d_a), atol=1e-3)",
            "assert not np.allclose(R, R_withS, atol=1e-3)",
        ],
    },

    {
        "id": "rt_h04",
        "name": "find_universal_triangles",
        "tier": "HARD",
        "trap_family": "index-space",
        "source": "data/sae-analysis/cross_arch_matching_v3.py:194",
        "prompt": r'''
def find_universal_triangles(llama_qwen, llama_mistral, mistral_qwen):
    """
    Identify SAE features that are universal across three model architectures:
    a feature triple (l, q, m) counts only when ALL THREE pairwise edges are
    present, i.e. the triangle is CLOSED.

    Each argument is a list of (first, second, sim) tuples, and the id
    ordering inside a tuple differs per list:
      llama_qwen    : (llama_feat, qwen_feat, sim)
      llama_mistral : (llama_feat, mistral_feat, sim)
      mistral_qwen  : (mistral_feat, qwen_feat, sim)

    Walk the llama->qwen edges. Keep a triple when the same llama feature also
    has a llama->mistral edge AND that mistral feature's own mistral->qwen
    partner is the very same qwen feature.

    Return a list of dicts, each with exactly these keys:
      "llama_feat", "qwen_feat", "mistral_feat",
      "cosim_llama_qwen", "cosim_llama_mistral", "cosim_mistral_qwen"
    sorted by DESCENDING mean of the three similarities.

    Finally, assert that each of "llama_feat", "qwen_feat" and "mistral_feat"
    is one-to-one across the returned triples; raise AssertionError otherwise.
    """
''',
        "reference": r'''
def find_universal_triangles(llama_qwen, llama_mistral, mistral_qwen):
    lq_by_l = {a: (b, s) for a, b, s in llama_qwen}
    lm_by_l = {a: (b, s) for a, b, s in llama_mistral}
    mq_by_m = {a: (b, s) for a, b, s in mistral_qwen}

    universal = []
    for lf, (qf, s_lq) in lq_by_l.items():
        if lf not in lm_by_l:
            continue
        mf, s_lm = lm_by_l[lf]
        if mq_by_m.get(mf, (None, None))[0] != qf:
            continue
        universal.append({
            "llama_feat": lf, "qwen_feat": qf, "mistral_feat": mf,
            "cosim_llama_qwen": s_lq, "cosim_llama_mistral": s_lm,
            "cosim_mistral_qwen": mq_by_m[mf][1],
        })
    universal.sort(key=lambda e: -(e["cosim_llama_qwen"] + e["cosim_llama_mistral"]
                                  + e["cosim_mistral_qwen"]) / 3)
    for k in ("llama_feat", "qwen_feat", "mistral_feat"):
        assert len({u[k] for u in universal}) == len(universal), k + " not one-to-one"
    return universal
''',
        "tests": [
            "# TRAP: mistral_qwen is keyed by the MISTRAL id, and its second element",
            "# is the qwen id. Keying it the other way round silently returns [] --",
            "# which is also the correct answer on real data, so the bug is invisible",
            "# unless a triangle is deliberately planted.",
            "r = find_universal_triangles([(1, 10, 0.9)], [(1, 100, 0.8)], [(100, 10, 0.7)])",
            "assert len(r) == 1, r",
            "assert r[0]['llama_feat'] == 1 and r[0]['qwen_feat'] == 10 and r[0]['mistral_feat'] == 100",
            "assert abs(r[0]['cosim_llama_qwen'] - 0.9) < 1e-12",
            "assert abs(r[0]['cosim_llama_mistral'] - 0.8) < 1e-12",
            "assert abs(r[0]['cosim_mistral_qwen'] - 0.7) < 1e-12",
            "assert set(r[0]) == {'llama_feat','qwen_feat','mistral_feat','cosim_llama_qwen','cosim_llama_mistral','cosim_mistral_qwen'}",
            "# an OPEN triangle (third edge points at a different qwen feature)",
            "assert find_universal_triangles([(1, 10, 0.9)], [(1, 100, 0.8)], [(100, 11, 0.7)]) == []",
            "# a missing llama->mistral edge",
            "assert find_universal_triangles([(1, 10, 0.9)], [], [(100, 10, 0.7)]) == []",
            "# a missing mistral->qwen edge",
            "assert find_universal_triangles([(1, 10, 0.9)], [(1, 100, 0.8)], []) == []",
            "assert find_universal_triangles([], [], []) == []",
            "# sort is by DESCENDING mean of the three similarities",
            "lq = [(1, 10, 0.50), (2, 20, 0.95), (3, 30, 0.70)]",
            "lm = [(1, 100, 0.50), (2, 200, 0.95), (3, 300, 0.70)]",
            "mq = [(100, 10, 0.50), (200, 20, 0.95), (300, 30, 0.70)]",
            "out = find_universal_triangles(lq, lm, mq)",
            "assert [u['llama_feat'] for u in out] == [2, 3, 1], out",
            "# a low-mean triple with one huge edge must still sort by the MEAN",
            "lq2 = [(1, 10, 0.99), (2, 20, 0.60)]",
            "lm2 = [(1, 100, 0.10), (2, 200, 0.60)]",
            "mq2 = [(100, 10, 0.10), (200, 20, 0.60)]",
            "out2 = find_universal_triangles(lq2, lm2, mq2)",
            "assert [u['llama_feat'] for u in out2] == [2, 1], out2",
            "# only closed triangles survive a mixed input",
            "lq3 = [(1, 10, 0.9), (2, 11, 0.9), (3, 12, 0.9)]",
            "lm3 = [(1, 100, 0.9), (2, 200, 0.9)]",
            "mq3 = [(100, 10, 0.9), (200, 99, 0.9)]",
            "out3 = find_universal_triangles(lq3, lm3, mq3)",
            "assert [u['llama_feat'] for u in out3] == [1], out3",
            "# the one-to-one guarantee is enforced, not assumed",
            "bad_lq = [(1, 10, 0.9), (2, 10, 0.9)]",
            "bad_lm = [(1, 100, 0.9), (2, 200, 0.9)]",
            "bad_mq = [(100, 10, 0.9), (200, 10, 0.9)]",
            "raised = False",
            "try:",
            "    find_universal_triangles(bad_lq, bad_lm, bad_mq)",
            "except AssertionError:",
            "    raised = True",
            "assert raised, 'duplicate qwen_feat must trigger the one-to-one assertion'",
        ],
    },

    {
        "id": "rt_h05",
        "name": "auxk_loss",
        "tier": "HARD",
        "trap_family": "order-of-operations",
        "source": "data/sae-runs/train_sae_aux.py:237",
        "fixture": True,
        "prompt": r'''
import numpy as np

def auxk_loss(x, W_enc, b_enc, W_dec, b_dec, k, dead_mask, k_aux, lambda_aux):
    """
    TopK-SAE reconstruction loss plus the auxiliary dead-latent revival term
    we added to stop a 57% dead-latent rate from suppressing universality
    recall.

    Main pass (tied pre-encoder bias, >= gate on the k-th largest, no ReLU):
        pre   = (x - b_dec) @ W_enc + b_enc
        acts  = pre where pre >= kth_largest(pre, k) per row, else 0
        recon = acts @ W_dec + b_dec
        main  = mean((x - recon) ** 2)          over ALL elements

    dead_mask is a (dict_size,) float array holding 1.0 for dead latents and
    0.0 for alive ones. When the number of dead latents is STRICTLY LESS than
    k_aux, the auxiliary term is skipped entirely and `main` is returned
    unchanged -- the loss is deliberately discontinuous in that respect.

    Otherwise, only DEAD latents may participate in a second top-k_aux pass
    that tries to reconstruct the main pass's residual:
        e         = x - recon
        pre_dead  = pre * dead_mask + (-1e9) * (1 - dead_mask)
        acts_aux  = pre where pre_dead >= kth_largest(pre_dead, k_aux) per row,
                    else 0      (note the VALUES come from pre, not pre_dead)
        aux_recon = acts_aux @ W_dec + b_dec    (the decoder bias is included
                                                 even though e is a residual)
        aux       = mean((e - aux_recon) ** 2)

    Returns the scalar float  main + lambda_aux * aux.
    """
''',
        "reference": r'''
import numpy as np

def auxk_loss(x, W_enc, b_enc, W_dec, b_dec, k, dead_mask, k_aux, lambda_aux):
    pre = (x - b_dec) @ W_enc + b_enc
    kth = np.partition(pre, -k, axis=1)[:, -k][:, None]
    acts = np.where(pre >= kth, pre, 0.0)
    recon = acts @ W_dec + b_dec
    main = float(((x - recon) ** 2).mean())

    n_dead = int(np.asarray(dead_mask).sum())
    if n_dead < k_aux:
        return main

    e = x - recon
    dead = np.asarray(dead_mask, dtype=np.float64)
    pre_dead = pre * dead + (-1e9) * (1.0 - dead)
    kth_a = np.partition(pre_dead, -k_aux, axis=1)[:, -k_aux][:, None]
    acts_aux = np.where(pre_dead >= kth_a, pre, 0.0)
    aux_recon = acts_aux @ W_dec + b_dec
    aux = float(((e - aux_recon) ** 2).mean())
    return main + lambda_aux * aux
''',
        "tests": [
            "sae = make_sae_fixture(0)",
            "W_enc, b_enc = sae['W_enc'].astype(np.float64), sae['b_enc'].astype(np.float64)",
            "W_dec, b_dec = sae['W_dec'].astype(np.float64), sae['b_dec'].astype(np.float64)",
            "x = make_acts(24, 32, seed=5).astype(np.float64)",
            "K, KA = 8, 12",
            "dead = np.zeros(256); dead[100:180] = 1.0",
            "# independent whole-computation reference",
            "pre = (x - b_dec) @ W_enc + b_enc",
            "kth = np.partition(pre, -K, axis=1)[:, -K][:, None]",
            "acts = np.where(pre >= kth, pre, 0.0)",
            "recon = acts @ W_dec + b_dec",
            "main = float(((x - recon) ** 2).mean())",
            "e = x - recon",
            "didx = np.flatnonzero(dead > 0)",
            "sub = pre[:, didx]",
            "kth_sub = np.partition(sub, -KA, axis=1)[:, -KA][:, None]",
            "full = np.zeros_like(pre)",
            "full[:, didx] = np.where(sub >= kth_sub, sub, 0.0)",
            "aux = float(((e - (full @ W_dec + b_dec)) ** 2).mean())",
            "got = auxk_loss(x, W_enc, b_enc, W_dec, b_dec, K, dead, KA, 0.1)",
            "assert abs(got - (main + 0.1 * aux)) < 1e-9, (got, main + 0.1 * aux)",
            "# TRAP: only DEAD latents may fire in the aux pass, however large the",
            "# alive pre-activations are. Restricting to `didx` is the whole point.",
            "assert int((full != 0).sum(axis=1).max()) <= KA",
            "assert np.all(full[:, dead == 0] == 0)",
            "# TRAP: aux_recon adds b_dec even though the target is a residual.",
            "aux_nob = float(((e - (full @ W_dec)) ** 2).mean())",
            "assert abs(aux - aux_nob) > 1e-9",
            "assert abs(got - (main + 0.1 * aux_nob)) > 1e-9",
            "# TRAP: the aux term is skipped entirely while n_dead < k_aux, and the",
            "# comparison is strict, so n_dead == k_aux DOES compute it.",
            "few = np.zeros(256); few[:11] = 1.0",
            "assert abs(auxk_loss(x, W_enc, b_enc, W_dec, b_dec, K, few, 12, 0.1) - main) < 1e-12",
            "exact = np.zeros(256); exact[:12] = 1.0",
            "v = auxk_loss(x, W_enc, b_enc, W_dec, b_dec, K, exact, 12, 0.1)",
            "assert abs(v - main) > 1e-9, 'n_dead == k_aux must NOT skip the aux term'",
            "none_dead = np.zeros(256)",
            "assert abs(auxk_loss(x, W_enc, b_enc, W_dec, b_dec, K, none_dead, 1, 0.1) - main) < 1e-12",
            "# lambda_aux scales only the auxiliary term",
            "assert abs(auxk_loss(x, W_enc, b_enc, W_dec, b_dec, K, dead, KA, 0.0) - main) < 1e-12",
            "g2 = auxk_loss(x, W_enc, b_enc, W_dec, b_dec, K, dead, KA, 0.2)",
            "assert abs((g2 - main) - 2.0 * (got - main)) < 1e-9",
            "# main loss is the elementwise MSE, not a per-token sum",
            "assert abs(main - float(((x - recon) ** 2).sum() / x.size)) < 1e-12",
            "assert isinstance(got, float)",
        ],
    },

    {
        "id": "rt_h06",
        "name": "parse_stream",
        "tier": "HARD",
        "trap_family": "streaming-boundary",
        "source": "benchmarks/streaming_receiver.py:77",
        "prompt": r'''
import struct
import numpy as np

def parse_stream(payload: bytes, hidden_dim: int) -> dict:
    """
    Receiver for our activation-streaming wire protocol.

    Every frame begins with an 8-byte big-endian header laid out as
    ">HIH": an unsigned 16-bit layer index, an unsigned 32-bit token_start,
    and an unsigned 16-bit token_count. There is no length field: the payload
    size is derived from token_count and `hidden_dim`.

    Three frame kinds, and the checks MUST be made in this order:

      1. END OF STREAM -- layer_idx == 0xFFFF AND token_start == 0xFFFFFFFF.
         Stop immediately. Anything after it is ignored. Its token_count field
         is filler and must not be read as a length.

      2. END OF LAYER -- layer_idx == 0xFFFF with any other token_start. Here
         token_start carries the index of the layer that just closed. No
         payload follows; token_count is again filler.

      3. DATA -- token_count * hidden_dim big-endian float16 values follow the
         header immediately. Each layer keeps its OWN token cursor starting at
         0; layers may interleave freely. A data frame whose token_start does
         not equal that layer's current cursor is a protocol violation: raise
         ValueError. Otherwise advance that layer's cursor by token_count.

    Stop cleanly when fewer than 8 bytes remain.

    Returns a dict:
      "totals"   : {layer_idx: total tokens decoded for that layer}
      "closed"   : list of layer indices closed by END OF LAYER, in order
      "eos"      : bool, whether an END OF STREAM frame was seen
      "checksum" : float, the sum of every decoded value, rounded to 4
    """
''',
        "reference": r'''
import struct
import numpy as np

_HDR = struct.Struct(">HIH")

def parse_stream(payload: bytes, hidden_dim: int) -> dict:
    totals = {}
    closed = []
    expected = {}
    eos = False
    checksum = 0.0
    pos = 0
    n = len(payload)
    while pos + 8 <= n:
        layer_idx, token_start, token_count = _HDR.unpack_from(payload, pos)
        pos += 8
        if layer_idx == 0xFFFF and token_start == 0xFFFFFFFF:
            eos = True
            break
        if layer_idx == 0xFFFF:
            closed.append(int(token_start))
            continue
        exp = expected.get(layer_idx, 0)
        if token_start != exp:
            raise ValueError("token_start %d != expected %d for layer %d"
                             % (token_start, exp, layer_idx))
        count = token_count * hidden_dim
        block = np.frombuffer(payload, dtype=">f2", count=count, offset=pos)
        pos += count * 2
        checksum += float(block.astype(np.float64).sum())
        expected[layer_idx] = exp + token_count
        totals[layer_idx] = totals.get(layer_idx, 0) + token_count
    return {"totals": totals, "closed": closed, "eos": eos,
            "checksum": round(checksum, 4)}
''',
        "tests": [
            "assert struct.calcsize('>HIH') == 8",
            "def frame(layer, start, count, arr):",
            "    return struct.pack('>HIH', layer, start, count) + np.asarray(arr, dtype=np.float32).astype('>f2').tobytes()",
            "def eol(layer):",
            "    return struct.pack('>HIH', 0xFFFF, layer, 0xFFFF)",
            "EOS = struct.pack('>HIH', 0xFFFF, 0xFFFFFFFF, 0xFFFF)",
            "a = np.arange(8, dtype=np.float32).reshape(2, 4)",
            "b = np.arange(4, dtype=np.float32).reshape(1, 4)",
            "s = frame(0, 0, 2, a) + frame(3, 0, 1, b) + frame(0, 2, 1, b) + eol(3) + EOS + b'GARBAGE'",
            "r = parse_stream(s, 4)",
            "assert r['totals'] == {0: 3, 3: 1}, r['totals']",
            "assert r['closed'] == [3]",
            "assert r['eos'] is True",
            "# checksum pins BIG-endian float16 decoding: 28 + 6 + 6",
            "assert abs(r['checksum'] - 40.0) < 1e-6, r['checksum']",
            "# TRAP: the EOS test must precede the EOL test. Checking EOL first",
            "# swallows the sentinel as a layer close and never reports eos.",
            "r2 = parse_stream(EOS, 4)",
            "assert r2['eos'] is True and r2['closed'] == [] and r2['totals'] == {}",
            "r3 = parse_stream(frame(1, 0, 1, b) + EOS + frame(1, 1, 1, b), 4)",
            "assert r3['eos'] is True and r3['totals'] == {1: 1} and r3['closed'] == []",
            "# TRAP: sentinel token_count is 0xFFFF filler, never a payload length",
            "assert parse_stream(eol(7) + EOS, 4)['closed'] == [7]",
            "# TRAP: cursors are per layer, so interleaving must not desynchronise",
            "s4 = frame(0, 0, 1, b) + frame(9, 0, 1, b) + frame(0, 1, 1, b) + frame(9, 1, 1, b) + EOS",
            "r4 = parse_stream(s4, 4)",
            "assert r4['totals'] == {0: 2, 9: 2}",
            "assert abs(r4['checksum'] - 24.0) < 1e-6",
            "# a global counter instead of per-layer cursors would raise here",
            "raised = False",
            "try:",
            "    parse_stream(frame(0, 0, 1, b) + frame(0, 5, 1, b), 4)",
            "except ValueError:",
            "    raised = True",
            "assert raised, 'out-of-order token_start must raise ValueError'",
            "# TRAP: little-endian float16 decodes to different numbers entirely",
            "le = struct.pack('>HIH', 0, 0, 1) + b.astype('<f2').tobytes()",
            "assert abs(parse_stream(le, 4)['checksum'] - 6.0) > 1e-3",
            "# truncated tail (fewer than 8 bytes) stops cleanly",
            "r5 = parse_stream(frame(0, 0, 1, b) + b'\\x00\\x01\\x02', 4)",
            "assert r5['totals'] == {0: 1} and r5['eos'] is False",
            "assert parse_stream(b'', 4) == {'totals': {}, 'closed': [], 'eos': False, 'checksum': 0.0}",
            "# negative and fractional values survive the round trip",
            "c = np.array([[-1.5, 0.25, -0.75, 2.0]], dtype=np.float32)",
            "assert abs(parse_stream(frame(2, 0, 1, c) + EOS, 4)['checksum'] - 0.0) < 1e-6",
        ],
    },

    {
        "id": "rt_h07",
        "name": "contrastive_label",
        "tier": "HARD",
        "trap_family": "tie-handling",
        "source": "scripts/extract_rules_v2.py:213",
        "prompt": r'''
from collections import Counter

_STRIP = "▁Ġ .,!?;:'\""

def contrastive_label(high_tokens, low_tokens, min_count=3):
    """
    Label an SAE feature by contrasting the tokens it fires hardest on against
    the tokens it barely fires on.

    high_tokens : token strings from the feature's top-activating positions,
                  ordered by DESCENDING activation
    low_tokens  : token strings from its low-activation positions

    Normalise every token by stripping the characters in _STRIP from both
    ends and lowercasing; drop tokens that normalise to the empty string.

    Score each candidate w that occurs at least `min_count` times among the
    normalised high tokens with this DELIBERATELY ASYMMETRIC smoothing --
    the +0.5 is applied only to the low side, to both its count and its
    total, and there is no vocabulary term anywhere:

        ratio(w) = (count_high(w) / total_high)
                   / ((count_low(w) + 0.5) / (total_low + 0.5))

    where total_high and total_low are the numbers of surviving normalised
    tokens in each list.

    Scan candidates in FIRST-APPEARANCE order within high_tokens and keep a
    candidate only when its ratio is STRICTLY greater than the best so far,
    starting from 0.0. Ties therefore keep the earlier candidate.

    Returns (best_token, best_ratio); (None, 0.0) when nothing qualifies.
    """
''',
        "reference": r'''
from collections import Counter

_STRIP = "▁Ġ .,!?;:'\""

def contrastive_label(high_tokens, low_tokens, min_count=3):
    hi = [t.strip(_STRIP).lower() for t in high_tokens]
    lo = [t.strip(_STRIP).lower() for t in low_tokens]
    hi = [t for t in hi if t]
    lo = [t for t in lo if t]
    ch = Counter(hi)
    cl = Counter(lo)
    total_high = len(hi)
    total_low = len(lo)
    best, best_ratio = None, 0.0
    for w in ch:
        cnt_h = ch[w]
        if cnt_h < min_count:
            continue
        ratio = (cnt_h / total_high) / ((cl[w] + 0.5) / (total_low + 0.5))
        if ratio > best_ratio:
            best, best_ratio = w, ratio
    return best, best_ratio
''',
        "tests": [
            "SP = '\\u2581'",
            "high = [SP + 'the'] * 4 + [SP + 'cat'] * 3 + [SP + 'dog'] * 2",
            "low = [SP + 'the'] * 10 + [SP + 'mouse'] * 10",
            "tok, ratio = contrastive_label(high, low)",
            "assert tok == 'cat', tok",
            "# TRAP: the smoothing is one-sided. Symmetric add-0.5 would give",
            "# (3+0.5)/(9+0.5) / (0.5/20.5) = 15.105..., not 13.666...",
            "assert abs(ratio - (3 / 9) / (0.5 / 20.5)) < 1e-9, ratio",
            "assert abs(ratio - 13.6666666667) < 1e-6, ratio",
            "assert abs(ratio - (3.5 / 9.5) / (0.5 / 20.5)) > 1.0",
            "# raising the gate above cat's support hands the label to 'the'",
            "assert contrastive_label(high, low, min_count=4)[0] == 'the'",
            "assert contrastive_label(high, low, min_count=5) == (None, 0.0)",
            "assert contrastive_label(high, low, min_count=99) == (None, 0.0)",
            "# TRAP: the min_count gate is applied BEFORE scoring, so a token that",
            "# would win by a factor of 130 is silently excluded when it is rare.",
            "h5 = ['q'] * 3 + ['r'] * 2",
            "l5 = ['q'] * 100",
            "t5, r5 = contrastive_label(h5, l5, min_count=3)",
            "assert t5 == 'q' and abs(r5 - 0.6) < 1e-9, (t5, r5)",
            "t6, r6 = contrastive_label(h5, l5, min_count=2)",
            "assert t6 == 'r' and abs(r6 - (2 / 5) / (0.5 / 100.5)) < 1e-9, (t6, r6)",
            "assert r6 > 80.0",
            "# TRAP: strictly-greater comparison keeps the EARLIER candidate on a tie",
            "t1, r1 = contrastive_label(['a'] * 3 + ['b'] * 3, [])",
            "assert t1 == 'a' and abs(r1 - 0.5) < 1e-12, (t1, r1)",
            "t2, r2 = contrastive_label(['b'] * 3 + ['a'] * 3, [])",
            "assert t2 == 'b' and abs(r2 - 0.5) < 1e-12, (t2, r2)",
            "# an empty low list must not divide by zero: (0+0.5)/(0+0.5) == 1",
            "assert contrastive_label(['z'] * 3, []) == ('z', 1.0)",
            "# normalisation folds markers, case and trailing punctuation together",
            "h2 = [SP + 'The', 'the.', 'THE', '\\u0120the']",
            "t3, _ = contrastive_label(h2, ['q'] * 5, min_count=4)",
            "assert t3 == 'the', t3",
            "# tokens that normalise to nothing are dropped from BOTH totals",
            "h3 = [SP, ' ', '...'] + ['w'] * 3",
            "t4, r4 = contrastive_label(h3, [])",
            "assert t4 == 'w'",
            "assert abs(r4 - (3 / 3) / (0.5 / 0.5)) < 1e-12, r4",
            "assert contrastive_label([], ['a']) == (None, 0.0)",
            "assert contrastive_label([], []) == (None, 0.0)",
            "# the rarer, more specific token wins over the frequent generic one",
            "h4 = ['rare'] * 3 + ['common'] * 20",
            "l4 = ['common'] * 200 + ['other'] * 200",
            "assert contrastive_label(h4, l4)[0] == 'rare'",
        ],
    },

    {
        "id": "rt_h08",
        "name": "depth_conservation_null",
        "tier": "HARD",
        "trap_family": "tie-handling",
        "source": "data/sae-analysis/recompute_corrections.py:156",
        "prompt": r'''
import numpy as np

def greedy_depth_match(a, b, tol=0.075):
    """
    Count how many of the relative depths in `a` can be paired with distinct
    relative depths in `b` within `tol`.

    Build every (distance, i, j) triple, sort it ascending as a TUPLE, and
    walk it: stop as soon as the distance exceeds `tol`; skip a triple whose i
    or j has already been consumed; otherwise consume both and count it.

    This is nearest-pair-first greedy, NOT an optimal assignment -- it can and
    does return fewer pairs than a Hungarian matcher on the same input, and
    that difference is exactly what the published claim rested on.
    """

def depth_conservation_null(a_layers, n_a, b_layers, n_b, tol=0.075,
                            n_perm=2000, seed=42, lo_a=0, lo_b=0):
    """
    Permutation null for "X of K circuit-critical head positions are conserved
    by relative depth".

    Relative depth is layer / n_layers, using each model's OWN layer count:
    a_layers are divided by n_a and b_layers by n_b.

    Compute the observed match count with greedy_depth_match, then draw
    `n_perm` random replicates: len(a_layers) layer indices uniform in
    [lo_a, n_a) divided by n_a, and len(b_layers) uniform in [lo_b, n_b)
    divided by n_b, both from np.random.default_rng(seed) created ONCE and
    advanced across the whole loop, drawing the `a` side before the `b` side
    on every iteration.

    The p-value is the fraction of replicates whose match count is GREATER
    THAN OR EQUAL TO the observed count.

    Returns a dict with, and only with:
      "observed_matches"  int
      "observed_rate"     observed_matches / len(a_layers), unrounded
      "mean_matches"      mean replicate count, rounded to 2
      "mean_rate"         mean count / len(a_layers), rounded to 3
      "p_value"           rounded to 4
      "n_permutations"    int
    """
''',
        "reference": r'''
import numpy as np

def greedy_depth_match(a, b, tol=0.075):
    pairs = sorted((abs(x - y), i, j) for i, x in enumerate(a) for j, y in enumerate(b))
    ua, ub, n = set(), set(), 0
    for d, i, j in pairs:
        if d > tol:
            break
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
        n += 1
    return n

def depth_conservation_null(a_layers, n_a, b_layers, n_b, tol=0.075,
                            n_perm=2000, seed=42, lo_a=0, lo_b=0):
    rng = np.random.default_rng(seed)
    ad = [l / n_a for l in a_layers]
    bd = [l / n_b for l in b_layers]
    obs = greedy_depth_match(ad, bd, tol)
    k = len(a_layers)
    hits = 0
    tot = 0
    for _ in range(n_perm):
        xa = rng.integers(lo_a, n_a, k) / n_a
        xb = rng.integers(lo_b, n_b, len(b_layers)) / n_b
        m = greedy_depth_match(list(xa), list(xb), tol)
        tot += m
        hits += (m >= obs)
    return {
        "observed_matches": int(obs),
        "observed_rate": obs / k,
        "mean_matches": round(tot / n_perm, 2),
        "mean_rate": round(tot / n_perm / k, 3),
        "p_value": round(hits / n_perm, 4),
        "n_permutations": int(n_perm),
    }
''',
        "tests": [
            "# TRAP: nearest-pair-first greedy loses a pairing that an optimal",
            "# assignment keeps. a1-b0 is the globally shortest edge, consuming b0,",
            "# after which a0 has nothing in range -- yet a0-b0 plus a1-b1 is legal.",
            "assert greedy_depth_match([0.0, 0.05], [0.04, 0.10], 0.06) == 1",
            "assert greedy_depth_match([0.0], [0.04], 0.06) == 1",
            "assert greedy_depth_match([0.0], [1.0], 0.075) == 0",
            "assert greedy_depth_match([], [0.1], 0.075) == 0",
            "assert greedy_depth_match([0.1], [], 0.075) == 0",
            "# tol boundary is inclusive (0.625 - 0.5 is exact in binary)",
            "assert greedy_depth_match([0.5], [0.625], 0.125) == 1",
            "assert greedy_depth_match([0.5], [0.626], 0.125) == 0",
            "# each side is consumed at most once",
            "assert greedy_depth_match([0.5, 0.5, 0.5], [0.5, 0.5], 0.01) == 2",
            "assert greedy_depth_match([0.5], [0.5, 0.5, 0.5], 0.01) == 1",
            "# relative depth uses each model's OWN layer count",
            "d = depth_conservation_null([14], 28, [12], 24, tol=0.01, n_perm=200, seed=1)",
            "assert d['observed_matches'] == 1",
            "d2 = depth_conservation_null([14], 28, [23], 24, tol=0.01, n_perm=200, seed=1)",
            "assert d2['observed_matches'] == 0",
            "assert set(d) == {'observed_matches','observed_rate','mean_matches','mean_rate','p_value','n_permutations'}",
            "# TRAP: the p-value uses >=, so an observed count of 0 gives exactly 1.0",
            "z = depth_conservation_null([0], 28, [23], 24, tol=0.001, n_perm=300, seed=4)",
            "assert z['observed_matches'] == 0",
            "assert z['p_value'] == 1.0, z['p_value']",
            "# a real ten-head comparison: deterministic, bounded, self-consistent",
            "ll = [15, 17, 13, 24, 19, 21, 18, 14, 27, 26]",
            "py = [10, 15, 22, 1, 21, 17, 10, 12, 16, 13]",
            "r = depth_conservation_null(ll, 28, py, 24, tol=0.075, n_perm=500, seed=42)",
            "assert r['n_permutations'] == 500",
            "assert 0 <= r['observed_matches'] <= 10",
            "assert abs(r['observed_rate'] - r['observed_matches'] / 10) < 1e-12",
            "assert 0.0 <= r['p_value'] <= 1.0",
            "assert 0.0 <= r['mean_matches'] <= 10.0",
            "assert abs(r['mean_rate'] - r['mean_matches'] / 10) < 0.01",
            "assert r['mean_matches'] > 0.0",
            "# a single rng advanced across the loop -> fully reproducible",
            "r_again = depth_conservation_null(ll, 28, py, 24, tol=0.075, n_perm=500, seed=42)",
            "assert r == r_again",
            "r_other = depth_conservation_null(ll, 28, py, 24, tol=0.075, n_perm=500, seed=43)",
            "assert r_other['n_permutations'] == 500",
            "# an 'informed' null restricted to the deep layer band matches more often",
            "lo = depth_conservation_null(ll, 28, py, 24, tol=0.075, n_perm=500, seed=42, lo_a=11, lo_b=9)",
            "assert lo['mean_matches'] >= r['mean_matches']",
            "assert isinstance(r['observed_matches'], int)",
        ],
    },

    {
        "id": "rt_h09",
        "name": "null_threshold",
        "tier": "HARD",
        "trap_family": "index-space",
        "source": "data/sae-analysis/cross_arch_matching_v3.py:152",
        "prompt": r'''
import numpy as np

def null_threshold(fp_a, fp_b, valid_a, valid_b, rng, n_perm=20, percentile=99.0):
    """
    Permutation-calibrated similarity threshold for cross-architecture SAE
    feature matching. A hand-picked tau of 0.80 turned out to sit BELOW this
    noise floor, which is why the threshold has to be calibrated.

    fp_a : (D_a, N) fingerprints, rows unit-norm
    fp_b : (D_b, N)
    valid_a, valid_b : boolean support masks over the rows
    rng  : a np.random.Generator, advanced across the whole loop

    On each of `n_perm` iterations:
      * draw a FRESH permutation of the N fingerprint columns and apply it to
        the valid rows of fp_b, destroying any real correspondence
      * re-normalise those permuted rows
      * find, in this permuted space, the reciprocal best matches between the
        valid rows of fp_a and the permuted valid rows of fp_b
      * collect the similarities of ONLY the reciprocal matches -- not every
        best match

    Concatenate every iteration's similarities into ONE pool and take the
    requested percentile of that pool. Do not average per-iteration
    percentiles.

    Returns (tau, all_values) where tau is a float and all_values is the
    pooled 1-D array.
    """
''',
        "reference": r'''
import numpy as np

def null_threshold(fp_a, fp_b, valid_a, valid_b, rng, n_perm=20, percentile=99.0):
    ia = np.flatnonzero(valid_a)
    ib = np.flatnonzero(valid_b)
    A = fp_a[ia]
    sims_null = []
    for _ in range(n_perm):
        perm = rng.permutation(fp_b.shape[1])
        Bp = fp_b[ib][:, perm]
        Bp = Bp / np.maximum(np.linalg.norm(Bp, axis=1, keepdims=True), 1e-12)
        sims = A @ Bp.T
        a2b = np.argmax(sims, axis=1)
        a2b_s = sims[np.arange(len(a2b)), a2b]
        b2a = np.argmax(sims.T, axis=1)
        mutual = b2a[a2b] == np.arange(len(a2b))
        sims_null.append(a2b_s[mutual])
    allv = np.concatenate(sims_null)
    return float(np.percentile(allv, percentile)), allv
''',
        "tests": [
            "rng0 = np.random.default_rng(0)",
            "def unit(m, n, s):",
            "    r = np.random.default_rng(s)",
            "    X = r.standard_normal((m, n))",
            "    return X / np.linalg.norm(X, axis=1, keepdims=True)",
            "FA = unit(30, 24, 1)",
            "FB = unit(26, 24, 2)",
            "va = np.ones(30, bool); va[:5] = False",
            "vb = np.ones(26, bool); vb[:3] = False",
            "tau, allv = null_threshold(FA, FB, va, vb, np.random.default_rng(7), n_perm=6)",
            "assert isinstance(tau, float)",
            "assert allv.ndim == 1 and allv.size > 0",
            "# TRAP: only RECIPROCAL matches enter the pool. Pooling every best match",
            "# would give exactly n_perm * n_valid_a entries.",
            "assert allv.size < 6 * int(va.sum()), (allv.size, 6 * int(va.sum()))",
            "assert allv.size <= 6 * min(int(va.sum()), int(vb.sum()))",
            "# TRAP: the percentile is of the POOLED array, not a mean of per-run",
            "# percentiles -- the returned tau must be exactly that.",
            "assert abs(tau - float(np.percentile(allv, 99.0))) < 1e-12",
            "t50, a50 = null_threshold(FA, FB, va, vb, np.random.default_rng(7), n_perm=6, percentile=50.0)",
            "assert np.array_equal(a50, allv)",
            "assert abs(t50 - float(np.percentile(allv, 50.0))) < 1e-12",
            "assert t50 <= tau",
            "# similarities are cosines of unit rows",
            "assert allv.max() <= 1.0 + 1e-9 and allv.min() >= -1.0 - 1e-9",
            "# the rng is threaded through, so the same seed reproduces exactly",
            "tb, ab = null_threshold(FA, FB, va, vb, np.random.default_rng(7), n_perm=6)",
            "assert abs(tb - tau) < 1e-12 and np.array_equal(ab, allv)",
            "tc, ac = null_threshold(FA, FB, va, vb, np.random.default_rng(8), n_perm=6)",
            "assert not (ac.size == allv.size and np.allclose(ac, allv))",
            "# TRAP: a FRESH permutation per iteration. Hoisting it out of the loop",
            "# would make the pool n_perm identical repeats of the first block.",
            "t1, p1 = null_threshold(FA, FB, va, vb, np.random.default_rng(7), n_perm=1)",
            "t2, p2 = null_threshold(FA, FB, va, vb, np.random.default_rng(7), n_perm=2)",
            "assert p2.size >= p1.size",
            "assert np.allclose(p2[:p1.size], p1)",
            "assert not (p2.size == 2 * p1.size and np.allclose(p2[p1.size:], p1))",
            "# masked-out rows never contribute: with two valid rows a side, each",
            "# iteration can add at most two similarities to the pool.",
            "va2 = np.zeros(30, bool); va2[[0, 1]] = True",
            "vb2 = np.zeros(26, bool); vb2[[4, 5]] = True",
            "t3, p3 = null_threshold(FA, FB, va2, vb2, np.random.default_rng(7), n_perm=5)",
            "assert 0 < p3.size <= 10, p3.size",
            "assert abs(t3 - float(np.percentile(p3, 99.0))) < 1e-12",
            "assert p3.max() <= 1.0 + 1e-9",
        ],
    },

    {
        "id": "rt_h10",
        "name": "build_pdf",
        "tier": "HARD",
        "trap_family": "index-space",
        "source": "papers/distributed-interp/submission/make_figures.py:64",
        "prompt": r'''
def build_pdf(objects, ):
    """
    Assemble a complete, valid PDF 1.4 file from a list of already-serialised
    object bodies. We emit paper figures as vector PDFs with no external
    library, so the cross-reference table has to be built by hand.

    `objects` is a list of bytes; objects[i] is the body of object number
    i + 1 (they are 1-indexed in a PDF).

    File layout:
      * the literal header  b"%PDF-1.4\n"
      * for each object, in order:  b"<num> 0 obj\n" + body + b"\nendobj\n"
      * the cross-reference section, then the trailer:

            xref
            0 <len(objects) + 1>
            0000000000 65535 f
            <offset of object 1, ten digits> 00000 n
            ...
            trailer
            << /Size <len(objects) + 1> /Root 1 0 R >>
            startxref
            <byte offset at which the xref section begins>
            %%EOF

    Every xref entry is EXACTLY 20 bytes: the ten-or-five digit field, a
    space, the second field, a space, the type letter, then a SPACE and a
    newline. Readers reject the file if an entry is any other length.

    The offset recorded for an object is the byte offset of the start of its
    "<num> 0 obj" line within the finished file, and the startxref value is
    the byte offset at which the "xref" keyword itself begins.

    Returns the complete file as bytes.
    """
''',
        "reference": r'''
def build_pdf(objects, ):
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objects, 1):
        offs.append(len(out))
        out += ("%d 0 obj\n" % i).encode() + o + b"\nendobj\n"
    xref = len(out)
    out += ("xref\n0 %d\n" % (len(objects) + 1)).encode()
    out += b"0000000000 65535 f \n"
    for o in offs:
        out += ("%010d 00000 n \n" % o).encode()
    out += ("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, xref)).encode()
    return bytes(out)
''',
        "tests": [
            "content = b'0 0 m 100 100 l S'",
            "objs = [",
            "    b'<< /Type /Catalog /Pages 2 0 R >>',",
            "    b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',",
            "    b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R >>',",
            "    b'<< /Length ' + str(len(content)).encode() + b' >>\\nstream\\n' + content + b'\\nendstream',",
            "    b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',",
            "]",
            "data = build_pdf(objs)",
            "assert isinstance(data, (bytes, bytearray))",
            "data = bytes(data)",
            "assert data.startswith(b'%PDF-1.4\\n')",
            "assert data.endswith(b'%%EOF\\n')",
            "assert data.count(b'endobj') == 5",
            "# locate the xref through startxref, exactly as a reader would",
            "i = data.rfind(b'startxref\\n')",
            "assert i != -1",
            "j = data.index(b'\\n', i + 10)",
            "sx = int(data[i + 10:j])",
            "# TRAP: startxref must point at the 'xref' keyword itself",
            "assert data[sx:sx + 5] == b'xref\\n', data[sx:sx + 12]",
            "hdr_end = data.index(b'\\n', sx + 5)",
            "assert data[sx + 5:hdr_end].split() == [b'0', b'6']",
            "tbl = data[hdr_end + 1:]",
            "# TRAP: 20-byte entries, including the space BEFORE the newline",
            "assert tbl[:20] == b'0000000000 65535 f \\n', tbl[:24]",
            "for k in range(len(objs)):",
            "    entry = tbl[20 * (k + 1):20 * (k + 2)]",
            "    assert len(entry) == 20, (k, entry)",
            "    assert entry[10:11] == b' ' and entry[11:16] == b'00000'",
            "    assert entry[16:17] == b' ' and entry[17:18] == b'n' and entry[18:20] == b' \\n'",
            "    off = int(entry[:10])",
            "    tag = ('%d 0 obj' % (k + 1)).encode()",
            "    # TRAP: the offset is captured BEFORE the object bytes are appended",
            "    assert data[off:off + len(tag)] == tag, (k, off, data[off:off + 20])",
            "assert b'/Size 6' in data and b'/Root 1 0 R' in data",
            "# offsets are strictly increasing and land inside the file",
            "offsets = [int(tbl[20 * (k + 1):20 * (k + 1) + 10]) for k in range(len(objs))]",
            "assert offsets == sorted(offsets) and len(set(offsets)) == len(objs)",
            "assert offsets[0] == 9 and max(offsets) < sx",
            "# the whole structure must scale with the object count",
            "one = bytes(build_pdf([b'<< /Type /Catalog >>']))",
            "assert b'/Size 2' in one",
            "i1 = one.rfind(b'startxref\\n')",
            "sx1 = int(one[i1 + 10:one.index(b'\\n', i1 + 10)])",
            "assert one[sx1:sx1 + 5] == b'xref\\n'",
            "e1 = one[one.index(b'\\n', sx1 + 5) + 1:]",
            "assert e1[:20] == b'0000000000 65535 f \\n'",
            "assert int(e1[20:30]) == 9 and one[9:16] == b'1 0 obj'",
        ],
    },
]


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------

def build_program(task, solution=None):
    """Assemble the full program that is executed for one task."""
    parts = []
    if task.get("fixture"):
        parts.append(SAE_FIXTURE_SRC)
    parts.append(solution if solution is not None else task["reference"])
    parts.append("\n".join(task["tests"]))
    parts.append("print('__TASK_OK__')")
    return "\n\n".join(parts)


def run_task(task, solution=None, timeout=180):
    """Execute one task in a subprocess. Returns (ok, detail)."""
    src = build_program(task, solution)
    fd, path = tempfile.mkstemp(suffix=".py", prefix="rt_%s_" % task["id"])
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(src)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT after %ss" % timeout
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode == 0 and "__TASK_OK__" in proc.stdout:
        return True, ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return False, "\n".join(tail[-6:]) if tail else "rc=%d" % proc.returncode


def check_solution(task_id, solution_src, timeout=180):
    """Grade a candidate solution string against one task by id."""
    task = next(t for t in REAL_TASKS if t["id"] == task_id)
    return run_task(task, solution=solution_src, timeout=timeout)


def validate(verbose=True):
    """Run every task against its own reference solution."""
    results = []
    for task in REAL_TASKS:
        ok, detail = run_task(task)
        results.append((task, ok, detail))
        if verbose:
            print("%-8s %-6s %-28s %s" % (
                task["id"], task["tier"], task["name"], "PASS" if ok else "FAIL"))
            if not ok:
                for line in detail.splitlines():
                    print("             | %s" % line)
    n_pass = sum(1 for _, ok, _ in results if ok)
    if verbose:
        print("")
        by_tier = {}
        by_family = {}
        for task in REAL_TASKS:
            by_tier[task["tier"]] = by_tier.get(task["tier"], 0) + 1
            by_family[task["trap_family"]] = by_family.get(task["trap_family"], 0) + 1
        print("tiers   : " + "  ".join("%s=%d" % (t, by_tier.get(t, 0)) for t in TIERS))
        print("families: " + "  ".join("%s=%d" % (f, by_family.get(f, 0)) for f in TRAP_FAMILIES))
        print("assertions: %d" % sum(
            len([a for a in t["tests"] if a.strip().startswith("assert")]) for t in REAL_TASKS))
        print("")
        print("%d/%d tasks pass with their reference solution" % (n_pass, len(REAL_TASKS)))
    return n_pass, len(REAL_TASKS), results


def _self_check():
    """Structural invariants that must hold regardless of execution."""
    ids = [t["id"] for t in REAL_TASKS]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    for t in REAL_TASKS:
        assert t["tier"] in TIERS, t["id"]
        assert t["trap_family"] in TRAP_FAMILIES, t["id"]
        assert ":" in t["source"], t["id"]
        assert t["prompt"].strip() and t["reference"].strip(), t["id"]
        assert len(t["tests"]) >= 5, t["id"]
        # a prompt must never leak the solution
        assert "def " in t["prompt"] or "class " in t["prompt"], t["id"]
        # no task may require emitting a markdown fence delimiter
        fence = "`" * 3
        assert fence not in t["prompt"], t["id"]
        assert fence not in t["reference"], t["id"]
        assert all(fence not in a for a in t["tests"]), t["id"]
        # heavy deps are banned at eval time
        blob = t["prompt"] + t["reference"] + "\n".join(t["tests"])
        for banned in ("import torch", "import mlx", "transformers", "sklearn",
                       "scipy", "statsmodels", "requests", "urllib"):
            assert banned not in blob, (t["id"], banned)


if __name__ == "__main__":
    _self_check()
    n_pass, n_total, results = validate()
    if "--json" in sys.argv:
        print(json.dumps([
            {"id": t["id"], "tier": t["tier"], "family": t["trap_family"],
             "name": t["name"], "pass": ok, "detail": d}
            for t, ok, d in results
        ], indent=1))
    sys.exit(0 if n_pass == n_total else 1)
