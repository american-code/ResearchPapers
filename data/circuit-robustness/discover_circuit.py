#!/usr/bin/env python3
"""Discover a circuit on an arbitrary dataset: patching sweep + ablation sweep.

Replaces the four dataset-hardcoded scripts in data/ioi/ with one that takes the
dataset as an argument, which is the whole point: the robustness protocol needs a
circuit discovered on D and scored on D-prime, so discovery cannot be a fixed list.

METRICS (both preserved from the published protocol so circuits stay comparable)

  patching   For each (l, h), run the CORRUPT prompt but splice in that head's clean
             SDPA output, and measure how much of the clean-vs-corrupt gap is recovered:
                 (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)
             High = that head alone carries much of the task signal.

  ablation   For each (l, h), mean-ablate it on the CLEAN prompt and measure the drop:
                 LD_clean - LD_ablated
             High = the rest of the model does not compensate for losing it.

  combined   Each metric min-max normalised to [0, 1] across all heads, then summed;
             top-k by that sum. This is the selection rule documented in
             data/ioi/cross-model-comparison.md, reproduced exactly.

The two metrics disagree often enough to matter -- patching finds heads that carry
signal, ablation finds heads nothing else replaces -- which is why the published
selection uses both rather than either.

COST. The patching sweep is n_layers x n_heads forward passes per batch: 672 for
Llama, 384 for Pythia, times ceil(n/batch). --limit defaults to 100 to match the
published protocol and keep this tractable; raising it scales runtime linearly.

Usage:
  python3 discover_circuit.py --model llama --dataset ioi-frame-D.json \
      --out circuits/llama-ioi-frame.json
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

import circuit_lib as C

HERE = Path(__file__).parent


def minmax(flat):
    lo, hi = min(flat), max(flat)
    if hi - lo < 1e-12:
        # A degenerate metric would otherwise contribute a constant to every head and
        # silently hand circuit selection entirely to the other metric.
        return [0.0] * len(flat), True
    return [(v - lo) / (hi - lo) for v in flat], False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(C.MODELS), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=100,
                    help="examples used (published protocol used 100)")
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    cfg = C.MODELS[args.model]
    nl, nh = cfg["n_layers"], cfg["n_heads"]

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = HERE / args.dataset
    ds = json.loads(ds_path.read_text())
    examples = ds["examples"][: args.limit]
    print(f"{ds_path.name}: {len(examples)} examples "
          f"({len({e['template_id'] for e in examples})} templates)")

    print(f"Loading {cfg['id']} …", flush=True)
    model, tokenizer = load(cfg["id"])
    attns = C.install(model, args.model)
    batches = C.tokenize(examples, tokenizer, args.batch)
    print(f"  {len(attns)} attention layers, {len(batches)} batches", flush=True)

    t0 = time.time()
    ld_clean, _ = C.run_clean_and_cache_mean(model, attns, batches)

    C.set_mode(attns, "normal")
    corr_lds = []
    for _, corr, last, a_ids, d_ids in batches:
        logits = model(corr)
        mx.eval(logits)
        corr_lds += C.logit_diff(logits, last, a_ids, d_ids)
    ld_corrupt = sum(corr_lds) / len(corr_lds)
    print(f"clean LD = {ld_clean:.4f}   corrupt LD = {ld_corrupt:.4f}", flush=True)

    gap = ld_clean - ld_corrupt
    if gap <= 0:
        # Without a clean-over-corrupt gap the model is not doing the task, and every
        # downstream number would be noise divided by noise.
        raise SystemExit(f"ABORT: clean LD ({ld_clean:.4f}) does not exceed corrupt "
                         f"({ld_corrupt:.4f}); the model does not perform this task.")

    # ── patching sweep ───────────────────────────────────────────────────────
    # Batches outer, heads inner. The clean cache depends only on the batch, so
    # caching once per batch and sweeping all heads against it costs
    # n_batches * (1 + n_layers*n_heads) forward passes instead of
    # n_batches * n_layers*n_heads * 2 -- half the work, identical arithmetic.
    tot = [[0.0] * nh for _ in range(nl)]
    cnt = 0
    for bi, (clean, corr, last, a_ids, d_ids) in enumerate(batches):
        C.set_mode(attns, "cache_clean")
        mx.eval(model(clean))
        for l in range(nl):
            for h in range(nh):
                C.set_patch_target(attns, l, h)
                logits = model(corr)
                mx.eval(logits)
                tot[l][h] += sum(C.logit_diff(logits, last, a_ids, d_ids))
        cnt += len(last)
        print(f"  patching batch {bi + 1}/{len(batches)}  ({time.time() - t0:.0f}s)",
              flush=True)
    patch = [[(tot[l][h] / cnt - ld_corrupt) / gap for h in range(nh)] for l in range(nl)]
    C.set_mode(attns, "normal")

    # ── ablation sweep ───────────────────────────────────────────────────────
    # The mean was cached during run_clean_and_cache_mean and is not recomputed, so
    # every head is ablated to the same reference the faithfulness runs will use.
    drop = [[0.0] * nh for _ in range(nl)]
    for l in range(nl):
        for h in range(nh):
            ld, _ = C.run_ablated(model, attns, batches, {(l, h)})
            drop[l][h] = ld_clean - ld
        print(f"  ablation layer {l + 1}/{nl}  ({time.time() - t0:.0f}s)", flush=True)

    # ── combined ranking ─────────────────────────────────────────────────────
    pf, p_degen = minmax([v for row in patch for v in row])
    df, d_degen = minmax([v for row in drop for v in row])
    if p_degen or d_degen:
        print(f"  WARNING: degenerate metric (patching={p_degen}, ablation={d_degen}) "
              f"-- selection driven by the other metric alone")
    combined = [(pf[i] + df[i], i // nh, i % nh) for i in range(nl * nh)]
    combined.sort(key=lambda r: -r[0])
    circuit = [[l, h] for _, l, h in combined[: args.top_k]]

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {
            "model": cfg["id"], "model_key": args.model,
            "dataset": str(ds_path), "dataset_meta": ds.get("meta", {}),
            "n_examples": len(examples), "n_layers": nl, "n_heads": nh,
            "clean_logit_diff": round(ld_clean, 4),
            "corrupt_logit_diff": round(ld_corrupt, 4),
            "top_k": args.top_k, "elapsed_sec": round(time.time() - t0, 1),
            "selection": "min-max normalise patching and ablation to [0,1] across all "
                         "heads, sum, take top-k",
            "mean_ablation_scope": "all clean examples (see circuit_lib docstring: the "
                                   "data/ioi harness used the final batch only)",
            "synthetic": False,
        },
        "circuit": circuit,
        "circuit_labels": [f"L{l}H{h}" for l, h in circuit],
        "combined_top": [{"head": f"L{l}H{h}", "combined": round(s, 4),
                          "patching": round(patch[l][h], 4),
                          "ablation_drop": round(drop[l][h], 4)}
                         for s, l, h in combined[: max(args.top_k * 2, 20)]],
        "patching_scores": [[round(v, 6) for v in row] for row in patch],
        "drop_scores": [[round(v, 6) for v in row] for row in drop],
    }, indent=1))
    print(f"\ncircuit: {[f'L{l}H{h}' for l, h in circuit]}")
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
