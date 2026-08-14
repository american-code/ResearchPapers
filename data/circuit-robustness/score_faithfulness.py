#!/usr/bin/env python3
"""Score a circuit on a dataset: faithfulness, minimality, completeness.

Same three quantities as data/ioi/run_circuit_faithfulness.py and the same
normalisation, but built on circuit_lib so that discovery and scoring share one
implementation. That matters more than it sounds: D and D-prime must be measured by
an identical procedure or the difference between them is partly the procedure. The
data/ioi harness pads per batch and caches the ablation mean from the final batch
only, both of which vary with the data, so it would have introduced a D-vs-D-prime
difference of its own on the non-IOI families.

  F(C)          = (LD(only C active) - LD_floor) / (LD_clean - LD_floor)
                  where LD_floor is every head mean-ablated. The normalisation
                  matters: an unnormalised ratio treats the all-ablated model as
                  scoring zero, when in fact it can score anything.
  minimality    F(C) - F(C \\ {v}) for each v in C
  completeness  |F(C \\ K) - F(M \\ K)| over random subsets K

Usage:
  python3 score_faithfulness.py --model llama --dataset ioi-frame-Dprime.json \
      --circuit-file circuits/llama-ioi-frame.json --out results/x.json
"""

import argparse
import json
import random
import time
from pathlib import Path

from mlx_lm import load

import circuit_lib as C

HERE = Path(__file__).parent
SEED = 42
N_COMPLETENESS_SUBSETS = 12


def resolve(p):
    q = Path(p)
    return q if q.is_absolute() else HERE / q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(C.MODELS), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--circuit-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()
    random.seed(SEED)

    cfg = C.MODELS[args.model]
    nl, nh = cfg["n_layers"], cfg["n_heads"]
    ds = json.loads(resolve(args.dataset).read_text())
    examples = ds["examples"][: args.limit]
    circuit = {tuple(x) for x in json.loads(resolve(args.circuit_file).read_text())["circuit"]}

    print(f"Loading {cfg['id']} …", flush=True)
    model, tokenizer = load(cfg["id"])
    attns = C.install(model, args.model)
    batches = C.tokenize(examples, tokenizer, args.batch)

    t0 = time.time()
    ld_clean, _ = C.run_clean_and_cache_mean(model, attns, batches)
    all_heads = {(l, h) for l in range(nl) for h in range(nh)}
    ld_floor, _ = C.run_ablated(model, attns, batches, all_heads)
    span = ld_clean - ld_floor
    if abs(span) < 1e-9:
        raise SystemExit("ABORT: clean and fully-ablated logit differences coincide; "
                         "F(C) would be a division by zero.")
    norm = lambda ld: (ld - ld_floor) / span

    print(f"clean {ld_clean:.4f}  floor {ld_floor:.4f}  |C|={len(circuit)}", flush=True)

    # faithfulness: everything OUTSIDE the circuit ablated
    ld_c, _ = C.run_ablated(model, attns, batches, all_heads - circuit)
    F = norm(ld_c)
    print(f"F(C) = {F:.4f}", flush=True)

    minimality = []
    for v in sorted(circuit):
        ld_v, _ = C.run_ablated(model, attns, batches, (all_heads - circuit) | {v})
        minimality.append({"head": f"L{v[0]}H{v[1]}", "F_without": round(norm(ld_v), 4),
                           "delta": round(F - norm(ld_v), 4)})

    completeness = []
    circ = sorted(circuit)
    for _ in range(N_COMPLETENESS_SUBSETS):
        k = random.randint(1, max(1, len(circ) // 2))
        K = set(random.sample(circ, k))
        ld_ck, _ = C.run_ablated(model, attns, batches, (all_heads - circuit) | K)
        ld_mk, _ = C.run_ablated(model, attns, batches, K)
        completeness.append({"K": [f"L{l}H{h}" for l, h in sorted(K)],
                             "F_C_minus_K": round(norm(ld_ck), 4),
                             "F_model_minus_K": round(norm(ld_mk), 4),
                             "gap": round(abs(norm(ld_ck) - norm(ld_mk)), 4)})

    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meta": {
            "model": cfg["id"], "model_key": args.model,
            "dataset": str(resolve(args.dataset)), "dataset_meta": ds.get("meta", {}),
            "circuit_source": str(resolve(args.circuit_file)),
            "circuit": [f"L{l}H{h}" for l, h in sorted(circuit)],
            "n_examples": len(examples), "seed": SEED, "synthetic": False,
            "normalization": "F(X) = (LD(X) - LD_all_ablated) / (LD_clean - LD_all_ablated)",
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "logit_diffs": {"clean": round(ld_clean, 4), "all_ablated": round(ld_floor, 4),
                        "circuit_only": round(ld_c, 4)},
        "faithfulness": round(F, 4),
        "minimality": minimality,
        "completeness": completeness,
        "completeness_gap_mean": round(
            sum(c["gap"] for c in completeness) / len(completeness), 4),
    }, indent=1))
    print(f"wrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
