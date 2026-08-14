#!/usr/bin/env python3
"""Compare cluster-gate arms at the byte level and at the verdict level.

Two levels, because they answer different questions and can disagree:

  byte level     did the model write the identical string?
  verdict level  did the pass/fail outcome change?

Byte-identical is the strong result: it means distribution is numerically inert and
a cluster-hosted arm can be compared against single-node arms without qualification.
Verdict-identical with byte differences is the weaker, still-usable result: the
comparison holds for pass@1 but not for anything token-level, and any paper using it
has to say so.

Verdicts come from EvalPlus, scored separately per its own protocol:
  ~/evalplus-env/bin/evalplus.evaluate --dataset humaneval --samples <arm>/humaneval-samples.jsonl
This script reads the eval_results.json EvalPlus writes beside the samples. If those
are absent it reports byte-level only rather than guessing.

Usage:
  compare_arms.py data/cluster-gate/runs A-mlx-single B-exo-1node C-exo-2node
"""
import json
import sys
from pathlib import Path


def load_raw(armdir):
    return {r["task_id"]: r for r in
            (json.loads(l) for l in (armdir / "raw.jsonl").read_text().splitlines() if l)}


def load_verdicts(armdir):
    """EvalPlus writes eval_results.json next to the samples file."""
    for name in ("eval_results.json", "humaneval-samples_eval_results.json"):
        p = armdir / name
        if p.exists():
            d = json.loads(p.read_text())
            out = {}
            for tid, entry in d.get("eval", {}).items():
                # EvalPlus shapes have shifted across versions; accept both.
                first = entry[0] if isinstance(entry, list) and entry else entry
                if isinstance(first, dict):
                    status = first.get("base_status") or first.get("status")
                    out[tid] = (status == "pass")
            if out:
                return out
    return None


def compare(a_name, a_raw, b_name, b_raw, a_v, b_v):
    ids = sorted(set(a_raw) & set(b_raw), key=lambda t: int(t.split("/")[1]))
    missing = (set(a_raw) ^ set(b_raw))
    same_bytes = [t for t in ids if a_raw[t]["sha256"] == b_raw[t]["sha256"]]
    diff = [t for t in ids if t not in set(same_bytes)]

    print(f"\n=== {a_name}  vs  {b_name} ===")
    if missing:
        print(f"  WARNING: {len(missing)} task(s) present in only one arm: "
              f"{sorted(missing)[:5]}")
    print(f"  byte-identical : {len(same_bytes)}/{len(ids)}"
          f"  ({100*len(same_bytes)/max(len(ids),1):.1f}%)")

    if diff:
        print(f"  differing      : {len(diff)}  e.g. {diff[:5]}")
        t = diff[0]
        # Show where the first divergence starts -- an early split means the very
        # first sampled token differed, which points at prompt/template or numerics;
        # a late split points at accumulated drift.
        x, y = a_raw[t]["raw"], b_raw[t]["raw"]
        i = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y)))
        print(f"  first divergence in {t} at char {i} of {len(x)}/{len(y)}")
        print(f"    {a_name}: ...{x[max(0,i-40):i+40]!r}")
        print(f"    {b_name}: ...{y[max(0,i-40):i+40]!r}")

    if a_v and b_v:
        vids = sorted(set(a_v) & set(b_v), key=lambda t: int(t.split("/")[1]))
        agree = [t for t in vids if a_v[t] == b_v[t]]
        flips = [(t, a_v[t], b_v[t]) for t in vids if a_v[t] != b_v[t]]
        print(f"  verdict-identical: {len(agree)}/{len(vids)}"
              f"   pass@1 {a_name}={sum(a_v[t] for t in vids)}/{len(vids)}"
              f"  {b_name}={sum(b_v[t] for t in vids)}/{len(vids)}")
        if flips:
            print(f"  flipped: {flips[:8]}")
    else:
        print("  verdicts: not scored yet (run evalplus.evaluate on both arms)")

    return len(same_bytes) == len(ids) and not missing


def main():
    root = Path(sys.argv[1])
    arms = sys.argv[2:]
    if len(arms) < 2:
        sys.exit("need at least two arm names")

    raw = {a: load_raw(root / a) for a in arms}
    verd = {a: load_verdicts(root / a) for a in arms}
    for a in arms:
        print(f"{a}: {len(raw[a])} completions, "
              f"verdicts {'yes' if verd[a] else 'no'}")

    results = {}
    for i in range(len(arms) - 1):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            results[f"{a} vs {b}"] = compare(a, raw[a], b, raw[b], verd[a], verd[b])

    print("\n=== gate ===")
    for pair, identical in results.items():
        print(f"  {pair:34} {'PASS (byte-identical)' if identical else 'DIFFERS'}")
    print("\nReading it:")
    print("  all pairs identical      -> the cluster is numerically just a bigger")
    print("                              machine; a cluster-hosted bf16 arm is sound")
    print("  A vs B differs           -> exo's serving stack, not distribution, is the")
    print("                              variable; fix or run every arm through exo")
    print("  B vs C differs           -> distribution itself perturbs output; a")
    print("                              cluster-only arm cannot be compared to")
    print("                              single-node arms except at the verdict level")


if __name__ == "__main__":
    main()
