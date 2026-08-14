#!/usr/bin/env python3
"""Collect protocol results into the paper's central table.

The column that decides the paper is F(D')/F(D) -- retention. Reporting F(D') alone
would confuse a circuit that was weak everywhere with one that was strong on its
discovery set and collapsed off it, and only the second is evidence of frame
dependence.

Clean LD on both splits is printed alongside, because a faithfulness drop is only
interpretable if the model still performs the task on D-prime. If clean LD collapses
too, the shift was not task-preserving and that row says nothing about circuits.

Usage: collect_results.py results/
"""

import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = []
    for d_file in sorted(root.glob("*-D.faith.json")):
        tag = d_file.name[: -len("-D.faith.json")]
        p_file = root / f"{tag}-Dprime.faith.json"
        if not p_file.exists():
            print(f"  (skipping {tag}: no D-prime result)")
            continue
        d, p = json.loads(d_file.read_text()), json.loads(p_file.read_text())
        rows.append({
            "tag": tag,
            "F_D": d["faithfulness"], "F_Dp": p["faithfulness"],
            "ld_D": d["logit_diffs"]["clean"], "ld_Dp": p["logit_diffs"]["clean"],
            "gap_D": d["completeness_gap_mean"], "gap_Dp": p["completeness_gap_mean"],
        })

    if not rows:
        print("no completed pairs found")
        return

    print(f"\n{'run':28} {'F(D)':>7} {'F(D-prime)':>11} {'retained':>9} "
          f"{'LD(D)':>7} {'LD(D-prime)':>12} {'behav':>7}")
    print("-" * 92)
    for r in rows:
        ret = r["F_Dp"] / r["F_D"] if abs(r["F_D"]) > 1e-9 else float("nan")
        # Behaviour retention gates interpretation of the faithfulness column.
        beh = r["ld_Dp"] / r["ld_D"] if abs(r["ld_D"]) > 1e-9 else float("nan")
        flag = "" if beh > 0.8 else "  <- TASK NOT PRESERVED"
        print(f"{r['tag']:28} {r['F_D']:7.3f} {r['F_Dp']:11.3f} {ret:9.2f} "
              f"{r['ld_D']:7.3f} {r['ld_Dp']:12.3f} {beh:7.2f}{flag}")

    ok = [r for r in rows if abs(r["ld_D"]) > 1e-9 and r["ld_Dp"] / r["ld_D"] > 0.8]
    print(f"\n{len(ok)}/{len(rows)} rows have a task-preserving shift (clean LD "
          f"retained >80%).")
    if ok:
        rets = [r["F_Dp"] / r["F_D"] for r in ok if abs(r["F_D"]) > 1e-9]
        if rets:
            print(f"Faithfulness retention across those: min {min(rets):.2f}, "
                  f"median {sorted(rets)[len(rets)//2]:.2f}, max {max(rets):.2f}")
    print("\nReading it: retention near 1.0 means the circuit transfers across frames, "
          "so\nthe published 0.663 -> 0.228 collapse was about a degenerate "
          "single-template\ndiscovery set. Retention near 0 means frame dependence is "
          "real and general.")


if __name__ == "__main__":
    main()
