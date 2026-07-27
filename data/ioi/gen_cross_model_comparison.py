"""
Cross-model IOI circuit comparison: Llama-3.2-3B vs Pythia-1.4B

Identifies top-10 circuit-critical heads per model from both patching and
ablation scores, maps to relative layer depth (layer / total_layers), and
reports which relative positions are shared vs model-specific.

Outputs: data/ioi/cross-model-comparison.md
"""
import json
from pathlib import Path

DATA_DIR   = Path(__file__).parent
OUTPUT_DIR = DATA_DIR.parent.parent / "data" / "ioi"   # same dir


def load_model_data(patching_path: Path, ablation_path: Path) -> dict:
    p = json.loads(patching_path.read_text())
    a = json.loads(ablation_path.read_text())

    meta = p["meta"]
    n_layers = meta["n_layers"]
    n_heads  = meta["n_heads"]

    patching_scores = p["patching_scores"]   # [n_layers][n_heads]
    drop_scores     = a["drop_scores"]       # [n_layers][n_heads]
    clean_ld        = a["meta"].get("clean_logit_diff_mean", float("nan"))

    # Flatten to (score, layer, head, rel_depth)
    patching_flat = [
        (patching_scores[l][h], l, h, l / n_layers)
        for l in range(n_layers)
        for h in range(n_heads)
    ]
    ablation_flat = [
        (drop_scores[l][h], l, h, l / n_layers)
        for l in range(n_layers)
        for h in range(n_heads)
    ]

    patching_top10 = sorted(patching_flat, reverse=True)[:10]
    ablation_top10 = sorted(ablation_flat, reverse=True)[:10]

    return {
        "model":          meta["model"],
        "n_layers":       n_layers,
        "n_heads":        n_heads,
        "n_examples":     meta["n_examples"],
        "clean_ld":       clean_ld,
        "patching_top10": patching_top10,   # (score, layer, head, rel_depth)
        "ablation_top10": ablation_top10,
    }


def rel_depth_bucket(rel: float, tol: float = 0.08) -> str:
    """Map relative depth to a named thirds bucket."""
    if rel < 1/3 - tol:
        return "early"
    elif rel < 2/3 - tol:
        return "middle"
    else:
        return "late"


def find_shared_positions(
    heads_a: list[tuple],
    heads_b: list[tuple],
    tol: float = 0.08,
) -> tuple[list, list, list]:
    """
    Match top-10 heads across models by relative layer depth ± tol.
    Returns (shared_pairs, only_a, only_b).
    shared_pairs: list of ((score_a, l_a, h_a, rel_a), (score_b, l_b, h_b, rel_b))
    """
    used_b = set()
    shared, only_a = [], []

    for entry_a in heads_a:
        rel_a = entry_a[3]
        match  = None
        best_d = tol + 1
        for j, entry_b in enumerate(heads_b):
            if j in used_b:
                continue
            d = abs(entry_a[3] - entry_b[3])
            if d <= tol and d < best_d:
                best_d = d
                match  = j
        if match is not None:
            shared.append((entry_a, heads_b[match]))
            used_b.add(match)
        else:
            only_a.append(entry_a)

    only_b = [heads_b[j] for j in range(len(heads_b)) if j not in used_b]
    return shared, only_a, only_b


def format_head(score: float, layer: int, head: int, rel: float, n_layers: int) -> str:
    bucket = rel_depth_bucket(rel)
    return f"L{layer:02d}·H{head:02d} (rel={rel:.2f}, {bucket}, score={score:+.4f})"


def generate_report(llama: dict, pythia: dict) -> str:
    tol = 0.08  # ±8% of total layers considered "same relative position"

    lines = [
        "# Cross-Model IOI Circuit Comparison: Llama-3.2-3B vs Pythia-1.4B",
        "",
        "Compares top-10 circuit-critical attention heads identified by activation",
        "patching and mean ablation on the IOI task across two architectures.",
        "Relative layer depth = `layer / total_layers` enables fair comparison.",
        "",
        "---",
        "",
        "## Model Specs",
        "",
        "| Property | Llama-3.2-3B | Pythia-1.4B |",
        "|---|---|---|",
        f"| HuggingFace ID | `{llama['model']}` | `{pythia['model']}` |",
        f"| Architecture | LlamaForCausalLM (GQA) | GPTNeoXForCausalLM (MHA) |",
        f"| Layers | {llama['n_layers']} | {pythia['n_layers']} |",
        f"| Attention heads | {llama['n_heads']} | {pythia['n_heads']} |",
        f"| Clean IOI logit diff (mean) | {llama['clean_ld']:.4f} | {pythia['clean_ld']:.4f} |",
        f"| N examples | {llama['n_examples']} | {pythia['n_examples']} |",
        "",
        "---",
        "",
        "## Top-10 Heads by Activation Patching",
        "",
        "Score = normalised logit-diff recovery. Higher = more causally important.",
        "",
        "### Llama-3.2-3B (28 layers × 24 heads)",
        "",
        "| Rank | Layer | Head | Rel depth | Bucket | Patching score |",
        "|---|---|---|---|---|---|",
    ]

    for rank, (score, l, h, rel) in enumerate(llama["patching_top10"], 1):
        lines.append(f"| {rank} | L{l:02d} | H{h:02d} | {rel:.3f} | {rel_depth_bucket(rel)} | {score:.4f} |")

    lines += [
        "",
        "### Pythia-1.4B (24 layers × 16 heads)",
        "",
        "| Rank | Layer | Head | Rel depth | Bucket | Patching score |",
        "|---|---|---|---|---|---|",
    ]

    for rank, (score, l, h, rel) in enumerate(pythia["patching_top10"], 1):
        lines.append(f"| {rank} | L{l:02d} | H{h:02d} | {rel:.3f} | {rel_depth_bucket(rel)} | {score:.4f} |")

    # Shared positions (patching)
    shared_p, only_llama_p, only_pythia_p = find_shared_positions(
        llama["patching_top10"], pythia["patching_top10"], tol=tol
    )

    lines += [
        "",
        f"### Shared relative positions (patching, ±{tol:.0%} tolerance)",
        "",
    ]
    if shared_p:
        lines.append("| Llama head | Pythia head | Δ rel depth |")
        lines.append("|---|---|---|")
        for (sa, la, ha, ra), (sb, lb, hb, rb) in shared_p:
            lines.append(
                f"| L{la:02d}·H{ha:02d} rel={ra:.2f} ({score_fmt(sa)}) "
                f"| L{lb:02d}·H{hb:02d} rel={rb:.2f} ({score_fmt(sb)}) "
                f"| {abs(ra-rb):.3f} |"
            )
    else:
        lines.append("_No shared relative positions within tolerance._")

    if only_llama_p:
        lines += ["", "**Llama-only patching positions:**"]
        for score, l, h, rel in only_llama_p:
            lines.append(f"- L{l:02d}·H{h:02d} rel={rel:.2f} ({rel_depth_bucket(rel)}) score={score:.4f}")

    if only_pythia_p:
        lines += ["", "**Pythia-only patching positions:**"]
        for score, l, h, rel in only_pythia_p:
            lines.append(f"- L{l:02d}·H{h:02d} rel={rel:.2f} ({rel_depth_bucket(rel)}) score={score:.4f}")

    # ── Ablation section ─────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Top-10 Heads by Mean Ablation",
        "",
        "Score = drop in mean IOI logit diff when head is mean-ablated.",
        "Positive = head contributes positively to IO identification.",
        "",
        "### Llama-3.2-3B",
        "",
        "| Rank | Layer | Head | Rel depth | Bucket | Ablation drop |",
        "|---|---|---|---|---|---|",
    ]

    for rank, (score, l, h, rel) in enumerate(llama["ablation_top10"], 1):
        lines.append(f"| {rank} | L{l:02d} | H{h:02d} | {rel:.3f} | {rel_depth_bucket(rel)} | {score:.4f} |")

    lines += [
        "",
        "### Pythia-1.4B",
        "",
        "| Rank | Layer | Head | Rel depth | Bucket | Ablation drop |",
        "|---|---|---|---|---|---|",
    ]

    for rank, (score, l, h, rel) in enumerate(pythia["ablation_top10"], 1):
        lines.append(f"| {rank} | L{l:02d} | H{h:02d} | {rel:.3f} | {rel_depth_bucket(rel)} | {score:.4f} |")

    shared_a, only_llama_a, only_pythia_a = find_shared_positions(
        llama["ablation_top10"], pythia["ablation_top10"], tol=tol
    )

    lines += [
        "",
        f"### Shared relative positions (ablation, ±{tol:.0%} tolerance)",
        "",
    ]
    if shared_a:
        lines.append("| Llama head | Pythia head | Δ rel depth |")
        lines.append("|---|---|---|")
        for (sa, la, ha, ra), (sb, lb, hb, rb) in shared_a:
            lines.append(
                f"| L{la:02d}·H{ha:02d} rel={ra:.2f} ({score_fmt(sa)}) "
                f"| L{lb:02d}·H{hb:02d} rel={rb:.2f} ({score_fmt(sb)}) "
                f"| {abs(ra-rb):.3f} |"
            )
    else:
        lines.append("_No shared relative positions within tolerance._")

    if only_llama_a:
        lines += ["", "**Llama-only ablation positions:**"]
        for score, l, h, rel in only_llama_a:
            lines.append(f"- L{l:02d}·H{h:02d} rel={rel:.2f} ({rel_depth_bucket(rel)}) drop={score:.4f}")

    if only_pythia_a:
        lines += ["", "**Pythia-only ablation positions:**"]
        for score, l, h, rel in only_pythia_a:
            lines.append(f"- L{l:02d}·H{h:02d} rel={rel:.2f} ({rel_depth_bucket(rel)}) drop={score:.4f}")

    # ── Consensus circuit (heads in both patching and ablation top-10) ────────
    lines += [
        "",
        "---",
        "",
        "## Consensus Circuit Heads",
        "",
        "Heads that appear in BOTH the patching top-10 AND the ablation top-10",
        "for the same model — these are the most robustly identified circuit members.",
        "",
    ]

    for label, data in [("Llama-3.2-3B", llama), ("Pythia-1.4B", pythia)]:
        pat_set = {(l, h) for _, l, h, _ in data["patching_top10"]}
        abl_set = {(l, h) for _, l, h, _ in data["ablation_top10"]}
        consensus = pat_set & abl_set

        # Build score lookup
        pat_scores = {(l, h): s for s, l, h, _ in data["patching_top10"]}
        abl_scores = {(l, h): s for s, l, h, _ in data["ablation_top10"]}
        n = data["n_layers"]

        lines.append(f"### {label} ({len(consensus)} consensus heads)")
        if consensus:
            lines.append("")
            lines.append("| Layer | Head | Rel depth | Bucket | Patching score | Ablation drop |")
            lines.append("|---|---|---|---|---|---|")
            for l, h in sorted(consensus):
                rel = l / n
                lines.append(
                    f"| L{l:02d} | H{h:02d} | {rel:.3f} | {rel_depth_bucket(rel)} "
                    f"| {pat_scores[(l,h)]:.4f} | {abl_scores[(l,h)]:.4f} |"
                )
        else:
            lines.append("")
            lines.append("_No heads appear in both top-10 lists._")
        lines.append("")

    # ── Interpretation ────────────────────────────────────────────────────────
    llama_pat_rels  = [rel for _, _, _, rel in llama["patching_top10"]]
    pythia_pat_rels = [rel for _, _, _, rel in pythia["patching_top10"]]
    llama_abl_rels  = [rel for _, _, _, rel in llama["ablation_top10"]]
    pythia_abl_rels = [rel for _, _, _, rel in pythia["ablation_top10"]]

    def bucket_counts(rels):
        e = sum(1 for r in rels if r < 1/3)
        m = sum(1 for r in rels if 1/3 <= r < 2/3)
        la = sum(1 for r in rels if r >= 2/3)
        return e, m, la

    le_p, lm_p, ll_p = bucket_counts(llama_pat_rels)
    pe_p, pm_p, pl_p = bucket_counts(pythia_pat_rels)
    le_a, lm_a, ll_a = bucket_counts(llama_abl_rels)
    pe_a, pm_a, pl_a = bucket_counts(pythia_abl_rels)

    n_shared_p = len(shared_p)
    n_shared_a = len(shared_a)

    lines += [
        "---",
        "",
        "## Interpretation",
        "",
        "### Layer distribution of top-10 heads",
        "",
        "| Model | Method | Early (<1/3) | Middle (1/3–2/3) | Late (≥2/3) |",
        "|---|---|---|---|---|",
        f"| Llama-3.2-3B | Patching | {le_p} | {lm_p} | {ll_p} |",
        f"| Pythia-1.4B | Patching | {pe_p} | {pm_p} | {pl_p} |",
        f"| Llama-3.2-3B | Ablation | {le_a} | {lm_a} | {ll_a} |",
        f"| Pythia-1.4B | Ablation | {pe_a} | {pm_a} | {pl_a} |",
        "",
        f"### Cross-model alignment (±{tol:.0%} relative depth tolerance)",
        "",
        f"- **Patching:** {n_shared_p}/10 positions shared, "
        f"{len(only_llama_p)} Llama-only, {len(only_pythia_p)} Pythia-only",
        f"- **Ablation:** {n_shared_a}/10 positions shared, "
        f"{len(only_llama_a)} Llama-only, {len(only_pythia_a)} Pythia-only",
        "",
    ]

    if n_shared_p >= 4 or n_shared_a >= 4:
        lines.append(
            "**Finding:** Substantial overlap in relative circuit positions across architectures, "
            "consistent with H2 (cross-model transfer) in the experimental brief. "
            "The IOI circuit appears to occupy conserved relative-depth slots despite "
            "different absolute layer counts (28 vs 24) and attention mechanisms (GQA vs MHA)."
        )
    elif n_shared_p >= 2 or n_shared_a >= 2:
        lines.append(
            "**Finding:** Partial overlap in relative circuit positions. "
            "Some functional slots appear conserved across architectures, "
            "but most critical heads occupy model-specific positions. "
            "H2 (cross-model transfer) receives weak support."
        )
    else:
        lines.append(
            "**Finding:** Minimal overlap in relative circuit positions. "
            "Critical heads for IOI are largely model-specific, suggesting "
            "the IOI circuit topology does not transfer across these architectures. "
            "H2 (cross-model transfer) is not supported."
        )

    lines += [
        "",
        "---",
        "",
        f"_Generated from real experimental data. "
        f"Llama: {llama['n_examples']} examples; Pythia: {pythia['n_examples']} examples. "
        f"Tolerance for shared-position matching: ±{tol:.0%} relative depth._",
    ]

    return "\n".join(lines)


def score_fmt(s: float) -> str:
    return f"{s:+.4f}"


def main():
    llama  = load_model_data(
        DATA_DIR / "patching-llama3b.json",
        DATA_DIR / "ablation-llama3b.json",
    )
    # Prior session produced pythia1b.json; our new scripts produce pythia1.4b.json.
    # Use whichever exists.
    for pat_name, abl_name in [
        ("patching-pythia1b.json",   "ablation-pythia1b.json"),
        ("patching-pythia1.4b.json", "ablation-pythia1.4b.json"),
    ]:
        if (DATA_DIR / pat_name).exists() and (DATA_DIR / abl_name).exists():
            pythia = load_model_data(DATA_DIR / pat_name, DATA_DIR / abl_name)
            break
    else:
        raise FileNotFoundError(
            "Pythia patching/ablation data not found. "
            "Run run_patching_pythia1b.py and run_ablation_pythia1b.py first."
        )

    report = generate_report(llama, pythia)
    out    = DATA_DIR / "cross-model-comparison.md"
    out.write_text(report)
    print(f"Saved → {out}")
    print(report)


if __name__ == "__main__":
    main()
