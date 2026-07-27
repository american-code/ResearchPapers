#!/usr/bin/env python3
"""
Generate (or load) IOI activation patching data for Llama-3.2-3B and produce
an SVG heatmap: rows = layers, columns = attention heads, colour = patching
importance (normalised logit-diff recovery).

Usage:
    python gen_patching_heatmap.py

Outputs:
    data/ioi/patching-llama3b.json   (synthetic placeholder if missing)
    figures/ioi-patching-heatmap-llama3b.svg
"""
import json, random
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # ResearchPapers/
DATA_DIR = ROOT / "data" / "ioi"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

JSON_PATH = DATA_DIR / "patching-llama3b.json"
SVG_PATH  = FIG_DIR / "ioi-patching-heatmap-llama3b.svg"

# ---------------------------------------------------------------------------
# Synthetic data generation
# Llama-3.2-3B: 28 layers, 24 attention heads per layer
# Scores = normalised logit-diff recovery ∈ [0, 1]
# ---------------------------------------------------------------------------
N_LAYERS = 28
N_HEADS  = 24

def make_synthetic_scores(seed: int = 42) -> list[list[float]]:
    rng = random.Random(seed)

    # Background: small positive noise
    scores = [[max(0.0, rng.gauss(0.025, 0.025)) for _ in range(N_HEADS)]
              for _ in range(N_LAYERS)]

    # IOI circuit heads (scaled from GPT-2 Small results to 28-layer depth)
    # Format: (layer, head, score)
    circuit_heads = [
        # Name-mover heads — late layers, highest patching importance
        (22,  7, 0.912),
        (24,  2, 0.871),
        (20, 14, 0.833),
        (26, 11, 0.571),
        (18, 20, 0.281),
        (25, 15, 0.262),
        # S-inhibition heads — middle layers
        (14,  9, 0.721),
        (16,  4, 0.682),
        (12, 17, 0.614),
        (11,  1, 0.231),
        # Induction heads — early-middle layers
        ( 8, 12, 0.374),
        ( 9,  6, 0.347),
        # Duplicate-token / previous-token heads — early layers
        ( 4,  3, 0.441),
        ( 3, 19, 0.403),
        ( 6, 22, 0.312),
        ( 5,  8, 0.198),
    ]
    for l, h, v in circuit_heads:
        scores[l][h] = v

    return scores


def ensure_data() -> dict:
    if JSON_PATH.exists():
        return json.loads(JSON_PATH.read_text())

    scores = make_synthetic_scores()
    data = {
        "meta": {
            "model": "mlx-community/Llama-3.2-3B-bf16",
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "n_examples": 100,
            "metric": "normalized_logit_diff_recovery",
            "description": (
                "Per-head activation patching. Score = mean over examples of "
                "(logit_diff_patched - logit_diff_corrupted) / "
                "(logit_diff_clean - logit_diff_corrupted). "
                "Corrupted = IO and S names swapped."
            ),
            "synthetic": True,
            "note": "Placeholder — run run_patching_llama3b.py to replace with real results.",
        },
        "patching_scores": scores,
    }
    JSON_PATH.write_text(json.dumps(data, indent=2))
    print(f"Created synthetic placeholder: {JSON_PATH}")
    return data


# ---------------------------------------------------------------------------
# Colour map — ColorBrewer "Reds" sequential (9 stops), white → dark red
# ---------------------------------------------------------------------------
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
            return f"#{r:02x}{g:02x}{b:02x}"
    return f"#{_CMAP[-1][1]:02x}{_CMAP[-1][2]:02x}{_CMAP[-1][3]:02x}"


# ---------------------------------------------------------------------------
# SVG layout constants
# ---------------------------------------------------------------------------
CELL          = 20        # px per grid cell
ML            = 50        # left margin  (layer labels)
MT_TITLE      = 28        # title height
MT_AXIS       = 45        # top margin after title (head labels)
MB            = 38        # bottom margin (layer axis label + tick labels)
MR            = 92        # right margin (colorbar + legend)

CB_W          = 16        # colorbar width
CB_H          = 220       # colorbar height
CB_OFFSET_X   = 18        # gap between grid and colorbar
CB_STEPS      = 100       # gradient rectangles


def generate_svg(scores: list[list[float]], meta: dict, top_n: int = 5) -> str:
    n_layers = len(scores)
    n_heads  = len(scores[0])

    grid_w = n_heads  * CELL
    grid_h = n_layers * CELL

    total_w = ML + grid_w + MR
    total_h = MT_TITLE + MT_AXIS + grid_h + MB

    # Identify top-N heads
    all_vals = [(scores[l][h], l, h) for l in range(n_layers) for h in range(n_heads)]
    top_heads = sorted(all_vals, reverse=True)[:top_n]
    top_set   = {(l, h) for _, l, h in top_heads}

    G = MT_TITLE + MT_AXIS  # y-origin of grid

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_w}" height="{total_h}" '
        f'font-family="Helvetica,Arial,sans-serif">',
        f'<rect width="{total_w}" height="{total_h}" fill="#ffffff"/>',
    ]

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    cx = ML + grid_w // 2
    model_label = meta.get("model", "").replace("mlx-community/", "")
    title = f"IOI Activation Patching — {model_label}"
    parts.append(
        f'<text x="{cx}" y="20" text-anchor="middle" '
        f'font-size="12" font-weight="bold" fill="#1a1a1a">{title}</text>'
    )
    if meta.get("synthetic"):
        parts.append(
            f'<text x="{cx}" y="36" text-anchor="middle" '
            f'font-size="9" font-style="italic" fill="#888">'
            f'[synthetic placeholder — run run_patching_llama3b.py for real results]</text>'
        )

    # ------------------------------------------------------------------
    # Axis labels
    # ------------------------------------------------------------------
    # "Attention Head" centred above grid
    parts.append(
        f'<text x="{ML + grid_w // 2}" y="{MT_TITLE + 14}" '
        f'text-anchor="middle" font-size="10" fill="#444">Attention Head</text>'
    )
    # "Layer" rotated left of grid
    ly = G + grid_h // 2
    lx = 13
    parts.append(
        f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="10" fill="#444" '
        f'transform="rotate(-90,{lx},{ly})">Layer</text>'
    )

    # ------------------------------------------------------------------
    # Tick labels — heads (top)
    # ------------------------------------------------------------------
    for h in range(n_heads):
        if h % 4 == 0 or h == n_heads - 1:
            x = ML + h * CELL + CELL // 2
            y = MT_TITLE + MT_AXIS - 8
            parts.append(
                f'<text x="{x}" y="{y}" text-anchor="middle" font-size="8" fill="#555">{h}</text>'
            )

    # ------------------------------------------------------------------
    # Tick labels — layers (left)
    # ------------------------------------------------------------------
    for l in range(n_layers):
        if l % 4 == 0 or l == n_layers - 1:
            x = ML - 4
            y = G + l * CELL + CELL // 2 + 3
            parts.append(
                f'<text x="{x}" y="{y}" text-anchor="end" font-size="8" fill="#555">{l}</text>'
            )

    # ------------------------------------------------------------------
    # Grid cells
    # ------------------------------------------------------------------
    for l in range(n_layers):
        for h in range(n_heads):
            v     = scores[l][h]
            color = val_to_hex(v)
            cx_   = ML + h * CELL
            cy_   = G  + l * CELL
            parts.append(
                f'<rect x="{cx_}" y="{cy_}" width="{CELL}" height="{CELL}" '
                f'fill="{color}" stroke="#e8e8e8" stroke-width="0.3"/>'
            )
            # Highlight top-N with white border + value label
            if (l, h) in top_set:
                parts.append(
                    f'<rect x="{cx_+1}" y="{cy_+1}" width="{CELL-2}" height="{CELL-2}" '
                    f'fill="none" stroke="white" stroke-width="1.8"/>'
                )
                parts.append(
                    f'<text x="{cx_ + CELL//2}" y="{cy_ + CELL//2 + 4}" '
                    f'text-anchor="middle" font-size="6" font-weight="bold" fill="white">'
                    f'{v:.2f}</text>'
                )

    # ------------------------------------------------------------------
    # Colorbar
    # ------------------------------------------------------------------
    cb_x = ML + grid_w + CB_OFFSET_X
    cb_y = G

    step_h = CB_H / CB_STEPS
    for i in range(CB_STEPS):
        v     = 1.0 - i / (CB_STEPS - 1)
        color = val_to_hex(v)
        y_pos = cb_y + i * step_h
        parts.append(
            f'<rect x="{cb_x}" y="{y_pos:.1f}" '
            f'width="{CB_W}" height="{step_h + 0.5:.2f}" fill="{color}"/>'
        )
    # border
    parts.append(
        f'<rect x="{cb_x}" y="{cb_y}" width="{CB_W}" height="{CB_H}" '
        f'fill="none" stroke="#999" stroke-width="0.8"/>'
    )
    # tick labels
    for v_tick in [1.0, 0.75, 0.5, 0.25, 0.0]:
        y_pos = cb_y + (1.0 - v_tick) * CB_H
        parts.append(
            f'<text x="{cb_x + CB_W + 4}" y="{y_pos + 3}" '
            f'font-size="8" fill="#444">{v_tick:.2f}</text>'
        )
    # colorbar title (two lines)
    cb_cx = cb_x + CB_W // 2
    parts.append(
        f'<text x="{cb_cx}" y="{cb_y + CB_H + 14}" '
        f'text-anchor="middle" font-size="8" fill="#444">logit-diff</text>'
    )
    parts.append(
        f'<text x="{cb_cx}" y="{cb_y + CB_H + 24}" '
        f'text-anchor="middle" font-size="8" fill="#444">recovery</text>'
    )

    # ------------------------------------------------------------------
    # Top-N legend below colorbar
    # ------------------------------------------------------------------
    leg_y = cb_y + CB_H + 44
    parts.append(
        f'<text x="{cb_x}" y="{leg_y}" font-size="8" font-weight="bold" fill="#333">Top {top_n}:</text>'
    )
    for rank, (v, l, h) in enumerate(top_heads):
        parts.append(
            f'<text x="{cb_x}" y="{leg_y + 12*(rank+1)}" font-size="7.5" fill="#444">'
            f'L{l:02d}·H{h:02d}  {v:.3f}</text>'
        )

    # ------------------------------------------------------------------
    # n_examples label bottom-right
    # ------------------------------------------------------------------
    n_ex = meta.get("n_examples", "?")
    parts.append(
        f'<text x="{total_w - 4}" y="{total_h - 4}" '
        f'text-anchor="end" font-size="7.5" fill="#aaa">n={n_ex} examples</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data   = ensure_data()
    meta   = data["meta"]
    scores = data["patching_scores"]

    svg = generate_svg(scores, meta, top_n=5)
    SVG_PATH.write_text(svg)
    print(f"Saved heatmap → {SVG_PATH}")
