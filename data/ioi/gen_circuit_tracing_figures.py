#!/usr/bin/env python3
"""
Generate publication-quality SVG figures for the IOI circuit-tracing paper.

Outputs to figures/circuit-tracing/:
  1. ioi-patching-heatmap-llama3b.svg  — per-head activation patching, Llama-3.2-3B
  2. ioi-cross-model-comparison.svg    — Llama-3.2-3B vs Pythia-1.4B side-by-side
  3. ioi-circuit-diagram.svg           — IOI subgraph schematic (functional head groups)

Color palette is consistent across all three figures.
"""
import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent.parent
DATA    = ROOT / "data" / "ioi"
FIG_DIR = ROOT / "figures" / "circuit-tracing"
FIG_DIR.mkdir(parents=True, exist_ok=True)

F = "Helvetica,Arial,sans-serif"

# ── Consistent color palette ─────────────────────────────────────────────────
C = {
    "nm":  "#c0392b",   # name-mover   — deep red
    "si":  "#d35400",   # S-inhibition — burnt orange
    "ind": "#2471a3",   # induction    — steel blue
    "dt":  "#1e8449",   # dup-token    — forest green
    "bg":  "#ffffff",
    "gr":  "#e8e8e8",
    "tx":  "#1a1a1a",
    "ax":  "#555555",
    "dm":  "#aaaaaa",
}

# Zone background fills (very light tints)
ZONE_BG = {"nm": "#fff2f1", "si": "#fff7ef", "ind": "#eff6ff", "dt": "#effff4"}

# Functional zones: (label, color_key, rel_lo, rel_hi_exclusive)
ROLES = [
    ("Name-mover",   "nm",  0.64, 1.01),
    ("S-inhibition", "si",  0.38, 0.64),
    ("Induction",    "ind", 0.25, 0.38),
    ("Dup-token",    "dt",  0.00, 0.25),
]

# ── Colormap: ColorBrewer "Reds" sequential ──────────────────────────────────
_REDS = [
    (0.000, 255, 245, 240), (0.125, 254, 224, 210),
    (0.250, 252, 187, 161), (0.375, 252, 146, 114),
    (0.500, 251, 106,  74), (0.625, 239,  59,  44),
    (0.750, 203,  24,  29), (0.875, 165,  15,  21),
    (1.000, 103,   0,  13),
]


def red(v: float, vmax: float = 1.0) -> str:
    v = max(0.0, min(1.0, v / max(vmax, 1e-9)))
    for i in range(len(_REDS) - 1):
        v0, r0, g0, b0 = _REDS[i]
        v1, r1, g1, b1 = _REDS[i + 1]
        if v <= v1 + 1e-9:
            t = (v - v0) / (v1 - v0) if v1 > v0 else 0.0
            return (f"#{int(r0+t*(r1-r0)):02x}"
                    f"{int(g0+t*(g1-g0)):02x}"
                    f"{int(b0+t*(b1-b0)):02x}")
    r, g, b = _REDS[-1][1], _REDS[-1][2], _REDS[-1][3]
    return f"#{r:02x}{g:02x}{b:02x}"


def colorbar_svg(x: int, y: int, h: int, w: int, vmax: float, steps: int = 80) -> str:
    parts = []
    sh = h / steps
    for i in range(steps):
        v = 1.0 - i / (steps - 1)
        parts.append(
            f'<rect x="{x}" y="{y + i*sh:.1f}" width="{w}" '
            f'height="{sh + 0.5:.2f}" fill="{red(v * vmax, vmax)}"/>'
        )
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="none" stroke="#999" stroke-width="0.8"/>'
    )
    for vt in [1.0, 0.75, 0.5, 0.25, 0.0]:
        yt = y + (1.0 - vt) * h
        parts.append(
            f'<text x="{x + w + 4}" y="{yt + 3:.1f}" '
            f'font-family="{F}" font-size="8" fill="{C["ax"]}">{vt * vmax:.3f}</text>'
        )
    cx = x + w // 2
    parts.append(
        f'<text x="{cx}" y="{y + h + 14}" text-anchor="middle" '
        f'font-family="{F}" font-size="8" fill="{C["ax"]}">logit-diff</text>'
    )
    parts.append(
        f'<text x="{cx}" y="{y + h + 24}" text-anchor="middle" '
        f'font-family="{F}" font-size="8" fill="{C["ax"]}">recovery</text>'
    )
    return "\n".join(parts)


def heatmap_cells(x0: int, y0: int, scores, nl: int, nh: int,
                  cell: int, vmax: float, highlight: set | None = None) -> str:
    parts = []
    for l in range(nl):
        for h in range(nh):
            v = scores[l][h]
            cx = x0 + h * cell
            cy = y0 + l * cell
            parts.append(
                f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" '
                f'fill="{red(v, vmax)}" stroke="{C["gr"]}" stroke-width="0.3"/>'
            )
            if highlight and (l, h) in highlight:
                parts.append(
                    f'<rect x="{cx+1}" y="{cy+1}" width="{cell-2}" height="{cell-2}" '
                    f'fill="none" stroke="white" stroke-width="1.5"/>'
                )
                fc = "white" if v > 0.07 else C["tx"]
                parts.append(
                    f'<text x="{cx + cell//2}" y="{cy + cell//2 + 3}" '
                    f'text-anchor="middle" font-family="{F}" font-size="6" '
                    f'font-weight="bold" fill="{fc}">{v:.3f}</text>'
                )
    parts.append(
        f'<rect x="{x0}" y="{y0}" width="{nh * cell}" height="{nl * cell}" '
        f'fill="none" stroke="#999" stroke-width="0.8"/>'
    )
    return "\n".join(parts)


# =============================================================================
# Figure 1 — IOI Patching Heatmap: Llama-3.2-3B
# =============================================================================

def fig1_heatmap() -> None:
    d    = json.loads((DATA / "patching-llama3b.json").read_text())
    meta = d["meta"]
    sc   = d["patching_scores"]
    nl, nh = meta["n_layers"], meta["n_heads"]  # 28, 24

    CELL = 20
    ML   = 82    # left margin (role bar + gap + layer ticks)
    MT   = 74    # top (title + subtitle + head axis)
    MB   = 52
    MR   = 158   # right (colorbar + legend)

    gw = nh * CELL   # 480
    gh = nl * CELL   # 560
    W  = ML + gw + MR
    H  = MT + gh + MB

    flat   = [(sc[l][h], l, h) for l in range(nl) for h in range(nh)]
    vmax   = max(v for v, _, _ in flat)
    top10  = {(l, h) for _, l, h in sorted(flat, reverse=True)[:10]}

    cx_grid = ML + gw // 2

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}">',
        f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>',
        # Title
        f'<text x="{cx_grid}" y="20" text-anchor="middle" font-size="13" '
        f'font-weight="bold" fill="{C["tx"]}">'
        f'IOI Activation Patching — Llama-3.2-3B</text>',
        f'<text x="{cx_grid}" y="36" text-anchor="middle" font-size="9" fill="{C["dm"]}">'
        f'n={meta["n_examples"]} examples · normalised logit-diff recovery · per attention head</text>',
        # Head axis label
        f'<text x="{cx_grid}" y="{MT - 6}" text-anchor="middle" '
        f'font-size="10" fill="{C["ax"]}">Attention head</text>',
    ]

    # Head tick marks
    for h in range(nh):
        if h % 4 == 0 or h == nh - 1:
            x = ML + h * CELL + CELL // 2
            p.append(
                f'<text x="{x}" y="{MT - 18}" text-anchor="middle" '
                f'font-size="8" fill="{C["ax"]}">{h}</text>'
            )

    # Layer axis label (rotated)
    lx, ly = 18, MT + gh // 2
    p.append(
        f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="10" fill="{C["ax"]}" '
        f'transform="rotate(-90,{lx},{ly})">Layer</text>'
    )

    # Layer tick marks + functional role bars
    for l in range(nl):
        if l % 4 == 0 or l == nl - 1:
            y = MT + l * CELL + CELL // 2 + 3
            p.append(
                f'<text x="{ML - 10}" y="{y}" text-anchor="end" '
                f'font-size="8" fill="{C["ax"]}">{l}</text>'
            )

    # Functional role bars (left edge, 5 px wide)
    for _, ck, rlo, rhi in ROLES:
        y_top = MT + int(rlo * nl) * CELL
        y_bot = MT + min(int(rhi * nl), nl) * CELL
        p.append(
            f'<rect x="34" y="{y_top}" width="5" height="{y_bot - y_top}" '
            f'fill="{C[ck]}" rx="2"/>'
        )

    # Heatmap
    p.append(heatmap_cells(ML, MT, sc, nl, nh, CELL, vmax, top10))

    # Colorbar
    cb_x = ML + gw + 18
    p.append(colorbar_svg(cb_x, MT, 224, 16, vmax))

    # Circuit role legend
    leg_x = cb_x
    leg_y = MT + 250
    p.append(
        f'<text x="{leg_x}" y="{leg_y}" font-size="9" font-weight="bold" fill="{C["tx"]}">Circuit role zones:</text>'
    )
    for i, (lbl, ck, rlo, rhi) in enumerate(ROLES):
        y    = leg_y + 14 * (i + 1)
        lrng = f"L{int(rlo * nl)}–{min(int(rhi * nl), nl) - 1}"
        p += [
            f'<rect x="{leg_x}" y="{y - 8}" width="9" height="9" fill="{C[ck]}" rx="2"/>',
            f'<text x="{leg_x + 13}" y="{y}" font-size="8" fill="{C["tx"]}">{lbl} ({lrng})</text>',
        ]

    # Top-5 head table
    t5_y = leg_y + 82
    p.append(
        f'<text x="{leg_x}" y="{t5_y}" font-size="9" font-weight="bold" fill="{C["tx"]}">Top 5 heads:</text>'
    )
    for rank, (v, l, h) in enumerate(sorted(flat, reverse=True)[:5], 1):
        rel  = l / nl
        rcol = next((C[ck] for _, ck, rlo, rhi in ROLES if rlo <= rel < rhi), C["dm"])
        p.append(
            f'<text x="{leg_x}" y="{t5_y + 12 * rank}" font-size="7.5" fill="{rcol}">'
            f'#{rank} L{l:02d}·H{h:02d}  {v:.4f}</text>'
        )

    p.append("</svg>")
    out = FIG_DIR / "ioi-patching-heatmap-llama3b.svg"
    out.write_text("\n".join(p))
    print(f"[1/3] Saved → {out}")


# =============================================================================
# Figure 2 — Cross-model comparison: Llama-3.2-3B vs Pythia-1.4B
# =============================================================================

def fig2_cross_model() -> None:
    ld  = json.loads((DATA / "patching-llama3b.json").read_text())
    pd_ = json.loads((DATA / "patching-pythia1b.json").read_text())
    l_sc = ld["patching_scores"];  p_sc = pd_["patching_scores"]
    l_nl, l_nh = ld["meta"]["n_layers"], ld["meta"]["n_heads"]   # 28, 24
    p_nl, p_nh = pd_["meta"]["n_layers"], pd_["meta"]["n_heads"]  # 24, 16

    vmax = max(
        max(l_sc[l][h] for l in range(l_nl) for h in range(l_nh)),
        max(p_sc[l][h] for l in range(p_nl) for h in range(p_nh)),
    )

    CELL = 14; GAP = 52
    MT = 90; MB = 52; ML = 56; MR = 108
    l_gw = l_nh * CELL; l_gh = l_nl * CELL   # 336 × 392
    p_gw = p_nh * CELL; p_gh = p_nl * CELL   # 224 × 336
    p_x0 = ML + l_gw + GAP

    W = ML + l_gw + GAP + p_gw + MR
    H = MT + max(l_gh, p_gh) + MB

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}">',
        f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>',
        # Main title
        f'<text x="{W // 2}" y="22" text-anchor="middle" font-size="13" '
        f'font-weight="bold" fill="{C["tx"]}">IOI Circuit Comparison: Llama-3.2-3B vs Pythia-1.4B</text>',
        f'<text x="{W // 2}" y="37" text-anchor="middle" font-size="9" fill="{C["dm"]}">'
        f'Shared colour scale · normalised logit-diff recovery · n=100 examples per model</text>',
        # Shared "Layer" y-axis label
        f'<text x="14" y="{MT + l_gh // 2}" text-anchor="middle" font-size="10" fill="{C["ax"]}" '
        f'transform="rotate(-90,14,{MT + l_gh // 2})">Layer</text>',
        # rel. depth column header (left of left panel)
        f'<text x="{ML - 24}" y="{MT - 14}" text-anchor="end" font-size="7" fill="{C["dm"]}">rel.</text>',
        f'<text x="{ML - 24}" y="{MT - 5}" text-anchor="end" font-size="7" fill="{C["dm"]}">depth</text>',
    ]

    def draw_panel(label: str, model_name: str, scores, nl: int, nh: int, x0: int) -> None:
        gw = nh * CELL
        gh = nl * CELL
        is_left = (x0 == ML)

        # Panel letter + model name
        p.append(
            f'<text x="{x0}" y="52" font-size="13" font-weight="bold" fill="{C["tx"]}">{label}</text>'
        )
        p.append(
            f'<text x="{x0 + gw // 2}" y="68" text-anchor="middle" font-size="11" '
            f'font-weight="bold" fill="{C["tx"]}">{model_name}</text>'
        )
        p.append(
            f'<text x="{x0 + gw // 2}" y="{MT - 5}" text-anchor="middle" '
            f'font-size="8" fill="{C["ax"]}">Attention head</text>'
        )

        # Head ticks
        for h in range(nh):
            if h % 4 == 0 or h == nh - 1:
                x = x0 + h * CELL + CELL // 2
                p.append(
                    f'<text x="{x}" y="{MT - 15}" text-anchor="middle" '
                    f'font-size="7" fill="{C["ax"]}">{h}</text>'
                )

        # Relative-depth dashed dividers at 1/3 and 2/3
        for frac in [1 / 3, 2 / 3]:
            y_ = MT + int(frac * nl) * CELL
            p.append(
                f'<line x1="{x0}" y1="{y_}" x2="{x0 + gw}" y2="{y_}" '
                f'stroke="{C["dm"]}" stroke-width="0.8" stroke-dasharray="4,3"/>'
            )

        # Layer ticks
        for l in range(nl):
            if l % 4 == 0 or l == nl - 1:
                y = MT + l * CELL + CELL // 2 + 3
                if is_left:
                    p.append(
                        f'<text x="{x0 - 8}" y="{y}" text-anchor="end" '
                        f'font-size="7" fill="{C["ax"]}">{l}</text>'
                    )
                    # Relative depth annotation
                    rel = l / nl
                    p.append(
                        f'<text x="{x0 - 24}" y="{y}" text-anchor="end" '
                        f'font-size="6.5" fill="{C["dm"]}">{rel:.2f}</text>'
                    )
                else:
                    p.append(
                        f'<text x="{x0 + gw + 6}" y="{y}" '
                        f'font-size="7" fill="{C["ax"]}">{l}</text>'
                    )

        # Heatmap
        p.append(heatmap_cells(x0, MT, scores, nl, nh, CELL, vmax))

        # Functional role bars (right of grid, 4 px wide)
        bar_x = x0 + gw + 2
        for _, ck, rlo, rhi in ROLES:
            y_top = MT + int(rlo * nl) * CELL
            y_bot = MT + min(int(rhi * nl), nl) * CELL
            p.append(
                f'<rect x="{bar_x}" y="{y_top}" width="4" height="{y_bot - y_top}" '
                f'fill="{C[ck]}" rx="1"/>'
            )

    draw_panel("(A)", "Llama-3.2-3B  (28L × 24H)", l_sc, l_nl, l_nh, ML)
    draw_panel("(B)", "Pythia-1.4B  (24L × 16H)",  p_sc, p_nl, p_nh, p_x0)

    # Shared colorbar
    cb_x  = p_x0 + p_gw + 22
    cb_h  = (max(l_gh, p_gh) * 2) // 3
    p.append(colorbar_svg(cb_x, MT, cb_h, 16, vmax))

    # Role legend
    leg_x = cb_x
    leg_y = MT + cb_h + 36
    p.append(
        f'<text x="{leg_x}" y="{leg_y}" font-size="8.5" font-weight="bold" fill="{C["tx"]}">Circuit roles:</text>'
    )
    for i, (lbl, ck, rlo, rhi) in enumerate(ROLES):
        y = leg_y + 13 * (i + 1)
        p += [
            f'<rect x="{leg_x}" y="{y - 8}" width="9" height="9" fill="{C[ck]}" rx="2"/>',
            f'<text x="{leg_x + 13}" y="{y}" font-size="7.5" fill="{C["tx"]}">'
            f'{lbl} (rel {rlo:.2f}–{min(rhi, 1.0):.2f})</text>',
        ]
    p.append(
        f'<text x="{leg_x}" y="{leg_y + 68}" font-size="7" fill="{C["dm"]}">'
        f'‒‒‒ rel. depth ⅓, ⅔</text>'
    )

    p.append("</svg>")
    out = FIG_DIR / "ioi-cross-model-comparison.svg"
    out.write_text("\n".join(p))
    print(f"[2/3] Saved → {out}")


# =============================================================================
# Figure 3 — IOI Circuit Diagram (path-patching subgraph)
# =============================================================================

def fig3_circuit_diagram() -> None:
    """
    Schematic of the IOI subgraph for Llama-3.2-3B.

    Y-axis: functional depth (top = late layers = output, bottom = early = input).
    X-axis: 4 key sequence positions (IO, S1, S2, END).
    Head groups shown as colored circles at their attending (query) position.
    Dashed arrows: attention K,V source (what the head reads from).
    Solid arrows: residual stream contribution flow.
    """
    d  = json.loads((DATA / "patching-llama3b.json").read_text())
    nl = d["meta"]["n_layers"]   # 28

    W, H = 680, 570
    LEGEND_W = 148
    TITLE_H  = 52
    INPUT_H  = 60

    # Diagram bounding box
    dx0 = 28
    dx1 = W - LEGEND_W - 8
    dy0 = TITLE_H
    dy1 = H - INPUT_H - 24
    dw  = dx1 - dx0
    dh  = dy1 - dy0

    # 4 sequence positions (x-coordinates)
    POS_LABELS = ["IO", "S₁", "S₂", "END"]
    POS_DESC   = ["Indirect\nobject", "Subject\n(1st)", "Subject\n(2nd)", "Final\ntoken"]
    pos_x = [dx0 + int((i + 0.5) * dw / 4) for i in range(4)]
    x_IO, x_S1, x_S2, x_END = pos_x

    def zone_y(rel_lo: float, rel_hi: float) -> tuple[int, int]:
        # rel=0 → bottom (dy1), rel=1 → top (dy0)
        rel_hi = min(rel_hi, 1.0)
        y_top = dy0 + int((1.0 - rel_hi) * dh)
        y_bot = dy0 + int((1.0 - rel_lo) * dh)
        return y_top, y_bot

    p: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="{F}">',
        '<defs>',
    ]
    # Arrow markers
    for ck in ["nm", "si", "ind", "dt"]:
        p.append(
            f'<marker id="a-{ck}" markerWidth="10" markerHeight="10" '
            f'refX="8" refY="3" orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L0,6 L9,3 z" fill="{C[ck]}"/></marker>'
        )
    p.append(
        '<marker id="a-flow" markerWidth="10" markerHeight="10" '
        'refX="8" refY="3" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#999"/></marker>'
    )
    p.append('</defs>')

    p += [
        f'<rect width="{W}" height="{H}" fill="{C["bg"]}"/>',
        f'<text x="{(dx0 + dx1) // 2}" y="24" text-anchor="middle" font-size="13" '
        f'font-weight="bold" fill="{C["tx"]}">IOI Circuit Subgraph — Llama-3.2-3B</text>',
        f'<text x="{(dx0 + dx1) // 2}" y="40" text-anchor="middle" font-size="9" fill="{C["dm"]}">'
        f'Functional head groups by relative layer depth · path-patching evidence</text>',
    ]

    # Zone backgrounds and left labels
    for lbl, ck, rlo, rhi in ROLES:
        yt, yb = zone_y(rlo, rhi)
        p.append(
            f'<rect x="{dx0}" y="{yt}" width="{dw}" height="{yb - yt}" '
            f'fill="{ZONE_BG[ck]}" rx="3"/>'
        )
        # Left zone label
        cy = (yt + yb) // 2
        p.append(
            f'<text x="{dx0 - 6}" y="{cy + 4}" text-anchor="end" '
            f'font-size="9" font-weight="bold" fill="{C[ck]}">{lbl}</text>'
        )
        # Right: layer range
        lr = f"L{int(rlo*nl)}–{min(int(rhi*nl), nl)-1}"
        p.append(
            f'<text x="{dx1 + 4}" y="{cy + 4}" font-size="7.5" fill="{C[ck]}">{lr}</text>'
        )

    # Vertical residual stream lines
    for px in pos_x:
        p.append(
            f'<line x1="{px}" y1="{dy0}" x2="{px}" y2="{dy1}" '
            f'stroke="{C["gr"]}" stroke-width="1.5" stroke-dasharray="2,2"/>'
        )

    # ── Head group nodes + attention arrows ──────────────────────────────────
    # (label, ck, query_pos, kv_pos, rel_lo, rel_hi, top_heads_str)
    HEAD_GROUPS = [
        ("NM",  "nm",  x_END, x_IO,  0.64, 1.01,
         "L24·H15 (0.148)\nL21·H20 (0.073)"),
        ("SI",  "si",  x_END, x_S2,  0.38, 0.64,
         "L15·H20 (0.240)†\nL14·H00 (0.111)"),
        ("IND", "ind", x_S2,  x_IO,  0.25, 0.38,
         "weak individual\npatching signal"),
        ("DT",  "dt",  x_S2,  x_S1,  0.00, 0.25,
         "weak individual\npatching signal"),
    ]

    node_pos: dict[str, tuple[int, int]] = {}
    R = 18  # circle radius

    for abbr, ck, qx, kvx, rlo, rhi, hlbl in HEAD_GROUPS:
        yt, yb = zone_y(rlo, rhi)
        cy = (yt + yb) // 2
        cx = qx
        node_pos[abbr] = (cx, cy)

        # Dashed attention arrow: KV source → node edge
        if kvx != qx:
            ex = cx - R - 2 if kvx < qx else cx + R + 2
            ctrl_y = cy - 28
            p.append(
                f'<path d="M{kvx},{cy} Q{(kvx+qx)//2},{ctrl_y} {ex},{cy}" '
                f'stroke="{C[ck]}" stroke-width="2" fill="none" '
                f'stroke-dasharray="5,3" marker-end="url(#a-{ck})"/>'
            )
            # K,V label at arc midpoint
            lx_ = (kvx + qx) // 2
            p.append(
                f'<text x="{lx_}" y="{ctrl_y - 4}" text-anchor="middle" '
                f'font-size="8" font-style="italic" fill="{C[ck]}">K,V</text>'
            )
            # Q label near head node
            p.append(
                f'<text x="{cx}" y="{cy + R + 14}" text-anchor="middle" '
                f'font-size="7.5" font-style="italic" fill="{C[ck]}">Q</text>'
            )

        # Node circle
        p.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="{C[ck]}" opacity="0.88"/>')
        p.append(
            f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" '
            f'font-size="9" font-weight="bold" fill="white">{abbr}</text>'
        )

        # Head label to the right of the node (or left for END-column nodes)
        lx_off = cx + R + 8
        for j, line in enumerate(hlbl.split("\n")):
            p.append(
                f'<text x="{lx_off}" y="{cy - 4 + j * 12}" '
                f'font-size="7.5" fill="{C[ck]}">{line}</text>'
            )

    # ── Residual stream contribution arrows ──────────────────────────────────
    # Dup-token at S2 → S-inhibition at S2 (via S2 residual stream)
    dt_cx, dt_cy = node_pos["DT"]
    si_cx, si_cy = node_pos["SI"]
    # They're both at x_S2 (the stream), so draw along x_S2
    p.append(
        f'<line x1="{x_S2}" y1="{dt_cy - R - 1}" x2="{x_S2}" y2="{si_cy + R + 1}" '
        f'stroke="#888" stroke-width="1.4" marker-end="url(#a-flow)"/>'
    )
    # S-inhibition at END → Name-mover at END (via END residual stream)
    nm_cx, nm_cy = node_pos["NM"]
    p.append(
        f'<line x1="{x_END}" y1="{si_cy - R - 1}" x2="{x_END}" y2="{nm_cy + R + 1}" '
        f'stroke="#888" stroke-width="1.4" marker-end="url(#a-flow)"/>'
    )
    # Induction at S2 → S-inhibition: both use S2 stream (IND at S2, SI reads K,V from S2)
    ind_cx, ind_cy = node_pos["IND"]
    si_yt, _ = zone_y(0.38, 0.64)
    p.append(
        f'<line x1="{x_S2}" y1="{ind_cy - R - 1}" x2="{x_S2}" y2="{si_yt + 4}" '
        f'stroke="#888" stroke-width="1.0" stroke-dasharray="3,2" '
        f'marker-end="url(#a-flow)" opacity="0.55"/>'
    )

    # ── Output arrow ─────────────────────────────────────────────────────────
    p.append(
        f'<line x1="{nm_cx}" y1="{nm_cy - R - 1}" x2="{nm_cx}" y2="{dy0 + 6}" '
        f'stroke="{C["nm"]}" stroke-width="2.2" marker-end="url(#a-nm)"/>'
    )
    p.append(
        f'<text x="{nm_cx}" y="{dy0 - 4}" text-anchor="middle" '
        f'font-size="9" font-weight="bold" fill="{C["nm"]}">P(IO) &gt; P(S)  ↑</text>'
    )

    # ── Input token labels ────────────────────────────────────────────────────
    input_y = dy1 + 18
    for px, lbl, desc in zip(pos_x, POS_LABELS, POS_DESC):
        p.append(f'<line x1="{px}" y1="{dy1}" x2="{px}" y2="{dy1 + 8}" stroke="{C["gr"]}" stroke-width="1.5"/>')
        p.append(
            f'<text x="{px}" y="{input_y}" text-anchor="middle" '
            f'font-size="11" font-weight="bold" fill="{C["tx"]}">{lbl}</text>'
        )
        for j, line in enumerate(desc.split("\n")):
            p.append(
                f'<text x="{px}" y="{input_y + 12 + j * 10}" text-anchor="middle" '
                f'font-size="7.5" fill="{C["dm"]}">{line}</text>'
            )

    # Sequence axis arrow and label
    arr_y = H - 8
    p += [
        f'<line x1="{pos_x[0] - 30}" y1="{arr_y}" x2="{pos_x[-1] + 30}" y2="{arr_y}" '
        f'stroke="{C["dm"]}" stroke-width="1" marker-end="url(#a-flow)"/>',
        f'<text x="{(pos_x[0] + pos_x[-1]) // 2}" y="{arr_y - 3}" '
        f'text-anchor="middle" font-size="8" fill="{C["dm"]}">sequence position</text>',
    ]

    # ── Legend (right panel) ──────────────────────────────────────────────────
    leg_x = dx1 + LEGEND_W // 2 - 50
    leg_y = dy0 + 8

    p.append(f'<text x="{leg_x}" y="{leg_y}" font-size="9" font-weight="bold" fill="{C["tx"]}">Head types:</text>')
    ROLE_NAMES = ["Name-mover", "S-inhibition", "Induction", "Dup-token"]
    ROLE_ABBRV = ["NM", "SI", "IND", "DT"]
    ROLE_CK    = ["nm", "si", "ind", "dt"]
    for i, (rn, ra, rck) in enumerate(zip(ROLE_NAMES, ROLE_ABBRV, ROLE_CK)):
        y = leg_y + 16 * (i + 1)
        p += [
            f'<circle cx="{leg_x + 7}" cy="{y - 4}" r="7" fill="{C[rck]}"/>',
            f'<text x="{leg_x + 7}" y="{y - 1}" text-anchor="middle" '
            f'font-size="6.5" font-weight="bold" fill="white">{ra}</text>',
            f'<text x="{leg_x + 18}" y="{y}" font-size="8" fill="{C["tx"]}">{rn}</text>',
        ]

    leg_y2 = leg_y + 90
    p.append(f'<text x="{leg_x}" y="{leg_y2}" font-size="9" font-weight="bold" fill="{C["tx"]}">Arrows:</text>')
    p += [
        f'<line x1="{leg_x}" y1="{leg_y2+12}" x2="{leg_x+32}" y2="{leg_y2+12}" '
        f'stroke="{C["si"]}" stroke-width="1.8" stroke-dasharray="5,3" marker-end="url(#a-si)"/>',
        f'<text x="{leg_x+36}" y="{leg_y2+16}" font-size="7.5" fill="{C["tx"]}">Attends to (K,V)</text>',
        f'<line x1="{leg_x}" y1="{leg_y2+28}" x2="{leg_x+32}" y2="{leg_y2+28}" '
        f'stroke="#888" stroke-width="1.4" marker-end="url(#a-flow)"/>',
        f'<text x="{leg_x+36}" y="{leg_y2+32}" font-size="7.5" fill="{C["tx"]}">Residual stream</text>',
        f'<text x="{leg_x+36}" y="{leg_y2+42}" font-size="7.5" fill="{C["tx"]}">contribution</text>',
    ]

    leg_y3 = leg_y2 + 62
    p += [
        f'<text x="{leg_x}" y="{leg_y3}" font-size="7.5" font-weight="bold" fill="{C["tx"]}">Scores:</text>',
        f'<text x="{leg_x}" y="{leg_y3+12}" font-size="7" fill="{C["dm"]}">Normalised logit-diff</text>',
        f'<text x="{leg_x}" y="{leg_y3+22}" font-size="7" fill="{C["dm"]}">recovery (n=100)</text>',
        f'<text x="{leg_x}" y="{leg_y3+38}" font-size="7" fill="{C["dm"]}">Note (†): SI heads have</text>',
        f'<text x="{leg_x}" y="{leg_y3+48}" font-size="7" fill="{C["dm"]}">highest individual</text>',
        f'<text x="{leg_x}" y="{leg_y3+58}" font-size="7" fill="{C["dm"]}">patching score in</text>',
        f'<text x="{leg_x}" y="{leg_y3+68}" font-size="7" fill="{C["dm"]}">Llama-3.2-3B (unlike</text>',
        f'<text x="{leg_x}" y="{leg_y3+78}" font-size="7" fill="{C["dm"]}">GPT-2 where NM leads).</text>',
    ]

    p.append("</svg>")
    out = FIG_DIR / "ioi-circuit-diagram.svg"
    out.write_text("\n".join(p))
    print(f"[3/3] Saved → {out}")


# =============================================================================

if __name__ == "__main__":
    print(f"Generating figures → {FIG_DIR}")
    fig1_heatmap()
    fig2_cross_model()
    fig3_circuit_diagram()
    print("Done.")
