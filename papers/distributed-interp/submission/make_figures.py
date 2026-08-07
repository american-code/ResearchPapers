#!/usr/bin/env python3
"""
Generate the figures for the two SAE-infrastructure papers.

No matplotlib dependency: emits vector PDF directly.

  convergence-5k  : two-worker vs single-node SAE loss trajectories
                    -> papers/distributed-interp/submission/figures/
  matching-null   : observed vs permutation-null match similarity, per pair
                    -> papers/sae-comparison/submission/figures/

The matching figure belongs to the sae-comparison paper, which is where the
cross-architecture matching analysis is reported; it is generated here because
this script already loads the shared matching report.
"""

import json
from pathlib import Path

WS = Path("/Users/melton/ResearchPapers")
OUT = WS / "papers/distributed-interp/submission/figures"
OUT_MATCHING = WS / "papers/sae-comparison/submission/figures"
for _d in (OUT, OUT_MATCHING):
    _d.mkdir(parents=True, exist_ok=True)


# ── minimal PDF writer ───────────────────────────────────────────────────────

class PDF:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.ops = []

    def _c(self, rgb):
        return f"{rgb[0]:.3f} {rgb[1]:.3f} {rgb[2]:.3f}"

    def line(self, x1, y1, x2, y2, rgb=(0, 0, 0), lw=0.8, dash=None):
        d = f"[{dash}] 0 d " if dash else "[] 0 d "
        self.ops.append(f"q {d}{lw} w {self._c(rgb)} RG {x1:.2f} {y1:.2f} m "
                        f"{x2:.2f} {y2:.2f} l S Q")

    def polyline(self, pts, rgb=(0, 0, 0), lw=1.0, dash=None):
        if len(pts) < 2:
            return
        d = f"[{dash}] 0 d " if dash else "[] 0 d "
        p = f"{pts[0][0]:.2f} {pts[0][1]:.2f} m " + " ".join(
            f"{x:.2f} {y:.2f} l" for x, y in pts[1:])
        self.ops.append(f"q {d}{lw} w {self._c(rgb)} RG {p} S Q")

    def rect(self, x, y, w, h, rgb=(0, 0, 0), fill=False, lw=0.8):
        op = "f" if fill else "S"
        col = "rg" if fill else "RG"
        self.ops.append(f"q {lw} w {self._c(rgb)} {col} "
                        f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op} Q")

    def text(self, x, y, s, size=8, rgb=(0, 0, 0), anchor="start", rot=0):
        s = (s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)"))
        wid = len(s) * size * 0.5
        if anchor == "middle":
            x -= wid / 2
        elif anchor == "end":
            x -= wid
        if rot:
            import math
            c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
            tm = f"{c:.4f} {sn:.4f} {-sn:.4f} {c:.4f} {x:.2f} {y:.2f} Tm"
        else:
            tm = f"1 0 0 1 {x:.2f} {y:.2f} Tm"
        self.ops.append(f"BT /F1 {size} Tf {self._c(rgb)} rg {tm} ({s}) Tj ET")

    def save(self, path):
        content = "\n".join(self.ops).encode("latin-1", "replace")
        objs = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.w} {self.h}] "
             f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>").encode(),
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n"
            + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        out = bytearray(b"%PDF-1.4\n")
        offs = []
        for i, o in enumerate(objs, 1):
            offs.append(len(out))
            out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
        for o in offs:
            out += f"{o:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n"
                f"{xref}\n%%EOF\n").encode()
        Path(path).write_bytes(out)
        print(f"wrote {path} ({len(out)} bytes)")


BLUE = (0.16, 0.35, 0.60)
ORANGE = (0.85, 0.42, 0.10)
GREY = (0.55, 0.55, 0.55)
DARK = (0.15, 0.15, 0.15)


def fig_convergence():
    import math
    d = json.loads((WS / "data/distributed-sae-training-validation.json").read_text())
    dist = [(c["step"], c["loss"]) for c in d["distributed"]["curve"]]
    base = [(c["step"], c["loss"]) for c in d["baseline"]["curve"]]

    W, H = 470, 200
    p = PDF(W, H)
    # ── left panel: log-scale full trajectory ──
    L, B, PW, PH = 42, 38, 178, 132
    ymin, ymax = math.log10(0.015), math.log10(1.5)

    def X(s):
        return L + PW * s / 5000

    def Y(v):
        return B + PH * (math.log10(v) - ymin) / (ymax - ymin)

    p.rect(L, B, PW, PH, GREY, lw=0.5)
    for dec, lab in ((0.02, "0.02"), (0.1, "0.1"), (1.0, "1.0")):
        p.line(L, Y(dec), L + PW, Y(dec), (0.88, 0.88, 0.88), 0.4)
        p.text(L - 4, Y(dec) - 2.5, lab, 7, DARK, "end")
    for s in (0, 2500, 5000):
        p.line(X(s), B, X(s), B - 3, GREY, 0.5)
        p.text(X(s), B - 12, f"{s:,}", 7, DARK, "middle")
    p.polyline([(X(s), Y(v)) for s, v in base], BLUE, 1.1)
    p.polyline([(X(s), Y(v)) for s, v in dist], ORANGE, 1.1, dash="2 2")
    p.text(L + PW / 2, B + PH + 8, "full trajectory (log scale)", 7.5, DARK, "middle")
    p.text(L + PW / 2, B - 24, "training step", 7.5, DARK, "middle")
    p.text(L - 30, B + PH / 2, "loss", 7.5, DARK, "middle", rot=90)

    # ── right panel: linear zoom, steps 500+ ──
    L2 = L + PW + 62
    z_d = [(s, v) for s, v in dist if s >= 500]
    z_b = [(s, v) for s, v in base if s >= 500]
    lo = min(min(v for _, v in z_d), min(v for _, v in z_b)) * 0.998
    hi = max(max(v for _, v in z_d), max(v for _, v in z_b)) * 1.002

    def X2(s):
        return L2 + PW * (s - 500) / 4500

    def Y2(v):
        return B + PH * (v - lo) / (hi - lo)

    p.rect(L2, B, PW, PH, GREY, lw=0.5)
    for frac in (0, 0.5, 1.0):
        v = lo + frac * (hi - lo)
        p.line(L2, Y2(v), L2 + PW, Y2(v), (0.88, 0.88, 0.88), 0.4)
        p.text(L2 - 4, Y2(v) - 2.5, f"{v:.4f}", 7, DARK, "end")
    for s in (500, 2500, 5000):
        p.line(X2(s), B, X2(s), B - 3, GREY, 0.5)
        p.text(X2(s), B - 12, f"{s:,}", 7, DARK, "middle")
    p.polyline([(X2(s), Y2(v)) for s, v in z_b], BLUE, 1.1)
    p.polyline([(X2(s), Y2(v)) for s, v in z_d], ORANGE, 1.1, dash="2 2")
    p.text(L2 + PW / 2, B + PH + 8, "steps 500-5,000 (linear scale)", 7.5, DARK, "middle")
    p.text(L2 + PW / 2, B - 24, "training step", 7.5, DARK, "middle")

    # legend
    p.line(L + 8, H - 12, L + 26, H - 12, BLUE, 1.1)
    p.text(L + 30, H - 14.5, "single-node (batch 512)", 7.5, DARK)
    p.line(L + 150, H - 12, L + 168, H - 12, ORANGE, 1.1, dash="2 2")
    p.text(L + 172, H - 14.5, "two-worker averaged (2 x 256)", 7.5, DARK)
    p.save(OUT / "convergence-5k.pdf")


def fig_matching_null():
    r = json.loads((WS / "data/sae-analysis/matching-v2/matching-report.json").read_text())
    pw = r["metrics"]["pearson"]["pairwise"]
    pairs = [("llama_qwen", "Llama-Qwen"), ("mistral_qwen", "Mistral-Qwen"),
             ("llama_mistral", "Llama-Mistral")]

    W, H = 440, 190
    p = PDF(W, H)
    L, B, PW, PH = 52, 44, 350, 118
    p.rect(L, B, PW, PH, GREY, lw=0.5)

    def X(v):
        return L + PW * (v - 0.2) / 0.8      # similarity axis 0.2 -> 1.0

    for v in (0.2, 0.4, 0.6, 0.8, 1.0):
        p.line(X(v), B, X(v), B - 3, GREY, 0.5)
        p.text(X(v), B - 12, f"{v:.1f}", 7, DARK, "middle")
    p.text(L + PW / 2, B - 24, "reciprocal-match similarity (Pearson r over 1,000 chunks)",
           7.5, DARK, "middle")

    row_h = PH / len(pairs)
    for i, (key, lab) in enumerate(pairs):
        y = B + PH - (i + 0.5) * row_h
        n = pw[key]["null"]
        p.text(L - 5, y - 2.5, lab, 7.5, DARK, "end")
        # null range bar: mean -> p99
        p.rect(X(n["null_mean"]), y - 5, X(n["null_p99"]) - X(n["null_mean"]), 10,
               (0.80, 0.84, 0.90), fill=True)
        p.line(X(n["null_mean"]), y - 7, X(n["null_mean"]), y + 7, BLUE, 1.0)
        p.line(X(n["null_p99"]), y - 7, X(n["null_p99"]), y + 7, BLUE, 1.4)
        p.text(X(n["null_mean"]), y + 9, "null mean", 6, BLUE, "middle")
        p.text(X(n["null_p99"]), y + 9, "null p99 = tau", 6, BLUE, "middle")
        # the discarded fixed threshold
        p.line(X(0.80), y - 7, X(0.80), y + 7, ORANGE, 1.2, dash="2 2")
        p.text(X(1.0) + 4, y - 2.5, f"{pw[key]['n_matches']} matches", 7, DARK)

    p.line(X(0.80), B, X(0.80), B + PH, ORANGE, 1.0, dash="2 2")
    p.text(X(0.80), B + PH + 6, "tau = 0.80 (naive protocol)", 7, ORANGE, "middle")
    p.text(L, H - 12, "Shaded: range between the permutation-null mean and its 99th "
                      "percentile.", 7, DARK)
    p.text(L, H - 21, "The naive fixed threshold falls inside the null range for every "
                      "pair.", 7, DARK)
    p.save(OUT_MATCHING / "matching-null.pdf")


if __name__ == "__main__":
    fig_convergence()
    fig_matching_null()
