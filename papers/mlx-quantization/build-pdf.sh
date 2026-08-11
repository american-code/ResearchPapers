#!/bin/bash
# Superseded. The paper is now LaTeX: submission/main.tex + submission/refs.bib.
#
#   cd submission && tectonic -X compile main.tex
#
# The markdown source (full-draft-v1.md) is retained as the drafting record and is
# no longer the build input. The old pandoc path is kept below, commented, only
# because its glyph warning is worth remembering: without -H tex-glyphs.tex,
# pandoc+tectonic SILENTLY DROPPED the Unicode exponents in p-values, so
# "9 × 10⁻⁶" rendered as "9 × 10" -- a plausible wrong number rather than an
# obvious failure. The LaTeX source avoids this by writing math as math.
set -euo pipefail
cd "$(dirname "$0")/submission"
tectonic -X compile main.tex
echo "built: submission/main.pdf"

# --- superseded pandoc path -------------------------------------------------
# pandoc full-draft-v1.md -o mlx-quantization-draft-v1.pdf \
#   --pdf-engine=tectonic -H tex-glyphs.tex \
#   --metadata title="..." --metadata author="..." \
#   -V geometry:margin=1in -V fontsize=10pt --toc --toc-depth=2
