#!/bin/bash
# Build the PDF. Requires pandoc + tectonic (no TeX install needed).
#
# The -H header is not optional: without it, pandoc+tectonic SILENTLY DROPS the
# Unicode math characters used throughout, which deletes the exponents from
# p-values — "9 × 10⁻⁶" renders as "9 × 10" and reads as a plausible wrong number
# rather than an obvious failure. Verify output after building.
set -euo pipefail
cd "$(dirname "$0")"
pandoc full-draft-v1.md \
  -o mlx-quantization-draft-v1.pdf \
  --pdf-engine=tectonic \
  --metadata title="What Does MLX 4-bit Cost? A Controlled Audit of Quantization for Code Generation on Apple Silicon" \
  --metadata author="J. Melton, American Code Labs (jmelton@americancode.org)" \
  --metadata date="$(date +%Y-%m-%d)" \
  -H tex-glyphs.tex \
  -V geometry:margin=1in -V fontsize=10pt \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  --toc --toc-depth=2
echo "built: mlx-quantization-draft-v1.pdf"
# Fail loudly if a load-bearing figure did not survive conversion.
if command -v pdftotext >/dev/null; then
  txt=$(pdftotext mlx-quantization-draft-v1.pdf - 2>/dev/null)
  for n in 134 0.52 2.51 9.1 59; do
    grep -q -- "$n" <<<"$txt" || { echo "VERIFY FAILED: '$n' missing from PDF"; exit 1; }
  done
  echo "verified: key figures present in output"
fi
