# arXiv Submission Checklist — SAE Comparison Paper

**Paper:** *Sparse Autoencoders as Feature Finders: A Systematic Comparison of Training Objectives, Architectures, and Evaluation Metrics*
**Target arXiv date:** October 2026 (concurrent with ICLR 2027 submission, September 24, 2026)
**arXiv categories:** cs.LG (primary), cs.AI (secondary)

---

## Phase 1: Complete Before Data Collection Begins

- [ ] Confirm layer targeting: verify that Llama-3.2-3B layer 15 is the correct SAE training target (cross-reference circuit-tracing paper companion results)
- [ ] IRB/ethics review: determine whether human interpretability annotation study requires institutional approval; initiate review if yes
- [ ] Annotator recruitment plan: identify annotation platform (e.g., Prolific, internal), set pay rate, prepare annotation interface
- [ ] Automated steering judge: choose and validate the judge model/classifier for steering fidelity evaluation; run pilot on 20 features to check for systematic biases
- [ ] Lock in dataset: choose the held-out activation set (e.g., The Pile validation split) and establish that it does not overlap with SAE training data

---

## Phase 2: Complete Before Submission Draft Is Finalized

### Data and Experiments

- [ ] SAE training: train all four architectures (L1, TopK, Gated, JumpReLU) at all three sparsity levels ($L_0 \in \{16, 32, 64\}$); save checkpoints to `data/sae-runs/`
- [ ] Feature overlap analysis: compute pairwise cosine similarity across all architecture pairs; save to `data/sae-runs/feature-overlap.json`
- [ ] Probe evaluation: run linear probe suite; save AUROC by task and architecture to `data/sae-runs/probe-results.json`
- [ ] Human annotation study: collect ratings; save raw data to `data/sae-runs/human-ratings.json`
- [ ] Steering fidelity: run activation steering experiments; save results to `data/steering/feature-steering-results.json`
- [ ] Statistical tests: run Wilcoxon tests and two-way ANOVA; save results

### Figures

- [ ] Figure 1: Architecture diagram (`figures/sae-comparison/architecture-diagram.svg`)
- [ ] Figure 2: Training curves (`figures/sae-comparison/training-curves.svg`)
- [ ] Figure 3: Probe AUROC bar chart (`figures/sae-comparison/probe-auroc-barplot.svg`)
- [ ] Figure 4: Feature activation examples (`figures/sae-comparison/feature-examples.svg`)
- [ ] Figure 5: Architecture × metric ranking matrix (`figures/sae-comparison/ranking-matrix.svg`) — **the central result figure; prioritize**

### Text

- [ ] Fill in all `[[PLACEHOLDER]]` entries in `papers/sae-comparison/full-draft-v1.md` with actual numbers
- [ ] Fill in all `\placeholder{}` entries in `papers/sae-comparison/submission/main.tex`
- [ ] Resolve all TODO entries in `references.bib`
- [ ] Verify Llama 3.2-3B citation (Meta technical report — confirm arXiv ID covers the 3.2 series)
- [ ] Verify JumpReLU citation (confirm separate paper or same paper as Gated SAE)
- [ ] Write full reproducibility statement (repository URL, seed list, compute estimate)

---

## Phase 3: arXiv Submission Steps

### Before uploading

- [ ] Download ICLR 2027 style file from iclr.cc when published; replace `iclr2024_conference.sty` proxy in `submission/`
- [ ] Confirm ICLR 2027 double-blind policy: arXiv posting concurrent with submission is standard practice but verify no deanonymization conflict (ICLR has historically permitted this; the paper should be submitted anonymously, with the arXiv version authored normally)
- [ ] Run complete LaTeX build: `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`
- [ ] Verify page count: main text ≤ 9 pages (ICLR 2027 submission limit, based on ICLR 2026 precedent — confirm when author guide publishes)
- [ ] Check all figures render correctly in PDF
- [ ] Remove all `\placeholder{}` commands (they should all be replaced with real content by this point)
- [ ] Remove `\tocite{}` commands (replace with `\citet{}` or `\citep{}`)
- [ ] Check for unresolved `[[PLACEHOLDER]]` or `[[NOTE]]` markers in the Markdown draft and the LaTeX source
- [ ] Spell-check and grammar-check full manuscript
- [ ] Verify abstract is self-contained (no undefined acronyms, no forward references)
- [ ] Ethics statement: confirm it accurately describes annotation study and data practices
- [ ] Reproducibility statement: confirm repository URL is live and code is committed

### Packaging for arXiv

arXiv requires the LaTeX source, not just the PDF. The submission package is this directory (`papers/sae-comparison/submission/`).

- [ ] All `.tex` files are in the root of the submission directory
- [ ] `references.bib` is present
- [ ] All figure files are in `submission/figures/` in PDF, PNG, or EPS format (SVG not supported by arXiv; convert all figures)
- [ ] Style file (`.sty`) is included in the submission directory
- [ ] No absolute paths in LaTeX source (all figure paths relative to submission directory)
- [ ] Test: zip the submission directory and compile from the zip to verify no missing dependencies

### Uploading

- [ ] Create arXiv account or log in at arxiv.org
- [ ] Start a new submission: select cs.LG (primary), cs.AI (secondary)
- [ ] Upload zip of submission directory
- [ ] Fill in title, abstract (plain text — no LaTeX math in abstract field), authors, and comments fields
- [ ] Comments field suggestion: "9 pages, 5 figures. Under review at ICLR 2027."
- [ ] Set the correct license (recommend CC BY 4.0 for open science)
- [ ] Preview the compiled PDF on arXiv's system before finalizing
- [ ] Submit

### After submission

- [ ] Record arXiv ID in `docs/publication-calendar.md` and `README.md`
- [ ] Share arXiv link with collaborators and on relevant community channels (Alignment Forum, interpretability Slack, etc.)
- [ ] Cross-post to Alignment Forum if framing for the safety community is desired

---

## Timing Relative to ICLR 2027

| Event | Date |
|---|---|
| ICLR 2027 abstract registration | September 19, 2026 (AoE) |
| ICLR 2027 full paper submission | September 24, 2026 (AoE) |
| arXiv posting (recommended: same day) | September 24, 2026 |
| ICLR 2027 notification | January 22, 2027 |
| arXiv revision (if major revisions needed post-review) | Post-notification |

**NeurIPS 2026 MI Workshop opportunity:** If the NeurIPS 2026 MI Workshop CFP publishes
before ~August 10, 2026, a 9-page long-paper version can be submitted by the projected
~August 22 deadline. This is the highest-fit near-term venue for the paper's central
finding. The workshop is non-archival; concurrent ICLR 2027 submission is permitted.

---

## Key File Locations

| File | Purpose |
|---|---|
| `papers/sae-comparison/full-draft-v1.md` | Markdown source (primary working draft) |
| `papers/sae-comparison/submission/main.tex` | LaTeX submission source |
| `papers/sae-comparison/submission/references.bib` | Bibliography |
| `papers/sae-comparison/submission/figures/` | Figure files (PDF/PNG/EPS for arXiv) |
| `data/sae-runs/` | SAE training checkpoints and eval results |
| `data/steering/` | Activation steering experiment results |
| `figures/sae-comparison/` | Source figures (SVG; convert to PDF for arXiv) |
