# Research Papers Workspace

Mechanistic interpretability and AI safety research. Three planned papers targeting
ICLR / NeurIPS 2026.

---

## Paper 1 — Circuit Tracing at Scale

**Working title:** *Circuit Tracing in Large Language Models: Scaling Mechanistic Explanations Beyond Toy Tasks*

**Directory:** `papers/circuit-tracing/`

**Hypothesis:** The circuit-level mechanisms identified in small models on controlled tasks
(e.g., IOI) generalize to larger models and naturalistic inputs, but with additional
redundancy structure that is obscured by standard activation patching. Targeted
path-patching combined with sparse decomposition recovers stable, transferable circuits.

**Target venue:** ICLR 2026 (main track — interpretability)

**Key data:** `data/ioi/`

---

## Paper 2 — SAE Architecture Comparison

**Working title:** *Sparse Autoencoders as Feature Finders: A Systematic Comparison of Training Objectives, Architectures, and Evaluation Metrics*

**Directory:** `papers/sae-comparison/`

**Hypothesis:** Reconstruction-loss-optimized SAEs and auxiliary-loss variants (e.g.,
TopK, Gated) recover overlapping but non-identical feature sets; the choice of
evaluation metric (probe accuracy, human interpretability ratings, downstream steering
fidelity) determines which architecture appears superior, revealing a hidden evaluation
confound that has driven inconsistent conclusions in prior work.

**Target venue:** NeurIPS 2026 (main track — representation learning)

**Key data:** `data/sae-runs/`, `data/steering/`

---

## Paper 3 — Distributed Interpretability

**Working title:** *Distributed Interpretability: Collaborative Feature Attribution Across Model Layers and Attention Heads*

**Directory:** `papers/distributed-interp/`

**Hypothesis:** Safety-relevant behaviors (e.g., refusal, deception detection) are not
localized to single components but emerge from distributed sub-networks spanning multiple
layers. A graph-based attribution method that propagates credit across the full residual
stream identifies these sub-networks more faithfully than layer-local methods, and
the resulting representations improve the robustness of safety classifiers.

**Target venue:** ICLR 2026 (safety & alignment track)

**Key data:** `data/safety-classifier/`, `data/steering/`

---

## Directory Layout

```
ResearchPapers/
├── papers/
│   ├── circuit-tracing/      # Paper 1 source, notebooks, drafts
│   ├── sae-comparison/       # Paper 2 source, notebooks, drafts
│   └── distributed-interp/   # Paper 3 source, notebooks, drafts
├── data/
│   ├── ioi/                  # Indirect Object Identification task data
│   ├── sae-runs/             # SAE training checkpoints and eval logs
│   ├── steering/             # Activation steering experiment data
│   └── safety-classifier/    # Safety classifier training and eval data
├── figures/                  # Publication-quality figures (shared)
├── benchmarks/               # Shared benchmark scripts and results
└── docs/                     # Notes, meeting logs, literature reviews
```
