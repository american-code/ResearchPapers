# Research Papers Workspace

Mechanistic interpretability and AI safety research. Three papers.

> **See [CORRECTIONS.md](CORRECTIONS.md)** (2026-07-30) for the audit of all three
> papers against their data. Two headline claims were withdrawn: cross-architecture
> feature universality (3,753 shared features → 0 under a permutation null) and IOI
> depth-zone conservation (80% match rate → not significant, p = 0.098/0.32). The
> hypotheses below are the *original* framings; the papers now report what the data
> supports, which in two of three cases is a negative result.

---

## Paper 1 — Circuit Tracing at Scale

**Working title:** *Circuit Tracing in Large Language Models: Scaling Mechanistic Explanations Beyond Toy Tasks*

**Directory:** `papers/circuit-tracing/`

**Hypothesis:** The circuit-level mechanisms identified in small models on controlled tasks
(e.g., IOI) generalize to larger models and naturalistic inputs, but with additional
redundancy structure that is obscured by standard activation patching. Targeted
path-patching combined with sparse decomposition recovers stable, transferable circuits.

**Target venue:** ICLR 2026 (main track — interpretability)

**Status:** Bottleneck structure and path-patching results hold. The depth-conservation
claim was withdrawn after permutation testing. Circuit-level faithfulness is still
unmeasured — the largest outstanding gap.

**Key data:** `data/ioi/`, `data/factual-assoc/`

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

**Status:** Rescoped. Only TopK SAEs were ever trained (no L1, no Gated, no JumpReLU),
on Llama/Mistral/Qwen rather than GPT-2/Pythia/Llama, and no probe, human-rating or
steering study was conducted — so the objective comparison in the hypothesis above has
no data behind it. The paper now reports TopK training dynamics: dictionary collapse,
dense-feature degeneracy, and held-out reconstruction.

**Key data:** `data/sae-runs/`, `data/sae-analysis/`

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

**Status:** Infrastructure results hold (bit-exact split, 1.25 GB/s throughput). Both
scientific results are negative: zero cross-architecture universal features under a
calibrated null, and a safety classifier at ROC-AUC 0.564 with a CI spanning chance.
The two-node Thunderbolt deployment is specified but not built.

**Key data:** `data/safety-classifier/`, `data/sae-analysis/matching-v2/`

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

## Key analysis scripts

| Script | Purpose |
|---|---|
| `data/sae-analysis/cross_arch_matching_v2.py` | Corrected cross-architecture matching: reciprocal one-to-one, closed triangles, permutation-calibrated threshold. Supersedes `cross_arch_matching.py`. |
| `data/sae-analysis/recompute_corrections.py` | Full-corpus and held-out SAE reconstruction, safety-classifier ROC-AUC, IOI depth-conservation null. |
| `papers/distributed-interp/submission/make_figures.py` | Convergence and null-distribution figures. |

Note: `data/activations/llama-3b-layer14/` and `data/sae-runs/llama-3b-layer14/` were
renamed from `-layer16` on 2026-07-30. They always contained layer-14 activations; the
old name did not match the data.
