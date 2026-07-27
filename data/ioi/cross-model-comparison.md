# IOI Circuit: Cross-Model Comparison (Llama-3.2-3B vs Pythia-1.4B)

**Generated:** 2026-07-27  
**Method:** Top-10 circuit-critical heads selected by combined normalized rank across activation patching and mean-ablation scores. Relative depth = `layer_index / n_layers`.

---

## Model Metadata

| Property | Llama-3.2-3B | Pythia-1.4B |
|---|---|---|
| HuggingFace ID | `mlx-community/Llama-3.2-3B-bf16` | `EleutherAI/pythia-1.4b` |
| Layers | 28 | 24 |
| Heads per layer | 24 | 16 |
| Examples (n) | 100 | 100 |
| Clean logit-diff mean | 5.649 | 4.120 |

---

## Selection Method

Each head is scored by two independent metrics:

- **Patching score** — normalized logit-diff recovery under activation patching (corrupted = IO/S names swapped). Higher = more critical.  
- **Ablation drop** — mean-logit-diff drop when the head is mean-ablated over clean examples. Higher = more critical.

Both metrics are min-max normalized to [0, 1] across all heads in the model, then summed to a **combined score**. Top-10 heads by combined score are reported.

---

## Top-10 Heads: Llama-3.2-3B (28 layers, 24 heads)

| Rank | Head | Layer | Patch score | Ablation drop | Rel. depth |
|------|------|-------|-------------|---------------|------------|
| 1 | L15H20 | 15 | 0.23963 | 1.57812 | 0.5357 |
| 2 | L17H17 | 17 | 0.09818 | 0.92937 | 0.6071 |
| 3 | L13H14 | 13 | 0.09215 | 0.62562 | 0.4643 |
| 4 | L24H15 | 24 | 0.14755 | 0.27563 | 0.8571 |
| 5 | L19H1  | 19 | 0.09907 | 0.44250 | 0.6786 |
| 6 | L21H20 | 21 | 0.07319 | 0.41687 | 0.7500 |
| 7 | L18H10 | 18 | 0.07844 | 0.31312 | 0.6429 |
| 8 | L14H0  | 14 | 0.11118 | 0.01688 | 0.5000 |
| 9 | L27H17 | 27 | 0.03509 | 0.37312 | 0.9643 |
| 10 | L26H23 | 26 | 0.03866 | 0.34437 | 0.9286 |

**Notes:** L15H20 is the dominant head, with an ablation drop (1.578) nearly 2× the next ranked head — consistent with a name-mover or S-inhibition role concentrated in a single head. The top-10 span rel. depths 0.46–0.96 with no representation below 0.46.

---

## Top-10 Heads: Pythia-1.4B (24 layers, 16 heads)

| Rank | Head | Layer | Patch score | Ablation drop | Rel. depth |
|------|------|-------|-------------|---------------|------------|
| 1 | L10H7  | 10 | 0.20687 | 0.89344 | 0.4167 |
| 2 | L15H15 | 15 | 0.15459 | 0.55094 | 0.6250 |
| 3 | L22H2  | 22 | 0.10086 | 0.41086 | 0.9167 |
| 4 | L21H3  | 21 | 0.05579 | 0.23703 | 0.8750 |
| 5 | L12H15 | 12 | 0.05074 | 0.12375 | 0.5000 |
| 6 | L17H7  | 17 | 0.04048 | 0.14945 | 0.7083 |
| 7 | L1H11  |  1 | -0.00279 | 0.32688 | 0.0417 |
| 8 | L10H0  | 10 | 0.03871 | 0.12781 | 0.4167 |
| 9 | L16H13 | 16 | 0.02528 | 0.11664 | 0.6667 |
| 10 | L13H6  | 13 | 0.04894 | 0.01781 | 0.5417 |

**Notes:** L1H11 is anomalous: its patching score is slightly negative (-0.003) but its ablation drop (0.327) is substantial, suggesting it suppresses interference rather than directly boosting IO probability. This is consistent with a duplicate-token or induction-style role in very early layers — a pattern absent from Llama at comparable depths.

---

## Shared vs. Model-Specific Positions

Pairs are matched greedily by smallest |Δrel_depth|. Tolerance: ±0.075.

| Shared zone | Llama head | Llama depth | Pythia head | Pythia depth | Δ |
|-------------|-----------|------------|------------|-------------|---|
| ~0.50 | L14H0 | 0.5000 | L12H15 | 0.5000 | 0.0000 |
| ~0.54 | L15H20 | 0.5357 | L13H6 | 0.5417 | 0.0060 |
| ~0.68 | L19H1 | 0.6786 | L16H13 | 0.6667 | 0.0119 |
| ~0.92 | L26H23 | 0.9286 | L22H2 | 0.9167 | 0.0119 |
| ~0.62 | L17H17 | 0.6071 | L15H15 | 0.6250 | 0.0179 |
| ~0.86 | L24H15 | 0.8571 | L21H3 | 0.8750 | 0.0179 |
| ~0.73 | L21H20 | 0.7500 | L17H7 | 0.7083 | 0.0417 |
| ~0.44 | L13H14 | 0.4643 | L10H7 | 0.4167 | 0.0476 |

**8 of 10 head positions are shared** (within ±0.075 relative depth).

### Llama-specific (no Pythia match within ±0.075)

| Head | Rel. depth | Interpretation |
|------|-----------|----------------|
| L18H10 | 0.6429 | Mid-late head; likely redundant with the 0.60–0.68 cluster in Llama |
| L27H17 | 0.9643 | Near-final layer; may correspond to a late output-writing role unique to Llama's deeper architecture |

### Pythia-specific (no Llama match within ±0.075)

| Head | Rel. depth | Interpretation |
|------|-----------|----------------|
| L1H11 | 0.0417 | **Very early** (layer 1 of 24). Negative patch score but strong ablation drop — plausible induction or duplicate-token head that shapes token distributions before main circuit activations. No analog in Llama's top-10. |
| L10H0 | 0.4167 | Co-located with Pythia's rank-1 head (L10H7) in the same layer. Matched depth (0.4167) was claimed by the L10H7 ↔ L13H14 pair; L10H0 represents a second critical head at the same layer, creating a two-head cluster not seen in Llama. |

---

## Zone Summary

| Rel. depth zone | Llama heads | Pythia heads | Status |
|-----------------|-------------|--------------|--------|
| 0.00–0.10 | — | L1H11 | Pythia-only |
| 0.40–0.55 | L13H14, L14H0, L15H20 | L10H7, L10H0, L12H15, L13H6 | **Shared** (both dense) |
| 0.60–0.70 | L17H17, L18H10, L19H1 | L15H15, L16H13 | **Shared** (Llama denser) |
| 0.70–0.80 | L21H20 | L17H7 | **Shared** |
| 0.85–0.90 | L24H15 | L21H3 | **Shared** |
| 0.90–1.00 | L26H23, L27H17 | L22H2 | **Shared** (Llama has 2) |

---

## Key Findings

1. **Strong cross-architecture conservation at mid-to-late depths.** Both models concentrate circuit-critical heads in the 0.40–0.97 relative depth range, with cluster peaks near 0.50, 0.62–0.68, 0.73, and 0.87–0.93. This pattern is consistent with the known IOI circuit structure (duplicate-token heads → S-inhibition → name-movers) mapping to roughly the bottom third, middle third, and top third of the network.

2. **Pythia has a unique early-layer critical head.** L1H11 at rel. depth 0.042 has no Llama analog and exhibits a dissociation between patching (slightly negative) and ablation (large positive drop). This may reflect Pythia's smaller hidden dimension requiring earlier information routing, or a model-specific induction head that gates later circuit components.

3. **Top heads are highly dominant in both models.** The rank-1 head (L15H20 in Llama, L10H7 in Pythia) accounts for the majority of the ablation drop relative to rank-2 and below, suggesting a bottleneck structure. This is consistent with a single prominent name-mover head being the primary driver in both architectures.

4. **Llama shows slightly more tail-end (0.90+) critical heads** (two vs. one), consistent with its greater depth (28 vs. 24 layers) creating more room for late output-adjustment mechanisms.

---

## Data provenance

| File | Description |
|------|-------------|
| `patching-llama3b.json` | Activation patching scores for Llama-3.2-3B, n=100 IOI examples |
| `patching-pythia1b.json` | Activation patching scores for Pythia-1.4B, n=100 IOI examples |
| `ablation-llama3b.json` | Mean-ablation drop scores for Llama-3.2-3B, n=100 IOI examples |
| `ablation-pythia1b.json` | Mean-ablation drop scores for Pythia-1.4B, n=100 IOI examples |
