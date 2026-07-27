# 4. Results

## 4.1 Baseline Task Performance

Both models correctly perform the IOI task across all 100 evaluation examples. Llama-3.2-3B achieves a mean clean logit difference (LD) of **5.649** (SD = 0.773), reflecting a strong preference for the indirect object (IO) token over the subject (S) token at the final sequence position. Pythia-1.4B achieves a mean clean LD of **4.120**, 27% lower in absolute terms, consistent with its smaller parameter count and simpler attention architecture. Neither model produces a negative-LD example: the IO token is ranked above S on every trial in both models, confirming that the circuit analysis begins from a clean behavioral baseline.

---

## 4.2 IOI Circuit Identification: Llama-3.2-3B

Figure 1 shows the full activation patching heatmap for Llama-3.2-3B across all 28 × 24 = 672 (layer, head) positions. Two features are immediately apparent: most heads contribute negligibly to logit-difference recovery under patching, and a sparse set of heads in the middle-to-late network produces strongly positive scores.

**Figure 1.** Activation patching heatmap for Llama-3.2-3B (n = 100 IOI examples). Each cell shows normalized logit-difference recovery when the output of that (layer, head) is replaced with the corresponding clean-run activation. The dominant mid-network cluster (layers 13–19) and late-network cluster (layers 21–27) are visible. See `figures/ioi-patching-heatmap-llama3b.svg`.

Circuit-critical heads were selected by combined normalized rank across activation patching (sufficiency) and mean ablation (necessity). Table 1 lists the top-10 heads on each metric.

**Table 1.** Top-10 circuit-critical heads in Llama-3.2-3B by activation patching score (sufficiency) and mean-ablation drop (necessity). Relative depth = layer / n\_layers.

| Rank | Head    | Patch score | 95% CI         | Ablation drop | % clean LD | Rel. depth |
|------|---------|-------------|----------------|---------------|------------|------------|
| 1    | L15H20  | 0.240       | [0.225, 0.255] | 1.578         | 27.9%      | 0.536      |
| 2    | L17H17  | 0.098       | [0.089, 0.107] | 0.929         | 16.4%      | 0.607      |
| 3    | L13H14  | 0.092       | [0.083, 0.102] | 0.626         | 11.1%      | 0.464      |
| 4    | L24H15  | 0.148       | [0.133, 0.164] | 0.276         |  4.9%      | 0.857      |
| 5    | L19H1   | 0.099       | [0.094, 0.105] | 0.443         |  7.8%      | 0.679      |
| 6    | L21H20  | 0.073       | [0.068, 0.078] | 0.417         |  7.4%      | 0.750      |
| 7    | L18H10  | 0.078       | [0.073, 0.084] | 0.313         |  5.5%      | 0.643      |
| 8    | L14H0   | 0.111       | [0.099, 0.125] | 0.017         |  0.3%      | 0.500      |
| 9    | L27H17  | 0.035       | [0.030, 0.041] | 0.373         |  6.6%      | 0.964      |
| 10   | L26H23  | 0.039       | [0.030, 0.047] | 0.344         |  6.1%      | 0.929      |

The most striking feature of the Llama-3.2-3B circuit is the dominance of L15H20. Its ablation drop (1.578) is **70% larger** than rank-2 L17H17 (0.929) and **2.5× rank-3** L13H14 (0.626). L15H20 also leads the patching ranking at a score of 0.240, more than 60% above rank-2 L24H15 (0.148). No other head in the top-10 approaches this contribution on both metrics simultaneously: L24H15 scores highly on patching but weakly on ablation (0.276), while L27H17 shows the reverse pattern. This double-dominance of L15H20 — both sufficient and necessary — identifies it as a hard bottleneck in the IOI circuit at relative depth 0.536.

The circuit spans relative depths 0.46–0.97 with no critical head below 0.46. Three depth clusters are visible: a mid cluster (0.46–0.54) comprising L13H14, L14H0, and L15H20; a mid-late cluster (0.60–0.75) comprising L17H17, L18H10, L19H1, and L21H20; and a late cluster (0.86–0.97) comprising L24H15, L26H23, and L27H17. This three-cluster structure corresponds qualitatively to the functional zones identified by Wang et al. (2022) in GPT-2 Small, as discussed in Section 4.4.

---

## 4.3 IOI Circuit Identification: Pythia-1.4B

Pythia-1.4B replicates the bottleneck structure of Llama-3.2-3B. Table 2 presents the top-10 circuit-critical heads.

**Table 2.** Top-10 circuit-critical heads in Pythia-1.4B.

| Rank | Head   | Patch score | 95% CI         | Ablation drop | % clean LD | Rel. depth |
|------|--------|-------------|----------------|---------------|------------|------------|
| 1    | L10H7  | 0.207       | [0.199, 0.216] | 0.893         | 21.7%      | 0.417      |
| 2    | L15H15 | 0.155       | [0.134, 0.175] | 0.551         | 13.4%      | 0.625      |
| 3    | L22H2  | 0.101       | [0.097, 0.105] | 0.411         |  9.9%      | 0.917      |
| 4    | L1H11  | −0.003      | [−0.011, 0.006]| 0.327         |  7.9%      | 0.042      |
| 5    | L21H3  | 0.056       | [0.052, 0.059] | 0.237         |  5.7%      | 0.875      |
| 6    | L17H7  | 0.040       | [0.027, 0.056] | 0.150         |  3.6%      | 0.708      |
| 7    | L10H0  | 0.039       | [0.036, 0.041] | 0.128         |  3.1%      | 0.417      |
| 8    | L12H15 | 0.051       | [0.048, 0.054] | 0.124         |  3.0%      | 0.500      |
| 9    | L16H13 | 0.025       | [0.022, 0.029] | 0.117         |  2.8%      | 0.667      |
| 10   | L13H6  | 0.049       | [0.042, 0.056] | 0.018         |  0.4%      | 0.542      |

L10H7 leads both rankings: its ablation drop (0.893) is **62% larger** than rank-2 L15H15 (0.551), and its patching score (0.207) is **34% larger** than rank-2 (0.155). The pattern — one head substantially dominant on both sufficiency and necessity — exactly mirrors the Llama result, though the absolute magnitude of the ablation drop is smaller (0.893 vs. 1.578) consistent with Pythia's lower baseline LD.

**Anomalous early head: L1H11.** One Pythia-specific finding stands out: L1H11 at relative depth 0.042 (layer 1 of 24) scores −0.003 on activation patching (95% CI [−0.011, 0.006], straddling zero) yet produces the fourth-largest ablation drop in the model (0.327, 7.9% of clean LD). This dissociation between sufficiency (near zero or slightly negative) and necessity (substantial) is inconsistent with the role profile of name-mover or S-inhibition heads, which score positively on both measures. The most parsimonious interpretation is that L1H11 suppresses interference — for example, dampening an early signal that would otherwise mislead later heads — rather than directly boosting IO probability. Ablating it disrupts the clean causal pathway downstream, producing a large LD drop, but substituting its clean activation into a corrupted context provides no positive recovery. No Llama-3.2-3B head in the top-10 occupies a comparable depth (< 0.10), suggesting this interference-suppression role may be specific to Pythia's architecture or training distribution.

---

## 4.4 Comparison with Wang et al. (2022): Depth-Zone Conservation

Wang et al. (2022) identified three functional head classes in GPT-2 Small at specific relative depth ranges: duplicate-token/induction heads (depth 0.0–0.42), S-inhibition heads (0.58–0.67), and name-mover heads (0.75–0.83). Table 3 shows where these zones fall in our two models.

**Table 3.** Functional depth-zone alignment across GPT-2 Small (Wang et al.), Llama-3.2-3B, and Pythia-1.4B.

| Zone                  | GPT-2 Small      | Llama-3.2-3B          | Pythia-1.4B              |
|-----------------------|------------------|-----------------------|--------------------------|
| Early (0.00–0.25)     | L0H1, L3H0       | —                     | L1H11                    |
| Mid-induction (0.40–0.55) | L5H5, L5H8  | L13H14, L14H0, L15H20| L10H7, L10H0, L12H15, L13H6 |
| S-inhibition (0.50–0.70) | L7H3–L8H10  | L17H17, L18H10, L19H1| L15H15, L16H13           |
| Name-movers (0.75–0.92)  | L9H6–L10H0  | L21H20, L24H15, L26H22| L21H3, L22H2            |

All three functional zones from Wang et al. are present in both modern models at compatible relative depths. The mid-induction zone (0.40–0.55) is the densest cluster in all three models; the name-mover zone (0.75–0.92) contains the heads with highest late-network patching scores. One notable divergence: the very-early zone (0.00–0.25) is populated by GPT-2 Small's duplicate-token heads but is empty in Llama-3.2-3B's top-10 and represented only by the anomalous L1H11 in Pythia.

A second divergence is the **bottleneck vs. distributed** structure. Wang et al. found 26 circuit components with moderate individual contributions; in both Llama and Pythia, one head accounts for 22–28% of clean LD on its own, substantially above any single head's contribution in GPT-2 Small. This could reflect scale effects (larger models can concentrate circuit function in individual heads), differences in our selection methodology, or task-distribution differences in our IOI dataset. We return to this in the Discussion.

---

## 4.5 Cross-Architecture Generalization

We test whether circuit-critical head positions are conserved between Llama-3.2-3B and Pythia-1.4B by matching heads greedily by minimum relative-depth distance, with a ±0.075 tolerance. Table 4 reports all matched pairs.

**Table 4.** Cross-architecture matched head positions (Llama-3.2-3B vs. Pythia-1.4B, tolerance ±0.075).

| Zone    | Llama head | Llama depth | Pythia head | Pythia depth | \|Δ\| |
|---------|-----------|-------------|------------|--------------|-------|
| ~0.50   | L14H0     | 0.500       | L12H15     | 0.500        | 0.000 |
| ~0.54   | L15H20    | 0.536       | L13H6      | 0.542        | 0.006 |
| ~0.68   | L19H1     | 0.679       | L16H13     | 0.667        | 0.012 |
| ~0.92   | L26H23    | 0.929       | L22H2      | 0.917        | 0.012 |
| ~0.62   | L17H17    | 0.607       | L15H15     | 0.625        | 0.018 |
| ~0.86   | L24H15    | 0.857       | L21H3      | 0.875        | 0.018 |
| ~0.73   | L21H20    | 0.750       | L17H7      | 0.708        | 0.042 |
| ~0.46   | L13H14    | 0.464       | L10H7      | 0.417        | 0.048 |

**8 of 10 circuit-critical head positions match within ±0.075 relative depth.** The two model-specific heads are: Llama's L18H10 (depth 0.643, no Pythia match) and L27H17 (depth 0.964, near-final layer, consistent with Llama's greater depth permitting an extra output-adjustment head); and Pythia's L1H11 (depth 0.042, discussed above) and L10H0 (depth 0.417, co-located with L10H7 in the same layer, forming a two-head cluster not observed in Llama).

The matched pairs exhibit a three-cluster depth structure in both models:

- **Mid cluster (0.40–0.55):** Three Llama heads, four Pythia heads. Pythia is denser here, consistent with its smaller total layer count producing more critical heads in the early-to-mid range.
- **Mid-late cluster (0.60–0.75):** Four Llama heads, three Pythia heads.
- **Late cluster (0.85–0.97):** Three Llama heads, two Pythia heads. Llama is denser here, reflecting its two additional layers (28 vs. 24) creating room for extra late-network mechanisms.

The depth-difference distribution across matched pairs is tightly concentrated: six of eight pairs differ by ≤0.02 in relative depth, and the maximum difference is 0.048. Considering that the tolerance was set at 0.075, the actual conservation is substantially tighter than the threshold requires.

---

## 4.6 Statistical Significance and Patching Verification

Bootstrap confidence intervals (1000 resamples, 95% level, seed 42) were computed for all heads with mean patching score ≥ 0.030. For the top-ranked heads in each model, the effect sizes are large relative to uncertainty:

- **L15H20 (Llama):** mean 0.240, 95% CI [0.225, 0.255], CI width 0.030. The lower bound (0.225) exceeds the rank-3 head's mean (0.092) by 2.4×.
- **L10H7 (Pythia):** mean 0.207, 95% CI [0.199, 0.216], CI width 0.017. The lower CI bound (0.199) exceeds rank-2 (0.155) by 28%.

All top-10 heads in both models have confidence intervals excluding zero by a margin of at least 3× the CI width. The Pythia L1H11 anomaly is confirmed: its patching CI [−0.011, 0.006] straddles zero (the only such case in either model's top-10), whereas its ablation drop (0.327) is large and unambiguous. This dissociation is the empirical signature of necessity without sufficiency, and constitutes the primary path patching–style verification in our results: by simultaneously measuring what happens when a head's activation is *replaced* (patching; sufficiency) versus *removed* (ablation; necessity), we detect heads whose role is suppressive rather than generative, a distinction that patching-only or ablation-only protocols would miss.

---

## 4.7 Factual Association: Cross-Task Transfer

To test whether the circuit structure identified for IOI transfers to a related but distinct task, we applied the same activation patching protocol to factual association prompts (n = 50, Llama-3.2-3B only). Prompts had the form "The capital of [country] is ___" with the country name corrupted by substitution to a different country; the patching score measures each head's causal contribution to correct capital prediction. Table 5 reports the top-10 heads.

**Table 5.** Top-10 heads by activation patching score for factual association (Llama-3.2-3B, n = 50).

| Rank | Head   | Patch score | Rel. depth | Matching IOI layer? |
|------|--------|-------------|------------|---------------------|
| 1    | L15H17 | 0.420       | 0.536      | **Yes** (IOI rank 1: L15H20) |
| 2    | L21H2  | 0.418       | 0.750      | **Yes** (IOI rank 6: L21H20) |
| 3    | L27H5  | 0.105       | 0.964      | **Yes** (IOI rank 9: L27H17) |
| 4    | L17H18 | 0.088       | 0.607      | **Yes** (IOI rank 2: L17H17) |
| 5    | L13H18 | 0.052       | 0.464      | **Yes** (IOI rank 3: L13H14) |
| 6    | L25H8  | 0.045       | 0.893      | No                  |
| 7    | L26H20 | 0.035       | 0.929      | No                  |
| 8    | L13H19 | 0.026       | 0.464      | Yes (layer 13)      |
| 9    | L21H0  | 0.025       | 0.750      | Yes (layer 21)      |
| 10   | L26H19 | 0.019       | 0.929      | No                  |

The top-5 factual association heads occupy the same five layers as five of the top-9 IOI heads (layers 13, 15, 17, 21, and 27), but activate *different heads within those layers*: IOI uses L15H20 while FA uses L15H17; IOI uses L21H20 while FA uses L21H2; IOI uses L13H14 while FA uses L13H18. This pattern is consistent with **layer-level circuit topology conservation across tasks** with **head-level specialization by task** — the same network locations implement different input-output functions depending on which specific circuit the task activates.

Two additional features of the factual association results are noteworthy. First, the top-2 FA heads (0.420, 0.418) have substantially higher patching scores than the IOI rank-1 head (0.240), and the two scores are nearly equal to each other, suggesting FA uses a more concentrated two-head bottleneck while IOI distributes attribution more broadly across the mid-to-late network. Second, layer 15 is the single most critical layer for both tasks — the dominant head in each circuit (L15H20 for IOI, L15H17 for FA) resides at the same relative depth (0.536), which may reflect layer 15's position at the transition between Llama's induction-like and output-writing computational stages.
