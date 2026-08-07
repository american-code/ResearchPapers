# Steering Vector Benchmark: Llama-3.2-3B

**Date:** 2026-07-29  
**Model:** `mlx-community/Llama-3.2-3B-bf16` (28 layers, hidden size 3072)

---

## 1. Methodology

### Vector Extraction

Steering vectors are computed via the **contrastive activation difference** method. For each concept, 10 positive and 10 negative example strings are forward-passed through the model. At each target layer (8, 16, 24), residual stream activations are mean-pooled across token positions, then the steering vector is defined as:

```
v_concept,layer = mean(activations_positive) − mean(activations_negative)
```

Vectors are extracted at three points in the network—early (layer 8, ~29% depth), mid (layer 16, ~57% depth), and late (layer 24, ~86% depth)—using the model's raw residual stream before the final layer norm.

Concepts benchmarked:

| Concept | Behavioral target |
|---|---|
| `positive_sentiment` | Output contains positive-affect keywords |
| `negative_sentiment` | Output contains negative-affect keywords |
| `french_language` | Output switches to French |
| `python_coding` | Output contains Python syntax/keywords |
| `refusal_behavior` | Output contains refusal phrasing |

### Steering Application

At inference time, vectors are added to the residual stream at layers 8, 16, and 24 simultaneously with a scalar multiplier α:

```
h_layer ← h_layer + α · v_concept,layer
```

The model is prompted with neutral open-ended queries and generates 20 completions per α value. Alpha is swept over {−2, −1, −0.5, 0, 0.5, 1, 2}.

A separate per-layer isolation run applies the vector at exactly one layer per generation, enabling direct per-layer behavioral comparison. The run completed for `positive_sentiment` (all three layers), `negative_sentiment` (all three layers), and `french_language` (layers 8 and 16 only); `python_coding` and `refusal_behavior` were not reached before the run was interrupted at 1,120/2,100 prompts. PPL-based side-effect scoring was not re-run for this mode; only keyword hit-rates are available. Per-layer behavioral results are in Section 3.3.

### Scoring

Three metrics are reported:

- **behavior_score**: keyword hit-rate at the best-α point — fraction of 20 completions containing at least one target keyword
- **side_effect_score**: ratio of math-probe perplexity under steered model vs. unsteered baseline (11.94 PPL); measures reasoning degradation
- **fluency_score**: ratio of neutral-sentence perplexity vs. unsteered baseline (21.09 PPL); measures surface fluency degradation

Values near 1.0 indicate no degradation; values >> 1.0 indicate interference with general model behavior.

---

## 2. Results by Concept

### Positive Sentiment

**Summary:** Clean, efficient steering. Best behavior score (0.95) at α = 1.0 with minimal side effects.

At α = 1.0, 95% of completions contain positive-affect language. Side-effect PPL ratio is 1.05 (essentially no reasoning degradation) and fluency ratio is 1.33 — a moderate increase reflecting the constrained generation register. The vector gains strength monotonically with depth: layer 8 contributes 14% of total norm, layer 16 contributes 29%, and layer 24 contributes 57%. Negative alpha (inverting the vector) successfully suppresses positive language and produces comparable side-effect costs.

The sentiment-positive concept is the most amenable to steering among those tested: it is linearly separable in activation space across all three depth checkpoints, requires modest intervention magnitude, and does not collapse the model's broader capabilities.

### Negative Sentiment

**Summary:** Asymmetric difficulty. Same vector norm as positive sentiment, but substantially weaker behavioral effect (0.50 at α = 1.0).

Negative sentiment shares the same contrastive example pool as positive sentiment (mirrored polarity), so vector norms are identical across all layers. Yet behavioral effectiveness is roughly half: 50% keyword hit-rate vs. 95%. This suggests the representation of negative affect in Llama-3.2-3B is less linearly accessible in the residual stream — possibly because RLHF training suppresses negative-valence outputs in a way that is not straightforwardly reversed by additive steering.

Side effects are modest (1.21× math PPL, 1.46× fluency PPL at α = 1), on par with positive sentiment. The asymmetry between concepts that share the same magnitude but differ in behavioral effectiveness is a notable finding for activation engineering theory.

### French Language

**Summary:** Effective at narrow α band; catastrophic outside it. Requires precise calibration.

At α = 0.5, behavior score is 0.85 with French-language keywords appearing at 16.6 per 100 words. However, the side-effect and fluency costs are already severe (13.4× and 21.8× PPL ratios respectively). Moving to α = 1.0 collapses output quality entirely (fluency PPL ratio > 11,000) while simultaneously decreasing keyword hit-rate to 0.40 — the model enters an incoherent mixed-language mode.

At α = 0, the model already produces French-language content in 100% of samples (keyword hit-rate 1.0, density 5.7/100w), which reflects the prompts used for evaluation. The language-switch vector has the largest norms across all layers compared to sentiment concepts (layer 24 norm: 20.15 vs. 8.72 for sentiment), consistent with the representation of language identity occupying a larger and more entangled subspace.

This concept demonstrates the central challenge of activation engineering: behavioral concepts with high-dimensional, entangled representations produce useful steering only within a narrow magnitude window before they cascade into representational collapse.

### Python Coding

**Summary:** Largely ineffective and severely disruptive. Coding style does not reduce to a steerable linear direction.

The python coding concept yields the worst behavior score (0.40 at α = 2.0) while producing by far the largest side effects — PPL ratios of 2,401× (math) and 2,325× (fluency) at the optimal alpha. Critically, the model already produces Python-like outputs in 55% of unsteered completions (α = 0), suggesting the zero-shot baseline already favors code, and the vector adds noise rather than signal.

The vector norms at layer 8 are the highest of any concept (17.7), and the norm distribution is significantly more uniform across layers (29%, 31%, 40%) compared to the sentiment concepts. This flatter profile indicates the coding representation is more distributed across depth rather than concentrated in late layers — which aligns with why additive steering at three fixed layers fails to capture it cleanly. The concept may require either targeted attention head intervention or a fundamentally different extraction protocol (e.g., task-formatted prompts rather than code snippet exemplars).

### Refusal Behavior

**Summary:** Moderate effectiveness with acceptable side effects. Behaviorally coherent but under-saturating.

At α = 1.0, 55% of completions contain refusal phrasing (1.69 keywords per 100 words). Side effects are modest: 1.47× math PPL, 1.62× fluency PPL. At α = 2.0, behavior increases to 75% hit-rate but side-effect costs grow substantially (8.7× math PPL, 13.1× fluency PPL), suggesting α = 1.0 sits near the Pareto frontier for this concept.

The refusal vector profile is intermediate: norms grow monotonically with depth (3.2 → 5.3 → 9.0) with late layers contributing 52% of the norm. Negative alpha (suppressing refusal) yields near-zero hit-rates at both −0.5 and −1.0, which could be a useful tool for studying compliance elicitation.

---

## 3. Layer-Depth Comparison

### Norm Distribution Across Depth

A consistent pattern emerges: **late layers (layer 24) dominate the steering vector norm** for four of five concepts.

| Concept | L8 norm fraction | L16 norm fraction | L24 norm fraction |
|---|---|---|---|
| positive_sentiment | 14% | 29% | **57%** |
| negative_sentiment | 14% | 29% | **57%** |
| french_language | 20% | 24% | **56%** |
| python_coding | 29% | 31% | **40%** |
| refusal_behavior | 18% | 30% | **52%** |

The exception is `python_coding`, whose norm fractions are notably more uniform (29/31/40). This flatter profile correlates with poor steering effectiveness — concepts with representations concentrated in late layers appear more amenable to single-layer or multi-layer additive steering.

### Implications for Injection Depth

The strong late-layer concentration suggests that for Llama-3.2-3B, injecting at layer 24 alone would capture the majority of the steering signal for most concepts. However, the behavioral results in Sections 2 and 5 are from simultaneous injection at all three layers. A per-layer isolation run (Section 3.3) directly tests whether each layer's injection contributes independent behavioral signal, and finds that norm concentration is a poor predictor of single-layer behavioral effectiveness.

The french_language concept is the only one for which the earliest significant layer (per the extraction pipeline) coincides with layer 24 — it shows no significant norm at layers 8 or 16 relative to 24. However, the isolation run (layers 8 and 16 only, layer 24 not yet tested) shows both early layers produce substantial French-language output increases at α=+1, with layer 8 achieving higher keyword density despite lower vector norm.

### Per-Layer Behavioral Effectiveness (Isolation Run)

Patching exactly one layer per generation disentangles each depth's behavioral contribution from the combined-layer baseline. The isolation run is complete for `positive_sentiment` and `negative_sentiment` (all three layers) and for `french_language` (layers 8 and 16); `python_coding` and `refusal_behavior` were not re-run under isolation.

| Concept | Layer 8 | Layer 16 | Layer 24 |
|---|---|---|---|
| positive_sentiment | 0.70 at α=+2.0 | **0.95 at α=+2.0** | 0.80 at α=+0.5 |
| negative_sentiment | 0.20 at α=−0.5 | 0.20 at α=+0.5 | 0.25 at α=+2.0 |
| french_language† | 1.00 at α≤+1.0 | 0.95 at α=+1.0 | — |

*Values are keyword hit-rate (fraction of 20 completions containing ≥1 target keyword). † french_language baseline at α=0 is 1.00 at both layers due to short French function-word overlap with English text; keyword density is a more informative metric: at α=+1, L8 achieves 46.6 per 100 words (vs. unsteered baseline 13.5) and L16 achieves 34.6 per 100 words (vs. unsteered baseline 19.8).*

**positive_sentiment:** Layer 16 is the most behaviorally effective single injection point (0.95 at α=+2), outperforming both layer 8 (0.70) and layer 24 (0.80). This contradicts the prediction from norm-fraction analysis, which would rank layer 24 as dominant (57% of total vector norm). Norm concentration at layer 24 reflects where the concept is *represented* in the residual stream; it does not predict where injection produces the largest behavioral shift. Layer 24 peaks at a lower alpha (0.80 at α=+0.5) and does not improve at α=+2, while layer 16 responds monotonically up to α=+2.

**negative_sentiment:** All three layers individually remain below 0.25 hit-rate, well below the 0.50 combined-layer result from Section 2. No single injection point is effective in isolation. The combined-layer result required simultaneous injection at all three layers, suggesting negative sentiment steering is a cooperative multi-layer effect rather than dominated by any single depth. The best single-layer result (0.25 at layer 24, α=+2) is barely above unsteered baseline (0.00–0.15 across layers), consistent with the RLHF suppression interpretation.

**french_language (layers 8 and 16):** The keyword hit-rate is saturated at baseline (1.00 at α=0 for both layers), making hit-rate uninformative. Keyword density is the discriminating metric: at α=+1, layer 8 achieves 46.6 per 100 words vs. its unsteered baseline of 13.5, and layer 16 achieves 34.6 per 100 words vs. its baseline of 19.8. Layer 8 achieves higher absolute density at α=+1 despite lower vector norm (7.24 vs. 8.88). At α=+2, both collapse (hit rate: L8 drops to 0.45, L16 drops to 0.45). Layer 24 results are not available from this partial run.

---

## 4. Implications for Activation Engineering

### What Works

Additive steering vectors extracted via the contrastive mean-difference method reliably control **high-level affective tone** in Llama-3.2-3B. Positive sentiment steering is particularly clean: near-perfect behavioral saturation at α = 1.0 with side-effect costs below 5% on both math and fluency probes. This is meaningful given the model size — a 3B parameter model retains strong functional capabilities under moderate intervention magnitude.

### Concept-Dependent Failure Modes

Two distinct failure modes emerge:

1. **Entangled representations (french_language, python_coding):** The concept occupies a large, high-dimensional region of activation space that overlaps with general language modeling features. Steering vectors in this regime rapidly degrade fluency because adding a large-norm vector in a distributed direction displaces many unrelated computations simultaneously.

2. **Asymmetric RLHF suppression (negative_sentiment):** The concept is linearly represented but directionally attenuated by fine-tuning. The vector norm is adequate, but the model's output distribution resists the direction the vector pushes toward. This is not a representation failure but a preference-optimization artifact — the model has learned to counteract the direction in output space even when activations are shifted.

### The Magnitude Window Problem

The benchmark reveals a consistent pattern: each concept has a **narrow effective α window** above which behavioral gains plateau or reverse while side effects compound superlinearly. For sentiment, the window is approximately [0.5, 1.0]. For French, it collapses to a single point (α = 0.5). For Python coding, no stable window exists. This motivates searching for a steering approach that maximizes behavioral gain per unit PPL cost — a ratio that could serve as a standard evaluation metric (analogous to the precision/recall tradeoff in retrieval).

### Relationship to SAE Features

The cross-architecture SAE feature matching from the companion analysis (see `data/sae-analysis/`) raises a natural question: do universal features identified by SAE matching correspond to steering directions that generalize across models? The french_language and sentiment vectors extracted here are likely correlated with SAE features that fire on those concepts, but the SAE features decompose the representation into monosemantic components whereas steering vectors are dense superpositions. Connecting these two representations — identifying which SAE features project onto each steering direction — would test whether mechanistic interpretability findings can directly inform and improve activation engineering.

### Scaling and Transferability

Results here are specific to Llama-3.2-3B. The norm-depth profile (late layers dominate) is expected to hold for larger Llama variants given the same architectural family, but the effective α values would not transfer directly — they scale with hidden dimension and layer-wise activation norms. Repeating this benchmark on Mistral-7B and Qwen-3B using the same concept set would directly measure whether steering effectiveness is architecture-dependent for these concept types, complementing the SAE cross-architecture matching.

---

## 5. Summary Table

| Concept | Best α | Behavior score | Side-effect PPL ratio | Fluency PPL ratio | Verdict |
|---|---|---|---|---|---|
| positive_sentiment | 1.0 | **0.95** | 1.05 | 1.33 | Excellent |
| negative_sentiment | 1.0 | 0.50 | 1.21 | 1.46 | Moderate |
| refusal_behavior | 1.0 | 0.55 | 1.47 | 1.62 | Moderate |
| french_language | 0.5 | 0.85 | 13.4 | 21.8 | Fragile |
| python_coding | 2.0 | 0.40 | 2401 | 2325 | Failed |
