# 4. Results

> **Scope note.** This section reports results from three TopK SAE runs on Llama-3.2-3B, Mistral-7B-v0.3, and Qwen2.5-3B. Two of the three SAEs were stopped well short of the training target (Llama: 10k/50k steps; Mistral: 1k/~50k steps on a 50k-token subset). No L1 or Gated SAE variants were trained. The evaluation confound analysis (Section 4.4 in the outline) has not been conducted and is omitted here. Metrics marked as "batch FVE" are single-batch measurements from the training log, not held-out evaluation; they fluctuate significantly step to step and should be read as rough indicators rather than precise reconstruction quality estimates.

---

## 4.1 SAE Training Metrics

Table 1 reports training metrics for all three TopK SAEs at their final checkpoints. "Batch FVE" is $1 - \|x - \hat{x}\|_2^2 / \|x\|_2^2$ computed on the batch at the last logged training step; it varies substantially across batches (standard deviation on the order of 0.02–0.05 per-batch within a single run) and is not a stable summary statistic. No held-out evaluation set was used; held-out FVE values are not available.

"Dead features" is the count of dictionary features that did not activate on any sample in the last 5,000 training steps (the `dead_5k` field from the training log, which tracks a sliding window of 25 log intervals at 200-step logging frequency). The training code includes no auxiliary dead-feature revival loss; dead features receive no gradient signal and remain dead.

**Table 1.** TopK SAE training metrics.

| Model | Layer | $k$ (L0) | Training steps | Batch FVE (final step) | Dead features (last 5k steps) |
|---|---|---|---|---|---|
| Qwen2.5-3B | 18 | 128 | 50,000 (complete) | 0.984 | 9,401 / 16,384 (57.4%) |
| Llama-3.2-3B | 14 | 128 | 10,000 (partial, 20%) | 0.992 | 6,273 / 16,384 (38.3%) |
| Mistral-7B-v0.3 | 16 | 128 | 1,000 (very partial, ~2%) | 0.965 | 339 / 16,384 (2.1%) |

The high dead-feature rates for Qwen (57.4%) and Llama (38.3%) reflect the absence of an auxiliary revival loss. Without such a loss, TopK training naturally allows many features to go permanently unactivated once they fall below the top-$k$ threshold on every training example, as the encoder gradient for those features is zero. The Mistral dead-feature rate is low (2.1%) because training ran for only 1,000 steps — too few for most features to specialize or go permanently silent.

The batch FVE values should be read with caution. For Qwen, the last ten logged batch FVE values range from 0.895 to 0.992 across consecutive log steps, reflecting the high variance of single-batch estimates. The final-step values in the table are point estimates, not averages.

The Mistral SAE was trained on a 4-bit quantized version of Mistral-7B-v0.3 and on only 50,000 source tokens (compared to 500,000 for Llama and Qwen). Both factors limit the comparability of Mistral SAE features to those of the other two models.

---

## 4.2 Cross-Architecture Feature Matching

Cross-architecture matching used the chunk-averaged activation fingerprint method described in Section 3.4: 100-dimensional fingerprints computed over 50,000 tokens of wikitext-103, with a cosine similarity threshold of 0.80. Many-to-one matches are allowed.

**Table 2.** Pairwise cross-architecture feature matches (cosim ≥ 0.80) and three-way universal features.

| Pair | Matched pairs (cosim ≥ 0.80) |
|---|---|
| Llama-3.2-3B ↔ Qwen2.5-3B | 5,923 |
| Mistral-7B ↔ Qwen2.5-3B | 8,099 |
| Llama-3.2-3B ↔ Mistral-7B | 12,174 |
| **Three-way universal features** | **3,753** |

The Venn breakdown of matched features is: 11,155 Llama-only, 13,784 Qwen-only, 8,087 Mistral-only; 414 Llama–Qwen exclusive pairs; 1,261 Llama–Mistral exclusive pairs; 307 Qwen–Mistral exclusive pairs; and 3,753 appearing in all three models.

The high pairwise counts (5,923–12,174) relative to dictionary size (16,384) and the large number of three-way universal features (3,753, or ~23% of the Qwen dictionary) are notable, but their interpretation is complicated by methodological limitations. Many matched feature pairs show cosine similarity exactly 1.0 across all three pairwise comparisons. As discussed in Section 3.4, a 100-dimensional fingerprint derived from only 50,000 evaluation tokens is low-dimensional relative to the dictionary; features that happen to activate in the same single 500-token chunk across all three models will yield identical fingerprints and cosim = 1.0 by construction. This artifact inflates both match counts and the apparent universality rate.

The Llama–Mistral pair has the highest match count (12,174), which is unexpected given that Mistral's SAE was trained for only 1,000 steps. One possible explanation is that early-training SAE features in both models respond to broad distributional properties of the residual stream (e.g., overall activation norms, dominant principal components) before specializing into semantic or syntactic features; such broad features would be easy to match via activation pattern similarity. This interpretation is speculative and would require further investigation.

**What these numbers do not establish.** The matching results show that pairs of features across models have correlated chunk-averaged activation patterns. They do not establish semantic correspondence, interpretability, or meaningful functional equivalence. Semantic labeling of matched feature pairs — which would require inspecting maximally activating token contexts and assigning linguistic or functional labels — was not completed (see Section 3.5).

---

## 4.3 Top Universal Feature Pairs

Table 3 lists the ten feature triplets with the highest mean pairwise cosine similarity across all three model pairs among the 3,753 three-way universal features. Feature indices are given for all three models. Cosine similarity values are from the wikitext-103 evaluation corpus.

**Table 3.** Top-10 three-way universal feature triplets by mean pairwise cosim.

| Rank | Llama feat | Qwen feat | Mistral feat | cosim (LQ) | cosim (MQ) | cosim (LM) | Mean cosim |
|---|---|---|---|---|---|---|---|
| 1 | 12 | 8144 | 148 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 823 | 4597 | 106 | 1.000 | 1.000 | 1.000 | 1.000 |
| 3 | 959 | 11876 | 642 | 1.000 | 1.000 | 1.000 | 1.000 |
| 4 | 1296 | 10557 | 917 | 1.000 | 1.000 | 1.000 | 1.000 |
| 5 | 1449 | 3283 | 565 | 1.000 | 1.000 | 1.000 | 1.000 |
| 6 | 1831 | 3283 | 565 | 1.000 | 1.000 | 1.000 | 1.000 |
| 7 | 2224 | 4597 | 106 | 1.000 | 1.000 | 1.000 | 1.000 |
| 8 | 2266 | 4638 | 60 | 1.000 | 1.000 | 1.000 | 1.000 |
| 9 | 2749 | 10557 | 917 | 1.000 | 1.000 | 1.000 | 1.000 |
| 10 | 2761 | 10557 | 917 | 1.000 | 1.000 | 1.000 | 1.000 |

All top-10 entries show cosim = 1.0 in all three pairwise comparisons. Several entries share the same Qwen or Mistral feature index (e.g., Qwen feature 10557 and Mistral feature 917 appear as the matched counterpart of multiple distinct Llama features). This confirms the many-to-one matching artifact: the 100-dimensional fingerprint space collapses many Llama features onto the same Qwen or Mistral feature, because multiple Llama features fire in similar chunk patterns over the small evaluation corpus.

**Semantic labels are not available.** The feature examples files in `data/sae-analysis/` contain the top-20 maximally activating token contexts for each feature in each model, extracted from 500,000 evaluation tokens. Inspection of those files shows that the most-activated Qwen feature (feature 13961, which fires on 97% of tokens) activates strongly on wikitext-103 section-header tokens (`= = Section heading = =`), suggesting a high-frequency structural marker rather than a semantically specific feature. However, systematic semantic labeling of all 3,753 universal features — or even the top-ranked pairs — was not completed and is left for future work.

---

## 4.4 Summary of Limitations

The following limitations are primary constraints on the conclusions that can be drawn from this study:

1. **Partial SAE training.** The Llama SAE reached only 20% of its training target, and the Mistral SAE reached less than 2%. Features from partial SAEs may not yet have specialized into interpretable representations.

2. **Mistral confounds.** The Mistral model was loaded at 4-bit quantization (no bf16 checkpoint was accessible), and its SAE was trained on a 10× smaller corpus than the other two models. Cross-architecture comparisons involving Mistral are consequently less reliable than the Llama–Qwen comparison.

3. **High dead-feature rates.** Without an auxiliary dead-feature revival loss, 57% of Qwen features and 38% of Llama features show no activation in the last 5,000 training steps. These features contribute nothing to the learned dictionary. A future run should add an auxiliary loss (e.g., Gao et al., 2024) or a re-initialization scheme.

4. **Batch-level FVE only.** No held-out evaluation set was used; reconstruction quality is reported only as single-batch estimates from the training log, which fluctuate substantially.

5. **Small, low-dimensional evaluation corpus for matching.** The 50,000-token, 100-chunk evaluation corpus produces a 100-dimensional fingerprint per feature, which is insufficient to distinguish genuine functional correspondence from coincidental co-activation. The cosim = 1.0 artifact in the top-ranked matches reflects this limitation.

6. **No semantic labeling.** The interpretability of matched features has not been verified through maximal-activation inspection or probe-accuracy evaluation.
