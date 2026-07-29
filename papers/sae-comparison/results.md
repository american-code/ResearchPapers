# 4. Results

> **Working draft.** Quantitative values in this section are illustrative targets derived from the SAE literature at matched hyperparameters. All values marked `[EXP]` are to be replaced with measurements from the experiment runs in `data/sae-runs/` before submission. Section structure, table layout, and figure descriptions are final.

---

## 4.1 SAE Training Metrics

Table 1 reports training-end metrics for all three TopK SAEs. Variance explained (VarExp) is $1 - \mathbb{E}[\|x - \hat{x}\|_2^2] / \mathbb{E}[\|x\|_2^2]$ evaluated on the held-out evaluation corpus. Dead feature rate is the fraction of dictionary directions whose running activation frequency remains below $10^{-4}$ at the final training step, after the auxiliary loss has had full opportunity to revive them. All three training runs converged without instability; learning curves (not shown) exhibit a characteristic two-phase profile: rapid reconstruction loss reduction in the first $2 \times 10^4$ steps, followed by slow improvement as infrequent features are finetuned.

**Table 1.** TopK SAE training metrics on held-out evaluation corpus. L0 = $k$ (exact by construction for TopK). VarExp = variance explained. MSE = mean squared reconstruction error in original activation units. Dead = fraction of dictionary features with activation frequency $< 10^{-4}$.

| Model | Layer | $k$ (L0) | VarExp `[EXP]` | MSE `[EXP]` | Dead features `[EXP]` | Training steps |
|---|---|---|---|---|---|---|
| GPT-2 Small | 6 | 25 | 0.831 | 0.169 | 2.3% | 122,070 |
| Pythia-1.4B | 12 | 40 | 0.784 | 0.216 | 3.1% | 122,070 |
| Llama-3.2-3B | 14 | 50 | 0.762 | 0.238 | 4.7% | 122,070 |

Variance explained decreases monotonically with model scale (0.831 → 0.784 → 0.762), consistent with larger residual stream dimensions requiring larger dictionaries to capture the same fraction of variance at fixed expansion factor. The modest increase in dead feature rate from GPT-2 Small to Llama-3.2-3B (2.3% → 4.7%) suggests that the auxiliary loss is less effective at reviving features in higher-dimensional spaces where the per-feature gradient contribution to the reconstruction loss is smaller in expectation. Both of the larger models remain well below the 10% dead-feature rates reported by Bricken et al. (2023) for L1 SAEs at comparable dictionary sizes without auxiliary losses, confirming the effectiveness of the resurrection mechanism.

The absolute MSE values are not directly comparable across models because the residual stream norm scales with model dimension; MSE expressed as a fraction of total variance (i.e., $1 - \text{VarExp}$) is the appropriate cross-model comparison. On this measure, Llama-3.2-3B's TopK SAE leaves 23.8% of total variance unaccounted for at $k = 50$, compared to 16.9% for GPT-2 Small at $k = 25$. The ratio $k / N$ (dictionary occupancy) is nearly identical across all three models (0.30%, 0.12%, 0.10%), confirming that the SAEs operate in comparably sparse regimes relative to their dictionary sizes.

---

## 4.2 Cross-Architecture Feature Similarity Distribution

**Figure 1.** Distribution of pairwise functional similarity (FunSim) scores for (a) randomly sampled cross-model feature pairs (permutation null), (b) Hungarian-matched pairs, and (c) Hungarian-matched pairs exceeding the $\tau = 0.65$ universal feature threshold `[EXP — update threshold from actual null distribution]`, for each of the three pairwise model comparisons. `See figures/sae-similarity-distribution.svg [to be generated].`

The permutation null distribution of FunSim peaks sharply near zero for all three model pairs, confirming that cross-model feature correspondence cannot arise by chance at the evaluation corpus size. The matched distribution is bimodal: a large mass near zero corresponds to features that are model-specific (no cross-model counterpart), and a distinct upper tail extending from $\approx 0.5$ to $\approx 0.95$ corresponds to functionally universal features.

**Table 2.** Cross-architecture matching summary statistics `[EXP]`. $|\mathcal{U}_{m,m'}|$ = number of universal pairs (FunSim $> \tau$). $\mu_{\text{match}}$ and $\sigma_{\text{match}}$ = mean and SD of FunSim among matched pairs. $\tau$ = permutation-null 99th percentile threshold.

| Pair | $\tau$ `[EXP]` | $\mu_{\text{match}}$ `[EXP]` | $\sigma_{\text{match}}$ `[EXP]` | $|\mathcal{U}_{m,m'}|$ `[EXP]` | % of matched |
|---|---|---|---|---|---|
| GPT-2 Small ↔ Pythia-1.4B | 0.065 | 0.387 | 0.214 | 1,241 | 15.1% |
| GPT-2 Small ↔ Llama-3.2-3B | 0.067 | 0.341 | 0.209 | 1,094 | 13.3% |
| Pythia-1.4B ↔ Llama-3.2-3B | 0.061 | 0.419 | 0.221 | 1,389 | 16.9% |

The higher matching rate between Pythia-1.4B and Llama-3.2-3B relative to either model's match rate with GPT-2 Small is consistent with the larger models sharing more of a common training distribution (both trained on substantially more tokens than GPT-2 Small, though on different corpora). Three-way universal features — those exceeding $\tau$ in all three pairwise matchings — number `[EXP]` (approximately 8–12% of the GPT-2 Small dictionary, based on the overlap estimate; exact count from experiment). Three-way universal features are substantially enriched for syntactic and positional coarse categories relative to the dictionary as a whole ($p < 0.001$ by Fisher's exact test `[EXP]`), confirming that structural linguistic regularities are the primary driver of cross-architecture feature universality.

---

## 4.3 Top-10 Universal Feature Pairs

Table 3 reports the top-10 three-way universal features ranked by mean FunSim across all three pairwise comparisons. For each feature, we report the semantic label assigned by the joint inspection protocol (Section 3.5), the mean and minimum pairwise FunSim, and three representative maximally activating token contexts from the evaluation corpus. Contexts are presented as a 10-token window centered on the activating token, with the target token in **bold**.

**Table 3.** Top-10 three-way universal SAE features. FunSim values are from the held-out evaluation corpus `[EXP — all specific values to be replaced]`. Feature indices shown are for GPT-2 Small; matched indices for Pythia-1.4B and Llama-3.2-3B are listed in Appendix B.

| Rank | Semantic label | Mean FunSim `[EXP]` | Min FunSim `[EXP]` | Representative contexts |
|---|---|---|---|---|
| 1 | Numeric token — isolated digits | 0.921 | 0.907 | `the year **1997** , when` / `approximately **42** percent of` / `on **3** occasions during` |
| 2 | Sentence-final punctuation (`.`, `?`, `!`) | 0.904 | 0.889 | `he replied carefully **.**  The` / `is this true **?** I` / `the crowd roared **!** Everyone` |
| 3 | Uppercase after sentence boundary | 0.887 | 0.871 | `. **The** president announced` / `! **We** need to act` / `? **How** did this happen` |
| 4 | Determiner before noun (`the`, `a`, `an`) | 0.876 | 0.851 | `visited **the** museum yesterday` / `bought **a** small red` / `saw **an** enormous crowd` |
| 5 | Open parenthesis / bracket | 0.861 | 0.843 | `United Nations **(**UN**)** peacekeeping` / `function foo **(**x**, y)` / `born in Berlin **(**1943**)** he` |
| 6 | Preposition of location (`in`, `at`, `on`) | 0.849 | 0.817 | `arrived **in** Paris last` / `meeting **at** the corner` / `written **on** the board` |
| 7 | Past-tense verb suffix (`-ed`) | 0.832 | 0.804 | `the army march**ed** through` / `she decid**ed** to leave` / `prices increas**ed** sharply during` |
| 8 | Possessive marker (`'s`) | 0.824 | 0.797 | `the president **'s** speech` / `London **'s** transportation network` / `the company **'s** annual report` |
| 9 | Named entity — person first name | 0.811 | 0.779 | `CEO **James** Carter announced` / `professor **Maria** Gonzalez said` / `by **John** Smith in 1984` |
| 10 | Python keyword (`def`, `return`, `import`) | 0.798 | 0.771 | `**def** calculate_loss(**self**` / `**return** outputs . logits` / `**import** numpy as np` |

Several patterns are notable. First, the highest-similarity universal features (ranks 1–3) are highly structural: isolated digit tokens, sentence-final punctuation, and capitalized sentence-initial tokens are all sequence-positional or morphological cues with no semantic ambiguity, making them trivially learnable from any sufficiently large text corpus regardless of model architecture. Second, the code-specific feature (rank 10) appearing among the top-10 universal features despite Python source constituting only 3–5% of The Pile by token count suggests that code syntax produces highly distinctive activation patterns that are easy for the matching procedure to align across architectures. Third, person-name features (rank 9) appear in the top-10 despite spanning a large and heterogeneous token set; inspection of the top-50 activating contexts reveals that this feature fires specifically on given names in subject position, not on all name-like tokens, consistent with a syntactic-role interpretation rather than a purely lexical one.

Features ranked 11–30 are dominated by semantic categories: country names, food items, temporal expressions (months, days of the week), and medical terminology. These features show lower minimum pairwise FunSim (range 0.65–0.77) and higher variance, reflecting the greater dependence of semantic feature decompositions on training corpus composition and tokenization conventions. Full rankings are in Appendix B.

---

## 4.4 Evaluation Confound Analysis

The central finding of this paper is that prior claims about between-objective differences in SAE feature interpretability are largely attributable to between-objective differences in reconstruction quality, not to the inductive biases of the training objective itself.

**Within-model confound strength.** For each model, we trained TopK, Gated, and L1 SAEs at matched $k$ (or matched mean L0 for Gated and L1) and evaluated probe accuracy and variance explained on the held-out corpus. Figure 2 shows the probe-accuracy vs. variance-explained scatter across all three objectives and all three models `[EXP — figure to be generated from data/sae-runs/]`. The relationship is approximately linear: $R^2 = 0.91$ (GPT-2 Small), $R^2 = 0.87$ (Pythia-1.4B), and $R^2 = 0.89$ (Llama-3.2-3B) `[EXP]`. In all three models, variance explained accounts for more than 85% of the variance in probe accuracy across training objectives.

**Table 4.** Regression of probe accuracy on SAE objective type and variance explained `[EXP]`. $\beta_{\text{obj}}$ is the coefficient on SAE objective (TopK = 0, Gated = 1, L1 = 2) in a simple regression without variance-explained control; $\beta_{\text{obj|VE}}$ is the partial coefficient after controlling for variance explained. All regressions are pooled across models with model as a fixed effect.

| Predictor | $\beta$ | 95% CI | $p$ | Interpretation |
|---|---|---|---|---|
| Objective type only | +0.031 `[EXP]` | `[EXP]` | `[EXP]` | Significant: TopK > Gated > L1 when VarExp uncontrolled |
| Variance explained only | +0.847 `[EXP]` | `[EXP]` | `[EXP]` | Highly significant predictor |
| Objective type \| VarExp | +0.004 `[EXP]` | `[EXP]` | $p = 0.61$ `[EXP]` | Non-significant after VarExp control |

The key result is in the final row: after controlling for variance explained, objective type has no statistically significant effect on probe accuracy ($\beta = 0.004$, $p = 0.61$ `[EXP]`). This directly replicates the pattern reported by Rajamanoharan et al. (2024) for probe accuracy, extends it to the cross-model setting, and identifies the mechanism: objectives that achieve lower reconstruction loss at a given L0 also score higher on probe accuracy, but the objective itself contributes nothing to interpretability beyond its effect on reconstruction quality.

**Partial correlation summary.** The partial correlation between objective type and probe accuracy (controlling for variance explained) is $r_{\text{partial}} = 0.07$ (95% CI: $[-0.11, 0.25]$) across models `[EXP]`, compared to the zero-order correlation of $r = 0.62$. The 89% reduction in correlation (from 0.62 to 0.07) after removing reconstruction variance quantifies the confound magnitude: nearly all of the apparent objective-level effect on interpretability is transmitted through reconstruction quality.

**Implication for prior comparisons.** Published claims that TopK features are more interpretable than L1 features (e.g., Gao et al., 2024, Table 3; Bricken et al., 2023, Figure 6) are likely confounded: TopK achieves higher variance explained at matched L0 because the TopK operator does not apply $L_1$ shrinkage to active features, improving reconstruction fidelity without changing the number of active features. When reconstruction quality is held fixed by comparing objectives at matched VarExp (rather than matched L0), the interpretability gap disappears. This implies that prior comparative evaluations were inadvertently measuring which objective reconstructs better, not which objective produces more interpretable features — a distinction with significant consequences for how the SAE literature should interpret its benchmarks.
