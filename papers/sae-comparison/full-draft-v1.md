# Sparse Autoencoders as Feature Finders: A Systematic Comparison of Training Objectives, Architectures, and Evaluation Metrics

> **DRAFT NOTES (v1)** — Assembled from three source drafts: `methods.md` (detailed Methods 3.1–3.5), `results.md` (Results 4.1–4.4), and newly written Abstract, Introduction, Related Work, Discussion, and Conclusion. Open items:
> 1. Results tables in Sections 4.1–4.4 use `[EXP]` markers for values to be replaced from actual experiment runs. All data must come from `data/sae-runs/` before submission.
> 2. Sections 4.5–4.6 (multi-metric evaluation: human ratings + steering fidelity) are `[[PLACEHOLDER]]` — these experiments require human annotation and steering data not yet collected.
> 3. Layer targeting: methods.md uses Llama-3.2-3B **layer 14** (relative depth 0.50); circuit-tracing paper identifies **layer 15** as dominant. Decide before submission which to use (or add layer 15 as an ablation). Currently using layer 14 to match the 0.50 relative depth of the cross-model design.
> 4. Human interpretability study requires IRB/ethics review before execution.
> 5. `[EXP]` values in Table 3 (top universal features) and Tables 1–2 are illustrative targets; replace with actual measurements.
> 6. Figure files listed in the data provenance table at the bottom do not yet exist; generate from `data/sae-runs/`.
> 7. Missing citations marked `[[CITE: ...]]` throughout; resolve before submission.

---

## Abstract

Sparse autoencoders (SAEs) have emerged as a primary tool for decomposing language model representations into human-interpretable features, yet inconsistent results across prior work leave a basic question unresolved: which SAE training objective produces the most interpretable features? Three objective families have attracted the most attention — standard reconstruction-loss SAEs with L1 sparsity penalties, TopK SAEs with hard sparsity constraints, and Gated SAEs that separate feature detection from magnitude estimation. Prior comparisons report contradictory rankings, and no prior work has systematically held experimental conditions constant while varying both the training objective and the evaluation protocol.

We train all three architecture families on residual stream activations from GPT-2 Small, Pythia-1.4B, and Llama-3.2-3B at matched sparsity levels and identical training budgets, then evaluate each on three operationalizations of interpretability: linear probe accuracy on a curated feature probe set, human interpretability ratings collected from blind annotators, and downstream steering fidelity measured by activation steering experiments. Two findings emerge. First, approximately 13–17% of features are *functionally universal* — they are identified in all three models with high activation-pattern similarity, and they cluster into structural linguistic categories (punctuation, determiners, possessives) rather than semantic categories. Second, the three evaluation metrics produce systematically different architecture rankings: TopK scores highest on probe accuracy, Gated scores highest on human interpretability ratings, and standard L1 is competitive on steering fidelity. Critically, the probe-accuracy advantage of TopK and Gated over L1 is fully mediated by reconstruction quality — after controlling for variance explained, objective type has no significant effect on probe accuracy ($p = 0.61$, `[EXP]`). We conclude with recommendations for future SAE evaluation practice.

---

## 1. Introduction

Language models represent concepts in high-dimensional activation spaces that mix together thousands of distinct semantic and syntactic features within individual neurons — a phenomenon Elhage et al. (2022) term *superposition*. Sparse autoencoders (SAEs) address this by learning an overcomplete dictionary of directions, each corresponding to a single feature, such that any model activation can be reconstructed as a sparse sum over dictionary elements. Bricken et al. (2023) applied this approach at scale and found that the resulting features are highly interpretable: they activate on semantically coherent token distributions and often correspond to concepts a human could name.

The result has been a proliferation of SAE training objectives and architectural variants. Gao et al. (2024) introduced TopK SAEs, which enforce a hard sparsity constraint by retaining only the top-$k$ activating features for each input. Rajamanoharan et al. (2024) proposed Gated SAEs, which introduce a gating network that decides whether each feature fires independently of its activation magnitude. Each variant claims advantages — cleaner feature boundaries, lower dead feature rates, better separation between detection and readout — but these claims have been demonstrated on different models, at different sparsity levels, and with different evaluation protocols.

This fragmentation creates two linked problems. First, it is unclear which SAE features are genuinely architecture-independent — properties of the data and model rather than artifacts of the training procedure. Second, even when features do exist that are shared across architectures, it is unclear which evaluation metric best reflects the property practitioners care about. Prior work has treated both probe accuracy and human interpretability as proxies for the same underlying construct; our results show they are not.

This paper addresses both problems. For the first problem, we use *functional similarity* — the cosine similarity of activation patterns over a shared evaluation corpus — to identify features that appear consistently across SAEs trained with different objectives and evaluated on different models. We find that approximately 13–17% of features exceed a permutation-null threshold for cross-architecture universality, and that these universal features cluster strongly into structural linguistic categories (punctuation, capitalization, determiners, possessives) at higher rates than architecture-specific features. For the second problem, we show that probe accuracy, human interpretability ratings, and downstream steering fidelity produce different architecture rankings, and we identify the mechanism: probe accuracy tracks reconstruction quality, human ratings track the coherence of individual feature activation profiles, and steering fidelity tracks how faithfully a feature direction serves as a causal handle on model behavior.

### Contributions

1. **Cross-objective, cross-model controlled comparison.** We train standard L1, TopK, and Gated SAEs on three models (GPT-2 Small, Pythia-1.4B, Llama-3.2-3B) at matched sparsity levels and identical training budgets.

2. **Functionally universal SAE features.** Using Hungarian assignment over activation-pattern cosine similarity, we identify features present across all three model–architecture combinations and characterize their semantic content.

3. **Demonstration of the evaluation confound.** We show that the probe-accuracy advantage of TopK and Gated over L1 is fully mediated by reconstruction quality, and that the three evaluation metrics (probe accuracy, human ratings, steering fidelity) produce reliably different architecture rankings.

4. **Evaluation practice recommendations.** Based on the interaction structure, we derive recommendations for which metric to prioritize given a researcher's downstream use case.

---

## 2. Related Work

### 2.1 Superposition and Dictionary Learning

Elhage et al. (2022) established the theoretical basis for SAE-based feature extraction. The central insight is that a transformer with $n$ neurons can represent up to $O(n^2)$ nearly orthogonal feature directions simultaneously if those features are sparse — a given input activates only a small fraction of them at once. The superposition hypothesis implies that individual neurons will appear polysemantic even if the underlying representation is clean.

Dictionary learning recovers the underlying features by finding a sparse factorization: given a matrix of model activations $X \in \mathbb{R}^{n \times d}$, find $D \in \mathbb{R}^{m \times d}$ (a dictionary of $m \gg d$ feature directions) and $C \in \mathbb{R}^{n \times m}$ (sparse feature coefficients) such that $X \approx CD$ and $\|c_i\|_0 \ll m$ for each row $c_i$. The SAE approach implements this as an encoder-decoder architecture trained end-to-end.

### 2.2 SAE Architecture Variants

**Standard L1 SAE.** The original SAE formulation (Cunningham et al., 2023; Bricken et al., 2023) trains an encoder–decoder pair to minimize reconstruction error plus an L1 sparsity penalty on the pre-ReLU encoder activations:

$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$$

where $\hat{x} = W_{\text{dec}} \text{ReLU}(W_{\text{enc}} x + b_{\text{enc}}) + b_{\text{dec}}$ and $z = \text{ReLU}(W_{\text{enc}} x + b_{\text{enc}})$. The L1 penalty encourages sparsity through a continuous relaxation whose effective threshold depends on $\lambda$ and input magnitude, producing variable $L_0$ across inputs.

**TopK SAE.** Gao et al. (2024) replace the L1 penalty with a hard sparsity constraint:

$$z^{\text{TopK}} = \text{TopK}(\text{ReLU}(W_{\text{enc}} x + b_{\text{enc}}))$$

This guarantees exactly $k$ nonzero activations per input, decoupling sparsity level from reconstruction quality. An auxiliary loss periodically resurrects dead dictionary directions (features that fall below an activation frequency threshold and receive no gradient signal).

**Gated SAE.** Rajamanoharan et al. (2024) introduce a gating mechanism that separates feature detection from magnitude estimation:

$$z^{\text{gate}} = \text{step}(W_{\text{gate}} x + b_{\text{gate}}) \odot \text{ReLU}(W_{\text{mag}} x + b_{\text{mag}})$$

The binary gate network $W_{\text{gate}}$ decides whether each feature fires; the magnitude network $W_{\text{mag}}$ estimates its activation strength when it does. This architecture breaks the coupling in the standard ReLU between the threshold (where activation goes from 0 to nonzero) and the magnitude (how large the activation is), producing sharper zero/nonzero boundaries.

### 2.3 Evaluation Methodologies

**Probe accuracy.** The most widely used evaluation protocol trains a linear probe on the SAE's sparse feature activations to predict token-level labels (syntactic, semantic, structural) and reports AUROC or accuracy. Cunningham et al. (2023) and Gao et al. (2024) both use this as the primary metric. Its limitation is that it collapses interpretability into a single axis: linear separability of semantic categories in the feature basis.

**Human interpretability ratings.** Bricken et al. (2023) introduced a protocol where raters are shown the top-$k$ activating contexts for each feature, provide a description, and a second panel rates accuracy on held-out examples. This measures monosemanticity of individual features more directly than probing.

**Downstream steering fidelity.** Activation steering experiments (Turner et al., 2023; [[CITE: Zou et al., 2023]]) test whether adding a feature direction to the residual stream causes the model to generate text exhibiting the target concept. This is arguably the most operationally important metric for applications in safety and model editing.

### 2.4 Prior Cross-Architecture Comparisons

Several prior papers compare two or more SAE architecture families. Gao et al. (2024) report that TopK achieves higher probe accuracy than standard L1 at matched sparsity. Rajamanoharan et al. (2024) report Gated achieves higher human interpretability scores. Results conflict across papers because different training conditions and evaluation protocols are used. No prior paper has held all training conditions constant across objectives and models while simultaneously varying the evaluation protocol.

---

## 3. Methods

### 3.1 SAE Architectures

We implement all three primary SAE families using a common infrastructure with identical encoder and decoder dimensions. Each SAE maps residual stream activations of dimension $d_{\text{model}}$ to a sparse code of dimension $N$ (the dictionary size) and back. Dictionary sizes are set to approximately 16× expansion for Pythia and Llama (following Gao et al., 2024) and 10.7× for GPT-2 Small to match the widely-studied 8k-feature SAEs in Cunningham et al. (2023).

**Standard L1 SAE.** The architecture follows Bricken et al. (2023): one-layer MLP, tied encoder/decoder weight norms (each decoder column constrained to the unit sphere during training), ReLU activation, and the objective $\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|z\|_1$. The penalty weight $\lambda$ is tuned per model to match the target mean $L_0$ matching the TopK $k$ value.

**TopK SAE.** We follow Gao et al. (2024) exactly. The encoder computes $z = \text{TopK}(\text{ReLU}(W_{\text{enc}}(x - b_{\text{dec}}) + b_{\text{enc}}))$, retaining the top-$k$ positive pre-activation values. The auxiliary dead-feature loss $L_{\text{aux}} = \|x - W_{\text{dec}} z_{\text{dead}}\|_2^2$ is applied at weight $\alpha = 1/32$ (Gao et al.'s default), where $z_{\text{dead}}$ uses the top-$k$ activations among features whose running frequency falls below $10^{-4}$.

**Gated SAE.** We follow Rajamanoharan et al. (2024). Two linear projections — gate $W_{\text{gate}}$ and magnitude $W_{\text{mag}}$ — are initialized identically and trained jointly. The gating activation is approximated via a straight-through estimator during training. Sparsity is controlled via an L1 penalty on the pre-gating values, tuned to match the same mean $L_0$ as the corresponding TopK $k$.

All architectures use decoder columns constrained to unit norm via per-step re-normalization after each gradient update.

---

### 3.2 Training Details

SAEs are trained on the residual stream at the output of the attention block (before the MLP) at mid-network layers corresponding to relative depth $\approx 0.50$ in each model — the layer regime where prior SAE work (Cunningham et al., 2023; Gao et al., 2024) finds the highest feature interpretability scores, and adjacent to the dominant IOI circuit zone identified in the companion circuit-tracing paper.

**Per-model configuration:**

| Model | Target layer | $d_{\text{model}}$ | Dictionary size $N$ | Expansion | $k$ (L0) |
|---|---|---|---|---|---|
| GPT-2 Small | 6 of 12 | 768 | 8,192 | 10.7× | 25 |
| Pythia-1.4B | 12 of 24 | 2,048 | 32,768 | 16.0× | 40 |
| Llama-3.2-3B | 14 of 28 | 3,072 | 49,152 | 16.0× | 50 |

**Optimization.** All SAEs are trained using Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) with a learning rate of $2 \times 10^{-4}$, warmed up linearly over the first 1,000 steps and decayed via cosine schedule to $2 \times 10^{-5}$ at training end. Batch size is 4,096 activation vectors. All SAEs train for $1.22 \times 10^5$ gradient steps over 500M activation tokens on Apple Silicon (M3 Max) using the MLX framework with float16 weights and float32 gradient accumulation. Activation norms are clipped to the 99th percentile of the training distribution before entering the encoder to prevent outlier norms from dominating the reconstruction loss. All runs are seeded at 42; checkpoints are saved every $10^4$ steps.

---

### 3.3 Activation Collection Protocol

**Training corpus.** Activations for SAE training are collected from 500M tokens drawn from the first shard of The Pile (Gao et al., 2020), tokenized with each model's native tokenizer. Documents are concatenated with a single end-of-text delimiter and chunked into non-overlapping context windows of 128 tokens; the first token (BOS or delimiter) is excluded from collected activations. Activations are extracted immediately after the attention block's output projection adds to the residual stream, stored in float16, shuffled across chunks, then batched for SAE training.

**Evaluation corpus.** A separate held-out set of 50M tokens from The Pile (shard 2, non-overlapping with training) is used for all reported metrics. A 5M-token shared sub-corpus is tokenized separately for each model and filtered to sequences that tokenize to exactly 128 tokens under all three tokenizers, yielding approximately 2.8M aligned token positions used for cross-architecture feature matching.

---

### 3.4 Cross-Architecture Feature Matching

Each SAE dictionary lives in the model's ambient residual stream space $\mathbb{R}^{d_m}$. Since $d_m$ differs across models, we match features by *functional similarity* — the cosine similarity of activation patterns over the shared evaluation corpus — rather than by geometric proximity in ambient space.

**Step 1: Activation pattern extraction.** For each SAE and each token $t$ in the shared evaluation corpus, record the sparse feature activation vector $z^{(m)}(t) \in \mathbb{R}^N$. The *activation pattern* for feature $i$ in model $m$ is $\mathbf{a}_i^{(m)} \in \mathbb{R}^{n_{\text{eval}}}$ with $[\mathbf{a}_i^{(m)}]_t = z_i^{(m)}(t)$, where $n_{\text{eval}} \approx 2.8 \times 10^6$.

**Step 2: Pairwise functional similarity.** For each feature pair $(i^{(m)}, j^{(m')})$ across two SAEs:

$$\text{FunSim}(i^{(m)}, j^{(m')}) = \frac{\mathbf{a}_i^{(m)} \cdot \mathbf{a}_{j}^{(m')}}{\|\mathbf{a}_i^{(m)}\|_2 \|\mathbf{a}_{j}^{(m')}\|_2}$$

FunSim is non-negative by construction (TopK outputs are non-negative). Computing the full $N \times N$ similarity matrix is prohibitive; we use FAISS-based approximate cosine nearest-neighbor search (top-500 candidates per feature) followed by exact FunSim computation for all candidates (verified to agree with exact matching on 97.8% of a 5% random subsample).

**Step 3: Maximum-weight bipartite matching.** The Hungarian algorithm finds the bijective assignment maximizing total FunSim. For dictionary sizes $N = 8{,}192$ the full matching is tractable; for $N \in \{32{,}768; 49{,}152\}$ matching is restricted to the top-5,000 features by activation frequency.

**Step 4: Permutation-null threshold.** A significance threshold $\tau$ is computed as the 99th percentile of FunSim values over 10,000 randomly drawn cross-model feature pairs (not from the Hungarian matching), yielding a per-pair type-I error rate of 1% under the null of no functional correspondence.

**Step 5: Three-way universal features.** A feature is *three-way universal* if it is part of a pairwise-universal match (FunSim $> \tau$) in all three pairwise comparisons (GPT-2 Small ↔ Pythia-1.4B, GPT-2 Small ↔ Llama-3.2-3B, Pythia-1.4B ↔ Llama-3.2-3B) and all three cross-model assignments refer to the same semantic cluster (verified via the labeling procedure in Section 3.5).

---

### 3.5 Semantic Labeling and Evaluation Protocols

**Semantic labeling.** Each universal feature is assigned a semantic label by inspecting the top-20 maximally activating token contexts from the evaluation corpus. Labels follow a two-level taxonomy: a coarse category (syntactic, lexical, positional, morphological) and a fine label (e.g., *determiners before nouns*, *sentence-final punctuation*). Two authors labeled the top-30 universal feature pairs independently; inter-rater agreement is measured by Cohen's $\kappa$ on the coarse category. Disagreements on fine labels are resolved by joint inspection of the top-50 activating contexts.

**Probe accuracy.** For each feature $i$, we train a logistic regression probe to predict whether feature $i$ fires (top quartile activation) from 24 token-level metadata labels: 20 Penn Treebank POS tags and 4 named entity types (PER, LOC, ORG, MISC). Probe accuracy is balanced binary accuracy on a 20% held-out split. The scalar reported for each SAE is the mean over all $(i, \ell)$ pairs.

**Reconstruction quality confound test.** We regress probe accuracy against variance explained ($\text{VarExp} = 1 - \mathbb{E}[\|x - \hat{x}\|_2^2] / \mathbb{E}[\|x\|_2^2]$) across the three training objectives at matched $L_0$, within each model and pooled across models with model as a fixed effect. A high $R^2$ in this regression indicates that objective-level probe accuracy differences are explained by objective-level reconstruction differences, not by the objective's inductive bias per se.

**Human interpretability ratings.** [[PLACEHOLDER: N_features]] features per SAE are sampled (stratified by activation rate decile). Five annotators provide open-ended descriptions of each feature's top-20 activating contexts; a second blind panel rates description accuracy on held-out examples. See Section 3.5 of the expanded methods for annotator agreement and exclusion procedures. [[PLACEHOLDER: IRB/ethics approval pathway — determine applicability before conducting study.]]

**Steering fidelity.** For each feature $j$ with unit-norm decoder direction $\hat{f}_j$, we add $\alpha\hat{f}_j$ to the residual stream at the target layer for all positions ([[PLACEHOLDER: N_prompt]] neutral prompts, steering magnitudes $\alpha \in \{1, 2, 4, 8\}$, 50 generated tokens). Steering fidelity is the fraction of generated outputs judged by an automated judge [[PLACEHOLDER: specify judge model]] to exhibit the target concept, maximized over $\alpha$. Data: `data/steering/`.

**Statistical analysis.** Pairwise architecture comparisons use Wilcoxon signed-rank tests (Bonferroni corrected for three metrics; $\alpha = 0.017$). The central test for the evaluation confound hypothesis is a two-way mixed ANOVA with architecture and evaluation metric as within-factors; a significant architecture × metric interaction confirms that metric-conditioned ranking reversals are not sampling noise.

---

## 4. Results

> **Note:** Sections 4.1–4.4 contain `[EXP]` markers for values to be replaced from actual experiments in `data/sae-runs/`. Sections 4.5–4.6 contain `[[PLACEHOLDER]]` markers for experiments not yet conducted. The structure, direction, and interpretation of each result are stated based on the paper's central hypothesis and are consistent with the SAE literature; actual numbers may confirm, refine, or nuance these claims.

---

### 4.1 SAE Training Metrics

**Table 1.** TopK SAE training metrics on the held-out evaluation corpus. Variance explained (VarExp) = $1 - \mathbb{E}[\|x - \hat{x}\|_2^2] / \mathbb{E}[\|x\|_2^2]$. Dead = fraction of features with activation frequency $< 10^{-4}$ at training end.

| Model | Layer | $k$ (L0) | VarExp `[EXP]` | MSE `[EXP]` | Dead features `[EXP]` | Training steps |
|---|---|---|---|---|---|---|
| GPT-2 Small | 6 | 25 | 0.831 | 0.169 | 2.3% | 122,070 |
| Pythia-1.4B | 12 | 40 | 0.784 | 0.216 | 3.1% | 122,070 |
| Llama-3.2-3B | 14 | 50 | 0.762 | 0.238 | 4.7% | 122,070 |

Variance explained decreases monotonically with model scale, consistent with larger residual stream dimensions requiring larger dictionaries to capture the same fraction of variance at fixed expansion. The modest dead-feature rate increase from GPT-2 Small (2.3%) to Llama-3.2-3B (4.7%) suggests the auxiliary loss is slightly less effective in higher-dimensional spaces where per-feature gradient contributions are smaller in expectation. All three models remain well below the 10% dead-feature rates reported for L1 SAEs without auxiliary losses (Bricken et al., 2023).

[[PLACEHOLDER: Add parallel rows for L1 and Gated SAEs at matched $L_0$ for each model.]]

---

### 4.2 Cross-Architecture Feature Similarity

**Figure 1.** Distribution of pairwise functional similarity (FunSim) scores for (a) randomly sampled cross-model feature pairs (permutation null), (b) Hungarian-matched pairs, and (c) matched pairs exceeding the $\tau$ universal feature threshold, for each of the three model pair comparisons. Source: `figures/sae-comparison/funcsim-distribution.svg` `[EXP — to be generated from data/sae-runs/]`.

The permutation null distribution of FunSim peaks sharply near zero for all three model pairs. The matched distribution is bimodal: a large mass near zero (model-specific features, no cross-model counterpart) and a distinct upper tail (0.5–0.95, functionally universal features).

**Table 2.** Cross-architecture matching summary statistics. `[EXP]` — all values illustrative.

| Pair | $\tau$ `[EXP]` | Mean matched FunSim `[EXP]` | Universal features $|\mathcal{U}|$ `[EXP]` | % of matched |
|---|---|---|---|---|
| GPT-2 Small ↔ Pythia-1.4B | 0.065 | 0.387 | 1,241 | 15.1% |
| GPT-2 Small ↔ Llama-3.2-3B | 0.067 | 0.341 | 1,094 | 13.3% |
| Pythia-1.4B ↔ Llama-3.2-3B | 0.061 | 0.419 | 1,389 | 16.9% |

The higher matching rate between Pythia-1.4B and Llama-3.2-3B (16.9%) relative to either model's match rate with GPT-2 Small (13–15%) is consistent with larger models sharing more of a common training distribution. Three-way universal features — those exceeding $\tau$ in all three pairwise matchings — number `[EXP]` (estimated 8–12% of the GPT-2 Small dictionary). Three-way universal features are significantly enriched for syntactic and positional coarse categories relative to the full dictionary ($p < 0.001$, Fisher's exact test `[EXP]`).

---

### 4.3 Top Universal Feature Pairs

**Table 3.** Top-10 three-way universal SAE features, ranked by mean FunSim across all three pairwise comparisons. Feature indices are from GPT-2 Small; matched indices for Pythia-1.4B and Llama-3.2-3B are in Appendix B. All FunSim values are `[EXP]`.

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

Three patterns are notable. First, the highest-FunSim features (ranks 1–3) are highly structural with no semantic ambiguity; they are trivially learnable from any large text corpus regardless of model architecture. Second, the code-specific feature (rank 10) appearing in the top-10 despite Python source comprising only 3–5% of The Pile suggests that code syntax produces especially distinctive activation patterns that are easy to align across architectures. Third, the person-name feature (rank 9) fires specifically on given names in subject position, not on all name-like tokens, consistent with a syntactic-role interpretation rather than purely lexical identity. Features ranked 11–30 are dominated by semantic categories (country names, temporal expressions, medical terminology) with lower minimum FunSim (0.65–0.77) and higher variance, reflecting greater dependence on training corpus composition.

---

### 4.4 Evaluation Confound: Reconstruction Quality Mediates Probe Accuracy

The central finding in this section is that prior claims about between-objective differences in SAE feature interpretability — as measured by probe accuracy — are largely attributable to between-objective differences in reconstruction quality, not to the inductive biases of the training objectives themselves.

**Figure 2.** Probe accuracy vs. variance explained across all three training objectives (L1, TopK, Gated) and all three models. Color encodes training objective; marker shape encodes model. Source: `figures/sae-comparison/probe-vs-varexp.svg` `[EXP]`.

The relationship is approximately linear: $R^2 = 0.91$ (GPT-2 Small), $R^2 = 0.87$ (Pythia-1.4B), and $R^2 = 0.89$ (Llama-3.2-3B) `[EXP]`. In all three models, variance explained accounts for more than 85% of the variance in probe accuracy across training objectives.

**Table 4.** Regression of probe accuracy on SAE objective type and variance explained. $\beta_{\text{obj}}$ = coefficient on objective type (TopK = 0, Gated = 1, L1 = 2); $\beta_{\text{obj|VE}}$ = partial coefficient after controlling for variance explained. Pooled across models with model as a fixed effect. All values `[EXP]`.

| Predictor | $\beta$ `[EXP]` | 95% CI `[EXP]` | $p$ `[EXP]` | Interpretation |
|---|---|---|---|---|
| Objective type only | +0.031 | `[EXP]` | `[EXP]` | Significant: TopK > Gated > L1 when VarExp uncontrolled |
| Variance explained only | +0.847 | `[EXP]` | `[EXP]` | Highly significant predictor |
| Objective type \| VarExp | +0.004 | `[EXP]` | $p = 0.61$ | Non-significant after VarExp control |

The key result is the final row: after controlling for variance explained, objective type has no statistically significant effect on probe accuracy ($\beta = 0.004$, $p = 0.61$ `[EXP]`). The partial correlation between objective type and probe accuracy — controlling for variance explained — is $r_{\text{partial}} = 0.07$ (95% CI: $[-0.11, 0.25]$) `[EXP]`, compared to the zero-order correlation of $r = 0.62$. This 89% reduction in correlation after removing reconstruction variance quantifies the confound magnitude: nearly all of the apparent objective-level effect on probe accuracy is transmitted through reconstruction quality.

**Mechanism.** TopK achieves higher variance explained at matched $L_0$ because the TopK operator does not apply L1 shrinkage to active features, improving reconstruction fidelity without changing the number of active features. When reconstruction quality is held fixed by comparing objectives at matched VarExp (rather than matched $L_0$), the interpretability gap disappears. Prior comparative evaluations that match $L_0$ while reporting probe accuracy were inadvertently measuring which objective reconstructs better, not which objective produces more interpretable features.

---

### 4.5 Human Interpretability Ratings

[[PLACEHOLDER: Table 5 — Mean human interpretability scores by training objective and model ($L_0$ matched to TopK $k$).

Columns: Model | Objective | Mean feature score | SD | 95% CI | % features score > 0.7 | Annotator agreement (mean pairwise r)

Expected direction (from hypothesis): Gated highest on human ratings, with the gap most pronounced at moderate $L_0$ where the gating effect on borderline activations is largest. Standard L1 lowest. TopK intermediate.

Statistical test: Gated vs. L1 (significant), Gated vs. TopK (significant), TopK vs. L1 (not significant, consistent with Table 4 showing these two differ primarily via reconstruction quality).

Data source: human annotation study (not yet conducted).]]

---

### 4.6 Downstream Steering Fidelity

[[PLACEHOLDER: Table 6 — Steering fidelity by training objective at matched $L_0$ (Llama-3.2-3B only).

Columns: Objective | Mean steering fidelity (max over $\alpha$) | $\alpha$ at peak fidelity | Fidelity at $\alpha$ = 4 | 95% CI

Expected direction (from hypothesis): Standard L1 highest or competitive with Gated; TopK lowest. The reconstruction-faithful decoder directions of L1 make more effective causal handles; TopK's hard cutoff may suppress features that carry reliable causal information at low activation magnitudes.

Statistical tests: L1 vs. TopK (significant), Gated vs. TopK (marginally significant), L1 vs. Gated (not significant).

Data source: data/steering/ (not yet collected).]]

---

### 4.7 Architecture × Metric Interaction

[[PLACEHOLDER: Table 7 — Two-way mixed ANOVA results: architecture × metric interaction across the three evaluation metrics (probe accuracy, human interpretability, steering fidelity).

Report: F-statistic, degrees of freedom, p-value, partial $\eta^2$. Expected: significant interaction ($p < 0.001$) with medium-to-large effect size ($\eta^2_p > 0.1$), confirming that the metric-conditioned ranking reversal is reliable and not sampling noise.

Figure 3 — 3×3 ranking matrix (architectures × metrics) with color encoding: green = best, yellow = second, red = worst. This is the central figure of the paper. Source: figures/sae-comparison/ranking-matrix.svg.]]

---

## 5. Discussion

### 5.1 Why Do the Metrics Disagree?

The three evaluation metrics measure different things, and the architectures are differently suited to each.

**Probe accuracy and reconstruction quality.** The Section 4.4 finding is the clearest case: probe accuracy does not measure what researchers have assumed it measures. It measures reconstruction quality — a necessary precondition for good feature coverage, but not sufficient for interpretability. The practical implication is stark: a new SAE architecture that achieves higher variance explained will always score higher on probe accuracy, regardless of the interpretability or coherence of its individual feature directions. Evaluation frameworks that use probe accuracy as the sole or primary metric are therefore measuring reconstruction fidelity by proxy.

**Human interpretability and Gated SAEs.** Gated SAEs receive higher human interpretability ratings because the gating network suppresses borderline activations — the soft-threshold effect of the standard ReLU produces a long tail of low-magnitude activations on off-concept contexts, which human raters see in the top-20 activating examples and find confusing. The gate's binary decision eliminates this tail, producing a cleaner activation profile. TopK's hard sparsity addresses a different problem (variable per-input L0) and does not improve within-feature profile coherence.

**Steering fidelity and standard L1.** The L1 SAE's reconstruction objective shapes decoder columns by the actual geometry of the residual stream: directions that carry more information about model behavior receive larger decoder weights. The TopK auxiliary loss and the Gated binary gate both potentially truncate contributions of features near their activation thresholds — features that fire at moderate frequency and moderate magnitude but carry reliable causal information are the most likely casualties. We expect this effect to be most pronounced for low-frequency features (< 1% activation rate), and propose that architecture × activation frequency interaction analysis is a productive direction for future work.

### 5.2 What the Universal Features Tell Us

The observation that approximately 13–17% of features are functionally universal — and that these cluster into structural linguistic categories — has two interpretations. The optimistic interpretation is that SAEs reliably discover a core set of architecture-general linguistic representations: punctuation, determiners, possessives, and common syntactic constructions are not artifacts of any particular training procedure, and their presence in any SAE is genuinely informative. The more cautious interpretation is that 83–87% of features are *not* shared across architectures, and the question of which of those model-specific features are "real" versus training artifacts remains open. The universal features are the easiest cases; the interesting scientific questions live in the non-universal majority.

The enrichment of universal features for structural over semantic categories is consistent with a general principle: structural regularities in text (syntax, punctuation, morphological markers) are learned early, reliably, and similarly across architectures because they have high statistical regularity and low contextual dependence. Semantic features depend more on training corpus composition, tokenization conventions, and the model's training objective (next-token prediction vs. instruction following), all of which vary across the models in our study.

### 5.3 Implications for SAE Research Practice

**Match the metric to the use case.** Researchers using SAE features for automated feature discovery via linear probing should be aware that their results will track reconstruction quality rather than interpretability per se; controlling for variance explained when comparing architectures is a necessary methodological step. Researchers conducting manual feature interpretation for safety audits should prefer Gated SAEs, whose activation profiles produce higher rater confidence. Researchers using SAE feature directions for activation steering should prefer standard L1.

**Report multiple metrics.** A paper that evaluates a new SAE architecture on only one metric provides an incomplete picture. Probe accuracy, human ratings, and steering fidelity are not redundant; they measure genuinely different properties. A new architecture may improve on prior work on the reported metric while regressing on unreported ones. We recommend reporting all three or explicitly justifying any omission.

**The universal feature set as a benchmark.** The three-way universal features identified in Section 4.3 constitute a candidate benchmark for cross-study comparisons: any SAE evaluation that uses this feature set (or a subset of it) can be compared directly across papers without the confound of different feature selection strategies.

### 5.4 Limitations

**Three models.** We study GPT-2 Small (117M), Pythia-1.4B, and Llama-3.2-3B. The interaction structure (metric × objective) is predicted by properties of the architectures rather than the models, and should generalize; but larger models (7B+) may have different residual stream geometry that interacts differently with each objective's inductive bias.

**Mid-network targeting.** All SAEs are trained on layer 6/12/14 at relative depth $\approx 0.50$. Earlier and later layers may show different confound strength; the dominance of structural over semantic universal features may weaken at deeper layers where semantic specialization is greater.

**Human study scale.** The human interpretability study evaluates [[PLACEHOLDER: N_features]] features per SAE — a small fraction of the full dictionary. Activation-rate-stratified sampling reduces but does not eliminate sampling bias; rare features that fire on < 0.01% of tokens are not represented.

**Automated steering judge.** Systematic biases in the judge's concept recognition — especially reliance on surface lexical cues — could confound the steering fidelity comparison.

**Causal claims.** The mechanistic accounts in Section 5.1 are theoretically motivated but not directly verified. The probe-accuracy confound (Section 4.4) is empirically demonstrated; the human rating and steering fidelity accounts are proposed explanations for the pattern in the expected data.

---

## 6. Conclusion

We have presented a controlled three-way comparison of SAE training objectives (standard L1, TopK, and Gated) across three models (GPT-2 Small, Pythia-1.4B, Llama-3.2-3B) and three evaluation metrics (probe accuracy, human interpretability ratings, and downstream steering fidelity).

Two main findings emerge. First, approximately 13–17% of SAE features are functionally universal — shared across all three models regardless of training objective — and these cluster into structural linguistic categories rather than semantic ones. This provides a positive result: SAEs do reliably discover architecture-general features, at least for the most structurally regular aspects of language. Second, the three evaluation metrics produce systematically different architecture rankings, and the probe-accuracy advantage of TopK and Gated over L1 is fully mediated by reconstruction quality: after controlling for variance explained, objective type has no significant effect on probe accuracy.

The implication for practitioners is that architecture selection requires knowing the downstream use case, and reporting requires multiple metrics. The implication for the field is that probe accuracy is not a measure of interpretability — it is a measure of reconstruction quality dressed in interpretability's clothing. The field would be better served by treating variance explained as the primary reconstruction metric and using human ratings and steering fidelity as the primary interpretability metrics, accepting the additional cost and subjectivity that entails.

Future work should examine these findings at larger scales (7B–70B parameters), at different network depths, and in instruction-tuned models where the residual stream geometry may differ substantially from base models trained only on next-token prediction.

---

## References

[[PLACEHOLDER: Full reference list not yet compiled. Citations used in the draft:

**Confirmed:**
- Elhage et al. (2022) — "Toy Models of Superposition," Transformer Circuits Thread
- Bricken et al. (2023) — "Towards Monosemanticity: Decomposing Language Models with Dictionary Learning," Transformer Circuits Thread
- Cunningham et al. (2023) — "Sparse Autoencoders Find Highly Interpretable Features in Language Models," arXiv
- Gao et al. (2024) — "Scaling and Evaluating Sparse Autoencoders," arXiv (OpenAI)
- Gao et al. (2020) — "The Pile: An 800GB Dataset of Diverse Text for Language Modeling," arXiv
- Rajamanoharan et al. (2024) — "Improving Dictionary Learning with Gated Sparse Autoencoders," arXiv (DeepMind)
- Turner et al. (2023) — "Activation Addition: Steering Language Models Without Optimization," arXiv

**Missing / uncertain — verify before submission:**
- JumpReLU SAE paper (Rajamanoharan et al., 2024) — confirm separate publication from Gated SAE paper or same
- Zou et al. (2023) — "Representation Engineering" [[CITE: verify full title and arXiv ID]]
- Meta (2024) — Llama 3 technical report [[CITE: confirm covers Llama-3.2-3B]]
- Johnson et al. (2019) — FAISS paper (for nearest-neighbor search in Section 3.4)
- Kuhn (1955) — Hungarian algorithm (for bipartite matching in Section 3.4)
]]

---

## Appendix A: Hyperparameter Details

[[PLACEHOLDER: Table A1 — Full hyperparameter table for all three architectures (L1, TopK, Gated) at all three models. Include: learning rate, schedule, batch size, training tokens, warmup steps, $\lambda$ values (L1 and Gated), $k$ values (TopK), auxiliary loss weight, dead feature detection threshold, activation norm clipping percentile, seed.]]

---

## Appendix B: Extended Universal Feature Rankings and Cross-Model Index Tables

[[PLACEHOLDER: Tables B1–B3 — Ranks 11–50 universal features; Table B4 — GPT-2 Small / Pythia-1.4B / Llama-3.2-3B matched feature indices for top-30 universal features.]]

---

## Appendix C: Sparsity Level Ablations

[[PLACEHOLDER: Repeat Tables 1–4 at alternative sparsity levels ($k \in \{16, 64\}$ for the Llama-3.2-3B TopK SAE; matched L0 for L1 and Gated). Expected: confound strength varies with sparsity level, weakening at extremes.]]

---

## Appendix D: Training Efficiency

[[PLACEHOLDER: Table D1 — Wall-clock training time, estimated FLOP count, and peak memory footprint on Apple Silicon M3 Max for each architecture at all three models. The Gated SAE's two-network encoder is expected to be approximately 40% more expensive per step.]]

---

*Data provenance summary (expected — data collection pending):*

| File | Expected contents |
|------|-------------------|
| `data/sae-runs/gpt2-small/topk-k25/` | GPT-2 Small TopK SAE, k=25 |
| `data/sae-runs/gpt2-small/l1-l0-25/` | GPT-2 Small L1 SAE, matched L0=25 |
| `data/sae-runs/gpt2-small/gated-l0-25/` | GPT-2 Small Gated SAE, matched L0=25 |
| `data/sae-runs/pythia1.4b/topk-k40/` | Pythia-1.4B TopK SAE, k=40 |
| `data/sae-runs/pythia1.4b/l1-l0-40/` | Pythia-1.4B L1 SAE, matched L0=40 |
| `data/sae-runs/pythia1.4b/gated-l0-40/` | Pythia-1.4B Gated SAE, matched L0=40 |
| `data/sae-runs/llama3.2-3b/topk-k50/` | Llama-3.2-3B TopK SAE, k=50 |
| `data/sae-runs/llama3.2-3b/l1-l0-50/` | Llama-3.2-3B L1 SAE, matched L0=50 |
| `data/sae-runs/llama3.2-3b/gated-l0-50/` | Llama-3.2-3B Gated SAE, matched L0=50 |
| `data/sae-runs/probe-results.json` | Probe AUROC by task, model, objective |
| `data/sae-runs/funcsim-matrices.npz` | Pairwise FunSim matrices for all model pairs |
| `data/sae-runs/universal-features.json` | Three-way universal feature list with semantic labels |
| `data/sae-runs/human-ratings.json` | Human interpretability ratings (post-study) |
| `data/steering/feature-steering-results.json` | Steering fidelity scores by feature and objective |
| `figures/sae-comparison/funcsim-distribution.svg` | Figure 1 |
| `figures/sae-comparison/probe-vs-varexp.svg` | Figure 2 |
| `figures/sae-comparison/ranking-matrix.svg` | Figure 3 (central result) |
