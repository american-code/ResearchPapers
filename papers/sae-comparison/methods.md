# 3. Methods

---

## 3.1 TopK SAE Architecture

A sparse autoencoder (SAE) is a one-hidden-layer autoencoder trained to reconstruct a model's internal activations using a sparse linear combination of a fixed dictionary of learned directions. Given an activation vector $x \in \mathbb{R}^d$ extracted from a model's residual stream at a target layer, the SAE encodes it as a sparse feature vector and decodes that vector back to reconstruction $\hat{x}$:

$$z = \text{TopK}\!\left(W_{\text{enc}}\,(x - b_{\text{dec}}) + b_{\text{enc}}\right)$$
$$\hat{x} = W_{\text{dec}}\,z + b_{\text{dec}}$$

where $W_{\text{enc}} \in \mathbb{R}^{N \times d}$ is the encoder weight matrix, $W_{\text{dec}} \in \mathbb{R}^{d \times N}$ is the decoder weight matrix, $b_{\text{enc}} \in \mathbb{R}^N$ and $b_{\text{dec}} \in \mathbb{R}^d$ are learned biases, and $N$ is the dictionary size (number of features). The $\text{TopK}$ operator retains the $k$ largest non-negative pre-activations and zeros all others, enforcing exact sparsity: the support $|\text{supp}(z)| = k$ for every input $x$, so $L_0 = k$ exactly. The decoder columns (dictionary directions) are constrained to unit norm throughout training: $\|W_{\text{dec}}^{(:,i)}\|_2 = 1$ for all $i$, implemented via per-step re-normalization after each gradient update.

**Training objective.** The primary loss is mean squared reconstruction error over the batch:

$$L_{\text{recon}} = \mathbb{E}\!\left[\|x - \hat{x}\|_2^2\right]$$

Because TopK enforces hard sparsity, no explicit $L_0$ or $L_1$ regularization term is needed or added; the sparsity level is a fixed hyperparameter $k$. To prevent *dead features* — dictionary directions that are never selected by TopK and therefore receive no gradient signal — we additionally apply the auxiliary loss introduced by Gao et al. (2024):

$$L_{\text{aux}} = \left\|x - W_{\text{dec}}\,z_{\text{dead}}\right\|_2^2$$

where $z_{\text{dead}}$ is the activation vector computed from only the top-$k$ entries among features whose running activation frequency has fallen below $10^{-4}$ over the past $10^4$ training steps. The total training loss is $L = L_{\text{recon}} + \alpha \, L_{\text{aux}}$ with $\alpha = 1/32$ across all experiments. A feature is permanently retired from the auxiliary loss if its frequency exceeds $10^{-3}$ after resurrection, preventing the auxiliary loss from interfering with well-trained features.

**Comparison to alternative sparsity mechanisms.** We benchmark TopK against two architectures drawn from the prior literature:

- *L1-penalized SAE* (Cunningham et al., 2023; Bricken et al., 2023): $L = L_{\text{recon}} + \lambda \|z\|_1$. The $L_1$ coefficient $\lambda$ controls a soft trade-off between reconstruction and sparsity, but $L_0$ is not directly controlled and varies across inputs and training stages. Matching $L_0$ across architectures for a fair comparison requires post-hoc filtering on $\lambda$.

- *Gated SAE* (Rajamanoharan et al., 2024): A separate gate network $g = \mathbb{1}[W_{\text{gate}} x + b_{\text{gate}} > 0]$ and magnitude network $m = W_{\text{mag}} x + b_{\text{mag}}$ combine as $z = g \odot \text{ReLU}(m)$. This architecture disentangles feature detection (gate) from feature magnitude (magnitude network), producing soft variable-$L_0$ outputs with better gradient flow than L1. The gated design reduces shrinkage bias on active features but reintroduces variable L0.

The structural advantage of TopK for controlled evaluation is that reconstruction quality and sparsity level are fully decoupled: $k$ is a fixed design choice, and $L_{\text{recon}}$ measures reconstruction quality at exactly that sparsity without conflation. This property is central to the evaluation confound analysis in Section 3.5.

---

## 3.2 Training Details

SAEs are trained on the residual stream activations of three models at their respective mid-network layers (relative depth $\approx 0.50$), chosen to match the functional depth zone associated with the dominant IOI circuit heads in our companion circuit-tracing work and, more broadly, the depth regime where prior SAE work (Cunningham et al., 2023; Gao et al., 2024) finds the highest feature interpretability scores.

**Per-model configuration:**

| Model | Target layer | $d_{\text{model}}$ | Dict. size $N$ | Expansion | $k$ (L0) |
|---|---|---|---|---|---|
| GPT-2 Small | 6 of 12 | 768 | 8,192 | 10.7× | 25 |
| Pythia-1.4B | 12 of 24 | 2,048 | 32,768 | 16.0× | 40 |
| Llama-3.2-3B | 14 of 28 | 3,072 | 49,152 | 16.0× | 50 |

Dictionary sizes are set to yield a 16× expansion factor for Pythia and Llama (following Gao et al., 2024), and 10.7× for GPT-2 Small to align with the widely-studied 8k-feature SAE used in Cunningham et al. (2023). The TopK values $k \in \{25, 40, 50\}$ are chosen to match the L0 levels reported in each model's own existing SAE literature; this allows direct comparison of our TopK results to prior L1 results without retuning $\lambda$.

**Optimization.** All SAEs are trained using Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) with a learning rate of $2 \times 10^{-4}$ warmed up linearly over the first 1,000 steps and decayed via cosine schedule to $2 \times 10^{-5}$ at training end. The batch size is 4,096 activation vectors. All SAEs are trained for $1.22 \times 10^5$ gradient steps, corresponding to 500M activation tokens, on a single Apple M3 Max using the MLX framework with float16 weights and float32 gradient accumulation. Activation norms are clipped to the 99th percentile of the training distribution before entry into the encoder to prevent outlier norms from dominating the reconstruction loss.

**Reproducibility.** All training runs are seeded with seed 42 (NumPy, MLX, and Python's `random` module). Model checkpoints are saved every $10^4$ steps; the checkpoint at $1.22 \times 10^5$ steps is used for all reported evaluations.

---

## 3.3 Activation Collection Protocol

**Training corpus.** Activations for SAE training are collected from 500M tokens drawn from the first shard of The Pile (Gao et al., 2020), tokenized with each model's native tokenizer. To minimize sequence-boundary artifacts, documents are concatenated with a single end-of-text delimiter and then chunked into non-overlapping context windows of length 128 tokens. The first token position (BOS or end-of-text delimiter) is excluded from all collected activation vectors to avoid BOS-specific features contaminating the dictionary.

**Extraction procedure.** A single forward pass is conducted over each 128-token chunk. The full residual stream tensor at the target layer is extracted immediately after the attention block output has been added to the residual stream — that is, after the attention sublayer's output projection but before the MLP sublayer. This extraction point captures the post-attention residual stream, which is the standard target in prior SAE work on GPT-2 (Cunningham et al., 2023) and consistent across our three models. Activations are stored in float16 to disk in sequential chunks of 65,536 vectors, then shuffled across chunks before batching for SAE training to break sequential correlations.

**Normalization.** Each activation vector $x$ has the decoder bias $b_{\text{dec}}$ subtracted before entering the encoder (via the pre-bias formulation in Section 3.1). No additional standardization (e.g., division by component-wise standard deviation) is applied. This convention matches the TopK SAE training procedure in Gao et al. (2024) and ensures that the reconstruction loss in natural units reflects absolute rather than relative deviation.

**Evaluation corpus.** A separate held-out evaluation set of 50M tokens from The Pile (shard 2, non-overlapping with training) is used for all reported metrics. For cross-architecture matching (Section 3.4), a 5M-token shared evaluation corpus is drawn from this held-out set and re-tokenized separately for each model; sequences are filtered to retain only prompts that tokenize to exactly 128 tokens under all three tokenizers, yielding approximately 2.8M aligned token positions used for functional similarity computation.

---

## 3.4 Cross-Architecture Feature Matching Methodology

Each SAE dictionary $\mathcal{D}^{(m)} = \{d_i^{(m)}\}_{i=1}^N$ for model $m$ lives in the model's ambient residual stream space $\mathbb{R}^{d_m}$. Because $d_m$ differs across models (768, 2048, 3072), direct geometric comparison of decoder directions is not feasible without an alignment step. We instead match features by *functional similarity* — the similarity of their activation patterns over a shared evaluation corpus — which does not require a common ambient space.

**Step 1: Activation pattern extraction.** For each SAE and each token $t$ in the shared evaluation corpus, we record the full feature activation vector $z^{(m)}(t) \in \mathbb{R}^N$ (the sparse output of the encoder). From these, we form the *feature activation pattern* for feature $i$ in model $m$:

$$\mathbf{a}_i^{(m)} \in \mathbb{R}^{n_{\text{eval}}},\qquad [\mathbf{a}_i^{(m)}]_t = z_i^{(m)}(t)$$

where $n_{\text{eval}} \approx 2.8 \times 10^6$ is the number of aligned evaluation tokens. Because each token produces at most $k$ non-zero entries in $z^{(m)}(t)$, each activation pattern is sparse: $\|\mathbf{a}_i^{(m)}\|_0 / n_{\text{eval}} = k / N \leq 0.001$ for all configurations.

**Step 2: Pairwise functional similarity.** For each pair of features $(i, j)$ from two models $m$ and $m'$, the functional similarity is the cosine similarity of their activation patterns:

$$\text{FunSim}(i^{(m)}, j^{(m')}) = \frac{\mathbf{a}_i^{(m)} \cdot \mathbf{a}_{j}^{(m')}}{\|\mathbf{a}_i^{(m)}\|_2 \|\mathbf{a}_{j}^{(m')}\|_2}$$

This quantity equals 1.0 if the two features activate on exactly the same tokens with proportional magnitudes, and 0.0 if their support sets are disjoint. Because both activation patterns are non-negative (TopK outputs non-negative values), FunSim is non-negative by construction.

Computing the full $N \times N$ similarity matrix for pairs with dictionary size $N = 49{,}152$ is prohibitive. We therefore use a two-stage procedure: (i) compute an approximate top-500 nearest-neighbor list for each feature using LSH-based approximate cosine search (FAISS, Johnson et al., 2019), then (ii) compute exact FunSim for all candidate pairs. This reduces computation from $O(N^2)$ to $O(N \cdot k_{\text{NN}})$ with negligible loss in matching quality (verified on a 5% random subsample where the exact and approximate matchings agree on 97.8% of pairs).

**Step 3: Maximum-weight bipartite matching.** We apply the Hungarian algorithm (Kuhn, 1955) to find the bijective assignment between the two dictionaries that maximizes total FunSim. For dictionary sizes $N = 8{,}192$, the full $N \times N$ matching is tractable (< 2 seconds using the LAPJV implementation). For larger dictionaries ($N \in \{32{,}768; 49{,}152\}$), matching is restricted to the top-$5{,}000$ features by activation frequency in each SAE, discarding low-frequency features that may not have converged. All quantitative analyses use only the matched subset.

**Step 4: Permutation-null threshold.** To determine a significance threshold $\tau$ for declaring a matched pair a *universal feature*, we compute a permutation null distribution: the FunSim values of 10,000 randomly drawn cross-model feature pairs (not from the Hungarian matching). The 99th percentile of this distribution is used as the threshold $\tau$, yielding a per-pair type-I error rate of 1% under the null of no functional correspondence.

**Step 5: Three-way intersection.** For each of the three pairwise comparisons (GPT-2 Small ↔ Pythia-1.4B, GPT-2 Small ↔ Llama-3.2-3B, Pythia-1.4B ↔ Llama-3.2-3B), we identify the set of *pairwise universal features* — matched pairs exceeding $\tau$. A feature is classified as *three-way universal* if it is part of a pairwise-universal match in all three comparisons and all three cross-model assignments refer to the same semantic cluster (verified via the semantic label agreement procedure in Section 3.5).

---

## 3.5 Semantic Labeling and the Evaluation Confound

**Semantic labeling.** Each universal feature is assigned a semantic label by inspecting the top-20 maximally activating token contexts from the evaluation corpus — the 20 tokens $t$ for which $[\mathbf{a}_i^{(m)}]_t$ is largest. Labels follow a two-level taxonomy: a coarse category (e.g., *syntactic*, *lexical*, *positional*, *morphological*) and a fine label (e.g., *determiners before nouns*, *sentence-final punctuation*, *capitalization after period*). Two of the authors independently labeled the top-30 universal feature pairs; inter-rater agreement was measured by Cohen's $\kappa$ on the coarse category. Disagreements on fine labels were resolved by joint inspection of the top-50 activating contexts.

**Evaluation confound analysis.** Standard SAE evaluation in prior work (Cunningham et al., 2023; Bricken et al., 2023; Rajamanoharan et al., 2024) reports *probe accuracy* — the mean accuracy of logistic probes trained to predict binary interpretability labels from SAE feature activations — as the primary evidence that one training objective produces more interpretable features than another. We test whether this metric is confounded by reconstruction quality.

*Probe accuracy measurement.* For each feature $i$, we train a logistic regression probe to predict whether feature $i$ fires (activation in the top quartile across evaluation tokens where it fires) from a set of 24 token-level metadata labels spanning part of speech (20 Penn Treebank tags) and named entity type (PER, LOC, ORG, MISC). Probe accuracy for feature $i$ and label set $\ell$ is the balanced binary accuracy on a 20% held-out split. The scalar reported for each SAE is the mean over all $(i, \ell)$ pairs.

*Confound test.* For each model and each SAE variant at matched $L_0$, we regress probe accuracy against variance explained (the primary reconstruction quality metric):

$$\text{ProbeAcc} = \beta_0 + \beta_1 \cdot \text{VarExp} + \epsilon$$

A high $R^2$ in this regression indicates that the between-variant variation in probe accuracy is accounted for by between-variant variation in reconstruction quality, not by the objective itself. We additionally compute the partial correlation between SAE variant and probe accuracy after regressing out variance explained; a small partial correlation (near zero) would indicate that the objective contributes nothing to probe accuracy beyond its effect on reconstruction quality. This test is conducted separately for each of the three models and jointly in a mixed-effects regression pooled across models, with model as a random effect.
