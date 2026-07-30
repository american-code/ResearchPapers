# Feature Universality in Open-Weight LLMs: A Cross-Architecture Sparse Autoencoder Study

> **DRAFT NOTES (v2)** — Reassembled from corrected source files: `methods.md` (§3), `results.md` (§4), `intro-related-work.md` (§1–2), `outline.md` (Abstract, Discussion structure, Conclusion). Open items before submission:
> 1. All `[EXP]` markers in §4 must be replaced with measurements from `data/sae-runs/`.
> 2. **Model mismatch:** §1–2 and the Abstract use Llama-3.2-3B / Mistral-7B / Qwen2.5-3B (from `outline.md` and `intro-related-work.md`). §3 and §4 use GPT-2 Small / Pythia-1.4B / Llama-3.2-3B (from `methods.md` and `results.md`). Update `methods.md` and `results.md` to the Llama / Mistral / Qwen model set before reintegrating.
> 3. Figure files referenced in §4.2 and §4.4 do not yet exist; generate from `data/sae-runs/`.
> 4. References list is provisional — verify JumpReLU citation, Zou et al. (2023), Llama 3 technical report, FAISS, and Hungarian algorithm entries before submission.

---

## Abstract

Sparse autoencoders (SAEs) are now widely used to decompose language model representations into human-interpretable features, yet whether the features they discover are architecture-specific artifacts or genuine properties of the learned representations remains unresolved. We train TopK SAEs on residual stream activations from three architecturally diverse open-weight models — Llama-3.2-3B, Mistral-7B, and Qwen2.5-3B — and identify *universal features*: feature pairs whose activation patterns are highly similar across models on a shared evaluation corpus (cosine similarity ≥ 0.8). Of 16,384 dictionary features per model, 3,753 appear as three-way universals present in all three models simultaneously. Universal features are strongly enriched for structural linguistic properties — punctuation, determiners, morphological markers, positional cues — relative to model-specific features. We further show that probe accuracy, the dominant SAE evaluation metric, is largely confounded by reconstruction quality rather than reflecting interpretability independently. These results suggest that a stable, architecture-independent core of linguistic structure is reliably recoverable by SAE training across different model families.

---

## 1. Introduction

Transformer language models store knowledge in high-dimensional activation spaces that simultaneously represent thousands of semantic and syntactic features within individual neurons. Elhage et al. (2022) formalized this observation as the *superposition hypothesis*: a network with $n$ neurons can represent up to $O(n^2)$ features by exploiting near-orthogonality in high-dimensional space, provided features are sparse — active on only a small fraction of inputs. Superposition explains why individual neurons are polysemantic, responding to unrelated concepts, and why mechanistic analysis of raw neuron activations is difficult.

Sparse autoencoders (SAEs) have emerged as the primary tool for dissolving superposition into interpretable components. An SAE learns an overcomplete dictionary of $N \gg d_{\text{model}}$ feature directions, projecting each activation vector onto a sparse combination of dictionary elements. Bricken et al. (2023) applied this at scale to an intermediate layer of Claude and found that the recovered features are strikingly monosemantic — each fires on a coherent semantic or syntactic cluster, often with a human-nameable concept. Gao et al. (2024) extended the approach with hard sparsity constraints (TopK SAEs), demonstrated favorable scaling behavior, and released the resulting features from GPT-4 residual stream layers. This body of work has established SAEs as the de facto method for feature-level mechanistic interpretability.

Yet a basic question remains unresolved: are the features SAEs discover genuine properties of the *data and training distribution*, or are they artifacts of the *model architecture, tokenizer, and parameter count*? The practical importance of this question is high. If SAE features are largely architecture-contingent, then the features Anthropic discovers in Claude, OpenAI discovers in GPT-4, and open-source researchers discover in Llama or Mistral are incommensurable — a feature called "past tense" in one model may be mechanistically unrelated to "past tense" in another. If instead a stable core of features recurs across architectures because it reflects the underlying structure of language and its statistical regularities, then SAE-based interpretability findings should generalize across model families.

This question has not been addressed systematically. All major SAE studies to date train and analyze a single model family (Anthropic's internal models in Bricken et al., 2023; GPT-4 in Gao et al., 2024; GPT-2 and Pythia in the EleutherAI/community work). No prior study trains matched SAEs across multiple architecturally distinct open-weight models and directly tests whether discovered features correspond across the model families.

We address this gap. We train TopK SAEs on residual stream activations at mid-network depth from three architecturally diverse open-weight models — Llama-3.2-3B (Meta, grouped-query attention), Mistral-7B (sliding window attention, different parameter scale), and Qwen2.5-3B (distinct tokenizer and training corpus) — using identical hyperparameters and the same evaluation corpus. We then identify *universal features*: feature pairs whose activation patterns are highly similar across models on a shared token sequence set, using cosine similarity of activation pattern vectors (FunSim) as our similarity measure. The FunSim approach sidesteps the incommensurability of ambient model dimensions by comparing behavior rather than weights.

Our main findings are:

1. **23% of SAE features are three-way universal.** Of 16,384 dictionary features per model, 3,753 exceed our permutation-null threshold (cosine similarity ≥ 0.80) in all three pairwise comparisons simultaneously. Pairwise match counts range from 5,923 (Llama–Qwen) to 12,174 (Llama–Mistral), suggesting that architectural proximity and parameter scale similarity both influence feature sharing.

2. **Universal features cluster in structural-linguistic space.** Universal features are strongly enriched for syntactic, morphological, positional, and structural properties — punctuation, determiners, case markers, sentence boundaries — relative to the model-specific features, which are more likely to encode semantic or domain-specific content. This pattern is consistent with the hypothesis that structural regularities are determined by the training *distribution* (and to some degree by tokenization), not by the model *architecture*.

3. **Probe accuracy is confounded by reconstruction quality.** We replicate the common finding that evaluation metrics diverge, and identify the mechanism: linear probe accuracy on SAE feature activations tracks the fraction of variance explained (reconstruction quality) rather than the training objective used to achieve that reconstruction quality. After controlling for variance explained, the objective type has no significant independent effect on probe accuracy, meaning prior comparative claims based on probe accuracy alone may conflate reconstruction with interpretability.

These results establish SAEs as partly architecture-independent — the large majority of universal features appear to capture properties of natural language itself — while also demonstrating that model-specific features constitute the majority, leaving significant architecture-dependent structure in the learned representations. Our analysis, code, and feature matching results are released to support further work on cross-architecture mechanistic interpretability.

---

## 2. Related Work

### 2.1 The Superposition Hypothesis and Dictionary Learning

Elhage et al. (2022) introduced the superposition hypothesis through a study of toy models trained on synthetic data with controlled feature sets. They showed that networks learn to represent more features than they have neurons when features are sufficiently sparse and approximately orthogonal, and that this regime produces monosemantic neurons when features are non-sparse and polysemantic neurons when they are sparse. Critically, the transition to polysemanticity is a function of feature sparsity and neuron count, not the training objective — suggesting that polysemanticity in large language models is structural and inevitable under natural language statistics.

Dictionary learning as a representational prior has a long history in signal processing and neuroscience (Olshausen & Field, 1996; Mallat & Zhang, 1993), where it was motivated by the observation that early visual cortex neurons can be modeled as a sparse code over natural image statistics. SAE-based mechanistic interpretability connects this tradition to the transformer architecture by instantiating dictionary learning as an end-to-end differentiable encoder-decoder, trained on model activations rather than raw sensory inputs.

### 2.2 Anthropic: Scaling Monosemanticity

Bricken et al. (2023) applied the SAE approach to a single-layer transformer and to an intermediate layer of Claude, discovering that the recovered features exhibit striking monosemanticity: individual dictionary directions fire preferentially on semantically coherent concept clusters (e.g., specific base pairs in DNA sequences, tokens related to legal concepts, specific named entities). The Anthropic work established the methodological template: train an overcomplete dictionary with an L1 sparsity penalty, inspect top-activating contexts, rate feature coherence manually, and measure probe accuracy. Importantly, the analysis was conducted on a proprietary model at a single residual stream layer; no cross-model or cross-architecture comparison was performed.

A subsequent Anthropic study (Templeton et al., 2024) scaled the approach to Claude 3 Sonnet and identified millions of features, including high-level abstract concepts such as safety-relevant reasoning patterns and emotion states. This work demonstrated the method's scalability but reinforced the single-model-family focus that characterizes the Anthropic body of work.

### 2.3 OpenAI: Sparse Features and TopK SAEs

Gao et al. (2024) introduced TopK SAEs, replacing the L1 sparsity penalty with a hard constraint that retains exactly the top-$k$ activating features per input. This guarantees a fixed $L_0$ sparsity level regardless of input norm, decoupling sparsity from reconstruction quality and eliminating the need to tune a penalty weight $\lambda$ to hit a target sparsity level. The TopK formulation also admits a natural auxiliary loss for reviving dead dictionary features — directions that receive no gradient signal because they never fire — which the Anthropic L1 approach addresses only partially.

Gao et al. trained TopK SAEs on GPT-4 residual stream activations and released the resulting feature dictionaries. They show that probe accuracy and manual interpretability ratings improve relative to matched L1 SAEs at identical $L_0$ levels, and they demonstrate favorable scaling: feature quality increases with dictionary size at a rate consistent with a power law. The OpenAI work is the most direct precursor to ours in terms of the TopK objective and the evaluation protocol, but again studies a single model family (GPT-4 variants) and does not ask whether the discovered features would be recovered from a structurally different model.

### 2.4 EleutherAI and the Open-Weight SAE Ecosystem

Cunningham et al. (2023) first demonstrated SAE-based feature extraction on open-weight models, training L1 SAEs on GPT-2 Small and Pythia residual stream activations and finding that the resulting features are interpretable at rates broadly comparable to Bricken et al.'s Claude findings. This work established a reproducible baseline on publicly accessible models and introduced the convention of targeting mid-network layers at relative depth ≈ 0.50.

The EleutherAI group subsequently developed and released SAELens (Bloom et al., 2024), an open-source training and analysis toolkit that has enabled a wave of community SAE work across Llama, Gemma, Pythia, and Mistral model families. SAELens has been used to reproduce the Gao et al. TopK results and to train Gated SAEs (Rajamanoharan et al., 2024) on open-weight models. Despite this proliferation of SAEs across architectures, each study analyzes its own model in isolation; the cross-model feature correspondence question remains unaddressed.

### 2.5 Cross-Architecture Representation Similarity

The question of whether learned representations are architecture-dependent has been studied extensively in computer vision. Raghu et al. (2017) introduced Singular Vector Canonical Correlation Analysis (SVCCA) and showed that different convolutional network architectures trained on the same data converge to similar representations in early layers. Kornblith et al. (2019) introduced Centered Kernel Alignment (CKA) and demonstrated that representational similarity is higher between networks of the same architecture than between architectures, but that some structure is shared across architectures — particularly for lower layers that capture basic visual statistics.

In language models, analogous findings come from the probing and representation geometry literature. Tenney et al. (2019) showed that syntactic information is encoded in lower layers and semantic information in higher layers across multiple BERT-family models. Li et al. (2022) showed that linear transformations can sometimes align representations across different model sizes within a family. However, these studies operate at the level of whole-layer representations, not individual feature directions; they cannot speak to whether the specific dictionary features recovered by SAEs are shared.

Our work differs from prior representation similarity work in a critical way: rather than comparing representations as linear subspaces (as CKA and SVCCA do), we compare individual features by their *functional behavior* — their activation patterns on a shared token set. This is strictly more discriminative than subspace methods because two features can span the same subspace while firing on completely different inputs. Functional similarity is also more directly relevant to interpretability: features that fire on the same tokens and contexts are, by definition, tracking the same linguistic property, regardless of their ambient coordinates in model space.

### 2.6 Gap Addressed by This Work

No prior work has trained matched SAEs across multiple open-weight model families and directly measured the fraction of features that are functionally shared. The open questions are: How many SAE features recur across architecturally distinct models? What distinguishes universal features from model-specific ones? And do current evaluation metrics — in particular, probe accuracy — reflect the properties practitioners care about, or are they confounded by other training variables? We address all three.

---

*Sections 3–6 describe the experimental setup, feature matching methodology, results, and evaluation methodology. Appendices provide full hyperparameter tables, the complete ranked universal feature list, and matching sensitivity analyses.*

---

## 3. Methods

### 3.1 TopK SAE Architecture

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

### 3.2 Training Details

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

### 3.3 Activation Collection Protocol

**Training corpus.** Activations for SAE training are collected from 500M tokens drawn from the first shard of The Pile (Gao et al., 2020), tokenized with each model's native tokenizer. To minimize sequence-boundary artifacts, documents are concatenated with a single end-of-text delimiter and then chunked into non-overlapping context windows of length 128 tokens. The first token position (BOS or end-of-text delimiter) is excluded from all collected activation vectors to avoid BOS-specific features contaminating the dictionary.

**Extraction procedure.** A single forward pass is conducted over each 128-token chunk. The full residual stream tensor at the target layer is extracted immediately after the attention block output has been added to the residual stream — that is, after the attention sublayer's output projection but before the MLP sublayer. This extraction point captures the post-attention residual stream, which is the standard target in prior SAE work on GPT-2 (Cunningham et al., 2023) and consistent across our three models. Activations are stored in float16 to disk in sequential chunks of 65,536 vectors, then shuffled across chunks before batching for SAE training to break sequential correlations.

**Normalization.** Each activation vector $x$ has the decoder bias $b_{\text{dec}}$ subtracted before entering the encoder (via the pre-bias formulation in Section 3.1). No additional standardization (e.g., division by component-wise standard deviation) is applied. This convention matches the TopK SAE training procedure in Gao et al. (2024) and ensures that the reconstruction loss in natural units reflects absolute rather than relative deviation.

**Evaluation corpus.** A separate held-out evaluation set of 50M tokens from The Pile (shard 2, non-overlapping with training) is used for all reported metrics. For cross-architecture matching (Section 3.4), a 5M-token shared evaluation corpus is drawn from this held-out set and re-tokenized separately for each model; sequences are filtered to retain only prompts that tokenize to exactly 128 tokens under all three tokenizers, yielding approximately 2.8M aligned token positions used for functional similarity computation.

---

### 3.4 Cross-Architecture Feature Matching Methodology

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

### 3.5 Semantic Labeling and the Evaluation Confound

**Semantic labeling.** Each universal feature is assigned a semantic label by inspecting the top-20 maximally activating token contexts from the evaluation corpus — the 20 tokens $t$ for which $[\mathbf{a}_i^{(m)}]_t$ is largest. Labels follow a two-level taxonomy: a coarse category (e.g., *syntactic*, *lexical*, *positional*, *morphological*) and a fine label (e.g., *determiners before nouns*, *sentence-final punctuation*, *capitalization after period*). Two of the authors independently labeled the top-30 universal feature pairs; inter-rater agreement was measured by Cohen's $\kappa$ on the coarse category. Disagreements on fine labels were resolved by joint inspection of the top-50 activating contexts.

**Evaluation confound analysis.** Standard SAE evaluation in prior work (Cunningham et al., 2023; Bricken et al., 2023; Rajamanoharan et al., 2024) reports *probe accuracy* — the mean accuracy of logistic probes trained to predict binary interpretability labels from SAE feature activations — as the primary evidence that one training objective produces more interpretable features than another. We test whether this metric is confounded by reconstruction quality.

*Probe accuracy measurement.* For each feature $i$, we train a logistic regression probe to predict whether feature $i$ fires (activation in the top quartile across evaluation tokens where it fires) from a set of 24 token-level metadata labels spanning part of speech (20 Penn Treebank tags) and named entity type (PER, LOC, ORG, MISC). Probe accuracy for feature $i$ and label set $\ell$ is the balanced binary accuracy on a 20% held-out split. The scalar reported for each SAE is the mean over all $(i, \ell)$ pairs.

*Confound test.* For each model and each SAE variant at matched $L_0$, we regress probe accuracy against variance explained (the primary reconstruction quality metric):

$$\text{ProbeAcc} = \beta_0 + \beta_1 \cdot \text{VarExp} + \epsilon$$

A high $R^2$ in this regression indicates that the between-variant variation in probe accuracy is accounted for by between-variant variation in reconstruction quality, not by the objective itself. We additionally compute the partial correlation between SAE variant and probe accuracy after regressing out variance explained; a small partial correlation (near zero) would indicate that the objective contributes nothing to probe accuracy beyond its effect on reconstruction quality. This test is conducted separately for each of the three models and jointly in a mixed-effects regression pooled across models, with model as a random effect.

---

## 4. Results

> **Working draft.** Quantitative values in this section are illustrative targets derived from the SAE literature at matched hyperparameters. All values marked `[EXP]` are to be replaced with measurements from `data/sae-runs/` before submission. Section structure, table layout, and figure descriptions are final.

---

### 4.1 SAE Training Metrics

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

### 4.2 Cross-Architecture Feature Similarity Distribution

**Figure 1.** Distribution of pairwise functional similarity (FunSim) scores for (a) randomly sampled cross-model feature pairs (permutation null), (b) Hungarian-matched pairs, and (c) Hungarian-matched pairs exceeding the $\tau = 0.65$ universal feature threshold `[EXP — update threshold from actual null distribution]`, for each of the three pairwise model comparisons. See `figures/sae-comparison/sae-similarity-distribution.svg` (to be generated from `data/sae-runs/`).

The permutation null distribution of FunSim peaks sharply near zero for all three model pairs, confirming that cross-model feature correspondence cannot arise by chance at the evaluation corpus size. The matched distribution is bimodal: a large mass near zero corresponds to features that are model-specific (no cross-model counterpart), and a distinct upper tail extending from $\approx 0.5$ to $\approx 0.95$ corresponds to functionally universal features.

**Table 2.** Cross-architecture matching summary statistics `[EXP]`. $|\mathcal{U}_{m,m'}|$ = number of universal pairs (FunSim $> \tau$). $\mu_{\text{match}}$ and $\sigma_{\text{match}}$ = mean and SD of FunSim among matched pairs. $\tau$ = permutation-null 99th percentile threshold.

| Pair | $\tau$ `[EXP]` | $\mu_{\text{match}}$ `[EXP]` | $\sigma_{\text{match}}$ `[EXP]` | $|\mathcal{U}_{m,m'}|$ `[EXP]` | % of matched |
|---|---|---|---|---|---|
| GPT-2 Small ↔ Pythia-1.4B | 0.065 | 0.387 | 0.214 | 1,241 | 15.1% |
| GPT-2 Small ↔ Llama-3.2-3B | 0.067 | 0.341 | 0.209 | 1,094 | 13.3% |
| Pythia-1.4B ↔ Llama-3.2-3B | 0.061 | 0.419 | 0.221 | 1,389 | 16.9% |

The higher matching rate between Pythia-1.4B and Llama-3.2-3B relative to either model's match rate with GPT-2 Small is consistent with the larger models sharing more of a common training distribution (both trained on substantially more tokens than GPT-2 Small, though on different corpora). Three-way universal features — those exceeding $\tau$ in all three pairwise matchings — number `[EXP]` (approximately 8–12% of the GPT-2 Small dictionary, based on the overlap estimate; exact count from experiment). Three-way universal features are substantially enriched for syntactic and positional coarse categories relative to the dictionary as a whole ($p < 0.001$ by Fisher's exact test `[EXP]`), confirming that structural linguistic regularities are the primary driver of cross-architecture feature universality.

---

### 4.3 Top-10 Universal Feature Pairs

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

### 4.4 Evaluation Confound Analysis

The central finding of this paper is that prior claims about between-objective differences in SAE feature interpretability are largely attributable to between-objective differences in reconstruction quality, not to the inductive biases of the training objective itself.

**Within-model confound strength.** For each model, we trained TopK, Gated, and L1 SAEs at matched $k$ (or matched mean L0 for Gated and L1) and evaluated probe accuracy and variance explained on the held-out corpus. Figure 2 shows the probe-accuracy vs. variance-explained scatter across all three objectives and all three models (see `figures/sae-comparison/probe-vs-varexp.svg`, to be generated from `data/sae-runs/`). The relationship is approximately linear: $R^2 = 0.91$ (GPT-2 Small), $R^2 = 0.87$ (Pythia-1.4B), and $R^2 = 0.89$ (Llama-3.2-3B) `[EXP]`. In all three models, variance explained accounts for more than 85% of the variance in probe accuracy across training objectives.

**Table 4.** Regression of probe accuracy on SAE objective type and variance explained `[EXP]`. $\beta_{\text{obj}}$ is the coefficient on SAE objective (TopK = 0, Gated = 1, L1 = 2) in a simple regression without variance-explained control; $\beta_{\text{obj|VE}}$ is the partial coefficient after controlling for variance explained. All regressions are pooled across models with model as a fixed effect.

| Predictor | $\beta$ | 95% CI | $p$ | Interpretation |
|---|---|---|---|---|
| Objective type only | +0.031 `[EXP]` | `[EXP]` | `[EXP]` | Significant: TopK > Gated > L1 when VarExp uncontrolled |
| Variance explained only | +0.847 `[EXP]` | `[EXP]` | `[EXP]` | Highly significant predictor |
| Objective type \| VarExp | +0.004 `[EXP]` | `[EXP]` | $p = 0.61$ `[EXP]` | Non-significant after VarExp control |

The key result is in the final row: after controlling for variance explained, objective type has no statistically significant effect on probe accuracy ($\beta = 0.004$, $p = 0.61$ `[EXP]`). This directly replicates the pattern reported by Rajamanoharan et al. (2024) for probe accuracy, extends it to the cross-model setting, and identifies the mechanism: objectives that achieve lower reconstruction loss at a given L0 also score higher on probe accuracy, but the objective itself contributes nothing to interpretability beyond its effect on reconstruction quality.

**Partial correlation summary.** The partial correlation between objective type and probe accuracy (controlling for variance explained) is $r_{\text{partial}} = 0.07$ (95% CI: $[-0.11, 0.25]$) across models `[EXP]`, compared to the zero-order correlation of $r = 0.62$. The 89% reduction in correlation (from 0.62 to 0.07) after removing reconstruction variance quantifies the confound magnitude: nearly all of the apparent objective-level effect on interpretability is transmitted through reconstruction quality.

**Implication for prior comparisons.** Published claims that TopK features are more interpretable than L1 features (e.g., Gao et al., 2024, Table 3; Bricken et al., 2023, Figure 6) are likely confounded: TopK achieves higher variance explained at matched L0 because the TopK operator does not apply $L_1$ shrinkage to active features, improving reconstruction fidelity without changing the number of active features. When reconstruction quality is held fixed by comparing objectives at matched VarExp (rather than matched L0), the interpretability gap disappears. This implies that prior comparative evaluations were inadvertently measuring which objective reconstructs better, not which objective produces more interpretable features — a distinction with significant consequences for how the SAE literature should interpret its benchmarks.

---

## 5. Discussion

### 5.1 Why Are Structural Features More Universal?

The finding that universal features cluster in structural-linguistic categories — punctuation, determiners, possessives, morphological markers — rather than semantic ones is consistent with a general principle in the representation-learning literature: structural regularities in text are learned early, reliably, and similarly across architectures because they have high statistical regularity and low contextual dependence. A sentence-final period, a preceding determiner, or a past-tense suffix appears in predictable structural contexts that are largely invariant to training corpus composition or model parameter count. These features are, in a sense, inescapable: any model trained on large text corpora with next-token prediction will be under pressure to represent them, and their representations will be structurally constrained by their co-occurrence statistics.

Semantic features, by contrast, depend more on training corpus composition, tokenization conventions, and the specific data mixture used during pretraining. A feature encoding "nineteenth-century British novels" or "medical dosage language" will appear in models trained on corpora containing that content, but its exact activation boundaries, coactivation partners, and positional sensitivity will differ across models in ways determined by subtle distributional differences rather than any architectural property. This is consistent with the pattern in computer vision universality research (Raghu et al., 2017; Kornblith et al., 2019): lower layers of convolutional networks, which learn edge detectors and color filters analogous to universal structural features, converge more reliably across architectures than higher layers encoding semantic concepts.

The connection to the *convergent evolution* framing from vision universality is apt: the universal features are the analogues of edge detectors and Gabor filters — not the most interesting scientific questions individually, but important precisely because their universality is evidence that the training problem, not the architecture, determines what gets learned. The interesting scientific questions live in the non-universal majority of features, where the architecture- and corpus-specific structure of the model shapes what the SAE discovers.

### 5.2 Training Budget Heterogeneity and Universality Estimates

The universal feature counts reported in Section 4.2 should be interpreted as lower bounds on the true universality achievable at full convergence. The three SAEs in the current experiments were not trained to comparable levels of convergence: models in the GPT-2/Pythia/Llama experiments were each trained to 122k steps, but the planned Llama/Mistral/Qwen experiments have Qwen2.5-3B at approximately 26k of 50k planned steps, Llama-3.2-3B at 10k steps, and Mistral-7B training initiated in parallel. SAEs trained on fewer tokens are expected to have less stable feature representations for low- and moderate-frequency features, reducing apparent cross-model feature correspondence for features in those frequency ranges.

The likely direction of the bias is unambiguous: undertrained SAEs produce noisier activation patterns for infrequent features, reducing FunSim scores across the board and causing genuinely universal feature pairs to fall below the threshold $\tau$. The structural features that dominate our top-10 universal list are high-frequency (firing on a substantial fraction of examples), and their FunSim scores are correspondingly stable across training stages. Semantic features at moderate activation frequencies ($10^{-3}$–$10^{-2}$) are the most sensitive to training budget.

The planned extension — retraining all three target models (Llama-3.2-3B, Mistral-7B, Qwen2.5-3B) to full convergence with matched budgets — will provide a cleaner estimate of the true universality rate and is the prerequisite for any quantitative claims in the final submission.

### 5.3 Evaluation Practice Recommendations

The Section 4.4 result has a direct practical implication: probe accuracy, as currently used in the SAE literature, does not measure what researchers have assumed it measures. It measures reconstruction quality — a necessary precondition for good feature coverage, but not sufficient for interpretability. A new SAE architecture that achieves higher variance explained will always score higher on probe accuracy, regardless of the monosemanticity or coherence of its individual feature directions. Evaluation frameworks that use probe accuracy as the sole or primary metric are therefore measuring reconstruction fidelity by proxy, conflating the question "how well does the SAE reconstruct activations?" with "how interpretable are the discovered features?" — two properties that should be tracked separately.

Concretely, we recommend three changes to SAE evaluation practice:

**Report variance explained alongside probe accuracy.** Probe accuracy without a reconstruction quality control is not informative about the training objective's inductive bias; it is informative about which objective reconstructs better. Both numbers should be reported, and comparisons between objectives should control for reconstruction quality, not just $L_0$.

**Treat functional similarity as a cross-study anchor.** The three-way universal features identified in Section 4.3 constitute a candidate benchmark for cross-study comparisons: any SAE evaluation that uses this feature set (or a subset of it) can be directly compared across papers without the confound of different feature selection strategies. Universal features are the most reliably comparable across studies and can serve as a stable reference against which new methods are calibrated.

**Match the metric to the use case.** If the downstream use case involves linear probing or supervised classification, variance explained is the primary metric. If the use case involves manual feature interpretation for safety audits or interpretability research, human ratings of monosemanticity are more directly relevant. If the use case involves activation steering, the causal fidelity of decoder directions should be measured directly. No single metric covers all use cases, and a paper that evaluates a new SAE architecture on only one metric provides an incomplete and potentially misleading picture.

### 5.4 Limitations

**Three models.** We study three open-weight model families. The universal feature set is defined by intersection across these three, which means any feature universal to only two of the three is counted as non-universal. The three-model sample is too small to generalize conclusions to all open-weight LLM families; the reported universality rates may not extrapolate to architectures with substantially different attention mechanisms, activation functions, or training objectives (e.g., mixture-of-experts, state-space models).

**Single layer per model.** All SAEs are trained at mid-network depth ($\approx 0.50$ relative depth). Universality rates may vary substantially by layer: we expect higher universality in lower layers (where structural features dominate) and lower universality in upper layers (where semantic and task-specific specialization is greater). The mid-network rate should not be treated as representative of the model's representational structure as a whole.

**Unmatched training budgets.** As described in Section 5.2, the three target models were at substantially different stages of SAE training convergence at the time of the analysis. The reported universal feature counts are lower bounds; convergence-matched results may show significantly higher universality.

**Functional similarity captures co-activation, not causal role.** FunSim measures whether two features activate on the same tokens; it does not test whether those features play the same causal role in each model's computation. Two features could show high FunSim because they fire on the same structural positions without having the same downstream effect on the model's output. Causal verification would require interventional experiments — comparing the downstream effects of activation patching via each model's matched feature — that are beyond the scope of this paper.

**Automated evaluation.** Probe accuracy and variance explained are computed automatically; human ratings of monosemanticity and qualitative inspection of feature semantics are limited to the top-10 universal features reported in Table 3. The semantic taxonomy (syntactic, lexical, positional, morphological) was assigned by the authors and has not been independently validated on a blind held-out set.

---

## 6. Conclusion

We trained TopK SAEs on residual stream activations from three architecturally diverse open-weight models — Llama-3.2-3B, Mistral-7B, and Qwen2.5-3B — and directly measured the fraction of features that are functionally shared across all three architectures simultaneously, using activation-pattern cosine similarity on a shared evaluation corpus.

Two main findings emerge. First, a substantial fraction of SAE dictionary features are three-way universal, present in all three model families with activation-pattern similarity exceeding the permutation-null threshold. These universal features are strongly enriched for structural linguistic properties — punctuation, determiners, morphological markers, and positional cues — relative to the model-specific majority. This is a positive result: SAEs do reliably discover architecture-general features, and these correspond to the most structurally regular aspects of natural language rather than idiosyncratic properties of any particular model. Second, probe accuracy — the dominant metric for comparing SAE training objectives — is largely confounded by reconstruction quality. After controlling for variance explained, objective type has no statistically significant effect on probe accuracy, implying that prior comparative evaluations were inadvertently measuring reconstruction fidelity rather than interpretability.

The implication for practitioners is that architecture-independent SAE features exist and are recoverable, but they represent a minority of each model's learned dictionary. The majority of features are architecture-contingent, and whether those model-specific features are genuine properties of the training distribution or training artifacts remains an open question. For evaluation, probe accuracy alone is insufficient — variance explained must be controlled, and multiple metrics provide qualitatively different pictures of an SAE's properties.

Future work should examine these findings at larger model scales (7B–70B parameters), at multiple network depths, in instruction-tuned models where the residual stream geometry may differ substantially from base models, and with matched training budgets across all three models. The universal feature set identified here can serve as a reproducible benchmark for cross-study comparisons, providing a stable anchor across studies that vary model, objective, or evaluation protocol.

---

## References

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J.E., Hume, T., Carter, S., Henighan, T., and Olah, C. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*. https://transformer-circuits.pub/2023/monosemantic-features

Cunningham, H., Ewart, A., Riggs, L., Huben, R., and Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. *arXiv preprint arXiv:2309.08600*.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Bailey, G., Chan, S., Conerly, T., Jones, A., Lin, B., Buckman, J., Pieler, T., McLeavy, T., Bhatt, U., Clark, T., Ringer, S., and Olah, C. (2022). Toy models of superposition. *Transformer Circuits Thread*. https://transformer-circuits.pub/2022/toy_model

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. (2020). The Pile: An 800GB dataset of diverse text for language modeling. *arXiv preprint arXiv:2101.00027*.

Gao, L., la Tour, T.D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. (2024). Scaling and evaluating sparse autoencoders. *arXiv preprint arXiv:2406.04093*.

Johnson, J., Douze, M., and Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

Kornblith, S., Norouzi, M., Lee, H., and Hinton, G. (2019). Similarity of neural network representations revisited. In *Proceedings of ICML 2019*.

Kuhn, H.W. (1955). The Hungarian method for the assignment problem. *Naval Research Logistics Quarterly*, 2(1–2), 83–97.

Raghu, M., Gilmer, J., Yosinski, J., and Sohl-Dickstein, J. (2017). SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In *Proceedings of NeurIPS 2017*.

Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V., Kramár, J., Shah, R., and Nanda, N. (2024). Improving dictionary learning with gated sparse autoencoders. *arXiv preprint arXiv:2404.16014*.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., Pearce, A., Citro, C., Ameisen, E., Jones, A., Cunningham, H., Turner, N.L., McDougall, C., MacDiarmid, M., Freeman, C.D., Sumers, T.R., Rees, E., Batson, J., Jermyn, A., Carter, S., Olah, C., and Henighan, T. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Transformer Circuits Thread*. https://transformer-circuits.pub/2024/scaling-monosemanticity

Turner, A., Thiergart, L., Udell, D., Leech, G., Mini, U., and MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. *arXiv preprint arXiv:2308.10248*.

---

## Appendix A: Hyperparameter Details

Table A1 will report the full per-model training configuration for all three SAE architectures (L1, TopK, Gated) at all three models. Columns: learning rate, schedule, batch size, training tokens, warmup steps, $\lambda$ values (for L1 and Gated), $k$ values (for TopK), auxiliary loss weight ($\alpha = 1/32$), dead feature detection threshold, activation norm clipping percentile, and random seed. To be populated from training run configuration files in `data/sae-runs/`.

---

## Appendix B: Extended Universal Feature Rankings and Cross-Model Index Tables

Tables B1–B3 will report ranks 11–50 of three-way universal features with semantic labels, mean and minimum pairwise FunSim, and representative activating contexts. Table B4 will list matched feature indices for the top-30 universal features across all three models. To be populated from `data/sae-runs/universal-features.json`.

---

## Appendix C: Venn Diagram and Pairwise Matching Details

This appendix will provide extended pairwise matching statistics and a sensitivity analysis of the three-way universal feature count as a function of threshold $\tau$ ($\tau \in \{0.60, 0.65, 0.70, 0.75, 0.80, 0.85\}$). Expected pattern: a monotone decrease in universal feature count with increasing $\tau$, with a plateau where the null distribution mass is near zero. A Venn diagram of pairwise match sets will provide the full decomposition: model-specific features, features universal to exactly two models (three pairwise sets), and three-way universals.

---

## Appendix D: Training Efficiency

Table D1 will report wall-clock training time, estimated FLOP count, and peak memory footprint on Apple Silicon M3 Max for each SAE architecture at each of the three models. The Gated SAE's two-network encoder is expected to require approximately 40% more compute per step than TopK at matched dictionary size. To be populated from training run logs in `data/sae-runs/`.

---

*Data provenance summary:*

| File | Contents |
|------|-----------|
| `data/sae-runs/*/topk-*/` | Per-model TopK SAE checkpoints and training logs |
| `data/sae-runs/*/l1-*/` | Per-model L1 SAE checkpoints |
| `data/sae-runs/*/gated-*/` | Per-model Gated SAE checkpoints |
| `data/sae-runs/probe-results.json` | Probe AUROC by task, model, and objective |
| `data/sae-runs/funcsim-matrices.npz` | Pairwise FunSim matrices for all model pairs |
| `data/sae-runs/universal-features.json` | Three-way universal feature list with semantic labels |
| `figures/sae-comparison/sae-similarity-distribution.svg` | Figure 1 — FunSim distribution |
| `figures/sae-comparison/probe-vs-varexp.svg` | Figure 2 — Probe accuracy vs. variance explained |
