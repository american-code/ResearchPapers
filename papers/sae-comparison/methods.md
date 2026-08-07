# 3. Methods

---

## 3.1 TopK SAE Architecture

A sparse autoencoder (SAE) is a one-hidden-layer autoencoder trained to reconstruct a model's internal activations using a sparse linear combination of a fixed dictionary of learned directions. Given an activation vector $x \in \mathbb{R}^d$ extracted from a model's residual stream at a target layer, the SAE encodes it as a sparse feature vector and decodes that vector back to reconstruction $\hat{x}$:

$$z = \text{TopK}\!\left(W_{\text{enc}}\,(x - b_{\text{dec}}) + b_{\text{enc}}\right)$$
$$\hat{x} = W_{\text{dec}}\,z + b_{\text{dec}}$$

where $W_{\text{enc}} \in \mathbb{R}^{N \times d}$ is the encoder weight matrix, $W_{\text{dec}} \in \mathbb{R}^{d \times N}$ is the decoder weight matrix, $b_{\text{enc}} \in \mathbb{R}^N$ and $b_{\text{dec}} \in \mathbb{R}^d$ are learned biases, and $N$ is the dictionary size (number of features). The $\text{TopK}$ operator retains the $k$ largest non-negative pre-activations and zeros all others, enforcing exact sparsity: the support $|\text{supp}(z)| = k$ for every input $x$, so $L_0 = k$ exactly. The decoder columns (dictionary directions) are constrained to unit norm throughout training: $\|W_{\text{dec}}^{(:,i)}\|_2 = 1$ for all $i$, implemented via per-step re-normalization after each gradient update.

**Training objective.** The sole training loss is mean squared reconstruction error over the batch:

$$L = \mathbb{E}\!\left[\|x - \hat{x}\|_2^2\right]$$

Because TopK enforces hard sparsity, no explicit $L_0$ or $L_1$ regularization term is needed or added; the sparsity level is a fixed hyperparameter $k$. No auxiliary dead-feature revival loss is used. Features that are never selected by the TopK operator receive no encoder gradient and can go permanently unactivated — a known failure mode of TopK training without auxiliary losses (Gao et al., 2024). The high dead-feature rates observed in two of the three runs (Section 4.4) reflect this absence.

All experiments in this paper use the TopK architecture exclusively. We do not compare against L1-penalized or Gated SAE variants; the focus is on cross-architecture feature universality under a single controlled training objective.

---

## 3.2 Training Details

SAEs are trained on the residual stream activations of three models, each at their 50% depth layer (relative depth $= 0.50$). All three SAEs share the same architecture hyperparameters: dictionary size $N = 16{,}384$ and TopK parameter $k = 128$. The expansion factors differ because the models have different residual stream dimensions.

**Per-model configuration:**

| Model | Weights | Target layer | $d_{\text{model}}$ | Dict. size $N$ | Expansion | $k$ (L0) | Training steps | Training tokens |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | bf16 | 14 of 28 | 3,072 | 16,384 | 5.33× | 128 | 10,000 of 50,000 | 500,000 |
| Mistral-7B | 4-bit | 16 of 32 | 4,096 | 16,384 | 4.00× | 128 | 1,000 of 50,000 | 50,000 |
| Qwen2.5-3B | bf16 | 18 of 36 | 2,048 | 16,384 | 8.00× | 128 | 50,000 of 50,000 | 500,000 |

**Training status.** The Qwen2.5-3B SAE completed its full 50,000-step training run. The Llama-3.2-3B SAE was stopped at 10,000 steps (20% of target) for the cross-architecture analysis reported here; training is ongoing. The Mistral-7B SAE ran for only 1,000 steps on a reduced 50,000-token corpus subset, using 4-bit quantized model weights (mlx-community/Mistral-7B-v0.3-4bit), because a bf16 checkpoint was not available without authenticated Hugging Face access. Both the Llama and Mistral checkpoints are substantially partial; their effect on the reported results is discussed in Section 4.4.

**Optimization.** All SAEs are trained using Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) with a peak learning rate of $10^{-4}$ warmed up linearly over the first 500 steps (100 steps for Mistral) and decayed via cosine schedule to $5 \times 10^{-6}$ at training end. The batch size is 2,048 activation vectors. All training runs are seeded with seed 42 (NumPy, MLX, and Python's `random` module). Training was performed on a single Apple M3 Max using the MLX framework with float16 activations.

**Reproducibility.** The checkpoints used for all analyses in this paper are: `checkpoint_step_010000.npz` (Llama-3.2-3B), `checkpoint_final.npz` (Mistral-7B, equivalent to `checkpoint_step_001000.npz`), and `checkpoint_final.npz` (Qwen2.5-3B).

---

## 3.3 Activation Collection Protocol

**Training corpus.** Activations for SAE training are collected from WikiText-103 (wikitext-103-raw-v1 train split; Merity et al., 2016), tokenized with each model's native tokenizer. Documents are concatenated and chunked into non-overlapping context windows of 512 tokens. Llama-3.2-3B and Qwen2.5-3B activations were collected from 500,000 tokens each; Mistral-7B activations were collected from 50,000 tokens, reflecting the slower throughput of 4-bit quantized inference.

**Extraction procedure.** A single forward pass is conducted over each 512-token chunk. The full residual stream tensor is extracted at the target layer — defined as the layer index at exactly 50% of total model depth (layer 14 of 28 for Llama-3.2-3B, 16 of 32 for Mistral-7B, 18 of 36 for Qwen2.5-3B) — immediately after the attention block output has been added to the residual stream (post-attention, pre-MLP). Activations are stored in float16.

**Normalization.** Each activation vector $x$ has the decoder bias $b_{\text{dec}}$ subtracted before entering the encoder (via the pre-bias formulation in Section 3.1). No additional standardization is applied.

**Evaluation corpus.** Cross-architecture feature matching (Section 3.4) uses the same WikiText-103 train split, evaluated over 50,000 tokens (100 non-overlapping chunks of 500 tokens each). This corpus is not held-out from the SAE training data for Llama and Qwen; a proper held-out evaluation would require a disjoint corpus split. This is a noted limitation of the current analysis (Section 4.4).

---

## 3.4 Cross-Architecture Feature Matching Methodology

Each SAE dictionary $\mathcal{D}^{(m)} = \{d_i^{(m)}\}_{i=1}^N$ for model $m$ lives in the model's ambient residual stream space $\mathbb{R}^{d_m}$. Because $d_m$ differs across models (3,072, 4,096, 2,048), direct geometric comparison of decoder directions requires an alignment step. We use two complementary methods.

**Method A: Chunk-averaged activation pattern cosine similarity (primary).** For each SAE and each 500-token chunk $c$ in the evaluation corpus, we record the mean feature activation vector $\bar{z}^{(m)}(c) \in \mathbb{R}^N$ — the average over all token positions in the chunk of the sparse encoder output. For each pair of models $(m, m')$, the similarity between feature $i$ from model $m$ and feature $j$ from model $m'$ is the cosine similarity of their chunk-averaged activation profiles across all 100 evaluation chunks:

$$\text{ChunkSim}(i^{(m)}, j^{(m')}) = \frac{\bar{\mathbf{a}}_i^{(m)} \cdot \bar{\mathbf{a}}_{j}^{(m')}}{\|\bar{\mathbf{a}}_i^{(m)}\|_2 \|\bar{\mathbf{a}}_{j}^{(m')}\|_2}$$

where $\bar{\mathbf{a}}_i^{(m)} \in \mathbb{R}^{100}$ stacks the chunk-averaged activations of feature $i$ across the 100 chunks. A feature pair $(i^{(m)}, j^{(m')})$ is declared a *match* if $\text{ChunkSim} \geq 0.8$ and $j$ is the nearest neighbor of $i$ in model $m'$ (or symmetrically). The threshold 0.8 was set a priori as a conservative criterion; no permutation null calibration was performed.

**Method B: Decoder weight cosine similarity with Procrustes alignment (supplementary, Llama↔Mistral only).** Because $d_{\text{Llama}} \neq d_{\text{Mistral}}$, direct geometric comparison of decoder directions requires first aligning the two ambient spaces. We fit a Procrustes rotation $R^* \in \mathbb{R}^{d_{\text{Mistral}} \times d_{\text{Llama}}}$ by minimizing $\|R\,W_{\text{dec}}^{\text{Llama}} - W_{\text{dec}}^{\text{Mistral}}\|_F$ over the 12,174 activation-pattern-matched pairs from Method A, using SVD. The decoder weight cosine similarity under the fitted alignment is then computed for each Llama feature against its nearest Mistral neighbor in the aligned space. This method is reported as a geometric consistency check for the Llama↔Mistral pair; it is not used for three-way universality classification.

**Three-way intersection.** A feature is classified as *three-way universal* if it is part of a pairwise match (Method A, $\text{ChunkSim} \geq 0.8$) in all three pairwise comparisons — Llama↔Qwen, Mistral↔Qwen, and Llama↔Mistral — and the three matched counterparts form a mutually consistent triple (each pair of the triple is also a direct mutual nearest neighbor match). Counts and Venn breakdowns are reported in Section 4.3.
