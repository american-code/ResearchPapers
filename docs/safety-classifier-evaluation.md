# Safety Classifier Evaluation: SAE Feature-Based Harm Detection

**Date:** 2026-07-29  
**Model:** Llama-3.2-3B (mlx-community/Llama-3.2-3B-bf16)  
**SAE checkpoint:** `sae-runs/llama-3b-layer14/checkpoint_step_010000.npz`  
**Weights:** real

---

## 1. Design and Methodology

### 1.1 Motivation

This experiment tests whether a sparse autoencoder (SAE) trained on Llama-3.2-3B residual activations can serve as the backbone for a lightweight, interpretable safety classifier. The appeal of the approach is mechanistic: rather than training a separate classifier head on top of dense activations, we identify *which latent features* a model activates when processing harmful content, then use those features directly as a detection signal. This provides a natural explanation of *why* a prompt is flagged.

The classifier sits at three stages: (1) feature labeling — categorizing SAE features by semantic role, (2) harm scoring — aggregating activations of "potentially-harmful" features into a scalar signal, and (3) threshold calibration — sweeping thresholds across a balanced eval set to characterize the precision/recall trade-off.

### 1.2 SAE Configuration

The SAE is a TopK sparse autoencoder (k=128) with dictionary size 16,384, trained for 10,000 steps on approximately 500k tokens drawn from WikiText-103. It encodes layer-14 residual stream activations from Llama-3.2-3B (d_model=3,072), placing it at roughly the 50% depth point of the 28-layer model. The encoder applies:

```
pre_act = (x - b_dec) @ W_enc + b_enc
features = ReLU(pre_act)
```

with TopK sparsity enforced during training.

### 1.3 Feature Labeling

**Corpus:** WikiText-103 training split (~103M tokens), subsampled to 500k tokens.

**Selection:** All 16,384 SAE features are ranked by activation frequency over the corpus. The top-200 features by frequency are selected for labeling, on the hypothesis that high-frequency features capture the model's primary representational vocabulary.

**Pass 1** counts how many tokens each feature fires on (firing defined as being in the top-k activations for that token). **Pass 2** collects the top-10 maximum-activating token positions per feature and replays the WikiText corpus to decode 25-token context windows around each position.

**Heuristic labeling** applies weighted regex patterns across eight categories: `factual`, `emotional-positive`, `emotional-negative`, `potentially-harmful`, `refusal-related`, `stylistic`, `code-related`, and `other`. Each context window is scored by counting pattern matches weighted by specificity. The winning category must exceed 40% of the total signal; otherwise the feature falls back to `factual` (appropriate given the Wikipedia source corpus). A confidence score is the winning category's share of total weighted matches.

**Label distribution (top-200 features):**

| Label | Count | % |
|---|---|---|
| factual | 173 | 86.5% |
| stylistic | 24 | 12.0% |
| code-related | 2 | 1.0% |
| **potentially-harmful** | **1** | **0.5%** |
| other categories | 0 | 0.0% |

Non-other coverage: 100%.

### 1.4 Eval Dataset

A balanced dataset of 200 prompt–response pairs: 100 safe, 100 unsafe. Safe examples span five categories (factual Q&A, coding, writing, math, general knowledge). Unsafe examples span eight harm categories:

| Unsafe Category | N |
|---|---|
| violence | 15 |
| weapons | 15 |
| hate_speech | 15 |
| cybercrime | 10 |
| fraud | 10 |
| privacy_violation | 10 |
| dangerous_misinformation | 10 |
| sexual_exploitation | 10 |
| self_harm | 5 |

All examples include a realistic model response: safe examples have helpful factual answers; unsafe examples have explicitly harmful responses (for the purpose of measuring whether the *response* text elevates the harm score).

### 1.5 Harm Scoring

For each example, the full prompt+response is tokenized and truncated to 512 tokens. A partial forward pass through Llama-3.2-3B terminates at layer 14, yielding residual activations of shape (T, 3072). These are encoded through the SAE encoder, producing pre-activation values of shape (T, 16384). For each "potentially-harmful" feature, we take the max activation over all token positions. The harm score is the sum of these per-feature maxima:

```
harm_score = Σ_{f ∈ harmful_features} max_t( ReLU(pre_act[t, f]) )
```

### 1.6 Threshold Sweep

Thresholds are swept from 0.10 to 2.00 in steps of 0.05 (39 points). At each threshold, predictions are `unsafe` if `harm_score > threshold`, `safe` otherwise. Precision, recall, F1, and false-positive rate (FPR) are computed against ground-truth labels.

---

## 2. Results

### 2.1 Score Distributions

| Statistic | Value |
|---|---|
| Overall min | 0.168 |
| Overall max | 1.987 |
| Overall mean | 0.842 |
| Overall std | 0.333 |
| **Unsafe mean** | **0.882** |
| **Safe mean** | **0.802** |

The safe and unsafe distributions are nearly identical. The gap between means (0.08) is a small fraction of the standard deviation (0.33), making reliable discrimination impossible.

### 2.2 Threshold Sweep (Selected Points)

| Threshold | Precision | Recall | F1 | FPR |
|---|---|---|---|---|
| 0.10 | 0.500 | 1.000 | 0.667 | 1.000 |
| 0.50 | 0.476 | 0.800 | 0.597 | 0.880 |
| 0.70 | 0.488 | 0.630 | 0.550 | 0.660 |
| 0.90 | 0.580 | 0.470 | 0.519 | 0.340 |
| 1.10 | 0.732 | 0.300 | 0.426 | 0.110 |
| 1.30 | 1.000 | 0.180 | 0.305 | 0.000 |
| 1.50 | 1.000 | 0.030 | 0.058 | 0.000 |

**Best threshold by F1:** 0.10, with F1=0.667. However, this operating point predicts *every* example as unsafe (TP=100, FP=100, TN=0, FN=0), making it equivalent to a trivial "always-unsafe" rule with no discrimination ability.

The highest-F1 non-degenerate operating point is approximately threshold=0.90 (F1=0.519, precision=0.580, recall=0.470, FPR=0.340). This is barely above the random-classifier baseline of F1=0.500 on a balanced dataset.

At threshold ≥ 1.30, precision reaches 1.00 (no false positives) but recall collapses to 18% or below, catching fewer than 1 in 5 unsafe examples.

### 2.3 Most Discriminative Features

The five features with the greatest mean-activation difference between unsafe and safe examples are:

| Feature ID | Label | Discriminability | Unsafe mean | Safe mean |
|---|---|---|---|---|
| 5528 | factual | +0.576 | 4.863 | 4.287 |
| 8092 | factual | +0.416 | 3.020 | 2.604 |
| 12776 | stylistic | +0.381 | 2.995 | 2.614 |
| 15429 | factual | +0.380 | 1.965 | 1.585 |
| 3394 | factual | +0.316 | 2.496 | 2.179 |

No "potentially-harmful" feature appears in the top discriminators. The discriminability differences are modest (0.3–0.6 units) relative to activation magnitudes (2–5 units), confirming that the safe/unsafe score separation is driven by general distributional drift across text types rather than harm-specific features.

### 2.4 The Single Harmful Feature

Feature 15040 (rank 143, activation frequency 9.9% over WikiText tokens) is the only feature assigned the "potentially-harmful" label. Its top max-activating contexts from WikiText are military history passages:

> *"…artillery, antitank, and air defence regiments of divisions provided specialised fire support…"*

> *"…Hook-Sickle spear and Wang Gui, the Yanyue Dao. All of them learn the skill of archery…"*

This feature responds to weapons-adjacent military encyclopedic text rather than to operationally harmful instructions. It fires broadly (9.9% of tokens) because military history is well-represented in Wikipedia. As a result, it produces elevated scores on both safe factual prompts (e.g., history questions) and unsafe prompts.

---

## 3. Failure Mode Analysis

### 3.1 Corpus Mismatch in Feature Labeling

The fundamental problem is that the labeling corpus (WikiText-103) contains almost no explicitly harmful content. The top-200 features by activation frequency represent the model's most commonly-used representational vocabulary for encyclopedic prose — factual, stylistic, and occasionally code-related. Harmful concepts, by contrast, are rare in Wikipedia and therefore produce rare, low-frequency SAE features that fall outside the top-200 selection window.

This is a corpus selection error: the labeling corpus should be aligned with the target distribution (harmful vs. safe prompts and responses), not with the SAE training corpus.

### 3.2 Frequency Bias in Feature Selection

Selecting features by activation frequency biases selection toward general-purpose semantic features that fire on everything. High-frequency features are not necessarily the most *discriminative* features for any downstream task — they are simply the most common. Safety-relevant features are likely low-frequency by design: they activate specifically on harmful content, which is rare in the training corpus. These features would rank in the bottom quartile of activation frequency and are invisible to the current selection strategy.

A better strategy is to select features by *differential frequency*: features that fire more on harmful examples than on safe examples in a safety-relevant corpus.

### 3.3 Single-Feature Signal Collapse

With only one harmful feature, the harm score is a scalar drawn from a single noisy channel. The token-max aggregation discards all information about activation patterns across positions, and the sum over features reduces to a single value. Any single SAE feature captures only one facet of a concept; harmful requests invoke complex semantic patterns that should require multiple features in combination.

### 3.4 Layer Choice and Representational Maturity

Layer 14 sits at the midpoint of Llama-3.2-3B's 28-layer stack. Research on residual stream geometry (e.g., Logit Lens, tuned lens analyses) suggests that early-to-mid layers represent syntactic and surface-level features, while later layers (layers 20–28) increasingly represent semantic content relevant to the model's output distribution — including intent and topic. Safety-relevant representations may not be well-separated at layer 14.

### 3.5 False Positive Pattern

At the threshold=0.90 operating point, the false positives (safe examples predicted as unsafe) are predominantly from the `factual_qa` category with harm scores of 0.47–1.01. These examples discuss topics with surface-level lexical overlap with feature 15040's military-weapons vocabulary (e.g., questions about historical conflicts, chemistry, or physics). The classifier has no way to distinguish a factual educational question from an operationally harmful one when operating on a single broadly-firing feature.

### 3.6 No False Negatives at Low Threshold

At the degenerate threshold=0.10, all examples receive harm_score > 0.10 — including all safe examples. This happens because feature 15040 fires on nearly 10% of all tokens; most text passages contain at least one token that activates it above 0.10.

---

## 4. Implications for the Research Program

These results are informative rather than merely negative. They establish a concrete diagnosis of what the current SAE training and feature-labeling approach cannot do out of the box, and motivate specific architectural changes.

**For the SAE paper:** The experiment demonstrates that a classifier built on top of an SAE trained on a neutral corpus is not safety-capable without explicit re-targeting. This motivates a proposed extension: train a small auxiliary SAE (or fine-tune the encoder head) on a mixed corpus of harmful and safe conversations to surface harm-relevant features explicitly. The current pipeline infrastructure (two-pass frequency analysis, context replay, threshold sweep) is reusable once the corpus is replaced.

**For feature labeling:** Replace WikiText with a safety-relevant corpus — e.g., AdvBench, HarmBench, or a curated red-team dataset paired with safe examples. Alternatively, use contrastive activation patching: run paired safe/unsafe inputs through the model and select features that show the largest mean-activation difference as candidates for harm labeling.

**For threshold calibration:** The threshold sweep infrastructure is correct and reusable. Once feature labeling improves to yield O(10–50) harm-relevant features rather than 1, the score distribution should separate enough to find a useful operating point. Target: F1 ≥ 0.80 at FPR ≤ 0.10.

**For the paper section on safety applications:** This experiment can be framed as a *baseline* that motivates more sophisticated approaches. The honest result — a well-executed pipeline that fails due to corpus mismatch and feature scarcity — is scientifically more valuable than a cherry-picked positive result, because it exposes the conditions under which SAE-based classifiers will and will not work.

---

## 5. Summary

| Metric | Value |
|---|---|
| Model | Llama-3.2-3B, layer 14 |
| SAE dict size | 16,384 (TopK k=128) |
| Features labeled | 200 (top by frequency) |
| Harmful features identified | **1** |
| Eval set size | 200 (100 safe / 100 unsafe) |
| Best operating point F1 | 0.667 (degenerate: predicts all unsafe) |
| Best non-degenerate F1 | ~0.519 at threshold 0.90 |
| Random baseline F1 | 0.500 (balanced dataset) |
| Conclusion | **Classifier fails to separate safe/unsafe** |

The primary bottleneck is feature label scarcity caused by corpus mismatch. The pipeline architecture is sound; the labeling corpus must be replaced with safety-relevant data before re-evaluation.
