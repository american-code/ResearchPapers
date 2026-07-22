# Circuit Tracing Experiment Brief

## Overview

This brief defines the experimental protocol for replicating and extending the Indirect Object Identification (IOI) circuit analysis from Wang et al. (2022) using modern open-weight models. The central question is whether the mechanistic findings from GPT-2 Small generalize to larger, differently-trained architectures.

---

## 1. Testable Hypothesis

**H1 (Replication):** The IOI circuit in Llama-3.2-3B replicates the functional architecture identified in Wang et al. (2022), specifically:
- A set of S-inhibition heads that suppress the subject token in the residual stream.
- Name mover heads in later layers that copy the indirect object name to the logit output.
- Duplicate token heads and induction heads that flag the repeated subject token.

**H1a (Sufficiency):** The identified circuit is causally sufficient to reproduce IOI behavior — patching only circuit components into a corrupted forward pass recovers the model's correct prediction.

**H1b (Necessity):** The identified circuit is causally necessary — ablating only circuit components degrades IOI performance to near-chance, while ablating non-circuit components does not.

**H2 (Cross-model transfer):** The same functional circuit exists in Pythia-1.4B, though the specific head indices and layer positions may differ from both GPT-2 Small and Llama-3.2-3B.

---

## 2. Evaluation Metric

### Primary Metric: Logit Difference (LD)

The logit difference measures the model's preference for the indirect object (IO) token over the subject (S) token at the final token position:

```
LD = logit(IO) − logit(S)
```

- **Baseline LD** (clean forward pass): expected to be positive (IO > S).
- **Corrupted LD** (mean-ablated or patched forward pass): expected to drop toward zero or negative.
- **Recovered LD** (patching circuit components into corrupted run): the degree to which LD returns to baseline quantifies circuit sufficiency.

### Normalized Recovery Score

To facilitate cross-model comparison, normalize recovered LD:

```
Recovery = (LD_patched − LD_corrupted) / (LD_clean − LD_corrupted)
```

A score of 1.0 indicates full recovery; 0.0 indicates no recovery above the corrupted baseline.

### Secondary Metrics

- **Top-1 accuracy:** whether the IO token is ranked first at the prediction position.
- **Probability ratio:** `P(IO) / P(S)` to assess confidence independently of logit scale.

### Dataset

Use the IOI dataset from Wang et al. (2022): templated sentences of the form:

> *"When Mary and John went to the store, John gave a drink to ___"*

Generate N ≥ 500 prompts with balanced name pairs and sentence templates. Crucially, include an ABC (three-name) control condition to isolate pure copying from name-position heuristics.

---

## 3. Required Interventions

### 3.1 Activation Patching

Activation patching identifies which components causally mediate IOI behavior by replacing activations in a corrupted forward pass with their clean counterparts, one component at a time.

**Protocol:**

1. Run the **clean** forward pass on the IOI prompt; cache all intermediate activations.
2. Construct a **corrupted** prompt by swapping the IO and S names, or by replacing names with random tokens from a fixed distribution.
3. Run the corrupted forward pass, substituting in the cached clean activation for a single component (attention head output, MLP output, or residual stream position).
4. Measure LD on the corrupted-plus-patch run.
5. Repeat for every (layer, head, token position) triple.

**Output:** A heatmap of Recovery scores across all layers and heads, aggregated over the IOI dataset. High-recovery components constitute candidate circuit members.

**Granularity levels:**
- Coarse: full attention layer outputs and MLP outputs per layer.
- Fine: individual attention head outputs (per head, per token position).
- Residual stream: per-position, per-layer residual stream patch.

### 3.2 Mean Ablation

Mean ablation replaces a component's activation with its mean value computed over a reference distribution (corrupted prompts or a separate random-sentence corpus). This tests necessity without requiring a paired corrupted prompt.

**Protocol:**

1. Compute the **mean activation** for each attention head output and MLP output over N_ref ≥ 200 reference sentences.
2. For each component under test, run the clean forward pass but substitute the component's output with its precomputed mean.
3. Measure LD degradation relative to the clean baseline.
4. Report components where ablation causes LD to drop by more than a threshold (e.g., > 20% of clean LD).

**Reference distribution options:**
- Corrupted IOI prompts (name-swapped) — isolates IOI-specific computation.
- Random sentences drawn from the Pile — tests for general language modeling residuals.

### 3.3 Path Patching (Optional Extension)

For higher-resolution attribution, implement path patching (Goldowsky-Dill et al., 2023) to measure the causal contribution of specific information-flow paths (e.g., head A → residual → head B) rather than individual components in isolation.

---

## 4. Models

### 4.1 Primary Model — Llama-3.2-3B

| Property | Value |
|---|---|
| Architecture | LlamaForCausalLM |
| Parameters | 3.21B |
| Layers | 28 |
| Attention heads | 24 |
| KV heads | 8 (GQA) |
| Hidden dim | 3072 |
| Intermediate dim | 8192 |
| Tokenizer | BPE (128k vocab) |
| Training data | ~9T tokens (undisclosed mix) |
| HuggingFace ID | `meta-llama/Llama-3.2-3B` |

**Notes:** Llama-3.2-3B uses Grouped Query Attention (GQA), which reduces the number of distinct key-value projections per layer. Patching must account for the 8-head KV structure; the 24 query heads share KV across groups of 3. Circuit analysis should track query heads as the primary unit.

### 4.2 Comparison Model — Pythia-1.4B

| Property | Value |
|---|---|
| Architecture | GPTNeoXForCausalLM |
| Parameters | 1.41B |
| Layers | 24 |
| Attention heads | 16 |
| Hidden dim | 2048 |
| Intermediate dim | 8192 |
| Tokenizer | BPE (50k vocab) |
| Training data | The Pile (825 GB, fully documented) |
| HuggingFace ID | `EleutherAI/pythia-1.4b` |

**Notes:** Pythia uses standard multi-head attention (no GQA), making it more directly comparable to the GPT-2 Small architecture from Wang et al. (2022). Its fully documented training data (The Pile) allows reference distribution construction from known in-distribution sentences.

### 4.3 Model Selection Rationale

Llama-3.2-3B is chosen as the primary model because it is a widely deployed, frontier-adjacent architecture trained with modern techniques (RoPE, SwiGLU, GQA, RMSNorm). Replication here would establish that IOI circuits are architecture-general, not artifacts of the GPT-2 training regime.

Pythia-1.4B serves as a controlled comparison: its documented training data, standard attention, and publicly available training checkpoints enable ablation over training dynamics if needed (using the Pythia checkpoint suite at 0–143K steps).

---

## 5. Implementation Notes

- Use **TransformerLens** for both models. Confirm that `HookedTransformer.from_pretrained()` supports `meta-llama/Llama-3.2-3B` with the current release; fall back to manual hook registration on the HuggingFace model if needed.
- Patching experiments should be run on GPU with `torch.no_grad()` and `model.eval()`.
- Cache clean activations in fp32 to avoid precision artifacts during patch substitution.
- Report all results as mean ± standard error over the IOI dataset with bootstrap confidence intervals (N_bootstrap = 1000).

---

## 6. Expected Deliverables

1. Activation patching heatmaps for both models (layer × head × token position).
2. Mean ablation necessity scores for all candidate circuit components.
3. Identified circuit heads with functional labels (name mover, S-inhibition, duplicate token, induction) for both models.
4. Cross-model comparison table: circuit head positions in GPT-2 Small (Wang et al.) vs. Pythia-1.4B vs. Llama-3.2-3B.
5. Recovery curves: LD recovered as circuit components are added back incrementally (sufficiency test).

---

## References

- Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small. *arXiv:2211.00593*.
- Goldowsky-Dill, N., MacLeod, C., Sato, L., & Arora, A. (2023). Localizing Model Behavior with Path Patching. *arXiv:2304.05969*.
- Biderman, S., et al. (2023). Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling. *ICML 2023*.
- Elhage, N., et al. (2021). A Mathematical Framework for Transformer Circuits. *Transformer Circuits Thread*.
