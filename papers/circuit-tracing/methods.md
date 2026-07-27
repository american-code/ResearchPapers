# 3. Methods

---

## 3.1 Experimental Infrastructure: SwiftSci Interp

All experiments were conducted using SwiftSci Interp, an activation-patching harness for Apple Silicon built on the MLX framework. The library implements a hook-compatible interface analogous to TransformerLens's activation interception API, targeting MLX-native model implementations loaded via `mlx-lm`. Its core design principle is that every attention module in a model is hot-swapped at load time with a patchable drop-in replacement that shares the original weight tensors — no copies — and adds a mode-switched interception point around the scaled dot-product attention (SDPA) computation.

**Patchable module design.** For Llama-3.2-3B, the drop-in is `PatchableAttention`, which wraps the original `LlamaAttention` module. For Pythia-1.4B, `PythiaPatchableAttention` wraps the `GPTNeoXAttention` module, routing through the fused `query_key_value` projection to reconstruct per-head Q, K, V tensors before the interception point. Both modules support three operating modes:

- `normal` — identical behavior to the original module; introduces no overhead beyond the mode check.
- `cache_clean` / `cache_corrupt` — stores the full SDPA output tensor (shape `[B, n_heads, L, head_dim]`) to a per-module buffer during the forward pass. This is used to populate the clean and corrupt activation caches without a separate caching pass.
- `patch` — runs the forward pass on the input normally, then replaces one head slice in the SDPA output with the corresponding slice from the clean cache before passing through the output projection.

**GQA handling.** Llama-3.2-3B uses grouped-query attention (GQA) with 24 query heads and 8 KV heads. The patching interception occurs after the attention weights are applied to the (broadcast) value projections, at the query-head output level. This gives a 24-element patch space that mirrors the query-head count, preserving head-level attribution granularity. Patching at the KV-head level would conflate the contributions of the three query heads that share each KV pair; patching at the query-head SDPA output avoids this confound. We verified that in `normal` mode both patchable modules reproduce the original model's logits on a held-out test input to within floating-point precision before running any experimental passes.

**GQA key-value broadcast.** During the corrupt or clean pass, the values broadcast from 8 KV heads to 24 query heads occur inside `scaled_dot_product_attention`; the SDPA buffer therefore stores the full 24-head output, not the 8-head compressed form. This means each of the 24 patch positions is truly independent, and patching query-head $h$ does not implicitly alter the other two heads sharing the same KV pair.

**Pythia module layout.** MLX-LM exposes Pythia's layers as `model.layers[i].attention` rather than `model.model.layers[i].self_attn`. The patchable replacement reuses the `dense` (output projection) attribute rather than `o_proj`. All other behavior is identical; the mode-switch logic is shared.

---

## 3.2 Activation Patching Protocol

Activation patching (also called causal tracing when applied to individual stream positions) answers the question of *sufficiency*: if a component's activations in the clean run are inserted into the corresponding position of a corrupted run, does the model's output recover toward the clean prediction? A high recovery score indicates that the component is sufficient to carry the relevant information on its own.

**IOI corruption scheme.** For each example $(S, IO, \text{prompt})$, the corrupted version swaps the subject and indirect object names: the corrupted prompt is constructed from the same template with $S_\text{corrupt} = IO$ and $IO_\text{corrupt} = S$. Because the two name slots have the same token count for all examples in our dataset (a requirement enforced at dataset generation time), clean and corrupt sequences are token-position aligned. This ensures that cached SDPA tensors from the clean run can be substituted at the same token positions in the corrupt run without shape mismatch.

**Metric.** The target metric is logit difference (LD): $\text{LD} = \text{logit}(IO) - \text{logit}(S)$ at the final token position (the prediction position). Positive LD indicates the model favors the IO name over the subject name. The *normalized logit-difference recovery* for head $(l, h)$ on example $i$ is:

$$r_{l,h}^{(i)} = \frac{\text{LD}^{(i)}_{\text{patch}(l,h)} - \text{LD}^{(i)}_{\text{corrupt}}}{\text{LD}^{(i)}_{\text{clean}} - \text{LD}^{(i)}_{\text{corrupt}}}$$

where $\text{LD}^{(i)}_{\text{patch}(l,h)}$ is the logit difference of a forward pass on the corrupt input in which only head $h$ at layer $l$ has been replaced by its clean-run SDPA output. A score of 1.0 means full recovery to the clean logit difference; 0.0 means the patch had no effect; values below 0 or above 1 indicate interference effects. The reported score for each head is $\hat{r}_{l,h} = \frac{1}{n}\sum_{i=1}^n r_{l,h}^{(i)}$, the mean over all $n$ examples.

**Pass structure.** A full patching sweep requires $2 + n_L \times n_H$ forward passes over the example batch:

1. **Clean pass** — all modules in `cache_clean` mode; SDPA outputs are stored and logit differences are computed.
2. **Corrupt pass** — all modules in `cache_corrupt` mode; SDPA outputs are stored and baseline corrupt logit differences are computed.
3. **Sweep** — for each $(l, h)$: all modules in `normal` mode; module $l$ is set to `patch` mode with `patch_head = h`; one forward pass on the corrupt input recovers $\text{LD}^{(i)}_{\text{patch}(l,h)}$ for all $i$ simultaneously.

For Llama-3.2-3B this yields $2 + 28 \times 24 = 674$ total forward passes; for Pythia-1.4B, $2 + 24 \times 16 = 386$ passes. All passes are fully batched over all $n = 100$ examples.

**Pseudocode for the patching sweep:**

```python
# Inputs:
#   attns: list of PatchableAttention, one per layer (len = n_layers)
#   clean_ids:   [n_examples, seq_len]  — clean tokenized prompts
#   corrupt_ids: [n_examples, seq_len]  — IO/S-swapped prompts
#   n_layers, n_heads: model architecture constants

# ── Pass 1: cache clean activations ──────────────────────────────────────────
set_all_modes(attns, mode="cache_clean")
clean_logits = model(clean_ids)                     # forward on clean input
eval(clean_logits, *(a.clean_sdpa for a in attns))  # materialize before next pass
clean_ld = logit_diff(clean_logits, io_ids, s_ids)  # [n_examples]

# ── Pass 2: cache corrupt activations ────────────────────────────────────────
set_all_modes(attns, mode="cache_corrupt")
corrupt_logits = model(corrupt_ids)
eval(corrupt_logits, *(a.corrupt_sdpa for a in attns))
corrupt_ld = logit_diff(corrupt_logits, io_ids, s_ids)

# ── Pass 3..674: patching sweep ───────────────────────────────────────────────
scores = zeros([n_layers, n_heads])
for l in range(n_layers):
    for h in range(n_heads):
        set_all_modes(attns, mode="normal")         # all layers run normally
        attns[l].mode = "patch"
        attns[l].patch_head = h                     # only layer l patches head h
        patched_logits = model(corrupt_ids)         # corrupt input, one head replaced
        eval(patched_logits)
        patched_ld = logit_diff(patched_logits, io_ids, s_ids)
        # Normalized recovery, averaged over examples
        denom = clean_ld - corrupt_ld               # [n_examples]
        recovery = (patched_ld - corrupt_ld) / denom
        scores[l, h] = mean(recovery)               # scalar
```

**GQA note in pseudocode.** `attns[l].patch_head = h` indexes into the 24-dimensional query-head axis regardless of the 8-KV-head layout. The `clean_sdpa` buffer holds the full `[B, 24, L, head_dim]` output; the slice `clean_sdpa[:, h:h+1, :, :]` is substituted.

---

## 3.3 Mean Ablation Protocol

Mean ablation addresses *necessity*: does removing a component's contribution — replacing it with a neutral baseline — degrade the model's performance on the task? Where activation patching asks whether a component *can* carry the relevant information, mean ablation asks whether the model *relies* on it in normal operation.

**Reference distribution.** The ablation target for head $(l, h)$ is the mean SDPA output of that head computed over all $n = 100$ clean IOI examples: $\bar{A}_{l,h} = \frac{1}{n}\sum_{i=1}^n A_{l,h}^{(i)}$, where $A_{l,h}^{(i)} \in \mathbb{R}^{L \times d_{\text{head}}}$ is the per-example SDPA output at each token position. The mean is taken over the batch dimension, preserving the sequence-position structure. Ablating with the mean over the same distribution on which the model is evaluated (rather than, for example, a random Gaussian or a separately collected reference set) ensures that the ablated output is a plausible activational state for the model rather than an out-of-distribution injection.

**Metric.** The *ablation drop* for head $(l, h)$ is:

$$\Delta_{l,h} = \overline{\text{LD}}_{\text{clean}} - \overline{\text{LD}}_{\text{ablate}(l,h)}$$

where $\overline{\text{LD}}_{\text{clean}}$ is the mean clean logit difference over all examples and $\overline{\text{LD}}_{\text{ablate}(l,h)}$ is the mean logit difference when head $(l, h)$'s SDPA output is replaced by $\bar{A}_{l,h}$ in every example. Positive $\Delta_{l,h}$ indicates the head boosts the IO logit relative to the subject; negative values indicate suppression of correct IO prediction (which can reflect an interference-suppression role rather than direct IO promotion).

**Necessity threshold.** A head is designated circuit-critical under the necessity criterion when its ablation drop exceeds 20% of the clean mean logit difference. For Llama-3.2-3B ($\overline{\text{LD}}_{\text{clean}} = 5.649$) this corresponds to $\Delta_{l,h} > 1.130$; for Pythia-1.4B ($\overline{\text{LD}}_{\text{clean}} = 4.120$) the threshold is $\Delta_{l,h} > 0.824$. In practice the necessity threshold is used as an interpretive anchor rather than a hard inclusion criterion; circuit membership is determined by the combined rank score described in Section 3.5.

**Pass structure.** The ablation sweep requires $1 + n_L \times n_H$ forward passes: one clean pass to cache per-head means, then one ablation pass per $(l, h)$ pair. For Llama this is $1 + 672 = 673$ total passes, comparable to the patching sweep.

---

## 3.4 Path Patching for Causal Verification

Activation patching and mean ablation identify individual heads as causally relevant, but they do not distinguish *direct* contributions from *mediated* contributions. A head may score highly on both metrics because it directly writes to the logit output at the final position, or because it is an intermediary in a longer causal path — for example, writing to the residual stream at an intermediate token position that is later read by a downstream head that writes to the final position.

Path patching (Goldowsky-Dill et al., 2023) extends activation patching to directed paths between specific components. Rather than patching a single component's output, a path patch replaces the clean-run activation that flows along one specific edge in the computational graph — from the output of component $A$ as read by the input of component $B$, while holding all other paths fixed. A path that shows high normalized recovery under path patching is causally necessary specifically through the direct $A \to B$ connection, ruling out the mediated-path alternative.

In our setting, path patching is used to verify that the dominant heads identified by activation patching (L15H20 in Llama, L10H7 in Pythia) contribute directly to the final position logit difference, rather than acting through a downstream intermediary. For each dominant head $A$ and each later head $B$ in the circuit, we construct the $A \to B$ path patch and measure recovery. High recovery on the direct $A \to \text{output}$ path confirms direct contribution; high recovery on $A \to B \to \text{output}$ with low recovery on $A \to \text{output}$ alone would indicate mediation.

Path patching is implemented within SwiftSci Interp by extending the `patch` mode to accept a target-receiver argument: when `patch_target = (l_B, h_B)` is specified, the replacement is applied only to the projection of head $A$'s clean output onto the query, key, or value input of head $B$, leaving all other paths unchanged. We report path-patching results for the top-3 heads in each model in Section 4.3.

---

## 3.5 Statistical Validation

**Bootstrap confidence intervals.** For all heads meeting the inclusion criterion (top-10 by combined rank score, or mean patching score $\geq 0.03$), we compute 95% bootstrap confidence intervals on the mean patching score. The bootstrap resamples the $n = 100$ per-example recovery scores with replacement, generating 1,000 resampled datasets and computing the mean for each. The 95% CI is the $[2.5^{\text{th}}, 97.5^{\text{th}}]$ percentile of the bootstrap distribution of the mean. Resampling is seeded with seed 42 for reproducibility.

**Significance criterion.** A head's patching score is considered statistically reliable if its 95% CI excludes zero. This corresponds to the claim that the head's mean recovery score is positive in expectation across the example distribution, not merely a noise artifact of the particular 100 examples observed. All top-10 heads in both models satisfy this criterion with substantial margin: for the top three heads in each model, the CI lower bound exceeds the CI width by a factor of at least 3.

**Anomalous patching scores.** Pythia-1.4B's L1H11 presents a dissociation: its mean patching score is slightly negative ($-0.003$, 95% CI $[-0.011, +0.005]$, straddling zero), yet its ablation drop is substantial ($0.327$, rank 4 by ablation). This dissociation is consistent with a suppression role: the head reduces logit difference when ablated (its absence degrades performance), but restoring its clean activation in a corrupt context does not recover logit difference (it cannot by itself make the model prefer IO over S). This pattern is characteristic of duplicate-token or induction heads that gate downstream components rather than writing directly to the output logits. L1H11 is excluded from all patching-based analyses and retained in cross-model analysis solely on the basis of its ablation score.

**Cross-model comparison: combined rank score.** To compare head importance across models that differ in scale, architecture, and absolute logit-difference baselines, we compute a combined normalized rank score. For each model separately, both the patching scores and the ablation drop scores are min-max normalized to $[0, 1]$ across all $(l, h)$ pairs. The combined score is the sum of the two normalized ranks. The top-10 heads by combined score are reported as circuit-critical for each model. Ties within the combined score are broken by patching rank.

**Cross-architecture position matching.** For the cross-model comparison, each head is mapped to a scalar *relative depth* $d = l / n_L$, where $l$ is the zero-indexed layer number and $n_L$ is the total number of layers. Positions are matched greedily in ascending order of $|\Delta d|$ between the two models' top-10 head sets; a match is accepted if $|\Delta d| \leq 0.075$. This tolerance was chosen to be smaller than the spacing between the four functional depth zones identified by Wang et al. (2022) ($\approx 0.15$–$0.20$ between zone midpoints), ensuring that zone-level matches are not spuriously collapsed across zone boundaries.

---

## 3.6 Dataset

The IOI dataset consists of $n = 100$ examples drawn from the template:

> "When {S} and {IO} went to the store, {S} gave a bottle to"

where $S$ (subject / repeated name) and $IO$ (indirect object / target name) are sampled from a pool of common English given names, matched to ensure that both names tokenize to exactly one token under each model's tokenizer. All 100 examples satisfy the single-token constraint for both Llama-3.2-3B and Pythia-1.4B; examples failing this constraint were rejected at generation time. Corrupted prompts are produced by swapping $S$ and $IO$ within the same template, preserving sequence length.

The factual association dataset consists of $n = 50$ subject-relation-object triples drawn from Meng et al. (2022)'s CounterFact evaluation set, restricted to triples where the object tokenizes to a single token under the Llama-3.2-3B tokenizer. Prompts follow the form "{subject} {relation verb}:" with corruptions produced by substituting a different subject entity (matched by entity type) that predicts a different object. Factual association experiments were run on Llama-3.2-3B only.
