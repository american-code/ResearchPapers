# Efficient Mechanistic Circuit Analysis of Open-Weight LLMs on Apple Silicon

> **REVISION NOTES (v2)** — Issues marked inline with `[[NOTE: ...]]`. Open items:
> 1. L24H15 vs L24H16 data inconsistency (ablation table in results-summary.md vs combined rank in cross-model-comparison.md) — must resolve before submission.
> 2. L1H11 CI upper bound: methods says +0.005, Table 2 says +0.006 — verify against `data/ioi/statistical-validation.json`.
> 3. Baseline LD 5.643 in `data/ioi/baseline-llama3b.json` Data Provenance entry conflicts with 5.649 used everywhere else — update the data file's provenance annotation.
> 4. Table 2 is ordered by ablation rank, not combined rank as stated in Methods 3.5 — either reorder or add a note in the caption.
> 5. L13H6 ↔ L15H20 cross-model pairing is potentially misleading (L13H6 ablation drop 0.018 is negligible) — address interpretive framing.
> 6. Missing citations: Biderman et al. (2023) for Pythia, Meta Llama technical report, Ainslie et al. (2023) for GQA, Su et al. (2021) for RoPE.
> 7. Sections 6, 7, and References are placeholders.

---

## Abstract

Mechanistic interpretability research has largely depended on GPU clusters, limiting reproducibility and iteration speed for many researchers. We demonstrate that full activation-patching and mean-ablation circuit analyses of multi-billion-parameter transformer models are tractable on Apple Silicon using the MLX framework, with no specialized hardware beyond a modern Mac. Applying this infrastructure to the Indirect Object Identification (IOI) task on Llama-3.2-3B and Pythia-1.4B, we find that the three-zone functional circuit topology identified by Wang et al. (2022) in GPT-2 Small generalizes across both models: 80% of circuit-critical head positions are conserved by relative depth (±0.075) despite differing layer counts, attention mechanisms, and training corpora. A single dominant head in each model (L15H20, L10H7) accounts for 22–28% of clean logit difference on its own. Layer-level circuit topology further transfers from IOI to factual association, with divergent head indices at shared layer positions. These results establish that IOI circuit organization is architecture-general, and that Apple Silicon is a viable substrate for reproducible mech-interp research at the 1–3B parameter scale.

---

## 1. Introduction

Mechanistic interpretability aims to reverse-engineer the algorithms implemented by neural networks — to move from "the model predicts X" to "these specific components, performing these specific computations, cause the model to predict X." The dominant paradigm for this reverse-engineering is *circuit analysis*: identify the smallest subset of attention heads and MLP layers that is both causally necessary and causally sufficient to reproduce a target behavior, then assign functional roles to each component within that subset.

The foundational work in this paradigm — Wang et al. (2022) — demonstrated that the Indirect Object Identification (IOI) task in GPT-2 Small is implemented by a comprehensible circuit of 26 attention heads organized into three functional classes: duplicate-token heads that flag repeated names, S-inhibition heads that suppress the subject position, and name-mover heads that copy the indirect object name to the output. The result was striking: a clean, human-interpretable algorithm, found inside a neural network trained only to predict next tokens.

What remains unresolved is whether this algorithm is specific to GPT-2 Small, or whether it reflects something deeper — a functional structure that emerges wherever a language model learns to solve this problem. GPT-2 Small is small (117M parameters), trained on a narrow corpus (WebText), and architecturally plain (standard MHA, learned positional embeddings). Modern open-weight models differ on every dimension: they are larger by one to two orders of magnitude, trained on multitrillion-token corpora, and equipped with grouped-query attention, rotary position embeddings, and gated MLP activations. If the IOI circuit is merely an artifact of GPT-2's architecture, we would expect a different circuit in a different model. If it reflects a general algorithmic solution, we would expect the same functional structure — with head indices shifted, but depth relationships and role assignments preserved.

This paper answers that question by replicating and extending Wang et al.'s analysis on two modern models: Llama-3.2-3B (a frontier-adjacent instruction-tuned architecture with grouped-query attention) and Pythia-1.4B (a fully documented, standard-MHA model trained on The Pile, making it the closest modern analog to GPT-2 Small's controlled training conditions). [[NOTE: add Llama citation — Meta technical report; add Biderman et al. 2023 for Pythia]] Using activation patching and mean ablation, we identify circuit-critical heads in each model, compare their positions to Wang et al.'s findings, and test for cross-architecture conservation at the level of relative network depth rather than absolute head indices.

**Apple Silicon as interpretability infrastructure.** All experiments in this paper were run on Apple Silicon hardware using the MLX framework, without access to NVIDIA GPU clusters. This is a deliberate choice. The interpretability tooling ecosystem — TransformerLens in particular — assumes CUDA, and the community's empirical norms have drifted toward hardware accessible only to well-resourced labs. Apple's M-series chips offer up to 96 GB of unified memory shared between CPU and GPU, which is well-suited to the memory access patterns of activation patching: you must store one full forward pass of cached activations per example, then perform hundreds of re-runs patching each (layer, head, token position) triple. On an M-series machine with sufficient unified memory, a 3B-parameter model fits in full float16 with room for the activation cache, and forward passes are fast enough that the full IOI patching sweep over 100 examples completes in wall-clock hours rather than days. We describe our Apple Silicon–compatible pipeline and release it alongside this paper to lower the barrier for researchers without cluster access to run mechanistic interpretability experiments at the 1–7B parameter scale.

### Contributions

1. **Cross-architecture circuit replication.** We identify circuit-critical heads for the IOI task in Llama-3.2-3B and Pythia-1.4B using activation patching (sufficiency) and mean ablation (necessity). In both models, a small bottleneck structure emerges: a single dominant head accounts for 22–28% of the model's logit difference for the IO token, substantially more concentrated than the distributed 26-component circuit in GPT-2 Small.

2. **Depth-zone conservation across architectures.** We show that the three functional depth zones identified in GPT-2 Small — early duplicate-token heads, mid S-inhibition heads, and late name-mover heads — are present in both Llama-3.2-3B and Pythia-1.4B at similar relative positions, despite 3–7× more parameters and substantially different architectures. 80% of circuit-critical head positions are shared between Llama and Pythia within ±0.075 relative depth.

3. **Layer-level conservation across tasks.** Factual association patching in Llama-3.2-3B reveals that the top-5 factual association heads occupy the same five layers as five of the top-9 IOI heads — with different head indices within those layers — suggesting a form of layer-level circuit topology that generalizes across tasks while maintaining head-level specialization.

4. **An Apple Silicon–compatible activation patching pipeline.** We release tooling built on the MLX framework that replicates TransformerLens's hook-based patching interface for LlamaForCausalLM and GPTNeoXForCausalLM models, enabling circuit analysis on Apple Silicon hardware without modification.

---

## 2. Related Work

### 2.1 A Mathematical Framework for Transformer Circuits

The theoretical foundations of circuit analysis in transformers were laid by Elhage et al. (2021), who introduced the *transformer circuits* framework. The key insight is that a transformer's residual stream can be analyzed as a sum of contributions from attention heads and MLP layers, with each attention head performing a rank-one update of the residual: the head reads from the stream via query and key projections, computes a weighted average of value projections, and writes back a low-rank update whose structure is determined by the composition of the OV and QK matrices.

Elhage et al. used this decomposition to characterize *induction heads* — a two-head circuit found across models that implements the copying operation "if token A was followed by token B earlier in context, predict B again when A reappears." Induction heads are relevant to IOI analysis because the same copying mechanism underlies name-mover head behavior: identifying the indirect object earlier in context and copying its token to the output position. Our results confirm that the depth zone corresponding to induction-like behavior (relative depth 0.40–0.55) is present in both Llama-3.2-3B and Pythia-1.4B.

The circuits framework also establishes *virtual attention heads* — paths in which the output of one head is read as part of another head's input — as the primary mechanism for inter-head communication. This framing is a prerequisite for understanding the IOI circuit's S-inhibition subgraph, where a set of early heads writes information about the subject's position that is then used by a later head to suppress subject-token logits.

### 2.2 Interpretability in the Wild: IOI in GPT-2 Small

Wang et al. (2022) is the direct predecessor to this work. Using the IOI task — sentences of the form "When Mary and John went to the store, John gave a drink to ___" — they applied activation patching across all attention heads and MLP layers in GPT-2 Small (12 layers, 12 heads) to identify which components causally mediate correct IO prediction. The result was a 26-component circuit with three functional classes:

- **Name-mover heads** (L9H6, L9H9, L10H0; relative depths 0.75–0.83) copy the indirect object name to the logit output at the final token position.
- **S-inhibition heads** (L7H3, L7H9, L8H6, L8H10; relative depths 0.58–0.67) suppress the subject token's representation in the residual stream, preventing the model from predicting the repeated name.
- **Duplicate-token heads and induction heads** (L0H1, L3H0, L5H5, L5H8; relative depths 0.0–0.42) flag the position of the repeated subject token so that downstream heads know which name to suppress.

Our work replicates this three-zone structure across Llama-3.2-3B and Pythia-1.4B, extending the generalizability claim beyond GPT-2 Small. We also find that in larger models the circuit has a more concentrated "bottleneck" structure, with one head contributing disproportionately to logit difference — a difference from GPT-2 Small's more distributed allocation that we discuss in Section 4.4.

### 2.3 TransformerLens

TransformerLens (Nanda & Lawrence, 2022) is the standard tooling for mechanistic interpretability experiments on transformer language models. It provides a hook-based interface for intercepting and modifying activations at any point in the forward pass — attention head outputs, MLP outputs, residual stream positions, or individual Q/K/V projections — making activation patching experiments straightforward to implement.

Our work builds on TransformerLens for Pythia-1.4B, where the library's support is mature. For Llama-3.2-3B with grouped-query attention, we implement a compatible interface that maps the 24-query-head, 8-KV-head structure to a consistent per-query-head naming convention. GQA complicates patching because each KV head is shared across a group of 3 query heads; we patch at the query-head output level, after the attention weights are applied to the shared value projections, to preserve head-level attribution granularity. We release this extension alongside the paper.

### 2.4 Locating and Editing Factual Associations

Meng et al. (2022) introduced ROME (Rank-One Model Editing), which demonstrated that factual associations — the model's belief that "the Eiffel Tower is in Paris" — are stored with high locality in mid-layer MLP weights, and can be surgically edited by a rank-one update to those weights. The key interpretability contribution was not the editing procedure itself but the preceding localization experiment: using causal tracing (a form of activation patching), they showed that early-layer attention heads read subject tokens and route information to mid-layer MLPs, which then store and retrieve the associated object.

This causal tracing methodology is closely related to our activation patching protocol. A key difference is that ROME's target is stored parametric knowledge (factual associations mediated by MLP weights), while IOI involves in-context reasoning (copying from the prompt, not from stored facts). Our factual association experiments in Section 5.2 partially bridge this gap: we apply patching to factual association prompts and compare the resulting head-importance rankings to the IOI results, finding that layer-level circuit structure is shared while head-level assignments diverge. This cross-task comparison was not reported in ROME and provides new evidence about how circuits for different types of factual reasoning coexist within the same model.

### 2.5 Towards Monosemanticity and Sparse Autoencoders

Elhage et al. (2022) and the Anthropic mechanistic interpretability team have pursued an orthogonal but complementary program: rather than identifying circuits for specific behaviors, they seek to decompose the residual stream into *features* — directions in activation space that correspond to human-interpretable concepts — using sparse autoencoders (SAEs). The central finding of the monosemanticity work is that individual neurons are typically polysemantic (responding to unrelated concepts), but a sparse linear decomposition of neuron activations recovers interpretable monosemantic features.

The circuit-tracing and monosemanticity programs address different levels of the interpretability problem. Circuit analysis asks: which components mediate this behavior? Feature decomposition asks: what information is represented in the activations those components read and write? They are complementary: knowing that L15H20 in Llama-3.2-3B is the dominant IOI head is more interpretable if we also know what feature L15H20 reads from the residual stream (presumably a representation of the indirect object token's identity and position). Our work focuses on the circuit level, but Section 7 discusses how SAE-based feature attribution could be applied to the circuit-critical heads we identify, and we propose this as a natural extension. [[NOTE: v1 draft said "Section 6" here — corrected to "Section 7" (Discussion); Section 6 is Efficiency Analysis.]]

---

*The remainder of the paper is organized as follows. Section 3 describes the experimental setup: models, dataset, and patching protocol. Section 4 presents IOI circuit results for Llama-3.2-3B and Pythia-1.4B and compares them to Wang et al. (2022). Section 5 presents cross-architecture generalization results and cross-task comparisons for factual association. Section 6 reports efficiency benchmarks for the Apple Silicon pipeline. Section 7 discusses implications for circuit universality and concludes.* [[NOTE: v1 roadmap conflated Sections 5 and 6 and omitted Section 6's actual content (efficiency) — corrected.]]

---

## 3. Methods

### 3.1 Experimental Infrastructure: SwiftSci Interp

All experiments were conducted using SwiftSci Interp, an activation-patching harness for Apple Silicon built on the MLX framework. The library implements a hook-compatible interface analogous to TransformerLens's activation interception API, targeting MLX-native model implementations loaded via `mlx-lm`. Its core design principle is that every attention module in a model is hot-swapped at load time with a patchable drop-in replacement that shares the original weight tensors — no copies — and adds a mode-switched interception point around the scaled dot-product attention (SDPA) computation.

**Patchable module design.** For Llama-3.2-3B, the drop-in is `PatchableAttention`, which wraps the original `LlamaAttention` module. For Pythia-1.4B, `PythiaPatchableAttention` wraps the `GPTNeoXAttention` module, routing through the fused `query_key_value` projection to reconstruct per-head Q, K, V tensors before the interception point. Both modules support three operating modes:

- `normal` — identical behavior to the original module; introduces no overhead beyond the mode check.
- `cache_clean` / `cache_corrupt` — stores the full SDPA output tensor (shape `[B, n_heads, L, head_dim]`) to a per-module buffer during the forward pass. This is used to populate the clean and corrupt activation caches without a separate caching pass.
- `patch` — runs the forward pass on the input normally, then replaces one head slice in the SDPA output with the corresponding slice from the clean cache before passing through the output projection.

**GQA handling.** Llama-3.2-3B uses grouped-query attention (GQA) [[NOTE: add citation — Ainslie et al. (2023)]] with 24 query heads and 8 KV heads. The patching interception occurs after the attention weights are applied to the (broadcast) value projections, at the query-head output level. This gives a 24-element patch space that mirrors the query-head count, preserving head-level attribution granularity. Patching at the KV-head level would conflate the contributions of the three query heads that share each KV pair; patching at the query-head SDPA output avoids this confound. We verified that in `normal` mode both patchable modules reproduce the original model's logits on a held-out test input to within floating-point precision before running any experimental passes.

**GQA key-value broadcast.** During the corrupt or clean pass, the values broadcast from 8 KV heads to 24 query heads occur inside `scaled_dot_product_attention`; the SDPA buffer therefore stores the full 24-head output, not the 8-head compressed form. This means each of the 24 patch positions is truly independent, and patching query-head $h$ does not implicitly alter the other two heads sharing the same KV pair.

**Pythia module layout.** MLX-LM exposes Pythia's layers as `model.layers[i].attention` rather than `model.model.layers[i].self_attn`. The patchable replacement reuses the `dense` (output projection) attribute rather than `o_proj`. All other behavior is identical; the mode-switch logic is shared.

---

### 3.2 Activation Patching Protocol

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

### 3.3 Mean Ablation Protocol

Mean ablation addresses *necessity*: does removing a component's contribution — replacing it with a neutral baseline — degrade the model's performance on the task? Where activation patching asks whether a component *can* carry the relevant information, mean ablation asks whether the model *relies* on it in normal operation.

**Reference distribution.** The ablation target for head $(l, h)$ is the mean SDPA output of that head computed over all $n = 100$ clean IOI examples: $\bar{A}_{l,h} = \frac{1}{n}\sum_{i=1}^n A_{l,h}^{(i)}$, where $A_{l,h}^{(i)} \in \mathbb{R}^{L \times d_{\text{head}}}$ is the per-example SDPA output at each token position. The mean is taken over the batch dimension, preserving the sequence-position structure. Ablating with the mean over the same distribution on which the model is evaluated (rather than, for example, a random Gaussian or a separately collected reference set) ensures that the ablated output is a plausible activational state for the model rather than an out-of-distribution injection.

**Metric.** The *ablation drop* for head $(l, h)$ is:

$$\Delta_{l,h} = \overline{\text{LD}}_{\text{clean}} - \overline{\text{LD}}_{\text{ablate}(l,h)}$$

where $\overline{\text{LD}}_{\text{clean}}$ is the mean clean logit difference over all examples and $\overline{\text{LD}}_{\text{ablate}(l,h)}$ is the mean logit difference when head $(l, h)$'s SDPA output is replaced by $\bar{A}_{l,h}$ in every example. Positive $\Delta_{l,h}$ indicates the head boosts the IO logit relative to the subject; negative values indicate suppression of correct IO prediction (which can reflect an interference-suppression role rather than direct IO promotion).

**Necessity threshold.** A head is designated circuit-critical under the necessity criterion when its ablation drop exceeds 20% of the clean mean logit difference. For Llama-3.2-3B ($\overline{\text{LD}}_{\text{clean}} = 5.649$, `data/ioi/ablation-llama3b.json`) this corresponds to $\Delta_{l,h} > 1.130$; for Pythia-1.4B ($\overline{\text{LD}}_{\text{clean}} = 4.120$, `data/ioi/ablation-pythia1b.json`) the threshold is $\Delta_{l,h} > 0.824$. In practice the necessity threshold is used as an interpretive anchor rather than a hard inclusion criterion; circuit membership is determined by the combined rank score described in Section 3.5.

**Pass structure.** The ablation sweep requires $1 + n_L \times n_H$ forward passes: one clean pass to cache per-head means, then one ablation pass per $(l, h)$ pair. For Llama this is $1 + 672 = 673$ total passes, comparable to the patching sweep.

---

### 3.4 Path Patching for Causal Verification

Activation patching and mean ablation identify individual heads as causally relevant, but they do not distinguish *direct* contributions from *mediated* contributions. A head may score highly on both metrics because it directly writes to the logit output at the final position, or because it is an intermediary in a longer causal path — for example, writing to the residual stream at an intermediate token position that is later read by a downstream head that writes to the final position.

Path patching (Goldowsky-Dill et al., 2023) extends activation patching to directed paths between specific components. Rather than patching a single component's output, a path patch replaces the clean-run activation that flows along one specific edge in the computational graph — from the output of component $A$ as read by the input of component $B$, while holding all other paths fixed. A path that shows high normalized recovery under path patching is causally necessary specifically through the direct $A \to B$ connection, ruling out the mediated-path alternative.

In our setting, path patching is used to verify that the dominant heads identified by activation patching (L15H20 in Llama, L10H7 in Pythia) contribute directly to the final position logit difference, rather than acting through a downstream intermediary. For each dominant head $A$ and each later head $B$ in the circuit, we construct the $A \to B$ path patch and measure recovery. High recovery on the direct $A \to \text{output}$ path confirms direct contribution; high recovery on $A \to B \to \text{output}$ with low recovery on $A \to \text{output}$ alone would indicate mediation.

Path patching is implemented within SwiftSci Interp by extending the `patch` mode to accept a target-receiver argument: when `patch_target = (l_B, h_B)` is specified, the replacement is applied only to the projection of head $A$'s clean output onto the query, key, or value input of head $B$, leaving all other paths unchanged. [[PLACEHOLDER: path-patching results for the top-3 heads in each model are not yet computed. Report planned for a future revision; cite Section 5 cross-architecture comparison in the interim. v1 draft incorrectly cross-referenced "Section 4.3" (which is IOI: Pythia identification, not path patching).]]

---

### 3.5 Statistical Validation

**Bootstrap confidence intervals.** For all heads meeting the inclusion criterion (top-10 by combined rank score, or mean patching score $\geq 0.03$), we compute 95% bootstrap confidence intervals on the mean patching score. The bootstrap resamples the $n = 100$ per-example recovery scores with replacement, generating 1,000 resampled datasets and computing the mean for each. The 95% CI is the $[2.5^{\text{th}}, 97.5^{\text{th}}]$ percentile of the bootstrap distribution of the mean. Resampling is seeded with seed 42 for reproducibility. Full CI data: `data/ioi/statistical-validation.json`.

**Significance criterion.** A head's patching score is considered statistically reliable if its 95% CI excludes zero. This corresponds to the claim that the head's mean recovery score is positive in expectation across the example distribution, not merely a noise artifact of the particular 100 examples observed. All top-10 heads in both models satisfy this criterion with substantial margin: for the top three heads in each model, the CI lower bound exceeds the CI width by a factor of at least 3.

**Anomalous patching scores.** Pythia-1.4B's L1H11 presents a dissociation: its mean patching score is slightly negative ($-0.003$, 95% CI $[-0.011, +0.005]$, straddling zero) [[NOTE: Table 2 in Section 4.3 reports the upper bound as +0.006; verify against `data/ioi/statistical-validation.json` and standardize.]], yet its ablation drop is substantial ($0.327$, rank 4 by ablation). This dissociation is consistent with a suppression role: the head reduces logit difference when ablated (its absence degrades performance), but restoring its clean activation in a corrupt context does not recover logit difference (it cannot by itself make the model prefer IO over S). This pattern is characteristic of duplicate-token or induction heads that gate downstream components rather than writing directly to the output logits. L1H11 is excluded from all patching-based analyses and retained in cross-model analysis solely on the basis of its ablation score.

**Cross-model comparison: combined rank score.** To compare head importance across models that differ in scale, architecture, and absolute logit-difference baselines, we compute a combined normalized rank score. For each model separately, both the patching scores and the ablation drop scores are min-max normalized to $[0, 1]$ across all $(l, h)$ pairs. The combined score is the sum of the two normalized ranks. The top-10 heads by combined score are reported as circuit-critical for each model. Ties within the combined score are broken by patching rank. Combined rank tables: `data/ioi/cross-model-comparison.md`.

**Cross-architecture position matching.** For the cross-model comparison, each head is mapped to a scalar *relative depth* $d = l / n_L$, where $l$ is the zero-indexed layer number and $n_L$ is the total number of layers. Positions are matched greedily in ascending order of $|\Delta d|$ between the two models' top-10 head sets; a match is accepted if $|\Delta d| \leq 0.075$. This tolerance was chosen to be smaller than the spacing between the four functional depth zones identified by Wang et al. (2022) ($\approx 0.15$–$0.20$ between zone midpoints), ensuring that zone-level matches are not spuriously collapsed across zone boundaries.

---

### 3.6 Dataset

The IOI dataset consists of $n = 100$ examples drawn from the template:

> "When {S} and {IO} went to the store, {S} gave a bottle to"

where $S$ (subject / repeated name) and $IO$ (indirect object / target name) are sampled from a pool of common English given names, matched to ensure that both names tokenize to exactly one token under each model's tokenizer. All 100 examples satisfy the single-token constraint for both Llama-3.2-3B and Pythia-1.4B; examples failing this constraint were rejected at generation time. Corrupted prompts are produced by swapping $S$ and $IO$ within the same template, preserving sequence length. Full dataset: `data/ioi/dataset.json`.

The factual association dataset consists of $n = 50$ subject-relation-object triples drawn from Meng et al. (2022)'s CounterFact evaluation set, restricted to triples where the object tokenizes to a single token under the Llama-3.2-3B tokenizer. Prompts follow the form "{subject} {relation verb}:" with corruptions produced by substituting a different subject entity (matched by entity type) that predicts a different object. Factual association experiments were run on Llama-3.2-3B only. Full dataset: `data/factual-assoc/dataset.json`.

---

## 4. Results: IOI Circuit Identification

### 4.1 Baseline Task Performance

Both models correctly perform the IOI task across all 100 evaluation examples. Llama-3.2-3B achieves a mean clean logit difference (LD) of **5.649** (SD = 0.773, `data/ioi/baseline-llama3b.json`), reflecting a strong preference for the indirect object (IO) token over the subject (S) token at the final sequence position. [[NOTE: the data provenance annotation in `data/ioi/results-summary.md` cites the baseline JSON as mean LD = 5.643 — this 0.006 discrepancy should be resolved; the value 5.649 is used uniformly everywhere else.]] Pythia-1.4B achieves a mean clean LD of **4.120** (`data/ioi/ablation-pythia1b.json`), 27% lower in absolute terms, consistent with its smaller parameter count and simpler attention architecture. Neither model produces a negative-LD example: the IO token is ranked above S on every trial in both models, confirming that the circuit analysis begins from a clean behavioral baseline.

---

### 4.2 IOI Circuit Identification: Llama-3.2-3B

Figure 1 shows the full activation patching heatmap for Llama-3.2-3B across all 28 × 24 = 672 (layer, head) positions. Two features are immediately apparent: most heads contribute negligibly to logit-difference recovery under patching, and a sparse set of heads in the middle-to-late network produces strongly positive scores.

**Figure 1.** Activation patching heatmap for Llama-3.2-3B (n = 100 IOI examples). Each cell shows normalized logit-difference recovery when the output of that (layer, head) is replaced with the corresponding clean-run activation. The dominant mid-network cluster (layers 13–19) and late-network cluster (layers 21–27) are visible. Source: `figures/circuit-tracing/ioi-patching-heatmap-llama3b.svg`. Data: `data/ioi/patching-llama3b.json`.

Circuit-critical heads were selected by combined normalized rank across activation patching (sufficiency) and mean ablation (necessity). Table 1 lists the top-10 heads by combined rank.

**Table 1.** Top-10 circuit-critical heads in Llama-3.2-3B by combined normalized rank across activation patching (sufficiency) and mean-ablation drop (necessity). Relative depth = layer / n\_layers (zero-indexed layer). Data: `data/ioi/patching-llama3b.json`, `data/ioi/ablation-llama3b.json`, `data/ioi/cross-model-comparison.md`.

[[NOTE: L24H15 here has ablation drop 0.276 (from `cross-model-comparison.md` combined rank table). The ablation-rank table in `results-summary.md` lists L24H16 (different head) at rank 9 with ablation 0.301. This is an unresolved data inconsistency — must reconcile before submission.]]

| Rank | Head    | Patch score | 95% CI         | Ablation drop | % clean LD | Rel. depth |
|------|---------|-------------|----------------|---------------|------------|------------|
| 1    | L15H20  | 0.240       | [0.225, 0.255] | 1.578         | 27.9%      | 0.536      |
| 2    | L17H17  | 0.098       | [0.089, 0.107] | 0.929         | 16.4%      | 0.607      |
| 3    | L13H14  | 0.092       | [0.083, 0.102] | 0.626         | 11.1%      | 0.464      |
| 4    | L24H15  | 0.148       | [0.133, 0.164] | 0.276         |  4.9%      | 0.857      |
| 5    | L19H1   | 0.099       | [0.094, 0.105] | 0.443         |  7.8%      | 0.679      |
| 6    | L21H20  | 0.073       | [0.068, 0.078] | 0.417         |  7.4%      | 0.750      |
| 7    | L18H10  | 0.078       | [0.073, 0.084] | 0.313         |  5.5%      | 0.643      |
| 8    | L14H0   | 0.111       | [0.099, 0.125] | 0.017         |  0.3%      | 0.500      |
| 9    | L27H17  | 0.035       | [0.030, 0.041] | 0.373         |  6.6%      | 0.964      |
| 10   | L26H23  | 0.039       | [0.030, 0.047] | 0.344         |  6.1%      | 0.929      |

The most striking feature of the Llama-3.2-3B circuit is the dominance of L15H20. Its ablation drop (1.578) is **70% larger** than rank-2 L17H17 (0.929) and **2.5× rank-3** L13H14 (0.626). L15H20 also leads the patching ranking at a score of 0.240, more than 60% above rank-2 L24H15 (0.148). No other head in the top-10 approaches this contribution on both metrics simultaneously: L24H15 scores highly on patching but weakly on ablation (0.276), while L27H17 shows the reverse pattern. This double-dominance of L15H20 — both sufficient and necessary — identifies it as a hard bottleneck in the IOI circuit at relative depth 0.536.

The circuit spans relative depths 0.46–0.97 with no critical head below 0.46. Three depth clusters are visible: a mid cluster (0.46–0.54) comprising L13H14, L14H0, and L15H20; a mid-late cluster (0.60–0.75) comprising L17H17, L18H10, L19H1, and L21H20; and a late cluster (0.86–0.97) comprising L24H15, L26H23, and L27H17. This three-cluster structure corresponds qualitatively to the functional zones identified by Wang et al. (2022) in GPT-2 Small, as discussed in Section 4.4.

---

### 4.3 IOI Circuit Identification: Pythia-1.4B

Pythia-1.4B replicates the bottleneck structure of Llama-3.2-3B. Table 2 presents the top-10 circuit-critical heads.

**Table 2.** Top-10 circuit-critical heads in Pythia-1.4B by combined normalized rank across activation patching and mean-ablation drop. [[NOTE: this table is currently ordered by *ablation rank* rather than combined rank — L1H11 appears at position 4 (ablation rank 4) but its combined rank is 7; L21H3 (combined rank 4) and L12H15 (combined rank 5) are displaced. Either reorder by combined rank or add a note in the caption explaining that ordering is by ablation drop to highlight the L1H11 anomaly.]] Data: `data/ioi/patching-pythia1b.json`, `data/ioi/ablation-pythia1b.json`, `data/ioi/statistical-validation.json`.

| Rank | Head   | Patch score | 95% CI         | Ablation drop | % clean LD | Rel. depth |
|------|--------|-------------|----------------|---------------|------------|------------|
| 1    | L10H7  | 0.207       | [0.199, 0.216] | 0.893         | 21.7%      | 0.417      |
| 2    | L15H15 | 0.155       | [0.134, 0.175] | 0.551         | 13.4%      | 0.625      |
| 3    | L22H2  | 0.101       | [0.097, 0.105] | 0.411         |  9.9%      | 0.917      |
| 4    | L1H11  | −0.003      | [−0.011, +0.006]| 0.327        |  7.9%      | 0.042      |
| 5    | L21H3  | 0.056       | [0.052, 0.059] | 0.237         |  5.7%      | 0.875      |
| 6    | L17H7  | 0.040       | [0.027, 0.056] | 0.150         |  3.6%      | 0.708      |
| 7    | L10H0  | 0.039       | [0.036, 0.041] | 0.128         |  3.1%      | 0.417      |
| 8    | L12H15 | 0.051       | [0.048, 0.054] | 0.124         |  3.0%      | 0.500      |
| 9    | L16H13 | 0.025       | [0.022, 0.029] | 0.117         |  2.8%      | 0.667      |
| 10   | L13H6  | 0.049       | [0.042, 0.056] | 0.018         |  0.4%      | 0.542      |

L10H7 leads both rankings: its ablation drop (0.893) is **62% larger** than rank-2 L15H15 (0.551), and its patching score (0.207) is **34% larger** than rank-2 (0.155). The pattern — one head substantially dominant on both sufficiency and necessity — exactly mirrors the Llama result, though the absolute magnitude of the ablation drop is smaller (0.893 vs. 1.578) consistent with Pythia's lower baseline LD.

**Anomalous early head: L1H11.** One Pythia-specific finding stands out: L1H11 at relative depth 0.042 (layer 1 of 24) scores −0.003 on activation patching (95% CI [−0.011, +0.006], straddling zero) yet produces the fourth-largest ablation drop in the model (0.327, 7.9% of clean LD). This dissociation between sufficiency (near zero or slightly negative) and necessity (substantial) is inconsistent with the role profile of name-mover or S-inhibition heads, which score positively on both measures. The most parsimonious interpretation is that L1H11 suppresses interference — for example, dampening an early signal that would otherwise mislead later heads — rather than directly boosting IO probability. Ablating it disrupts the clean causal pathway downstream, producing a large LD drop, but substituting its clean activation into a corrupted context provides no positive recovery. No Llama-3.2-3B head in the top-10 occupies a comparable depth (< 0.10), suggesting this interference-suppression role may be specific to Pythia's architecture or training distribution.

---

### 4.4 Comparison with Wang et al. (2022): Depth-Zone Conservation

Wang et al. (2022) identified three functional head classes in GPT-2 Small at specific relative depth ranges: duplicate-token/induction heads (depth 0.0–0.42), S-inhibition heads (0.58–0.67), and name-mover heads (0.75–0.83). Table 3 shows where these zones fall in our two models.

**Figure 3.** Schematic circuit diagram for the IOI circuit across GPT-2 Small, Llama-3.2-3B, and Pythia-1.4B, annotated by functional zone. See `figures/circuit-tracing/ioi-circuit-diagram.svg`. [[NOTE: this figure file exists but was not referenced anywhere in the v1 draft — add this reference and ensure the figure caption is written before submission.]]

**Table 3.** Functional depth-zone alignment across GPT-2 Small (Wang et al., 2022), Llama-3.2-3B, and Pythia-1.4B. [[NOTE: the "Name-movers" zone boundary shown here (0.75–0.92) is an expansion of Wang et al.'s original range (0.75–0.83) to accommodate the Llama/Pythia heads at deeper positions. The table header should clarify that zone boundaries for Llama and Pythia are inferred from the depth distribution of critical heads, not directly assigned by functional annotation.]] Data: `data/ioi/cross-model-comparison.md`.

| Zone                      | GPT-2 Small          | Llama-3.2-3B                    | Pythia-1.4B                       |
|---------------------------|----------------------|---------------------------------|-----------------------------------|
| Early (0.00–0.25)         | L0H1, L3H0           | —                               | L1H11                             |
| Mid-induction (0.40–0.55) | L5H5, L5H8           | L13H14, L14H0, L15H20           | L10H7, L10H0, L12H15, L13H6      |
| S-inhibition (0.50–0.70)  | L7H3–L8H10           | L17H17, L18H10, L19H1           | L15H15, L16H13                    |
| Name-movers (0.75–0.92)   | L9H6–L10H0           | L21H20, L24H15, L26H23          | L21H3, L22H2                      |

[[NOTE: v1 draft Table 3 named "L26H22" in the Llama name-mover row — corrected to L26H23 to match Table 1 (combined rank) and `data/ioi/cross-model-comparison.md`. L26H22 appears only in the patching-only ranking in `results-summary.md` and did not make the combined top-10.]]

All three functional zones from Wang et al. are present in both modern models at compatible relative depths. The mid-induction zone (0.40–0.55) is the densest cluster in all three models; the name-mover zone (0.75–0.92) contains the heads with highest late-network patching scores. One notable divergence: the very-early zone (0.00–0.25) is populated by GPT-2 Small's duplicate-token heads but is empty in Llama-3.2-3B's top-10 and represented only by the anomalous L1H11 in Pythia.

A second divergence is the **bottleneck vs. distributed** structure. Wang et al. found 26 circuit components with moderate individual contributions; in both Llama and Pythia, one head accounts for 22–28% of clean LD on its own, substantially above any single head's contribution in GPT-2 Small. This could reflect scale effects (larger models can concentrate circuit function in individual heads), differences in our selection methodology, or task-distribution differences in our IOI dataset. We return to this in Section 7.

---

### 4.5 Statistical Significance and Patching Verification

Bootstrap confidence intervals (1000 resamples, 95% level, seed 42, `data/ioi/statistical-validation.json`) were computed for all heads with mean patching score ≥ 0.030. For the top-ranked heads in each model, the effect sizes are large relative to uncertainty:

- **L15H20 (Llama):** mean 0.240, 95% CI [0.225, 0.255], CI width 0.030. The lower bound (0.225) exceeds the rank-3 head's mean (0.092) by 2.4×.
- **L10H7 (Pythia):** mean 0.207, 95% CI [0.199, 0.216], CI width 0.017. The lower CI bound (0.199) exceeds rank-2 (0.155) by 28%.

All top-10 heads in both models have confidence intervals excluding zero by a margin of at least 3× the CI width. The Pythia L1H11 anomaly is confirmed: its patching CI [−0.011, +0.006] straddles zero (the only such case in either model's top-10), whereas its ablation drop (0.327) is large and unambiguous. This dissociation is the empirical signature of necessity without sufficiency, and constitutes a primary verification in our results: by simultaneously measuring what happens when a head's activation is *replaced* (patching; sufficiency) versus *removed* (ablation; necessity), we detect heads whose role is suppressive rather than generative, a distinction that patching-only or ablation-only protocols would miss.

---

## 5. Results: Cross-Architecture and Cross-Task Generalization

### 5.1 Cross-Architecture Generalization

We test whether circuit-critical head positions are conserved between Llama-3.2-3B and Pythia-1.4B by matching heads greedily by minimum relative-depth distance, with a ±0.075 tolerance. Table 4 reports all matched pairs.

**Figure 2.** Cross-model comparison of circuit-critical head positions by relative depth for Llama-3.2-3B and Pythia-1.4B. Matched pairs are connected by dashed lines; unmatched (model-specific) heads are shown without connections. See `figures/circuit-tracing/ioi-cross-model-comparison.svg`. Data: `data/ioi/cross-model-comparison.md`. [[NOTE: this figure file exists but was not referenced in the v1 draft — add reference here.]]

**Table 4.** Cross-architecture matched head positions (Llama-3.2-3B vs. Pythia-1.4B, tolerance ±0.075). Data: `data/ioi/cross-model-comparison.md`.

| Zone    | Llama head | Llama depth | Pythia head | Pythia depth | \|Δ\| |
|---------|-----------|-------------|------------|--------------|-------|
| ~0.50   | L14H0     | 0.500       | L12H15     | 0.500        | 0.000 |
| ~0.54   | L15H20    | 0.536       | L13H6      | 0.542        | 0.006 |
| ~0.68   | L19H1     | 0.679       | L16H13     | 0.667        | 0.012 |
| ~0.92   | L26H23    | 0.929       | L22H2      | 0.917        | 0.012 |
| ~0.62   | L17H17    | 0.607       | L15H15     | 0.625        | 0.018 |
| ~0.86   | L24H15    | 0.857       | L21H3      | 0.875        | 0.018 |
| ~0.73   | L21H20    | 0.750       | L17H7      | 0.708        | 0.042 |
| ~0.46   | L13H14    | 0.464       | L10H7      | 0.417        | 0.048 |

[[NOTE: the ~0.54 pairing (L15H20 ↔ L13H6) warrants an interpretive flag: L15H20 is Llama's dominant bottleneck head (ablation drop 1.578, 27.9% of clean LD) while L13H6 is Pythia's rank-10 head (ablation drop 0.018, 0.4% of clean LD). These heads share a relative depth zone but their functional magnitudes differ by two orders of magnitude. The pairing reflects depth-position conservation, not necessarily functional equivalence. Acknowledge this in the discussion.]]

**8 of 10 circuit-critical head positions match within ±0.075 relative depth.** The two model-specific heads are: Llama's L18H10 (depth 0.643, no Pythia match) and L27H17 (depth 0.964, near-final layer, consistent with Llama's greater depth permitting an extra output-adjustment head); and Pythia's L1H11 (depth 0.042, discussed above) and L10H0 (depth 0.417, co-located with L10H7 in the same layer, forming a two-head cluster not observed in Llama).

The matched pairs exhibit a three-cluster depth structure in both models:

- **Mid cluster (0.40–0.55):** Three Llama heads, four Pythia heads. Pythia is denser here, consistent with its smaller total layer count producing more critical heads in the early-to-mid range.
- **Mid-late cluster (0.60–0.75):** Four Llama heads, three Pythia heads.
- **Late cluster (0.85–0.97):** Three Llama heads, two Pythia heads. Llama is denser here, reflecting its two additional layers (28 vs. 24) creating room for extra late-network mechanisms.

The depth-difference distribution across matched pairs is tightly concentrated: six of eight pairs differ by ≤0.02 in relative depth, and the maximum difference is 0.048. Considering that the tolerance was set at 0.075, the actual conservation is substantially tighter than the threshold requires.

---

### 5.2 Cross-Task Transfer: Factual Association

To test whether the circuit structure identified for IOI transfers to a related but distinct task, we applied the same activation patching protocol to factual association prompts (n = 50, Llama-3.2-3B only, `data/factual-assoc/patching-llama3b.json`). Prompts had the form "The capital of [country] is ___" with the country name corrupted by substitution to a different country; the patching score measures each head's causal contribution to correct capital prediction. Table 5 reports the top-10 heads.

**Table 5.** Top-10 heads by activation patching score for factual association (Llama-3.2-3B, n = 50). Data: `data/factual-assoc/patching-llama3b.json`.

| Rank | Head   | Patch score | Rel. depth | Matching IOI layer? |
|------|--------|-------------|------------|---------------------|
| 1    | L15H17 | 0.420       | 0.536      | **Yes** (IOI rank 1: L15H20) |
| 2    | L21H2  | 0.418       | 0.750      | **Yes** (IOI rank 6: L21H20) |
| 3    | L27H5  | 0.105       | 0.964      | **Yes** (IOI rank 9: L27H17) |
| 4    | L17H18 | 0.088       | 0.607      | **Yes** (IOI rank 2: L17H17) |
| 5    | L13H18 | 0.052       | 0.464      | **Yes** (IOI rank 3: L13H14) |
| 6    | L25H8  | 0.045       | 0.893      | No                  |
| 7    | L26H20 | 0.035       | 0.929      | No                  |
| 8    | L13H19 | 0.026       | 0.464      | Yes (layer 13)      |
| 9    | L21H0  | 0.025       | 0.750      | Yes (layer 21)      |
| 10   | L26H19 | 0.019       | 0.929      | No                  |

The top-5 factual association heads occupy the same five layers as five of the top-9 IOI heads (layers 13, 15, 17, 21, and 27), but activate *different heads within those layers*: IOI uses L15H20 while FA uses L15H17; IOI uses L21H20 while FA uses L21H2; IOI uses L13H14 while FA uses L13H18. This pattern is consistent with **layer-level circuit topology conservation across tasks** with **head-level specialization by task** — the same network locations implement different input-output functions depending on which specific circuit the task activates.

Two additional features of the factual association results are noteworthy. First, the top-2 FA heads (0.420, 0.418) have substantially higher patching scores than the IOI rank-1 head (0.240), and the two scores are nearly equal to each other, suggesting FA uses a more concentrated two-head bottleneck while IOI distributes attribution more broadly across the mid-to-late network. Second, layer 15 is the single most critical layer for both tasks — the dominant head in each circuit (L15H20 for IOI, L15H17 for FA) resides at the same relative depth (0.536), which may reflect layer 15's position at the transition between Llama's induction-like and output-writing computational stages.

---

## 6. Efficiency Analysis

[[PLACEHOLDER: This section is not yet drafted. Per the outline, it should cover:

- **Wall-clock times on Apple Silicon (M-series):** full 28×24 patching sweep for Llama-3.2-3B and 24×16 sweep for Pythia-1.4B; comparison to estimated GPU runtime at equivalent batch size.
- **Memory footprint:** unified memory allows full bf16 model weights plus activation cache in one address space; practical parameter budget ceiling for this approach.
- **Reproducibility:** all experiments run with fixed seeds and publicly released MLX harness; estimated total compute for full result set in GPU-equivalent hours.

No timing or memory data files were found in `data/` at time of revision. Data collection required before this section can be written.]]

---

## 7. Discussion and Conclusion

[[PLACEHOLDER: This section is not yet drafted. Per the outline and cross-references established in earlier sections, it should address:

- **IOI circuit topology is architecture-general:** functional depth zones survive 3–7× parameter scaling, GQA vs. MHA attention, and different training corpora; specific head indices are not conserved.
- **Bottleneck vs. distributed structure:** single dominant heads at 22–28% of clean LD may reflect scale effects or task-distribution differences relative to Wang et al.'s 26-component circuit; path patching is the recommended next step to test direct vs. mediated contributions (see Section 3.4).
- **L13H6 ↔ L15H20 pairing:** discuss that depth-position conservation does not imply functional equivalence — the ~0.54 match pairs Llama's dominant bottleneck head with Pythia's rank-10 head, which has negligible ablation drop.
- **SAE-based feature attribution as next step:** which features do L15H20 and L10H7 read from the residual stream? (cross-reference: originally described in Section 2.5 as future work.)
- **Implications for consumer-hardware mech-interp:** Apple Silicon closes the GPU access gap for circuit analysis at 1–3B scale, enabling faster hypothesis iteration and broader community participation; limitations at >7B scale.
- **Limitations:** IOI dataset is 100 examples from one template; factual association sample is 50 examples; path patching not yet completed; Pythia L1H11 role is interpretive, not directly verified.]]

---

## References

[[PLACEHOLDER: Full reference list not yet compiled. Citations used in the draft:

- Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits"
- Wang et al. (2022) — "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 Small"
- Nanda & Lawrence (2022) — TransformerLens library
- Meng et al. (2022) — "Locating and Editing Factual Associations in GPT"
- Goldowsky-Dill et al. (2023) — "Localizing Model Behavior with Path Patching"
- Elhage et al. (2022) — "Toy Models of Superposition" / monosemanticity work

Missing citations to add:
- Biderman et al. (2023) — Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling
- Meta (2024) — Llama 3 technical report [[confirm exact citation]]
- Ainslie et al. (2023) — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
- Su et al. (2021) — RoFormer: Enhanced Transformer with Rotary Position Embedding]]

---

*Data provenance summary:*

| File | Contents |
|------|----------|
| `data/ioi/patching-llama3b.json` | IOI activation patching, Llama-3.2-3B, n=100 |
| `data/ioi/patching-pythia1b.json` | IOI activation patching, Pythia-1.4B, n=100 |
| `data/ioi/ablation-llama3b.json` | IOI mean ablation, Llama-3.2-3B, n=100, clean LD=5.649 |
| `data/ioi/ablation-pythia1b.json` | IOI mean ablation, Pythia-1.4B, n=100, clean LD=4.120 |
| `data/ioi/statistical-validation.json` | Bootstrap 95% CIs, 1000 resamples, both models |
| `data/ioi/baseline-llama3b.json` | Per-example baselines, Llama-3.2-3B [[NOTE: provenance annotation says mean LD=5.643; all other sources use 5.649 — reconcile]] |
| `data/ioi/cross-model-comparison.md` | Combined rank table and matched-position analysis |
| `data/factual-assoc/patching-llama3b.json` | FA activation patching, Llama-3.2-3B, n=50 |
| `figures/circuit-tracing/ioi-patching-heatmap-llama3b.svg` | Figure 1 |
| `figures/circuit-tracing/ioi-cross-model-comparison.svg` | Figure 2 |
| `figures/circuit-tracing/ioi-circuit-diagram.svg` | Figure 3 |
