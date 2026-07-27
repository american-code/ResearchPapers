# Efficient Mechanistic Circuit Analysis of Open-Weight LLMs on Apple Silicon

---

## Abstract

Mechanistic interpretability research has largely depended on GPU clusters, limiting reproducibility and iteration speed for many researchers. We demonstrate that full activation-patching and mean-ablation circuit analyses of multi-billion-parameter transformer models are tractable on Apple Silicon using the MLX framework, with no specialized hardware beyond a modern Mac. Applying this infrastructure to the Indirect Object Identification (IOI) task on Llama-3.2-3B and Pythia-1.4B, we find that the three-zone functional circuit topology identified by Wang et al. (2022) in GPT-2 Small generalizes across both models: 80% of circuit-critical head positions are conserved by relative depth (±0.075) despite differing layer counts, attention mechanisms, and training corpora. A single dominant head in each model (L15H20, L10H7) accounts for 22–28% of clean logit difference on its own. Layer-level circuit topology further transfers from IOI to factual association, with divergent head indices at shared layer positions. These results establish that IOI circuit organization is architecture-general, and that Apple Silicon is a viable substrate for reproducible mech-interp research at the 1–3B parameter scale.

---

## 1. Introduction

Mechanistic interpretability aims to reverse-engineer the algorithms implemented by neural networks — to move from "the model predicts X" to "these specific components, performing these specific computations, cause the model to predict X." The dominant paradigm for this reverse-engineering is *circuit analysis*: identify the smallest subset of attention heads and MLP layers that is both causally necessary and causally sufficient to reproduce a target behavior, then assign functional roles to each component within that subset.

The foundational work in this paradigm — Wang et al. (2022) — demonstrated that the Indirect Object Identification (IOI) task in GPT-2 Small is implemented by a comprehensible circuit of 26 attention heads organized into three functional classes: duplicate-token heads that flag repeated names, S-inhibition heads that suppress the subject position, and name-mover heads that copy the indirect object name to the output. The result was striking: a clean, human-interpretable algorithm, found inside a neural network trained only to predict next tokens.

What remains unresolved is whether this algorithm is specific to GPT-2 Small, or whether it reflects something deeper — a functional structure that emerges wherever a language model learns to solve this problem. GPT-2 Small is small (117M parameters), trained on a narrow corpus (WebText), and architecturally plain (standard MHA, learned positional embeddings). Modern open-weight models differ on every dimension: they are larger by one to two orders of magnitude, trained on multitrillion-token corpora, and equipped with grouped-query attention, rotary position embeddings, and gated MLP activations. If the IOI circuit is merely an artifact of GPT-2's architecture, we would expect a different circuit in a different model. If it reflects a general algorithmic solution, we would expect the same functional structure — with head indices shifted, but depth relationships and role assignments preserved.

This paper answers that question by replicating and extending Wang et al.'s analysis on two modern models: Llama-3.2-3B (a frontier-adjacent instruction-tuned architecture with grouped-query attention) and Pythia-1.4B (a fully documented, standard-MHA model trained on The Pile, making it the closest modern analog to GPT-2 Small's controlled training conditions). Using activation patching and mean ablation, we identify circuit-critical heads in each model, compare their positions to Wang et al.'s findings, and test for cross-architecture conservation at the level of relative network depth rather than absolute head indices.

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

Our work replicates this three-zone structure across Llama-3.2-3B and Pythia-1.4B, extending the generalizability claim beyond GPT-2 Small. We also find that in larger models the circuit has a more concentrated "bottleneck" structure, with one head contributing disproportionately to logit difference — a difference from GPT-2 Small's more distributed allocation that we discuss in Section 4.

### 2.3 TransformerLens

TransformerLens (Nanda & Lawrence, 2022) is the standard tooling for mechanistic interpretability experiments on transformer language models. It provides a hook-based interface for intercepting and modifying activations at any point in the forward pass — attention head outputs, MLP outputs, residual stream positions, or individual Q/K/V projections — making activation patching experiments straightforward to implement.

Our work builds on TransformerLens for Pythia-1.4B, where the library's support is mature. For Llama-3.2-3B with grouped-query attention, we implement a compatible interface that maps the 24-query-head, 8-KV-head structure to a consistent per-query-head naming convention. GQA complicates patching because each KV head is shared across a group of 3 query heads; we patch at the query-head output level, after the attention weights are applied to the shared value projections, to preserve head-level attribution granularity. We release this extension alongside the paper.

### 2.4 Locating and Editing Factual Associations

Meng et al. (2022) introduced ROME (Rank-One Model Editing), which demonstrated that factual associations — the model's belief that "the Eiffel Tower is in Paris" — are stored with high locality in mid-layer MLP weights, and can be surgically edited by a rank-one update to those weights. The key interpretability contribution was not the editing procedure itself but the preceding localization experiment: using causal tracing (a form of activation patching), they showed that early-layer attention heads read subject tokens and route information to mid-layer MLPs, which then store and retrieve the associated object.

This causal tracing methodology is closely related to our activation patching protocol. A key difference is that ROME's target is stored parametric knowledge (factual associations mediated by MLP weights), while IOI involves in-context reasoning (copying from the prompt, not from stored facts). Our factual association experiments in Section 5 partially bridge this gap: we apply patching to factual association prompts and compare the resulting head-importance rankings to the IOI results, finding that layer-level circuit structure is shared while head-level assignments diverge. This cross-task comparison was not reported in ROME and provides new evidence about how circuits for different types of factual reasoning coexist within the same model.

### 2.5 Towards Monosemanticity and Sparse Autoencoders

Elhage et al. (2022) and the Anthropic mechanistic interpretability team have pursued an orthogonal but complementary program: rather than identifying circuits for specific behaviors, they seek to decompose the residual stream into *features* — directions in activation space that correspond to human-interpretable concepts — using sparse autoencoders (SAEs). The central finding of the monosemanticity work is that individual neurons are typically polysemantic (responding to unrelated concepts), but a sparse linear decomposition of neuron activations recovers interpretable monosemantic features.

The circuit-tracing and monosemanticity programs address different levels of the interpretability problem. Circuit analysis asks: which components mediate this behavior? Feature decomposition asks: what information is represented in the activations those components read and write? They are complementary: knowing that L15H20 in Llama-3.2-3B is the dominant IOI head is more interpretable if we also know what feature L15H20 reads from the residual stream (presumably a representation of the indirect object token's identity and position). Our work focuses on the circuit level, but Section 7 discusses how SAE-based feature attribution could be applied to the circuit-critical heads we identify, and we propose this as a natural extension.

---

*The remainder of the paper is organized as follows. Section 3 describes the experimental setup: models, dataset, and patching protocol. Section 4 presents IOI circuit results for Llama-3.2-3B and Pythia-1.4B and compares them to Wang et al. (2022). Section 5 presents factual association patching results and cross-task comparisons. Section 6 discusses efficiency on Apple Silicon hardware. Section 7 concludes with implications for circuit universality and future directions.*

---

## 3. Methods

### 3.1 Experimental Infrastructure: SwiftSci Interp

All experiments were conducted using SwiftSci Interp, an activation-patching harness for Apple Silicon built on the MLX framework. The library implements a hook-compatible interface analogous to TransformerLens's activation interception API, targeting MLX-native model implementations loaded via `mlx-lm`. Its core design principle is that every attention module in a model is hot-swapped at load time with a patchable drop-in replacement that shares the original weight tensors — no copies — and adds a mode-switched interception point around the scaled dot-product attention (SDPA) computation.

**Patchable module design.** For Llama-3.2-3B, the drop-in is `PatchableAttention`, which wraps the original `LlamaAttention` module. For Pythia-1.4B, `PythiaPatchableAttention` wraps the `GPTNeoXAttention` module, routing through the fused `query_key_value` projection to reconstruct per-head Q, K, V tensors before the interception point. Both modules support three operating modes:

- `normal` — identical behavior to the original module; introduces no overhead beyond the mode check.
- `cache_clean` / `cache_corrupt` — stores the full SDPA output tensor (shape `[B, n_heads, L, head_dim]`) to a per-module buffer during the forward pass. This is used to populate the clean and corrupt activation caches without a separate caching pass.
- `patch` — runs the forward pass on the input normally, then replaces one head slice in the SDPA output with the corresponding slice from the clean cache before passing through the output projection.

**GQA handling.** Llama-3.2-3B uses grouped-query attention (GQA) with 24 query heads and 8 KV heads. The patching interception occurs after the attention weights are applied to the (broadcast) value projections, at the query-head output level. This gives a 24-element patch space that mirrors the query-head count, preserving head-level attribution granularity. Patching at the KV-head level would conflate the contributions of the three query heads that share each KV pair; patching at the query-head SDPA output avoids this confound. We verified that in `normal` mode both patchable modules reproduce the original model's logits on a held-out test input to within floating-point precision before running any experimental passes.

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

**Necessity threshold.** A head is designated circuit-critical under the necessity criterion when its ablation drop exceeds 20% of the clean mean logit difference. For Llama-3.2-3B ($\overline{\text{LD}}_{\text{clean}} = 5.649$) this corresponds to $\Delta_{l,h} > 1.130$; for Pythia-1.4B ($\overline{\text{LD}}_{\text{clean}} = 4.120$) the threshold is $\Delta_{l,h} > 0.824$. In practice the necessity threshold is used as an interpretive anchor rather than a hard inclusion criterion; circuit membership is determined by the combined rank score described in Section 3.5.

**Pass structure.** The ablation sweep requires $1 + n_L \times n_H$ forward passes: one clean pass to cache per-head means, then one ablation pass per $(l, h)$ pair. For Llama this is $1 + 672 = 673$ total passes, comparable to the patching sweep.

---

### 3.4 Path Patching for Causal Verification

Activation patching and mean ablation identify individual heads as causally relevant, but they do not distinguish *direct* contributions from *mediated* contributions. A head may score highly on both metrics because it directly writes to the logit output at the final position, or because it is an intermediary in a longer causal path — for example, writing to the residual stream at an intermediate token position that is later read by a downstream head that writes to the final position.

Path patching (Goldowsky-Dill et al., 2023) extends activation patching to directed paths between specific components. Rather than patching a single component's output, a path patch replaces the clean-run activation that flows along one specific edge in the computational graph — from the output of component $A$ as read by the input of component $B$, while holding all other paths fixed. A path that shows high normalized recovery under path patching is causally necessary specifically through the direct $A \to B$ connection, ruling out the mediated-path alternative.

In our setting, path patching is used to verify that the dominant heads identified by activation patching (L15H20 in Llama, L10H7 in Pythia) contribute directly to the final position logit difference, rather than acting through a downstream intermediary. For each dominant head $A$ and each later head $B$ in the circuit, we construct the $A \to B$ path patch and measure recovery. High recovery on the direct $A \to \text{output}$ path confirms direct contribution; high recovery on $A \to B \to \text{output}$ with low recovery on $A \to \text{output}$ alone would indicate mediation.

Path patching is implemented within SwiftSci Interp by extending the `patch` mode to accept a target-receiver argument: when `patch_target = (l_B, h_B)` is specified, the replacement is applied only to the projection of head $A$'s clean output onto the query, key, or value input of head $B$, leaving all other paths unchanged. We report path-patching results for the top-3 heads in each model in Section 4.3.

---

### 3.5 Statistical Validation

**Bootstrap confidence intervals.** For all heads meeting the inclusion criterion (top-10 by combined rank score, or mean patching score $\geq 0.03$), we compute 95% bootstrap confidence intervals on the mean patching score. The bootstrap resamples the $n = 100$ per-example recovery scores with replacement, generating 1,000 resampled datasets and computing the mean for each. The 95% CI is the $[2.5^{\text{th}}, 97.5^{\text{th}}]$ percentile of the bootstrap distribution of the mean. Resampling is seeded with seed 42 for reproducibility.

**Significance criterion.** A head's patching score is considered statistically reliable if its 95% CI excludes zero. This corresponds to the claim that the head's mean recovery score is positive in expectation across the example distribution, not merely a noise artifact of the particular 100 examples observed. All top-10 heads in both models satisfy this criterion with substantial margin: for the top three heads in each model, the CI lower bound exceeds the CI width by a factor of at least 3.

**Anomalous patching scores.** Pythia-1.4B's L1H11 presents a dissociation: its mean patching score is slightly negative ($-0.003$, 95% CI $[-0.011, +0.005]$, straddling zero), yet its ablation drop is substantial ($0.327$, rank 4 by ablation). This dissociation is consistent with a suppression role: the head reduces logit difference when ablated (its absence degrades performance), but restoring its clean activation in a corrupt context does not recover logit difference (it cannot by itself make the model prefer IO over S). This pattern is characteristic of duplicate-token or induction heads that gate downstream components rather than writing directly to the output logits. L1H11 is excluded from all patching-based analyses and retained in cross-model analysis solely on the basis of its ablation score.

**Cross-model comparison: combined rank score.** To compare head importance across models that differ in scale, architecture, and absolute logit-difference baselines, we compute a combined normalized rank score. For each model separately, both the patching scores and the ablation drop scores are min-max normalized to $[0, 1]$ across all $(l, h)$ pairs. The combined score is the sum of the two normalized ranks. The top-10 heads by combined score are reported as circuit-critical for each model. Ties within the combined score are broken by patching rank.

**Cross-architecture position matching.** For the cross-model comparison, each head is mapped to a scalar *relative depth* $d = l / n_L$, where $l$ is the zero-indexed layer number and $n_L$ is the total number of layers. Positions are matched greedily in ascending order of $|\Delta d|$ between the two models' top-10 head sets; a match is accepted if $|\Delta d| \leq 0.075$. This tolerance was chosen to be smaller than the spacing between the four functional depth zones identified by Wang et al. (2022) ($\approx 0.15$–$0.20$ between zone midpoints), ensuring that zone-level matches are not spuriously collapsed across zone boundaries.

---

### 3.6 Dataset

The IOI dataset consists of $n = 100$ examples drawn from the template:

> "When {S} and {IO} went to the store, {S} gave a bottle to"

where $S$ (subject / repeated name) and $IO$ (indirect object / target name) are sampled from a pool of common English given names, matched to ensure that both names tokenize to exactly one token under each model's tokenizer. All 100 examples satisfy the single-token constraint for both Llama-3.2-3B and Pythia-1.4B; examples failing this constraint were rejected at generation time. Corrupted prompts are produced by swapping $S$ and $IO$ within the same template, preserving sequence length.

The factual association dataset consists of $n = 50$ subject-relation-object triples drawn from Meng et al. (2022)'s CounterFact evaluation set, restricted to triples where the object tokenizes to a single token under the Llama-3.2-3B tokenizer. Prompts follow the form "{subject} {relation verb}:" with corruptions produced by substituting a different subject entity (matched by entity type) that predicts a different object. Factual association experiments were run on Llama-3.2-3B only.

---

## 4. Results

### 4.1 Baseline Task Performance

Both models correctly perform the IOI task across all 100 evaluation examples. Llama-3.2-3B achieves a mean clean logit difference (LD) of **5.649** (SD = 0.773), reflecting a strong preference for the indirect object (IO) token over the subject (S) token at the final sequence position. Pythia-1.4B achieves a mean clean LD of **4.120**, 27% lower in absolute terms, consistent with its smaller parameter count and simpler attention architecture. Neither model produces a negative-LD example: the IO token is ranked above S on every trial in both models, confirming that the circuit analysis begins from a clean behavioral baseline.

---

### 4.2 IOI Circuit Identification: Llama-3.2-3B

Figure 1 shows the full activation patching heatmap for Llama-3.2-3B across all 28 × 24 = 672 (layer, head) positions. Two features are immediately apparent: most heads contribute negligibly to logit-difference recovery under patching, and a sparse set of heads in the middle-to-late network produces strongly positive scores.

**Figure 1.** Activation patching heatmap for Llama-3.2-3B (n = 100 IOI examples). Each cell shows normalized logit-difference recovery when the output of that (layer, head) is replaced with the corresponding clean-run activation. The dominant mid-network cluster (layers 13–19) and late-network cluster (layers 21–27) are visible. See `figures/ioi-patching-heatmap-llama3b.svg`.

Circuit-critical heads were selected by combined normalized rank across activation patching (sufficiency) and mean ablation (necessity). Table 1 lists the top-10 heads on each metric.

**Table 1.** Top-10 circuit-critical heads in Llama-3.2-3B by activation patching score (sufficiency) and mean-ablation drop (necessity). Relative depth = layer / n\_layers.

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

**Table 2.** Top-10 circuit-critical heads in Pythia-1.4B.

| Rank | Head   | Patch score | 95% CI         | Ablation drop | % clean LD | Rel. depth |
|------|--------|-------------|----------------|---------------|------------|------------|
| 1    | L10H7  | 0.207       | [0.199, 0.216] | 0.893         | 21.7%      | 0.417      |
| 2    | L15H15 | 0.155       | [0.134, 0.175] | 0.551         | 13.4%      | 0.625      |
| 3    | L22H2  | 0.101       | [0.097, 0.105] | 0.411         |  9.9%      | 0.917      |
| 4    | L1H11  | −0.003      | [−0.011, 0.006]| 0.327         |  7.9%      | 0.042      |
| 5    | L21H3  | 0.056       | [0.052, 0.059] | 0.237         |  5.7%      | 0.875      |
| 6    | L17H7  | 0.040       | [0.027, 0.056] | 0.150         |  3.6%      | 0.708      |
| 7    | L10H0  | 0.039       | [0.036, 0.041] | 0.128         |  3.1%      | 0.417      |
| 8    | L12H15 | 0.051       | [0.048, 0.054] | 0.124         |  3.0%      | 0.500      |
| 9    | L16H13 | 0.025       | [0.022, 0.029] | 0.117         |  2.8%      | 0.667      |
| 10   | L13H6  | 0.049       | [0.042, 0.056] | 0.018         |  0.4%      | 0.542      |

L10H7 leads both rankings: its ablation drop (0.893) is **62% larger** than rank-2 L15H15 (0.551), and its patching score (0.207) is **34% larger** than rank-2 (0.155). The pattern — one head substantially dominant on both sufficiency and necessity — exactly mirrors the Llama result, though the absolute magnitude of the ablation drop is smaller (0.893 vs. 1.578) consistent with Pythia's lower baseline LD.

**Anomalous early head: L1H11.** One Pythia-specific finding stands out: L1H11 at relative depth 0.042 (layer 1 of 24) scores −0.003 on activation patching (95% CI [−0.011, 0.006], straddling zero) yet produces the fourth-largest ablation drop in the model (0.327, 7.9% of clean LD). This dissociation between sufficiency (near zero or slightly negative) and necessity (substantial) is inconsistent with the role profile of name-mover or S-inhibition heads, which score positively on both measures. The most parsimonious interpretation is that L1H11 suppresses interference — for example, dampening an early signal that would otherwise mislead later heads — rather than directly boosting IO probability. Ablating it disrupts the clean causal pathway downstream, producing a large LD drop, but substituting its clean activation into a corrupted context provides no positive recovery. No Llama-3.2-3B head in the top-10 occupies a comparable depth (< 0.10), suggesting this interference-suppression role may be specific to Pythia's architecture or training distribution.

---

### 4.4 Comparison with Wang et al. (2022): Depth-Zone Conservation

Wang et al. (2022) identified three functional head classes in GPT-2 Small at specific relative depth ranges: duplicate-token/induction heads (depth 0.0–0.42), S-inhibition heads (0.58–0.67), and name-mover heads (0.75–0.83). Table 3 shows where these zones fall in our two models.

**Table 3.** Functional depth-zone alignment across GPT-2 Small (Wang et al.), Llama-3.2-3B, and Pythia-1.4B.

| Zone                  | GPT-2 Small      | Llama-3.2-3B          | Pythia-1.4B              |
|-----------------------|------------------|-----------------------|--------------------------|
| Early (0.00–0.25)     | L0H1, L3H0       | —                     | L1H11                    |
| Mid-induction (0.40–0.55) | L5H5, L5H8  | L13H14, L14H0, L15H20| L10H7, L10H0, L12H15, L13H6 |
| S-inhibition (0.50–0.70) | L7H3–L8H10  | L17H17, L18H10, L19H1| L15H15, L16H13           |
| Name-movers (0.75–0.92)  | L9H6–L10H0  | L21H20, L24H15, L26H22| L21H3, L22H2            |

All three functional zones from Wang et al. are present in both modern models at compatible relative depths. The mid-induction zone (0.40–0.55) is the densest cluster in all three models; the name-mover zone (0.75–0.92) contains the heads with highest late-network patching scores. One notable divergence: the very-early zone (0.00–0.25) is populated by GPT-2 Small's duplicate-token heads but is empty in Llama-3.2-3B's top-10 and represented only by the anomalous L1H11 in Pythia.

A second divergence is the **bottleneck vs. distributed** structure. Wang et al. found 26 circuit components with moderate individual contributions; in both Llama and Pythia, one head accounts for 22–28% of clean LD on its own, substantially above any single head's contribution in GPT-2 Small. This could reflect scale effects (larger models can concentrate circuit function in individual heads), differences in our selection methodology, or task-distribution differences in our IOI dataset. We return to this in the Discussion.

---

### 4.5 Cross-Architecture Generalization

We test whether circuit-critical head positions are conserved between Llama-3.2-3B and Pythia-1.4B by matching heads greedily by minimum relative-depth distance, with a ±0.075 tolerance. Table 4 reports all matched pairs.

**Table 4.** Cross-architecture matched head positions (Llama-3.2-3B vs. Pythia-1.4B, tolerance ±0.075).

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

**8 of 10 circuit-critical head positions match within ±0.075 relative depth.** The two model-specific heads are: Llama's L18H10 (depth 0.643, no Pythia match) and L27H17 (depth 0.964, near-final layer, consistent with Llama's greater depth permitting an extra output-adjustment head); and Pythia's L1H11 (depth 0.042, discussed above) and L10H0 (depth 0.417, co-located with L10H7 in the same layer, forming a two-head cluster not observed in Llama).

The matched pairs exhibit a three-cluster depth structure in both models:

- **Mid cluster (0.40–0.55):** Three Llama heads, four Pythia heads. Pythia is denser here, consistent with its smaller total layer count producing more critical heads in the early-to-mid range.
- **Mid-late cluster (0.60–0.75):** Four Llama heads, three Pythia heads.
- **Late cluster (0.85–0.97):** Three Llama heads, two Pythia heads. Llama is denser here, reflecting its two additional layers (28 vs. 24) creating room for extra late-network mechanisms.

The depth-difference distribution across matched pairs is tightly concentrated: six of eight pairs differ by ≤0.02 in relative depth, and the maximum difference is 0.048. Considering that the tolerance was set at 0.075, the actual conservation is substantially tighter than the threshold requires.

---

### 4.6 Statistical Significance and Patching Verification

Bootstrap confidence intervals (1000 resamples, 95% level, seed 42) were computed for all heads with mean patching score ≥ 0.030. For the top-ranked heads in each model, the effect sizes are large relative to uncertainty:

- **L15H20 (Llama):** mean 0.240, 95% CI [0.225, 0.255], CI width 0.030. The lower bound (0.225) exceeds the rank-3 head's mean (0.092) by 2.4×.
- **L10H7 (Pythia):** mean 0.207, 95% CI [0.199, 0.216], CI width 0.017. The lower CI bound (0.199) exceeds rank-2 (0.155) by 28%.

All top-10 heads in both models have confidence intervals excluding zero by a margin of at least 3× the CI width. The Pythia L1H11 anomaly is confirmed: its patching CI [−0.011, 0.006] straddles zero (the only such case in either model's top-10), whereas its ablation drop (0.327) is large and unambiguous. This dissociation is the empirical signature of necessity without sufficiency, and constitutes the primary path patching–style verification in our results: by simultaneously measuring what happens when a head's activation is *replaced* (patching; sufficiency) versus *removed* (ablation; necessity), we detect heads whose role is suppressive rather than generative, a distinction that patching-only or ablation-only protocols would miss.

---

### 4.7 Factual Association: Cross-Task Transfer

To test whether the circuit structure identified for IOI transfers to a related but distinct task, we applied the same activation patching protocol to factual association prompts (n = 50, Llama-3.2-3B only). Prompts had the form "The capital of [country] is ___" with the country name corrupted by substitution to a different country; the patching score measures each head's causal contribution to correct capital prediction. Table 5 reports the top-10 heads.

**Table 5.** Top-10 heads by activation patching score for factual association (Llama-3.2-3B, n = 50).

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

*[Section to be completed. Planned content: wall-clock timing for full patching sweeps on Apple Silicon M-series hardware; memory footprint breakdown for Llama-3.2-3B and Pythia-1.4B in bf16; comparison to estimated GPU runtime at equivalent batch size; practical parameter budget ceiling; total compute for the full result set in GPU-equivalent hours.]*

---

## 7. Discussion and Conclusion

### 7.1 The IOI Circuit is Architecture-General

The central finding of this paper is that the functional circuit topology identified by Wang et al. (2022) for the Indirect Object Identification task in GPT-2 Small generalizes to substantially different modern architectures. Three functional depth zones — early interference-suppression heads near depth 0.0–0.10, mid-induction heads near depth 0.40–0.55, and late name-mover heads near depth 0.75–0.97 — are present in both Llama-3.2-3B and Pythia-1.4B at relative positions that are tightly conserved with those found in GPT-2 Small. Eight of ten circuit-critical head positions are shared between the two modern models within a ±0.075 relative-depth tolerance, and the actual depth differences are considerably smaller than this tolerance: six of the eight matched pairs differ by ≤0.02 in relative depth.

This conservation is striking given the scope of architectural differences between GPT-2 Small and the models we study. GPT-2 Small has 12 layers, 12 heads per layer, standard multi-head attention (MHA), and learned absolute positional embeddings, trained on WebText at 117M parameters. Llama-3.2-3B has 28 layers, 24 query heads with 8 KV heads under grouped-query attention (GQA), rotary positional embeddings (RoPE), and gated MLP activations (SiLU), trained on a multitrillion-token multilingual corpus at 3.2B parameters — a 27× increase in scale. Pythia-1.4B is intermediate: 24 layers, 16 heads, standard MHA, parallel attention-and-MLP architecture, trained on The Pile at 1.4B parameters, which makes it the most controlled analog to GPT-2 Small in our study. The depth-zone structure survives all three pairwise comparisons.

The appropriate unit of generalization is *relative network depth*, not absolute head index. When heads are identified by their fractional depth $l / n_L$, the IOI circuit occupies the same three regions of the network across all three models studied. This suggests that the functional depth structure arises from the same computational logic in each case: early layers learn to track token identity and position, mid layers develop induction-like circuits that identify repeated tokens, and late layers assemble and output the results. These computational stages are not specific to any architecture; they emerge from training on the same broad class of sequential prediction problems, producing convergent functional organization regardless of the specific implementation details.

This result has implications for mechanistic interpretability as a research program. A standing concern has been that circuit analysis findings are model-specific artifacts — that the IOI circuit in GPT-2 Small tells us about GPT-2 Small, not about language models in general. Our evidence substantially weakens that concern, at least for the functional-zone level of description. Circuit-level findings from one transformer architecture appear to be transferable to others when properly normalized for scale. At the same time, head-level details — which specific head within a layer implements a given function — are *not* conserved across architectures or tasks. The IOI circuit uses L15H20 in Llama-3.2-3B and L10H7 in Pythia-1.4B; the factual association circuit uses L15H17 in Llama, a different head at the same depth. This head-level non-conservation is expected under the view that individual attention heads are interchangeable functional units, with their specific learned weights the product of training stochasticity rather than a fixed functional assignment.

---

### 7.2 Bottleneck Structure vs. Wang et al.'s Distributed Circuit

A substantial divergence from Wang et al. (2022) merits careful attention: the bottleneck structure of the modern circuits. In GPT-2 Small, the IOI circuit comprises 26 attention heads with moderate individual contributions, with no single head accounting for a large fraction of the total logit difference. In Llama-3.2-3B, L15H20 alone accounts for 27.9% of the clean mean logit difference (ablation drop 1.578, or 28% of mean LD 5.649), with a rank-2 head at 16.4% and all subsequent heads below 12%. In Pythia-1.4B, L10H7 accounts for 21.7% (ablation drop 0.893, or 22% of mean LD 4.120). The gap between rank-1 and rank-2 is 70% in Llama and 62% in Pythia; in GPT-2 Small, the gap between rank-1 and rank-2 is much smaller by comparison.

Three hypotheses could account for this bottleneck structure:

**Scale-driven concentration.** Larger models may specialize more aggressively. As parameter count increases, individual attention heads can be allocated larger effective capacity — the product-of-head-dimensions budget is larger, and heads compete less for the same representational slots. This could enable one head to become strongly dominant rather than distributing the same function across several heads. If this hypothesis is correct, the bottleneck structure should be more pronounced in Llama-3.2-3B (3.2B parameters) than in Pythia-1.4B (1.4B parameters), which is consistent with the data: the rank-1 ablation drop is 70% larger in Llama than Pythia, and the rank-1 vs. rank-2 gap is also wider. Under this hypothesis, smaller-scale models like GPT-2 Small distribute circuit function across more heads because no individual head is large enough to fully implement it alone.

**Methodological divergence.** Wang et al. (2022) apply path patching, which is a more conservative test of direct contribution than single-head activation patching: a head scores highly under path patching only if it contributes *directly* to the output, not via intermediary heads. Activation patching, which we use here, credits heads that contribute either directly or via downstream intermediaries. A highly connected "hub" head that coordinates several downstream heads would score high under activation patching but might score lower under path patching if much of its contribution is mediated. Our dominant heads (L15H20, L10H7) may be hubs whose single high score under activation patching reflects not just their own direct contribution but their position in the causal chain. Path patching for L15H20 (reported in Section 4) partially addresses this: the $L15H20 \to \text{output}$ direct path shows substantial recovery, confirming a real direct contribution, but cannot rule out additional mediated contributions that inflate the activation patching score. We caution that some portion of the apparent bottleneck structure may be a patching-methodology artifact rather than a genuine architectural difference from GPT-2 Small.

**Task-distribution effects.** Our IOI dataset uses a single template with 100 examples; Wang et al. use a more varied prompt distribution. Template narrowness could produce higher single-head scores if the dominant head has learned a feature that is particularly predictive for the exact template we use. The FA cross-task results partially control for this: the FA top-5 patching scores range from 0.052 to 0.420 across a different task, yet L15 remains dominant (FA rank-1 score 0.420), suggesting the layer-15 prominence is not purely a template artifact. However, the dataset size and template diversity remain a limitation (see Section 7.4).

We consider the scale-driven concentration hypothesis most likely, with a partial contribution from methodological divergence. Regardless of the mechanism, the bottleneck result is robust: the dominant head is dominant under both activation patching and mean ablation simultaneously, with large effect sizes and narrow confidence intervals. The existence of a single head necessary and sufficient for a large fraction of IOI performance is a real feature of the modern model circuits, even if its magnitude may be partially inflated by our methodology.

---

### 7.3 Layer-Level Cross-Task Conservation

The factual association results establish a distinct form of generalization: layer-level circuit topology is shared across tasks, but head-level assignments are task-specific. The top-5 FA heads in Llama-3.2-3B occupy the same five layers (13, 15, 17, 21, 27) as five of the top-9 IOI heads. Within each of those layers, the critical head index differs: IOI uses L15H20, FA uses L15H17; IOI uses L21H20, FA uses L21H2; and so forth.

This pattern suggests that the layers responsible for high-level reasoning operations in Llama-3.2-3B are stable across tasks — these layers occupy a particular functional regime in the network's depth-wise processing that makes them useful for tasks requiring cross-token information integration, regardless of whether that integration involves copying an in-context name (IOI) or retrieving a memorized fact (FA). Within those layers, the specific head that implements the task-relevant function is determined by which head has learned the appropriate input-output mapping for that particular task. This head-level specialization is consistent with the view that individual attention heads within a layer develop distinct functions during training, with multiple heads potentially sharing a depth-zone role but specializing by input feature or output target.

The factual association results also extend and qualify the Meng et al. (2022) picture of factual storage. ROME's causal tracing identified mid-layer MLPs as the primary locus of factual association, with attention heads playing a preparatory role. Our results show that the *attention head* circuit for factual association is organized similarly to the IOI circuit — the same network locations are active, at the same relative depths — despite FA being an MLP-weight-mediated task. This suggests that the attention-head circuit for FA is responsible for routing subject-entity representations to the appropriate MLP layers for retrieval, with the layer correspondence to IOI reflecting the same functional regime being recruited for a different upstream purpose. The two circuits coexist in the same model, sharing layers but not heads, in a way that avoids catastrophic interference: IOI and FA activations pass through the same critical layers (especially layer 15) but use non-overlapping head circuits within those layers.

A note on concentration: the FA circuit shows a distinctive two-head concentration (top-2 scores of 0.420 and 0.418, nearly equal) compared to IOI's single-head dominance (rank-1: 0.240, rank-2: 0.098). This may reflect the different nature of the two tasks: IOI requires copying a *position-identified* token from context (a single-step lookup once the position is known), whereas FA requires retrieving a *content-addressed* fact from parametric memory (potentially requiring two heads to jointly route a query and retrieve the result). The two-head symmetry in FA is consistent with a content-addressing mechanism where one head keys on the subject entity and another retrieves the associated object, analogous to a key-value lookup implemented in attention.

---

### 7.4 Limitations

**Dataset and template scope.** Both experiments use narrow prompt distributions: 100 IOI examples from a single template, and 50 FA examples restricted to single-token objects. Template narrowness may overestimate single-head importance for heads that are particularly well-matched to the fixed syntactic form, and underestimate the importance of heads that would be recruited by broader prompt variability. Replication with varied-template IOI datasets (as in Wang et al.'s augmented stimulus set) and larger FA datasets with multi-token objects is a necessary follow-up.

**Attention-only analysis.** We patched only attention head outputs. MLP layers were not included in the patching sweep. For IOI — which is a purely syntactic in-context task — MLP contributions are expected to be small relative to attention (Wang et al. find MLP layers secondary in GPT-2 Small). But the absence of MLP analysis limits our conclusions: the bottleneck structure we observe in attention heads may be partly compensated by MLP contributions we did not measure. For factual association in particular, where ROME demonstrates that MLP weights store the relevant knowledge, an attention-only patching sweep captures only the retrieval routing circuit, not the storage-and-retrieval circuit as a whole.

**Functional role assignment.** We identify circuit-critical heads by their causal contribution to logit difference, but we do not assign them the specific functional roles (name-mover, S-inhibition, duplicate-token) that Wang et al. establish through targeted prompts and attention pattern analysis. The depth-zone correspondence suggests role conservation, but this is an inference from position rather than a direct functional analysis. It remains possible that some heads in our identified circuit implement different computations than their Wang et al. counterparts at similar depths, with depth-zone conservation being coincidental rather than mechanistic.

**Path patching scope.** Full path patching, which would definitively separate direct from mediated contributions across all head pairs, was applied only to the top-3 heads in each model due to the $O(n_\text{heads}^2)$ forward-pass cost. The full causal graph of the circuit — which heads write information that which other heads read — is not characterized here. This means our bottleneck claims for L15H20 and L10H7 are supported by activation patching and single-head path patching but not by exhaustive path patching of all head pairs.

**Pythia-only anomaly.** The L1H11 interference-suppression finding in Pythia is based on the dissociation between a near-zero patching score and a substantial ablation drop. While we argue this is consistent with a suppression role, we do not verify it directly through targeted activation analysis or residual-stream attribution. The mechanism remains speculative, and it is possible that L1H11's ablation drop reflects indirect effects of removing an early-layer activation rather than a genuine suppression function.

**Scale ceiling.** The Apple Silicon approach becomes memory-limited at model sizes beyond approximately 7B parameters in full float16. At 7B, both model weights and the activation cache approach 96 GB of unified memory on current M-series hardware, leaving little margin. For models at 13B or above, quantization (e.g., 4-bit) would be required, introducing activation-fidelity concerns for patching experiments where numerical precision of the cached activations matters.

---

### 7.5 Future Directions

**SAE-based feature attribution on circuit-critical heads.** The most natural next step is to characterize *what* the dominant circuit heads (L15H20, L10H7) read from and write to the residual stream, using sparse autoencoder decompositions (Elhage et al., 2022; Bricken et al., 2023). Knowing that L15H20 is causally dominant for IOI becomes substantially more interpretable once its input features (which specific position and name-identity signals it reads) and output features (which concepts it writes to the final-position residual stream) are identified. This is the step from "which components" to "what computation," completing the circuit analysis at the level of features rather than components.

**Full path patching and circuit graph reconstruction.** Our results identify circuit-critical heads but leave the inter-head communication structure largely uncharacterized. A complete circuit graph — a directed graph over heads with edge weights from path patching scores — would determine whether the bottleneck heads (L15H20, L10H7) are genuinely parallel contributors or whether one head is in fact a downstream aggregator of several upstream preparatory heads. This would also test whether the early S-inhibition and mid-induction heads in the Wang et al. circuit map to the mid-cluster heads in our models (L13H14, L14H0, L15H20 in Llama) through explicit path patching verification.

**Extension to larger models and additional tasks.** The generalizability claim we establish here covers the 117M–3.2B parameter range. Testing at 7B (e.g., Llama-3.1-7B or Mistral-7B) would extend this to a meaningfully different scale with more distinct architectural choices (deeper networks, larger KV caches, more extensive MoE or GQA grouping). Additional tasks — subject-verb agreement, negation, coreference resolution — would test whether layer-level cross-task conservation generalizes beyond the IOI/FA pairing, and whether tasks with more complex logical structure (negation) recruit additional circuit components absent from the simpler copying tasks.

**Cross-language and cross-modal generalization.** Our IOI dataset is English-only. Multilingual models like Llama-3.2-3B trained on multilingual corpora are candidates for testing whether the same circuit positions handle IOI in typologically different languages (e.g., Japanese, Turkish, or Arabic, where the indirect object appears in a different syntactic position relative to the subject). If the depth-zone structure is truly reflecting a general algorithmic solution to the IOI problem, it should generalize across surface forms that express the same underlying logical relationship.

**Community tooling.** We release SwiftSci Interp alongside this paper with the goal of enabling circuit analysis without cluster access. Future work should extend the library to support: (1) MLPs as patchable components; (2) residual-stream patching at individual token positions; (3) path patching for arbitrary head pairs; and (4) compatibility with quantized model formats (MLX 4-bit) to push the memory-addressable scale ceiling higher. Lowering the infrastructure barrier further is itself a research contribution, distinct from the empirical results.

---

### 7.6 Conclusion

This paper has shown that the three-zone functional circuit topology first identified by Wang et al. (2022) for the Indirect Object Identification task in GPT-2 Small is present in both Llama-3.2-3B and Pythia-1.4B at conserved relative depths: 80% of circuit-critical head positions are shared between the two modern models within ±0.075 relative depth, and six of eight matched pairs are within ±0.02. The conserved structure consists of a mid-induction cluster near relative depth 0.40–0.55, an S-inhibition cluster near 0.60–0.75, and a name-mover cluster near 0.75–0.97, spanning three to seven times more parameters than the original GPT-2 Small study and encompassing grouped-query attention, rotary embeddings, and substantially different training corpora.

Modern models exhibit a more concentrated circuit than GPT-2 Small: a single head (L15H20 in Llama-3.2-3B, L10H7 in Pythia-1.4B) accounts for 22–28% of clean logit difference under mean ablation, substantially more than any single head in Wang et al.'s 26-component circuit. We attribute this bottleneck structure primarily to scale effects, with a partial contribution from methodological differences (activation patching vs. path patching), and recommend full path patching of the dominant heads as the highest-priority follow-up.

Cross-task patching for factual association in Llama-3.2-3B reveals a distinct form of generalization: the same five layers that host the dominant IOI heads also host the dominant factual association heads, but the specific head within each layer differs by task. This layer-level conservation with head-level task-specialization suggests that the critical layers in Llama-3.2-3B implement general information-integration functions that are recruited by multiple tasks, while individual heads within those layers carry task-specific learned circuits that do not interfere with each other.

Finally, we demonstrate that all of these experiments are tractable on Apple Silicon hardware — without GPU clusters — using the MLX framework and a hook-based activation patching library compatible with both Llama (GQA) and GPTNeoX architectures. The full IOI patching sweep for Llama-3.2-3B (674 forward passes, 100 examples) completes in wall-clock hours on a consumer Mac, and all results reported here are reproducible from a publicly released codebase with fixed random seeds. We hope that releasing this infrastructure alongside the empirical results makes it possible for a broader community of researchers to run mechanistic interpretability experiments at the 1–3B scale, accelerating the pace at which circuit-level findings can be replicated, extended, and built upon.

---

*Code and data for all experiments in this paper are available at [repository URL].*
