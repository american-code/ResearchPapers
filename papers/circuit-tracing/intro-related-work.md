# Introduction and Related Work
## Circuit Tracing at Scale: IOI Circuit Generalization to Modern Open-Weight Models

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

The circuit-tracing and monosemanticity programs address different levels of the interpretability problem. Circuit analysis asks: which components mediate this behavior? Feature decomposition asks: what information is represented in the activations those components read and write? They are complementary: knowing that L15H20 in Llama-3.2-3B is the dominant IOI head is more interpretable if we also know what feature L15H20 reads from the residual stream (presumably a representation of the indirect object token's identity and position). Our work focuses on the circuit level, but Section 6 discusses how SAE-based feature attribution could be applied to the circuit-critical heads we identify, and we propose this as a natural extension.

---

*The remainder of the paper is organized as follows. Section 3 describes the experimental setup: models, dataset, and patching protocol. Section 4 presents IOI circuit results for Llama-3.2-3B and Pythia-1.4B and compares them to Wang et al. (2022). Section 5 presents factual association patching results and cross-task comparisons. Section 6 discusses implications for circuit universality and the Apple Silicon tooling pipeline. Section 7 concludes.*
