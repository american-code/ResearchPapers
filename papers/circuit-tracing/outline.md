# Efficient Mechanistic Circuit Analysis of Open-Weight LLMs on Apple Silicon

## Abstract

Mechanistic interpretability research has largely depended on GPU clusters, limiting reproducibility and iteration speed for many researchers. We demonstrate that full activation-patching and mean-ablation circuit analyses of multi-billion-parameter transformer models are tractable on Apple Silicon using the MLX framework, with no specialized hardware beyond a modern Mac. Applying this infrastructure to the Indirect Object Identification (IOI) task on Llama-3.2-3B and Pythia-1.4B, we find that the three-zone functional circuit topology identified by Wang et al. (2022) in GPT-2 Small generalizes across both models: 80% of circuit-critical head positions are conserved by relative depth (±0.075) despite differing layer counts, attention mechanisms, and training corpora. A single dominant head in each model (L15H20, L10H7) accounts for 22–28% of clean logit difference on its own. Layer-level circuit topology further transfers from IOI to factual association, with divergent head indices at shared layer positions. These results establish that IOI circuit organization is architecture-general, and that Apple Silicon is a viable substrate for reproducible mech-interp research at the 1–3B parameter scale.

---

## Section Outline

### 1. Introduction

- Motivation: circuit analysis is computationally gated; GPU access is a barrier to reproducibility and rapid iteration in mech-interp research.
- Paper scope: demonstrate end-to-end circuit tracing (activation patching + mean ablation) at 1–3B scale on Apple Silicon, and use it to test cross-architecture generalization of the IOI circuit.
- Contributions listed: (1) MLX-based patching harness, (2) IOI circuit results for Llama-3.2-3B and Pythia-1.4B, (3) cross-model and cross-task generalization findings.

### 2. Background

- IOI task and Wang et al. (2022) circuit: functional head classes (name movers, S-inhibition, duplicate-token/induction), logit difference metric, and the three-zone depth structure in GPT-2 Small.
- Activation patching and mean ablation: formal definitions of normalized recovery score and necessity criterion; relationship to path patching (Goldowsky-Dill et al., 2023).
- Apple Silicon and MLX: unified memory architecture, MLX lazy evaluation model, and how both properties interact with the repeated forward-pass structure of patching sweeps (674 passes for Llama-3.2-3B).

### 3. Methods

- Patching protocol: clean/corrupt pass pair per example (IO/S name swap), per-(layer, head) patch substitution via `PatchableAttention`, normalized recovery aggregated over n=100 IOI examples with 1000-resample bootstrap CIs.
- Mean ablation protocol: component outputs replaced by mean over reference (name-swapped) distribution; necessity threshold set at >20% clean logit difference drop.
- Cross-model comparison: relative depth normalization (`layer / n_layers`), greedy position matching at ±0.075 tolerance, combined-score head selection (min-max normalized patching + ablation ranks summed).

### 4. Results: IOI Circuit Identification

- Llama-3.2-3B circuit: 10 critical heads spanning relative depths 0.46–0.97; L15H20 is dominant with ablation drop 1.578 (27.9% of clean LD), 70% larger than rank-2; bottleneck rather than distributed structure.
- Pythia-1.4B circuit: analogous bottleneck at L10H7 (21.7% of clean LD); anomalous early head L1H11 (depth 0.042) shows necessity without sufficiency — plausible interference-suppression role absent from Llama.
- Replication of Wang et al. depth zones: early (0.00–0.25), mid-induction (0.40–0.55), S-inhibition (0.50–0.70), and name-mover (0.75–0.97) zones all present in both models at preserved relative positions.

### 5. Results: Cross-Architecture and Cross-Task Generalization

- Cross-architecture: 8 of 10 critical head positions match within ±0.075 relative depth across Llama and Pythia; three-cluster structure (mid / mid-late / late) reproduced in both, with density differences consistent with layer count (Pythia denser at mid, Llama denser at late).
- Cross-task transfer to factual association (n=50, Llama-3.2-3B): top-5 factual association heads occupy the same five layers as five of the top-9 IOI heads; head indices diverge within shared layers, consistent with layer-level but not head-level conservation.
- Magnitude and concentration differences: FA top-2 heads score 0.420/0.418 vs. IOI rank-1 at 0.240, suggesting factual association uses a more concentrated circuit; layer 15 is the dominant layer for both tasks.

### 6. Efficiency Analysis

- Wall-clock times on Apple Silicon (M-series): full 28×24 patching sweep for Llama-3.2-3B and 24×16 sweep for Pythia-1.4B; comparison to estimated GPU runtime at equivalent batch size.
- Memory footprint: unified memory allows full bf16 model weights plus activation cache in one address space; practical parameter budget ceiling for this approach.
- Reproducibility: all experiments run with fixed seeds and publicly released MLX harness; estimated total compute for full result set in GPU-equivalent hours.

### 7. Discussion and Conclusion

- IOI circuit topology is architecture-general: functional depth zones survive 3–7× parameter scaling, GQA vs. MHA attention, and different training corpora; specific head indices are not conserved.
- Bottleneck vs. distributed structure: single dominant heads at 22–28% of clean LD may reflect scale effects or task-distribution differences relative to Wang et al.'s 26-component circuit; path patching is the recommended next step to test direct vs. mediated contributions.
- Implications for consumer-hardware mech-interp: Apple Silicon closes the GPU access gap for circuit analysis at 1–3B scale, enabling faster hypothesis iteration and broader community participation; limitations at >7B scale discussed.
