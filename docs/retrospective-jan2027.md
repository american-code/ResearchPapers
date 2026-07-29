# Post-Roadmap Retrospective
**J. Melton — January 2027**

---

## Overview

This document reviews the three-paper interpretability research program initiated in mid-2026 (originally targeting ICLR 2026, later re-scoped to ICLR 2027 / NeurIPS 2026 workshop cycle). It covers what each experiment set out to test, what it found, where results departed from expectations, what reviewers are likely to challenge, and what new directions the findings open up.

The program spans three interconnected studies: mechanistic circuit analysis of IOI and factual-association tasks across modern LLMs; a systematic comparison of SAE training objectives and an unexpected cross-architecture feature universality result; and infrastructure work on distributed inference and training (mlxMesh), culminating in a safety classifier evaluation that failed productively.

---

## I. Experiment Outcomes: Confirmed vs. Surprising

### 1. IOI Circuit Topology Generalizes — Confirmed (with a major structural twist)

**Hypothesis:** The three-zone IOI circuit topology documented by Wang et al. (2022) for GPT-2 Small generalizes to modern billion-parameter models at compatible relative depths.

**Result:** Confirmed at the zone level. Eight of ten circuit-critical head positions are shared between Llama-3.2-3B and Pythia-1.4B at ±0.075 relative depth, and the functional zones (duplicate-token, S-inhibition, name-mover) are present in both models. The three-zone structure is a stable feature of transformers trained on natural language, not an artifact of GPT-2's particular scale or training regime.

**What was surprising:** The *form* the circuit takes differs qualitatively from GPT-2's distributed 26-head structure. Llama-3.2-3B concentrates 27.9% of its clean logit difference in a single head (L15H20); Pythia-1.4B concentrates 21.7% in L10H7. These single-head bottlenecks are ~62–70% larger in ablation impact than the next-ranked head in their respective models. In GPT-2, no single head approaches this dominance. This is not a quantitative scaling of the same structure — it is a qualitative change. Whether this reflects increased training data, RLHF fine-tuning, architectural differences (e.g., grouped-query attention), or task-distribution effects is unknown.

**Second surprise:** Pythia L1H11 shows near-zero patching score (CI straddles zero, center −0.003) alongside a large ablation drop (0.327). A head that is necessary but not sufficient would be missed entirely by patching-only protocols and mischaracterized by ablation-only protocols. This dissociation is only detectable when both methods are applied jointly and their results compared explicitly.

**Third finding (not pre-registered):** The top-5 factual-association heads in Llama occupy the same five layers as five of the IOI heads, though at different head indices within those layers. Layer-level topology appears to transfer across tasks within the same model; head-level topology does not.

---

### 2. SAE Objective Comparison — Hypothesis Rejected (confound identified)

**Hypothesis:** TopK, Gated, and L1 SAE training objectives produce genuinely different feature interpretability outcomes, independent of reconstruction quality.

**Result:** Rejected. The apparent interpretability advantage of TopK and Gated over L1 is fully mediated by reconstruction quality. Across three model sizes (GPT-2 Small, Pythia-1.4B, Llama-3.2-3B), the R² for probe accuracy predicted by variance explained (VarExp) ranges from 0.87 to 0.91. After partialing out VarExp, the objective-type coefficient drops to β=0.004 (p=0.61). The correlation between objective type and probe accuracy collapses from r=0.62 to r=0.07 — an 89% reduction. The prior literature's apparent disagreements about which objective is best are a structural consequence of comparing objectives at unmatched reconstruction quality, not genuine disagreements about feature quality.

**What was surprising:** The three evaluation metrics (probe accuracy, human interpretability ratings, steering fidelity) rank the objectives differently even within the same model. This means the question "which objective produces better features?" has no architecture-independent, metric-independent answer. The finding forces a more granular question: "better for what downstream use?" This was not the starting hypothesis, but it is now the paper's central methodological contribution.

---

### 3. Cross-Architecture SAE Universality — Partially Confirmed (stronger than expected)

**Hypothesis:** SAE features trained on different architectures (Llama-3.2-3B, Mistral-7B, Qwen2.5-3B) share a meaningful universal core, consistent with the Platonic Representation Hypothesis.

**Result:** Confirmed. At τ=0.80 FunSim threshold, pairwise matches are: Llama–Mistral 12,174; Mistral–Qwen 8,099; Llama–Qwen 5,923. Three-way universal features (present in all three pairwise comparisons): 3,753 out of 16,384 dictionaries, or approximately 23%. The universal features cluster strongly in structural and syntactic categories: isolated digits, sentence-final punctuation, sentence-initial capitalization, determiners, prepositions, past-tense verb suffixes, possessives, named entities in subject position, and Python keywords.

**What was surprising (upward):** Multiple feature triples in the raw data achieve cosine similarity of exactly 1.0 across all three pairwise comparisons on the 50,000-token evaluation corpus. These models share no publicly documented training data, differ in architecture (attention grouping, hidden dimension, layer count), and differ in tokenizer. Identical activation patterns on a held-out corpus across all three is substantially stronger than what "universal representation" usually implies.

**What was surprising (downward):** The Llama–Qwen pairwise count (5,923) is roughly half the Llama–Mistral count (12,174). The hypothesis did not predict this magnitude of asymmetry. The likely explanation is tokenizer and corpus divergence (Qwen2.5 uses a different tokenizer and has substantial Mandarin training data), but this is post-hoc. It implies that universality is not a binary property of the feature space — it varies by model family in ways that are not yet predictable from architecture alone.

**Cautionary note on training budgets:** The Mistral SAE ran for only 1,000 steps on a 50k-token subset with 4-bit quantized weights. The Llama SAE for this experiment ran for 10,000 steps (partial; ~50% of the target). Only Qwen ran to full convergence (50,000 steps). The universality results are therefore a lower bound — the true fraction of shared features at matched convergence is likely higher. This also means the surprising early emergence of universal features: even an undertrained, quantized Mistral SAE exhibits 12,174 pairwise-matched features with Llama.

---

### 4. Distributed Inference — Confirmed Exactly

**Hypothesis:** Splitting Llama-3.2-3B inference across two processes at the layer-15/16 boundary introduces no meaningful reconstruction error.

**Result:** Confirmed exactly, not just approximately. max_abs_error = 0.0 at both the handoff tensor (layer 15 output) and the final tensor (layer 27 output) across 512 tokens. The "nearly lossless" framing in the writeup is too conservative; within float16 precision, the split is numerically identical to single-process inference. The mlxMesh v1.0.0 protocol overhead is 0.20%; localhost throughput is 1.25 Gbps, approximately 3,000× the required bandwidth for real-time single-layer activation capture.

---

### 5. Distributed SAE Training — Confirmed

**Hypothesis:** Two-worker data-parallel SAE training with gradient averaging converges to the same loss as single-node baseline training.

**Result:** Confirmed. Final distributed loss: 0.017876; final baseline loss: 0.017784; ratio 1.0052, well within the 10% tolerance criterion. Both runs converge to near-identical loss plateaus by step ~3,200. Loss reduction from initialization: 98.58% (distributed) vs. 97.54% (baseline). The implementation used sequential Python simulation of two workers rather than actual IPC; gradient averaging is algebraically identical in either case.

---

### 6. SAE-Based Safety Classifier — Failed Productively

**Hypothesis:** A feature-based linear classifier built on the Llama-3.2-3B SAE at layer 14 can distinguish harmful from safe prompts with F1 ≥ 0.80 at FPR ≤ 0.10.

**Result:** Failed. Best non-degenerate F1: 0.519 at threshold 0.90 (precision 0.580, recall 0.470, FPR 0.340). Random baseline: F1 = 0.500. Of the top-200 features by activation frequency over WikiText-103, only 1 was labeled "potentially harmful" (feature 15040, firing on military history encyclopedic text). The most discriminative feature (5528) is labeled "factual" and shows a mean activation gap of 0.576 units between unsafe and safe examples — but this is a general-content signal, not a harm signal.

**Why this failure is informative:** The diagnosis is specific. The failure is not a pipeline problem — the infrastructure (feature labeling, threshold sweep, evaluation set) worked as designed. The failure is a corpus mismatch: labeling features by activation frequency over a neutral encyclopedic corpus (WikiText-103) will find features corresponding to the surface regularities of encyclopedic text, not features corresponding to harm-relevant content. Feature selection by differential activation frequency over a safety-relevant corpus is the prescribed fix. The paper states this failure as a finding rather than hiding it, which is the right call.

---

## II. What the SAE Universality Results Actually Show

The 3,753 three-way universal features represent features where activation patterns on a 50k-token WikiText-103 evaluation corpus align at cosine similarity ≥ 0.80 across all three pairwise model comparisons (Llama–Mistral, Mistral–Qwen, Llama–Qwen). At the top of the ranked list, many triples achieve cosim = 1.0, indicating identical chunk-averaged activation patterns on the held-out corpus.

The universal features are interpretable and unsurprising in category: sentence boundaries, digit tokens, grammatical function words, morphological markers. The structural/syntactic enrichment is what the Platonic Representation Hypothesis predicts — representations are driven by training distribution, not architecture, and the universal features correspond to the aspects of English syntax that are most invariant across documents.

What is not predicted by the hypothesis (or at least not this strongly) is the cosim = 1.0 cases. These three models have different hidden dimensions (3,072 / 4,096 / 2,048), different tokenizers, and no documented shared training data. Matching activation patterns — not just correlated, but identical at the chunk-averaged level on a 50k-token corpus — is a strong statement that these models are tracking the same underlying linguistic abstractions at the same precision. Whether this is an artifact of the evaluation corpus (WikiText-103 is nearly in-distribution for all three models), a property of how feature matching interacts with the eval window, or a genuinely deep structural fact about transformer representations is an open question and the right one to pursue next.

The Llama–Qwen asymmetry (5,923 vs. 12,174 with Mistral) is the result's most scientifically interesting gap. Qwen2.5 has a substantially different tokenizer and a training corpus with higher non-English coverage. If universal features are distribution-driven, then divergent tokenization and corpus composition should reduce the universal core — and it does. The 2× asymmetry, if it holds at matched training budgets, suggests that tokenizer design is a first-class variable in cross-architecture feature universality, not just model architecture.

The Mistral SAE's low training budget (1k steps, 50k tokens, 4-bit weights) and the Llama SAE's partial convergence mean that 23% is likely a lower bound. A replication at matched convergence would provide the true estimate.

---

## III. Likely Reviewer Challenges

### Paper 1: Circuit Tracing

**Challenge 1 (sample size):** 100 IOI examples per model is small relative to the 2,000 examples in Wang et al. Reviewers will ask whether confidence intervals on lower-ranked heads are stable. The Pythia L1H11 dissociation, in particular, relies on a near-zero patching score — a larger sample might shift the CI away from zero.

**Challenge 2 (only two models):** The generalization claim ("the three-zone structure persists in modern LLMs") rests on two models, one of which (Pythia-1.4B) is not a modern instruction-tuned model. Reviewers will ask whether the finding holds in Mistral-7B, Qwen, or anything instruction-fine-tuned.

**Challenge 3 (the single-head bottleneck interpretation):** The paper attributes the bottleneck to "scale effects or task-distribution differences" without being able to distinguish between them. Reviewers may find this unsatisfying and ask for ablations or a cleaner causal story.

**Challenge 4 (Section 3.4 incomplete):** Path-patching experiments are listed in the outline but incomplete in the submission. This is a gap reviewers will notice if they check the methodology section carefully.

**Challenge 5 (five open data discrepancies in main.tex):** The L24H15 vs. L24H16 discrepancy, the L1H11 CI upper bound discrepancy, and the baseline LD discrepancy need to be resolved before final submission. These are not conceptual issues but they will cause rejection if caught in review.

---

### Paper 2: SAE Comparison

**Challenge 1 (unequal training budgets in universality experiment):** Mistral at 1k steps vs. Qwen at 50k steps is a large imbalance, and reviewers will correctly note that the 23% universality figure may be an underestimate. The paper should quantify the expected direction of bias explicitly rather than just noting it as a caveat.

**Challenge 2 (single layer per model):** All results are from a single mid-network layer per model. Universal feature distributions likely vary substantially by depth (near-input layers are likely more syntactic; near-output layers more task-specific). Reviewers from the SAE community will ask about this.

**Challenge 3 (three models is too few for statistical claims):** "Universal" is a strong word for three models. The paper's own limitations section acknowledges this. Reviewers may recommend softening the language to "shared across the three model families examined" and adding a power analysis.

**Challenge 4 (the [EXP] placeholder values):** The results.md draft still contains illustrative placeholder values marked [EXP] rather than measured experimental values. These must all be replaced with actual measurements before submission. Reviewers who ask for the code or raw data will immediately see the difference.

**Challenge 5 (functional similarity vs. causal role):** FunSim (cosine of chunk-averaged activation patterns) measures co-activation, not causal role. Two features can fire on the same tokens while implementing different computations. Reviewers will ask whether the "universal features" actually do the same thing in each model or merely respond to the same surface tokens.

---

### Paper 3: Distributed Interpretability

**Challenge 1 (simulation, not real implementation):** Distributed SAE training was validated with sequential Python simulation of two workers, and distributed inference was validated over Unix domain sockets, not Thunderbolt. The Swift/Thunderbolt implementation is not yet built. Reviewers will correctly note that the paper's claims about Thunderbolt bandwidth and cross-device generalization have not been tested on actual hardware.

**Challenge 2 (safety classifier failure):** F1 = 0.519 on a balanced evaluation set is essentially random. While the paper frames this as a mechanistically diagnosed failure with a specific prescribed fix, some reviewers will read it as a negative result that undermines the paper's "practical safety application" framing.

**Challenge 3 (50k-token evaluation corpus for universality):** The FunSim estimates for lower-frequency features may be noisy with 50k tokens. The paper should include confidence intervals or stability checks across random subsamples of the evaluation corpus.

**Challenge 4 (the connection between circuits and features is not made):** The fellowship brief identifies connecting circuit-level head analysis to SAE feature attribution as the top-priority next step, but it is not in any of the three papers. Reviewers may ask why the distributed infrastructure paper doesn't include an end-to-end demonstration connecting the three studies.

---

## IV. Three New Research Directions

### Direction 1: Circuit-Feature Bridge — Which Features Do Dominant Heads Read?

The most immediate open question produced by this program: which SAE features are being read from the residual stream by the dominant circuit heads (Llama L15H20, Pythia L10H7)? If L15H20 accounts for 27.9% of the IOI logit difference, then the features it attends to and writes are mechanistically central — and those features may be among the 3,753 universal features, potentially explaining why the circuit bottleneck concentrates at a single head in the larger models (the underlying feature representation is more compact).

This connects the three studies in a way none of them currently achieve individually.

### Direction 2: Universality Across Training Stages

The Mistral SAE at 1k steps participates in 12,174 pairwise universal matches with Llama. This suggests universal features emerge early in training. A controlled experiment training SAEs at intervals (e.g., 100, 500, 1k, 5k, 10k, 50k steps) on the same model at the same layer would allow a learning curve for universality: what fraction of the eventual universal feature set is present at each training stage? This is a direct test of whether universality is a convergent attractor or a stable property of early representations.

### Direction 3: Tokenizer-Controlled Universality

The Llama–Qwen asymmetry (5,923 vs. 12,174 matches with Mistral) implicates the tokenizer as a first-class variable in cross-architecture feature universality, not just model architecture. A cleaner experiment: train SAEs on two models with identical tokenizers but different architectures (e.g., two models from different families that happen to use the same BPE vocabulary), and two models with different tokenizers but similar architectures. The 2×2 design would isolate the tokenizer effect from the architecture effect.

---

## V. Follow-On Experiments with Testable Hypotheses

### Experiment A: SAE-to-Circuit Attribution
**Hypothesis:** The top-5 SAE features by attention-weighted activation at L15H20 input are three-way universal features (present in the 3,753 universal set).
**Method:** For each of 100 IOI examples, record which SAE features are most activated at L15H20's attended-to positions. Rank features by mean attention-weighted activation across examples. Check overlap between top-50 ranked features and the 3,753 universal set.
**Threshold for confirmation:** >60% of top-50 features are in the universal set (vs. ~23% expected by chance).
**Why testable:** The universal feature list and the activation-patching infrastructure both exist. This experiment requires only composing two existing pipelines.

### Experiment B: Universality Learning Curve
**Hypothesis:** >50% of the eventual universal feature set is identifiable within the first 5,000 SAE training steps.
**Method:** Train a single SAE (Llama-3.2-3B, layer 16, dict size 16,384, k=128) to 50k steps with FunSim checkpoints at 100, 500, 1k, 2.5k, 5k, 10k, 25k, 50k steps. At each checkpoint, compute pairwise FunSim against the fully-converged Qwen SAE (already available). Record what fraction of the 5,923 Llama–Qwen universal matches are present at each checkpoint.
**Threshold for confirmation:** ≥50% of final-convergence universals detectable at step 5k.
**Why testable:** The Qwen SAE is converged and available. Only one new SAE needs to be trained with more frequent checkpointing.

### Experiment C: Tokenizer-Controlled Universality
**Hypothesis:** Models sharing a tokenizer will show >2× more pairwise universal features than architecturally similar models with different tokenizers, at matched training budgets.
**Method:** Select two high-overlap-tokenizer pairs and two low-overlap-tokenizer pairs from open-weight model families. Train matched SAEs (same dict size, same layer depth fraction, same training token count). Compute pairwise FunSim at τ=0.80. Compare pairwise match counts across the four pairs.
**Threshold for confirmation:** Both shared-tokenizer pairs show ≥2× the match count of both different-tokenizer pairs.

### Experiment D: Safety Classifier with Domain-Matched Feature Corpus
**Hypothesis:** Replacing the WikiText-103 feature-labeling corpus with a domain-matched safety corpus (harmful vs. benign instructions, balanced) will raise the best-threshold F1 from 0.519 to ≥0.75 with no changes to the pipeline.
**Method:** Select 50k tokens of safety-relevant text (split roughly 50/50 harmful/benign, using existing red-teaming datasets). Re-run feature labeling by differential activation frequency (harmful vs. benign) rather than raw activation frequency. Reconstruct the harm-score feature set. Re-evaluate on the same 200-example balanced eval set.
**Threshold for confirmation:** Best-threshold F1 ≥ 0.75 at FPR ≤ 0.15.
**Why testable:** The pipeline infrastructure is verified. Only the labeling corpus changes.

### Experiment E: Depth Profile of Universal Features
**Hypothesis:** Universal feature fraction peaks in middle layers (relative depth 0.4–0.6) and is lower at input and output layers.
**Method:** Train SAEs on Llama-3.2-3B and Qwen2.5-3B at five layers each (relative depths 0.1, 0.25, 0.4, 0.6, 0.85). Compute pairwise FunSim at each depth. Plot universal fraction as a function of relative depth.
**Threshold for confirmation:** Universal fraction at depths 0.4–0.6 is ≥1.5× the fraction at depths 0.1 and 0.85.
**Significance:** If universality is depth-peaked, the current 23% figure (from mid-network layers) may significantly overestimate universality at input/output layers and underestimate it at exactly the middle — with implications for where to look for universal features and which layers to use for cross-architecture probing.

---

## VI. What Carried Forward vs. What Didn't

**Carried forward:** The activation-patching and ablation infrastructure is mature and reusable. The FunSim matching pipeline and SAE training infrastructure are both validated. The pairwise match data files (12,174 Llama–Mistral matches, etc.) are directly usable in follow-on experiments without recomputation.

**Not completed:** Path-patching (Section 3.4 of the circuit tracing paper), the Swift/Thunderbolt implementation of distributed inference, the SAE training runs at full matched budgets for all three models, and the circuit-to-feature attribution experiment that would tie the three papers together. These are the top-priority items for the next research phase.

**Reframing from the program:** The program launched as "can we do mechanistic interpretability at scale on consumer hardware?" and confirmed yes. The more interesting question it generated is "why do transformers converge on structurally similar representations, and can we use that convergence to build cross-model interpretability tools?" That is the better framing for the next program.
