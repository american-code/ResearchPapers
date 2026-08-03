# Post-Roadmap Retrospective
**J. Melton — January 2027**

---

## Overview

This document reviews the three-paper interpretability research program initiated in mid-2026 (originally targeting ICLR 2026, later re-scoped to ICLR 2027 / NeurIPS 2026 workshop cycle). It covers what each experiment set out to test, what it found, where results departed from expectations, what reviewers are likely to challenge, and what new directions the findings open up.

The program spans three interconnected studies: mechanistic circuit analysis of IOI and factual-association tasks across modern LLMs; a systematic comparison of SAE training objectives and a cross-architecture feature universality experiment that returned a null result under corrected analysis; and infrastructure work on distributed inference and training (mlxMesh), culminating in a safety classifier evaluation that failed productively.

---

## I. Experiment Outcomes: Confirmed vs. Surprising

### 1. IOI Circuit Topology Generalizes — Confirmed (with a major structural twist)

**Hypothesis:** The three-zone IOI circuit topology documented by Wang et al. (2022) for GPT-2 Small generalizes to modern billion-parameter models at compatible relative depths.

**Result:** Confirmed at the zone level. Eight of ten circuit-critical head positions are shared between Llama-3.2-3B and Pythia-1.4B at ±0.075 relative depth, and the functional zones (duplicate-token, S-inhibition, name-mover) are present in both models. The three-zone structure is a stable feature of transformers trained on natural language, not an artifact of GPT-2's particular scale or training regime.

**What was surprising:** The *form* the circuit takes differs qualitatively from GPT-2's distributed 26-head structure. Llama-3.2-3B concentrates 27.9% of its clean logit difference in a single head (L15H20); Pythia-1.4B concentrates 21.7% in L10H7. These single-head bottlenecks are ~62–70% larger in ablation impact than the next-ranked head in their respective models. In GPT-2, no single head approaches this dominance. This is not a quantitative scaling of the same structure — it is a qualitative change. Whether this reflects increased training data, RLHF fine-tuning, architectural differences (e.g., grouped-query attention), or task-distribution effects is unknown.

**Second surprise:** Pythia L1H11 shows near-zero patching score (CI straddles zero, center −0.003) alongside a large ablation drop (0.327). A head that is necessary but not sufficient would be missed entirely by patching-only protocols and mischaracterized by ablation-only protocols. This dissociation is only detectable when both methods are applied jointly and their results compared explicitly.

**Third finding (not pre-registered):** The top-5 factual-association heads in Llama occupy the same five layers as five of the IOI heads, though at different head indices within those layers. Layer-level topology appears to transfer across tasks within the same model; head-level topology does not.

---

### 2. SAE Objective Comparison — Not Executed

**Hypothesis:** TopK, Gated, and L1 SAE training objectives produce genuinely different feature interpretability outcomes, independent of reconstruction quality.

**Result: Not executed.** The multi-objective SAE comparison was planned but never carried out. Only a single training objective (TopK, k=128) was used across all three models trained in this program. The models were Llama-3.2-3B, Mistral-7B-v0.3, and Qwen2.5-3B — not GPT-2 Small and Pythia-1.4B as the original hypothesis specified. Statistical claims that appeared in an early draft of this document (R²=0.87–0.91, β=0.004, p=0.61, r=0.62→r=0.07) were illustrative placeholder values; they were never computed from data and should be disregarded entirely.

**What the SAE training actually produced:** The three TopK SAEs were trained to support the cross-architecture universality analysis (Section 3), not an objective comparison. All three SAEs completed full 50,000-step training on 500k tokens: Qwen locally; Llama and Mistral on lab-02 (checkpoints timestamped 2026-07-29, retrieved subsequently). Earlier analyses ran on partial checkpoints (Llama: 10k steps; Mistral: 1k steps / 50k tokens / 4-bit quantized), which inflated apparent match counts due to undifferentiated features and an uncalibrated similarity threshold. The v2 cross-architecture universality analysis on full checkpoints, using a corrected algorithm (reciprocal one-to-one matching, closed-triangle requirement, permutation-calibrated threshold), yields 1 three-way universal feature — statistically indistinguishable from the permutation null (p ≈ 0.14).

The objective-comparison hypothesis — whether training objective produces interpretability differences independent of reconstruction quality — remains untested and is a candidate experiment for the next phase.

---

### 3. Cross-Architecture SAE Universality — Not Confirmed (null result under corrected v2 analysis)

**Hypothesis:** SAE features trained on different architectures (Llama-3.2-3B, Mistral-7B, Qwen2.5-3B) share a meaningful universal core, consistent with the Platonic Representation Hypothesis.

**Result:** Not confirmed. The initial v1 analysis (τ = 0.80, non-reciprocal matching, unclosed triangles) reported 3,753 three-way universals (~23%), but three independent algorithmic defects inflated that count: many-to-one matching (one Mistral feature was counted as the match for up to 137 distinct Llama features), unclosed triangles (67% of v1 triples had no verified third edge), and a threshold that sits at or below the permutation-null noise floor for every model pair. The corrected v2 analysis (reciprocal one-to-one matching, closed-triangle requirement, permutation-calibrated τ per pair, 5% minimum chunk-support floor) on fully-trained checkpoints yields pairwise matches of 15 (Llama–Mistral), 23 (Mistral–Qwen), and 16 (Llama–Qwen) — on the order of expected false-positive counts. Three-way universal features: 1, against a permutation-null expectation of 0.15 (Pr(≥1 | null) ≈ 0.14). The single apparent universal is statistically indistinguishable from a false positive.

**What was surprising (about the analysis process):** The v1 analysis produced apparent cosim = 1.0 triples — features with literally identical activation patterns across all three models on the evaluation corpus. These turned out to be artifacts of degenerate low-support fingerprints: features firing on fewer than 5% of evaluation chunks, where a handful of coincident activations dominate the chunk-average and drive cosine similarity to 1.0. The v2 analysis imposes a 5% minimum-chunk-support floor that eliminates these. The 'cosim = 1.0 as evidence for strong universality' interpretation in earlier drafts is therefore withdrawn — it was a methodological artifact, not a finding about the models.

**What was surprising (about the v1-to-v2 correction):** The dramatic pairwise asymmetry in the v1 analysis (Llama–Qwen at 5,923 vs. Llama–Mistral at 12,174, a 2× gap) disappeared entirely under the corrected analysis. v2 pairwise counts are 15–23 per pair with no meaningful spread. The v1 asymmetry was an artifact of many-to-one matching and the uncalibrated threshold, not a real property of the representations. The hypothesis that Qwen's tokenizer and Mandarin training data reduce cross-architecture overlap remains scientifically motivated but is no longer supported by the current data — there is insufficient signal above the noise floor to detect it.

**Note on training budgets (resolved):** All three SAEs completed full convergence (50,000 steps, 500k tokens each). Llama and Mistral final checkpoints were completed on lab-02 by 2026-07-29 and retrieved subsequently; Qwen's local copy was already the full run. The partial-checkpoint results cited in earlier drafts (3,753 three-way universals from Llama at 10k steps and Mistral at 1k steps) are superseded. The v2 null result is not an artifact of undertraining — the full-checkpoint analysis removes that confound and the count moves from 0 (partial) to 1 (full), remaining indistinguishable from null.

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

The corrected v2 result is a null: 1 three-way universal feature across the three models (Llama-3.2-3B, Mistral-7B, Qwen2.5-3B), against a permutation-null expectation of 0.15 and Pr(≥1 | null) ≈ 0.14. The apparent 3,753-feature result from the v1 analysis was produced by three independent algorithmic defects. Many-to-one matching allowed a single Mistral feature to be the recorded nearest neighbor of up to 137 distinct Llama features, making reported "match counts" exceed dictionary size and rendering coverage percentages meaningless. Unclosed triangles meant 67% of v1 triples had no verified third edge. And the fixed threshold τ = 0.80 was chosen by hand without calibration — under a chunk-permutation null, the mean reciprocal-match cosine similarity between unrelated features is 0.61–0.74 at v2 evaluation resolution, so τ = 0.80 sits below the noise floor for every model pair.

The pairwise match counts under the corrected analysis are 15 (Llama–Mistral), 23 (Mistral–Qwen), and 16 (Llama–Qwen) against calibrated per-pair thresholds (τ ranging from 0.95 to 0.98). Expected false-positive counts at these thresholds are 7–9 per pair, meaning the observed counts are only 1.7–3.3× enriched over chance — not statistically significant. The v1 pairwise counts (12,174 / 8,099 / 5,923) were inflated by many-to-one matching; as a check, Qwen's v1 Venn regions summed to 18,258 entries in a 16,384-feature dictionary. The v2 Venn counts partition each dictionary exactly.

The cosim = 1.0 triples that appeared to be the strongest evidence of universality in the v1 analysis were artifacts of degenerate low-support fingerprints. Features firing on fewer than 5% of evaluation chunks have activation vectors dominated by a few coincident chunks, driving cosine similarity to 1.0 without indicating genuine representational overlap. The v2 analysis enforces a 5% minimum-chunk-support floor; no cosim = 1.0 triples survive.

The dramatic v1 tokenizer-asymmetry finding (Llama–Qwen 5,923 vs. Llama–Mistral 12,174, a 2× gap attributed to Qwen's different tokenizer and Mandarin training corpus) also does not survive correction. v2 pairwise counts show 15–23 matches per pair with no meaningful asymmetry. The hypothesis that tokenizer divergence modulates cross-architecture universality is still scientifically motivated — it is just not supported by the current data at the current evaluation scale. The null result is the finding: even fully-converged SAEs on three architectures trained on similar distributions share essentially no reliably-matched features at a calibrated similarity threshold.

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

**Challenge 1 (null universality result):** The v2 analysis yields 1 three-way universal feature (p ≈ 0.14 vs. null) — a null result, not the originally reported 23%. The paper must clearly report this as a null finding and explain the v1 algorithmic defects that inflated the earlier count. Reviewers will ask whether the null reflects a true absence of universality or an underpowered experiment (insufficient evaluation tokens, wrong similarity metric, too few models). The paper should address each alternative explanation explicitly and discuss what scale of universality the current setup could reliably detect if it existed.

**Challenge 2 (single layer per model):** All results are from a single mid-network layer per model. Universal feature distributions likely vary substantially by depth (near-input layers are likely more syntactic; near-output layers more task-specific). Reviewers from the SAE community will ask about this.

**Challenge 3 (three models is too few for statistical claims):** "Universal" is a strong word for three models. The paper's own limitations section acknowledges this. Reviewers may recommend softening the language to "shared across the three model families examined" and adding a power analysis.

**Challenge 4 (single training objective, missing comparison):** All three SAEs were trained with a single objective (TopK, k=128). The paper's framing as an objective comparison study is therefore unsupported by the experiments actually run. Reviewers will notice the mismatch immediately. The paper either needs to be reframed as a cross-architecture universality study (dropping the objective comparison framing entirely) or the missing training runs (Gated and L1 variants at matched training budgets) need to be completed and added.

**Challenge 5 (functional similarity vs. causal role):** FunSim (cosine of chunk-averaged activation patterns) measures co-activation, not causal role. Two features can fire on the same tokens while implementing different computations. Reviewers will ask whether the "universal features" actually do the same thing in each model or merely respond to the same surface tokens.

---

### Paper 3: Distributed Interpretability

**Challenge 1 (simulation, not real implementation):** Distributed SAE training was validated with sequential Python simulation of two workers, and distributed inference was validated over Unix domain sockets, not Thunderbolt. The Swift/Thunderbolt implementation is not yet built. Reviewers will correctly note that the paper's claims about Thunderbolt bandwidth and cross-device generalization have not been tested on actual hardware.

**Challenge 2 (safety classifier failure):** F1 = 0.519 on a balanced evaluation set is essentially random. While the paper frames this as a mechanistically diagnosed failure with a specific prescribed fix, some reviewers will read it as a negative result that undermines the paper's "practical safety application" framing.

**Challenge 3 (evaluation corpus for universality):** The v2 analysis uses 500k evaluation tokens (10× the original 50k), which substantially improves fingerprint stability for active features. However, with a null result (1 three-way universal, p ≈ 0.14), the primary reviewer concern shifts from fingerprint noise to statistical power. The paper should report what size universal feature set the current setup could reliably detect, and whether 500k tokens and three models are sufficient to falsify the Platonic Representation Hypothesis at any effect size.

**Challenge 4 (the connection between circuits and features is not made):** The fellowship brief identifies connecting circuit-level head analysis to SAE feature attribution as the top-priority next step, but it is not in any of the three papers. Reviewers may ask why the distributed infrastructure paper doesn't include an end-to-end demonstration connecting the three studies.

---

## IV. Three New Research Directions

### Direction 1: Circuit-Feature Bridge — Which Features Do Dominant Heads Read?

The most immediate open question produced by this program: which SAE features are being read from the residual stream by the dominant circuit heads (Llama L15H20, Pythia L10H7)? If L15H20 accounts for 27.9% of the IOI logit difference, then the features it attends to and writes are mechanistically central — and those features may be among the small pool of pairwise-matched features (15–23 per pair under v2 calibrated analysis), potentially explaining why the circuit bottleneck concentrates at a single head in the larger models (the underlying feature representation is more compact).

This connects the three studies in a way none of them currently achieve individually.

### Direction 2: Universality Across Training Stages

The v2 analysis finds only 15–23 pairwise matches per pair at calibrated threshold, consistent with chance across fully-trained SAEs. A controlled experiment training SAEs at intervals (e.g., 100, 500, 1k, 5k, 10k, 50k steps) on the same model at the same layer would reveal whether pairwise match counts grow toward a signal threshold as training proceeds, or remain flat — which would indicate no detectable signal at any stage. The earlier partial-checkpoint analyses showed dramatically inflated match counts due to v1 defects; this learning-curve experiment, run under the v2 analysis, would provide the first clean test of whether universality is a convergent attractor or simply absent at this model scale and dictionary size.

### Direction 3: Tokenizer-Controlled Universality

The v1 analysis showed a 2× asymmetry between Llama–Qwen and Llama–Mistral pairwise counts, attributed to Qwen's different tokenizer and Mandarin training corpus. The corrected v2 analysis shows no such asymmetry (15–23 matches per pair, no meaningful spread). Whether that is because the tokenizer effect is real but smaller than the noise floor, or because there is no effect, cannot be determined from the current data. A cleaner experiment to test the hypothesis: train SAEs on two models with identical tokenizers but different architectures (e.g., two models from different families that happen to use the same BPE vocabulary), and two models with different tokenizers but similar architectures. The 2×2 design would isolate the tokenizer effect from architecture — and the current null result makes it a sharper question than the v1 numbers implied.

---

## V. Follow-On Experiments with Testable Hypotheses

### Experiment A: SAE-to-Circuit Attribution
**Hypothesis:** The top-5 SAE features by attention-weighted activation at L15H20 input are among the pairwise-matched features identified by the v2 analysis (15–23 per pair, out of 16,384).
**Method:** For each of 100 IOI examples, record which SAE features are most activated at L15H20's attended-to positions. Rank features by mean attention-weighted activation across examples. Check overlap between top-50 ranked features and the v2 pairwise match sets.
**Threshold for confirmation:** Any overlap between top-50 features and the v2 pairwise-matched sets is strong evidence of circuit-feature linkage, given that those sets comprise ~0.1% of the dictionary. Enrichment ≥5× over null rate would be highly significant.
**Why testable:** The universal feature list and the activation-patching infrastructure both exist. This experiment requires only composing two existing pipelines.

### Experiment B: Universality Learning Curve
**Hypothesis:** >50% of the eventual universal feature set is identifiable within the first 5,000 SAE training steps.
**Method:** Train a single SAE (Llama-3.2-3B, layer 16, dict size 16,384, k=128) to 50k steps with FunSim checkpoints at 100, 500, 1k, 2.5k, 5k, 10k, 25k, 50k steps. At each checkpoint, compute pairwise FunSim against the fully-converged Qwen SAE (already available) under the v2 calibrated threshold. Record the pairwise match count at each stage, and whether the 16 eventual Llama–Qwen v2 matches are detectable before full convergence.
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
**Significance:** The current result is a null at mid-network layers (~1 three-way universal at calibrated threshold). If universality is depth-peaked, mid-network layers might paradoxically be harder to compare (higher-level, more model-specific features) while input/output layers show stronger universality in basic token-processing features. A depth profile would determine whether the null result generalizes across depths or is specific to the sampled layers, with direct implications for where cross-architecture probing experiments should be targeted.

---

## VI. What Carried Forward vs. What Didn't

**Carried forward:** The activation-patching and ablation infrastructure is mature and reusable. The FunSim matching pipeline and SAE training infrastructure are both validated. The v2 pairwise match data files (data/sae-analysis/matching-v2/) are directly usable in follow-on experiments without recomputation.

**Not completed:** Path-patching (Section 3.4 of the circuit tracing paper), the Swift/Thunderbolt implementation of distributed inference, and the circuit-to-feature attribution experiment that would tie the three papers together. These are the top-priority items for the next research phase. (SAE training at full matched budgets for all three models is now complete.)

**Reframing from the program:** The program launched as "can we do mechanistic interpretability at scale on consumer hardware?" and confirmed yes. The more interesting question it generated is "why do transformers converge on structurally similar representations, and can we use that convergence to build cross-model interpretability tools?" That is the better framing for the next program.
