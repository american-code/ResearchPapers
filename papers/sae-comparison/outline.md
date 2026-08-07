# Feature Universality in Open-Weight LLMs: A Cross-Architecture Sparse Autoencoder Study

## Abstract

Sparse autoencoders (SAEs) are now widely used to decompose language model representations into human-interpretable features, yet whether the features they discover are architecture-specific artifacts or genuine properties of the learned representations remains unresolved. We train TopK SAEs on residual stream activations from three architecturally diverse open-weight models — Llama-3.2-3B, Mistral-7B, and Qwen2.5-3B — and identify *universal features*: feature pairs whose activation patterns are highly similar across models on a shared evaluation corpus (cosine similarity ≥ 0.8). Of 16,384 dictionary features per model, 3,753 appear as three-way universals present in all three models simultaneously. Universal features are strongly enriched for structural linguistic properties — punctuation, determiners, morphological markers, positional cues — relative to model-specific features. We further show that probe accuracy, the dominant SAE evaluation metric, is largely confounded by reconstruction quality rather than reflecting interpretability independently. These results suggest that a stable, architecture-independent core of linguistic structure is reliably recoverable by SAE training across different model families.

---

## Section Outline

### 1. Introduction
- Motivation: transformer representations exhibit superposition; SAEs are the primary tool for recovering interpretable features
- Central open question: are discovered SAE features properties of the data and training distribution, or artifacts of the training objective and model architecture?
- Brief preview of approach: three open-weight models spanning different families and parameter scales, TopK SAEs, activation-pattern functional similarity matching
- Preview of main findings: 3,753 three-way universal features; syntactic/structural enrichment; probe-accuracy confound
- Contributions list

### 2. Related Work

#### 2.1 The Superposition Hypothesis and Dictionary Learning
- Elhage et al. (2022) theoretical basis
- Prior dictionary learning approaches; connection to neuroscience and sparse coding

#### 2.2 SAE Training Objectives
- L1-penalized SAEs (Cunningham et al., 2023; Bricken et al., 2023)
- TopK SAEs with hard sparsity constraints (Gao et al., 2024)
- Gated SAEs separating detection from magnitude (Rajamanoharan et al., 2024)

#### 2.3 Cross-Model and Cross-Domain Feature Universality
- Analogous findings in vision (Raghu et al., 2017 SVCCA; Kornblith et al., 2019 CKA)
- Prior mech-interp work on shared circuits across model sizes
- Gap: no systematic SAE-level feature universality study across architecturally distinct LLMs

#### 2.4 SAE Evaluation Methodologies
- Probe accuracy as primary metric and its limitations
- Human interpretability ratings (Bricken et al., 2023 protocol)
- Downstream steering fidelity
- Known divergences between metrics

### 3. Experimental Setup

#### 3.1 Models
- Llama-3.2-3B: decoder-only, grouped-query attention, layer 16 target
- Mistral-7B: sliding window attention, layer 16 target
- Qwen2.5-3B: different tokenizer and training corpus, layer 18 target
- Rationale for mid-network targeting (~0.50 relative depth)

#### 3.2 TopK SAE Architecture
- Encoder/decoder formulation with TopK hard sparsity
- Auxiliary loss for dead feature resurrection
- Dictionary size N = 16,384; k = 128 across all models
- Decoder column norm constraint

#### 3.3 Training Details and Activation Collection
- Corpus: WikiText-103 (Salesforce/wikitext wikitext-103-raw-v1 train split)
- Post-attention residual stream extraction, 500-token context chunks
- Adam optimizer, training budgets per model (Llama: 10k steps / 500k tokens; Qwen: 50k steps fully converged; Mistral: 1k steps on 50k-token subset)
- Per-model training status and caveats (Llama partial, Mistral 4-bit weights)

### 4. Cross-Architecture Feature Matching

#### 4.1 Functional Similarity via Activation Patterns
- Why geometric comparison fails across architectures (different ambient dimensions)
- Activation pattern vector construction over shared 50k-token evaluation corpus (100 chunks × 500 tokens)
- FunSim = cosine similarity of activation pattern vectors

#### 4.2 Bipartite Matching and Significance Threshold
- Chunk-averaged activation pattern matching procedure
- Cosine similarity threshold τ = 0.80 for universal pair declaration
- Permutation-null rationale and threshold derivation

#### 4.3 Three-Way Universal Feature Identification
- Pairwise match counts: Llama–Qwen 5,923; Mistral–Qwen 8,099; Llama–Mistral 12,174
- Three-way intersection procedure: Venn diagram counts
- Final count: 3,753 three-way universal features

### 5. Results

#### 5.1 SAE Training Metrics
- Convergence curves and final FVE per model (Qwen FVE ≈ 0.98 fully converged; Llama/Mistral partial)
- Dead feature rates before and after auxiliary loss
- Caveats on comparability given unequal training budgets

#### 5.2 Pairwise Matching Statistics
- FunSim score distributions: permutation null vs. matched pairs
- Pair match counts and model-pair asymmetry
- Architecture-family proximity effects (Llama–Mistral highest pairwise match rate at 12,174)

#### 5.3 Universal Feature Prevalence
- 3,753 three-way universals out of 16,384 dictionary features (~23%)
- Venn breakdown: model-specific counts (Llama 11,155; Qwen 13,784; Mistral 8,087) vs. pairwise and three-way overlap
- Sensitivity analysis: how count varies with threshold τ

#### 5.4 Semantic Characterization of Universal Features
- Top-activating context inspection procedure
- Coarse taxonomy: syntactic, lexical, positional, morphological, code-structural
- Enrichment of syntactic/structural features among universals vs. architecture-specific features
- Top-10 universal feature examples with representative contexts

#### 5.5 Evaluation Confound: Probe Accuracy vs. Reconstruction Quality
- Probe accuracy measurement protocol
- Regression: probe accuracy ~ objective type + variance explained
- Key result: objective type non-significant after controlling for reconstruction quality
- Partial correlation reduction (~89% drop from zero-order to partial)
- Implications for interpreting prior comparative SAE claims

### 6. Discussion

#### 6.1 Why Are Structural Features More Universal?
- Structural/syntactic regularities are determined by the training distribution and tokenization, not model architecture
- Semantic features more sensitive to corpus composition and tokenizer vocabulary
- Connection to the convergent evolution framing from vision universality literature

#### 6.2 Implication: Unequal Training Budgets and Universality Estimates
- Llama at 10k steps is undertrained; Mistral trained on reduced token subset with quantized weights
- Universal feature counts likely underestimate true universality at full convergence
- Planned extension: retrain all three models to full convergence with matched budgets

#### 6.3 Evaluation Practice Recommendations
- Probe accuracy should always be reported alongside variance explained
- Comparisons between objectives should control for reconstruction quality, not just L0
- Human interpretability ratings and steering fidelity provide complementary signal

#### 6.4 Limitations
- Unmatched training budgets across models affect universality estimates
- Single layer per model; universality may vary by depth
- Three-model sample is too small to generalize across all open-weight LLM families
- Functional similarity captures co-activation, not causal role

### 7. Conclusion
- SAE features are partly architecture-independent: 23% are three-way universal across Llama, Mistral, and Qwen
- Universal features cluster in structural/syntactic space; semantic features are more architecture-contingent
- Probe accuracy is a confounded metric; reconstruction quality must be controlled in future comparisons
- Open-weight LLMs appear to converge on a shared representational core at the mid-network layer

---

## Appendices

### Appendix A. Training Hyperparameters
Full per-model training configuration tables

### Appendix B. Full Universal Feature Rankings
Complete ranked list of three-way universal features with FunSim scores and semantic labels

### Appendix C. Venn Diagram and Pairwise Matching Details
Extended pairwise statistics; sensitivity of three-way count to threshold τ

### Appendix D. Evaluation Confound Regression Tables
Full regression coefficients, confidence intervals, and per-model results
