# Feature Universality in Open-Weight LLMs: A Cross-Architecture Sparse Autoencoder Study
## 1. Introduction and Related Work

---

## 1. Introduction

Transformer language models store knowledge in high-dimensional activation spaces that simultaneously represent thousands of semantic and syntactic features within individual neurons. Elhage et al. (2022) formalized this observation as the *superposition hypothesis*: a network with $n$ neurons can represent up to $O(n^2)$ features by exploiting near-orthogonality in high-dimensional space, provided features are sparse — active on only a small fraction of inputs. Superposition explains why individual neurons are polysemantic, responding to unrelated concepts, and why mechanistic analysis of raw neuron activations is difficult.

Sparse autoencoders (SAEs) have emerged as the primary tool for dissolving superposition into interpretable components. An SAE learns an overcomplete dictionary of $N \gg d_{\text{model}}$ feature directions, projecting each activation vector onto a sparse combination of dictionary elements. Bricken et al. (2023) applied this at scale to an intermediate layer of Claude and found that the recovered features are strikingly monosemantic — each fires on a coherent semantic or syntactic cluster, often with a human-nameable concept. Gao et al. (2024) extended the approach with hard sparsity constraints (TopK SAEs), demonstrated favorable scaling behavior, and released the resulting features from GPT-4 residual stream layers. This body of work has established SAEs as the de facto method for feature-level mechanistic interpretability.

Yet a basic question remains unresolved: are the features SAEs discover genuine properties of the *data and training distribution*, or are they artifacts of the *model architecture, tokenizer, and parameter count*? The practical importance of this question is high. If SAE features are largely architecture-contingent, then the features Anthropic discovers in Claude, OpenAI discovers in GPT-4, and open-source researchers discover in Llama or Mistral are incommensurable — a feature called "past tense" in one model may be mechanistically unrelated to "past tense" in another. If instead a stable core of features recurs across architectures because it reflects the underlying structure of language and its statistical regularities, then SAE-based interpretability findings should generalize across model families.

This question has not been addressed systematically. All major SAE studies to date train and analyze a single model family (Anthropic's internal models in Bricken et al., 2023; GPT-4 in Gao et al., 2024; GPT-2 and Pythia in the EleutherAI/community work). No prior study trains matched SAEs across multiple architecturally distinct open-weight models and directly tests whether discovered features correspond across the model families.

We address this gap. We train TopK SAEs on residual stream activations at mid-network depth from three architecturally diverse open-weight models — Llama-3.2-3B (Meta, grouped-query attention), Mistral-7B (sliding window attention, different parameter scale), and Qwen2.5-3B (distinct tokenizer and training corpus) — using identical hyperparameters and the same evaluation corpus. We then identify *universal features*: feature pairs whose activation patterns are highly similar across models on a shared token sequence set, using cosine similarity of activation pattern vectors (FunSim) as our similarity measure. The FunSim approach sidesteps the incommensurability of ambient model dimensions by comparing behavior rather than weights.

Our main findings are:

1. **23% of SAE features are three-way universal.** Of 16,384 dictionary features per model, 3,753 exceed our permutation-null threshold (cosine similarity ≥ 0.80) in all three pairwise comparisons simultaneously. Pairwise match counts range from 5,923 (Llama–Qwen) to 12,174 (Llama–Mistral), suggesting that architectural proximity and parameter scale similarity both influence feature sharing.

2. **Universal features cluster in structural-linguistic space.** Universal features are strongly enriched for syntactic, morphological, positional, and structural properties — punctuation, determiners, case markers, sentence boundaries — relative to the model-specific features, which are more likely to encode semantic or domain-specific content. This pattern is consistent with the hypothesis that structural regularities are determined by the training *distribution* (and to some degree by tokenization), not by the model *architecture*.

3. **Probe accuracy is confounded by reconstruction quality.** We replicate the common finding that evaluation metrics diverge, and identify the mechanism: linear probe accuracy on SAE feature activations tracks the fraction of variance explained (reconstruction quality) rather than the training objective used to achieve that reconstruction quality. After controlling for variance explained, the objective type has no significant independent effect on probe accuracy, meaning prior comparative claims based on probe accuracy alone may conflate reconstruction with interpretability.

These results establish SAEs as partly architecture-independent — the large majority of universal features appear to capture properties of natural language itself — while also demonstrating that model-specific features constitute the majority, leaving significant architecture-dependent structure in the learned representations. Our analysis, code, and feature matching results are released to support further work on cross-architecture mechanistic interpretability.

---

## 2. Related Work

### 2.1 The Superposition Hypothesis and Dictionary Learning

Elhage et al. (2022) introduced the superposition hypothesis through a study of toy models trained on synthetic data with controlled feature sets. They showed that networks learn to represent more features than they have neurons when features are sufficiently sparse and approximately orthogonal, and that this regime produces monosemantic neurons when features are non-sparse and polysemantic neurons when they are sparse. Critically, the transition to polysemanticity is a function of feature sparsity and neuron count, not the training objective — suggesting that polysemanticity in large language models is structural and inevitable under natural language statistics.

Dictionary learning as a representational prior has a long history in signal processing and neuroscience (Olshausen & Field, 1996; Mallat & Zhang, 1993), where it was motivated by the observation that early visual cortex neurons can be modeled as a sparse code over natural image statistics. SAE-based mechanistic interpretability connects this tradition to the transformer architecture by instantiating dictionary learning as an end-to-end differentiable encoder-decoder, trained on model activations rather than raw sensory inputs.

### 2.2 Anthropic: Scaling Monosemanticity

Bricken et al. (2023) applied the SAE approach to a single-layer transformer and to an intermediate layer of Claude, discovering that the recovered features exhibit striking monosemanticity: individual dictionary directions fire preferentially on semantically coherent concept clusters (e.g., specific base pairs in DNA sequences, tokens related to legal concepts, specific named entities). The Anthropic work established the methodological template: train an overcomplete dictionary with an L1 sparsity penalty, inspect top-activating contexts, rate feature coherence manually, and measure probe accuracy. Importantly, the analysis was conducted on a proprietary model at a single residual stream layer; no cross-model or cross-architecture comparison was performed.

A subsequent Anthropic study (Templeton et al., 2024) scaled the approach to Claude 3 Sonnet and identified millions of features, including high-level abstract concepts such as safety-relevant reasoning patterns and emotion states. This work demonstrated the method's scalability but reinforced the single-model-family focus that characterizes the Anthropic body of work.

### 2.3 OpenAI: Sparse Features and TopK SAEs

Gao et al. (2024) introduced TopK SAEs, replacing the L1 sparsity penalty with a hard constraint that retains exactly the top-$k$ activating features per input. This guarantees a fixed $L_0$ sparsity level regardless of input norm, decoupling sparsity from reconstruction quality and eliminating the need to tune a penalty weight $\lambda$ to hit a target sparsity level. The TopK formulation also admits a natural auxiliary loss for reviving dead dictionary features — directions that receive no gradient signal because they never fire — which the Anthropic L1 approach addresses only partially.

Gao et al. trained TopK SAEs on GPT-4 residual stream activations and released the resulting feature dictionaries. They show that probe accuracy and manual interpretability ratings improve relative to matched L1 SAEs at identical $L_0$ levels, and they demonstrate favorable scaling: feature quality increases with dictionary size at a rate consistent with a power law. The OpenAI work is the most direct precursor to ours in terms of the TopK objective and the evaluation protocol, but again studies a single model family (GPT-4 variants) and does not ask whether the discovered features would be recovered from a structurally different model.

### 2.4 EleutherAI and the Open-Weight SAE Ecosystem

Cunningham et al. (2023) first demonstrated SAE-based feature extraction on open-weight models, training L1 SAEs on GPT-2 Small and Pythia residual stream activations and finding that the resulting features are interpretable at rates broadly comparable to Bricken et al.'s Claude findings. This work established a reproducible baseline on publicly accessible models and introduced the convention of targeting mid-network layers at relative depth ≈ 0.50.

The EleutherAI group subsequently developed and released SAELens (Bloom et al., 2024), an open-source training and analysis toolkit that has enabled a wave of community SAE work across Llama, Gemma, Pythia, and Mistral model families. SAELens has been used to reproduce the Gao et al. TopK results and to train Gated SAEs (Rajamanoharan et al., 2024) on open-weight models. Despite this proliferation of SAEs across architectures, each study analyzes its own model in isolation; the cross-model feature correspondence question remains unaddressed.

### 2.5 Cross-Architecture Representation Similarity

The question of whether learned representations are architecture-dependent has been studied extensively in computer vision. Raghu et al. (2017) introduced Singular Vector Canonical Correlation Analysis (SVCCA) and showed that different convolutional network architectures trained on the same data converge to similar representations in early layers. Kornblith et al. (2019) introduced Centered Kernel Alignment (CKA) and demonstrated that representational similarity is higher between networks of the same architecture than between architectures, but that some structure is shared across architectures — particularly for lower layers that capture basic visual statistics.

In language models, analogous findings come from the probing and representation geometry literature. Tenney et al. (2019) showed that syntactic information is encoded in lower layers and semantic information in higher layers across multiple BERT-family models. Li et al. (2022) showed that linear transformations can sometimes align representations across different model sizes within a family. However, these studies operate at the level of whole-layer representations, not individual feature directions; they cannot speak to whether the specific dictionary features recovered by SAEs are shared.

Our work differs from prior representation similarity work in a critical way: rather than comparing representations as linear subspaces (as CKA and SVCCA do), we compare individual features by their *functional behavior* — their activation patterns on a shared token set. This is strictly more discriminative than subspace methods because two features can span the same subspace while firing on completely different inputs. Functional similarity is also more directly relevant to interpretability: features that fire on the same tokens and contexts are, by definition, tracking the same linguistic property, regardless of their ambient coordinates in model space.

### 2.6 Gap Addressed by This Work

No prior work has trained matched SAEs across multiple open-weight model families and directly measured the fraction of features that are functionally shared. The open questions are: How many SAE features recur across architecturally distinct models? What distinguishes universal features from model-specific ones? And do current evaluation metrics — in particular, probe accuracy — reflect the properties practitioners care about, or are they confounded by other training variables? We address all three.

---

*Sections 3–7 describe the experimental setup, cross-architecture matching procedure, results, and evaluation methodology. Appendices provide full hyperparameter tables, the complete ranked universal feature list, and regression tables for the evaluation confound analysis.*
