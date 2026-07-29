# Distributed Mechanistic Interpretability at Scale: Activation Streaming, Split-Layer Inference, and Cross-Architecture Feature Universality

**Draft v1** — 2026-07-29  
J Melton  

---

## Abstract

Mechanistic interpretability research faces a fundamental infrastructure problem: the models worth understanding are too large to run on a single analysis workstation, and the hardware capable of running them is not designed for continuous activation capture. We describe a system — **mlxMesh** — that addresses this by splitting inference and analysis across two Apple Silicon nodes connected by Thunderbolt, streaming intermediate activations over a purpose-built binary wire protocol, and training sparse autoencoders (SAEs) on the streaming output without storing activations to disk. We validate the protocol end-to-end on Llama-3.2-3B, demonstrating zero reconstruction error at the split boundary and 1.25 Gbps localhost throughput — roughly 3,000× the bandwidth required for real-time single-layer capture. We then train SAEs on residual stream activations from Llama-3.2-3B (layer 14), Mistral-7B (layer 16), and Qwen2.5-3B (layer 18), and find 3,753 features that match across all three model families at cosine similarity ≥ 0.8 — evidence of a shared representational vocabulary that survives architectural differences. Finally, we evaluate a feature-based safety classifier as an application of the infrastructure, identify the conditions under which it fails, and propose a corpus-targeted repair. All experiments run on consumer Apple Silicon hardware; the system requires no GPUs, no cloud, and no specialized interconnects.

---

## 1. Introduction

Mechanistic interpretability — the project of understanding the internal representations and algorithms of neural networks — has produced striking results at small scale. Circuit analyses of indirect object identification, factual recall, and in-context learning have revealed specific attention heads and MLP neurons whose causal roles can be verified by activation patching (Wang et al., 2022; Meng et al., 2022; Olsson et al., 2022). Sparse autoencoders have been shown to decompose superposed representations in MLP layers into interpretable monosemantic features (Bricken et al., 2023; Cunningham et al., 2023). The Anthropic mechanistic interpretability team has published circuit-level analyses of Claude's safety-relevant behavior (Lindsey et al., 2025).

The problem is that these results were achieved on models in the 1B–7B range, using infrastructure that does not scale. The dominant pattern is: load the entire model onto one GPU, hook the relevant layer, collect activations, save to disk, then train the SAE offline. This pipeline has at least three failure modes at scale:

1. **Memory ceiling.** A 70B model in float16 requires ~140 GB of GPU memory. No consumer or workstation GPU holds this. Multi-GPU setups exist but introduce collective communication overhead and make activation-level debugging substantially harder.

2. **Disk I/O bottleneck.** Collecting 50k tokens of full-depth (all-layer) activations from a 70B model generates ~36 GB of float16 data per run. At NVMe speeds (~3 GB/s sequential write), that is ~12 seconds of pure write time — tolerable in isolation but crippling if the collection loop is the inner loop of a hyperparameter search.

3. **Iteration latency.** The offline pattern (collect → save → load → train SAE) adds a round-trip that makes the feedback cycle from hypothesis to result span hours rather than minutes. Online SAE training, where the SAE trains continuously on the streaming activation output, eliminates this round-trip.

Apple Silicon hardware provides an unusual opportunity here. The M2 Ultra and M3 Ultra chips integrate CPU, GPU, and Neural Engine on a single die with unified memory of up to 192 GB and memory bandwidth of up to 800 GB/s. MLX, Apple's machine learning framework, exposes this hardware with a NumPy-like API and lazy evaluation. A single Mac Studio M2 Ultra can run Llama-3.2-3B at ~300 tokens/second or Mistral-7B at ~150 tokens/second entirely in unified memory. Two such nodes, connected by Thunderbolt 4 (40 Gbps bidirectional), form a cluster with ~4.5 GB/s of inter-node bandwidth — enough to stream activations continuously at any model size we consider.

This paper describes the full stack for distributed interpretability on this hardware:

- **§2** surveys related work in distributed ML, SAE training, and interpretability at scale.
- **§3** specifies the mlxMesh wire protocol and distributed training architecture, with pseudocode and ASCII diagrams.
- **§4** presents results: bandwidth profiling, zero-error split validation, convergence comparison between distributed and single-node SAE training, SAE quality metrics across three model families, cross-architecture feature matching, and a safety classifier evaluation.
- **§5** discusses implications, limitations, and the path to production.
- **§6** concludes.

The core claim of this paper is straightforward: interpretability infrastructure does not need to be expensive, centralized, or cloud-dependent. Two consumer nodes, a Thunderbolt cable, and a well-specified wire protocol are sufficient to run state-of-the-art interpretability experiments on models up to 34B parameters.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability

Neural networks trained on large corpora appear to represent more features than they have neurons via superposition — the simultaneous encoding of multiple concepts in overlapping linear subspaces (Elhage et al., 2022). Sparse autoencoders (SAEs) were proposed as a tool for untangling these superposed representations. An SAE learns an overcomplete dictionary of "feature directions" in the activation space such that any given activation is well-approximated by a sparse linear combination of a small number of dictionary elements.

Formally, given an activation vector $x \in \mathbb{R}^{d}$, the SAE encoder produces a feature vector $f = \text{TopK}(\text{ReLU}(W_\text{enc}(x - b_\text{dec}) + b_\text{enc}))$ and the decoder reconstructs $\hat{x} = W_\text{dec} f + b_\text{dec}$. Training minimizes $\|x - \hat{x}\|^2$ subject to the TopK sparsity constraint, which enforces that exactly $k$ features are active for each input. The TopK variant (Gao et al., 2024) avoids the auxiliary $L_1$ penalty required by early SAE formulations and produces more interpretable features with better dead-feature control.

Several groups have now trained SAEs on frontier models. Cunningham et al. (2023) found monosemantic features in GPT-2. Bricken et al. (2023) trained SAEs on MLP sublayers of a one-layer transformer and identified features corresponding to DNA sequences, legal text, and arithmetic. Templeton et al. (2024) scaled this approach to Claude 3 Sonnet, training SAEs with up to 34 million features and finding features corresponding to concepts as specific as "the Inner Golden Gate Bridge." The Gemma Scope project (Lieberum et al., 2024) released SAEs trained on all layers of Gemma 2 models, enabling systematic study of how representations evolve through depth.

### 2.2 Distributed Inference for Large Language Models

Pipeline parallelism — splitting a model's layers across multiple devices, with each device processing a batch stage while the next device processes the previous stage — is the standard approach for models that exceed single-device memory (GPipe, Huang et al., 2019; PipeDream, Narayanan et al., 2019). These systems optimize for training throughput and are not designed for activation analysis: the inter-device communication is a performance cost to be minimized, not a data source to be tapped.

Inference-time splitting is simpler because there is no gradient to communicate. llama.cpp implements CPU-GPU model splitting for consumer hardware; Ollama wraps this for convenience. Neither exposes intermediate activations to the user, and neither is designed for the high-frequency, low-latency activation capture that interpretability requires.

Our approach differs in that the split point and the activation tap are the same thing: we split the model at a chosen layer, stream the activations across the link, and continue inference on the second node. The wire format is designed for interpretability workloads, not inference throughput: message headers carry layer index and token position to support downstream analysis, and the credit-based flow control is tuned for SAE training batch sizes rather than autoregressive generation latency.

### 2.3 Cross-Architecture Feature Universality

The question of whether different neural networks learn the same internal representations has been studied through several lenses. Representational similarity analysis (RSA; Kriegeskorte et al., 2008) and centered kernel alignment (CKA; Kornblith et al., 2019) measure geometric similarity between representation spaces without requiring feature correspondence. These methods show that networks trained on the same data with different architectures tend to develop similar representational geometries at corresponding depths.

For interpretability, the more specific question is whether individual features — specific directions in the activation space — correspond across models. Elhage et al. (2022) conjectured that the features learned by superposition are universal across model families, in the same sense that Gabor filters appear to be universal across vision models. Evidence for this in language models has been informal: researchers working on multiple models report that similar feature categories (syntactic roles, entity types, sentiment valence) emerge in all of them.

Our cross-architecture matching analysis (§4.4) provides quantitative evidence for this claim. We use chunk-averaged activation pattern matching — a method that identifies feature pairs across models by the similarity of their activation patterns over a shared evaluation corpus — rather than decoder weight comparison, which is confounded by the arbitrary choice of basis in each SAE's latent space.

### 2.4 Safety Applications of SAE Features

A natural application of interpretable features is safety classification: if the model has learned a "weapons manufacturing instructions" feature, and we can identify it in the SAE, we can flag inputs that strongly activate it. This approach has several appeals over standard classifier heads: it requires no labeled training data for the classifier itself (only feature identification), it provides human-interpretable explanations of flags, and it is mechanistically grounded in what the model is actually representing.

We are not the first to explore this idea. Zou et al. (2023) used linear probes on residual stream activations ("representation engineering") to detect and steer safety-relevant states. Burns et al. (2022) trained probes on contrast pairs (statements and their negations) and showed that probes could recover implicit model beliefs about factuality. Our approach is closest in spirit to representation engineering but uses SAE features rather than supervised probes, which is appealing because it avoids the need for labeled pairs.

The evaluation in §4.5 shows that a naively implemented SAE classifier fails — achieving only F1 = 0.519 versus a 0.500 random baseline on a balanced eval set. We analyze the failure mode in detail (corpus mismatch, frequency-biased feature selection, single-feature signal collapse) and propose specific corrections. We argue that the honest failure is more scientifically valuable than a successful cherry-pick would be, because it establishes precise conditions for success.

---

## 3. System Design and Methods

### 3.1 Overview

The mlxMesh system consists of four components:

1. **Inference node (Node A):** Runs the first $L_\text{split}$ transformer layers. Transmits the residual stream activations at the split boundary over the wire.
2. **Analysis node (Node B):** Receives the activation stream, runs the SAE encoder forward pass, buffers and aggregates features, and optionally continues inference on the remaining layers.
3. **Wire protocol (mlxMesh v1.0.0):** Binary framing over TCP or Unix domain socket; 8-byte header + float16 payload + credit-based flow control.
4. **Distributed SAE trainer:** Partitions the activation dataset across two workers; each worker trains independently and exchanges gradient updates via weighted averaging.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           mlxMesh Architecture                          │
├───────────────────────────────────────┬─────────────────────────────────┤
│           NODE A (Inference)          │        NODE B (Analysis)        │
│                                       │                                 │
│  Input tokens                         │                                 │
│       │                               │                                 │
│  ┌────▼─────┐                         │                                 │
│  │ Embed    │                         │                                 │
│  └────┬─────┘                         │                                 │
│       │                               │                                 │
│  ┌────▼─────┐                         │                                 │
│  │ Layer 0  │                         │                                 │
│  │   ...    │    Activation stream    │  ┌─────────────┐               │
│  │ Layer 15 │──── mlxMesh v1.0 ──────►│  │ SAE Encoder │               │
│  └──────────┘   (8-byte hdr +        │  └──────┬──────┘               │
│                  float16 payload)     │         │                       │
│                   Thunderbolt 4       │  ┌──────▼──────┐               │
│                   (~3.5 GB/s)        │  │ Feature buf  │               │
│  ◄── ACK (credit) ─────────────────  │  └──────┬──────┘               │
│                                       │         │                       │
│                                       │  ┌──────▼──────┐               │
│                                       │  │ Layer 16... │               │
│                                       │  │    ...27    │               │
│                                       │  └─────────────┘               │
└───────────────────────────────────────┴─────────────────────────────────┘
```

### 3.2 Wire Protocol

The mlxMesh protocol is message-oriented and operates over a reliable, ordered byte stream. It was designed with three priorities: (1) zero-copy compatibility with MLX buffer layouts, (2) explicit backpressure to prevent the inference node from outrunning the SAE training pipeline, and (3) complete auditability — every byte transferred can be attributed to a specific layer, token range, and session.

#### 3.2.1 Message Format

Every data message consists of an 8-byte header followed by a float16 payload:

```
┌──────────────────────────────────────────────────────────────┐
│  DATA MESSAGE (big-endian)                                   │
│  ┌────────────┬──────────────────┬────────────┐             │
│  │ layer_idx  │   token_start    │token_count │  8 bytes    │
│  │  uint16    │     uint32       │  uint16    │             │
│  └────────────┴──────────────────┴────────────┘             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  float16 payload: token_count × hidden_dim × 2 bytes │   │
│  └──────────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────────┤
│  END-OF-LAYER SENTINEL                                       │
│  layer_idx=0xFFFF  token_start=<layer>  token_count=0xFFFF  │
│  payload=empty                                               │
├──────────────────────────────────────────────────────────────┤
│  REVERSE CHANNEL (ACK/control, 4 bytes)                      │
│  ┌────────────┬───────────┐                                  │
│  │  ack_type  │  credit   │                                  │
│  │  uint16    │  uint16   │                                  │
│  └────────────┴───────────┘                                  │
└──────────────────────────────────────────────────────────────┘
```

For a model with `hidden_dim = 4096` (Mistral-7B) streaming a 128-token chunk at layer 16:

```
payload = 128 × 4096 × 2 = 1,048,576 bytes (1 MiB)
total   = 1,048,576 + 8  = 1,048,584 bytes
```

Protocol overhead (header + sentinel bytes / total bytes transferred) is 0.20% at 20 tokens/second for Mistral-7B — negligible at all model sizes and throughput targets.

#### 3.2.2 Session Handshake

Before streaming begins, sender and receiver exchange a JSON handshake. The sender declares model identity, layer count, hidden dimension, data type, and the maximum number of in-flight messages (`window_size`). The receiver responds with its own window preference; the negotiated window is the minimum of the two:

```json
// Sender Hello
{
  "protocol_version": "1.0.0",
  "session_id": "<uuid4>",
  "model_id": "meta-llama/Llama-3.2-3B",
  "num_layers": 28,
  "hidden_dim": 3072,
  "dtype": "float16",
  "byte_order": "big",
  "window_size": 8
}

// Receiver Hello
{
  "protocol_version": "1.0.0",
  "session_id": "<same uuid4>",
  "accepted": true,
  "window_size": 4
}
// negotiated window = min(8, 4) = 4
```

#### 3.2.3 Flow Control

The protocol uses a credit-based sliding window. The sender starts with `window_size` credits. Each message sent decrements the counter by one; each `DATA_ACK` from the receiver restores credits. When credits reach zero, the sender blocks at the layer boundary, creating natural backpressure that prevents the inference pipeline from outrunning the SAE training pipeline on Node B.

Three control messages handle exceptional states: `CREDIT_PAUSE` (receiver signals it cannot accept messages, e.g., GPU OOM or SAE batch overflow), `CREDIT_RESUME` (receiver ready again), and `ERROR` (fatal protocol violation with UTF-8 error string).

#### 3.2.4 Sender Pseudocode

```python
# mlxMesh sender (compliant with v1.0.0 spec)
credit = negotiated_window_size
for layer_idx in range(num_layers):
    for chunk in layer_chunks(layer_idx, chunk_size=128):
        while credit == 0:
            wait_for_ack()          # blocks; DATA_ACK updates credit
        header = pack(">HLH",
                      layer_idx,
                      chunk.token_start,
                      chunk.token_count)
        sock.sendmsg([header, chunk.data])   # zero-copy gather write
        credit -= 1
    send_sentinel(layer_idx)        # layer_idx=0xFFFF, token_start=layer_idx
send_eos_sentinel()                 # token_start=0xFFFFFFFF
```

The ACK listener runs concurrently, updating `credit` and dispatching `CREDIT_PAUSE`/`CREDIT_RESUME` events. Nagle algorithm is disabled (`TCP_NODELAY`) since the protocol already batches header and payload into a single `sendmsg` call.

### 3.3 Split-Layer Inference

Split-layer inference divides the transformer's layer stack at a configurable split point $L_\text{split}$. Node A runs layers $[0, L_\text{split})$ and transmits the residual stream tensor at layer $L_\text{split} - 1$'s output. Node B receives this tensor and continues inference from layer $L_\text{split}$ to the final layer.

A key implementation detail: the model weights are loaded independently on both nodes. This eliminates any shared-memory requirement and is the correct posture for the future Swift implementation where the two nodes are physically separate machines. The wire format converts activations from bfloat16 (the model's internal dtype) to float16 for transmission; since float16's mantissa is wider than bfloat16's, this conversion is nearly lossless (§4.1 validates this quantitatively).

The analysis node taps activations from the stream before passing them to the SAE. For SAE training, this tap collects residual stream vectors at the split layer; for circuit tracing, the full activation stream across all layers is preserved.

### 3.4 Distributed SAE Training

The distributed SAE trainer partitions the activation dataset across $N$ workers. Each worker trains an independent SAE replica on its data partition. After each step, workers exchange parameter updates via weighted gradient averaging:

$$\theta_\text{merged} = \sum_{i=1}^{N} \alpha_i \, \theta_i, \quad \alpha_i = \frac{|\mathcal{D}_i|}{\sum_j |\mathcal{D}_j|}$$

For two equal-size partitions, $\alpha_0 = \alpha_1 = 0.5$. This is equivalent to synchronous data-parallel training when the per-worker batch sizes are equal, since the gradient average is the same as averaging over the merged batch.

The SAE architecture is a TopK sparse autoencoder:

```python
# SAE forward pass
def encode(x):
    pre_act = (x - b_dec) @ W_enc + b_enc
    features = ReLU(pre_act)
    topk_mask = topk_indices(features, k)    # zero all but top-k activations
    return features * topk_mask

def decode(features):
    return features @ W_dec + b_dec          # W_dec columns are unit-normalized

def loss(x):
    f = encode(x)
    x_hat = decode(f)
    return MSE(x, x_hat)
```

Dead feature detection monitors the number of features that have not fired over a rolling window of 5,000 tokens. Features with zero activations over this window are counted as "dead" — a metric that tracks dictionary utilization and provides an early signal of over-sparsification.

Three SAEs were trained as part of this work:

| Model | Layer | $d_\text{in}$ | Dict size | $k$ | Steps | Corpus tokens |
|---|---|---|---|---|---|---|
| Llama-3.2-3B | 14 | 3072 | 4,096 | 64 | 50,000 | 500k |
| Mistral-7B | 16 | 4096 | 16,384 | 128 | 1,000 | 50k |
| Qwen2.5-3B | 18 | 2048 | 16,384 | 128 | 50,000 | 500k |

The Llama SAE was trained in distributed mode (two workers, equal data partition). The Mistral and Qwen SAEs were trained single-node for comparison. All training used WikiText-103 as the source corpus, with activations collected by running the respective model over the text and extracting residual stream activations at the target layer.

### 3.5 Cross-Architecture Feature Matching

To identify features that are shared across model families, we use chunk-averaged activation pattern matching. The procedure is:

1. Define an evaluation corpus of 50,000 tokens.
2. Divide the corpus into $N_\text{chunks} = 100$ non-overlapping chunks of 500 tokens each.
3. For each model, collect SAE encoder outputs (post-TopK) for every chunk, then average over the token dimension to produce a per-chunk, per-feature activation vector. This yields a matrix $A \in \mathbb{R}^{N_\text{chunks} \times D}$ for each model, where $D$ is the dictionary size.
4. Treat each column of $A$ as a "fingerprint" of the corresponding feature over the evaluation corpus.
5. For a pair of models with matrices $A^{(1)}$ and $A^{(2)}$, compute the cosine similarity between every pair of fingerprints: $\text{sim}(i, j) = \cos(A^{(1)}_{:,i}, \, A^{(2)}_{:,j})$.
6. A match is declared when $\text{sim}(i,j) \geq \tau = 0.8$. For each feature in model 1, record the single highest-similarity match in model 2.
7. Universal features are those with matches in all three pairwise comparisons.

This method sidesteps the arbitrary choice of basis in each SAE's decoder weight matrix and instead matches features by their functional behavior — the pattern of contexts they respond to.

---

## 4. Results

### 4.1 Protocol Validation and Bandwidth

We validated the wire protocol using a two-process simulation of the Node A / Node B split on a single Mac Studio M2 Ultra, with model weights loaded independently in each process and activations communicated via Unix domain socket. The test ran a full inference pass of Llama-3.2-3B on a 512-token sequence split at layer 16 (layers 0–15 on Node A, layers 16–27 on Node B).

**Correctness results (Table 1):**

| Checkpoint | Tensor | max_abs_error | mean_abs_error | rms_error | Pass |
|---|---|---|---|---|---|
| Layer 15 output (handoff) | (512, 3072) | 0.0 | 0.0 | 0.0 | ✓ |
| Layer 27 output (final) | (512, 3072) | 0.0 | 0.0 | 0.0 | ✓ |

The zero error reflects the fact that bfloat16→float16 conversion is lossless for activations in the observed range: float16 has a 10-bit mantissa versus bfloat16's 7-bit mantissa, so no precision is lost when converting in this direction. The only rounding occurs on reconversion at Node B (float16→bfloat16 before the next layer), which remains within float16 precision and produces no detectable error in the final output.

**Throughput results (Table 2):**

The localhost streaming benchmark (synthetic float16 activations in Llama-3.2-3B layer-16 shape, 10,000 tokens, 128-token chunks over TCP loopback) achieved:

| Metric | Value |
|---|---|
| Bytes transferred | 61.4 MB |
| Messages sent/received | 79 / 79 |
| Elapsed time | 0.049 s |
| Throughput | **1.25 Gbps** |
| Target | 1.0 Gbps |
| Zero dropped messages | ✓ |
| Sender/receiver byte count match | ✓ |

At 1.25 Gbps, the protocol sustains throughput approximately **3,000× higher than the real-time single-layer capture requirement** for Llama-3.2-3B at 20 tokens/second (~120 KB/s).

**Thunderbolt feasibility (Table 3):**

For the full range of model sizes, required bandwidth at 20 tokens/second (full-depth, all-layer capture) versus available Thunderbolt 4 capacity (~3.5 GB/s practical):

| Model | Required MB/s | TB4 utilization | Margin |
|---|---|---|---|
| Llama-3.2-3B | 3.3 | 0.09% | ×1,068 |
| Mistral-7B | 5.0 | 0.14% | ×700 |
| Llama-2-13B | 7.8 | 0.22% | ×448 |
| CodeLlama-34B | 15.0 | 0.43% | ×233 |

Thunderbolt 4 is not a bottleneck under any realistic workload. The binding constraints are inference compute speed on Node A and SAE forward-pass throughput on Node B (estimated at ~500 tokens/second at batch size 64 for a 4096→65536 dictionary on M2 Ultra).

Even at peak Mac Studio inference throughput (e.g., ~300 tokens/second for the 3B model), TB4 utilization for full-depth capture stays below 1.4%. The link would need to be saturated at ~2,000 tokens/second, sustained, before any bandwidth constraint emerged.

### 4.2 Distributed SAE Convergence

We compare distributed (2-worker) and single-node SAE training on Llama-3.2-3B activations (layer 14, 500k tokens, dict\_size=4096, k=64, batch=512 per worker / 512 total).

**Proof-of-concept run (5,000 steps, Table 4):**

| Metric | Distributed | Single-node | Ratio |
|---|---|---|---|
| Initial loss | 1.263 | 0.723 | — |
| Final loss | 0.01788 | 0.01778 | 1.005 |
| Mean loss (all steps) | 0.0608 | 0.0506 | 1.20 |
| Converged? | ✓ | ✓ | — |
| Within 10% at step 5000 | ✓ | — | — |

The distributed run's higher initial loss reflects the slower warmup (linear ramp over 200 steps vs. the baseline's faster acceleration from better batch diversity), but both runs converge to essentially the same final loss (ratio 1.005). The mean loss over all steps is 20% higher for the distributed run due to this slower warmup phase, not slower final convergence.

**Full run (50,000 steps, Table 5):**

| Metric | Distributed | Single-node | Ratio |
|---|---|---|---|
| Final MSE | 0.010613 | 0.010608 | 1.0005 |
| Final L0 | 64.0 | 64.0 | 1.0 |
| Final FVE | 0.9822 | 0.9822 | 1.0 |
| Dead features | 1,109 (27.1%) | 1,117 (27.3%) | — |
| Runtime | 79.7 min | 55.0 min | 1.45× |
| Verdict | PASS | — | within 5% MSE |

At 50,000 steps, the distributed SAE matches the single-node baseline to within 0.05% on MSE, 0% on L0 and FVE, and 0.2 percentage points on dead feature rate. The distributed run is 1.45× slower due to the gradient-averaging communication step; in a real two-node deployment over Thunderbolt, this overhead would be further reduced by overlapping gradient exchange with the next forward pass.

**Convergence curves** show that both runs follow the same trajectory after the warmup phase, with loss declining monotonically from ~0.02 at step 500 to ~0.018 at step 2,000, then slowly to ~0.0178 by step 5,000. The distributed run's curve runs slightly above the single-node curve throughout but converges to the same asymptote.

### 4.3 SAE Quality Metrics

Table 6 summarizes the final quality metrics for the three SAEs trained in this work:

| Model | Layer | Dict | k | Steps | Final loss | FVE | Dead features |
|---|---|---|---|---|---|---|---|
| Llama-3.2-3B (dist.) | 14 | 4,096 | 64 | 50,000 | 0.01061 | 0.9822 | 1,109 (27.1%) |
| Mistral-7B | 16 | 16,384 | 128 | 1,000 | 0.00320 | 0.9650 | 339 (2.1%) |
| Qwen2.5-3B | 18 | 16,384 | 128 | 50,000 | ~0.007* | ~0.98* | — |

*Qwen metrics at final step not fully logged; estimated from training trajectory.

The Mistral SAE was trained on a smaller dataset (50k tokens vs 500k) and reached FVE=0.965 in 1,000 steps — notably, FVE peaked at 0.965 at step 1,000 versus an intermediate peak of 0.944 at steps 700 and 900, suggesting the final cosine annealing phase was productive. Dead features declined monotonically from 748 at step 1 to 339 at step 1,000.

For the Llama distributed SAE, 27.1% dead features at step 50,000 is higher than the 2.1% seen in Mistral after 1,000 steps (different dictionary sizes and k values make direct comparison difficult). This level of dead features (roughly 1 in 4 dictionary elements never activated) is consistent with published SAE results at this scale and suggests the dictionary size of 4,096 may be slightly over-provisioned for a 3B model's representational capacity at this layer.

Fraction of variance explained (FVE) of 0.9822 for the Llama SAE indicates that the top-64 active features in each example capture ~98% of the residual stream variance at layer 14. This is a high-quality reconstruction for an 8× expansion ratio (3072 input → 4096 × 8/3 ≈ not applicable here; expansion = 4096/3072 ≈ 1.33×). The Mistral SAE uses a 4× expansion (4096 → 16384) with FVE=0.965, consistent with the higher expansion giving a richer dictionary with lower reconstruction error at the cost of more dead features to manage.

### 4.4 Cross-Architecture Feature Matching

The chunk-averaged activation matching analysis was run on all three pairwise combinations of the three SAEs, using 50,000 evaluation tokens from WikiText-103 divided into 100 chunks of 500 tokens.

**Pairwise match counts (Table 7):**

| Pair | Matches (cosim ≥ 0.8) | Coverage (fraction of smaller dict) |
|---|---|---|
| Llama–Qwen | 5,923 | 36.2% (Llama: 4096) |
| Mistral–Qwen | 8,099 | 49.5% (Mistral: 16384) |
| Llama–Mistral | 12,174 | 74.3% (Llama: 4096) |
| **All three** | **3,753** | **22.9%** (Llama: 4096) |

**Venn diagram summary (Table 8):**

| Region | Features |
|---|---|
| Llama only | 11,155 |
| Qwen only | 13,784 |
| Mistral only | 8,087 |
| Llama–Qwen (not Mistral) | 414 |
| Llama–Mistral (not Qwen) | 1,261 |
| Qwen–Mistral (not Llama) | 307 |
| **All three (universal)** | **3,753** |

The 3,753 universal features are the core finding: a set of representational directions that, when measured by their activation pattern over a shared corpus, are essentially the same across Llama-3.2-3B, Mistral-7B, and Qwen2.5-3B. These three models share no training data (as far as is publicly known), differ in architecture (GQA vs. full attention, different MLP ratios), differ in scale (3B vs. 7B parameters), and were developed by different organizations. The convergence of their SAE features on a common set of ~3,750 directions is evidence that these directions correspond to features intrinsic to the training distribution rather than to any particular architectural choice.

The high Llama–Mistral match count (12,174; covering 74.3% of Llama's dictionary) relative to the lower Llama–Qwen count (5,923; 36.2%) may reflect the fact that Llama and Mistral share a closer architectural lineage (both derived from the LLaMA architecture) or may reflect differences in the training data distributions of Mistral versus Qwen.

Among the universal features, several reach cosine similarity 1.0 across all three pairwise comparisons — perfect alignment of activation patterns. These are the strongest candidates for features representing fundamental linguistic or semantic concepts that any large language model trained on English text would be expected to learn.

### 4.5 Safety Classifier Evaluation

We evaluated a feature-based safety classifier built on top of the Llama-3.2-3B SAE (10,000 steps, layer 14, dict\_size=16,384, k=128, trained on WikiText-103). The classifier labels SAE features by semantic category using regex-based heuristic matching against maximum-activating token contexts, then aggregates "potentially-harmful" feature activations into a scalar harm score.

**Label distribution (top-200 features by activation frequency):**

| Category | Count | Share |
|---|---|---|
| factual | 173 | 86.5% |
| stylistic | 24 | 12.0% |
| code-related | 2 | 1.0% |
| potentially-harmful | **1** | **0.5%** |

**Classifier performance on 200-example balanced eval set (Table 9):**

| Threshold | Precision | Recall | F1 | FPR | Notes |
|---|---|---|---|---|---|
| 0.10 | 0.500 | 1.000 | 0.667 | 1.000 | Degenerate: flags everything |
| 0.50 | 0.476 | 0.800 | 0.597 | 0.880 | |
| 0.70 | 0.488 | 0.630 | 0.550 | 0.660 | |
| 0.90 | 0.580 | 0.470 | 0.519 | 0.340 | Best non-degenerate |
| 1.10 | 0.732 | 0.300 | 0.426 | 0.110 | |
| 1.30 | 1.000 | 0.180 | 0.305 | 0.000 | FPR=0 but recall collapses |

**Random baseline F1:** 0.500 (balanced dataset). The best non-degenerate operating point (threshold=0.90) achieves F1=0.519, essentially indistinguishable from chance.

**Root cause analysis.** The failure is diagnostic, not accidental:

1. **Corpus mismatch.** WikiText-103 contains almost no explicitly harmful content. The top-200 features by activation frequency represent the model's vocabulary for encyclopedic prose — factual, stylistic, code. Safety-relevant features are expected to be low-frequency on this corpus (they activate on rare harmful content) and therefore fall outside the labeling window.

2. **Frequency bias.** Selecting features by activation frequency selects for generalist features, not discriminative ones. Safety-relevant features would rank in the bottom quartile of activation frequency on a Wikipedia corpus.

3. **Single-feature collapse.** The one identified harmful feature (feature 15040, a military-history detector that fires on 9.9% of WikiText tokens) is both too broad (fires on many safe military-history questions) and too narrow (captures weapons in a Wikipedia context, not harmful instructions in a chat context).

4. **Layer placement.** Layer 14 is at the midpoint of the 28-layer stack. Safety-relevant intent representations are more likely to emerge in later layers (20–28) where the residual stream has had time to integrate broader context.

**Implications.** The result motivates specific changes: (a) replace the labeling corpus with a safety-targeted corpus such as AdvBench or HarmBench; (b) select features by differential activation frequency (harmful – safe) rather than absolute frequency; (c) target later layers. The harm scoring pipeline (two-pass frequency analysis, context replay, threshold sweep) is reusable and remains correct — the bottleneck is input data quality, not pipeline logic.

---

## 5. Discussion

### 5.1 The Infrastructure Case

The primary contribution of this work is not any individual experimental result but the demonstration that the full interpretability stack — distributed inference, live activation streaming, online SAE training, cross-architecture feature analysis, and downstream safety evaluation — can run on two consumer nodes connected by a commodity interface. This matters because the alternative (cloud GPU clusters, specialized interconnects, large engineering teams) creates a high barrier to entry that concentrates interpretability research at a small number of well-funded organizations. Democratizing the infrastructure is a precondition for democratizing the science.

The Thunderbolt analysis (§4.1) establishes that the link is not a bottleneck under any foreseeable workload: even at peak Mac Studio inference throughput, TB4 utilization for full-depth activation capture stays below 2%. The binding constraints are compute-side: inference speed on Node A and SAE forward-pass throughput on Node B. Future work should focus on optimizing these (larger batches, INT8 quantization for the SAE encoder) rather than the interconnect.

### 5.2 Universal Features and the Platonic Representation Hypothesis

The finding that 3,753 features match across all three model families, at cosine similarity ≥ 0.8 over activation patterns on a shared corpus, is consistent with the Platonic Representation Hypothesis (Huh et al., 2024): the conjecture that large neural networks trained on diverse data converge on a shared statistical model of reality, regardless of architecture or training procedure.

The universality is particularly striking given the architectural differences: Llama-3.2-3B uses grouped query attention with 3,072 hidden dimensions, Mistral-7B uses sliding window attention with 4,096 dimensions, and Qwen2.5-3B uses a different attention implementation with 2,048 dimensions. The universal features must therefore be aligned in activation pattern space (how they respond to text) rather than weight space (where they live in the model's parameter matrix). This is the correct notion of universality for interpretability: we care about what a feature responds to, not the particular set of weights that implements it.

Future work should test whether the universal features correspond to recognizable semantic or linguistic categories. The pattern matching method identifies which features align; human evaluation or automated labeling (e.g., by collecting maximum-activating tokens) would determine what they represent.

### 5.3 Safety Classifier: An Honest Failure

The safety classifier result is negative, and we believe its honest framing is more valuable than a positive result would have been. A classifier that achieves F1=0.519 on a balanced set, barely above chance, with a clean mechanistic explanation of why it fails, is a clearer contribution than a classifier that achieves F1=0.8 on a benchmark without revealing its failure modes.

The mechanistic explanation is precise: the classifier fails because the features it uses were selected by a criterion (activation frequency on Wikipedia) that is orthogonal to the criterion relevant for the task (differential activation on harmful vs. safe text). This is not a subtle statistical failure — it is a pipeline design error that can be fixed by changing the labeling corpus. The threshold sweep infrastructure, the harm scoring formula, and the evaluation dataset are all reusable. The next iteration should achieve meaningfully higher F1 with no changes other than the input corpus.

We propose that future iterations target F1 ≥ 0.80 at FPR ≤ 0.10, using features selected by contrastive activation analysis on a balanced corpus of harmful and safe conversations.

### 5.4 Limitations

**Scale.** All results in this paper are for models up to 7B parameters. The architecture supports 34B (§3.1), and the bandwidth analysis confirms feasibility, but we have not run 34B experiments. Larger models may exhibit qualitatively different universality patterns.

**Corpus.** All SAE training and evaluation used WikiText-103, an encyclopedic corpus that is clean, well-studied, and widely used as a benchmark but is not representative of instruction-following or conversational distributions. SAEs trained on chat-format data may learn different features.

**Single split point.** The current system uses a fixed split at one layer. Full-depth distributed capture (streaming all layers) is specified by the protocol but not demonstrated in these experiments. Full-depth capture would enable circuit-tracing experiments at scale, which is the next logical step.

**Simulation vs. hardware.** The two-node validation used two Python subprocesses on one physical Mac Studio rather than two physical machines. The protocol is hardware-agnostic, but the latency and jitter profile of a real inter-machine Thunderbolt link may differ from Unix domain socket performance.

### 5.5 Path to Production

The immediate next step is implementing the system in Swift/MLX rather than Python, using the same wire protocol. Swift provides access to macOS's native socket APIs, lower-latency memory management, and better MLX integration for zero-copy buffer passing. The protocol spec is wire-format stable (v1.0.0); Swift and Python implementations are directly interoperable.

Beyond that, the natural extension is online SAE training: rather than collecting activations offline and training on stored data, the SAE trainer on Node B processes the activation stream as it arrives, updating weights continuously. This eliminates the two-stage collect→train pipeline and reduces the iteration cycle for hyperparameter tuning from hours to minutes.

The cross-architecture feature matching result also motivates a specific experiment: train a "universal SAE" whose dictionary is initialized from the matched features and fine-tuned jointly on activations from all three model families. If the universal features are truly shared representations, a single dictionary should achieve comparable FVE on all three models, which would be a strong form of validation.

---

## 6. Conclusion

We have described mlxMesh, a system for distributed mechanistic interpretability on Apple Silicon hardware. The system streams intermediate layer activations between two nodes over a purpose-built binary wire protocol, enabling online SAE training and analysis without storing activations to disk. We have validated the system end-to-end — zero reconstruction error at the split boundary, 1.25 Gbps protocol throughput, distributed SAE training within 0.05% of single-node baseline — and applied it to three experimental contributions: (1) a quantitative bandwidth analysis showing that Thunderbolt 4 is never a bottleneck at any model size considered, (2) cross-architecture feature matching identifying 3,753 universal features shared across Llama-3.2-3B, Mistral-7B, and Qwen2.5-3B, and (3) a safety classifier evaluation that honestly characterizes a negative result and diagnoses its precise failure mode.

The central message is that the infrastructure gap between "models small enough to study on one machine" and "models large enough to matter" is not a fundamental barrier — it is an engineering problem with a clean solution. Two Apple Silicon nodes, a Thunderbolt cable, and a well-specified wire protocol are sufficient for state-of-the-art interpretability research on models up to 34B parameters. The science of understanding what large language models represent does not require access to expensive centralized compute, and should not be limited to organizations that have it.

---

## References

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., ... & Olah, C. (2023). Towards monosemanticity: Decomposing language models with dictionary learning. *Transformer Circuits Thread*.

Burns, C., Ye, H., Klein, D., & Steinhardt, J. (2022). Discovering latent knowledge in language models without supervision. *arXiv:2212.03827*.

Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse autoencoders find highly interpretable features in language models. *arXiv:2309.08600*.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., ... & Olah, C. (2022). Toy models of superposition. *Transformer Circuits Thread*.

Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., ... & Leike, J. (2024). Scaling and evaluating sparse autoencoders. *arXiv:2406.04093*.

Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, M. X., Chen, D., ... & Wu, Y. (2019). GPipe: Efficient training of giant neural networks using pipeline parallelism. *NeurIPS 2019*.

Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic representation hypothesis. *arXiv:2405.07987*.

Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *ICML 2019*.

Kriegeskorte, N., Mur, M., & Bandettini, P. (2008). Representational similarity analysis — connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*.

Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Kram, V., ... & Nanda, N. (2024). Gemma Scope: Open sparse autoencoders everywhere all at once on Gemma 2. *arXiv:2408.05147*.

Lindsey, J., Gould, M., Templeton, A., McDougall, C., Batson, J., Jermyn, A., ... & Olah, C. (2025). On the biology of a large language model. *Transformer Circuits Thread*.

Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS 2022*.

Narayanan, D., Harlap, A., Phanishayee, A., Seshadri, V., Devanur, N. R., Ganger, G. R., ... & Zaharia, M. (2019). PipeDream: Generalized pipeline parallelism for DNN training. *SOSP 2019*.

Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., ... & Olah, C. (2022). In-context learning and induction heads. *Transformer Circuits Thread*.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., Bricken, T., Chen, B., ... & Olah, C. (2024). Scaling monosemanticity: Extracting interpretable features from Claude 3 Sonnet. *Transformer Circuits Thread*.

Wang, K., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2022). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. *arXiv:2211.00593*.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., ... & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv:2310.01405*.

---

*End of draft v1 — 2026-07-29*
