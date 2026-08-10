# Activation Streaming Bandwidth Analysis

Version: 1.0  
Date: 2026-07-29  
Context: ActStream two-node Mac Studio topology over Thunderbolt

---

## 1. Scope

This document quantifies the data volumes and bandwidth demands of real-time activation streaming for transformer models in the 3B–34B parameter range. The target deployment is two Mac Studio nodes connected directly via Thunderbolt cable: one node runs inference; the other captures, buffers, and analyzes the activation stream (e.g. for SAE training or circuit tracing).

All calculations assume the ActStream wire protocol (8-byte header + float16 payload per chunk) documented in `activation-streaming-protocol.md`. Data type is float16 throughout (2 bytes per element). Streaming residual-stream activations only; attention patterns and MLP intermediates are not included unless noted.

---

## 2. Model Architecture Parameters

These are the actual or canonical configurations for each size class. The 3B and 7B entries match models currently in use on this project.

| Size | Representative model | `hidden_dim` | `num_layers` | Mem (fp16) |
|------|---------------------|:------------:|:------------:|:----------:|
| 3B   | Llama-3.2-3B        | 3072         | 28           | ~6 GB      |
| 7B   | Mistral-7B          | 4096         | 32           | ~14 GB     |
| 13B  | Llama-2-13B         | 5120         | 40           | ~26 GB     |
| 34B  | CodeLlama-34B       | 8192         | 48           | ~68 GB     |

---

## 3. Per-Layer Data Volume

One residual-stream activation vector per token per layer:

```
bytes_per_token_per_layer = hidden_dim × 2
```

| Model | hidden_dim | Bytes / token / layer | KiB |
|-------|:----------:|:---------------------:|:---:|
| 3B    | 3072       | 6,144                 | 6.00 |
| 7B    | 4096       | 8,192                 | 8.00 |
| 13B   | 5120       | 10,240                | 10.00 |
| 34B   | 8192       | 16,384                | 16.00 |

For a single-layer tap (e.g. layer 16 only, as used in the current Mistral SAE run), multiply the table above by 1.

---

## 4. Full-Depth Capture Volume

Streaming activations from all layers for a single token:

```
bytes_per_token_full = hidden_dim × 2 × num_layers
```

| Model | Bytes / token (all layers) | KiB    | MiB (approx) |
|-------|:--------------------------:|:------:|:------------:|
| 3B    | 172,032                    | 168.00 | 0.164        |
| 7B    | 262,144                    | 256.00 | 0.250        |
| 13B   | 409,600                    | 400.00 | 0.391        |
| 34B   | 786,432                    | 768.00 | 0.750        |

**Batch sizing examples** (full-depth, float16):

| Model | 1k tokens | 10k tokens | 50k tokens |
|-------|:---------:|:----------:|:----------:|
| 3B    | 168 MiB   | 1.64 GiB   | 8.20 GiB   |
| 7B    | 256 MiB   | 2.50 GiB   | 12.5 GiB   |
| 13B   | 390 MiB   | 3.81 GiB   | 19.1 GiB   |
| 34B   | 750 MiB   | 7.32 GiB   | 36.6 GiB   |

---

## 5. Required Bandwidth at 20 Tokens / Second

Real-time generation at 20 tok/s means all layer activations for each token must be transmitted within each 50 ms window.

```
bandwidth = bytes_per_token_full × tokens_per_second
```

### 5.1 Full-depth capture (all layers)

| Model | MB/s   | Gbps   | Token window (50 ms) |
|-------|:------:|:------:|:--------------------:|
| 3B    | 3.28   | 0.0262 | 168 KiB to transmit  |
| 7B    | 5.00   | 0.0400 | 256 KiB to transmit  |
| 13B   | 7.81   | 0.0625 | 400 KiB to transmit  |
| 34B   | 15.00  | 0.1200 | 768 KiB to transmit  |

### 5.2 Single-layer tap (one layer only)

| Model | MB/s   | Gbps    |
|-------|:------:|:-------:|
| 3B    | 0.117  | 0.00094 |
| 7B    | 0.156  | 0.00125 |
| 13B   | 0.195  | 0.00156 |
| 34B   | 0.313  | 0.00250 |

### 5.3 Scaled throughput: bandwidth at N tokens/sec (full-depth, 7B)

| Tokens/sec | MB/s  | Gbps  | Use case                         |
|:----------:|:-----:|:-----:|----------------------------------|
| 20         | 5.0   | 0.040 | Interactive generation, realtime |
| 100        | 25.0  | 0.200 | Fast batch prefill               |
| 500        | 125   | 1.00  | Sustained SAE training run       |
| 2000       | 500   | 4.00  | Near-peak Mac Studio throughput  |
| 5000       | 1250  | 10.0  | Theoretical upper bound          |

---

## 6. Protocol Overhead

Using the ActStream wire format: 8-byte header + 8-byte sentinel per layer per chunk.

At 20 tok/s, with one chunk per layer per token (chunk size = 1 token):

| Model | Messages/sec | Header bytes/sec | Sentinel bytes/sec | Overhead fraction |
|-------|:------------:|:----------------:|:------------------:|:-----------------:|
| 3B    | 560          | 4,480            | 4,480              | 0.26%             |
| 7B    | 640          | 5,120            | 5,120              | 0.20%             |
| 13B   | 800          | 6,400            | 6,400              | 0.16%             |
| 34B   | 960          | 7,680            | 7,680              | 0.10%             |

Protocol overhead is negligible at all model sizes and throughput targets. Batching tokens into larger chunks (e.g. 128 tokens/chunk as recommended in §10.1 of the protocol spec) reduces message rate by 128× and drops overhead to an unmeasurable level.

---

## 7. Thunderbolt Feasibility Analysis

### 7.1 Available bandwidth

| Interface       | Bidirectional | Usable (practical) | Notes                          |
|-----------------|:-------------:|:------------------:|--------------------------------|
| Thunderbolt 3   | 40 Gbps       | ~4.5 GB/s          | Mac Studio M1/M2 Studio ports  |
| Thunderbolt 4   | 40 Gbps       | ~4.5 GB/s          | Same electrical spec as TB3    |
| Thunderbolt 5   | 120 Gbps      | ~13 GB/s           | Not yet on current Mac Studios |
| 10GbE (backup)  | 10 Gbps       | ~1.1 GB/s          | Built-in on Mac Studio Ultra   |

Mac Studio M2 Ultra ships with Thunderbolt 4. Effective unidirectional data transfer rate in practice is approximately **3–4 GB/s** (accounting for PCIe encapsulation overhead and OS socket stack).

### 7.2 Utilization at 20 tok/s

```
utilization = bandwidth_required / TB4_practical (3.5 GB/s)
```

| Model | Required MB/s | TB4 utilization | Headroom factor |
|-------|:-------------:|:---------------:|:---------------:|
| 3B    | 3.28          | 0.094%          | ×1068           |
| 7B    | 5.00          | 0.143%          | ×700            |
| 13B   | 7.81          | 0.223%          | ×448            |
| 34B   | 15.00         | 0.429%          | ×233            |

**Thunderbolt is ~3 orders of magnitude overprovisioned** for activation streaming at 20 tok/s across all model sizes. The link would need to be running at 2,000–7,000 tokens/sec before TB4 utilization reaches even 10%.

### 7.3 Utilization at peak Mac Studio throughput

Estimated peak generation throughput on Mac Studio M2 Ultra (MLX, single-node):

| Model | Estimated peak tok/s | Required MB/s | TB4 utilization |
|-------|:--------------------:|:-------------:|:---------------:|
| 3B    | ~300                 | 49            | 1.4%            |
| 7B    | ~150                 | 37.5          | 1.1%            |
| 13B   | ~80                  | 31.3          | 0.90%           |
| 34B   | ~30                  | 22.5          | 0.64%           |

Even at peak inference throughput, TB4 utilization stays below 2%. The link is not a bottleneck under any realistic workload.

### 7.4 Latency

Thunderbolt's PCIe tunnel latency is approximately 1–5 µs at the hardware level. With macOS TCP or Unix domain socket overhead, end-to-end per-message latency is approximately 10–100 µs.

At 20 tok/s (50 ms/token), all layer activations must cross the link within the token window:

| Model | Data per token | Time to transmit @ 3.5 GB/s | Budget (50 ms) | Margin   |
|-------|:--------------:|:---------------------------:|:--------------:|:--------:|
| 3B    | 168 KiB        | 0.047 ms                    | 50 ms          | ×1064    |
| 7B    | 256 KiB        | 0.071 ms                    | 50 ms          | ×704     |
| 13B   | 400 KiB        | 0.111 ms                    | 50 ms          | ×450     |
| 34B   | 768 KiB        | 0.213 ms                    | 50 ms          | ×235     |

Transmission time is under 0.25 ms for any model. The token budget is consumed by inference compute, not data transfer.

---

## 8. Actual Bottlenecks

In order of likelihood, the real constraints are:

### 8.1 Inference compute speed

The inference node (Mac Studio 1) must generate activations faster than the analysis node can consume them. For SAE training targeting 50k tokens at 20 tok/s, collection takes ~42 minutes. This is the primary throughput ceiling.

### 8.2 Memory bandwidth for activation readout

Reading activations from MLX GPU buffers into a transmit buffer requires memory bandwidth. On M2 Ultra (800 GB/s unified memory bandwidth), reading 768 KiB per token (34B) costs:

```
768 KiB / 800 GB/s ≈ 0.001 ms per token
```

Memory bandwidth is not a bottleneck.

### 8.3 Receiver-side processing

The analysis node must deserialize, validate, and enqueue activations fast enough to avoid triggering CREDIT_PAUSE. For SAE feature matching or online SAE training, the forward pass of the SAE itself may become the binding constraint.

Estimated SAE forward throughput (4096 → 65536 features, MLX):

- At batch size 64 tokens: ~500 tok/s
- At batch size 1 token: ~50 tok/s (single-sample overhead dominates)

Recommendation: buffer 64–256 tokens on the receiver before dispatching SAE batches.

### 8.4 34B model on single Mac Studio

A 34B model at fp16 requires ~68 GB of weights. Mac Studio M2 Ultra (192 GB) can hold this comfortably; Mac Studio M1 Ultra (128 GB) can too (128 - 68 = 60 GB for KV cache and activations). Both node configurations are viable for 34B inference.

---

## 9. Deployment Configurations

### 9.1 Minimal (single-layer SAE training)

```
Mac Studio A (inference)  ──TB4──  Mac Studio B (SAE analysis)
  Llama-3.2-3B                       SAE: 3072 → 49152 features
  Layer 16 tap only                  ~120 KB/s data rate
  20 tok/s generation                Buffer: 64-token batches
```

Data rate: **~120 KB/s**. Any connection works, including 1GbE.

### 9.2 Full-depth capture for circuit tracing (7B)

```
Mac Studio A (inference)  ──TB4──  Mac Studio B (circuit tracing)
  Mistral-7B                         All 32 layers captured
  All layers streamed                Per-token: 256 KiB
  20 tok/s generation                5 MB/s sustained, 50 ms window
```

Data rate: **5 MB/s**. Comfortably within any link including 1GbE.

### 9.3 High-throughput SAE dataset collection (7B)

```
Mac Studio A (inference)  ──TB4──  Mac Studio B (disk write + SAE)
  Mistral-7B                         50k tokens, layer 16 only
  Layer 16 tap only                  Total: ~1.2 GB
  ~150 tok/s max                     Expected runtime: ~5.5 min
```

Data rate: **~1.2 GB / 5.5 min ≈ 3.6 MB/s**. No link constraint.

---

## 10. Summary

| Model | Per-token/layer | Per-token full | 20 tok/s BW | TB4 utilization |
|-------|:---------------:|:--------------:|:-----------:|:---------------:|
| 3B    | 6 KiB           | 168 KiB        | 3.3 MB/s    | 0.09%           |
| 7B    | 8 KiB           | 256 KiB        | 5.0 MB/s    | 0.14%           |
| 13B   | 10 KiB          | 400 KiB        | 7.8 MB/s    | 0.22%           |
| 34B   | 16 KiB          | 768 KiB        | 15.0 MB/s   | 0.43%           |

**Verdict: Thunderbolt is not a constraint for any configuration considered here.** Even 34B full-depth capture at 20 tok/s consumes less than 0.5% of TB4 capacity. The binding constraints are inference speed on the sending node and SAE forward-pass throughput on the receiving node. A 10GbE fallback would also be adequate for all use cases.

The activation streaming link could absorb 200–1000× the workloads described here before approaching saturation. Bandwidth planning effort should be directed at receiver-side memory management and batch sizing for the SAE pipeline, not the wire.

---

*See also: `activation-streaming-protocol.md` for wire format details.*
