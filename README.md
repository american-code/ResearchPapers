# Research Papers — American Code Labs

Mechanistic interpretability and quantization research, run end to end on consumer
Apple Silicon. No GPU cluster, no cloud. Four published papers and two in progress.

Every result below is measured and reproducible from this repository. Where a result is
negative, it is reported as the finding.

---

## Published

Cite the **concept DOI** — it always resolves to the latest version. Version DOIs are
pinned to one deposit and go stale on revision.

### What does MLX 4-bit cost?
*A Controlled Audit of Quantization for Code Models on Apple Silicon*
`papers/mlx-quantization/` · [10.5281/zenodo.21906720](https://doi.org/10.5281/zenodo.21906720)

Five instruction-tuned code models, 1.5B to 8B, on HumanEval+ and MBPP+, comparing bf16
against 4-bit weights quantized from those same weights. One harness, one machine, one
variable.

- 4-bit loses on **220** problems and wins on **125** (McNemar *p* < 10⁻⁶)
- The cost **shrinks as models grow**: +0.52 points per billion parameters. Phi-4-mini
  (3.8B) loses 6.5 points; Qwen2.5-Coder-7B loses 1.2
- Activation-aware quantization (AWQ) does not recover the loss and is statistically
  indistinguishable from the uncalibrated default (133–116, *p* = 0.31) while costing up
  to 1h51m to build
- Differencing a local 4-bit score against a vendor's published bf16 figure is unsound:
  the harness-mismatch term is the same order as the effect and has no stable sign

### Efficient Mechanistic Circuit Analysis of Open-Weight LLMs on Apple Silicon
`papers/circuit-tracing/` · [10.5281/zenodo.21906706](https://doi.org/10.5281/zenodo.21906706)

IOI circuits in Llama-3.2-3B and Pythia-1.4B, identified by activation patching and mean
ablation and validated as sets.

- Both models concentrate the task far more than GPT-2 Small: one head carries **22–28%**
  of clean logit difference, against a distributed 26-component circuit
- Faithful but incomplete — 0.663 (Llama) and 0.854 (Pythia), with completeness gaps of
  0.28 and 0.22
- **Faithfulness on one sentence frame is not generality.** On 15 templates it falls to
  **0.228** and **−0.028** while both models still perform the task above 90% of their
  original logit difference
- **Cross-model depth-position conservation is indistinguishable from chance** (null 59%,
  *p* = 0.098; layer-band-restricted null 69%, *p* = 0.32)

### TopK Sparse Autoencoders Across Three Model Architectures
`papers/sae-comparison/` · [10.5281/zenodo.21906712](https://doi.org/10.5281/zenodo.21906712)

TopK SAEs trained to a common specification on Llama-3.2-3B, Mistral-7B-v0.3 and
Qwen2.5-3B, with cross-architecture feature matching calibrated against a permutation null.

- **A positive control establishes the method's ceiling.** Two SAEs differing only in
  training seed agree on **8%** of matchable features at 108× the null. The instrument
  works; 8% is the most any study of this kind can report
- Cross-architecture agreement is **0.18–0.33%** — near-zero against a working instrument
- **Standard quality metrics are blind.** Loss, FVE, L0 and dead-feature count cannot
  distinguish dictionaries that disagree about 92% of their content
- Dictionary collapse and dense-feature degeneracy severe enough to invalidate
  frequency-based feature selection

### Distributed Mechanistic Interpretability at Scale (ActStream)
`papers/distributed-interp/` · [10.5281/zenodo.21906710](https://doi.org/10.5281/zenodo.21906710)

A two-node system that splits inference and analysis, streams activations over a
purpose-built binary protocol, and trains SAEs on the stream without staging to disk.

- Thunderbolt sustains **2.150 GB/s** — 86% of its negotiated rate, 26× the LAN path,
  and only 3.0× below in-machine memory
- Tail latency is the consequential figure: p99 within 1.5× of median, where the LAN
  spreads to 10.5× with a 52 ms tail
- Bit-exact split boundary; two-worker distributed SAE training matches the single-node
  baseline to within **0.05%** on MSE
- A 40-feature safety classifier reaches ROC-AUC **0.7206** held out. An LLM judge scores
  higher (0.8545), but the two agree on only 54% of examples (κ = 0.083) — close to
  orthogonal failure modes

---

## In progress

Both drafts render every unmeasured value as a red placeholder and carry a title-page
banner counting them. No number in either is estimated or illustrative.

### A Perturbation Ladder for SAE Dictionaries
`papers/sae-perturbation-ladder/`

How much of a learned dictionary survives seed, optimizer and precision changes — one
matching procedure, one denominator, so the rungs are comparable to each other. All arms
50,000 steps on the same corpus.

| rung | Pearson | cosine |
|---|---|---|
| ceiling — seed 42 vs 123 | 9.07% (123×) | 10.99% (150×) |
| precision — bf16 vs local 4-bit | 7.70% (104×) | 10.30% (138×) |
| optimizer — Adam vs Muon | *running* | *running* |
| floor — cross-architecture | 0.18–0.33% | — |

4-bit quantization retains 85% (Pearson) or 94% (cosine) of the seed ceiling: it perturbs
the dictionary somewhat more than reseeding, and both sit far above the architecture floor.
Complements Duan (arXiv:2606.03002), which holds a dictionary fixed and asks whether its
features still fire; this asks whether the dictionary one would *learn* is the same.
The two use different denominators and are not numerically comparable.

### Frame Dependence: Circuit Robustness Under Task-Preserving Shift
`papers/circuit-robustness/`

Whether a circuit discovered on diverse sentence frames transfers to frames it was not
discovered on. Published results conflict, and the candidate reconciliation is that the
*discovery* set decides it — which requires a shift holding template **count** fixed while
changing only their identity. Twelve matched-diversity dataset pairs across four task
families (IOI, factual recall, subject–verb agreement, induction) and three shift axes.

---

## Backlog

Parked ideas, with prior-art status and cost, are in
[docs/research-backlog.md](docs/research-backlog.md). Nothing there is started.

---

## Layout

```
ResearchPapers/
├── papers/
│   ├── mlx-quantization/          published
│   ├── circuit-tracing/           published
│   ├── sae-comparison/            published
│   ├── distributed-interp/        published
│   ├── sae-perturbation-ladder/   draft
│   └── circuit-robustness/        draft
├── data/
│   ├── ioi/                       IOI datasets, patching and ablation harnesses
│   ├── circuit-v2/                faithfulness, minimality, completeness
│   ├── circuit-robustness/        D/D' shift pairs and protocol implementation
│   ├── sae-runs/                  SAE training scripts, checkpoints, logs
│   ├── sae-analysis/              feature matching, null calibration
│   ├── seed-stability/            seed control (original corpus)
│   ├── seed-stability-newcorpus/  seed control (matched corpus)
│   ├── quant-interp/              bf16 vs 4-bit dictionary matching
│   ├── slm-benchmark/             quantization benchmark harness
│   ├── cluster-gate/              distributed-inference determinism gate
│   ├── safety-classifier/         safety classifier training and eval
│   └── activations/               collected residual-stream activations
├── figures/
└── docs/
```

## Key scripts

| Script | Purpose |
|---|---|
| `data/sae-runs/train_sae.py` | TopK SAE trainer. Trained every Adam arm; sha256 `b701c090…` |
| `data/sae-runs/train_sae_muon.py` | Optimizer-capable variant. Kept separate so the Adam arms and the repo trainer stay byte-identical; see `trainer-muon.diff` |
| `data/sae-analysis/quant_interp_match.py` | Reciprocal feature matching with permutation-calibrated τ |
| `data/circuit-robustness/circuit_lib.py` | Both architectures × four intervention modes, dataset-agnostic |
| `data/circuit-robustness/generate_shift_pairs.py` | D/D′ generation with single-token, length-parity and template-balance constraints |
| `data/circuit-robustness/discover_circuit.py` | Patching + ablation sweep → circuit file |
| `data/ioi/run_circuit_faithfulness.py` | Faithfulness, minimality, completeness |
| `data/slm-benchmark/evalplus_generate.py` | Greedy generation for the quantization audit |

## Reproducibility notes

**Activation collection is a versioned artifact, not a script.** Activations behind the
originally published SAEs cannot be regenerated from the released collection code — the
metadata schemas differ and a hash over the leading 3.072 × 10⁹ bytes does not match.
Every arm in the perturbation ladder is therefore trained on a corpus regenerated and
committed as part of that work.

**Two harness requirements**, both silent when violated. The mean-ablation reference must
be computed over the full example set, not per batch; and sequences must be padded to one
global length, since the cached mean carries a sequence dimension. Datasets of
near-uniform length hide both.

**Quantized arms are converted locally**, never downloaded. A community upload cannot be
verified as to base revision or conversion settings, and a provenance gap in the variable
under test makes the experiment unreadable.

All runs are seeded (42 unless stated) on a single Apple M1 Max with 32 GB unified memory
via MLX.

See [CORRECTIONS.md](CORRECTIONS.md) for the 2026-07-30 audit of the papers against their
underlying data.
