# Outline — What Does MLX 4-bit Cost? A Controlled Audit for Code Generation

Status: **skeleton**, started 2026-08-08. Data collection in progress (2 of 5 pairs complete).

## Claim

Practitioners running local code models on Apple Silicon overwhelmingly use
`mlx-community/*-4bit` weights, and have no published figure for what that
quantization costs on code benchmarks. We measure it under a controlled harness, and
show that the obvious way to estimate it — subtracting a measured 4-bit score from a
vendor's published bf16 number — overstates the cost by roughly 2.4x because the
harness-mismatch term is larger than the effect.

## Why this gap exists

Three facts, each verified during this work rather than assumed:

1. Apple publishes a bf16→q4 delta for exactly two models on exactly one benchmark
   (MMLU Pro), and no code benchmark. `mlx-community` conversion cards publish no
   accuracy numbers at all.
2. `mlx_lm.convert --q-bits 4` is naive round-to-nearest with no calibration, and it
   quantizes token embeddings and `lm_head` as well — a weaker configuration than the
   GPTQ/AWQ methods behind the published ~0.5–2 pt loss figures.
3. EvalPlus, the standard harness, **does not run on macOS out of the box**: it caps
   executed code with `RLIMIT_AS`, which Darwin rejects, and every problem errors.
   Every MLX user is on macOS. This is a plausible partial explanation for why the
   intersection is empty.

## Contributions

- **C1.** First paired bf16-vs-MLX-4-bit measurement on code benchmarks, same harness,
  same machine, one variable changed. [2/5 pairs complete]
- **C2.** Quantification of the vendor-subtraction confound: the naive method reports
  −6.1 pts where the controlled comparison gives −2.5.
- **C3.** Evidence that `mlx-community` uploads are **not** equivalent to local
  `mlx_lm convert` output — the upload scored higher. [1 model; needs replication]
- **C4.** A harness-defect catalogue: five independent ways a code-eval harness
  silently scores correct answers as failures, with measured magnitudes (one protocol
  choice moved MBPP pass@1 by 59 points).
- **C5.** An uncontaminated 26-task domain suite drawn from real research code, which
  discriminates far more sharply than HumanEval across this model range.
- **C6.** Determinism result: local MLX greedy decoding is bit-reproducible across
  processes, licensing n=1. The 47.6% non-determinism reported at temperature=0 by
  Ouyang et al. (TOSEM 2025) is an API-serving artifact.

## Section plan

1. Introduction — the gap, and why it is not merely unmeasured but hard to measure
2. Related work — EvalPlus, contamination, quantization literature, the empty cell
3. Methods — harness, determinism gate, provenance control, the domain suite
4. Results: quantization cost — the paired tables
5. Results: the confound — vendor subtraction vs controlled comparison
6. Results: provenance — community uploads vs local conversion
7. Harness fragility — the defect catalogue as a methodological finding
8. Discussion — what a practitioner should conclude; limits
9. Threats to validity

## Open items before submission

- [x] Complete the bf16/4-bit matrix — all 5 pairs measured under 0.29.1. Pooled
      p < 10⁻⁶; cost shrinks with model size at +0.52 pts per B.
- [x] Add a calibrated arm. AWQ measured on 3 of 5 models under a unified 0.31.3.
      **Calibration does not close the gap** — AWQ still strongly below bf16
      (p=0.0018) and not significantly different from RTN head-to-head
      (119 vs 103, p=0.31).
      NOTE: an interim read of Granite-8B's HumanEval cell alone suggested AWQ
      "actively hurts"; its MBPP cell reversed that (+0.3 vs bf16). Per-benchmark
      variance exceeds the method difference. Do not draw directional quantizer
      conclusions from one benchmark.
- [x] Phi-4-mini AWQ is impossible — `mlx_lm.quant.awq` does not support phi3. The
      model with the largest RTN cost is the one we cannot test calibration on.
- [x] Establish that the runtime version is not a confound (162/164 identical
      verdicts, p=0.50), so the 0.29.1 matrix stands as a cross-version replication.
- [ ] AWQ for phi4mini, qwen7b, granite8b (running). These are the larger models where
      RTN cost least — the place calibration would most plausibly show an effect.
- [ ] `peak_memory_gb` still untrustworthy — reports 4-bit using *more* memory than
      bf16 on qwen3b (6.55 vs 6.47 GB) while qwen1.5b is sane (1.65 vs 3.48). Omitted
      from the draft; fix the collection path or drop the claim entirely.
- [ ] C3 (community uploads beat local conversion) is now in tension with §4.4's
      head-to-head. Either replicate it on a second model or downgrade it from a
      contribution to an observation.
- [ ] GPTQ/DWQ remain unmeasured. Decide whether "calibration does not help" needs
      more than one calibrated method to stand.
- [ ] Abstract, introduction, related work, discussion still `[SKELETON]`.
