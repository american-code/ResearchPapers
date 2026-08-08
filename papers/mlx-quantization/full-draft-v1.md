# What Does MLX 4-bit Cost? A Controlled Audit of Quantization for Code Generation on Apple Silicon

**DRAFT — SKELETON. Started 2026-08-08.**

**Data completeness: 2 of 5 bf16/4-bit pairs measured.** Every number below with a value
is measured and traceable to an EvalPlus per-problem results file. Cells marked
`[PENDING]` have no data — they are not estimates, defaults, or placeholders, and no
number should be inferred for them. Sections marked `[SKELETON]` are structure only.

---

## Abstract

`[SKELETON]` — write last, once the remaining arms land.

Target claim: on a 32 GB Apple Silicon host, converting a small code model to 4-bit
with MLX's default round-to-nearest quantizer costs a small but statistically
detectable amount of accuracy on HumanEval and MBPP, while roughly doubling throughput.
The naive way to estimate that cost — differencing a locally measured 4-bit score
against the vendor's published bf16 figure — overstates it by about 2.4x, because
harness mismatch is larger than the effect being measured.

---

## 1. Introduction

`[SKELETON]`

Framing beats to hit:

- Local code models on Apple Silicon are consumed almost entirely as
  `mlx-community/*-4bit` weights. That is the artifact practitioners actually run.
- No published figure states what that costs on a code benchmark. Apple's only
  bf16→q4 accuracy table covers two models on MMLU Pro; the conversion cards carry no
  accuracy numbers; the one MLX effort that does run HumanEval compares against
  uniform-4bit rather than bf16.
- The question is not merely unanswered but structurally hard to answer: the standard
  harness does not run on the only operating system these weights target (§3.2).
- The obvious shortcut — compare your 4-bit run to the model card — is wrong by more
  than the effect size (§5).

---

## 2. Related Work

`[SKELETON]`

- **Benchmark adequacy.** EvalPlus (Liu et al., NeurIPS 2023) adds 80x/35x more tests
  to HumanEval/MBPP; pass@k falls 19.3–28.9% across 26 models. We report both base and
  plus variants throughout for exactly this reason.
- **Contamination.** HumanEval and MBPP are demonstrably present in common pretraining
  corpora (12.2% verbatim in The Pile, 18.9% in The Stack; 50–63% semantic overlap).
  This bounds what an absolute score means, but is largely *neutralized* by our design:
  both arms are the same model, so contamination cancels in the paired difference.
- **Quantization and code.** Calibrated 4-bit weight-only methods (AQLM, GPTQ-style
  W4A16) cost ~0.5–2 pass@1 points in the 1B–34B range. Whether code is
  disproportionately sensitive is contested. Crucially, MLX's default is *not* one of
  these methods (§3.3).
- **Scaling.** Low-bit degradation grows with training tokens and shrinks with model
  size, placing modern heavily-trained 1–8B models in the worst quadrant.
- **The empty cell.** No published HumanEval/MBPP for stock `mlx-community` 4-bit
  weights against a bf16 baseline, from Apple, mlx-community, or a third party.

---

## 3. Methods

### 3.1 Design

One variable. For each model we evaluate two arms — bf16 and 4-bit — through an
identical harness, on the same host, with identical decoding. Vendor published numbers
are used **only** as a sanity check that our harness reproduces the standard protocol,
never as a term in any difference.

### 3.2 Harness

EvalPlus 0.3.1 (HumanEval+ 164 tasks, MBPP+ 378 tasks), rather than a bespoke harness.
This is a deliberate reversal: we began with a custom implementation and abandoned it
after finding five independent defects (§7).

**EvalPlus does not run on macOS as shipped.** `reliability_guard` caps executed code
via `RLIMIT_AS`, which Darwin rejects, and every problem fails with
`ValueError: current limit exceeds maximum limit`. The documented escape hatch is
`EVALPLUS_MAX_MEMORY_BYTES=-1`. We note this because every MLX user is on macOS, and it
plausibly contributes to the absence of published MLX code-benchmark numbers.

Generation runs under system Python 3.9 (where `mlx-lm` is installed) and scoring under
Python 3.14 (where EvalPlus installs cleanly); EvalPlus decouples the two by design.

### 3.3 Quantization and provenance

4-bit arms are produced **locally** by `mlx_lm.convert --quantize --q-bits 4` from the
same weights the bf16 arm evaluates, rather than downloaded from `mlx-community`. A
community upload cannot be verified as to base revision or conversion settings, which
would confound the single variable under test. §6 shows this is not a pedantic concern.

MLX's default quantizer is affine round-to-nearest: per group of 64, scale and bias
from min/max, no calibration data, no error compensation. Both `nn.Linear` and
`nn.Embedding` define `to_quantized`, so **token embeddings and `lm_head` are quantized
to 4 bits as well** — where GGUF Q4_K_M keeps them at Q6_K and GPTQ/AWQ pipelines
conventionally leave them in fp16. Nominal "4-bit" is therefore not one thing.

### 3.4 Decoding and the determinism gate

Greedy, `temp=0.0`, n=1, max 768 new tokens. n=1 requires determinism, which we
verified rather than assumed: 20 HumanEval problems generated twice within a process
and twice across separate processes produced **40/40 byte-identical completions**.

This matters because Ouyang et al. (TOSEM 2025) report that 47.6% of HumanEval tasks
yield non-identical outputs at temperature 0. Our result localizes that finding: it is
an artifact of API serving (batching, load balancing, kernel selection), not a property
of greedy decoding. Local MLX greedy is bit-reproducible. Throughput varied 193–208
tok/s across those identical runs — speed is noisy, tokens are not.

### 3.5 Statistics

Arms are evaluated on identical problems, so observations are paired. We use McNemar's
exact test on discordant pairs, not a difference of proportions. Reporting `b` (bf16
solved, 4-bit failed) and `c` (the reverse) throughout.

### 3.6 Domain task suite

Public benchmarks are saturated and skewed easy — PythonSaga finds 84.8% of HumanEval
and 89.6% of MBPP problems are "Easy", with MBPP containing no "Hard" problems at all.
We therefore add 26 tasks derived from functions in our own research codebase
(SAE training, circuit tracing, activation streaming, statistical nulls): 8 EASY,
8 MEDIUM, 10 HARD, 347 assertions.

Each task was validated in both directions: it must pass with a known-correct reference
solution, **and** reject 21 hand-written plausible-but-wrong implementations. This
two-sided validation caught one task whose stated trap was mathematically vacuous — it
would have scored every model correct — and one that was unanswerable by construction.

### 3.7 Host

Apple Silicon, 32 GB unified memory. This ceiling excludes DeepSeek-Coder-V2-Lite from
the bf16 arm (31.4 GB weights); it is reported 4-bit-only and contributes to no delta.

---

## 4. Results: quantization cost

### 4.1 Paired comparison

Measured, complete pairs only:

| Model | Benchmark | bf16 | 4-bit RTN | Δ (pts) | b | c | p (McNemar) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | HumanEval | 0.671 | 0.616 | −5.5 | 14 | 5 | 0.064 |
| | HumanEval+ | 0.622 | 0.579 | −4.3 | 14 | 7 | 0.189 |
| | MBPP | 0.704 | 0.680 | −2.4 | 21 | 12 | 0.163 |
| | MBPP+ | 0.624 | 0.590 | −3.4 | 19 | 6 | 0.015 |
| | **pooled base** | | | | 35 | 17 | **0.018** |
| | **pooled plus** | | | | 33 | 13 | **0.005** |
| Qwen2.5-Coder-3B † | HumanEval | 0.841 | 0.811 | −3.1 | 12 | 7 | 0.359 |
| | HumanEval+ | 0.793 | 0.768 | −2.4 | 12 | 8 | 0.503 |
| | MBPP | 0.772 | 0.728 | −4.5 | 33 | 16 | 0.021 |
| | MBPP+ | 0.653 | 0.619 | −3.4 | 30 | 17 | 0.079 |
| | **pooled base** | | | | 45 | 23 | **0.010** |
| | **pooled plus** | | | | 42 | 25 | **0.050** |
| Phi-4-mini | all | `[PENDING]` | `[PENDING]` | | | | |
| Qwen2.5-Coder-7B | all | `[PENDING]` | `[PENDING]` | | | | |
| Granite-8B-Code | all | `[PENDING]` | `[PENDING]` | | | | |

† Qwen-3B's 4-bit arm is an `mlx-community` upload, not locally converted. Given §6,
this is a different recipe from the other arms and the row must be re-run before it can
be pooled with them. It is shown because it is real, not because it is final.

**No pooled cross-model p-value is stated in this draft.** Pooling a
provenance-matched arm with a community-sourced one mixes two quantizers, and §6
indicates that difference is not negligible.

### 4.2 bf16 baselines against published figures

Sanity check only — never differenced against a 4-bit score:

| Model | Ours (bf16) | Published | Δ |
|---|---|---|---|
| Qwen2.5-Coder-1.5B | 0.671 | 0.707 | −3.6 |
| Qwen2.5-Coder-3B | 0.841 | 0.841 | **0.0** |
| Qwen2.5-Coder-7B | 0.896 | 0.884 | +1.2 |
| Phi-4-mini | 0.665 | 0.744 | −7.9 |
| Granite-8B-Code | 0.591 | n/a ‡ | — |

Agreement is close across the Qwen family and exact at 3B, which is the strongest
available evidence that the harness reproduces the vendor protocol. Phi-4-mini's larger
gap is consistent with Microsoft evaluating through an internal pipeline whose prompt
format and shot count are unspecified. ‡ IBM publishes HumanEvalSynthesize (57.9 Python)
rather than vanilla HumanEval; our 59.1 is near it but not the same benchmark.

### 4.3 Throughput

| Model | bf16 tok/s | 4-bit tok/s | speedup |
|---|---|---|---|
| Qwen2.5-Coder-1.5B | 89.9 | 161.9 | 1.8× |
| Qwen2.5-Coder-3B | 47.7 | 119.8 | 2.5× |
| Phi-4-mini | 39.2 | `[PENDING]` | |
| Qwen2.5-Coder-7B | 22.8 | `[PENDING]` | |
| Granite-8B-Code | 19.6 | `[PENDING]` | |

**Peak memory is deliberately omitted.** The collected figures are not trustworthy —
they report 4-bit consuming *more* memory than bf16 on Qwen-3B (6.55 vs 6.47 GB) while
being sane on Qwen-1.5B (1.65 vs 3.48 GB). The collection path needs fixing before any
memory claim is made.

---

## 5. Results: the vendor-subtraction confound

The intuitive estimate of quantization cost — measure 4-bit locally, subtract the
model card's bf16 figure — is available to any practitioner and is wrong.

| Method | Implied cost, Qwen-1.5B HumanEval |
|---|---|
| Naive: published bf16 (0.707) − measured 4-bit (0.616) | **−9.1 pts** |
| Controlled: measured bf16 (0.671) − measured 4-bit (0.616) | **−5.5 pts** |

The naive figure absorbs the entire harness-mismatch term (−3.6 pts here), which is of
the same order as the effect. The sign of that term is not even stable: our harness
scores *below* published on HumanEval for Qwen-1.5B but *above* published on MBPP,
ruling out a simple "our prompt is worse" explanation and showing the bias cannot be
corrected with a constant.

`[EXPAND]` — restate with the qwen3b row once it is provenance-matched.

---

## 6. Results: community uploads are not local conversions

`mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit` and a local
`mlx_lm.convert --q-bits 4` of the same base model, evaluated identically:

| Benchmark | community | self-converted | Δ | p |
|---|---|---|---|---|
| HumanEval | 0.646 | 0.616 | −3.0 | 0.180 |
| MBPP | 0.690 | 0.680 | −1.1 | 0.455 |
| pooled | | | | 0.108 |

Not significant on one model, but consistently in one direction: the upload is better.
"mlx-community 4-bit" is therefore not a synonym for `mlx_lm convert --q-bits 4`, and
results measured on one should not be reported of the other.

**This needs replication on a second model before it is a claim.** If it holds, it has
two consequences: it is the number practitioners actually want (they run the uploads),
and it means our self-converted arms are a *lower bound* on the quality of what people
deploy.

---

## 7. Harness fragility as a finding

Five independent defects were found in a custom HumanEval/MBPP harness before it was
abandoned. Every one silently scored **correct** model answers as failures.

| # | Defect | Measured effect |
|---|---|---|
| 1 | MBPP prompt never conveyed the required function name, which `test_list` asserts exactly | MBPP pass@1 **0.047 → 0.640** |
| 2 | One system prompt demanded a bare body — correct for HumanEval's stub, fatal for MBPP which has none | SyntaxError on every compliant answer |
| 3 | `test_imports` never emitted | 13/257 unconditional NameErrors |
| 4 | Unindented body appended to a HumanEval stub | SyntaxError on correct answers |
| 5 | Extractor's non-greedy fence match truncated any answer containing a ``` delimiter | one task unanswerable by construction |

Defect 1 is the headline: a single undocumented protocol choice moved pass@1 by 59
points. EvalPlus's own MBPP+ prompts include the assertion, so this was a deviation
from the accepted protocol rather than a hard benchmark. This corroborates Szalontai et
al. (*Software*, 2025), who reproduced only 12 of 35 published results and attributed
failures to benchmark-variant confusion, temperature misconfiguration, and unspecified
precision.

The practical rule this suggests: **a benchmark number is not evidence until the
harness has been shown to score a known-correct reference answer as passing.** Our
domain suite enforces this in both directions (§3.6).

`[EXPAND]` — argue this belongs in the paper rather than an appendix: the field has no
code-native harness-sensitivity study, and this is a measured instance.

---

## 8. Results: domain task suite

Measured, bf16 arms:

| Model | Domain suite | HumanEval |
|---|---|---|
| Qwen2.5-Coder-1.5B | 2/26 | 0.671 |
| Qwen2.5-Coder-3B | 5/26 | 0.841 |
| Granite-8B-Code | 6/26 | 0.591 |
| Phi-4-mini | 7/26 | 0.665 |
| DeepSeek-V2-Lite (4-bit) | 7/26 | 0.762 |
| Qwen2.5-Coder-7B | **11/26** | 0.896 |

The suite spreads this model range 5.5x (2 → 11) where HumanEval spreads it 1.3x
(0.671 → 0.896). Best model solves 42% of tasks; smallest solves 8%. Consistent with
the PythonSaga finding that public benchmarks are dominated by easy problems, and
supports using an uncontaminated domain suite as the discriminating instrument.

`[EXPAND]` — per-tier and per-trap-family breakdown; which trap families survive
quantization. Requires the pending 4-bit arms.

---

## 9. Discussion

`[SKELETON]`

- What a practitioner should conclude, stated as a decision rather than a p-value.
- Cost is small relative to a ~2x throughput gain and a large memory reduction — but
  "small" is measured, not assumed, and it grows with model scale in our two pairs
  (−1.7 avg at 1.5B, −3.3 at 3B).
- Do not compare your local numbers to a model card.
- If you run community uploads, you are not running what `mlx_lm convert` produces.

---

## 10. Threats to validity

- **Incomplete matrix.** 2 of 5 pairs. Conclusions are provisional.
- **Provenance mismatch on Qwen-3B** (§4.1†).
- **Single quantizer.** MLX RTN only. The calibrated variants (`mlx_lm awq/gptq/dwq`)
  are unmeasured, so we cannot say whether the cost is intrinsic to 4-bit or specific
  to the uncalibrated default. This is the most important missing arm.
- **n=1 greedy.** Justified by the determinism gate (§3.4), but a single sample per
  problem cannot estimate pass@k.
- **Single host.** One machine, one MLX version. No claim about other Apple Silicon
  configurations.
- **Memory figures unusable** (§4.3).
- **Operational fragility.** The three missing arms failed twice for unrelated
  infrastructure reasons — a runner bug that pre-created the conversion output
  directory (which `mlx_lm convert` rejects), and a task daemon that silently accepted
  work without persisting it. Neither affects the validity of measured numbers, but
  both are recorded because they determined what got measured.

---

## References

`[SKELETON]` — EvalPlus (Liu et al. 2023); PythonSaga; Szalontai et al. 2025; Ouyang et
al. TOSEM 2025; Giagnorio et al. 2025; Kurtić et al. ACL 2025; Shi & Ding 2025; Fang et
al. 2025; low-bit scaling laws; Qwen2.5-Coder tech report; Phi-4-mini tech report;
Granite Code tech report; DeepSeek-Coder-V2 tech report; mlx-lm BENCHMARKS.md and
LEARNED_QUANTS.md.
