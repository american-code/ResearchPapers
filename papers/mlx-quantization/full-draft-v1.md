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

All five pairs measured. Δ is the mean of the HumanEval and MBPP deltas; `b` and `c`
are discordant pairs pooled over both benchmarks (bf16-only and 4-bit-only solves).

| Model | Params | HE bf16 | HE 4bit | MBPP bf16 | MBPP 4bit | mean Δ | b | c | p |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | 1.5B | 0.671 | 0.616 | 0.704 | 0.680 | −3.9 | 35 | 17 | **0.018** |
| Qwen2.5-Coder-3B † | 3.0B | 0.841 | 0.811 | 0.772 | 0.728 | −3.8 | 45 | 23 | **0.010** |
| Phi-4-mini | 3.8B | 0.665 | 0.616 | 0.646 | 0.563 | **−6.5** | 91 | 52 | **0.0014** |
| Qwen2.5-Coder-7B | 7.0B | 0.896 | 0.878 | 0.849 | 0.844 | −1.2 | 16 | 11 | 0.442 |
| Granite-8B-Code | 8.0B | 0.591 | 0.579 | 0.672 | 0.648 | −1.8 | 33 | 22 | 0.177 |
| **All five pooled** | | | | | | | **220** | **125** | **< 10⁻⁶** |

† Qwen-3B's 4-bit arm is an `mlx-community` upload, not locally converted (see §6).
Pooled figures are reported both with and without it in the final version; its
exclusion does not change the sign or significance of the pooled result.

**Two results, not one.**

1. **The cost is real.** 220 discordant pairs favor bf16 against 125 favoring 4-bit,
   p < 10⁻⁶. Individually only three of five models reach significance, which is why
   single-model studies of this effect are underpowered — an important observation in
   its own right.

2. **The cost shrinks with model size.** Regressing mean Δ on parameter count gives
   **+0.52 points per billion**: Phi-4-mini (3.8B) loses 6.5 points while Qwen-7B
   loses 1.2. This is consistent with the quantization literature's finding that
   larger models are more resilient, and it is the practically actionable result — the
   models most often chosen *because* they are small are the ones 4-bit hurts most.

`[EXPAND]` — Phi-4-mini is the outlier in both directions: largest quantization cost
and largest gap to its published bf16 figure. Worth investigating whether its
architecture (or its tokenizer/embedding sizing) interacts with 4-bit embedding
quantization, given §3.3.

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

### 4.4 Calibration does not rescue 4-bit

MLX's default quantizer is uncalibrated round-to-nearest (§3.3), which is weaker than
the GPTQ/AWQ-class methods behind the published ~0.5–2 pt loss figures. The obvious
hypothesis is that the cost we measure is an artifact of that default rather than a
property of 4-bit. We tested it directly by adding an activation-aware (AWQ) arm at the
same bit width and group size.

All three arms below ran under a single mlx-lm 0.31.3 (see §4.5 on why the runtime
changed and why it is not a confound). **2 of 5 models complete; `[PENDING]` elsewhere.**

| Model | Benchmark | bf16 | 4-bit RTN | 4-bit AWQ | RTN Δ | AWQ Δ |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-1.5B | HumanEval | 0.683 | 0.622 | 0.628 | −6.1 | −5.5 |
| | MBPP | 0.704 | 0.688 | 0.683 | −1.6 | −2.1 |
| Qwen2.5-Coder-3B | HumanEval | 0.854 | 0.829 | 0.835 | −2.4 | −1.8 |
| | MBPP | 0.775 | 0.712 | 0.738 | −6.3 | −3.7 |
| Granite-8B-Code | HumanEval | 0.598 | 0.585 | 0.549 | −1.2 | −4.9 |
| | MBPP | 0.677 | 0.648 | **0.680** | −2.9 | **+0.3** |
| Qwen2.5-Coder-7B | HumanEval | 0.896 | 0.878 | 0.890 | −1.8 | −0.6 |
| | MBPP | 0.854 | 0.844 | 0.841 | −1.1 | −1.3 |
| Phi-4-mini | | 0.665 | 0.616 | **n/a** ‡ | −4.9 | |

‡ **Phi-4-mini cannot have an AWQ arm.** `mlx_lm.quant.awq` raises
`NotImplementedError: AWQ support for phi3 models NYI`. This is a hard limitation of
the tool, not a transient failure. It matters because Phi-4-mini carries the *largest*
RTN cost in the matrix (§4.1), so it is precisely the model where "a better quantizer
would fix this" was most plausible — and it is the one model where we cannot test it.
Reported as excluded with cause rather than silently dropped.

Pooled McNemar over all 8 cells (4 models × 2 benchmarks):

| Comparison | wins A | wins B | p |
|---|---|---|---|
| bf16 vs RTN | 134 (bf16) | 70 (RTN) | **0.000009** |
| bf16 vs AWQ | 119 (bf16) | 72 (AWQ) | **0.00083** |
| **RTN vs AWQ (head-to-head)** | 133 (AWQ) | 116 (RTN) | **0.31** |

**Both quantizers lose to bf16 by a wide statistical margin. Neither beats the other.**
The head-to-head has now been stable at p≈0.31 across three successive additions of
data, and Qwen-7B's own head-to-head is 14–13 (p=1.00) — as close to a tie as the test
can express.

The practical conclusion is therefore not "use a better quantizer". It is that at 4
bits and these scales **the loss is a property of the bit width, not of the algorithm
that gets you there.**

**Per-benchmark variance exceeds the method difference.** Granite-8B is the clearest
case: AWQ is 4.9 points *below* bf16 on HumanEval but *matches* it on MBPP (+0.3),
while RTN loses on both. On the same model with the same weights, the two benchmarks
disagree about which quantizer is better.

This is the section's central caution, and it was learned the hard way. An earlier
draft, written when only Granite-8B's HumanEval cell had finished, concluded that
calibration "actively hurts". MBPP reversed that within the hour. **A single benchmark
on a single model does not support a directional claim about quantizer quality** —
the same failure this paper documents in §7, arriving from a different direction.

If this survives the remaining three models, the cost is **intrinsic to 4-bit at these
scales**, not an artifact of MLX shipping an uncalibrated default. That is the more
useful conclusion for a practitioner — "4-bit costs a few points regardless of how
carefully you quantize" is actionable in a way that "switch quantizers" would not be.

It also contradicts a hypothesis we formed earlier from §6, where an mlx-community
upload outscored a local RTN conversion; that had suggested recipe quality mattered.
The head-to-head here says it does not, at least between RTN and AWQ. Both can be true
— the community uploads may differ from local `convert` in some way other than
calibration — but the simpler reading is that §6's effect was noise, and it is reported
as non-significant there.

**Cost of the calibrated arm.** AWQ quantization scales worse than linearly: 18 minutes
at 1.5B, ~1h30m at 7B, 1h51m at 8B — roughly 6× the time for 5.3× the parameters, as
the grid search over scaling factors compounds with layer count. Inference throughput is
unchanged (Granite-8B 48.2 vs 48.3 tok/s; Qwen-7B 57.2 vs 57.3), so the price is paid
entirely at build time and buys no runtime benefit either.

**Domain suite.** The 26-task suite tracks the same ordering but with a wider spread,
and is the one place AWQ looks consistently worse:

| Model | bf16 | RTN | AWQ |
|---|---|---|---|
| Qwen-7B | 12/26 | 11/26 | **8/26** |
| Granite-8B | 7/26 | 6/26 | 7/26 |

Qwen-7B's AWQ arm solves four fewer tasks than bf16 while its HumanEval score is within
0.6 points. With only 26 tasks this is a small-N observation, not a result — but it is
the direction worth checking if the suite is expanded, since it hints that aggregate
pass@1 on saturated benchmarks may hide capability loss the harder tasks expose.

**One asymmetry to record rather than smooth over:** AWQ sets `model.embed_tokens` to
`group_size: 32` while RTN uses 64 uniformly. So "RTN vs AWQ" is not a pure method
comparison — the embedding treatment differs too. Given §3.3 (MLX quantizes embeddings
and `lm_head` at all, unlike GGUF Q4_K_M and GPTQ/AWQ pipelines elsewhere), this is a
plausible contributor and should not be attributed to calibration alone.

### 4.5 Runtime version is not a confound

The calibrated quantizers exist only in mlx-lm ≥ 0.30, while the matrix in §4.1 was
measured under 0.29.1. The original plan was to produce AWQ weights with a newer
mlx-lm in an isolated venv and evaluate them on the pinned runtime, keeping inference
constant. That failed: 0.31.3 writes a per-layer quantization config that 0.29.1
cannot parse. We rejected writing a compatibility shim, because misreading a
quantization spec would produce an AWQ column that looks correct and means something
else.

Instead we tested the assumption being protected. Same weights, same harness, only the
mlx-lm version differing:

| | 0.29.1 | 0.31.3 | Δ | discordant | p |
|---|---|---|---|---|---|
| Qwen-1.5B HumanEval | 0.671 | 0.683 | +1.2 | 0 old-only / 2 new-only | 0.50 |
| Qwen-1.5B HumanEval+ | 0.622 | 0.634 | +1.2 | 0 / 2 | 0.50 |

**162 of 164 per-problem verdicts are identical.** The runtime shifts 2 problems; the
effect under study accounts for 345 discordant pairs across the matrix. The version is
therefore not a confound at the resolution that matters, and the three-arm results run
under a single 0.31.3.

The 0.29.1 matrix is retained and reported (§4.1) rather than discarded — it is a
cross-version replication, which is more than the design originally promised. Every
summary now records `mlx_lm_version`, `python`, and `runs_base` so no score can be
silently attributed to the wrong inference stack.

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
- **Calibrated arm is partial.** AWQ is measured on 2 of 5 models (§4.4). GPTQ and DWQ
  remain unmeasured; the head-to-head so far is RTN vs AWQ only, so "calibration does
  not help" is supported for one calibrated method, not for calibration in general.
- **AWQ and RTN differ in embedding group size** (32 vs 64), so §4.4's head-to-head is
  not a pure method contrast.
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
