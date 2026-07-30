# Corrections log — 2026-07-30

Record of defects found in an audit of the three papers against their underlying data,
and what was done about each. Ordered by severity.

Every claim below was checked against the data files, not against prose. Where a
correction required recomputation, the script that produces the new number is named.

---

## 1. Cross-architecture feature universality: 3,753 → 0

**Affected:** `papers/distributed-interp` (headline result), `papers/sae-comparison`

`data/sae-analysis/cross_arch_matching.py` reported 3,753 features shared across
Llama-3.2-3B, Mistral-7B and Qwen2.5-3B. Three independent defects inflated this:

1. **Many-to-one matching.** `nearest_neighbor_match()` returned the *union* of the
   A→B and B→A nearest-neighbour sets, so one feature could be the recorded partner of
   arbitrarily many others. One Mistral feature was the match for 137 distinct Llama
   features. Reported "match counts" exceeded the dictionary size, making the derived
   coverage percentages meaningless (Llama–Mistral: 12,174 pairs but only 4,769 unique
   Llama features, i.e. 29.1% real coverage, not 74.3%).
2. **Unclosed triangles.** `find_universal_features()` chained llama→qwen→mistral and
   only recorded the llama–mistral similarity "if available", never requiring it.
   2,520 of 3,753 triples (67%) had no verified third edge; 306 lacked the field
   entirely. The 3,753 triples collapsed onto 644 distinct Qwen and 341 distinct
   Mistral features.
3. **Uncalibrated threshold.** τ = 0.80 was chosen by hand. Under a chunk-permutation
   null, mean reciprocal-match similarity between *unrelated* features is 0.39–0.57
   and the 99th percentile is 0.83–0.94. **τ = 0.80 sits at or below the noise floor.**

Additionally, the Venn counts did not partition the dictionary: Qwen's regions summed
to 18,258 of a 16,384-feature dictionary (+1,874); Llama +199; Mistral −2,976.

**Fix.** `data/sae-analysis/cross_arch_matching_v2.py` implements reciprocal
(mutual-nearest-neighbour) one-to-one matching, a closed-triangle requirement, a 5%
fingerprint-support floor, a permutation-calibrated threshold, and centred fingerprints
(Pearson rather than cosine, removing the positive-orthant baseline). Fingerprint
resolution raised from 100 chunks / 50k tokens to 1,000 chunks / 500k tokens — at the
original resolution the null p99 reaches 0.98–0.995 and the statistic has no
discriminative power at all.

**Result:** three-way universal features = **0** (null expectation 0.15). Pairwise:
22 / 10 / 12 matches against 7.0 / 4.9 / 5.6 expected false positives. Region counts
now partition each dictionary exactly (asserted in code).

Outputs: `data/sae-analysis/matching-v2/`. v1 retained for comparison.

---

## 2. sae-comparison stated results that did not exist

**Affected:** `papers/sae-comparison`

The abstract and conclusion asserted findings — "13–17% functionally universal",
"TopK scores highest on probe accuracy; Gated highest on human ratings", "*p* = 0.61" —
while every Results table in the body was a `\placeholder{}` marked *illustrative*,
*not yet collected*, or *not yet conducted*. 24 PLACEHOLDER blocks and 3 CITE markers
rendered in the compiled PDF. The abstract quoted *p* = 0.61 with an inline
`\placeholder{exact value from experiment}` attached.

Every experimental premise was also contradicted by the data:

| Claimed | Actual |
|---|---|
| GPT-2 Small, Pythia-1.4B, Llama-3.2-3B | Llama-3.2-3B, Mistral-7B, Qwen2.5-3B |
| L1, TopK, Gated (+ JumpReLU) | **only TopK trained** |
| 500M tokens of The Pile + 50M held-out | 500k tokens WikiText-103, no held-out |
| Human study, 5 annotators | never conducted |
| 16× / 10.7× expansion | 5.33× / 4.0× / 8.0× |
| ICLR 2027 (header) | NeurIPS 2026 (README) |

**Fix.** Rescoped to what the data supports: a study of TopK SAE training dynamics
across three architectures (dictionary collapse, dense-feature degeneracy, held-out
reconstruction, the negative matching result). New §5 "Planned Experiments Not
Conducted" enumerates every missing experiment. Ethics statement describing
compensation and institutional approval for the non-existent human study removed.
All placeholders eliminated.

---

## 3. IOI depth-conservation claim is not significant

**Affected:** `papers/circuit-tracing` (abstract, contribution 2, conclusion)

"80% of circuit-critical head positions are conserved by relative depth (±0.075)" was
reported with no null test. Reimplementing the paper's own greedy matcher over 20,000
random draws (`recompute_corrections.py`):

| | Mean matches | Rate | Pr(≥ 8/10) |
|---|---|---|---|
| Observed | 8.00 | 80.0% | — |
| Uniform null | 5.88 | 58.8% | **0.098** |
| Informed null (d ≥ 0.40) | 6.87 | 68.7% | **0.32** |

Neither null rejects. ±0.075 spans ~2 layers in each model; 10 heads drawn from ~17
usable layers cannot avoid landing within tolerance. The matching is also functionally
uninformative — it pairs Llama's L15H20 (ablation drop 1.578) with Pythia's L13H6
(0.018), while Pythia's actual dominant head L10H7 is matched to Llama's rank-3.

**Fix.** New §6.1.1 reports both nulls, explains why the tolerance is too permissive,
and states what a valid test would require. Abstract, contributions and conclusion
rewritten to withdraw the quantitative claim while retaining the descriptive
three-cluster observation.

---

## 4. Safety classifier: no threshold-free metric

**Affected:** `papers/distributed-interp` §6.5

Only a threshold sweep was reported, which cannot distinguish "score carries signal but
threshold is wrong" from "score carries no signal". Computed from the stored
per-example scores:

- **ROC-AUC 0.564, 95% bootstrap CI [0.484, 0.645] — includes chance**
- Average precision 0.657 (baseline 0.500); Cohen's *d* = 0.238

The classifier is not merely weak but statistically indistinguishable from chance at
*n* = 200. Root-cause analysis also missed the real driver: the top-200 features by
frequency are degenerate dense directions (34 fire on >50% of tokens; the 200 carry 40%
of all activation mass), so frequency-based selection draws almost entirely from
non-features.

**Fix.** New Table 12 and accompanying analysis; root-cause item 2 rewritten; the
unsupported "target F1 ≥ 0.80" projection removed.

---

## 5. Unit error in the throughput benchmark

**Affected:** `papers/distributed-interp`, `benchmarks/activation-streaming-localhost.json`

61,440,632 bytes ÷ 0.049126 s = **1.25 GB/s = 10.0 Gbps**. Both the paper and the
`throughput_gbps` JSON field reported "1.25 Gbps" — the value was a byte-rate, the label
was wrong, an 8× discrepancy. The derived "≈3,000× the real-time requirement" matched
neither reading (correct: ~10,200× at GB/s, ~1,300× at Gbps).

Also corrected: protocol overhead stated as 0.20% (actual: 8 bytes on a 1 MiB payload =
7.6×10⁻⁴%, off by ~260×); Thunderbolt-4 practical bandwidth quoted as 4.5 GB/s in §1
and 3.5 GB/s in Table 4 (standardised on 3.5).

---

## 6. Claims outrunning what was run

**Affected:** `papers/distributed-interp`

- The abstract described "two Apple Silicon nodes connected by Thunderbolt". **No
  Thunderbolt experiment was ever performed** — all measurements are single-machine.
  Disclosed only in Limitations, five pages later. Moved into the abstract.
- "Distributed" SAE training runs **two workers sequentially in one process**
  (`distributed-sae-training-validation.json`: *"Workers simulated sequentially in
  Python"*). Limitations disclosed the two-*subprocess* protocol test but not this.
  Table caption and Limitations now state it plainly.
- **Split inference is 1.84× slower** than single-process (132.1 s vs 71.7 s). Present
  in the data, never reported. Now reported.
- The initial-loss explanation ("slower warmup … linear ramp over 200 steps vs. the
  baseline's faster acceleration") **contradicted the paper's own curves**, which show
  identical learning rates at every logged step. Replaced with the actual cause: the
  data partition gives the two runs different first batches.
- The float16 losslessness argument cited mantissa width while omitting that float16's
  5-bit exponent gives it a much narrower dynamic range. The run did not instrument
  activation magnitudes, so the precondition is **unverified** — now flagged as such
  rather than asserted. *(No substitute measurement was fabricated; the model weights
  are not available locally to re-measure.)*

---

## 7. Reconstruction quality reported from single batches

**Affected:** both SAE papers

Reported FVE was the last logged training batch. Batch FVE spans 0.013 (Llama), 0.021
(Qwen) and **0.364** (Mistral) across the final ten logged steps — a single reading is
a draw from a wide distribution. Recomputed over the full 500k-token corpus:

| Model | Reported (batch) | Corrected (full corpus) | Held-out |
|---|---|---|---|
| Llama-3.2-3B | 0.992 | 0.9865 | none (53 epochs) |
| Qwen2.5-3B | 0.984 | 0.9839 | none (205 epochs) |
| Mistral-7B | 0.9649 | 0.9273 | **0.9252** |

Mistral was trained on only the first 50k of its 500k-token dump, so tokens 50k–500k
are a **genuine held-out split** — the only one in the workspace. Held-out FVE 0.9252
vs in-sample 0.9273 (0.2 pt gap): that SAE is not overfitting. The Mistral figure was
previously overstated by 3.8 points.

---

## 8. Circuit-tracing: statistical over-claim and misdescribed data

**Affected:** `papers/circuit-tracing`

- §5.5 claimed *"All top-10 heads in both models have CIs excluding zero by a margin of
  at least 3× the CI width."* False for four heads by the paper's own tables: L26H23
  (1.8×), L27H17 (2.7×), L17H7 (0.9×), and L1H11 (CI straddles zero, ranked #4).
  Corrected to the per-head breakdown, plus a new paragraph on the absence of
  multiple-comparison correction across 672/384 tested heads and the winner's-curse
  bias from selecting and estimating on the same 100 examples.
- **Factual-association dataset was misdescribed** as "drawn from Meng et al. (2022)'s
  CounterFact evaluation set". It is 25 hand-built `capital_of` + 25
  `official_language_of` items. §6.2 also described only the capitals, silently omitting
  half the dataset. Both corrected.
- **Path patching was reported as "in preparation"** in three places while
  `data/ioi/path-patching-llama3b-real.json` contained 10 real edges at *n* = 100. Now
  reported as new §6.2 with the full edge table, including two suppressive edges
  (L15H20→L17H17, score −0.113) that support a negative-name-mover reading.
- **Three values for one quantity**: mean clean logit difference was 5.649 (paper),
  5.6431 (`baseline-llama3b.json`), 5.668 (`path-patching-llama3b-real.json`).
  Standardised on 5.643; dependent percentages recomputed (L15H20 27.9%→28.0%,
  L17H17 16.4%→16.5%).
- Llama-3.2-3B called "instruction-tuned"; the model used is the base bf16 checkpoint.
- Added a Related Work subsection covering the prior art whose omission reviewers would
  flag: Merullo et al. 2024 (cross-task circuit reuse — closest prior work to §6.3),
  Lieberum et al. 2023 (circuit analysis at 70B), Conmy et al. 2023 (ACDC),
  Zhang & Nanda 2024 and Heimersheim & Nanda 2024 (patching best practices),
  Hanna et al. 2023.
- **Faithfulness/completeness/minimality is still absent** and is now the first item in
  Limitations. Per-head patching scores sum to 1.013, so they cannot substitute for a
  joint circuit measurement. This requires model forward passes and **was not run** —
  the weights are not available locally.
- `REPO_URL_HERE` ×2 replaced.

---

## 9. Data hygiene

- `qwen-3b-layer18/metrics_final.json` recorded `training_elapsed_s: 0.2` for a
  50,000-step run — an artifact of a resume invocation that loaded the final checkpoint
  and exited. **Not fabrication**: `training.jsonl` shows 14,446.8 s of real training.
  Corrected, with provenance noted.
- `distributed-capture-validation.json` note claimed *"Non-zero error arises from…"*
  while every reported error was exactly 0.0. Rewritten.
- Mistral `metadata.json` had `n_tokens_written: 50000` alongside
  `activations_shape: [500000, 4096]`, and pointed `activations_file` at the 50k
  subset — understating available data and hiding the held-out split. Corrected.
- `mistral7b-feature-examples.json` still carried `error: no_sae_checkpoint` from before
  the Mistral SAE existed. Marked superseded.
- Schema drift: `baseline-llama3b.json` used `s_name`, `dataset.json` used
  `subject_name`. Unified on `subject_name`.
- **Directory rename**: `llama-3b-layer16/` → `llama-3b-layer14/` (both
  `data/activations/` and `data/sae-runs/`), plus `collect_llama3b_layer16.py`. Every
  config, log and metadata file in those directories records `source_layer: 14`; the
  `layer16` name was a trap for anyone reproducing. 25 referencing files updated.

---

## Still outstanding

These require model forward passes. No model weights are present in the local HF cache,
so none could be run here.

1. **IOI circuit faithfulness/completeness/minimality** — joint ablation of the
   identified circuit. The single most important missing experiment in circuit-tracing.
2. **Multi-template IOI dataset** with balanced ABBA/BABA name ordering. All 100 current
   examples share one frame and one ordering.
3. **Pythia path patching** — only Llama was measured.
4. **Retrain the Mistral SAE** to parity (1,000 steps → 50,000 on 500k tokens; ~17 h at
   the observed 1.24 s/step).
5. **Retrain all SAEs with an auxiliary dead-feature revival loss** and a reserved
   held-out split. Fixes the collapse documented in §7 and the missing held-out data.
6. **Instrument activation dynamic range** at the split boundary to verify the float16
   losslessness precondition (§6).
7. **Larger, benchmark-sourced safety eval set** drawn directly from AdvBench/HarmBench
   rather than "inspired by" them, with a second annotator.

## Regenerating

```bash
python3 data/sae-analysis/cross_arch_matching_v2.py    # ~8 min
python3 data/sae-analysis/recompute_corrections.py     # ~13 min
python3 papers/distributed-interp/submission/make_figures.py
for p in circuit-tracing distributed-interp sae-comparison; do
  (cd papers/$p/submission && tectonic -X compile main.tex)
done
```
