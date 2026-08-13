# Seed-stability control — results

Run 2026-08-12 on lab-02. Three TopK-SAEs on Llama-3.2-3B layer 14, identical
activations (500k tokens, WikiText-103), identical config (dict 16,384, k=128,
50k steps, batch 2048, lr 1e-4, warmup 500). Only `--seed` differs: 42 (the
checkpoint in sae-comparison), 123, 456. Matching procedure copied verbatim from
`cross_arch_matching_v3.py`.

## Headline

| Pair | Matched | % of matchable | Enrichment over null |
|---|---|---|---|
| seed123 ↔ seed456 | 965 | 7.93% | 108× |
| seed42 ↔ seed123 | 978 | 8.06% | 109× |
| seed42 ↔ seed456 | 1002 | 8.26% | 112× |
| **cross-model (paper)** | **15–23** | **~0.1%** | **1.7–3.2×** |

Pearson metric; cosine gives 1087–1211 matches, 8.9–10.0%, 121–135×.

## 1. The matching method works

This was the question. When feature sharing is guaranteed by construction, the
procedure recovers it at **108–112× over the permutation null**, against 1.7–3.2×
across models. The detector is not blind.

**Consequence for sae-comparison:** the near-zero cross-architecture result is a
fact about the models, not an artifact of a broken instrument. That was the
alternative explanation we could not previously exclude, and it is now excluded.

## 2. But the ceiling is 8%, not 100%

Two SAEs trained on *identical activations from the same model*, differing only in
seed, share only **~8% of their matchable features**. Roughly 92% of each
dictionary does not survive a reseed.

**This rescales the paper's central claim.** Cross-model sharing should be reported
against the within-model ceiling, not against the dictionary size: 0.1% observed
against an 8% ceiling — about 80× below what is achievable — rather than 0.1%
against 100%. The gap remains large and the conclusion stands, but the correct
denominator is far smaller than the paper currently implies.

It also means no cross-model universality study using this class of method can
report more than ~8%, whatever the models. That is a bound on the literature, not
just on this paper.

## 3. The method finds near-duplicates, exactly as the power analysis predicted

Similarity distribution of the matched pairs (seed123↔seed456): min 0.983,
median **0.991**, 59% above 0.99, 27% above 0.995.

sae-comparison's planted-signal analysis put the minimum detectable effect size at
correlation 0.95–1.00. This is the empirical confirmation on real data: the
procedure is not finding "similar" features, it is finding features that are
essentially identical. Anything genuinely shared but realised at 0.9 is invisible,
and now we can say so from measurement rather than simulation.

## 4. The reproducible 8% is itself only half-reproducible

Of seed42's features, 978 matched seed123 and 1002 matched seed456 — but the two
sets overlap on only **515 features, 52.7% of the smaller set (Jaccard 0.352)**.

So there is no stable "core dictionary" of robust features. Different reseeds
recover overlapping but substantially different subsets. Reproducibility is not a
property a given feature has; it is closer to a coin flip conditioned on the pair
of runs being compared.

## 5. The quality metrics cannot see any of this

The three runs are indistinguishable by every metric the SAE literature reports:

| Seed | Final loss | FVE | L0 | dead (5k window) |
|---|---|---|---|---|
| 42 (published) | 0.006174 | 0.9939 | 128.0 | 3,753 |
| 123 | 0.006171 | 0.9951 | 128.0 | 3,600 |
| 456 | 0.006182 | 0.9898 | 128.0 | 3,696 |

Loss agrees to 0.2%, FVE to half a percentage point, L0 exactly, dead count to 4%.
By the standard report, these are the same SAE trained three times.

**They share 8% of their features.** Reconstruction quality, sparsity and dead-feature
count are jointly blind to the identity of the dictionary. This is the paper's existing
thesis — that final-checkpoint metrics miss what matters — carried one step further
than it currently goes: the metrics do not merely miss the collapse dynamics, they
cannot distinguish two dictionaries that disagree about 92% of their content.

## 6. Two incidental replications

**Collapse is a property of the procedure, not a seed accident.** Dead-on-eval
counts: 3,344 (seed 42, the published figure), 3,206 (seed 123), 3,274 (seed 456)
— within 4% across seeds. sae-comparison's collapse magnitude replicates.

**The fixed threshold is below the noise floor, demonstrated on known ground
truth.** The same-model null mean is 0.757, so τ = 0.80 sits barely above it. At
τ = 0.80 the procedure returns ~5,450–5,950 matches (about 45% of matchable)
against ~965 at the calibrated threshold. The paper argued this from permutation
nulls; here it is shown on a case where we know sharing is real, and the fixed
threshold still inflates the count roughly sixfold.

## Caveats

- Fingerprints are computed on the same corpus the SAEs trained on, matching the
  paper's protocol. The 8% ceiling is therefore in-sample; a held-out ceiling
  could differ.
- One model, one layer. Whether the ~8% ceiling is characteristic of TopK SAEs
  generally, or specific to this configuration, is unmeasured.
- Seed controls both weight init and batch order. These are not separated, so this
  measures total retraining variation rather than either source alone.

## Files

- `seed-stability-report.json` — full metrics, both metrics, all three pairs
- `{cosine,pearson}_{pair}_matches.json` — matched feature index pairs with similarity
- Checkpoints on lab-02: `data/sae-runs/llama-3b-layer14-seed{123,456}/`
- Scripts: `data/sae-runs/run_seed_stability.sh`,
  `data/sae-analysis/seed_stability_match.py`
