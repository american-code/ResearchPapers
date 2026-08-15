# Research backlog

Ideas parked deliberately, not abandoned. Nothing here is started; the two papers in
flight (see README) take priority.

Each entry states the question, why this workspace is positioned for it, what it would
cost, and — where known — what already exists in the literature. That last field is
mandatory: two of the three topics selected in August 2026 turned out to be
substantially anticipated by work that used none of the phrases the bibliometric survey
searched on. Phrase-count gap analysis measures vocabulary, not activity. Read before
committing.

---

## 1. Does a transformer represent argument structure geometrically?

**Question.** Verbs carry a valence — intransitive (*sleep*, 1 argument), transitive
(*kick*, 2), ditransitive (*give*, 3). This is a structural property, not a lexical one.
Does a trained model encode it as a recoverable direction in the residual stream, and is
that encoding causal?

**Why it is worth asking here.** The cross-architecture result in `sae-comparison` is
that *features* do not transfer between model families: 0.18–0.33% of matchable
features agree, against an 8–9% same-model ceiling. That is a claim about lexical or
semantic features. It says nothing about whether *structural* properties transfer. If
argument arity is represented in a way that survives the same matching procedure that
killed feature universality, that is a positive result carved out of a published
negative one — and it sharpens rather than contradicts it.

**Pilot, roughly in order.**

1. Probe: verbs of known valence, look for a linear direction separating 1/2/3 in the
   residual stream. Cheap, and if there is nothing here the rest is moot.
2. SAE features: does any feature fire on valence rather than on lexical identity?
   Dictionaries and the calibrated matching procedure already exist.
3. Causality: patch the candidate direction and test whether the model's argument
   expectations change. `data/circuit-robustness/circuit_lib.py` already performs this
   class of intervention on both architectures.
4. Universality: run the direction through the cross-architecture matching used in
   `sae-comparison`. This is the step that makes it a paper rather than a probe result.

**Cost.** Low. The verbs are already present in the agreement and IOI datasets. A probe
plus a patching run, no training job.

**Prior art to check before starting.** Structural-probing literature (Hewitt & Manning
and successors) established that syntax trees are recoverable from contextual
embeddings; the specific question of *argument arity as a causal direction*, and whether
it transfers across architectures, is what needs a real literature check. Do not rely on
phrase counts.

**Origin.** Arrived at from the observation that a noun behaves like one thing and a
verb like three. Assigning a scalar per part of speech does not work — it is strictly
less information than a POS tag, which models recover in early layers — but the
underlying intuition is a rediscovery of the type structure in DisCoCat
(Coecke, Sadrzadeh & Clark 2010), where a noun is a vector and a transitive verb is an
order-3 tensor in N ⊗ S ⊗ N. Three slots, not three outcomes. DisCoCat itself was
empirically overtaken by transformers and is not proposed here as an architecture; the
useful residue is the type structure as an object of measurement.

---

## 2. Wall-clock cost of geometric latent reasoning

**Question.** Geometric Latent Reasoning (arXiv:2606.02248) reports 75–85% fewer
generated tokens on GSM8K and MATH500 by replacing discrete reasoning steps with
continuous paths through embedding space. The authors state they did **not** measure
wall-clock latency, citing implementation differences. Does the token reduction survive
as a real speedup on real hardware?

**Why it is worth asking here.** This workspace already has a validated generation
harness, a determinism gate, per-problem verdict reporting, and a documented refusal to
subtract numbers measured under different conditions. Measuring what someone else
asserted is the recurring shape of the useful work here.

**Cost.** Moderate. Qwen3-0.6B and 1.7B are the published scales and fit lab-02
comfortably. The work is reimplementing or obtaining the transition head, not training a
model.

**Caveat.** A negative result is likely and is still worth publishing: fewer tokens at
higher per-token cost can be net slower, which is exactly the kind of thing the field
assumes away. Note also that GLR's accuracy advantage appears only under constrained
budgets — at unlimited budget, standard chain-of-thought often exceeds it.

---

## 3. Sophontic / "geometric density" — watch only, do not start

Julian D. Michels (Sophontic) claims reasoning performance exceeding models up to 60×
larger by directly training internal geometry. As of 2026-08-15 this is unevaluable:
zero arXiv entries, no model name, no benchmark data, no code or weights, and the
site's `/model`, `/evaluation` and `/research` paths return 404. The claim is
parameter-efficiency, not inference speed, and is not the 1000× figure that circulates.

Revisit if a technical report or weights appear. Nothing to measure until then.

---

## 4. MoE and MLA quantization on Apple Silicon

Reduced from a full study after finding that distributed-inference nondeterminism is
already characterized (arXiv:2506.09501 establishes that GPU count affects
reproducibility, with bf16 worst). What survives is narrower and still unclaimed:
DeepSeek-Coder-V2-Lite bf16 (31.4 GB, `DeepseekV2ForCausalLM`, MLA with
`kv_lora_rank` 512) versus local 4-bit, which fills the row the quantization paper had
to report as 4-bit-only. Scripts exist at `data/cluster-gate/`; the determinism gate is
required as internal methodology before trusting a cluster-hosted bf16 arm, but is no
longer a contribution in itself.
