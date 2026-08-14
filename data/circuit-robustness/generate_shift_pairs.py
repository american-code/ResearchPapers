#!/usr/bin/env python3
"""Generate task-preserving D / D-prime dataset pairs for circuit robustness testing.

WHY THIS EXISTS
---------------
The circuit-tracing paper measures faithfulness falling from 0.663 to 0.228 (Llama)
and 0.854 to -0.028 (Pythia) when the circuit is re-evaluated on a 15-template set.
That paper states plainly that the mechanism is unverified, and the reason it cannot
be verified from those runs is that the comparison changes two things at once:

    discovery set : 1 template            evaluation set : 15 templates
                    ^^^^^^^^^^                             ^^^^^^^^^^^^
                    template COUNT and template IDENTITY both change

So a collapse is consistent with two very different stories:

  (a) circuits are frame-specific -- a circuit found on any frame fails on other
      frames, and the field's faithfulness numbers are routinely overstated;
  (b) the DISCOVERY set was degenerate -- with one frame, token position is perfectly
      confounded with syntactic role, so part of what was called a circuit is really
      "attend to position 4", and it fails anywhere that coincidence breaks.

These have opposite prescriptions. Under (a), faithfulness must always be reported on
held-out data. Under (b), the fix is upstream: discover on diverse data and the
problem largely goes away. Neither can be selected from a 1-vs-15 comparison.

This generator produces the missing condition: matched-diversity, disjoint-frame
splits. D and D-prime have the SAME number of templates and share NONE of them, so
diversity is held fixed and frame identity is the only thing that moves.

    F(C_D) on D          in-distribution reference
    F(C_D) on D-prime    the number that decides between (a) and (b)

If the second roughly matches the first, the original collapse was about discovery-set
degeneracy, and "discover on diverse data" is a concrete, checkable prescription. If it
still collapses, frame dependence is real and general. Both outcomes are publishable;
what is not publishable is the current ambiguity.

SHIFT AXES
----------
--shift frame    templates disjoint, entity pool shared    (syntactic frame moves)
--shift entity   templates shared, entity pools disjoint   (lexical content moves)
--shift both     both disjoint                             (upper bound on shift)

Separating these matters because they are different claims. A circuit that survives
new names but not new frames is a syntactic-position circuit; one that survives new
frames but not new names has memorised lexical items. Reporting only "held-out" would
merge them.

TASK FAMILIES
-------------
ioi         indirect-object identification         (the existing task, extended)
factual     capital-city recall                    (retrieval rather than routing)
agreement   subject-verb number agreement          (long-range syntactic dependency)
induction   in-context copying of a paired token   (the canonical induction circuit)

Four families rather than one because "circuits are fragile" measured only on IOI is a
statement about IOI. Each family emits the same record shape, so one harness scores all
of them.

OUTPUT SHAPE
------------
Each record carries `answer_token` / `distractor_token` as the canonical field names,
plus `io_name` / `subject_name` as aliases so that data/ioi/run_circuit_faithfulness.py
consumes these files unmodified -- its logit_diff() reads those two keys and computes
logit(answer) - logit(distractor) at the final position, which is the right metric for
every family here.

Usage:
  python3 generate_shift_pairs.py --family ioi --shift frame --n 200
  python3 generate_shift_pairs.py --all
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEED = 42

TOKENIZERS = [
    ("llama", "mlx-community/Llama-3.2-3B-bf16"),
    ("pythia", "EleutherAI/pythia-1.4b"),
]

# ── IOI ──────────────────────────────────────────────────────────────────────
# Bank A is the 15-template set from data/ioi/generate_dataset_v2.py, reproduced so
# that D matches the published evaluation set exactly. Bank B is new and disjoint:
# different verbs, different objects, different locations, different opening
# conjunctions. Clause SHAPE is deliberately held constant across banks -- both are
# "[conj] {first} and {second} [vp], {subject} [vt] [obj] to" -- because the point is
# to vary the frame while preserving the task, and changing the syntax as well would
# reintroduce the confound this file exists to remove.
IOI_A = [
    ("When {first} and {second} went to the store,", "{subject} gave a bottle to"),
    ("When {first} and {second} went to the park,", "{subject} handed a ball to"),
    ("After {first} and {second} finished lunch,", "{subject} passed the plate to"),
    ("While {first} and {second} were at the office,", "{subject} sent an email to"),
    ("Then {first} and {second} had a long argument,", "{subject} apologised to"),
    ("When {first} and {second} arrived at the party,", "{subject} offered a drink to"),
    ("After {first} and {second} left the museum,", "{subject} showed the ticket to"),
    ("While {first} and {second} waited at the station,", "{subject} lent a phone to"),
    ("When {first} and {second} got to the beach,", "{subject} threw a towel to"),
    ("After {first} and {second} met at the cafe,", "{subject} read the letter to"),
    ("When {first} and {second} were working late,", "{subject} brought coffee to"),
    ("While {first} and {second} sat in the garden,", "{subject} described the plan to"),
    ("After {first} and {second} watched the film,", "{subject} explained the ending to"),
    ("When {first} and {second} visited the library,", "{subject} returned the book to"),
    ("While {first} and {second} walked downtown,", "{subject} pointed the way to"),
]
IOI_B = [
    ("Because {first} and {second} shared a taxi,", "{subject} paid the fare for"),
    ("Once {first} and {second} reached the summit,", "{subject} tossed a rope to"),
    ("Since {first} and {second} joined the choir,", "{subject} sang the part for"),
    ("Before {first} and {second} boarded the train,", "{subject} carried the case for"),
    ("As {first} and {second} toured the factory,", "{subject} mailed the badge to"),
    ("Because {first} and {second} entered the contest,", "{subject} wrote the speech for"),
    ("Once {first} and {second} settled the bill,", "{subject} returned the change to"),
    ("Since {first} and {second} rented the cabin,", "{subject} lit the fire for"),
    ("Before {first} and {second} started the meeting,", "{subject} poured the water for"),
    ("As {first} and {second} crossed the bridge,", "{subject} waved the flag at"),
    ("Because {first} and {second} missed the bus,", "{subject} called a cab for"),
    ("Once {first} and {second} opened the shop,", "{subject} counted the stock for"),
    ("Since {first} and {second} planted the field,", "{subject} drove the tractor for"),
    ("Before {first} and {second} sold the boat,", "{subject} cleaned the deck for"),
    ("As {first} and {second} painted the hall,", "{subject} moved the ladder for"),
]
IOI_NAMES_A = ["Mary", "John", "Alice", "Bob", "Sarah", "Michael", "Emma", "David",
               "Laura", "James", "Anna", "Paul", "Kate", "Steve", "Julia", "Mark",
               "Lisa"]
IOI_NAMES_B = ["Carol", "Chris", "Peter", "Susan", "Daniel", "Helen", "George",
               "Rachel", "Simon", "Claire", "Andrew", "Nicole", "Robert", "Diana",
               "Kevin", "Sophie", "Tom"]

# ── Factual recall ───────────────────────────────────────────────────────────
CAPITALS = {
    "France": "Paris", "England": "London", "Germany": "Berlin", "Italy": "Rome",
    "Spain": "Madrid", "Russia": "Moscow", "Japan": "Tokyo", "Austria": "Vienna",
    "Greece": "Athens", "Ireland": "Dublin", "Norway": "Oslo", "Portugal": "Lisbon",
    "Poland": "Warsaw", "Egypt": "Cairo", "Cuba": "Havana", "Peru": "Lima",
}
FACTUAL_A = [
    "The capital of {e} is", "The capital city of {e} is",
    "{e}'s capital is", "The seat of government of {e} is",
    "Travellers to {e} usually land in its capital,",
    "On a map of {e}, the capital is marked as",
]
FACTUAL_B = [
    "If you fly to {e}, you arrive in the capital,",
    "The government of {e} sits in the city of",
    "Ask anyone in {e} to name the capital and they will say",
    "The largest administrative centre of {e} is",
    "Diplomats posted to {e} live in the capital,",
    "The parliament of {e} meets in",
]

# ── Subject-verb agreement ───────────────────────────────────────────────────
# The correct continuation is "are" for a plural head noun and "is" for singular; the
# distractor is the other one. Each template puts a NUMBER-MISMATCHED attractor noun
# between the head and the verb, which is the standard construction for testing
# whether the model tracks the syntactic head rather than the nearest noun.
AGREEMENT_A = [
    ("The {head} near the {attr}", "the {head} near the {attr}"),
    ("The {head} beside the {attr}", "the {head} beside the {attr}"),
    ("The {head} behind the {attr}", "the {head} behind the {attr}"),
    ("The {head} under the {attr}", "the {head} under the {attr}"),
    ("The {head} across from the {attr}", "the {head} across from the {attr}"),
    ("The {head} in front of the {attr}", "the {head} in front of the {attr}"),
]
AGREEMENT_B = [
    ("The {head} that the {attr} blocked", "the {head} that the {attr} blocked"),
    ("The {head} which the {attr} covered", "the {head} which the {attr} covered"),
    ("The {head} the {attr} concealed", "the {head} the {attr} concealed"),
    ("The {head} left by the {attr}", "the {head} left by the {attr}"),
    ("The {head} placed against the {attr}", "the {head} placed against the {attr}"),
    ("The {head} found beneath the {attr}", "the {head} found beneath the {attr}"),
]
NOUNS_PLURAL_A = ["keys", "books", "cups", "boxes", "chairs", "lamps"]
NOUNS_SINGULAR_A = ["cabinet", "table", "shelf", "window", "door", "desk"]
NOUNS_PLURAL_B = ["plates", "towels", "coins", "ropes", "bottles", "tools"]
NOUNS_SINGULAR_B = ["counter", "bench", "cupboard", "gate", "wall", "drawer"]

# ── Induction ────────────────────────────────────────────────────────────────
# A B ... A -> B. The frame axis here is the DELIMITER and the framing text; the
# entity axis is the token vocabulary the pairs are drawn from.
INDUCTION_A = ["{a} {b}", "{a} -> {b}", "{a} = {b}", "{a} : {b}",
               "{a} | {b}", "{a} , {b}"]
INDUCTION_B = ["{a} / {b}", "{a} ; {b}", "{a} ~ {b}", "{a} * {b}",
               "{a} + {b}", "{a} # {b}"]
# Common concrete nouns rather than the NATO alphabet: nearly all of NATO is
# multi-token under Llama-3's tokenizer (bravo, kilo, sierra, zulu ... all split),
# which left the entity-shifted split with one usable word. The induction circuit is
# indifferent to what the tokens mean -- it copies -- so the only real requirement is
# that they are single tokens, and ordinary words satisfy it far more often.
IND_VOCAB = ["cat", "dog", "tree", "house", "car", "book", "door", "fish", "bird",
             "stone", "river", "cloud", "chair", "table", "green", "blue", "black",
             "white", "north", "south", "east", "west", "gold", "iron", "salt",
             "milk", "bread", "water", "fire", "wind", "snow", "rain", "moon",
             "star", "road", "field", "hill", "lake", "wood", "glass", "paper",
             "metal", "cloth", "horse", "sheep", "wheat", "brick", "rope"]


def single_token(words, skip_filter=False):
    """Keep words that are exactly one token, with a leading space, under EVERY
    target tokenizer.

    Multi-token answers would silently break the metric: logit_diff reads a single
    token id at the final position, so a two-token answer would be scored on its
    first piece only -- which for many names is a shared prefix and therefore not
    diagnostic at all.
    """
    if skip_filter:
        print("  WARNING: tokenizer filter skipped; answers not verified single-token")
        return sorted(words)
    from transformers import AutoTokenizer
    keep = set(words)
    for label, repo in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(repo)
        ok = {w for w in words if len(tok.encode(" " + w, add_special_tokens=False)) == 1}
        if keep - ok:
            print(f"  {label}: dropping {sorted(keep - ok)}")
        keep &= ok
    return sorted(keep)


_TOK_CACHE = {}


def length_matched(records, skip_filter=False):
    """Keep only records whose clean and corrupt prompts tokenize to the SAME length
    under every target tokenizer.

    Activation patching caches per-head SDPA output on a clean forward pass and splices
    it into a corrupt pass at the same tensor positions. That correspondence is only
    meaningful if position i means the same thing in both. IOI corruption swaps two
    single-token names and is length-preserving by construction, but the other families
    substitute different entities -- "France" and "Portugal" are not the same number of
    tokens -- and a length mismatch would shift every position after the substitution.
    The harness pads to a common length, so nothing would error; the patching scores
    would just be quietly measuring the wrong heads.
    """
    if skip_filter:
        return records, 0
    from transformers import AutoTokenizer
    toks = []
    for label, repo in TOKENIZERS:
        if repo not in _TOK_CACHE:
            _TOK_CACHE[repo] = AutoTokenizer.from_pretrained(repo)
        toks.append(_TOK_CACHE[repo])
    kept = []
    for r in records:
        if all(len(t.encode(r["prompt"], add_special_tokens=True))
               == len(t.encode(r["corrupt_prompt"], add_special_tokens=True))
               for t in toks):
            kept.append(r)
    return kept, len(records) - len(kept)


def rec(i, tmpl_id, prompt, corrupt, answer, distractor, **extra):
    d = {
        "id": i, "template_id": tmpl_id,
        "prompt": prompt, "corrupt_prompt": corrupt,
        "answer_token": answer, "distractor_token": distractor,
        # Aliases so data/ioi/run_circuit_faithfulness.py reads these files as-is.
        "io_name": answer, "subject_name": distractor,
    }
    d.update(extra)
    return d


def build_ioi(templates, names, n, rng):
    out = []
    for i in range(n):
        t = i % len(templates)
        opening, cont = templates[t]
        order = "ABBA" if (i // len(templates)) % 2 == 0 else "BABA"
        io, s = rng.sample(names, 2)
        first, second = (io, s) if order == "ABBA" else (s, io)
        prompt = opening.format(first=first, second=second) + " " + cont.format(subject=s)
        cf, cs = (s, io) if order == "ABBA" else (io, s)
        corrupt = opening.format(first=cf, second=cs) + " " + cont.format(subject=io)
        out.append(rec(i, t, prompt, corrupt, io, s, order=order))
    return out


def build_factual(templates, entities, n, rng):
    out = []
    for i in range(n):
        t = i % len(templates)
        e, other = rng.sample(entities, 2)
        # Corruption swaps in a different country, so the correct answer for the
        # corrupted prompt is the distractor -- the same role-swap logic as IOI.
        out.append(rec(i, t, templates[t].format(e=e), templates[t].format(e=other),
                       CAPITALS[e], CAPITALS[other], entity=e))
    return out


def build_agreement(templates, plurals, singulars, n, rng):
    out = []
    for i in range(n):
        t = i % len(templates)
        frame, _ = templates[t]
        plural_head = (i // len(templates)) % 2 == 0
        if plural_head:
            head, attr, ans, dis = rng.choice(plurals), rng.choice(singulars), "are", "is"
            c_head, c_attr = rng.choice(singulars), rng.choice(plurals)
        else:
            head, attr, ans, dis = rng.choice(singulars), rng.choice(plurals), "is", "are"
            c_head, c_attr = rng.choice(plurals), rng.choice(singulars)
        out.append(rec(i, t, frame.format(head=head, attr=attr),
                       frame.format(head=c_head, attr=c_attr), ans, dis,
                       head_number="plural" if plural_head else "singular"))
    return out


def build_induction(templates, vocab, n, rng):
    out = []
    for i in range(n):
        t = i % len(templates)
        pat = templates[t]
        a, b, c, d = rng.sample(vocab, 4)
        # Two pairs, then the first key repeated: the completion requires copying b,
        # not d, so the distractor is the other pair's value.
        prompt = f"{pat.format(a=a, b=b)} {pat.format(a=c, b=d)} {pat.format(a=a, b='').rstrip()}"
        corrupt = f"{pat.format(a=a, b=d)} {pat.format(a=c, b=b)} {pat.format(a=a, b='').rstrip()}"
        out.append(rec(i, t, prompt, corrupt, b, d))
    return out


FAMILIES = {
    "ioi": dict(frames=(IOI_A, IOI_B), ents=(IOI_NAMES_A, IOI_NAMES_B)),
    "factual": dict(frames=(FACTUAL_A, FACTUAL_B),
                    ents=(sorted(CAPITALS)[:8], sorted(CAPITALS)[8:])),
    "agreement": dict(frames=(AGREEMENT_A, AGREEMENT_B),
                      ents=((NOUNS_PLURAL_A, NOUNS_SINGULAR_A),
                            (NOUNS_PLURAL_B, NOUNS_SINGULAR_B))),
    "induction": dict(frames=(INDUCTION_A, INDUCTION_B),
                      ents=(IND_VOCAB, [])),
}


OVERSAMPLE = 6


def balanced_take(records, n, n_templates):
    """Take n records round-robin across template ids.

    Taking the first n after length filtering would silently skew the template mix --
    the filter does not drop uniformly, since some frames place the substitution where
    it more often changes token count. A skewed D or D-prime would confound frame
    identity with frame frequency, which is the confound this whole file removes.
    """
    by_t = {}
    for r in records:
        by_t.setdefault(r["template_id"], []).append(r)
    out, t = [], 0
    while len(out) < n and any(by_t.values()):
        bucket = by_t.get(t % n_templates)
        if bucket:
            out.append(bucket.pop(0))
        t += 1
        if t > n_templates * (n + 1):
            break
    for i, r in enumerate(out):
        r["id"] = i
    return out


def build_filtered(builder, n, n_templates, rng, skip_filter, label):
    """Overgenerate, drop length-mismatched pairs, then take a template-balanced n."""
    raw = builder(n * OVERSAMPLE)
    kept, dropped = length_matched(raw, skip_filter)
    out = balanced_take(kept, n, n_templates)
    if len(out) < n:
        print(f"  WARNING: {label} yielded {len(out)}/{n} length-matched examples")
    if dropped:
        print(f"  {label}: dropped {dropped}/{len(raw)} on token-length mismatch")
    return out


def make_split(family, shift, n, rng, skip_filter):
    """Return (D_records, Dprime_records, meta).

    D always uses bank A of whichever axes are held fixed. Under --shift frame the
    entity pool is the UNION of both banks for both splits, so neither split gets a
    smaller vocabulary than the other -- an unequal pool size would be a second
    difference between D and D-prime and would make the comparison unfair in the
    direction of the hypothesis.
    """
    fa, fb = FAMILIES[family]["frames"]
    ea, eb = FAMILIES[family]["ents"]
    shift_frame = shift in ("frame", "both")
    shift_ent = shift in ("entity", "both")

    d_frames, p_frames = (fa, fb) if shift_frame else (fa, fa)

    if family == "agreement":
        # Same filter-then-split discipline as below: 'cupboard' is multi-token under
        # Pythia, and filtering fixed banks independently left them at 6 and 5.
        plur = single_token(sorted(set(ea[0]) | set(eb[0])), skip_filter)
        sing = single_token(sorted(set(ea[1]) | set(eb[1])), skip_filter)
        if shift_ent:
            d_ent, p_ent = (plur[0::2], sing[0::2]), (plur[1::2], sing[1::2])
        else:
            d_ent = p_ent = (plur, sing)
        D = build_filtered(lambda m: build_agreement(d_frames, d_ent[0], d_ent[1], m, rng),
                           n, len(d_frames), rng, skip_filter, "D")
        P = build_filtered(lambda m: build_agreement(p_frames, p_ent[0], p_ent[1], m, rng),
                           n, len(p_frames), rng, skip_filter, "D'")
    else:
        pool = sorted(set(ea) | set(eb))
        if family == "factual":
            # Filter on the ANSWER (the capital), not the entity: the capital is what
            # the metric reads a logit for.
            keep = set(single_token([CAPITALS[c] for c in pool], skip_filter))
            pool = [c for c in pool if CAPITALS[c] in keep]
        else:
            pool = single_token(pool, skip_filter)
        if shift_ent:
            # Filter FIRST, then split -- not the reverse. Splitting into fixed banks
            # and filtering each independently lets tokenizer attrition fall unevenly:
            # the NATO alphabet this originally used left one bank with 5 usable words
            # and the other with 1. An unequal pool is itself a difference between D
            # and D-prime, so the halves are cut from the survivors and interleaved by
            # sorted position rather than blocked alphabetically.
            d_ent, p_ent = pool[0::2], pool[1::2]
        else:
            d_ent = p_ent = pool
        if min(len(d_ent), len(p_ent)) < 4:
            raise SystemExit(f"{family}: too few usable entities "
                             f"(D={len(d_ent)}, D'={len(p_ent)})")
        builder = {"ioi": build_ioi, "factual": build_factual,
                   "induction": build_induction}[family]
        D = build_filtered(lambda m: builder(d_frames, d_ent, m, rng),
                           n, len(d_frames), rng, skip_filter, "D")
        P = build_filtered(lambda m: builder(p_frames, p_ent, m, rng),
                           n, len(p_frames), rng, skip_filter, "D'")

    meta = {
        "family": family, "shift": shift, "n_examples": n, "seed": SEED,
        "n_templates_D": len(d_frames), "n_templates_Dprime": len(p_frames),
        "frames_disjoint": shift_frame, "entities_disjoint": shift_ent,
        "metric": "logit(answer_token) - logit(distractor_token) at final position",
    }
    return D, P, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=sorted(FAMILIES), default="ioi")
    ap.add_argument("--shift", choices=["frame", "entity", "both"], default="frame")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--all", action="store_true",
                    help="every family x every shift axis")
    ap.add_argument("--skip-tokenizer-filter", action="store_true")
    args = ap.parse_args()

    jobs = ([(f, s) for f in sorted(FAMILIES) for s in ("frame", "entity", "both")]
            if args.all else [(args.family, args.shift)])

    for family, shift in jobs:
        rng = random.Random(SEED)
        print(f"\n=== {family} / shift={shift} ===")
        D, P, meta = make_split(family, shift, args.n, rng, args.skip_tokenizer_filter)
        for tag, recs in (("D", D), ("Dprime", P)):
            path = DATA_DIR / f"{family}-{shift}-{tag}.json"
            path.write_text(json.dumps(
                {"meta": {**meta, "split": tag}, "examples": recs}, indent=1))
            print(f"  {path.name}: {len(recs)} examples, "
                  f"{len({r['template_id'] for r in recs})} templates")
        overlap = ({r["prompt"] for r in D} & {r["prompt"] for r in P})
        # A frame-shifted split that shares prompts would silently measure nothing.
        print(f"  prompt overlap D n D': {len(overlap)}"
              + ("  <-- EXPECTED 0" if overlap and shift != "entity" else ""))


if __name__ == "__main__":
    main()
