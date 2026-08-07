#!/usr/bin/env python3
"""
Multi-template IOI dataset with balanced name ordering.

The v1 dataset (dataset.json) uses a single sentence frame,
"When {S} and {IO} went to the store, {S} gave a bottle to", over 20 names, with
one name ordering throughout. Wang et al. (2022) use 15 templates and balance both
ABBA and BABA orderings precisely because a single frame confounds circuit structure
with positional structure: a head that looks circuit-critical may be responding to a
fixed token position rather than to the IOI computation.

This generator produces:
  - 15 sentence templates spanning different verbs, objects and locations
  - balanced ABBA / BABA name ordering (50/50)
  - a name pool filtered to names that are single-token under BOTH tokenizers
  - corrupted counterparts produced by swapping S and IO, preserving length

ABBA:  "When {A} and {B} went to the store, {B} gave a bottle to"   -> IO = A
BABA:  "When {B} and {A} went to the store, {B} gave a bottle to"   -> IO = A

In both cases the subject (the repeated name) is B and the indirect object is A;
the orderings differ in whether the IO appears first or second in the opening clause.

Usage:
  python3 generate_dataset_v2.py --n 200
Output: data/ioi/dataset-v2.json
"""

import argparse
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent
SEED = 42

# (opening clause with {first}/{second}, continuation with {subject})
TEMPLATES = [
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

# Candidate pool; filtered at generation time to names that are single-token under
# every tokenizer we target.
NAME_POOL = [
    "Mary", "John", "Alice", "Bob", "Sarah", "Michael", "Emma", "David", "Laura",
    "James", "Anna", "Paul", "Kate", "Steve", "Julia", "Mark", "Lisa", "Tom",
    "Carol", "Chris", "Peter", "Susan", "Daniel", "Helen", "George", "Rachel",
    "Simon", "Claire", "Andrew", "Nicole", "Robert", "Diana", "Kevin", "Sophie",
]

TOKENIZERS = [
    ("llama", "mlx-community/Llama-3.2-3B-bf16"),
    ("pythia", "EleutherAI/pythia-1.4b"),
]


def single_token_names(pool: list[str]) -> list[str]:
    """Keep only names that tokenize to exactly one token, with a leading space,
    under every target tokenizer."""
    from transformers import AutoTokenizer
    keep = set(pool)
    for label, repo in TOKENIZERS:
        tok = AutoTokenizer.from_pretrained(repo)
        ok = set()
        for n in pool:
            ids = tok.encode(" " + n, add_special_tokens=False)
            if len(ids) == 1:
                ok.add(n)
        dropped = keep - ok
        if dropped:
            print(f"  {label}: dropping {sorted(dropped)}")
        keep &= ok
    return sorted(keep)


def build(names: list[str], n: int, rng: random.Random) -> list[dict]:
    out = []
    for i in range(n):
        tmpl_idx = i % len(TEMPLATES)
        opening, cont = TEMPLATES[tmpl_idx]
        order = "ABBA" if (i // len(TEMPLATES)) % 2 == 0 else "BABA"

        io_name, s_name = rng.sample(names, 2)      # A = IO, B = S
        first, second = (io_name, s_name) if order == "ABBA" else (s_name, io_name)

        if "{subject}" in opening:                   # single-clause template
            prompt = opening.format(first=first, second=second, subject=s_name)
        else:
            prompt = opening.format(first=first, second=second) + " " + cont.format(subject=s_name)

        # corruption: swap the roles of IO and S throughout, preserving length
        c_first, c_second = (s_name, io_name) if order == "ABBA" else (io_name, s_name)
        if "{subject}" in opening:
            corrupt = opening.format(first=c_first, second=c_second, subject=io_name)
        else:
            corrupt = (opening.format(first=c_first, second=c_second) + " "
                       + cont.format(subject=io_name))

        out.append({
            "id": i,
            "template_id": tmpl_idx,
            "order": order,
            "prompt": prompt,
            "corrupt_prompt": corrupt,
            "io_name": io_name,
            "subject_name": s_name,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="dataset-v2.json")
    ap.add_argument("--skip-tokenizer-filter", action="store_true",
                    help="use the raw pool (offline); the filter needs both tokenizers")
    args = ap.parse_args()
    rng = random.Random(SEED)

    if args.skip_tokenizer_filter:
        names = sorted(NAME_POOL)
        print("WARNING: tokenizer filter skipped; names not verified single-token")
    else:
        print("Filtering names to single-token under all target tokenizers …")
        names = single_token_names(NAME_POOL)
    print(f"  {len(names)} usable names: {names}")
    if len(names) < 4:
        raise SystemExit("too few single-token names to build a dataset")

    examples = build(names, args.n, rng)
    by_order = {"ABBA": 0, "BABA": 0}
    for e in examples:
        by_order[e["order"]] += 1

    doc = {
        "meta": {
            "version": 2,
            "description": ("Multi-template IOI dataset with balanced ABBA/BABA name "
                            "ordering. Supersedes dataset.json, which used a single "
                            "template and one ordering."),
            "n_examples": len(examples),
            "n_templates": len(TEMPLATES),
            "n_names": len(names),
            "names": names,
            "order_balance": by_order,
            "seed": SEED,
            "templates": [o + (" " + c if "{subject}" not in o else "")
                          for o, c in TEMPLATES],
            "corruption_strategy": "swap IO and subject roles, preserving sequence length",
        },
        "examples": examples,
    }
    out = DATA_DIR / args.out
    out.write_text(json.dumps(doc, indent=1))
    print(f"\n{len(examples)} examples, {len(TEMPLATES)} templates, "
          f"order balance {by_order}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
