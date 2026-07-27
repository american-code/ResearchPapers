"""
Generate the IOI (indirect object identification) dataset for circuit tracing.

Prompt template: "When {S} and {IO} went to the store, {S} gave a bottle to"
- S  = subject (appears twice)
- IO = indirect object (appears once; correct completion)

Each record stores:
  prompt        - the full prompt string
  io_name       - the IO name (correct next-token prediction)
  subject_name  - the subject name
  io_token_pos  - 0-indexed position of the IO token in the token sequence
  s1_token_pos  - position of the first subject token
  s2_token_pos  - position of the second subject token
  last_token_pos- position of the final "to" token (query position)
  tokens        - full token list for verification
"""

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Names: all confirmed single-token in GPT-2 / cl100k tokenisers
# ---------------------------------------------------------------------------
NAMES = [
    "Mary", "John", "Alice", "Bob", "Sarah",
    "Michael", "James", "Emma", "David", "Lisa",
    "Tom", "Kate", "Chris", "Anna", "Mark",
    "Laura", "Paul", "Julia", "Steve", "Carol",
]
assert len(NAMES) == 20

# ---------------------------------------------------------------------------
# Tokeniser: use GPT-2 via transformers if available, else tiktoken, else stub
# ---------------------------------------------------------------------------
def load_tokeniser():
    try:
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        def encode(text):
            return tok.convert_ids_to_tokens(tok.encode(text))
        return encode, "gpt2-transformers"
    except Exception:
        pass
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def encode(text):
            return [enc.decode([t]) for t in enc.encode(text)]
        return encode, "gpt2-tiktoken"
    except Exception:
        pass
    # Fallback: whitespace + punctuation split — good enough to get positions
    # right for single-token names in simple prompts.
    import re
    def encode(text):
        # Approximate GPT-2 byte-pair by treating each word (with leading space)
        # as a token, which is accurate for this fixed template + single-token names.
        tokens = []
        for m in re.finditer(r"\S+|\s+\S+", text):
            tokens.append(m.group())
        return tokens
    return encode, "fallback-whitespace"


encode, tokeniser_name = load_tokeniser()


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
def make_example(s_name: str, io_name: str, idx: int) -> dict:
    prompt = f"When {s_name} and {io_name} went to the store, {s_name} gave a bottle to"
    tokens = encode(prompt)

    # Locate token positions by scanning for the name strings.
    # Names are single tokens; leading-space variants differ by tokeniser.
    def find_name_positions(tokens, name):
        positions = []
        for i, t in enumerate(tokens):
            if t.strip() == name:
                positions.append(i)
        return positions

    s_positions  = find_name_positions(tokens, s_name)
    io_positions = find_name_positions(tokens, io_name)

    if len(s_positions) != 2:
        raise ValueError(f"Expected 2 subject positions for '{s_name}', got {s_positions} in {tokens}")
    if len(io_positions) != 1:
        raise ValueError(f"Expected 1 IO position for '{io_name}', got {io_positions} in {tokens}")

    return {
        "id": idx,
        "prompt": prompt,
        "io_name": io_name,
        "subject_name": s_name,
        "s1_token_pos": s_positions[0],
        "io_token_pos": io_positions[0],
        "s2_token_pos": s_positions[1],
        "last_token_pos": len(tokens) - 1,   # final "to"
        "n_tokens": len(tokens),
        "tokens": tokens,
    }


def generate(n: int = 100, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    # Build a pool of all ordered (S, IO) pairs, then sample without replacement
    all_pairs = [(s, io) for s in NAMES for io in NAMES if s != io]  # 20*19 = 380 pairs
    chosen = rng.sample(all_pairs, n)
    for idx, (s_name, io_name) in enumerate(chosen):
        examples.append(make_example(s_name, io_name, idx))
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_path = Path(__file__).parent / "dataset.json"
    examples = generate(100)

    payload = {
        "meta": {
            "n_examples": len(examples),
            "template": "When {S} and {IO} went to the store, {S} gave a bottle to",
            "names": NAMES,
            "tokeniser": tokeniser_name,
            "seed": 42,
        },
        "examples": examples,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(examples)} examples → {out_path}  (tokeniser: {tokeniser_name})")

    # ------------------------------------------------------------------
    # Spot-check: print 5 examples
    # ------------------------------------------------------------------
    print("\n--- Spot-check (5 examples) ---\n")
    for ex in examples[:5]:
        toks = ex["tokens"]
        print(f"[{ex['id']:03d}] {ex['prompt']}")
        print(f"      subject={ex['subject_name']!r}  IO={ex['io_name']!r}")
        print(f"      s1@{ex['s1_token_pos']} → {toks[ex['s1_token_pos']]!r}")
        print(f"      io@{ex['io_token_pos']} → {toks[ex['io_token_pos']]!r}")
        print(f"      s2@{ex['s2_token_pos']} → {toks[ex['s2_token_pos']]!r}")
        print(f"      last@{ex['last_token_pos']} → {toks[ex['last_token_pos']]!r}")
        print(f"      tokens ({ex['n_tokens']}): {toks}")
        print()
