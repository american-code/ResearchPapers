"""Generate 50-example factual association dataset for activation patching.

Two relation types:
  capital_of:           "The capital of {subject} is"
  official_language_of: "The official language of {subject} is"

Each example has a clean (subject→object) and corrupt (corrupt_subject→corrupt_object)
prompt. Corruption swaps the subject for a different entity with a different answer.
"""
import json
from pathlib import Path

TEMPLATES = {
    "capital_of":           "The capital of {subject} is",
    "official_language_of": "The official language of {subject} is",
}

RAW_FACTS = [
    # ── capital_of (25) ────────────────────────────────────────────────────────
    # fmt: off
    {"relation": "capital_of", "subject": "France",      "object": "Paris",      "corrupt_subject": "Germany",     "corrupt_object": "Berlin"},
    {"relation": "capital_of", "subject": "Germany",     "object": "Berlin",     "corrupt_subject": "France",      "corrupt_object": "Paris"},
    {"relation": "capital_of", "subject": "Japan",       "object": "Tokyo",      "corrupt_subject": "China",       "corrupt_object": "Beijing"},
    {"relation": "capital_of", "subject": "China",       "object": "Beijing",    "corrupt_subject": "Japan",       "corrupt_object": "Tokyo"},
    {"relation": "capital_of", "subject": "Italy",       "object": "Rome",       "corrupt_subject": "Greece",      "corrupt_object": "Athens"},
    {"relation": "capital_of", "subject": "Greece",      "object": "Athens",     "corrupt_subject": "Italy",       "corrupt_object": "Rome"},
    {"relation": "capital_of", "subject": "Russia",      "object": "Moscow",     "corrupt_subject": "Spain",       "corrupt_object": "Madrid"},
    {"relation": "capital_of", "subject": "Spain",       "object": "Madrid",     "corrupt_subject": "Russia",      "corrupt_object": "Moscow"},
    {"relation": "capital_of", "subject": "Ireland",     "object": "Dublin",     "corrupt_subject": "Portugal",    "corrupt_object": "Lisbon"},
    {"relation": "capital_of", "subject": "Portugal",    "object": "Lisbon",     "corrupt_subject": "Ireland",     "corrupt_object": "Dublin"},
    {"relation": "capital_of", "subject": "Denmark",     "object": "Copenhagen", "corrupt_subject": "Sweden",      "corrupt_object": "Stockholm"},
    {"relation": "capital_of", "subject": "Sweden",      "object": "Stockholm",  "corrupt_subject": "Denmark",     "corrupt_object": "Copenhagen"},
    {"relation": "capital_of", "subject": "Norway",      "object": "Oslo",       "corrupt_subject": "Poland",      "corrupt_object": "Warsaw"},
    {"relation": "capital_of", "subject": "Poland",      "object": "Warsaw",     "corrupt_subject": "Norway",      "corrupt_object": "Oslo"},
    {"relation": "capital_of", "subject": "Hungary",     "object": "Budapest",   "corrupt_subject": "Romania",     "corrupt_object": "Bucharest"},
    {"relation": "capital_of", "subject": "Romania",     "object": "Bucharest",  "corrupt_subject": "Hungary",     "corrupt_object": "Budapest"},
    {"relation": "capital_of", "subject": "Egypt",       "object": "Cairo",      "corrupt_subject": "Turkey",      "corrupt_object": "Ankara"},
    {"relation": "capital_of", "subject": "Turkey",      "object": "Ankara",     "corrupt_subject": "Egypt",       "corrupt_object": "Cairo"},
    {"relation": "capital_of", "subject": "India",       "object": "Delhi",      "corrupt_subject": "Thailand",    "corrupt_object": "Bangkok"},
    {"relation": "capital_of", "subject": "Thailand",    "object": "Bangkok",    "corrupt_subject": "India",       "corrupt_object": "Delhi"},
    {"relation": "capital_of", "subject": "Austria",     "object": "Vienna",     "corrupt_subject": "Belgium",     "corrupt_object": "Brussels"},
    {"relation": "capital_of", "subject": "Belgium",     "object": "Brussels",   "corrupt_subject": "Austria",     "corrupt_object": "Vienna"},
    {"relation": "capital_of", "subject": "Netherlands", "object": "Amsterdam",  "corrupt_subject": "France",      "corrupt_object": "Paris"},
    {"relation": "capital_of", "subject": "Ukraine",     "object": "Kiev",       "corrupt_subject": "Belarus",     "corrupt_object": "Minsk"},
    {"relation": "capital_of", "subject": "Morocco",     "object": "Rabat",      "corrupt_subject": "Tunisia",     "corrupt_object": "Tunis"},
    # ── official_language_of (25) ──────────────────────────────────────────────
    {"relation": "official_language_of", "subject": "France",      "object": "French",     "corrupt_subject": "Spain",       "corrupt_object": "Spanish"},
    {"relation": "official_language_of", "subject": "Spain",       "object": "Spanish",    "corrupt_subject": "France",      "corrupt_object": "French"},
    {"relation": "official_language_of", "subject": "Germany",     "object": "German",     "corrupt_subject": "Italy",       "corrupt_object": "Italian"},
    {"relation": "official_language_of", "subject": "Italy",       "object": "Italian",    "corrupt_subject": "Germany",     "corrupt_object": "German"},
    {"relation": "official_language_of", "subject": "Russia",      "object": "Russian",    "corrupt_subject": "Japan",       "corrupt_object": "Japanese"},
    {"relation": "official_language_of", "subject": "Japan",       "object": "Japanese",   "corrupt_subject": "Russia",      "corrupt_object": "Russian"},
    {"relation": "official_language_of", "subject": "China",       "object": "Chinese",    "corrupt_subject": "Korea",       "corrupt_object": "Korean"},
    {"relation": "official_language_of", "subject": "Korea",       "object": "Korean",     "corrupt_subject": "China",       "corrupt_object": "Chinese"},
    {"relation": "official_language_of", "subject": "Brazil",      "object": "Portuguese", "corrupt_subject": "Spain",       "corrupt_object": "Spanish"},
    {"relation": "official_language_of", "subject": "Egypt",       "object": "Arabic",     "corrupt_subject": "Germany",     "corrupt_object": "German"},
    {"relation": "official_language_of", "subject": "Sweden",      "object": "Swedish",    "corrupt_subject": "Denmark",     "corrupt_object": "Danish"},
    {"relation": "official_language_of", "subject": "Denmark",     "object": "Danish",     "corrupt_subject": "Sweden",      "corrupt_object": "Swedish"},
    {"relation": "official_language_of", "subject": "Greece",      "object": "Greek",      "corrupt_subject": "Turkey",      "corrupt_object": "Turkish"},
    {"relation": "official_language_of", "subject": "Turkey",      "object": "Turkish",    "corrupt_subject": "Greece",      "corrupt_object": "Greek"},
    {"relation": "official_language_of", "subject": "Netherlands", "object": "Dutch",      "corrupt_subject": "Italy",       "corrupt_object": "Italian"},
    {"relation": "official_language_of", "subject": "Portugal",    "object": "Portuguese", "corrupt_subject": "France",      "corrupt_object": "French"},
    {"relation": "official_language_of", "subject": "Mexico",      "object": "Spanish",    "corrupt_subject": "Brazil",      "corrupt_object": "Portuguese"},
    {"relation": "official_language_of", "subject": "Australia",   "object": "English",    "corrupt_subject": "France",      "corrupt_object": "French"},
    {"relation": "official_language_of", "subject": "Norway",      "object": "Norwegian",  "corrupt_subject": "Finland",     "corrupt_object": "Finnish"},
    {"relation": "official_language_of", "subject": "Finland",     "object": "Finnish",    "corrupt_subject": "Norway",      "corrupt_object": "Norwegian"},
    {"relation": "official_language_of", "subject": "Austria",     "object": "German",     "corrupt_subject": "France",      "corrupt_object": "French"},
    {"relation": "official_language_of", "subject": "Morocco",     "object": "Arabic",     "corrupt_subject": "Germany",     "corrupt_object": "German"},
    {"relation": "official_language_of", "subject": "Poland",      "object": "Polish",     "corrupt_subject": "Italy",       "corrupt_object": "Italian"},
    {"relation": "official_language_of", "subject": "Argentina",   "object": "Spanish",    "corrupt_subject": "Brazil",      "corrupt_object": "Portuguese"},
    {"relation": "official_language_of", "subject": "Ukraine",     "object": "Ukrainian",  "corrupt_subject": "Russia",      "corrupt_object": "Russian"},
    # fmt: on
]


def main() -> None:
    data_dir = Path(__file__).parent
    examples = []
    for i, f in enumerate(RAW_FACTS):
        template = TEMPLATES[f["relation"]]
        examples.append({
            "id":             i,
            "relation":       f["relation"],
            "subject":        f["subject"],
            "object":         f["object"],
            "corrupt_subject": f["corrupt_subject"],
            "corrupt_object":  f["corrupt_object"],
            "prompt":         template.format(subject=f["subject"]),
            "corrupt_prompt": template.format(subject=f["corrupt_subject"]),
        })

    dataset = {
        "meta": {
            "n_examples": len(examples),
            "templates": TEMPLATES,
            "corruption_strategy": "subject_swap",
            "description": (
                "50 factual association examples (25 capital_of + 25 official_language_of). "
                "Each example pairs a clean prompt (subject→object) with a corrupt prompt "
                "(corrupt_subject→corrupt_object). Patching metric: normalised logit-diff "
                "recovery: (logit(object) - logit(corrupt_object)) at last token position."
            ),
        },
        "examples": examples,
    }

    out = data_dir / "dataset.json"
    out.write_text(json.dumps(dataset, indent=2))
    print(f"Saved {len(examples)} examples → {out}")
    for rel in TEMPLATES:
        n = sum(1 for e in examples if e["relation"] == rel)
        print(f"  {rel}: {n}")


if __name__ == "__main__":
    main()
