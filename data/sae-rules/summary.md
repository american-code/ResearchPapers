# SAE Neurosymbolic Rule Extraction — Summary

Generated: 2026-07-29 16:01:27

## Dataset & Models

| Model | Layer | d_in | dict_size | Corpus |
|---|---|---|---|---|
| Mistral-7B-v0.3 (4-bit) | 16 | 4096 | 16384 | wikitext-103-raw-v1 |
| Llama-3.2-3B (bf16)     | 14 | 3072 | 16384 | wikitext-103-raw-v1 |

Token budget: 100,000 per model (of 500,000 collected)

## Mistral-7B-layer16

- Active features (mean_freq > 0.01): **13806** / 16384
- Decision tree overall accuracy: **0.432**
- POS classes modelled: ADJ, FUNC, NOUN, OTHER, VERB

### Top 20 Most Interpretable Features

| feature_id | label | mean_freq | max_act | best_POS | precision |
|---|---|---|---|---|---|
| 15025 | token-" | 0.9657 | 7.55 | NOUN | 0.431 |
| 2678 | token-" | 0.9484 | 10.41 | NOUN | 0.431 |
| 7554 | token-" | 0.9385 | 10.06 | NOUN | 0.430 |
| 10973 | token-" | 0.9151 | 7.03 | NOUN | 0.431 |
| 10665 | token-" | 0.9114 | 10.15 | NOUN | 0.430 |
| 11923 | token-" | 0.9097 | 7.58 | NOUN | 0.429 |
| 12197 | token-1 | 0.8920 | 1.46 | NOUN | 0.431 |
| 12668 | function-word | 0.8711 | 10.71 | NOUN | 0.429 |
| 11010 | token-" | 0.8260 | 10.22 | NOUN | 0.430 |
| 7573 | function-word | 0.8213 | 1.14 | NOUN | 0.429 |
| 10909 | token-" | 0.8064 | 1.65 | NOUN | 0.431 |
| 8686 | function-word | 0.7964 | 1.47 | NOUN | 0.431 |
| 7566 | token-" | 0.7892 | 8.70 | NOUN | 0.431 |
| 16213 | token-description | 0.7870 | 0.78 | NOUN | 0.430 |
| 15407 | token-carried | 0.7696 | 3.72 | NOUN | 0.431 |
| 12246 | number | 0.6999 | 1.35 | NOUN | 0.428 |
| 9876 | token-( | 0.6840 | 0.60 | NOUN | 0.427 |
| 1372 | token-m | 0.6814 | 1.37 | NOUN | 0.427 |
| 14454 | token-, | 0.6792 | 1.44 | NOUN | 0.430 |
| 6708 | punctuation | 0.6642 | 0.90 | NOUN | 0.432 |

## Llama-3.2-3B-layer14

- Active features (mean_freq > 0.01): **14383** / 16384
- Decision tree overall accuracy: **0.432**
- POS classes modelled: ADJ, FUNC, NOUN, OTHER, VERB

### Top 20 Most Interpretable Features

| feature_id | label | mean_freq | max_act | best_POS | precision |
|---|---|---|---|---|---|
| 5528 | token-schools | 0.9951 | 7.07 | NOUN | 0.430 |
| 11027 | function-word | 0.9885 | 4.95 | NOUN | 0.430 |
| 14897 | token- | 0.9860 | 6.31 | NOUN | 0.430 |
| 13830 | token-of | 0.9817 | 5.11 | NOUN | 0.431 |
| 12776 | token-way | 0.9725 | 4.62 | NOUN | 0.430 |
| 8092 | token-j | 0.9123 | 4.22 | NOUN | 0.430 |
| 538 | token-travel | 0.9121 | 4.00 | NOUN | 0.432 |
| 464 | token-, | 0.9095 | 2.92 | NOUN | 0.431 |
| 12775 | token-concluded | 0.9082 | 4.21 | NOUN | 0.431 |
| 10048 | token-agricultural | 0.8978 | 4.48 | NOUN | 0.431 |
| 15250 | token-ress | 0.8725 | 5.36 | NOUN | 0.430 |
| 15636 | token-les | 0.8249 | 3.30 | NOUN | 0.431 |
| 7646 | token-farm | 0.8108 | 2.74 | NOUN | 0.429 |
| 1947 | token-as | 0.8033 | 1.41 | NOUN | 0.431 |
| 3807 | token-of | 0.8031 | 2.69 | NOUN | 0.431 |
| 2309 | function-word | 0.8011 | 4.99 | NOUN | 0.431 |
| 15992 | token-operations | 0.7976 | 3.81 | NOUN | 0.431 |
| 15964 | token-nominated | 0.7923 | 4.00 | NOUN | 0.431 |
| 802 | function-word | 0.7817 | 1.72 | NOUN | 0.431 |
| 12911 | token-ty | 0.7789 | 3.31 | NOUN | 0.431 |

## Top 5 Shared Features (Cross-Model)

| label | Mistral feat | Llama feat | freq-ratio sim |
|---|---|---|---|
| punctuation | 6708 | 12873 | 0.999 |
| token-are | 3576 | 6428 | 0.987 |
| number | 12246 | 9955 | 0.927 |
| token-for | 3380 | 6325 | 0.919 |
| token-3 | 16266 | 6044 | 0.893 |

## Example Rules in Plain English

- **Mistral-7B-layer16**: when SAE feature 12861 fires (activation > 0), the token is likely **NOUN** (precision 43%, recall 57%, support 12,369 tokens)
- **Mistral-7B-layer16**: when SAE feature 15654 fires (activation > 0), the token is likely **NOUN** (precision 43%, recall 53%, support 11,394 tokens)
- **Mistral-7B-layer16**: when SAE feature 14487 fires (activation > 0), the token is likely **NOUN** (precision 43%, recall 55%, support 11,866 tokens)

## Notes

- Cross-model cosine similarity is undefined (Mistral d_in=4096 ≠ Llama d_in=3072). Shared features identified via label-matching + activation frequency ratio.
- Mistral weights are 4-bit quantized (mlx-community); Llama weights are bf16. Quantization is a potential confound.
- POS labels are heuristic (suffix/function-word lookup), not gold-standard tagging.
