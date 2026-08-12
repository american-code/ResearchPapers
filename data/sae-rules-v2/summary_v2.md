# SAE Neurosymbolic Rule Extraction v2 — Summary

Generated: 2026-07-29 18:50:11

## What was fixed from v1

1. **Probe targets**: POS replaced by NER label, dependency role, is-NE, sentence-position quartile (spacy)
2. **Feature labeling**: Contrastive PMI-based labels (HIGH top-100 vs LOW random-100) instead of non-contrastive top-token frequency
3. **Cross-model**: CKA in token space + Pearson correlation (not direct cosine; d_in differs)

## Probe AUC Scores

| Model | Target | Class | AUC |
|---|---|---|---|
| mistral | ner | person | 0.895 |
| mistral | ner | org | 0.890 |
| mistral | ner | gpe | 0.955 |
| mistral | ner | date | 0.875 |
| mistral | ner | number | 0.918 |
| mistral | ner | other | 0.779 |
| mistral | dep | nsubj | 0.887 |
| mistral | dep | dobj | 0.921 |
| mistral | dep | prep | 0.822 |
| mistral | dep | det | 0.998 |
| mistral | dep | punct | 0.993 |
| mistral | dep | other | 0.771 |
| mistral | is_ne | NE | 0.786 |
| mistral | pos_q | Q1 | 0.736 |
| mistral | pos_q | Q2 | 0.818 |
| mistral | pos_q | Q3 | 0.825 |
| mistral | pos_q | Q4 | 0.810 |
| llama | ner | person | 0.983 |
| llama | ner | org | 0.984 |
| llama | ner | gpe | 0.995 |
| llama | ner | date | 0.974 |
| llama | ner | number | 0.987 |
| llama | ner | other | 0.942 |
| llama | dep | nsubj | 0.979 |
| llama | dep | dobj | 0.992 |
| llama | dep | prep | 0.950 |
| llama | dep | det | 1.000 |
| llama | dep | punct | 1.000 |
| llama | dep | other | 0.932 |
| llama | is_ne | NE | 0.907 |
| llama | pos_q | Q1 | 0.885 |
| llama | pos_q | Q2 | 0.941 |
| llama | pos_q | Q3 | 0.948 |
| llama | pos_q | Q4 | 0.933 |

## Top 10 Most Discriminative Features per Model

| Model | Feature | Contrastive Label | Variance | Sparsity |
|---|---|---|---|---|
| mistral | 5787 | @ | 2.2241 | 0.004 |
| mistral | 8076 | @ | 1.7986 | 0.004 |
| mistral | 10355 | in | 1.6713 | 0.004 |
| mistral | 8864 | @ | 1.6557 | 0.004 |
| mistral | 15191 | in | 1.6402 | 0.004 |
| mistral | 7946 | in | 1.6125 | 0.004 |
| mistral | 8061 | 1 | 1.6041 | 0.004 |
| mistral | 1557 | in | 1.5985 | 0.004 |
| mistral | 3554 | 1 | 1.5305 | 0.004 |
| mistral | 14393 | in | 1.4950 | 0.004 |
| llama | 8544 | with | 10.6567 | 0.002 |
| llama | 10115 | with | 10.4262 | 0.002 |
| llama | 1552 | with | 10.1138 | 0.002 |
| llama | 9747 | with | 9.6738 | 0.002 |
| llama | 14107 | with | 9.2926 | 0.002 |
| llama | 1487 | with | 9.1033 | 0.002 |
| llama | 5858 | with | 9.1020 | 0.002 |
| llama | 5953 | of | 9.0410 | 0.002 |
| llama | 11299 | with_the | 8.8564 | 0.002 |
| llama | 8767 | with | 8.8207 | 0.002 |

## Top Feature-to-Concept Rules (plain English)

- **mistral** feature 4796 (coef=+0.526) → predicts `ner=person` with AUC 0.89
- **mistral** feature 9578 (coef=+0.480) → predicts `ner=org` with AUC 0.89
- **mistral** feature 11125 (coef=+0.645) → predicts `ner=gpe` with AUC 0.95
- **mistral** feature 11125 (coef=+0.692) → predicts `ner=date` with AUC 0.87
- **mistral** feature 15293 (coef=+1.086) → predicts `ner=number` with AUC 0.92
- **mistral** feature 366 (coef=-0.527) → predicts `ner=other` with AUC 0.78
- **mistral** feature 2432 (coef=+0.502) → predicts `dep=nsubj` with AUC 0.89
- **mistral** feature 8392 (coef=+0.522) → predicts `dep=dobj` with AUC 0.92
- **mistral** feature 2555 (coef=+0.534) → predicts `dep=prep` with AUC 0.82
- **mistral** feature 5114 (coef=+0.064) → predicts `dep=det` with AUC 1.00
- **mistral** feature 8458 (coef=+0.368) → predicts `dep=punct` with AUC 0.99
- **mistral** feature 8392 (coef=-0.568) → predicts `dep=other` with AUC 0.77
- **mistral** feature 366 (coef=+0.541) → predicts `is_ne=NE` with AUC 0.79
- **mistral** feature 1296 (coef=-0.621) → predicts `pos_q=Q1` with AUC 0.74
- **mistral** feature 15293 (coef=+0.726) → predicts `pos_q=Q2` with AUC 0.82
- **mistral** feature 15125 (coef=+0.546) → predicts `pos_q=Q3` with AUC 0.82
- **mistral** feature 11473 (coef=+0.551) → predicts `pos_q=Q4` with AUC 0.81
- **llama** feature 2240 (coef=-0.642) → predicts `ner=person` with AUC 0.98
- **llama** feature 5121 (coef=+0.540) → predicts `ner=org` with AUC 0.98
- **llama** feature 12516 (coef=+0.626) → predicts `ner=gpe` with AUC 1.00
- **llama** feature 2445 (coef=+0.600) → predicts `ner=date` with AUC 0.97
- **llama** feature 3445 (coef=+0.830) → predicts `ner=number` with AUC 0.99
- **llama** feature 13749 (coef=-0.652) → predicts `ner=other` with AUC 0.94
- **llama** feature 9559 (coef=-0.442) → predicts `dep=nsubj` with AUC 0.98
- **llama** feature 3511 (coef=+0.398) → predicts `dep=dobj` with AUC 0.99
- **llama** feature 2844 (coef=-0.476) → predicts `dep=prep` with AUC 0.95
- **llama** feature 13108 (coef=-0.271) → predicts `dep=det` with AUC 1.00
- **llama** feature 8826 (coef=+0.470) → predicts `dep=punct` with AUC 1.00
- **llama** feature 13749 (coef=-0.532) → predicts `dep=other` with AUC 0.93
- **llama** feature 8525 (coef=-0.601) → predicts `is_ne=NE` with AUC 0.91
- **llama** feature 3092 (coef=+0.643) → predicts `pos_q=Q1` with AUC 0.88
- **llama** feature 1697 (coef=+0.599) → predicts `pos_q=Q2` with AUC 0.94
- **llama** feature 11022 (coef=-0.581) → predicts `pos_q=Q3` with AUC 0.95
- **llama** feature 686 (coef=-0.534) → predicts `pos_q=Q4` with AUC 0.93

## Cross-Model Alignment

- **Linear CKA score** (n=8000): **0.016921**
- Feature pairs with Pearson r > 0.5: **2**

### Top 5 Cross-Model Feature Matches

| Mistral ID | Llama ID | Pearson r | Mistral label | Llama label |
|---|---|---|---|---|
| 1917 | 15327 | 0.684 | game | was |
| 8369 | 4425 | 0.552 | were | of |

## What to Try Next

- **More probe targets**: coreference distance, sentence depth (parse tree depth), argument structure
- **Sparse probing**: L1-regularized logistic regression to identify the minimal feature set
- **Layer sweep**: run the same analysis at layers 8 and 24 to see where semantics crystallize
- **Cross-architecture clustering**: use the matched feature pairs to seed cross-model concept clusters
- **Causal intervention**: apply activation patches on top-matched features to test causal role
