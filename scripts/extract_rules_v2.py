#!/usr/bin/env python3
"""
Neurosymbolic rule extraction v2 — semantic probing + contrastive labeling.
Fixes v1: POS→semantics targets, non-contrastive labels, CKA cross-model alignment.
"""

import os, sys, json, time, gc
import numpy as np
from pathlib import Path
from collections import Counter

REPO     = Path("/Users/lab-02/ResearchPapers")
OUT      = REPO / "data/sae-rules-v2"
OUT.mkdir(parents=True, exist_ok=True)

ACT_M    = REPO / "data/activations/mistral-7b-layer16/activations.npy"
ACT_L    = REPO / "data/activations/llama-3b-layer16/activations.npy"
META_M   = REPO / "data/activations/mistral-7b-layer16/metadata.json"
META_L   = REPO / "data/activations/llama-3b-layer16/metadata.json"
SAE_M    = REPO / "data/sae-runs/mistral-7b-layer16/checkpoint_final.npz"
SAE_L    = REPO / "data/sae-runs/llama-3b-layer16/checkpoint_final.npz"

N_TOKENS        = 100_000
N_CKA           = 8_000    # subsample for CKA (full 100k × 16384 is ~100GB kernel)
BATCH_SIZE      = 2_000
TOP_VAR         = 500      # top features by activation variance
TOP_CONTRASTIVE = 200      # top features for cross-model Pearson matching
HIGH_K          = 100
LOW_K           = 100
MIN_SPARSITY    = 0.005
MAX_SPARSITY    = 0.40

NER_CLASSES  = ["person", "org", "gpe", "date", "number", "other"]
DEP_CLASSES  = ["nsubj", "dobj", "prep", "det", "punct", "other"]
POSQ_CLASSES = ["Q1", "Q2", "Q3", "Q4"]
NE_CLASSES   = ["non-NE", "NE"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0: SAE encode raw activations → (N_TOKENS, 16384) float16
# ─────────────────────────────────────────────────────────────────────────────
def encode_sae(act_path, meta_path, sae_path, label):
    meta  = json.load(open(meta_path))
    shape = tuple(meta["activations_shape"])
    dtype = meta["activations_dtype"]
    log(f"[{label}] memmap {act_path.name}: shape={shape} dtype={dtype}")
    raw   = np.memmap(act_path, dtype=dtype, mode="r", shape=shape)

    ckpt  = np.load(sae_path)
    W_enc = ckpt["W_enc"].astype(np.float32)   # (d_in, 16384)
    b_enc = ckpt["b_enc"].astype(np.float32)   # (16384,)
    D     = b_enc.shape[0]

    log(f"[{label}] Encoding {N_TOKENS} tokens → SAE features (D={D}) ...")
    out = np.zeros((N_TOKENS, D), dtype=np.float16)
    for s in range(0, N_TOKENS, BATCH_SIZE):
        e     = min(s + BATCH_SIZE, N_TOKENS)
        batch = raw[s:e].astype(np.float32)
        z     = np.maximum(0.0, batch @ W_enc + b_enc)
        out[s:e] = z.astype(np.float16)
        if (s // BATCH_SIZE) % 10 == 0:
            log(f"  {e}/{N_TOKENS}")
    del raw, ckpt, W_enc, b_enc
    return out     # (N_TOKENS, 16384)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Token strings + spacy semantic labels
# ─────────────────────────────────────────────────────────────────────────────
def get_tokens_and_labels(tokenizer_name):
    import spacy
    from transformers import AutoTokenizer
    from datasets import load_dataset

    log(f"Loading tokenizer: {tokenizer_name}")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    log("Loading wikitext-103 ...")
    ds   = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                        split="train", trust_remote_code=False)
    texts = [t for t in ds["text"] if t.strip()]
    full  = " ".join(texts)

    log("Tokenizing full text with offset mapping ...")
    # HF tokenizers can't do offset_mapping on huge strings in one shot.
    # Tokenize in 50k-char chunks, adjusting char offsets.
    token_ids_all  = []
    char_offsets_all = []
    chunk_size_chars = 50_000
    pos = 0
    while len(token_ids_all) < N_TOKENS and pos < len(full):
        chunk = full[pos: pos + chunk_size_chars]
        enc   = tok(chunk, return_offsets_mapping=True, add_special_tokens=False)
        for tid, (cs, ce) in zip(enc["input_ids"], enc["offset_mapping"]):
            token_ids_all.append(tid)
            char_offsets_all.append((cs + pos, ce + pos))
            if len(token_ids_all) >= N_TOKENS:
                break
        pos += chunk_size_chars - 200   # 200-char overlap to avoid split-token seams

    token_ids_all  = token_ids_all[:N_TOKENS]
    char_offsets_all = char_offsets_all[:N_TOKENS]
    token_strs = [tok.decode([tid]) for tid in token_ids_all]

    # Determine how much text we actually need
    max_char = char_offsets_all[-1][1] if char_offsets_all else 0
    text_needed = full[: max_char + 1]
    log(f"  {len(token_strs)} tokens spanning {max_char:,} chars")

    # Run spacy in 100k-char chunks, build char→label array
    log("Running spacy on corpus ...")
    nlp = spacy.load("en_core_web_sm")
    NER_MAP = {
        "PERSON": 0, "ORG": 1, "GPE": 2, "DATE": 3,
        "CARDINAL": 4, "ORDINAL": 4, "MONEY": 4, "PERCENT": 4, "QUANTITY": 4,
    }
    DEP_MAP = {
        "nsubj": 0, "nsubjpass": 0, "csubj": 0,
        "dobj": 1, "attr": 1, "oprd": 1,
        "pobj": 2, "prep": 2,
        "det": 3,
        "punct": 4,
    }

    # We need labels for every character index up to max_char.
    # Store 4 label arrays indexed by char position (only for chars we need).
    # To save memory, store as compressed: process chunk by chunk, mark tokens as we go.
    char_ner  = np.full(max_char + 2, 5, dtype=np.int8)
    char_dep  = np.full(max_char + 2, 5, dtype=np.int8)
    char_ne   = np.zeros(max_char + 2, dtype=np.int8)
    char_q    = np.zeros(max_char + 2, dtype=np.int8)

    SP_CHUNK  = 80_000
    sp_pos    = 0
    while sp_pos <= max_char:
        chunk_end = min(sp_pos + SP_CHUNK, len(text_needed))
        chunk     = text_needed[sp_pos:chunk_end]
        try:
            doc = nlp(chunk)
        except Exception as e:
            log(f"  spacy error at {sp_pos}: {e}")
            sp_pos += SP_CHUNK
            continue
        for sent in doc.sents:
            sp_toks = list(sent)
            n_sp    = len(sp_toks)
            for i, st in enumerate(sp_toks):
                q      = min(int(i / max(n_sp, 1) * 4), 3)
                ner_v  = NER_MAP.get(st.ent_type_, 5) if st.ent_type_ else 5
                dep_v  = DEP_MAP.get(st.dep_, 5)
                ne_v   = 1 if st.ent_iob_ in ("B", "I") else 0
                ci_s   = st.idx + sp_pos
                ci_e   = ci_s + len(st.text)
                if ci_e > max_char + 2:
                    ci_e = max_char + 2
                if ci_s < max_char + 2:
                    char_ner[ci_s:ci_e] = ner_v
                    char_dep[ci_s:ci_e] = dep_v
                    char_ne [ci_s:ci_e] = ne_v
                    char_q  [ci_s:ci_e] = q
        sp_pos += SP_CHUNK - 500   # overlap to avoid sentence-boundary misses
        if sp_pos % 400_000 < SP_CHUNK:
            log(f"  spacy progress: {sp_pos:,}/{max_char:,} chars")

    log("  Aligning labels to token positions ...")
    ner_labels = np.array([int(char_ner[cs]) for cs, ce in char_offsets_all], dtype=np.int8)
    dep_labels = np.array([int(char_dep[cs]) for cs, ce in char_offsets_all], dtype=np.int8)
    is_ne      = np.array([int(char_ne [cs]) for cs, ce in char_offsets_all], dtype=np.int8)
    pos_q      = np.array([int(char_q  [cs]) for cs, ce in char_offsets_all], dtype=np.int8)

    del char_ner, char_dep, char_ne, char_q
    return token_strs, ner_labels, dep_labels, is_ne, pos_q


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Contrastive feature labeling (top 500 by variance)
# ─────────────────────────────────────────────────────────────────────────────
def contrastive_feature_labeling(sae_acts, token_strs, label):
    """Returns list of feature dicts, sorted by variance desc (top TOP_VAR)."""
    log(f"[{label}] Computing feature statistics ...")
    A = sae_acts.astype(np.float32)   # (N, D)
    N, D = A.shape

    mean_act  = A.mean(axis=0)        # (D,)
    var_act   = A.var(axis=0)         # (D,)
    sparsity  = (A > 0.1).mean(axis=0)  # fraction of tokens where feat > 0.1

    top_idx = np.argsort(var_act)[::-1][:TOP_VAR].tolist()
    log(f"  Top {TOP_VAR} features selected by variance")

    rng = np.random.default_rng(42)
    features = []

    for rank, fid in enumerate(top_idx):
        col = A[:, fid]

        # HIGH group: top-100 indices by activation
        high_indices = np.argpartition(col, -HIGH_K)[-HIGH_K:]
        high_indices = high_indices[np.argsort(col[high_indices])[::-1]]

        # LOW group: 100 random tokens where activation < 0.01
        low_candidates = np.where(col < 0.01)[0]
        if len(low_candidates) >= LOW_K:
            low_indices = rng.choice(low_candidates, LOW_K, replace=False)
        else:
            low_indices = low_candidates

        # Contrastive label via frequency ratio (PMI proxy)
        def bigrams_from_indices(idxs, window=1):
            words = []
            for i in idxs:
                w = token_strs[i].strip().lower().strip("▁Ġ .,!?;:'\"")
                if w:
                    words.append(w)
                # include bigram with next token
                if i + 1 < N_TOKENS:
                    w2 = token_strs[i+1].strip().lower().strip("▁Ġ .,!?;:'\"")
                    if w and w2:
                        words.append(f"{w}_{w2}")
            return words

        high_words = bigrams_from_indices(high_indices)
        low_words  = bigrams_from_indices(low_indices)

        high_cnt = Counter(high_words)
        low_cnt  = Counter(low_words)
        total_low = max(sum(low_cnt.values()), 1)
        total_high = max(sum(high_cnt.values()), 1)

        best_label = "unknown"
        best_ratio = 0.0
        for word, cnt_h in high_cnt.items():
            if cnt_h < 3:
                continue
            cnt_l = low_cnt.get(word, 0)
            # frequency ratio with Laplace smoothing
            ratio = (cnt_h / total_high) / ((cnt_l + 0.5) / (total_low + 0.5))
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = word

        features.append({
            "feature_id":       int(fid),
            "variance_rank":    rank,
            "contrastive_label": best_label,
            "top_tokens":  [{"token": token_strs[i], "activation": float(col[i])}
                            for i in high_indices[:20]],
            "low_tokens":  [{"token": token_strs[i]} for i in low_indices[:10]],
            "mean_act":    float(mean_act[fid]),
            "variance":    float(var_act[fid]),
            "sparsity":    float(sparsity[fid]),
        })

        if rank % 100 == 0:
            log(f"  [{label}] feature {rank}/{TOP_VAR}: id={fid} label='{best_label}' "
                f"var={var_act[fid]:.4f} sparsity={sparsity[fid]:.3f}")

    return features, mean_act, var_act, sparsity


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Semantic rule extraction — LogReg + DecisionTree per probe target
# ─────────────────────────────────────────────────────────────────────────────
def semantic_rule_extraction(sae_acts, var_act, sparsity, ner_labels, dep_labels,
                             is_ne, pos_q, model_label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    # Filter to features with sparsity in [MIN_SPARSITY, MAX_SPARSITY]
    mask = (sparsity >= MIN_SPARSITY) & (sparsity <= MAX_SPARSITY)
    feat_ids = np.where(mask)[0]
    log(f"[{model_label}] {len(feat_ids)} features pass sparsity filter "
        f"[{MIN_SPARSITY}, {MAX_SPARSITY}]")

    if len(feat_ids) == 0:
        log("  No features pass filter — skipping rule extraction")
        return {}

    A_full = sae_acts[:, feat_ids].astype(np.float32)   # (N, n_feats)

    # Subsample for LR fitting: saga on 100k × 5k features takes hours; 30k is representative
    LR_SAMPLE = 30_000
    rng_lr = np.random.default_rng(7)
    lr_idx = rng_lr.choice(N_TOKENS, min(LR_SAMPLE, N_TOKENS), replace=False)
    lr_idx.sort()
    A_lr = A_full[lr_idx]

    targets = {
        "ner":      (ner_labels, NER_CLASSES),
        "dep":      (dep_labels, DEP_CLASSES),
        "is_ne":    (is_ne,      NE_CLASSES),
        "pos_q":    (pos_q,      POSQ_CLASSES),
    }

    all_results = {}
    for tname, (y_full, class_names) in targets.items():
        y    = y_full[lr_idx]
        log(f"  [{model_label}] Fitting {tname} probe (n={len(y)}, feats={len(feat_ids)}) ...")
        # lbfgs is faster than saga for dense data; no n_jobs needed
        lr = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs",
                                multi_class="ovr", random_state=42)
        try:
            lr.fit(A_lr, y)
        except Exception as e:
            log(f"    LR failed: {e}")
            continue

        y_pred_proba = lr.predict_proba(A_lr)
        classes      = lr.classes_
        n_classes    = len(classes)

        auc_per_class = {}
        if n_classes == 2:
            auc_per_class[class_names[classes[1]] if classes[1] < len(class_names) else str(classes[1])] = \
                float(roc_auc_score(y, y_pred_proba[:, 1]))
        else:
            y_bin = label_binarize(y, classes=classes)
            for i, cls in enumerate(classes):
                cn = class_names[cls] if cls < len(class_names) else str(cls)
                if y_bin[:, i].sum() > 0:
                    try:
                        auc_per_class[cn] = float(roc_auc_score(y_bin[:, i], y_pred_proba[:, i]))
                    except:
                        auc_per_class[cn] = 0.5

        log(f"    AUC: { {k: round(v,3) for k,v in auc_per_class.items()} }")

        # Top-5 predictive features per class
        top_features_per_class = {}
        coef = lr.coef_   # (n_classes, n_feats) or (1, n_feats)
        for ci, cls in enumerate(classes):
            if coef.shape[0] == 1:
                c = coef[0]
            else:
                c = coef[ci]
            top5_local = np.argsort(np.abs(c))[::-1][:5]
            cn = class_names[cls] if cls < len(class_names) else str(cls)
            top_features_per_class[cn] = [
                {"feature_id": int(feat_ids[j]), "coefficient": float(c[j])}
                for j in top5_local
            ]

        # DecisionTree
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        try:
            dt.fit(A_lr, y)
            feat_names = [f"f{fid}" for fid in feat_ids]
            tree_text  = export_text(dt, feature_names=feat_names, max_depth=5)
        except Exception as e:
            log(f"    DT failed: {e}")
            tree_text = f"ERROR: {e}"

        # Save rules file
        out_path = OUT / f"rules_v2_{model_label}_{tname}.txt"
        with open(out_path, "w") as fp:
            fp.write(f"# Semantic Rules — {model_label} — target: {tname}\n\n")
            fp.write("## AUC-ROC per class\n")
            for cn, auc in auc_per_class.items():
                fp.write(f"  {cn}: {auc:.4f}\n")
            fp.write("\n## Top-5 Predictive Features per Class (by |coefficient|)\n")
            for cn, feats in top_features_per_class.items():
                fp.write(f"\n  {cn}:\n")
                for f in feats:
                    fp.write(f"    feature {f['feature_id']:5d}  coef={f['coefficient']:+.4f}\n")
            fp.write("\n## Decision Tree Rules\n```\n")
            fp.write(tree_text[:8000])
            fp.write("\n```\n")
        log(f"    Saved {out_path.name}")

        all_results[tname] = {
            "auc_per_class":       auc_per_class,
            "top_features_per_class": top_features_per_class,
        }

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Cross-model alignment — CKA + Pearson correlation
# ─────────────────────────────────────────────────────────────────────────────
def cross_model_alignment(sae_m, sae_l, features_m, features_l):
    """
    CKA in token space: X=(n×16384) Mistral, Y=(n×16384) Llama.
    Uses subsampled n=N_CKA tokens.
    Also finds Pearson r matches for top-200 Mistral features vs all Llama features.
    """
    log(f"Cross-model alignment: CKA (n={N_CKA}) + Pearson correlation ...")

    rng = np.random.default_rng(0)
    idx = rng.choice(N_TOKENS, N_CKA, replace=False)
    idx.sort()

    X = sae_m[idx].astype(np.float32)   # (N_CKA, 16384) Mistral
    Y = sae_l[idx].astype(np.float32)   # (N_CKA, 16384) Llama

    # Center columns
    X -= X.mean(axis=0, keepdims=True)
    Y -= Y.mean(axis=0, keepdims=True)

    # Linear CKA: ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    # Compute via: tr(X X^T Y Y^T) = ||X^T Y||_F^2
    # But (16384×16384) is too large. Use: ||X^T Y||_F^2 = sum_ij (X^T Y)_ij^2
    # = tr(Y^T X X^T Y) = ||X^T Y||_F^2
    # We can compute this as: frobenius_sq = sum((X.T @ Y)**2)
    # X.T @ Y is (16384 × 16384) — at float32 that's 1GB. Too big.
    # Use chunked: compute row-by-row of X^T (columns of X), each row × Y → scalar
    # ||X^T Y||_F^2 = sum_i ||x_i^T Y||^2 where x_i is column i of X (row i of X^T)
    # = sum_i ||X[:,i] @ Y||^2  (X[:,i] is shape N_CKA, dot with Y rows → shape 16384)
    # But sum over 16384 such vectors = expensive. Use a different chunked approach.

    # Better: ||X^T Y||_F^2 = trace(X X^T Y Y^T)
    # Let Kx = X X^T (N_CKA × N_CKA), Ky = Y Y^T (N_CKA × N_CKA)
    # These are (8000 × 8000) = 64M elements × 4 bytes = 256MB. Feasible.
    # Linear CKA = trace(Kx Ky) / (||Kx||_F * ||Ky||_F)

    log("  Computing kernel matrices (N_CKA × N_CKA) ...")
    Kx = X @ X.T   # (N_CKA, N_CKA)
    Ky = Y @ Y.T

    def frob(M):
        return float(np.sqrt((M * M).sum()))

    def trace_prod(A, B):
        # trace(A B) = sum_ij A_ij * B_ji = sum_ij A_ij * B^T_ij = (A*B^T).sum()
        # But B^T for symmetric B is just B
        return float((A * B).sum())

    numerator = trace_prod(Kx, Ky)
    denom     = frob(Kx) * frob(Ky)
    cka_score = float(numerator / (denom + 1e-10))
    log(f"  Linear CKA score: {cka_score:.6f}")

    del Kx, Ky, X, Y
    gc.collect()

    # Pearson correlation matching: top-200 Mistral features vs all Llama features
    log(f"  Pearson correlation matching (top {TOP_CONTRASTIVE} Mistral features) ...")

    # Use all N_TOKENS for Pearson (more robust)
    Am = sae_m.astype(np.float32)   # (N_TOKENS, 16384)
    Al = sae_l.astype(np.float32)

    # Variance-rank top-200 Mistral feature IDs
    var_m = Am.var(axis=0)
    top200_m = np.argsort(var_m)[::-1][:TOP_CONTRASTIVE]

    # Build label lookup for features
    label_m = {f["feature_id"]: f["contrastive_label"] for f in features_m}
    label_l = {f["feature_id"]: f["contrastive_label"] for f in features_l}

    matched = []
    var_l    = Al.var(axis=0)
    top200_l = np.argsort(var_l)[::-1][:TOP_CONTRASTIVE * 2]  # search only top-400 Llama feats

    # Normalize columns for fast Pearson computation
    def norm_cols(M):
        mu  = M.mean(axis=0)
        std = M.std(axis=0)
        std[std < 1e-8] = 1.0
        return (M - mu) / std

    Am_norm = norm_cols(Am[:, top200_m])        # (N, 200)
    Al_norm = norm_cols(Al[:, top200_l])        # (N, 400)
    corr    = (Am_norm.T @ Al_norm) / N_TOKENS  # (200, 400)

    for i, mid in enumerate(top200_m):
        row = corr[i]
        best_j = int(np.argmax(row))
        best_r = float(row[best_j])
        lid    = int(top200_l[best_j])
        if best_r > 0.5:
            matched.append({
                "mistral_id":    int(mid),
                "llama_id":      lid,
                "pearson_r":     round(best_r, 4),
                "mistral_label": label_m.get(int(mid), "?"),
                "llama_label":   label_l.get(lid, "?"),
            })

    matched.sort(key=lambda x: -x["pearson_r"])
    log(f"  Found {len(matched)} cross-model feature pairs with r > 0.5")

    return {
        "cka_score":       cka_score,
        "n_cka_tokens":    N_CKA,
        "matched_features": matched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Summary
# ─────────────────────────────────────────────────────────────────────────────
def write_summary(all_results, alignment):
    lines = [
        "# SAE Neurosymbolic Rule Extraction v2 — Summary\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## What was fixed from v1\n\n",
        "1. **Probe targets**: POS replaced by NER label, dependency role, is-NE, sentence-position quartile (spacy)\n",
        "2. **Feature labeling**: Contrastive PMI-based labels (HIGH top-100 vs LOW random-100) instead of non-contrastive top-token frequency\n",
        "3. **Cross-model**: CKA in token space + Pearson correlation (not direct cosine; d_in differs)\n\n",
        "## Probe AUC Scores\n\n",
        "| Model | Target | Class | AUC |\n",
        "|---|---|---|---|\n",
    ]

    for mname in ["mistral", "llama"]:
        probe_results = all_results[mname]["probes"]
        for tname, tdata in probe_results.items():
            for cls, auc in tdata["auc_per_class"].items():
                lines.append(f"| {mname} | {tname} | {cls} | {auc:.3f} |\n")

    lines += [
        "\n## Top 10 Most Discriminative Features per Model\n\n",
        "| Model | Feature | Contrastive Label | Variance | Sparsity |\n",
        "|---|---|---|---|---|\n",
    ]
    for mname in ["mistral", "llama"]:
        feats = all_results[mname]["features"][:10]
        for f in feats:
            lines.append(
                f"| {mname} | {f['feature_id']} | {f['contrastive_label']} "
                f"| {f['variance']:.4f} | {f['sparsity']:.3f} |\n"
            )

    lines += [
        "\n## Top Feature-to-Concept Rules (plain English)\n\n",
    ]
    for mname in ["mistral", "llama"]:
        probe_results = all_results[mname]["probes"]
        for tname, tdata in probe_results.items():
            for cls, feats in tdata["top_features_per_class"].items():
                if not feats:
                    continue
                top = feats[0]
                auc = tdata["auc_per_class"].get(cls, 0.0)
                if auc > 0.65:
                    lines.append(
                        f"- **{mname}** feature {top['feature_id']} (coef={top['coefficient']:+.3f}) "
                        f"→ predicts `{tname}={cls}` with AUC {auc:.2f}\n"
                    )

    lines += [
        f"\n## Cross-Model Alignment\n\n",
        f"- **Linear CKA score** (n={alignment['n_cka_tokens']}): **{alignment['cka_score']:.6f}**\n",
        f"- Feature pairs with Pearson r > 0.5: **{len(alignment['matched_features'])}**\n\n",
        "### Top 5 Cross-Model Feature Matches\n\n",
        "| Mistral ID | Llama ID | Pearson r | Mistral label | Llama label |\n",
        "|---|---|---|---|---|\n",
    ]
    for m in alignment["matched_features"][:5]:
        lines.append(
            f"| {m['mistral_id']} | {m['llama_id']} | {m['pearson_r']:.3f} "
            f"| {m['mistral_label']} | {m['llama_label']} |\n"
        )

    lines += [
        "\n## What to Try Next\n\n",
        "- **More probe targets**: coreference distance, sentence depth (parse tree depth), argument structure\n",
        "- **Sparse probing**: L1-regularized logistic regression to identify the minimal feature set\n",
        "- **Layer sweep**: run the same analysis at layers 8 and 24 to see where semantics crystallize\n",
        "- **Cross-architecture clustering**: use the matched feature pairs to seed cross-model concept clusters\n",
        "- **Causal intervention**: apply activation patches on top-matched features to test causal role\n",
    ]

    with open(OUT / "summary_v2.md", "w") as f:
        f.writelines(lines)
    log("Saved summary_v2.md")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=== SAE Neurosymbolic Rule Extraction v2 ===")
    log(f"Output: {OUT}")

    # 0. Encode activations through SAE
    cached_m = OUT / "activations_mistral.npy"
    cached_l = OUT / "activations_llama.npy"
    if cached_m.exists() and cached_l.exists():
        log("Loading cached SAE activations ...")
        sae_m = np.load(cached_m)
        sae_l = np.load(cached_l)
    else:
        meta_m = json.load(open(META_M))
        meta_l = json.load(open(META_L))
        sae_m  = encode_sae(ACT_M, META_M, SAE_M, "mistral")
        np.save(cached_m, sae_m)
        log(f"Saved {cached_m.name}")
        sae_l  = encode_sae(ACT_L, META_L, SAE_L, "llama")
        np.save(cached_l, sae_l)
        log(f"Saved {cached_l.name}")

    log(f"SAE activations: mistral={sae_m.shape} llama={sae_l.shape}")

    # 1. Token strings + spacy labels
    cached_toks  = OUT / "tokens_mistral.json"
    cached_lbls  = OUT / "labels_mistral.npz"
    if cached_toks.exists() and cached_lbls.exists():
        log("Loading cached tokens and labels ...")
        token_strs  = json.load(open(cached_toks))
        lbl         = np.load(cached_lbls)
        ner_labels  = lbl["ner"]
        dep_labels  = lbl["dep"]
        is_ne       = lbl["is_ne"]
        pos_q       = lbl["pos_q"]
    else:
        token_strs, ner_labels, dep_labels, is_ne, pos_q = \
            get_tokens_and_labels("mlx-community/Mistral-7B-v0.3-4bit")
        json.dump(token_strs, open(cached_toks, "w"))
        np.savez(OUT / "labels_mistral.npz",
                 ner=ner_labels, dep=dep_labels, is_ne=is_ne, pos_q=pos_q)
        # Llama uses same corpus so same labels
        np.savez(OUT / "labels_llama.npz",
                 ner=ner_labels, dep=dep_labels, is_ne=is_ne, pos_q=pos_q)
        log("Saved tokens and labels")

    log(f"NER dist:  { {NER_CLASSES[i]: int((ner_labels==i).sum()) for i in range(6)} }")
    log(f"DEP dist:  { {DEP_CLASSES[i]: int((dep_labels==i).sum()) for i in range(6)} }")
    log(f"IS_NE:     NE={int(is_ne.sum())} non-NE={int((is_ne==0).sum())}")

    all_results = {}

    for mname, sae_acts in [("mistral", sae_m), ("llama", sae_l)]:
        log(f"\n{'='*60}")
        log(f"Processing: {mname}")
        log(f"{'='*60}")

        # 2. Contrastive feature labeling
        features, mean_act, var_act, sparsity = \
            contrastive_feature_labeling(sae_acts, token_strs, mname)
        json.dump(features, open(OUT / f"features_v2_{mname}.json", "w"), indent=2)
        log(f"Saved features_v2_{mname}.json ({len(features)} features)")

        # 3. Semantic rule extraction
        probes = semantic_rule_extraction(
            sae_acts, var_act, sparsity,
            ner_labels, dep_labels, is_ne, pos_q,
            mname
        )

        all_results[mname] = {"features": features, "probes": probes}

    # Collect probe metrics
    probe_metrics = {
        m: {tname: tdata["auc_per_class"] for tname, tdata in all_results[m]["probes"].items()}
        for m in all_results
    }
    json.dump(probe_metrics, open(OUT / "probe_metrics.json", "w"), indent=2)
    log("Saved probe_metrics.json")

    # 4. Cross-model alignment
    alignment = cross_model_alignment(
        sae_m, sae_l,
        all_results["mistral"]["features"],
        all_results["llama"]["features"],
    )
    json.dump(alignment, open(OUT / "alignment.json", "w"), indent=2)
    log("Saved alignment.json")

    # 5. Summary
    write_summary(all_results, alignment)

    elapsed = time.time() - t0
    log(f"\n=== DONE in {elapsed/60:.1f} min ===")
    log(f"Outputs in {OUT}:")
    for p in sorted(OUT.iterdir()):
        log(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    sys.exit(0)


if __name__ == "__main__":
    main()
