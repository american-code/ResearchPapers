#!/usr/bin/env python3
"""
Neurosymbolic rule extraction from SAE features (Mistral-7B + Llama-3.2-3B).
Produces: features_<model>.json, rules_<model>.txt, comparison.json, summary.md
"""

import os, sys, json, time, heapq
import numpy as np
from pathlib import Path

REPO      = Path("/Users/lab-02/ResearchPapers")
SAE_M     = REPO / "data/sae-runs/mistral-7b-layer16/checkpoint_final.npz"
SAE_L     = REPO / "data/sae-runs/llama-3b-layer16/checkpoint_final.npz"
ACT_M     = REPO / "data/activations/mistral-7b-layer16/activations.npy"
ACT_L     = REPO / "data/activations/llama-3b-layer16/activations.npy"
OUT       = REPO / "data/sae-rules"
OUT.mkdir(parents=True, exist_ok=True)

MAX_TOKENS    = 100_000   # use first 100k tokens from corpus
BATCH_SIZE    = 2_000     # tokens per encoding batch
TOP_FEATURES  = 200       # features to fully analyze
TREE_FEATURES = 64        # features for decision tree
TOP_CONTEXTS  = 20        # top contexts per feature
TREE_SAMPLE   = 50_000    # tokens for decision tree fitting


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Step 1: load SAE weights ──────────────────────────────────────────────────
def load_sae(path):
    log(f"Loading SAE: {path.name}")
    ckpt = np.load(path)
    d = {k: ckpt[k] for k in ckpt.files}
    log(f"  W_enc {d['W_enc'].shape}  W_dec {d['W_dec'].shape}")
    return d


# ── Step 2: tokenize corpus for context labels ────────────────────────────────
def load_tokens(tokenizer_name, max_tokens):
    log(f"Loading wikitext-2 and tokenizing with {tokenizer_name} ...")
    try:
        from transformers import AutoTokenizer
        from datasets import load_dataset
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        ds  = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                           split="train", trust_remote_code=False)
        texts = [t for t in ds["text"] if t.strip()]
        full  = " ".join(texts)
        ids   = tok.encode(full, add_special_tokens=False)[:max_tokens]
        strs  = [tok.decode([i]) for i in ids]
        log(f"  Got {len(strs)} tokens")
        return strs
    except Exception as e:
        log(f"  Tokenizer failed ({e}), using word-split fallback")
        from datasets import load_dataset
        ds    = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        words = " ".join([t for t in ds["text"] if t.strip()]).split()[:max_tokens]
        log(f"  Got {len(words)} words via split")
        return words


# ── POS heuristics ────────────────────────────────────────────────────────────
_FUNC = {
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "must","can","of","in","to","for","on","at","by","with","from","that",
    "this","it","he","she","they","we","you","i","and","or","but","not",
    "no","so","if","then","than","as","which","who","what","when","where",
    "its","their","our","your","my","his","her","also","been","into","out",
    "up","about","after","before","over","under","between","through",
}
_VERB_SUF = ("ing","ed","ify","ize","ise","ate","ify")
_ADJ_SUF  = ("al","ful","less","ous","ive","ish","able","ible","ic","ary")
_NEG      = {"not","no","never","neither","nor","without","nothing","nobody"}

def heuristic_pos(tok: str) -> str:
    t = tok.lower().strip(" \t\n▁Ġ_")
    if not t:
        return "OTHER"
    if not t[0].isalpha():
        return "OTHER"
    if t in _NEG:
        return "FUNC"
    if t in _FUNC:
        return "FUNC"
    if tok and tok[0].isupper() and not tok.startswith(("Ġ","▁")):
        return "NOUN"   # likely proper noun
    if any(t.endswith(s) for s in _VERB_SUF):
        return "VERB"
    if any(t.endswith(s) for s in _ADJ_SUF):
        return "ADJ"
    return "NOUN"

def auto_label(top_tokens):
    if not top_tokens:
        return "unknown"
    strs = [t["token"].lower().strip(" ▁Ġ") for t in top_tokens[:8]]
    nums    = sum(1 for s in strs if s.replace(",","").replace(".","").isdigit())
    punct   = sum(1 for s in strs if s and all(not c.isalnum() for c in s))
    func    = sum(1 for s in strs if s in _FUNC)
    neg     = sum(1 for s in strs if s in _NEG)
    prop    = sum(1 for t in top_tokens[:8]
                  if t["token"] and t["token"][0].isupper()
                  and not t["token"].startswith(("Ġ","▁")))
    code_kw = {"def","class","return","import","if","else","for","while",
               "function","var","let","const","int","void","public","private",
               "null","true","false","print","self","init","type"}
    code    = sum(1 for s in strs if s in code_kw)

    if code >= 2:    return "code-keyword"
    if neg >= 2:     return "negation"
    if nums >= 3:    return "number"
    if punct >= 4:   return "punctuation"
    if func >= 4:    return "function-word"
    if prop >= 4:    return "proper-noun"
    best = strs[0] if strs else "unk"
    return f"token-{best[:15]}"


# ── Step 3 & 4: streaming SAE encode + top-k context tracking ─────────────────
def stream_encode(sae, act_path, token_strs, max_tokens, batch_size):
    """
    Single pass over activations:
      - accumulates mean_freq and max_activation for all 16384 features
      - maintains top-TOP_CONTEXTS heap per feature (for top TOP_FEATURES later)
      - builds per-token label array for first TREE_SAMPLE tokens (top-64 feats)
    Returns:
      mean_freq  (dict_size,)
      topk_heaps list of heaps indexed by feature (only built for top features — done in 2 passes)
      tree_X     (TREE_SAMPLE, TREE_FEATURES) float32
      tree_y     (TREE_SAMPLE,) str
    """
    W_enc = sae["W_enc"].astype(np.float32)   # (d_in, dict_size)
    b_enc = sae["b_enc"].astype(np.float32)   # (dict_size,)
    dict_size = b_enc.shape[0]

    log(f"  PASS 1: computing mean freq for {dict_size} features over {max_tokens} tokens ...")
    import json as _json
    _meta = _json.load(open(act_path.parent / "metadata.json"))
    _shape = tuple(_meta["activations_shape"])
    _dtype = _meta["activations_dtype"]
    acts  = np.memmap(act_path, dtype=_dtype, mode="r", shape=_shape)
    n_tok = min(max_tokens, acts.shape[0])

    freq_sum  = np.zeros(dict_size, dtype=np.float64)
    max_act   = np.zeros(dict_size, dtype=np.float32)
    n_seen    = 0

    for start in range(0, n_tok, batch_size):
        batch = acts[start : start + batch_size].astype(np.float32)
        Z     = np.maximum(0.0, batch @ W_enc + b_enc)   # (batch, dict_size)
        freq_sum += (Z > 0).sum(axis=0)
        np.maximum(max_act, Z.max(axis=0), out=max_act)
        n_seen += len(batch)
        if (start // batch_size) % 10 == 0:
            log(f"    pass1 {n_seen}/{n_tok}")

    mean_freq = freq_sum / n_seen
    top_feature_ids = np.argsort(mean_freq)[::-1][:TOP_FEATURES].tolist()
    top64_ids       = top_feature_ids[:TREE_FEATURES]

    log(f"  PASS 2: collecting top contexts & tree features for top {TOP_FEATURES} features ...")
    # min-heaps: (activation_value, token_index) per feature
    heaps = {fid: [] for fid in top_feature_ids}

    # tree data
    n_tree   = min(TREE_SAMPLE, n_tok, len(token_strs))
    tree_X   = np.zeros((n_tree, TREE_FEATURES), dtype=np.float32)
    tree_y   = np.array([heuristic_pos(token_strs[i]) for i in range(n_tree)])

    top_ids_arr   = np.array(top_feature_ids, dtype=np.int32)
    top64_arr     = np.array(top64_ids,       dtype=np.int32)

    for start in range(0, n_tok, batch_size):
        end   = min(start + batch_size, n_tok)
        batch = acts[start:end].astype(np.float32)
        Z     = np.maximum(0.0, batch @ W_enc + b_enc)

        Z_top  = Z[:, top_ids_arr]          # (batch, 200)
        Z_top64 = Z[:, top64_arr]           # (batch, 64)

        # update heaps
        for local_i in range(len(batch)):
            global_i = start + local_i
            for j, fid in enumerate(top_feature_ids):
                v = float(Z_top[local_i, j])
                if v == 0.0:
                    continue
                h = heaps[fid]
                if len(h) < TOP_CONTEXTS:
                    heapq.heappush(h, (v, global_i))
                elif v > h[0][0]:
                    heapq.heapreplace(h, (v, global_i))

        # fill tree_X
        overlap_start = start
        overlap_end   = min(end, n_tree)
        if overlap_start < n_tree:
            s = overlap_start - start
            e = overlap_end   - start
            tree_X[overlap_start:overlap_end] = Z_top64[s:e]

        if (start // batch_size) % 10 == 0:
            log(f"    pass2 {end}/{n_tok}")

    return mean_freq, max_act, heaps, top_feature_ids, top64_ids, tree_X, tree_y


# ── Step 4: build feature records ─────────────────────────────────────────────
def build_features(mean_freq, max_act_all, heaps, top_ids, token_strs):
    features = []
    for fid in top_ids:
        heap = heaps[fid]
        top_toks = []
        for (act_val, tok_idx) in sorted(heap, reverse=True):
            if tok_idx < len(token_strs):
                top_toks.append({"token": token_strs[tok_idx], "activation": float(act_val)})
        label = auto_label(top_toks)
        features.append({
            "feature_id":     int(fid),
            "top_tokens":     top_toks,
            "label":          label,
            "mean_freq":      float(mean_freq[fid]),
            "max_activation": float(max_act_all[fid]),
        })
    return features


# ── Step 5: decision tree rules ───────────────────────────────────────────────
def extract_rules(model_name, tree_X, tree_y, top64_ids):
    log(f"  Fitting DecisionTree for {model_name} on {len(tree_y)} tokens ...")
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.metrics import accuracy_score, classification_report

    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(tree_X, tree_y)
    y_pred = clf.predict(tree_X)
    acc    = accuracy_score(tree_y, y_pred)
    report = classification_report(tree_y, y_pred, output_dict=True, zero_division=0)
    feat_names = [f"feat_{i}" for i in top64_ids]
    tree_text  = export_text(clf, feature_names=feat_names)
    log(f"    accuracy: {acc:.3f}  classes: {clf.classes_.tolist()}")

    # per-feature precision/recall for each POS class
    per_feat_rules = []
    for j, fid in enumerate(top64_ids):
        col = tree_X[:, j]
        binary = (col > 0).astype(np.int8)
        for cls in clf.classes_:
            pos = (tree_y == cls).astype(np.int8)
            tp = int(((binary == 1) & (pos == 1)).sum())
            fp = int(((binary == 1) & (pos == 0)).sum())
            fn = int(((binary == 0) & (pos == 1)).sum())
            if tp < 5:
                continue
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            if prec > 0.25:
                per_feat_rules.append({
                    "feature_id": int(fid),
                    "pos_class":  cls,
                    "precision":  round(float(prec), 4),
                    "recall":     round(float(rec), 4),
                    "support":    tp,
                })
    per_feat_rules.sort(key=lambda x: -x["precision"])

    return {
        "accuracy":          round(float(acc), 4),
        "classes":           clf.classes_.tolist(),
        "per_class_report":  report,
        "tree_text":         tree_text,
        "per_feature_rules": per_feat_rules,
    }

def write_rules_txt(path, model_name, tree_result):
    with open(path, "w") as f:
        f.write(f"# Symbolic Rules — {model_name}\n\n")
        f.write(f"## Decision Tree (top-{TREE_FEATURES} SAE features → POS class)\n\n")
        f.write(f"Overall accuracy: {tree_result['accuracy']:.3f}\n")
        f.write(f"Classes: {', '.join(tree_result['classes'])}\n\n")
        f.write("```\n" + tree_result["tree_text"] + "```\n\n")
        f.write("## Top Per-Feature Rules (by precision)\n\n")
        f.write(f"{'feat_id':>10}  {'POS':6}  {'prec':>6}  {'rec':>6}  {'support':>7}\n")
        f.write("-" * 50 + "\n")
        for r in tree_result["per_feature_rules"][:60]:
            f.write(f"{r['feature_id']:>10}  {r['pos_class']:<6}  "
                    f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['support']:>7}\n")


# ── Step 6: cross-model comparison ────────────────────────────────────────────
def cross_model_compare(feat_m, feat_l):
    """
    Mistral d_in=4096, Llama d_in=3072 → different embedding spaces.
    Direct cosine similarity is undefined across architectures.
    We use label + frequency-ratio matching as a proxy.
    """
    log("Cross-model comparison via label + frequency matching ...")

    def index_by_label(feats):
        d = {}
        for f in feats:
            d.setdefault(f["label"], []).append(f)
        return d

    idx_m = index_by_label(feat_m)
    idx_l = index_by_label(feat_l)

    shared, mistral_only, llama_only = [], [], []
    all_labels = set(idx_m) | set(idx_l)

    for lbl in all_labels:
        ms = idx_m.get(lbl, [])
        ls = idx_l.get(lbl, [])
        if ms and ls:
            bm = max(ms, key=lambda x: x["mean_freq"])
            bl = max(ls, key=lambda x: x["mean_freq"])
            lo = min(bm["mean_freq"], bl["mean_freq"])
            hi = max(bm["mean_freq"], bl["mean_freq"], 1e-10)
            sim = round(lo / hi, 4)
            shared.append({
                "mistral_id": bm["feature_id"],
                "llama_id":   bl["feature_id"],
                "similarity": sim,
                "label":      lbl,
                "note": "label+freq-ratio (d_in mismatch precludes direct cosine)",
            })
        elif ms:
            for f in ms:
                mistral_only.append({"feature_id": f["feature_id"], "label": lbl})
        else:
            for f in ls:
                llama_only.append({"feature_id": f["feature_id"], "label": lbl})

    shared.sort(key=lambda x: -x["similarity"])
    log(f"  shared={len(shared)}  mistral_only={len(mistral_only)}  llama_only={len(llama_only)}")
    return {"shared": shared, "mistral_only": mistral_only, "llama_only": llama_only}


# ── Step 7: summary ───────────────────────────────────────────────────────────
def write_summary(models, comparison):
    lines = [
        "# SAE Neurosymbolic Rule Extraction — Summary\n\n",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
        "## Dataset & Models\n\n",
        "| Model | Layer | d_in | dict_size | Corpus |\n",
        "|---|---|---|---|---|\n",
        "| Mistral-7B-v0.3 (4-bit) | 16 | 4096 | 16384 | wikitext-103-raw-v1 |\n",
        "| Llama-3.2-3B (bf16)     | 14 | 3072 | 16384 | wikitext-103-raw-v1 |\n\n",
        f"Token budget: {MAX_TOKENS:,} per model (of 500,000 collected)\n\n",
    ]

    for mname, data in models.items():
        feats   = data["features"]
        tree    = data["tree"]
        mf      = data["mean_freq"]
        active  = int((mf > 0.01).sum())

        feat_prec = {}
        for r in tree["per_feature_rules"]:
            fid = r["feature_id"]
            if fid not in feat_prec or r["precision"] > feat_prec[fid][1]:
                feat_prec[fid] = (r["pos_class"], r["precision"])

        lines += [
            f"## {mname}\n\n",
            f"- Active features (mean_freq > 0.01): **{active}** / {len(mf)}\n",
            f"- Decision tree overall accuracy: **{tree['accuracy']:.3f}**\n",
            f"- POS classes modelled: {', '.join(tree['classes'])}\n\n",
            "### Top 20 Most Interpretable Features\n\n",
            "| feature_id | label | mean_freq | max_act | best_POS | precision |\n",
            "|---|---|---|---|---|---|\n",
        ]
        for feat in feats[:20]:
            fid    = feat["feature_id"]
            bp, bp_prec = feat_prec.get(fid, ("—", 0.0))
            lines.append(
                f"| {fid} | {feat['label']} | {feat['mean_freq']:.4f} | "
                f"{feat['max_activation']:.2f} | {bp} | {bp_prec:.3f} |\n"
            )
        lines.append("\n")

    lines += [
        "## Top 5 Shared Features (Cross-Model)\n\n",
        "| label | Mistral feat | Llama feat | freq-ratio sim |\n",
        "|---|---|---|---|\n",
    ]
    for s in comparison["shared"][:5]:
        lines.append(
            f"| {s['label']} | {s['mistral_id']} | {s['llama_id']} | {s['similarity']:.3f} |\n"
        )

    lines += ["\n## Example Rules in Plain English\n\n"]
    all_rules = []
    for mname, data in models.items():
        for r in data["tree"]["per_feature_rules"][:5]:
            all_rules.append((mname, r))
    all_rules.sort(key=lambda x: -x[1]["precision"])
    for mname, r in all_rules[:3]:
        lines.append(
            f"- **{mname}**: when SAE feature {r['feature_id']} fires (activation > 0), "
            f"the token is likely **{r['pos_class']}** "
            f"(precision {r['precision']:.0%}, recall {r['recall']:.0%}, "
            f"support {r['support']:,} tokens)\n"
        )

    lines += [
        "\n## Notes\n\n",
        "- Cross-model cosine similarity is undefined (Mistral d_in=4096 ≠ Llama d_in=3072). "
        "Shared features identified via label-matching + activation frequency ratio.\n",
        "- Mistral weights are 4-bit quantized (mlx-community); "
        "Llama weights are bf16. Quantization is a potential confound.\n",
        "- POS labels are heuristic (suffix/function-word lookup), not gold-standard tagging.\n",
    ]

    with open(OUT / "summary.md", "w") as f:
        f.writelines(lines)
    log("Wrote summary.md")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log("=== SAE Neurosymbolic Rule Extraction ===")

    sae_m = load_sae(SAE_M)
    sae_l = load_sae(SAE_L)

    # Tokenize corpus (use Mistral tokenizer as primary; both use same corpus)
    tok_strs = load_tokens("mlx-community/Mistral-7B-v0.3-4bit", MAX_TOKENS)
    # Pad/truncate to MAX_TOKENS
    if len(tok_strs) < MAX_TOKENS:
        tok_strs += [f"<pad_{i}>" for i in range(MAX_TOKENS - len(tok_strs))]
    tok_strs = tok_strs[:MAX_TOKENS]

    results = {}

    for model_name, sae, act_path in [
        ("Mistral-7B-layer16", sae_m, ACT_M),
        ("Llama-3.2-3B-layer14", sae_l, ACT_L),
    ]:
        log(f"\n{'='*60}")
        log(f"Processing {model_name}")
        log(f"{'='*60}")

        mean_freq, max_act_all, heaps, top_ids, top64_ids, tree_X, tree_y = \
            stream_encode(sae, act_path, tok_strs, MAX_TOKENS, BATCH_SIZE)

        features = build_features(mean_freq, max_act_all, heaps, top_ids, tok_strs)

        slug = model_name.split("-")[0].lower()
        feat_path = OUT / f"features_{slug}.json"
        with open(feat_path, "w") as f:
            json.dump(features, f, indent=2)
        log(f"Saved {feat_path.name}")

        tree = extract_rules(model_name, tree_X, tree_y, top64_ids)
        rule_path = OUT / f"rules_{slug}.txt"
        write_rules_txt(rule_path, model_name, tree)
        log(f"Saved {rule_path.name}")

        results[model_name] = {
            "features":  features,
            "tree":      tree,
            "mean_freq": mean_freq,
        }

    log("\nStep 6: cross-model comparison")
    comparison = cross_model_compare(
        results["Mistral-7B-layer16"]["features"],
        results["Llama-3.2-3B-layer14"]["features"],
    )
    comp_path = OUT / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    log(f"Saved {comp_path.name}")

    log("\nStep 7: writing summary")
    write_summary(results, comparison)

    elapsed = time.time() - t0
    log(f"\n=== DONE in {elapsed:.0f}s ===")
    log(f"Outputs: {OUT}/")
    for p in sorted(OUT.iterdir()):
        log(f"  {p.name}  ({p.stat().st_size // 1024} KB)")
    sys.exit(0)


if __name__ == "__main__":
    main()
