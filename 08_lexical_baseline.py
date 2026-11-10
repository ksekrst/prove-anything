"""Step 8 (CPU, no model, seconds). How far does word identity alone get?

For each target, a bag-of-words logistic regression (unigrams and bigrams, lower-cased,
the "I feel:" suffix removed) is trained on the same first-person core set with the same
5 folds over sentence-set numbers as step 02, and scored by held-out AUC, target vs all
controls and target vs each control category. This is the lexical floor: if words alone
separate a target, high AUC from the residual stream says little about anything else.

Writes results/lexical_baseline.csv and prints a table. Compare its heldout_auc column with
heldout_auc in summary_all_models.csv for the trained and the random-init models.

Usage: python 08_lexical_baseline.py
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import config
from common import load_datasets, load_targets


def strip(p):
    return p.replace(" I feel:", "").strip()


def heldout(texts, y, sets, extra_eval=None):
    texts, y, sets = np.asarray(texts), np.asarray(y), np.asarray(sets)
    uniq = sorted(set(sets))
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    scores = np.zeros(len(y))
    for tr_i, te_i in kf.split(uniq):
        tr = np.isin(sets, [uniq[i] for i in tr_i])
        te = np.isin(sets, [uniq[i] for i in te_i])
        vec = CountVectorizer(ngram_range=(1, 2), lowercase=True, binary=True)
        Xtr = vec.fit_transform(texts[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, y[tr])
        scores[te] = clf.decision_function(vec.transform(texts[te]))
    return scores


def main():
    data = load_datasets()
    targets = load_targets()
    rows = []
    for tname, T in targets.items():
        core = data[T["core_1P"]]["sentences"]
        texts = [strip(s["prompt"]) for s in core]
        cats = np.array([s["category"] for s in core])
        sets = [s["set"] for s in core]
        tcats, ccats = list(T["target_cats"]), list(T["control_cats"])
        keep = np.isin(cats, tcats + ccats)
        y = np.isin(cats, tcats).astype(int)
        sc = heldout(np.array(texts)[keep], y[keep], np.array(sets)[keep])
        cats_k, y_k = cats[keep], y[keep]
        row = {"target": tname, "heldout_auc": roc_auc_score(y_k, sc)}
        for c in ccats:
            m = np.isin(cats_k, tcats + [c])
            row[f"auc_vs_{c}"] = roc_auc_score(y_k[m], sc[m])
        row["auc_min_over_controls"] = min(row[f"auc_vs_{c}"] for c in ccats)
        spec = list(ccats) if tname == "pain" else [c for c in ccats if c not in {"A1", "B", "C1", "C2", "D", "E"}]
        m = np.isin(cats_k, tcats + spec)
        row["auc_vs_specific"] = roc_auc_score(y_k[m], sc[m])

        # explicit sentences scored by a classifier trained on the whole (indirect) core
        exp = [s for s in T["standalone"] if s.endswith("_EXPLICIT_1P")]
        if exp:
            vec = CountVectorizer(ngram_range=(1, 2), lowercase=True, binary=True)
            clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(np.array(texts)[keep]), y_k)
            etexts = [strip(s["prompt"]) for s in data[exp[0]]["sentences"]]
            ctexts = [t for t, c in zip(texts, cats) if c in ccats]
            s_e = clf.decision_function(vec.transform(etexts))
            s_c = clf.decision_function(vec.transform(ctexts))
            row["auc_explicit_vs_controls"] = roc_auc_score([1] * len(s_e) + [0] * len(s_c),
                                                            np.concatenate([s_e, s_c]))
        rows.append(row)

    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "lexical_baseline.csv", index=False)
    cols = ["target", "heldout_auc", "auc_vs_specific", "auc_min_over_controls"] + \
           (["auc_explicit_vs_controls"] if "auc_explicit_vs_controls" in df else [])
    print("Bag-of-words logistic regression, held-out AUC (same folds as step 02):")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
