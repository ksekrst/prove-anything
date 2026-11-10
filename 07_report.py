"""Step 7. Collects every result into results/REPORT.md: one table per test with the four
targets side by side, per model. Reads whatever exists; missing steps are noted.

Usage: python 07_report.py
"""
import json
from pathlib import Path

import pandas as pd

import config
from common import load_targets


def md_table(df, floatfmt="{:.2f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(floatfmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    R = config.RESULTS_DIR
    targets = load_targets()
    tnames = list(targets)
    out = [f"# Results\n", f"Targets: {', '.join(targets[t]['label'] for t in tnames)}\n"]

    # ---- 0. the main comparison -------------------------------------------------------
    sp = R / "summary_all_models.csv"
    lexp = R / "lexical_baseline.csv"
    if sp.exists():
        S0 = pd.read_csv(sp)
        if "heldout_vs_specific" in S0:
            ft = S0[S0.readout == "final_token"]
            piv = ft.pivot_table(index="target", columns="model", values="heldout_vs_specific")
            rnd = [c for c in piv.columns if c.startswith("Random")]
            trn = [c for c in piv.columns if not c.startswith("Random")]
            piv = piv[rnd + trn]
            if lexp.exists():
                lx = pd.read_csv(lexp).set_index("target")
                if "auc_vs_specific" in lx:
                    piv.insert(0, "words only", lx["auc_vs_specific"])
            piv = piv.reindex([t for t in tnames if t in piv.index])
            out.append("## 0. Main comparison: held-out AUC against each target's own controls (final token)\n")
            out.append("Each target against the controls written for it: for pain, the paper's five control "
                       "categories; for the new targets, the two target-specific ones (for the indirect targets "
                       "MATCH and ABSENT). Vectors fitted without the scored fold. 'words only' is the "
                       "bag-of-words classifier of step 08 on the same folds; the Random_* columns are the "
                       "untrained networks. What pretraining adds is the trained column minus the Random column.\n")
            out.append(md_table(piv.reset_index()))
            out.append("")

    # ---- 1. separation --------------------------------------------------------------
    if sp.exists():
        S = pd.read_csv(sp)
        out.append("## 1. Does a direction separate the target from its controls? (step 02)\n")
        out.append("Held-out AUC from the 5-fold layer choice; AUC vs all controls at the chosen layer; the "
                   "lowest AUC against any single control category; z of the target in first and third person; "
                   "z of the standalone 'cause present, state explicitly absent' set (reference: first-person core).\n")
        for readout in ("final_token", "mean"):
            sub = S[S.readout == readout].copy()
            if sub.empty:
                continue
            if "z_state_absent" not in sub:      # results from before this column existed
                zcols = [c for c in sub.columns if c.startswith("z_")]
                sub["z_state_absent"] = sub[zcols].bfill(axis=1).iloc[:, 0] if zcols else float("nan")
            view = sub[["model", "target", "layer", "heldout_auc", "auc_vs_all_controls", "auc_min_over_controls",
                        "target_z_1P", "target_z_3P", "z_state_absent"]]
            out.append(f"### {readout}\n")
            out.append(md_table(view))
            out.append("")

        ind = S[S.target.str.endswith("_ind") & (S.readout == "final_token")].copy()
        if not ind.empty:
            out.append("## 1b. Indirect targets: matched controls and generalisation (step 02, final token)\n")
            out.append("heldout_vs_MATCH: target vs the same scenes with nobody affected, held out. heldout_vs_ABSENT: "
                       "target vs 'cause present, state explicitly absent' (these contain the state words, the targets "
                       "do not), held out. auc_vs_MATCH is the in-sample version, kept to show how much it inflates. "
                       "auc_explicit_vs_controls: the explicit v1 sentences, never seen when "
                       "fitting, scored by the indirect vector against the core controls.\n")
            cols = [c for c in ["model", "target", "layer", "heldout_auc", "heldout_vs_MATCH", "heldout_vs_ABSENT",
                                "auc_vs_MATCH", "z_explicit", "auc_explicit_vs_controls"] if c in ind]
            out.append(md_table(ind[cols]))
            out.append("")

    lex = R / "lexical_baseline.csv"
    if lex.exists():
        L = pd.read_csv(lex)
        out.append("## 1c. Lexical floor: bag-of-words logistic regression, same folds (step 08)\n")
        out.append("What word identity alone achieves on each core set. Compare with heldout_auc above.\n")
        cols = [c for c in ["target", "heldout_auc", "auc_vs_specific", "auc_min_over_controls", "auc_vs_MATCH", "auc_vs_ABSENT",
                            "auc_explicit_vs_controls"] if c in L]
        out.append(md_table(L[cols]))
        out.append("")

    # ---- per model sections ----------------------------------------------------------
    model_dirs = sorted(p for p in R.iterdir() if p.is_dir() and (p / "activations.pt").exists())
    for mdir in model_dirs:
        name = mdir.name
        out.append(f"\n## {name}\n")

        cos = mdir / "validation" / "cosine_final_token.csv"
        if cos.exists():
            C = pd.read_csv(cos, index_col=0)
            keep = [c for c in C.index if "pooled controls" in c]
            out.append("Cosine similarity among the target directions (final token, reference layer):\n")
            sub = C.loc[keep, keep].copy()
            sub.index = [i.split(" (")[0] for i in sub.index]
            sub.columns = [i.split(" (")[0] for i in sub.columns]
            out.append(md_table(sub.reset_index().rename(columns={"index": ""})))
            out.append("")
        xa = mdir / "validation" / "cross_auc_final_token.csv"
        if xa.exists():
            X = pd.read_csv(xa, index_col=0)
            out.append("Cross-projection: AUC of each column target's sentences vs its controls, scored by the row target's vector:\n")
            out.append(md_table(X.reset_index().rename(columns={"index": ""})))
            out.append("")

        sbs = mdir / "unembedding" / "side_by_side.md"
        if sbs.exists():
            out.append("### Unembedding readout (step 03)\n")
            out.append("\n".join(sbs.read_text(encoding="utf-8").splitlines()[1:]))
            out.append("")

        sdir = mdir / "steering"
        if sdir.exists():
            out.append("### Steering ladder (step 04)\n")
            rows = []
            for t in tnames:
                rp, dp = sdir / f"{t}_rates.csv", sdir / f"{t}_dose.json"
                if not rp.exists():
                    continue
                rates = pd.read_csv(rp)
                dose = json.load(open(dp, encoding="utf-8"))["dose"] if dp.exists() else None
                for cond in ("target", "random"):
                    r = rates[(rates.condition == cond) & (rates.coeff == dose)]
                    if len(r):
                        r = r.iloc[0]
                        rows.append({"target": t, "condition": cond, "dose": dose, "strict_rate": r.strict,
                                     "broad_rate": r.broad, "coherent": r.coherent})
                r0 = rates[(rates.condition == "target") & (rates.coeff == 0)]
                if len(r0):
                    r0 = r0.iloc[0]
                    rows.append({"target": t, "condition": "unsteered", "dose": 0.0, "strict_rate": r0.strict,
                                 "broad_rate": r0.broad, "coherent": r0.coherent})
            if rows:
                out.append("Share of the 50 neutral-prompt continuations using the target's strict keywords "
                           "(the paper's parser is 'pain/painful/hurt(s/ing)'), at the auto-picked dose:\n")
                out.append(md_table(pd.DataFrame(rows)))
                out.append("")
            for t in tnames:
                cp, dp = sdir / f"{t}.csv", sdir / f"{t}_dose.json"
                if not cp.exists():
                    continue
                df = pd.read_csv(cp)
                dose = json.load(open(dp, encoding="utf-8"))["dose"] if dp.exists() else df.coeff.max()
                ex = df[(df.condition == "target") & (df.coeff == dose)].head(3)
                out.append(f"**{targets[t]['label']}**, coefficient {dose}:\n")
                for _, r in ex.iterrows():
                    out.append(f"- *{r['prompt']}* {str(r['generation'])[:260].replace(chr(10), ' ')}")
                out.append("")

        ip = mdir / "injection" / "summary.csv"
        if ip.exists():
            out.append("### Injection detection (step 05)\n")
            I = pd.read_csv(ip)
            out.append(md_table(I))
            out.append("")

        bp = mdir / "buttons" / "summary.csv"
        if bp.exists():
            out.append("### Button task (step 06): share choosing the first-listed option\n")
            B = pd.read_csv(bp)
            piv = B.pivot_table(index=["target", "pair"], columns="arm", values="chose_first_option")
            piv.columns = [f"arm: {c}" for c in piv.columns]
            out.append(md_table(piv.reset_index()))
            out.append("")

    (R / "REPORT.md").write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {R / 'REPORT.md'}")


if __name__ == "__main__":
    main()
