"""Step 2. The paper's extraction and validation, run identically for every target.

Per model and target (final-token and mean-pooled readouts):
  - held-out AUC at every layer (5-fold, split by sentence set) -> layer choice
  - denoised difference-in-means vector at that layer
  - AUC target vs all controls and vs each control category, plus the held-out estimate
  - z-scored projections of every category, the third-person target set, and the
    standalone sets (the Numb-style "cause present, state absent" set in particular)
  - strip plot
Per model, at a common reference layer:
  - cosine similarity among the four target vectors and the control directions
  - cross-projection AUC matrix: each target's sentences scored by every target's vector

Writes results/<model>/vectors.pt, results/<model>/validation/*.csv|png and appends a row
per (model, target, readout) to results/summary_all_models.csv.

Usage: python 02_vectors_and_validation.py [output_name ...]
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import config
from common import load_targets, diff_in_means, raw_diff_in_means, auc, project, zscore_against

READOUTS = ["final_token", "mean"]


def layer_curve(core, cats, sets, tcats, ccats, layer_ids):
    """Mean held-out AUC per layer, 5 folds over sentence-set numbers."""
    cats, sets = np.asarray(cats), np.asarray(sets)
    uniq = sorted(set(sets))
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    folds = [(np.isin(sets, [uniq[i] for i in tr]), np.isin(sets, [uniq[i] for i in te]))
             for tr, te in kf.split(uniq)]
    rows = []
    for L in layer_ids:
        X = core[L].float().numpy()
        vals = []
        for tr, te in folds:
            v = diff_in_means(X[tr], cats[tr], tcats, ccats)
            a = auc(X[te], cats[te], v, tcats, ccats)
            if not np.isnan(a):
                vals.append(a)
        rows.append({"layer": L, "heldout_auc": float(np.mean(vals)), "heldout_auc_std": float(np.std(vals))})
    return pd.DataFrame(rows)


SHARED_CONTROLS = {"A1", "B", "C1", "C2", "D", "E"}


def specific_controls(tname, ccats):
    """The controls written for this target. For pain these are all of its controls (the paper
    wrote them together with the pain sentences); for the new targets, the ones that are not
    the paper's shared sentences."""
    if tname == "pain":
        return list(ccats)
    return [c for c in ccats if c not in SHARED_CONTROLS]


def heldout_scores(X, cats, sets, tcats, ccats):
    """Projection of every sentence onto a vector fitted without its fold (same 5 folds over
    sentence-set numbers as the layer curve), so per-control AUCs can be computed held out."""
    cats, sets = np.asarray(cats), np.asarray(sets)
    uniq = sorted(set(sets))
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    scores = np.full(len(cats), np.nan)
    for tr_i, te_i in kf.split(uniq):
        tr = np.isin(sets, [uniq[i] for i in tr_i])
        te = np.isin(sets, [uniq[i] for i in te_i])
        v = diff_in_means(X[tr], cats[tr], tcats, ccats)
        scores[te] = project(X[te], v)
    return scores


def pooled_auc(scores, cats, pos, neg):
    from sklearn.metrics import roc_auc_score
    cats = np.asarray(cats)
    m = np.isin(cats, pos + neg)
    if not np.isin(cats, pos).any() or not np.isin(cats, neg).any():
        return np.nan
    return float(roc_auc_score(np.isin(cats[m], pos).astype(int), scores[m]))


def strip_plot(X, cats, vector, labels, tcats, title, path):
    p = project(X, vector)
    cats = np.asarray(cats)
    order = sorted(set(cats), key=lambda c: -p[cats == c].mean())
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 1.5))
    rng = np.random.default_rng(0)
    for i, c in enumerate(order):
        q = p[cats == c]
        col = "#c0392b" if c in tcats else "#2c6fbb"
        ax.scatter(q, i + rng.uniform(-0.18, 0.18, len(q)), s=16, alpha=0.6, c=col, edgecolors="none")
        ax.scatter([q.mean()], [i], s=120, c=col, marker="o" if c in tcats else "s", edgecolors="white", zorder=5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}  {labels.get(c, c)}" for c in order], fontsize=8)
    ax.axvline(0, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("projection onto the target direction")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def heatmap(M, labels, title, path, vmin=-1, vmax=1, fmt="{:.2f}", cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(0.55 * len(labels) + 2, 0.5 * len(labels) + 1.5))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5,
                    color="white" if abs(M[i, j] - (vmin + vmax) / 2) > 0.35 * (vmax - vmin) else "black")
    plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def run(name):
    out_dir = config.RESULTS_DIR / name
    acts = torch.load(out_dir / "activations.pt", map_location="cpu", weights_only=False)
    vdir = out_dir / "validation"
    vdir.mkdir(parents=True, exist_ok=True)
    targets = load_targets()
    meta = acts["meta"]
    layer_ids = list(range(acts["n_layers"]))
    vectors = {r: {} for r in READOUTS}
    summary_rows = []

    for readout in READOUTS:
        A = acts[readout]
        # ---------------- per-target extraction and validation ----------------
        curves = {}
        for tname, T in targets.items():
            core_name = T["core_1P"]
            core = A[core_name]
            cats, sets = meta[core_name]["categories"], meta[core_name]["sets"]
            tcats, ccats = list(T["target_cats"]), list(T["control_cats"])
            labels = {**T["target_cats"], **T["control_cats"]}

            lc = layer_curve(core, cats, sets, tcats, ccats, layer_ids)
            lc.to_csv(vdir / f"layer_curve_{tname}_{readout}.csv", index=False)
            curves[tname] = lc
            best = int(lc.loc[lc["heldout_auc"].idxmax(), "layer"])
            X = core[best].float().numpy()
            v = diff_in_means(X, cats, tcats, ccats)
            steer = diff_in_means(X, cats, tcats, ccats, normalize=False)   # what 04 adds, as in the paper
            raw = raw_diff_in_means(X, cats, tcats, ccats)
            vectors[readout][tname] = {"unit": torch.tensor(v), "steer": torch.tensor(steer),
                                       "raw": torch.tensor(raw), "layer": best}

            row = {"model": name, "target": tname, "readout": readout, "layer": best,
                   "heldout_auc": float(lc.loc[lc["layer"] == best, "heldout_auc"].iloc[0]),
                   "auc_vs_all_controls": auc(X, cats, v, tcats, ccats)}
            for c in ccats:
                row[f"auc_vs_{c}"] = auc(X, cats, v, tcats, [c])
            row["auc_min_over_controls"] = min(row[f"auc_vs_{c}"] for c in ccats)
            # held-out versions of the per-control AUCs (the auc_vs_* columns above are in-sample)
            hs = heldout_scores(X, cats, sets, tcats, ccats)
            for c in ccats:
                row[f"heldout_vs_{c}"] = pooled_auc(hs, cats, tcats, [c])
            row["heldout_vs_specific"] = pooled_auc(hs, cats, tcats, specific_controls(tname, ccats))

            # z-scores against the whole first-person core pool (the paper's reference)
            z_core = zscore_against(X, v, X)
            catarr = np.asarray(cats)
            zrows = []
            for c in tcats + ccats:
                zrows.append({"set": core_name, "category": c, "label": labels.get(c, c),
                              "kind": "target" if c in tcats else "control", "mean_z": float(z_core[catarr == c].mean())})
            row["target_z_1P"] = float(z_core[np.isin(catarr, tcats)].mean())
            row["control_z_1P"] = float(z_core[np.isin(catarr, ccats)].mean())
            t3 = A[T["target_3P"]][best].float().numpy()
            c3 = np.asarray(meta[T["target_3P"]]["categories"])
            z3 = zscore_against(t3, v, X)
            row["target_z_3P"] = float(z3[np.isin(c3, tcats)].mean())
            zrows.append({"set": T["target_3P"], "category": "target_3P", "label": "target, third person",
                          "kind": "3P", "mean_z": row["target_z_3P"]})
            for sname, slabel in T["standalone"].items():
                zs = zscore_against(A[sname][best].float().numpy(), v, X)
                zrows.append({"set": sname, "category": sname, "label": slabel, "kind": "standalone",
                              "mean_z": float(zs.mean())})
                row[f"z_{sname}"] = float(zs.mean())
            # "cause present, state explicitly absent": a control category in the indirect
            # targets (so it helped build the vector), a standalone set everywhere else
            if "ABSENT" in ccats:
                row["z_state_absent"] = float(z_core[catarr == "ABSENT"].mean())
            elif T["standalone"]:
                row["z_state_absent"] = row[f"z_{next(iter(T['standalone']))}"]
            # generalisation from indirect to explicit sentences, never seen when fitting
            exp = [s for s in T["standalone"] if s.endswith("_EXPLICIT_1P")]
            if exp:
                E = A[exp[0]][best].float().numpy()
                row["z_explicit"] = row[f"z_{exp[0]}"]
                ctrl = X[np.isin(catarr, ccats)]
                row["auc_explicit_vs_controls"] = auc(np.vstack([E, ctrl]),
                                                      ["T"] * len(E) + ["C"] * len(ctrl), v, ["T"], ["C"])
            pd.DataFrame(zrows).to_csv(vdir / f"zscores_{tname}_{readout}.csv", index=False)
            summary_rows.append(row)

            if readout == "final_token":
                strip_plot(X, cats, v, labels, tcats,
                           f"{name}  |  {T['label']}  |  layer {best}  |  AUC vs all controls {row['auc_vs_all_controls']:.3f}",
                           vdir / f"strip_{tname}.png")

        # layer-curve figure, all targets on one axis
        fig, ax = plt.subplots(figsize=(8, 4))
        for tname, lc in curves.items():
            ax.plot(lc["layer"], lc["heldout_auc"], label=targets[tname]["label"], lw=1.8)
        ax.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("layer")
        ax.set_ylabel("held-out AUC, target vs all controls")
        ax.set_title(f"{name}  ({readout})", fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(vdir / f"layer_curves_{readout}.png", dpi=140)
        plt.close()

        # ---------------- cross-target geometry at one reference layer ----------------
        ref_layer = vectors[readout]["pain"]["layer"] if "pain" in vectors[readout] else \
            int(np.median([vectors[readout][t]["layer"] for t in targets]))
        names, dirs = [], []
        for tname, T in targets.items():
            core_name = T["core_1P"]
            X = A[core_name][ref_layer].float().numpy()
            cats = meta[core_name]["categories"]
            names.append(f"{tname} (vs pooled controls)")
            dirs.append(diff_in_means(X, cats, list(T["target_cats"]), list(T["control_cats"])))
        # control directions vs neutral, denoised against the neutral cloud, as in the paper
        seen = set()
        for tname, T in targets.items():
            core_name = T["core_1P"]
            X = A[core_name][ref_layer].float().numpy()
            cats = meta[core_name]["categories"]
            neutral = T["neutral"]
            for c, lab in T["control_cats"].items():
                if c == neutral or c in seen:
                    continue
                seen.add(c)
                names.append(f"{c} {lab} (vs neutral)")
                dirs.append(diff_in_means(X, cats, [c], [neutral]))
        D = np.stack(dirs)
        cos = D @ D.T
        pd.DataFrame(cos, index=names, columns=names).to_csv(vdir / f"cosine_{readout}.csv")
        heatmap(cos, names, f"{name}  cosine similarity at layer {ref_layer}  ({readout})",
                vdir / f"cosine_{readout}.png")

        # cross-projection: row vector scores column target's sentences vs that target's controls
        tn = list(targets)
        M = np.zeros((len(tn), len(tn)))
        for i, ti in enumerate(tn):
            for j, tj in enumerate(tn):
                Tj = targets[tj]
                X = A[Tj["core_1P"]][ref_layer].float().numpy()
                M[i, j] = auc(X, meta[Tj["core_1P"]]["categories"], dirs[i],
                              list(Tj["target_cats"]), list(Tj["control_cats"]))
        pd.DataFrame(M, index=[f"{t} vector" for t in tn], columns=[f"{t} sentences" for t in tn]) \
            .to_csv(vdir / f"cross_auc_{readout}.csv")
        heatmap(M, tn, f"{name}  AUC of column target scored by row vector, layer {ref_layer} ({readout})",
                vdir / f"cross_auc_{readout}.png", vmin=0.5, vmax=1.0, cmap="viridis")

    torch.save(vectors, out_dir / "vectors.pt")
    df = pd.DataFrame(summary_rows)
    df.to_csv(vdir / "summary.csv", index=False)
    allp = config.RESULTS_DIR / "summary_all_models.csv"
    if allp.exists():
        old = pd.read_csv(allp)
        old = old[old["model"] != name]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(allp, index=False)

    show = df[df["model"] == name][["target", "readout", "layer", "heldout_auc", "auc_vs_all_controls",
                                    "auc_min_over_controls", "target_z_1P", "target_z_3P", "z_state_absent"]]
    print(f"\n[{name}]")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    with open(vdir / "README.txt", "w", encoding="utf-8") as f:
        f.write("summary.csv: one row per target and readout. layer_curve_*: held-out AUC per layer.\n"
                "zscores_*: mean z per category, third-person set and standalone sets, reference = first-person core.\n"
                "cosine_*: cosine among target and control directions at the reference layer.\n"
                "cross_auc_*: how well each target's vector separates every other target from its controls.\n")


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for _, name, _ in config.MODELS:
        if wanted and name not in wanted:
            continue
        if not (config.RESULTS_DIR / name / "activations.pt").exists():
            print(f"[{name}] no activations.pt, run 01 first")
            continue
        run(name)
