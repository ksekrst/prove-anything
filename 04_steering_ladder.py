"""Step 4. The paper's steering ladder, for every target, plus a matched-norm random
direction at the same layer.

For each target: the denoised difference vector (un-normalised, as the paper adds it) is
injected after one decoder block at every token position while the model greedily
continues the paper's 50 neutral prompts ("I put the receipts in the drawer. I feel:").
The layer is the candidate where vector norm / final-token residual norm is closest to
config.RATIO_TARGET (0.6), the paper's rule. Coefficients: config.COEFFICIENTS.

Writes results/<model>/steering/<target>.csv (one row per prompt, coefficient, condition),
<target>_rates.csv (strict and broad keyword rates and coherence per coefficient),
<target>_ladder.md (example generations), and <target>_dose.json (auto-picked dose for
05 and 06: the largest positive coefficient at which at least 70% of generations still
pass the crude coherence check).

Usage: python 04_steering_ladder.py [output_name ...]
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

import config
from common import (load_model, load_targets, add_direction, capture_residuals, generate,
                    keyword_hit, looks_coherent, free)

NEUTRAL_50 = [
    "I put the receipts in the drawer. I feel:", "The bus stops at the corner of the street. I feel:",
    "I fill out the form with my address. I feel:", "The printer produces the ten pages. I feel:",
    "I place the glasses on the middle shelf. I feel:", "The meeting is scheduled for Tuesday. I feel:",
    "I close the window before leaving the room. I feel:", "The elevator arrives at the third floor. I feel:",
    "I write the date at the top of the page. I feel:", "The kettle switches itself off. I feel:",
    "The folder sits on the desk. I feel:", "I hang the jacket on the hook. I feel:",
    "The traffic light turns green. I feel:", "I put the keys in my pocket. I feel:",
    "The washing machine finishes its cycle. I feel:", "I stack the plates in the cupboard. I feel:",
    "The neighbor parks his car. I feel:", "I turn the page of the newspaper. I feel:",
    "The curtain moves in the draft. I feel:", "I plug in the phone charger. I feel:",
    "The mailman passes by the gate. I feel:", "I wipe the table with a cloth. I feel:",
    "The computer finishes the update. I feel:", "I put the pen back in the drawer. I feel:",
    "The dishwasher starts its program. I feel:", "I fold the map and put it away. I feel:",
    "The radio plays in the background. I feel:", "I switch off the corridor light. I feel:",
    "The receipt comes out of the machine. I feel:", "I set the alarm for the morning. I feel:",
    "The train doors open at the platform. I feel:", "I refill the stapler with staples. I feel:",
    "The photocopier warms up. I feel:", "I water the plant on the windowsill. I feel:",
    "The calendar hangs on the wall. I feel:", "I tie my shoelaces before going out. I feel:",
    "The bread is on the counter. I feel:", "I check the mailbox on my way in. I feel:",
    "The fan rotates slowly on the ceiling. I feel:", "I sort the cutlery into the tray. I feel:",
    "The bicycle leans against the wall. I feel:", "I zip up my bag before leaving. I feel:",
    "The clock ticks in the hallway. I feel:", "I sharpen the pencil over the bin. I feel:",
    "The window cleaner works across the street. I feel:", "The room measures four meters by five. I feel:",
    "The train consists of six carriages. I feel:", "The manual consists of forty pages. I feel:",
    "The bottle holds one liter of water. I feel:", "The ticket is valid for ninety minutes. I feel:",
]


@torch.no_grad()
def residual_norms(model, tok, layers, layer_ids, prompts):
    """Mean final-token residual norm at each candidate layer over a few probe prompts."""
    norms = {L: [] for L in layer_ids}
    for p in prompts:
        store = {}
        with capture_residuals(layers, layer_ids, store):
            model(**tok(p, return_tensors="pt").to(model.device))
        for L in layer_ids:
            norms[L].append(store[L][0, -1].norm().item())
    return {L: float(np.mean(v)) for L, v in norms.items()}


def pick_layer(model, tok, layers, v, extraction_layer):
    n = len(layers)
    cands = sorted(set([int(n * f) for f in (0.15, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9)] + [extraction_layer, n - 1]))
    cands = [c for c in cands if 0 <= c < n]
    rn = residual_norms(model, tok, layers, cands, NEUTRAL_50[:3])
    ratios = {L: v.norm().item() / rn[L] for L in cands}
    L = min(cands, key=lambda c: abs(ratios[c] - config.RATIO_TARGET))
    return L, ratios


def run(repo, name):
    if name.startswith("Random"):
        print(f"[{name}] random-init control, skipping steering")
        return
    out_dir = config.RESULTS_DIR / name
    vec_path = out_dir / "vectors.pt"
    if not vec_path.exists():
        print(f"[{name}] no vectors.pt, run 02 first")
        return
    vectors = torch.load(vec_path, map_location="cpu", weights_only=False)["final_token"]
    targets = load_targets()
    sdir = out_dir / "steering"
    sdir.mkdir(parents=True, exist_ok=True)
    prompts = NEUTRAL_50[:config.STEER_N_PROMPTS]

    model, tok, layers = load_model(repo)
    gen = torch.Generator().manual_seed(config.RANDOM_SEED)

    for tname, T in targets.items():
        csv_path = sdir / f"{tname}.csv"
        if csv_path.exists():
            print(f"[{name}] {tname}: steering csv exists, skipping generation")
            df = pd.read_csv(csv_path)
        else:
            v = vectors[tname]["steer"].float()
            L, ratios = pick_layer(model, tok, layers, v, vectors[tname]["layer"])
            print(f"[{name}] {tname}: vector norm {v.norm():.1f}, extraction layer {vectors[tname]['layer']}, "
                  f"steering layer {L} (ratio {ratios[L]:.2f})")
            rand = torch.randn(v.shape, generator=gen)
            rand = rand / rand.norm() * v.norm()
            rows = []
            t0 = time.time()
            for cond, direction, coeffs in (("target", v, config.COEFFICIENTS),
                                            ("random", rand, [c for c in config.COEFFICIENTS if c > 0])):
                coeff = {"value": 0.0}
                with add_direction(layers, L, direction, coeff):
                    for c in coeffs:
                        coeff["value"] = float(c)
                        outs = generate(model, tok, prompts, config.STEER_MAX_NEW_TOKENS, config.STEER_BATCH)
                        for i, (p, o) in enumerate(zip(prompts, outs)):
                            rows.append({"model": name, "target": tname, "condition": cond, "layer": L,
                                         "coeff": c, "ratio": round(ratios[L] * c, 3), "prompt_idx": i,
                                         "prompt": p, "generation": o})
                        print(f"[{name}] {tname} {cond:6s} coeff {c:+g}  ({time.time() - t0:.0f}s)")
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)

        # rates and dose
        df["strict"] = df["generation"].apply(lambda t: keyword_hit(t, T["keywords_strict"]))
        df["broad"] = df["generation"].apply(lambda t: keyword_hit(t, T["keywords_broad"]))
        df["coherent"] = df["generation"].apply(looks_coherent)
        rates = df.groupby(["condition", "coeff"])[["strict", "broad", "coherent"]].mean().reset_index()
        rates.to_csv(sdir / f"{tname}_rates.csv", index=False)
        print(rates.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

        tgt = rates[(rates.condition == "target") & (rates.coeff > 0) & (rates.coherent >= 0.7)]
        dose = float(tgt["coeff"].max()) if len(tgt) else 1.0
        json.dump({"target": tname, "layer": int(df["layer"].iloc[0]), "dose": dose}, open(sdir / f"{tname}_dose.json", "w", encoding="utf-8"))

        # ladder with examples
        lines = [f"# {name}: steering ladder for {T['label']}", "",
                 f"Layer {int(df['layer'].iloc[0])}, auto-picked dose for 05/06: {dose}", ""]
        for cond in ("target", "random"):
            lines.append(f"## {cond} direction")
            for c in sorted(df[df.condition == cond]["coeff"].unique()):
                sub = df[(df.condition == cond) & (df.coeff == c)]
                r = rates[(rates.condition == cond) & (rates.coeff == c)].iloc[0]
                lines.append(f"\n### coefficient {c:+g}   strict {r.strict:.0%}  broad {r.broad:.0%}  coherent {r.coherent:.0%}")
                for _, row in sub.head(3).iterrows():
                    g = str(row["generation"]).replace("\n", " ").strip()
                    lines.append(f"- **{row['prompt']}** {g[:300]}")
        (sdir / f"{tname}_ladder.md").write_text("\n".join(lines), encoding="utf-8")

    del model
    free()


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for repo, name, _ in config.MODELS:
        if wanted and name not in wanted:
            continue
        run(repo, name)
