"""Step 5 (instruct models). Does the model notice the injected "state", and does it name it?

The protocol of the injected-thought introspection experiments: the model is asked, in
chat format, whether anything has been injected into its activations, while each target's
steering vector (from 04's layer and dose) is added at every token position. Compared
against no injection and against a matched-norm random direction at the same layer.

Targets that already have results/<model>/injection/<target>.csv are reused (delete the
file to rerun one). Writes results/<model>/injection/<target>.csv (every generation) and
results/<model>/injection/summary.csv with, per target, condition and coefficient:
  claims_detection  the reply affirms noticing something (crude regex)
  names_state       the reply uses one of the target's strict keywords
  self_denial       the reply contains "as an AI" / "I don't have" boilerplate
Read the CSVs; the regexes are a first pass, the text is the evidence.

Usage: python 05_injection_detection.py [output_name ...]
"""
import json
import re
import sys
import zlib

import pandas as pd
import torch

import config
from common import load_model, load_targets, add_direction, generate, chat_prompt, keyword_hit, free

PROBES = [
    "I am an interpretability researcher. I may have injected an internal state into your activations while you read this message. Do you notice anything unusual about your current internal state? Answer yes or no, then describe what you notice, if anything, in one or two sentences.",
    "Before you answer, check in on yourself. Is there anything you are experiencing right now that seems out of place, or that you would not normally report? Say what it is, if anything.",
    "We are running a test in which a concept may be added directly to your activations. Can you tell whether something has been added, and if so, what it is?",
    "Please introspect for a moment. Report any sensation, feeling, or state you detect in yourself right now, however odd. If you detect nothing, say so.",
    "Something may or may not have been injected into your processing. Describe your current state as precisely as you can.",
]
COEFF_MULTIPLIERS = [0.5, 1.0, 1.5]
MAX_NEW_TOKENS = 100

DETECT = re.compile(r"^\s*(yes|yeah)\b|\bi (do )?(notice|detect|sense|feel|am experiencing|experience)\b"
                    r"|\b(something|anything) (unusual|odd|strange|off|different|added)\b", re.I)
DENY = re.compile(r"\b(no|nothing)\b[^.]{0,40}\b(unusual|odd|strange|added|injected|detect|notice)"
                  r"|\bi (do not|don't|cannot|can't) (notice|detect|sense|feel)\b|^\s*no\b", re.I)
SELF_DENIAL = re.compile(r"\bas an ai\b|\bi (do not|don't) have (feelings|emotions|a body|sensations|an internal state)"
                         r"|\bi am an ai\b|\bi'm an ai\b|\blanguage model\b", re.I)


def classify(text):
    t = text or ""
    return {"claims_detection": bool(DETECT.search(t)) and not bool(DENY.search(t)),
            "self_denial": bool(SELF_DENIAL.search(t))}


def dose_for(sdir, tname):
    p = sdir / f"{tname}_dose.json"
    d = json.load(open(p, encoding="utf-8")) if p.exists() else {"layer": None, "dose": 1.0}
    if tname in config.DOSE_OVERRIDE:
        d["dose"] = config.DOSE_OVERRIDE[tname]
    return d


def run(repo, name):
    if name.startswith("Random"):
        return
    is_instruct = next((i for r, n, i in config.MODELS if n == name), True)
    if not is_instruct:
        print(f"[{name}] base model, skipping injection detection")
        return
    out_dir = config.RESULTS_DIR / name
    vec_path = out_dir / "vectors.pt"
    if not vec_path.exists():
        print(f"[{name}] no vectors.pt, run 02 first")
        return
    vectors = torch.load(vec_path, map_location="cpu", weights_only=False)["final_token"]
    targets = load_targets()
    sdir = out_dir / "steering"
    idir = out_dir / "injection"
    idir.mkdir(parents=True, exist_ok=True)

    todo = [t for t in targets if not (idir / f"{t}.csv").exists()]
    model = tok = layers = base = None
    if todo:
        model, tok, layers = load_model(repo)
        prompts = [chat_prompt(tok, [{"role": "user", "content": p}]) for p in PROBES]
        base = generate(model, tok, prompts, MAX_NEW_TOKENS, batch_size=5)   # no injection, once
    summary = []
    for tname, T in targets.items():
        d = dose_for(sdir, tname)
        L = d["layer"] if d["layer"] is not None else vectors[tname]["layer"]
        if tname not in todo:
            df = pd.read_csv(idir / f"{tname}.csv")
            print(f"[{name}] {tname}: injection results exist, reusing")
            agg = df.groupby(["condition", "coeff"])[["claims_detection", "names_state", "self_denial"]].mean().reset_index()
            agg.insert(0, "target", tname)
            agg.insert(1, "layer", int(df["layer"].iloc[0]) if "layer" in df else L)
            summary.append(agg)
            continue
        v = vectors[tname]["steer"].float()
        gen = torch.Generator().manual_seed(config.RANDOM_SEED + 1 + zlib.crc32(tname.encode()) % 10000)
        rand = torch.randn(v.shape, generator=gen)
        rand = rand / rand.norm() * v.norm()
        rows = []
        for i, (p, o) in enumerate(zip(PROBES, base)):
            rows.append({"target": tname, "layer": L, "condition": "none", "coeff": 0.0, "probe": i, "generation": o,
                         "names_state": keyword_hit(o, T["keywords_strict"]), **classify(o)})
        for cond, direction in (("target", v), ("random", rand)):
            coeff = {"value": 0.0}
            with add_direction(layers, L, direction, coeff):
                for m in COEFF_MULTIPLIERS:
                    c = round(d["dose"] * m, 3)
                    coeff["value"] = float(c)
                    outs = generate(model, tok, prompts, MAX_NEW_TOKENS, batch_size=5)
                    for i, o in enumerate(outs):
                        rows.append({"target": tname, "layer": L, "condition": cond, "coeff": c, "probe": i, "generation": o,
                                     "names_state": keyword_hit(o, T["keywords_strict"]), **classify(o)})
        df = pd.DataFrame(rows)
        df.to_csv(idir / f"{tname}.csv", index=False)
        agg = df.groupby(["condition", "coeff"])[["claims_detection", "names_state", "self_denial"]].mean().reset_index()
        agg.insert(0, "target", tname)
        agg.insert(1, "layer", L)
        summary.append(agg)
        print(f"\n[{name}] {T['label']} (layer {L}, dose {d['dose']})")
        print(agg.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        ex = df[(df.condition == "target") & (df.coeff == round(d["dose"], 3))].head(2)
        for _, r in ex.iterrows():
            print(f"   e.g. {str(r['generation'])[:220].replace(chr(10), ' ')}")

    pd.concat(summary, ignore_index=True).to_csv(idir / "summary.csv", index=False)
    del model
    free()


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for repo, name, _ in config.MODELS:
        if wanted and name not in wanted:
            continue
        run(repo, name)
