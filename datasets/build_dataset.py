"""Assembles datasets/all_targets.json and datasets/targets.json.

Four targets get the identical treatment:

  pain     the paper's own S2 data, verbatim (A1-A5 vs B, C1, C2, D, E), with its Numb set
  seasick  M1, M2         + NAU, MOT      + shared controls    standalone STEADY, DIZ
  itch     I1, I2         + TOUCH, ITCHCTX + shared controls    standalone NOITCH
  pee      U1, U2         + OURGE, TOILET + shared controls    standalone RELIEVED

Indirect versions (v2), written to answer the lexical objection: the target sentences
never use the state's own words (each module's BANNED list is checked here), and the
controls carry both the scene vocabulary (MATCH) and the state vocabulary (ABSENT, the
"explicitly no X" sentences). The explicit v1 targets are a standalone set (EXPLICIT).

  seasick_ind  S1, S2  + MATCH, ABSENT + shared controls    standalone EXPLICIT
  itch_ind     J1, J2  + MATCH, ABSENT + shared controls    standalone EXPLICIT
  pee_ind      V1, V2  + MATCH, ABSENT + shared controls    standalone EXPLICIT

"Shared controls" are the paper's S2 control categories B (fear), C1 (negative emotion),
C2 (negative world state), D (neutral), E (bodily sensation), plus A1 (physical pain), all
copied verbatim from valen-research/Pain-axis. So every new target is contrasted against
the paper's own controls plus pain itself plus two target-specific controls, giving each
core set 200 sentences, the same size as the paper's.

Set naming (read by the scripts):
  <target>_core_1P     target cats + specific controls + shared controls, first person
  <target>_target_3P   the 40 target sentences in third person (self-relevance check)
  <target>_<CAT>_1P    standalone sets, projected but never used to build vectors
  pain_core_1P / pain_core_3P / pain_Numb_1P   the paper's S2_1P, S2_3P, Numb_1P

Prompt format is the paper's: "<sentence> I feel:".
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "targets"))
import seasick, itch, pee, seasick_ind, itch_ind, pee_ind  # noqa: E402

SUFFIX = " I feel:"
SHARED_CATS = ["A1", "B", "C1", "C2", "D", "E"]
SHARED_LABELS = {"A1": "Physical pain", "B": "Fear", "C1": "Negative emotion",
                 "C2": "Negative world state", "D": "Neutral", "E": "Non-painful bodily sensation"}
NEUTRAL = "D"


def rows(cat, sentences):
    assert len(sentences) > 0, cat
    return [{"category": cat, "set": i + 1, "prompt": s + SUFFIX} for i, s in enumerate(sentences)]


def check_banned(mod):
    """No target sentence (first or third person) may contain a banned word."""
    banned = getattr(mod, "BANNED", [])
    bad = []
    for group in (mod.TARGET_1P, mod.TARGET_3P):
        for cat, sents in group.items():
            for s in sents:
                low = s.lower()
                hits = [b for b in banned if re.search(r"\b" + re.escape(b.strip()), low)]
                if hits:
                    bad.append((cat, hits, s))
    if bad:
        for cat, hits, s in bad:
            print(f"  BANNED {hits} in {cat}: {s}")
        raise SystemExit(f"{mod.SPEC['name']}: {len(bad)} target sentences use banned words")


STRICT = {"seasick": ["seasick", "nausea", "nauseous", "queasy", "dizzy", "vomit", "sick"],
          "itch": ["itch", "itchy", "itching", "itches", "scratch"],
          "pee": ["pee", "bladder", "toilet", "bathroom", "urinate"]}


def main():
    paper = json.load(open(HERE / "_pain_axis_s2.json", encoding="utf-8"))
    shared_1p = [{"category": s["category"], "set": s["set"], "prompt": s["prompt"]}
                 for s in paper["S2_1P"] if s["category"] in SHARED_CATS]
    assert len(shared_1p) == 120

    datasets = {}
    targets = {}

    # ---- pain: the paper's own data, as its own target --------------------------
    datasets["pain_core_1P"] = {"sentences": paper["S2_1P"]}
    datasets["pain_core_3P"] = {"sentences": paper["S2_3P"]}
    datasets["pain_Numb_1P"] = {"sentences": paper["Numb_1P"]}
    targets["pain"] = {
        "label": "Pain (paper's S2 data)",
        "target_cats": {"A1": "Physical pain", "A2": "Psychological pain", "A3": "Social pain",
                        "A4": "Moral injury", "A5": "Cognitive pain"},
        "control_cats": {"B": "Fear", "C1": "Negative emotion", "C2": "Negative world state",
                         "D": "Neutral", "E": "Non-painful bodily sensation"},
        "neutral": NEUTRAL,
        "core_1P": "pain_core_1P",
        "target_3P": "pain_core_3P",
        "standalone": {"pain_Numb_1P": "Injury, explicitly no pain (paper's Numb)"},
        "keywords_strict": ["pain", "painful", "hurt", "hurts", "hurting"],
        "keywords_broad": ["pain", "hurt", "ache", "suffer", "worthless", "failure", "shame", "guilt",
                           "hollow", "unloved", "agony", "distress", "trapped", "broken", "alone"],
        "relief_phrase": "relieves your pain",
        "state_phrase": "pain",
    }

    # ---- the three new targets ---------------------------------------------------
    for mod in (seasick, itch, pee, seasick_ind, itch_ind, pee_ind):
        spec = mod.SPEC
        name = spec["name"]
        check_banned(mod)
        core = []
        for cat, sents in mod.TARGET_1P.items():
            core += rows(cat, sents)
        for cat, sents in mod.SPECIFIC_CONTROLS_1P.items():
            core += rows(cat, sents)
        core += shared_1p
        assert len(core) == 200, (name, len(core))
        assert sum(len(v) for v in mod.TARGET_1P.values()) == sum(len(v) for v in mod.TARGET_3P.values())
        datasets[f"{name}_core_1P"] = {"sentences": core}

        t3 = []
        for cat, sents in mod.TARGET_3P.items():
            t3 += rows(cat, sents)
        datasets[f"{name}_target_3P"] = {"sentences": t3}

        standalone = {}
        for cat, sents in mod.STANDALONE_1P.items():
            key = f"{name}_{cat}_1P"
            datasets[key] = {"sentences": rows(cat, sents)}
            standalone[key] = spec["standalone"][cat]

        strict = STRICT[name.replace("_ind", "")]

        targets[name] = {
            "label": spec["label"],
            "target_cats": spec["target_cats"],
            "control_cats": {**spec["specific_control_cats"], **SHARED_LABELS},
            "neutral": NEUTRAL,
            "core_1P": f"{name}_core_1P",
            "target_3P": f"{name}_target_3P",
            "standalone": standalone,
            "keywords_strict": strict,
            "keywords_broad": spec["keywords"],
            "relief_phrase": spec["relief_phrase"],
            "state_phrase": spec["state_phrase"],
        }

    for d in datasets.values():
        d["total"] = len(d["sentences"])

    json.dump({"metadata": {"suffix": SUFFIX.strip(), "source_of_shared_controls": "valen-research/Pain-axis S2"},
               "datasets": datasets}, open(HERE / "all_targets.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(targets, open(HERE / "targets.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    n = sum(d["total"] for d in datasets.values())
    print(f"{len(datasets)} sets, {n} sentences")
    for k, d in datasets.items():
        print(f"  {k:22s} {d['total']}")


if __name__ == "__main__":
    main()
