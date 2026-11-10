"""Step 1. Residual-stream activations at every layer for every sentence set.

For each model in config.MODELS, writes results/<model>/activations.pt:
  {"final_token": {set_name: {layer: (n, d)}}, "mean": {...}, "meta": {set_name: {categories, sets}},
   "n_layers", "d_model", "repo"}
Tensors are stored in float16 to keep the file small. If activations.pt exists, only
sets that are missing from it (or changed size) are extracted and merged in.

Usage:  python 01_extract_activations.py            (all models in config)
        python 01_extract_activations.py Gemma_2_2B_base   (one output name)
"""
import sys
import time

import torch

import config
from common import load_datasets, load_model, extract_activations, free


def run(repo, name):
    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "activations.pt"
    datasets = load_datasets()
    old = None
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=False)
        def changed(k, v):
            m = old["meta"].get(k)
            if m is None or len(m["categories"]) != len(v["sentences"]):
                return True
            return "prompts" in m and m["prompts"] != [s["prompt"] for s in v["sentences"]]
        datasets = {k: v for k, v in datasets.items() if changed(k, v)}
        if not datasets:
            print(f"[{name}] activations.pt already has every set, skipping")
            return
        print(f"[{name}] activations.pt exists; extracting only the {len(datasets)} new or changed sets")
    print(f"[{name}] loading {repo} ({config.DTYPE}, {config.DEVICE})")
    model, tok, layers = load_model(repo)
    n_layers = len(layers)
    layer_ids = list(range(n_layers))
    d_model = model.config.hidden_size
    print(f"[{name}] {n_layers} layers, d_model {d_model}")

    if old is not None:
        final_all, mean_all, meta = old["final_token"], old["mean"], old["meta"]
    else:
        final_all, mean_all, meta = {}, {}, {}
    t0 = time.time()
    for set_name, ds in datasets.items():
        prompts = [s["prompt"] for s in ds["sentences"]]
        final, mean = extract_activations(model, tok, prompts, layers, layer_ids, batch_size=1)
        final_all[set_name] = {L: final[L].half() for L in layer_ids}
        mean_all[set_name] = {L: mean[L].half() for L in layer_ids}
        meta[set_name] = {"categories": [s["category"] for s in ds["sentences"]],
                          "sets": [s["set"] for s in ds["sentences"]],
                          "prompts": prompts}
        print(f"[{name}]   {set_name:22s} {len(prompts):4d} sentences  ({time.time() - t0:.0f}s)")

    torch.save({"final_token": final_all, "mean": mean_all, "meta": meta,
                "n_layers": n_layers, "d_model": d_model, "repo": repo}, path)
    print(f"[{name}] wrote {path}")
    del model
    free()


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for repo, name, _ in config.MODELS:
        if wanted and name not in wanted:
            continue
        run(repo, name)
