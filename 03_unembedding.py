"""Step 3. Vocabulary readout of each target vector through the unembedding matrix.

For each target's unit vector (final-token readout, from 02), the dot product with every
row of the output embedding, as in the paper (raw W_U), plus a variant that first applies
the final RMSNorm scale, which is closer to what the logits actually see.

Writes results/<model>/unembedding/<target>.csv (top and bottom 40 tokens, both variants)
and results/<model>/unembedding/side_by_side.md with the top 20 per target in one table.

Usage: python 03_unembedding.py [output_name ...]
"""
import sys

import pandas as pd
import torch

import config
from common import load_model, load_targets, free

TOP_N = 40


def norm_scale(model):
    """Final norm weight as a multiplicative scale; Gemma's RMSNorm uses (1 + w)."""
    for attr in ("model.norm", "model.language_model.norm", "transformer.ln_f"):
        obj = model
        try:
            for p in attr.split("."):
                obj = getattr(obj, p)
        except AttributeError:
            continue
        w = obj.weight.detach().float().cpu()
        if "gemma" in type(model).__name__.lower():
            return 1.0 + w
        return w
    return None


def run(repo, name):
    if name.startswith("Random"):
        print(f"[{name}] random-init control, skipping unembedding")
        return
    out_dir = config.RESULTS_DIR / name
    vec_path = out_dir / "vectors.pt"
    if not vec_path.exists():
        print(f"[{name}] no vectors.pt, run 02 first")
        return
    vectors = torch.load(vec_path, map_location="cpu", weights_only=False)["final_token"]
    targets = load_targets()
    udir = out_dir / "unembedding"
    udir.mkdir(parents=True, exist_ok=True)

    model, tok, _ = load_model(repo)
    W = model.get_output_embeddings().weight.detach().float().cpu()      # (vocab, d)
    scale = norm_scale(model)
    del model
    free()

    side = {}
    for tname, T in targets.items():
        v = vectors[tname]["unit"].float()
        rows = []
        for variant, vv in (("raw", v), ("normed", v * scale if scale is not None else v)):
            logits = W @ vv
            top = torch.topk(logits, TOP_N)
            bot = torch.topk(-logits, TOP_N)
            for rank, (i, s) in enumerate(zip(top.indices.tolist(), top.values.tolist())):
                rows.append({"variant": variant, "end": "promoted", "rank": rank + 1,
                             "token": tok.decode([i]), "score": s})
            for rank, (i, s) in enumerate(zip(bot.indices.tolist(), (-bot.values).tolist())):
                rows.append({"variant": variant, "end": "suppressed", "rank": rank + 1,
                             "token": tok.decode([i]), "score": s})
        df = pd.DataFrame(rows)
        df.to_csv(udir / f"{tname}.csv", index=False)
        raw_top = df[(df.variant == "raw") & (df.end == "promoted")].head(20)["token"].tolist()
        raw_bot = df[(df.variant == "raw") & (df.end == "suppressed")].head(10)["token"].tolist()
        side[tname] = (raw_top, raw_bot)
        print(f"[{name}] {tname:8s} promoted: {', '.join(repr(t) for t in raw_top[:12])}")
        print(f"[{name}] {tname:8s} suppressed: {', '.join(repr(t) for t in raw_bot[:8])}")

    names = list(side)
    lines = [f"# {name}: unembedding readout (raw W_U, layer per target from 02)\n",
             "| rank | " + " | ".join(targets[t]["label"] for t in names) + " |",
             "|---|" + "---|" * len(names)]
    for r in range(20):
        lines.append(f"| {r + 1} | " + " | ".join(f"`{side[t][0][r]}`" for t in names) + " |")
    lines.append("\n**Suppressed (bottom 10)**\n")
    lines.append("| rank | " + " | ".join(targets[t]["label"] for t in names) + " |")
    lines.append("|---|" + "---|" * len(names))
    for r in range(10):
        lines.append(f"| {r + 1} | " + " | ".join(f"`{side[t][1][r]}`" for t in names) + " |")
    (udir / "side_by_side.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    wanted = set(sys.argv[1:])
    for repo, name, _ in config.MODELS:
        if wanted and name not in wanted:
            continue
        run(repo, name)
