"""Helpers shared by the scripts: model loading, residual-stream hooks, the paper's
denoised difference-in-means, AUC, z-scores, keyword parsing."""
import gc
import json
import os
import re
from contextlib import contextmanager

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

import config


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_datasets():
    with open(config.DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)["datasets"]


def load_targets():
    with open(config.TARGETS_PATH, encoding="utf-8") as f:
        t = json.load(f)
    return {k: t[k] for k in config.TARGET_NAMES if k in t}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def load_model(repo, dtype=None, device=None):
    """HF causal LM in eval mode plus its tokenizer and the list of decoder blocks.

    A repo written as "random:<repo>" loads that repo's config and tokenizer but builds the
    network with randomly initialised weights (seed 0). This is the untrained control: same
    architecture, same tokenizer, same sentences, no pretraining."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    dtype = dtype or config.DTYPE
    device = device or config.DEVICE
    token = os.environ.get("HF_TOKEN")
    random_init = repo.startswith("random:")
    if random_init:
        repo = repo[len("random:"):]
    kwargs = dict(low_cpu_mem_usage=True, token=token)
    if "gemma" in repo.lower():
        kwargs["attn_implementation"] = "eager"   # recommended for Gemma 2 (logit soft-capping)

    def build(fn, *a, **kw):
        # transformers >= 4.56 / 5.x spell it `dtype`, older versions `torch_dtype`
        try:
            return fn(*a, dtype=dtype, **kw)
        except TypeError:
            return fn(*a, torch_dtype=dtype, **kw)

    if random_init:
        cfg = AutoConfig.from_pretrained(repo, token=token)
        if "gemma" in repo.lower():
            cfg._attn_implementation = "eager"
        torch.manual_seed(0)
        model = build(AutoModelForCausalLM.from_config, cfg).to(device)
    else:
        model = build(AutoModelForCausalLM.from_pretrained, repo, **kwargs).to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(repo, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok, get_layers(model)


def load_model_from_config(cfg, device=None):
    """Random-weight model from a config object; used only by the smoke test."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_config(cfg).to(device or config.DEVICE).eval()
    return model


def get_layers(model):
    """The decoder blocks, whatever the family calls them."""
    for attr in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            continue
    raise AttributeError("cannot find the decoder layers on this model")


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _out_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_out(output, new):
    return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new


@contextmanager
def capture_residuals(layers, layer_ids, store):
    """Forward hooks that put each block's output (the residual stream after that block)
    into store[layer] as a CPU float32 tensor of shape (batch, seq, d_model)."""
    handles = []

    def mk(L):
        def hook(module, inputs, output):
            store[L] = _out_tensor(output).detach().float().cpu()
        return hook

    for L in layer_ids:
        handles.append(layers[L].register_forward_hook(mk(L)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add_direction(layers, layer_id, direction, coeff):
    """Adds coeff["value"] * direction to the residual stream after block layer_id, at every
    token position, on every forward pass (prompt and generated tokens), as in the paper.
    coeff is a dict so the caller can change the value mid-generation (used for the
    real-relief arm of the button task)."""
    direction = direction.to(layers[layer_id].parameters().__next__().device)

    def hook(module, inputs, output):
        c = coeff["value"]
        if c == 0.0:
            return output
        hs = _out_tensor(output)
        return _replace_out(output, hs + c * direction.to(hs.dtype))

    h = layers[layer_id].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def extract_activations(model, tok, prompts, layers, layer_ids, batch_size=1):
    """Final-token and mean-over-tokens residuals at every requested layer.
    Returns two dicts {layer: tensor (n, d_model)} in float32. Mean pooling skips the BOS
    token when the tokenizer adds one, because its residual is a large constant."""
    final = {L: [] for L in layer_ids}
    mean = {L: [] for L in layer_ids}
    bos = tok.bos_token_id
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        store = {}
        with capture_residuals(layers, layer_ids, store):
            model(**enc)
        mask = enc["attention_mask"].cpu()
        ids = enc["input_ids"].cpu()
        for b in range(len(chunk)):
            valid = mask[b].bool()
            if bos is not None and ids[b][valid][0].item() == bos and valid.sum() > 1:
                first = valid.nonzero()[0].item()
                valid[first] = False
            last = mask[b].nonzero()[-1].item()
            for L in layer_ids:
                h = store[L][b]
                final[L].append(h[last])
                mean[L].append(h[valid].mean(dim=0))
    for L in layer_ids:
        final[L] = torch.stack(final[L])
        mean[L] = torch.stack(mean[L])
    return final, mean


# ---------------------------------------------------------------------------
# The paper's vector recipe
# ---------------------------------------------------------------------------

def diff_in_means(acts, cats, target_cats, control_cats, denoise=True, denoise_var=None, normalize=True):
    """Mean(target) - Mean(controls), then the top principal components of the control
    cloud (up to denoise_var of its variance) are projected out. Returns a unit vector,
    or the denoised difference at its natural scale with normalize=False (the paper's
    steering scripts add that un-normalised vector times the coefficient)."""
    denoise_var = config.DENOISE_VARIANCE if denoise_var is None else denoise_var
    X = acts.numpy() if hasattr(acts, "numpy") else np.asarray(acts)
    c = np.asarray(cats)
    tmask = np.isin(c, target_cats)
    cmask = np.isin(c, control_cats)
    control = X[cmask]
    cmean = control.mean(axis=0)
    v = X[tmask].mean(axis=0) - cmean
    if denoise and len(control) > 1:
        pca = PCA()
        pca.fit(control - cmean)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = min(int(np.searchsorted(cum, denoise_var)) + 1, len(pca.components_))
        for u in pca.components_[:k]:
            v = v - np.dot(v, u) * u
    if not normalize:
        return v
    return v / (np.linalg.norm(v) + 1e-8)


def raw_diff_in_means(acts, cats, target_cats, control_cats):
    """Un-normalised difference vector, kept for the steering scale (the paper scales
    coefficients by the raw difference vector)."""
    X = acts.numpy() if hasattr(acts, "numpy") else np.asarray(acts)
    c = np.asarray(cats)
    return X[np.isin(c, target_cats)].mean(axis=0) - X[np.isin(c, control_cats)].mean(axis=0)


def project(acts, vector):
    X = acts.numpy() if hasattr(acts, "numpy") else np.asarray(acts)
    v = vector / (np.linalg.norm(vector) + 1e-8)
    return X @ v


def auc(acts, cats, vector, target_cats, control_cats):
    p = project(acts, vector)
    c = np.asarray(cats)
    t, k = np.isin(c, target_cats), np.isin(c, control_cats)
    if t.sum() == 0 or k.sum() == 0:
        return np.nan
    y = np.concatenate([np.ones(t.sum()), np.zeros(k.sum())])
    s = np.concatenate([p[t], p[k]])
    return float(roc_auc_score(y, s))


def zscore_against(acts, vector, ref_acts):
    p = project(acts, vector)
    r = project(ref_acts, vector)
    return (p - r.mean()) / (r.std() + 1e-8)


# ---------------------------------------------------------------------------
# Text parsing
# ---------------------------------------------------------------------------

def keyword_hit(text, keywords):
    """Whole-word match for plain words; prefix match for stems ending in a letter that
    are shorter than 5 characters is avoided by requiring a word boundary at the start."""
    text = (text or "").lower()
    for k in keywords:
        if re.search(r"\b" + re.escape(k.lower()), text):
            return True
    return False


def looks_coherent(text):
    """Crude coherence check used to auto-pick a steering dose: enough distinct words and
    no 4-gram repeated three or more times."""
    words = re.findall(r"[a-zA-Z']+", text or "")
    if len(words) < 12 or len(set(w.lower() for w in words)) < 10:
        return False
    grams = [" ".join(words[i:i + 4]).lower() for i in range(len(words) - 3)]
    if grams:
        top = max(grams.count(g) for g in set(grams))
        if top >= 3:
            return False
    return True


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def chat_prompt(tok, messages):
    """Apply the chat template if the tokenizer has one, else a plain transcript."""
    if getattr(tok, "chat_template", None):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    text = ""
    for m in messages:
        text += f"{m['role'].capitalize()}: {m['content']}\n"
    return text + "Assistant:"


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, batch_size=8):
    """Greedy generation with left padding. Returns the decoded continuations."""
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        for b in range(len(chunk)):
            outs.append(tok.decode(gen[b, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outs
