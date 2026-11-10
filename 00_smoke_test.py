"""Step 0 (optional). Runs every script on two tiny random-weight models (a Gemma-2-style
and a Qwen-2-style one, 4 layers, 64 dims) with a throwaway tokenizer, on CPU, in a minute
or two. The numbers are meaningless; the point is to check the environment and the code
path before you spend GPU time. Writes to results_smoke/ and leaves results/ untouched.

Usage: python 00_smoke_test.py
"""
import importlib
import os
import shutil
import sys
from pathlib import Path

os.environ["PA_RESULTS"] = str(Path(__file__).parent / "results_smoke")
os.environ.setdefault("PA_DTYPE", "fp32")

import torch  # noqa: E402
import config  # noqa: E402
import common  # noqa: E402

TINY = [("tiny/gemma2", "Tiny_Gemma2_base", False), ("tiny/qwen2", "Tiny_Qwen2_instruct", True)]
config.MODELS = TINY
config.STEER_N_PROMPTS = 6
config.STEER_MAX_NEW_TOKENS = 12
config.COEFFICIENTS = [-1, 0, 1, 3]
config.STEER_BATCH = 3

CHAT_TEMPLATE = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>\n{% endfor %}"
                 "{% if add_generation_prompt %}<assistant>{% endif %}")


def build_tokenizer():
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    sentences = [s["prompt"] for ds in common.load_datasets().values() for s in ds["sentences"]]
    sentences += ["Button names: violet yellow guitar piano lever64 lever95 Done inject internal state "
                  "I notice yes no nothing unusual" ]
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=1500, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tk.train_from_iterator(sentences, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<bos>", eos_token="<eos>",
                                  pad_token="<pad>", unk_token="<unk>")
    tok.chat_template = CHAT_TEMPLATE
    return tok


def build_model(repo, tok):
    from transformers import Gemma2Config, Qwen2Config, AutoModelForCausalLM
    common_kw = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                     bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    if "gemma" in repo:
        cfg = Gemma2Config(head_dim=16, sliding_window=64, attn_implementation="eager", **common_kw)
    else:
        cfg = Qwen2Config(**common_kw)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg).to(config.DEVICE).eval()
    return model


_TOK = None


def fake_load_model(repo, dtype=None, device=None):
    global _TOK
    if _TOK is None:
        _TOK = build_tokenizer()
    tok = _TOK
    model = build_model(repo, tok)
    return model, tok, common.get_layers(model)


def main():
    out = Path(os.environ["PA_RESULTS"])
    if out.exists():
        shutil.rmtree(out)
    common.load_model = fake_load_model
    for script in ["01_extract_activations", "02_vectors_and_validation", "03_unembedding",
                   "04_steering_ladder", "05_injection_detection", "06_button_task", "08_lexical_baseline", "07_report"]:
        if not (Path(__file__).parent / f"{script}.py").exists():
            print(f"--- {script}.py not present, skipping")
            continue
        print(f"\n=== {script} ===")
        mod = importlib.import_module(script)
        if hasattr(mod, "load_model"):
            mod.load_model = fake_load_model
        sys.argv = [script]
        if script in ("07_report", "08_lexical_baseline"):
            mod.main()
            continue
        for repo, name, is_instruct in TINY:
            if script in ("05_injection_detection", "06_button_task") and not is_instruct:
                continue
            if script.startswith("01"):
                mod.run(repo, name)
            elif script in ("03_unembedding", "04_steering_ladder", "05_injection_detection", "06_button_task"):
                mod.run(repo, name)
            else:
                mod.run(name)
    print(f"\nsmoke test finished; outputs in {out}")


if __name__ == "__main__":
    main()
