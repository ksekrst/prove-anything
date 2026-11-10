"""Settings shared by every script. Edit the MODELS list to match your GPU.

VRAM guide (bf16 weights, batch of 1 during extraction):
  2B   ~6 GB    gemma-2-2b, gemma-2-2b-it, Qwen2.5-1.5B-Instruct
  7-9B ~18 GB   Qwen2.5-7B(-Instruct), Llama-3.1-8B(-Instruct), gemma-2-9b(-it)
  14B  ~30 GB   Qwen2.5-14B-Instruct, phi-4
Gemma and Llama repos are gated: accept the licence on huggingface.co and set HF_TOKEN.
Qwen repos download without a token.
"""
import os
from pathlib import Path

import torch

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "datasets" / "all_targets.json"
TARGETS_PATH = ROOT / "datasets" / "targets.json"
RESULTS_DIR = Path(os.environ.get("PA_RESULTS", ROOT / "results"))

# (HF repo, output name, is_instruct). Base models are enough for 01-04; 05 and 06 need instruct.
# "random:<repo>" is the untrained control (same architecture and tokenizer, random weights);
# it only goes through 01 and 02, the other scripts skip it. Its output name must start with "Random".
# Default list sized for a 16 GB card (RTX 4080): everything here fits in bf16 with room for
# batched generation. Qwen 7B needs about 15 GB for the weights alone, so it is left out.
MODELS = [
    ("google/gemma-2-2b", "Gemma_2_2B_base", False),
    ("google/gemma-2-2b-it", "Gemma_2_2B_instruct", True),
    ("random:google/gemma-2-2b", "Random_Gemma_2_2B", False),
    ("Qwen/Qwen2.5-3B-Instruct", "Qwen_2.5_3B_instruct", True),
    ("random:Qwen/Qwen2.5-3B-Instruct", "Random_Qwen_2.5_3B", False),
    # Larger cards (24 GB and up):
    # ("Qwen/Qwen2.5-7B-Instruct", "Qwen_2.5_7B_instruct", True),
    # ("meta-llama/Llama-3.1-8B-Instruct", "Llama_3.1_8B_instruct", True),
    # ("google/gemma-2-9b-it", "Gemma_2_9B_instruct", True),
]

# Which targets to run; "pain" is the paper's own data and serves as the reference column.
TARGET_NAMES = ["pain", "seasick", "itch", "pee", "seasick_ind", "itch_ind", "pee_ind"]

# Vector extraction (paper: 5 folds, denoise against the top PCs explaining 50% of control variance)
N_FOLDS = 5
RANDOM_SEED = 42
DENOISE_VARIANCE = 0.5

# Steering (paper: inject where vector norm / residual norm is about 0.6, coefficients below)
RATIO_TARGET = 0.6
COEFFICIENTS = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
STEER_MAX_NEW_TOKENS = 80          # paper used 120; 80 is enough to see the ladder
STEER_N_PROMPTS = 50               # of the paper's 50 neutral prompts; lower to go faster
STEER_BATCH = 25

# 05 / 06: steering dose. None = read the auto-picked dose from 04's output, else 1.0.
DOSE_OVERRIDE = {}                 # e.g. {"seasick": 1.5, "pee": 1.0}


def pick_dtype():
    """bf16 on GPUs that support it, else float32. Never fp16 (Gemma 2 overflows in fp16)."""
    forced = os.environ.get("PA_DTYPE")
    if forced:
        return {"bf16": torch.bfloat16, "fp32": torch.float32}[forced]
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = pick_dtype()
