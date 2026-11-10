#!/usr/bin/env bash
# Runs the whole pipeline for every model in config.MODELS. Each step skips work that is
# already on disk, so it is safe to rerun after an interruption.
set -e
cd "$(dirname "$0")"
python 01_extract_activations.py "$@"
python 02_vectors_and_validation.py "$@"
python 03_unembedding.py "$@"
python 04_steering_ladder.py "$@"
python 05_injection_detection.py "$@"
python 06_button_task.py "$@"
python 08_lexical_baseline.py
python 07_report.py
