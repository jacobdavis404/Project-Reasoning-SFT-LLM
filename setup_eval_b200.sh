#!/bin/bash
# Adds evaluation packages (vllm + grader) to the B200 env, so eval shards can run on Owl.
# vllm 0.10.0 is built against torch 2.7.1 / CUDA 12.8 -> matches reasoning-sft-b200 exactly.
# Run on the OWL login node:  bash ~/Project-Reasoning-SFT-LLM/setup_eval_b200.sh   (~5-10 min)
set -e
source "$HOME/Project-Reasoning-SFT-LLM/env.sh"
PY="$HOME/.conda/envs/reasoning-sft-b200/bin/python"; PIP="$PY -m pip"
[ -x "$PY" ] || { echo "!! B200 env missing: run setup_env_b200.sh first"; exit 1; }
PINS=(torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "transformers<=4.55.0" "tokenizers<=0.21.1" "numpy<2.0.0")
$PIP install vllm==0.10.0 word2number "${PINS[@]}"
$PIP install --no-deps latex2sympy2==1.9.1
$PIP install "antlr4-python3-runtime==4.9.3"
$PY - <<'PY'
import torch, vllm, transformers
from latex2sympy2 import latex2sympy
print("torch", torch.__version__, "| vllm", vllm.__version__, "| transformers", transformers.__version__)
print("grader check latex2sympy(1/2) =", latex2sympy(r"\frac{1}{2}"))
PY
echo "Next: GPU sanity test (see STEPS)."
