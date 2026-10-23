#!/bin/bash
# Training-only env for Owl's B200 GPUs (Blackwell needs a CUDA 12.8 PyTorch build).
# Same pinned versions as setup_env.sh, except torch comes from the cu128 index; no vllm/flash-attn
# (eval stays on Tinkercliffs). Run on the OWL login node:  bash ~/Project-Reasoning-SFT-LLM/setup_env_b200.sh
set -e
source "$HOME/Project-Reasoning-SFT-LLM/env.sh"
module load Miniconda3
ENV_DIR="$HOME/.conda/envs/reasoning-sft-b200"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  rm -rf "$ENV_DIR"
  conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.10
fi
PY="$ENV_DIR/bin/python"; PIP="$PY -m pip"
PINS=(torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "transformers<=4.55.0" "tokenizers<=0.21.1" "numpy<2.0.0")
echo ">>> [1/4] torch 2.7.1 (CUDA 12.8 build, supports B200 / sm_100)"
$PIP install -U pip setuptools wheel packaging ninja psutil
$PIP install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
echo ">>> [2/4] LLaMA-Factory"
cd "$LF_DIR"
$PIP install -r requirements.txt "${PINS[@]}"
$PIP install -e ".[torch,metrics]" --no-build-isolation "${PINS[@]}"
echo ">>> [3/4] DeepSpeed"
$PIP install deepspeed==0.16.8 huggingface_hub "${PINS[@]}"
echo ">>> [4/4] CHECK"
$PY - <<'PY'
import torch, transformers, deepspeed, llamafactory
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
print("arch list:", torch.cuda.get_arch_list())
print("sm_100 supported:", "sm_100" in torch.cuda.get_arch_list())
print("transformers", transformers.__version__, "| deepspeed", deepspeed.__version__, "| llamafactory", llamafactory.__version__)
PY
