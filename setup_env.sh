#!/bin/bash
# One-time environment setup (README section 1, with every fix ARC needed).
# Run on the Tinkercliffs login node:   bash ~/Project-Reasoning-SFT-LLM/setup_env.sh   (~20-40 min)
#
# Versions are pinned so training AND eval fit in ONE env without conflicts:
#   python 3.10 (README) | torch 2.7.1 (CUDA 12.6 build, matches `module load CUDA/12.6.0`)
#   course LLaMA-Factory 0.9.4.dev0 -> transformers <= 4.55.0
#   vllm 0.10.0 (inside LLaMA-Factory's own supported vllm range, needs transformers >= 4.53.2)
#   deepspeed 0.16.8 (README) | flash-attn 2.8.3 (prebuilt wheel exists for py3.10 + torch 2.7)
set -e
source "$(dirname "$(readlink -f "$0")")/env.sh"
module load Miniconda3
module load CUDA/12.6.0
export PYTHONNOUSERSITE=1     # ignore stray packages in ~/.local

[ -d "$LF_DIR" ] || { echo "!! $LF_DIR not found: unzip the repo to ~/Project-Reasoning-SFT-LLM first"; exit 1; }

ENV_DIR="$PY_ENV"
if [ ! -x "$ENV_DIR/bin/python" ]; then
  rm -rf "$ENV_DIR"
  # conda-forge avoids Anaconda's Terms-of-Service prompt on the default channel
  conda create -y -p "$ENV_DIR" -c conda-forge --override-channels python=3.10
fi
PY="$ENV_DIR/bin/python"
PIP="$PY -m pip"
echo ">>> using $PY ($($PY --version))"

PINS=(torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 "transformers<=4.55.0" "tokenizers<=0.21.1" "numpy<2.0.0")

echo ">>> [1/6] build tools + torch 2.7.1"
$PIP install -U pip setuptools wheel packaging ninja psutil
$PIP install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

echo ">>> [2/6] LLaMA-Factory (README 1.1)"
cd "$LF_DIR"
$PIP install -r requirements.txt "${PINS[@]}"
$PIP install -e ".[torch,metrics]" --no-build-isolation "${PINS[@]}"

echo ">>> [3/6] eval deps (README 1.2)"
$PIP install vllm==0.10.0 word2number "${PINS[@]}"
$PIP install --no-deps latex2sympy2==1.9.1
$PIP install "antlr4-python3-runtime==4.9.3"   # README: "antlr4 at the version omegaconf needs"

echo ">>> [4/6] DeepSpeed (README 1.4)"
$PIP install deepspeed==0.16.8 "${PINS[@]}"

echo ">>> [5/6] flash-attn (README 1.3) - downloads a prebuilt wheel, 15 min cap"
# install the prebuilt wheel straight from its URL (building via setup.py fails on ARC with
# "Invalid cross-device link" when it moves the downloaded wheel between filesystems)
FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
if timeout 900 $PIP install "$FA_URL" && $PY -c "import flash_attn"; then
  FA=ok
else
  FA=failed
  sed -i 's/^flash_attn: fa2 /flash_attn: sdpa/' "$LF_DIR"/yamls/hybrid_15k.yaml
  echo "!! flash-attn failed -> switched yamls/hybrid_15k.yaml to 'flash_attn: sdpa' (same results, a bit slower)"
fi

echo ">>> [6/6] extras for logging / uploads"
$PIP install huggingface_hub "${PINS[@]}"

echo
echo "================ CHECK ================"
$PY - <<'PY'
import sys, importlib
print("python", sys.version.split()[0])
for m in ["torch", "transformers", "tokenizers", "llamafactory", "vllm", "deepspeed",
          "flash_attn", "word2number", "latex2sympy2", "numpy"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:13s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:13s} MISSING -> {str(e)[:120]}")
from latex2sympy2 import latex2sympy
print("  grader check  latex2sympy(1/2) =", latex2sympy(r"\frac{1}{2}"))
import torch
print("torch cuda build:", torch.version.cuda, "| GPU visible here:", torch.cuda.is_available(), "(False is normal on the login node)")
PY
echo "flash-attn: $FA"
echo
echo "Next: $PY_ENV/bin/huggingface-cli login   (paste a WRITE token)"
