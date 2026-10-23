#!/bin/bash
# One training job; submitted and chained by submit_hybrid.sh (don't sbatch by hand).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
set -e
source "$HOME/Project-Reasoning-SFT-LLM/env.sh"
PY_ENV="${TRAIN_PY_ENV:-$PY_ENV}"              # B200 jobs use the cu128 env
module load CUDA/12.6.0 2>/dev/null || true
export PATH="$PY_ENV/bin:$PATH"
cd "$LF_DIR"
echo "host: $(hostname) | date: $(date) | python: $(which python) | out: $OUT | overrides: ${LF_OVERRIDES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
if [ -f "$OUT/train_results.json" ]; then echo "training already finished in $OUT - nothing to do"; exit 0; fi
FORCE_TORCHRUN=1 llamafactory-cli train yamls/hybrid_15k.yaml output_dir="$OUT" ${LF_OVERRIDES:-}
echo "=== job finished: $(date) ==="
