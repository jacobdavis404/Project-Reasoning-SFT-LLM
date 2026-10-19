#!/bin/bash

#SBATCH --job-name=advanced-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=ece-6524-spring2026
#SBATCH --gres=gpu:1
#SBATCH --output=advanced_smoke_%j.out

module load CUDA/12.6.0

ENV=/home/jacobdavis404/.conda/envs/reasoning-sft
export PATH="$ENV/bin:$PATH"

cd /home/jacobdavis404/reasoning-sft/Project-Reasoning-SFT-LLM/LLaMA-Factory

echo "=== Environment ==="
which python
python --version

echo "=== GPU ==="
nvidia-smi

echo "=== Advanced 15K smoke test ==="
FORCE_TORCHRUN=1 llamafactory-cli train yamls/advanced_15k_smoke.yaml
