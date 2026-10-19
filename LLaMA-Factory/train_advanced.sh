#!/bin/bash

#SBATCH --job-name=qwen-advanced15k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=ece-6524-spring2026
#SBATCH --gres=gpu:1
#SBATCH --output=advanced15k_%j.out

module load CUDA/12.6.0

ENV=/home/jacobdavis404/.conda/envs/reasoning-sft
export PATH="$ENV/bin:$PATH"

cd /home/jacobdavis404/reasoning-sft/Project-Reasoning-SFT-LLM/LLaMA-Factory

echo "=== Advanced 15K SFT ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
which python
python --version

echo
echo "=== GPU ==="
nvidia-smi

echo
echo "=== Starting training ==="
FORCE_TORCHRUN=1 llamafactory-cli train yamls/advanced_15k.yaml

echo
echo "=== Job finished ==="
date
