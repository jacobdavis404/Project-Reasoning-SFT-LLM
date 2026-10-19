#!/bin/bash

#SBATCH --job-name=qwen-random15k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=ece-6524-spring2026
#SBATCH --gres=gpu:1
#SBATCH --output=random15k_%j.out

module load CUDA/12.6.0

ENV=/home/jacobdavis404/.conda/envs/reasoning-sft
export PATH="$ENV/bin:$PATH"

cd /home/jacobdavis404/reasoning-sft/Project-Reasoning-SFT-LLM/LLaMA-Factory

echo "======================================"
echo "Random 15K Qwen2.5 SFT - RESUME"
echo "======================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
python --version

echo
echo "=== GPU ==="
nvidia-smi

echo
echo "=== Resuming from checkpoint-200 ==="

FORCE_TORCHRUN=1 llamafactory-cli train yamls/random_15k.yaml

echo
echo "=== Job finished ==="
date
