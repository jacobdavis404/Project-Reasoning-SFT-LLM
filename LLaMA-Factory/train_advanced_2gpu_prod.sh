#!/bin/bash

#SBATCH --job-name=qwen-adv-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --partition=h200_normal_q
#SBATCH --account=ece-6524-spring2026
#SBATCH --gres=gpu:2
#SBATCH --output=advanced15k_2gpu_%j.out

module load CUDA/12.6.0

ENV=/home/jacobdavis404/.conda/envs/reasoning-sft
export PATH="$ENV/bin:$PATH"

cd /home/jacobdavis404/reasoning-sft/Project-Reasoning-SFT-LLM/LLaMA-Factory

nvidia-smi

FORCE_TORCHRUN=1 llamafactory-cli train yamls/advanced_15k_2gpu_prod.yaml
