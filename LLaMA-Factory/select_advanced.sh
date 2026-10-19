#!/bin/bash

#SBATCH --job-name=select-advanced15k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=ece-6524-spring2026
#SBATCH --output=advanced_selection_%j.out

module load Miniconda3

ENV=/home/jacobdavis404/.conda/envs/reasoning-sft
export PATH="$ENV/bin:$PATH"

cd /home/jacobdavis404/reasoning-sft/Project-Reasoning-SFT-LLM/LLaMA-Factory

python select_advanced_15k.py
