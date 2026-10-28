#!/bin/bash
#SBATCH --job-name=rnd-cn_math_2024
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:15:00
#SBATCH --partition=a100_normal_q
#SBATCH --account=ece-6524-spring2026
#SBATCH --gres=gpu:1
#SBATCH --output=random_cn_math_2024_%j.out

module load CUDA/12.6.0

cd /home/jacobdavis404/reasoning-sft/Project-Reasoning-SFT-LLM/eval

/home/jacobdavis404/.conda/envs/reasoning-eval/bin/python eval.py   --model_name_or_path "/home/jacobdavis404/reasoning-sft/Project-Reasoning-SFT-LLM/LLaMA-Factory/saves/qwen25_3b_instruct/random_15k"   --data_name "cn_math_2024"   --prompt_type "qwen-instruct"   --temperature 0.6   --top_p 0.95   --start_idx 0   --end_idx -1   --n_sampling 8   --k 1   --split "test"   --max_tokens 32768   --seed 0   --surround_with_messages   --output_dir "./outputs_random15k_separate"   --completions_save_dir "./outputs_random15k_separate/completions"   --gpu_memory_utilization 0.96
