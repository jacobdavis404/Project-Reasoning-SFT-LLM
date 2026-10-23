#!/bin/bash
# One eval shard (benchmark slice [S, E)). Submitted by eval_sharded.sh; don't sbatch by hand.
# Generation settings are identical to eval_single.sh (leaderboard setup).
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1
set -e
source "$HOME/Project-Reasoning-SFT-LLM/env.sh"
PY_ENV="${EVAL_PY_ENV:-$PY_ENV}"             # B200 shards use the cu128 env
module load CUDA/12.6.0 2>/dev/null || true
export PATH="$PY_ENV/bin:$PATH"
echo "host: $(hostname) | python: $(which python) | model: $MODEL | $B [$S,$E)"
cd "$EVAL_DIR"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python eval.py \
  --model_name_or_path "$MODEL" \
  --data_name "$B" \
  --prompt_type "qwen-instruct" \
  --temperature 0.6 \
  --top_p 0.95 \
  --start_idx "$S" \
  --end_idx "$E" \
  --n_sampling 8 \
  --k 1 \
  --split "test" \
  --max_tokens 32768 \
  --seed 0 \
  --surround_with_messages \
  --output_dir "$OUTPUT_DIR" \
  --completions_save_dir "$OUTPUT_DIR/completions" \
  --gpu_memory_utilization "${GPU_MEM_UTIL:-0.85}"
