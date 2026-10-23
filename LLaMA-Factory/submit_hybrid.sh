#!/bin/bash
# Submit the curated 3-epoch training as a chain of resumable jobs on one GPU type.
#   bash submit_hybrid.sh h200 2      # Tinkercliffs H200   (run from a Tinkercliffs terminal)
#   bash submit_hybrid.sh a100 3      # Tinkercliffs A100   (run from a Tinkercliffs terminal)
#   bash submit_hybrid.sh b200 2      # Owl B200            (run from an OWL terminal)
# Optional env: TIME=10:00:00 (default 23:50:00), QOS=..., SUFFIX=_10h (separate folder + job name)
# Jobs with the same name never run at the same time (singleton) and each resumes from the
# latest checkpoint in saves/qwen25_3b_instruct/hybrid_15k_<gpu><suffix>.
set -e
source "$HOME/Project-Reasoning-SFT-LLM/env.sh"
GPU="$1"; N="${2:-2}"; EXP=""
case "$GPU" in
  a100) PART=a100_normal_q; QOS=${QOS:-tc_a100_normal_short} ;;
  h200) PART=h200_normal_q; QOS=${QOS:-tc_h200_normal_short} ;;
  b200) PART=b200_normal_q; QOS=${QOS:-owl_b200_normal_short}
        EXP=",TRAIN_PY_ENV=$HOME/.conda/envs/reasoning-sft-b200,LF_OVERRIDES=flash_attn=sdpa" ;;
  *) echo "usage: bash submit_hybrid.sh a100|h200|b200 [num_chained_jobs]"; exit 1 ;;
esac
OUT="saves/qwen25_3b_instruct/hybrid_15k_$GPU${SUFFIX:-}"
NAME="sft-hybrid-$GPU${SUFFIX:-}"
mkdir -p "$LF_DIR/logs"
DEP="--dependency=singleton"; IDS=""
for ((i=1; i<=N; i++)); do
  J=$(sbatch --parsable -A "$ACCOUNT" -p "$PART" --qos="$QOS" --time=${TIME:-23:50:00} $DEP \
      --job-name="$NAME" --output="$LF_DIR/logs/%x-%j.out" \
      --export=ALL,OUT="$OUT"$EXP "$LF_DIR/train_hybrid.sh")
  IDS="$IDS $J"; DEP="--dependency=afterany:$J,singleton"
done
echo "$NAME -> $OUT : chained jobs$IDS"
