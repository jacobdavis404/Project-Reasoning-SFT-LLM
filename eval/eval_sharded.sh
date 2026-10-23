#!/bin/bash
# Evaluate a model on all 8 benchmarks as many small shard jobs (default 30 problems = 240 generations),
# so long-answer models never lose a benchmark to a time limit. Run with bash (not sbatch):
#
#   bash eval_sharded.sh <model> <tag> <a100|h200|b200> [shard_size]     (b200: run from an OWL terminal)
#   bash eval_sharded.sh Qwen/Qwen2.5-3B-Instruct baseline a100
#   bash eval_sharded.sh $LF_DIR/saves/qwen25_3b_instruct/hybrid_15k_a100 hybrid a100
#
# Re-running the same command only submits shards that have no result yet (safe to repeat after failures).
# Optional: BENCHES="aime amc" to limit benchmarks, TIME=4:00:00 per shard.
# Results: eval/outputs_<tag>/   Score: python score_results.py tag=outputs_<tag> ...
set -e
source "$HOME/Project-Reasoning-SFT-LLM/env.sh"
MODEL="${1%/}"; TAG="$2"; GPU="$3"; SIZE="${4:-30}"
EXP=""
case "$GPU" in
  a100) PART=a100_normal_q; QOS=tc_a100_normal_short ;;
  h200) PART=h200_normal_q; QOS=tc_h200_normal_short ;;
  b200) PART=b200_normal_q; QOS=owl_b200_normal_short; EXP=",EVAL_PY_ENV=$HOME/.conda/envs/reasoning-sft-b200" ;;
  a30)  PART=a30_normal_q;  QOS=fal_a30_normal_short;  EXP=",GPU_MEM_UTIL=0.80" ;;
  l40s) PART=l40s_normal_q; QOS=fal_l40s_normal_short ;;
  *) sed -n 2,13p "$0"; exit 1 ;;
esac
[ -n "$MODEL" ] && [ -n "$TAG" ] || { sed -n 2,13p "$0"; exit 1; }
BENCHES="${BENCHES:-aime amc cn_math_2024 gpqa kaoyan minerva math olympiadbench}"
OUTPUT_DIR="$EVAL_DIR/outputs_$TAG"; mkdir -p "$OUTPUT_DIR/logs"
MNAME=$(echo "$MODEL" | awk -F/ '{n=NF; s=$n; if(n>1) s=$(n-1)"/"s; if(n>2) s=$(n-2)"/"s; print s}')
sub=0; skip=0
for B in $BENCHES; do
  N=$(grep -c . "$EVAL_DIR/data/$B/test.jsonl")
  for ((S=0; S<N; S+=SIZE)); do
    E=$(( S+SIZE < N ? S+SIZE : N ))
    [ -f "$OUTPUT_DIR/$MNAME/$B/test_qwen-instruct_t0.6_k8_s${S}_e${E}.jsonl" ] && { skip=$((skip+1)); continue; }
    sbatch -A "$ACCOUNT" -p "$PART" --qos="$QOS" --time="${TIME:-4:00:00}" \
      --job-name="ev-$TAG-$B-$S" --output="$OUTPUT_DIR/logs/%x-%j.out" \
      --export=ALL,MODEL="$MODEL",B="$B",S="$S",E="$E",OUTPUT_DIR="$OUTPUT_DIR"$EXP \
      "$EVAL_DIR/eval_shard_job.sh" >/dev/null
    sub=$((sub+1))
  done
done
echo "$TAG on $GPU: submitted $sub shard jobs, $skip already done  (watch: squeue -u \$USER -h | grep -c ev-$TAG)"
