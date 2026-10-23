# Sourced by every script below. Nothing to edit if the repo is at ~/Project-Reasoning-SFT-LLM.
export REPO="$HOME/Project-Reasoning-SFT-LLM"
export LF_DIR="$REPO/LLaMA-Factory"
export EVAL_DIR="$REPO/eval"
export ENV_NAME="reasoning-sft"                       # same env name the team's scripts use
export PY_ENV="$HOME/.conda/envs/$ENV_NAME"
export ACCOUNT="tml_2026"
export HF_USER="enriquemaldo"
export HYBRID_HF_REPO="enriquemaldo/acereason-hybrid15k"
export PYTHONNOUSERSITE=1                             # never use stray packages in ~/.local
[ -n "$OUT" ] && export HF_DATASETS_CACHE="$HOME/.cache/hf_datasets/$(basename "$OUT")"
true
