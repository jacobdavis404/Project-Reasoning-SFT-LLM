# Curated (hybrid) 15K run, from scratch on Tinkercliffs (H200 + A100)

Account `tml_2026`. Everything runs from the Tinkercliffs terminal
(https://ood.arc.vt.edu/pun/sys/shell/ssh/tinkercliffs2.arc.vt.edu, VT VPN on).
Files added to the team repo for this run:

| File | Purpose |
|---|---|
| `env.sh` | Paths + account, sourced by every script |
| `setup_env.sh` | One-time conda env `reasoning-sft` (pinned versions that work on ARC) |
| `LLaMA-Factory/get_hybrid_15k.sh` | Downloads the curated set from HF and registers `acereason_hybrid_15k` |
| `LLaMA-Factory/yamls/hybrid_15k.yaml` | Training config, same hyperparameters as `random_15k.yaml` |
| `LLaMA-Factory/submit_hybrid.sh`, `train_hybrid.sh` | Chained, auto-resuming training jobs |
| `eval/eval_sharded.sh`, `eval_shard_job.sh` | Eval split into small jobs (identical settings to `eval_single.sh`) |
| `eval/score_results.py` | pass@8 / pass@1 tables from the result files |

## 1. Put the repo in place (5 min)
Upload this zip in OnDemand (Files -> Home Directory -> Upload), then:
```bash
cd ~ && unzip -q Project-Reasoning-SFT-LLM.zip && ls ~/Project-Reasoning-SFT-LLM
```

## 2. Environment (one time, 20-40 min, login node)
```bash
module load Miniconda3
bash ~/Project-Reasoning-SFT-LLM/setup_env.sh
```
The CHECK at the end must show a version for every package (no `MISSING`) and
`grader check latex2sympy(1/2) = 1/2`. `GPU visible here: False` is normal on the login node.

## 3. Hugging Face login + downloads (5 min)
```bash
~/.conda/envs/reasoning-sft/bin/huggingface-cli login      # WRITE token; answer n to git credential
~/.conda/envs/reasoning-sft/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-3B-Instruct')"
bash ~/Project-Reasoning-SFT-LLM/LLaMA-Factory/get_hybrid_15k.sh
```
Expect `rows=15000 empty=0 ... think=15000` and `registered acereason_hybrid_15k`.

## 4. Training: submit on both GPU types (3 epochs, ~5,625 steps)
```bash
cd ~/Project-Reasoning-SFT-LLM/LLaMA-Factory
bash submit_hybrid.sh h200 2      # ~25-30h on H200 -> 2 chained jobs
bash submit_hybrid.sh a100 3      # ~48h on A100   -> 3 chained jobs
squeue -u $USER
```
The two chains write to separate folders (`saves/qwen25_3b_instruct/hybrid_15k_h200`, `..._a100`).
**As soon as one starts running (`R`), cancel the other chain** so you don't train twice:
`scancel -n sft-hybrid-a100` or `scancel -n sft-hybrid-h200`.
Each job resumes automatically from the latest 200-step checkpoint; the log shows
`Resuming training from .../checkpoint-XXXX`.

Check progress (step count and time left):
```bash
grep -o "[0-9]*/[0-9]* \[[^]]*\]" ~/Project-Reasoning-SFT-LLM/LLaMA-Factory/logs/sft-hybrid-*.out | tail -1
```

## 5. Baseline eval (any time, short answers -> fast)
```bash
cd ~/Project-Reasoning-SFT-LLM/eval
bash eval_sharded.sh Qwen/Qwen2.5-3B-Instruct baseline a100 100
```

## 6. Curated model eval (after training finishes)
```bash
cd ~/Project-Reasoning-SFT-LLM/eval
bash eval_sharded.sh ~/Project-Reasoning-SFT-LLM/LLaMA-Factory/saves/qwen25_3b_instruct/hybrid_15k_h200 hybrid a100
bash eval_sharded.sh ~/Project-Reasoning-SFT-LLM/LLaMA-Factory/saves/qwen25_3b_instruct/hybrid_15k_h200 hybrid h200
```
Use `hybrid_15k_a100` instead if the A100 chain is the one that trained. Run only ONE of the two lines
(the script skips shards that are finished, not ones still queued). If that queue is slow,
`scancel -u $USER -t PENDING` and run the other line. Re-running a line resubmits only missing/failed shards.

## 7. Score + upload
```bash
cd ~/Project-Reasoning-SFT-LLM/eval
~/.conda/envs/reasoning-sft/bin/python score_results.py baseline=outputs_baseline hybrid=outputs_hybrid
~/.conda/envs/reasoning-sft/bin/python - <<'PY'
from huggingface_hub import HfApi
a = HfApi(); r = "enriquemaldo/qwen25-3b-hybrid15k-3ep"
a.create_repo(r, exist_ok=True)
a.upload_folder(folder_path="/home/enriquem/Project-Reasoning-SFT-LLM/LLaMA-Factory/saves/qwen25_3b_instruct/hybrid_15k_h200",
                repo_id=r, ignore_patterns=["checkpoint-*", "checkpoint-*/**"])
print("https://huggingface.co/" + r)
PY
```
`results.md` holds the tables for the report.
