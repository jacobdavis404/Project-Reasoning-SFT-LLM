# Curated (hybrid) 15K run: Qwen2.5-3B-Instruct

**Model:** https://huggingface.co/enriquemaldo/qwen25-3b-hybrid15k-3ep
**Training data:** https://huggingface.co/datasets/enriquemaldo/acereason-hybrid15k (15,000 samples, LLaMA-Factory format)

## Data
15,000 samples from the 100K AceReason-1.1-SFT subset (10,776 math / 4,224 code), grouped into 128 topic
clusters and restricted to a middle length band (2,705-8,134 tokens per sample, median 5,426).
All samples contain a complete `<think>` trace; 10,779 end in a `\boxed{}` answer.
(Selection script: see the teammate who built `hybrid_15000_seed42.json`.)

## Training (identical hyperparameters to the random-15K run)
| Setting | Value |
|---|---|
| Method | Full SFT, LLaMA-Factory, DeepSpeed ZeRO-3, bf16 |
| Epochs / LR / schedule / warmup | 3 / 5e-6 / cosine / 0 |
| Batch | 1 per GPU x 8 grad. accum. x 1 GPU = 8 (5,625 steps) |
| cutoff_len / template | 16,384 / qwen |
| Speed-only differences vs random | no CPU offload, FlashAttention-2, A100 |
| Hardware / time | 1x A100-80GB, ~15h (resumable 10h SLURM jobs, checkpoint every 100 steps) |
| Mean train loss per epoch | 0.612 / 0.537 / 0.503 (all steps: 0.5508) |

Config: `LLaMA-Factory/yamls/hybrid_15k.yaml` - SLURM: `LLaMA-Factory/submit_hybrid.sh`, `train_hybrid.sh` - data: `LLaMA-Factory/get_hybrid_15k.sh` - full steps: `STEPS.md`.
Loss curve: `training_loss.png`, per-step log: `trainer_log.jsonl`.

## Evaluation
Course `eval/eval.py` with the `eval_single.sh` (leaderboard) settings: temperature 0.6, top-p 0.95,
8 samples, max 32,768 tokens, seed 0. Benchmarks were split into 30-problem shards
(`eval/eval_sharded.sh`, `--start_idx/--end_idx`) so no job hit a time limit; scored with `eval/score_results.py`.
AVG = mean of CN-Math, Kaoyan, AMC, Minerva, OlympiadBench, GPQA (leaderboard set minus AIME25).

| pass@8 | AIME24 | MATH500 | CN_MATH_24 | KAOYAN | AMC23 | MINERVA | OLYMPIAD | GPQA | AVG |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B-Instruct (baseline) | 0.167 | 0.846 | 0.333 | 0.578 | 0.775 | 0.482 | 0.498 | 0.747 | 0.569 |
| **Curated hybrid 15K** | 0.200 | 0.876 | 0.433 | 0.603 | 0.750 | 0.474 | 0.551 | 0.727 | **0.590** |

| pass@1 | AIME24 | MATH500 | CN_MATH_24 | KAOYAN | AMC23 | MINERVA | OLYMPIAD | GPQA | AVG |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | 0.062 | 0.650 | 0.108 | 0.225 | 0.419 | 0.308 | 0.281 | 0.301 | 0.274 |
| **Curated hybrid 15K** | 0.050 | 0.646 | 0.129 | 0.273 | 0.403 | 0.265 | 0.284 | 0.217 | **0.262** |

Answer statistics (curated): median generation length 12.5K-39K characters vs 1.3K-2.4K for the baseline;
76-94% of generations end in a `\boxed{}` answer vs 95-100% for the baseline. Full tables: `results_tables.md`.

## Findings
1. Curated beats the base model on the leaderboard metric: pass@8 AVG 0.569 -> 0.590 (+0.021); largest gains on OlympiadBench, CN-Math, Kaoyan, MATH-500.
2. pass@1 is about flat (0.274 -> 0.262): more problems solved within 8 tries, single attempts not more reliable.
3. Answers are 10-20x longer and 7-24% never reach a `\boxed{}` answer, likely running into the 32K limit; every training example was <= 8K tokens.
4. GPQA (science) drops most on pass@1 (0.301 -> 0.217); the curated set contains no science data.
5. AIME/CN-Math (30 problems) and AMC (40) are noisy: one problem = 2.5-3.3 points.
