# Project Reasoning SFT LLM — Group 6

## Project Overview

This project investigates how training-data selection affects the reasoning performance of `Qwen/Qwen2.5-3B-Instruct`.

Starting from the AceReason-1.1-SFT 100K subset, Group 6 constructed two 15,000-example training datasets:

1. **Random 15K** — a uniformly sampled baseline dataset.
2. **Advanced / Curated 15K** — a deliberately selected dataset intended to improve the quality and diversity of the reasoning examples used for fine-tuning.

Both datasets were used to independently fine-tune `Qwen/Qwen2.5-3B-Instruct`.

The resulting models are compared against the original base model on eight reasoning benchmarks using the same evaluation settings.

---

## Dataset Selection

### Random 15K Baseline

The Random 15K dataset was created by uniformly sampling 15,000 examples from the AceReason-1.1-SFT 100K dataset.

A fixed random seed of `42` was used to make the selection reproducible.

This dataset serves as the baseline for measuring whether a more deliberate data-selection strategy improves downstream reasoning performance.

### Advanced / Curated 15K

The Advanced / Curated dataset contains 15,000 examples selected from the same AceReason-1.1-SFT 100K source dataset.

Unlike the Random 15K baseline, the curated dataset uses a deliberate selection strategy intended to prioritize stronger and more diverse reasoning examples.

Using the same number of examples in both datasets allows the evaluation to focus on the effect of **data selection strategy** rather than differences in training-set size.

The final Group 6 curated dataset is published on Hugging Face as:

`enriquemaldo/acereason-hybrid15k`

---

## Training Configuration

Both fine-tuned models begin independently from:

`Qwen/Qwen2.5-3B-Instruct`

The primary training configuration used for the project is:

- Training method: Full supervised fine-tuning
- Epochs: `3`
- Learning rate: `5e-6`
- Learning-rate scheduler: Cosine
- Warmup: `0`
- Per-device batch size: `1`
- Effective batch size: `8`
- Maximum sequence length: `16384`
- Precision: `bf16`
- DeepSpeed: ZeRO Stage 3

The Random 15K and Advanced / Curated 15K models were trained independently from the original base model rather than sequentially from one another.

---

## Evaluation Methodology

All models are evaluated on the same eight reasoning benchmarks:

- AIME 2024
- AMC 2023
- MATH-500
- Minerva Math
- OlympiadBench
- CN Math 2024
- Kaoyan
- GPQA-Diamond

For each benchmark, the model generates eight responses per problem.

The primary reported metric is **Pass@1**.

The evaluation scripts also report an `Acc` value, but this value represents whether at least one of the eight generated responses is correct and is therefore more optimistic than Pass@1. For consistency with the leaderboard requirements, Group 6 reports Pass@1.

All models are evaluated using the following settings:

- Temperature: `0.6`
- Top-p: `0.95`
- Samples per problem: `8`
- Reported metric: `Pass@1`
- Maximum generation length: `32768`
- Random seed: `0`
- System prompt: `Please reason step by step, and put your final answer within \boxed{}.`

---

## Evaluation Results Group 6

| Model                      | AIME 2024 | AMC 2023 | MATH-500 | Minerva Math | OlympiadBench | CN Math 2024 | Kaoyan | GPQA-Diamond |
| -------------------------- | --------- | -------- | -------- | ------------ | ------------- | ------------ | ------ | ------------ |
| Base Qwen2.5-3B-Instruct   | 0.062     | 0.419    | 0.650    | 0.308        | 0.281         | 0.108        | 0.225  | 0.301        |
| Random 15K SFT             | 0.0917    | 0.3531   | —        | —            | —             | 0.1042       | —      | —            |
| Advanced / Curated 15K SFT | 0.050     | 0.403    | 0.646    | 0.265        | 0.284         | 0.129        | 0.273  | 0.217        |

### Average Performance

| Model                      | Mean Pass@1 |
| -------------------------- | ----------- |
| Base Qwen2.5-3B-Instruct   | 0.294       |
| Random 15K SFT             | —           |
| Advanced / Curated 15K SFT | 0.283       |

The Random 15K evaluation is still in progress. The remaining benchmark scores and final mean Pass@1 will be added once those evaluations complete.

---

## Preliminary Results

The currently available results show that fine-tuning does not uniformly improve performance across every benchmark.

The Advanced / Curated model performs better than the base model on some benchmarks, including CN Math 2024 and Kaoyan, while performing below the base model on several others.

The Advanced / Curated model currently has a mean Pass@1 of `0.283`, compared with `0.294` for the base model.

This suggests that the effects of supervised fine-tuning are benchmark-dependent and that selecting higher-quality or more diverse examples does not necessarily guarantee an improvement in aggregate reasoning performance.

A complete comparison between the Random 15K and Advanced / Curated 15K strategies will be added after the remaining Random 15K evaluations finish.

---

## Hugging Face Artifacts

### Datasets

- Random 15K: [https://huggingface.co/datasets/jacobdavis404/AceReason-Random-15K](https://huggingface.co/datasets/jacobdavis404/AceReason-Random-15K)
- Advanced / Curated 15K: [https://huggingface.co/datasets/enriquemaldo/acereason-hybrid15k](https://huggingface.co/datasets/enriquemaldo/acereason-hybrid15k)

### Models

- Base model: [https://huggingface.co/Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- Random 15K SFT: [https://huggingface.co/jacobdavis404/Qwen2.5-3B-Instruct-Random15K-SFT](https://huggingface.co/jacobdavis404/Qwen2.5-3B-Instruct-Random15K-SFT)
- Advanced / Curated 15K SFT: [https://huggingface.co/enriquemaldo/qwen25-3b-hybrid15k-3ep](https://huggingface.co/enriquemaldo/qwen25-3b-hybrid15k-3ep)

---

## Reproducibility

The repository contains the scripts and configuration files used for:

- Random dataset selection
- Advanced / curated dataset selection
- Fine-tuning
- Evaluation
- Benchmark scoring

Large datasets and trained model weights are hosted on Hugging Face rather than committed directly to the GitHub repository.

All reported model comparisons use the same evaluation configuration to ensure that the Base, Random 15K, and Advanced / Curated 15K models are evaluated consistently.
