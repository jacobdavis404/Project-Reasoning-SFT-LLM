"""Diversity + difficulty selection."""
# note that current code expects to be run like this:
# python select_data.py prepare --input acereason11_100k.jsonl --out work --tokenizer Qwen/Qwen2.5-3B --max-seq-len 8192 --device cuda
# this means that it expects acereason11_100k.jsonl to be in the same working directory
# playing around with seed and other values totally encouraged, you can check all the settings at the bottom
# running this will create a directory called work, after that if you want to generate some sample groups, run:
# python select_data.py select --work work --method hybrid
# for a hybrid method group, and then to get a random baseline run:
# python select_data.py select --work work --method random
# if things don't seem to work, it may have to do with library versioning, so you may need to set up a venv
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_id(prompt):
    # Preserve case and internal whitespace: they can be significant in code.
    return hashlib.sha256(prompt.strip().replace("\r\n", "\n").encode()).hexdigest()


def quotas(counts, total, power=1.0):
    """Capacity-constrained apportionment with exact total, deterministic ties."""
    counts = np.asarray(counts, dtype=int)
    if total < 0 or total > counts.sum() or np.any(counts < 0):
        raise ValueError("Insufficient eligible rows for requested sample size")
    result = np.zeros(len(counts), dtype=int)
    while result.sum() < total:
        available = counts - result
        active = available > 0
        weights = np.where(active, counts.astype(float) ** power, 0)
        ideal = (total - result.sum()) * weights / weights.sum()
        add = np.minimum(np.floor(ideal).astype(int), available)
        if not add.any():
            # Assign one to largest fractional share; repeat after capacity update.
            add[np.argmax(np.where(active, ideal, -1))] = 1
        result += add
    return result


def length_scores(rows):
    """Middle-high response-length preference, ranked separately by category."""
    from scipy.stats import rankdata
    scores = np.empty(len(rows))
    cats = np.array([r["category"] for r in rows])
    for cat in sorted(set(cats)):
        ix = np.flatnonzero(cats == cat)
        lengths = np.array([rows[i]["response_tokens"] for i in ix])
        rank = (rankdata(lengths, method="average") - .5) / len(ix)
        scores[ix] = np.exp(-.5 * ((rank - .65) / .25) ** 2)
    return scores


def prepare(a):
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pool.json").exists():
        raise ValueError("Use a new output directory to avoid stale features/scores")
    raw = read_jsonl(a.input)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if a.chat_template:
        tok.chat_template = Path(a.chat_template).read_text()
    if not tok.chat_template:
        raise ValueError("Provide --chat-template matching your SFT formatter")
    excluded = set()
    if a.exclude:
        excluded = {prompt_id(r[a.prompt_key]) for r in read_jsonl(a.exclude)}
    rng = np.random.default_rng(a.seed)
    rows, seen = [], set()
    dropped = dict(empty=0, excluded=0, overlong=0, duplicate=0)
    # Randomized order avoids favoring the first response for repeated prompts.
    for original_index in rng.permutation(len(raw)):
        r = raw[original_index]
        p, answer = r[a.prompt_key], r[a.response_key]
        if not isinstance(p, str) or not isinstance(answer, str):
            raise ValueError("Prompt and response columns must contain strings")
        if not p.strip() or not answer.strip():
            dropped["empty"] += 1
            continue
        uid = prompt_id(p)
        if uid in excluded:
            dropped["excluded"] += 1
            continue
        messages = [{"role": "user", "content": p},
                    {"role": "assistant", "content": answer}]
        n = len(tok.apply_chat_template(messages, tokenize=True,
                                       add_generation_prompt=False))
        if n > a.max_seq_len:
            dropped["overlong"] += 1
            continue
        if uid in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(uid)
        rows.append(dict(id=uid, original_index=int(original_index), prompt=p,
                         response=answer, messages=messages,
                         category=str(r.get(a.category_key, "unknown")),
                         source=str(r.get(a.source_key, "unknown")),
                         response_tokens=len(tok.encode(answer, add_special_tokens=False)),
                         total_tokens=n, original=r))
    if len(rows) < a.minimum_pool:
        raise ValueError(f"Only {len(rows)} eligible prompts; drops={dropped}. "
                         "Reconsider context budget; do not silently truncate solutions.")

    encoder = SentenceTransformer(a.embedding_model, device=a.device)
    # Chunk long prompts, then mean-pool chunk vectors: no silent tail truncation.
    chunk_size = min(encoder.max_seq_length - 8, a.chunk_tokens)
    if chunk_size <= 0:
        raise ValueError("Invalid encoder context length")
    embeddings = []
    for start in range(0, len(rows), 256):
        block = rows[start:start + 256]
        chunks, owners = [], []
        for j, row in enumerate(block):
            ids = encoder.tokenizer.encode(row["prompt"], add_special_tokens=False)
            for k in range(0, max(1, len(ids)), chunk_size):
                chunks.append(encoder.tokenizer.decode(ids[k:k + chunk_size]))
                owners.append(j)
        vectors = encoder.encode(chunks, batch_size=a.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)
        aggregate = np.zeros((len(block), vectors.shape[1]), dtype=np.float32)
        np.add.at(aggregate, owners, vectors)
        aggregate /= np.maximum(np.linalg.norm(aggregate, axis=1, keepdims=True), 1e-12)
        embeddings.append(aggregate)
        print(f"Embedded {start + len(block)}/{len(rows)}", flush=True)
    embeddings = np.concatenate(embeddings)
    cats = np.array([r["category"] for r in rows])
    for cat in sorted(set(cats)):
        ix = np.flatnonzero(cats == cat)
        k = min(a.clusters_per_category, len(ix))
        km = MiniBatchKMeans(n_clusters=k, random_state=a.seed,
                            n_init=10, batch_size=2048)
        labels = km.fit_predict(embeddings[ix])
        for i, label in zip(ix, labels):
            rows[i]["cluster"] = int(label)
    write_jsonl(out / "pool.json", rows)
    np.save(out / "embeddings.npy", embeddings)
    manifest = dict(arguments=vars(a), input_sha256=hashlib.sha256(
        Path(a.input).read_bytes()).hexdigest(), eligible=len(rows), dropped=dropped)
    (out / "prepare.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def difficulty(rows, scores_path):
    fallback = length_scores(rows)
    if not scores_path:
        return fallback
    records = read_jsonl(scores_path)
    by_id = {r["id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate score IDs")
    if set(by_id) - {r["id"] for r in rows}:
        raise ValueError("Scores contain IDs outside this pool")
    result = fallback.copy()
    for i, row in enumerate(rows):
        s = by_id.get(row["id"])
        if not s:
            continue
        n, c = s["trials"], s["correct"]
        if not isinstance(n, int) or not isinstance(c, int) or not 0 <= c <= n:
            raise ValueError("Require integer 0 <= correct <= trials")
        if not n:
            continue
        p = (c + 1) / (n + 2)  # Beta(1,1) smoothing, not pass@K.
        learnable = np.exp(-.5 * ((p - .35) / .22) ** 2)
        reliability = n / (n + 4)
        result[i] = reliability * learnable + (1 - reliability) * fallback[i]
    return result


def choose(rows, embeddings, n, method, seed, scores_path=None, alpha=.5):
    rng = np.random.default_rng(seed)
    if n > len(rows) or n <= 0:
        raise ValueError("Invalid requested sample size")
    if method == "random":
        return rng.choice(len(rows), n, replace=False).tolist()
    score = difficulty(rows, scores_path)
    if method == "difficulty":
        return np.lexsort((rng.random(len(rows)), -score))[:n].tolist()
    cats = np.array([r["category"] for r in rows])
    categories, counts = np.unique(cats, return_counts=True)
    selected = []
    for cat, cat_n in zip(categories, quotas(counts, n)):
        ix = np.flatnonzero(cats == cat)
        labels = np.array([rows[i]["cluster"] for i in ix])
        clusters, sizes = np.unique(labels, return_counts=True)
        for cluster, count in zip(clusters, quotas(sizes, int(cat_n), alpha)):
            candidates = ix[labels == cluster]
            if method == "diversity":
                selected.extend(rng.choice(candidates, count, replace=False).tolist())
                continue
            # Greedy MMR inside a semantic cluster; never build an NxN matrix.
            x = embeddings[candidates]
            redundancy = np.zeros(len(candidates), dtype=float)
            used = np.zeros(len(candidates), dtype=bool)
            jitter = rng.uniform(0, 1e-8, len(candidates))
            for _ in range(count):
                utility = .8 * score[candidates] + .2 * (1 - redundancy) + jitter
                utility[used] = -np.inf
                j = int(np.argmax(utility))
                used[j] = True
                selected.append(int(candidates[j]))
                redundancy = np.maximum(redundancy, np.clip(x @ x[j], 0, 1))
    assert len(selected) == n and len(set(selected)) == n
    rng.shuffle(selected)
    return selected


def select(a):
    root = Path(a.work)
    rows = read_jsonl(root / "pool.json")
    embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
    if len(rows) != len(embeddings):
        raise ValueError("Misaligned embeddings")
    chosen = choose(rows, embeddings, a.n, a.method, a.seed, a.scores, a.alpha)
    selected = [rows[i] for i in chosen]
    output = root / f"{a.method}_{a.n}_seed{a.seed}.json"
    write_jsonl(output, selected)
    from collections import Counter
    report = dict(arguments=vars(a), count=len(selected),
                  total_tokens=sum(r["total_tokens"] for r in selected),
                  response_tokens=sum(r["response_tokens"] for r in selected),
                  categories=dict(Counter(r["category"] for r in selected)),
                  sources=dict(Counter(r["source"] for r in selected)),
                  covered_clusters=len({(r["category"], r["cluster"]) for r in selected}),
                  length_quantiles=np.quantile([r["response_tokens"] for r in selected],
                                                [0, .25, .5, .75, 1]).tolist())
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(output)
    print(json.dumps(report, indent=2))


def probe(a):
    """Generate K attempts; use an explicit trusted grading adapter."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    module, function = a.verifier.split(":")
    verify = getattr(importlib.import_module(module), function)
    rows = read_jsonl(Path(a.work) / "pool.json")
    if a.limit:
        ix = np.random.default_rng(a.seed).choice(len(rows), min(a.limit, len(rows)), False)
        rows = [rows[i] for i in ix]
    output = Path(a.output)
    if output.exists():
        raise ValueError("Use a new score output file (prevent mixed probe settings)")
    tok = AutoTokenizer.from_pretrained(a.model)
    if a.chat_template:
        tok.chat_template = Path(a.chat_template).read_text()
    if not tok.chat_template:
        raise ValueError("Provide the same --chat-template used in SFT")
    llm = LLM(model=a.model, max_model_len=a.context, seed=a.seed,
              tensor_parallel_size=a.tensor_parallel_size,
              gpu_memory_utilization=a.gpu_memory_utilization)
    params = SamplingParams(n=a.k, temperature=.7, top_p=.95,
                            max_tokens=a.max_new_tokens, seed=a.seed)
    output.with_suffix(".config.json").write_text(json.dumps(vars(a), indent=2))
    with output.open("w") as f:
        for start in range(0, len(rows), a.batch_size):
            block = rows[start:start + a.batch_size]
            prompts = [tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}], tokenize=False,
                add_generation_prompt=True) for r in block]
            if any(len(tok.encode(p, add_special_tokens=False)) + a.max_new_tokens > a.context
                   for p in prompts):
                raise ValueError("Probe context too short; increase --context")
            outputs = llm.generate(prompts, params)
            for row, result in zip(block, outputs):
                correct = trials = unknown = truncated = 0
                for attempt in result.outputs:
                    if attempt.finish_reason == "length":
                        truncated += 1
                        continue  # Censored, not evidence of intrinsic difficulty.
                    verdict = verify(row["original"], attempt.text)
                    if verdict is None:
                        unknown += 1
                    elif type(verdict) is bool:
                        trials += 1
                        correct += int(verdict)
                    else:
                        raise TypeError("Verifier must return bool or None")
                f.write(json.dumps(dict(id=row["id"], correct=correct, trials=trials,
                                        unknown=unknown, truncated=truncated)) + "\n")
            f.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--input", required=True, help="Exact assignment pool, JSONL")
    p.add_argument("--out", required=True)
    p.add_argument("--prompt-key", default="input")
    p.add_argument("--response-key", default="output")
    p.add_argument("--category-key", default="category")
    p.add_argument("--source-key", default="source")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-3B")
    p.add_argument("--chat-template")
    p.add_argument("--max-seq-len", type=int, required=True)
    p.add_argument("--minimum-pool", type=int, default=15000)
    p.add_argument("--exclude", help="Held-out prompts JSONL; same prompt key")
    p.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    p.add_argument("--device", default="cpu")
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--clusters-per-category", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("select")
    s.add_argument("--work", required=True)
    s.add_argument("--method", choices=["random", "difficulty", "diversity", "hybrid"],
                   default="hybrid")
    s.add_argument("--n", type=int, default=15000)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--alpha", type=float, default=.5)
    s.add_argument("--scores")
    p = sub.add_parser("probe")
    p.add_argument("--work", required=True)
    p.add_argument("--model", required=True, help="Exact checkpoint BEFORE fine-tuning")
    p.add_argument("--verifier", required=True, help="Python module:function")
    p.add_argument("--output", required=True)
    p.add_argument("--chat-template")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="0=all; pilot uses random subset")
    p.add_argument("--context", type=int, default=8192)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=.85)
    p.add_argument("--seed", type=int, default=42)
    a = parser.parse_args()
    {"prepare": prepare, "select": select, "probe": probe}[a.command](a)


if __name__ == "__main__":
    main()
