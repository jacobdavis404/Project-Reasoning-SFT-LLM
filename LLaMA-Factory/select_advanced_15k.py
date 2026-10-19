import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


SEED = 42
TARGET_SIZE = 15000
N_CLUSTERS = 200

random.seed(SEED)
np.random.seed(SEED)

src = Path("../data_download/data/acereason11_100k.json")
dst = Path("data/acereason_advanced_15k.json")
stats_dst = Path("data/acereason_advanced_15k_stats.json")

print(f"Loading {src} ...")
with src.open("r") as f:
    data = json.load(f)

print(f"Original examples: {len(data)}")


# ---------------------------------------------------------
# 1. QUALITY FILTERING
# ---------------------------------------------------------
filtered = []

for i, ex in enumerate(data):
    instruction = str(ex.get("instruction", "")).strip()
    inp = str(ex.get("input", "")).strip()
    output = str(ex.get("output", "")).strip()

    if not instruction:
        continue

    if not output:
        continue

    # Remove obviously tiny responses that provide almost no reasoning.
    if len(output) < 100:
        continue

    item = dict(ex)
    item["_original_index"] = i
    item["_output_length"] = len(output)
    item["_selection_text"] = (instruction + "\n" + inp).strip()

    filtered.append(item)

print(f"After basic quality filtering: {len(filtered)}")


# ---------------------------------------------------------
# 2. DIFFICULTY / LENGTH FILTERING
#
# Output length is used as a simple proxy for reasoning depth.
# Remove extreme tails while retaining medium and harder examples.
# ---------------------------------------------------------
lengths = np.array([x["_output_length"] for x in filtered])

low_cutoff = np.percentile(lengths, 10)
high_cutoff = np.percentile(lengths, 98)

difficulty_filtered = [
    x for x in filtered
    if low_cutoff <= x["_output_length"] <= high_cutoff
]

print(f"10th percentile output length: {low_cutoff:.0f}")
print(f"98th percentile output length: {high_cutoff:.0f}")
print(f"After difficulty/length filtering: {len(difficulty_filtered)}")


# ---------------------------------------------------------
# 3. DIVERSITY REPRESENTATION
#
# TF-IDF gives us a lightweight semantic/topic representation
# without requiring another large neural model.
# ---------------------------------------------------------
texts = [x["_selection_text"] for x in difficulty_filtered]

print("Building TF-IDF features ...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
)

X = vectorizer.fit_transform(texts)

print(f"TF-IDF matrix: {X.shape}")


# ---------------------------------------------------------
# 4. CLUSTER INTO DIVERSE PROBLEM GROUPS
# ---------------------------------------------------------
print(f"Clustering into {N_CLUSTERS} groups ...")

kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    batch_size=2048,
    n_init=3,
)

labels = kmeans.fit_predict(X)

clusters = {}
for idx, label in enumerate(labels):
    clusters.setdefault(int(label), []).append(idx)

cluster_sizes = [len(v) for v in clusters.values()]

print(
    f"Cluster sizes: min={min(cluster_sizes)}, "
    f"median={np.median(cluster_sizes):.1f}, "
    f"max={max(cluster_sizes)}"
)


# ---------------------------------------------------------
# 5. BALANCED SELECTION
#
# Aim for equal coverage across clusters.
# Within each cluster favor moderately longer outputs,
# which acts as a difficulty signal.
# ---------------------------------------------------------
per_cluster = TARGET_SIZE // N_CLUSTERS
remainder = TARGET_SIZE % N_CLUSTERS

selected_indices = []
unused_indices = []

for cluster_id in sorted(clusters):
    members = clusters[cluster_id]

    # Sort by reasoning/output length, hardest first.
    members = sorted(
        members,
        key=lambda idx: difficulty_filtered[idx]["_output_length"],
        reverse=True,
    )

    n_take = per_cluster + (1 if cluster_id < remainder else 0)

    selected_indices.extend(members[:n_take])
    unused_indices.extend(members[n_take:])


# If small clusters prevent us from reaching exactly 15K,
# fill from the remaining candidate pool.
if len(selected_indices) < TARGET_SIZE:
    needed = TARGET_SIZE - len(selected_indices)

    unused_indices = sorted(
        unused_indices,
        key=lambda idx: difficulty_filtered[idx]["_output_length"],
        reverse=True,
    )

    selected_indices.extend(unused_indices[:needed])


# In the unlikely event equal allocation produced more than target.
selected_indices = selected_indices[:TARGET_SIZE]

selected = []

for idx in selected_indices:
    ex = dict(difficulty_filtered[idx])

    # Remove temporary analysis fields.
    ex.pop("_original_index", None)
    ex.pop("_output_length", None)
    ex.pop("_selection_text", None)

    selected.append(ex)

random.shuffle(selected)

assert len(selected) == TARGET_SIZE

print(f"Final selected examples: {len(selected)}")


# ---------------------------------------------------------
# SAVE DATASET
# ---------------------------------------------------------
dst.parent.mkdir(parents=True, exist_ok=True)

with dst.open("w") as f:
    json.dump(selected, f, ensure_ascii=False)

stats = {
    "seed": SEED,
    "source_examples": len(data),
    "after_quality_filter": len(filtered),
    "after_length_filter": len(difficulty_filtered),
    "target_examples": TARGET_SIZE,
    "num_clusters": N_CLUSTERS,
    "tfidf_max_features": 20000,
    "tfidf_ngram_range": [1, 2],
    "low_output_length_percentile": 10,
    "high_output_length_percentile": 98,
    "low_output_length_cutoff": float(low_cutoff),
    "high_output_length_cutoff": float(high_cutoff),
    "selection_strategy": (
        "Quality filtering + output-length difficulty filtering + "
        "TF-IDF MiniBatchKMeans diversity clustering + balanced "
        "cluster sampling favoring longer reasoning outputs."
    ),
}

with stats_dst.open("w") as f:
    json.dump(stats, f, indent=2)

print()
print(f"Dataset saved to: {dst}")
print(f"Selection metadata saved to: {stats_dst}")
