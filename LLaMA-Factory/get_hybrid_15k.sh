#!/bin/bash
# Download the team's curated set from Hugging Face (already in LLaMA-Factory alpaca format with the
# eval system prompt) and register it as `acereason_hybrid_15k`. Run once on the login node.
set -e
source "$HOME/Project-Reasoning-SFT-LLM/env.sh"
cd "$LF_DIR"
"$PY_ENV/bin/python" - <<'PY'
import json, os, shutil, statistics
from huggingface_hub import hf_hub_download
p = hf_hub_download(os.environ["HYBRID_HF_REPO"], "data/train.json", repo_type="dataset")
shutil.copy(p, "data/acereason_hybrid_15k.json")
d = json.load(open("data/acereason_hybrid_15k.json"))
assert sorted(d[0]) == ["input", "instruction", "output", "system"], sorted(d[0])
info = json.load(open("data/dataset_info.json"))
info["acereason_hybrid_15k"] = {"file_name": "acereason_hybrid_15k.json",
    "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}}
json.dump(info, open("data/dataset_info.json", "w"), indent=2, ensure_ascii=False)
L = sorted(len(x["instruction"]) + len(x["output"]) for x in d)
print(f"rows={len(d)}  empty={sum(1 for x in d if not x['instruction'] or not x['output'])}  "
      f"boxed={sum(chr(92)+'boxed' in x['output'] for x in d)}  think={sum('</think>' in x['output'] for x in d)}  "
      f"chars p50={L[len(L)//2]} max={L[-1]}")
print("system prompt:", d[0]["system"])
print("registered acereason_hybrid_15k -> data/acereason_hybrid_15k.json")
PY
