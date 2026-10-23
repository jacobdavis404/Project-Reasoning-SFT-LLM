"""Score eval runs straight from the result .jsonl files (works for sharded and unsharded runs).

  cd ~/Project-Reasoning-SFT-LLM/eval
  python score_results.py baseline=outputs_baseline hybrid=outputs_hybrid random=outputs_random

Multiple dirs can be merged under one tag with commas: hybrid=outputs_hybrid,outputs_hybrid_h200
Per benchmark it reports pass@8 ("Acc" in the course logs: solved if any of 8 samples is correct),
pass@1 (mean correctness over the 8 samples), coverage (problems scored / total), plus answer
statistics: % of generations containing \\boxed{} and median generation length in characters.
Writes results.md next to where it is run.
"""
import glob
import json
import os
import re
import statistics
import sys

BENCHES = ["aime", "math", "cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]
NAMES = {"aime": "AIME24", "math": "MATH500", "cn_math_2024": "CN_MATH_24", "kaoyan": "KAOYAN",
         "amc": "AMC23", "minerva": "MINERVA", "olympiadbench": "OLYMPIAD", "gpqa": "GPQA"}
AVG_SET = ["cn_math_2024", "kaoyan", "amc", "minerva", "olympiadbench", "gpqa"]
EVAL_DIR = os.environ.get("EVAL_DIR", os.path.dirname(os.path.abspath(__file__)))
SE = re.compile(r"_s(\d+)_e(\d+)\.jsonl$")


def total(b):
    p = os.path.join(EVAL_DIR, "data", b, "test.jsonl")
    try:
        return sum(1 for l in open(p) if l.strip())
    except OSError:
        return None


def score(dirs, b):
    rows = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", b, "*.jsonl"), recursive=True):
            m = SE.search(f)
            if not m or "/completions/" in f:
                continue
            s = int(m.group(1))
            for i, line in enumerate(open(f, encoding="utf-8")):
                if line.strip():
                    rows[s + i] = json.loads(line)   # problem index -> result (dedupes overlaps)
    if not rows:
        return None
    r = list(rows.values())
    acc = sum(bool(x.get("is_correct")) for x in r) / len(r)
    p1 = statistics.mean(statistics.mean(x["answers_correctness"]) if x.get("answers_correctness")
                         else float(bool(x.get("is_correct"))) for x in r)
    gens = [g for x in r for g in x.get("generated_responses", [])]
    boxed = sum("\\boxed" in g for g in gens) / len(gens) if gens else None
    med = int(statistics.median(len(g) for g in gens)) if gens else None
    return {"acc": acc, "p1": p1, "n": len(r), "N": total(b), "boxed": boxed, "med": med}


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    runs = {}
    for arg in sys.argv[1:]:
        tag, ds = arg.split("=", 1)
        runs[tag] = {b: score(ds.split(","), b) for b in BENCHES}

    def cell(res, key, fmt):
        if res is None:
            return "–"
        v = fmt.format(res[key]) if res[key] is not None else "–"
        if key in ("acc", "p1") and res["N"] and res["n"] < res["N"]:
            v += f" ({res['n']}/{res['N']})"
        return v

    out = []
    for key, title, fmt in [("acc", "pass@8 (leaderboard metric)", "{:.3f}"),
                            ("p1", "pass@1 (mean over 8 samples)", "{:.3f}"),
                            ("boxed", "% generations with a \\boxed{} answer", "{:.0%}"),
                            ("med", "median generation length (chars)", "{:,}")]:
        out += [f"**{title}**", "",
                "| Model | " + " | ".join(NAMES[b] for b in BENCHES) + (" | AVG |" if key in ("acc", "p1") else " |"),
                "|" + "---|" * (len(BENCHES) + (2 if key in ("acc", "p1") else 1))]
        for tag, res in runs.items():
            row = f"| {tag} | " + " | ".join(cell(res[b], key, fmt) for b in BENCHES)
            if key in ("acc", "p1"):
                ok = [res[b] for b in AVG_SET if res[b] and (not res[b]["N"] or res[b]["n"] == res[b]["N"])]
                row += f" | {sum(x[key] for x in ok) / len(ok):.3f} |" if len(ok) == len(AVG_SET) else " | – |"
            else:
                row += " |"
            out.append(row)
        out.append("")
    out.append("(x/N) = partial coverage. AVG = mean of " + ", ".join(NAMES[b] for b in AVG_SET)
               + " (leaderboard set minus AIME25), only when all six are complete.")
    text = "\n".join(out)
    print(text)
    open("results.md", "w").write(text + "\n")
    print("\n-> results.md")


if __name__ == "__main__":
    main()
