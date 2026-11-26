import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


results = {r["id"]: r for r in read_rows(Path("data/results_finetuned_selfrag.jsonl"))}
judgments = read_rows(Path("data/judgments_finetuned_selfrag.jsonl"))

causes = Counter()
examples = defaultdict(list)

for j in judgments:
    if j["label"] != "answerable" or j["verdict"] == "CORRECT":
        continue
    r = results[j["id"]]
    gold = r["gold_chunk"]

    if gold not in r["retrieved"]:
        cause = "1. search missed the right chunk"
    elif gold not in (r["relevant"] or []):
        cause = "2. relevance check dropped the right chunk"
    elif r["support"] == "no":
        cause = "3. support check rejected the answer"
    else:
        cause = "4. answered, but the answer was wrong (answer quality)"

    causes[cause] += 1
    if len(examples[cause]) < 2:
        examples[cause].append(j["question"])

print("Answerable questions the fine-tuned version did not get right:",
      sum(causes.values()))
for cause in sorted(causes):
    print(f"\n{cause}: {causes[cause]}")
    for q in examples[cause]:
        print("   -", q[:100])