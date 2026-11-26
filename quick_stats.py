import json
from pathlib import Path

RESULTS = Path("data/results_normal_rag.jsonl")

rows = []
with open(RESULTS, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

answerable = [r for r in rows if r["label"] == "answerable"]
unanswerable = [r for r in rows if r["label"] == "not_answerable"]

print("Total results:   ", len(rows), "(expected 159)")
print("Answerable:      ", len(answerable))
print("Not answerable:  ", len(unanswerable))

# 1. Did the search find the right chunk?
hits = sum(r["gold_chunk"] in r["retrieved"] for r in answerable)
print(f"\nSearch found the right chunk: {hits}/{len(answerable)}"
      f" ({100 * hits / max(len(answerable), 1):.0f}%)")

# 2. On not-answerable questions, did the model admit it can't answer?
PHRASES = [
    "not mention", "does not", "doesn't", "do not", "no information",
    "not specified", "not provided", "not stated", "not state", "cannot",
    "can't", "unable", "not contain", "not included", "not given",
    "not covered", "not available",
]


def admits(answer):
    a = answer.lower()
    return any(p in a for p in PHRASES)


admitted = [r for r in unanswerable if admits(r["model_answer"])]
print(f"\nNot-answerable questions where the model said it can't answer: "
      f"{len(admitted)}/{len(unanswerable)}"
      f" ({100 * len(admitted) / max(len(unanswerable), 1):.0f}%)")

print("\nExamples where it did NOT admit (probably made up an answer):")
shown = 0
for r in unanswerable:
    if not admits(r["model_answer"]) and shown < 5:
        print("\nQ:", r["question"])
        print("A:", r["model_answer"][:300])
        shown += 1