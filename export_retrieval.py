import json
from pathlib import Path

from hybrid import hybrid_search

EVAL = Path("data/eval_set.jsonl")
OUT = Path("data/eval_retrieved.json")

rows = []
with open(EVAL, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        results = hybrid_search(r["question"], k=5)
        rows.append({
            "id": r["id"],
            "question": r["question"],
            "label": r["label"],
            "gold_answer": r["answer"],
            "gold_chunk": r["chunk_id"],
            "passages": [{"id": c["id"], "text": c["text"]} for c in results],
        })

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False)

print("Saved", len(rows), "questions to", OUT)