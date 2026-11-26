import json
from pathlib import Path

from selfrag_prompt import selfrag_answer

EVAL = Path("data/eval_set.jsonl")
OUT = Path("data/results_prompt_selfrag.jsonl")

rows = []
with open(EVAL, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

done = set()
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["id"])

with open(OUT, "a", encoding="utf-8") as out:
    for r in rows:
        if r["id"] in done:
            continue
        t = selfrag_answer(r["question"])
        out.write(json.dumps({
            "id": r["id"],
            "question": r["question"],
            "label": r["label"],
            "gold_answer": r["answer"],
            "gold_chunk": r["chunk_id"],
            "retrieved": t.get("retrieved"),
            "relevant": t.get("relevant"),
            "draft": t.get("draft"),
            "support": t.get("support"),
            "model_answer": t["final"],
        }, ensure_ascii=False) + "\n")
        out.flush()
        print(r["id"], "done")

print("Saved to", OUT)