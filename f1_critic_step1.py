import json
from pathlib import Path

from selfrag_prompt import generate

RES = Path("data/results_finetuned_selfrag.jsonl")
RET = Path("data/eval_retrieved.json")
OUT = Path("data/hybrid_drafts.json")

# Text of every retrieved chunk, per question
with open(RET, encoding="utf-8") as f:
    passage_text = {
        r["id"]: {p["id"]: p["text"] for p in r["passages"]}
        for r in json.load(f)
    }

rows = []
with open(RES, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

out = []
for i, r in enumerate(rows, start=1):
    relevant_ids = (r["relevant"] or [])[:3]
    texts = [passage_text[r["id"]][cid] for cid in relevant_ids]

    draft = None
    if texts:
        draft = generate(r["question"], [{"text": t} for t in texts])

    out.append({
        "id": r["id"],
        "question": r["question"],
        "label": r["label"],
        "gold_answer": r["gold_answer"],
        "gold_chunk": r["gold_chunk"],
        "retrieved": r["retrieved"],
        "relevant": r["relevant"],
        "texts": texts,
        "draft": draft,
    })
    print(i, "done")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("Saved", len(out), "items to", OUT)