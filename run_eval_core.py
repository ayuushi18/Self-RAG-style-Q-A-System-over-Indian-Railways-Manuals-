import json
from pathlib import Path

from azure_search import hybrid_search_azure
from core import selfrag_events

EVAL = Path("data/eval_set.jsonl")
OUT = Path("data/results_azure_selfrag.jsonl")

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
    for n, r in enumerate(rows, start=1):
        if r["id"] in done:
            continue

        retrieved, relevant, final = [], [], {}
        for ev in selfrag_events(r["question"], hybrid_search_azure):
            if ev["step"] == "retrieved":
                retrieved = [c["id"] for c in ev["chunks"]]
            elif ev["step"] == "relevance":
                relevant = [c["id"] for c, f in zip(ev["chunks"], ev["flags"]) if f]
            elif ev["step"] == "final":
                final = ev

        out.write(json.dumps({
            "id": r["id"],
            "question": r["question"],
            "label": r["label"],
            "gold_answer": r["answer"],
            "gold_chunk": r["chunk_id"],
            "retrieved": retrieved,
            "relevant": relevant,
            "draft": final.get("draft"),
            "support": final.get("support"),
            "model_answer": final.get("answer", ""),
        }, ensure_ascii=False) + "\n")
        out.flush()
        print(n, "done")

print("Saved to", OUT)