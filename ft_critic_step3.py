import json
from pathlib import Path

NOT_FOUND = "I cannot find this in the documents."

with open("data/hybrid_drafts.json", encoding="utf-8") as f:
    drafts = json.load(f)

support = {}
with open("data/hybrid_support.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            support[r["id"]] = r["support"]

out_path = Path("data/results_hybrid_selfrag.jsonl")
with open(out_path, "w", encoding="utf-8") as out:
    for d in drafts:
        label = support.get(d["id"], "none")
        final = d["draft"] if (label == "fully" and d["draft"]) else NOT_FOUND
        out.write(json.dumps({
            "id": d["id"],
            "question": d["question"],
            "label": d["label"],
            "gold_answer": d["gold_answer"],
            "gold_chunk": d["gold_chunk"],
            "retrieved": d["retrieved"],
            "relevant": d["relevant"],
            "draft": d["draft"],
            "support": label,
            "model_answer": final,
        }, ensure_ascii=False) + "\n")

print("Saved", len(drafts), "results to", out_path)