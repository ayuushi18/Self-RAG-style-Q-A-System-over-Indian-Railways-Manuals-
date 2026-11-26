import json
import re
from pathlib import Path

ANSWERABLE = Path("data/reviewed_questions.jsonl")
NOT_ANSWERABLE = Path("data/not_answerable_checked.jsonl")
OUT = Path("data/eval_set.jsonl")


def read_rows(path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def clean_question(q):
    """Remove a leading number like '30  ' or '30. '."""
    return re.sub(r"^\s*\d+\s*[\.\):\-]?\s*", "", q).strip()


rows = []

for r in read_rows(ANSWERABLE):
    if r.get("dropped"):
        continue
    rows.append({
        "question": clean_question(r["question"]),
        "answer": r["answer"],
        "label": "answerable",
        "chunk_id": r["chunk_id"],
    })

for r in read_rows(NOT_ANSWERABLE):
    if r.get("dropped"):
        continue
    rows.append({
        "question": clean_question(r["question"]),
        "answer": None,
        "label": "not_answerable",
        "chunk_id": None,
    })

# Remove duplicate questions and give each one an id
seen = set()
final = []
for r in rows:
    if r["question"] in seen:
        continue
    seen.add(r["question"])
    r["id"] = f"q{len(final) + 1:03d}"
    final.append(r)

with open(OUT, "w", encoding="utf-8") as f:
    for r in final:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

answerable = sum(r["label"] == "answerable" for r in final)
print("Answerable:    ", answerable)
print("Not answerable:", len(final) - answerable)
print("Saved to", OUT)

print("\nSample not-answerable questions:")
shown = 0
for r in final:
    if r["label"] == "not_answerable" and shown < 3:
        print(" -", r["question"])
        shown += 1