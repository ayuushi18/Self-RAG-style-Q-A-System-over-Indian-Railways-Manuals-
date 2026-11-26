import json
from pathlib import Path

CHUNKS = Path("data/chunks.jsonl")
DRAFTS = Path("data/draft_questions.jsonl")
REVIEWED = Path("data/reviewed_questions.jsonl")
OUT_DIR = Path("data/batches")
BATCH_SIZE = 10

PROMPT = """You are a strict reviewer of a question-answering test set about Indian Railways rules.

I will give you items. Each item has an ID, a CHUNK (the source text), a QUESTION and an ANSWER.

For each item, choose one verdict:
- KEEP: the answer is correct and fully stated in the chunk, and the question makes sense on its own (it never says "this passage" and needs no missing context).
- EDIT: the question is good, but the answer is slightly wrong, incomplete or too long, or the question is ambiguous. Give a corrected question and/or answer.
- DROP: the answer is not in the chunk, or it is wrong, or the question only makes sense with the chunk in front of you, or it is trivial.

Rules:
- Use ONLY the chunk. Do not use outside knowledge.
- If the chunk is cut off before the answer, answer DROP.
- Be strict. When unsure, choose EDIT or DROP, not KEEP.
- Any corrected answer must use only facts stated in the chunk.

Output a JSON list only, with no other text. One object per item:
{"id": "...", "verdict": "KEEP" or "EDIT" or "DROP", "new_question": null or "...", "new_answer": null or "...", "reason": "one short sentence"}

Here are the items:
"""


def read_rows(path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


chunks = {c["id"]: c for c in read_rows(CHUNKS)}
done = {r["chunk_id"] for r in read_rows(REVIEWED)}
todo = [d for d in read_rows(DRAFTS) if d["chunk_id"] not in done]

OUT_DIR.mkdir(parents=True, exist_ok=True)
for old in OUT_DIR.glob("batch_*.txt"):
    old.unlink()

count = 0
for start in range(0, len(todo), BATCH_SIZE):
    items = todo[start:start + BATCH_SIZE]
    text = PROMPT
    for n, d in enumerate(items, start=1):
        text += (
            f"\nITEM {n}\n"
            f"ID: {d['chunk_id']}\n"
            f"CHUNK: {chunks[d['chunk_id']]['text']}\n"
            f"QUESTION: {d['question']}\n"
            f"ANSWER: {d['answer']}\n"
        )
    count += 1
    (OUT_DIR / f"batch_{count:02d}.txt").write_text(text, encoding="utf-8")

print(len(todo), "drafts ->", count, "batch files in", OUT_DIR)