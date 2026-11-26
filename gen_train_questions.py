import json
import random
import re
from pathlib import Path

from llm import ask

CHUNKS = Path("data/chunks.jsonl")
EVAL = Path("data/eval_set.jsonl")
OUT = Path("data/train_questions.jsonl")
SKIPPED = Path("data/train_skipped.txt")
HOW_MANY = 250   # chunks to process this run; test with 5, then raise to 250

SYSTEM = """You help build training data for a question answering system about Indian Railways rules.
Read the passage and write:
(a) TWO different questions that this passage fully answers. Give each a short answer (1-2 sentences) using only facts in the passage.
(b) ONE question that sounds like it belongs to the same topic and that a real passenger or railway staff member could ask, but whose specific answer (a number, amount, time period or name) is NOT given in this passage.
Every question must make sense on its own. Never say "according to the passage" or "in this text".
If the passage is only a table, a list of names or a contents list, return {"skip": true}.
Return JSON only:
{"answerable": [{"question": "...", "answer": "..."}, {"question": "...", "answer": "..."}], "unanswerable": [{"question": "..."}]}"""


def is_mostly_text(text):
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) >= 0.75


def neighbours(chunk_id):
    """railway1-107 -> {railway1-106, railway1-107, railway1-108}"""
    m = re.match(r"(.+)-(\d+)$", chunk_id)
    source, n = m.group(1), int(m.group(2))
    return {f"{source}-{n + d}" for d in (-1, 0, 1)}


def parse_json(text):
    try:
        return json.loads(text[text.find("{"): text.rfind("}") + 1])
    except Exception:
        return None


chunks = []
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Chunks that must NOT be used for training
excluded = set()
with open(EVAL, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("chunk_id"):
            excluded |= neighbours(r["chunk_id"])
print("Chunks excluded to protect the test set:", len(excluded))

candidates = [c for c in chunks if c["id"] not in excluded and is_mostly_text(c["text"])]
random.seed(11)
random.shuffle(candidates)

done = set()
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["chunk_id"])
if SKIPPED.exists():
    done |= set(SKIPPED.read_text(encoding="utf-8").split())

processed = len(done)
with open(OUT, "a", encoding="utf-8") as out, open(SKIPPED, "a", encoding="utf-8") as skip:
    for c in candidates:
        if processed >= HOW_MANY:
            break
        if c["id"] in done:
            continue

        reply = ask([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Passage:\n" + c["text"]},
        ])
        data = parse_json(reply or "")
        processed += 1

        if not data or data.get("skip"):
            skip.write(c["id"] + "\n")
            skip.flush()
            print("Skipped", c["id"])
            continue

        rows = []
        for q in data.get("answerable", []):
            rows.append({"chunk_id": c["id"], "kind": "answerable",
                         "question": q["question"], "answer": q["answer"]})
        for q in data.get("unanswerable", []):
            rows.append({"chunk_id": c["id"], "kind": "unanswerable",
                         "question": q["question"], "answer": None})
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        out.flush()
        print(c["id"], "->", len(rows), "questions")

print("Done. Chunks processed so far:", processed)