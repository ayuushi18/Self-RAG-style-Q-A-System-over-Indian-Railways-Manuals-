import json
from pathlib import Path

CHUNKS = Path("data/chunks.jsonl")
DRAFTS = Path("data/draft_questions.jsonl")
OUT = Path("data/reviewed_questions.jsonl")

chunks = {}
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        chunks[c["id"]] = c

drafts = []
with open(DRAFTS, encoding="utf-8") as f:
    for line in f:
        drafts.append(json.loads(line))

# Skip questions you already reviewed
done = set()
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["chunk_id"])

with open(OUT, "a", encoding="utf-8") as out:
    for d in drafts:
        if d["chunk_id"] in done:
            continue
        print("=" * 70)
        print("CHUNK", d["chunk_id"], "(page", d["page"], ")")
        print(chunks[d["chunk_id"]]["text"])
        print("-" * 70)
        print("QUESTION:", d["question"])
        print("ANSWER:  ", d["answer"])
        choice = input("[k]eep / [e]dit / [d]rop / [q]uit: ").strip().lower()
        if choice == "q":
            break
        d["dropped"] = (choice == "d")
        if choice == "e":
            d["question"] = input("New question (Enter keeps old): ").strip() or d["question"]
            d["answer"] = input("New answer (Enter keeps old): ").strip() or d["answer"]
        d["reviewed"] = True
        out.write(json.dumps(d, ensure_ascii=False) + "\n")
        out.flush()

print("Review saved to", OUT)