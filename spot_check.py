import json
import random
import shutil
from pathlib import Path

import re


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def best_sentences(text, answer, n=2):
    """Find the sentences in the chunk that share the most words with the answer."""
    a = words(answer)
    sentences = re.split(r"(?<=[.;:])\s+", text)
    ranked = sorted(sentences, key=lambda s: len(a & words(s)), reverse=True)
    return ranked[:n]

CHUNKS = Path("data/chunks.jsonl")
REVIEWED = Path("data/reviewed_questions.jsonl")
BACKUP = Path("data/reviewed_questions.backup.jsonl")
HOW_MANY = 30

shutil.copy(REVIEWED, BACKUP)  # safety copy

rows = []
with open(REVIEWED, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

chunks = {}
with open(CHUNKS, encoding="utf-8") as f:
    for line in f:
        c = json.loads(line)
        chunks[c["id"]] = c

random.seed(7)
candidates = [i for i, r in enumerate(rows) if not r.get("dropped")]
sample = random.sample(candidates, min(HOW_MANY, len(candidates)))

checked = 0
problems = 0
try:
    for i in sample:
        r = rows[i]
        print("=" * 70)
        print("CHUNK", r["chunk_id"], "(page", r["page"], ")")
        print(chunks[r["chunk_id"]]["text"])
        print("-" * 70)
        print("QUESTION:", r["question"])
        print("ANSWER:  ", r["answer"] or "(EMPTY - fix or drop)")
        print("\nCHUNK SAYS (best matching sentences):")
        for s in best_sentences(chunks[r["chunk_id"]]["text"], r["answer"]):
            print("  *", s[:300])
        choice = input("[k]eep / [e]dit / [d]rop / [q]uit: ").strip().lower()
        if choice == "q":
            break
        checked += 1
        if choice == "d":
            r["dropped"] = True
            problems += 1
        elif choice == "e":
            r["question"] = input("New question (Enter keeps old): ").strip() or r["question"]
            r["answer"] = input("New answer (Enter keeps old): ").strip() or r["answer"]
            problems += 1
        r["spot_checked"] = True
finally:
    with open(REVIEWED, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nChecked:", checked, " Problems found:", problems)