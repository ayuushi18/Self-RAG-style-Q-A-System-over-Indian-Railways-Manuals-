import json
import re
from pathlib import Path

from hybrid import hybrid_search, tokenize

IN = Path("data/not_answerable.txt")
OUT = Path("data/not_answerable_checked.jsonl")

questions = []
with open(IN, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(line.strip())

# Skip questions you already checked
done = set()
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["question"])


def best_sentences(text, question, n=2):
    """Pick the sentences that share the most words with the question."""
    q_words = set(tokenize(question))
    sentences = re.split(r"(?<=[.;:])\s+", text)
    ranked = sorted(
        sentences, key=lambda s: len(q_words & set(tokenize(s))), reverse=True
    )
    return ranked[:n]


with open(OUT, "a", encoding="utf-8") as out:
    for q in questions:
        if q in done:
            continue

        results = hybrid_search(q, k=3)
        print("=" * 70)
        print("QUESTION:", q)
        for i, c in enumerate(results, start=1):
            print(f"\n--- Result {i}: {c['id']} (page {c['page']}) ---")
            for s in best_sentences(c["text"], q):
                print("  *", s[:300])

        while True:
            choice = input(
                "\n[k]eep as not answerable / [d]rop / [f]ull text / [q]uit: "
            ).strip().lower()
            if choice == "f":
                for i, c in enumerate(results, start=1):
                    print(f"\n### Result {i}: {c['id']}\n{c['text']}")
                continue
            break

        if choice == "q":
            break

        row = {
            "question": q,
            "answer": None,
            "label": "not_answerable",
            "reviewed": True,
            "dropped": choice != "k",
        }
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()

print("Saved to", OUT)