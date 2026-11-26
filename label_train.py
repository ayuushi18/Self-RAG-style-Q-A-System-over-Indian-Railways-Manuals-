import json
import random
import re
from pathlib import Path

from hybrid import hybrid_search, chunk_by_id
from selfrag_prompt import judge_relevance, generate, check_support

TRAIN_Q = Path("data/train_questions.jsonl")
EVAL = Path("data/eval_set.jsonl")
OUT = Path("data/train_labels.jsonl")
HOW_MANY = 1000   # questions to label this run; test with 5, then set to 1000


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def read_rows(path):
    rows = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


eval_word_sets = [words(r["question"]) for r in read_rows(EVAL)]
questions = read_rows(TRAIN_Q)

# Remove training questions that are too similar to a test question
clean = []
for q in questions:
    w = words(q["question"])
    if any(jaccard(w, e) >= 0.6 for e in eval_word_sets):
        continue
    clean.append(q)
print("Training questions:", len(questions), "-> after test-set check:", len(clean))

done = {r["question"] for r in read_rows(OUT)}
random.seed(5)
processed = len(done)

with open(OUT, "a", encoding="utf-8") as out:
    for q in clean:
        if processed >= HOW_MANY:
            break
        if q["question"] in done:
            continue

        question = q["question"]
        src = chunk_by_id[q["chunk_id"]]

        results = hybrid_search(question, k=5)
        flags = judge_relevance(question, results)
        relevant = [c for c, f in zip(results, flags) if f]

        examples = []

        if q["kind"] == "answerable":
            # Good answer, checked against its source chunk
            context = [src] + [c for c in relevant if c["id"] != src["id"]]
            context = context[:3]
            label = check_support(question, q["answer"], context)
            examples.append({
                "origin": "gold", "answer": q["answer"],
                "passages": [c["id"] for c in context], "label": label,
            })

            # Bad answer: written from passages that are not the source chunk
            if random.random() < 0.5:
                wrong = [c for c in results if c["id"] != src["id"]][:3]
                if wrong:
                    draft = generate(question, wrong)
                    label = check_support(question, draft, wrong)
                    examples.append({
                        "origin": "wrong_passages", "answer": draft,
                        "passages": [c["id"] for c in wrong], "label": label,
                    })
        else:
            # Decoy-style question: write a draft and see if it is supported
            context = (relevant or results)[:3]
            draft = generate(question, context)
            label = check_support(question, draft, context)
            examples.append({
                "origin": "decoy_draft", "answer": draft,
                "passages": [c["id"] for c in context], "label": label,
            })

        out.write(json.dumps({
            "question": question,
            "kind": q["kind"],
            "source_chunk": q["chunk_id"],
            "gold_answer": q["answer"],
            "retrieved": [c["id"] for c in results],
            "relevance": flags,
            "support_examples": examples,
        }, ensure_ascii=False) + "\n")
        out.flush()
        processed += 1
        print(processed, "-", question[:70])

# Summary of what we collected
rows = read_rows(OUT)
support = {}
rel_true = rel_total = 0
for r in rows:
    for e in r["support_examples"]:
        support[e["label"]] = support.get(e["label"], 0) + 1
    rel_true += sum(r["relevance"])
    rel_total += len(r["relevance"])
print("\nQuestions labeled:", len(rows))
print("Support labels:", support)
print(f"Chunks marked relevant: {rel_true}/{rel_total}")