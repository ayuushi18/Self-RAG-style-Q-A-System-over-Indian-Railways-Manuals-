import json
import random
from pathlib import Path

from llm import ask
from sft_prompts import (
    retrieve_prompt, relevance_prompt, generate_prompt, support_prompt,
)

LABELS = Path("data/train_labels.jsonl")
CHUNKS = Path("data/chunks.jsonl")
OUT_TRAIN = Path("data/sft_train.jsonl")
OUT_VAL = Path("data/sft_val.jsonl")


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


chunk_text = {c["id"]: c["text"] for c in read_rows(CHUNKS)}
rows = read_rows(LABELS)

# Drop rows damaged by the content filter (empty answers)
rows = [r for r in rows
        if all((e.get("answer") or "").strip() for e in r["support_examples"])]
print("Usable labeled questions:", len(rows))

random.seed(3)

# Questions that do NOT need the manuals (for the RETRIEVE task)
reply = ask([{"role": "user", "content": (
    "Write 60 different short messages a user might send to a chatbot that do NOT need "
    "any document to answer: greetings, thanks, simple maths, general knowledge, writing help, "
    "jokes. They must not be about Indian Railways rules. "
    "Return only a JSON list of strings."
)}]) or ""
try:
    no_questions = json.loads(reply[reply.find("["): reply.rfind("]") + 1])
    no_questions = [q for q in no_questions if isinstance(q, str) and q.strip()]
except Exception:
    no_questions = []
    print("WARNING: could not read the list of no-retrieval questions")
print("No-retrieval questions:", len(no_questions))

# Hold out 10% of questions for validation (kept apart from training)
val_questions = set(random.sample([r["question"] for r in rows], len(rows) // 10))
if no_questions:
    val_questions |= set(random.sample(no_questions, max(1, len(no_questions) // 10)))

examples = []   # (question, example)


def add(question, user, assistant):
    examples.append((question, {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}))


# RETRIEVE
for r in random.sample(rows, min(150, len(rows))):
    add(r["question"], retrieve_prompt(r["question"]), "Yes")
for q in no_questions:
    add(q, retrieve_prompt(q), "No")

# RELEVANCE (balanced)
pos, neg = [], []
for r in rows:
    for cid, flag in zip(r["retrieved"], r["relevance"]):
        (pos if flag else neg).append((r["question"], chunk_text[cid]))
n = min(600, len(pos), len(neg))
for q, text in random.sample(pos, n):
    add(q, relevance_prompt(q, text), "Relevant")
for q, text in random.sample(neg, n):
    add(q, relevance_prompt(q, text), "Irrelevant")

# ANSWER (only good, fully supported gold answers)
answer_pool = []
for r in rows:
    if r["kind"] != "answerable":
        continue
    gold = [e for e in r["support_examples"] if e["origin"] == "gold" and e["label"] == "fully"]
    if gold:
        answer_pool.append((r["question"], r["gold_answer"],
                            [chunk_text[c] for c in gold[0]["passages"]]))
for q, ans, texts in random.sample(answer_pool, min(400, len(answer_pool))):
    add(q, generate_prompt(q, texts), ans)

# SUPPORT (Supported vs Unsupported, balanced by downsampling)
supported, unsupported = [], []
for r in rows:
    for e in r["support_examples"]:
        item = (r["question"], e["answer"], [chunk_text[c] for c in e["passages"]])
        (supported if e["label"] == "fully" else unsupported).append(item)
for q, ans, texts in random.sample(supported, min(250, len(supported))):
    add(q, support_prompt(q, ans, texts), "Supported")
for q, ans, texts in unsupported:
    add(q, support_prompt(q, ans, texts), "Unsupported")

# Split and save
random.shuffle(examples)
train = [ex for q, ex in examples if q not in val_questions]
val = [ex for q, ex in examples if q in val_questions]

for path, data in ((OUT_TRAIN, train), (OUT_VAL, val)):
    with open(path, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def task_counts(data):
    counts = {}
    for ex in data:
        task = ex["messages"][0]["content"].split("\n")[0]
        counts[task] = counts.get(task, 0) + 1
    return counts


print("\nTrain examples:", len(train), task_counts(train))
print("Validation examples:", len(val), task_counts(val))
print("Saved to", OUT_TRAIN, "and", OUT_VAL)