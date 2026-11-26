import json


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, (l for l in f if l.strip()))}


normal = read_rows("data/judgments_normal_rag.jsonl")
prompt = read_rows("data/judgments_prompt_selfrag.jsonl")

print("Questions where Normal RAG made up an answer but Self-RAG refused:\n")
for qid, n in normal.items():
    p = prompt.get(qid)
    if p and n["label"] == "not_answerable" \
            and n["verdict"] == "ANSWERED" and p["verdict"] == "ABSTAINED":
        print("-", n["question"])
        print("    Normal RAG said:", n["model_answer"][:150])
        print("    Self-RAG said:  ", p["model_answer"][:100])