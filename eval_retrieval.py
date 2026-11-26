import json
import re

from hybrid import hybrid_search
from azure_search import hybrid_search_azure


def neighbours(chunk_id):
    m = re.match(r"(.+)-(\d+)$", chunk_id)
    source, n = m.group(1), int(m.group(2))
    return {f"{source}-{n + d}" for d in (-1, 0, 1)}


rows = []
with open("data/eval_set.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["label"] == "answerable":
            rows.append(r)

for name, search in (("Old (Chroma + BM25)", hybrid_search),
                     ("Azure AI Search", hybrid_search_azure)):
    exact = near = 0
    for r in rows:
        ids = [c["id"] for c in search(r["question"], k=5)]
        if r["chunk_id"] in ids:
            exact += 1
        if neighbours(r["chunk_id"]) & set(ids):
            near += 1
    print(f"{name}: right chunk in top 5: {exact}/{len(rows)}, "
          f"right chunk or a neighbour: {near}/{len(rows)}")