import json
import re

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

PREFIX = "Represent this sentence for searching relevant passages: "
STOPWORDS = {
    "what", "is", "the", "a", "an", "of", "for", "to", "in", "on", "and",
    "or", "do", "does", "how", "much", "i", "if", "my", "can", "are", "be",
    "used", "form",
}

FORM_CODE = re.compile(r"com\s*[./|]+\s*[a-z]\s*-?\s*\d+(?:\s*-\s*[a-z])?")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
collection = chromadb.PersistentClient(path="data/chroma").get_collection("railways")

chunks = []
with open("data/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
chunk_by_id = {c["id"]: c for c in chunks}


def tokenize(text):
    text = text.lower()
    # Turn each form code into one word, e.g. "com./r-14" -> "comr14"
    codes = [re.sub(r"[^a-z0-9]", "", m) for m in FORM_CODE.findall(text)]
    words = re.findall(r"[a-z0-9]+", text)
    return [w for w in words if w not in STOPWORDS] + codes


# Build the keyword index once, when the file loads
bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])


def vector_rank(question, n=20):
    """Chunk ids ranked by meaning."""
    q_vec = model.encode([PREFIX + question], normalize_embeddings=True)
    result = collection.query(query_embeddings=q_vec.tolist(), n_results=n)
    return result["ids"][0]


def keyword_rank(question, n=20):
    """Chunk ids ranked by matching words."""
    scores = bm25.get_scores(tokenize(question))
    top = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:n]
    return [chunks[i]["id"] for i in top]


def hybrid_search(question, k=5):
    """Combine both rankings. Returns the best k chunks."""
    points = {}
    for ranking in (vector_rank(question), keyword_rank(question)):
        for rank, chunk_id in enumerate(ranking):
            points[chunk_id] = points.get(chunk_id, 0) + 1 / (60 + rank + 1)
    best = sorted(points, key=points.get, reverse=True)[:k]
    return [chunk_by_id[c] for c in best]


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type q to quit): ")
        if question.strip().lower() == "q":
            break
        for i, c in enumerate(hybrid_search(question, k=5), start=1):
            has_code = "R-14" in c["text"]
            print(f"\n--- Result {i}: {c['id']} (page {c['page']}) contains 'R-14': {has_code} ---")
            print(c["text"][:300], "...")