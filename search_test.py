import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_collection("railways")

PREFIX = "Represent this sentence for searching relevant passages: "

while True:
    question = input("\nAsk a question (or type q to quit): ")
    if question.strip().lower() == "q":
        break
    q_vec = model.encode([PREFIX + question], normalize_embeddings=True)
    results = collection.query(query_embeddings=q_vec.tolist(), n_results=3)
    for i in range(3):
        meta = results["metadatas"][0][i]
        print(f"\n--- Result {i + 1} ({meta['source']}, page {meta['page']}) ---")
        print(results["documents"][0][i])