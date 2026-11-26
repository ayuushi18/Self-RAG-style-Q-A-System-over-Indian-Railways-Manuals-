import json
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Load the chunks
chunks = []
with open("data/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
texts = [c["text"] for c in chunks]

# Turn every chunk into numbers (takes a few minutes on a laptop)
embeddings = model.encode(
    texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
)

# Store them in a local vector database
client = chromadb.PersistentClient(path="data/chroma")
try:
    client.delete_collection("railways")   # so re-running starts fresh
except Exception:
    pass
collection = client.create_collection("railways", metadata={"hnsw:space": "cosine"})

collection.add(
    ids=[c["id"] for c in chunks],
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=[{"source": c["source"], "page": c["page"]} for c in chunks],
)
print("Stored", collection.count(), "chunks")