import json
import os
import re

from dotenv import load_dotenv
load_dotenv()

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
)

from llm import embed

ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
KEY = os.environ["AZURE_SEARCH_ADMIN_KEY"]
INDEX = os.environ["AZURE_SEARCH_INDEX"]

FORM_CODE = re.compile(r"com\s*[./|]+\s*[a-z]\s*-?\s*\d+(?:\s*-\s*[a-z])?", re.I)


def normalize_codes(text):
    """'Com./R-14' becomes 'comr14' so keyword search treats it as one word."""
    return FORM_CODE.sub(lambda m: re.sub(r"[^a-z0-9]", "", m.group(0).lower()), text)


chunks = []
with open("data/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Embed every chunk (in batches)
vectors = []
for start in range(0, len(chunks), 16):
    batch = [c["text"] for c in chunks[start:start + 16]]
    vectors.extend(embed(batch))
    print("Embedded", len(vectors), "of", len(chunks))
dims = len(vectors[0])
print("Embedding size:", dims)

# Create the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="text", type=SearchFieldDataType.String),
    SearchableField(name="text_search", type=SearchFieldDataType.String),
    SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
    SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True),
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=dims,
        vector_search_profile_name="vec-profile",
    ),
]
vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
    profiles=[VectorSearchProfile(name="vec-profile", algorithm_configuration_name="hnsw")],
)
index = SearchIndex(name=INDEX, fields=fields, vector_search=vector_search)
SearchIndexClient(ENDPOINT, AzureKeyCredential(KEY)).create_or_update_index(index)
print("Index ready:", INDEX)

# Upload the chunks
docs = [{
    "id": c["id"],
    "text": c["text"],
    "text_search": normalize_codes(c["text"]),
    "source": c["source"],
    "page": c["page"],
    "embedding": v,
} for c, v in zip(chunks, vectors)]

client = SearchClient(ENDPOINT, INDEX, AzureKeyCredential(KEY))
for start in range(0, len(docs), 100):
    client.upload_documents(docs[start:start + 100])
print("Uploaded", len(docs), "chunks")