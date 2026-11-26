import os
import re

from dotenv import load_dotenv
load_dotenv()

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from llm import embed

FORM_CODE = re.compile(r"com\s*[./|]+\s*[a-z]\s*-?\s*\d+(?:\s*-\s*[a-z])?", re.I)


def normalize_codes(text):
    return FORM_CODE.sub(lambda m: re.sub(r"[^a-z0-9]", "", m.group(0).lower()), text)


_client = SearchClient(
    os.environ["AZURE_SEARCH_ENDPOINT"],
    os.environ["AZURE_SEARCH_INDEX"],
    AzureKeyCredential(os.environ["AZURE_SEARCH_ADMIN_KEY"]),
)


def hybrid_search_azure(question, k=5):
    """Hybrid search: meaning (vector) and keywords together, merged by Azure."""
    vector = embed([question])[0]
    results = _client.search(
        search_text=normalize_codes(question),
        search_fields=["text_search"],
        vector_queries=[VectorizedQuery(
            vector=vector, k_nearest_neighbors=50, fields="embedding",
        )],
        top=k,
        select=["id", "text", "source", "page"],
    )
    return [{"id": r["id"], "text": r["text"], "source": r["source"], "page": r["page"]}
            for r in results]