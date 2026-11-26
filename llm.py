import os
import time

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

_client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/") + "/openai/v1/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)
DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]


def ask(messages):
    """Send messages to the model. Retries a few times if Azure is busy."""
    for attempt in range(4):
        try:
            response = _client.chat.completions.create(
                model=DEPLOYMENT,
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 5 * (attempt + 1)
            print("Retry in", wait, "seconds ->", str(e)[:100])
            time.sleep(wait)
    return None

_resource_name = os.environ["AZURE_OPENAI_ENDPOINT"].split("//")[-1].split("/")[0].split(".")[0]

_embed_client = OpenAI(
    base_url=f"https://{_resource_name}.openai.azure.com/openai/v1/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
)


def embed(texts):
    """Turn a list of texts into a list of embedding vectors."""
    response = _embed_client.embeddings.create(
        model=os.environ["AZURE_EMBEDDING_DEPLOYMENT"],
        input=texts,
    )
    return [d.embedding for d in response.data]