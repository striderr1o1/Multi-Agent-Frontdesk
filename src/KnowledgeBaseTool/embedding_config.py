from dotenv import load_dotenv
import httpx
load_dotenv()
import os
import requests
from google import genai
from google.genai import types

client = genai.Client()  # reads GOOGLE_API_KEY from env

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBEDDING_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
# Pinecone index is 1024-dim. This model natively outputs 2048 but is trained
# with Matryoshka Representation Learning, so truncating to the first N values
# is valid and keeps the vector usable. See model card on Hugging Face.
EMBEDDING_DIMENSIONS = 1024


def get_google_embeddings(chunks):
    result = client.models.embed_content(
          model="gemini-embedding-001",
          contents=chunks,
          config=types.EmbedContentConfig(output_dimensionality=1024),
    )
    embeddings = [e.values for e in result.embeddings]  # one vector per chunk
    return embeddings


async def get_openrouter_embeddings(chunks):
    # Embed a list of text chunks via OpenRouter, returns one vector per chunk.
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_EMBEDDING_MODEL,
        # this model is multimodal (VL), so each input is a content block list
        "input": [
            {"content": [{"type": "text", "text": chunk}]} for chunk in chunks
        ],
        "encoding_format": "float",
        # best-effort hint; ignored by providers that don't support it, which is
        # why we also truncate below to guarantee the final size.
        "dimensions": EMBEDDING_DIMENSIONS,
    }

#    response = requests.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload)
    async with httpx.AsyncClient() as client:
        response = await client.post(OPENROUTER_EMBEDDINGS_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()["data"]
    # Truncate each vector to 1024 (Matryoshka) so it matches the Pinecone index.
    # Cast to float: OpenRouter can return whole numbers as ints, but Pinecone
    # requires every value to be a float.
    embeddings = [
        [float(v) for v in item["embedding"][:EMBEDDING_DIMENSIONS]] for item in data
    ]
    return embeddings


# bottle neck: according to claude, if chunks are greater than 100 than google embedding model will reject it so go with batch size
# also need to look up normalization
