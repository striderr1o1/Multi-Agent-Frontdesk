import os
from dotenv import load_dotenv
from pinecone import AsyncPinecone, ServerlessSpec
from utils.exceptions import RetrievalError
from .embedding_config import get_openrouter_embeddings
load_dotenv()
class Retrieval:
    def __init__(self):
        # AsyncPinecone is an async context manager, so it can't be built once
        # here in a sync __init__ - _get_results opens one per call instead.
        self.api_key = os.environ.get('PINECONE_API_KEY')
        self.index_name = os.environ.get('PINECONE_INDEX_NAME')
        return

    async def retrieve(self, query, namespace):
        embeddings = await self._create_embeddings(query)
        results = await self._get_results(embeddings, namespace)
        return results
    async def _create_embeddings(self, query):
        try:
            # Embed the query with the SAME model used at ingestion so the
            # query vector lives in the same space as the stored chunks.
            # get_openrouter_embeddings takes a list and returns one vector
            # per item, so pass the query as a single-element list and unwrap.
            embeddings = await get_openrouter_embeddings([query])
            embeddings = embeddings[0]
            return embeddings
        except Exception:
            raise RetrievalError('retrieval.py: error in creating retrieval embeddings')
    async def _get_results(self, embeddings, namespace):
        try:
            async with AsyncPinecone(api_key=self.api_key) as pc:
                index = await pc.index(host=os.environ.get('INDEX_URL_PINECONE'))
                async with index:
                    results = await index.query(
                        namespace=namespace,
                        vector=embeddings,
                        top_k=5,
                        include_metadata=True,
                        include_values=False
                    )

            return results
        except Exception:
            raise RetrievalError('retrieval.py: Error in getting retrieval results')


