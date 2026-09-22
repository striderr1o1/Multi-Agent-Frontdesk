import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
from pinecone import AsyncPinecone, ServerlessSpec
from utils.exceptions import IngestionError
from services.supabase_db_functions import (
    get_pinecone_id_from_supabase,
    insert_ingestion_into_supabase,
)
from .embedding_config import get_google_embeddings, get_openrouter_embeddings
load_dotenv()

class Ingestion:
    def __init__(self, supabase_client=None, user_id=None): #configuration
        # the authenticated client comes down from routes/ingestion.py so the
        # ingestion record is written as the signed-in business, not the admin key
        self.supabase_client = supabase_client
        self.user_id = user_id
        self.filepath = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap = 200)
        ##self.embedding = OllamaEmbeddings(
        ##    model=os.environ.get("EMBEDDING_MODEL")
        ##)
        # AsyncPinecone is an async context manager, so it can't be built once
        # here in a sync __init__ - each pinecone call opens one per operation,
        # the same way retrieval.py does.
        self.api_key = os.environ.get('PINECONE_API_KEY')
        self.index_name = os.environ.get('PINECONE_INDEX_NAME')
        # _create_index has to be awaited, so it moved out of __init__ into
        # ingest_document; this flag keeps it to one check per object the way
        # constructing it used to
        self._index_ready = False

    async def ingest_document(self, filepath, namespace):
         #main function
        await self._create_index()
        self.filepath = filepath
        doc = self._load_document()
        chunks = self._create_chunks(doc)
        str_chunks, metadatas = self._convert_doc_chunks_to_str(chunks)
        embeddings_list = await self._create_embeddings_from_chunks(str_chunks)
        vectors_list, vector_ids = self._preparing_ingestions(embeddings_list, str_chunks, metadatas)
        #send to database with record name
        await self._store_in_vectordb(vectors_list, namespace)
        await self._record_ingestion(vector_ids)
        return

    async def _record_ingestion(self, vector_ids): #write the ingestion row in supabase
        # no client means the caller didn't pass one (e.g. an internal call with no
        # signed-in user) - the pinecone upsert is still the point, so don't block it
        if self.supabase_client is None or not self.user_id:
            return
        try:
            pc_id = await get_pinecone_id_from_supabase(self.supabase_client, self.user_id)
            if not pc_id:
                raise IngestionError(f'No pinecone_data_table row for user {self.user_id}')
            # source_name is the uploaded filename, not the temp path it was copied to
            source_name = os.path.basename(self.filepath) if self.filepath else None
            vector_ids_json = {"vector_ids_list": vector_ids}
            return await insert_ingestion_into_supabase(
                self.supabase_client, self.user_id, pc_id, source_name, vector_ids_json
            )
        except IngestionError:
            raise
        except Exception as e:
            raise IngestionError(f'Error in ingestion.py _record_ingestion(). Details: {e}')

    async def delete_ingestion_source(self, record_ids, namespace_name): #drop one source's vectors
        try:
            # pinecone caps delete-by-id at 1000 ids per call, so a large pdf's
            # chunk list has to go in batches the same way the upsert does
            batch_size = 1000
            async with AsyncPinecone(api_key=self.api_key) as pc:
                index = await pc.index(host=os.environ.get('INDEX_URL_PINECONE'))
                # the index client holds its own connection pool - closing pc
                # does not close it, so it needs its own `async with`
                async with index:
                    for i in range(0, len(record_ids), batch_size):
                        await index.delete(
                            ids = record_ids[i: i+batch_size],
                            namespace = namespace_name
                        )

            return len(record_ids)
        except Exception as e:
            raise IngestionError(f'Ingestion.py -> Error deleting from vector store, delete_ingestion_source(). Details: {e}')

    def _load_document(self): #load the document
        try:
            self.loader = PyPDFLoader(self.filepath)
            pdf_docs = self.loader.load()
            return pdf_docs
        except Exception:
            raise IngestionError('Unable to load document in ingestion.py _load_document()')

    def _create_chunks(self, document_text):# create the chunks
        try:
            chunks = self.text_splitter.split_documents(document_text)
        # need to only convert page content to string
            return chunks
        except Exception:
            raise IngestionError('Chunking error in ingestion.py _create_chunks()')

    def _convert_doc_chunks_to_str(self, chunks): #convert chunks to string type chunks
        try:
            str_chunks = []
            metadatas = []
            for chunk in chunks:
                string_chunk = str(chunk.page_content)
                str_chunks.append(string_chunk)
                #adding string chunk in metadata
                chunk.metadata["chunk_text"] = string_chunk
                metadatas.append(chunk.metadata)
                print(len(str_chunks))
    
            return str_chunks, metadatas
        except Exception:
            raise IngestionError('Error in ingestion.py _convert_doc_chunks_to_str()')


    async def _create_embeddings_from_chunks(self, text_chunks): # create embeddings out of string chunks
            # Previous Ollama embeddings (uncomment self.embedding in __init__ to switch back):
            # embeddings = self.embedding.embed_documents(text_chunks)
            
        try:
            BATCH_SIZE = 99
            embeddings = []
            for i in range(0, len(text_chunks), BATCH_SIZE):
                chunks = text_chunks[i:i+BATCH_SIZE]
                # embeddings_from_chunk = get_google_embeddings(chunks)
                embeddings_from_chunk = await get_openrouter_embeddings(chunks)
                print("Embedding chunks: ", len(chunks))
                embeddings = embeddings + embeddings_from_chunk
                #time.sleep(70)
            return embeddings
        except Exception as e:
            raise IngestionError(f'Error in creating embeddings, ingestion.py,_create_embeddings_from_chunks(). Details: {e}')

    async def _create_index(self): #initialize index if not already
        if self._index_ready:
            return
        try:
            async with AsyncPinecone(api_key=self.api_key) as pc:
                if not await pc.has_index(self.index_name):
                    await pc.create_index(
                        name=self.index_name,
                        vector_type="dense",
                        dimension=1024, #according to model
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        ),
                        deletion_protection="disabled",
                        tags={
                            "environment": "development"
                        }
                    )
            self._index_ready = True
        except Exception:
            raise IngestionError('Error in creating index pinecone, ingestion.py _create_index()')

    def _preparing_ingestions(self, embeddings_list, string_chunks, metadatas): #prepare vectors for ingestion
        try:
            length_of_embeddings = len(embeddings_list)
            length_of_chunkslist = len(string_chunks)
            vectors_list = []
            vector_ids = []
            if length_of_chunkslist == length_of_embeddings:
                for i in range(0, length_of_chunkslist):
                    vector = {
                        'id': f'{metadatas[i]["source"]}#chunk: {i}',
                        "values": embeddings_list[i],
                        "metadata": metadatas[i]
                            }
                    vectors_list.append(vector)
                    vector_ids.append(vector['id'])

            return vectors_list, vector_ids
        except Exception:
            raise IngestionError(f'Error in ingestion.py _preparing_ingestions()')

        
    async def _store_in_vectordb(self, vectors_list, namespace_name): #upsert in pinecone vector store
        try:
            batch_size = 100
            async with AsyncPinecone(api_key=self.api_key) as pc:
                index = await pc.index(host=os.environ.get('INDEX_URL_PINECONE'))
                # every batch goes through the one index client, so the
                # connection pool is opened and torn down once per document
                async with index:
                    for i in range(0, len(vectors_list), batch_size):
                        await index.upsert(
                            namespace = namespace_name,
                            vectors = vectors_list[i: i+batch_size]
                        )

            return
        except Exception as e:
            raise IngestionError(f'Ingestion.py -> Error in storing in vector store, _store_in_vectordb(). Details: {e}')

