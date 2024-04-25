import glob
from llama_index.core import Settings
from llmsherpa.readers.layout_reader import Block
from llama_index.core.schema import Document
from llmsherpa.readers import LayoutPDFReader
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext

import logging
import sys
import os

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# init graph db
# url = "neo4j://localhost:7687"
# username = "neo4j"
# password = "yasuotruong"
# database = "neo4j"
# DATA_PATH = "pdf"

# Settings.llm = Ollama(model="gemma:2b", temperature=0.0)
# Settings.embed_model = get_ollama_embedding()
# Settings.chunk_size = 512

# # retrieve from graph db
# query_engine = loaded_index.as_retriever()
# print("done retrieve")

# # generate
# res = query_engine.retrieve(
#     "Tell me about the connection between communicative VL and vaccinecompliance"
# )
# print(res)


class GraphIngestion:
    DEFAULT_EMBEDDING_DIMENSION = 1536
    LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    DATA_PATH = "/*.pdf"

    def __init__(self, model, embedding_model, logging_level=logging.INFO):
        # cfg
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        # internal variables
        self.vector_store = None
        self.index = None

        # If not set, default is OpenAI's GPT-3.5
        Settings.llm = model
        Settings.embed_model = embedding_model

    def init_graph_db(
        self,
        username,
        password,
        url,
        database="neo4j",
        embedding_dimension=DEFAULT_EMBEDDING_DIMENSION,
        index_name="yasuo_index",
        hybrid_search=True,
    ):
        neo4j_vector = Neo4jVectorStore(
            username=username,
            password=password,
            url=url,
            database=database,
            embedding_dimension=embedding_dimension,
            index_name=index_name,
            hybrid_search=hybrid_search,
        )
        self.vector_store = neo4j_vector

        if embedding_dimension != self.DEFAULT_EMBEDDING_DIMENSION:
            # TODO: create custom index for new dimension
            pass

        return neo4j_vector

    def process_ingestion(self, data_path=DATA_PATH, k=1, show_progress=True):
        # use llm sherpa to read pdf
        pdf_reader = LayoutPDFReader(self.LLM_SHERPA_API_URL)
        documents = []

        # iterate through all pdf files in data path
        file_count = 0
        for file in glob.glob(self.DATA_PATH):
            # limit number of files to ingest
            if file_count >= k:
                break
            file_count += 1

            doc = pdf_reader.read_pdf(file)
            block: Block
            for block in doc.chunks():
                metadata = {
                    "page": block.page_idx,
                    "source": os.path.basename(file),
                    "tag": block.tag,
                }
                document = Document(text=block.to_context_text(), metadata=metadata)
                documents.append(document)
        logging.info("done load documents")

        self.index = self.embed_documents(documents, show_progress=show_progress)
        logging.info("done embed")

        return self.index

    def embed_documents(self, documents, show_progress=True):
        # load storage context using initialized vector store
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=show_progress,
        )

        return index

    def get_latest_index(self):
        return VectorStoreIndex.from_vector_store(self.vector_store)

    def get_latest_index(
        self,
        username,
        password,
        url,
        database,
        index_name="yasuo_index",
        hybrid_search=True,
    ):
        vector_store = Neo4jVectorStore(
            username=username,
            password=password,
            url=url,
            database=database,
            index_name=index_name,
            hybrid_search=hybrid_search,
        )

        return VectorStoreIndex.from_vector_store(vector_store=vector_store)


if __name__ == "__main__":
    i = GraphIngestion()
    i.init_graph_db()
    i.process_ingestion()
