import glob
import logging
import os
import sys

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers.layout_reader import Block


class GraphIngestion:
    DEFAULT_EMBEDDING_DIMENSION = 1536
    LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    DATA_PATH = "pdf/*.pdf"

    def __init__(self, model=None, embedding_model=None, logging_level=logging.INFO):
        # cfg
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        # internal variables
        self.vector_store = None
        self.index = None

        # If not set, default is OpenAI's GPT-3.5
        if model is not None and embedding_model is not None:
            Settings.llm = model
            Settings.embed_model = embedding_model

    def init_graph_db(
        self,
        username="neo4j",
        password="yasuotruong",
        url="neo4j://localhost:7687",
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

    def process_ingestion(self, data_path=DATA_PATH, k=100, show_progress=True):
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
        username="neo4j",
        password="yasuotruong",
        url="neo4j://localhost:7687",
        database="neo4j",
        embedding_dimension=DEFAULT_EMBEDDING_DIMENSION,
        index_name="yasuo_index",
        hybrid_search=True,
    ):
        vector_store = Neo4jVectorStore(
            username=username,
            password=password,
            url=url,
            database=database,
            index_name=index_name,
            embedding_dimension=embedding_dimension,
            hybrid_search=hybrid_search,
        )

        return VectorStoreIndex.from_vector_store(vector_store=vector_store)


if __name__ == "__main__":
    i = GraphIngestion()
    i.init_graph_db()
    i.process_ingestion()
