import glob
import logging
import math
import os
import shutil
from typing import Sequence

import chromadb

from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding


from llama_index.core.ingestion import IngestionPipeline
from llama_index.extractors.entity import EntityExtractor
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.extractors import (
    KeywordExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers.layout_reader import Block

from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")

logger = logging.getLogger(__name__)


class Ingestion:
    DATA_PATH = "./ingestion/pdf"
    CHROMA_PATH = "chroma"
    LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

    def __init__(self, with_openai: bool = False):
        Settings.llm = Ollama(model="gemma:2b")
        if with_openai:
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=API_KEY,
            )
        else:
            Settings.embed_model = OllamaEmbedding(model_name="snowflake-arctic-embed")

    def load_documents(self, k=math.inf):
        pdf_reader = LayoutPDFReader(self.LLM_SHERPA_API_URL)
        documents = []
        i = 0
        for file in glob.glob(self.DATA_PATH + "/*.pdf"):
            if i >= k:
                break
            i += 1
            logger.info(f"{file=}, {i=}")
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
        # self.print_documents(documents)
        return documents

    def extract_metadata(self, documents: list[Document]):
        transformations = [
            # TitleExtractor(),
            # QuestionsAnsweredExtractor(),
            # SummaryExtractor(summaries=["prev", "self"]),
            KeywordExtractor(),
            EntityExtractor(device="cpu"),
        ]
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(show_progress=True, documents=documents)
        return nodes

    def split_documents(self, documents: list[Document]):
        node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
        # Extract nodes from documents
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(nodes)} chunks")
        return nodes

    def add_to_chroma(
        self,
        db_name: str = "default_db",
        nodes: Sequence[BaseNode] = [],
        documents: list[Document] = [],
    ):
        db = chromadb.PersistentClient(path=self.CHROMA_PATH)
        chroma_collection = db.get_or_create_collection(db_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        for document in documents:
            index.insert(document)
        index.storage_context.persist(self.CHROMA_PATH)
        return index

    def read_from_chroma(self, db_name: str = "default_db"):
        db = chromadb.PersistentClient(path=self.CHROMA_PATH)
        chroma_collection = db.get_or_create_collection(db_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store)
        return index

        # db = chromadb.PersistentClient(path=self.CHROMA_PATH)
        # chroma_collection = db.get_or_create_collection(db_name)
        # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        # storage_context = StorageContext.from_defaults(
        #     persist_dir=self.CHROMA_PATH, vector_store=vector_store
        # )
        # index = load_index_from_storage(storage_context)
        # return index

    def clear_database(self):
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)

    def print_documents(self, documents: list[Document]):
        for document in documents:
            logger.info("--- New Page ---")
            # logger.info(document.metadata)
            logger.info(document.text)
