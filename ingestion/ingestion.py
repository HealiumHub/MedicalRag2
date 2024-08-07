import glob
import logging
import math
import os
import shutil
from typing import List, Sequence

import chromadb

from llama_index.core import Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from langchain.schema import Document as LangChainDocument

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

from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

from dotenv import load_dotenv

from const import EmbeddingConfig

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "huggingface/ls-da3m0ns/bge_large_medical"
)
logger = logging.getLogger(__name__)


class Ingestion:
    DATA_PATH = "./ingestion/pdf_new"
    CHROMA_PATH = "chroma"
    # LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    LLM_SHERPA_API_URL = "http://localhost:5010/api/parseDocument?renderFormat=all"

    def __init__(
        self,
        embedding_model_name: str,
    ):
        Settings.embed_model = self.__get_embed_model(embedding_model_name)

    def __get_embed_model(self, embedding_model_name: str):
        if embedding_model_name.startswith(EmbeddingConfig.OPENAI_PREFIX):
            return OpenAIEmbedding(
                model=embedding_model_name.replace(EmbeddingConfig.OPENAI_PREFIX, ""),
                api_key=API_KEY,
            )
        elif embedding_model_name.startswith(EmbeddingConfig.HUGGINGFACE_PREFIX):
            return HuggingFaceEmbedding(
                model_name=embedding_model_name.replace(
                    EmbeddingConfig.HUGGINGFACE_PREFIX, ""
                )
            )
        elif embedding_model_name.startswith(EmbeddingConfig.OLLAMA_PREFIX):
            return OllamaEmbedding(
                model_name=embedding_model_name.replace(
                    EmbeddingConfig.OLLAMA_PREFIX, ""
                )
            )
        else:
            raise ValueError(f"Unknown embedding model: {embedding_model_name}")

    def load_documents(self, k=math.inf):
        # pdf_reader = LayoutPDFReader(self.LLM_SHERPA_API_URL)
        documents = []
        i = 0
        for file in glob.glob(self.DATA_PATH + "/*.pdf"):
            print(file)
            if i >= k:
                break
            i += 1
            logger.info(f"{file=}, {i=}")
            loader = LLMSherpaFileLoader(
                file_path=file,
                new_indent_parser=True,
                apply_ocr=True,
                strategy="chunks",
                llmsherpa_api_url="http://localhost:5010/api/parseDocument?renderFormat=all",
            )
            docs: List[LangChainDocument] = loader.load()
            # doc = pdf_reader.read_pdf(file)

            for doc in docs:
                document = Document(text=doc.page_content, metadata=doc.metadata)
                documents.append(document)

            # block: Block
            # for block in doc.chunks():
            #     metadata = {
            #         "page": block.page_idx,
            #         "file_name": os.path.basename(file),
            #         "tag": block.tag,
            #     }
            #     document = Document(text=block.to_context_text(), metadata=metadata)
            #     documents.append(document)
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

    def sentence_window_split(self, documents: list[Document]):
        # create the sentence window node parser w/ default settings
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = node_parser.get_nodes_from_documents(documents)
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


ingestion_index = Ingestion(embedding_model_name=EMBEDDING_MODEL_NAME)
