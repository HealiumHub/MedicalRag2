import faiss
import glob
import logging
import math
import os
import shutil
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore

from const import EmbeddingConfig

# from ingestion import Ingestion
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llmsherpa.readers import LayoutPDFReader
from llama_index.core.node_parser import SimpleNodeParser
from llmsherpa.readers.layout_reader import Block
from llama_index.core.schema import BaseNode, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv
load_dotenv()


import glob
import logging
import math
import os
import shutil
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core import (
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore

from llama_index.embeddings.openai import OpenAIEmbedding
from const import EmbeddingConfig
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llmsherpa.readers import LayoutPDFReader
from llama_index.core.node_parser import SimpleNodeParser
from llmsherpa.readers.layout_reader import Block
from llama_index.core.schema import BaseNode, Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "huggingface/ls-da3m0ns/bge_large_medical"
)
logger = logging.getLogger(__name__)

class FaissIngestion():
    LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
    FAISS_PATH = "faiss"
    DATA_PATH = "./ingestion/pdf_new"

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
            HuggingFaceEmbedding(
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
        
    def create_index(self, nodes = []):
        if len(nodes) == 0:
            print("Nodes cannot be empty")
            return 
        
        faiss_index = faiss.IndexFlatL2(1536)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex(
            nodes = nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        index.storage_context.persist(self.FAISS_PATH)
        return index

    def load_index(self):
        print("Loading index")
        vector_store = FaissVectorStore.from_persist_dir(self.FAISS_PATH)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, 
            persist_dir = self.FAISS_PATH
        )
        index = load_index_from_storage(storage_context=storage_context)
        return index 
    
    def clear_database(self):
        if os.path.exists(self.FAISS_PATH):
            shutil.rmtree(self.FAISS_PATH)


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
                    "file_name": os.path.basename(file),
                    "tag": block.tag,
                }
                document = Document(text=block.to_context_text(), metadata=metadata)
                documents.append(document)
        # self.print_documents(documents)
        return documents

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

    def clear_database(self):
        if os.path.exists(self.FAISS_PATH):
            shutil.rmtree(self.FAISS_PATH)

faiss_instance = FaissIngestion(embedding_model_name='openai/text-embedding-3-small')