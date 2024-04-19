import os
import shutil
import chromadb

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from sympy import true

from embedding import get_ollama_embedding


class Ingestion:
    DATA_PATH = "./ingestion/pdf"
    CHROMA_PATH = "chroma"

    def load_documents(self):
        document_loader = SimpleDirectoryReader(self.DATA_PATH, num_files_limit=10)
        documents = document_loader.load_data(true)
        print(len(documents))
        return documents

    def split_documents(self, documents: list[Document]):
        node_parser = SimpleNodeParser.from_defaults(chunk_size=1024)
        # Extract nodes from documents
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"Split {len(documents)} documents into {len(nodes)} chunks")
        return nodes

    def add_to_chroma(self, nodes: list[BaseNode]):
        db = chromadb.PersistentClient(path=self.CHROMA_PATH)
        chroma_collection = db.get_or_create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=get_ollama_embedding(),
        )
        return index

    def __calculate_chunk_ids(self, chunks):

        # This will create IDs like "data/monopoly.pdf:6:2"
        # Page Source : Page Number : Chunk Index

        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id
        return chunks

    def clear_database(self):
        if os.path.exists(self.CHROMA_PATH):
            shutil.rmtree(self.CHROMA_PATH)


if __name__ == "__main__":
    ingestion = Ingestion()
    documents = ingestion.load_documents()
    nodes = ingestion.split_documents(documents)
    ingestion.add_to_chroma(nodes)
