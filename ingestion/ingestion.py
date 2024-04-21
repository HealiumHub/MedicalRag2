import glob
import math
import os
import shutil
import chromadb

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, Document
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers.layout_reader import Block

from embedding import get_ollama_embedding


class Ingestion:
    DATA_PATH = "./ingestion/pdf"
    CHROMA_PATH = "chroma"
    LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"

    def load_documents(self, k=math.inf):
        pdf_reader = LayoutPDFReader(self.LLM_SHERPA_API_URL)
        documents = []
        i = 0
        for file in glob.glob(self.DATA_PATH + "/*.pdf"):
            if i >= k:
                break
            i += 1
            print(file, "k=", i)
            doc = pdf_reader.read_pdf(file)
            block: Block
            for block in doc.chunks():
                document = Document(text=block.to_context_text())
                documents.append(document)
        # self.print_documents(documents)
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

    def print_documents(self, documents: list[Document]):
        for document in documents:
            print("--- New Page ---")
            # print(document.metadata)
            print(document.text)


if __name__ == "__main__":
    ingestion = Ingestion()
    # ingestion.clear_database()
    documents = ingestion.load_documents(k=1)
    # nodes = ingestion.split_documents(documents)
    # ingestion.add_to_chroma(nodes)
