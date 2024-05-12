import os
import glob
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from llmsherpa.readers import LayoutPDFReader
from llmsherpa.readers.layout_reader import Block
from langchain_core.documents import Document

from typing import List
import logging
import sys

DEFAULT_EMBEDDING_DIMENSION = 1536
LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
DATA_PATH = "pdf/*.pdf"


class KGConstruct:
    def __init__(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
        self.llm_graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[
                "Medical_concept",
                "Country",
                "Institution",
                "Person",
                "Unknown",
            ],
            allowed_relationships=[
                "HELPS",
                "BOOST",
                "REDUCES",
                "RELATED_TO",
                "ABSORBS",
                "CONTAINS",
                "CAUSES",
                "TREATS",
                "HAS",
                "UNKNOWN_RELATIONSHIP",
            ],
            strict_mode=True,
            node_properties=["source", "page", "tag", "text"],
        )
        logging.info("LLM initialized")

        self.graph = Neo4jGraph(
            url="neo4j://localhost:7687",
            username="neo4j",
            password="yasuotruong",
            database="neo4j",
        )

        logging.info("Graph initialized")

    def process(self):
        # use llm sherpa to read pdf
        pdf_reader = LayoutPDFReader(LLM_SHERPA_API_URL)
        documents = []

        for file in glob.glob(DATA_PATH):
            doc = pdf_reader.read_pdf(file)
            block: Block
            logging.info(f"Processing file: {os.path.basename(file)}")

            block_idx = 0
            for block in doc.chunks():
                print(f"Processing block {block_idx} on page {block.page_idx}")

                metadata = {
                    "page": block.page_idx,
                    "source": os.path.basename(file),
                    "tag": block.tag,
                    "text": block.to_context_text(),
                }

                # add metadata to context text
                page_content = block.to_context_text() + " " + str(metadata)

                documents = [Document(page_content=page_content)]
                graph_documents = self.llm_graph_transformer.convert_to_graph_documents(
                    documents
                )

                logging.info(
                    f"Text extracted from the document: {block.to_context_text()}"
                )
                logging.info(f"Nodes:{graph_documents[0].nodes}")
                logging.info(f"Relationships:{graph_documents[0].relationships}")

                self.graph.add_graph_documents(graph_documents, include_source=False)
                block_idx += 1

        self._clear_lonely_nodes()
        logging.info("done process")

    def _clear_lonely_nodes(self):
        # Delete single node
        self.graph.query(
            """MATCH (n)
            WHERE NOT (n)-[]-()
            DELETE n
        """
        )


if __name__ == "__main__":
    kg_construct = KGConstruct()
    kg_construct.process()
