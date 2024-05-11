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

DEFAULT_EMBEDDING_DIMENSION = 1536
LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
DATA_PATH = "pdf/*.pdf"

graph = Neo4jGraph(
    url="neo4j://localhost:7687",
    username="neo4j",
    password="yasuotruong",
    database="neo4j",
)
print("Graph initialized")

from langchain_core.prompts import ChatPromptTemplate

# template = ChatPromptTemplate.from_template(
#     """
#     You are an expert in diabetes.
#     Take a deep breath and analyze the document. Extract the information and create the graph.
#     """
# )


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
llm_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=[
        "Disease",
        "DietarySupplements",
        "Treatment",
        "Biomarker",
        "Mushroom",
        "BioactiveComponents",
    ],
    allowed_relationships=["CAUSES", "TREATS", "BOOSTS", "REDUCES", "PREVENTS"],
)
print("LLM initialized")


# use llm sherpa to read pdf
pdf_reader = LayoutPDFReader(LLM_SHERPA_API_URL)
documents = []

# iterate through all pdf files in data path
file_count = 0
for file in glob.glob(DATA_PATH):
    doc = pdf_reader.read_pdf(file)
    block: Block
    print(f"Processing file: {os.path.basename(file)}")

    block_idx = 0
    for block in doc.chunks():
        print(f"Processing block {block_idx} on page {block.page_idx}")
        metadata = {
            "page": block.page_idx,
            "source": os.path.basename(file),
            "tag": block.tag,
        }
        documents = [Document(page_content=block.to_context_text())]
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        print(f"Text extracted from the document: {block.to_context_text()}")
        print(f"Nodes:{graph_documents[0].nodes}")
        print(f"Relationships:{graph_documents[0].relationships}")
        graph.add_graph_documents(graph_documents, include_source=True)

        block_idx += 1
