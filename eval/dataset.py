import logging
from typing import List

from llama_index.core.schema import Document
import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    generate_qa_embedding_pairs,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from ingestion.dataloader import ingestion_index

nest_asyncio.apply()

print("Generating QA dataset")

# NOTE: Can't get all from chroma so set high top-k
nodes = (
    ingestion_index.read_from_chroma().as_retriever(similarity_top_k=1000).retrieve("")
)
nodes = [node.node for node in nodes]
# documents = ingestion_index.load_documents(k=1)  # smart chunking
# nodes = ingestion_index.sentence_window_split(documents=documents)
print(f"Number of nodes: {len(nodes)}")
# print(f"Nodes: {nodes}")

llm = OpenAI(model="gpt-3.5-turbo")

# # TODO: Only 1Q-1Chunk rn, better to have 1Q-multiple chunks (esp from different docs).
qa_dataset = generate_qa_embedding_pairs(nodes, llm=llm, num_questions_per_chunk=1)
qa_dataset.save_json("pg_eval_dataset.json")
