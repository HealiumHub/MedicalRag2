import json
import logging
import random
import re
import uuid
from typing import List

import nest_asyncio
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    generate_qa_embedding_pairs,
)
from llama_index.core.llms.utils import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, MetadataMode, NodeWithScore, TextNode
from llama_index.llms.openai import OpenAI
from tqdm import tqdm

from ingestion.dataloader import ingestion_index

nest_asyncio.apply()

print("Generating QA dataset")

DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""


def generate_qa_embedding_pairs_custom(
    nodes: List[TextNode],
    llm: LLM,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}

    # Pair up 2 random nodes and generate questions based on them.
    node_pairs: list[tuple[TextNode, TextNode]] = []
    for node in nodes:
        node_pairs.append([node, nodes[random.randint(0, len(nodes) - 1)]])

    for node_pair in tqdm(node_pairs):
        query = qa_generate_prompt_tmpl.format(
            context_str=f"{node_pair[0].get_content(metadata_mode=MetadataMode.NONE)}\n{node_pair[1].get_content(metadata_mode=MetadataMode.NONE)}",
            num_questions_per_chunk=num_questions_per_chunk,
        )
        response = llm.complete(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_pair[0].id_, node_pair[1].id_]

    # construct dataset
    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )


list_of_key_words = [
    "diabetes",
    "mushroom",
    "insulin",
    "diabetic retinopathy",
    "genetic factors",
    "type 2 diabetes",
]

# NOTE: Can't get all from chroma so set high top-k
final_list_of_nodes: list[NodeWithScore] = []
for key_word in list_of_key_words:
    nodes = (
        ingestion_index.read_from_chroma()
        .as_retriever(similarity_top_k=5000)
        .retrieve(key_word)
    )
    nodes: List[NodeWithScore] = [
        node
        for node in nodes
        if len(node.get_content()) > 30
        and "References" not in node.get_content()
        and "CONTRIBUTIONS" not in node.get_content()
        and "et al." not in node.get_content()
        and node.score > 0.4
    ][:10]
    final_list_of_nodes.extend(nodes)
    print(f"Number of nodes for {key_word}: {len(nodes)}")

# Deduplicate nodes
final_list_of_nodes = list(
    {node.id_: node.node for node in final_list_of_nodes}.values()
)
print(f"Nodes: {json.dumps(final_list_of_nodes, indent=4, default=str)}")
print(f"Number of nodes after deduplication: {len(final_list_of_nodes)}")

llm = OpenAI(model="gpt-3.5-turbo")

# # TODO: Only 1Q-1Chunk rn, better to have 1Q-multiple chunks (esp from different docs).
qa_dataset = generate_qa_embedding_pairs(
    final_list_of_nodes, llm=llm, num_questions_per_chunk=1
)
qa_dataset.save_json("pg_eval_dataset.json")
