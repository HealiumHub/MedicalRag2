import asyncio
from typing import List

import pandas as pd
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    RetrievalEvalResult,
    RetrieverEvaluator,
)
from llama_index.core.evaluation.benchmarks import BeirEvaluator
from llama_index.core.evaluation.retrieval.base import BaseRetrievalEvaluator
from llama_index.core.schema import (
    Document,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
    TextNode,
)
from llama_index.core.indices.base_retriever import BaseRetriever

from ingestion.dataloader import ingestion_index
from postretrieve.rerank import Reranker

# Resources:
# - https://docs.llamaindex.ai/en/stable/examples/evaluation/retrieval/retriever_eval/
# - https://docs.llamaindex.ai/en/stable/examples/evaluation/BeirEvaluation/

# TODO: Needs to read from the same db used to create the dataset. Currently different -> all is wrong
retriever = ingestion_index.read_from_chroma().as_retriever(similarity_top_k=50)


class RerankedRetriever(BaseRetriever):
    def __init__(self, retriever: BaseRetriever):
        super().__init__()
        self.retriever = retriever
        self.reranker = Reranker()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return super()._retrieve(query_bundle)

    async def aretrieve(self, query: str):
        retrieved_nodes = await self.retriever.aretrieve(query)
        return self.rerank(query, retrieved_nodes)

    def rerank(self, query: str, retrieved_nodes: list[NodeWithScore]):
        # Rerank the retrieved nodes
        retrieved_nodes = self.reranker.get_top_k(query, retrieved_nodes, 5)
        return retrieved_nodes


metrics = ["mrr", "hit_rate"]
qa_dataset = EmbeddingQAFinetuneDataset.from_json("pg_eval_dataset.json")
reranked_retriever = RerankedRetriever(retriever)
retriever_evaluator: BaseRetrievalEvaluator = RetrieverEvaluator.from_metric_names(
    metrics, retriever=reranked_retriever
)

# try it out on an entire dataset
eval_results = asyncio.run(retriever_evaluator.aevaluate_dataset(qa_dataset))


def format_results_to_table(name, eval_results: list[RetrievalEvalResult]):
    """Display results from evaluate."""
    metric_dicts = []
    for eval_result in eval_results:
        # llamaindex don't support recall & precision metrics so we'll do it ourselves
        row_json = eval_result.dict()
        retrieved_ids = eval_result.retrieved_ids
        expected_ids = eval_result.expected_ids

        # Calculate fp, fn, tp, tn based on retrieved_ids and expected_ids
        row_json["false positive"] = len(set(retrieved_ids) - set(expected_ids))
        row_json["false negative"] = len(set(expected_ids) - set(retrieved_ids))
        row_json["true positive"] = len(set(retrieved_ids) & set(expected_ids))
        row_json["true negative"] = len(set(retrieved_ids) & set(expected_ids))

        # Calculate precision, recall, f1
        row_json["precision"] = row_json["true positive"] / (
            row_json["true positive"] + row_json["false positive"] + 1e-10
        )
        row_json["recall"] = row_json["true positive"] / (
            row_json["true positive"] + row_json["false negative"] + 1e-10
        )
        row_json["f1"] = (
            2
            * (row_json["precision"] * row_json["recall"])
            / (row_json["precision"] + row_json["recall"] + 1e-10)
        )

        # Get mrr & hit rate
        metric_results = eval_result.metric_vals_dict
        row_json.update(metric_results)

        metric_dicts.append(row_json)

    full_df = pd.DataFrame(metric_dicts)
    return full_df


df = format_results_to_table("retriever", eval_results)
df.to_csv("retriever_eval_results.csv", index=False)
print(df.describe())
