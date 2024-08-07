import logging
from typing import List

from llama_index.core.schema import NodeWithScore

from ingestion.ingestion import ingestion_index
from models.types import Source
from postretrieve.rerank import Reranker
from retrievals.retrieval import Retrieval
from llama_index.core.vector_stores.types import (
    VectorStoreQueryMode,
)
from llama_index.core.postprocessor import MetadataReplacementPostProcessor


logger = logging.getLogger(__name__)


@Retrieval.register
class DeepRetrievalApi:
    # Retrieve using deep models.

    def __init__(self, **kwargs):
        index = ingestion_index.read_from_chroma()
        self.retriever = index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID, **kwargs
        )

    def search(self, queries: list[str]):
        # create set formatted_response
        documentIdSet = set()
        formatted_response = []

        for q in queries:
            response: List[NodeWithScore] = self.retriever.retrieve(q)
            print(f"chroma_retrieval | query {q} | response {response}")
            response = Reranker().get_top_k(q, response, k=5)
            processor = MetadataReplacementPostProcessor(target_metadata_key="window")
            response = processor.postprocess_nodes(response)

            # print(f"chroma_retrieval | query {q} | response {response}")

            for x in response:
                if x.node_id not in documentIdSet:
                    documentIdSet.add(x.node_id)
                    source = Source(
                        id=x.node_id,
                        doi=x.metadata.get("doi", ""),
                        file_name=x.metadata.get("file_name", ""),
                        page=x.metadata.get("page", 1),
                        content=x.get_content(),
                        score=round(x.get_score(), 2),
                    )
                    formatted_response.append(source)
                else:
                    # search the document in formatted_response and update the score with greates
                    for doc in formatted_response:
                        if doc.id == x.node_id:
                            doc.score = max(doc.score, round(x.get_score(), 2))
                            break
        # logger.info(json.dumps(formatted_response, indent=4, default=str))
        return formatted_response
