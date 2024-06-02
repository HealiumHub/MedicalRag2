import logging
from typing import List

from llama_index.core.schema import NodeWithScore

from ingestion.dataloader import ingestion_index
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
        processor = MetadataReplacementPostProcessor(target_metadata_key="window")

        relevant_chunks: List[NodeWithScore] = []
        for q in queries:
            chunks: List[NodeWithScore] = self.retriever.retrieve(q)
            
            # Dedupe the response
            for x in chunks:
                if x.node_id not in documentIdSet:
                    documentIdSet.add(x.node_id)
                    relevant_chunks.append(x)
                else:
                    # search the document in formatted_response and update the score with greates
                    for doc in relevant_chunks:
                        if doc.node_id == x.node_id:
                            doc.score = max(doc.score, round(x.get_score(), 2))
                            break

        relevant_chunks = Reranker().get_top_k(q, relevant_chunks, k=10)
        relevant_chunks = processor.postprocess_nodes(relevant_chunks)

        list_source = []
        for x in relevant_chunks:
            source = Source(
                id=x.node_id,
                doi=x.metadata.get("doi", ""),
                file_name=x.metadata.get("file_name", ""),
                page=x.metadata.get("page", ""),
                content=x.get_content(),
                score=round(x.get_score(), 2),
            )
            list_source.append(source)

        # logger.info(json.dumps(list_source, indent=4, default=str))
        return list_source
