import logging

from ingestion.ingestion import Ingestion
from models.types import Source
from retrievals.retrieval import Retrieval
from llama_index.core.vector_stores.types import (
    VectorStoreQueryMode,
)

logger = logging.getLogger(__name__)


@Retrieval.register
class DeepRetrievalApi:
    # Retrieve using deep models.

    def __init__(self, **kwargs):
        index = Ingestion(with_openai=True).read_from_chroma()
        self.retriever = index.as_retriever(
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            **kwargs
        )

    def search(self, queries: list[str]):
        # create set formatted_response
        documentIdSet = set()
        formatted_response = []

        for q in queries:
            response = self.retriever.retrieve(q)

            for x in response:
                if x.node_id not in documentIdSet:
                    documentIdSet.add(x.node_id)
                    source = Source(
                        id=x.node_id,
                        doi=x.metadata.get("doi", ""),
                        file_name=x.metadata.get("file_name", ""),
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
