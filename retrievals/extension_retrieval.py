from llama_index.core.vector_stores.types import (
    VectorStoreQueryMode,
)

from ingestion.ingestion import ingestion_index
from models.types import Source
from preretrieve.expansion.langchain.expansion import QueryExpansion
from retrievals.retrieval import Retrieval


@Retrieval.register
class ExtensionRetrievalApi:
    # Retrieve using deep models.

    def __init__(self, **kwargs):
        index = ingestion_index.read_from_chroma()
        self.retriever = index.as_retriever(
            similarity_top_k=6,
            vector_store_query_mode=VectorStoreQueryMode.HYBRID,
            **kwargs,
        )

    def search(self, queries):
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

    def search_v0(self, query):
        response = self.retriever.retrieve(query)
        formatted_response: list[Source] = []
        for x in response:
            source = Source(
                id=x.node_id,
                doi=x.metadata.get("doi", ""),
                file_name=x.metadata.get("file_name", ""),
                content=x.get_content(),
                score=round(x.get_score(), 2),
            )
            formatted_response.append(source)
        return formatted_response
