from enum import Enum
from retrievals.chroma_retrieval import DeepRetrievalApi
from retrievals.graph_retrieval import GraphRetrievalApi
from retrievals.retrieval import Retrieval


class RetrievalApiEnum(str, Enum):
    CHROMA_RETRIEVAL = "CHROMA_RETRIEVAL"
    NEO4J_RETRIEVAL = "NEO4J_RETRIEVAL"

    @staticmethod
    def get_retrieval(retrieval_type: str, **kwargs) -> Retrieval:
        if retrieval_type == RetrievalApiEnum.NEO4J_RETRIEVAL:
            return GraphRetrievalApi(**kwargs)
        elif retrieval_type == RetrievalApiEnum.CHROMA_RETRIEVAL:
            return DeepRetrievalApi(**kwargs)
        else:
            raise ValueError("Invalid retrieval API")