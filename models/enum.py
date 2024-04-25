from enum import Enum
from retrievals.chroma_retrieval import DeepRetrievalApi
from retrievals.graph_retrieval import GraphRetrievalApi
from retrievals.retrieval import Retrieval


class RetrievalApiEnum(str, Enum):
    CHROMA_RETRIEVAL = "CHROMA_RETRIEVAL"
    NEO4J_RETRIEVAL = "NEO4J_RETRIEVAL"

    @staticmethod
    def get_retrieval(str) -> Retrieval:
        if str == RetrievalApiEnum.NEO4J_RETRIEVAL:
            return GraphRetrievalApi()
        elif str == RetrievalApiEnum.CHROMA_RETRIEVAL:
            return DeepRetrievalApi()
        else:
            raise ValueError("Invalid retrieval API")
