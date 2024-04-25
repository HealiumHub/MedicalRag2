from ingestion.ingestion import Ingestion
from retrievals.retrieval import Retrieval
from ingestion.ingestion import Ingestion
from models.types import Source

@Retrieval.register
class DeepRetrievalApi:
    # Retrieve using deep models.

    def __init__(self):
        index = Ingestion(with_openai=True).read_from_chroma()
        self.retriever = index.as_retriever(similarity_top_k=6)

    def search(self, query):  # -> list:
        response = self.retriever.retrieve(query)
        print("Debug retrieval.py:13", response)
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
