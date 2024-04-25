import json
from ingestion.graph_embedding import GraphIngestion

from retrievals.retrieval import Retrieval
from models.types import Source


@Retrieval.register
class GraphRetrievalApi:
    def __init__(self):
        graph_ingestion = GraphIngestion()
        graph_ingestion.init_graph_db()
        index = graph_ingestion.get_latest_index(
            username="neo4j",
            password="yasuotruong",
            database="neo4j",
            url="neo4j://localhost:7687",
        )
        self.retriever = index.as_retriever()

    def search(self, query):
        response = self.retriever.retrieve(query)
        formatted_response: list[Source] = []
        for x in response:
            print("\n content1 \n", json.dumps(x.node.to_dict(), indent=4))
            source = Source(
                id=x.node_id,
                doi=x.metadata.get("doi", ""),
                file_name=x.metadata.get("source", ""),
                content=x.get_content(),
                score=round(x.get_score(), 2),
            )
            formatted_response.append(source)
        return formatted_response