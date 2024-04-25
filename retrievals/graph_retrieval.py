from ingestion.graph_embedding import GraphIngestion
import logging


class GraphRetrievalApi:
    # Retrieve using deep models.

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
        logging.info("done retrieval", extra={"resp": response})

        formatted_response = []
        for x in response:
            formatted_response.append(
                {
                    "id": x.node_id,
                    # TODO: Populate metatdata.
                    "doi": x.metadata.get("doi", ""),
                    "file_name": x.metadata.get("file_name", "").replace(".pdf", ""),
                    "title": x.metadata.get("title", ""),
                    "content": x.get_content(),
                    "score": round(x.get_score(), 2),
                }
            )
        return formatted_response
