import json
import logging
from typing import List

from ingestion.graph_embedding import GraphIngestion
from models.types import Source
from retrievals.retrieval import Retrieval
from neo4j import GraphDatabase
from neo4j import Record
from neo4j import EagerResult

import logging


@Retrieval.register
class GraphRetrievalApi:
    def __init__(self, **kwargs):
        logging.info("Initializing GraphRetrievalApi")
        graph_ingestion = GraphIngestion()
        graph_ingestion.init_graph_db()
        index = graph_ingestion.get_latest_index(
            username="neo4j",
            password="yasuotruong",
            database="neo4j",
            url="neo4j://localhost:7687",
        )
        self.driver = GraphDatabase.driver(
            "neo4j://localhost:7687", auth=("neo4j", "yasuotruong")
        )
        logging.info("Initialized Neo4j")
        self.retriever = index.as_retriever()

    def close(self):
        self.driver.close()

    def search(self, queries: list[str] = []):
        response = self.retriever.retrieve(queries[0])

        formatted_response: list[Source] = []
        for x in response:
            # logging.info(json.dumps(x.node.to_dict()["relationships"], indent=4))
            for rel in x.node.relationships.values():

                node_id = rel.to_dict()["node_id"]
                logging.info(f"traverse relationship {rel.to_json()}")
                records, summary, keys = self.driver.execute_query(
                    f"""MATCH (n:Chunk)
                    WHERE n.ref_doc_id = "{node_id}"
                    OR n.id = "{node_id}"
                    RETURN n LIMIT 1""",
                )

                if len(records) < 1:
                    logging.error(
                        f"Expected 1 record, got {len(records)} | id: {node_id}"
                    )
                    logging.info(records)
                    continue
                rel_node = dict(records[0].data()["n"])
                text = rel_node.get("text", "")
                logging.info(f"related node {rel_node}")
                logging.info(f"Related text: {rel_node.get('text')}")

                # TODO find a way to find these information
                source = Source(
                    id="x",
                    doi="",
                    file_name="x",
                    content=text,
                    score=round(0.3, 2),
                )
                formatted_response.append(source)

            source = Source(
                id=x.node_id,
                doi=x.metadata.get("doi", ""),
                file_name=x.metadata.get("source", ""),
                content=x.get_content(),
                score=round(x.get_score(), 2),
            )
            formatted_response.append(source)

        logging.info(f"total formatted response: {len(formatted_response)}")
        self.close()  # close db conn
        return formatted_response
