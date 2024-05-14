from retrievals.retrieval import Retrieval
import logging

from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from retrievals.graph.kg_retrieval_prompt import CYPHER_GENERATION_TEMPLATE

logger = logging.getLogger(__name__)


@Retrieval.register
class KGRetrievalApi:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

        self.graph = Neo4jGraph(
            url="neo4j://localhost:7687",
            username="neo4j",
            password="yasuotruong",
            database="neo4j",
            refresh_schema=True,
        )

        self.vector_store = Neo4jVector.from_existing_index(
            embedding=OpenAIEmbeddings(),
            url="neo4j://localhost:7687",
            username="neo4j",
            password="yasuotruong",
            database="neo4j",
            index_name="yasuo_index",
            text_node_property="text",
        )

    def search(self, queries: list[str]):
        formatted_response = []

        # rewrite queries to graph query
        rewrite_queries = []
        for q in queries:
            # use LLM to rewrite queries
            rewrite_queries.append(self.rewrite_question_to_graph_query(q))

        # retrieve from vector store
        for i in range(len(rewrite_queries)):
            vector_query = queries[i]
            vector_response = self.vector_retriever(vector_query)

            rewrite_query = rewrite_queries[i]
            relationship_response = self.kg_retriever(rewrite_query)

            fused_response = self.fuse_response(vector_response, relationship_response)
            formatted_response.append(fused_response)

        return formatted_response

    def rewrite_question_to_graph_query(self, query: str):
        # use LLM to rewrite queries
        return ""

    def vector_retriever(self, query: str):
        # retrieve from vector store
        pass

    def kg_retriever(self, query: str):
        # gen cypher query

        # retrieve from graph

        # parse response to nlp format
        pass

    def fuse_response(self, vector_response, relationship_response):
        # fuse response from vector and relationship retrieval
        pass
