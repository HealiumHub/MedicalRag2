from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.document import Document

hybrid_db = Neo4jVector.from_existing_graph(
    text_node_properties=["title", "content"],
    embedding_node_property="embedding",
    embedding=OpenAIEmbeddings(),
    node_label="Document",
    url="neo4j://localhost:7687",
    username="neo4j",
    password="yasuotruong",
    database="neo4j",
)

print("done loading hybrid db")

query = "what mushroom contains beta-glucans that can reduce blood glucose?"
docs_with_score = hybrid_db.similarity_search_with_score(query, k=5)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)
