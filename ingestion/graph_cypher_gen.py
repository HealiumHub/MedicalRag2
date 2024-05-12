import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate


graph = Neo4jGraph(
    url="neo4j://localhost:7687",
    username="neo4j",
    password="yasuotruong",
    refresh_schema=True,
)

CYPHER_GENERATION_TEMPLATE = """
Task:Generate Cypher statement to query a graph database.
Instructions:
Use only Cypher to query the graph database.
Use only the provided relationship types and properties in the schema.
Generate longest path between two nodes.
Use apoc.cypherRuntimeboxed to limit the query execution time to 2 seconds.
Do not use any other relationship types or properties that are not provided.
Don't use the node label

Schema:
{schema}
# What is the relationship between Hyponatremia and Nocturnal Enuresis
CALL apoc.cypher.runTimeboxed("MATCH p=(start:Node)-[:RELATIONSHIP*1..20]-(end:Node)
WHERE start.id CONTAINS 'Hyponatremia' AND end.id CONTAINS 'Nocturnal Enuresis'
RETURN p
ORDER BY length(p) DESC
LIMIT 1", {{}}, 2000)
#

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)
chain_language_example = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2),
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

# question:
# what mushroom contains Ergosterol that can reduce blood glucose?
# what mushroom contains beta-glucans that can reduce blood glucose?
# what can reduces urine causes by demopressin?
# how many treatments are there for diabetes?
# how many mushrooms can reduce blood glucose?
# what can Lentinus Edodes do?
# list all mushrooms that has effect to diabetes
# which mushroom can boost insulin?
# which mushroom has the therapeutic effect against metabolic syndrome

q = "What is the relationship between Hyponatremia and Nocturnal Enuresis".title()

graph_result = chain_language_example.invoke({"query": q})
print("##########\n")
print(graph_result)
