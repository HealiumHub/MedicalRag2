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
Do not use any other relationship types or properties that are not provided.
Use WHERE clause and contains to get more results.
Take a deep breath and double check the cypher query. Ensure that it is correct.
Schema:
{schema}
Cypher examples:
# List 25 bioactive components?
MATCH (n:`Bioactivecomponents`) RETURN n LIMIT 25

# Which mushrooms that can reduce blood glucose?
MATCH (n:Mushroom)-[r:REDUCES]-(b:Biomarker) WHERE b.id CONTAINS "Blood Glucose" RETURN n, b


Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

The question is:
{question}"""
CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
)
chain_language_example = GraphCypherQAChain.from_llm(
    ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    graph=graph,
    verbose=True,
    cypher_prompt=CYPHER_GENERATION_PROMPT,
)

existing_index_return = Neo4jVector.from_existing_index(
    embedding=OpenAIEmbeddings(),
    url="neo4j://localhost:7687",
    username="neo4j",
    password="yasuotruong",
    database="neo4j",
    index_name="yasuo_index",
    text_node_property="text",
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

q = "what mushroom contains beta-glucans that can reduce blood glucose?"

vector_results = existing_index_return.similarity_search(q, k=5)

vector_result = ""
for doc in vector_results:
    vector_result += doc.page_content + " "

print("##########")
print("Vector result: ", vector_result)
print("##########")


graph_result = chain_language_example.invoke({"query": q})
print("##########")
print(graph_result)
print("##########")

final_prompt = f"""
You are a helpful medical question-answering agent. Your task is to analyze
and synthesize information from two sources: the top result from a similarity search
(unstructured information) and relevant data from a graph database (structured information).
Given the user's query: {q}, provide a meaningful and efficient answer based
on the insights derived from the following data:

Unstructured information: {vector_result}.
Structured information: {graph_result} 
"""


from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": final_prompt,
        }
    ],
    model="gpt-3.5-turbo",
)

print("##########")
print("##########")
print("##########")
print("\n-- F I N A L  A N S W E R --\n")


answer = chat_completion.choices[0].message.content.strip()
print(answer)
