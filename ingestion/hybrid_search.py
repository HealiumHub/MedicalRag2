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
Do not use any other relationship types or properties that are not provided.
Do not use the node label
Do not use relationship types
Keeps everything generics

Schema:
{schema}

Examples:
# What is the relationship between Hyponatremia and Nocturnal Enuresis
MATCH p=(start)-[*1..5]-(end)
WHERE start.id CONTAINS 'Hyponatremia' AND end.id CONTAINS 'Nocturnal Enuresis'
RETURN p
ORDER BY length(p) DESC
LIMIT 1
# What is the result of Plasma Arginine-Vasopressin Concentrations and Urine Concentration Inability
MATCH p=(start)-[*1..5]-(end)
WHERE start.id CONTAINS 'Plasma Arginine-Vasopressin Concentrations' AND end.id CONTAINS 'Urine Concentration Inability'
RETURN p
ORDER BY length(p) DESC
LIMIT 1
# What causes diabetes
MATCH p=(start)-[:CASES]->(end)
WHERE start.id CONTAINS 'Diabetes'
RETURN p

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

# p1
# q: what is the result of Plasma Arginine-Vasopressin Concentrations and Urine Concentration Inability
# rr: List all relationships between Plasma Arginine-Vasopressin and Urine

# p2
# q: what is the effect of Primary Monosymptomatic Enuresis on Restoration Of Desmopressin?
# rr: List all relationships between Monosymptomatic Enuresis and Restoration Of Desmopressin

# p3
# q: what is the effect of Primary Monosymptomatic Enuresis on Restoration Of Desmopressin and Nasal Mucosa?
# rr: List all relationships between Monosymptomatic Enuresis and Restoration Of Desmopressin and Nasal Mucosa

# p4
# q: what is the relationship between Mild-To-Moderate Hemophilia A and Diabetes Insipidus
# rr: List all relationships between Hemophilia A and Diabetes

# p5
# q: what is the relationship between Vasopressin and Diabetes Retinopathy?
# rr: List all relationships between Vasopressin and Diabetes

q = "What are 5 latest discoveries of diabetes"
rewrite_q = "List all relationships of diabetes"

vector_results = existing_index_return.similarity_search(q, k=5)

vector_result = ""
for doc in vector_results:
    vector_result += doc.page_content + " "

print("##########")
print("Vector result: ", vector_result)
print("##########")


graph_result = chain_language_example.invoke({"query": rewrite_q})
# graph_result = ""
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
