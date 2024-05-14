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
