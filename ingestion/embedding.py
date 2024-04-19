from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_ollama_embedding(model: str = "snowflake-arctic-embed"):
    embeddings = OllamaEmbeddings(model=model)
    return embeddings
