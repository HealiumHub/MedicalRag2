from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_ollama_embedding(model: str = "nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=model)
    return embeddings
