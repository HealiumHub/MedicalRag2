from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_ollama_embedding(model: str = "all-minilm:latest"):
    embeddings = OllamaEmbeddings(model=model)
    return embeddings
