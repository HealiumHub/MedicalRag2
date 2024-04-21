import glob
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llmsherpa.readers.layout_reader import Block
from llama_index.core.schema import Document
from llmsherpa.readers import LayoutPDFReader
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core import KnowledgeGraphIndex

from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext

from IPython.display import Markdown, display

from embedding import get_ollama_embedding


import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# init graph db
url = "neo4j://localhost:7687"
username ="neo4j"
password = "yasuotruong"
database = "neo4j"
DATA_PATH = "pdf"

# Settings.llm = Ollama(model="gemma:2b", temperature=0.0)
# Settings.embed_model = get_ollama_embedding()
Settings.chunk_size = 512


neo4j_vector = Neo4jVectorStore(
  username, 
  password, 
  url, 
  embedding_dimension=1536,
  index_name="yasuo_index",
  hybrid_search=True,
)

storage_context = StorageContext.from_defaults(vector_store=neo4j_vector)
print("done init")

# load document
LLM_SHERPA_API_URL = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
pdf_reader = LayoutPDFReader(LLM_SHERPA_API_URL)
documents = []
i = 0
for file in glob.glob(DATA_PATH + "/*.pdf"):
  if i >= 1:
    break
  i += 1
  print(file, "k=", i)
  doc = pdf_reader.read_pdf(file)
  block: Block
  for block in doc.chunks():
    document = Document(text=block.to_context_text())
    documents.append(document)
print("done load documents")

# embed to graph db
index = VectorStoreIndex.from_documents(
  documents,
  storage_context=storage_context,
  # max_triplets_per_chunk=2,
  # include_embeddings=True,
  show_progress=True
)
print("done embed")

# load existing index
# index = Neo4jVectorStore(
#   username, 
#   password, 
#   url, 
#   database,
#   index_name="yasuo_index",
#   hybrid_search=True
# )
# loaded_index = VectorStoreIndex.from_vector_store(index)

# retrieve from graph db
query_engine = index.as_query_engine()
print("done retrieve")

# generate
response = query_engine.query("Tell me about the connection between communicative VL and vaccinecompliance ")
print(response)
display(Markdown(f"<b>{response}</b>"))
  