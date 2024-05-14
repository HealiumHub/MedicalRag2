from faiss_ingestion import FaissIngestion

faiss = FaissIngestion(embedding_model_name="huggingface/")
faiss.clear_database()
documents = faiss.load_documents(k=1)  # smart chunking
nodes = faiss.sentence_window_split(documents=documents)
# nodes = ingestion.split_documents(documents)
# nodes = ingestion.extract_metadata(documents=documents)
faiss.create_index(nodes=nodes)