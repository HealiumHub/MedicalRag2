from faiss_ingestion import faiss_instance

faiss = faiss_instance
faiss.clear_database()
documents = faiss.load_documents(k=6)  # smart chunking
nodes = faiss.sentence_window_split(documents=documents)
# nodes = ingestion.split_documents(documents)
# nodes = ingestion.extract_metadata(documents=documents)
faiss.create_index(nodes=nodes)