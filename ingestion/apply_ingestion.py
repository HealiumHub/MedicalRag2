from ingestion import Ingestion


ingestion = Ingestion(with_openai=True)
ingestion.clear_database()
documents = ingestion.load_documents(k=6)  # smart chunking
nodes = ingestion.sentence_window_split(documents=documents)
# nodes = ingestion.split_documents(documents)
# nodes = ingestion.extract_metadata(documents=documents)
ingestion.add_to_chroma(nodes=nodes)
