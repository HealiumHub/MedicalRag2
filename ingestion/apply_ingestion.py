from ingestion import Ingestion


ingestion = Ingestion(with_openai=True)
ingestion.clear_database()
documents = ingestion.load_documents(k=1)  # smart chunking
# nodes = ingestion.split_documents(documents)
# nodes = ingestion.extract_metadata(documents=documents)
ingestion.add_to_chroma(documents=documents)
