from ingestion import Ingestion


ingestion = Ingestion()
ingestion.clear_database()
documents = ingestion.load_documents(k=10)  # smart chunking
# nodes = ingestion.split_documents(documents)
# nodes = ingestion.extract_metadata(documents=documents)
ingestion.add_to_chroma(documents=documents)
