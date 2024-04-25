# MedicalRag2

## Setup

Run the following command

1. Install the requirement

```bash
./setup.sh
```

2. Unzip all the pdf file (optional)  
   _Note: the zip file must be in the `ingestion/pdf` directory_

```bash
./unzip.sh
```

3. Setup environment key for OpenAI in parent folder `.env` (optional)

```
OPENAI_API_KEY=xxx
```

4. Run this command to ingest the pdf to the chromaDB

```bash
python3 ingestion/apply_ingestion.py
```

4. You should download [ollama](https://ollama.com/) and pull the required LLM  
   This is a few that you can test out:

   ```bash
    ollama pull phi3:latest
    ollama pull llama3:8b
    ollama pull llama3:instruct
    ollama pull gemma:2b
    ollama pull gemma:2b-instruct
    ollama pull gemma:7b-instruct
    ollama pull gemma:latest
   ```

5. Run this command to start the application

```bash
streamlit run app.py
```
