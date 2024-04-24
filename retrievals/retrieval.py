from ingestion.ingestion import Ingestion


class DeepRetrievalApi:
    # Retrieve using deep models.

    def __init__(self):
        index = Ingestion().read_from_chroma()
        self.retriever = index.as_retriever(similarity_top_k=6)

    def search(self, query):
        response = self.retriever.retrieve(query)

        formatted_response = []
        for x in response:
            formatted_response.append(
                {
                    "id": x.node_id,
                    # TODO: Populate metatdata.
                    "doi": x.metadata.get("doi", ""),
                    "file_name": x.metadata.get("file_name", "").replace(".pdf", ""),
                    "title": x.metadata.get("title", ""),
                    "content": x.get_content(),
                    "score": round(x.get_score(), 2),
                }
            )
        return formatted_response
