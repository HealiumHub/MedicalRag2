import logging
import os

import torch
from llama_index.core.schema import NodeWithScore
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from models.types import Source

logger = logging.getLogger(__name__)
reranking_model = os.getenv("RERANKING_MODEL", "ncbi/MedCPT-Cross-Encoder")


class Reranker:
    def __init__(self, model_name: str = "ncbi/MedCPT-Cross-Encoder"):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def rerank(
        self, query: str, chunks: list[Source | NodeWithScore]
    ) -> list[Source | NodeWithScore]:
        # combine query article into pairs
        articles = [
            chunk.content if isinstance(chunk, Source) else chunk.get_content()
            for chunk in chunks
        ]
        pairs = [[query, article] for article in articles]

        # infer scores
        with torch.no_grad():
            encoded = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )

            # tensor([  6.9363,  -8.2063,  -8.7692, -12.3450, -10.4416, -15.8475])
            logits = self.model(**encoded).logits.squeeze(dim=1)

            # Convert to 0-1 range through sigmoid
            scores = torch.sigmoid(logits).tolist()
            logger.info(f"{logits=}\n{scores=}")

        # Set the new scores for each chunk
        for i, chunk in enumerate(chunks):
            chunk.score = scores[i]

        return sorted(chunks, key=lambda x: x.score, reverse=True)

    def get_top_k(
        self, query: str, chunks: list[Source | NodeWithScore], k: int = 5
    ) -> list[Source | NodeWithScore]:
        # Avoid modifying the original list
        chunks = chunks.copy()
        return self.rerank(query, chunks)[: min(k, len(chunks))]
