from typing import Optional
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from .config import config


class EmbeddingModel:

    def __init__(self, model_id: Optional[str] = None) -> None:
        self.model_id: str = model_id or config.EMBED_MODEL_ID
        self.embedding: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
            model_name=self.model_id
        )

    def get_embedding(self) -> Embeddings:
        return self.embedding
