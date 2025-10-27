import os
from pydantic import BaseModel, Field


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Config(BaseModel):
    HF_TOKEN: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    EMBED_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_HOST: str = Field(default_factory=lambda: os.getenv("MILVUS_HOST", "localhost"))
    VECTOR_PORT: str = Field(default_factory=lambda: os.getenv("MILVUS_PORT", "19530"))
    VECTOR_DB_COLLECTION: str = Field(default_factory=lambda: os.getenv("VECTOR_DB_COLLECTION", "docling_demo"))
    GROQ_API_KEY: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GEN_MODEL_ID: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GEN_TEMPERATURE: float = 0.05
    TOP_K: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "5")))

    class Config:
        arbitrary_types_allowed = True

config = Config()
