import os
from pydantic import BaseModel, Field, field_validator
from langchain_docling.loader import ExportType


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Config(BaseModel):
    HF_TOKEN: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    FILE_PATH: str = Field(default="https://arxiv.org/pdf/2408.09869")
    EMBED_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_HOST: str = Field(default_factory=lambda: os.getenv("MILVUS_HOST", "localhost"))
    VECTOR_PORT: str = Field(default_factory=lambda: os.getenv("MILVUS_PORT", "19530"))
    VECTOR_DB_COLLECTION: str = "docling_demo"
    GROQ_API_KEY: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GEN_MODEL_ID: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    GEN_TEMPERATURE: float = 0.05
    TOP_K: int = 3

    class Config:
        arbitrary_types_allowed = True

config = Config()
