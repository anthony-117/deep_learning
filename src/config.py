import os
from pydantic import BaseModel, Field, field_validator
from langchain_docling.loader import ExportType


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Config(BaseModel):
    HF_TOKEN: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    FILE_PATH: str = Field(default="https://arxiv.org/pdf/2408.09869")
    EMBED_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    GEN_MODEL_ID: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    EXPORT_TYPE: str = ExportType.DOC_CHUNKS
    VECTOR_URI: str = Field(default_factory=lambda: os.getenv("MILVUS_URI", ""))

    class Config:
        arbitrary_types_allowed = True

    @field_validator("VECTOR_URI")
    def ensure_vector_uri_present(cls, value: str) -> str:
        if not value:
            raise ValueError("VECTOR_URI is required (set MILVUS_URI environment variable or pass it explicitly).")
        return value

config = Config()
