import os
from pydantic import BaseModel, Field
from typing import Optional


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ScraperSourceConfig(BaseModel):
    """Configuration for individual scraper sources."""
    enabled: bool = True
    max_results: int = 100
    rate_limit_requests: int = 10
    rate_limit_period: int = 60  # seconds
    cache_ttl: int = 3600  # 1 hour
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    parallel_downloads: int = 5
    api_key: Optional[str] = None


class ScraperConfig(BaseModel):
    """Configuration for paper scraping system."""
    # Output settings
    DOWNLOAD_DIR: str = Field(default_factory=lambda: os.getenv("SCRAPER_DOWNLOAD_DIR", "downloads/papers"))

    # Source configurations
    ARXIV: ScraperSourceConfig = Field(
        default_factory=lambda: ScraperSourceConfig(
            enabled=True,
            rate_limit_requests=10,
            max_results=100
        )
    )
    BIORXIV: ScraperSourceConfig = Field(
        default_factory=lambda: ScraperSourceConfig(
            enabled=True,
            rate_limit_requests=30,
            max_results=100
        )
    )
    MEDRXIV: ScraperSourceConfig = Field(
        default_factory=lambda: ScraperSourceConfig(
            enabled=True,
            rate_limit_requests=30,
            max_results=100
        )
    )
    ENGRXIV: ScraperSourceConfig = Field(
        default_factory=lambda: ScraperSourceConfig(
            enabled=False,  # Disabled by default
            rate_limit_requests=20,
            max_results=50,
            api_key=os.getenv("OSF_API_KEY")
        )
    )
    CHEMRXIV: ScraperSourceConfig = Field(
        default_factory=lambda: ScraperSourceConfig(
            enabled=True,
            rate_limit_requests=30,
            max_results=100
        )
    )

    # Global settings
    ENABLE_CACHE: bool = True
    PARALLEL_SOURCES: bool = True
    DEDUPLICATE_RESULTS: bool = True


class Config(BaseModel):
    HF_TOKEN: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    EMBED_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_HOST: str = Field(default_factory=lambda: os.getenv("MILVUS_HOST", "localhost"))
    VECTOR_PORT: str = Field(default_factory=lambda: os.getenv("MILVUS_PORT", "19530"))
    VECTOR_DB_COLLECTION: str = Field(default_factory=lambda: os.getenv("VECTOR_DB_COLLECTION", "docling_demo"))
    GROQ_API_KEY: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GEN_MODEL_ID: str = "openai/gpt-oss-120b"
    GEN_TEMPERATURE: float = 0.05
    GEN_RATE_LIMIT: float = 0.5  # requests per second (30 requests per minute = 0.5 req/sec)
    TOP_K: int = Field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    STORAGE_DIR: str = Field(default_factory=lambda: os.getenv("STORAGE_DIR", os.path.join(os.getcwd(), "processed")))
    DOCUMENT_STORE_PATH: str = Field(default_factory=lambda: os.getenv("DOCUMENT_STORE_PATH", os.path.join(os.getcwd(), "documents.db")))

    # Chunking configuration
    MAX_CHUNK_SIZE: int = Field(default_factory=lambda: int(os.getenv("MAX_CHUNK_SIZE", "1000")))

    # Paper scraping configuration
    SCRAPER: ScraperConfig = Field(default_factory=ScraperConfig)

    class Config:
        arbitrary_types_allowed = True

config = Config()
