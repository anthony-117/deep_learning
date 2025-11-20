from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from src import ScraperSourceConfig


class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None
    orcid: Optional[str] = None


class PaperMetadata(BaseModel):
    id: str  # Source-specific ID
    source: str  # "arxiv", "biorxiv", "medrxiv", "engrxiv"

    document_id: Optional[str] = None

    # Core information
    title: str
    authors: List[Author] = Field(default_factory=list)
    abstract: str = ""

    # Dates
    published_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None

    # URLs and identifiers
    pdf_url: Optional[str] = None
    html_url: Optional[str] = None
    doi: Optional[str] = None

    # Categorization
    categories: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    subjects: List[str] = Field(default_factory=list)

    # Metrics (if available)
    citation_count: Optional[int] = None
    download_count: Optional[int] = None

    # Source-specific metadata (flexible field)
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        authors_str = ", ".join([a.name for a in self.authors[:3]])
        if len(self.authors) > 3:
            authors_str += f" et al. ({len(self.authors)} total)"
        return f"[{self.source}] {self.title}\n  Authors: {authors_str}\n  ID: {self.id}"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BaseScraper(ABC):
    # todo i don't like config is not strongly typed
    def __init__(self, config: ScraperSourceConfig):
        self.config = config
        self.name = self._get_name()

    @abstractmethod
    def _get_name(self) -> str:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        **filters
    ) -> List[PaperMetadata]:
        pass

    @abstractmethod
    def download_paper(
        self,
        paper: PaperMetadata,
        output_dir: str
    ) -> Path:
        pass

    @abstractmethod
    def get_paper_metadata(self, paper_id: str) -> PaperMetadata:
        pass

    def validate_query(self, query: str) -> bool:
        if not query or not query.strip():
            return False
        return True

    def get_supported_categories(self) -> List[str]:
        return []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"