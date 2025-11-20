from operator import add
from typing import TypedDict, Optional, Annotated, Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    question: str
    generation: Optional[str]
    documents: list[Document]
    steps: Annotated[list[str], add]
    rewrite_count: int
    relevance_scores: list[float]
    answer_grounded: bool
    chat_history: list[BaseMessage]  # Conversation memory

    # Scraping-related fields
    needs_scraping: bool  # Whether user is requesting to scrape papers
    scraping_params: Optional[dict[str, Any]]  # Extracted scraping parameters
    scraped_papers: Optional[list[Any]]  # List of scraped papers
    scraped_count: int  # Number of papers scraped
