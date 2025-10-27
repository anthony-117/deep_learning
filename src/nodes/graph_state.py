from operator import add
from typing import TypedDict, Optional, Annotated

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
