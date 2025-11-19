from typing import Optional, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

from .vectordb import VectorStore
from .llm import LLMModel
from .nodes import (
    analyze_query,
    improve_prompt,
    retrieve_documents,
    grade_documents,
    generate_answer,
    check_hallucination,
    rewrite_query,
    decide_after_grading,
    decide_after_hallucination,
    detect_scraping_request,
    extract_scraping_params,
    scrape_and_ingest_papers,
    generate_scraping_summary,
    GraphState,
)

class RAGGraph:
    """
    LangGraph-based RAG workflow with self-correction and query rewriting.
    """

    def __init__(
            self,
            vector_store: VectorStore,
            llm: LLMModel,
            max_rewrites: int = 2,
            relevance_threshold: float = 0.5
    ):
        """
        Initialize the RAG graph.

        Args:
            vector_store: VectorStore instance for document retrieval
            llm: LLM instance for generation
            max_rewrites: Maximum number of query rewrites allowed
            relevance_threshold: Minimum relevance score for documents
        """
        self.vector_store = vector_store
        self.llm = llm.get_llm()
        self.system_prompt = llm.get_system_prompt()
        self.max_rewrites = max_rewrites
        self.relevance_threshold = relevance_threshold

        # Build the graph
        self.graph = self._build_graph()
        self.app = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with scraping capability."""
        workflow = StateGraph(GraphState)

        # Add scraping nodes
        workflow.add_node("detect_scraping", detect_scraping_request(self.llm))
        workflow.add_node("extract_params", extract_scraping_params(self.llm))
        workflow.add_node("scrape_papers", scrape_and_ingest_papers(self.vector_store))
        workflow.add_node("scraping_summary", generate_scraping_summary(self.llm))

        # Add existing RAG nodes
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("improve_prompt", improve_prompt(self.llm))
        workflow.add_node("retrieve", retrieve_documents(self.vector_store))
        workflow.add_node("grade_documents", grade_documents(self.llm))
        workflow.add_node("generate", generate_answer(self.llm, self.system_prompt))
        workflow.add_node("check_hallucination", check_hallucination(self.llm))
        workflow.add_node("rewrite_query", rewrite_query(self.llm))

        # Set entry point - now starts with scraping detection
        workflow.set_entry_point("detect_scraping")

        # Decision after scraping detection
        workflow.add_conditional_edges(
            "detect_scraping",
            self._decide_after_scraping_detection,
            {
                "scrape": "extract_params",
                "normal": "analyze_query"
            }
        )

        # Scraping flow
        workflow.add_edge("extract_params", "scrape_papers")
        workflow.add_edge("scrape_papers", "scraping_summary")
        workflow.add_edge("scraping_summary", END)

        # Normal RAG flow
        workflow.add_edge("analyze_query", "improve_prompt")
        workflow.add_edge("improve_prompt", "retrieve")
        workflow.add_edge("retrieve", "grade_documents")

        # Conditional edge after grading
        workflow.add_conditional_edges(
            "grade_documents",
            decide_after_grading(self.max_rewrites),
            {
                "generate": "generate",
                "rewrite": "rewrite_query",
                "end": END
            }
        )

        workflow.add_edge("generate", "check_hallucination")

        # Conditional edge after hallucination check
        workflow.add_conditional_edges(
            "check_hallucination",
            decide_after_hallucination(self.max_rewrites),
            {
                "end": END,
                "rewrite": "rewrite_query"
            }
        )

        workflow.add_edge("rewrite_query", "improve_prompt")

        return workflow

    def _decide_after_scraping_detection(self, state: GraphState) -> Literal["scrape", "normal"]:
        """
        Decide whether to scrape papers or proceed with normal RAG.

        Args:
            state: Current graph state

        Returns:
            "scrape" if user wants to scrape papers, "normal" otherwise
        """
        needs_scraping = state.get("needs_scraping", False)
        return "scrape" if needs_scraping else "normal"

    def invoke(self, question: str, chat_history: Optional[list[BaseMessage]] = None) -> dict:
        """
        Run the RAG graph with a question and optional chat history.

        Args:
            question: User's question
            chat_history: Optional list of previous messages for context

        Returns:
            Dictionary with answer, documents, and processing steps
        """
        initial_state: GraphState = {
            "question": question,
            "generation": None,
            "documents": [],
            "steps": [],
            "rewrite_count": 0,
            "relevance_scores": [],
            "answer_grounded": False,
            "chat_history": chat_history or [],
            # Scraping fields
            "needs_scraping": False,
            "scraping_params": None,
            "scraped_papers": None,
            "scraped_count": 0,
        }

        # Run the graph
        result = self.app.invoke(initial_state)

        return {
            "question": result["question"],
            "answer": result.get("generation", "I couldn't find a relevant answer."),
            "documents": result["documents"],
            "steps": result["steps"],
            "rewrites": result["rewrite_count"],
            "grounded": result.get("answer_grounded", False),
            "scraped_papers": result.get("scraped_papers"),
            "scraped_count": result.get("scraped_count", 0),
        }

    def stream(self, question: str, chat_history: Optional[list[BaseMessage]] = None):
        """
        Stream the RAG graph execution step-by-step.

        Args:
            question: User's question
            chat_history: Optional chat history

        Yields:
            State updates at each step of the graph
        """
        initial_state: GraphState = {
            "question": question,
            "generation": None,
            "documents": [],
            "steps": [],
            "rewrite_count": 0,
            "relevance_scores": [],
            "answer_grounded": False,
            "chat_history": chat_history or [],
            # Scraping fields
            "needs_scraping": False,
            "scraping_params": None,
            "scraped_papers": None,
            "scraped_count": 0,
        }

        for output in self.app.stream(initial_state):
            yield output