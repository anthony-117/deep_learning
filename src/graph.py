from typing import Optional
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
        """Build the LangGraph workflow."""
        workflow = StateGraph(GraphState)

        # Add nodes using the imported node functions
        workflow.add_node("analyze_query", analyze_query)
        workflow.add_node("improve_prompt", improve_prompt(self.llm))
        workflow.add_node("retrieve", retrieve_documents(self.vector_store))
        workflow.add_node("grade_documents", grade_documents(self.llm))
        workflow.add_node("generate", generate_answer(self.llm, self.system_prompt))
        workflow.add_node("check_hallucination", check_hallucination(self.llm))
        workflow.add_node("rewrite_query", rewrite_query(self.llm))

        # Set entry point
        workflow.set_entry_point("analyze_query")

        # Add edges
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
            "chat_history": chat_history or []
        }

        # Run the graph
        result = self.app.invoke(initial_state)

        return {
            "question": result["question"],
            "answer": result.get("generation", "I couldn't find a relevant answer."),
            "documents": result["documents"],
            "steps": result["steps"],
            "rewrites": result["rewrite_count"],
            "grounded": result.get("answer_grounded", False)
        }

    def stream(self, question: str):
        """
        Stream the RAG graph execution step-by-step.

        Args:
            question: User's question

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
            "chat_history": []
        }

        for output in self.app.stream(initial_state):
            yield output