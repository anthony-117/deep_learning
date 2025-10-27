from .graph_state import GraphState

from ..vectordb import VectorStore


def retrieve_documents(vector_store: "VectorStore"):
    """
    Factory function that creates a retrieve documents node.

    Args:
        vector_store: VectorStore instance for document retrieval

    Returns:
        A function that retrieves documents from the vector store
    """
    def _retrieve(state: "GraphState") -> "GraphState":
        """Retrieve relevant documents from vector store."""
        question = state["question"]
        documents = vector_store.search(question)

        steps = [f"Retrieved {len(documents)} documents"]

        return {
            **state,
            "documents": documents,
            "steps": steps
        }

    return _retrieve