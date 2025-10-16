from typing import Literal

from .graph_state import GraphState


def decide_after_grading(max_rewrites: int):
    """
    Factory function that creates a decision function for after document grading.

    Args:
        max_rewrites: Maximum number of query rewrites allowed

    Returns:
        A function that decides the next step after grading
    """
    def _decide(state: "GraphState") -> Literal["generate", "rewrite", "end"]:
        """
        Decide next step after document grading.

        Returns:
            - "generate" if we have relevant documents
            - "rewrite" if no relevant docs and rewrites available
            - "end" if no relevant docs and max rewrites reached
        """
        if state["documents"]:
            return "generate"

        if state["rewrite_count"] < max_rewrites:
            return "rewrite"

        return "end"

    return _decide


def decide_after_hallucination(max_rewrites: int):
    """
    Factory function that creates a decision function for after hallucination check.

    Args:
        max_rewrites: Maximum number of query rewrites allowed

    Returns:
        A function that decides the next step after hallucination check
    """
    def _decide(state: "GraphState") -> Literal["end", "rewrite"]:
        """
        Decide next step after hallucination check.

        Returns:
            - "end" if answer is grounded or max rewrites reached
            - "rewrite" if answer is hallucinated and rewrites available
        """
        if state["answer_grounded"]:
            return "end"

        if state["rewrite_count"] < max_rewrites:
            return "rewrite"

        return "end"

    return _decide