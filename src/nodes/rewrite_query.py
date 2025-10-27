
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .graph_state import GraphState
from langchain_core.language_models import BaseChatModel


def rewrite_query(llm: "BaseChatModel"):
    """
    Factory function that creates a query rewrite node.

    Args:
        llm: Language model for query rewriting

    Returns:
        A function that rewrites queries to improve retrieval
    """
    def _rewrite(state: "GraphState") -> "GraphState":
        """Rewrite the query to improve retrieval."""
        question = state["question"]

        # Query rewrite prompt
        rewrite_prompt = ChatPromptTemplate.from_template(
            """You are a question re-writer that converts an input question to a better version that is optimized
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.

            Here is the initial question:
            {question}

            Formulate an improved question:"""
        )

        chain = rewrite_prompt | llm | StrOutputParser()

        better_question = chain.invoke({"question": question})

        steps = [f"Rewrote query (attempt {state['rewrite_count'] + 1})"]

        return {
            **state,
            "question": better_question,
            "rewrite_count": state["rewrite_count"] + 1,
            "steps": steps
        }

    return _rewrite