from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

from .graph_state import GraphState

def improve_prompt(llm: BaseChatModel):
    """
    Factory function that creates a prompt improvement node.

    This node enhances the user's query by considering conversation history
    to generate a better search query for the vector database.

    Args:
        llm: Language model for prompt improvement

    Returns:
        A function that improves queries based on conversation context
    """

    def _improve(state: "GraphState") -> "GraphState":
        """
        Improve the user's query by considering conversation history.

        This creates a standalone, context-aware search query that:
        - Resolves pronouns and references from history
        - Incorporates relevant context from previous exchanges
        - Optimizes the query for vector search
        """
        question = state["question"]
        chat_history = state.get("chat_history", [])

        # If no chat history, return the original question
        if not chat_history:
            steps = ["No chat history - using original query"]
            return {
                **state,
                "steps": steps
            }

        # Format chat history
        history_text = ""
        if chat_history:
            for msg in chat_history[-6:]:  # Last 3 exchanges (6 messages)
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"

        # Prompt improvement template
        improve_prompt_template = ChatPromptTemplate.from_template(
            """Given the following conversation history and a follow-up question,
            rephrase the follow-up question to be a standalone question that can be used
            to search a vector database effectively.

            The standalone question should:
            - Resolve any pronouns or references to previous context
            - Include relevant context from the conversation history
            - Be clear and specific for semantic search
            - Focus on the core information need

            Conversation History:
            {history}

            Follow-up Question: {question}

            Standalone Question:"""
        )

        chain = improve_prompt_template | llm | StrOutputParser()

        improved_question = chain.invoke({
            "history": history_text,
            "question": question
        })

        steps = [f"Improved query with conversation context: '{improved_question}'"]

        return {
            **state,
            "question": improved_question,
            "steps": steps
        }

    return _improve
