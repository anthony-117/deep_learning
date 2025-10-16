from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .graph_state import GraphState
from langchain_core.language_models import BaseChatModel


def grade_documents(llm: "BaseChatModel"):
    """
    Factory function that creates a grade documents node.

    Args:
        llm: Language model for grading

    Returns:
        A function that grades document relevance
    """
    def _grade(state: "GraphState") -> "GraphState":
        """
        Grade the relevance of retrieved documents.

        Uses an LLM to determine if documents are relevant to the question.
        """
        question = state["question"]
        documents = state["documents"]

        # Grading prompt
        grade_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing relevance of a retrieved document to a user question.

            Retrieved document:
            {document}

            User question: {question}

            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.

            Provide only 'yes' or 'no' as output."""
        )

        chain = grade_prompt | llm | StrOutputParser()

        # Score each document
        relevance_scores = []
        filtered_docs = []

        for doc in documents:
            score = chain.invoke({
                "document": doc.page_content,
                "question": question
            })

            if "yes" in score.lower():
                relevance_scores.append(1.0)
                filtered_docs.append(doc)
            else:
                relevance_scores.append(0.0)

        steps = [f"Graded documents: {len(filtered_docs)}/{len(documents)} relevant"]

        return {
            **state,
            "documents": filtered_docs,
            "relevance_scores": relevance_scores,
            "steps": steps
        }

    return _grade