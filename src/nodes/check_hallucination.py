from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .graph_state import GraphState


def check_hallucination(llm: BaseChatModel):
    def _check(state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        hallucination_prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing whether an answer is grounded in / supported by a set of facts.

            Set of facts:
            {documents}

            Answer: {generation}

            Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in the facts.
            Provide only 'yes' or 'no' as output."""
        )

        chain = hallucination_prompt | llm | StrOutputParser()

        context = "\n\n".join([doc.page_content for doc in documents])
        score = chain.invoke({
            "documents": context,
            "generation": generation
        })

        is_grounded = "yes" in score.lower()
        steps = [f"Hallucination check: {'passed' if is_grounded else 'failed'}"]

        return {
            **state,
            "answer_grounded": is_grounded,
            "steps": steps
        }

    return _check
