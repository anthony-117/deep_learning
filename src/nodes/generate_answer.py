from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage

from .graph_state import GraphState
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import  BaseChatPromptTemplate


def generate_answer(llm: BaseChatModel, system_prompt: BaseChatPromptTemplate):
    def _generate(state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", [])

        # Format documents
        context = "\n\n".join([doc.page_content for doc in documents])

        # Format chat history
        history_text = ""
        if chat_history:
            history_text = "\n\nConversation History:\n"
            for msg in chat_history[-4:]:  # Last 2 exchanges (4 messages)
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                history_text += f"{role}: {msg.content}\n"

        # Generation prompt with conversation context
        chain = system_prompt | llm | StrOutputParser()

        generation = chain.invoke({
            "question": question,
            "context": context,
            "history": history_text
        })

        steps = ["Generated answer from documents with conversation context"]

        return {
            **state,
            "generation": generation,
            "steps": steps
        }

    return _generate