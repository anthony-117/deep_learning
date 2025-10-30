from langchain_core.prompts import ChatPromptTemplate, BaseChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

try:
    from langchain_groq import ChatGroq  # type: ignore
except Exception:  # pragma: no cover
    ChatGroq = None  # type: ignore

from .config import config


class LLMModel:

    def __init__(self) -> None:
        if ChatGroq and config.GROQ_API_KEY:
            self.llm: BaseChatModel | RunnableLambda = ChatGroq(
                api_key=config.GROQ_API_KEY,
                model=config.GEN_MODEL_ID,
                temperature=config.GEN_TEMPERATURE,
            )
        else:
            # Fallback runnable that returns a clear message when LLM is not configured
            self.llm = RunnableLambda(lambda _: "LLM not configured. Set GROQ_API_KEY or update LLM backend.")

        self.system_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
            """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            Consider the conversation history to provide contextual answers.
            If you don't know the answer, just say that you don't know.
            Keep the answer concise and relevant to the conversation.
            {history}
            Question: {question}
    
            Retrieved Context: {context}
    
            Answer:"""
        )

    def get_llm(self) -> BaseChatModel:
        return self.llm

    def get_system_prompt(self) -> BaseChatPromptTemplate:
        return self.system_prompt
