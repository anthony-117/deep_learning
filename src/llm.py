from langchain_core.prompts import ChatPromptTemplate, BaseChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel

from .config import config


class LLMModel:

    def __init__(self) -> None:
        self.llm: BaseChatModel = ChatGroq(
            api_key = config.GROQ_API_KEY,
            model = config.GEN_MODEL_ID,
            temperature = config.GEN_TEMPERATURE,
        )

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
