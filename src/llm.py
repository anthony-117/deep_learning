from langchain_core.prompts import ChatPromptTemplate
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
            "Context information is below.\n"
            "---------------------\n"
            "{context}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {input}\n"
            "Answer:\n"
        )

    def get_llm(self) -> BaseChatModel:
        return self.llm

    def get_prompt(self) -> ChatPromptTemplate:
        return self.system_prompt
