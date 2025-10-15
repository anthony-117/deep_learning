from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from .config import config


class LLM:

    def __init__(self) -> None:
        self.llm: HuggingFaceEndpoint = HuggingFaceEndpoint(
            repo_id=config.GEN_MODEL_ID,
            huggingfacehub_api_token=config.HF_TOKEN,
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

    def get_llm(self) -> HuggingFaceEndpoint:
        return self.llm

    def get_prompt(self) -> ChatPromptTemplate:
        return self.system_prompt
