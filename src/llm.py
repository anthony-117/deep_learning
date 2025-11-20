from langchain_core.prompts import ChatPromptTemplate, BaseChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from .config import config


class LLMModel:

    def __init__(self) -> None:
        # Create rate limiter from config (30 requests per minute = 0.5 req/sec)
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=config.GEN_RATE_LIMIT,
            check_every_n_seconds=0.1,  # Check rate limit every 100ms
            max_bucket_size=5,  # Allow small bursts of up to 5 requests
        )

        self.llm: BaseChatModel = ChatGroq(
            api_key = config.GROQ_API_KEY,
            model = config.GEN_MODEL_ID,
            temperature = config.GEN_TEMPERATURE,
            rate_limiter = rate_limiter,
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
