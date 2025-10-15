from typing import Optional
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from .config import config


class VectorStore:

    def __init__(
        self,
        embedding: HuggingFaceEmbeddings,
        collection_name: str = "docling_demo"
    ) -> None:
        self.embedding: HuggingFaceEmbeddings = embedding
        self.collection_name: str = collection_name
        self.vectorstore: Optional[Milvus] = None


    def create_from_documents(
        self,
        documents: list[Document],
        drop_old: bool = True
    ) -> Milvus:
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embedding,
            collection_name=self.collection_name,
            connection_args={"uri": config.VECTOR_URI},
            index_params={"index_type": "FLAT"},
            drop_old=drop_old,
        )
        return self.vectorstore

    def add_documents(self, documents: list[Document]) -> list[str]:
        if not self.vectorstore:
            raise RuntimeError(
                "Vector store not initialized. Call create_from_documents() first."
            )

        if not documents:
            raise ValueError("Documents list cannot be empty")

        return self.vectorstore.add_documents(documents)

    def get_retriever(self, top_k: int = 5) -> VectorStoreRetriever:
        if not self.vectorstore:
            raise RuntimeError(
                "Vector store not initialized. Call create_from_documents() first."
            )

        return self.vectorstore.as_retriever(search_kwargs={"k": top_k})

    def similarity_search(
        self,
        query: str,
        k: int = 5
    ) -> list[Document]:
        if not self.vectorstore:
            raise RuntimeError(
                "Vector store not initialized. Call create_from_documents() first."
            )

        return self.vectorstore.similarity_search(query, k=k)
