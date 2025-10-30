from typing import Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus

from langchain_core.embeddings import Embeddings

from pymilvus import utility, connections

from .config import config


class VectorStore:

    def __init__(
            self,
            embedding: Embeddings,
            collection_name: str = config.VECTOR_DB_COLLECTION,
            drop_old: bool = False,
    ) -> None:

        self.embedding = embedding
        self.collection_name: str = collection_name
        self.vectorstore: Optional[Milvus] = None

        # Try to connect and check if collection exists
        try:
            connections.connect(
                alias="default",
                host=config.VECTOR_HOST,
                port=config.VECTOR_PORT
            )

            if utility.has_collection(self.collection_name):
                # Connect to existing collection
                self.vectorstore = Milvus(
                    embedding_function=embedding,
                    collection_name=self.collection_name,
                    connection_args={"host": config.VECTOR_HOST, "port": config.VECTOR_PORT},
                )
        except Exception:
            # If connection fails, vectorstore remains None
            # Will be initialized later via create_from_documents
            pass

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
            connection_args={"host": config.VECTOR_HOST, "port": config.VECTOR_PORT},
            index_params={
                "index_type": "FLAT",
                "metric_type": "IP"  # use "L2" for Euclidean
            },
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

    def search(
            self,
            query: str,
            k: int = config.TOP_K
    ) -> list[Document]:
        if not self.vectorstore:
            raise RuntimeError(
                "Vector store not initialized. Call create_from_documents() first."
            )

        return self.vectorstore.similarity_search(query, k=k)
