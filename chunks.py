# rag_processor.py
import os
from dotenv import load_dotenv

# --- PDF Processing ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Vector Store and Embeddings ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import other vector stores conditionally
try:
    from langchain_community.vectorstores import Qdrant
except ImportError:
    Qdrant = None

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None

try:
    from langchain_community.vectorstores import Pinecone
except ImportError:
    Pinecone = None

try:
    from langchain_community.vectorstores import Weaviate
except ImportError:
    Weaviate = None

try:
    from langchain_community.vectorstores import Milvus
except ImportError:
    Milvus = None

# --- LLM and RAG Chain ---
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

class RAGProcessor:
    """
    A class to handle the entire RAG pipeline from PDF processing to answering queries.
    Supports multiple vector databases through environment configuration.
    """
    def __init__(self, groq_api_key: str, temperature: float = 0.1, vector_db: str = None):
        """
        Initializes the RAG processor with configurable vector database.

        Args:
            groq_api_key (str): The Groq API key.
            temperature (float): The temperature for the LLM's generation.
            vector_db (str): Vector database to use (overrides env variable).
        """
        if not groq_api_key:
            raise ValueError("Groq API key is required.")
        
        self.vector_store = None
        self.retrieval_chain = None
        self.vector_db = vector_db or os.getenv('VECTOR_DB', 'faiss').lower()
        
        # 1. Initialize the LLM with the fixed model name
        self.llm = ChatGroq(
            temperature=temperature,
            model_name="openai/gpt-oss-20b", # <-- MODEL IS HARCODED HERE
            api_key=groq_api_key
        )
        
        # 2. Initialize the Embedding Model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # 3. Define the RAG Prompt Template
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "{context}"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        print(f"RAG Processor initialized with model 'openai/gpt-oss-20b' and vector DB '{self.vector_db}'.")

    def setup_rag_pipeline(self, pdf_path: str, chunk_size: int, chunk_overlap: int, top_k: int):
        """
        Processes a PDF file to build the RAG pipeline using custom settings.
        
        Args:
            pdf_path (str): The file path to the PDF.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between adjacent chunks.
            top_k (int): The number of relevant chunks to retrieve.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
            
        print(f"Processing PDF: {pdf_path}...")
        
        # 1. Load the PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # 2. Split the document into chunks with configurable overlap
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"PDF split into {len(chunks)} chunks with size {chunk_size} and overlap {chunk_overlap}.")
        
        # 3. Create a vector store from the chunks
        print(f"Creating {self.vector_db} vector store...")
        self.vector_store = self._create_vector_store(chunks)
        print(f"{self.vector_db.title()} vector store created successfully.")
        
        # 4. Create the core RAG chain
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # 5. Create the retrieval chain with a configurable retriever
        retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})
        
        self.retrieval_chain = create_retrieval_chain(
            retriever, 
            question_answer_chain
        )
        print(f"RAG pipeline is ready. Retriever will use top_k={top_k}.")

    def _create_vector_store(self, chunks):
        """
        Create vector store based on the configured database type.

        Args:
            chunks: Document chunks to index

        Returns:
            Vector store instance
        """
        if self.vector_db == 'faiss':
            return FAISS.from_documents(
                documents=chunks,
                embedding=self.embedding_model
            )

        elif self.vector_db == 'qdrant':
            if Qdrant is None:
                raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")

            url = os.getenv('QDRANT_URL', 'http://localhost:6333')
            api_key = os.getenv('QDRANT_API_KEY')
            collection_name = 'pdf_documents'

            return Qdrant.from_documents(
                chunks,
                self.embedding_model,
                url=url,
                api_key=api_key,
                collection_name=collection_name
            )

        elif self.vector_db == 'chroma':
            if Chroma is None:
                raise ImportError("chromadb not installed. Run: pip install chromadb")

            persist_directory = './chroma_db'
            collection_name = 'pdf_documents'

            return Chroma.from_documents(
                chunks,
                self.embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name
            )

        elif self.vector_db == 'pinecone':
            if Pinecone is None:
                raise ImportError("pinecone-client not installed. Run: pip install pinecone-client")

            import pinecone

            api_key = os.getenv('PINECONE_API_KEY')
            environment = os.getenv('PINECONE_ENVIRONMENT')
            index_name = 'pdf-documents'

            if not api_key or not environment:
                raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set")

            pinecone.init(api_key=api_key, environment=environment)
            return Pinecone.from_documents(chunks, self.embedding_model, index_name=index_name)

        elif self.vector_db == 'weaviate':
            if Weaviate is None:
                raise ImportError("weaviate-client not installed. Run: pip install weaviate-client")

            import weaviate

            url = os.getenv('WEAVIATE_URL', 'http://localhost:8080')
            api_key = os.getenv('WEAVIATE_API_KEY')

            client = weaviate.Client(
                url=url,
                auth_client_secret=weaviate.AuthApiKey(api_key) if api_key else None
            )

            return Weaviate.from_documents(
                chunks,
                self.embedding_model,
                client=client,
                by_text=False
            )

        elif self.vector_db == 'milvus':
            if Milvus is None:
                raise ImportError("pymilvus not installed. Run: pip install pymilvus")

            host = os.getenv('MILVUS_HOST', 'localhost')
            port = os.getenv('MILVUS_PORT', '19530')
            collection_name = 'pdf_documents'

            connection_args = {
                'host': host,
                'port': port
            }

            return Milvus.from_documents(
                chunks,
                self.embedding_model,
                collection_name=collection_name,
                connection_args=connection_args
            )

        else:
            raise ValueError(f"Unsupported vector database: {self.vector_db}. Supported: faiss, qdrant, chroma, pinecone, weaviate, milvus")

    def ask_question(self, query: str) -> dict:
        """
        Asks a question to the RAG pipeline and returns the response.
        """
        if not self.retrieval_chain:
            raise RuntimeError("RAG pipeline has not been set up. Call setup_rag_pipeline() first.")
        
        print(f"Invoking chain with query: '{query}'")
        response = self.retrieval_chain.invoke({"input": query})
        return response