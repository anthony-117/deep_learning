# rag_processor.py
import os
from dotenv import load_dotenv

# --- PDF Processing ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Vector Store and Embeddings ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional embedding providers (install as needed)
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from langchain_cohere import CohereEmbeddings
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


# --- LLM and RAG Chain ---
from langchain_groq import ChatGroq

# Cerebras LLM (uses OpenAI-compatible API)
try:
    from langchain_openai import ChatOpenAI
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

class RAGProcessor:
    """
    A class to handle the entire RAG pipeline from PDF processing to answering queries.
    Supports both Groq and Cerebras LLM providers.
    """
    def __init__(self, llm_provider: str, llm_model: str, groq_api_key: str = None,
                 cerebras_api_key: str = None, temperature: float = 0.1):
        """
        Initializes the RAG processor with flexible LLM provider support.

        Args:
            llm_provider (str): The LLM provider ('groq' or 'cerebras').
            llm_model (str): The model name for the chosen provider.
            groq_api_key (str): The Groq API key (required if provider is 'groq').
            cerebras_api_key (str): The Cerebras API key (required if provider is 'cerebras').
            temperature (float): The temperature for the LLM's generation.
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
        self.vector_store = None
        self.retrieval_chain = None
        
        # 1. Initialize the LLM based on provider
        if llm_provider == "groq":
            if not groq_api_key:
                raise ValueError("Groq API key is required when using Groq provider.")
            self.llm = ChatGroq(
                temperature=temperature,
                model_name=llm_model,
                api_key=groq_api_key
            )
        elif llm_provider == "cerebras":
            if not CEREBRAS_AVAILABLE:
                raise ImportError("Cerebras LLM not available. Install with: pip install langchain-openai")
            if not cerebras_api_key:
                raise ValueError("Cerebras API key is required when using Cerebras provider.")
            self.llm = ChatOpenAI(
                temperature=temperature,
                model=llm_model,
                api_key=cerebras_api_key,
                base_url="https://api.cerebras.ai/v1"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}. Supported providers: groq, cerebras")
        
        # 2. Initialize the Embedding Model based on environment variables
        self.embedding_model = self._create_embedding_model()
        
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
        print(f"RAG Processor initialized with LLM: '{llm_provider}/{llm_model}' and embeddings: {self._get_embedding_info()}")

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
        print("Creating vector store...")
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model
        )
        print("Vector store created successfully.")
        
        # 4. Create the core RAG chain
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # 5. Create the retrieval chain with a configurable retriever
        retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})
        
        self.retrieval_chain = create_retrieval_chain(
            retriever, 
            question_answer_chain
        )
        print(f"RAG pipeline is ready. Retriever will use top_k={top_k}.")

    def ask_question(self, query: str) -> dict:
        """
        Asks a question to the RAG pipeline and returns the response.
        """
        if not self.retrieval_chain:
            raise RuntimeError("RAG pipeline has not been set up. Call setup_rag_pipeline() first.")
        
        print(f"Invoking chain with query: '{query}'")
        response = self.retrieval_chain.invoke({"input": query})
        return response

    def _create_embedding_model(self):
        """
        Creates an embedding model based on environment variables.

        Environment Variables:
        - EMBEDDING_PROVIDER: huggingface, openai, cohere (default: huggingface)
        - EMBEDDING_MODEL: model name (default: all-MiniLM-L6-v2)
        - EMBEDDING_DEVICE: cpu, cuda (default: cpu)
        - OPENAI_API_KEY: required for OpenAI embeddings
        - COHERE_API_KEY: required for Cohere embeddings
        """
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").lower()
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        device = os.getenv("EMBEDDING_DEVICE", "cpu")

        print(f"Creating embedding model: Provider={provider}, Model={model_name}, Device={device}")

        if provider == "huggingface":
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )

        elif provider == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI embeddings not available. Install with: pip install langchain-openai")

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI embeddings")

            return OpenAIEmbeddings(
                model=model_name,
                api_key=api_key
            )

        elif provider == "cohere":
            if not COHERE_AVAILABLE:
                raise ImportError("Cohere embeddings not available. Install with: pip install langchain-cohere")

            api_key = os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("COHERE_API_KEY environment variable is required for Cohere embeddings")

            return CohereEmbeddings(
                model=model_name,
                cohere_api_key=api_key
            )

        else:
            raise ValueError(f"Unsupported embedding provider: {provider}. "
                           f"Supported providers: huggingface, openai, cohere")

    def _get_embedding_info(self):
        """
        Returns a string describing the current embedding configuration.
        """
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return f"{provider}/{model_name}"