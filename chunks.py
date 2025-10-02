# rag_processor.py
import os
from dotenv import load_dotenv

# --- PDF Processing ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Vector Store and Embeddings ---
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    This version is locked to use the 'openai/gpt-oss-20b' model.
    """
    def __init__(self, groq_api_key: str, temperature: float = 0.1):
        """
        Initializes the RAG processor. The model is hardcoded.
        
        Args:
            groq_api_key (str): The Groq API key.
            temperature (float): The temperature for the LLM's generation.
        """
        if not groq_api_key:
            raise ValueError("Groq API key is required.")
        
        self.vector_store = None
        self.retrieval_chain = None
        
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
        print("RAG Processor initialized with fixed model 'openai/gpt-oss-20b'.")

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

    def setup_rag_pipeline_multiple(self, pdf_paths: list, chunk_size: int, chunk_overlap: int, top_k: int):
        """
        Processes multiple PDF files to build the RAG pipeline using custom settings.

        Args:
            pdf_paths (list): List of file paths to PDF files.
            chunk_size (int): The size of each text chunk.
            chunk_overlap (int): The overlap between adjacent chunks.
            top_k (int): The number of relevant chunks to retrieve.
        """
        if not pdf_paths:
            raise ValueError("No PDF paths provided")

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        print(f"Processing {len(pdf_paths)} PDF files...")

        all_chunks = []

        # Process each PDF file
        for i, pdf_path in enumerate(pdf_paths):
            print(f"Processing file {i+1}/{len(pdf_paths)}: {pdf_path}")

            # 1. Load the PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Add source information to each document
            for doc in documents:
                doc.metadata['source_file'] = os.path.basename(pdf_path)

            # 2. Split the document into chunks with configurable overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            all_chunks.extend(chunks)

            print(f"File {os.path.basename(pdf_path)} split into {len(chunks)} chunks.")

        print(f"Total chunks from all files: {len(all_chunks)}")

        # 3. Create a vector store from all chunks
        print("Creating vector store from all documents...")
        self.vector_store = FAISS.from_documents(
            documents=all_chunks,
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
        print(f"RAG pipeline is ready with {len(pdf_paths)} files. Retriever will use top_k={top_k}.")

    def ask_question(self, query: str) -> dict:
        """
        Asks a question to the RAG pipeline and returns the response.
        """
        if not self.retrieval_chain:
            raise RuntimeError("RAG pipeline has not been set up. Call setup_rag_pipeline() first.")
        
        print(f"Invoking chain with query: '{query}'")
        response = self.retrieval_chain.invoke({"input": query})
        return response