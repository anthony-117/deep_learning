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
        self.top_k = 4  # Default value
        
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

        # 2.5. Enhance chunks with position information
        for i, chunk in enumerate(chunks):
            # Add chunk position info
            chunk.metadata['chunk_id'] = i
            # Calculate approximate line numbers based on content
            lines_before = chunk.page_content.count('\n')
            chunk.metadata['approx_lines'] = lines_before + 1
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
        self.top_k = top_k  # Store the top_k value
        retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})

        self.retrieval_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )
        print(f"RAG pipeline is ready. Retriever will use top_k={top_k}.")

    def ask_question(self, query: str) -> dict:
        """
        Asks a question to the RAG pipeline and returns the response with ordered context.
        """
        if not self.retrieval_chain:
            raise RuntimeError("RAG pipeline has not been set up. Call setup_rag_pipeline() first.")

        print(f"Invoking chain with query: '{query}'")

        # Get documents with similarity scores
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)

        # Create ordered context with relevance scores
        ordered_context = []
        for i, (doc, score) in enumerate(docs_with_scores):
            ordered_context.append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "relevance_percentage": round((1 - score) * 100, 2)  # Convert distance to relevance %
            })

        # Get the standard response from the chain
        response = self.retrieval_chain.invoke({"input": query})

        # Replace the context with our ordered version
        response["ordered_context"] = ordered_context

        return response