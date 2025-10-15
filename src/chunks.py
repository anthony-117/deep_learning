# rag_processor.py
import os
import pandas as pd
import numpy as np
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# --- Enhanced PDF Processing ---
import fitz  # PyMuPDF for advanced PDF parsing
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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

# Cerebras LLM (uses OpenAI-compatible API)
try:
    from langchain_openai import ChatOpenAI
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

# --- Optional advanced libraries for table extraction ---
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    print("Camelot not available. Install with: pip install camelot-py[cv]")

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    print("Tabula not available. Install with: pip install tabula-py")

try:
    from PIL import Image
    import cv2
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Vision libraries not available. Install with: pip install pillow opencv-python")

# Load environment variables from .env file
load_dotenv()

# Import the logger
try:
    from rag_logger import RAGLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    print("RAGLogger not available. Logging disabled.")

class RAGProcessor:
    """
    A class to handle the entire RAG pipeline from PDF processing to answering queries.
    Supports both Groq and Cerebras LLM providers.
    """
    def __init__(self, llm_provider: str, llm_model: str, groq_api_key: str = None,
                 cerebras_api_key: str = None, temperature: float = 0.1, vector_db: str = None):
        """
        Initializes the RAG processor with flexible LLM provider support.

        Args:
            llm_provider (str): The LLM provider ('groq' or 'cerebras').
            llm_model (str): The model name for the chosen provider.
            groq_api_key (str): The Groq API key (required if provider is 'groq').
            cerebras_api_key (str): The Cerebras API key (required if provider is 'cerebras').
            temperature (float): The temperature for the LLM's generation.
            vector_db (str): Vector database to use (overrides env variable).
        """
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        
        self.vector_store = None
        self.retrieval_chain = None
        self.top_k = 4  # Default value
        self.vector_db = vector_db or os.getenv('VECTOR_DB', 'faiss').lower()

        # Initialize logging
        self.logger = None
        self.session_id = None
        self.config_id = None
        if LOGGER_AVAILABLE:
            self.logger = RAGLogger()

        # Store config for logging
        self.config = {
            'llm_provider': llm_provider,
            'model': llm_model,
            'temp': temperature,
            'embedding_provider': os.getenv("EMBEDDING_PROVIDER", "huggingface"),
            'embedding_model': os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            'embedding_device': os.getenv("EMBEDDING_DEVICE", "cpu"),
            'vector_db': self.vector_db
        }

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
        
        # 3. Define the enhanced RAG Prompt Template for multimodal content
        system_prompt = (
            "you are an assistant for question-answering tasks. "
            "use the following retrieved context to answer the question. "
            "the context may include text, tables, and descriptions of images/diagrams. "
            "when referencing tables, preserve their structure in your response. "
            "when discussing images or diagrams, mention the visual elements described. "
            "if you don't know the answer, just say that you don't know. "
            "keep the answer comprehensive but concise.\n\n"
            "{context}"
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        print("RAG Processor initialized with enhanced multimodal capabilities.")

    def extract_tables_from_page(self, pdf_path: str, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from a specific PDF page using multiple methods."""
        tables = []

        # Method 1: Camelot (if available)
        if CAMELOT_AVAILABLE:
            try:
                camelot_tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))
                for i, table in enumerate(camelot_tables):
                    if table.df.shape[0] > 1 and table.df.shape[1] > 1:
                        tables.append({
                            'method': 'camelot',
                            'page': page_num,
                            'table_id': f"camelot_{page_num}_{i}",
                            'dataframe': table.df,
                            'confidence': getattr(table, 'accuracy', 0.0)
                        })
            except Exception as e:
                print(f"Camelot extraction failed for page {page_num}: {e}")

        # Method 2: Tabula (if available)
        if TABULA_AVAILABLE:
            try:
                tabula_tables = tabula.read_pdf(pdf_path, pages=page_num + 1, multiple_tables=True)
                for i, df in enumerate(tabula_tables):
                    if df.shape[0] > 1 and df.shape[1] > 1:
                        tables.append({
                            'method': 'tabula',
                            'page': page_num,
                            'table_id': f"tabula_{page_num}_{i}",
                            'dataframe': df,
                            'confidence': 0.8
                        })
            except Exception as e:
                print(f"Tabula extraction failed for page {page_num}: {e}")

        return tables

    def extract_images_from_page(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract and analyze images from a PDF page."""
        images = []

        if not VISION_AVAILABLE:
            return images

        try:
            image_list = page.get_images()

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)

                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        width, height = pix.width, pix.height

                        # Basic image analysis
                        size_category = self.categorize_image_size(width, height)
                        description = f"Image on page {page_num + 1}: {size_category}, dimensions {width}x{height}px"

                        images.append({
                            'page': page_num,
                            'image_id': f"img_{page_num}_{img_index}",
                            'width': width,
                            'height': height,
                            'description': description,
                            'size_category': size_category
                        })

                    pix = None

                except Exception as e:
                    print(f"Error processing image {img_index} on page {page_num}: {e}")

        except Exception as e:
            print(f"Error extracting images from page {page_num}: {e}")

        return images

    def categorize_image_size(self, width: int, height: int) -> str:
        """Categorize image by size to understand its likely importance."""
        area = width * height

        if area < 10000:
            return "small icon/symbol"
        elif area < 100000:
            return "medium figure/diagram"
        else:
            return "large chart/diagram"

    def format_table_as_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to readable text format."""
        try:
            df_clean = df.fillna('')
            table_text = df_clean.to_string(index=False, max_rows=50)

            if len(df) > 50:
                table_text += f"\n... (Table continues with {len(df)} total rows)"

            return table_text

        except Exception as e:
            print(f"Error formatting table: {e}")
            return str(df)

    def process_pdf_with_advanced_extraction(self, pdf_path: str) -> List[Document]:
        """Process PDF with enhanced extraction of tables and images."""
        documents = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text
                text = page.get_text()

                # Extract tables
                tables = self.extract_tables_from_page(pdf_path, page_num)

                # Extract images
                images = self.extract_images_from_page(page, page_num)

                # Create main text document
                if text.strip():
                    text_doc = Document(
                        page_content=text,
                        metadata={
                            'source': pdf_path,
                            'page': page_num,
                            'content_type': 'text',
                            'source_file': os.path.basename(pdf_path)
                        }
                    )
                    documents.append(text_doc)

                # Create table documents
                for table in tables:
                    table_text = self.format_table_as_text(table['dataframe'])
                    if table_text.strip():
                        table_doc = Document(
                            page_content=f"TABLE on page {page_num + 1}:\n{table_text}",
                            metadata={
                                'source': pdf_path,
                                'page': page_num,
                                'content_type': 'table',
                                'table_id': table['table_id'],
                                'extraction_method': table['method'],
                                'confidence': table.get('confidence', 0.0),
                                'source_file': os.path.basename(pdf_path)
                            }
                        )
                        documents.append(table_doc)

                # Create image documents
                for image in images:
                    image_doc = Document(
                        page_content=f"IMAGE on page {page_num + 1}: {image['description']}",
                        metadata={
                            'source': pdf_path,
                            'page': page_num,
                            'content_type': 'image',
                            'image_id': image['image_id'],
                            'size_category': image['size_category'],
                            'dimensions': f"{image['width']}x{image['height']}",
                            'source_file': os.path.basename(pdf_path)
                        }
                    )
                    documents.append(image_doc)

            doc.close()

        except Exception as e:
            print(f"Advanced extraction failed for {pdf_path}, using fallback: {e}")
            # Fallback to basic extraction
            loader = PyPDFLoader(pdf_path)
            fallback_docs = loader.load()
            for doc in fallback_docs:
                doc.metadata['source_file'] = os.path.basename(pdf_path)
                doc.metadata['content_type'] = 'text_fallback'
            documents.extend(fallback_docs)

        return documents

    def smart_chunk_documents(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """Smart chunking that respects content types and structure."""
        chunks = []

        # Separate documents by type
        text_docs = [doc for doc in documents if doc.metadata.get('content_type') in ['text', 'text_fallback']]
        table_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'table']
        image_docs = [doc for doc in documents if doc.metadata.get('content_type') == 'image']

        # Process text documents with standard chunking
        if text_docs:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            text_chunks = text_splitter.split_documents(text_docs)
            chunks.extend(text_chunks)

        # Keep tables as single chunks (don't split them)
        for table_doc in table_docs:
            if len(table_doc.page_content) > chunk_size * 2:
                # If table is very large, create a summary chunk
                lines = table_doc.page_content.split('\n')
                summary = '\n'.join(lines[:20]) + f"\n... (Large table with {len(lines)} total lines)"
                table_doc.page_content = summary
            chunks.append(table_doc)

        # Keep image descriptions as single chunks
        chunks.extend(image_docs)

        return chunks

    def setup_rag_pipeline_enhanced(self, pdf_paths: List[str], chunk_size: int, chunk_overlap: int, top_k: int):
        """Set up RAG pipeline with enhanced PDF processing for tables and images."""
        if not pdf_paths:
            raise ValueError("No PDF paths provided")

        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        print(f"Processing {len(pdf_paths)} PDF files with enhanced extraction...")
        all_documents = []

        for pdf_path in pdf_paths:
            print(f"Processing with advanced extraction: {os.path.basename(pdf_path)}")
            documents = self.process_pdf_with_advanced_extraction(pdf_path)
            all_documents.extend(documents)

        print(f"Total documents extracted: {len(all_documents)}")

        # Smart chunking
        all_chunks = self.smart_chunk_documents(all_documents, chunk_size, chunk_overlap)
        print(f"Created {len(all_chunks)} chunks with smart chunking")

        # Create vector store
        self.vector_store = FAISS.from_documents(
            documents=all_chunks,
            embedding=self.embedding_model
        )

        # Create retrieval chain
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})

        self.retrieval_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Update config with pipeline settings and log configuration
        self.config.update({
            'chunk_size': chunk_size,
            'overlap': chunk_overlap,
            'top_k': top_k,
            'enhanced': True
        })

        if self.logger:
            self.config_id = self.logger.log_configuration(self.config)
            self.session_id = self.logger.create_session(self.config_id, "Enhanced Pipeline Session")

        print(f"Enhanced RAG pipeline ready with {len(pdf_paths)} files and {len(all_chunks)} chunks.")

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
        print(f"Creating {self.vector_db} vector store...")
        self.vector_store = self._create_vector_store(chunks)
        print(f"{self.vector_db.title()} vector store created successfully.")
        
        # 4. Create the core RAG chain
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        
        # 5. Create the retrieval chain with a configurable retriever
        self.top_k = top_k  # Store the top_k value
        retriever = self.vector_store.as_retriever(search_kwargs={'k': top_k})

        self.retrieval_chain = create_retrieval_chain(
            retriever,
            question_answer_chain
        )

        # Update config with pipeline settings and log configuration
        self.config.update({
            'chunk_size': chunk_size,
            'overlap': chunk_overlap,
            'top_k': top_k,
            'enhanced': False
        })

        if self.logger:
            self.config_id = self.logger.log_configuration(self.config)
            self.session_id = self.logger.create_session(self.config_id, "Basic Pipeline Session")

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

        # Update config with pipeline settings and log configuration
        self.config.update({
            'chunk_size': chunk_size,
            'overlap': chunk_overlap,
            'top_k': top_k,
            'enhanced': False
        })

        if self.logger:
            self.config_id = self.logger.log_configuration(self.config)
            self.session_id = self.logger.create_session(self.config_id, "Multiple Files Pipeline Session")

        print(f"RAG pipeline is ready with {len(pdf_paths)} files. Retriever will use top_k={top_k}.")

    def ask_question(self, query: str) -> dict:
        """
        Asks a question to the RAG pipeline and returns the response with ordered context.
        """
        if not self.retrieval_chain:
            raise RuntimeError("RAG pipeline has not been set up. Call setup_rag_pipeline() first.")

        print(f"Invoking chain with query: '{query}'")

        import time
        start_time = time.time()

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

        # Calculate response time
        response_time = time.time() - start_time

        # Replace the context with our ordered version
        response["ordered_context"] = ordered_context

        # Log the query-response if logger is available
        if self.logger and self.session_id and self.config_id:
            try:
                self.logger.log_query_response(
                    session_id=self.session_id,
                    config_id=self.config_id,
                    question=query,
                    answer=response.get("answer", ""),
                    retrieved_chunks=ordered_context,
                    response_time=response_time
                )
            except Exception as e:
                print(f"Logging failed: {e}")

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