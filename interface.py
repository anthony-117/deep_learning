# streamlit_app.py
import streamlit as st
from chunks import RAGProcessor # Import the updated class
import os

# --- Page Configuration ---
st.set_page_config(page_title="Customizable PDF Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("Chat with your PDF using Groq and RAG")

# --- Load API Key ---
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.warning("GROQ_API_KEY is not set. Please set it as an environment variable or in a .env file.")

# --- Session State Management ---
if 'rag_processor' not in st.session_state:
    st.session_state.rag_processor = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# To store the configuration used for the current session
if "config" not in st.session_state:
    st.session_state.config = None

# --- Sidebar for PDF Upload and Configuration ---
with st.sidebar:
    st.header("1. Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    st.header("2. Configure RAG Pipeline")
    with st.expander("Settings", expanded=True):
        # LLM Configuration
        st.subheader("LLM Settings")

        # LLM Provider Selection
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["groq", "cerebras"],
            index=0,
            help="Choose your LLM provider. Make sure you have the required API keys set in your .env file."
        )

        # Model selection based on LLM provider
        llm_models = {
            "groq": [
                "openai/gpt-oss-20b",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768"
            ],
            "cerebras": [
                "gpt-oss-120b",
                "llama3.1-8b",
                "llama3.1-70b",
                "llama-3.3-70b"
                "llama-4-maverick-17b-128e-instruct",
                "llama-4-scout-17b-16e-instruct",
                "qwen-3-32b",
                "qwen-3-235b-a22b-instruct-2507",
                "qwen-3-235b-a22b-thinking-2507",
            ]
        }

        llm_model = st.selectbox(
            "LLM Model",
            options=llm_models[llm_provider],
            index=0,
            help=f"Select the model for {llm_provider}"
        )

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

        # API Key validation for LLM
        if llm_provider == "groq":
            groq_key = os.getenv("GROQ_API_KEY")
            if not groq_key:
                st.warning("⚠️ GROQ_API_KEY not found in environment variables")
        elif llm_provider == "cerebras":
            cerebras_key = os.getenv("CEREBRAS_API_KEY")
            if not cerebras_key:
                st.warning("⚠️ CEREBRAS_API_KEY not found in environment variables")

        # Embedding Configuration
        st.subheader("Embedding Settings")

        # Embedding Provider Selection
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["huggingface", "openai", "cohere"],
            index=0,
            help="Choose your embedding provider. Make sure you have the required API keys set in your .env file."
        )

        # Model selection based on provider
        embedding_models = {
            "huggingface": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "all-MiniLM-L12-v2",
                "BAAI/bge-small-en-v1.5",
                "BAAI/bge-base-en-v1.5",
                "sentence-transformers/all-distilroberta-v1"
            ],
            "cohere": [
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v2.0",
                "embed-english-light-v3.0"
            ]
        }

        embedding_model = st.selectbox(
            "Embedding Model",
            options=embedding_models[embedding_provider],
            index=0,
            help=f"Select the embedding model for {embedding_provider}"
        )

        # Device selection (only for HuggingFace)
        if embedding_provider == "huggingface":
            embedding_device = st.selectbox(
                "Device",
                options=["cpu", "cuda"],
                index=0,
                help="Choose CPU or CUDA for HuggingFace models"
            )
        else:
            embedding_device = "cpu"  # Not applicable for API-based providers

        # API Key validation
        if embedding_provider == "openai":
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                st.warning("⚠️ OPENAI_API_KEY not found in environment variables")
        elif embedding_provider == "cohere":
            cohere_key = os.getenv("COHERE_API_KEY")
            if not cohere_key:
                st.warning("⚠️ COHERE_API_KEY not found in environment variables")
        else:  # huggingface
            st.info("ℹ️ HuggingFace embeddings run locally - no API key required")

        # RAG Configuration
        st.subheader("RAG Settings")
        chunk_size = st.slider("Chunk Size", min_value=256, max_value=2048, value=1000, step=128)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=512, value=200, step=32)
        top_k = st.slider("Top 'K' Chunks", min_value=1, max_value=10, value=4, step=1)
        
        # Add a check to ensure overlap is less than chunk size
        if chunk_overlap >= chunk_size:
            st.warning("Chunk Overlap should be smaller than Chunk Size.")

    if uploaded_file and st.button("Process PDF with Settings"):
        if chunk_overlap >= chunk_size:
            st.error("Processing failed: Chunk Overlap must be smaller than Chunk Size.")
        else:
            with st.spinner("Processing PDF with your settings... This may take a moment."):
                # Save the uploaded file temporarily to pass its path
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    # Set environment variables for embedding configuration
                    os.environ["EMBEDDING_PROVIDER"] = embedding_provider
                    os.environ["EMBEDDING_MODEL"] = embedding_model
                    os.environ["EMBEDDING_DEVICE"] = embedding_device

                    # Store current configuration
                    current_config = {
                        "llm_provider": llm_provider, "model": llm_model, "temp": temperature,
                        "chunk_size": chunk_size, "overlap": chunk_overlap, "top_k": top_k,
                        "embedding_provider": embedding_provider, "embedding_model": embedding_model,
                        "embedding_device": embedding_device
                    }
                    st.session_state.config = current_config

                    # Initialize and setup the RAG pipeline with parameters from the UI
                    processor = RAGProcessor(
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        groq_api_key=groq_api_key if llm_provider == "groq" else None,
                        cerebras_api_key=os.getenv("CEREBRAS_API_KEY") if llm_provider == "cerebras" else None,
                        temperature=temperature
                    )
                    processor.setup_rag_pipeline(
                        pdf_path=uploaded_file.name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k
                    )
                    
                    st.session_state.rag_processor = processor
                    
                    # Reset chat history
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"PDF '{uploaded_file.name}' processed! How can I help you?"}
                    ]
                    st.success("PDF processed successfully!")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    # Clean up the temporary file
                    if os.path.exists(uploaded_file.name):
                        os.remove(uploaded_file.name)

# --- Main Chat Interface ---

# Display initial message if no PDF is processed yet
if not st.session_state.rag_processor:
    st.info("Please upload a PDF and click 'Process PDF with Settings' in the sidebar to start.")
else:
    # Display the configuration that was used for the current chat session
    config = st.session_state.config
    st.info(f"**Current Configuration:** LLM: `{config.get('llm_provider', 'N/A')}/{config['model']}`, Temp: `{config['temp']}`, "
            f"Embedding: `{config.get('embedding_provider', 'N/A')}/{config.get('embedding_model', 'N/A')}`, "
            f"Chunk Size: `{config['chunk_size']}`, Overlap: `{config['overlap']}`, Top K: `{config['top_k']}`")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
             with st.expander("Show Retrieved Context"):
                st.json(message["context"])

# Chat input for the user
if prompt := st.chat_input("Ask a question about your document...", disabled=not st.session_state.rag_processor):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_processor.ask_question(prompt)
            answer = response.get('answer', "Sorry, I couldn't find an answer.")
            context = response.get('context', {})
            
            st.markdown(answer)

            with st.expander("Show Retrieved Context"):
                st.json(context)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "context": context
            })