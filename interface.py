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
        model_name = "openai/gpt-oss-20b"
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

        # Vector Database Configuration
        st.subheader("Vector Database")
        vector_db_options = ['faiss', 'qdrant', 'chroma', 'weaviate', 'milvus']
        selected_vector_db = st.selectbox(
            "Choose Vector Database",
            options=vector_db_options,
            index=0,
            help="Select the vector database for document storage and retrieval"
        )

        # Show requirements for selected vector database
        if selected_vector_db != 'faiss':
            db_requirements = {
                'qdrant': "Requires: pip install qdrant-client",
                'chroma': "Requires: pip install chromadb",
                'pinecone': "Requires: pip install pinecone-client + API keys",
                'weaviate': "Requires: pip install weaviate-client",
                'milvus': "Requires: pip install pymilvus + running Milvus server"
            }
            st.info(f"📦 {db_requirements.get(selected_vector_db, '')}")

            if selected_vector_db in ['pinecone', 'qdrant', 'weaviate']:
                st.warning("⚠️ Make sure to configure the required environment variables in .env file")

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
                    # Store current configuration
                    current_config = {
                        "model": "openai/gpt-oss-20b", "temp": temperature, "chunk_size": chunk_size,
                        "overlap": chunk_overlap, "top_k": top_k, "vector_db": selected_vector_db
                    }
                    st.session_state.config = current_config

                    # Initialize and setup the RAG pipeline with parameters from the UI
                    processor = RAGProcessor(
                        groq_api_key=groq_api_key,
                        temperature=temperature,
                        vector_db=selected_vector_db
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
    st.info(f"**Current Configuration:** Model: `{config['model']}`, Vector DB: `{config['vector_db']}`, Temp: `{config['temp']}`, "
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