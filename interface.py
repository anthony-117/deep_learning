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
    st.header("1. Upload your Documents")

    # Initialize session state for upload modal
    if 'show_upload_modal' not in st.session_state:
        st.session_state.show_upload_modal = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'folder_path' not in st.session_state:
        st.session_state.folder_path = None
    if 'upload_mode' not in st.session_state:
        st.session_state.upload_mode = None

    # Upload button to open modal
    if st.button("📎 Upload Documents", type="primary", use_container_width=True):
        st.session_state.show_upload_modal = True

    # Display current upload status
    if st.session_state.uploaded_files:
        st.success(f"✅ {len(st.session_state.uploaded_files)} file(s) selected")
    elif st.session_state.folder_path:
        if os.path.exists(st.session_state.folder_path):
            pdf_count = len([f for f in os.listdir(st.session_state.folder_path) if f.lower().endswith('.pdf')])
            st.success(f"✅ Folder selected ({pdf_count} PDF files)")
        else:
            st.error("❌ Selected folder no longer exists")
    else:
        st.info("📄 No documents selected")

    uploaded_files = st.session_state.uploaded_files
    folder_path = st.session_state.folder_path

    st.header("2. Configure RAG Pipeline")
    with st.expander("Settings", expanded=True):
        # Processing Mode
        st.subheader("Processing Mode")
        use_enhanced_processing = st.checkbox(
            "Enable Enhanced Processing",
            value=True,
            help="Extract tables, images, and diagrams in addition to text"
        )

        if use_enhanced_processing:
            st.info("📊 Enhanced mode extracts tables, images, and structured content")
        else:
            st.info("📄 Basic mode extracts text content only")

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
                "llama-3.3-70b",
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
            options=["huggingface", "cohere"],
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

        if embedding_provider == "cohere":
            cohere_key = os.getenv("COHERE_API_KEY")
            if not cohere_key:
                st.warning("⚠️ COHERE_API_KEY not found in environment variables")
        else:  # huggingface
            st.info("ℹ️ HuggingFace embeddings run locally - no API key required")

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

    # Determine if we have files to process
    has_files = (uploaded_files and len(uploaded_files) > 0) or (folder_path and os.path.exists(folder_path) and any(f.lower().endswith('.pdf') for f in os.listdir(folder_path)))

    if has_files and st.button("Process Documents with Settings"):
        if chunk_overlap >= chunk_size:
            st.error("Processing failed: Chunk Overlap must be smaller than Chunk Size.")
        else:
            with st.spinner("Processing documents with your settings... This may take a moment."):
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
                        "embedding_device": embedding_device, "vector_db": selected_vector_db
                    }
                    st.session_state.config = current_config

                    # Initialize the RAG processor
                    processor = RAGProcessor(
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        groq_api_key=groq_api_key if llm_provider == "groq" else None,
                        cerebras_api_key=os.getenv("CEREBRAS_API_KEY") if llm_provider == "cerebras" else None,
                        temperature=temperature,
                        vector_db=selected_vector_db
                    )

                    # Process files based on upload type and processing type
                    if folder_path:
                        # Process all PDFs in the folder
                        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                        file_paths = [os.path.join(folder_path, f) for f in pdf_files]

                        if use_enhanced_processing:
                            processor.setup_rag_pipeline_enhanced(
                                pdf_paths=file_paths,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k
                            )
                            processing_type = "Enhanced"
                        else:
                            processor.setup_rag_pipeline_multiple(
                                pdf_paths=file_paths,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k
                            )
                            processing_type = "Basic"

                        file_count = len(pdf_files)
                        success_message = f"{processing_type} processing completed: {file_count} PDF files from folder!"

                    else:
                        # Process uploaded files (single or multiple)
                        temp_paths = []

                        for i, uploaded_file in enumerate(uploaded_files):
                            temp_path = f"temp_file_{i}_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)

                        if use_enhanced_processing and len(temp_paths) >= 1:
                            # Use enhanced processing for single or multiple files
                            processor.setup_rag_pipeline_enhanced(
                                pdf_paths=temp_paths,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k
                            )
                            processing_type = "Enhanced"
                        elif len(temp_paths) == 1:
                            # Single file - use basic method
                            processor.setup_rag_pipeline(
                                pdf_path=temp_paths[0],
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k
                            )
                            processing_type = "Basic"
                        else:
                            # Multiple files - use basic multiple method
                            processor.setup_rag_pipeline_multiple(
                                pdf_paths=temp_paths,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k
                            )
                            processing_type = "Basic"

                        file_count = len(uploaded_files)
                        success_message = f"{processing_type} processing completed: {file_count} PDF file{'s' if file_count > 1 else ''}!"

                        # Clean up temporary files
                        for temp_path in temp_paths:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

                    st.session_state.rag_processor = processor

                    # Reset chat history
                    st.session_state.messages = [
                        {"role": "assistant", "content": f"{success_message} How can I help you?"}
                    ]
                    st.success(success_message)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    # Clean up any temporary files in case of error
                    if 'temp_paths' in locals():
                        for temp_path in temp_paths:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)

# --- Main Chat Interface ---

# Display initial message if no documents are processed yet
if not st.session_state.rag_processor:
    st.info("Please upload documents and click 'Process Documents with Settings' in the sidebar to start.")
else:
    # Display the configuration that was used for the current chat session
    config = st.session_state.config
    processing_mode = "Enhanced" if config.get('enhanced', False) else "Basic"
    st.info(f"**Current Configuration:** Mode: `{processing_mode}`, LLM: `{config.get('llm_provider', 'N/A')}/{config['model']}`,  Vector DB: `{config.get('vector_db', 'N/A')}`, Temp: `{config['temp']}`, "
            f"Embedding: `{config.get('embedding_provider', 'N/A')}/{config.get('embedding_model', 'N/A')}`, "
            f"Chunk Size: `{config['chunk_size']}`, Overlap: `{config['overlap']}`, Top K: `{config['top_k']}`")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "ordered_context" in message and message["ordered_context"]:
            with st.expander("Show Retrieved Context (Ordered by Relevance)"):
                for chunk in message["ordered_context"]:
                    st.markdown(f"**Rank {chunk['rank']} - Relevance: {chunk['relevance_percentage']}%**")
                    st.markdown(f"*Similarity Score: {chunk['similarity_score']:.4f}*")
                    with st.container():
                        st.text_area(
                            f"Chunk {chunk['rank']} Content",
                            value=chunk['content'],
                            height=100,
                            disabled=True,
                            key=f"hist_chunk_{chunk['rank']}_{hash(chunk['content'][:50])}"
                        )
                    if chunk['metadata']:
                        page_num = chunk['metadata'].get('page', 'Unknown')
                        source_file = chunk['metadata'].get('source', 'Unknown')
                        chunk_id = chunk['metadata'].get('chunk_id', 'Unknown')
                        approx_lines = chunk['metadata'].get('approx_lines', 'Unknown')
                        filename = os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown'

                        st.markdown(f"""
                        <div style="background-color: #e8f4fd; border: 1px solid #1f77b4; padding: 12px; border-radius: 6px; margin: 8px 0; color: #1f77b4;">
                            <div style="font-weight: bold; margin-bottom: 4px;">
                                📄 <span style="color: #d62728;">{filename}</span> |
                                📍 Page <span style="color: #d62728;">{page_num + 1 if isinstance(page_num, int) else page_num}</span> |
                                📝 ~{approx_lines} lines |
                                🔍 Chunk #{chunk_id}
                            </div>
                            <div style="font-size: 0.85em; color: #666; margin-top: 4px;">
                                Text Preview: "{chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}"
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.divider()
        elif "context" in message:
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
            ordered_context = response.get('ordered_context', [])

            st.markdown(answer)

            with st.expander("Show Retrieved Context (Ordered by Relevance)"):
                if ordered_context:
                    for chunk in ordered_context:
                        st.markdown(f"**Rank {chunk['rank']} - Relevance: {chunk['relevance_percentage']}%**")
                        st.markdown(f"*Similarity Score: {chunk['similarity_score']:.4f}*")
                        with st.container():
                            st.text_area(
                                f"Chunk {chunk['rank']} Content",
                                value=chunk['content'],
                                height=100,
                                disabled=True,
                                key=f"chunk_{chunk['rank']}_{hash(chunk['content'][:50])}"
                            )
                        if chunk['metadata']:
                            page_num = chunk['metadata'].get('page', 'Unknown')
                            source_file = chunk['metadata'].get('source', 'Unknown')
                            chunk_id = chunk['metadata'].get('chunk_id', 'Unknown')
                            approx_lines = chunk['metadata'].get('approx_lines', 'Unknown')
                            filename = os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown'

                            st.markdown(f"""
                            <div style="background-color: #e8f4fd; border: 1px solid #1f77b4; padding: 12px; border-radius: 6px; margin: 8px 0; color: #1f77b4;">
                                <div style="font-weight: bold; margin-bottom: 4px;">
                                    📄 <span style="color: #d62728;">{filename}</span> |
                                    📍 Page <span style="color: #d62728;">{page_num + 1 if isinstance(page_num, int) else page_num}</span> |
                                    📝 ~{approx_lines} lines |
                                    🔍 Chunk #{chunk_id}
                                </div>
                                <div style="font-size: 0.85em; color: #666; margin-top: 4px;">
                                    Text Preview: "{chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}"
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.divider()
                else:
                    st.write("No context retrieved.")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "ordered_context": ordered_context
            })

# --- Upload Modal ---
if st.session_state.show_upload_modal:
    @st.dialog("📁 Upload Documents")
    def upload_modal():
        st.markdown("### Choose how you want to upload your documents:")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "📁\n\n**Upload Folder**\n\nSelect a folder containing PDF files",
                use_container_width=True,
                type="secondary",
                key="folder_upload_btn"
            ):
                st.session_state.upload_mode = "folder"
                st.rerun()

        with col2:
            if st.button(
                "📄\n\n**Upload Files**\n\nSelect individual PDF files",
                use_container_width=True,
                type="secondary",
                key="file_upload_btn"
            ):
                st.session_state.upload_mode = "files"
                st.rerun()

        # Handle folder upload
        if st.session_state.upload_mode == "folder":
            st.markdown("---")
            st.markdown("### 📁 Folder Upload")

            # Use file uploader for multiple files to simulate folder upload
            uploaded_folder_files = st.file_uploader(
                "Choose all PDF files from your folder:",
                type="pdf",
                accept_multiple_files=True,
                help="Select all PDF files from the folder you want to process"
            )

            col1, col2, col3 = st.columns([1, 1, 1])

            if uploaded_folder_files:
                st.success(f"✅ {len(uploaded_folder_files)} file(s) selected")
                for file in uploaded_folder_files:
                    st.write(f"📄 {file.name}")

                with col2:
                    if st.button("✅ Select Files", type="primary", use_container_width=True):
                        st.session_state.uploaded_files = uploaded_folder_files
                        st.session_state.folder_path = None
                        st.session_state.show_upload_modal = False
                        st.session_state.upload_mode = None
                        st.success("Files selected!")
                        st.rerun()

            with col3:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_upload_modal = False
                    st.session_state.upload_mode = None
                    st.rerun()

        # Handle file upload
        elif st.session_state.upload_mode == "files":
            st.markdown("---")
            st.markdown("### 📄 File Upload")

            uploaded_files = st.file_uploader(
                "Choose PDF files:",
                type="pdf",
                accept_multiple_files=True,
                help="You can select multiple PDF files"
            )

            col1, col2, col3 = st.columns([1, 1, 1])

            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} file(s) selected")
                for file in uploaded_files:
                    st.write(f"📄 {file.name}")

                with col2:
                    if st.button("✅ Upload Files", type="primary", use_container_width=True):
                        st.session_state.uploaded_files = uploaded_files
                        st.session_state.folder_path = None
                        st.session_state.show_upload_modal = False
                        st.session_state.upload_mode = None
                        st.success("Files uploaded!")
                        st.rerun()

            with col3:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_upload_modal = False
                    st.session_state.upload_mode = None
                    st.rerun()

        # Close button at bottom
        if st.session_state.upload_mode is None:
            if st.button("❌ Close", use_container_width=True):
                st.session_state.show_upload_modal = False
                st.rerun()

    upload_modal()