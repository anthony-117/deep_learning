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
        model_name = "openai/gpt-oss-20b"
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

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
                    # Store current configuration
                    current_config = {
                        "model": "openai/gpt-oss-20b", "temp": temperature, "chunk_size": chunk_size,
                        "overlap": chunk_overlap, "top_k": top_k, "enhanced": use_enhanced_processing
                    }
                    st.session_state.config = current_config

                    # Initialize the RAG processor
                    processor = RAGProcessor(
                        groq_api_key=groq_api_key,
                        temperature=temperature
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
    st.info(f"**Current Configuration:** Mode: `{processing_mode}`, Model: `{config['model']}`, Temp: `{config['temp']}`, "
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