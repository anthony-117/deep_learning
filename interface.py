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

    # Upload mode selection
    upload_mode = st.radio(
        "Choose upload mode:",
        ["Single PDF File", "Multiple PDF Files", "Folder Path"],
        help="Select how you want to provide documents for processing"
    )

    uploaded_files = None
    folder_path = None

    if upload_mode == "Single PDF File":
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            uploaded_files = [uploaded_file]
    elif upload_mode == "Multiple PDF Files":
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    else:  # Folder Path
        folder_path = st.text_input(
            "Enter folder path containing PDF files:",
            placeholder="/path/to/your/pdf/folder",
            help="Enter the full path to a folder containing PDF files"
        )
        if folder_path:
            if os.path.exists(folder_path):
                if os.path.isdir(folder_path):
                    try:
                        all_files = os.listdir(folder_path)
                        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]

                        if pdf_files:
                            st.success(f"Found {len(pdf_files)} PDF files in the folder")
                            st.write("Files found:", pdf_files[:5])  # Show first 5 files
                            if len(pdf_files) > 5:
                                st.write(f"... and {len(pdf_files) - 5} more files")
                        else:
                            non_pdf_files = [f for f in all_files if not f.startswith('.')]
                            if non_pdf_files:
                                st.warning(f"No PDF files found. Found {len(non_pdf_files)} other files (only PDF files are supported)")
                            else:
                                st.warning("Folder is empty or contains only hidden files")
                    except PermissionError:
                        st.error("Permission denied: Cannot access the specified folder")
                else:
                    st.error("Path exists but is not a directory")
            else:
                st.error("Folder path does not exist")

    st.header("2. Configure RAG Pipeline")
    with st.expander("Settings", expanded=True):
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
                        "overlap": chunk_overlap, "top_k": top_k
                    }
                    st.session_state.config = current_config

                    # Initialize the RAG processor
                    processor = RAGProcessor(
                        groq_api_key=groq_api_key,
                        temperature=temperature
                    )

                    # Process files based on upload mode
                    if upload_mode == "Folder Path":
                        # Process all PDFs in the folder
                        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                        file_paths = [os.path.join(folder_path, f) for f in pdf_files]

                        processor.setup_rag_pipeline_multiple(
                            pdf_paths=file_paths,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k
                        )

                        file_count = len(pdf_files)
                        success_message = f"Processed {file_count} PDF files from folder!"

                    else:
                        # Process uploaded files (single or multiple)
                        temp_paths = []

                        for i, uploaded_file in enumerate(uploaded_files):
                            temp_path = f"temp_file_{i}_{uploaded_file.name}"
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)

                        if len(temp_paths) == 1:
                            # Single file - use existing method
                            processor.setup_rag_pipeline(
                                pdf_path=temp_paths[0],
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k
                            )
                        else:
                            # Multiple files - use new method
                            processor.setup_rag_pipeline_multiple(
                                pdf_paths=temp_paths,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                top_k=top_k
                            )

                        file_count = len(uploaded_files)
                        success_message = f"Processed {file_count} PDF file{'s' if file_count > 1 else ''}!"

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
    st.info(f"**Current Configuration:** Model: `{config['model']}`, Temp: `{config['temp']}`, "
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