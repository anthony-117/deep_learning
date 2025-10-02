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
        model_name = "openai/gpt-oss-20b" , 
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

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
                        "overlap": chunk_overlap, "top_k": top_k
                    }
                    st.session_state.config = current_config

                    # Initialize and setup the RAG pipeline with parameters from the UI
                    processor = RAGProcessor(
                         groq_api_key=groq_api_key,
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
    st.info(f"**Current Configuration:** Model: `{config['model']}`, Temp: `{config['temp']}`, "
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
                        filename = os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown'

                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 8px; border-radius: 4px; margin: 5px 0;">
                            <strong>📄 Source:</strong> {filename} | <strong>📍 Page:</strong> {page_num + 1 if page_num != 'Unknown' else 'Unknown'}
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
                            filename = os.path.basename(source_file) if source_file != 'Unknown' else 'Unknown'

                            st.markdown(f"""
                            <div style="background-color: #f0f2f6; padding: 8px; border-radius: 4px; margin: 5px 0;">
                                <strong>📄 Source:</strong> {filename} | <strong>📍 Page:</strong> {page_num + 1 if page_num != 'Unknown' else 'Unknown'}
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