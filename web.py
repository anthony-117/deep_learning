import streamlit as st
import os
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

from src.embedding import EmbeddingModel
from src.llm import LLMModel
from src.vectordb import VectorStore
from src.graph import RAGGraph


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# Node name to user-friendly display mapping
NODE_DISPLAY_NAMES = {
    "detect_scraping": ("🔍 Detecting Intent", "Analyzing your request..."),
    "extract_params": ("📋 Extracting Parameters", "Getting scraping details..."),
    "scrape_papers": ("📥 Scraping Papers", "Fetching papers from sources..."),
    "scraping_summary": ("📊 Generating Summary", "Summarizing scraped papers..."),
    "analyze_query": ("🔍 Analyzing Query", "Understanding your question..."),
    "improve_prompt": ("✨ Improving Query", "Optimizing search terms..."),
    "retrieve": ("📚 Retrieving Documents", "Searching vector database..."),
    "grade_documents": ("⚖️ Grading Documents", "Evaluating document relevance..."),
    "generate": ("✍️ Generating Answer", "Creating response..."),
    "check_hallucination": ("✅ Verifying Answer", "Checking answer accuracy..."),
    "rewrite_query": ("🔄 Rewriting Query", "Refining search..."),
}


def is_image_document(doc) -> tuple[bool, str]:
    """
    Check if a document is an image based on its metadata.

    Returns:
        tuple: (is_image: bool, file_path: str)
    """
    if not doc.metadata:
        return False, ""

    # Check mimetype
    if "dl_meta" in doc.metadata and "origin" in doc.metadata["dl_meta"]:
        origin = doc.metadata["dl_meta"]["origin"]
        mimetype = origin.get("mimetype", "")
        if mimetype.startswith("image/"):
            file_path = doc.metadata.get("file_path", "")
            return True, file_path

    return False, ""


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components."""
    try:
        # Initialize components
        with st.spinner("Initializing RAG system..."):
            embedding_model = EmbeddingModel()
            llm = LLMModel()
            vector_store = VectorStore(
                embedding=embedding_model.get_embedding(),
            )

            # Note: VectorStore needs to be populated with documents first
            # If vectorstore is not initialized, we'll handle it gracefully
            try:
                rag_graph = RAGGraph(
                    vector_store=vector_store,
                    llm=llm,
                    max_rewrites=2,
                    relevance_threshold=0.5
                )
                return rag_graph
            except RuntimeError as e:
                st.warning(f"Vector store not initialized: {e}")
                st.info("Please run the ingestion pipeline first to populate the vector store.")
                return None

    except Exception as e:
        st.error(f"Error initializing RAG system: {e}")
        return None


def main():
    st.title("RAG Chatbot")
    st.markdown("Ask questions about your documents!")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Display environment status
        st.subheader("Environment Status")
        groq_api_key_set = "Yes" if os.getenv("GROQ_API_KEY") else "No"
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")

        st.markdown(f"**GROQ_API_KEY:** {groq_api_key_set}")
        st.markdown(f"**Milvus Host:** `{milvus_host}`")
        st.markdown(f"**Milvus Port:** `{milvus_port}`")

        st.divider()

        # Clear chat button
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        st.divider()

        # Advanced settings
        with st.expander("Advanced Settings"):
            show_steps = st.checkbox("Show processing steps", value=False)
            show_sources = st.checkbox("Show source documents", value=True)
            st.session_state.show_steps = show_steps
            st.session_state.show_sources = show_sources

    # Initialize RAG system
    rag_graph = initialize_rag_system()

    if rag_graph is None:
        st.warning("RAG system not ready. Please check the configuration above.")
        st.stop()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display metadata if available
            if "metadata" in message and message["role"] == "assistant":
                metadata = message["metadata"]

                # Show processing steps
                if st.session_state.get("show_steps", False) and "steps" in metadata:
                    with st.expander("Processing Steps"):
                        for step in metadata["steps"]:
                            st.text(f"- {step}")

                # Show source documents
                if st.session_state.get("show_sources", True) and "documents" in metadata:
                    docs = metadata["documents"]
                    if docs:
                        with st.expander(f"Source Documents ({len(docs)})"):
                            for i, doc in enumerate(docs):
                                # Check if this is an image document
                                is_image, image_path = is_image_document(doc)

                                # Use appropriate icon
                                doc_icon = "🖼️" if is_image else "📄"
                                st.markdown(f"### {doc_icon} Document {i+1}")

                                # Display metadata in a user-friendly way
                                if doc.metadata:
                                    if "title" in doc.metadata:
                                        st.markdown(f"**Title:** {doc.metadata['title']}")

                                    # PDF URL as clickable link
                                    if "pdf_url" in doc.metadata and doc.metadata["pdf_url"]:
                                        pdf_url = doc.metadata["pdf_url"]
                                        st.markdown(f"**PDF:** [Open PDF]({pdf_url}) 📄")

                                    # Extract page and location info from dl_meta
                                    if "dl_meta" in doc.metadata:
                                        dl_meta = doc.metadata["dl_meta"]

                                        # Get filename
                                        if "origin" in dl_meta and "filename" in dl_meta["origin"]:
                                            filename = dl_meta["origin"]["filename"]
                                            st.markdown(f"**File:** `{filename}`")

                                        # Get page numbers and headings
                                        if "doc_items" in dl_meta and dl_meta["doc_items"]:
                                            # Collect unique page numbers from all items
                                            pages = set()
                                            for item in dl_meta["doc_items"]:
                                                if "prov" in item:
                                                    for prov in item["prov"]:
                                                        if "page_no" in prov:
                                                            pages.add(prov["page_no"])

                                            if pages:
                                                pages_sorted = sorted(list(pages))
                                                if len(pages_sorted) == 1:
                                                    st.markdown(f"**Page:** {pages_sorted[0]}")
                                                else:
                                                    st.markdown(f"**Pages:** {', '.join(map(str, pages_sorted))}")

                                        # Get headings if available
                                        if "headings" in dl_meta and dl_meta["headings"]:
                                            headings_str = " > ".join(dl_meta["headings"])
                                            st.markdown(f"**Section:** {headings_str}")

                                    # Domain
                                    if "domain" in doc.metadata:
                                        st.markdown(f"**Domain:** `{doc.metadata['domain']}`")

                                    # URL as clickable link
                                    if "url" in doc.metadata:
                                        url = doc.metadata["url"]
                                        # Add protocol if not present
                                        if not url.startswith(("http://", "https://")):
                                            full_url = f"https://{url}"
                                        else:
                                            full_url = url
                                        st.markdown(f"**HTML:** [{url}]({full_url})")

                                # Display image or content
                                if is_image and image_path and Path(image_path).exists():
                                    st.markdown("**Image:**")
                                    st.image(image_path, use_container_width=True)
                                    if doc.page_content:
                                        st.markdown("**Description:**")
                                        st.markdown(f"> {doc.page_content}")
                                else:
                                    st.markdown("**Content:**")
                                    st.markdown(f"> {doc.page_content}")

                                if i < len(docs) - 1:
                                    st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            # Build chat history from session messages
            chat_history = []
            last_role = None

            for msg in st.session_state.messages[:-1]:  # Exclude current message
                content = msg.get("content", "")
                role = msg.get("role", "")

                # Skip messages with empty or None content
                if not content or not role:
                    continue

                # Skip consecutive messages from the same role to maintain alternation
                if last_role == role:
                    continue

                if role == "user":
                    chat_history.append(HumanMessage(content=content))
                    last_role = "user"
                elif role == "assistant":
                    chat_history.append(AIMessage(content=content))
                    last_role = "assistant"

            try:
                # Create status container for real-time updates
                status_container = st.empty()
                answer_container = st.empty()

                # Stream the RAG graph execution
                result = None
                current_node = None

                for output in rag_graph.stream(prompt, chat_history=chat_history):
                    # Extract the node name from the output
                    # Output is a dict with node name as key
                    for node_name, node_state in output.items():
                        current_node = node_name

                        # Get display name and description
                        if node_name in NODE_DISPLAY_NAMES:
                            display_name, description = NODE_DISPLAY_NAMES[node_name]

                            # Update status display
                            with status_container:
                                st.info(f"**{display_name}**\n\n{description}")

                        # Store the latest result
                        result = node_state

                # Clear status after completion
                status_container.empty()

                # Extract answer from final result
                if result:
                    answer = result.get("generation", "I couldn't find a relevant answer.")
                else:
                    answer = "I couldn't find a relevant answer."

                # Display answer
                with answer_container:
                    st.markdown(answer)

                # Prepare metadata
                metadata = {
                    "steps": result.get("steps", []) if result else [],
                    "documents": result.get("documents", []) if result else [],
                    "rewrites": result.get("rewrite_count", 0) if result else 0,
                    "grounded": result.get("answer_grounded", False) if result else False
                }

                # Show processing steps
                if st.session_state.get("show_steps", False) and metadata["steps"]:
                    with st.expander("Processing Steps"):
                        for step in metadata["steps"]:
                            st.text(f"- {step}")
                        st.caption(f"Query rewrites: {metadata['rewrites']}")
                        st.caption(f"Answer grounded: {'Yes' if metadata['grounded'] else 'No'}")

                # Show source documents
                if st.session_state.get("show_sources", True) and metadata["documents"]:
                    docs = metadata["documents"]
                    with st.expander(f"Source Documents ({len(docs)})"):
                        for i, doc in enumerate(docs):
                            # Check if this is an image document
                            is_image, image_path = is_image_document(doc)

                            # Use appropriate icon
                            doc_icon = "🖼️" if is_image else "📄"
                            st.markdown(f"### {doc_icon} Document {i+1}")

                            # Display metadata in a user-friendly way
                            if doc.metadata:
                                if "title" in doc.metadata:
                                    st.markdown(f"**Title:** {doc.metadata['title']}")

                                if "source" in doc.metadata:
                                    st.markdown(f"**Source:** `{doc.metadata['source']}`")

                                # PDF URL as clickable link
                                if "pdf_url" in doc.metadata and doc.metadata["pdf_url"]:
                                    pdf_url = doc.metadata["pdf_url"]
                                    st.markdown(f"**PDF:** [Open PDF]({pdf_url}) 📄")

                                # Extract page and location info from dl_meta
                                if "dl_meta" in doc.metadata:
                                    dl_meta = doc.metadata["dl_meta"]

                                    # Get page numbers and headings
                                    if "doc_items" in dl_meta and dl_meta["doc_items"]:
                                        # Collect unique page numbers from all items
                                        pages = set()
                                        for item in dl_meta["doc_items"]:
                                            if "prov" in item:
                                                for prov in item["prov"]:
                                                    if "page_no" in prov:
                                                        pages.add(prov["page_no"])

                                        if pages:
                                            pages_sorted = sorted(list(pages))
                                            if len(pages_sorted) == 1:
                                                st.markdown(f"**Page:** {pages_sorted[0]}")
                                            else:
                                                st.markdown(f"**Pages:** {', '.join(map(str, pages_sorted))}")

                                    # Get headings if available
                                    if "headings" in dl_meta and dl_meta["headings"]:
                                        headings_str = " > ".join(dl_meta["headings"])
                                        st.markdown(f"**Section:** {headings_str}")

                                # Domain
                                if "domain" in doc.metadata:
                                    st.markdown(f"**Domain:** `{doc.metadata['domain']}`")

                                # URL as clickable link
                                if "url" in doc.metadata:
                                    url = doc.metadata["url"]
                                    # Add protocol if not present
                                    if not url.startswith(("http://", "https://")):
                                        full_url = f"https://{url}"
                                    else:
                                        full_url = url
                                    st.markdown(f"**HTML:** [{url}]({full_url})")

                            # Display image or content
                            if is_image and image_path and Path(image_path).exists():
                                st.markdown("**Image:**")
                                st.image(image_path, use_container_width=True)
                                if doc.page_content:
                                    st.markdown("**Description:**")
                                    st.markdown(f"> {doc.page_content}")
                            else:
                                st.markdown("**Content:**")
                                st.markdown(f"> {doc.page_content}")

                            if i < len(docs) - 1:
                                st.divider()

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


if __name__ == "__main__":
    main()