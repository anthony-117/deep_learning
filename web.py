import streamlit as st
import os

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
                collection_name="docling_demo"
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
                                st.markdown(f"**Document {i+1}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                if doc.metadata:
                                    st.json(doc.metadata)
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
            with st.spinner("Thinking..."):
                try:
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

                    # Invoke the RAG graph with conversation history
                    result = rag_graph.invoke(prompt, chat_history=chat_history)

                    # Extract answer
                    answer = result.get("answer", "I couldn't find a relevant answer.")

                    # Display answer
                    st.markdown(answer)

                    # Prepare metadata
                    metadata = {
                        "steps": result.get("steps", []),
                        "documents": result.get("documents", []),
                        "rewrites": result.get("rewrites", 0),
                        "grounded": result.get("grounded", False)
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
                                st.markdown(f"**Document {i+1}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                if doc.metadata:
                                    st.json(doc.metadata)
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