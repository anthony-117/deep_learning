import streamlit as st
import os
import io
import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

from src.embedding import EmbeddingModel
from src.llm import LLMModel
from src.vectordb import VectorStore
from src.graph import RAGGraph
from src.audio import GroqWhisperSTT, OpenAITTS, AudioConfig


# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


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


@st.cache_resource
def initialize_audio_system():
    """Initialize the audio system components."""
    try:
        # Check if required API keys are available
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if not groq_key:
            st.warning("GROQ_API_KEY not set. Speech-to-text will be unavailable.")
            return None, None

        if not openai_key:
            st.warning("OPENAI_API_KEY not set. Text-to-speech will be unavailable.")
            return None, None

        # Initialize audio config
        audio_config = AudioConfig()

        # Initialize STT and TTS
        stt_model = GroqWhisperSTT(audio_config)
        tts_model = OpenAITTS(audio_config)

        return stt_model, tts_model

    except Exception as e:
        st.error(f"Error initializing audio system: {e}")
        return None, None


def main():
    st.title("RAG Chatbot")
    st.markdown("Ask questions about your documents!")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")

        # Display environment status
        st.subheader("Environment Status")
        groq_api_key_set = "Yes" if os.getenv("GROQ_API_KEY") else "No"
        openai_api_key_set = "Yes" if os.getenv("OPENAI_API_KEY") else "No"
        milvus_host = os.getenv("MILVUS_HOST", "localhost")
        milvus_port = os.getenv("MILVUS_PORT", "19530")

        st.markdown(f"**GROQ_API_KEY:** {groq_api_key_set}")
        st.markdown(f"**OPENAI_API_KEY:** {openai_api_key_set}")
        st.markdown(f"**Milvus Host:** `{milvus_host}`")
        st.markdown(f"**Milvus Port:** `{milvus_port}`")

        st.divider()

        # Audio settings
        st.subheader("Audio Settings")

        # Initialize audio system
        stt_model, tts_model = initialize_audio_system()

        if stt_model and tts_model:
            # Enable audio features
            enable_audio = st.checkbox("Enable Audio Features", value=True)
            st.session_state.enable_audio = enable_audio

            if enable_audio:
                # TTS voice selection
                available_voices = tts_model.get_available_voices()
                selected_voice = st.selectbox(
                    "TTS Voice",
                    options=available_voices,
                    index=0,
                    help="Select the voice for text-to-speech"
                )
                st.session_state.tts_voice = selected_voice

                # TTS speed
                tts_speed = st.slider(
                    "TTS Speed",
                    min_value=0.25,
                    max_value=2.0,
                    value=1.0,
                    step=0.25,
                    help="Adjust the speed of speech synthesis"
                )
                st.session_state.tts_speed = tts_speed

                # Auto-play TTS
                auto_play_tts = st.checkbox(
                    "Auto-play responses",
                    value=False,
                    help="Automatically play audio for assistant responses"
                )
                st.session_state.auto_play_tts = auto_play_tts
        else:
            st.session_state.enable_audio = False
            st.info("Audio features unavailable. Please set GROQ_API_KEY and OPENAI_API_KEY.")

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

    # Store audio models in session state if available
    if st.session_state.get("enable_audio", False):
        if "stt_model" not in st.session_state:
            st.session_state.stt_model = stt_model
        if "tts_model" not in st.session_state:
            st.session_state.tts_model = tts_model

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Add TTS playback button for assistant messages
            if message["role"] == "assistant" and st.session_state.get("enable_audio", False):
                audio_key = f"audio_{idx}"

                # Create columns for TTS controls
                col1, col2 = st.columns([1, 5])

                with col1:
                    if st.button("🔊 Play", key=f"play_{idx}"):
                        # Generate TTS audio
                        with st.spinner("Generating audio..."):
                            try:
                                tts_model = st.session_state.tts_model

                                # Update TTS config with current settings
                                tts_model.config.tts_voice = st.session_state.get("tts_voice", "alloy")
                                tts_model.config.tts_speed = st.session_state.get("tts_speed", 1.0)

                                # Generate audio
                                audio_bytes = tts_model.synthesize(message["content"])

                                # Store in session state
                                st.session_state[audio_key] = audio_bytes
                            except Exception as e:
                                st.error(f"TTS Error: {str(e)}")

                # Display audio player if audio was generated
                if audio_key in st.session_state:
                    st.audio(st.session_state[audio_key], format="audio/mp3")

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

    # Chat input with voice button
    col1, col2 = st.columns([10, 1])

    with col1:
        # Regular chat input
        prompt = st.chat_input("Ask a question about your documents...")

    with col2:
        # Voice recording button (only if audio is enabled)
        if st.session_state.get("enable_audio", False):
            if st.button("🎤", key="voice_button", help="Click to record", use_container_width=True):
                st.session_state.recording_active = not st.session_state.get("recording_active", False)
                st.rerun()

    # Minimal voice recorder
    if st.session_state.get("recording_active", False):
        # Show recording indicator
        st.markdown("🔴 **Recording... Click stop when done**")

        # Audio input for recording
        audio_input = st.audio_input("Record", label_visibility="collapsed", key="audio_recorder")

        # Stop button
        if st.button("⏹ Stop & Transcribe", key="stop_recording", type="primary"):
            if audio_input is not None:
                with st.spinner("Transcribing..."):
                    try:
                        stt_model = st.session_state.stt_model
                        transcribed_text = stt_model.transcribe(audio_input)

                        # Display what was said
                        st.success(f"**You said:** {transcribed_text}")

                        # Store and close
                        st.session_state.voice_prompt = transcribed_text
                        st.session_state.recording_active = False
                        st.rerun()

                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                        st.session_state.recording_active = False
            else:
                st.warning("No audio recorded yet")

    # Check if we have a voice prompt to process
    if st.session_state.get("voice_prompt"):
        prompt = st.session_state.pop("voice_prompt")
        # Show what was transcribed before processing
        if prompt:
            st.info(f"💬 **Transcribed:** {prompt}")

    if prompt:
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
                                # Check if this is an image document
                                is_image, image_path = is_image_document(doc)

                                # Use appropriate icon
                                doc_icon = "🖼️" if is_image else "📄"
                                st.markdown(f"### {doc_icon} Document {i+1}")

                                # Display metadata in a user-friendly way
                                if doc.metadata:
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
                                        st.markdown(f"**Source:** [{url}]({full_url})")

                                    # Origin info
                                    if "dl_meta" in doc.metadata and "origin" in doc.metadata["dl_meta"]:
                                        origin = doc.metadata["dl_meta"]["origin"]
                                        if "mimetype" in origin:
                                            st.markdown(f"**Type:** `{origin['mimetype']}`")
                                        if "filename" in origin:
                                            st.markdown(f"**File:** `{origin['filename']}`")

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

                    # Auto-play TTS if enabled
                    if st.session_state.get("enable_audio", False) and st.session_state.get("auto_play_tts", False):
                        try:
                            tts_model = st.session_state.tts_model

                            # Update TTS config with current settings
                            tts_model.config.tts_voice = st.session_state.get("tts_voice", "alloy")
                            tts_model.config.tts_speed = st.session_state.get("tts_speed", 1.0)

                            # Generate and play audio
                            with st.spinner("Generating audio..."):
                                audio_bytes = tts_model.synthesize(answer)
                                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                        except Exception as e:
                            st.warning(f"TTS auto-play failed: {str(e)}")

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()