"""Gradio web interface for RAG Chatbot with audio support."""

import os
import asyncio
import gradio as gr
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage

from src.embedding import EmbeddingModel
from src.llm import LLMModel
from src.vectordb import VectorStore
from src.graph import RAGGraph
from src.audio import GroqWhisperSTT, HuggingFaceTTS, OpenAITTS, AudioConfig


# Global variables for models
rag_graph = None
stt_model = None
tts_model = None
audio_config = None


def initialize_rag_system():
    """Initialize the RAG system components."""
    global rag_graph
    try:
        embedding_model = EmbeddingModel()
        llm = LLMModel()
        vector_store = VectorStore(embedding=embedding_model.get_embedding())

        rag_graph = RAGGraph(
            vector_store=vector_store,
            llm=llm,
            max_rewrites=2,
            relevance_threshold=0.5
        )
        return "✅ RAG system initialized successfully"
    except RuntimeError as e:
        return f"⚠️ Vector store not initialized: {e}\nPlease run the ingestion pipeline first."
    except Exception as e:
        return f"❌ Error initializing RAG system: {e}"


def initialize_audio_system():
    """Initialize the audio system components."""
    global stt_model, tts_model, audio_config

    messages = []

    # Initialize audio config
    audio_config = AudioConfig()

    # Initialize STT
    if not audio_config.groq_api_key:
        messages.append("⚠️ GROQ_API_KEY not set. Speech-to-text unavailable.")
        stt_model = None
    else:
        try:
            stt_model = GroqWhisperSTT(audio_config)
            messages.append("✅ STT initialized (Groq Whisper)")
        except Exception as e:
            messages.append(f"❌ STT initialization failed: {e}")
            stt_model = None

    # Initialize TTS based on provider
    tts_provider = audio_config.tts_provider
    tts_model_name = audio_config.tts_model

    if tts_provider == "huggingface":
        if not audio_config.huggingface_api_key:
            messages.append("⚠️ HUGGINGFACE_API_KEY not set. Text-to-speech unavailable.")
            tts_model = None
        else:
            try:
                tts_model = HuggingFaceTTS(audio_config)
                messages.append(f"✅ TTS initialized (Hugging Face: {tts_model_name})")
            except Exception as e:
                messages.append(f"❌ TTS initialization failed: {e}")
                tts_model = None

    elif tts_provider == "openai":
        if not audio_config.openai_api_key:
            messages.append("⚠️ OPENAI_API_KEY not set. Text-to-speech unavailable.")
            tts_model = None
        else:
            try:
                tts_model = OpenAITTS(audio_config)
                messages.append(f"✅ TTS initialized (OpenAI: {tts_model_name})")
            except Exception as e:
                messages.append(f"❌ TTS initialization failed: {e}")
                tts_model = None
    elif tts_provider == "local":
        try:
            from src.audio import LocalTTS
            tts_model = LocalTTS(audio_config)
            messages.append(f"✅ TTS initialized (Local: {tts_model_name})")
        except Exception as e:
            messages.append(f"❌ TTS initialization failed: {e}")
            tts_model = None
    else:
        messages.append(f"⚠️ Unknown TTS provider: {tts_provider}")
        tts_model = None

    return "\n".join(messages) if messages else "✅ Audio system initialized"


def is_image_document(doc) -> tuple[bool, str]:
    """Check if a document is an image based on its metadata."""
    if not doc.metadata:
        return False, ""

    if "dl_meta" in doc.metadata and "origin" in doc.metadata["dl_meta"]:
        origin = doc.metadata["dl_meta"]["origin"]
        mimetype = origin.get("mimetype", "")
        if mimetype.startswith("image/"):
            file_path = doc.metadata.get("file_path", "")
            return True, file_path

    return False, ""


def format_source_documents(docs):
    """Format source documents for display."""
    if not docs:
        return ""

    output = f"\n\n📚 **Source Documents ({len(docs)})**\n\n"

    for i, doc in enumerate(docs, 1):
        is_image, image_path = is_image_document(doc)
        doc_icon = "🖼️" if is_image else "📄"

        output += f"### {doc_icon} Document {i}\n\n"

        if doc.metadata:
            if "title" in doc.metadata:
                output += f"**Title:** {doc.metadata['title']}\n\n"

            if "pdf_url" in doc.metadata and doc.metadata["pdf_url"]:
                output += f"**PDF:** [{doc.metadata['pdf_url']}]({doc.metadata['pdf_url']})\n\n"

            if "dl_meta" in doc.metadata:
                dl_meta = doc.metadata["dl_meta"]

                if "origin" in dl_meta and "filename" in dl_meta["origin"]:
                    output += f"**File:** `{dl_meta['origin']['filename']}`\n\n"

                if "doc_items" in dl_meta and dl_meta["doc_items"]:
                    pages = set()
                    for item in dl_meta["doc_items"]:
                        if "prov" in item:
                            for prov in item["prov"]:
                                if "page_no" in prov:
                                    pages.add(prov["page_no"])

                    if pages:
                        pages_sorted = sorted(list(pages))
                        if len(pages_sorted) == 1:
                            output += f"**Page:** {pages_sorted[0]}\n\n"
                        else:
                            output += f"**Pages:** {', '.join(map(str, pages_sorted))}\n\n"

                if "headings" in dl_meta and dl_meta["headings"]:
                    headings_str = " > ".join(dl_meta["headings"])
                    output += f"**Section:** {headings_str}\n\n"

            if "domain" in doc.metadata:
                output += f"**Domain:** `{doc.metadata['domain']}`\n\n"

            if "url" in doc.metadata:
                url = doc.metadata["url"]
                if not url.startswith(("http://", "https://")):
                    url = f"https://{url}"
                output += f"**URL:** [{doc.metadata['url']}]({url})\n\n"

        output += f"**Content:**\n> {doc.page_content}\n\n"
        output += "---\n\n"

    return output


def transcribe_audio(audio_file):
    """Transcribe audio file to text."""
    if audio_file is None:
        return ""

    if stt_model is None:
        return "❌ STT not available. Please set GROQ_API_KEY."

    try:
        # Debug: Print audio file info
        print(f"[DEBUG] Audio file received: {audio_file}")
        print(f"[DEBUG] Audio file type: {type(audio_file)}")

        # Check if audio file exists and has content
        if isinstance(audio_file, str):
            from pathlib import Path
            path = Path(audio_file)
            if path.exists():
                file_size = path.stat().st_size
                print(f"[DEBUG] Audio file size: {file_size} bytes")
                if file_size == 0:
                    return "❌ Audio file is empty"
            else:
                return "❌ Audio file does not exist"

        transcribed_text = stt_model.transcribe(audio_file)
        print(f"[DEBUG] Transcribed text: {transcribed_text}")
        return transcribed_text
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Transcription error: {str(e)}"


def generate_speech(text, voice):
    """Generate speech from text."""
    if not text or not text.strip():
        return None

    if tts_model is None:
        return None

    try:
        # Update voice setting
        tts_model.config.tts_voice = voice

        # Generate audio
        audio_bytes = tts_model.synthesize(text)

        # Save to temporary file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(audio_bytes)
        temp_file.close()

        return temp_file.name
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def chat(message, history, show_sources, voice_selection, auto_play):
    """Process chat message and return response."""
    print(f"[DEBUG] chat() called with message: {message}")

    if not message or not message.strip():
        print("[DEBUG] Empty message, returning")
        return history, None

    if rag_graph is None:
        print("[DEBUG] RAG graph not initialized")
        error = (message, "❌ RAG system not initialized. Please check the configuration.")
        if history is None:
            history = []
        history.append(error)
        return history, None

    try:
        # Initialize history if None
        if history is None:
            history = []

        print(f"[DEBUG] Building chat history from {len(history)} previous messages")

        # Build chat history
        chat_history = []
        for user_msg, assistant_msg in history:
            if user_msg:
                chat_history.append(HumanMessage(content=user_msg))
            if assistant_msg:
                chat_history.append(AIMessage(content=assistant_msg))

        print(f"[DEBUG] Invoking RAG graph...")
        # Invoke RAG graph
        result = rag_graph.invoke(message, chat_history=chat_history)
        print(f"[DEBUG] RAG graph returned result: {result.keys() if result else 'None'}")

        # Extract answer
        answer = result.get("answer", "I couldn't find a relevant answer.") if result else "I couldn't find a relevant answer."

        # Ensure answer is not None
        if answer is None:
            answer = "I couldn't find a relevant answer."

        print(f"[DEBUG] Answer: {answer[:100] if len(answer) > 100 else answer}...")

        # Add source documents if enabled
        if show_sources:
            docs = result.get("documents", [])
            if docs:
                print(f"[DEBUG] Adding {len(docs)} source documents")
                answer += format_source_documents(docs)

        # Add to history
        history.append((message, answer))
        print(f"[DEBUG] Updated history, now has {len(history)} messages")

        # Generate TTS if auto-play is enabled
        audio_output = None
        if auto_play and tts_model:
            print("[DEBUG] Generating TTS audio")
            # Extract just the answer text (without sources) for TTS
            answer_text = result.get("answer", "")
            audio_output = generate_speech(answer_text, voice_selection)

        print(f"[DEBUG] Returning history with {len(history)} messages")
        return history, audio_output

    except Exception as e:
        print(f"[DEBUG] Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        error_msg = f"❌ Error: {str(e)}"
        if history is None:
            history = []
        history.append((message, error_msg))
        return history, None


def process_voice_input(audio, history, show_sources, voice_selection, auto_play):
    """Process voice input and generate response."""
    if audio is None:
        return history, "", None

    # Transcribe audio
    transcribed_text = transcribe_audio(audio)

    if transcribed_text.startswith("❌"):
        return history, transcribed_text, None

    # Process the transcribed text
    history, audio_output = chat(transcribed_text, history, show_sources, voice_selection, auto_play)

    return history, f"💬 You said: {transcribed_text}", audio_output


def toggle_recording(is_recording):
    """Toggle recording state and update button appearance."""
    new_state = not is_recording
    if new_state:
        # Starting to record
        return new_state, gr.update(elem_classes=["mic-btn", "recording"])
    else:
        # Stopped recording
        return new_state, gr.update(elem_classes=["mic-btn"])


def play_response_audio(voice_selection):
    """Generate audio for the last assistant response."""
    # This will be called by clicking the "Play Response" button
    # We'll need to get the last message from history
    return None


# Initialize systems on startup
rag_status = initialize_rag_system()
audio_status = initialize_audio_system()


# Create Gradio interface with custom CSS
custom_css = """
.mic-btn {
    width: 60px !important;
    height: 60px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-size: 28px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0px 4px 15px rgba(102, 126, 234, 0.4) !important;
    cursor: pointer !important;
}

.mic-btn:hover {
    transform: scale(1.1) !important;
    background: linear-gradient(135deg, #5568d3 0%, #6340a0 100%) !important;
    box-shadow: 0px 6px 20px rgba(102, 126, 234, 0.6) !important;
}

.recording {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    animation: pulse 1.5s ease-in-out infinite !important;
}

.recording:hover {
    background: linear-gradient(135deg, #e082ea 0%, #e4465b 100%) !important;
}

@keyframes pulse {
    0%   { transform: scale(1.00); box-shadow: 0 0 0 rgba(245, 87, 108, 0.4); }
    50%  { transform: scale(1.12); box-shadow: 0 0 20px rgba(245, 87, 108, 0.8); }
    100% { transform: scale(1.00); box-shadow: 0 0 0 rgba(245, 87, 108, 0.4); }
}
"""

with gr.Blocks(title="RAG Chatbot", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# 🤖 RAG Chatbot")
    gr.Markdown("Ask questions about your documents using text or voice!")

    # State for recording
    is_recording = gr.State(False)

    with gr.Row():
        # Main chat area
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
                show_copy_button=True,
                bubble_full_width=False
            )

            with gr.Row():
                with gr.Column(scale=9):
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Ask a question about your documents...",
                        show_label=False,
                        container=False
                    )

                with gr.Column(scale=1, min_width=60):
                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎤",
                        show_label=False,
                        container=False,
                        elem_classes=["voice-audio"]
                    )

            transcription_output = gr.Textbox(
                label="Transcription",
                visible=True,
                interactive=False
            )

            # Audio output for TTS
            audio_output = gr.Audio(
                label="Assistant Response (Audio)",
                visible=True,
                autoplay=True
            )

            # Manual TTS generation
            with gr.Row():
                generate_audio_btn = gr.Button("🔊 Play Last Response", size="sm")

        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Configuration")

            # Environment status
            with gr.Accordion("Environment Status", open=False):
                gr.Markdown(f"""
                **GROQ_API_KEY:** {"✅ Set" if os.getenv("GROQ_API_KEY") else "❌ Not set"}

                **OPENAI_API_KEY:** {"✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not set"}

                **Milvus Host:** `{os.getenv("MILVUS_HOST", "localhost")}`

                **Milvus Port:** `{os.getenv("MILVUS_PORT", "19530")}`
                """)

            # Audio settings
            gr.Markdown("### 🎵 Audio Settings")

            voice_selection = gr.Dropdown(
                choices=["af_heart", "af_bella", "af_sarah", "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis"],
                value="af_heart",
                label="TTS Voice",
                info="Select voice for text-to-speech (Kokoro voices)"
            )

            auto_play = gr.Checkbox(
                label="Auto-play responses",
                value=False,
                info="Automatically generate audio for responses"
            )

            gr.Markdown("### 📋 Display Settings")

            show_sources = gr.Checkbox(
                label="Show source documents",
                value=True,
                info="Display source documents with answers"
            )

            # System status
            with gr.Accordion("System Status", open=True):
                gr.Markdown(f"**RAG System:**\n{rag_status}")
                gr.Markdown(f"**Audio System:**\n{audio_status}")

            # Clear chat button
            clear_btn = gr.Button("🗑️ Clear Chat History", variant="secondary")

    # Event handlers

    # Text input
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot, show_sources, voice_selection, auto_play],
        outputs=[chatbot, audio_output],
        queue=True
    ).then(
        lambda: "",
        None,
        msg_input,
        queue=False
    )

    # Voice input - process when audio changes
    voice_input.change(
        fn=process_voice_input,
        inputs=[voice_input, chatbot, show_sources, voice_selection, auto_play],
        outputs=[chatbot, transcription_output, audio_output],
        queue=True
    )

    # Clear chat
    clear_btn.click(
        lambda: ([], "", None),
        None,
        [chatbot, transcription_output, audio_output]
    )

    # Generate audio for last response
    def generate_last_response_audio(history, voice):
        if not history:
            return None

        last_response = history[-1][1]
        # Extract just the answer (before sources)
        if "📚 **Source Documents" in last_response:
            last_response = last_response.split("📚 **Source Documents")[0].strip()

        return generate_speech(last_response, voice)

    generate_audio_btn.click(
        fn=generate_last_response_audio,
        inputs=[chatbot, voice_selection],
        outputs=audio_output
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )