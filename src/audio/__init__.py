"""Audio module for TTS and STT functionality."""

from src.audio.stt import STTModel, GroqWhisperSTT
from src.audio.tts import TTSModel, OpenAITTS, HuggingFaceTTS, LocalTTS, F5TTS
from src.audio.config import AudioConfig

__all__ = [
    "STTModel",
    "GroqWhisperSTT",
    "TTSModel",
    "OpenAITTS",
    "HuggingFaceTTS",
    "LocalTTS",
    "F5TTS",  # Backwards compatibility
    "AudioConfig",
]