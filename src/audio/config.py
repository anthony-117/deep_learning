"""Audio configuration."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Configuration for audio models."""

    # STT Configuration
    stt_provider: str = "groq"  # groq, openai, local
    stt_model: str = "whisper-large-v3-turbo"  # Groq Whisper model
    stt_language: Optional[str] = None  # Auto-detect if None

    # TTS Configuration
    tts_provider: str = "local"  # openai, huggingface, local
    tts_model: str = "hexgrad/kokoro-82M"  # HF model: SWivid/F5-TTS, coqui/XTTS-v2, hexgrad/Kokoro-82M, etc.
    tts_voice: str = "af_heart"  # Voice name (model-specific: kokoro uses af_heart, af_bella, etc.)
    tts_speed: float = 1.0  # 0.25 to 4.0
    tts_base_url: Optional[str] = None  # Custom base URL for TTS API (OpenAI only)

    # API Keys
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None

    def __post_init__(self):
        """Load API keys and URLs from environment if not provided."""
        if self.groq_api_key is None:
            self.groq_api_key = os.getenv("GROQ_API_KEY")

        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if self.huggingface_api_key is None:
            self.huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

        # Load TTS model from environment if set
        if env_tts_model := os.getenv("TTS_MODEL"):
            self.tts_model = env_tts_model

        # Load TTS provider from environment if set
        if env_tts_provider := os.getenv("TTS_PROVIDER"):
            self.tts_provider = env_tts_provider

    @property
    def openai_voices(self) -> list[str]:
        """Available OpenAI voices."""
        return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

    def validate(self) -> bool:
        """Validate configuration."""
        if self.stt_provider == "groq" and not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not set for STT")

        if self.tts_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set for TTS")

        if self.tts_provider == "huggingface" and not self.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set for TTS")

        if self.tts_provider == "openai" and self.tts_voice not in self.openai_voices:
            raise ValueError(f"Invalid OpenAI voice: {self.tts_voice}")

        if not 0.25 <= self.tts_speed <= 4.0:
            raise ValueError("TTS speed must be between 0.25 and 4.0")

        return True