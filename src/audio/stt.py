"""Speech-to-Text implementations."""

import io
from pathlib import Path
from typing import BinaryIO, Union

from groq import Groq

from src.audio.base import STTModel
from src.audio.config import AudioConfig


class GroqWhisperSTT(STTModel):
    """Groq Whisper implementation for Speech-to-Text."""

    def __init__(self, config: AudioConfig = None):
        """
        Initialize Groq Whisper STT.

        Args:
            config: Audio configuration
        """
        self.config = config or AudioConfig()
        self.config.validate()

        self.client = Groq(api_key=self.config.groq_api_key)
        self.model = self.config.stt_model

    def transcribe(self, audio: Union[BinaryIO, Path, bytes]) -> str:
        """
        Transcribe audio to text using Groq Whisper.

        Args:
            audio: Audio data as file-like object, path, or bytes

        Returns:
            Transcribed text

        Raises:
            ValueError: If audio format is not supported
            Exception: If transcription fails
        """
        try:
            # Handle different input types
            if isinstance(audio, bytes):
                # Convert bytes to file-like object
                audio_file = io.BytesIO(audio)
                audio_file.name = "audio.wav"  # Groq needs a filename
            elif isinstance(audio, Path):
                # Open file
                audio_file = open(audio, "rb")
            elif isinstance(audio, str):
                # Treat as path
                audio_file = open(audio, "rb")
            else:
                # Assume it's already a file-like object
                audio_file = audio

            # Transcribe using Groq
            transcription = self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.model,
                language=self.config.stt_language,
                response_format="text",
            )

            # Close file if we opened it
            if isinstance(audio, (Path, str)):
                audio_file.close()

            return transcription

        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")