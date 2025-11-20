"""Base classes for audio processing."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Union


class STTModel(ABC):
    """Abstract base class for Speech-to-Text models."""

    @abstractmethod
    def transcribe(self, audio: Union[BinaryIO, Path, bytes]) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio data as file-like object, path, or bytes

        Returns:
            Transcribed text
        """
        pass


class TTSModel(ABC):
    """Abstract base class for Text-to-Speech models."""

    @abstractmethod
    def synthesize(self, text: str, voice: str = None) -> bytes:
        """
        Synthesize text to speech.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (provider-specific)

        Returns:
            Audio data as bytes
        """
        pass

    @abstractmethod
    def get_available_voices(self) -> list[str]:
        """
        Get list of available voice identifiers.

        Returns:
            List of voice names/IDs
        """
        pass