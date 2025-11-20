"""Text-to-Speech implementations."""

from openai import OpenAI
from huggingface_hub import InferenceClient

from src.audio.base import TTSModel
from src.audio.config import AudioConfig


class OpenAITTS(TTSModel):
    """OpenAI TTS implementation for Text-to-Speech."""

    def __init__(self, config: AudioConfig = None):
        """
        Initialize OpenAI TTS.

        Args:
            config: Audio configuration
        """
        self.config = config or AudioConfig()
        self.config.validate()

        # Initialize OpenAI client with custom base URL if provided
        client_kwargs = {"api_key": self.config.openai_api_key}
        if self.config.tts_base_url:
            client_kwargs["base_url"] = self.config.tts_base_url

        self.client = OpenAI(**client_kwargs)
        self.model = self.config.tts_model

    def synthesize(self, text: str, voice: str = None) -> bytes:
        """
        Synthesize text to speech using OpenAI TTS.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (alloy, echo, fable, onyx, nova, shimmer)

        Returns:
            Audio data as bytes (MP3 format)

        Raises:
            ValueError: If voice is invalid
            Exception: If synthesis fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Use provided voice or default from config
        selected_voice = voice or self.config.tts_voice

        # Validate voice
        if selected_voice not in self.config.openai_voices:
            raise ValueError(
                f"Invalid voice: {selected_voice}. "
                f"Available voices: {', '.join(self.config.openai_voices)}"
            )

        try:
            # Generate speech using OpenAI
            response = self.client.audio.speech.create(
                model=self.model,
                voice=selected_voice,
                input=text,
                speed=self.config.tts_speed,
            )

            # Return audio bytes
            return response.content

        except Exception as e:
            raise Exception(f"Speech synthesis failed: {str(e)}")

    def get_available_voices(self) -> list[str]:
        """
        Get list of available OpenAI voice identifiers.

        Returns:
            List of voice names
        """
        return self.config.openai_voices

    def synthesize_to_file(self, text: str, output_path: str, voice: str = None) -> None:
        """
        Synthesize text to speech and save to file.

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice identifier

        Raises:
            ValueError: If text is empty or voice is invalid
            Exception: If synthesis or file writing fails
        """
        audio_bytes = self.synthesize(text, voice)

        try:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            raise Exception(f"Failed to write audio file: {str(e)}")


class HuggingFaceTTS(TTSModel):
    """
    Hugging Face TTS implementation using Inference SDK.

    Supports various models:
    - SWivid/F5-TTS
    - coqui/XTTS-v2
    - hexgrad/Kokoro-82M
    - And any other HF text-to-speech model
    """

    def __init__(self, config: AudioConfig = None):
        """
        Initialize Hugging Face TTS.

        Args:
            config: Audio configuration with model specified in tts_model
        """
        self.config = config or AudioConfig()

        # Initialize Hugging Face Inference Client
        if not self.config.huggingface_api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set for Hugging Face TTS")

        self.model_name = self.config.tts_model
        self.client = InferenceClient(
            model=self.model_name,
            token=self.config.huggingface_api_key
        )

    def synthesize(self, text: str, voice: str = None) -> bytes:
        """
        Synthesize text to speech using Hugging Face model.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (model-specific, may not be used)

        Returns:
            Audio data as bytes (format depends on model)

        Raises:
            ValueError: If text is empty
            Exception: If synthesis fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Use text_to_speech method from Hugging Face SDK
            audio_bytes = self.client.text_to_speech(text)

            return audio_bytes

        except Exception as e:
            raise Exception(f"Speech synthesis failed with {self.model_name}: {str(e)}")

    def get_available_voices(self) -> list[str]:
        """
        Get list of available voice identifiers.
        F5-TTS uses default voice.

        Returns:
            List with single default voice
        """
        return ["default"]

    def synthesize_to_file(self, text: str, output_path: str, voice: str = None) -> None:
        """
        Synthesize text to speech and save to file.

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice identifier (not used)

        Raises:
            ValueError: If text is empty
            Exception: If synthesis or file writing fails
        """
        audio_bytes = self.synthesize(text, voice)

        try:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            raise Exception(f"Failed to write audio file: {str(e)}")


# Backwards compatibility alias
F5TTS = HuggingFaceTTS


class LocalTTS(TTSModel):
    """
    Local TTS implementation using kokoro library.

    Supports Kokoro TTS models (e.g., hexgrad/kokoro-82M).
    """

    def __init__(self, config: AudioConfig = None):
        """
        Initialize Local TTS with kokoro.

        Args:
            config: Audio configuration with model name/path in tts_model
        """
        self.config = config or AudioConfig()

        # Import kokoro
        try:
            from kokoro import KPipeline
        except ImportError:
            raise ValueError("kokoro library required for local TTS. Install with: pip install kokoro>=0.9.2")

        self.model_name = self.config.tts_model

        # Initialize Kokoro pipeline
        # Kokoro uses language codes: 'a' for American English
        try:
            print(f"[LocalTTS] Loading Kokoro model: {self.model_name}")
            self.pipe = KPipeline(lang_code='a')
            print(f"[LocalTTS] Kokoro model loaded successfully!")
        except Exception as e:
            raise ValueError(f"Failed to load Kokoro model '{self.model_name}': {str(e)}")

    def synthesize(self, text: str, voice: str = None) -> bytes:
        """
        Synthesize text to speech using Kokoro.

        Args:
            text: Text to convert to speech
            voice: Voice identifier (af_heart, af_bella, af_sarah, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis)

        Returns:
            Audio data as bytes (WAV format)

        Raises:
            ValueError: If text is empty
            Exception: If synthesis fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            import io
            import soundfile as sf
            import numpy as np

            # Use provided voice or default
            selected_voice = voice or self.config.tts_voice or "af_heart"

            # Generate speech - kokoro returns a generator of Result objects
            audio_samples = []
            for result_obj in self.pipe(text, voice=selected_voice):
                # Result objects have an 'audio' attribute containing the audio data
                if hasattr(result_obj, 'audio'):
                    chunk = result_obj.audio
                else:
                    chunk = result_obj

                # Convert chunk to numpy array
                if isinstance(chunk, (list, tuple)):
                    audio_samples.extend(chunk)
                elif isinstance(chunk, np.ndarray):
                    audio_samples.extend(chunk.flatten().tolist())
                else:
                    # Try to convert to array
                    try:
                        chunk_arr = np.asarray(chunk).flatten()
                        audio_samples.extend(chunk_arr.tolist())
                    except:
                        audio_samples.append(float(chunk))

            # Convert to numpy array
            if len(audio_samples) > 0:
                audio = np.array(audio_samples, dtype=np.float32)
            else:
                # Empty audio
                audio = np.array([], dtype=np.float32)

            # Convert to bytes (Kokoro outputs at 24kHz)
            buffer = io.BytesIO()
            sf.write(buffer, audio, 24000, format='WAV')
            buffer.seek(0)

            return buffer.read()

        except Exception as e:
            raise Exception(f"Speech synthesis failed with Kokoro: {str(e)}")

    def get_available_voices(self) -> list[str]:
        """
        Get list of available voice identifiers.

        Returns:
            List with single default voice
        """
        return ["default"]

    def synthesize_to_file(self, text: str, output_path: str, voice: str = None) -> None:
        """
        Synthesize text to speech and save to file.

        Args:
            text: Text to convert to speech
            output_path: Path to save audio file
            voice: Voice identifier (not used)

        Raises:
            ValueError: If text is empty
            Exception: If synthesis or file writing fails
        """
        audio_bytes = self.synthesize(text, voice)

        try:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
        except Exception as e:
            raise Exception(f"Failed to write audio file: {str(e)}")