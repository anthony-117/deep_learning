# Audio Module

This module provides Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities for the RAG chatbot.

## Features

### Speech-to-Text (STT)
- **Provider**: Groq Whisper (whisper-large-v3-turbo)
- **Supported formats**: WAV, MP3, M4A, OGG, FLAC
- **Language**: Auto-detection (configurable)

### Text-to-Speech (TTS)
- **Provider**: OpenAI TTS
- **Model**: tts-1 (or tts-1-hd for higher quality)
- **Available voices**: alloy, echo, fable, onyx, nova, shimmer
- **Configurable speed**: 0.25x to 4.0x

## Configuration

### Environment Variables

Set the following environment variables:

```bash
export GROQ_API_KEY="your-groq-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### Audio Configuration

```python
from src.audio import AudioConfig

config = AudioConfig(
    # STT settings
    stt_provider="groq",
    stt_model="whisper-large-v3-turbo",
    stt_language=None,  # Auto-detect

    # TTS settings
    tts_provider="openai",
    tts_model="tts-1",
    tts_voice="alloy",
    tts_speed=1.0,
)
```

## Usage

### Speech-to-Text

```python
from src.audio import GroqWhisperSTT, AudioConfig

# Initialize
config = AudioConfig()
stt = GroqWhisperSTT(config)

# Transcribe from file
text = stt.transcribe("path/to/audio.wav")

# Transcribe from bytes
with open("audio.mp3", "rb") as f:
    audio_bytes = f.read()
text = stt.transcribe(audio_bytes)
```

### Text-to-Speech

```python
from src.audio import OpenAITTS, AudioConfig

# Initialize
config = AudioConfig()
tts = OpenAITTS(config)

# Generate audio
audio_bytes = tts.synthesize("Hello, how can I help you today?")

# Save to file
tts.synthesize_to_file("Hello world!", "output.mp3")

# Change voice
audio_bytes = tts.synthesize("Hello!", voice="nova")

# Get available voices
voices = tts.get_available_voices()
print(voices)  # ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer']
```

## Web UI Integration

The audio module is integrated into the Streamlit web UI with the following features:

### Voice Input
- Click the microphone icon to record audio
- Audio is transcribed automatically using Groq Whisper
- Transcribed text populates the chat input

### Voice Output
- Click the "🔊 Play" button on any assistant message to generate TTS audio
- Audio player with playback controls appears below the message
- Optional auto-play: automatically generate and play audio for new responses

### Settings (Sidebar)
- **Enable Audio Features**: Toggle audio functionality on/off
- **TTS Voice**: Select from 6 available voices
- **TTS Speed**: Adjust playback speed (0.25x - 2.0x)
- **Auto-play responses**: Automatically play audio for new responses

## Architecture

```
src/audio/
├── __init__.py          # Module exports
├── base.py              # Abstract base classes (STTModel, TTSModel)
├── config.py            # AudioConfig dataclass
├── stt.py              # Speech-to-Text implementations
├── tts.py              # Text-to-Speech implementations
└── README.md           # This file
```

## Dependencies

- `groq>=0.11.0` - For Groq Whisper STT
- `openai>=1.58.1` - For OpenAI TTS
- `streamlit>=1.50.0` - For web UI integration

## Error Handling

Both STT and TTS implementations include comprehensive error handling:

- API key validation
- Audio format validation
- Network error handling
- Rate limiting handling
- Clear error messages for debugging

## Performance Considerations

### STT (Speech-to-Text)
- Groq Whisper is very fast (typically <1 second for short audio)
- Supports streaming for real-time transcription
- Handles various audio formats efficiently

### TTS (Text-to-Speech)
- OpenAI TTS generates audio quickly (typically <2 seconds)
- Audio is cached in session state to avoid regeneration
- MP3 format for optimal file size

## Future Enhancements

Potential improvements for the audio module:

1. **Additional Providers**:
   - ElevenLabs for higher-quality TTS
   - Local Whisper for offline STT
   - Coqui TTS for open-source TTS

2. **Features**:
   - Voice activity detection (VAD)
   - Real-time streaming STT
   - Audio preprocessing (noise reduction)
   - Voice cloning/customization

3. **UI Improvements**:
   - Keyboard shortcuts for voice input
   - Visual waveform display
   - Audio trimming/editing
   - Multi-language support UI