# Voice Processing Module for Healthcare Applications

## Introduction

The Voice Processing Module provides comprehensive speech-to-text and text-to-speech capabilities for healthcare applications, with a specific focus on breast cancer information systems. This module enables natural voice interaction with AI assistants, making information more accessible through both speech recognition and audio responses.

Designed with a focus on reliability and ease of integration, the module combines state-of-the-art speech recognition through OpenAI's Whisper models with robust text-to-speech synthesis using Google's TTS service.

## Table of Contents

1. [Core Features](#core-features)
2. [Technical Architecture](#technical-architecture)
3. [Advanced Voice Processor](#advanced-voice-processor)
4. [Voice Interface Components](#voice-interface-components)
5. [Speech Recognition Pipeline](#speech-recognition-pipeline)
6. [Text-to-Speech Synthesis](#text-to-speech-synthesis)
7. [Integration with Streamlit](#integration-with-streamlit)
8. [Hardware Acceleration](#hardware-acceleration)
9. [Example Usage](#example-usage)
10. [Technical Reference](#technical-reference)

## Core Features

The Voice Processing Module offers several key capabilities:

1. **Speech Recognition**: Transcribe spoken audio to text using OpenAI's Whisper models
2. **Text-to-Speech**: Convert text responses to natural-sounding speech using Google TTS
3. **Hardware Acceleration**: Automatic detection and utilization of available GPU/MPS acceleration
4. **Interactive Recording**: Streamlit components for audio recording with visual feedback
5. **Seamless Integration**: Easy integration with chat interfaces and other Streamlit components
6. **Accessibility Focus**: Makes AI healthcare information accessible to users who prefer voice interaction

## Technical Architecture

The module is built around several key components:

1. **AdvancedVoiceProcessor**: Core class that handles model loading, audio processing, and transcription
2. **Audio Recording Functions**: Utilities for capturing audio from microphone input
3. **Transcription Pipeline**: Integration with Hugging Face Transformers for Whisper-based transcription
4. **TTS Components**: Text-to-speech synthesis using Google's gTTS
5. **Streamlit UI Components**: Ready-to-use interface elements for voice interaction

The architecture prioritizes reliability, with graceful fallbacks and comprehensive error handling to ensure the system remains functional even when certain components are unavailable.

## Advanced Voice Processor

The `AdvancedVoiceProcessor` class is the central component of the module, handling the initialization of speech recognition models and processing of audio input:

```python
class AdvancedVoiceProcessor:
    def __init__(self, whisper_model_size="tiny"):
        # Save configuration
        self.whisper_model_size = whisper_model_size
        
        # Detect device
        if torch.backends.mps.is_available():
            self.device = "mps"  # Use Metal acceleration on Macs with Apple Silicon
        elif torch.cuda.is_available():
            self.device = "cuda"  # NVIDIA GPU
        else:
            self.device = "cpu"
            
        # Initialize Whisper model for speech recognition (ASR)
        try:
            # Use specified model size
            self.whisper_model_id = f"openai/whisper-{self.whisper_model_size}"
            
            self.whisper_processor = AutoProcessor.from_pretrained(self.whisper_model_id)
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.whisper_model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Transfer model to selected device
            self.whisper_model.to(self.device)
            
            # Whisper Pipeline
            self.asr = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                return_timestamps=False,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=self.device,
            )
        except Exception as e:
            self.asr = None
```

The class automatically detects available hardware acceleration (CUDA for NVIDIA GPUs or MPS for Apple Silicon) and configures models accordingly. It also provides graceful error handling to ensure the application can continue running even if model initialization fails.

### Key Methods

The `AdvancedVoiceProcessor` provides several essential methods:

#### Audio Recording

```python
def record_audio(self, duration=10, sample_rate=16000):
    """
    Record audio from microphone
    
    Args:
        duration (int): Maximum recording duration in seconds
        sample_rate (int): Audio sample rate in Hz
        
    Returns:
        tuple: (file_path, audio_array) - Path to saved audio file and numpy array of audio data
    """
```

This method handles microphone access, audio buffering, and user interface elements for recording status.

#### Audio Transcription

```python
def transcribe_audio(self, audio_path=None, audio_array=None, sample_rate=16000):
    """
    Transcribe audio using Whisper
    
    Args:
        audio_path (str, optional): Path to audio file
        audio_array (numpy.ndarray, optional): Audio data as numpy array
        sample_rate (int): Sample rate of audio in Hz
        
    Returns:
        str: Transcribed text
    """
```

This method accepts either a file path or audio array and returns the transcribed text using the Whisper model.

## Voice Interface Components

The module provides ready-to-use Streamlit components for integrating voice capabilities into applications:

### Audio Recorder and Transcriber

```python
def audio_recorder_and_transcriber():
    """
    Simplified component to record audio and transcribe it
    
    Returns:
        str: Transcribed text or None if unsuccessful
    """
```

This function provides a complete user interface flow for audio recording and transcription, managing the various states of the process (ready, recording, transcribing, done).

### Voice Controls for Sidebar

```python
def add_voice_controls_to_sidebar():
    """
    Add voice controls to sidebar
    
    Returns:
        dict: Configuration settings for voice processing
    """
```

This function adds a comprehensive set of voice processing controls to the Streamlit sidebar, allowing users to:
- Enable/disable voice processing
- Toggle automatic reading of responses
- Select the Whisper model size (tiny, base, small)

### Voice Interface for Chat

```python
def add_voice_interface_to_chat(messages=None, on_voice_input=None):
    """
    Add voice interface to chat
    
    Args:
        messages (list, optional): List of chat messages
        on_voice_input (callable, optional): Callback function for voice input
        
    Returns:
        str: Voice input if successful, None otherwise
    """
```

This function integrates voice capabilities directly into chat interfaces, providing a button to initiate voice input and handling the callback when transcription completes.

## Speech Recognition Pipeline

The speech recognition pipeline uses OpenAI's Whisper models through the Hugging Face Transformers library:

1. **Model Loading**: Initialize the Whisper model and processor
2. **Audio Recording**: Capture audio from the microphone using sounddevice
3. **Preprocessing**: Convert audio to the format expected by Whisper
4. **Transcription**: Process audio through the Whisper pipeline
5. **Post-processing**: Clean up and return the transcribed text

The module supports different Whisper model sizes:
- `tiny`: Fastest, lowest accuracy, minimal resource requirements
- `base`: Balance of speed and accuracy, moderate resource requirements
- `small`: Higher accuracy, slower, higher resource requirements

## Text-to-Speech Synthesis

The module utilizes Google's Text-to-Speech (gTTS) service for speech synthesis:

```python
def gtts_generate_speech(text, output_path=None):
    """
    Simple function to generate speech using gTTS
    
    Args:
        text (str): Text to convert to speech
        output_path (str, optional): Path to save audio file
        
    Returns:
        str: Path to generated audio file or None if unsuccessful
    """
```

This function:
1. Normalizes input text (removing excess whitespace, etc.)
2. Truncates very long text to prevent issues
3. Generates speech using gTTS
4. Saves the audio to a file
5. Returns the path to the audio file

For playback in Streamlit applications, another helper function is provided:

```python
def generate_and_play_speech(text):
    """
    Generate and play speech from text
    
    Args:
        text (str): Text to convert to speech
        
    Returns:
        str: Path to audio file if successful, None otherwise
    """
```

This function handles the entire process from text to audible speech, including error handling and user feedback.

## Integration with Streamlit

The module is designed specifically for seamless integration with Streamlit applications, providing UI components and state management for voice interactions.

### State Management

The module uses Streamlit's session state to track the status of voice processing:

```python
if 'voice_state' not in st.session_state:
    st.session_state.voice_state = 'ready'
```

Possible states include:
- `ready`: Ready to begin recording
- `recording`: Currently recording audio
- `transcribing`: Processing recorded audio
- `done`: Transcription completed

### UI Components

The module provides several Streamlit UI components:

- Progress bars for recording status
- Audio playback for recorded speech
- Success/error messages for process status
- Buttons for initiating recording and generating speech
- Settings controls in the sidebar

### Configuration Options

Users can configure various aspects of voice processing:

- Enable/disable voice functionality
- Select Whisper model size
- Enable automatic reading of responses
- Adjust recording duration

## Hardware Acceleration

The module automatically detects and leverages available hardware acceleration:

```python
# Detect device
if torch.backends.mps.is_available():
    self.device = "mps"  # Use Metal acceleration on Macs with Apple Silicon
elif torch.cuda.is_available():
    self.device = "cuda"  # NVIDIA GPU
else:
    self.device = "cpu"
```

This enables optimized performance across different hardware configurations:

- **NVIDIA GPUs**: Uses CUDA acceleration with FP16 precision
- **Apple Silicon**: Uses Metal Performance Shaders (MPS) for acceleration
- **CPU-only**: Falls back to CPU processing with appropriate configurations

## Example Usage

### Basic Voice Recording and Transcription

```python
import streamlit as st
from voice_processor import audio_recorder_and_transcriber

st.title("Voice Transcription Demo")

st.write("Click the button below to start recording")
transcription = audio_recorder_and_transcriber()

if transcription:
    st.success("Transcription complete:")
    st.write(transcription)
```

### Text-to-Speech Example

```python
import streamlit as st
from voice_processor import gtts_generate_speech

st.title("Text-to-Speech Demo")

text_input = st.text_area("Enter text to convert to speech", 
                        "Hello, I'm the virtual assistant for breast cancer information.")

if st.button("Generate Speech"):
    audio_path = gtts_generate_speech(text_input)
    if audio_path:
        st.audio(audio_path)
```

### Complete Voice Chat Integration

```python
import streamlit as st
from voice_processor import add_voice_controls_to_sidebar, add_voice_interface_to_chat

st.title("Voice-Enabled Chat Demo")

# Add voice controls to sidebar
voice_config = add_voice_controls_to_sidebar()

# Chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Text input
prompt = st.chat_input("Type your message here")

# Handle text input
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Process message and generate response...

# Add voice interface
def on_voice_input(text):
    st.session_state.messages.append({"role": "user", "content": text})
    # Process message and generate response...

add_voice_interface_to_chat(
    messages=st.session_state.messages,
    on_voice_input=on_voice_input
)
```

## Technical Reference

### Dependencies

- `torch`: PyTorch for neural network operations
- `transformers`: Hugging Face Transformers for Whisper models
- `soundfile`: Audio file I/O
- `sounddevice`: Audio recording
- `gtts`: Google Text-to-Speech
- `streamlit`: UI framework

### Core Classes and Functions

#### AdvancedVoiceProcessor

```python
class AdvancedVoiceProcessor:
    def __init__(self, whisper_model_size="tiny")
    def record_audio(self, duration=10, sample_rate=16000)
    def transcribe_audio(self, audio_path=None, audio_array=None, sample_rate=16000)
```

#### Text-to-Speech Functions

```python
def gtts_generate_speech(text, output_path=None)
def generate_and_play_speech(text)
```

#### Streamlit Integration

```python
def audio_recorder_and_transcriber()
def add_voice_controls_to_sidebar()
def add_voice_interface_to_chat(messages=None, on_voice_input=None)
def check_voice_capabilities()
```

---

This Voice Processing Module enables natural speech interaction in healthcare applications, making information more accessible to users who prefer voice interaction or have difficulties with traditional text input methods. By combining state-of-the-art speech recognition with reliable text-to-speech capabilities, the module enhances the user experience of breast cancer information systems.
