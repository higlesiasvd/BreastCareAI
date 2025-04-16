import streamlit as st
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
import tempfile
import os
import time
import subprocess
import base64
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq

# Try simpler approach for TTS
try:
    import gtts
    from gtts import gTTS
    gtts_available = True
except ImportError:
    gtts_available = False

class AdvancedVoiceProcessor:
    def __init__(self, whisper_model_size="tiny"):
        # Save configuration
        self.whisper_model_size = whisper_model_size
        
        # Detect device
        if torch.backends.mps.is_available():
            self.device = "mps"  # Use Metal acceleration on Macs with Apple Silicon
            st.info("Using Metal acceleration (MPS) for Apple Silicon")
        elif torch.cuda.is_available():
            self.device = "cuda"  # NVIDIA GPU
            st.info("Using CUDA acceleration for NVIDIA GPU")
        else:
            self.device = "cpu"
            st.info("Using CPU for processing (no compatible GPU detected)")
            
        st.info(f"Initializing voice processing models on {self.device}...")
        
        # Initialize Whisper model for speech recognition (ASR)
        try:
            # Use specified model size
            self.whisper_model_id = f"openai/whisper-{self.whisper_model_size}"
            st.info(f"Loading Whisper model: {self.whisper_model_id}")
            
            self.whisper_processor = AutoProcessor.from_pretrained(self.whisper_model_id)
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.whisper_model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Transfer model to selected device
            if self.device == "mps":
                try:
                    self.whisper_model.to(self.device)
                except Exception as e:
                    st.warning(f"Could not use MPS for Whisper, using CPU instead: {str(e)}")
                    self.device = "cpu"
                    self.whisper_model.to("cpu")
            else:
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
            
            st.success("✅ Whisper model loaded successfully")
        except Exception as e:
            st.error(f"Error loading Whisper model: {str(e)}")
            self.asr = None
            
        # Check TTS availability
        if gtts_available:
            st.success("✅ Google TTS available")
        else:
            st.warning("⚠️ Google TTS not available. Install with: pip install gtts")
    
    def record_audio(self, duration=10, sample_rate=16000):
        """
        Record audio from microphone
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()
        
        status = st.empty()
        progress_bar = st.progress(0)
        status.info("Preparing microphone...")
        
        audio_data = []
        
        def callback(indata, frame_count, time_info, status):
            if status:
                print(f"Error in callback: {status}")
            audio_data.append(indata.copy())
        
        if 'stop_button_pressed' not in st.session_state:
            st.session_state.stop_button_pressed = False
        
        def stop_recording_callback():
            st.session_state.stop_button_pressed = True
        
        st.button("⏹️ Stop Recording", 
                 on_click=stop_recording_callback,
                 key=f"stop_button_{temp_file.name}")
        
        status.info("🎙️ Recording... Speak now")
        
        try:
            with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
                start_time = time.time()
                for i in range(100):
                    if st.session_state.stop_button_pressed or time.time() - start_time > duration:
                        break
                    progress_bar.progress(i/100)
                    time.sleep(duration/100)
            
            st.session_state.stop_button_pressed = False
            
            if audio_data:
                status.info("Processing audio...")
                audio_array = np.concatenate(audio_data, axis=0)
                audio_array = audio_array.flatten()
                
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                sf.write(temp_file.name, audio_array, sample_rate)
                status.success("✅ Audio recorded")
                return temp_file.name, audio_array
            else:
                status.error("No audio detected")
                return None, None
        except Exception as e:
            status.error(f"Error recording: {str(e)}")
            return None, None
    
    def transcribe_audio(self, audio_path=None, audio_array=None, sample_rate=16000):
        """
        Transcribe audio using Whisper
        """
        if self.asr is None:
            st.error("Speech recognition model not available")
            return ""
            
        if audio_path is None and audio_array is None:
            st.error("Audio file or audio array required")
            return ""
            
        try:
            with st.spinner("Transcribing audio with Whisper..."):
                generate_kwargs = {
                    "task": "transcribe",
                    "language": "en"
                }
                
                if audio_path:
                    result = self.asr(audio_path, generate_kwargs=generate_kwargs)
                else:
                    result = self.asr({"array": audio_array, "sampling_rate": sample_rate}, generate_kwargs=generate_kwargs)
                
                transcription = result["text"]
                return transcription
        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            import traceback
            st.code(traceback.format_exc(), language="python")
            return ""

def gtts_generate_speech(text, output_path=None):
    """Simple function to generate speech using gTTS"""
    st.write("Attempting to generate speech with Google TTS...")
    
    if not gtts_available:
        st.error("Google TTS not available. Install with: pip install gtts")
        return None
    
    if not output_path:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()
        output_path = temp_file.name
    
    # Normalize text
    text = text.replace("\n", " ").strip()
    
    # Limit text 
    if len(text) > 500:
        text = text[:500] + "... Continue reading on screen for more information."
    
    try:
        # Create gTTS object with explicit lang setting
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to file
        tts.save(output_path)
        st.success(f"✅ Speech generated with Google TTS to {output_path}")
        return output_path
    except Exception as e:
        st.error(f"Google TTS error: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return None

def audio_recorder_and_transcriber():
    """
    Simplified component to record audio and transcribe it
    """
    if 'voice_state' not in st.session_state:
        st.session_state.voice_state = 'ready'
    
    whisper_model = st.session_state.get('whisper_model', 'tiny')
    
    voice_container = st.container()
    
    with voice_container:
        if st.session_state.voice_state == 'ready':
            st.write("📣 Press the button to record audio")
            if st.button("🎙️ Record audio", key="record_simple"):
                st.session_state.voice_state = 'recording'
                st.experimental_rerun()
                
        elif st.session_state.voice_state == 'recording':
            st.write("🔴 Recording... Speak now")
            progress = st.progress(0)
            
            try:
                processor = AdvancedVoiceProcessor(whisper_model_size=whisper_model)
                audio_path, audio_array = processor.record_audio(duration=10)
                
                if audio_path:
                    st.session_state.audio_path = audio_path
                    st.session_state.voice_state = 'transcribing'
                    st.experimental_rerun()
                else:
                    st.error("Could not record audio")
                    st.session_state.voice_state = 'ready'
            except Exception as e:
                st.error(f"Error in recording: {str(e)}")
                st.session_state.voice_state = 'ready'
                
        elif st.session_state.voice_state == 'transcribing':
            st.write("⏳ Processing audio...")
            
            try:
                if 'audio_path' in st.session_state:
                    st.audio(st.session_state.audio_path)
                
                processor = AdvancedVoiceProcessor(whisper_model_size=whisper_model)
                transcription = processor.transcribe_audio(audio_path=st.session_state.audio_path)
                
                if transcription:
                    st.session_state.transcription = transcription
                    st.session_state.voice_state = 'done'
                    st.experimental_rerun()
                else:
                    st.error("Could not transcribe audio")
                    st.session_state.voice_state = 'ready'
            except Exception as e:
                st.error(f"Error in transcription: {str(e)}")
                st.session_state.voice_state = 'ready'
                
        elif st.session_state.voice_state == 'done':
            st.success(f"🎯 Transcribed text: {st.session_state.transcription}")
            st.audio(st.session_state.audio_path)
            
            if st.button("🔄 New recording", key="reset_recording"):
                st.session_state.voice_state = 'ready'
                st.experimental_rerun()
            
            return st.session_state.transcription
    
    return None

def generate_and_play_speech(text):
    """
    Generate and play speech from text - simplified version
    """
    audio_path = None
    
    with st.status("Generating speech...", expanded=True) as status:
        st.write(f"Processing text: {text[:100]}..." if len(text) > 100 else f"Processing text: {text}")
        
        # Use Google TTS directly
        audio_path = gtts_generate_speech(text)
        
        if audio_path:
            st.write("✅ Audio generated successfully")
            st.audio(audio_path)
            status.update(label="✅ Speech generated", state="complete")
            return audio_path
        else:
            st.error("Could not generate audio with any available method")
            status.update(label="❌ Speech generation failed", state="error")
            return None

def check_voice_capabilities():
    """
    Check if required models and libraries are available
    """
    try:
        import torch
        import sounddevice
        import transformers
        
        tts_available = gtts_available
        
        if torch.backends.mps.is_available():
            device = "mps (Apple Silicon)"
        elif torch.cuda.is_available():
            device = f"cuda (GPU: {torch.cuda.get_device_name(0)})"
        else:
            device = "cpu"
        
        if tts_available:
            return True, f"Voice processing available on {device} with Google TTS"
        else:
            return False, f"Voice processing partially available (missing TTS) on {device}"
            
    except ImportError as e:
        return False, f"Missing dependencies: {str(e)}"

def add_voice_controls_to_sidebar():
    """
    Add voice controls to sidebar - simplified version
    """
    st.sidebar.subheader("Voice Processing")
    
    voice_available, status_msg = check_voice_capabilities()
    
    if voice_available:
        st.sidebar.success(status_msg)
        enabled = st.sidebar.checkbox("Enable voice processing", value=True)
        
        if enabled:
            auto_read = st.sidebar.checkbox("Automatically read responses", value=False)
            
            st.session_state.voice_enabled = enabled
            st.session_state.auto_read_responses = auto_read
            
            whisper_model = st.sidebar.selectbox(
                "Whisper model",
                ["tiny", "base", "small"],
                index=0
            )
            
            st.session_state.whisper_model = whisper_model
            
            return {
                "enabled": enabled,
                "auto_read_responses": auto_read,
                "whisper_model": whisper_model
            }
        else:
            st.session_state.voice_enabled = False
            st.session_state.auto_read_responses = False
            return {"enabled": False}
    else:
        st.sidebar.warning(status_msg)
        st.sidebar.info("To enable voice processing, install the required dependencies with:\n\n```\npip install transformers torch torchaudio soundfile sounddevice librosa gtts\n```")
        st.session_state.voice_enabled = False
        st.session_state.auto_read_responses = False
        return {"enabled": False}

def add_voice_interface_to_chat(messages=None, on_voice_input=None):
    """
    Add voice interface to chat 
    """
    session_id = int(time.time() * 1000)
    
    st.subheader("Voice Interface")
    
    
    if st.button("🎙️ Record Voice", key=f"voice_button_{session_id}"):
        voice_input = audio_recorder_and_transcriber()
        
        if voice_input and on_voice_input:
            on_voice_input(voice_input)
            return voice_input
    
    return None

if __name__ == "__main__":
    st.title("Voice Processing Demonstration")
    
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = "tiny"
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = True
    if 'auto_read_responses' not in st.session_state:
        st.session_state.auto_read_responses = False
    
    st.markdown("""
    This is a simplified voice processing demo focusing on the most reliable components.
    
    ## Features
    - Speech recognition with Whisper
    - Text-to-speech with Google TTS
    - Explicit debugging
    """)
    
    voice_config = add_voice_controls_to_sidebar()
    
    tab1, tab2 = st.tabs(["Speech Recognition", "Text-to-Speech"])
    
    with tab1:
        st.subheader("Speech-to-Text Transcription")
        transcription = audio_recorder_and_transcriber()
        
        if transcription:
            st.info("Transcribed text:")
            st.code(transcription)
    
    with tab2:
        st.subheader("Text-to-Speech Synthesis (Direct)")
        text_input = st.text_area("Enter text to convert to speech", 
                                 "Hello, I'm the virtual assistant for breast cancer information. How can I help you today?", 
                                 height=100)
        
        if st.button("Generate speech"):
            # Use the simplified direct approach
            audio_path = gtts_generate_speech(text_input)
            if audio_path:
                st.audio(audio_path)