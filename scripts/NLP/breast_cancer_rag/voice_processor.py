import streamlit as st
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
import tempfile
import os
import time
import librosa
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from TTS.api import TTS

class AdvancedVoiceProcessor:
    def __init__(self, use_gpu=None):
        """
        Inicializa el procesador de voz avanzado con modelos de última generación
        
        Args:
            use_gpu: True/False para forzar uso de GPU, None para detección automática
        """
        # Detectar si estamos en Apple Silicon y configurar dispositivo apropiado
        if torch.backends.mps.is_available():
            self.device = "mps"  # Usar aceleración Metal en Mac con Apple Silicon
            st.info("Usando aceleración Metal (MPS) para Apple Silicon")
        elif torch.cuda.is_available():
            self.device = "cuda"  # GPU NVIDIA
            st.info("Usando aceleración CUDA para GPU NVIDIA")
        else:
            self.device = "cpu"
            st.info("Usando CPU para procesamiento (no se detectó GPU compatible)")
            
        st.info(f"Inicializando modelos de procesamiento de voz en {self.device}...")
        
        # Inicializar modelo Whisper para reconocimiento de voz (ASR)
        try:
            # Usar modelo small para balance entre rendimiento y precisión
            self.whisper_model_id = "openai/whisper-tiny"  # Opciones: tiny, base, small, medium, large
            self.whisper_processor = AutoProcessor.from_pretrained(self.whisper_model_id)
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.whisper_model_id, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            
            # Transferir modelo al dispositivo seleccionado
            if self.device == "mps":
                # Para Apple Silicon, algunos componentes pueden no ser compatibles con MPS
                # así que verificamos y usamos CPU como fallback si es necesario
                try:
                    self.whisper_model.to(self.device)
                except Exception as e:
                    st.warning(f"No se pudo usar MPS para Whisper, usando CPU: {str(e)}")
                    self.device = "cpu"
                    self.whisper_model.to("cpu")
            else:
                self.whisper_model.to(self.device)
            
            # Pipeline de Whisper (más simple de usar)
            self.asr = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                tokenizer=self.whisper_processor.tokenizer,
                feature_extractor=self.whisper_processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=False,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=self.device,
            )
            
            st.success("✅ Modelo Whisper cargado correctamente")
        except Exception as e:
            st.error(f"Error al cargar modelo Whisper: {str(e)}")
            self.asr = None
            
        # Inicializar modelo TTS (text to speech)
        try:
            # Intenta varios modelos TTS hasta encontrar uno disponible
            models_to_try = [
                "tts_models/es/css10/vits",
                "tts_models/es/mai/tacotron2-DDC",
                "tts_models/multilingual/multi-dataset/your_tts",
            ]
            
            self.tts = None
            for model_name in models_to_try:
                try:
                    self.tts = TTS(model_name=model_name, progress_bar=True)
                    st.success(f"Modelo TTS cargado: {model_name}")
                    break
                except Exception as model_e:
                    st.warning(f"No se pudo cargar {model_name}, intentando otro modelo...")
            
            if self.tts is None:
                raise Exception("No se pudo cargar ningún modelo TTS")
                
        except Exception as e:
            st.error(f"Error al cargar modelo TTS: {str(e)}")
            self.tts = None
    
    def record_audio(self, duration=10, sample_rate=16000):
        """
        Graba audio desde el micrófono (versión simplificada)
        """
        # Crear un archivo temporal
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_file.close()
        
        # Preparar UI
        status = st.empty()
        progress_bar = st.progress(0)
        status.info("Preparando micrófono...")
        
        # Configuración de grabación
        audio_data = []
        
        def callback(indata, frame_count, time_info, status):
            if status:
                print(f"Error en callback: {status}")
            audio_data.append(indata.copy())
        
        # Inicializar variable de control
        stop_recording = False
        if 'stop_button_pressed' not in st.session_state:
            st.session_state.stop_button_pressed = False
        
        # Función para el botón de detener
        def stop_recording_callback():
            st.session_state.stop_button_pressed = True
        
        # Botón para detener grabación
        st.button("⏹️ Detener grabación", 
                 on_click=stop_recording_callback,
                 key=f"stop_button_{temp_file.name}")  # Usar nombre de archivo como clave única
        
        # Iniciar grabación
        status.info("🎙️ Grabando... Habla ahora")
        
        try:
            with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
                # Mostrar progreso mientras grabamos
                start_time = time.time()
                for i in range(100):
                    if st.session_state.stop_button_pressed or time.time() - start_time > duration:
                        break
                    progress_bar.progress(i/100)
                    time.sleep(duration/100)
            
            # Resetear el botón para la próxima vez
            st.session_state.stop_button_pressed = False
            
            # Procesar audio grabado
            if audio_data:
                status.info("Procesando audio...")
                audio_array = np.concatenate(audio_data, axis=0)
                audio_array = audio_array.flatten()
                
                # Normalizar
                if np.max(np.abs(audio_array)) > 0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                # Guardar a archivo
                sf.write(temp_file.name, audio_array, sample_rate)
                status.success("✅ Audio grabado")
                return temp_file.name, audio_array
            else:
                status.error("No se detectó audio")
                return None, None
        except Exception as e:
            status.error(f"Error al grabar: {str(e)}")
            return None, None
    
    def transcribe_audio(self, audio_path=None, audio_array=None, sample_rate=16000):
        """
        Transcribe audio usando Whisper
        
        Args:
            audio_path: Ruta al archivo de audio (opcional)
            audio_array: Array numpy con datos de audio (opcional)
            sample_rate: Tasa de muestreo del audio
            
        Returns:
            text: Texto transcrito
        """
        if self.asr is None:
            st.error("Modelo de reconocimiento de voz no disponible")
            return ""
            
        if audio_path is None and audio_array is None:
            st.error("Se requiere archivo de audio o array de audio")
            return ""
            
        try:
            with st.spinner("Transcribiendo audio con Whisper..."):
                # Configurar parámetros de generación sin usar 'language' directamente
                generate_kwargs = {
                    "task": "transcribe",
                    # Eliminar "batch_size" de aquí
                }
                # En versiones más nuevas, se especifica el idioma de esta forma
                if audio_path:
                    result = self.asr(audio_path, generate_kwargs=generate_kwargs)
                else:
                    result = self.asr({"array": audio_array, "sampling_rate": sample_rate}, generate_kwargs=generate_kwargs)
        # Si la salida está en inglés, podemos intentar forzar el idioma de otra manera
        # en una segunda pasada si es necesario
                # Obtener el texto transcrito
                transcription = result["text"]
                return transcription
        except Exception as e:
            st.error(f"Error al transcribir audio: {str(e)}")
            return ""
    
    def text_to_speech(self, text, output_path=None):
        """
        Convierte texto a voz usando modelo avanzado
        
        Args:
            text: Texto a convertir en voz
            output_path: Ruta de salida (opcional)
            
        Returns:
            path: Ruta al archivo de audio generado
        """
        if self.tts is None:
            st.error("Modelo TTS no disponible")
            return None
            
        # Crear archivo temporal si no se especifica salida
        if output_path is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            output_path = temp_file.name
            
        try:
            with st.spinner("Generando voz..."):
                # Normalizar el texto para evitar errores con caracteres especiales
                text = text.replace("\n", " ").strip()
                
                # Limitar texto para procesamiento más rápido 
                # (especialmente en CPU)
                if len(text) > 500:
                    text = text[:500] + "... Continúa leyendo en la pantalla para más información."
                
                # Generar audio
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker=None,  # No hay hablantes específicos para este modelo
                    language="es"
                )
                
                return output_path
        except Exception as e:
            st.error(f"Error al generar voz: {str(e)}")
            return None

def audio_recorder_and_transcriber():
    """
    Componente simplificado para grabar audio y transcribirlo
    """
    # Usar un método más sencillo para evitar problemas de estado
    if 'voice_state' not in st.session_state:
        st.session_state.voice_state = 'ready'
    
    # Contenedor principal para el componente
    voice_container = st.container()
    
    with voice_container:
        if st.session_state.voice_state == 'ready':
            st.write("📣 Presiona el botón para grabar audio")
            if st.button("🎙️ Grabar audio", key="record_simple"):
                # Cambiar estado e iniciar grabación
                st.session_state.voice_state = 'recording'
                st.experimental_rerun()
                
        elif st.session_state.voice_state == 'recording':
            # Mostrar que está grabando
            st.write("🔴 Grabando... Habla ahora")
            progress = st.progress(0)
            
            try:
                # Crear procesador y grabar
                processor = AdvancedVoiceProcessor()
                audio_path, audio_array = processor.record_audio(duration=10)  # Duración fija para simplificar
                
                if audio_path:
                    # Guardar en estado
                    st.session_state.audio_path = audio_path
                    st.session_state.voice_state = 'transcribing'
                    st.experimental_rerun()
                else:
                    st.error("No se pudo grabar audio")
                    st.session_state.voice_state = 'ready'
            except Exception as e:
                st.error(f"Error en grabación: {str(e)}")
                st.session_state.voice_state = 'ready'
                
        elif st.session_state.voice_state == 'transcribing':
            st.write("⏳ Procesando audio...")
            
            try:
                # Reproducir audio grabado
                if 'audio_path' in st.session_state:
                    st.audio(st.session_state.audio_path)
                
                # Transcribir
                processor = AdvancedVoiceProcessor()
                transcription = processor.transcribe_audio(audio_path=st.session_state.audio_path)
                
                if transcription:
                    st.session_state.transcription = transcription
                    st.session_state.voice_state = 'done'
                    st.experimental_rerun()
                else:
                    st.error("No se pudo transcribir el audio")
                    st.session_state.voice_state = 'ready'
            except Exception as e:
                st.error(f"Error en transcripción: {str(e)}")
                st.session_state.voice_state = 'ready'
                
        elif st.session_state.voice_state == 'done':
            # Mostrar resultado
            st.success(f"🎯 Texto transcrito: {st.session_state.transcription}")
            st.audio(st.session_state.audio_path)
            
            # Botón para reiniciar
            if st.button("🔄 Nueva grabación", key="reset_recording"):
                st.session_state.voice_state = 'ready'
                st.experimental_rerun()
            
            # Devolver el texto transcrito
            return st.session_state.transcription
    
    # Si no llegamos al estado 'done', no hay texto
    return None

# Función para generar y reproducir voz a partir de texto
def generate_and_play_speech(text):
    """
    Genera y reproduce voz a partir de texto
    
    Args:
        text: Texto a convertir en voz
    """
    processor = AdvancedVoiceProcessor()
    
    # Limitar texto para evitar procesamiento excesivo
    max_length = 500
    if len(text) > max_length:
        display_text = text[:max_length] + "..."
        speak_text = text[:max_length] + " La respuesta continúa en pantalla."
    else:
        display_text = text
        speak_text = text
    
    with st.status("Generando respuesta por voz"):
        st.write(f"Procesando: {display_text[:100]}...")
        
        # Generar audio
        audio_path = processor.text_to_speech(speak_text)
        
        if audio_path:
            st.write("✅ Audio generado correctamente")
            # Reproducir audio
            st.audio(audio_path)
        else:
            st.error("No se pudo generar el audio")

# Función para verificar disponibilidad y estado de los modelos de voz
def check_voice_capabilities():
    """
    Comprueba si los modelos y bibliotecas necesarios están disponibles
    
    Returns:
        available (bool): True si el procesamiento de voz está disponible
        status (str): Mensaje de estado
    """
    try:
        # Verificar disponibilidad de bibliotecas
        import torch
        import sounddevice
        import transformers
        
        # Verificar si TTS está disponible
        try:
            from TTS.api import TTS
            tts_available = True
        except ImportError:
            tts_available = False
        
        # Determinar mejor dispositivo para procesamiento
        if torch.backends.mps.is_available():
            device = "mps (Apple Silicon)"
        elif torch.cuda.is_available():
            device = f"cuda (GPU: {torch.cuda.get_device_name(0)})"
        else:
            device = "cpu"
        
        # Evaluar estado general
        if tts_available:
            return True, f"Procesamiento de voz disponible en {device}"
        else:
            return False, f"Procesamiento de voz parcialmente disponible (falta TTS) en {device}"
            
    except ImportError as e:
        return False, f"Faltan dependencias: {str(e)}"

# Función conveniente para integrar en la aplicación principal
def add_voice_controls_to_sidebar():
    """
    Añade controles de voz a la barra lateral de la aplicación
    
    Returns:
        dict: Configuración de voz (enabled, auto_read_responses)
    """
    st.sidebar.subheader("Procesamiento de Voz")
    
    # Verificar capacidades de voz
    voice_available, status_msg = check_voice_capabilities()
    
    if voice_available:
        st.sidebar.success(status_msg)
        enabled = st.sidebar.checkbox("Habilitar procesamiento de voz", value=True)
        
        if enabled:
            auto_read = st.sidebar.checkbox("Leer respuestas automáticamente", value=False)
            st.sidebar.info("Usando modelos Whisper y TTS para procesamiento de voz")
            
            # Opciones avanzadas
            with st.sidebar.expander("Opciones avanzadas de voz"):
                whisper_model = st.selectbox(
                    "Modelo Whisper",
                    ["tiny", "base", "small", "medium", "large"],
                    index=2  # Predeterminado: small
                )
                
                voice_speed = st.slider(
                    "Velocidad de voz",
                    0.5, 2.0, 1.0, 0.1
                )
            
            return {
                "enabled": enabled,
                "auto_read_responses": auto_read,
                "whisper_model": whisper_model,
                "voice_speed": voice_speed
            }
        else:
            return {"enabled": False}
    else:
        st.sidebar.warning(status_msg)
        st.sidebar.info("Para habilitar el procesamiento de voz, instala las dependencias necesarias con:\n\n```\npip install transformers torch torchaudio soundfile sounddevice librosa TTS\n```")
        return {"enabled": False}

# Función principal para añadir interfaz de voz a la conversación
def add_voice_interface_to_chat(messages=None, on_voice_input=None):
    """
    Añade interfaz de voz al chat
    
    Args:
        messages: Lista de mensajes de chat (opcional)
        on_voice_input: Función de callback para entrada de voz (opcional)
        
    Returns:
        voice_input: Texto reconocido si se usó entrada de voz
    """
    # Generar ID único para esta sesión
    import time
    session_id = int(time.time() * 1000)
    
    # Añadir controles de grabación de voz
    st.subheader("Entrada por Voz")
    voice_col1, voice_col2 = st.columns([3,2])
    
    with voice_col1:
        if st.button("🎙️ Hacer pregunta por voz", key=f"voice_button_{session_id}"):
            voice_input = audio_recorder_and_transcriber()
            
            if voice_input and on_voice_input:
                # Llamar al callback con la entrada de voz
                on_voice_input(voice_input)
                
                return voice_input
    
    with voice_col2:
        auto_read_responses = st.checkbox("🔊 Leer respuestas", value=False, key=f"voice_auto_read_{session_id}")
        
        # Si hay mensajes previos y auto_read está activado, leer último mensaje
        if auto_read_responses and messages and len(messages) > 0:
            last_message = messages[-1]
            if last_message["role"] == "assistant":
                generate_and_play_speech(last_message["content"])
    
    return None

# Si se ejecuta como script principal, mostrar demo
if __name__ == "__main__":
    st.title("Demostración de procesamiento de voz")
    
    st.markdown("""
    Esta es una demostración del módulo de procesamiento de voz para integrar con la aplicación RAG.
    
    ## Funcionalidades
    - Reconocimiento de voz con Whisper
    - Síntesis de voz con TTS
    - Optimizado para Apple Silicon
    
    Prueba las funciones abajo:
    """)
    
    tab1, tab2 = st.tabs(["Reconocimiento de Voz", "Síntesis de Voz"])
    
    with tab1:
        st.subheader("Transcripción de Voz a Texto")
        transcription = audio_recorder_and_transcriber()
        
        if transcription:
            st.info("Texto transcrito:")
            st.code(transcription)
    
    with tab2:
        st.subheader("Síntesis de Voz")
        text_input = st.text_area("Ingresa texto para convertir a voz", 
                                 "Hola, soy el asistente virtual para información sobre cáncer de mama. ¿En qué puedo ayudarte hoy?", 
                                 height=100)
        
        if st.button("Generar voz"):
            generate_and_play_speech(text_input)

## error:Error al transcribir audio: The following model_kwargs are not used by the model: ['batch_size'] (note: typos in the generate arguments will also show up in this list)
