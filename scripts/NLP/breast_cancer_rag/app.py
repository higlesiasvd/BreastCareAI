import streamlit as st
import tempfile
import os
import sys
import pickle
from datetime import datetime
import re
import pandas as pd

# Page settings
st.set_page_config(page_title="Sistema RAG para Cáncer de Mama", page_icon="🎗️", layout="wide")
st.title("🎗️ Sistema de RAG para Asesoramiento sobre Cáncer de Mama")
st.markdown("Accede a información médica verificada sobre cáncer de mama")

# Create required directories if they don't exist
os.makedirs("vectorstores", exist_ok=True)

# Medical disclaimer - Important for ethical and legal considerations
def show_medical_disclaimer():
    with st.expander("⚠️ Información importante sobre esta herramienta", expanded=True):
        st.markdown("""
        Esta aplicación proporciona información educativa basada en documentos médicos verificados, 
        pero **no sustituye el consejo médico profesional**. Consulte siempre con su equipo médico 
        antes de tomar decisiones sobre su salud.
        
        La información se extrae automáticamente de los documentos cargados y, aunque se hace todo 
        lo posible para garantizar su precisión, puede contener errores o estar desactualizada.
        """)
        must_accept = st.checkbox("Entiendo que esta herramienta es solo informativa y no reemplaza el consejo médico")
        return must_accept

# Show disclaimer at the beginning
disclaimer_accepted = show_medical_disclaimer()
if not disclaimer_accepted:
    st.warning("Por favor, acepte el aviso para continuar usando la aplicación.")
    st.stop()

# Import basic dependencies
try:
    from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chat_models import ChatOllama
    from langchain.chains import RetrievalQA, ConversationalRetrievalChain
    from langchain.vectorstores import FAISS
    from langchain.embeddings import OllamaEmbeddings
    from langchain.memory import ConversationBufferMemory
    st.success("✅ Dependencias cargadas correctamente")
except Exception as e:
    st.error(f"Error de importación: {str(e)}")
    st.info("Instala las dependencias necesarias con: pip install langchain langchain-community pypdf unstructured faiss-cpu pdf2image pandas")
    st.stop()

# Verify if Ollama is installed and running
try:
    import subprocess
    result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if "llama3:8b" in result.stdout:
        st.success("✅ Modelo llama3:8b detectado correctamente")
    else:
        st.warning("⚠️ No se encontró llama3:8b. Asegúrate de tenerlo instalado")
        st.code("ollama pull llama3:8b", language="bash")
except Exception as e:
    st.error(f"Error al verificar Ollama: {str(e)}")
    st.info("Asegúrate de que Ollama esté instalado y ejecutándose")
    st.stop()

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = "General"
if 'collections' not in st.session_state:
    st.session_state.collections = {
        "General": {"files": [], "vectorstore": None, "last_updated": None},
        "Diagnóstico": {"files": [], "vectorstore": None, "last_updated": None},
        "Tratamientos": {"files": [], "vectorstore": None, "last_updated": None},
        "Post-operatorio": {"files": [], "vectorstore": None, "last_updated": None},
        "Nutrición": {"files": [], "vectorstore": None, "last_updated": None}
    }
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'patient_profile' not in st.session_state:
    st.session_state.patient_profile = {
        "age": 45,
        "stage": "Pre-diagnóstico",
        "preferences": ["Información básica"]
    }

# Persistent storage functions
def save_vectorstore(vs, collection_name):
    """Save vectorstore to disk"""
    if vs:
        os.makedirs("vectorstores", exist_ok=True)
        filename = f"vectorstores/{collection_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pkl"
        try:
            with open(filename, "wb") as f:
                pickle.dump(vs, f)
            st.session_state.collections[collection_name]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            return True
        except Exception as e:
            st.error(f"Error al guardar la base de vectores: {str(e)}")
            return False
    return False

def load_vectorstore(collection_name):
    """Load vectorstore from disk"""
    # Find most recent file for this collection
    files = [f for f in os.listdir("vectorstores") if f.startswith(collection_name.lower().replace(' ', '_'))]
    if not files:
        return None
    
    # Sort by date (newest first)
    files.sort(reverse=True)
    try:
        with open(os.path.join("vectorstores", files[0]), "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar la base de vectores: {str(e)}")
        return None

# Medical glossary for automatic term highlighting
medical_terms = {
    "mastectomía": "Extirpación quirúrgica de una mama o parte de ella",
    "metástasis": "Propagación del cáncer desde su sitio original a otras partes del cuerpo",
    "carcinoma": "Tipo de cáncer que comienza en las células epiteliales",
    "biopsia": "Extracción de una pequeña muestra de tejido para examinarla",
    "mamografía": "Radiografía de las mamas para detectar cáncer",
    "quimioterapia": "Tratamiento con fármacos para destruir células cancerosas",
    "radioterapia": "Uso de radiación para destruir células cancerosas",
    "lumpectomía": "Cirugía para extirpar solo el tumor y parte del tejido normal circundante",
    "estadificación": "Proceso para determinar la extensión del cáncer",
    "ganglio centinela": "Primer ganglio linfático donde se propagaría el cáncer",
    "hormonoterapia": "Tratamiento que bloquea o elimina hormonas para detener o ralentizar el crecimiento del cáncer",
    "reconstrucción mamaria": "Cirugía para recrear la forma de la mama tras una mastectomía",
    "marcador tumoral": "Sustancia en sangre que puede indicar la presencia de cáncer",
    "oncólogo": "Médico especializado en el tratamiento del cáncer",
    "triple negativo": "Tipo de cáncer de mama que no tiene receptores de estrógeno, progesterona ni HER2",
    "BRCA1": "Gen cuyas mutaciones aumentan el riesgo de cáncer de mama y ovario",
    "BRCA2": "Gen cuyas mutaciones aumentan el riesgo de cáncer de mama y ovario",
    "mamografía digital": "Tipo de mamografía que utiliza rayos X para crear imágenes digitales",
    "ecografía mamaria": "Uso de ondas sonoras para crear imágenes del tejido mamario",
    "resonancia magnética": "Técnica de imagen que utiliza campos magnéticos y ondas de radio",
}

def highlight_medical_terms(text):
    """Resalta términos médicos y ofrece definiciones"""
    for term, definition in medical_terms.items():
        # Case insensitive search for the term
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(f'<span title="{definition}" style="color:#FF4B4B;font-weight:bold">{term}</span>', text)
    
    # Wrap in markdown to enable HTML rendering
    return f"<div>{text}</div>"

def verify_medical_accuracy(response):
    """Verifica la precisión médica de las respuestas generadas"""
    # Check if response contains absolute medical claims
    contains_absolute = any(phrase in response.lower() for phrase in [
        "siempre cura", "garantiza", "100% efectivo", "nunca falla", "única solución"
    ])
    
    # Check if response mentions limitations
    mentions_limitations = any(phrase in response.lower() for phrase in [
        "consulte con su médico", "puede variar", "caso específico", "individual", 
        "según su situación", "opciones disponibles"
    ])
    
    # Check if response avoids personal medical advice
    avoids_personal_advice = not any(phrase in response.lower() for phrase in [
        "debería tomar", "le recomiendo que", "su mejor opción es", "usted debe", 
        "tiene que hacerse"
    ])
    
    # Determine confidence level
    if contains_absolute or not avoids_personal_advice:
        confidence = "Baja"
    elif not mentions_limitations:
        confidence = "Media"
    else:
        confidence = "Alta"
    
    # Add disclaimer based on confidence
    if confidence == "Baja":
        disclaimer = """
        ⚠️ **Aviso importante**: Esta respuesta puede contener afirmaciones médicas absolutas o consejos personalizados.
        Recuerde que esta es una herramienta informativa y debe consultar con profesionales médicos para su caso específico.
        """
    elif confidence == "Media":
        disclaimer = """
        ⚠️ **Nota**: Esta respuesta proporciona información general. Su situación médica personal puede requerir
        consideraciones específicas. Consulte siempre con su equipo médico.
        """
    else:
        disclaimer = """
        ℹ️ Esta información es educativa. Cada persona es diferente y requiere atención personalizada
        de profesionales médicos cualificados.
        """
    
    return confidence, disclaimer

# Sidebar configuration with patient profile and document collections
with st.sidebar:
    st.header("Configuración")
    
    # Patient profile section
    st.subheader("Perfil del Paciente")
    with st.expander("Configurar perfil"):
        st.session_state.patient_profile["age"] = st.number_input(
            "Edad", 
            min_value=18, 
            max_value=100, 
            value=st.session_state.patient_profile["age"]
        )
        
        st.session_state.patient_profile["stage"] = st.selectbox(
            "Etapa",
            ["Pre-diagnóstico", "Recién diagnosticado", "En tratamiento", "Post-tratamiento", "Superviviente"],
            index=["Pre-diagnóstico", "Recién diagnosticado", "En tratamiento", "Post-tratamiento", "Superviviente"].index(st.session_state.patient_profile["stage"])
        )
        
        st.session_state.patient_profile["preferences"] = st.multiselect(
            "Preferencias de información",
            ["Información básica", "Detalles técnicos", "Opciones de tratamiento", "Estudios clínicos"],
            default=st.session_state.patient_profile["preferences"]
        )
    
    # Collections management
    st.subheader("Colecciones de Documentos")
    
    collection_options = list(st.session_state.collections.keys()) + ["Nueva colección..."]
    selected_collection = st.selectbox(
        "Seleccionar colección",
        collection_options,
        index=collection_options.index(st.session_state.current_collection) if st.session_state.current_collection in collection_options else 0
    )
    
    # Handle new collection creation
    if selected_collection == "Nueva colección...":
        new_collection_name = st.text_input("Nombre de la nueva colección")
        if st.button("Crear colección") and new_collection_name:
            if new_collection_name not in st.session_state.collections:
                st.session_state.collections[new_collection_name] = {
                    "files": [], 
                    "vectorstore": None,
                    "last_updated": None
                }
                st.session_state.current_collection = new_collection_name
                st.success(f"Colección '{new_collection_name}' creada")
                st.experimental_rerun()
    else:
        st.session_state.current_collection = selected_collection
    
    # Show collection info
    if st.session_state.current_collection in st.session_state.collections:
        collection = st.session_state.collections[st.session_state.current_collection]
        st.write(f"Documentos en esta colección: {len(collection['files'])}")
        if collection["last_updated"]:
            st.write(f"Última actualización: {collection['last_updated']}")
    
    # Load saved vectorstore if available
    if st.button("Cargar colección guardada"):
        with st.spinner("Cargando colección guardada..."):
            loaded_vs = load_vectorstore(st.session_state.current_collection)
            if loaded_vs:
                st.session_state.collections[st.session_state.current_collection]["vectorstore"] = loaded_vs
                st.session_state.vectorstore = loaded_vs
                st.success(f"✅ Colección '{st.session_state.current_collection}' cargada correctamente")
            else:
                st.warning(f"No se encontró una colección guardada para '{st.session_state.current_collection}'")
    
    # PDF loading method and other configs
    st.subheader("Configuración del Procesamiento")
    
    pdf_loader_type = st.radio(
        "Método para cargar PDFs",
        ["PyPDFLoader (rápido)", "UnstructuredPDFLoader (robusto)"],
        index=0
    )
    
    chunk_size = st.slider(
        "Tamaño de chunks",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )
    
    chunk_overlap = st.slider(
        "Superposición de chunks",
        min_value=0,
        max_value=500,
        value=100,
        step=50
    )
    
    k_retrievals = st.slider(
        "Número de chunks a recuperar",
        min_value=1,
        max_value=8,
        value=4,
        step=1
    )
    
    temperature = st.slider(
        "Temperatura",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1
    )

    st.markdown("---")
    st.markdown("### Sobre esta aplicación")
    st.info("Esta app fue desarrollada para crear un RAG especializado en cáncer de mama, permitiendo almacenar documentación y guías médicas y responder a preguntas basadas en evidencia científica.")

# Function to process PDF files
def process_pdf(file, temp_dir):
    """Process a PDF file and return documents"""
    file_path = os.path.join(temp_dir, file.name)
    
    # Save file to temporary directory
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Load the PDF file using the selected loader
    try:
        if "PyPDFLoader" in pdf_loader_type:
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredPDFLoader(file_path)
        
        documents = loader.load()
        
        # Add metadata to documents - important for source tracking and categorization
        for doc in documents:
            doc.metadata["collection"] = st.session_state.current_collection
            doc.metadata["file_name"] = file.name
            doc.metadata["date_added"] = datetime.now().strftime("%Y-%m-%d")
        
        st.write(f"Extraídos {len(documents)} segmentos de {file.name}")
        return documents
    except Exception as e:
        st.error(f"Error al procesar {file.name}: {str(e)}")
        
        # Try alternative method if the first one fails
        try:
            st.info(f"Intentando método alternativo para {file.name}...")
            if "PyPDFLoader" in pdf_loader_type:
                loader = UnstructuredPDFLoader(file_path)
            else:
                loader = PyPDFLoader(file_path)
            
            documents = loader.load()
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata["collection"] = st.session_state.current_collection
                doc.metadata["file_name"] = file.name
                doc.metadata["date_added"] = datetime.now().strftime("%Y-%m-%d")
                
            st.write(f"Extraídos {len(documents)} segmentos con método alternativo")
            return documents
        except Exception as e2:
            st.error(f"Error con método alternativo: {str(e2)}")
            return []

# Main content
st.header("1. Cargar Documentos")
uploaded_files = st.file_uploader(
    "Carga tus archivos PDF con información sobre cáncer de mama",
    type=["pdf"],
    accept_multiple_files=True
)

# Show currently loaded collection
st.caption(f"Colección actual: **{st.session_state.current_collection}**")

process_button = st.button("Procesar Documentos")

# Document processing
if process_button and uploaded_files:
    with st.spinner("Procesando documentos..."):
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each PDF
            all_docs = []
            processed_files = []
            
            for file in uploaded_files:
                with st.status(f"Procesando {file.name}..."):
                    docs = process_pdf(file, temp_dir)
                    if docs:
                        all_docs.extend(docs)
                        processed_files.append(file.name)
            
            if all_docs:
                # Split text into chunks
                with st.status("Dividiendo texto en chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = text_splitter.split_documents(all_docs)
                    st.write(f"Creados {len(chunks)} chunks de texto")
                
                # Create embeddings with Ollama
                with st.status("Creando embeddings con Ollama..."):
                    try:
                        embeddings = OllamaEmbeddings(model="llama3:8b")
                        
                        # Create vectorstore
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                        
                        # Update collection in session state
                        st.session_state.collections[st.session_state.current_collection]["files"].extend(processed_files)
                        st.session_state.collections[st.session_state.current_collection]["vectorstore"] = vectorstore
                        st.session_state.vectorstore = vectorstore
                        
                        # Save vectorstore to disk
                        if save_vectorstore(vectorstore, st.session_state.current_collection):
                            st.success("✅ Base de vectores guardada en disco")
                        
                        st.success(f"✅ Procesados {len(processed_files)} PDFs con éxito")
                    except Exception as e:
                        st.error(f"Error al crear embeddings: {str(e)}")
                        st.info("Verificar que Ollama está funcionando correctamente")

# Show processed files for current collection
current_collection_data = st.session_state.collections.get(st.session_state.current_collection, {})
if current_collection_data.get("files"):
    st.header("2. Documentos Procesados")
    st.subheader(f"Colección: {st.session_state.current_collection}")
    
    for name in current_collection_data.get("files", []):
        st.markdown(f"- `{name}`")
    
    # Section for asking questions about the documents
    st.header("3. Preguntas sobre Cáncer de Mama")
    
    # Frequently asked questions section
    st.subheader("Preguntas Frecuentes")
    
    # Customize FAQs based on patient stage
    stage = st.session_state.patient_profile["stage"]
    
    # Define FAQs based on patient stage
    if stage == "Pre-diagnóstico":
        faq_questions = [
            "¿Cuáles son los factores de riesgo para cáncer de mama?",
            "¿Cómo se realiza el autoexamen de mama?",
            "¿Qué síntomas debo vigilar?",
            "¿Cuándo debo hacerme una mamografía?"
        ]
    elif stage == "Recién diagnosticado":
        faq_questions = [
            "¿Cuáles son las opciones de tratamiento disponibles?",
            "¿Qué significan los diferentes estadios del cáncer?",
            "¿Cómo puedo prepararme para la cirugía?",
            "¿Qué preguntas debo hacer a mi oncólogo?"
        ]
    elif stage == "En tratamiento":
        faq_questions = [
            "¿Cómo manejar los efectos secundarios de la quimioterapia?",
            "¿Qué alimentos son recomendables durante el tratamiento?",
            "¿Cómo puedo cuidar mi piel durante la radioterapia?",
            "¿Qué ejercicios son seguros durante el tratamiento?"
        ]
    elif stage == "Post-tratamiento":
        faq_questions = [
            "¿Cuándo y cómo se realizan los controles posteriores?",
            "¿Qué puedo hacer para reducir el riesgo de recurrencia?",
            "¿Cómo manejar el linfedema?",
            "¿Cuándo puedo retomar mis actividades normales?"
        ]
    else:  # Superviviente
        faq_questions = [
            "¿Qué controles necesito a largo plazo?",
            "¿Cómo manejar el miedo a la recurrencia?",
            "¿Qué ejercicios son recomendables para supervivientes?",
            "¿Cómo puedo apoyar a otras personas con cáncer de mama?"
        ]
    
    # Display FAQs in a grid
    cols = st.columns(2)
    for i, question in enumerate(faq_questions):
        with cols[i % 2]:
            if st.button(question, key=f"faq_{i}"):
                st.session_state.current_question = question
    
    # Question input
    user_question = st.text_input(
        "Escribe tu pregunta:",
        value=st.session_state.current_question
    )
    
    # Clear the current question after it's been used
    if user_question == st.session_state.current_question:
        st.session_state.current_question = ""
        
    # Process the question
    if user_question and current_collection_data.get("vectorstore"):
        with st.spinner("Generando respuesta..."):
            try:
                # Create LLM
                llm = ChatOllama(model="llama3:8b", temperature=temperature)
                
                # Create conversational chain with memory
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=current_collection_data["vectorstore"].as_retriever(
                        search_kwargs={"k": k_retrievals}
                    ),
                    memory=st.session_state.memory,
                    return_source_documents=True
                )
                
                # Modify the question to include patient context
                contextualized_question = f"""
                Pregunta de un paciente {st.session_state.patient_profile['age']} años, 
                en fase de '{st.session_state.patient_profile['stage']}',
                con preferencias por '{', '.join(st.session_state.patient_profile['preferences'])}':
                {user_question}
                """
                
                # Generate answer
                import time
                start_time = time.time()
                response = qa_chain({"question": contextualized_question})
                end_time = time.time()
                
                # Verify medical accuracy
                confidence, disclaimer = verify_medical_accuracy(response["answer"])
                
                # Show answer with highlighted medical terms
                st.markdown("### Respuesta:")
                
                # Add color-coding based on confidence
                confidence_color = {
                    "Alta": "green",
                    "Media": "orange",
                    "Baja": "red"
                }
                
                st.markdown(f"""
                <div style="border-left: 5px solid {confidence_color[confidence]}; padding-left: 10px;">
                {highlight_medical_terms(response["answer"])}
                </div>
                """, unsafe_allow_html=True)
                
                # Show disclaimer based on verification
                st.markdown(disclaimer)
                
                # Show context used to generate the answer
                with st.expander("Ver fuentes utilizadas"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Fuente {i+1}**: {doc.metadata.get('file_name', 'Desconocido')}")
                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                        st.markdown("---")
                
                st.info(f"Respuesta generada en {(end_time - start_time):.2f} segundos")
                
                # Show additional resources
                st.subheader("Recursos Adicionales")
                resources_cols = st.columns(3)
                
                with resources_cols[0]:
                    st.info("📋 **Guías Clínicas**\nRecursos oficiales para pacientes y familias.")
                    if st.button("Ver guías"):
                        st.markdown("""
                        - [Asociación Española Contra el Cáncer](https://www.contraelcancer.es/es/todo-sobre-cancer/tipos-cancer/cancer-mama)
                        - [American Cancer Society](https://www.cancer.org/es/cancer/tipos/cancer-de-seno.html)
                        - [Breast Cancer Now](https://breastcancernow.org/information-support)
                        """)
                
                with resources_cols[1]:
                    st.success("🏥 **Centros Especializados**\nCentros de referencia para cáncer de mama.")
                    if st.button("Buscar centros"):
                        st.markdown("""
                        Para encontrar centros especializados en su área, consulte:
                        
                        - El directorio de la Sociedad Española de Oncología Médica
                        - Unidades CSUR del Sistema Nacional de Salud
                        - Centros con certificación EUSOMA (European Society of Breast Cancer Specialists)
                        """)
                
                with resources_cols[2]:
                    st.warning("👩‍⚕️ **Apoyo Psicológico**\nRecursos de apoyo emocional durante el proceso.")
                    if st.button("Opciones de apoyo"):
                        st.markdown("""
                        - Grupos de apoyo locales para pacientes
                        - Servicios de psicooncología en hospitales
                        - Asociaciones de pacientes como FECMA (Federación Española de Cáncer de Mama)
                        - Líneas telefónicas de apoyo: AECC (900 100 036)
                        """)
                
                # Follow-up system
                st.subheader("Mi Plan de Seguimiento")
                
                tabs = st.tabs(["Calendario", "Medicación", "Ejercicios", "Nutrición"])
                
                with tabs[0]:
                    cal_cols = st.columns(2)
                    with cal_cols[0]:
                        st.date_input("Próxima cita médica")
                    with cal_cols[1]:
                        st.time_input("Hora de la cita")
                    
                    # Simple table for appointments
                    if "appointments" not in st.session_state:
                        st.session_state.appointments = []
                    
                    new_appointment = st.text_input("Descripción de la cita")
                    if st.button("Guardar cita") and new_appointment:
                        st.session_state.appointments.append(new_appointment)
                    
                    if st.session_state.appointments:
                        for i, appt in enumerate(st.session_state.appointments):
                            st.text(f"{i+1}. {appt}")
                
                with tabs[1]:
                    # Simple medication tracker
                    med_cols = st.columns(3)
                    with med_cols[0]:
                        med_name = st.text_input("Nombre del medicamento")
                    with med_cols[1]:
                        med_dose = st.text_input("Dosis")
                    with med_cols[2]:
                        med_freq = st.selectbox(
                            "Frecuencia", 
                            ["Una vez al día", "Dos veces al día", "Tres veces al día", "Cada 12 horas", "Semanal"]
                        )
                    
                    if st.button("Agregar medicamento") and med_name:
                        if "medications" not in st.session_state:
                            st.session_state.medications = []
                        st.session_state.medications.append({"name": med_name, "dose": med_dose, "frequency": med_freq})
                    
                    if "medications" in st.session_state and st.session_state.medications:
                        # Create a dataframe for better display
                        med_df = pd.DataFrame(st.session_state.medications)
                        st.dataframe(med_df)
                
                with tabs[2]:
                    st.markdown("""
                    ### Ejercicios Recomendados
                    
                    En función de su etapa de tratamiento, consulte con su médico antes de iniciar cualquier rutina de ejercicios. 
                    Algunos ejercicios que pueden ser beneficiosos:
                    
                    - Caminatas suaves (comenzando con 10-15 minutos)
                    - Ejercicios de movilidad para el brazo y hombro
                    - Yoga adaptado para pacientes de cáncer de mama
                    - Natación (especialmente en etapas de recuperación)
                    
                    **Importante**: Evite ejercicios de alto impacto sin aprobación médica.
                    """)
                
                with tabs[3]:
                    st.markdown("""
                    ### Recomendaciones Nutricionales
                    
                    Una alimentación equilibrada es fundamental durante todo el proceso:
                    
                    - Priorice alimentos frescos y no procesados
                    - Incluya proteínas magras (pollo, pescado, legumbres)
                    - Consuma abundantes frutas y verduras
                    - Mantenga una buena hidratación (8 vasos de agua al día)
                    - Limite el consumo de alcohol y evite el tabaco
                    
                    Durante la quimioterapia, puede ser útil:
                    - Comidas pequeñas y frecuentes
                    - Alimentos a temperatura ambiente
                    - Evitar olores fuertes
                    """)
                
            except Exception as e:
                st.error(f"Error al generar respuesta: {str(e)}")
                st.code(str(e), language="python")
else:
    st.info("Carga y procesa documentos PDF primero para poder hacer preguntas sobre ellos, o selecciona una colección guardada.")

# Debug information
with st.expander("Información de depuración"):
    st.write(f"Python version: {sys.version}")
    st.write(f"Directorio actual: {os.getcwd()}")
    st.write(f"Estado de la sesión: {list(st.session_state.keys())}")