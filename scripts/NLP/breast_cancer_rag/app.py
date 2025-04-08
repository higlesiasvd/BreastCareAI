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
os.makedirs("knowledge_base", exist_ok=True)

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
    st.session_state.current_collection = "Base de conocimiento"
if 'collections' not in st.session_state:
    st.session_state.collections = {
        "Base de conocimiento": {"files": [], "vectorstore": None, "last_updated": None},
        "General": {"files": [], "vectorstore": None, "last_updated": None},
        "Diagnóstico": {"files": [], "vectorstore": None, "last_updated": None},
        "Tratamientos": {"files": [], "vectorstore": None, "last_updated": None},
        "Post-operatorio": {"files": [], "vectorstore": None, "last_updated": None},
        "Nutrición": {"files": [], "vectorstore": None, "last_updated": None}
    }
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'patient_profile' not in st.session_state:
    st.session_state.patient_profile = {
        "age": 45,
        "stage": "Pre-diagnóstico",
        "preferences": ["Información básica"]
    }
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state.knowledge_base_loaded = False

if 'pdf_loader_type' not in st.session_state:
    st.session_state.pdf_loader_type = "PyPDFLoader (rápido)"

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "all-minilm"

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

# Knowledge base functions
def create_knowledge_base_folder():
    """Create knowledge_base folder if it doesn't exist"""
    os.makedirs("knowledge_base", exist_ok=True)
    return os.path.exists("knowledge_base")

def get_knowledge_base_pdfs():
    """Get list of PDFs in the knowledge_base folder"""
    if not os.path.exists("knowledge_base"):
        return []
    return [f for f in os.listdir("knowledge_base") if f.lower().endswith(".pdf")]

def process_knowledge_base():
    """Process all PDFs in the knowledge_base folder and create vectorstore"""
    pdfs = get_knowledge_base_pdfs()
    if not pdfs:
        st.warning("No hay PDFs en la carpeta knowledge_base.")
        return None
    
    # Crear un contenedor para mostrar el progreso
    progress_container = st.container()
    progress_container.subheader("Progreso del procesamiento")
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    metrics_col1, metrics_col2, metrics_col3 = progress_container.columns(3)
    
    total_docs = 0
    processed_docs = 0
    total_chunks = 0
    
    with st.spinner(f"Procesando {len(pdfs)} PDFs de la base de conocimiento..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            all_docs = []
            
            # Primera pasada para contar documentos
            status_text.text("Contando documentos totales...")
            for pdf_name in pdfs:
                pdf_path = os.path.join("knowledge_base", pdf_name)
                try:
                    if "PyPDFLoader" in st.session_state.pdf_loader_type:
                        loader = PyPDFLoader(pdf_path)
                        # Solo contar páginas para PyPDFLoader
                        sample_docs = loader.load()
                        total_docs += len(sample_docs)
                    else:
                        # Para UnstructuredPDFLoader es más complejo contar páginas
                        total_docs += 1  # Contamos cada PDF como un documento
                except Exception as e:
                    pass
            
            metrics_col1.metric("Total PDFs", len(pdfs))
            metrics_col2.metric("Total documentos", total_docs)
            
            # Segunda pasada para procesar
            status_text.text("Procesando documentos...")
            for i, pdf_name in enumerate(pdfs):
                pdf_path = os.path.join("knowledge_base", pdf_name)
                
                try:
                    status_text.text(f"Procesando: {pdf_name}")
                    
                    # Usar el valor de la sesión en lugar de acceder directamente
                    if "PyPDFLoader" in st.session_state.pdf_loader_type:
                        loader = PyPDFLoader(pdf_path)
                    else:
                        loader = UnstructuredPDFLoader(pdf_path)
                    
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["collection"] = "Base de conocimiento"
                        doc.metadata["file_name"] = pdf_name
                        doc.metadata["date_added"] = datetime.now().strftime("%Y-%m-%d")
                    
                    all_docs.extend(docs)
                    processed_docs += len(docs)
                    metrics_col3.metric("Documentos procesados", processed_docs, delta=len(docs))
                    
                    # Actualizar barra de progreso basada en PDFs procesados
                    progress = min(1.0, (i + 1) / len(pdfs))
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error al procesar {pdf_name}: {str(e)}")
            
            if all_docs:
                # Split text into chunks
                status_text.text("Dividiendo texto en chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = text_splitter.split_documents(all_docs)
                total_chunks = len(chunks)
                metrics_col3.metric("Chunks generados", total_chunks)
                
                # Create embeddings and vectorstore
                try:
                    status_text.text("Generando embeddings con Ollama...")
                    embeddings = OllamaEmbeddings(model=st.session_state.embedding_model)
                    
                    # Mostrar progreso al crear los embeddings
                    with st.spinner("Creando vectorstore... (puede tardar unos minutos)"):
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                    
                    # Save to disk
                    status_text.text("Guardando vectorstore en disco...")
                    save_vectorstore(vectorstore, "Base de conocimiento")
                    
                    status_text.text("✅ Procesamiento completado con éxito")
                    progress_bar.progress(1.0)
                    
                    return vectorstore
                except Exception as e:
                    status_text.text(f"❌ Error al crear embeddings: {str(e)}")
                    st.error(f"Error al crear embeddings: {str(e)}")
                    return None
            return None

def add_to_knowledge_base(file):
    """Add a file to the knowledge_base folder"""
    create_knowledge_base_folder()
    file_path = os.path.join("knowledge_base", file.name)
    
    try:
        # Save file to knowledge base
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        return True
    except Exception as e:
        st.error(f"Error al añadir {file.name} a la base de conocimiento: {str(e)}")
        return False

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

# Carga inicial de la base de conocimiento
if not st.session_state.knowledge_base_loaded:
    # Create knowledge_base folder if it doesn't exist
    create_knowledge_base_folder()
    
    # Try to load pre-processed vectorstore
    loaded_kb = load_vectorstore("Base de conocimiento")
    
    if loaded_kb:
        st.session_state.collections["Base de conocimiento"] = {
            "files": get_knowledge_base_pdfs(),
            "vectorstore": loaded_kb,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.current_collection = "Base de conocimiento"
        st.session_state.vectorstore = loaded_kb
        st.session_state.knowledge_base_loaded = True
    else:
        # Process knowledge base if vectorstore doesn't exist
        vs = process_knowledge_base()
        if vs:
            st.session_state.collections["Base de conocimiento"] = {
                "files": get_knowledge_base_pdfs(),
                "vectorstore": vs,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.current_collection = "Base de conocimiento"
            st.session_state.vectorstore = vs
            st.session_state.knowledge_base_loaded = True

# Function to process PDF files
def process_pdf(file, temp_dir):
    """Process a PDF file and return documents"""
    file_path = os.path.join(temp_dir, file.name)
    
    # Save file to temporary directory
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Load the PDF file using the selected loader
    try:
        if "PyPDFLoader" in st.session_state.pdf_loader_type:
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
    
    embedding_model = st.selectbox(
        "Modelo para embeddings",
        ["all-minilm", "nomic-embed-text", "llama3:8b"],
        index=["all-minilm", "nomic-embed-text", "llama3:8b"].index(st.session_state.embedding_model) 
            if st.session_state.embedding_model in ["all-minilm", "nomic-embed-text", "llama3:8b"] else 0
    )
    st.session_state.embedding_model = embedding_model

    st.info("""
    **Recomendación de modelos:**
    - **all-minilm**: Rápido y eficiente, especializado en embeddings (recomendado)
    - **nomic-embed-text**: Alta calidad para textos, si está disponible
    - **llama3:8b**: Modelo general más grande, puede ser más lento
    """)
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
    
    # Show knowledge base status
    st.subheader("Base de Conocimiento")
    kb_files = get_knowledge_base_pdfs()
    st.write(f"PDFs en base de conocimiento: {len(kb_files)}")
    if kb_files:
        if st.button("Ver contenido de la base de conocimiento"):
            for file in kb_files:
                st.write(f"- {file}")
                
    if st.button("Recargar base de conocimiento"):
        with st.spinner("Recargando base de conocimiento..."):
            kb_vs = process_knowledge_base()
            if kb_vs:
                st.session_state.collections["Base de conocimiento"] = {
                    "files": get_knowledge_base_pdfs(), 
                    "vectorstore": kb_vs,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.success("✅ Base de conocimiento recargada")
            else:
                st.warning("No se pudo cargar la base de conocimiento")
    
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
    index=0 if st.session_state.pdf_loader_type == "PyPDFLoader (rápido)" else 1
    )
    st.session_state.pdf_loader_type = pdf_loader_type
    
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

# Main content
st.header("1. Cargar Documentos")
uploaded_files = st.file_uploader(
    "Carga tus archivos PDF con información sobre cáncer de mama",
    type=["pdf"],
    accept_multiple_files=True
)

# Show currently loaded collection
st.caption(f"Colección actual: **{st.session_state.current_collection}**")

# Option to add to knowledge base
add_to_kb = st.checkbox("Añadir a la base de conocimiento permanente", value=False)

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
                    # Add to knowledge base if requested
                    if add_to_kb:
                        if add_to_knowledge_base(file):
                            st.success(f"✅ {file.name} añadido a la base de conocimiento")
                    
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
                        
                        # Get collection to update
                        target_collection = st.session_state.current_collection
                        
                        # Create vectorstore
                        if target_collection in st.session_state.collections and st.session_state.collections[target_collection]["vectorstore"]:
                            # Add to existing vectorstore
                            existing_vs = st.session_state.collections[target_collection]["vectorstore"]
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                            
                            # Merge vectorstores
                            existing_vs.merge_from(vectorstore)
                            vectorstore = existing_vs
                        else:
                            # Create new vectorstore
                            vectorstore = FAISS.from_documents(chunks, embeddings)
                        
                        # Update collection in session state
                        if target_collection not in st.session_state.collections:
                            st.session_state.collections[target_collection] = {
                                "files": [],
                                "vectorstore": None,
                                "last_updated": None
                            }
                        
                        st.session_state.collections[target_collection]["files"].extend(processed_files)
                        st.session_state.collections[target_collection]["vectorstore"] = vectorstore
                        st.session_state.collections[target_collection]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                        st.session_state.vectorstore = vectorstore
                        
                        # Save vectorstore to disk
                        if save_vectorstore(vectorstore, target_collection):
                            st.success("✅ Base de vectores guardada en disco")
                        
                        st.success(f"✅ Procesados {len(processed_files)} PDFs con éxito")
                        
                        # If added to KB, update the knowledge base collection
                        if add_to_kb and "Base de conocimiento" in st.session_state.collections:
                            refresh_kb = process_knowledge_base()
                            if refresh_kb:
                                st.success("✅ Base de conocimiento actualizada")
                                
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
                
                # Create conversational chain with memory y output_key especificado
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=current_collection_data["vectorstore"].as_retriever(
                        search_kwargs={"k": k_retrievals}
                    ),
                    memory=st.session_state.memory,
                    return_source_documents=True,
                    output_key="answer"
                )
                
                # Modify the question to include patient context
                # Añadir instrucciones para el modelo sobre idioma y análisis de sentimiento
                contextualized_question = f"""
                INSTRUCCIONES IMPORTANTES:
                1. Responde SIEMPRE en español, independientemente del idioma de la pregunta.
                2. Realiza un análisis de sentimiento de la pregunta y adapta tu tono adecuadamente.
                3. Estructura tus respuestas en este formato:
                - RESPUESTA DIRECTA: Breve respuesta a la pregunta principal
                - CONTEXTO AMPLIADO: Información detallada y explicativa 
                - RECOMENDACIONES PRÁCTICAS: Pasos concretos o sugerencias aplicables
                - IMPORTANTE: Advertencias o consideraciones especiales si existen
                4. Cuando cites información, indica claramente de qué documento proviene.
                5. Si hay información contradictoria entre las fuentes, señálalo transparentemente.
                6. Expresa el nivel de consenso médico sobre el tema (alto, moderado o bajo).
                7. Si la pregunta está fuera del ámbito de la documentación, indícalo claramente y ofrece información general basada en consensos médicos verificados.
                8. Responde a todas las partes de preguntas múltiples o complejas.
                9. Usa ejemplos prácticos cuando sea apropiado para mejorar la comprensión.
                10. Incluye referencias a las secciones relevantes de los documentos consultados.

                CONTEXTO DEL PACIENTE:
                Paciente de {st.session_state.patient_profile['age']} años, 
                en fase de '{st.session_state.patient_profile['stage']}',
                preferencias de información: '{', '.join(st.session_state.patient_profile['preferences'])}',
                historial de consultas previas: {', '.join(st.session_state.previous_topics) if 'previous_topics' in st.session_state else 'Ninguna previa'}.

                PREGUNTA DEL PACIENTE:
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