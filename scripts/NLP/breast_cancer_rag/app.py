## LAB EXERCISE 03 - Hugo Iglesias Pombo

import streamlit as st #streamlit for the web app (user interface)
import tempfile #tempfile for temporary file storage (i use it to store the uploaded PDFs)
import os #os for file path manipulation (i use it to save the uploaded PDFs in a temporary directory)
import sys #sys for system-specific parameters and functions (i use it to check the python version or the current working directory) 
# Page settings
st.set_page_config(page_title="PDF RAG System", page_icon="📄", layout="wide") #this line is for set page metadata ()
st.title("📄 Sistema de RAG para PDFs")
st.markdown("Carga tus PDFs y haz preguntas sobre ellos usando llama3:8b")

# Import basic dependencies
try:
    from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader # for loading PDFs fast and robustly respectively
    from langchain.text_splitter import RecursiveCharacterTextSplitter # for splitting text into chunks
    from langchain.chat_models import ChatOllama # connects langchain with Ollama for using the llama3:8b model
    from langchain.chains import RetrievalQA # for question answering, it builds a chain that retrieves relevant documents and generates answers
    from langchain.vectorstores import FAISS # for storing and retrieving vectors (embeddings) efficiently, with this we can search the embeddings faster and more efficiently
    from langchain.embeddings import OllamaEmbeddings # for generating embeddings using the llama3:8b model
    st.success("✅ Dependencias cargadas correctamente")
except Exception as e: # if any of the imports fail, it will show an error message and stop the app
    st.error(f"Error de importación: {str(e)}")
    st.info("Instala las dependencias necesarias con: pip install langchain langchain-community pypdf unstructured faiss-cpu pdf2image")
    st.stop()

# I verify if Ollama is installed and running
try:
    import subprocess # subprocess for running shell commands 
    result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)  # run the command 'ollama list' to check if the model is available stdout is the standard output and stderr is the standard error
    if "llama3:8b" in result.stdout: # check if the model is in the output 
        st.success("✅ Modelo llama3:8b detectado correctamente") 
    else:
        st.warning("⚠️ No se encontró llama3:8b. Asegúrate de tenerlo instalado")
        st.code("ollama pull llama3:8b", language="bash") # this line is for pull the model from the server (if you don't have it installed)
except Exception as e: # if any of the commands fail, it will show an error message and stop the app
    st.error(f"Error al verificar Ollama: {str(e)}")
    st.info("Asegúrate de que Ollama esté instalado y ejecutándose")
    st.stop()

# Sidebar configuration
with st.sidebar:
    st.header("Configuración") # this line is for set the sidebar header 
    
    # PDF loading method selection, st.radio creates a radio button for selecting the PDF loading method (a radio button is a type of input that allows the user to select one option from a list of options)
    pdf_loader_type = st.radio( 
        "Método para cargar PDFs",
        ["PyPDFLoader (rápido)", "UnstructuredPDFLoader (robusto)"],
        index=0
    )
    
    # Chunks configuration for splitting the text into smaller parts (chunks) for processing
    chunk_size = st.slider( 
        "Tamaño de chunks",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )
    # overlap is the amount of text that is shared between two consecutive chunks
    # this is useful for maintaining context between chunks
    chunk_overlap = st.slider(
        "Superposición de chunks",
        min_value=0,
        max_value=500,
        value=100,
        step=50
    )
    
    # Retrieval configuration for setting the number of chunks to retrieve and the temperature for the model
    # temperature is a parameter that controls the randomness of the model's output
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
    st.info("Esta app fue desarrollada por Hugo Iglesias Pombo para la asignatura de NLP con la idea de crear un RAG para almacenar documentación y guías médicas y responder a preguntas en base a ellas")

# Main app configuration
if 'vectorstore' not in st.session_state: # this line is for check if the vectorstore is in the session state (with session state i mean the memory of the app) if  not, it will create it
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state: # this line is for check if the processed_files is in the session state (if not, it will create it)
    st.session_state.processed_files = []

# Function to process PDF files
def process_pdf(file, temp_dir): 
    """Process a PDF file and return documents"""
    file_path = os.path.join(temp_dir, file.name) # this line is for create the file path in the temporary directory
    
    # Guardar archivo en directorio temporal
    with open(file_path, "wb") as f: # this line is for open the file in binary mode (wb) and write the file to the temporary directory
        f.write(file.getbuffer()) # this line is for write the file to the temporary directory
    
    # Load the PDF file using the selected loader
    try:
        if "PyPDFLoader" in pdf_loader_type: # this line is for check if the pdf_loader_type is PyPDFLoader (if it is, it will use PyPDFLoader to load the PDF)
            loader = PyPDFLoader(file_path) # this line is for create the loader with PyPDFLoader
        else:
            loader = UnstructuredPDFLoader(file_path) # this line is for create the loader with UnstructuredPDFLoader
        
        documents = loader.load() # this line is for load the documents from the PDF file
        st.write(f"Extraídos {len(documents)} segmentos de {file.name}") # this line is for show the number of segments extracted from the PDF file
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
            st.write(f"Extraídos {len(documents)} segmentos con método alternativo")
            return documents
        except Exception as e2:
            st.error(f"Error con método alternativo: {str(e2)}")
            return []

# Loading documents section
st.header("1. Cargar Documentos")
# File uploader for PDF files (is from streamlit, it allows the user to upload files from their local machine)
uploaded_files = st.file_uploader(
    "Carga tus archivos PDF",
    type=["pdf"], # this line is for set the file type to pdf
    accept_multiple_files=True # this line is for allow multiple files to be uploaded
)

process_button = st.button("Procesar Documentos")

# Document processing
if process_button and uploaded_files: # this line is for check if the process button is clicked and if there are uploaded files
    with st.spinner("Procesando documentos..."): # this line is for show a spinner while the documents are being processed
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir: # this line is for create a temporary directory to store the uploaded files
            # Process each PDF
            all_docs = [] # this line is for create a list to store all the documents extracted from the PDFs
            st.session_state.processed_files = [] # this line is for create a list to store the names of the processed files
            
            for file in uploaded_files: # this line is for iterate over the uploaded files
                with st.status(f"Procesando {file.name}..."): # this line is for show the status of the processing
                    docs = process_pdf(file, temp_dir) # this line is for process the PDF file and extract the documents
                    if docs:
                        all_docs.extend(docs) # this line is for extend the all_docs list with the documents extracted from the PDF file
                        st.session_state.processed_files.append(file.name) # this line is for append the name of the processed file to the processed_files list
            
            if all_docs: # this line is for check if there are any documents extracted from the PDFs
                # Split text into chunks
                with st.status("Dividiendo texto en chunks..."): # this line is for show the status of the chunking
                    # Split the text into chunks using RecursiveCharacterTextSplitter, a class that splits the text into smaller parts (chunks) for processing
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    chunks = text_splitter.split_documents(all_docs)
                    st.write(f"Creados {len(chunks)} chunks de texto")
                
                # Create embeddings with Ollama
                with st.status("Creando embeddings con Ollama..."):
                    try:
                        embeddings = OllamaEmbeddings(model="llama3:8b") # this line is for create the embeddings with the llama3:8b model, ollamaembeddings is a class that generates embeddings using the llama3:8b model
                        
                        # Crear vectorstore
                        vectorstore = FAISS.from_documents(chunks, embeddings) # this line is for create the vectorstore with the chunks and the embeddings, FAISS is a class that stores and retrieves vectors (embeddings) efficiently with this we can search the embeddings faster and more efficiently
                        st.session_state.vectorstore = vectorstore
                        
                        st.success(f"✅ Procesados {len(st.session_state.processed_files)} PDFs con éxito")
                    except Exception as e:
                        st.error(f"Error al crear embeddings: {str(e)}")
                        st.info("Verificar que Ollama está funcionando correctamente")

# Show processed files
if st.session_state.processed_files:
    st.header("2. Documentos Procesados")
    for name in st.session_state.processed_files:
        st.markdown(f"- `{name}`")
    
    # Section for asking questions about the documents
    st.header("3. Preguntas sobre los documentos")
    question = st.text_input("Escribe tu pregunta acerca de los documentos")
    
    if question and st.session_state.vectorstore: # this line is for check if the question is not empty and if the vectorstore is in the session state (if it is, it will create it)
        with st.spinner("Generando respuesta..."):
            try:
                # Create LLM
                llm = ChatOllama(model="llama3:8b", temperature=temperature) # this line is for create the LLM with the llama3:8b model and the temperature set in the sidebar
                
                # Create RetrievalQA chain
                # RetrievalQA is a class that builds a chain that retrieves relevant documents and generates answers
                # from the retrieved documents, it uses the LLM to generate the answer
                qa_chain = RetrievalQA.from_chain_type( # this line is for create the qa_chain with the LLM and the vectorstore
                    llm=llm,
                    chain_type="stuff", # this line is for set the chain type to stuff (it means that it will use the LLM to generate the answer)
                    retriever=st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": k_retrievals} 
                    ), # this line is for set the retriever to the vectorstore and set the number of chunks to retrieve
                    return_source_documents=True
                ) # return_source_documents=True means that it will return the documents used to generate the answer
                
                # Generate answer
                import time # this line is for import the time module to measure the time taken to generate the answer
                start_time = time.time()
                response = qa_chain({"query": question})
                end_time = time.time()
                
                # Show answer
                st.markdown("### Respuesta:")
                st.markdown(response["result"])
                
                # Show context used to generate the answer
                with st.expander("Ver contexto utilizado"): # this line is for create an expander to show the context used to generate the answer
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Fuente {i+1}**") # this line is for show the source of the document
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content) # this line is for show the content of the document (if the content is too long, it will show only the first 500 characters)
                        st.markdown("---")
                
                st.info(f"Respuesta generada en {(end_time - start_time):.2f} segundos")
            except Exception as e:
                st.error(f"Error al generar respuesta: {str(e)}")
                st.code(str(e), language="python")
else:
    st.info("Carga y procesa documentos PDF primero para poder hacer preguntas sobre ellos.")

# Debug information
with st.expander("Información de depuración"): #this line is for create an expander to show the debug information, i mean the information about the app
    st.write(f"Python version: {sys.version}") 
    st.write(f"Directorio actual: {os.getcwd()}")
    st.write(f"Estado de la sesión: {list(st.session_state.keys())}") # this line is for show the session state keys (the keys are the names of the variables in the session state)