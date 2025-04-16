import streamlit as st

import tempfile
import os
import sys
import pickle
from datetime import datetime
import re
import pandas as pd
from langchain.memory import ConversationBufferWindowMemory

# Try to import voice processing modules if available
try:
    from voice_processor import add_voice_controls_to_sidebar, add_voice_interface_to_chat, audio_recorder_and_transcriber
    voice_available = True
except ImportError:
    voice_available = False

# Page settings
st.set_page_config(page_title="RAG System for Breast Cancer", page_icon="🎗️", layout="wide")
st.title("🎗️ RAG System for Breast Cancer Counseling")
st.markdown("Access verified medical information about breast cancer")

# Custom styles to enhance the chat interface
st.markdown("""
<style>
div.stChatMessage {
    padding: 10px;
    border-radius: 15px;
    margin-bottom: 10px;
}
div.stChatMessage[data-testid="stChatMessageContent"] p {
    font-size: 16px !important;
}
div.stChatMessage[data-testid="stChatMessage"] div[data-testid="stChatMessageContent"] {
    padding: 10px;
}
div.stChatMessage.user {
    background-color: #e6f7ff;
    text-align: right;
}
div.stChatMessage.assistant {
    background-color: #f0f2f6;
}
</style>
""", unsafe_allow_html=True)

# Create required directories if they don't exist
os.makedirs("vectorstores", exist_ok=True)
os.makedirs("knowledge_base", exist_ok=True)

# Medical disclaimer - Important for ethical and legal considerations
def show_medical_disclaimer():
    """Display an important medical disclaimer and require user acceptance"""
    with st.expander("⚠️ Important Information About This Tool", expanded=True):
        st.markdown("""
        This application provides educational information based on verified medical documents, 
        but **does not replace professional medical advice**. Always consult with your medical team 
        before making decisions about your health.
        
        Information is automatically extracted from loaded documents, and while every effort 
        is made to ensure accuracy, it may contain errors or be outdated.
        """)
        must_accept = st.checkbox("I understand that this tool is informational only and does not replace medical advice")
        return must_accept

# Show disclaimer at the beginning
disclaimer_accepted = show_medical_disclaimer()
if not disclaimer_accepted:
    st.warning("Please accept the disclaimer to continue using the application.")
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
    from langchain.schema import BaseRetriever
    from langchain.schema.document import Document
    from typing import List, Dict, Any
    st.success("✅ Dependencies loaded successfully")
except Exception as e:
    st.error(f"Import error: {str(e)}")
    st.info("Install the required dependencies with: pip install langchain langchain-community pypdf unstructured faiss-cpu pdf2image pandas")
    st.stop()

# Verify if Ollama is installed and running
try:
    import subprocess
    result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    available_models = []
    
    # Check for available models
    if "llama3:8b" in result.stdout:
        available_models.append("llama3:8b")
        st.success("✅ Model llama3:8b detected successfully")
    else:
        st.warning("⚠️ llama3:8b not found. Make sure you have it installed")
        st.code("ollama pull llama3:8b", language="bash")
    
    # Check if the phi2-breast-cancer model is available
    if "phi2-breast-cancer" in result.stdout:
        available_models.append("phi2-breast-cancer")
        st.success("✅ Model phi2-breast-cancer detected successfully")
    else:
        st.warning("⚠️ phi2-breast-cancer not found. If you want to use this model, make sure it's installed")
        
    if not available_models:
        st.error("❌ No compatible models found. Install at least one of the required models.")
        st.stop()
        
except Exception as e:
    st.error(f"Error checking Ollama: {str(e)}")
    st.info("Make sure Ollama is installed and running")
    st.stop()

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'current_collection' not in st.session_state:
    st.session_state.current_collection = "Knowledge Base"
if 'collections' not in st.session_state:
    st.session_state.collections = {
        "Knowledge Base": {"files": [], "vectorstore": None, "last_updated": None},
        "General": {"files": [], "vectorstore": None, "last_updated": None},
        "Diagnosis": {"files": [], "vectorstore": None, "last_updated": None},
        "Treatments": {"files": [], "vectorstore": None, "last_updated": None},
        "Post-operative": {"files": [], "vectorstore": None, "last_updated": None},
        "Nutrition": {"files": [], "vectorstore": None, "last_updated": None}
    }
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
    )
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'patient_profile' not in st.session_state:
    st.session_state.patient_profile = {
        "age": 45,
        "stage": "Pre-diagnosis",
        "preferences": ["Basic Information"]
    }
if 'knowledge_base_loaded' not in st.session_state:
    st.session_state.knowledge_base_loaded = False
if 'previous_topics' not in st.session_state:
    st.session_state.previous_topics = []

if 'pdf_loader_type' not in st.session_state:
    st.session_state.pdf_loader_type = "PyPDFLoader (fast)"

if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "all-minilm"

# Add response model to the session
if 'llm_model' not in st.session_state:
    # Set default model based on availability
    if "llama3:8b" in available_models:
        st.session_state.llm_model = "llama3:8b"
    elif "phi2-breast-cancer" in available_models:
        st.session_state.llm_model = "phi2-breast-cancer"
    else:
        st.session_state.llm_model = available_models[0] if available_models else "llama3:8b"

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
            st.error(f"Error saving vector database: {str(e)}")
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
        st.error(f"Error loading vector database: {str(e)}")
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
        st.warning("No PDFs in the knowledge_base folder.")
        return None
    
    # Create a container to show progress
    progress_container = st.container()
    progress_container.subheader("Processing Progress")
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    metrics_col1, metrics_col2, metrics_col3 = progress_container.columns(3)
    
    total_docs = 0
    processed_docs = 0
    total_chunks = 0
    
    with st.spinner(f"Processing {len(pdfs)} PDFs from knowledge base..."):
        with tempfile.TemporaryDirectory() as temp_dir:
            all_docs = []
            
            # First pass to count documents
            status_text.text("Counting total documents...")
            for pdf_name in pdfs:
                pdf_path = os.path.join("knowledge_base", pdf_name)
                try:
                    if "PyPDFLoader" in st.session_state.pdf_loader_type:
                        loader = PyPDFLoader(pdf_path)
                        # Only count pages for PyPDFLoader
                        sample_docs = loader.load()
                        total_docs += len(sample_docs)
                    else:
                        # For UnstructuredPDFLoader it's more complex to count pages
                        total_docs += 1  # Count each PDF as one document
                except Exception as e:
                    pass
            
            metrics_col1.metric("Total PDFs", len(pdfs))
            metrics_col2.metric("Total documents", total_docs)
            
            # Second pass to process
            status_text.text("Processing documents...")
            for i, pdf_name in enumerate(pdfs):
                pdf_path = os.path.join("knowledge_base", pdf_name)
                
                try:
                    status_text.text(f"Processing: {pdf_name}")
                    
                    # Use the session value instead of accessing directly
                    if "PyPDFLoader" in st.session_state.pdf_loader_type:
                        loader = PyPDFLoader(pdf_path)
                    else:
                        loader = UnstructuredPDFLoader(pdf_path)
                    
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata["collection"] = "Knowledge Base"
                        doc.metadata["file_name"] = pdf_name
                        doc.metadata["date_added"] = datetime.now().strftime("%Y-%m-%d")
                    
                    all_docs.extend(docs)
                    processed_docs += len(docs)
                    metrics_col3.metric("Processed documents", processed_docs, delta=len(docs))
                    
                    # Update progress bar based on processed PDFs
                    progress = min(1.0, (i + 1) / len(pdfs))
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing {pdf_name}: {str(e)}")
            
            if all_docs:
                # Split text into chunks
                status_text.text("Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=100
                )
                chunks = text_splitter.split_documents(all_docs)
                total_chunks = len(chunks)
                metrics_col3.metric("Generated chunks", total_chunks)
                
                # Create embeddings and vectorstore
                try:
                    status_text.text("Generating embeddings with Ollama...")
                    embeddings = OllamaEmbeddings(model=st.session_state.embedding_model)
                    
                    # Show progress when creating embeddings
                    with st.spinner("Creating vectorstore... (this may take a few minutes)"):
                        vectorstore = FAISS.from_documents(chunks, embeddings)
                    
                    # Save to disk
                    status_text.text("Saving vectorstore to disk...")
                    save_vectorstore(vectorstore, "Knowledge Base")
                    
                    status_text.text("✅ Processing completed successfully")
                    progress_bar.progress(1.0)
                    
                    return vectorstore
                except Exception as e:
                    status_text.text(f"❌ Error creating embeddings: {str(e)}")
                    st.error(f"Error creating embeddings: {str(e)}")
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
        st.error(f"Error adding {file.name} to knowledge base: {str(e)}")
        return False

# Medical glossary for automatic term highlighting
medical_terms = {
    "mastectomy": "Surgical removal of a breast or part of it",
    "metastasis": "Spread of cancer from its original site to other parts of the body",
    "carcinoma": "Type of cancer that begins in epithelial cells",
    "biopsy": "Removal of a small tissue sample for examination",
    "mammography": "X-ray of the breasts to detect cancer",
    "chemotherapy": "Treatment with drugs to destroy cancer cells",
    "radiotherapy": "Use of radiation to destroy cancer cells",
    "lumpectomy": "Surgery to remove only the tumor and part of the surrounding normal tissue",
    "staging": "Process to determine the extent of cancer",
    "sentinel node": "First lymph node where cancer would spread",
    "hormone therapy": "Treatment that blocks or removes hormones to stop or slow cancer growth",
    "breast reconstruction": "Surgery to recreate the breast shape after mastectomy",
    "tumor marker": "Substance in blood that may indicate the presence of cancer",
    "oncologist": "Doctor specializing in cancer treatment",
    "triple negative": "Type of breast cancer that lacks estrogen, progesterone, and HER2 receptors",
    "BRCA1": "Gene whose mutations increase the risk of breast and ovarian cancer",
    "BRCA2": "Gene whose mutations increase the risk of breast and ovarian cancer",
    "digital mammography": "Type of mammography that uses X-rays to create digital images",
    "breast ultrasound": "Use of sound waves to create images of breast tissue",
    "MRI": "Imaging technique that uses magnetic fields and radio waves",
}

def summarize_conversation(messages, llm_model="llama3:8b"):
    """Generate a summary of the current conversation"""
    if len(messages) < 4:  # Don't summarize short conversations
        return None
        
    # Convert messages to text format
    conversation_text = "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content'][:200]}" 
                                for m in messages[-10:]])  # Use only the last 10 messages
    
    # Create prompt to summarize
    summarize_prompt = f"""
    Below is a conversation between a user and an assistant about breast cancer.
    Summarize the key points discussed so far in 3-5 brief points.
    Don't add new information. Be concise.
    
    CONVERSATION:
    {conversation_text}
    
    KEY POINTS OF THE CONVERSATION (3-5 points):
    """
    
    try:
        # Get summary using the model
        llm = ChatOllama(model=llm_model, temperature=0.3)
        summary = llm.predict(summarize_prompt)
        return summary
    except Exception as e:
        st.warning(f"Could not generate summary: {str(e)}")
        return None

# Initialize conversation summary if it doesn't exist
if 'conversation_summary' not in st.session_state:
    st.session_state.conversation_summary = None

# Function to update the summary periodically
def update_conversation_summary():
    """Update the conversation summary when needed"""
    # Update the summary every 5 messages after the first 10
    if (len(st.session_state.messages) >= 10 and 
        len(st.session_state.messages) % 5 == 0):
        with st.spinner("Updating conversation context..."):
            conversation_summary = summarize_conversation(
                st.session_state.messages, 
                st.session_state.llm_model
            )
            if conversation_summary:
                st.session_state.conversation_summary = conversation_summary

# Improved document retrieval function
def enhanced_retrieval(vectorstore, current_question, previous_messages, previous_topics, k=4):
    """
    Enhanced document retrieval that takes into account the full conversation
    
    Args:
        vectorstore: Vector database for searching
        current_question: Current user question
        previous_messages: Conversation message history
        previous_topics: List of previous topics
        k: Number of documents to retrieve
        
    Returns:
        List of relevant documents
    """
    # Extract recent user questions
    recent_questions = []
    for msg in previous_messages[-10:]:  # Last 10 messages
        if msg["role"] == "user":
            recent_questions.append(msg["content"])
    
    # Detect medical terms in the current question
    medical_keywords = []
    current_question_lower = current_question.lower()
    for term in medical_terms.keys():
        if term.lower() in current_question_lower:
            medical_keywords.append(term)
    
    # Build enriched query for search
    enriched_query = current_question
    
    # Add recent conversation context if it exists
    if recent_questions and len(recent_questions) > 1:
        # Use only the last question different from the current one
        for q in reversed(recent_questions[:-1]):  # Excludes current question
            if q != current_question:
                enriched_query += f" Related to previous question: {q}"
                break
    
    # Add medical terms to improve retrieval
    if medical_keywords:
        enriched_query += f" Important medical terms: {', '.join(medical_keywords)}"
    
    # Perform two searches and combine results
    # 1. Search with enriched query
    docs_enriched = vectorstore.similarity_search(enriched_query, k=k)
    
    # 2. Search with just the question (as backup)
    docs_simple = vectorstore.similarity_search(current_question, k=k//2)
    
    # Combine and remove duplicates
    all_docs = []
    doc_contents = set()
    
    # First add documents from enriched query
    for doc in docs_enriched:
        if doc.page_content not in doc_contents:
            all_docs.append(doc)
            doc_contents.add(doc.page_content)
    
    # Then add unique documents from simple search
    for doc in docs_simple:
        if doc.page_content not in doc_contents:
            all_docs.append(doc)
            doc_contents.add(doc.page_content)
    
    # Limit to k documents
    return all_docs[:k]

def highlight_medical_terms(text):
    """Highlight medical terms and offer definitions"""
    for term, definition in medical_terms.items():
        # Case insensitive search for the term
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(f'<span title="{definition}" style="color:#FF4B4B;font-weight:bold">{term}</span>', text)
    
    # Wrap in markdown to enable HTML rendering
    return f"<div>{text}</div>"

def verify_medical_accuracy(response):
    """Verify the medical accuracy of generated responses"""
    # Check if response contains absolute medical claims
    contains_absolute = any(phrase in response.lower() for phrase in [
        "always cures", "guarantees", "100% effective", "never fails", "only solution"
    ])
    
    # Check if response mentions limitations
    mentions_limitations = any(phrase in response.lower() for phrase in [
        "consult with your doctor", "may vary", "specific case", "individual", 
        "according to your situation", "available options"
    ])
    
    # Check if response avoids personal medical advice
    avoids_personal_advice = not any(phrase in response.lower() for phrase in [
        "you should take", "I recommend that you", "your best option is", "you must", 
        "you have to get"
    ])
    
    # Determine confidence level
    if contains_absolute or not avoids_personal_advice:
        confidence = "Low"
    elif not mentions_limitations:
        confidence = "Medium"
    else:
        confidence = "High"
    
    # Add disclaimer based on confidence
    if confidence == "Low":
        disclaimer = """
        ⚠️ **Important notice**: This response may contain absolute medical claims or personalized advice.
        Remember this is an informational tool and you should consult with medical professionals for your specific case.
        """
    elif confidence == "Medium":
        disclaimer = """
        ⚠️ **Note**: This response provides general information. Your personal medical situation may require
        specific considerations. Always consult with your medical team.
        """
    else:
        disclaimer = """
        ℹ️ This information is educational. Each person is different and requires personalized care
        from qualified medical professionals.
        """
    
    return confidence, disclaimer

# Fixed PrecomputedRetriever class that properly implements the BaseRetriever interface
class PrecomputedRetriever(BaseRetriever):
    """A retriever that returns pre-computed documents regardless of the query."""
    
    def __init__(self, documents: List[Document]):
        """Initialize with the list of documents to be returned."""
        super().__init__()
        self._docs = documents
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Return pre-computed documents regardless of the query."""
        return self._docs
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """Asynchronously return pre-computed documents."""
        return self._docs

# Initial knowledge base loading
if not st.session_state.knowledge_base_loaded:
    # Create knowledge_base folder if it doesn't exist
    create_knowledge_base_folder()
    
    # Try to load pre-processed vectorstore
    loaded_kb = load_vectorstore("Knowledge Base")
    
    if loaded_kb:
        st.session_state.collections["Knowledge Base"] = {
            "files": get_knowledge_base_pdfs(),
            "vectorstore": loaded_kb,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.current_collection = "Knowledge Base"
        st.session_state.vectorstore = loaded_kb
        st.session_state.knowledge_base_loaded = True
    else:
        # Process knowledge base if vectorstore doesn't exist
        vs = process_knowledge_base()
        if vs:
            st.session_state.collections["Knowledge Base"] = {
                "files": get_knowledge_base_pdfs(),
                "vectorstore": vs,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.current_collection = "Knowledge Base"
            st.session_state.vectorstore = vs
            st.session_state.knowledge_base_loaded = True

# Function to reset conversation memory
def reset_conversation_memory():
    """Reset the conversation and its memory"""
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
        k=5
    )
    st.session_state.previous_topics = []
    if 'conversation_summary' in st.session_state:
        del st.session_state.conversation_summary
    st.success("Conversation reset")
    st.experimental_rerun()

# Function to prepare contextualized prompt
def prepare_contextualized_prompt(question, patient_profile, previous_topics, messages):
    """
    Prepare a contextualized prompt that includes relevant information
    from previous conversation and patient profile.
    """
    # Get conversation summary if it exists
    conversation_summary = st.session_state.get('conversation_summary', None)
    summary_text = f"""PREVIOUS CONVERSATION SUMMARY:
{conversation_summary}

""" if conversation_summary else ""
    
    # References to previous messages (last exchanges)
    last_exchanges = []
    if len(messages) >= 4:  # Get the last two exchanges (2 user-assistant pairs)
        for i in range(max(len(messages)-4, 0), len(messages), 2):
            if i+1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i+1]
                if user_msg["role"] == "user" and assistant_msg["role"] == "assistant":
                    # Try to shorten very long messages
                    user_content = user_msg["content"]
                    if len(user_content) > 300:
                        user_content = user_content[:300] + "..."
                    
                    assistant_content = assistant_msg["content"]
                    if len(assistant_content) > 300:
                        assistant_content = assistant_content[:300] + "..."
                    
                    last_exchanges.append(f"""USER: {user_content}
ASSISTANT: {assistant_content}""")
    
    recent_context = ""
    if last_exchanges:
        recent_context = f"""RECENT EXCHANGES:
{last_exchanges[-1]}

"""
    
    # Collector of important medical terms
    important_terms = []
    medical_terms_lower = {term.lower(): term for term in medical_terms.keys()}
    
    # Look for medical terms mentioned recently
    for msg in messages[-6:]:
        msg_content = msg["content"].lower()
        for term_lower, term in medical_terms_lower.items():
            if term_lower in msg_content and term not in important_terms:
                important_terms.append(term)
    
    medical_context = ""
    if important_terms:
        medical_context = f"""KEY MEDICAL TERMS MENTIONED:
{', '.join(important_terms)}

"""
    
    # Create the final contextualized prompt
    prompt = f"""
IMPORTANT INSTRUCTIONS:
1. ALWAYS respond in English, using precise but understandable medical language.
2. Maintain CONSISTENCY with the previous conversation and make explicit references to what has already been discussed.
3. Structure your responses in this format:
   - DIRECT ANSWER: Brief clear initial response
   - EXPANDED CONTEXT: Detailed explanatory information 
   - PRACTICAL RECOMMENDATIONS: Applicable suggestions when appropriate
4. If the question is related to previous topics, clearly establish the connection.
5. Adapt your level of detail according to the patient's stage and preferences.
6. If there is no information on the topic in the documentation, clearly indicate this.
7. Use exclusively correct and precise medical terms.

{summary_text}{recent_context}{medical_context}
PATIENT CONTEXT:
Patient aged {patient_profile['age']}, 
in '{patient_profile['stage']}' phase,
information preferences: '{', '.join(patient_profile['preferences'])}',
history of previous consultations: {', '.join(previous_topics[-5:]) if previous_topics else 'None previous'}.

CURRENT PATIENT QUESTION:
{question}
"""
    
    return prompt

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
        
        st.write(f"Extracted {len(documents)} segments from {file.name}")
        return documents
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        
        # Try alternative method if the first one fails
        try:
            st.info(f"Trying alternative method for {file.name}...")
            if "PyPDFLoader" in st.session_state.pdf_loader_type:
                loader = UnstructuredPDFLoader(file_path)
            else:
                loader = PyPDFLoader(file_path)
            
            documents = loader.load()
            
            # Add metadata to documents
            for doc in documents:
                doc.metadata["collection"] = st.session_state.current_collection
                doc.metadata["file_name"] = file.name
                doc.metadata["date_added"] = datetime.now().strftime("%Y-%m-%d")
                
            st.write(f"Extracted {len(documents)} segments with alternative method")
            return documents
        except Exception as e2:
            st.error(f"Error with alternative method: {str(e2)}")
            return []

# Sidebar configuration with patient profile and document collections
with st.sidebar:
    st.header("Configuration")
    
    # Conversation control
    st.subheader("Conversation Control")
    if st.button("New Conversation"):
        # Reset memory and messages
        reset_conversation_memory()
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5
        )
        st.session_state.previous_topics = []
        st.success("Conversation reset")
        st.experimental_rerun()
    
    # Export conversation
    if st.session_state.messages:
        chat_export = "\n\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            "Download conversation", 
            chat_export, 
            file_name=f"breast-cancer-conversation-{datetime.now().strftime('%Y%m%d-%H%M')}.txt"
        )
    
    # Patient profile section
    st.subheader("Patient Profile")
    with st.expander("Configure profile"):
        st.session_state.patient_profile["age"] = st.number_input(
            "Age", 
            min_value=18, 
            max_value=100, 
            value=st.session_state.patient_profile["age"]
        )
        
        st.session_state.patient_profile["stage"] = st.selectbox(
            "Stage",
            ["Pre-diagnosis", "Recently diagnosed", "In treatment", "Post-treatment", "Survivor"],
            index=["Pre-diagnosis", "Recently diagnosed", "In treatment", "Post-treatment", "Survivor"].index(st.session_state.patient_profile["stage"])
        )
        
        st.session_state.patient_profile["preferences"] = st.multiselect(
            "Information preferences",
            ["Basic Information", "Technical details", "Treatment options", "Clinical studies"],
            default=st.session_state.patient_profile["preferences"]
        )
    
    # Model selection for responses
    st.subheader("Response Model")
    llm_models = [model for model in available_models if model in ["llama3:8b", "phi2-breast-cancer"]]
    if llm_models:
        selected_llm = st.selectbox(
            "Model for answering questions",
            llm_models,
            index=llm_models.index(st.session_state.llm_model) if st.session_state.llm_model in llm_models else 0
        )
        st.session_state.llm_model = selected_llm
        
        # Model descriptions
        if selected_llm == "llama3:8b":
            st.info("**Llama 3 8B**: Meta's base model with good general performance in English.")
        elif selected_llm == "phi2-breast-cancer":
            st.success("**Phi-2 Breast Cancer**: Specialized model with fine-tuning for breast cancer. May offer more specific responses adapted to the oncological medical context.")
    else:
        st.error("No compatible models found. Make sure you have llama3:8b or phi2-breast-cancer installed in Ollama.")
    
    embedding_model = st.selectbox(
        "Embedding model",
        ["all-minilm", "nomic-embed-text", "llama3:8b"],
        index=["all-minilm", "nomic-embed-text", "llama3:8b"].index(st.session_state.embedding_model) 
            if st.session_state.embedding_model in ["all-minilm", "nomic-embed-text", "llama3:8b"] else 0
    )
    st.session_state.embedding_model = embedding_model

    st.info("""
    **Model recommendations:**
    - **all-minilm**: Fast and efficient, specialized in embeddings (recommended)
    - **nomic-embed-text**: High quality for texts, if available
    - **llama3:8b**: Larger general model, may be slower
    """)
    # Collections management
    st.subheader("Document Collections")
    
    collection_options = list(st.session_state.collections.keys()) + ["New collection..."]
    selected_collection = st.selectbox(
        "Select collection",
        collection_options,
        index=collection_options.index(st.session_state.current_collection) if st.session_state.current_collection in collection_options else 0
    )
    
    # Handle new collection creation
    if selected_collection == "New collection...":
        new_collection_name = st.text_input("New collection name")
        if st.button("Create collection") and new_collection_name:
            if new_collection_name not in st.session_state.collections:
                st.session_state.collections[new_collection_name] = {
                    "files": [], 
                    "vectorstore": None,
                    "last_updated": None
                }
                st.session_state.current_collection = new_collection_name
                st.success(f"Collection '{new_collection_name}' created")
                st.experimental_rerun()
    else:
        st.session_state.current_collection = selected_collection
    
    # Show collection info
    if st.session_state.current_collection in st.session_state.collections:
        collection = st.session_state.collections[st.session_state.current_collection]
        st.write(f"Documents in this collection: {len(collection['files'])}")
        if collection["last_updated"]:
            st.write(f"Last update: {collection['last_updated']}")
    
    # Show knowledge base status
    st.subheader("Knowledge Base")
    kb_files = get_knowledge_base_pdfs()
    st.write(f"PDFs in knowledge base: {len(kb_files)}")
    if kb_files:
        if st.button("View knowledge base content"):
            for file in kb_files:
                st.write(f"- {file}")
                
    if st.button("Reload knowledge base"):
        with st.spinner("Reloading knowledge base..."):
            kb_vs = process_knowledge_base()
            if kb_vs:
                st.session_state.collections["Knowledge Base"] = {
                    "files": get_knowledge_base_pdfs(), 
                    "vectorstore": kb_vs,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.success("✅ Knowledge base reloaded")
            else:
                st.warning("Could not load knowledge base")
    
    # Load saved vectorstore if available
    if st.button("Load saved collection"):
        with st.spinner("Loading saved collection..."):
            loaded_vs = load_vectorstore(st.session_state.current_collection)
            if loaded_vs:
                st.session_state.collections[st.session_state.current_collection]["vectorstore"] = loaded_vs
                st.session_state.vectorstore = loaded_vs
                st.success(f"✅ Collection '{st.session_state.current_collection}' loaded successfully")
            else:
                st.warning(f"No saved collection found for '{st.session_state.current_collection}'")

    # Voice control configuration
    if voice_available:
        voice_config = add_voice_controls_to_sidebar()
        use_voice = voice_config.get("enabled", False)
    else:
        use_voice = False
    
    # PDF loading method and other configs
    st.subheader("Processing Configuration")
    
    pdf_loader_type = st.radio(
    "PDF loading method",
    ["PyPDFLoader (fast)", "UnstructuredPDFLoader (robust)"],
    index=0 if st.session_state.pdf_loader_type == "PyPDFLoader (fast)" else 1
    )
    st.session_state.pdf_loader_type = pdf_loader_type
    
    chunk_size = st.slider(
        "Chunk size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )
    
    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=500,
        value=100,
        step=50
    )
    
    k_retrievals = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=8,
        value=4,
        step=1
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1
    )

    st.markdown("---")
    st.markdown("### About this application")
    st.info("This app was developed to create a RAG specialized in breast cancer, allowing the storage of medical documentation and guidelines and answering questions based on scientific evidence.")

# Main content - Sections: 1. Document Upload, 2. Chat Interface
tab1, tab2 = st.tabs(["📄 Upload Documents", "💬 Conversation"])

# Tab 1: Document Upload
with tab1:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF files with breast cancer information",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Show currently loaded collection
    st.caption(f"Current collection: **{st.session_state.current_collection}**")

    # Option to add to knowledge base
    add_to_kb = st.checkbox("Add to permanent knowledge base", value=False)

    process_button = st.button("Process Documents")

    # Document processing
    if process_button and uploaded_files:
        with st.spinner("Processing documents..."):
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Process each PDF
                all_docs = []
                processed_files = []
                
                for file in uploaded_files:
                    with st.status(f"Processing {file.name}..."):
                        # Add to knowledge base if requested
                        if add_to_kb:
                            if add_to_knowledge_base(file):
                                st.success(f"✅ {file.name} added to knowledge base")
                        
                        docs = process_pdf(file, temp_dir)
                        if docs:
                            all_docs.extend(docs)
                            processed_files.append(file.name)
                
                if all_docs:
                    # Split text into chunks
                    with st.status("Splitting text into chunks..."):
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks = text_splitter.split_documents(all_docs)
                        st.write(f"Created {len(chunks)} text chunks")
                    
                    # Create embeddings with Ollama
                    with st.status("Creating embeddings with Ollama..."):
                        try:
                            embeddings = OllamaEmbeddings(model=st.session_state.embedding_model)
                            
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
                                st.success("✅ Vector database saved to disk")
                            
                            st.success(f"✅ Processed {len(processed_files)} PDFs successfully")
                            
                            # If added to KB, update the knowledge base collection
                            if add_to_kb and "Knowledge Base" in st.session_state.collections:
                                refresh_kb = process_knowledge_base()
                                if refresh_kb:
                                    st.success("✅ Knowledge base updated")
                                    
                        except Exception as e:
                            st.error(f"Error creating embeddings: {str(e)}")
                            st.info("Verify that Ollama is running correctly")

    # Show processed files for current collection
    current_collection_data = st.session_state.collections.get(st.session_state.current_collection, {})
    if current_collection_data.get("files"):
        st.subheader("Processed Documents")
        st.caption(f"Collection: {st.session_state.current_collection}")
        
        for name in current_collection_data.get("files", []):
            st.markdown(f"- `{name}`")

# Tab 2: Chat Interface
with tab2:
    st.header("Breast Cancer Conversation")
    
    # Show information about the model in use
    model_info_container = st.container()
    with model_info_container:
        model_col1, model_col2 = st.columns([1, 2])
        with model_col1:
            st.subheader("Model in use:")
        with model_col2:
            if st.session_state.llm_model == "llama3:8b":
                st.info("🦙 **Llama 3 8B** - Meta's base model")
            elif st.session_state.llm_model == "phi2-breast-cancer":
                st.success("🎗️ **Phi2-Breast-Cancer** - Specialized model with fine-tuning for breast cancer")
            else:
                st.warning(f"⚠️ {st.session_state.llm_model} - Alternative model")
    
    # Customize FAQs based on patient stage
    stage = st.session_state.patient_profile["stage"]
    
    # Define FAQs based on patient stage
    if stage == "Pre-diagnosis":
        faq_questions = [
            "What are the risk factors for breast cancer?",
            "How do I perform a breast self-exam?",
            "What symptoms should I watch for?",
            "When should I get a mammogram?"
        ]
    elif stage == "Recently diagnosed":
        faq_questions = [
            "What treatment options are available?",
            "What do the different cancer stages mean?",
            "How can I prepare for surgery?",
            "What questions should I ask my oncologist?"
        ]
    elif stage == "In treatment":
        faq_questions = [
            "How can I manage chemotherapy side effects?",
            "What foods are recommended during treatment?",
            "How can I care for my skin during radiation therapy?",
            "What exercises are safe during treatment?"
        ]
    elif stage == "Post-treatment":
        faq_questions = [
            "When and how are follow-up check-ups done?",
            "What can I do to reduce the risk of recurrence?",
            "How do I manage lymphedema?",
            "When can I resume normal activities?"
        ]
    else:  # Survivor
        faq_questions = [
            "What long-term check-ups do I need?",
            "How do I manage fear of recurrence?",
            "What exercises are recommended for survivors?",
            "How can I support others with breast cancer?"
        ]
    
    # Display FAQs in a expandable section
    with st.expander("Frequently Asked Questions", expanded=False):
        cols = st.columns(2)
        for i, question in enumerate(faq_questions):
            with cols[i % 2]:
                if st.button(question, key=f"faq_{i}"):
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                        
                    # Add question to history
                    st.session_state.messages.append({"role": "user", "content": question})
                    
                    # Process this question at the end, before chat_input
                    if question not in st.session_state.previous_topics:
                        st.session_state.previous_topics.append(question)
                    
                    # Reload the page to show the question and generate the response
                    st.experimental_rerun()
                        
    # Container for chat history
    chat_container = st.container()
    
    # Show previous messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    # For assistant responses, keep format with highlighted medical terms
                    st.markdown(highlight_medical_terms(message["content"]), unsafe_allow_html=True)
                else:
                    # For user messages, show normal text
                    st.markdown(message["content"])
    # Voice interface
    if voice_available:
        voice_interface_result = add_voice_interface_to_chat(
            messages=st.session_state.messages,
            on_voice_input=lambda text: st.session_state.messages.append({"role": "user", "content": text})
        )
        
        #
        if len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant":
            if st.button("🔊 READ THIS RESPONSE"):
                text_to_read = st.session_state.messages[-1]["content"]
                from voice_processor import gtts_generate_speech
                audio_path = gtts_generate_speech(text_to_read)
                if audio_path:
                    st.audio(audio_path)
        # If there's already a transcription, show it
        if 'transcription' in st.session_state and st.session_state.transcription:
            voice_text = st.session_state.transcription
            
            # Clean for next time
            temp_transcription = voice_text
            st.session_state.transcription = None
            st.session_state.recording_state = "idle"
            
            # Add to history and process
            st.session_state.messages.append({"role": "user", "content": temp_transcription})
            if temp_transcription not in st.session_state.previous_topics:
                st.session_state.previous_topics.append(temp_transcription)
            st.experimental_rerun()
        else:
            # Show recording interface
            try:
                # Show recording interface
                voice_text = audio_recorder_and_transcriber()
            except Exception as e:
                st.error(f"Error in audio recording: {str(e)}")
                st.error("Technical details for debugging:")
                st.code(str(e))
                # Avoid automatic refresh
                import time
                time.sleep(10)
                voice_text = None
    
    # User input using chat_input
    user_question = st.chat_input("Type your question about breast cancer")
    
    # Process user question (from chat_input or FAQ button)
    if user_question:
        # Add question to history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Update previous topics for context
        if user_question not in st.session_state.previous_topics:
            st.session_state.previous_topics.append(user_question)

        # Update conversation summary if needed
        update_conversation_summary()

        # Reload to show the question
        st.experimental_rerun()
    
    # Process last question in history if it doesn't have a response yet
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # Get the last unanswered question
        last_question = st.session_state.messages[-1]["content"]
        
        # Show the user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(last_question)

        # Update conversation summary if needed
        update_conversation_summary()
        
        # Process the question if we have a vectorstore
        current_collection_data = st.session_state.collections.get(st.session_state.current_collection, {})
        if current_collection_data.get("vectorstore"):
            with st.spinner("Generating response..."):
                try:
                    # Create LLM with the selected model
                    llm = ChatOllama(model=st.session_state.llm_model, temperature=temperature)
                    
                    # Prepare the contextualized prompt
                    contextualized_question = prepare_contextualized_prompt(
                        last_question,
                        st.session_state.patient_profile,
                        st.session_state.previous_topics,
                        st.session_state.messages
                    )
                    
                    # Get relevant documents with enhanced retrieval
                    relevant_docs = enhanced_retrieval(
                        current_collection_data["vectorstore"],
                        last_question,
                        st.session_state.messages,
                        st.session_state.previous_topics,
                        k=k_retrievals
                    )
                    
                    # Create the custom retriever
                    custom_retriever = PrecomputedRetriever(documents=relevant_docs)
                    
                    # Create the chain with the custom retriever
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=custom_retriever,
                        memory=st.session_state.memory,
                        return_source_documents=True,
                        output_key="answer"
                    )
                    
                    # Generate response
                    import time
                    start_time = time.time()
                    response = qa_chain({"question": contextualized_question})
                    end_time = time.time()
                    
                    # Verify medical accuracy
                    confidence, disclaimer = verify_medical_accuracy(response["answer"])
                    
                    # Add response to history
                    st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
                    
                    # Show response in chat with highlighted medical terms
                    with chat_container:
                        with st.chat_message("assistant"):
                            # Add color according to confidence level
                            confidence_color = {
                                "High": "green",
                                "Medium": "orange",
                                "Low": "red"
                            }
                            
                            st.markdown(f"""
                            <div style="border-left: 5px solid {confidence_color[confidence]}; padding-left: 10px;">
                            {highlight_medical_terms(response["answer"])}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show disclaimer based on verification
                            st.markdown(disclaimer)
                            
                            # Show generation time
                            st.caption(f"Response generated in {(end_time - start_time):.2f} seconds using {st.session_state.llm_model} model")
                            
                            # Show sources in an expander
                            with st.expander("View sources used"):
                                for i, doc in enumerate(response["source_documents"]):
                                    st.markdown(f"**Source {i+1}**: {doc.metadata.get('file_name', 'Unknown')}")
                                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                    st.markdown("---")
                    
                    # Show additional resources below the response
                    with chat_container:
                        # Create columns for additional resources
                        st.subheader("Additional Resources")
                        resources_cols = st.columns(3)
                        
                        with resources_cols[0]:
                            st.info("📋 **Clinical Guidelines**\nOfficial resources for patients and families.")
                            if st.button("View guidelines", key="guides_btn"):
                                st.markdown("""
                                - [National Cancer Institute](https://www.cancer.gov/types/breast)
                                - [American Cancer Society](https://www.cancer.org/cancer/breast-cancer.html)
                                - [Breast Cancer Now](https://breastcancernow.org/information-support)
                                """)
                        
                        with resources_cols[1]:
                            st.success("🏥 **Specialized Centers**\nReference centers for breast cancer.")
                            if st.button("Find centers", key="centers_btn"):
                                st.markdown("""
                                To find specialized centers in your area, consult:
                                
                                - The American Society of Clinical Oncology directory
                                - National Cancer Institute Designated Cancer Centers
                                - Breast centers with NAPBC accreditation
                                """)
                        
                        with resources_cols[2]:
                            st.warning("👩‍⚕️ **Psychological Support**\nEmotional support resources during the process.")
                            if st.button("Support options", key="support_btn"):
                                st.markdown("""
                                - Local patient support groups
                                - Hospital psycho-oncology services
                                - Patient associations like Susan G. Komen Foundation
                                - Cancer Support Community Helpline: 1-888-793-9355
                                """)
                    
                except Exception as e:
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.error(f"Error generating response: {str(e)}")
                            st.code(str(e), language="python")
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    st.warning("No processed documents to answer your question. Please upload and process documents in the 'Upload Documents' tab first.")

# Debug information
with st.expander("Debug information", expanded=False):
    st.write(f"Python version: {sys.version}")
    st.write(f"Current directory: {os.getcwd()}")
    st.write(f"Session state: {list(st.session_state.keys())}")
    st.write(f"Number of messages in history: {len(st.session_state.messages)}")
    st.write(f"Previous topics: {st.session_state.previous_topics}")
    st.write(f"Current LLM model: {st.session_state.llm_model}")
    st.write(f"Embedding model: {st.session_state.embedding_model}")
    
    # View messages in detail
    if st.checkbox("View complete message history"):
        for i, msg in enumerate(st.session_state.messages):
            st.write(f"Message {i+1} - Role: {msg['role']}")
            st.text(msg['content'][:200] + "..." if len(msg['content']) > 200 else msg['content'])