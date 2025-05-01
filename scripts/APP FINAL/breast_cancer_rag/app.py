import streamlit as st
import matplotlib.pyplot as plt
import tempfile
import os
import sys
import pickle
from datetime import datetime
import re
import pandas as pd
from langchain.memory import ConversationBufferWindowMemory
import numpy as np

# Module imports - keeping original structure
import calendar_integration
import medication_reminders
import med_detection
import medical_glossary

# Import specialized modules
from breast_segmentation_module import BreastSegmentationModel
import io
from PIL import Image

# Import the vision explainer module
from vision_explainer import VisionExplainer

# Import the BIRADS classifier module
from birads_wrbs import BIRADSClassifierWRBS

# Try to import voice processing modules if available
try:
    from voice_processor import add_voice_controls_to_sidebar, add_voice_interface_to_chat, audio_recorder_and_transcriber
    voice_available = True
except ImportError:
    voice_available = False

# Page settings
st.set_page_config(page_title="BreastCare AI", page_icon="🎗️", layout="wide")
st.title("🎗️ BreastCareAI - Breast Cancer Information and Counseling")
st.caption("Your trusted AI companion for breast health guidance and support")

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

/* Profile card styling only - removed sidebar styling that caused issues */
.profile-card {
    background-color: #f0f7ff;
    border-radius: 10px;
    padding: 10px;
    border-left: 4px solid #4e8cff;
    margin-bottom: 15px;
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

# Verify Google Calendar dependencies
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    st.success("✅ Dependencies of Google Calendar loaded successfully")
except Exception as e:
    st.warning(f"Some Google Calendar dependencies are not available: {str(e)}")
    st.info("To enable integration with Google Calendar, install: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

# Verify if Ollama is installed and running
try:
    import subprocess
    result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    available_models = []
    
    # Check for available models
    if "llama3:8b" in result.stdout:
        available_models.append("llama3:8b")
    else:
        st.warning("⚠️ llama3:8b not found. Make sure you have it installed")
        st.code("ollama pull llama3:8b", language="bash")
    
    # Check if the phi2-breast-cancer model is available
    if "phi2-breast-cancer" in result.stdout:
        available_models.append("phi2-breast-cancer")
    else:
        st.warning("⚠️ phi2-breast-cancer not found. If you want to use this model, make sure it's installed")

    if "breast-cancer-llama3" in result.stdout:
        available_models.append("breast-cancer-llama3")
    else:
        st.warning("⚠️ breast-cancer-llama3 not found. If you want to use this model, execute :\n`ollama create breast-cancer-llama3 -f ./Modelfile`")
        
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

# Initialize breast segmentation model
segmentation_model = BreastSegmentationModel(model_path="breast_segmentation_model.pth")

# Initialize the vision explainer - uses local Ollama model
vision_explainer = VisionExplainer(model_name="llama3.2-vision")

# Add response model to the session
if 'llm_model' not in st.session_state:
    # Set default model based on availability
    if "breast-cancer-llama3" in available_models:  
        st.session_state.llm_model = "breast-cancer-llama3"
    elif "llama3:8b" in available_models:
        st.session_state.llm_model = "llama3:8b"
    elif "phi2-breast-cancer" in available_models:
        st.session_state.llm_model = "phi2-breast-cancer"
    else:
        st.session_state.llm_model = available_models[0] if available_models else "llama3:8b"

# --------------------------
# HELPER FUNCTIONS
# --------------------------

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

# --------------------------
# SIDEBAR - REORGANIZED FOR BETTER UX
# --------------------------
with st.sidebar:
    st.header("Configuration")
    
    # 1. Prominently display Reset App button at the top
    st.subheader("🔁 Reset Application")
    if st.button("New Conversation", use_container_width=True):
        reset_conversation_memory()
    
    # 2. Patient profile with improved visualization
    st.subheader("👤 Patient Profile")
    
    # Create a call-to-action for the user to set up their profile
    st.info("⚠️ Please configure your patient profile to get personalized information.")
    
    with st.expander("Configure your profile"):
        st.session_state.patient_profile["age"] = st.number_input(
            "Age", 
            min_value=18, 
            max_value=100, 
            value=st.session_state.patient_profile["age"]
        )
        
        st.session_state.patient_profile["stage"] = st.selectbox(
            "Cancer Stage",
            ["Pre-diagnosis", "Recently diagnosed", "In treatment", "Post-treatment", "Survivor"],
            index=["Pre-diagnosis", "Recently diagnosed", "In treatment", "Post-treatment", "Survivor"].index(st.session_state.patient_profile["stage"])
        )
        
        st.session_state.patient_profile["preferences"] = st.multiselect(
            "Information preferences",
            ["Basic Information", "Technical details", "Treatment options", "Clinical studies"],
            default=st.session_state.patient_profile["preferences"]
        )
        
        if st.button("Save Profile"):
            st.success("Profile saved successfully!")
            st.session_state.profile_configured = True
    
    # 3. Model configuration
    st.subheader("🗣️ Model Configuration")
    
    # Model selection for responses
    llm_models = [model for model in available_models if model in ["breast-cancer-llama3","llama3:8b", "phi2-breast-cancer"]]
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
            st.success("**Phi-2 Breast Cancer**: Specialized model with fine-tuning for breast cancer.")
        elif selected_llm == "breast-cancer-llama3":
            st.success("**Breast Cancer Llama 3**: Custom model with system prompt for oncological context.")
    else:
        st.error("No compatible models found. Make sure you have a model installed in Ollama.")
    
    # Embedding model selection
    embedding_model = st.selectbox(
        "Embedding model",
        ["all-minilm", "nomic-embed-text", "llama3:8b"],
        index=["all-minilm", "nomic-embed-text", "llama3:8b"].index(st.session_state.embedding_model) 
            if st.session_state.embedding_model in ["all-minilm", "nomic-embed-text", "llama3:8b"] else 0
    )
    st.session_state.embedding_model = embedding_model
    
    # Temperature setting
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher values create more diverse responses, lower values are more focused and deterministic"
    )
    
    # 4. Collections management (in an expander to save space)
    with st.expander("📁 Document Collections"):
        st.subheader("Document Collections")
        
        # Simplified collection management - start with just Knowledge Base
        selected_collection = st.selectbox(
            "Select collection",
            ["Knowledge Base", "New collection..."],
            index=0
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
            st.write(f"Documents in collection: {len(collection['files'])}")
            if collection["last_updated"]:
                st.write(f"Last update: {collection['last_updated']}")
    
    # 5. Advanced Settings in an expander
    with st.expander("⚙️ Advanced Settings"):
        # PDF processing options
        st.subheader("Processing Options")
        
        pdf_loader_type = st.radio(
            "PDF loading method",
            ["PyPDFLoader (fast)", "UnstructuredPDFLoader (robust)"],
            index=0 if st.session_state.pdf_loader_type == "PyPDFLoader (fast)" else 1,
            help="PyPDFLoader is faster but simpler, UnstructuredPDFLoader handles complex PDFs better"
        )
        st.session_state.pdf_loader_type = pdf_loader_type
        
        chunk_size = st.slider(
            "Chunk size",
            min_value=500,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing. Larger chunks provide more context but less precision"
        )
        
        chunk_overlap = st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=500,
            value=100,
            step=50,
            help="Overlap between chunks to maintain context across splits"
        )
        
        k_retrievals = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=8,
            value=4,
            step=1,
            help="How many document chunks to retrieve for each question"
        )
        
    # 6. Knowledge Base section
    with st.expander("🧠 Knowledge Base"):
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

    # Export conversation option
    if st.session_state.messages:
        chat_export = "\n\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            "Download Conversation History", 
            chat_export, 
            file_name=f"breast-cancer-conversation-{datetime.now().strftime('%Y%m%d-%H%M')}.txt",
            use_container_width=True
        )
    
    # Voice control configuration
    use_voice = False
    
    # 8. Help and Documentation
    st.subheader("🧭 About")
    st.info("""
    This app provides a RAG system specialized in breast cancer information.
    It allows storing medical documentation and guidelines, and answering 
    questions based on scientific evidence.
    """)
    
    # Add GitHub link
    st.markdown("[Visit the project on GitHub](https://github.com/higlesiasvd/breast-cancer-analysis.git)")
    
    # Add simple link to the PDF user guide
    st.subheader("📖 User Guide")
    st.markdown("[Download User Guide PDF](https://github.com/higlesiasvd/breast-cancer-analysis/blob/main/docs/user_guide.pdf)")
    st.caption("The user guide contains instructions, FAQs, and tips for using the application effectively.")

# --------------------------
# MAIN TABS - REORGANIZED
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Conversation & Documents", 
    "🩺 Breast Ultrasound Analysis", 
    "🗓️ Calendar", 
    "💊 Medication", 
    "📚 Medical Glossary"
])

# --------------------------
# TAB 1: CONVERSATION & DOCUMENTS (COMBINED)
# --------------------------
with tab1:
    
    # Document Upload Section in an expander
    with st.expander("📄 Upload Documents", expanded=False):
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files with breast cancer information",
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
    
    # Conversation Section - Main part of the tab
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
            elif st.session_state.llm_model == "breast-cancer-llama3":
                st.success("🎗️ **Breast-Cancer-Llama3** - Personalized model that avoids confusing case studies with personal information")
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
        
        # Read response aloud option
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
                    
                    # Store for potential actions
                    st.session_state.last_answer = response["answer"]
                    st.session_state.last_question = last_question
                    
                except Exception as e:
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.error(f"Error generating response: {str(e)}")
                            st.code(str(e), language="python")
        else:
            with chat_container:
                with st.chat_message("assistant"):
                    st.warning("No processed documents to answer your question. Please upload and process documents first (use the 'Upload Documents' section above).")

# --------------------------
# TAB 2: BREAST ULTRASOUND SEGMENTATION - MOVED TO SECOND POSITION
# --------------------------
with tab2:
    st.header("🩺 Breast Ultrasound Segmentation & BI-RADS Analysis")
    st.write("Upload an ultrasound image to segment breast tissue, visualize the result, and classify with BI-RADS using a weighted rule-based approach.")
    
    # Informational box about the technology
    with st.expander("ℹ️ About Breast Ultrasound Segmentation & BI-RADS Classification", expanded=False):
        st.markdown("""
        ### Breast Ultrasound Segmentation & BI-RADS Analysis
        
        This tool combines AI-powered segmentation and BI-RADS classification using a weighted rule-based approach based on imaging features:
        
        **Segmentation Features:**
        - AI-powered segmentation of breast tissue using a U-Net neural network
        - Adjustable threshold for fine-tuning detection sensitivity 
        - Dice coefficient calculation to evaluate segmentation quality
        
        **BI-RADS Classification Features:**
        - Automatic feature extraction from segmented images
        - BI-RADS classification using weighted scoring algorithms
        - Generated clinical report with findings and recommendations
        
        **What is BI-RADS?**
        The Breast Imaging-Reporting and Data System (BI-RADS) is a standardized classification system ranging from:
        - BI-RADS 1: Negative (no significant abnormality)
        - BI-RADS 2: Benign finding
        - BI-RADS 3: Probably benign
        - BI-RADS 4: Suspicious (A: low, B: moderate, C: high suspicion)
        - BI-RADS 5: Highly suggestive of malignancy
        
        **Classification Approach**
        The system uses a weighted scoring system where:
        - Variables: Features like shape, margin, orientation
        - Weights: Importance of each feature based on medical literature
        - Scores: Values that indicate benign (-) or suspicious (+) characteristics
        
        This approach aligns with clinical guidelines and reduces false positives.
        
        **Note:** This tool is for educational purposes only and should not replace professional medical evaluation.
        """)

    # File uploader
    uploaded_image = st.file_uploader(
        "Upload your breast ultrasound image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:
        # Load the image
        image = Image.open(uploaded_image).convert('L')  # Convert to grayscale

        # Attempt to save image temporarily to enable ground truth mask detection
        temp_image_path = None
        try:
            # Create temporary file path
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, uploaded_image.name)
            
            # Save uploaded image to temp file
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
                
            st.caption(f"Processing image: {uploaded_image.name}")
            
        except Exception as e:
            st.warning(f"Could not save temporary file: {str(e)}")
            temp_image_path = None

        # Allow user to adjust threshold
        threshold = st.slider(
            "Segmentation Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=0.5, 
            step=0.05,
            help="Adjust the sensitivity of the segmentation. Lower values include more tissue, higher values are more selective."
        )

        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Ultrasound Image")
            st.image(image, caption="Uploaded Ultrasound", use_column_width=True)

        # Segmentation Prediction
        with st.spinner('Running segmentation model...'):
            mask, prob_map = segmentation_model.predict(image, threshold=threshold)
            overlay_image = segmentation_model.overlay_mask(image, mask)
            
            # Calculate metrics - now with image path and name for auto ground truth detection
            metrics = segmentation_model.evaluate_prediction(
                mask, 
                image_path=temp_image_path,
                image_name=uploaded_image.name
            )
            area_percentage = metrics['area_ratio'] * 100
            
            # Store true mask if available for later use with BI-RADS classification
            true_mask = None
            if 'mask_source' in metrics:
                st.success(f"Found ground truth mask: {os.path.basename(metrics['mask_source'])}")
                # Load the true mask
                try:
                    import numpy as np
                    true_mask = Image.open(metrics['mask_source']).convert('L')
                    true_mask = np.array(true_mask) > 127  # Convert to binary
                except Exception as e:
                    st.warning(f"Could not load ground truth mask: {str(e)}")

        with col2:
            st.subheader("Segmentation Overlay")
            st.image(overlay_image, caption="Overlay with Segmentation Mask", use_column_width=True)
        
        # Metrics section
        st.subheader("Segmentation Metrics")
        metrics_cols = st.columns(3)
        
        with metrics_cols[0]:
            st.metric(
                label="Area Coverage", 
                value=f"{area_percentage:.1f}%",
                help="Percentage of the image identified as breast tissue"
            )
            
            # Indicate normal case (no lesion)
            if area_percentage < 1.0:
                st.success("Normal case detected (no significant lesion)")
        
        with metrics_cols[1]:
            if 'dice' in metrics:
                # We have a ground truth comparison
                dice_value = metrics['dice']
                
                if 'normal_case' in metrics and metrics['normal_case']:
                    # Handle normal case specially
                    st.metric(
                        label="Dice Coefficient", 
                        value=f"{dice_value:.3f}",
                        help="Perfect score for normal case (no lesion in image and ground truth)"
                    )
                    st.info("Normal case: Both prediction and ground truth show no lesion")
                else:
                    # Regular case with lesion
                    st.metric(
                        label="Dice Coefficient", 
                        value=f"{dice_value:.3f}",
                        help="Measure of overlap between predicted and true segmentation. Higher is better."
                    )
                    
                    # Add a visual indicator of segmentation quality
                    if dice_value > 0.8:
                        quality = "Excellent"
                        color = "green"
                    elif dice_value > 0.7:
                        quality = "Good"
                        color = "lightgreen"
                    elif dice_value > 0.5:
                        quality = "Fair"
                        color = "orange"
                    else:
                        quality = "Poor"
                        color = "red"
                        
                    st.markdown(f"<div style='color:{color};font-weight:bold;'>Quality: {quality}</div>", unsafe_allow_html=True)
            else:
                st.info("No ground truth mask found for Dice calculation")
                st.caption("System searched in temporary directory and BUSI dataset")
        
        with metrics_cols[2]:
            if 'sensitivity' in metrics and 'specificity' in metrics:
                if 'normal_case' in metrics and metrics['normal_case']:
                    # Special display for normal case
                    st.metric(
                        label="Sensitivity & Specificity", 
                        value="1.000",
                        help="Perfect scores for normal case detection"
                    )
                    st.success("Model correctly identified normal tissue")
                else:
                    # Regular metrics display
                    st.metric(
                        label="Sensitivity", 
                        value=f"{metrics['sensitivity']:.3f}",
                        help="True positive rate (recall)"
                    )
                    st.metric(
                        label="Specificity", 
                        value=f"{metrics['specificity']:.3f}",
                        help="True negative rate"
                    )
            elif 'dice' in metrics:
                st.metric(
                    label="IoU (Jaccard)", 
                    value=f"{metrics['iou']:.3f}",
                    help="Intersection over Union between predicted and true segmentation"
                )
        
        # Show probability map
        st.subheader("Probability Map")
        fig, ax = plt.subplots()
        im = ax.imshow(prob_map, cmap='viridis')
        ax.axis('off')
        fig.colorbar(im, ax=ax, label='Probability')
        st.pyplot(fig)
        
        # Explainability section - runs automatically
        st.subheader("Model Explainability")
        
        with st.spinner("Generating explainability visualization..."):
            # Generate Grad-CAM visualization
            cam_overlay, attention_map = segmentation_model.generate_gradcam(image, threshold=threshold)
            
            # Generate comprehensive explanation
            explanation_img = segmentation_model.explain_segmentation_result(image, mask, prob_map)
            
            # Analyze feature importance
            importance_stats = segmentation_model.analyze_feature_importance(image, mask)
            
            # Display explanations
            st.subheader("Model Attention (Grad-CAM)")
            st.image(cam_overlay, caption="Regions the model focused on", use_column_width=True)
            
            st.subheader("Comprehensive Explanation")
            st.image(explanation_img, caption="Multi-panel explanation", use_column_width=True)
            
            # Display feature importance statistics
            st.subheader("Feature Importance Analysis")

            # Create columns for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Max Importance", 
                    value=f"{importance_stats['max_importance']:.3f}",
                    help="Highest attention value in the image. Higher values indicate stronger focus on certain areas."
                )
                boundary_focus = importance_stats['boundary_focus'] * 100
                st.metric(
                    label="Boundary Focus", 
                    value=f"{boundary_focus:.1f}%", 
                    delta="Edge-focused" if boundary_focus > 60 else "Region-focused",
                    help="Indicates whether the model focuses more on region boundaries (edges) or interiors. Values closer to 100% indicate stronger focus on boundaries."
                )

            with col2:
                st.metric(
                    label="Mean Importance", 
                    value=f"{importance_stats['mean_importance']:.3f}",
                    help="Average attention value across the entire image. Indicates overall attention intensity."
                )
                
                if 'normalized_entropy' in importance_stats:
                    st.metric(
                        label="Attention Entropy", 
                        value=f"{importance_stats['normalized_entropy']:.3f}",
                        help="Measures how uniformly distributed the model's attention is (0-1 scale). Values closer to 1 indicate more dispersed attention, values closer to 0 show more focused attention on specific regions."
                    )
                else:
                    st.metric(
                        label="Importance Entropy", 
                        value=f"{importance_stats['importance_entropy']:.3f}",
                        help="Measures how uniformly attention is distributed. Higher values indicate more dispersed attention, lower values show more focused attention."
                    )

            with col3:
                st.metric(
                    label="Contiguity", 
                    value=f"{importance_stats['importance_contiguity']:.3f}", 
                    delta="Focused" if importance_stats['importance_contiguity'] > 0.7 else "Dispersed",
                    help="Measures how connected or clustered the areas of attention are. Values closer to 1 indicate a single concentrated area of focus, while lower values indicate multiple separated regions of attention."
                )
                st.metric(
                    label="Top 10% Importance", 
                    value=f"{importance_stats['importance_top10_percent']:.3f}",
                    help="Average attention value in the most attended 10% of the image. Shows intensity of the model's peak attention areas."
                )
            
            # Technical explanation
            with st.expander("About Grad-CAM Explainability", expanded=False):
                st.markdown("""
                ### Gradient-weighted Class Activation Mapping (Grad-CAM)
                
                Grad-CAM is a technique that uses the gradients flowing into the final convolutional layer 
                to produce a coarse localization map highlighting important regions that influenced the model's decisions.
                
                **How it works:**
                1. Forward pass through the network to get predictions
                2. Compute gradients of the output with respect to feature maps of a convolutional layer
                3. Global average pooling of gradients to get weights for each feature map
                4. Weighted combination of feature maps followed by ReLU
                
                **Interpretation:**
                - **Red/yellow areas**: Regions the model focused on most
                - **Blue areas**: Regions with minimal influence on the prediction
                
                **Metrics explained:**
                - **Boundary Focus**: Whether the model focuses more on edges (100%) or regions (0%)
                - **Contiguity**: How concentrated the model's attention is (higher = more focused)
                - **Importance Entropy**: Measures how uniformly attention is distributed
                
                This explainability helps understand what image features the model considers important
                for segmentation decisions.
                """)
        
        # BI-RADS wrbs Section
        st.header("🏥 BI-RADS Classification")
        
        with st.expander("About BI-RADS Classification", expanded=True):
            st.markdown("""
            ### BI-RADS Classification with Weighted Scoring
            
            This analysis classifies breast lesions into BI-RADS categories using a weighted scoring approach based on imaging features.
            
            **How it works:**
            1. Features are automatically extracted from the segmented image
            2. Each feature is assigned a score indicating whether it favors benignity or malignancy
            3. Features are weighted by their clinical importance (margin 25%, shape 20%, etc.)
            4. The total weighted score determines the BI-RADS category
            
            **Available algorithms:**
            - **Standard Scoring**: Direct weighted scoring based on extracted features
            - **Fuzzy Logic Scoring**: Uses linguistic variables and fuzzy rules to handle uncertainty
            
            This approach aligns with clinical guidelines, reducing false positives while maintaining sensitivity for suspicious findings.
            """)
            
        # Algorithm selection
        st.subheader("Classification Algorithm")
        wrbs_algorithm = st.radio(
            "Select algorithm approach:",
            ["Weighted Scoring", "Fuzzy Logic Scoring"],
            horizontal=True,
            help="Standard uses weighted feature scoring. Fuzzy Logic uses linguistic variables and more complex rule combinations."
        )
        
        # Map radio button selection to algorithm parameter
        algorithm_map = {
            "Weighted Scoring": "weighted",
            "Fuzzy Logic Scoring": "fuzzy"
        }
        
        # Run BI-RADS classification
        with st.spinner("Analyzing features and determining BI-RADS category..."):
            # Initialize the classifier
            birads_classifier = BIRADSClassifierWRBS()
            
            # Extract features from segmentation
            birads_classifier.extract_features_from_segmentation(np.array(image), mask)
            
            # Get selected algorithm
            selected_algorithm = algorithm_map[wrbs_algorithm]
            
            # Use the new classification method with selected algorithm
            birads_category, confidence, report, detailed_results = birads_classifier.classify(algorithm=selected_algorithm)
            
        # Display BI-RADS results
        st.subheader("BI-RADS Classification Results")
        
        # Style for BI-RADS category
        birads_color = {
            'BIRADS0': "gray",
            'BIRADS1': "green",
            'BIRADS2': "green",
            'BIRADS3': "orange",
            'BIRADS4A': "orange",
            'BIRADS4B': "orange",
            'BIRADS4C': "red",
            'BIRADS5': "red"
        }.get(birads_category, "blue")
        
        birads_descriptions = {
            'BIRADS0': "BI-RADS 0: Incomplete - Need additional imaging",
            'BIRADS1': "BI-RADS 1: Negative",
            'BIRADS2': "BI-RADS 2: Benign finding",
            'BIRADS3': "BI-RADS 3: Probably benign",
            'BIRADS4A': "BI-RADS 4A: Low suspicion for malignancy",
            'BIRADS4B': "BI-RADS 4B: Moderate suspicion for malignancy",
            'BIRADS4C': "BI-RADS 4C: High suspicion for malignancy",
            'BIRADS5': "BI-RADS 5: Highly suggestive of malignancy"
        }
        
        st.markdown(f"<h3 style='color:{birads_color};'>{birads_descriptions.get(birads_category, birads_category)}</h3>", unsafe_allow_html=True)
        st.metric("Confidence Level", f"{confidence:.2f}")
        
        # Display extracted features
        st.subheader("Extracted Features")
        
        # Define feature names mapping
        feature_names = {
            'shape': 'Shape',
            'margin': 'Margin',
            'orientation': 'Orientation',
            'echogenicity': 'Echogenicity',
            'posterior': 'Posterior Features',
            'size_mm': 'Size (mm)',
            'boundaries': 'Boundaries',
            'texture': 'Texture'
        }
        
        # Create columns for feature display
        feature_col1, feature_col2 = st.columns(2)
        
        # Process features for display
        features_to_display = []
        for feature, value in birads_classifier.variables.items():
            if value is not None:
                feature_name = feature_names.get(feature, feature)
                
                # Special handling for size if it's a dictionary
                if feature == 'size_mm' and isinstance(value, dict):
                    features_to_display.append({
                        'name': feature_name,
                        'display_name': f"{feature_name} (approx.)" if value.get('approximate', False) else feature_name,
                        'value': f"{value.get('value', 0):.2f}",
                        'weight': birads_classifier.feature_weights.get(feature, 0),
                        'column': len(features_to_display) % 2  # 0 for col1, 1 for col2
                    })
                else:
                    features_to_display.append({
                        'name': feature_name,
                        'display_name': feature_name,
                        'value': value,
                        'weight': birads_classifier.feature_weights.get(feature, 0),
                        'column': len(features_to_display) % 2  # 0 for col1, 1 for col2
                    })
        
        # Sort by weight (most important first)
        features_to_display.sort(key=lambda x: x['weight'], reverse=True)
        
        # Display in alternating columns
        for feature in features_to_display:
            col = feature_col1 if feature['column'] == 0 else feature_col2
            col.metric(feature['display_name'], feature['value'])
        
        # Display total score and decisive features
        st.subheader("Classification Analysis")
        score_col1, score_col2 = st.columns(2)
        
        with score_col1:
            # Display total score
            total_score = birads_classifier.metadata.get('total_score', 0)
            st.metric("Total Score", f"{total_score:.3f}")
            
            # Score interpretation
            if total_score < -0.2:
                st.success("Score indicates benign finding")
            elif total_score < 0.1:
                st.info("Score indicates probably benign finding")
            elif total_score < 0.3:
                st.warning("Score indicates low to moderate suspicion")
            else:
                st.error("Score indicates high suspicion")
        
        with score_col2:
            # Show decisive features
            decisive_features = birads_classifier.metadata.get('decisive_features', [])
            if decisive_features:
                st.write("**Decisive Features:**")
                for feature in decisive_features:
                    readable_feature = feature_names.get(feature, feature)
                    value = birads_classifier.variables.get(feature)
                    # Use emoji to indicate if increases or decreases suspicion
                    feature_value = value
                    if feature == 'size_mm' and isinstance(value, dict):
                        feature_value = f"{value.get('value', 0):.1f} mm"
                    
                    # Get score for this feature if available
                    score = 0
                    if hasattr(birads_classifier, 'feature_scores') and feature in birads_classifier.feature_scores and feature_value in birads_classifier.feature_scores[feature]:
                        score = birads_classifier.feature_scores[feature][feature_value]
                    elif 'feature_scores' in birads_classifier.metadata and feature in birads_classifier.metadata['feature_scores']:
                        if 'score' in birads_classifier.metadata['feature_scores'][feature]:
                            score = birads_classifier.metadata['feature_scores'][feature]['score']
                    
                    emoji = "✅" if score < 0 else "⚠️" if score > 0 else "➖"
                    st.write(f"{emoji} {readable_feature}: {feature_value}")
        
        # Display limitations if any exist
        limitations = birads_classifier.metadata.get('limitations', [])
        if limitations:
            with st.expander("Limitations in Assessment", expanded=True):
                for limitation in limitations:
                    st.info(f"• {limitation}")
        
        # Display constraint satisfaction results
        st.subheader("BI-RADS Category Scores")
        
        # Convert results to format suitable for chart
        chart_data = {k: v for k, v in detailed_results.items()}
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        categories = list(chart_data.keys())
        values = list(chart_data.values())
        
        # Generate colors
        colors = ['green' if cat == birads_category else 'lightgray' for cat in categories]
        
        # Create bar chart
        bars = ax.bar(categories, values, color=colors)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
        
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')
        ax.set_title('BI-RADS Category Scores')
        plt.tight_layout()
        
        # Display chart
        st.pyplot(fig)
        
        # AI Explanation Section
        st.subheader("📋 AI Analysis & Explanation")
        
        with st.spinner("Generating AI analysis of the segmentation results..."):
            try:
                # Generate explanation using the vision model - now with importance_stats and BI-RADS included
                explanation = vision_explainer.explain_segmentation(
                    original_image=image,
                    segmented_image=overlay_image,
                    metrics=metrics,
                    true_mask=true_mask,  # Pass the ground truth mask if available
                    attention_map=attention_map,
                    birads_category=birads_category,
                    birads_confidence=confidence
                )
                
                # Show the combined image created for the vision model
                temp_combined_path = os.path.join(tempfile.gettempdir(), "combined_segmentation.png")
                if os.path.exists(temp_combined_path):
                    combined_img = Image.open(temp_combined_path)
                    st.image(combined_img, caption="Combined view analyzed by AI", use_column_width=True)
                
                # Display the AI explanation
                st.markdown(explanation)
                
                # Add disclaimer
                st.info("⚠️ This AI-generated analysis is for educational purposes only and should not replace professional medical evaluation.")
            except Exception as e:
                st.error(f"Error generating AI explanation: {str(e)}")
                st.info("If the error persists, make sure Ollama is running with the 'llama3:vision' model available.")
        
        # Display full report
        st.subheader("BI-RADS Clinical Report")
        st.text_area("Detailed Report", report, height=400)
        
        # Export options
        st.subheader("Export Results")
        export_col1, export_col2, export_col3 = st.columns(3)
        
        with export_col1:
            # Convert overlay to bytes for download
            overlay_bytes = io.BytesIO()
            overlay_image.save(overlay_bytes, format='PNG')
            
            st.download_button(
                label="Download Segmentation Overlay",
                data=overlay_bytes.getvalue(),
                file_name="breast_ultrasound_segmentation.png",
                mime="image/png"
            )
            
        with export_col2:
            # Option to download the BI-RADS report
            st.download_button(
                label="Download BI-RADS Report",
                data=report,
                file_name=f"birads_report_{birads_category.lower()}.txt",
                mime="text/plain"
            )
            
        with export_col3:
            # Add option to download a combined report
            if st.button("Generate Combined Report"):
                # Combine all analyses into one comprehensive report
                combined_report = f"""
                # Breast Ultrasound Analysis Report

                **Date:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
                
                ## Segmentation Results
                - **Threshold used:** {threshold}
                - **Area coverage:** {area_percentage:.1f}%
                """
                
                # Add Dice score if available
                if 'dice' in metrics:
                    combined_report += f"""
                    - **Dice coefficient:** {metrics['dice']:.3f}
                    - **IoU (Jaccard):** {metrics['iou']:.3f}
                    - **Sensitivity:** {metrics['sensitivity']:.3f}
                    - **Specificity:** {metrics['specificity']:.3f}
                    """
                
                # Add feature importance metrics
                combined_report += f"""
                ## Model Explainability Metrics
                - **Boundary Focus:** {boundary_focus:.1f}% ({'Edge-focused' if boundary_focus > 60 else 'Region-focused'})
                - **Attention Contiguity:** {importance_stats['importance_contiguity']:.3f} ({'Focused' if importance_stats['importance_contiguity'] > 0.7 else 'Dispersed'})
                - **Attention Entropy:** {importance_stats.get('normalized_entropy', importance_stats.get('importance_entropy', 0)):.3f}
                """
                
                # Add BI-RADS section
                combined_report += f"""
                ## BI-RADS Classification
                - **Category:** {birads_descriptions.get(birads_category, birads_category)}
                - **Confidence:** {confidence:.2f}
                - **Algorithm used:** {wrbs_algorithm}
                - **Total score:** {total_score:.3f}
                - **Execution time:** {birads_classifier.metadata['execution_time']:.3f} seconds
                
                ### Extracted Features
                """
                
                # Add extracted features
                for feature, value in birads_classifier.variables.items():
                    if value is not None:
                        feature_name = feature_names.get(feature, feature.capitalize())
                        if feature == 'size_mm' and isinstance(value, dict):
                            combined_report += f"- **{feature_name}:** {value.get('value', 0):.2f} (approximate)\n"
                        else:
                            combined_report += f"- **{feature_name}:** {value}\n"
                
                # Add BI-RADS report
                combined_report += f"""
                
                ## Detailed BI-RADS Report
                
                {report}
                
                ## AI Explanation
                
                {explanation if 'explanation' in locals() else "AI explanation not available."}
                
                ## Disclaimer
                
                This analysis is based on algorithmic interpretation and should be reviewed by a healthcare professional.
                It is not a substitute for clinical judgment or professional medical advice.
                """
                
                # Convert report to bytes for download
                report_bytes = io.BytesIO(combined_report.encode())
                
                st.download_button(
                    label="Download Combined Report (Markdown)",
                    data=report_bytes.getvalue(),
                    file_name="breast_ultrasound_complete_analysis.md",
                    mime="text/markdown"
                )
        
        # Clean up temporary files
        try:
            # Only try to clean up if temp_image_path exists and is defined
            if 'temp_image_path' in locals() and temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except Exception as e:
            st.warning(f"Error cleaning up temporary files: {str(e)}")
    else:
        st.info("Please upload an ultrasound image to begin analysis.")
        
        # Show example images
        st.subheader("Example Results")
        st.markdown("""
        This tool provides:
        
        1. **AI Segmentation**: Identifies regions of interest in breast ultrasound images
        2. **BI-RADS Classification**: Uses weighted scoring to classify findings
        3. **Model Explainability**: Shows how the AI model arrived at its decisions
        4. **Clinical Report**: Generates a comprehensive analysis with recommendations
        
        Upload your own image to try the complete analysis pipeline.
        """)
                        
        # Clean up temporary files
        try:
            # Only try to clean up if temp_image_path exists and is defined
            if 'temp_image_path' in locals() and temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except Exception as e:
            st.warning(f"Error cleaning up temporary files: {str(e)}")

# --------------------------
# TAB 3: CALENDAR
# --------------------------
with tab3:
    # Show saved questions from the conversation
    if 'saved_questions' in st.session_state and st.session_state.saved_questions:
        st.subheader("Questions saved from conversation")
        for i, question in enumerate(st.session_state.saved_questions):
            st.markdown(f"{i+1}. {question}")
        
        if st.button("Clear saved questions"):
            st.session_state.saved_questions = []
            st.experimental_rerun()
    
    # Calendar integration
    calendar_integration.calendar_management_ui()

# --------------------------
# TAB 4: MEDICATION
# --------------------------
with tab4:
    # Call to the main function of the medication reminders module
    medication_reminders.medication_reminders_ui()

# --------------------------
# TAB 5: MEDICAL GLOSSARY
# --------------------------
with tab5:
    # Call to the medical glossary module
    medical_glossary.medical_glossary_ui()

# --------------------------
# MEDICAL TERMS ANALYSIS 
# --------------------------
st.markdown("---")
st.header("🔍 Medical Terms Analysis")

# Verificar si hay mensajes del asistente para analizar
has_assistant = False
last_assistant_message = None

if 'messages' in st.session_state and st.session_state.messages:
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant":
            has_assistant = True
            last_assistant_message = msg["content"]
            break

if has_assistant:
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.caption("Analyze the latest assistant response to identify and explain medical terminology")
    
    with col2:
        if st.button("Analyze Now", key="inline_medical_btn", use_container_width=True):
            with st.spinner("Analyzing medical terms..."):
                try:
                    detector = med_detection.get_medical_terms_detector()
                    detected_terms = detector.detect_medical_terms(last_assistant_message)
                    
                    if detected_terms:
                        # Inicializar saved_medical_terms si no existe
                        if 'saved_medical_terms' not in st.session_state:
                            st.session_state.saved_medical_terms = []
                        
                        # Añadir solo términos nuevos
                        terms_added = 0
                        current_terms = [t['term'] for t in st.session_state.saved_medical_terms]
                        
                        for term in detected_terms:
                            if term['term'] not in current_terms:
                                st.session_state.saved_medical_terms.append(term)
                                terms_added += 1
                        
                        # Guardar resultados en session_state
                        st.session_state.inline_med_results = {
                            "terms": detected_terms,
                            "terms_added": terms_added,
                            "html": detector.format_results_for_display(detected_terms)
                        }
                        st.experimental_rerun()
                    else:
                        st.info("No medical terms detected in the latest response.")
                except Exception as e:
                    st.error(f"Error analyzing terms: {str(e)}")
else:
    st.info("No assistant responses available to analyze. Start a conversation first.")

# Mostrar resultados si existen
if 'inline_med_results' in st.session_state:
    results = st.session_state.inline_med_results
    
    if results["terms"]:
        st.success(f"Found {len(results['terms'])} medical terms. {results['terms_added']} new terms saved to glossary.")
        
        with st.expander("Medical terms found:", expanded=True):
            st.markdown(results["html"], unsafe_allow_html=True)
            
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("Clear Results", key="clear_inline_results", use_container_width=True):
                    del st.session_state.inline_med_results
                    st.experimental_rerun()
            
            with col1:
                st.info("All terms have been automatically saved to the Medical Glossary tab.")
    else:
        st.info("No medical terms detected in the last response.")
        
        if st.button("Dismiss", key="dismiss_no_terms_inline"):
            del st.session_state.inline_med_results
            st.experimental_rerun()

st.markdown("---")


# --------------------------
# DEBUG INFORMATION - ONLY VISIBLE WHEN EXPANDED
# --------------------------
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