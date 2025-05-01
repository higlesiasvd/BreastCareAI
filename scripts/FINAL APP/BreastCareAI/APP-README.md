# BreastCareAI: Breast Cancer Information and Counseling System

![BreastCareAI Logo](/Volumes/Proyecto_Hugo/breast-cancer-analysis/utils/BreastCareAI-logo-transparent-enhanced.png)

## Overview

BreastCareAI is a comprehensive application providing interactive, personalized information and analysis for breast cancer patients, survivors, and healthcare professionals. It combines advanced AI technologies including Retrieval-Augmented Generation (RAG), medical image analysis, and natural language processing to deliver evidence-based information, diagnostic assistance, and personalized guidance.

The application integrates multiple specialized modules to create a holistic support system that addresses various aspects of the breast cancer journey, from diagnosis through treatment and into survivorship.

### Key Features

- **AI-Powered Conversations**: Contextualized, evidence-based information retrieval using RAG technologies
- **Breast Ultrasound Analysis**: Advanced segmentation and BI-RADS classification of ultrasound images
- **Calendar Integration**: Medical appointment scheduling with question preparation
- **Medication Management**: Personalized medication reminder system
- **Medical Terminology Assistance**: Automatic detection and explanation of complex medical terms
- **Voice Interaction**: Optional voice input/output capabilities for accessibility
- **Patient Profile**: Personalized content based on diagnosis stage and preferences

## Technical Architecture

BreastCareAI is built on a modular architecture with Streamlit as the frontend framework and integrates multiple specialized components:

![Architecture Diagram](/Volumes/Proyecto_Hugo/breast-cancer-analysis/scripts/FINAL APP/BreastCareAI/arquitecture diagram.png)

### Core Technologies

- **Frontend**: Streamlit-based interactive web application
- **RAG Engine**: LangChain with Ollama embeddings and FAISS vector storage
- **LLM Integration**: Ollama for local inference using Llama 3:8b with system prompt, specialized breast cancer models
- **Document Processing**: LangChain document loaders with chunking optimization
- **Image Analysis**: Segmentation U-Net with attention, PyTorch-based, with explainability features (GRADCAM, llama3.2-vision, rule based system)
- **Data Persistence**: Local file system with JSON and pickle serialization
- **Voice Processing**: Whisper models, and Google TTS for voice interactions

## 📦 Modules

BreastCareAI consists of seven integrated modules, each handling specific functionality:

1. **[Breast Ultrasound Segmentation Module](BREAST_SEGMENTATION-README.md)**: AI-powered segmentation of breast ultrasound images
2. **[BI-RADS Classification Module](BIRADS-README.md)**: Weighted rule-based assessment of breast imaging findings
3. **[Vision Explainer Module](VISION_EXPLAINER-README.md)**: AI interpretation of segmentation and classification results
4. **[Calendar Integration Module](CALENDAR_INTEGRATION-README.md)**: Google Calendar integration for appointment management
5. **[Medication Reminder Module](MEDICATION_REMINDERS-README.md)**: Comprehensive medication scheduling system
6. **[Medical Glossary Module](MEDICAL_GLOSSARY-README.md)**: Detection and explanation of medical terminology
7. **[Voice Processing Module](VOICE_PROCESSING-README.md)**: Speech-to-text and text-to-speech capabilities
8. **[Web Scraper Module](DOWNLOAD_PAPERS-README.md)**: Collection of authoritative breast cancer guidelines

## 🚀 Installation and Setup

### Prerequisites

- Python 3.9+
- Ollama (for local LLM inference)
- 8GB+ RAM (16GB recommended for optimal performance)
- GPU acceleration recommended but not required
- Google account (for calendar integration)

### Required Models

The application requires the following models to be available through Ollama:

```bash
# Core LLM model for RAG
ollama pull llama3:8b

# Optional specialized models
ollama pull llama3.2-vision       # For image analysis explanations

# Embedding models (choose one)
ollama pull all-minilm            # Default embedding model
ollama pull nomic-embed-text      # Alternative embedding model
```

### Dependencies Installation

```bash
# Clone the repository
git clone https://github.com/higlesiasvd/breast-cancer-analysis.git
cd breast-cancer-analysis

# Install dependencies
pip install -r requirements.txt

# Optional voice processing dependencies
pip install sounddevice soundfile gtts

# Google Calendar integration dependencies (optional)
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### Setting Up Knowledge Base

1. Create a `knowledge_base` directory in the application root folder
2. Add PDF documents containing breast cancer information to this directory
3. The application will automatically process these documents at startup

### Google Calendar Setup (Optional)

For Google Calendar integration:

1. Create a Google Cloud project
2. Enable the Google Calendar API
3. Create OAuth 2.0 credentials
4. Download the credentials JSON file and place it in the application directory
5. Rename the file to `credentials.json`

## 🖥️ Running the Application

```bash
# Start Ollama in a separate terminal
ollama serve

# Run the Streamlit application
cd scripts/APP\ FINAL/breast_cancer_rag/
streamlit run app.py
```

The application will be available at http://localhost:8501

## 📱 User Interface

BreastCareAI features a tabbed interface with five main sections:

### 💬 Conversation & Documents Tab

- Interactive AI-powered chat interface
- Document upload and management
- Knowledge base administration
- Patient profile configuration

### 🩺 Breast Ultrasound Analysis Tab

- Ultrasound image segmentation
- BI-RADS classification
- Model explainability visualization
- Analysis reports generation

### 🗓️ Calendar Tab

- Appointment scheduling and management
- Question preparation for doctor visits
- Google Calendar integration
- Automatic reminder configuration

### 💊 Medication Tab

- Medication schedule management
- Customizable reminders
- Visualization of medication timeline
- Notification preferences

### 📚 Medical Glossary Tab

- Automated medical term detection
- Personal glossary of medical terms
- Search and filtering capabilities
- Multi-format export options

## 💡 Usage Examples

### RAG-Based Conversation

1. Upload breast cancer guidelines PDFs in the "Conversation & Documents" tab
2. Process the documents to create embeddings
3. Ask questions in natural language about breast cancer topics
4. The system will retrieve relevant information and generate contextualized responses
5. Medical terms in responses are automatically highlighted with explanations

### Breast Ultrasound Analysis

1. Navigate to the "Breast Ultrasound Analysis" tab
2. Upload an ultrasound image
3. Adjust the segmentation threshold if needed
4. View the segmentation results, metrics, and model explanations
5. Obtain BI-RADS classification with detailed feature analysis
6. Generate and download a comprehensive report

### Calendar Management

1. Go to the "Calendar" tab and authorize Google Calendar access
2. Create a new appointment with your healthcare provider
3. Add specific questions for your appointment
4. Configure reminders (email, popup)
5. The system will create a calendar event with your prepared questions

### Medication Management

1. Access the "Medication" tab
2. Add medications with dosage, frequency, and timing information
3. Set start/end dates and reminder preferences
4. View your medication schedule in a weekly calendar view
5. Sync with Google Calendar for integrated reminders

### Medical Glossary

1. During conversations, the system automatically detects and explains medical terms
2. View and manage your personal glossary in the "Medical Glossary" tab
3. Search for specific terms or filter by category
4. Add additional notes to terms
5. Export your personal glossary in various formats

## 🔧 Configuration Options

BreastCareAI provides numerous configuration options:

### Model Configuration

- **LLM Model Selection**: Choose between general (Llama 3) and specialized models
- **Embedding Model**: Select from available embedding models (all-minilm, llama3, nomic-embed)
- **Temperature**: Adjust response creativity vs. determinism

### Patient Profile

- **Age**: Indicate patient age for age-appropriate information
- **Cancer Stage**: Select from pre-diagnosis to survivor stages
- **Information Preferences**: Configure desired level of detail and topics

### Document Processing

- **PDF Loading Method**: Choose between fast (PyPDFLoader) or robust (UnstructuredPDFLoader) methods
- **Chunk Size**: Configure text chunk size for optimal retrieval (500-2000)
- **Chunk Overlap**: Set overlap between chunks (0-500)
- **Retrievals**: Adjust how many document chunks to retrieve per question (1-8)

### 🔒 Privacy and Ethics

BreastCareAI is designed with privacy and ethics as core principles:

- **Local Processing**: All AI inference happens locally using Ollama
- **No Data Sharing**: User data stays on the local device
- **Medical Disclaimers**: Clear disclaimers about the informational nature of the tool
- **Verification Systems**: Medical accuracy verification for generated content
- **Transparent Citations**: Source attribution for provided information

## 🛠 Troubleshooting

Common issues and solutions:

### Model Loading Issues

```
Error checking Ollama: [specific error]
```

**Solution**: Ensure Ollama is running (`ollama serve`) and the required models are installed.

### PDF Processing Errors

```
Error processing [filename]: [specific error]
```

**Solution**: Try the alternative PDF loading method in Advanced Settings or check if the PDF is corrupted.

### Calendar Integration Issues

```
Error connecting to Google Calendar: [specific error]
```

**Solution**: Verify your OAuth credentials and ensure you've completed the authorization flow.

### Memory Limitations

```
CUDA out of memory
```

**Solution**: Reduce the model size, adjust chunk size settings, or switch to CPU processing.

## 🤝 Contributing

Contributions to BreastCareAI are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

BreastCareAI is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The breast cancer patient community for providing valuable feedback
- Healthcare professionals who contributed to the validation process
- Open-source AI community for tools and models that made this possible
