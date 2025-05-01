# BreastCareAI: Comprehensive Breast Cancer Information and Analysis System

## Project Overview

BreastCareAI is an advanced application providing interactive, personalized information and analysis for breast cancer patients, survivors, and healthcare professionals. It combines state-of-the-art AI technologies including Retrieval-Augmented Generation (RAG), medical image analysis, and natural language processing to deliver evidence-based information, diagnostic assistance, and personalized guidance.

The application integrates multiple specialized modules to create a holistic support system that addresses various aspects of the breast cancer journey, from diagnosis through treatment and into survivorship.

## Key Features

- **AI-Powered Conversations**: Contextualized, evidence-based information retrieval using RAG technologies
- **Breast Ultrasound Analysis**: Advanced segmentation and BI-RADS classification of ultrasound images
- **Calendar Integration**: Medical appointment scheduling with question preparation
- **Medication Management**: Personalized medication reminder system
- **Medical Terminology Assistance**: Automatic detection and explanation of complex medical terms
- **Voice Interaction**: Optional voice input/output capabilities for accessibility
- **Patient Profile**: Personalized content based on diagnosis stage and preferences

## Technical Architecture

BreastCareAI is built on a modular architecture with Streamlit as the frontend framework and integrates multiple specialized components:

### Core Technologies

- **Frontend**: Streamlit-based interactive web application
- **RAG Engine**: LangChain with Ollama embeddings and FAISS vector storage
- **LLM Integration**: Ollama for local inference using Llama 3:8b with system prompt, specialized breast cancer models
- **Document Processing**: LangChain document loaders with chunking optimization
- **Image Analysis**: Segmentation U-Net with attention, PyTorch-based, with explainability features (GRADCAM, llama3.2-vision, rule based system)
- **Data Persistence**: Local file system with JSON and pickle serialization
- **Voice Processing**: Whisper models, and Google TTS for voice interactions

## Modules

BreastCareAI consists of several integrated modules, each handling specific functionality:

### 1. [Breast Ultrasound Segmentation Module](./scripts/FINAL%20APP/BreastCareAI/BREAST_SEGMENTATION-README.md)

This module implements an advanced deep learning architecture for automatic segmentation of breast lesions in ultrasound images. It features an Attention U-Net architecture specifically designed for breast ultrasound image segmentation, consisting of an encoder path that progressively downsamples the input image while increasing feature depth, a bottleneck that captures abstract representations, and a decoder path that restores spatial resolution with skip connections and attention mechanisms that focus on relevant features.

The model automatically detects available hardware acceleration (CUDA for NVIDIA GPUs or MPS for Apple Silicon) and configures models accordingly, with robust error handling for model loading failures. The segmentation pipeline includes image preprocessing, segmentation prediction, visualization with overlays, and comprehensive evaluation metrics to assess segmentation quality.

For detailed information, see the [Breast Ultrasound Segmentation README](./scripts/FINAL%20APP/BreastCareAI/BREAST_SEGMENTATION-README.md).

### 2. [BI-RADS Classification Module](./scripts/FINAL%20APP/BreastCareAI/BIRADS-README.md)

This module provides an automated classification system for breast ultrasound findings according to the Breast Imaging-Reporting and Data System (BI-RADS). It implements a Weighted Rule-Based System (WRBS) approach that automatically extracts features from segmented ultrasound images and applies a weighted scoring algorithm to classify findings into BI-RADS categories ranging from BI-RADS 1 (negative) to BI-RADS 5 (highly suggestive of malignancy), with BI-RADS 4 further subdivided into categories A, B, and C based on suspicion level.

The system's core components include feature extraction from segmented images, feature scoring based on medical literature, weighted score calculation, BI-RADS category assignment, and report generation. It extracts key diagnostic features like shape, margin, orientation, echogenicity, posterior features, size, boundary regularity, and texture.

For detailed information, see the [BI-RADS Classification README](./scripts/FINAL%20APP/BreastCareAI/BIRADS-README.md).

### 3. [Vision Explainer Module](./scripts/FINAL%20APP/BreastCareAI/VISION_EXPLAINER-README.md)

This module leverages AI vision models to provide comprehensive explanations of breast ultrasound segmentation results. It bridges the gap between advanced image segmentation algorithms and clinical interpretation by automatically generating radiological-style reports from segmentation outputs, metrics, and classification results. Using the Ollama framework with multimodal vision models, the explainer transforms complex technical outputs into accessible explanations that could help clinicians and patients better understand AI-assisted ultrasound analysis.

The module serves as an interpretability layer, combining original images, segmentation masks, evaluation metrics, and optional BI-RADS classification data to generate: a composite visual explanation combining multiple visualization elements, a structured radiological-style text report explaining the findings in accessible language, and an analysis of the AI model's behavior and decision-making process.

For detailed information, see the [Vision Explainer README](./scripts/FINAL%20APP/BreastCareAI/VISION_EXPLAINER-README.md).

### 4. [Calendar Integration Module](./scripts/FINAL%20APP/BreastCareAI/CALENDAR_INTEGRATION-README.md)

This module provides a framework for managing medical appointments and consultations. It enables users to seamlessly schedule appointments, prepare consultation questions, and integrate them directly with Google Calendar, making it easier to manage healthcare-related events. Designed with healthcare workflows in mind, the module offers specialized features for medical appointment management, including automatic question generation based on patient profiles and a comprehensive interface for preparing for consultations.

Key features include OAuth authentication for secure connection with Google Calendar API, appointment scheduling to create and manage medical appointments, question preparation tools for medical consultations, AI-generated questions based on patient profiles, multi-timezone support for scheduling across different timezones, and customizable email and popup reminders for appointments.

For detailed information, see the [Calendar Integration README](./scripts/FINAL%20APP/BreastCareAI/CALENDAR_INTEGRATION-README.md).

### 5. [Medication Reminder Module](./scripts/FINAL%20APP/BreastCareAI/MEDICATION_REMINDERS-README.md)

This module provides a comprehensive solution for managing medication schedules and generating reminders. Designed specifically for breast cancer patients and survivors, this module helps users track medications, set up recurring reminders, and visualize their medication schedule within a streamlined Streamlit interface. By integrating with Google Calendar, the system creates time-based reminders with customizable notifications, ensuring patients maintain adherence to complex medication regimens.

Key features include medication tracking to record details including name, dosage, and notes; flexible scheduling support for various dosing frequencies (daily, weekly, monthly); multiple daily doses configuration for medications taken multiple times per day; Google Calendar integration for automatic creation of recurring events with reminders; visual schedule showing upcoming medication doses; persistent storage using local JSON-based storage; customizable popup and email reminders; and a complete management interface to add, edit, view, and delete medications.

For detailed information, see the [Medication Reminder README](./scripts/FINAL%20APP/BreastCareAI/MEDICATION_REMINDERS-README.md).

### 6. [Medical Glossary Module](./scripts/FINAL%20APP/BreastCareAI/MEDICAL_GLOSSARY-README.md)

This module provides a comprehensive solution for identifying, explaining, and managing medical terminology. Designed specifically for breast cancer information systems, it combines intelligent term detection with an interactive glossary management interface, helping users understand complex medical language in context. This dual-module system bridges the gap between technical medical language and patient understanding, serving as an essential component in healthcare communication applications.

Core features include medical term detection that automatically identifies terms using both dictionary-based and AI-powered approaches; contextual explanations providing clear, accessible definitions with surrounding context; glossary management with an interactive interface for saving and organizing terms; multi-format exports in Markdown, CSV, PDF, and JSON; category classification that automatically categorizes terms by type; search and filtering tools to quickly find terms; optional BioBERT integration for advanced biomedical entity recognition; and visual highlighting of terms with definitions and context.

For detailed information, see the [Medical Glossary README](./scripts/FINAL%20APP/BreastCareAI/MEDICAL_GLOSSARY-README.md).

### 7. [Voice Processing Module](./scripts/FINAL%20APP/BreastCareAI/VOICE_PROCESSING-README.md)

This module provides comprehensive speech-to-text and text-to-speech capabilities. It enables natural voice interaction with AI assistants, making information more accessible through both speech recognition and audio responses. Designed with a focus on reliability and ease of integration, the module combines state-of-the-art speech recognition through OpenAI's Whisper models with robust text-to-speech synthesis using Google's TTS service.

Key features include speech recognition to transcribe spoken audio to text using Whisper models; text-to-speech conversion to natural-sounding speech using Google TTS; hardware acceleration with automatic detection and utilization of available GPU/MPS acceleration; interactive recording with Streamlit components for audio recording and visual feedback; seamless integration with chat interfaces and other Streamlit components; and an accessibility focus making AI healthcare information accessible to users who prefer voice interaction.

For detailed information, see the [Voice Processing README](./scripts/FINAL%20APP/BreastCareAI/VOICE_PROCESSING-README.md).

### 8. [Web Scraper Module](./scripts/FINAL%20APP/BreastCareAI/DOWNLOAD_PAPERS-README.md)

This module is a sophisticated Python tool for collecting and organizing breast cancer guidelines. It's designed to collect, process, and organize high-quality breast cancer guidelines and patient information resources from authoritative sources across the web. Built specifically to support Retrieval-Augmented Generation (RAG) systems in healthcare applications, the scraper intelligently identifies, filters, and categorizes medical information resources while preserving rich metadata to enhance retrieval capabilities.

Core features include multi-source scraping from medical authorities, PubMed, Google Scholar, and specialized organizations; resource classification that automatically categorizes content by type; audience detection identifying intended audience (healthcare providers vs. patients); quality assessment assigning priority scores based on source reputation; metadata enrichment extracting and augmenting metadata to enhance RAG system performance; duplicate detection eliminating redundant resources while preserving highest quality sources; PDF validation verifying integrity and content before downloading; structured organization creating a well-organized repository with consistent naming conventions; and RAG-optimized output generating specialized metadata for seamless integration with RAG systems.

For detailed information, see the [Web Scraper README](./scripts/FINAL%20APP/BreastCareAI/DOWNLOAD_PAPERS-README.md).

## Installation and Setup

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

## Running the Application

```bash
# Start Ollama in a separate terminal
ollama serve

# Run the Streamlit application
cd scripts/APP\ FINAL/breast_cancer_rag/
streamlit run app.py
```

The application will be available at http://localhost:8501

## User Interface

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

## Research & Development

The repository includes several experimental models that were developed and evaluated during the research phase:

### Segmentation Models

Three different approaches to breast image segmentation were explored:

#### [Basic U-Net for Mammogram Segmentation](./scripts/UNINTEGRATED%20MODELS/Segmentation/DISCARDED%20MODELS/N01_Unet/N01_UNET-README.md) (Discarded)

This project implemented a U-Net architecture for breast lesion segmentation using the CBIS-DDSM (Curated Breast Imaging Subset of Digital Database for Screening Mammography) dataset. Despite implementation of several advanced techniques, the model achieved limited performance (Dice coefficient of 0.04 on validation), highlighting the significant challenges associated with mammogram segmentation.

For detailed information, see the [U-Net README](./scripts/UNINTEGRATED%20MODELS/Segmentation/DISCARDED%20MODELS/N01_Unet/N01_UNET-README.md).

#### [GA-Optimized Attention U-Net](./scripts/UNINTEGRATED%20MODELS/Segmentation/DISCARDED%20MODELS/N02.1_Metaheuristic_Attention_Unet/N02_METAHEURISTIC_ATTENTION_UNET-README.md) (Discarded)

This implementation features a modified U-Net architecture incorporating attention mechanisms specifically designed for breast cancer lesion segmentation in ultrasound images. The architecture is optimized using a genetic algorithm for hyperparameter tuning and features a two-stage training process to maximize performance. Notably, while the genetic algorithm optimization did not improve Dice coefficient metrics compared to the standard implementation (it actually performed slightly lower), it achieved a dramatic reduction in training time—approximately 4 hours less.

For detailed information, see the [GA-Optimized Attention U-Net README](./scripts/UNINTEGRATED%20MODELS/Segmentation/DISCARDED%20MODELS/N02.1_Metaheuristic_Attention_Unet/N02_METAHEURISTIC_ATTENTION_UNET-README.md).

#### [Standard Attention U-Net](./scripts/UNINTEGRATED%20MODELS/Segmentation/FINAL%20MODEL/N02_Attention_Unet/N02_ATTENTION_UNET-README.md) (Final Model)

This implementation features an enhanced U-Net architecture with attention mechanisms specifically designed for breast cancer lesion segmentation in ultrasound images. The model achieves a high Dice coefficient score of 0.728, demonstrating excellent segmentation performance on the challenging task of breast lesion delineation. Unlike the genetic algorithm (GA) optimized variant, this standard implementation uses fixed hyperparameters and a straightforward training process, resulting in better segmentation metrics and a more reliable model.

For detailed information, see the [Standard Attention U-Net README](./scripts/UNINTEGRATED%20MODELS/Segmentation/FINAL%20MODEL/N02_Attention_Unet/N02_ATTENTION_UNET-README.md).

#### [Comparative Analysis of Segmentation Models](./scripts/UNINTEGRATED%20MODELS/Segmentation/SEGMENTATION-README.md)

This repository contains three distinct implementations of deep learning models for breast cancer segmentation in medical images, each with different approaches, architectures, and performance characteristics. These models represent an exploration of various techniques for addressing the challenging task of medical image segmentation, particularly in ultrasound and mammogram images.

For detailed information, see the [Segmentation Comparative Analysis README](./scripts/UNINTEGRATED%20MODELS/Segmentation/SEGMENTATION-README.md).

### Conversation Models

#### [MammaELIZA: Pattern-Based Breast Cancer Information Chatbot](./scripts/UNINTEGRATED%20MODELS/Conversation/breast_cancer_eliza/NLP01_ELIZA-README.md) (Unintegrated)

MammaELIZA is a rule-based conversational agent designed to provide information and support about breast cancer. Following the pattern-matching approach of the classic ELIZA program, this chatbot uses regular expressions to identify user queries and provide appropriate responses from a curated knowledge base. Although MammaELIZA proved functional in testing, it was ultimately replaced with transformer-based models for the final BreastCareAI system due to their greater flexibility and knowledge depth.

For detailed information, see the [MammaELIZA README](./scripts/UNINTEGRATED%20MODELS/Conversation/breast_cancer_eliza/NLP01_ELIZA-README.md).

## Configuration Options

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

## Privacy and Ethics

BreastCareAI is designed with privacy and ethics as core principles:

- **Local Processing**: All AI inference happens locally using Ollama
- **No Data Sharing**: User data stays on the local device
- **Medical Disclaimers**: Clear disclaimers about the informational nature of the tool
- **Verification Systems**: Medical accuracy verification for generated content
- **Transparent Citations**: Source attribution for provided information

## Troubleshooting

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

## Contributing

Contributions to BreastCareAI are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

BreastCareAI is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The breast cancer patient community for providing valuable feedback
- Healthcare professionals who contributed to the validation process
- Open-source AI community for tools and models that made this possible
