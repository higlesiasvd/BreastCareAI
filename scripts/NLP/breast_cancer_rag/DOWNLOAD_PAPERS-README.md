# Breast Cancer Guidelines Web Scraper

## Introduction

The Breast Cancer Guidelines Web Scraper is a sophisticated Python tool designed to collect, process, and organize high-quality breast cancer guidelines and patient information resources from authoritative sources across the web. Built specifically to support Retrieval-Augmented Generation (RAG) systems in healthcare applications, the scraper intelligently identifies, filters, and categorizes medical information resources while preserving rich metadata to enhance retrieval and generation capabilities.

This tool bridges the gap between scattered online medical guidelines and structured knowledge bases necessary for AI-powered healthcare applications, with special attention to different audience types, reading levels, and information categories.

## Table of Contents

1. [Core Features](#core-features)
2. [Technical Architecture](#technical-architecture)
3. [Source Selection Strategy](#source-selection-strategy)
4. [Resource Classification](#resource-classification)
5. [Metadata Enrichment](#metadata-enrichment)
6. [Smart Filtering](#smart-filtering)
7. [Data Persistence](#data-persistence)
8. [Integration with RAG Systems](#integration-with-rag-systems)
9. [Usage Guide](#usage-guide)
10. [Technical Reference](#technical-reference)

## Core Features

The Breast Cancer Guidelines Web Scraper provides several key capabilities:

1. **Multi-Source Scraping**: Collects guidelines from medical authorities, PubMed, Google Scholar, and specialized organizations
2. **Resource Classification**: Automatically categorizes content by type (clinical guidelines, patient information, etc.)
3. **Audience Detection**: Identifies intended audience (healthcare providers vs. patients)
4. **Quality Assessment**: Assigns priority scores based on source reputation and resource characteristics
5. **Metadata Enrichment**: Extracts and augments metadata to enhance RAG system performance
6. **Duplicate Detection**: Eliminates redundant resources while preserving the highest quality sources
7. **PDF Validation**: Verifies PDF integrity and content before downloading
8. **Structured Organization**: Creates a well-organized repository with consistent naming conventions
9. **RAG-Optimized Output**: Generates specialized metadata for seamless integration with RAG systems

## Technical Architecture

The scraper follows a multi-stage pipeline architecture:

1. **Collection Stage**: Gathers potential resources from diverse sources
2. **Filtering Stage**: Applies relevance and quality filters
3. **Enrichment Stage**: Adds metadata and classifies content
4. **Validation Stage**: Verifies PDF integrity and accessibility
5. **Persistence Stage**: Saves files and metadata in structured formats

This architecture prioritizes reliability, information quality, and comprehensive metadata collection to support downstream RAG applications.

## Source Selection Strategy

The scraper implements a tiered approach to source selection:

### Tier 1: Direct Access to Authoritative Guidelines
```python
def search_specific_guideline_urls():
    """Directly check known URLs that host breast cancer guidelines"""
    
    specific_urls = [
        # NCCN Guidelines
        {
            "url": "https://www.nccn.org/patients/guidelines/content/PDF/breast-invasive-patient.pdf",
            "title": "NCCN Guidelines for Patients: Breast Cancer - Invasive",
            "source": "NCCN",
            "resource_type": "Patient Guidelines"
        },
        # Additional specific URLs...
    ]
    
    # Validation and processing logic...
```

This function targets highly reliable, pre-identified resources with known URLs, providing the highest confidence resources.

### Tier 2: Guideline-Producing Organizations
```python
def search_guideline_organizations():
    """Search for breast cancer guidelines from guideline-producing organizations"""
    organizations = [
        {
            "name": "NCCN (National Comprehensive Cancer Network)",
            "base_url": "https://www.nccn.org",
            "search_path": "/patients/guidelines/cancers.aspx",
            "resource_type": "Patient Guidelines"
        },
        # Additional organizations...
    ]
    
    # Processing logic to extract guidelines from each organization...
```

This function systematically explores the websites of recognized medical authorities to find guidelines and patient resources.

### Tier 3: Scientific Databases
```python
def search_pubmed_guidelines():
    """Search for breast cancer guidelines in PubMed"""
    # Implementation for searching PubMed with specific queries...

def search_google_scholar_guidelines():
    """Search for breast cancer guidelines in Google Scholar"""
    # Implementation for searching Google Scholar with specific queries...
```

These functions query scientific databases with carefully crafted search terms to identify peer-reviewed guidelines and resources.

## Resource Classification

The scraper automatically categorizes resources into specific types:

```python
# Determine resource type based on title and abstract
resource_type = "Unknown"
if any(term in title.lower() or term in abstract.lower() for term in ["guideline", "guidance", "consensus"]):
    resource_type = "Clinical Guideline"
elif any(term in title.lower() or term in abstract.lower() for term in ["patient guide", "handbook", "information", "booklet", "leaflet"]):
    resource_type = "Patient Information"
elif any(term in title.lower() or term in abstract.lower() for term in ["algorithm", "pathway", "protocol", "decision aid"]):
    resource_type = "Clinical Tool"
```

This classification system distinguishes between:
1. **Clinical Guidelines**: Technical documents for healthcare providers
2. **Patient Guidelines**: Simplified guidelines adapted for patient understanding
3. **Patient Information**: Educational resources for patients and families
4. **Clinical Tools**: Decision aids, algorithms, and reference tools for clinicians

## Metadata Enrichment

The scraper enhances resource metadata to optimize RAG system performance:

```python
def add_metadata_to_guidelines(guidelines):
    """Add additional useful metadata to guidelines"""
    for guideline in guidelines:
        # Add audience classification if not present
        if "resource_type" in guideline:
            resource_type = guideline["resource_type"].lower()
            
            # Determine the target audience
            if "patient" in resource_type or "information" in resource_type:
                guideline["audience"] = "Patients"
            elif "clinical" in resource_type or "practice" in resource_type:
                guideline["audience"] = "Healthcare Providers"
            else:
                guideline["audience"] = "Mixed/Unknown"
        
        # Add priority ranking based on source reputation and resource type
        priority = 0
        
        # Source-based priority
        high_priority_sources = ["NCCN", "ASCO", "ESMO", "NICE", "WHO", "American Cancer Society"]
        medium_priority_sources = ["Breastcancer.org", "Susan G. Komen", "Cancer.Net", "CDC"]
        
        source = guideline.get("source", "")
        if any(org in source for org in high_priority_sources):
            priority += 3
        elif any(org in source for org in medium_priority_sources):
            priority += 2
        else:
            priority += 1
        
        # Type-based priority
        resource_type = guideline.get("resource_type", "")
        if "Guidelines" in resource_type:
            priority += 2
        elif "Patient Information" in resource_type:
            priority += 1
        
        guideline["priority"] = priority
```

Key enriched metadata includes:
1. **Audience**: Target readership (patients vs. healthcare professionals)
2. **Priority**: Quality and relevance score based on source reputation
3. **Reading Level**: Estimated complexity level of the content
4. **Resource Type**: Categorization of the content type
5. **Language**: Language of the document (primarily English)
6. **Download Date**: Date when the resource was obtained

## Smart Filtering

The scraper uses a sophisticated filtering mechanism to identify relevant resources:

```python
def is_relevant_guideline(url, title="", abstract=""):
    """Check if a PDF is a relevant guideline or information resource based on keywords"""
    text_to_check = (url + " " + title + " " + abstract).lower()
    
    # Check if it contains at least one keyword
    keyword_match = any(keyword.lower() in text_to_check for keyword in KEYWORDS)
    if not keyword_match:
        return False
    
    # Check if it has guideline indicators
    guideline_match = any(indicator.lower() in text_to_check for indicator in GUIDELINE_INDICATORS)
    if not guideline_match:
        return False
    
    # Additional check for breast cancer specificity
    if not ('breast' in text_to_check and ('cancer' in text_to_check or 'carcinoma' in text_to_check or 'neoplasm' in text_to_check)):
        return False
        
    return True
```

This multi-stage filtering ensures that only high-quality, relevant breast cancer guidelines and information resources are collected.

## Data Persistence

The scraper implements a comprehensive data persistence system:

```python
def download_pdfs(pdf_links, limit=100):
    """Download PDFs and save metadata"""
    # Implementation for downloading and organizing PDFs...
    
    # Update metadata file
    all_metadata = existing_metadata + metadata
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    # Create a separate JSON with just the RAG-optimized metadata
    rag_metadata = []
    for item in all_metadata:
        rag_item = {
            'filename': item.get('filename', ''),
            'title': item.get('title', ''),
            'source': item.get('source', ''),
            'resource_type': item.get('resource_type', ''),
            'audience': item.get('audience', ''),
            'reading_level': item.get('reading_level', ''),
            'summary': item.get('summary', ''),
            'language': item.get('language', 'en'),
            'download_date': item.get('download_date', '')
        }
        rag_metadata.append(rag_item)
    
    rag_metadata_file = os.path.join(OUTPUT_DIR, 'rag_metadata.json')
    with open(rag_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(rag_metadata, f, ensure_ascii=False, indent=2)
```

The system creates:
1. A structured directory of PDF files with consistent naming conventions
2. A comprehensive JSON metadata file with all extracted information
3. A simplified RAG-optimized metadata file focused on retrieval-enhancing attributes
4. A README file explaining the collection and its usage

## Integration with RAG Systems

The scraper is specifically designed to support RAG systems in healthcare applications:

1. **Consistent Naming**: Files follow a `{resource_type}_{source}_{title}.pdf` pattern for easy reference
2. **Enhanced Metadata**: Includes audience, reading level, and content type to improve retrieval relevance
3. **Reading Level Detection**: Enables matching content complexity to user needs
4. **Source Attribution**: Preserves source information for citation and attribution
5. **Simplified JSON**: Provides a RAG-optimized metadata file for direct integration

The `rag_metadata.json` file serves as a ready-to-use index for RAG systems, enabling intelligent document retrieval based on query context and user characteristics.

## Usage Guide

### Basic Usage

```python
# Run the full pipeline
python breast_cancer_guidelines_scraper.py
```

This will:
1. Search all configured sources for breast cancer guidelines
2. Filter for relevant and high-quality resources
3. Download up to 100 PDFs per execution
4. Generate comprehensive metadata files
5. Create a structured repository of guidelines

### Customizing the Search

To modify the search behavior, adjust the following constants:

```python
# Focus on patient information and clinical guidelines
KEYWORDS = [
    'breast cancer', 'breast carcinoma', 'mammary carcinoma',
    'patient guide', 'patient information', 'patient education',
    # Additional keywords...
]

# Guideline indicators to ensure we're getting proper guides
GUIDELINE_INDICATORS = [
    'guideline', 'guide', 'protocol', 'pathway', 'algorithm',
    # Additional indicators...
]
```

### Integrating with Your RAG System

```python
import json
import os

# Load the RAG-optimized metadata
with open('guidelines_to_review/rag_metadata.json', 'r') as f:
    rag_metadata = json.load(f)

# Example: Filter resources by audience type
patient_resources = [item for item in rag_metadata if item.get('audience') == 'Patients']
clinical_resources = [item for item in rag_metadata if item.get('audience') == 'Healthcare Providers']

# Example: Create a file path lookup dictionary
file_paths = {}
for item in rag_metadata:
    file_paths[item['filename']] = os.path.join('guidelines_to_review', item['filename'])

# Now you can use this in your document processing pipeline
```

## Technical Reference

### Core Functions

```python
def search_specific_guideline_urls():
    """Directly check known URLs that host breast cancer guidelines"""

def search_guideline_organizations():
    """Search for breast cancer guidelines from guideline-producing organizations"""

def search_pubmed_guidelines():
    """Search for breast cancer guidelines in PubMed"""

def search_google_scholar_guidelines():
    """Search for breast cancer guidelines in Google Scholar"""

def is_relevant_guideline(url, title="", abstract=""):
    """Check if a PDF is a relevant guideline or information resource based on keywords"""

def format_filename(s):
    """Convert a string to a safe filename"""

def check_pdf_validity(url):
    """Check if a URL points to a valid PDF"""

def download_pdfs(pdf_links, limit=100):
    """Download PDFs and save metadata"""

def add_metadata_to_guidelines(guidelines):
    """Add additional useful metadata to guidelines"""

def create_readme():
    """Create a README file explaining how to use the downloaded guidelines"""
```

### Configuration Constants

```python
# Headers for web requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0...',
    # Additional headers...
}

# Output directory for downloaded files
OUTPUT_DIR = 'guidelines_to_review'

# JSON file to store metadata
METADATA_FILE = os.path.join(OUTPUT_DIR, 'guideline_metadata.json')

# Keywords and indicators for filtering
KEYWORDS = [
    'breast cancer', 'breast carcinoma', 'mammary carcinoma',
    # Additional keywords...
]

GUIDELINE_INDICATORS = [
    'guideline', 'guide', 'protocol', 'pathway', 'algorithm',
    # Additional indicators...
]
```

### Generated Files

- **PDFs**: Individual guideline files in the `guidelines_to_review` directory
- **guideline_metadata.json**: Complete metadata for all resources
- **rag_metadata.json**: Simplified metadata optimized for RAG systems
- **README.md**: Documentation explaining the collection and its usage

### Dependencies

- `requests`: For HTTP requests and downloads
- `beautifulsoup4`: For HTML parsing
- `json`: For metadata storage
- `re`: For regular expression operations
- `datetime`: For timestamp generation
- `os`: For file system operations
- `random`: For request throttling
- `time`: For request throttling

---

This Breast Cancer Guidelines Web Scraper represents a specialized tool for collecting high-quality medical information resources about breast cancer. By creating a structured, metadata-rich repository of authoritative guidelines, it enables the development of more accurate, context-aware, and responsible AI-powered healthcare applications.
