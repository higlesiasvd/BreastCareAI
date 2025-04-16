# Breast Cancer Guidelines Collection

## Overview
This folder contains guidelines and information resources about breast cancer, specifically collected 
to provide reliable information for patient counseling and a RAG (Retrieval-Augmented Generation) system.

## Content Description
The guidelines are organized by type:
- **Clinical Guidelines**: Professional medical guidelines from organizations like NCCN, ASCO, and ESMO
- **Patient Information**: Resources specifically designed for patients
- **Patient Guidelines**: Simplified guidelines adapted for patient understanding
- **Clinical Tools**: Decision aids, algorithms, and other tools for clinicians

## Metadata
Two JSON files are included:
1. `guideline_metadata.json`: Complete metadata for all downloaded files
2. `rag_metadata.json`: Simplified metadata optimized for RAG systems

## Using with RAG Systems
To use these resources with a RAG system:
1. Index the PDFs using your preferred document processing pipeline
2. Use the `rag_metadata.json` to enhance your retrieval and generation
3. Consider categorizing responses based on the 'audience' and 'reading_level' fields

## Sources
Guidelines were collected from authoritative sources including:
- NCCN (National Comprehensive Cancer Network)
- ASCO (American Society of Clinical Oncology)
- ESMO (European Society for Medical Oncology)
- American Cancer Society
- National Institute for Health and Care Excellence (NICE)
- World Health Organization (WHO)
- And other reputable medical organizations

## Best Practices for RAG Implementation
When implementing a breast cancer counseling RAG system:
1. Prioritize recent guidelines (check the 'download_date' field)
2. Match the reading level to the user's needs
3. Clearly distinguish between patient information and clinical guidelines
4. Always include references to the source documents when providing information

Last updated: 2025-04-16
