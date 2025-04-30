# Medical Glossary and Terms Detection System

## Introduction

The Medical Glossary and Terms Detection System is a comprehensive solution for identifying, explaining, and managing medical terminology in healthcare applications. Designed specifically for breast cancer information systems, it combines intelligent term detection with an interactive glossary management interface, helping users understand complex medical language in context.

This dual-module system bridges the gap between technical medical language and patient understanding, serving as an essential component in healthcare communication applications. The integration of rule-based and AI-powered detection approaches ensures both precision and breadth in terminology identification.

## Table of Contents

1. [Core Features](#core-features)
2. [System Architecture](#system-architecture)
3. [Medical Terms Detector](#medical-terms-detector)
4. [Medical Glossary Management](#medical-glossary-management)
5. [Term Detection Methods](#term-detection-methods)
6. [User Interface Components](#user-interface-components)
7. [Export Capabilities](#export-capabilities)
8. [Integration with RAG Systems](#integration-with-rag-systems)
9. [Example Usage](#example-usage)
10. [Technical Reference](#technical-reference)

## Core Features

The Medical Glossary and Terms Detection System provides several key capabilities:

1. **Medical Term Detection**: Automatically identifies medical terms in text using both dictionary-based and AI-powered approaches
2. **Contextual Explanations**: Provides clear, accessible definitions with surrounding context from the source text
3. **Glossary Management**: Interactive interface for saving, categorizing, and organizing medical terms
4. **Multi-format Exports**: Export capabilities in Markdown, CSV, PDF, and JSON formats
5. **Category Classification**: Automatic categorization of terms by type (Diagnostic Procedures, Treatments, Conditions, etc.)
6. **Search and Filtering**: Tools to quickly find and filter saved terms
7. **BioBERT Integration**: Optional advanced biomedical entity recognition using transformer models
8. **Visual Highlighting**: Stylized display of terms with definitions and context

## System Architecture

The system consists of two integrated modules:

1. **Medical Terms Detector**: Core detection and explanation engine
   - Dictionary-based term detection
   - Optional BioBERT-powered named entity recognition
   - Context extraction and highlighting
   - Term categorization logic

2. **Medical Glossary UI**: User interface for managing detected terms
   - Term storage and organization
   - Category-based presentation
   - Search and filtering capabilities
   - Multi-format export functionality
   - Persistent storage through JSON serialization

This modular architecture allows for flexible deployment, with the detection engine usable independently of the UI components when needed.

## Medical Terms Detector

The `MedicalTermsDetector` class forms the core engine for identifying and explaining medical terminology:

```python
class MedicalTermsDetector:
    """
    Class for detecting and explaining medical terms in text using either:
    - a predefined medical glossary
    - or a pre-trained BioBERT model for named entity recognition (NER)
    """

    def __init__(self):
        """Initializes the detector with a glossary and optionally a transformer model."""
        # Predefined glossary initialization
        self.medical_terms_glossary = {
            "mammogram": "An X-ray image of the breast used to detect early signs of breast cancer.",
            "biopsy": "A procedure to remove a small sample of tissue for laboratory testing...",
            # Extensive glossary of breast cancer terms...
        }

        # Attempt to load BioBERT if available
        if transformers_available:
            self.ner_pipeline = self._load_biobert_model()
        else:
            self.ner_pipeline = None
```

The detector implements three primary detection methods:

1. **Glossary-based Detection**: Identifies terms by matching against a comprehensive predefined glossary
2. **BioBERT-powered NER** (optional): Uses biomedical transformer models to identify domain-specific entities
3. **Context Extraction**: Captures and highlights surrounding text to provide usage context

Each detected term includes:
- The term itself
- A clear definition or explanation
- Contextual usage from the source text
- Classification into appropriate medical categories

## Medical Glossary Management

The glossary management interface provides comprehensive tools for organizing and accessing saved medical terms:

```python
def medical_glossary_ui():
    """User interface for the medical glossary"""
    st.header("📚 Medical Glossary")
    
    # Initialize saved terms or load them from session_state
    if 'saved_medical_terms' not in st.session_state:
        st.session_state.saved_medical_terms = load_saved_terms()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["My Saved Terms", "Search & Filter", "Export Options"])
    
    # Tab implementations for viewing, searching, and exporting terms...
```

The interface organizes terms into categories for easier comprehension:
- **Diagnostic Procedures**: Terms related to tests and imaging (mammogram, biopsy, ultrasound, etc.)
- **Treatments**: Terms related to interventions (mastectomy, radiation therapy, chemotherapy, etc.)
- **Conditions**: Terms describing disease states (carcinoma, metastasis, etc.)
- **Other**: Miscellaneous medical terminology

Each term is presented in an expandable panel with options to view, edit, or remove entries from the personal glossary.

## Term Detection Methods

### Dictionary-Based Detection

The system includes a comprehensive glossary of over 40 breast cancer terms with detailed explanations:

```python
def detect_medical_terms(self, text):
    """Detects medical terms in text using the glossary and optionally BioBERT."""
    if not text:
        return []

    results = []
    text_lower = text.lower()

    # Glossary-based detection
    for term, definition in self.medical_terms_glossary.items():
        if term.lower() in text_lower:
            results.append({
                "term": term,
                "definition": definition,
                "context": self._get_context(text, term)
            })

    # Additional detection methods...
```

This approach ensures reliable identification of common terms with carefully crafted explanations tailored to patient understanding.

### BioBERT-Powered Detection

For more advanced applications, the system can leverage biomedical transformer models:

```python
@st.cache_resource
def _load_biobert_model(_self):
    """Loads the BioBERT model for biomedical named entity recognition (NER)."""
    try:
        model_name = "dmis-lab/biobert-base-cased-v1.1"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        return pipeline("ner", model=model, tokenizer=tokenizer)
    except Exception as e:
        return None
```

When enabled, this allows for:
- Recognition of rare or specialized terms not in the predefined glossary
- Identification of variant forms and related terminology
- Detection of emerging terminology in newer research

The system gracefully falls back to dictionary-based detection when transformer models are unavailable.

## User Interface Components

The system provides a rich set of UI components for interacting with medical terminology:

### Term Display and Organization

```python
# Display terms by category
for category, terms in terms_by_category.items():
    with st.expander(f"{category} ({len(terms)} terms)", expanded=True):
        for i, term in enumerate(terms):
            col1, col2, col3 = st.columns([3, 8, 1])
            with col1:
                st.markdown(f"**{term['term'].capitalize()}**")
            with col2:
                # Use 'definition' if available, fall back to 'explanation'
                definition = term.get('definition', term.get('explanation', 'No explanation available'))
                st.markdown(definition, unsafe_allow_html=True)
            with col3:
                if st.button("❌", key=f"delete_{category}_{i}"):
                    st.session_state.saved_medical_terms.remove(term)
                    save_terms(st.session_state.saved_medical_terms)
                    st.experimental_rerun()
            st.markdown("---")
```

Terms are organized in expandable sections by category, with each term displayed with its definition and management options.

### Search and Filtering

Advanced search capabilities allow users to quickly find terms:

```python
# Search and filter options
search_term = st.text_input("Search for a term:")

# Get all available categories
all_categories = list(set(term.get('category', 'Other') for term in st.session_state.saved_medical_terms))
selected_categories = st.multiselect("Filter by category:", all_categories, default=all_categories)

# Filter terms based on criteria
filtered_terms = []
for term in st.session_state.saved_medical_terms:
    category = term.get('category', 'Other')
    term_text = term['term'].lower()
    
    # Apply filters
    category_match = category in selected_categories
    search_match = not search_term or search_term.lower() in term_text or search_term.lower() in term.get('explanation', '').lower()
    
    if category_match and search_match:
        filtered_terms.append(term)
```

This enables users to filter by both keyword and category, making larger glossaries manageable.

### Styled Term Display

The system uses styled HTML to present terms in a visually appealing format:

```python
def format_results_for_display(self, terms):
    """Formats the detected terms for display in Streamlit UI as styled cards."""
    if not terms:
        return "No relevant medical terms detected."

    html_output = []
    for term in terms:
        term_name = term['term'].capitalize()
        definition = term.get('definition', "No detailed explanation available.")
        context = term.get('context', "")
        html_output.append(f"""
        <div style='background-color: #f8f9fa; border-left: 4px solid #4CAF50; 
                    margin-bottom: 10px; padding: 10px; border-radius: 4px;'>
            <span style='color: #2E7D32; font-weight: bold;'>{term_name}</span>
            <p style='margin-top: 5px; font-size: 0.9em;'>{definition}</p>
            <p style='margin-top: 5px; font-size: 0.85em; font-style: italic;'>{context}</p>
        </div>
        """)

    return "".join(html_output)
```

This creates visually distinct cards for each term with color coding, typography, and spacing that enhance readability.

## Export Capabilities

The system supports multiple export formats to accommodate different user needs:

### Markdown Export

```python
if export_format == "Markdown":
    md_content = "# My Medical Glossary\n\n"
    md_content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    
    # Group by category
    terms_by_category = {}
    for term in st.session_state.saved_medical_terms:
        category = term.get('category', 'Other')
        if category not in terms_by_category:
            terms_by_category[category] = []
        terms_by_category[category].append(term)
    
    for category, terms in terms_by_category.items():
        md_content += f"## {category}\n\n"
        for term in terms:
            md_content += f"### {term['term'].capitalize()}\n"
            md_content += f"{term.get('definition', term.get('explanation', 'No explanation available'))}\n\n"
    
    st.download_button(
        "Download Markdown",
        md_content,
        f"medical_glossary_{datetime.now().strftime('%Y%m%d')}.md",
        "text/markdown"
    )
```

### CSV Export

```python
elif export_format == "CSV":
    # Convert to DataFrame
    data = []
    for term in st.session_state.saved_medical_terms:
        data.append({
            "Term": term['term'].capitalize(),
            "Category": term.get('category', 'Other'),
            "Explanation": term.get('explanation', 'No explanation available')
        })
    
    df = pd.DataFrame(data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        "Download CSV",
        csv,
        f"medical_glossary_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv"
    )
```

### PDF Export

For more polished documentation, the system offers PDF export with styled formatting using ReportLab:

```python
# Create PDF in memory
buffer = io.BytesIO()
doc = SimpleDocTemplate(buffer, pagesize=letter)
styles = getSampleStyleSheet()

# Create custom styles
styles.add(ParagraphStyle(
    name='CategoryHeading',
    parent=styles['Heading2'],
    textColor=colors.darkblue,
    spaceAfter=12
))

# Build document with styled content
content = []
content.append(Paragraph("My Medical Glossary", styles['Title']))
# ...additional content generation

# Build PDF
doc.build(content)
```

### JSON Export

For data interoperability, the system supports JSON export:

```python
elif export_format == "JSON":
    # Export data as JSON
    json_data = json.dumps(st.session_state.saved_medical_terms, indent=2)
    
    st.download_button(
        "Download JSON",
        json_data,
        f"medical_glossary_{datetime.now().strftime('%Y%m%d')}.json",
        "application/json"
    )
```

## Integration with RAG Systems

The Medical Glossary system is designed to integrate with Retrieval-Augmented Generation (RAG) systems in healthcare applications:

1. **Term Explanation in Context**: When RAG systems produce content containing medical terminology, the detector can identify and explain those terms in place
2. **Persistence of Learned Terms**: Terms that users have looked up can be saved to their personal glossary for future reference
3. **Term Highlighting**: Medical terms can be visually highlighted within generated content to improve readability
4. **Context-Aware Definitions**: The system captures and displays the context in which terms are used

Example integration with a RAG-based chat system:

```python
import streamlit as st
from medical_terms_detector import get_medical_terms_detector

# Initialize detector
detector = get_medical_terms_detector()

# In a RAG chat system
if "messages" in st.session_state and st.session_state.messages:
    last_assistant_message = [m for m in st.session_state.messages if m["role"] == "assistant"][-1]
    
    if st.button("Explain medical terms in this response"):
        # Detect medical terms in the response
        terms = detector.detect_medical_terms(last_assistant_message["content"])
        
        # Display explanations
        st.markdown(detector.format_results_for_display(terms), unsafe_allow_html=True)
        
        # Option to save terms to glossary
        if terms and st.button("Save these terms to your glossary"):
            if "saved_medical_terms" not in st.session_state:
                st.session_state.saved_medical_terms = []
            
            # Add new terms to glossary
            for term in terms:
                if not any(t["term"] == term["term"] for t in st.session_state.saved_medical_terms):
                    st.session_state.saved_medical_terms.append(term)
            
            # Save to persistent storage
            save_terms(st.session_state.saved_medical_terms)
            st.success(f"Added {len(terms)} terms to your glossary")
```

## Example Usage

### Basic Term Detection

```python
import streamlit as st
from medical_terms_detector import get_medical_terms_detector

st.title("Medical Term Detector")

# Initialize detector
detector = get_medical_terms_detector()

# Get text input
text = st.text_area("Enter medical text to analyze:", 
                   "The patient underwent a mammogram followed by a biopsy. The results showed invasive ductal carcinoma.")

if st.button("Analyze Text"):
    # Detect medical terms
    terms = detector.detect_medical_terms(text)
    
    # Display results
    st.markdown(detector.format_results_for_display(terms), unsafe_allow_html=True)
```

### Glossary Management

```python
import streamlit as st
from medical_glossary import medical_glossary_ui

st.title("My Medical Glossary")

# Display the glossary UI
medical_glossary_ui()
```

### Complete Integration

```python
import streamlit as st
from medical_terms_detector import get_medical_terms_detector
from medical_glossary import medical_glossary_ui

st.title("Medical Terminology Assistant")

# Create tabs for different functions
tab1, tab2 = st.tabs(["Term Detector", "My Glossary"])

with tab1:
    # Initialize detector
    detector = get_medical_terms_detector()
    
    # Get text input
    text = st.text_area("Enter medical text to analyze:")
    
    if st.button("Analyze Text"):
        terms = detector.detect_medical_terms(text)
        st.markdown(detector.format_results_for_display(terms), unsafe_allow_html=True)
        
        # Option to save to glossary
        if terms and st.button("Save to Glossary"):
            if "saved_medical_terms" not in st.session_state:
                st.session_state.saved_medical_terms = []
            
            # Add terms to glossary
            for term in terms:
                if not any(t["term"] == term["term"] for t in st.session_state.saved_medical_terms):
                    st.session_state.saved_medical_terms.append(term)

with tab2:
    # Display the glossary UI
    medical_glossary_ui()
```

## Technical Reference

### Medical Terms Detector

```python
class MedicalTermsDetector:
    """Class for detecting and explaining medical terms in text"""
    
    def __init__(self):
        """Initialize the detector with glossary and optional transformer model"""
        
    def detect_medical_terms(self, text):
        """Detect medical terms in the given text"""
        
    def _get_context(self, text, term):
        """Extract context around the identified term"""
        
    def format_results_for_display(self, terms):
        """Format detected terms for display in the UI"""
```

### Medical Glossary UI

```python
def load_saved_terms():
    """Load saved medical terms from a JSON file"""
    
def save_terms(terms):
    """Save medical terms to a JSON file"""
    
def medical_glossary_ui():
    """User interface for the medical glossary"""
```

### Core Dependencies

- `streamlit`: For user interface components
- `pandas`: For data manipulation and CSV export
- `json`: For data persistence
- `datetime`: For timestamping exports
- `transformers` (optional): For BioBERT-powered term detection
- `reportlab` (optional): For PDF export capabilities

---

The Medical Glossary and Terms Detection System provides a powerful tool for bridging the gap between technical medical language and patient understanding. By automatically identifying and explaining medical terminology in context, and allowing users to build personalized glossaries, this system enhances healthcare communication and empowers patients in their medical journey.
