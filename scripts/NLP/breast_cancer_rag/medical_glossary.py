"""
medical_glossary.py

Streamlit UI module for managing, viewing, and exporting a glossary of medical terms.
This script allows users to:
- Load and save medical terms with explanations and categories.
- Interactively browse, filter, and remove saved glossary entries.
- Export the glossary in various formats (Markdown, CSV, PDF, JSON).
Used primarily in the Breast Cancer RAG app to support user understanding of medical language.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json
import os

def load_saved_terms():
    """Load saved medical terms from a JSON file"""
    if os.path.exists("medical_terms.json"):
        try:
            with open("medical_terms.json", "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading saved terms: {e}")
    return []

def save_terms(terms):
    """Save medical terms to a JSON file"""
    try:
        with open("medical_terms.json", "w") as f:
            json.dump(terms, f)
        return True
    except Exception as e:
        st.error(f"Error saving terms: {e}")
        return False

def medical_glossary_ui():
    """User interface for the medical glossary"""
    st.header("📚 Medical Glossary")
    
    # Initialize saved terms or load them from session_state
    if 'saved_medical_terms' not in st.session_state:
        st.session_state.saved_medical_terms = load_saved_terms()
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["My Saved Terms", "Search & Filter", "Export Options"])
    
    # Tab 1: Saved terms
    with tab1:
        if not st.session_state.saved_medical_terms:
            st.info("You haven't saved any medical terms yet. When reading through the conversation, click on 'Explain medical terms' to analyze and save terms for your reference.")
        else:
            # Sort terms by category
            terms_by_category = {}
            for term in st.session_state.saved_medical_terms:
                # Determinar categoría (si no existe, asignar 'Other')
                if 'category' in term:
                    category = term['category']
                else:
                    # Intenta categorizar basándose en el término
                    term_text = term['term'].lower()
                    if any(x in term_text for x in ['mammogram', 'biopsy', 'scan', 'test', 'imaging']):
                        category = 'Diagnostic Procedures'
                    elif any(x in term_text for x in ['therapy', 'treatment', 'surgery', 'mastectomy']):
                        category = 'Treatments'
                    elif any(x in term_text for x in ['cancer', 'tumor', 'carcinoma', 'invasive']):
                        category = 'Conditions'
                    else:
                        category = 'Other'
                    
                    # Añadir la categoría al término
                    term['category'] = category
                    
                if category not in terms_by_category:
                    terms_by_category[category] = []
                terms_by_category[category].append(term)
                
            # Display terms by category
            for category, terms in terms_by_category.items():
                with st.expander(f"{category} ({len(terms)} terms)", expanded=True):
                    for i, term in enumerate(terms):
                        col1, col2, col3 = st.columns([3, 8, 1])
                        with col1:
                            st.markdown(f"**{term['term'].capitalize()}**")
                        with col2:
                            # Usar 'definition' si está disponible, caer en 'explanation' para compatibilidad
                            definition = term.get('definition', term.get('explanation', 'No explanation available'))
                            st.markdown(definition, unsafe_allow_html=True)
                        with col3:
                            if st.button("❌", key=f"delete_{category}_{i}"):
                                st.session_state.saved_medical_terms.remove(term)
                                save_terms(st.session_state.saved_medical_terms)
                                st.experimental_rerun()
                        st.markdown("---")
            
            # Button to save terms
            if st.button("Save glossary to file"):
                if save_terms(st.session_state.saved_medical_terms):
                    st.success("Glossary saved successfully")
    
    # Tab 2: Search and filter
    with tab2:
        if not st.session_state.saved_medical_terms:
            st.info("No terms available to search. Add some terms first.")
        else:
            # Search and filter options
            search_term = st.text_input("Search for a term:")
            
            # Get all available categories
            all_categories = list(set(term.get('category', 'Other') for term in st.session_state.saved_medical_terms))
            selected_categories = st.multiselect("Filter by category:", all_categories, default=all_categories)
            
            # Filter terms
            filtered_terms = []
            for term in st.session_state.saved_medical_terms:
                category = term.get('category', 'Other')
                term_text = term['term'].lower()
                
                # Apply filters
                category_match = category in selected_categories
                search_match = not search_term or search_term.lower() in term_text or search_term.lower() in term.get('explanation', '').lower()
                
                if category_match and search_match:
                    filtered_terms.append(term)
            
            # Show results
            if filtered_terms:
                st.markdown(f"### Found {len(filtered_terms)} matching terms")
                for term in filtered_terms:
                    with st.expander(f"{term['term'].capitalize()}", expanded=False):
                        st.markdown(f"**Category:** {term.get('category', 'Other')}")
                        st.markdown(term.get('definition', term.get('explanation', 'No explanation available')), unsafe_allow_html=True)
            else:
                st.warning("No terms match your search criteria.")
    
    # Tab 3: Export options
    with tab3:
        if not st.session_state.saved_medical_terms:
            st.info("No terms available to export. Add some terms first.")
        else:
            st.subheader("Export your medical glossary")
            
            # Export options
            export_format = st.radio("Export format:", ["Markdown", "CSV", "PDF", "JSON"])
            
            # Generate file according to format
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
                        md_content += f"{term.get('definition', term.get('explanation', 'No explanation available'))}\n\n"
                        md_content += f"{st.markdown(term.get('definition', term.get('explanation', 'No explanation available')))}\n\n"
                
                st.download_button(
                    "Download Markdown",
                    md_content,
                    f"medical_glossary_{datetime.now().strftime('%Y%m%d')}.md",
                    "text/markdown"
                )
            
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
            
            elif export_format == "PDF":
                st.info("PDF export functionality requires the ReportLab package. You can install it with `pip install reportlab`.")
                
                try:
                    from reportlab.lib.pagesizes import letter
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                    from reportlab.lib import colors
                    import io
                    
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
                    styles.add(ParagraphStyle(
                        name='TermHeading',
                        parent=styles['Heading3'],
                        textColor=colors.darkgreen,
                        spaceBefore=8
                    ))
                    
                    # Create content
                    content = []
                    content.append(Paragraph("My Medical Glossary", styles['Title']))
                    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Italic']))
                    content.append(Spacer(1, 20))
                    
                    # Group by category
                    terms_by_category = {}
                    for term in st.session_state.saved_medical_terms:
                        category = term.get('category', 'Other')
                        if category not in terms_by_category:
                            terms_by_category[category] = []
                        terms_by_category[category].append(term)
                    
                    for category, terms in terms_by_category.items():
                        content.append(Paragraph(category, styles['CategoryHeading']))
                        
                        for term in terms:
                            content.append(Paragraph(term['term'].capitalize(), styles['TermHeading']))
                            content.append(Paragraph(term.get('explanation', 'No explanation available'), styles['Normal']))
                            content.append(Spacer(1, 10))
                    
                    # Build PDF
                    doc.build(content)
                    pdf_data = buffer.getvalue()
                    buffer.close()
                    
                    st.download_button(
                        "Download PDF",
                        pdf_data,
                        f"medical_glossary_{datetime.now().strftime('%Y%m%d')}.pdf",
                        "application/pdf"
                    )
                    
                except ImportError:
                    st.error("ReportLab is not installed. Please install it to enable PDF export.")
            
            elif export_format == "JSON":
                # Export data as JSON
                json_data = json.dumps(st.session_state.saved_medical_terms, indent=2)
                
                st.download_button(
                    "Download JSON",
                    json_data,
                    f"medical_glossary_{datetime.now().strftime('%Y%m%d')}.json",
                    "application/json"
                )

# For standalone testing
if __name__ == "__main__":
    st.set_page_config(page_title="Medical Glossary", page_icon="📚", layout="wide")
    
    # Test data
    if 'saved_medical_terms' not in st.session_state:
        st.session_state.saved_medical_terms = [
            {
                "term": "mammogram",
                "explanation": "An X-ray image of the breast used to detect early signs of breast cancer.",
                "category": "Diagnostic Procedures",
                "confidence": 0.9
            },
            {
                "term": "mastectomy",
                "explanation": "Surgery to remove all breast tissue from a breast as a way to treat or prevent breast cancer.",
                "category": "Treatments",
                "confidence": 0.95
            }
        ]
    
    medical_glossary_ui()