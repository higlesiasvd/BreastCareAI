"""
BI-RADS Classification Tab for Breast Ultrasound Analysis Application

This module defines a Streamlit tab for BI-RADS classification using
Weighted Rule-Based System (WRBS).

Part of the Intelligent Systems course project.
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import io
import os

from birads_wrbs import BIRADSClassifierWRBS


def add_birads_wrbs_tab():
    """
    Adds a BI-RADS classification tab to the Streamlit application using
    Weighted Rule-Based System.
    
    This tab allows users to upload ultrasound images, run segmentation,
    and get a BI-RADS classification using weighted scoring algorithms.
    """
    st.title("🏥 BI-RADS Classification using Weighted Rule-Based System")
    
    st.markdown("""
    This module uses a Weighted Rule-Based System (WRBS) to classify breast ultrasound lesions 
    into appropriate BI-RADS categories. The classification is based on features extracted from 
    the segmentation results.
    
    ### What is BI-RADS?
    The Breast Imaging-Reporting and Data System (BI-RADS) is a standardized classification system 
    for breast imaging findings. It categorizes lesions from BI-RADS 1 (negative) to BI-RADS 5 
    (highly suspicious for malignancy).
    """)
    
    with st.expander("How Weighted Rule-Based System is used for BI-RADS Classification", expanded=False):
        st.markdown("""
        ### Weighted Rule-Based System (WRBS) for BI-RADS Classification
        
        Our approach represents BI-RADS classification as a weighted scoring system where:
        
        - **Variables**: Lesion features such as shape, margin, orientation, etc.
        - **Feature Scores**: Values that indicate whether a feature favors benignity (-) or malignancy (+)
        - **Feature Weights**: Importance of each feature based on medical literature (e.g., margin 25%, shape 20%)
        
        The system extracts features automatically from the segmentation and calculates a weighted score to find 
        the most appropriate BI-RADS category.
        
        We implement two different algorithms:
        1. **Weighted Scoring**: Direct calculation of weighted feature scores
        2. **Fuzzy Logic Scoring**: Uses fuzzy logic to handle uncertainty in feature values
        
        This approach aligns with clinical guidelines and provides a transparent methodology for classification.
        """)
    
    # File upload section
    st.header("Upload Ultrasound Image")
    uploaded_image = st.file_uploader(
        "Upload a breast ultrasound image",
        type=["png", "jpg", "jpeg"]
    )
    
    # If no image uploaded, show examples
    if uploaded_image is None:
        st.info("Please upload an ultrasound image to begin analysis.")
        
        # Show placeholder example
        st.subheader("Example Analysis")
        cols = st.columns(3)
        with cols[0]:
            st.image("example_images/benign_example.png", caption="Example: Benign lesion", width=200)
        with cols[1]:
            st.image("example_images/malignant_example.png", caption="Example: Suspicious lesion", width=200)
        with cols[2]:
            st.image("example_images/normal_example.png", caption="Example: Normal tissue", width=200)
            
        st.markdown("""
        Upload your own ultrasound image to get a BI-RADS classification using weighted scoring algorithms.
        The system will:
        1. Segment the image to identify lesions
        2. Extract features from the segmentation
        3. Apply weighted scoring algorithms to determine the appropriate BI-RADS category
        4. Generate a detailed report with clinical recommendations
        """)
        return
    
    # When image is uploaded
    col1, col2 = st.columns(2)
    
    # Load and display original image
    with col1:
        image = Image.open(uploaded_image).convert('L')  # Convert to grayscale
        st.image(image, caption="Original Ultrasound Image", use_column_width=True)
    
    # Get segmentation model from session state
    segmentation_model = st.session_state.segmentation_model
    
    # Run segmentation
    with st.spinner("Running segmentation..."):
        # Save image to temp file for processing
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
            image.save(temp_file, format='PNG')
            temp_path = temp_file.name
        
        # Run segmentation
        mask, prob_map = segmentation_model.predict(image, threshold=0.5)
        overlay_image = segmentation_model.overlay_mask(image, mask)
        
        # Display segmentation
        with col2:
            st.image(overlay_image, caption="Segmented Image", use_column_width=True)
    
    # Algorithm selection
    st.subheader("Algorithm Selection")
    wrbs_algorithm = st.radio(
        "Select algorithm to use:",
        ["Weighted Scoring", "Fuzzy Logic Scoring"],
        horizontal=True
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
        
        # Use selected algorithm
        selected_algorithm = algorithm_map[wrbs_algorithm]
        birads_category, confidence, total_score, detailed_results = birads_classifier.calculate_birads_score(algorithm=selected_algorithm)
        
        # Generate report
        report = birads_classifier.generate_report(birads_category, confidence, total_score, detailed_results)
    
    # Display BI-RADS results
    st.header("BI-RADS Classification Results")
    
    # Style for BI-RADS category
    birads_color = {
        'BIRADS0': 'gray',
        'BIRADS1': 'green',
        'BIRADS2': 'green',
        'BIRADS3': 'orange',
        'BIRADS4A': 'orange',
        'BIRADS4B': 'orange',
        'BIRADS4C': 'red',
        'BIRADS5': 'red'
    }.get(birads_category, 'blue')
    
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
    feature_col1, feature_col2 = st.columns(2)
    
    # Map feature names to more readable format
    feature_names = {
        'shape': 'Shape',
        'margin': 'Margin',
        'orientation': 'Orientation',
        'echogenicity': 'Echogenicity',
        'posterior': 'Posterior Features',
        'calcifications': 'Calcifications',
        'vascularity': 'Vascularity',
        'size_mm': 'Size (mm)',
        'boundaries': 'Boundaries',
        'texture': 'Texture'
    }
    
    # Display features in two columns
    for i, (feature, value) in enumerate(birads_classifier.variables.items()):
        if value is not None:
            col = feature_col1 if i % 2 == 0 else feature_col2
            col.metric(feature_names.get(feature, feature), value)
    
    # Display weighted scoring results
    st.subheader("Weighted Scoring Results")
    
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
    ax.set_ylabel('Category Score')
    ax.set_title('BI-RADS Category Scores')
    plt.tight_layout()
    
    # Display chart
    st.pyplot(fig)
    
    # Display full report
    st.subheader("Clinical Report")
    st.text_area("Detailed Report", report, height=400)
    
    # Option to download report
    st.download_button(
        label="Download Report (TXT)",
        data=report,
        file_name=f"birads_report_{birads_category.lower()}.txt",
        mime="text/plain"
    )
    
    # Clean up temp file
    if 'temp_path' in locals() and os.path.exists(temp_path):
        os.unlink(temp_path)