# BI-RADS Classification using Weighted Rule-Based System (WRBS)

## Introduction

This document provides an in-depth explanation of the Weighted Rule-Based System (WRBS) implemented for BI-RADS classification in breast ultrasound images. The system automatically extracts features from segmented ultrasound images and applies a weighted scoring algorithm to classify findings according to the Breast Imaging-Reporting and Data System (BI-RADS) categories.

## Table of Contents

1. [Overview of the BI-RADS Classification System](#overview-of-the-bi-rads-classification-system)
2. [Technical Implementation](#technical-implementation)
   - [Core Components](#core-components)
   - [Feature Extraction](#feature-extraction)
   - [Weighted Scoring Algorithm](#weighted-scoring-algorithm)
   - [Fuzzy Logic Algorithm](#fuzzy-logic-algorithm)
3. [Integration with Streamlit UI](#integration-with-streamlit-ui)
4. [Usage Guide](#usage-guide)
5. [Code Documentation](#code-documentation)
6. [Clinical Significance](#clinical-significance)
7. [Future Improvements](#future-improvements)

## Overview of the BI-RADS Classification System

The Breast Imaging-Reporting and Data System (BI-RADS) is a standardized framework for classifying breast imaging findings. It categorizes lesions from BI-RADS 1 (negative) to BI-RADS 5 (highly suggestive of malignancy), with BI-RADS 4 further subdivided into categories A, B, and C based on suspicion level.

Our implementation uses a Weighted Rule-Based System (WRBS) approach to automatically classify breast ultrasound findings into BI-RADS categories based on imaging features.

## Technical Implementation

### Core Components

The system is implemented in the `BIRADSClassifierWRBS` class within `birads_wrbs.py`. This class represents the foundation of our classification approach and performs several key functions:

1. Feature extraction from segmented images
2. Feature scoring based on medical literature
3. Weighted score calculation
4. BI-RADS category assignment
5. Report generation

Here's the basic structure of the class:

```python
class BIRADSClassifierWRBS:
    """
    BI-RADS classification system based on weighted scoring of features
    identified in breast ultrasound images.
    
    This implementation uses a weighted scoring approach, aligning
    with medical literature and reducing false positives.
    """
    
    def __init__(self):
        # Initialize variables, domains, scoring systems, and thresholds
        
    def extract_features_from_segmentation(self, original_image, mask, pixel_spacing_mm=0.2):
        # Extract features from segmented image
        
    def calculate_birads_score(self, algorithm='weighted'):
        # Calculate BI-RADS score using selected algorithm
        
    def _calculate_weighted_score(self):
        # Standard weighted scoring algorithm
        
    def _calculate_fuzzy_logic_score(self):
        # Fuzzy logic algorithm with linguistic variables
        
    def generate_report(self, birads_category, confidence, total_score, detailed_results):
        # Generate clinical report
        
    def classify(self, algorithm='weighted'):
        # Main method combining score calculation and report generation
```

### Feature Extraction

The feature extraction process analyzes the segmented ultrasound image to quantify key diagnostic features. This is performed in the `extract_features_from_segmentation` method.

Key features extracted include:

1. **Shape** (round, oval, irregular, etc.)
2. **Margin** (circumscribed, indistinct, spiculated, etc.)
3. **Orientation** (parallel or non-parallel)
4. **Echogenicity** (anechoic, hypoechoic, etc.)
5. **Posterior Features** (enhancement, shadowing)
6. **Size in millimeters**
7. **Boundary Regularity**
8. **Texture** (homogeneous or heterogeneous)

Here's how shape extraction is implemented:

```python
# Extract shape information
eccentricity = largest_region.eccentricity
solidity = largest_region.solidity
extent = largest_region.extent

# Determine shape category based on established metrics
if eccentricity < 0.3 and solidity > 0.9:
    self.variables['shape'] = 'round'
elif eccentricity < 0.7 and solidity > 0.8:
    self.variables['shape'] = 'oval'
elif solidity < 0.7:
    self.variables['shape'] = 'irregular'
elif eccentricity > 0.8:
    self.variables['shape'] = 'lobular'
else:
    self.variables['shape'] = 'regular'
```

The feature extraction leverages scikit-image's `regionprops` to calculate shape metrics, boundary analysis, texture properties, and more. For complex features like echogenicity and posterior features, specialized methods are implemented:

- `_estimate_echogenicity`: Compares lesion intensity to fat or surrounding tissue
- `_analyze_posterior_features`: Examines the area beneath the lesion for shadowing or enhancement

### Weighted Scoring Algorithm

The core of the classification system is the weighted scoring algorithm implemented in `_calculate_weighted_score()`. This method:

1. Assigns scores to each feature based on medical literature
2. Weights these scores according to clinical importance
3. Calculates a normalized total score
4. Maps this score to the appropriate BI-RADS category

The feature scoring system assigns values to each feature variant:

```python
self.feature_scores = {
    'shape': {
        'round': -0.3,       # Very benign
        'oval': -0.2,        # Benign
        'regular': -0.1,     # Probably benign
        'lobular': 0.1,      # Slightly suspicious
        'irregular': 0.3     # Very suspicious
    },
    'margin': {
        'circumscribed': -0.3,    # Very benign
        'microlobulated': 0.1,    # Slightly suspicious
        'indistinct': 0.2,        # Moderately suspicious
        'angular': 0.25,          # Suspicious
        'spiculated': 0.4         # Very suspicious
    },
    # Other features similarly defined
}
```

Feature weights reflect their clinical importance:

```python
self.feature_weights = {
    'shape': 0.20,          # 20% of total score
    'margin': 0.25,         # 25% of total score - margin is a key predictor
    'orientation': 0.15,    # 15% of total score
    'echogenicity': 0.15,   # 15% of total score
    'posterior': 0.10,      # 10% of total score
    'boundaries': 0.05,     # 5% of total score
    'texture': 0.10         # 10% of total score
}
```

The algorithm calculates the weighted score as follows:

```python
# Calculate score for each feature
for feature, weight in self.feature_weights.items():
    value = self.variables.get(feature)
    
    # Only consider evaluated features
    if value is not None and feature != 'size_mm':
        # Get score for this feature
        score = self.feature_scores.get(feature, {}).get(value, 0)
        
        # Save for explanation
        feature_scores[feature] = {
            'value': value,
            'score': score,
            'weighted_score': score * weight
        }
        
        # Accumulate weighted score
        weighted_score += score * weight
        used_weights_sum += weight

# Normalize by used weights
if used_weights_sum > 0:
    total_score = weighted_score / used_weights_sum
else:
    total_score = 0
```

The final BI-RADS category is determined using thresholds:

```python
self.birads_thresholds = {
    'BIRADS1': -1.0,     # Score < -0.4 (no visible lesion)
    'BIRADS2': -0.2,     # Score < -0.1 (definitely benign)
    'BIRADS3': 0.0,      # Score < 0.1 (probably benign)
    'BIRADS4A': 0.1,     # Score < 0.2 (low suspicion)
    'BIRADS4B': 0.2,     # Score < 0.3 (moderate suspicion)
    'BIRADS4C': 0.3,     # Score < 0.4 (high suspicion)
    'BIRADS5': 0.4       # Score >= 0.4 (highly suggestive of malignancy)
}
```

The system also calculates a confidence level for each classification, providing clinicians with an assessment of how certain the classification is.

### Fuzzy Logic Algorithm

An alternative classification approach uses fuzzy logic, implemented in `_calculate_fuzzy_logic_score()`. This method:

1. Defines membership functions for each feature
2. Uses linguistic variables to handle uncertainty
3. Applies fuzzy rules to combine evidence
4. Defuzzifies to produce a final score

Here's a snippet showing how the fuzzy membership function for shape is implemented:

```python
def shape_suspicion(shape):
    if shape is None:
        return 0.0  # Not evaluated
    
    # Different fuzzy mapping compared to standard algorithm
    suspicion_map = {
        'round': 0.0,  # Very benign
        'oval': 0.1,   # Benign
        'regular': 0.2, # Slightly suspicious
        'lobular': 0.5, # Moderately suspicious
        'irregular': 0.9 # Very suspicious
    }
    return suspicion_map.get(shape, 0.5)
```

Fuzzy rules are implemented using conditional logic:

```python
# Rule 1: If multiple highly suspicious features, increase suspicion synergistically
high_suspicion_count = sum(1 for score in [shape_susp, margin_susp, orientation_susp] 
                          if score > 0.6)
if high_suspicion_count >= 2:
    combined_suspicion = max(shape_susp, margin_susp, orientation_susp) + 0.2
    suspicion_level = min(0.9, combined_suspicion)  # Cap at 0.9
```

This approach is particularly valuable for handling cases with uncertainty in feature assessment and for capturing the synergistic effect of multiple suspicious features.

## Integration with Streamlit UI

The WRBS classifier is integrated into a Streamlit-based user interface that allows users to upload breast ultrasound images, visualize segmentation, and obtain BI-RADS classifications.

The integration is implemented in two main components:

1. The main application file that includes the breast ultrasound segmentation tab
2. A dedicated module for the BI-RADS classification tab

Here's how the classifier is instantiated and used in the UI:

```python
# Initialize the classifier
birads_classifier = BIRADSClassifierWRBS()

# Extract features from segmentation
birads_classifier.extract_features_from_segmentation(np.array(image), mask)

# Use selected algorithm
selected_algorithm = algorithm_map[wrbs_algorithm]
birads_category, confidence, total_score, detailed_results = birads_classifier.calculate_birads_score(algorithm=selected_algorithm)

# Generate report
report = birads_classifier.generate_report(birads_category, confidence, total_score, detailed_results)
```

The UI provides options for selecting between the weighted scoring and fuzzy logic algorithms:

```python
wrbs_algorithm = st.radio(
    "Select algorithm approach:",
    ["Weighted Scoring", "Fuzzy Logic Scoring"],
    horizontal=True,
    help="Weighted Scoring uses direct feature scoring. Fuzzy Logic uses linguistic variables and more complex rule combinations."
)

# Map radio button selection to algorithm parameter
algorithm_map = {
    "Weighted Scoring": "weighted",
    "Fuzzy Logic Scoring": "fuzzy"
}
```

## Usage Guide

To use the BI-RADS WRBS classifier:

1. **Upload an ultrasound image**: The system accepts PNG, JPG, or JPEG formats.
2. **Run segmentation**: The system automatically segments the image to identify lesions.
3. **Select classification algorithm**: Choose between "Weighted Scoring" and "Fuzzy Logic Scoring".
4. **View results**: The system displays:
   - BI-RADS category with confidence level
   - Extracted features
   - Category scores visualization
   - Detailed clinical report

The system can be used via:

1. The integrated Streamlit application
2. Direct API calls to the `BIRADSClassifierWRBS` class

Example API usage:

```python
import numpy as np
from PIL import Image
from birads_wrbs import BIRADSClassifierWRBS

# Load image and segmentation mask
image = np.array(Image.open("ultrasound.jpg").convert('L'))
mask = np.array(segmentation_model.predict(image))  # From your segmentation model

# Initialize classifier
classifier = BIRADSClassifierWRBS()

# Extract features
features = classifier.extract_features_from_segmentation(image, mask)

# Classify using weighted scoring
category, confidence, report, details = classifier.classify(algorithm="weighted")

# Print results
print(f"BI-RADS Category: {category} (Confidence: {confidence:.2f})")
print(report)
```

## Code Documentation

### Main Class: BIRADSClassifierWRBS

```python
class BIRADSClassifierWRBS:
    """
    BI-RADS classification system based on weighted scoring of features
    identified in breast ultrasound images.
    """
    
    def __init__(self):
        """
        Initialize the BI-RADS classifier with variables, scoring weights, and thresholds.
        """
        # Initialize variables, domains, scoring systems, etc.
        
    def extract_features_from_segmentation(self, original_image, mask, pixel_spacing_mm=0.2):
        """
        Extract lesion features from segmentation results.
        
        Args:
            original_image: Original ultrasound image
            mask: Binary segmentation mask
            pixel_spacing_mm: Conversion factor from pixels to mm (default 0.2mm/pixel)
            
        Returns:
            Dict: Extracted feature values
        """
        # Feature extraction implementation
        
    def calculate_birads_score(self, algorithm='weighted'):
        """
        Calculate the weighted score based on extracted features
        and determine the corresponding BI-RADS category.
        
        Args:
            algorithm: Algorithm to use ('weighted' or 'fuzzy')
            
        Returns:
            Tuple: (BI-RADS category, confidence level, total score, detailed results)
        """
        # Score calculation implementation
        
    def generate_report(self, birads_category, confidence, total_score, detailed_results, patient_info=None):
        """
        Generate a detailed clinical report explaining the BI-RADS classification.
        
        Args:
            birads_category: Determined BI-RADS category
            confidence: Confidence level in the classification
            total_score: Total calculated score
            detailed_results: Detailed results for each category
            patient_info: Optional patient information
            
        Returns:
            str: Formatted clinical report
        """
        # Report generation implementation
        
    def classify(self, algorithm='weighted'):
        """
        Main method to classify the lesion into a BI-RADS category.
        Simplifies the API by combining score calculation and report generation.
        
        Args:
            algorithm: Algorithm to use ('weighted' or 'fuzzy')
            
        Returns:
            Tuple: (BI-RADS category, confidence level, detailed report, detailed results)
        """
        # Combined classification workflow
```

### Key Helper Methods

```python
def _get_real_size(self, size_value):
    """
    Helper method to extract the real size value if stored as a dict with metadata.
    """
    
def _estimate_echogenicity(self, original_image, mask, region):
    """
    Estimate echogenicity of lesion, attempting to use subcutaneous fat as reference when possible.
    """
    
def _analyze_posterior_features(self, original_image, mask, region):
    """
    Analyze the area posterior to the lesion to detect enhancement or shadowing.
    """
    
def _calculate_weighted_score(self):
    """
    Standard weighted scoring algorithm.
    """
    
def _calculate_fuzzy_logic_score(self):
    """
    Fuzzy logic algorithm using membership functions and linguistic variables.
    """
```

## Clinical Significance

The WRBS approach offers several advantages for clinical use:

1. **Evidence-based weights**: Feature weights are derived from medical literature and ACR guidelines
2. **Transparent scoring**: The system provides a clear explanation of how each feature contributes to the final score
3. **Confidence metrics**: Each classification includes a confidence level to guide clinical decision-making
4. **Customizable thresholds**: The thresholds for each BI-RADS category can be adjusted based on institutional preferences
5. **Comprehensive reporting**: The system generates detailed reports suitable for clinical documentation

The weighted scoring approach aligns with radiologists' mental model of BI-RADS assessment, making it more intuitive for clinical use compared to black-box approaches.

## Future Improvements

Potential enhancements to the WRBS system include:

1. **Machine learning integration**: Incorporating ML to refine feature weights based on clinical outcomes
2. **Dynamic weight adjustment**: Allowing weights to vary based on patient demographics or risk factors
3. **Temporal analysis**: Comparing current findings with previous examinations to detect changes
4. **Multimodal integration**: Combining ultrasound findings with mammography, MRI, or clinical data
5. **Interactive threshold adjustment**: Enabling radiologists to adjust thresholds interactively to see the impact on classification

The modularity of the WRBS approach makes it particularly well-suited for these enhancements, as individual components can be improved without redesigning the entire system.

---

This implementation represents a practical, transparent, and clinically aligned approach to BI-RADS classification in breast ultrasound imaging. By combining domain knowledge with computational techniques, it bridges the gap between purely human assessment and black-box AI systems.
