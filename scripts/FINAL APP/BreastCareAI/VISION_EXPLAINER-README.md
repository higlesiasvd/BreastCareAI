# Breast Ultrasound Vision Explainer Module

## Introduction

The Breast Ultrasound Vision Explainer Module leverages AI vision models to provide comprehensive, human-readable explanations of breast ultrasound segmentation results. This module bridges the gap between advanced image segmentation algorithms and clinical interpretation by automatically generating radiological-style reports from segmentation outputs, metrics, and classification results.

Using the Ollama framework with multimodal vision models, the explainer transforms complex technical outputs into accessible explanations that could help clinicians and patients better understand AI-assisted ultrasound analysis.

## Table of Contents

1. [Overview](#overview)
2. [Core Components](#core-components)
3. [Technical Implementation](#technical-implementation)
4. [Integration with Segmentation Pipeline](#integration-with-segmentation-pipeline)
5. [BI-RADS Integration](#bi-rads-integration)
6. [Visual Explanation Generation](#visual-explanation-generation)
7. [Customization Options](#customization-options)
8. [Example Usage](#example-usage)
9. [Technical Reference](#technical-reference)

## Overview

The Vision Explainer module serves as an interpretability layer on top of breast ultrasound segmentation results. By combining original images, segmentation masks, evaluation metrics, and optional BI-RADS classification data, it generates:

1. A composite visual explanation that combines multiple visualization elements
2. A structured radiological-style text report explaining the findings in accessible language
3. An analysis of the AI model's behavior and decision-making process

This explanation framework helps make AI-assisted breast ultrasound analysis more transparent, interpretable, and trustworthy for clinical use.

## Core Components

The module consists of the following core components:

### VisionExplainer Class

The primary class that manages the generation of explanations using Ollama vision models. It includes methods for:

- Checking Ollama model availability
- Creating composite visualization images
- Generating textual explanations
- Formatting and structuring the output

### Composite Visualization Generator

This component creates a combined image that presents multiple visualizations side by side:

- Original ultrasound image
- AI segmentation overlay (highlighted in red)
- Ground truth mask (highlighted in blue) if available
- Model attention heatmap (Grad-CAM visualization) if available
- Metrics panel with quantitative assessment data

### AI Report Generator

Using the Ollama Python library with multimodal vision models (e.g., Llama 3 Vision), this component:

- Analyzes the composite image
- Interprets metrics and visual patterns
- Generates a structured report following radiological reporting conventions
- Explains findings in accessible, non-technical language

## Technical Implementation

### VisionExplainer Class Initialization

The `VisionExplainer` class initializes with a specified Ollama vision model:

```python
def __init__(self, model_name="llama3:vision"):
    """
    Initialize the vision explainer
    Args:
        model_name (str): The name of the Ollama model to use
    """
    self.model_name = model_name
    # Check if the model is available in Ollama
    self._check_ollama_model()
```

Upon initialization, the class verifies that the specified model is available in the local Ollama installation, providing helpful feedback if it's not.

### Composite Image Creation

The `_create_combined_image_with_features` method generates a multi-panel visualization with all available information:

```python
def _create_combined_image_with_features(self, original_image, segmented_image, metrics, true_mask=None, attention_map=None):
    """
    Create a combined image with original, segmentation, ground truth mask, attention map, and metrics
    Args:
        original_image (PIL.Image): Original ultrasound image
        segmented_image (PIL.Image): Segmented overlay image
        metrics (dict): Dictionary of segmentation metrics
        true_mask (PIL.Image/np.ndarray, optional): Ground truth mask if available
        attention_map (PIL.Image, optional): Attention visualization (Grad-CAM)
    Returns:
        PIL.Image: Combined image for explanation
    """
    # Implementation details for creating a multi-panel visualization
    # ...
```

This method dynamically:
1. Determines how many panels to include based on available data
2. Resizes all images to consistent dimensions
3. Arranges images side by side in a logical sequence
4. Adds a metrics panel with numerical evaluation data
5. Labels each panel with descriptive titles
6. Returns a single composite image

### Explanation Generation

The main `explain_segmentation` method orchestrates the explanation process:

```python
def explain_segmentation(self, original_image, segmented_image, metrics, true_mask=None, attention_map=None,
                        birads_category=None, birads_confidence=None, prompt_template=None):
    """
    Generate an explanation of the segmentation results using Ollama
    Args:
        original_image (PIL.Image): Original ultrasound image
        segmented_image (PIL.Image): Segmented overlay image
        metrics (dict): Dictionary of segmentation metrics
        true_mask (PIL.Image/np.ndarray, optional): Ground truth mask if available
        attention_map (PIL.Image, optional): Attention visualization (Grad-CAM)
        birads_category (str, optional): BI-RADS classification result
        birads_confidence (float, optional): Confidence in BI-RADS classification
        prompt_template (str, optional): Custom prompt template
    Returns:
        str: AI-generated explanation of the segmentation results
    """
    # Implementation details for generating the explanation
    # ...
```

This method:
1. Creates the composite visualization image
2. Saves a temporary copy for later display
3. Constructs an appropriate prompt based on available data
4. Invokes the Ollama API with the image and prompt
5. Post-processes the response to ensure accuracy
6. Returns the formatted explanation

## Integration with Segmentation Pipeline

The Vision Explainer is designed to seamlessly integrate with the breast ultrasound segmentation pipeline. It accepts the outputs from the segmentation process, including:

- Original ultrasound image (`PIL.Image`)
- Segmentation overlay (`PIL.Image`)
- Segmentation metrics (`dict`)
- Ground truth mask (optional)
- Attention map from Grad-CAM (optional)

This flexibility allows it to function with varying levels of available data, adapting its output based on what's provided.

## BI-RADS Integration

The module supports integration with Breast Imaging-Reporting and Data System (BI-RADS) classification through optional parameters:

```python
birads_category=None, birads_confidence=None
```

When BI-RADS classification data is provided, the explainer:

1. Looks up the exact definition and risk level for the category
2. Ensures explanations adhere to the standardized BI-RADS descriptors
3. Includes a dedicated BI-RADS assessment section in the report
4. Verifies that the explanation correctly represents the BI-RADS category

The module maintains a comprehensive dictionary of BI-RADS categories with precise definitions:

```python
birads_definitions = {
    'BIRADS0': "BI-RADS 0: Incomplete - Need additional imaging. This is not a final assessment.",
    'BIRADS1': "BI-RADS 1: Negative. No abnormalities found. Risk of malignancy is essentially 0%.",
    'BIRADS2': "BI-RADS 2: Benign finding. Definitely benign lesions with 0% risk of malignancy.",
    'BIRADS3': "BI-RADS 3: Probably benign. Less than 2% risk of malignancy. Typically requires follow-up.",
    'BIRADS4A': "BI-RADS 4A: Low suspicion for malignancy. Risk of malignancy is 2-10%. Requires biopsy.",
    'BIRADS4B': "BI-RADS 4B: Moderate suspicion for malignancy. Risk of malignancy is 10-50%. Requires biopsy.",
    'BIRADS4C': "BI-RADS 4C: High suspicion for malignancy. Risk of malignancy is 50-95%. Requires biopsy.",
    'BIRADS5': "BI-RADS 5: Highly suggestive of malignancy. Risk of malignancy is >95%. Requires biopsy."
}
```

## Visual Explanation Generation

The composite visualization is a key component of the explanation process. It includes:

1. **Original Image Panel**: Shows the unprocessed ultrasound image
2. **Segmentation Panel**: Displays the segmentation overlay with red highlighting
3. **Ground Truth Panel** (optional): Shows the reference mask with blue highlighting
4. **Attention Map Panel** (optional): Visualizes where the model focused using a heatmap
5. **Metrics Panel**: Presents quantitative assessment data including:
   - Area coverage percentage
   - Dice coefficient (when ground truth is available)
   - IoU/Jaccard index (when ground truth is available)
   - Quality assessment (Excellent, Good, Fair, Poor)
   - Explainability metrics (Boundary Focus, Contiguity, Attention Entropy)

Example of the combined visualization:

```
┌────────────────┬────────────────┬────────────────┬────────────────┬────────────────┐
│                │                │                │                │                │
│                │                │                │                │ SEGMENTATION   │
│                │                │                │                │ METRICS        │
│                │                │                │                │                │
│  Original      │  AI            │  Ground        │  Model         │ Area: 23.5%    │
│  Ultrasound    │  Segmentation  │  Truth Mask    │  Attention     │ Dice: 0.856    │
│                │                │                │                │ IoU: 0.749     │
│                │                │                │                │ Quality: Good  │
│                │                │                │                │                │
│                │                │                │                │ EXPLAINABILITY │
│                │                │                │                │ METRICS        │
│                │                │                │                │                │
│                │                │                │                │ Boundary: 65.2%│
│                │                │                │                │ Contiguity: 0.8│
│                │                │                │                │ Entropy: 0.45  │
│                │                │                │                │                │
└────────────────┴────────────────┴────────────────┴────────────────┴────────────────┘
```

## Customization Options

The module offers several customization options:

### Prompt Templates

Users can provide custom prompt templates to guide the explanation generation:

```python
custom_prompt = """
Please analyze this breast ultrasound image and focus specifically on 
the texture patterns visible in the segmented region. Compare the 
segmentation with the ground truth and discuss any discrepancies.
"""

explanation = vision_explainer.explain_segmentation(
    original_image=image,
    segmented_image=overlay,
    metrics=metrics,
    true_mask=true_mask,
    prompt_template=custom_prompt
)
```

### System Prompts

The module uses a detailed system prompt to guide the AI's response format:

```python
system_prompt = """
You are a radiologist specialized in breast ultrasound analysis with extensive knowledge of the BI-RADS classification system.
IMPORTANT: The exact definitions of BI-RADS categories are:
- BI-RADS 0: Incomplete - Need additional imaging
- BI-RADS 1: Negative (0% risk of malignancy)
...
Your task is to provide a radiological-style report written in accessible language for a general audience.
IMPORTANT INSTRUCTIONS ABOUT YOUR RESPONSE FORMAT:
- Structure your report as a medical radiologist would, with clear sections
...
"""
```

This system prompt ensures that the generated explanations follow a consistent structure and adhere to proper medical terminology while remaining accessible.

### Quality Control

The module includes post-processing to ensure the accuracy of BI-RADS descriptions:

```python
# If incorrect description found, override it
if has_incorrect_description:
    # Create corrected BI-RADS section
    corrected_section = """BI-RADS ASSESSMENT
    The image has been classified as """ + birads_description + """ with a confidence of """ + f"{birads_confidence:.2f}" + """.
    ...
    """
```

This quality control mechanism prevents the model from generating inconsistent or incorrect BI-RADS descriptions, ensuring that the explanations remain medically accurate.

## Example Usage

Here's a complete example of how to use the Vision Explainer module:

```python
from PIL import Image
import numpy as np
from vision_explainer import VisionExplainer

# Load images and results from segmentation pipeline
original_image = Image.open("ultrasound.jpg").convert('L')
mask = np.load("segmentation_mask.npy")
metrics = {
    'area_ratio': 0.235,
    'dice': 0.856,
    'iou': 0.749,
    'sensitivity': 0.874,
    'specificity': 0.968,
    'importance_contiguity': 0.812,
    'boundary_focus': 0.652,
    'normalized_entropy': 0.45
}

# Create overlay from original image and mask
from breast_segmentation_module import BreastSegmentationModel
segmentation_model = BreastSegmentationModel("model_weights.pth")
overlay = segmentation_model.overlay_mask(original_image, mask)

# Get BI-RADS classification
from birads_wrbs import BIRADSClassifierWRBS
birads_classifier = BIRADSClassifierWRBS()
birads_category, confidence, _, _ = birads_classifier.calculate_birads_score()

# Initialize the vision explainer
explainer = VisionExplainer(model_name="llama3:vision")

# Generate explanation
explanation = explainer.explain_segmentation(
    original_image=original_image,
    segmented_image=overlay,
    metrics=metrics,
    birads_category=birads_category,
    birads_confidence=confidence
)

# Print or display the explanation
print(explanation)
```

## Technical Reference

### VisionExplainer Class

```python
class VisionExplainer:
    """
    Class for generating explanations of breast ultrasound segmentation results
    using Ollama's Python library with vision models.
    """
    
    def __init__(self, model_name="llama3:vision"):
        """
        Initialize the vision explainer
        Args:
            model_name (str): The name of the Ollama model to use
        """
        
    def _check_ollama_model(self):
        """Check if the specified model is available in Ollama"""
        
    def _create_combined_image_with_features(self, original_image, segmented_image, metrics, true_mask=None, attention_map=None):
        """
        Create a combined image with original, segmentation, ground truth mask, attention map, and metrics
        Args:
            original_image (PIL.Image): Original ultrasound image
            segmented_image (PIL.Image): Segmented overlay image
            metrics (dict): Dictionary of segmentation metrics
            true_mask (PIL.Image/np.ndarray, optional): Ground truth mask if available
            attention_map (PIL.Image, optional): Attention visualization (Grad-CAM)
        Returns:
            PIL.Image: Combined image for explanation
        """
        
    def explain_segmentation(self, original_image, segmented_image, metrics, true_mask=None, attention_map=None,
                            birads_category=None, birads_confidence=None, prompt_template=None):
        """
        Generate an explanation of the segmentation results using Ollama
        Args:
            original_image (PIL.Image): Original ultrasound image
            segmented_image (PIL.Image): Segmented overlay image
            metrics (dict): Dictionary of segmentation metrics
            true_mask (PIL.Image/np.ndarray, optional): Ground truth mask if available
            attention_map (PIL.Image, optional): Attention visualization (Grad-CAM)
            birads_category (str, optional): BI-RADS classification result
            birads_confidence (float, optional): Confidence in BI-RADS classification
            prompt_template (str, optional): Custom prompt template
        Returns:
            str: AI-generated explanation of the segmentation results
        """
```

### Metrics Dictionary Structure

The metrics dictionary expects the following structure:

```python
metrics = {
    'area_ratio': float,         # Percentage of image covered by segmentation (0.0-1.0)
    'dice': float,               # Dice coefficient (0.0-1.0) - optional
    'iou': float,                # IoU/Jaccard index (0.0-1.0) - optional
    'sensitivity': float,        # Sensitivity/recall (0.0-1.0) - optional
    'specificity': float,        # Specificity (0.0-1.0) - optional
    'normal_case': bool,         # Flag for normal case (no lesion) - optional
    
    # Explainability metrics - optional
    'max_importance': float,     # Maximum attention value
    'mean_importance': float,    # Average attention value
    'importance_entropy': float, # Raw entropy of attention distribution
    'normalized_entropy': float, # Normalized entropy (0.0-1.0)
    'importance_top10_percent': float,  # Average attention in top 10% regions
    'importance_contiguity': float,     # Spatial contiguity of attention (0.0-1.0)
    'boundary_focus': float      # Ratio of boundary to region attention (0.0-1.0)
}
```

Not all metrics are required; the module adapts to the available data.

---

The Breast Ultrasound Vision Explainer module represents a significant advancement in making AI-assisted breast ultrasound analysis more transparent, interpretable, and accessible. By generating comprehensive explanations that combine visual elements with structured reports, it helps bridge the gap between AI algorithms and clinical understanding, potentially improving both clinician and patient comprehension of AI-assisted findings.
