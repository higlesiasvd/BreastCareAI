"""
Breast Ultrasound Vision Explainer Module

This module provides an AI vision-based explanation of breast ultrasound segmentation results.
It combines the original ultrasound image, segmentation mask, and metrics to generate 
a comprehensive explanation using the Ollama Python library with vision models.
"""

import io
import os
import tempfile
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Import the Ollama Python library
try:
    import ollama
except ImportError:
    raise ImportError("Please install the Ollama Python library with: pip install ollama")

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
        self.model_name = model_name
        
        # Check if the model is available in Ollama
        self._check_ollama_model()
    
    def _check_ollama_model(self):
        """Check if the specified model is available in Ollama"""
        try:
            # List available models
            models = ollama.list()
            model_names = [model.get('name') for model in models.get('models', [])]
            
            if self.model_name not in model_names:
                print(f"⚠️ Model {self.model_name} not found in Ollama.")
                print(f"Available models: {', '.join(model_names)}")
                print(f"Please run 'ollama pull {self.model_name}' to download it.")
            else:
                print(f"✓ Model {self.model_name} is available")
                
        except Exception as e:
            print(f"⚠️ Error checking Ollama model: {str(e)}")
            print("Make sure the Ollama service is running and accessible.")
    
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
        # Resize images to be the same height
        height = 400
        
        # Calculate how many images we have and their space requirements
        num_images = 2  # Original and segmentation are always present
        if true_mask is not None:
            num_images += 1
        if attention_map is not None:
            num_images += 1
            
        # Calculate widths while maintaining aspect ratio
        width1 = int(original_image.width * height / original_image.height)
        
        # Create new image with appropriate width
        metrics_width = 400
        image_width = width1  # Base width for each image
        total_width = (image_width * num_images) + metrics_width
        
        combined = Image.new('RGB', (total_width, height), (255, 255, 255))
        
        # Resize and paste original image
        original_resized = original_image.resize((image_width, height))
        combined.paste(original_resized, (0, 0))
        
        # Resize and paste segmented image
        segmented_resized = segmented_image.resize((image_width, height))
        combined.paste(segmented_resized, (image_width, 0))
        
        current_position = image_width * 2
        
        # Paste true mask if available
        if true_mask is not None:
            # Ensure mask is a PIL Image
            if not isinstance(true_mask, Image.Image):
                # If it's a numpy array, convert to PIL Image
                if isinstance(true_mask, np.ndarray):
                    # If binary mask, convert to RGB with color
                    if true_mask.ndim == 2:
                        # Create colored mask (blue for ground truth)
                        h, w = true_mask.shape
                        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
                        colored_mask[true_mask > 0] = [0, 0, 255]  # Blue for ground truth
                        true_mask_img = Image.fromarray(colored_mask)
                    else:
                        true_mask_img = Image.fromarray(true_mask)
                else:
                    # If not a supported type, create a placeholder
                    true_mask_img = Image.new('RGB', (width1, height), (200, 200, 200))
            else:
                true_mask_img = true_mask
            
            # Resize mask to match original image
            true_mask_resized = true_mask_img.resize((image_width, height))
            
            # Paste into the combined image
            combined.paste(true_mask_resized, (current_position, 0))
            current_position += image_width
        
        # Paste attention map if available
        if attention_map is not None:
            # Convert to PIL Image if it's not already
            if not isinstance(attention_map, Image.Image):
                # Handle numpy array case
                if isinstance(attention_map, np.ndarray):
                    if attention_map.ndim == 2:
                        # Single channel attention map
                        import matplotlib.pyplot as plt
                        import matplotlib
                        matplotlib.use("Agg")  # Non-interactive backend
                        
                        # Apply colormap
                        cm = plt.get_cmap('jet')
                        colored_attention = cm(attention_map)
                        colored_attention = (colored_attention[:, :, :3] * 255).astype(np.uint8)
                        attention_img = Image.fromarray(colored_attention)
                    else:
                        # Already colored
                        attention_img = Image.fromarray(attention_map)
                else:
                    # Use as is
                    attention_img = attention_map
            else:
                attention_img = attention_map
                
            # Resize attention map
            attention_resized = attention_img.resize((image_width, height))
            
            # Paste into combined image
            combined.paste(attention_resized, (current_position, 0))
            current_position += image_width
        
        # Create a drawing context
        draw = ImageDraw.Draw(combined)
        
        # Add text for metrics panel
        try:
            # Try to load a nice font, fall back to default
            font = ImageFont.truetype("Arial", 14)
        except:
            font = ImageFont.load_default()
        
        # Add metrics text
        x_offset = current_position + 20
        y_offset = 20
        line_height = 25
        
        # Title
        draw.text((x_offset, y_offset), "SEGMENTATION METRICS", fill=(0, 0, 0), font=font)
        y_offset += line_height * 1.5
        
        # Area coverage
        area_percentage = metrics.get('area_ratio', 0) * 100
        draw.text((x_offset, y_offset), f"Area Coverage: {area_percentage:.1f}%", fill=(0, 0, 0), font=font)
        y_offset += line_height
        
        # Check if normal case
        if area_percentage < 1.0:
            draw.text((x_offset, y_offset), "Assessment: Normal tissue (no lesion)", fill=(0, 128, 0), font=font)
            y_offset += line_height
        
        # Add Dice coefficient if available
        if 'dice' in metrics:
            dice_value = metrics['dice']
            draw.text((x_offset, y_offset), f"Dice Coefficient: {dice_value:.3f}", fill=(0, 0, 0), font=font)
            y_offset += line_height
            
            # IoU value
            if 'iou' in metrics:
                draw.text((x_offset, y_offset), f"IoU (Jaccard): {metrics['iou']:.3f}", fill=(0, 0, 0), font=font)
                y_offset += line_height
            
            # Quality assessment
            quality = "Unknown"
            color = (0, 0, 0)
            
            if 'normal_case' in metrics and metrics['normal_case']:
                quality = "Perfect (Normal)"
                color = (0, 128, 0)  # Green
            elif dice_value > 0.8:
                quality = "Excellent"
                color = (0, 128, 0)  # Green
            elif dice_value > 0.7:
                quality = "Good" 
                color = (0, 192, 0)  # Light green
            elif dice_value > 0.5:
                quality = "Fair"
                color = (255, 128, 0)  # Orange
            else:
                quality = "Poor"
                color = (255, 0, 0)  # Red
                
            draw.text((x_offset, y_offset), f"Quality: {quality}", fill=color, font=font)
            y_offset += line_height
        
        # Add explainability metrics if available
        if 'max_importance' in metrics:
            y_offset += line_height/2
            draw.text((x_offset, y_offset), "EXPLAINABILITY METRICS", fill=(0, 0, 0), font=font)
            y_offset += line_height
            
            # Boundary focus
            if 'boundary_focus' in metrics:
                boundary_focus = metrics['boundary_focus'] * 100
                draw.text((x_offset, y_offset), f"Boundary Focus: {boundary_focus:.1f}%", fill=(0, 0, 0), font=font)
                y_offset += line_height
            
            # Contiguity
            if 'importance_contiguity' in metrics:
                contiguity = metrics['importance_contiguity']
                draw.text((x_offset, y_offset), f"Contiguity: {contiguity:.3f}", fill=(0, 0, 0), font=font)
                y_offset += line_height
            
            # Entropy
            if 'normalized_entropy' in metrics:
                entropy = metrics['normalized_entropy']
                draw.text((x_offset, y_offset), f"Attention Entropy: {entropy:.3f}", fill=(0, 0, 0), font=font)
                y_offset += line_height
        
        # Add a note for normal cases
        if 'normal_case' in metrics and metrics['normal_case']:
            y_offset += line_height/2
            draw.text((x_offset, y_offset), 
                    "Note: Both prediction and ground truth", fill=(0, 0, 0), font=font)
            y_offset += line_height
            draw.text((x_offset, y_offset), 
                    "show no lesions (normal case)", fill=(0, 0, 0), font=font)
        
        # Add titles to the images
        draw.text((image_width//2 - 60, 5), "Original Ultrasound", fill=(255, 255, 255), font=font)
        draw.text((image_width + image_width//2 - 80, 5), "AI Segmentation", fill=(255, 255, 255), font=font)
        
        current_title_pos = image_width * 2
        
        if true_mask is not None:
            draw.text((current_title_pos + image_width//2 - 80, 5), "Ground Truth Mask", fill=(255, 255, 255), font=font)
            current_title_pos += image_width
        
        if attention_map is not None:
            draw.text((current_title_pos + image_width//2 - 65, 5), "Model Attention", fill=(255, 255, 255), font=font)
        
        return combined

    def explain_segmentation(self, original_image, segmented_image, metrics, true_mask=None, attention_map=None, prompt_template=None):
        """
        Generate an explanation of the segmentation results using Ollama
        
        Args:
            original_image (PIL.Image): Original ultrasound image
            segmented_image (PIL.Image): Segmented overlay image
            metrics (dict): Dictionary of segmentation metrics including explainability metrics
            true_mask (PIL.Image/np.ndarray, optional): Ground truth mask if available
            attention_map (PIL.Image, optional): Attention visualization (Grad-CAM)
            prompt_template (str, optional): Custom prompt template
            
        Returns:
            str: AI-generated explanation of the segmentation results
        """
        # Create combined image for explanation
        combined_image = self._create_combined_image_with_features(
            original_image, 
            segmented_image, 
            metrics, 
            true_mask=true_mask,
            attention_map=attention_map
        )
        
        # Save the combined image to a temporary file for Streamlit to display later
        temp_img_path = os.path.join(tempfile.gettempdir(), "combined_segmentation.png")
        combined_image.save(temp_img_path)
        
        # Convert the image to bytes for Ollama
        img_buffer = io.BytesIO()
        combined_image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # Default prompt if none provided
        if prompt_template is None:
            is_normal = metrics.get('area_ratio', 0) * 100 < 1.0
            has_ground_truth = 'dice' in metrics
            has_mask = true_mask is not None
            has_explainability = 'max_importance' in metrics
            
            prompt_template = """
            Please analyze this breast ultrasound image and segmentation results as if you were a radiologist writing a report for a patient:
            - On the left is the original ultrasound image
            - Next is the AI-generated segmentation overlay (red areas show detected tissue of interest)
            """
            
            if has_mask:
                prompt_template += """
                - Also included is a ground truth mask for comparison (blue areas)
                """
                
            if attention_map is not None:
                prompt_template += """
                - The model attention map shows where the AI focused (heatmap)
                """
                
            prompt_template += """
            - On the right are the segmentation metrics
            
            Analyze all images collectively, not individually. Consider the relationship between the original, 
            the AI segmentation, and other available visualizations.
            
            Provide a comprehensive yet accessible explanation in the style of a radiological report but written 
            for a general audience without medical training. Your report should include:
            
            1. A description of what is visible in the ultrasound image
            2. An assessment of the AI segmentation quality and what it reveals
            3. An explanation of the metrics and what they mean in this specific case
            """
            
            if has_explainability:
                prompt_template += """
            4. An interpretation of the model's attention patterns and explainability metrics
                - Boundary Focus: Whether the model focuses on edges (high %) or regions (low %)
                - Contiguity: How concentrated the model's attention is (higher = more focused)
                - Attention Entropy: How evenly distributed the attention is (higher = more dispersed)
            """
                
            prompt_template += """
            5. A conclusion about what this analysis suggests (but avoid making specific diagnostic claims)
            
            Your explanation should be structured like a medical report but use language that a person without medical knowledge can understand.
            """
            
            # Add specific guidance based on metrics
            if is_normal:
                prompt_template += """
                This appears to be a normal case with minimal or no segmentation.
                Explain what it means when an ultrasound is classified as normal 
                and how the model's behavior is appropriate for this case.
                """
            elif has_ground_truth:
                prompt_template += f"""
                The Dice coefficient for this segmentation is {metrics['dice']:.3f}.
                Explain what this means about the quality of the segmentation
                and how well the model detected areas of interest compared to the reference.
                """
        
        # System prompt to guide the model's responses
        system_prompt = """
        You are a radiologist specialized in breast ultrasound analysis. You are examining 
        ultrasound images and their AI-generated segmentation results.
        
        Your task is to provide a radiological-style report written in accessible language for a general audience, only with the following structure, nothing more.
        
        IMPORTANT INSTRUCTIONS ABOUT YOUR RESPONSE FORMAT:
        - Structure your report as a medical radiologist would, with clear sections but without the Patient Information, clinical history, or other unnecessary details.
        - Begin with a "FINDINGS" section that objectively describes what is visible
        - Include a "TECHNICAL ASSESSMENT" section that evaluates the quality of the AI segmentation
        - Add an "INTERPRETATION" section that explains what the findings might indicate
        - If explainability metrics are provided, include an "AI BEHAVIOR ANALYSIS" section discussing how the model arrived at its results
        - End with an "IMPRESSION" section that summarizes the key points
        - Use medical terms when necessary but always explain them in parentheses
        - Be precise but accessible - explain concepts as you would to a patient
        - Maintain a professional and confident yet compassionate tone
        
        Remember that your analysis is for educational purposes only, not for diagnosis.
        """
        
        try:
            # Use the Ollama Python library for chat with image
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system', 
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt_template,
                        'images': [img_bytes]
                    }
                ]
            )
            
            # Extract the response
            if response and 'message' in response and 'content' in response['message']:
                return response['message']['content']
            else:
                return "Sorry, I couldn't generate an explanation for this segmentation."
                
        except Exception as e:
            print(f"Error using Ollama Python API: {str(e)}")
            
            # Return a helpful error message
            return f"""
            Error generating explanation: {str(e)}
            
            Please check:
            1. That the Ollama service is running
            2. That the model '{self.model_name}' is installed
            3. That you have the latest version of the Ollama Python library
            
            You can install/update with: pip install -U ollama
            """