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
    
    def _create_combined_image(self, original_image, segmented_image, metrics):
        """
        Create a combined image with original, segmentation, and metrics
        
        Args:
            original_image (PIL.Image): Original ultrasound image
            segmented_image (PIL.Image): Segmented overlay image
            metrics (dict): Dictionary of segmentation metrics
            
        Returns:
            PIL.Image: Combined image for explanation
        """
        # Resize images to be the same height
        height = 400
        
        # Calculate widths while maintaining aspect ratio
        width1 = int(original_image.width * height / original_image.height)
        width2 = int(segmented_image.width * height / segmented_image.height)
        
        # Resize images
        original_resized = original_image.resize((width1, height))
        segmented_resized = segmented_image.resize((width2, height))
        
        # Create new image that fits both side by side plus a metrics panel
        metrics_width = 400
        new_width = width1 + width2 + metrics_width
        combined = Image.new('RGB', (new_width, height), (255, 255, 255))
        
        # Paste the images
        combined.paste(original_resized, (0, 0))
        combined.paste(segmented_resized, (width1, 0))
        
        # Create a drawing context
        draw = ImageDraw.Draw(combined)
        
        # Add text for metrics panel
        try:
            # Try to load a nice font, fall back to default
            font = ImageFont.truetype("Arial", 14)
        except:
            font = ImageFont.load_default()
        
        # Add metrics text
        x_offset = width1 + width2 + 20
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
        
        # Add a note for normal cases
        if 'normal_case' in metrics and metrics['normal_case']:
            y_offset += line_height/2
            draw.text((x_offset, y_offset), 
                      "Note: Both prediction and ground truth", fill=(0, 0, 0), font=font)
            y_offset += line_height
            draw.text((x_offset, y_offset), 
                      "show no lesions (normal case)", fill=(0, 0, 0), font=font)
        
        # Add titles to the images
        draw.text((width1//2 - 60, 5), "Original Ultrasound", fill=(255, 255, 255), font=font)
        draw.text((width1 + width2//2 - 80, 5), "Segmentation Overlay", fill=(255, 255, 255), font=font)
        
        return combined

    def explain_segmentation(self, original_image, segmented_image, metrics, prompt_template=None):
        """
        Generate an explanation of the segmentation results using Ollama
        
        Args:
            original_image (PIL.Image): Original ultrasound image
            segmented_image (PIL.Image): Segmented overlay image
            metrics (dict): Dictionary of segmentation metrics
            prompt_template (str, optional): Custom prompt template
            
        Returns:
            str: AI-generated explanation of the segmentation results
        """
        # Create combined image for explanation
        combined_image = self._create_combined_image(original_image, segmented_image, metrics)
        
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
            
            prompt_template = """
            Please analyze this breast ultrasound image and segmentation results:
            - On the left is the original ultrasound image
            - In the middle is the AI-generated segmentation overlay (red areas show detected tissue of interest)
            - On the right are the segmentation metrics
            
            Provide a comprehensive yet concise explanation including:
            1. What do you see in the ultrasound image?
            2. How good is the segmentation quality?
            3. What do the metrics tell us?
            
            Keep your explanation medically accurate but accessible to non-experts.
            Avoid making specific diagnostic claims or suggesting treatments.
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
                and how well the model detected areas of interest.
                """
        
        # System prompt to guide the model's responses
        system_prompt = """
        You are a medical imaging expert specialized in breast ultrasound analysis. You are examining 
        ultrasound images and their AI-generated segmentation results.
        
        IMPORTANT INSTRUCTIONS ABOUT YOUR RESPONSE FORMAT:
        - Use a natural, conversational tone - DO NOT use numbered steps or bullet points
        - Write as if you were explaining to a medical student or patient
        - Never use phrases like "Step 1:", "Step 2:", etc.
        - Keep paragraphs short and use a flowing narrative style
        - Explain technical concepts in simple but precise language
        - Use an empathetic, reassuring tone that conveys expertise
        
        Content to include (but not in a structured format):
        - Brief description of what you observe in the ultrasound image
        - Assessment of the segmentation quality in relation to the original image
        - Explanation of what the metrics mean in this specific case
        - For normal cases, explain why no significant segmentation is appropriate
        - For abnormal cases, describe the segmentation pattern objectively
        
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
            1. Ollama service is running
            2. The model '{self.model_name}' is installed
            3. You have the latest version of the Ollama Python library
            
            You can install/update with: pip install -U ollama
            """