"""
Breast Ultrasound Segmentation Module
-------------------------------------
This module provides functionality for segmenting breast tissue in ultrasound images
using a U-Net model with attention mechanisms.

The module includes:
1. The neural network architecture (AttentionUnet)
2. A BreastSegmentationModel class for inference
3. Utility functions for visualizing segmentation results
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2

# Define neural network components
class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and dropout"""
    def __init__(self, input_channel, out_channel, dropout):
        super(ConvBlock, self).__init__()
        self.conv2d_1 = nn.Conv2d(input_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_1 = nn.BatchNorm2d(out_channel)
        self.relu_1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2d_2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.batchnorm_2 = nn.BatchNorm2d(out_channel)
        self.relu_2 = nn.ReLU()
        
    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)
        x = self.dropout(x)
        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = self.relu_2(x)
        return x


class SimpleAttention(nn.Module):
    """Simple attention mechanism for focusing on relevant features"""
    def __init__(self, in_channels):
        super(SimpleAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate attention map
        attn = self.conv(x)
        attn = self.sigmoid(attn)
        # Apply attention to the input
        return x * attn


class Encoder(nn.Module):
    """Encoder block with maxpooling"""
    def __init__(self, input_channel, out_channel, dropout):
        super(Encoder, self).__init__()
        self.conv2d_1 = ConvBlock(input_channel, out_channel, dropout)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        conv_out = self.conv2d_1(x)
        pool_out = self.maxpool(conv_out)
        return conv_out, pool_out


class Decoder(nn.Module):
    """Decoder block with transposed convolution and attention"""
    def __init__(self, input_channel, output_channel, dropout):
        super(Decoder, self).__init__()
        self.conv_t = nn.ConvTranspose2d(input_channel, output_channel, stride=2, kernel_size=2)
        self.attention = SimpleAttention(output_channel)
        self.conv2d_1 = ConvBlock(output_channel*2, output_channel, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, skip):
        x = self.conv_t(x)
        # Apply attention to skip connection
        skip_with_attention = self.attention(skip)
        x = torch.cat([x, skip_with_attention], dim=1)
        x = self.dropout(x)
        x = self.conv2d_1(x)
        return x


class AttentionUnet(nn.Module):
    """U-Net architecture with attention mechanisms for breast ultrasound segmentation"""
    def __init__(self, input_channel=1):
        super().__init__()
        self.encoder_1 = Encoder(input_channel, 64, 0.07)
        self.encoder_2 = Encoder(64, 128, 0.08)
        self.encoder_3 = Encoder(128, 256, 0.09)
        self.encoder_4 = Encoder(256, 512, 0.1)
        self.conv_block = ConvBlock(512, 1024, 0.11)
        # Decoders with attention
        self.decoder_1 = Decoder(1024, 512, 0.1)
        self.decoder_2 = Decoder(512, 256, 0.09)
        self.decoder_3 = Decoder(256, 128, 0.08)
        self.decoder_4 = Decoder(128, 64, 0.07)
        self.cls = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.relu = nn.Sigmoid()
        
    def forward(self, x):
        """U-Net forward pass with skip connections"""
        # Encoder
        x1, p1 = self.encoder_1(x)
        x2, p2 = self.encoder_2(p1)
        x3, p3 = self.encoder_3(p2)
        x4, p4 = self.encoder_4(p3)
        
        # Bottleneck
        x5 = self.conv_block(p4)
        
        # Decoder
        x6 = self.decoder_1(x5, x4)
        x7 = self.decoder_2(x6, x3)
        x8 = self.decoder_3(x7, x2)
        x9 = self.decoder_4(x8, x1)
        
        # Final Layer
        x_final = self.cls(x9)
        x_final = self.relu(x_final)
        
        return x_final


class BreastSegmentationModel:
    """
    Breast ultrasound segmentation model for inference
    
    This class loads a pre-trained U-Net model with attention mechanisms
    and provides methods for segmenting breast ultrasound images.
    """
    def __init__(self, model_path, device=None):
        """
        Initialize the breast segmentation model
        
        Args:
            model_path (str): Path to the pre-trained model weights
            device (str, optional): Device to run the model on. Defaults to cuda if available.
        """
        # Determine device (use CUDA if available)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Create model
        self.model = AttentionUnet(input_channel=1)
        
        try:
            # Load model weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Try direct loading
                    self.model.load_state_dict(checkpoint)
            else:
                # Try direct loading
                self.model.load_state_dict(checkpoint)
                
            # Move model to device and set to evaluation mode
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Create a dummy model that returns empty predictions
            # This allows the app to continue even if the model fails to load
            self.model = None
            print(f"WARNING: Using fallback empty model")
            
        # Image preprocessing parameters
        self.image_size = 128
    
    def preprocess_image(self, image):
        """
        Preprocess an input image for the model
        
        Args:
            image (PIL.Image): Input ultrasound image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Resize image
        image = image.resize((self.image_size, self.image_size))
        
        # Convert to numpy array and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor.to(self.device)
    
    def predict(self, image, threshold=0.5):
        """
        Generate a segmentation mask for an ultrasound image
        
        Args:
            image (PIL.Image): Input ultrasound image
            threshold (float, optional): Threshold for binary segmentation. Defaults to 0.5.
            
        Returns:
            tuple: (binary_mask, probability_map) as numpy arrays
        """
        # If model failed to load, return empty predictions
        if self.model is None:
            # Create empty prediction with same dimensions as input
            width, height = image.size
            empty_prediction = np.zeros((height, width), dtype=np.uint8)
            return empty_prediction, empty_prediction
            
        # Preprocess the image
        img_tensor = self.preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Extract probability map
        prob_map = output.squeeze().cpu().numpy()
        
        # Create binary mask
        binary_mask = (prob_map > threshold).astype(np.uint8)
        
        return binary_mask, prob_map
        
    def generate_visualization(self, image, threshold=0.5, return_figure=False):
        """
        Generate a complete visualization of the segmentation results
        
        Args:
            image (PIL.Image): Input ultrasound image
            threshold (float, optional): Threshold for binary segmentation. Defaults to 0.5.
            return_figure (bool, optional): Whether to return the matplotlib figure
                                           instead of saving to buffer. Defaults to False.
            
        Returns:
            dict: Dictionary containing visualization elements
                - 'original': Original image
                - 'mask': Binary segmentation mask
                - 'overlay': Image with overlaid segmentation
                - 'prob_map': Probability map
                - 'metrics': Dictionary of evaluation metrics
        """
        # Generate prediction
        mask, prob_map = self.predict(image, threshold=threshold)
        
        # Calculate metrics
        metrics = self.evaluate_prediction(mask)
        
        # Create overlay
        overlay = self.overlay_mask(image, mask)
        
        # Convert to images for return
        result = {
            'original': image,
            'mask': Image.fromarray((mask * 255).astype(np.uint8)),
            'overlay': overlay,
            'prob_map': prob_map,
            'metrics': metrics
        }
        
        return result
    
    def overlay_mask(self, image, mask, alpha=0.5, color=[255, 0, 0]):
        """
        Overlay the segmentation mask on the original image
        
        Args:
            image (PIL.Image): Original ultrasound image
            mask (np.ndarray): Binary segmentation mask
            alpha (float, optional): Transparency of the overlay. Defaults to 0.5.
            color (list, optional): RGB color for the mask. Defaults to [255, 0, 0] (red).
            
        Returns:
            PIL.Image: Image with overlaid segmentation mask
        """
        # Resize original image to match mask size
        image_resized = image.resize((mask.shape[1], mask.shape[0]))
        
        # Convert PIL image to numpy array
        img_array = np.array(image_resized)
        
        # Check if grayscale, convert to RGB
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
            img_array = np.concatenate([img_array, img_array, img_array], axis=2)
            
        # Ensure uint8 type for OpenCV
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8)
        
        # Create colored mask
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        colored_mask[mask > 0] = color
        
        # Create overlay
        overlay = cv2.addWeighted(
            img_array, 1, 
            colored_mask, alpha, 
            0
        )
        
        # Convert back to PIL image
        return Image.fromarray(overlay)
    
    def dice_coefficient(self, mask1, mask2, smooth=1e-6):
        """
        Calculate Dice coefficient between two binary masks
        
        The Dice coefficient is a measure of overlap between two segmentations.
        It ranges from 0 (no overlap) to 1 (perfect overlap).
        
        Args:
            mask1 (np.ndarray): First binary mask
            mask2 (np.ndarray): Second binary mask
            smooth (float): Smoothing factor to avoid division by zero
            
        Returns:
            float: Dice coefficient value between 0 and 1
        """
        # Ensure masks are binary
        mask1_bin = mask1.astype(bool)
        mask2_bin = mask2.astype(bool)
        
        # Calculate intersection and union
        intersection = np.logical_and(mask1_bin, mask2_bin).sum()
        union = mask1_bin.sum() + mask2_bin.sum()
        
        # Calculate Dice coefficient
        if union == 0:
            return 0.0
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice
    
    def evaluate_prediction(self, pred_mask, true_mask=None):
        """
        Evaluate the quality of a segmentation prediction
        
        Args:
            pred_mask (np.ndarray): Predicted binary mask
            true_mask (np.ndarray, optional): Ground truth mask for comparison
                                             Default is None (no evaluation)
                                             
        Returns:
            dict: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Calculate metrics if we have a ground truth mask
        if true_mask is not None:
            # Resize true mask to match prediction if needed
            if pred_mask.shape != true_mask.shape:
                # Convert to same size
                from skimage.transform import resize
                true_mask = resize(true_mask, pred_mask.shape, order=0, preserve_range=True).astype(pred_mask.dtype)
            
            # Calculate Dice coefficient
            dice = self.dice_coefficient(pred_mask, true_mask)
            metrics['dice'] = dice
        
        # Calculate area ratio (percentage of image segmented)
        area_ratio = pred_mask.sum() / pred_mask.size
        metrics['area_ratio'] = area_ratio
        
        return metrics