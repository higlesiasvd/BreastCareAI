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
import os

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
    
    def evaluate_prediction(self, pred_mask, true_mask=None, image_path=None, image_name=None):
        """
        Evaluate the quality of a segmentation prediction
        
        Args:
            pred_mask (np.ndarray): Predicted binary mask
            true_mask (np.ndarray, optional): Ground truth mask for comparison
            image_path (str, optional): Path to the original image
            image_name (str, optional): Name of the original image file
                                             
        Returns:
            dict: Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Try to find a ground truth mask if not provided
        if true_mask is None:
            # First, check standard naming convention (adding _mask suffix)
            if image_path is not None:
                try:
                    path_parts = image_path.rsplit('.', 1)
                    if len(path_parts) == 2:
                        mask_path = f"{path_parts[0]}_mask.{path_parts[1]}"
                        if os.path.exists(mask_path):
                            try:
                                true_mask = np.array(Image.open(mask_path).convert('L'))
                                true_mask = (true_mask > 127).astype(np.uint8)
                                metrics['mask_source'] = mask_path
                                print(f"Found mask using standard naming: {mask_path}")
                            except Exception as e:
                                print(f"Error loading standard mask: {str(e)}")
                except Exception as e:
                    print(f"Error creating standard mask path: {str(e)}")
            
            # If that didn't work, try direct dataset lookup in BUSI dataset path
            if true_mask is None and (image_name is not None or image_path is not None):
                try:
                    # Determine image name from path if not provided
                    if image_name is None and image_path is not None:
                        image_name = os.path.basename(image_path)
                    
                    if image_name:
                        # Try to find mask in BUSI dataset
                        busi_dataset_path = "/Volumes/Proyecto_Hugo/breast-cancer-analysis/datasets/Dataset_BUSI_with_GT"
                        
                        # Look through subdirectories for matching mask
                        for subdir in ['benign', 'malignant', 'normal']:
                            full_dir = os.path.join(busi_dataset_path, subdir)
                            if not os.path.exists(full_dir):
                                continue
                                
                            for root, dirs, files in os.walk(full_dir):
                                for file in files:
                                    # Match image name with mask file naming pattern
                                    if file.endswith('_mask.png') or file.endswith('_mask.jpg'):
                                        # Extract base name without _mask for comparison
                                        base_name = file.rsplit('_mask', 1)[0]
                                        img_base_name = image_name.rsplit('.', 1)[0]
                                        
                                        if base_name == img_base_name or base_name in img_base_name:
                                            mask_path = os.path.join(root, file)
                                            try:
                                                true_mask = np.array(Image.open(mask_path).convert('L'))
                                                true_mask = (true_mask > 127).astype(np.uint8)
                                                metrics['mask_source'] = mask_path
                                                print(f"Found mask in BUSI dataset: {mask_path}")
                                                break
                                            except Exception as e:
                                                print(f"Error loading BUSI mask: {str(e)}")
                                                
                                if true_mask is not None:
                                    break
                            if true_mask is not None:
                                break
                except Exception as e:
                    print(f"Error searching BUSI dataset: {str(e)}")
        
        # Calculate area ratio (percentage of image segmented)
        area_ratio = pred_mask.sum() / pred_mask.size
        metrics['area_ratio'] = area_ratio
        
        # Check if this is likely a normal image (no segmentation)
        is_normal_image = area_ratio < 0.01  # Less than 1% segmented is likely normal
        
        # Calculate metrics if we have a ground truth mask
        if true_mask is not None:
            # Resize true mask to match prediction if needed
            if pred_mask.shape != true_mask.shape:
                # Convert to same size
                from skimage.transform import resize
                true_mask = resize(true_mask, pred_mask.shape, order=0, preserve_range=True).astype(pred_mask.dtype)
            
            # Check if ground truth is also empty/nearly empty (normal case)
            true_mask_ratio = true_mask.sum() / true_mask.size
            gt_is_normal = true_mask_ratio < 0.01
            
            # Calculate Dice coefficient
            if is_normal_image and gt_is_normal:
                # Both prediction and ground truth are normal (no lesions)
                # In this case, Dice should be 1.0 (perfect agreement on "no lesion")
                dice = 1.0
                iou = 1.0
                sensitivity = 1.0  # All true positives were correctly identified (none in this case)
                specificity = 1.0  # All true negatives were correctly identified (entire image)
            else:
                # Calculate standard metrics when at least one has segmentation
                # Calculate Dice coefficient
                dice = self.dice_coefficient(pred_mask, true_mask)
                
                # Calculate Intersection over Union (IoU) / Jaccard Index
                intersection = np.logical_and(pred_mask, true_mask).sum()
                union = np.logical_or(pred_mask, true_mask).sum()
                iou = intersection / union if union > 0 else 0.0
                
                # Calculate sensitivity (recall) and specificity
                true_positive = np.logical_and(pred_mask, true_mask).sum()
                false_positive = np.logical_and(pred_mask, np.logical_not(true_mask)).sum()
                false_negative = np.logical_and(np.logical_not(pred_mask), true_mask).sum()
                true_negative = np.logical_and(np.logical_not(pred_mask), np.logical_not(true_mask)).sum()
                
                sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
                specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0.0
            
            metrics['dice'] = dice
            metrics['iou'] = iou
            metrics['sensitivity'] = sensitivity
            metrics['specificity'] = specificity
            
            # Add flag for normal case identification
            if is_normal_image and gt_is_normal:
                metrics['normal_case'] = True
        
        return metrics
    def get_activation_gradients(self, image, target_layer=None):
        """Compute activation gradients using the GradCAM approach"""
        if self.model is None:
            return None, None
            
        # Preprocess image
        img_tensor = self.preprocess_image(image)
        img_tensor.requires_grad = True  # Ensure gradients are enabled
        
        # Store activations and gradients
        activations = {}
        gradients = {}
        
        # Hook to capture activations during forward pass
        def forward_hook(module, input, output):
            activations['value'] = output
        
        # Hook to capture gradients during backward pass
        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]
        
        # Automatically determine target layer if not provided
        if target_layer is None:
            # For a U-Net, the last encoder or first decoder usually holds the most relevant information
            target_layer_options = ['decoder_4', 'decoder_1', 'conv_block']
        else:
            target_layer_options = [target_layer]
        
        # Try to find a valid layer
        target_module = None
        used_layer_name = None
        
        for layer_name in target_layer_options:
            for name, module in self.model.named_modules():
                if layer_name in name:
                    target_module = module
                    used_layer_name = name
                    break
            if target_module is not None:
                break
        
        if target_module is None:
            # Fallback: use any convolutional layer near the output
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d) and 'decoder' in name:
                    target_module = module
                    used_layer_name = name
                    break
        
        if target_module is None:
            print("Could not find a suitable convolutional layer for Grad-CAM")
            return None, None
        
        print(f"Using layer '{used_layer_name}' for Grad-CAM")
        
        # Register hooks
        handle_forward = target_module.register_forward_hook(forward_hook)
        handle_backward = target_module.register_full_backward_hook(backward_hook)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(img_tensor)
        
        # Create a more effective target for backpropagation
        # Aim to maximize regions predicted as part of the segment
        mask_pred = output.squeeze()
        
        # Create a target tensor to encourage positive predictions
        target = torch.ones_like(output)
        
        # Backward pass
        if output.requires_grad:
            output.backward(gradient=target)
        else:
            print("Output did not have requires_grad, recreating with gradients enabled")
            self.model.zero_grad()
            
            # Remove hooks
            handle_forward.remove()
            handle_backward.remove()
            
            # Recreate tensors with requires_grad=True
            img_tensor = self.preprocess_image(image)
            img_tensor.requires_grad = True
            
            # Re-register hooks
            handle_forward = target_module.register_forward_hook(forward_hook)
            handle_backward = target_module.register_full_backward_hook(backward_hook)
            
            # Forward and backward pass again
            output = self.model(img_tensor)
            output.backward(gradient=target)
        
        # Remove hooks
        handle_forward.remove()
        handle_backward.remove()
        
        # If we still get null values, fallback to simulated activation
        if 'value' not in activations or 'value' not in gradients:
            print("Gradients or activations are missing, using fallback approach")
            mask, prob_map = self.predict(image)
            
            pseudo_activation = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
            pseudo_gradient = torch.ones_like(pseudo_activation)
            
            return pseudo_activation, pseudo_gradient
        
        # Check if the returned tensors are not zero
        act_value = activations['value']
        grad_value = gradients['value']
        
        if torch.sum(act_value) == 0 or torch.sum(grad_value) == 0:
            print(f"Warning: activation or gradient has zero sum (act: {torch.sum(act_value).item()}, grad: {torch.sum(grad_value).item()})")
            mask, prob_map = self.predict(image)
            pseudo_activation = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0)
            pseudo_gradient = torch.ones_like(pseudo_activation)
            return pseudo_activation, pseudo_gradient
        
        return activations['value'], gradients['value']

    def generate_gradcam(self, image, threshold=0.5):
        """Generate Grad-CAM visualization for model explainability"""
        # Get activations and gradients
        activations, gradients = self.get_activation_gradients(image)
        
        if activations is None or gradients is None:
            print("Could not obtain activations or gradients, using probability map instead.")
            _, prob_map = self.predict(image, threshold=threshold)

            import cv2
            prob_map_resized = cv2.resize(prob_map, (image.width, image.height))
            heatmap = (prob_map_resized * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            img_array = np.array(image.convert('RGB'))
            img_cv = img_array[:, :, ::-1]  # Convert from RGB to BGR (OpenCV format)

            # Make sure both heatmap and image have the same dimensions
            if len(heatmap_colored.shape) == 2:
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2BGR)
            # Resize heatmap_colored if needed
            if heatmap_colored.shape[:2] != img_cv.shape[:2]:
                heatmap_colored = cv2.resize(heatmap_colored, (img_cv.shape[1], img_cv.shape[0]))

            # Now both are (height, width, 3) and uint8
            try:
                overlay = cv2.addWeighted(img_cv, 0.7, heatmap_colored, 0.3, 0)
            except cv2.error as e:
                print(f"Overlay error: {e}")
                overlay = img_cv.copy()

            # Try to overlay the heatmap on the image
            try:
                overlay = cv2.addWeighted(img_cv, 0.7, heatmap_colored, 0.3, 0)
            except cv2.error as e:
                print(f"Overlay error: {e}")
                overlay = img_cv.copy()

            overlay_rgb = overlay[:, :, ::-1]  # Convert back to RGB
            overlay_pil = Image.fromarray(overlay_rgb)

            return overlay_pil, prob_map_resized
        
        # Check if tensors contain meaningful values
        if torch.sum(activations) == 0:
            print("Warning: All activations are zero")
        if torch.sum(gradients) == 0:
            print("Warning: All gradients are zero")
        
        # Compute weights - global average of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Apply weights to activations
        cam = torch.sum(weights * activations, dim=1).squeeze()
        
        # Apply ReLU to focus on positively influential features
        cam = torch.nn.functional.relu(cam)
        
        # Normalize between 0 and 1
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
        else:
            print("Warning: CAM max is zero or negative")
            _, prob_map = self.predict(image, threshold=threshold)
            
            import cv2
            heatmap = (prob_map * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            img_array = np.array(image.convert('RGB'))
            img_cv = img_array[:, :, ::-1]  # RGB to BGR
            
            # Convert grayscale to 3-channel color if needed
            if len(heatmap_colored.shape) == 2 or heatmap_colored.shape[2] == 1:
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_GRAY2BGR)

            # Resize if dimensions don't match
            if heatmap_colored.shape[:2] != img_cv.shape[:2]:
                heatmap_colored = cv2.resize(heatmap_colored, (img_cv.shape[1], img_cv.shape[0]))
                
            overlay = cv2.addWeighted(img_cv, 0.7, heatmap_colored, 0.3, 0)
            overlay_rgb = overlay[:, :, ::-1]  # BGR to RGB
            overlay_pil = Image.fromarray(overlay_rgb)
            
            return overlay_pil, prob_map
        
        # Resize CAM to match original image size
        import torch.nn.functional as F
        cam = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        cam = F.interpolate(cam, size=(image.height, image.width), mode='bicubic', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = np.clip(cam, 0.0, 1.0)
        
        # Print stats for debugging
        print(f"CAM stats - min: {np.min(cam)}, max: {np.max(cam)}, mean: {np.mean(cam)}")
        
        # Create heatmap overlay
        import cv2
        heatmap = (cam * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_array = np.array(image.convert('RGB'))
        img_cv = img_array[:, :, ::-1]  # Convert from RGB to BGR

        # Make sure both images are the same size
        if heatmap_colored.shape[:2] != img_cv.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (img_cv.shape[1], img_cv.shape[0]))

        try:
            overlay = cv2.addWeighted(img_cv, 0.7, heatmap_colored, 0.3, 0)
        except cv2.error as e:
            print(f"Overlay error: {e}, using original image")
            overlay = img_cv.copy()

        overlay_rgb = overlay[:, :, ::-1]  # Convert back to RGB
        overlay_pil = Image.fromarray(overlay_rgb)

        return overlay_pil, cam

    def explain_segmentation_result(self, image, mask, prob_map):
        """Generate visual explanation of segmentation focusing on model behavior"""
        # Generate Grad-CAM to show model attention
        cam_overlay, attention_map = self.generate_gradcam(image)
        
        # Create a multi-panel explanation
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axs[0, 0].imshow(image, cmap='gray')
        axs[0, 0].set_title('Original Image')
        axs[0, 0].axis('off')
        
        # Segmentation overlay
        overlay = self.overlay_mask(image, mask)
        axs[0, 1].imshow(overlay)
        axs[0, 1].set_title('Segmentation Result')
        axs[0, 1].axis('off')
        
        # Grad-CAM visualization
        axs[1, 0].imshow(cam_overlay)
        axs[1, 0].set_title('Model Attention (Grad-CAM)')
        axs[1, 0].axis('off')
        
        # Probability map
        axs[1, 1].imshow(prob_map, cmap='viridis')
        axs[1, 1].set_title('Probability Map')
        axs[1, 1].axis('off')
        plt.colorbar(axs[1, 1].imshow(prob_map, cmap='viridis'), ax=axs[1, 1])
        
        # Add overall title
        plt.suptitle('Segmentation Explanation', fontsize=16)
        plt.tight_layout()
        
        # Save to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Load image from buffer
        explanation_img = Image.open(buf)
        
        return explanation_img

    def analyze_feature_importance(self, image, mask):
        """Analyze which image features most influenced the segmentation result"""
        # Generate grad-cam for feature importance
        _, attention_map = self.generate_gradcam(image)
        
        # Ensure positive values to avoid issues with the logarithm
        attention_map_safe = np.clip(attention_map, 1e-10, 1.0)
        
        if np.sum(attention_map_safe) > 0:
            attention_map_normalized = attention_map_safe / np.sum(attention_map_safe)
        else:
            attention_map_normalized = np.ones_like(attention_map_safe) / attention_map_safe.size
        
        # Calculate entropy correctly over the normalized distribution
        entropy = -np.sum(attention_map_normalized * np.log2(attention_map_normalized))
        
        # Limit entropy to a reasonable range (the theoretical maximum would be log2(N) where N is the number of pixels)
        max_theoretical_entropy = np.log2(attention_map.size)
        normalized_entropy = entropy / max_theoretical_entropy  # Normalized between 0-1
        
        importance_stats = {
            'max_importance': float(np.max(attention_map)),
            'mean_importance': float(np.mean(attention_map)),
            'importance_entropy': float(entropy),  
            'normalized_entropy': float(normalized_entropy),  # Normalized version for better interpretation (0-1)
            'importance_top10_percent': float(np.percentile(attention_map[attention_map > 0], 90) if np.any(attention_map > 0) else 0.0),
            'importance_contiguity': self._calculate_contiguity(attention_map > 0.5)
        }
        
        # Identify whether model focuses on boundaries or interiors
        boundary_focus = self._calculate_boundary_focus(mask, attention_map)
        importance_stats['boundary_focus'] = boundary_focus
        
        return importance_stats

    def _calculate_contiguity(self, mask):
        """Calculate how contiguous a mask is (higher = more contiguous)"""
        from scipy import ndimage
        if not np.any(mask):
            return 0
            
        # Label connected components
        labels, num = ndimage.label(mask)
        
        # Calculate size of each component
        component_sizes = np.bincount(labels.flatten())[1:]
        
        # Calculate contiguity as percentage of largest component
        if len(component_sizes) > 0:
            return float(np.max(component_sizes) / np.sum(component_sizes))
        return 0.0

    def _calculate_boundary_focus(self, mask, attention_map):
        """Calculate whether model focuses more on boundaries (1) or interiors (0)"""
        from scipy import ndimage
        if not np.any(mask):
            return 0
            
        # Get boundary of mask
        boundary = ndimage.binary_dilation(mask) & ~mask
        
        # Calculate mean attention on boundary vs interior
        boundary_attention = np.mean(attention_map[boundary])
        interior_attention = np.mean(attention_map[mask])
        
        # Return normalized ratio (0 = interior focus, 1 = boundary focus)
        if interior_attention > 0:
            ratio = boundary_attention / (boundary_attention + interior_attention)
            return float(ratio)
        return 0.5  # Default if interior has no attention