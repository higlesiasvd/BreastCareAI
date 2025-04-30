# Breast Ultrasound Segmentation Module

## Introduction

This document provides a comprehensive explanation of the Breast Ultrasound Segmentation Module, which implements advanced deep learning techniques to automatically segment breast lesions in ultrasound images. The module features an Attention U-Net architecture, robust evaluation metrics, and explainability components to provide clinicians with transparent, interpretable results.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Neural Network Components](#neural-network-components)
3. [BreastSegmentationModel Class](#breastsegmentationmodel-class)
4. [Image Segmentation Pipeline](#image-segmentation-pipeline)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Model Explainability](#model-explainability)
7. [Integration and Deployment](#integration-and-deployment)
8. [Technical Documentation](#technical-documentation)

## Architecture Overview

The Breast Ultrasound Segmentation Module implements an enhanced U-Net architecture with attention mechanisms specifically designed for breast ultrasound image segmentation. The architecture consists of:

1. **Encoder Path**: Progressively downsamples the input image while increasing feature depth
2. **Bottleneck**: Captures the most abstract representations of the input
3. **Decoder Path**: Progressively upsamples to restore spatial resolution
4. **Skip Connections with Attention**: Allows the model to focus on relevant features from earlier layers
5. **Output Layer**: Generates a probability map for each pixel (lesion vs. background)

This architecture is particularly effective for medical image segmentation as it can handle images with limited training data and preserves fine details critical for accurate lesion boundary delineation.

## Neural Network Components

### Convolutional Block

The `ConvBlock` forms the basic building block of the network and implements a sequence of:
- Two 3x3 convolutional layers
- Batch normalization for training stability
- ReLU activation functions
- Dropout for regularization

```python
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
```

The dropout rates gradually increase deeper into the network (from 0.07 to 0.11) to prevent overfitting in the more specialized feature layers.

### SimpleAttention

The `SimpleAttention` module implements a lightweight attention mechanism that helps the model focus on relevant features:

```python
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
```

This attention mechanism:
1. Generates a spatial attention map via 1x1 convolution
2. Applies sigmoid activation to produce values between 0-1
3. Multiplies the original feature map with the attention map
4. This highlights important regions and suppresses irrelevant ones

### Encoder and Decoder Blocks

The `Encoder` block performs feature extraction and downsampling:

```python
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
```

The `Decoder` block performs upsampling with attention-enhanced skip connections:

```python
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
```

The key innovation in the decoder block is applying attention to the skip connections, which helps the model selectively use the most relevant features from earlier layers.

### Attention U-Net Architecture

The complete architecture is implemented in the `AttentionUnet` class:

```python
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
```

The model features:
- Four encoding stages with increasing channel depth (64→128→256→512)
- A central bottleneck with 1024 channels
- Four decoding stages with attention-enhanced skip connections
- A final 1x1 convolutional layer with sigmoid activation

## BreastSegmentationModel Class

The `BreastSegmentationModel` class provides a high-level interface for using the model for inference and visualization. It handles:

1. Model loading and initialization
2. Image preprocessing
3. Segmentation prediction
4. Result visualization and evaluation
5. Explainability features

### Initialization

```python
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
```

The initialization is designed to be robust, handling different checkpoint formats and providing graceful degradation (empty predictions) if model loading fails.

### Image Preprocessing

```python
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
```

The preprocessing pipeline:
1. Resizes input images to 128x128 pixels
2. Normalizes pixel values to the range [0,1]
3. Adds batch and channel dimensions for the model
4. Transfers the tensor to the appropriate device (CPU/GPU)

### Segmentation Prediction

```python
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
```

The prediction function:
1. Handles the case where the model failed to load by returning empty predictions
2. Preprocesses the input image
3. Performs inference with gradient calculation disabled for efficiency
4. Returns both the binary mask (using a configurable threshold) and the probability map

### Visualization

The class provides methods for visualizing the segmentation results:

```python
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
```

This method creates a visually informative overlay of the segmentation result on the original image, with configurable color and transparency.

## Evaluation Metrics

The module implements comprehensive evaluation metrics for assessing segmentation quality:

```python
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
```

The `evaluate_prediction` method automatically finds ground truth masks when available and calculates multiple metrics:

```python
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
        # Search for ground truth mask using various strategies...
    
    # Calculate area ratio (percentage of image segmented)
    area_ratio = pred_mask.sum() / pred_mask.size
    metrics['area_ratio'] = area_ratio
    
    # Check if this is likely a normal image (no segmentation)
    is_normal_image = area_ratio < 0.01  # Less than 1% segmented is likely normal
    
    # Calculate metrics if we have a ground truth mask
    if true_mask is not None:
        # Calculate Dice coefficient, IoU, sensitivity, specificity...
        metrics['dice'] = dice
        metrics['iou'] = iou
        metrics['sensitivity'] = sensitivity
        metrics['specificity'] = specificity
    
    return metrics
```

This method is particularly sophisticated in how it handles different scenarios:
1. Automatically searches for ground truth masks in standard locations
2. Resizes masks to match if dimensions differ
3. Properly handles the "normal case" (images without lesions)
4. Calculates multiple complementary metrics (Dice, IoU, sensitivity, specificity)

## Model Explainability

A key strength of this module is its focus on explainability, implemented through several methods:

### Gradient-Based Attention Visualization (Grad-CAM)

```python
def get_activation_gradients(self, image, target_layer=None):
    """Compute activation gradients using the GradCAM approach"""
    # Implementation details...

def generate_gradcam(self, image, threshold=0.5):
    """Generate Grad-CAM visualization for model explainability"""
    # Implementation details...
```

The Grad-CAM implementation:
1. Hooks into the forward and backward passes of the network
2. Captures activations and gradients from a target layer
3. Computes weights by averaging gradients
4. Creates a heatmap showing which regions influenced the prediction
5. Includes robust fallback mechanisms when gradient calculation fails

### Comprehensive Visual Explanation

```python
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
```

This method creates a comprehensive 4-panel visualization showing:
1. The original ultrasound image
2. The segmentation result overlaid on the image
3. The regions the model focused on (Grad-CAM)
4. The probability map with a color scale

### Feature Importance Analysis

```python
def analyze_feature_importance(self, image, mask):
    """Analyze which image features most influenced the segmentation result"""
    # Generate grad-cam for feature importance
    _, attention_map = self.generate_gradcam(image)
    
    # Calculate various metrics on the attention map
    importance_stats = {
        'max_importance': float(np.max(attention_map)),
        'mean_importance': float(np.mean(attention_map)),
        'importance_entropy': float(entropy),  
        'normalized_entropy': float(normalized_entropy),
        'importance_top10_percent': float(np.percentile(attention_map[attention_map > 0], 90)),
        'importance_contiguity': self._calculate_contiguity(attention_map > 0.5)
    }
    
    # Identify whether model focuses on boundaries or interiors
    boundary_focus = self._calculate_boundary_focus(mask, attention_map)
    importance_stats['boundary_focus'] = boundary_focus
    
    return importance_stats
```

This method calculates quantitative metrics that describe how the model attends to the image:
1. **Max/Mean Importance**: The intensity of attention
2. **Entropy**: How diffuse vs. focused the attention is
3. **Contiguity**: Whether attention is concentrated in a single region
4. **Boundary Focus**: Whether the model emphasizes boundaries or interiors

These metrics provide insights into the model's decision-making process and can help clinicians understand when to trust or question the segmentation results.

## Integration and Deployment

The module is designed to be easily integrated into clinical workflows and applications:

1. **Streaming Inference**: The model processes one image at a time, suitable for real-time applications
2. **Graceful Degradation**: Provides fallback mechanisms when model loading fails
3. **Platform Agnostic**: Works on both CPU and GPU environments
4. **Lightweight Dependencies**: Core requirements are PyTorch, NumPy, OpenCV, and PIL

### Example Usage

```python
from PIL import Image
from breast_segmentation_module import BreastSegmentationModel

# Initialize model
model = BreastSegmentationModel(model_path="breast_segmentation_model.pth")

# Load image
image = Image.open("ultrasound.jpg").convert('L')  # Grayscale

# Generate segmentation
mask, prob_map = model.predict(image, threshold=0.5)

# Create visualization
overlay = model.overlay_mask(image, mask)
overlay.save("segmentation_result.png")

# Generate explanation
explanation = model.explain_segmentation_result(image, mask, prob_map)
explanation.save("segmentation_explanation.png")

# Calculate metrics
metrics = model.evaluate_prediction(mask)
print(f"Area coverage: {metrics['area_ratio']*100:.1f}%")
```

## Technical Documentation

### Core Methods

| Method | Description |
|--------|-------------|
| `__init__(model_path, device=None)` | Initialize model with pre-trained weights |
| `preprocess_image(image)` | Prepare image for inference |
| `predict(image, threshold=0.5)` | Generate segmentation mask and probability map |
| `overlay_mask(image, mask, alpha=0.5, color=[255,0,0])` | Overlay segmentation on original image |
| `evaluate_prediction(pred_mask, true_mask=None)` | Calculate evaluation metrics |
| `generate_gradcam(image, threshold=0.5)` | Generate attention visualization |
| `explain_segmentation_result(image, mask, prob_map)` | Create comprehensive explanation |
| `analyze_feature_importance(image, mask)` | Quantify model attention patterns |

### Helper Methods

| Method | Description |
|--------|-------------|
| `dice_coefficient(mask1, mask2)` | Calculate overlap between masks |
| `_calculate_contiguity(mask)` | Measure spatial cohesion of attention |
| `_calculate_boundary_focus(mask, attention_map)` | Determine focus on boundaries vs. interiors |
| `get_activation_gradients(image, target_layer)` | Extract internal model activations and gradients |

### Model Architecture

| Component | Layers | Parameters |
|-----------|--------|------------|
| Encoder 1 | 2 Conv + MaxPool | 64 filters, 0.07 dropout |
| Encoder 2 | 2 Conv + MaxPool | 128 filters, 0.08 dropout |
| Encoder 3 | 2 Conv + MaxPool | 256 filters, 0.09 dropout | 
| Encoder 4 | 2 Conv + MaxPool | 512 filters, 0.10 dropout |
| Bottleneck | 2 Conv | 1024 filters, 0.11 dropout |
| Decoder 1 | TransConv + Attention + 2 Conv | 512 filters, 0.10 dropout |
| Decoder 2 | TransConv + Attention + 2 Conv | 256 filters, 0.09 dropout |
| Decoder 3 | TransConv + Attention + 2 Conv | 128 filters, 0.08 dropout |
| Decoder 4 | TransConv + Attention + 2 Conv | 64 filters, 0.07 dropout |
| Output | 1x1 Conv + Sigmoid | 1 filter |

This architecture contains approximately 31 million trainable parameters and has been optimized specifically for breast ultrasound segmentation tasks.

---

This breast ultrasound segmentation module represents a state-of-the-art approach to lesion segmentation, combining deep learning with explainability features to enhance clinical decision-making. Its attention-based architecture and comprehensive evaluation metrics make it particularly suitable for integration into computer-aided diagnosis systems for breast cancer.
