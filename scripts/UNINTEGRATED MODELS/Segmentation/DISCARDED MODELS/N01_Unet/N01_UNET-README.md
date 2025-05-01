# Breast Cancer Segmentation with U-Net: CBIS-DDSM Dataset Implementation

## Overview

This project implements a U-Net architecture for breast lesion segmentation using the CBIS-DDSM (Curated Breast Imaging Subset of Digital Database for Screening Mammography) dataset. The goal is to automatically identify and segment regions of interest (ROIs) in mammogram images, which could assist radiologists in breast cancer detection and analysis.

Despite implementation of several advanced techniques, the model achieved limited performance (Dice coefficient of 0.04 on validation), highlighting the significant challenges associated with mammogram segmentation. This document details the implementation, challenges, and potential areas for improvement.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Model Architecture](#model-architecture)
3. [Implementation Details](#implementation-details)
4. [Data Augmentation](#data-augmentation)
5. [Training Process](#training-process)
6. [Loss Functions and Optimization](#loss-functions-and-optimization)
7. [Performance Evaluation](#performance-evaluation)
8. [Challenges and Limitations](#challenges-and-limitations)
9. [Potential Improvements](#potential-improvements)
10. [Usage Guide](#usage-guide)

## Dataset Description

The project utilizes the CBIS-DDSM dataset, which contains:

- Mammographic images from various screening examinations
- ROI mask images marking abnormalities (calcifications and masses)
- Associated patient and abnormality metadata

The dataset characteristics:
- High-resolution grayscale mammogram images
- Binary mask annotations for lesions
- Limited number of positive samples with significant class imbalance
- Includes both calcification and mass abnormalities
- Varying image quality and contrast
- Multiple views per patient (CC, MLO)

Data preprocessing steps included:
- Conversion to grayscale
- Resizing to a standard resolution (256×256 pixels)
- Normalization to [0,1] range
- Binary thresholding of masks
- Train/validation/test split (70%/10%/20%)

## Model Architecture

The segmentation model uses the U-Net architecture, which is particularly effective for biomedical image segmentation tasks:

### U-Net Architecture Details:

- **Encoder Path (Contracting)**:
  - Initial convolution to 64 channels
  - Four downsampling blocks with channel depths: 64→128→256→512→1024
  - Each block consists of two 3×3 convolutions with BatchNorm and ReLU
  - Max pooling with stride 2 for downsampling

- **Decoder Path (Expanding)**:
  - Four upsampling blocks
  - Each upsampling uses transposed convolution with stride 2
  - Skip connections from corresponding encoder layer
  - Two 3×3 convolutions with BatchNorm and ReLU after each concatenation
  - Gradual reduction of channels: 1024→512→256→128→64

- **Output Layer**:
  - 1×1 convolution to reduce to single channel
  - Sigmoid activation for binary segmentation output

### Building Blocks:

1. **DoubleConv**: Two consecutive (Conv2D → BatchNorm → ReLU) operations
2. **Down**: MaxPool2D followed by DoubleConv for encoder steps
3. **Up**: Transposed convolution followed by concatenation and DoubleConv
4. **OutConv**: Final 1×1 convolution layer

The model contains approximately 31 million parameters, making it a relatively large network for the dataset size.

## Implementation Details

The implementation utilizes PyTorch for deep learning operations:

- **Environment**:
  - PyTorch 2.x
  - Hardware acceleration using MPS (Metal Performance Shaders) for Apple Silicon or CUDA (NVIDIA) if available, with graceful fallback to CPU
  - Standard data science libraries (NumPy, Matplotlib, Pandas, OpenCV)

- **Custom Dataset Implementation**:
  - `MammogramDataset` class extending PyTorch's `Dataset`
  - On-the-fly loading and preprocessing of images
  - Enhanced version implemented with augmentation capabilities (`EnhancedMammogramDataset`)

- **Data Pipeline**:
  - Custom data loaders with configurable batch size
  - Robust error handling for corrupted images
  - Efficient memory management with image loading during iteration

### Enhanced Implementation:

The project includes an improved dataset implementation (`EnhancedMammogramDataset`) that adds:
- Robust error handling with fallback for corrupt images
- Extensive data augmentation techniques
- Smart augmentation focusing on regions with lesions
- Efficient image processing pipeline

## Data Augmentation

To address limited training data, several augmentation techniques were implemented:

### Basic Augmentations:
- **Horizontal & Vertical Flips**: 50% probability for each
- **Rotations**: Random 90° rotations (0°, 90°, 180°, 270°)
- **Contrast Adjustment**: Gamma correction with values between 0.8-1.2

### Advanced Augmentations:
- **Lesion-Focused Zooming**: 30% probability
  - Identifies lesion center
  - Creates a randomly sized crop centered on lesion
  - Resizes back to standard dimensions
  - Preserves lesion information while creating variation

### Implementation Details:
- Augmentations only applied during training
- Deterministic transformations for validation/test sets
- Consistent transformations applied to both image and mask
- Re-binarization of masks after geometric transformations
- Parameter-controlled augmentation intensity

## Training Process

The training process was implemented with careful monitoring and optimization:

### Training Setup:
- Batch size: 8
- Input image size: 256×256 pixels
- Epochs: 50 (with early stopping)
- Learning rate: Starting at 1e-3 with adaptive scheduling
- Hardware: MPS acceleration on Apple Silicon M4 Pro

### Enhanced Training Loop:
The `train_model_enhanced` function implements:
- Gradient clipping (max norm: 1.0)
- Early stopping with patience of 15 epochs
- Validation after each epoch
- Best model saving based on validation Dice score
- Detailed progress reporting
- Batch-level metrics for monitoring

### Learning Rate Schedule:
- Reduce-on-plateau scheduler
- Monitoring validation Dice coefficient
- Reduction factor: 0.5
- Patience: 7 epochs
- Minimum learning rate: 1e-6

## Loss Functions and Optimization

Multiple loss functions were explored to address the challenges of imbalanced segmentation:

### Loss Functions:
1. **Binary Cross-Entropy + Dice Loss**:
   - Combined BCE and Dice with weighting (initial implementation)
   - Balances pixel-wise accuracy and structural similarity

2. **Focal + Dice Loss** (Final implementation):
   - Focal Loss component addresses class imbalance by focusing on hard examples
   - Parameters: α=0.75, γ=2.0
   - Dice component ensures structural similarity
   - Weighted combination (60% Dice, 40% Focal)

### Optimization Strategy:
- **AdamW optimizer**:
  - Initial learning rate: 1e-3
  - Weight decay: 1e-4 for regularization
  - Beta parameters: Default (0.9, 0.999)

## Performance Evaluation

The model performance was evaluated using multiple metrics:

### Evaluation Metrics:
- **Dice Coefficient**: Primary metric, measures overlap between prediction and ground truth
- **Loss Curves**: Training and validation loss progression
- **Visual Assessment**: Qualitative analysis of segmentation outputs

### Results:
- **Final Dice Coefficient**: ~0.04 on validation set
- **Convergence**: Model converged but at a suboptimal performance level
- **Visual Analysis**: Segmentations showed limited correspondence with ground truth

### Performance Analysis:
- Low Dice coefficient indicates significant challenges in the segmentation task
- Loss curves showed learning but plateaued at suboptimal values
- Visual inspection confirmed limited segmentation quality

## Challenges and Limitations

Several factors contributed to the limited performance:

### Dataset Challenges:
1. **Extreme Class Imbalance**: Lesions typically occupy <1% of image area
2. **Limited Sample Size**: Relatively few examples for deep learning
3. **Image Variability**: Significant differences in image acquisition and quality
4. **Annotation Inconsistency**: Potential variations in mask creation
5. **Complex Tissue Patterns**: Breast tissue presents complex, variable patterns

### Technical Limitations:
1. **Resolution Constraints**: Downsampling from high-resolution images loses detail
2. **Limited Contextual Information**: U-Net receptive field may be insufficient
3. **Model Capacity**: Despite large parameter count, may need domain-specific design
4. **Binary Segmentation Challenges**: Sharp decision boundaries may be inappropriate
5. **Training Stability**: Observed oscillations in performance metrics

## Potential Improvements

Based on the results and challenges, several improvements could be explored:

### Model Architecture:
1. **Attention Mechanisms**: Add attention gates to focus on relevant features
2. **Pre-trained Encoders**: Utilize transfer learning with pre-trained backbones
3. **Multi-scale Processing**: Incorporate features at different resolutions
4. **Ensemble Approaches**: Combine multiple models for improved robustness

### Data Strategies:
1. **Advanced Preprocessing**: Contrast enhancement, noise reduction
2. **More Aggressive Augmentation**: Additional techniques like elastic deformations
3. **Synthetic Data Generation**: Use GANs to generate additional training samples
4. **Semi-supervised Approaches**: Utilize unlabeled mammograms

### Training Strategies:
1. **Curriculum Learning**: Start with easier examples
2. **Boundary-focused Loss**: Emphasize lesion boundaries in loss function
3. **Multi-task Learning**: Add auxiliary tasks like classification
4. **Progressive Resizing**: Train at multiple resolutions
5. **Two-stage Approach**: Detection followed by segmentation

## Usage Guide

### Environment Setup:
```python
# Required libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Select optimal device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
```

### Dataset Preparation:
```python
# Path configuration
dicom_data = pd.read_csv('/path/to/dicom_info.csv')
image_dir = '/path/to/jpeg_images'

# Image paths
cropped_images = dicom_data[dicom_data.SeriesDescription == 'cropped images'].image_path
ROI_mask_images = dicom_data[dicom_data.SeriesDescription == 'ROI mask images'].image_path

# Convert paths to absolute paths
cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))
ROI_mask_images = ROI_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', image_dir))

# Create data pairs and split
image_mask_pairs = list(zip(cropped_images, ROI_mask_images))
train_pairs, val_test_pairs = train_test_split(image_mask_pairs, test_size=0.3, random_state=42)
val_pairs, test_pairs = train_test_split(val_test_pairs, test_size=0.33, random_state=42)

# Extract separate lists
train_imgs, train_masks = zip(*train_pairs)
val_imgs, val_masks = zip(*val_pairs)
test_imgs, test_masks = zip(*test_pairs)
```

### Data Loading with Enhanced Dataset:
```python
# Configuration parameters
IMG_SIZE = 256
BATCH_SIZE = 8

# Enhanced dataset with augmentation
train_dataset = EnhancedMammogramDataset(train_imgs, train_masks, img_size=IMG_SIZE, is_train=True)
val_dataset = EnhancedMammogramDataset(val_imgs, val_masks, img_size=IMG_SIZE, is_train=False)
test_dataset = EnhancedMammogramDataset(test_imgs, test_masks, img_size=IMG_SIZE, is_train=False)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
```

### Model Training:
```python
# Initialize model
model = UNet(n_channels=1, n_classes=1)
model = model.to(device)

# Loss function and optimizer
criterion = FocalDiceLoss(alpha=0.75, gamma=2.0, dice_weight=0.6)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=7, factor=0.5, verbose=True)

# Train the model
trained_model, history = train_model_enhanced(
    model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50
)

# Evaluate on test set
final_dice = evaluate_model(trained_model, test_loader)
print(f"Final Dice coefficient on test set: {final_dice:.4f}")

# Save the model
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'test_dice': final_dice
}, 'unet_mammogram_final.pth')
```

### Visualization:
```python
# Plot training history
plot_training_history(history)

# Visualize predictions
visualize_predictions(trained_model, test_dataset, num_samples=5)
```

---

This implementation demonstrates the challenges of breast lesion segmentation in mammograms. While the achieved performance was limited (Dice coefficient of 0.04), the project provides a foundation for future improvements and highlights the need for specialized approaches when working with medical imaging data, particularly mammograms.

The low performance indicates that this specific task may require more specialized techniques beyond standard U-Net architectures, including domain-specific preprocessing, model architectures designed for extreme class imbalance, and potentially multi-stage approaches that combine detection and segmentation strategies.
