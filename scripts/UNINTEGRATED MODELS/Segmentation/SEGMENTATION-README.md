# Comparative Analysis of Breast Cancer Segmentation Models

This repository contains three distinct implementations of deep learning models for breast cancer segmentation in medical images, each with different approaches, architectures, and performance characteristics. These models represent an exploration of various techniques for addressing the challenging task of medical image segmentation, particularly in ultrasound and mammogram images.

## Models Overview

| Model                                                                                                                                                                    | Architecture                         | Dataset   | Best Dice Score | Training Time | Key Features                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ | --------- | --------------- | ------------- | ------------------------------------------------------------ |
| [Basic U-Net for Mammogram Segmentation](scripts/UNINTEGRATED%20MODELS/Segmentation/DISCARDED%20MODELS/N01_Unet/N01_UNET-README.md)                                         | Standard U-Net                       | CBIS-DDSM | 0.04            | ~3 hours      | Advanced data augmentation, focal loss                       |
| [GA-Optimized Attention U-Net](scripts/UNINTEGRATED%20MODELS/Segmentation/DISCARDED%20MODELS/N02.1_Metaheuristic_Attention_Unet/N02_METAHEURISTIC_ATTENTION_UNET-README.md) | Attention U-Net with GA optimization | BUSI      | 0.5625          | ~1 hours      | Genetic algorithm hyperparameter tuning, two-stage training  |
| [Standard Attention U-Net](scripts/UNINTEGRATED%20MODELS/Segmentation/FINAL%20MODEL/N02_Attention_Unet/N02_ATTENTION_UNET-README.md)                                        | Attention U-Net                      | BUSI      | 0.728           | ~5 hours     | Attention gates, progressive dropout, combined BCE-Dice loss |

## Key Differences and Findings

### Model Architecture Evolution

Our exploration began with a standard **U-Net architecture** applied to mammogram images, which provided limited results (Dice coefficient of 0.04). This highlighted the significant challenges in segmenting mammographic images, which have lower contrast and more complex tissue patterns than ultrasound images.

We then progressed to an **Attention U-Net architecture** for ultrasound images (BUSI dataset), which showed dramatic improvements. The addition of attention gates allowed the model to focus on relevant regions, addressing the key challenge of localizing small lesions within noisy ultrasound images.

We explored two variants of the Attention U-Net:

1. A **genetically optimized implementation** that prioritized training efficiency
2. A **standard implementation** that prioritized segmentation accuracy

### Performance Comparison

#### Basic U-Net (Mammogram Dataset)

- **Dice Score**: 0.04 (very low)
- **Key Challenges**: Extreme class imbalance, limited contextual information, complex tissue patterns
- **Main Insights**: Standard U-Net architectures struggle with mammogram segmentation; more specialized approaches or preprocessing may be required

#### GA-Optimized Attention U-Net (Ultrasound Dataset)

- **Dice Score**: 0.5625
- **Training Time**: ~1 hours (40% faster than standard implementation)
- **Optimized Hyperparameters**: Progressive dropout rates (0.07, 0.32, 0.41, 0.24), batch size of 4
- **Key Advantage**: Significantly faster training with acceptable performance
- **Use Case**: Resource-constrained environments, rapid prototyping, mobile applications

#### Standard Attention U-Net (Ultrasound Dataset)

- **Dice Score**: 0.728 (29.4% higher than GA version)
- **Training Time**: ~5 hours
- **Key Advantage**: Superior segmentation quality with more stable training
- **Use Case**: Clinical applications where segmentation accuracy is critical

### Training Strategies

Each model implementation explored different training strategies:

- **Basic U-Net**: Utilized focal loss to address class imbalance and lesion-focused augmentation
- **GA-Optimized Attention U-Net**: Employed genetic algorithm for hyperparameter optimization and a two-stage training process (main training followed by fine-tuning)
- **Standard Attention U-Net**: Used a straightforward training approach with progressive dropout and combined BCE-Dice loss

### Dataset Impact

A crucial finding from our exploration is the significant impact of the dataset choice on model performance:

- **CBIS-DDSM (Mammogram Dataset)**: Presented extreme challenges for segmentation models with very low performance
- **BUSI (Ultrasound Dataset)**: Allowed for much higher segmentation accuracy, likely due to better contrast between lesions and background tissue

## When to Use Each Model

### Basic U-Net for Mammograms

- Use for: Initial exploration of mammogram segmentation
- Not recommended for: Production applications requiring accurate segmentation
- Further research needed: Domain-specific architectures, advanced preprocessing, multi-stage approaches

### GA-Optimized Attention U-Net

- Use for: Resource-constrained environments, rapid prototyping, applications where training time is critical
- Best suited for: Edge devices, mobile applications, frequent retraining scenarios
- Trade-off: 40% faster training at the cost of ~16% lower Dice score

### Standard Attention U-Net

- Use for: Clinical applications where segmentation accuracy is paramount
- Best suited for: Diagnostic support systems, research applications
- Selected as the final model for integration into the BreastCareAI system

## Technical Implementation Highlights

### Attention Mechanism

Both Attention U-Net implementations share a similar attention mechanism:

```python
class SimpleAttention(nn.Module):
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

This mechanism generates spatial attention maps to help the model focus on relevant features and suppress irrelevant background information.

### Genetic Algorithm Optimization

The GA-optimized model uses the DEAP library to explore the hyperparameter space, including:

- Learning rate
- Weight decay
- Dropout rates (four different rates for encoder/decoder blocks)
- Batch size

This approach led to an interesting pattern of progressive dropout rates (0.07, 0.32, 0.41, 0.24) that outperformed uniform dropout strategies.

### Data Augmentation

The Basic U-Net model implements sophisticated data augmentation strategies, including:

- Basic transforms: horizontal/vertical flips, rotations, contrast adjustment
- Lesion-focused zooming: intelligent augmentation focused on regions with lesions

## Conclusion and Recommendations

Based on our exploration, we have selected the **Standard Attention U-Net** as the final model for integration into the BreastCareAI system due to its superior segmentation accuracy (Dice coefficient of 0.728).

For scenarios where computational efficiency is more critical than absolute segmentation accuracy, the **GA-Optimized Attention U-Net** provides a compelling alternative, offering a 40% reduction in training time with a reasonable performance trade-off.

Our experiments with mammogram segmentation using the **Basic U-Net** highlight the significant challenges in this domain and suggest that more specialized approaches are needed for effective mammogram segmentation.

## Future Directions

- Explore transformer-based architectures for further improvements in segmentation accuracy
- Investigate multi-task learning approaches combining segmentation with classification
- Develop ensemble methods combining multiple model outputs
- Extend to 3D ultrasound volumes for volumetric segmentation
- Explore additional modalities and fusion strategies

---

For detailed information about each model implementation, please refer to the individual README files linked in the overview table above.
