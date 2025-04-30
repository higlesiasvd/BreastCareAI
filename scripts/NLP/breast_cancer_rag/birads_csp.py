import numpy as np
from skimage import measure, filters, morphology
import time
from typing import Dict, List, Tuple, Any, Callable, Optional, Union


class BIRADSClassifierCSP:
    """
    BI-RADS classification system based on weighted scoring of features
    identified in breast ultrasound images.
    
    This improved implementation uses a weighted scoring approach instead of rigid rules,
    aligning better with medical literature and reducing false positives.
    """
    
    def __init__(self):
        """
        Initialize the BI-RADS classifier with variables, scoring weights, and thresholds.
        I've designed this to balance sensitivity and specificity by using a continuous 
        scoring system rather than strict rule-based classification.
        """
        # Define variables for lesion features - only those reliably detectable in ultrasound
        self.variables = {
            'shape': None,          # Shape: regular, oval, irregular, etc.
            'margin': None,         # Margin: circumscribed, indistinct, spiculated, etc.
            'orientation': None,    # Orientation: parallel, non-parallel
            'echogenicity': None,   # Echogenicity: hypoechoic, anechoic, etc.
            'posterior': None,      # Posterior features: enhancement, shadowing
            'size_mm': None,        # Size in millimeters (approximate)
            'boundaries': None,     # Boundary regularity
            'texture': None         # Internal texture homogeneity
        }
        
        # Define domains for each variable (possible values)
        self.domains = {
            'shape': ['regular', 'oval', 'round', 'irregular', 'lobular'],
            'margin': ['circumscribed', 'indistinct', 'angular', 'microlobulated', 'spiculated'],
            'orientation': ['parallel', 'non_parallel'],
            'echogenicity': ['anechoic', 'hypoechoic', 'isoechoic', 'hyperechoic', 'complex'],
            'posterior': ['enhancement', 'shadowing', 'combined', 'none'],
            'size_mm': range(1, 100),  # Size from 1 to 100 mm
            'boundaries': ['smooth', 'irregular', 'angular'],
            'texture': ['homogeneous', 'heterogeneous']
        }
        
        # Scoring system for features - based on medical literature and ACR guidelines
        # Positive values indicate higher suspicion of malignancy
        # Negative values favor benignity
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
            'orientation': {
                'parallel': -0.2,        # Benign
                'non_parallel': 0.2      # Suspicious
            },
            'echogenicity': {
                'anechoic': -0.3,        # Very benign (typical of cysts)
                'hyperechoic': -0.1,     # Generally benign
                'isoechoic': 0.0,        # Neutral
                'complex': 0.1,          # Slightly suspicious
                'hypoechoic': 0.2        # Moderately suspicious
            },
            'posterior': {
                'enhancement': -0.2,      # Generally benign
                'none': 0.0,              # Neutral
                'combined': 0.1,          # Slightly suspicious
                'shadowing': 0.2          # Moderately suspicious
            },
            'boundaries': {
                'smooth': -0.2,           # Benign
                'irregular': 0.1,         # Slightly suspicious
                'angular': 0.2            # Moderately suspicious
            },
            'texture': {
                'homogeneous': -0.1,      # Generally benign
                'heterogeneous': 0.2      # Moderately suspicious
            }
        }
        
        # Weighting of each feature in the final score - based on clinical significance
        self.feature_weights = {
            'shape': 0.20,          # 20% of total score
            'margin': 0.25,         # 25% of total score - margin is a key predictor
            'orientation': 0.15,    # 15% of total score
            'echogenicity': 0.15,   # 15% of total score
            'posterior': 0.10,      # 10% of total score
            'boundaries': 0.05,     # 5% of total score
            'texture': 0.10         # 10% of total score
        }
        
        # Thresholds for BI-RADS categories based on weighted score
        # These thresholds determine the final classification
        self.birads_thresholds = {
            'BIRADS1': -1.0,     # Score < -0.4 (no visible lesion)
            'BIRADS2': -0.2,     # Score < -0.1 (definitely benign)
            'BIRADS3': 0.0,      # Score < 0.1 (probably benign)
            'BIRADS4A': 0.1,     # Score < 0.2 (low suspicion)
            'BIRADS4B': 0.2,     # Score < 0.3 (moderate suspicion)
            'BIRADS4C': 0.3,     # Score < 0.4 (high suspicion)
            'BIRADS5': 0.4       # Score >= 0.4 (highly suggestive of malignancy)
        }
        
        # Additional metadata for reporting and debugging
        self.metadata = {
            'execution_time': 0,
            'confidence_level': 0,
            'limitations': [],       # Limitations in the analysis
            'decisive_features': [], # Features that most influenced classification
            'total_score': 0,        # Final calculated score
            'feature_scores': {}     # Contribution of each feature
        }
    
    def _get_real_size(self, size_value):
        """
        Helper method to extract the real size value if stored as a dict with metadata.
        
        Args:
            size_value: Size value, either direct or as a dictionary
            
        Returns:
            float: The actual size value
        """
        if isinstance(size_value, dict):
            return size_value.get('value', 0)
        return size_value
    
    def extract_features_from_segmentation(self, original_image, mask, pixel_spacing_mm=0.2):
        """
        Extract lesion features from segmentation results.
        This is the first step in the classification process, where we analyze the
        image and segmentation mask to determine shape, margins, etc.
        
        Args:
            original_image: Original ultrasound image
            mask: Binary segmentation mask
            pixel_spacing_mm: Conversion factor from pixels to mm (default 0.2mm/pixel)
            
        Returns:
            Dict: Extracted feature values
        """
        # Reset limitations tracking
        self.metadata['limitations'] = []
        
        # Convert inputs to numpy arrays if they aren't already
        if not isinstance(original_image, np.ndarray):
            original_image = np.array(original_image)
        
        if not isinstance(mask, np.ndarray):
            mask = np.array(mask)
            
        # Ensure mask is binary
        if mask.dtype != bool:
            mask = mask > 0
            
        # Resize mask to match original image dimensions if needed
        if original_image.shape[:2] != mask.shape[:2]:
            from skimage.transform import resize
            # Use resize with order=0 (nearest neighbor) to preserve binary nature
            mask = resize(mask.astype(float), original_image.shape[:2], order=0, preserve_range=True) > 0.5
            
        # If no lesion detected, set appropriate values
        if np.sum(mask) < 10:  # Less than 10 pixels
            self.variables['size_mm'] = {'value': 0, 'approximate': True}
            return self.variables
            
        # Label connected components to identify distinct regions
        labeled_mask = measure.label(mask)
        props = measure.regionprops(labeled_mask)
        
        # If no regions found (shouldn't happen if mask has pixels)
        if len(props) == 0:
            self.variables['size_mm'] = {'value': 0, 'approximate': True}
            return self.variables
            
        # Work with the largest region (main lesion)
        largest_region = max(props, key=lambda p: p.area)
        
        # Extract size information
        area_pixels = largest_region.area
        size_mm = np.sqrt(area_pixels) * pixel_spacing_mm  # Convert to mm
        self.variables['size_mm'] = {'value': float(size_mm), 'approximate': True}
        
        # Extract shape information - critical for BI-RADS assessment
        # Calculate shape metrics
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
        
        # Extract orientation - important for BI-RADS
        orientation = largest_region.orientation
        if abs(orientation) < np.pi/4:  # Within 45 degrees of horizontal
            self.variables['orientation'] = 'parallel'
        else:
            self.variables['orientation'] = 'non_parallel'
        
        # Extract margin and boundary information - critical for malignancy assessment
        # Calculate boundary metrics
        perimeter = largest_region.perimeter
        area = largest_region.area
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Get region boundary
        boundary = np.zeros_like(mask, dtype=bool)
        dilated = morphology.binary_dilation(mask)
        boundary = dilated & ~mask
        
        # Calculate boundary roughness - critical for margin assessment
        if np.sum(boundary) > 0:
            boundary_coords = np.argwhere(boundary)
            # Use distance variation as roughness metric
            centroid = largest_region.centroid
            distances = np.sqrt(np.sum((boundary_coords - centroid)**2, axis=1))
            roughness = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0
        else:
            roughness = 0
        
        # Determine margin and boundary categories based on circularity and roughness
        if circularity > 0.9 and roughness < 0.1:
            self.variables['margin'] = 'circumscribed'
            self.variables['boundaries'] = 'smooth'
        elif circularity > 0.75 and roughness < 0.2:
            self.variables['margin'] = 'microlobulated'
            self.variables['boundaries'] = 'irregular'
        elif roughness > 0.3:
            self.variables['margin'] = 'spiculated'
            self.variables['boundaries'] = 'angular'
        else:
            self.variables['margin'] = 'indistinct'
            self.variables['boundaries'] = 'irregular'
        
        # Extract texture information (requires original image)
        # Get the region in the original image
        masked_image = original_image.copy()
        if len(masked_image.shape) == 3:  # Color image
            for i in range(masked_image.shape[2]):
                masked_image[:,:,i] = masked_image[:,:,i] * mask
        else:  # Grayscale
            masked_image = masked_image * mask
        
        # Calculate texture metrics on the masked region
        if np.sum(mask) > 0:
            region_values = original_image[mask]
            texture_std = np.std(region_values)
            texture_range = np.max(region_values) - np.min(region_values) if len(region_values) > 0 else 0
            
            # Normalize by maximum possible range
            if np.max(original_image) > np.min(original_image):
                normalized_range = texture_range / (np.max(original_image) - np.min(original_image))
            else:
                normalized_range = 0
                
            # Calculate GLCM texture features if scikit-image is available
            try:
                from skimage.feature import graycomatrix, graycoprops
                
                # Get region bounding box for GLCM analysis
                min_row, min_col, max_row, max_col = largest_region.bbox
                roi = original_image[min_row:max_row, min_col:max_col]
                roi_mask = mask[min_row:max_row, min_col:max_col]
                
                # Apply mask to ROI before GLCM calculation (for older scikit-image versions)
                # This is a workaround for versions that don't support the mask parameter
                roi_masked = roi.copy()
                if not np.all(roi_mask):  # Only apply if mask isn't all True
                    # Set non-masked areas to 0 or another background value
                    roi_masked[~roi_mask] = 0
                
                # Scale to 0-255 and convert to uint8 for GLCM
                if np.max(roi_masked) > np.min(roi_masked):
                    roi_scaled = ((roi_masked - np.min(roi_masked)) / (np.max(roi_masked) - np.min(roi_masked)) * 255).astype(np.uint8)
                else:
                    roi_scaled = np.zeros_like(roi_masked, dtype=np.uint8)
                    
                # Only calculate if region is large enough
                if roi_scaled.shape[0] > 5 and roi_scaled.shape[1] > 5:
                    # Calculate GLCM without mask parameter
                    glcm = graycomatrix(roi_scaled, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
                    
                    # Calculate GLCM properties
                    contrast = graycoprops(glcm, 'contrast').mean()
                    homogeneity = graycoprops(glcm, 'homogeneity').mean()
                    
                    # Use homogeneity for texture classification
                    if homogeneity > 0.7:
                        self.variables['texture'] = 'homogeneous'
                    else:
                        self.variables['texture'] = 'heterogeneous'
                else:
                    # Default for small regions
                    self.variables['texture'] = 'homogeneous' if texture_std < 0.15 * np.mean(region_values) else 'heterogeneous'
            except (ImportError, ValueError, RuntimeError) as e:
                # Fallback if GLCM calculation fails
                print(f"GLCM calculation error: {str(e)}")
                self.variables['texture'] = 'homogeneous' if texture_std < 0.15 * np.mean(region_values) else 'heterogeneous'
        else:
            # Default if no valid region
            self.variables['texture'] = 'homogeneous'
        
        # Improved echogenicity estimation based on medical feedback
        self._estimate_echogenicity(original_image, mask, largest_region)
        
        # Improved detection of posterior features
        self._analyze_posterior_features(original_image, mask, largest_region)
        
        return self.variables
    
    def _estimate_echogenicity(self, original_image, mask, region):
        """
        Estimate echogenicity of lesion, attempting to use subcutaneous fat as reference when possible.
        The medical literature recommends comparing to fat for standardized assessment.
        
        Args:
            original_image: Original ultrasound image
            mask: Binary segmentation mask
            region: Region properties of the segmented area
        """
        if np.sum(mask) == 0:
            self.variables['echogenicity'] = None
            self.metadata['limitations'].append("Echogenicity could not be evaluated (no lesion detected)")
            return
            
        # Get average intensity in lesion
        lesion_mean = np.mean(original_image[mask])
        
        # Try to find subcutaneous fat as reference (typically near the top of the image)
        # This is a simplistic approach - proper fat detection would require additional algorithms
        height, width = original_image.shape[:2]
        fat_reference_found = False
        
        try:
            # Try to find an area in the upper part of the image not overlapping with the lesion
            # as a potential subcutaneous fat reference
            top_margin = height // 5  # Upper 20% of image
            
            # Create mask for potential fat region (upper part of image)
            fat_mask = np.zeros_like(mask)
            fat_mask[:top_margin, :] = True
            
            # Exclude lesion area from fat mask
            fat_mask = fat_mask & ~morphology.binary_dilation(mask, morphology.disk(5))
            
            # Check if we have enough potential fat area
            if np.sum(fat_mask) > 100:  # At least 100 pixels
                fat_mean = np.mean(original_image[fat_mask])
                fat_reference_found = True
                
                # Determine echogenicity relative to fat
                ratio = lesion_mean / fat_mean if fat_mean > 0 else 1
                
                if ratio < 0.3:
                    self.variables['echogenicity'] = 'anechoic'
                elif ratio < 0.7:
                    self.variables['echogenicity'] = 'hypoechoic'
                elif ratio < 1.3:
                    self.variables['echogenicity'] = 'isoechoic'
                else:
                    self.variables['echogenicity'] = 'hyperechoic'
            else:
                fat_reference_found = False
        except Exception as e:
            print(f"Error finding fat reference: {str(e)}")
            fat_reference_found = False
            
        # If fat reference wasn't found, fall back to surrounding tissue comparison
        if not fat_reference_found:
            # Add this as a limitation
            self.metadata['limitations'].append("Echogenicity evaluated against surrounding tissue (fat reference not found)")
            
            # Create dilated mask for surrounding tissue
            dilated_mask = morphology.binary_dilation(mask, morphology.disk(10))
            surrounding_mask = dilated_mask & ~mask
            
            if np.sum(surrounding_mask) > 0:
                surrounding_mean = np.mean(original_image[surrounding_mask])
                
                # Determine echogenicity relative to surrounding tissue
                ratio = lesion_mean / surrounding_mean if surrounding_mean > 0 else 1
                
                if ratio < 0.3:
                    self.variables['echogenicity'] = 'anechoic'
                elif ratio < 0.7:
                    self.variables['echogenicity'] = 'hypoechoic'
                elif ratio < 1.3:
                    self.variables['echogenicity'] = 'isoechoic'
                else:
                    self.variables['echogenicity'] = 'hyperechoic'
            else:
                # Default if can't compare to surroundings
                self.variables['echogenicity'] = 'hypoechoic'
                self.metadata['limitations'].append("Echogenicity approximated (could not evaluate against reference tissue)")
    
    def _analyze_posterior_features(self, original_image, mask, region):
        """
        Analyze the area posterior to the lesion to detect enhancement or shadowing.
        Posterior features are important diagnostic markers in breast ultrasound.
        
        Args:
            original_image: Original ultrasound image
            mask: Binary segmentation mask
            region: Region properties of the segmented area
        """
        # Default if analysis fails
        self.variables['posterior'] = None
        
        try:
            # Get image dimensions
            height, width = original_image.shape[:2]
            
            # Get region properties
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)
            centroid = region.centroid
            
            # Define posterior region (area below the lesion)
            posterior_margin = 10  # Pixels to check below the lesion
            posterior_start = min(bbox[2], height - 1)  # Bottom of lesion or image edge
            posterior_end = min(posterior_start + posterior_margin, height)
            
            if posterior_end <= posterior_start or posterior_start >= height - 5:
                # Lesion extends to bottom of image or too close to it
                self.metadata['limitations'].append("Posterior features not evaluated (lesion near image edge)")
                return
                
            # Create mask for posterior region
            posterior_mask = np.zeros_like(mask, dtype=bool)
            posterior_mask[posterior_start:posterior_end, bbox[1]:bbox[3]] = True
            
            # Create mask for lateral regions (same depth but beside the lesion)
            lateral_width = max(10, (bbox[3] - bbox[1]) // 4)  # Width proportional to lesion width
            
            lateral_left_mask = np.zeros_like(mask, dtype=bool)
            lateral_left_start = max(0, bbox[1] - lateral_width)
            lateral_left_end = max(0, bbox[1])
            if lateral_left_end > lateral_left_start:
                lateral_left_mask[posterior_start:posterior_end, lateral_left_start:lateral_left_end] = True
            
            lateral_right_mask = np.zeros_like(mask, dtype=bool)
            lateral_right_start = min(bbox[3], width - 1)
            lateral_right_end = min(lateral_right_start + lateral_width, width)
            if lateral_right_end > lateral_right_start:
                lateral_right_mask[posterior_start:posterior_end, lateral_right_start:lateral_right_end] = True
            
            # Combine lateral masks
            lateral_mask = lateral_left_mask | lateral_right_mask
            
            # Check if we have enough pixels in lateral and posterior regions
            if np.sum(posterior_mask) < 10 or np.sum(lateral_mask) < 10:
                self.metadata['limitations'].append("Posterior features approximated (limited comparison area)")
                return
            
            # Calculate average intensities
            posterior_mean = np.mean(original_image[posterior_mask])
            lateral_mean = np.mean(original_image[lateral_mask])
            
            # Calculate ratio of posterior to lateral intensity
            intensity_ratio = posterior_mean / lateral_mean if lateral_mean > 0 else 1.0
            
            # Determine posterior features based on ratio
            if intensity_ratio > 1.2:  # Significantly brighter
                self.variables['posterior'] = 'enhancement'
            elif intensity_ratio < 0.8:  # Significantly darker
                self.variables['posterior'] = 'shadowing'
            elif 0.8 <= intensity_ratio <= 1.2:  # Similar intensity
                self.variables['posterior'] = 'none'
            
        except Exception as e:
            print(f"Error analyzing posterior features: {str(e)}")
            self.metadata['limitations'].append(f"Posterior features not evaluated (error: {str(e)})")
    
    def calculate_birads_score(self, algorithm='standard'):
        """
        Calculate the weighted score based on extracted features
        and determine the corresponding BI-RADS category.
        
        Args:
            algorithm: Algorithm to use ('standard' or 'fuzzy')
            
        Returns:
            Tuple: (BI-RADS category, confidence level, total score, detailed results)
        """
        start_time = time.time()
        
        # Reset metadata
        self.metadata['decisive_features'] = []
        self.metadata['feature_scores'] = {}
        self.metadata['algorithm_used'] = algorithm
        
        # Redirect to specialized methods based on algorithm
        if algorithm == 'fuzzy':
            return self._calculate_fuzzy_score()
        else:
            return self._calculate_standard_score()

    # Then add this new method to the class:

    def _calculate_standard_score(self):
        """
        Standard algorithm using weighted scoring.
        
        Returns:
            Tuple: (BI-RADS category, confidence level, total score, detailed results)
        """
        # Initialize start_time at the beginning of the method
        start_time = time.time()
        # Check if we have enough evaluated features
        evaluated_features = [feat for feat, val in self.variables.items() if val is not None]
        
        # If not enough features, return BI-RADS 0
        if len(evaluated_features) < 3:
            self.metadata['confidence_level'] = 0.8
            self.metadata['total_score'] = 0
            self.metadata['limitations'].append(f"Incomplete classification: only {len(evaluated_features)} features evaluated")
            return 'BIRADS0', 0.8, 0, {'BIRADS0': 1.0}
        
        # Special handling for no lesion case
        size_mm = self._get_real_size(self.variables.get('size_mm', 0))
        if size_mm is not None and size_mm < 3:
            self.metadata['confidence_level'] = 0.9
            self.metadata['total_score'] = -0.5
            self.metadata['decisive_features'] = ['size_mm']
            return 'BIRADS1', 0.9, -0.5, {'BIRADS1': 1.0}
        
        # Check for simple cyst pattern (classic benign pattern)
        if (self.variables.get('shape') in ['round', 'oval'] and 
            self.variables.get('margin') == 'circumscribed' and 
            self.variables.get('echogenicity') == 'anechoic' and
            self.variables.get('posterior') == 'enhancement'):
            
            # A simple cyst is definitely benign (BI-RADS 2)
            self.metadata['confidence_level'] = 0.95
            self.metadata['total_score'] = -0.4
            self.metadata['decisive_features'] = ['Simple cyst pattern']
            return 'BIRADS2', 0.95, -0.4, {'BIRADS2': 1.0}
        
        # Calculate weighted score
        total_score = 0
        weighted_score = 0
        used_weights_sum = 0
        
        # Save individual scores for explanation
        feature_scores = {}
        
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
        
        # Adjust for size if available
        size_adjustment = 0
        if self.variables.get('size_mm') is not None:
            size_mm = self._get_real_size(self.variables.get('size_mm'))
            if size_mm < 5:
                size_adjustment = -0.05  # Very small lesions: slightly reduce suspicion
            elif size_mm > 20:
                size_adjustment = 0.05   # Large lesions: slightly increase suspicion
                
            # Size alone should not determine classification
            total_score += size_adjustment
            feature_scores['size_mm'] = {
                'value': size_mm,
                'adjustment': size_adjustment
            }
        
        # Save score and details in metadata
        self.metadata['total_score'] = total_score
        self.metadata['feature_scores'] = feature_scores
        
        # Determine BI-RADS category based on score
        birads_category = None
        confidence = 0
        
        # Detailed scores by category
        detailed_results = {}
        
        # Represent score as certainty level for each BI-RADS category
        for birads, threshold in sorted(self.birads_thresholds.items(), key=lambda x: x[1]):
            if total_score >= threshold:
                birads_category = birads
                
            # Calculate how close we are to the ideal threshold for this category
            if birads == 'BIRADS1':
                detailed_results[birads] = max(0, 1 - (total_score - threshold) * 5) if total_score < -0.2 else 0
            elif birads == 'BIRADS2':
                # More certainty if we're in the middle of the BI-RADS 2 range
                center = (self.birads_thresholds['BIRADS2'] + self.birads_thresholds['BIRADS3']) / 2
                detailed_results[birads] = max(0, 1 - abs(total_score - center) * 5) if -0.2 <= total_score < 0 else 0
            elif birads == 'BIRADS3':
                # More certainty if we're in the middle of the BI-RADS 3 range
                center = (self.birads_thresholds['BIRADS3'] + self.birads_thresholds['BIRADS4A']) / 2
                detailed_results[birads] = max(0, 1 - abs(total_score - center) * 5) if 0 <= total_score < 0.1 else 0
            elif birads == 'BIRADS4A':
                center = (self.birads_thresholds['BIRADS4A'] + self.birads_thresholds['BIRADS4B']) / 2
                detailed_results[birads] = max(0, 1 - abs(total_score - center) * 5) if 0.1 <= total_score < 0.2 else 0
            elif birads == 'BIRADS4B':
                center = (self.birads_thresholds['BIRADS4B'] + self.birads_thresholds['BIRADS4C']) / 2
                detailed_results[birads] = max(0, 1 - abs(total_score - center) * 5) if 0.2 <= total_score < 0.3 else 0
            elif birads == 'BIRADS4C':
                center = (self.birads_thresholds['BIRADS4C'] + self.birads_thresholds['BIRADS5']) / 2
                detailed_results[birads] = max(0, 1 - abs(total_score - center) * 5) if 0.3 <= total_score < 0.4 else 0
            elif birads == 'BIRADS5':
                detailed_results[birads] = max(0, 1 - (0.6 - total_score) * 5) if total_score >= 0.4 else 0
        
        # If no category found (shouldn't happen), use BIRADS0
        if not birads_category:
            birads_category = 'BIRADS0'
            confidence = 0.5
            detailed_results['BIRADS0'] = 0.5
        else:
            # Get confidence from detailed results
            confidence = detailed_results[birads_category]
        
        # Identify decisive features
        sorted_features = sorted(
            [(f, d['weighted_score']) for f, d in feature_scores.items() if 'weighted_score' in d],
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Top 3 most influential features
        self.metadata['decisive_features'] = [f[0] for f in sorted_features[:3] if f[0] != 'size_mm']
        
        # Record execution time
        self.metadata['execution_time'] = time.time() - start_time
        self.metadata['confidence_level'] = confidence
        
        return birads_category, confidence, total_score, detailed_results

    def _calculate_fuzzy_score(self):
        """
        Fuzzy algorithm using membership functions and linguistic variables.
        This implementation gives more weight to certain feature combinations
        and considers uncertainty in feature assessment.
        
        Returns:
            Tuple: (BI-RADS category, confidence level, total score, detailed results)
        """
        start_time = time.time()
        
        # Check if we have enough evaluated features
        evaluated_features = [feat for feat, val in self.variables.items() if val is not None]
        
        # If not enough features, return BI-RADS 0
        if len(evaluated_features) < 3:
            self.metadata['confidence_level'] = 0.8
            self.metadata['total_score'] = 0
            self.metadata['limitations'].append(f"Incomplete classification: only {len(evaluated_features)} features evaluated")
            return 'BIRADS0', 0.8, 0, {'BIRADS0': 1.0}
        
        # Define fuzzy membership functions for key features
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
        
        def margin_suspicion(margin):
            if margin is None:
                return 0.0  # Not evaluated
            
            # Different fuzzy mapping
            suspicion_map = {
                'circumscribed': 0.0,    # Very benign
                'microlobulated': 0.4,   # Moderately suspicious
                'indistinct': 0.6,       # Suspicious
                'angular': 0.7,          # Highly suspicious
                'spiculated': 0.9        # Very suspicious
            }
            return suspicion_map.get(margin, 0.5)
        
        def orientation_suspicion(orientation):
            if orientation is None:
                return 0.0  # Not evaluated
            return 0.1 if orientation == 'parallel' else 0.7
        
        def echogenicity_suspicion(echogenicity):
            if echogenicity is None:
                return 0.0  # Not evaluated
            
            suspicion_map = {
                'anechoic': 0.0,       # Very benign (cysts)
                'hyperechoic': 0.2,    # Generally benign
                'isoechoic': 0.3,      # Neutral
                'complex': 0.5,        # Moderately suspicious
                'hypoechoic': 0.6      # Suspicious
            }
            return suspicion_map.get(echogenicity, 0.3)
        
        def posterior_suspicion(posterior):
            if posterior is None:
                return 0.0  # Not evaluated
            
            suspicion_map = {
                'enhancement': 0.1,   # Benign
                'none': 0.3,          # Neutral
                'combined': 0.5,      # Moderately suspicious
                'shadowing': 0.7      # Suspicious
            }
            return suspicion_map.get(posterior, 0.3)
        
        # Special handle for simple cyst
        if (self.variables.get('shape') in ['round', 'oval'] and 
            self.variables.get('margin') == 'circumscribed' and 
            self.variables.get('echogenicity') == 'anechoic' and
            self.variables.get('posterior') == 'enhancement'):
            
            # Simple cyst is definitely benign (BI-RADS 2)
            self.metadata['confidence_level'] = 0.95
            self.metadata['total_score'] = -0.4
            self.metadata['decisive_features'] = ['Simple cyst pattern']
            return 'BIRADS2', 0.95, -0.4, {'BIRADS2': 1.0}
        
        # Special handle for clear malignancy
        if (self.variables.get('shape') == 'irregular' and 
            self.variables.get('margin') == 'spiculated' and
            self.variables.get('orientation') == 'non_parallel'):
            
            # Classic malignancy pattern
            self.metadata['confidence_level'] = 0.9
            self.metadata['total_score'] = 0.5
            self.metadata['decisive_features'] = ['Classic malignant pattern']
            return 'BIRADS5', 0.9, 0.5, {'BIRADS5': 0.9}
        
        # Calculate fuzzy suspicion score - more complex than standard approach
        suspicion_level = 0
        
        # Get individual suspicion scores
        shape_susp = shape_suspicion(self.variables.get('shape'))
        margin_susp = margin_suspicion(self.variables.get('margin'))
        orientation_susp = orientation_suspicion(self.variables.get('orientation'))
        echo_susp = echogenicity_suspicion(self.variables.get('echogenicity'))
        posterior_susp = posterior_suspicion(self.variables.get('posterior'))
        
        # Fuzzy combination rules - more complex than simple weighting
        
        # Rule 1: If multiple highly suspicious features, increase suspicion synergistically
        high_suspicion_count = sum(1 for score in [shape_susp, margin_susp, orientation_susp] 
                                if score > 0.6)
        if high_suspicion_count >= 2:
            combined_suspicion = max(shape_susp, margin_susp, orientation_susp) + 0.2
            suspicion_level = min(0.9, combined_suspicion)  # Cap at 0.9
        
        # Rule 2: If mixed features, use weighted average with higher weight on shape and margin
        else:
            # Different weights than standard algorithm
            weights = {
                'shape': 0.35 if shape_susp > 0 else 0,
                'margin': 0.30 if margin_susp > 0 else 0,
                'orientation': 0.15 if orientation_susp > 0 else 0,
                'echogenicity': 0.10 if echo_susp > 0 else 0,
                'posterior': 0.10 if posterior_susp > 0 else 0
            }
            
            # Calculate weighted suspicion
            total_weight = sum(weights.values())
            if total_weight > 0:
                suspicion_level = (weights['shape'] * shape_susp +
                                weights['margin'] * margin_susp +
                                weights['orientation'] * orientation_susp +
                                weights['echogenicity'] * echo_susp +
                                weights['posterior'] * posterior_susp) / total_weight
            else:
                suspicion_level = 0.3  # Default moderate suspicion with insufficient data
        
        # Size consideration is fuzzy - no exact thresholds
        size_mm = self._get_real_size(self.variables.get('size_mm', 0))
        if size_mm is not None:
            # Fuzzy size adjustment
            if size_mm < 5:
                # Very small lesions fuzzy adjustment
                suspicion_level = max(0, suspicion_level - 0.15)
            elif 5 <= size_mm < 10:
                # Small lesions slight reduction
                suspicion_level = max(0, suspicion_level - 0.05)
            elif size_mm > 20:
                # Large lesions increase suspicion more in fuzzy logic
                suspicion_level = min(0.9, suspicion_level + 0.1)
        
        # Map fuzzy suspicion level to BI-RADS category
        birads_category = None
        if suspicion_level < 0.15:
            birads_category = 'BIRADS2'
            fuzzy_score = -0.3
        elif 0.15 <= suspicion_level < 0.3:
            birads_category = 'BIRADS3'
            fuzzy_score = 0.05
        elif 0.3 <= suspicion_level < 0.45:
            birads_category = 'BIRADS4A'
            fuzzy_score = 0.15
        elif 0.45 <= suspicion_level < 0.6:
            birads_category = 'BIRADS4B'
            fuzzy_score = 0.25
        elif 0.6 <= suspicion_level < 0.75:
            birads_category = 'BIRADS4C'
            fuzzy_score = 0.35
        else:
            birads_category = 'BIRADS5'
            fuzzy_score = 0.45
        
        # Record for metadata
        self.metadata['fuzzy_suspicion_level'] = suspicion_level
        self.metadata['total_score'] = fuzzy_score
        
        # Feature scores for explanation
        feature_scores = {
            'shape': {'value': self.variables.get('shape'), 'score': shape_susp},
            'margin': {'value': self.variables.get('margin'), 'score': margin_susp},
            'orientation': {'value': self.variables.get('orientation'), 'score': orientation_susp},
            'echogenicity': {'value': self.variables.get('echogenicity'), 'score': echo_susp},
            'posterior': {'value': self.variables.get('posterior'), 'score': posterior_susp}
        }
        
        if size_mm is not None:
            feature_scores['size_mm'] = {'value': size_mm}
        
        self.metadata['feature_scores'] = feature_scores
        
        # Calculate detailed results - different from standard approach
        detailed_results = {}
        
        # Each category has a membership function
        detailed_results['BIRADS2'] = max(0, 1 - suspicion_level * 4) 
        detailed_results['BIRADS3'] = max(0, 1 - abs(suspicion_level - 0.2) * 5)
        detailed_results['BIRADS4A'] = max(0, 1 - abs(suspicion_level - 0.35) * 5)
        detailed_results['BIRADS4B'] = max(0, 1 - abs(suspicion_level - 0.5) * 5)
        detailed_results['BIRADS4C'] = max(0, 1 - abs(suspicion_level - 0.65) * 5)
        detailed_results['BIRADS5'] = max(0, (suspicion_level - 0.6) * 3)
        
        # Get confidence from detailed results
        confidence = detailed_results[birads_category]
        
        # Determine decisive features
        decisive_features = []
        if shape_susp > 0.3:
            decisive_features.append('shape')
        if margin_susp > 0.3:
            decisive_features.append('margin')
        if orientation_susp > 0.3:
            decisive_features.append('orientation')
        
        # More specific decisive feature names for fuzzy logic
        if len(decisive_features) == 0:
            if echo_susp > 0.3:
                decisive_features.append('echogenicity')
            if posterior_susp > 0.3:
                decisive_features.append('posterior')
        
        self.metadata['decisive_features'] = decisive_features[:3]  # Top 3
        self.metadata['execution_time'] = time.time() - start_time
        self.metadata['confidence_level'] = confidence
        
        return birads_category, confidence, fuzzy_score, detailed_results
    
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
        from datetime import datetime
        
        # Report header
        report = "BREAST ULTRASOUND REPORT\n"
        report += "=" * 40 + "\n\n"
        
        # Patient information if available
        if patient_info:
            report += f"Patient: {patient_info.get('name', 'Not specified')}\n"
            report += f"ID: {patient_info.get('id', 'Not specified')}\n"
            report += f"Age: {patient_info.get('age', 'Not specified')}\n"
        
        report += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        # Findings section
        report += "FINDINGS:\n"
        report += "-" * 40 + "\n"
        
        # Describe detected features
        for feature, value in self.variables.items():
            if value is not None:
                # Get human-readable feature name
                feature_name = {
                    'shape': 'Shape',
                    'margin': 'Margin',
                    'orientation': 'Orientation',
                    'echogenicity': 'Echogenicity',
                    'posterior': 'Posterior features',
                    'size_mm': 'Size (mm)',
                    'boundaries': 'Boundaries',
                    'texture': 'Texture'
                }.get(feature, feature.capitalize())
                
                # Handle size_mm with metadata
                if feature == 'size_mm' and isinstance(value, dict):
                    size_value = value.get('value', 'Unknown')
                    report += f"- {feature_name}: {size_value:.2f} (approximate)\n"
                else:
                    report += f"- {feature_name}: {value}\n"
        
        report += "\n"
        
        # Technical assessment section
        report += "TECHNICAL ASSESSMENT:\n"
        report += "-" * 40 + "\n"
        report += f"Analysis method: Weighted scoring system\n"
        report += f"Execution time: {self.metadata['execution_time']:.3f} seconds\n"
        report += f"Total score: {total_score:.3f} (range -1.0 to +1.0)\n"
        report += f"Confidence level: {confidence:.2f}\n\n"
        
        # Decisive features section
        report += "DECISIVE FEATURES:\n"
        report += "-" * 40 + "\n"
        
        # Show individual feature scores
        feature_scores = self.metadata.get('feature_scores', {})
        if feature_scores:
            for feature, details in feature_scores.items():
                if feature == 'size_mm':
                    if 'adjustment' in details:
                        adjustment = details['adjustment']
                        adjustment_text = "increases" if adjustment > 0 else "reduces"
                        report += f"- Size of {details['value']:.2f} mm: {adjustment_text} suspicion slightly ({adjustment:+.2f})\n"
                elif 'weighted_score' in details:
                    feature_name = {
                        'shape': 'Shape',
                        'margin': 'Margin',
                        'orientation': 'Orientation',
                        'echogenicity': 'Echogenicity',
                        'posterior': 'Posterior features',
                        'boundaries': 'Boundaries',
                        'texture': 'Texture'
                    }.get(feature, feature.capitalize())
                    
                    # Determine if feature favors benignity or malignancy
                    direction = "favors benignity" if details['score'] < 0 else "increases suspicion" if details['score'] > 0 else "neutral"
                    
                    report += f"- {feature_name} {details['value']}: {direction} (contribution: {details['weighted_score']:+.3f})\n"
        
        report += "\n"
        
        # Limitations section if any found
        if self.metadata.get('limitations'):
            report += "LIMITATIONS:\n"
            report += "-" * 40 + "\n"
            for limitation in self.metadata['limitations']:
                report += f"- {limitation}\n"
            report += "\n"
        
        # BI-RADS classification
        report += "BI-RADS CLASSIFICATION:\n"
        report += "-" * 40 + "\n"
        
        # Get BI-RADS description
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
        
        report += f"{birads_descriptions.get(birads_category, birads_category)}\n\n"
        
        # Clinical relevance notes
        if birads_category in ['BIRADS4A', 'BIRADS4B', 'BIRADS4C', 'BIRADS5']:
            report += "CLINICAL RELEVANCE:\n"
            report += "-" * 40 + "\n"
            
            risk_percentages = {
                'BIRADS4A': "2-10%",
                'BIRADS4B': "10-50%",
                'BIRADS4C': "50-95%",
                'BIRADS5': ">95%"
            }
            
            report += f"Risk of malignancy: {risk_percentages.get(birads_category, 'Unknown')}\n"
            report += f"Features supporting this classification:\n"
            
            # List suspicious features found
            suspicious_features = []
            if self.variables.get('shape') == 'irregular':
                suspicious_features.append("irregular shape")
            if self.variables.get('margin') in ['indistinct', 'microlobulated', 'angular', 'spiculated']:
                suspicious_features.append(f"{self.variables.get('margin')} margins")
            if self.variables.get('orientation') == 'non_parallel':
                suspicious_features.append("non-parallel orientation")
            if self.variables.get('posterior') == 'shadowing':
                suspicious_features.append("posterior shadowing")
            if self.variables.get('texture') == 'heterogeneous':
                suspicious_features.append("heterogeneous texture")
                
            if suspicious_features:
                for feature in suspicious_features:
                    report += f"- {feature}\n"
            else:
                report += "- Classification based on overall feature pattern\n"
                
            report += "\n"
        
        # Recommendations section
        report += "RECOMMENDATIONS:\n"
        report += "-" * 40 + "\n"
        
        # Add appropriate recommendations based on BI-RADS category
        recommendations = {
            'BIRADS0': "Additional imaging evaluation and/or comparison to prior examination is needed.",
            'BIRADS1': "Routine screening as per age-appropriate guidelines.",
            'BIRADS2': "Routine screening as per age-appropriate guidelines.",
            'BIRADS3': "Short-term follow-up (usually 6 months) recommended.",
            'BIRADS4A': "Tissue diagnosis should be considered. Biopsy is recommended.",
            'BIRADS4B': "Tissue diagnosis should be obtained. Biopsy is recommended.",
            'BIRADS4C': "Tissue diagnosis must be obtained. Biopsy is strongly recommended.",
            'BIRADS5': "Appropriate action should be taken. Biopsy and treatment planning are strongly recommended."
        }
        
        report += f"{recommendations.get(birads_category, 'Consult with a healthcare provider for appropriate next steps.')}\n\n"
        
        # Add disclaimer
        report += "DISCLAIMER:\n"
        report += "-" * 40 + "\n"
        report += "This analysis is based on algorithmic interpretation and should be reviewed by a healthcare professional. "
        report += "It is not a substitute for clinical judgment or professional medical advice.\n"
        report += "Size measurements are approximate and based on pixel-to-millimeter conversion. Calcifications and "
        report += "vascularity were not evaluated as they require specialized imaging techniques.\n"
        
        return report
    
    def classify(self, algorithm='standard'):
        """
        Main method to classify the lesion into a BI-RADS category.
        Simplifies the API by combining score calculation and report generation.
        
        Args:
            algorithm: Algorithm to use ('standard' or 'fuzzy')
            
        Returns:
            Tuple: (BI-RADS category, confidence level, detailed report, detailed results)
        """
        # Calculate BI-RADS score using the selected algorithm
        birads_category, confidence, total_score, detailed_results = self.calculate_birads_score(algorithm=algorithm)
        
        # Generate detailed report
        report = self.generate_report(birads_category, confidence, total_score, detailed_results)
        
        return birads_category, confidence, report, detailed_results