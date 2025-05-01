# dataset.py
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BUSIDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=(224, 224), augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment
        
        # Define transformaciones
        if self.augment:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5, brightness_limit=0.2, contrast_limit=0.2),
                A.GaussNoise(p=0.3),
                A.OneOf([
                    A.MotionBlur(p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.CLAHE(p=0.3),  # Contraste mejorado para imágenes de ultrasonido
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.CLAHE(p=1.0),  # Siempre aplicamos CLAHE para mejorar el contraste
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Cargar imagen y máscara
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Leer imagen y convertir a RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Leer máscara y asegurar que es binaria
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 128).astype(np.float32)
        
        # Aplicar transformaciones
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
            # Asegurar que la máscara tiene el formato correcto para el modelo
            mask = mask.unsqueeze(0)
        
        return image, mask