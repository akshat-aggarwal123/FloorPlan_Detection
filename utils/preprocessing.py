import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FloorplanPreprocessor:
    def _init_(self, input_size=512):
        self.input_size = input_size
        # Updated transform for B/W images
        self.transform = A.Compose([
            A.Resize(input_size, input_size),
            A.ToGray(p=1.0),  # Ensure grayscale
            A.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
            ToTensorV2()
        ])
        
    def preprocess_image(self, image_path):
        """Preprocess B/W floorplan image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to 3-channel for consistency
        
        transformed = self.transform(image=image)
        return transformed['image'].unsqueeze(0)
    
    # def create_label_masks(self, annotation_data, input_size):
    #     """Create ground truth masks from annotation data"""
    #     boundary_mask = np.zeros((input_size, input_size), dtype=np.uint8)
    #     room_mask = np.zeros((input_size, input_size), dtype=np.uint8)
        
    #     # This would be implemented based on your annotation format
    #     # For now, returning empty masks as placeholder
    #     return boundary_mask, room_mask