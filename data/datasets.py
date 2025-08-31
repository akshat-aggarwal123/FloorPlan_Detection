import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FloorplanDataset(Dataset):
    def __init__(self, data_dir, config, is_train=True):
        self.data_dir = data_dir
        self.config = config
        self.is_train = is_train
        
        # Get all image files
        self.image_files = [
            f for f in os.listdir(os.path.join(data_dir, 'images')) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Data augmentation
        if is_train:
            self.transform = A.Compose([
                A.Resize(config.INPUT_SIZE, config.INPUT_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(config.INPUT_SIZE, config.INPUT_SIZE),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load corresponding labels
        label_name = os.path.splitext(img_name)[0] + "_boundary.png"
        boundary_path = os.path.join(self.data_dir, 'boundary_labels', label_name)
        
        room_label_name = os.path.splitext(img_name)[0] + "_room.png"
        room_path = os.path.join(self.data_dir, 'room_labels', room_label_name)
        
        # Load labels
        if os.path.exists(boundary_path):
            boundary_label = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
        else:
            boundary_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            
        if os.path.exists(room_path):
            room_label = cv2.imread(room_path, cv2.IMREAD_GRAYSCALE)
        else:
            room_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        transformed = self.transform(image=image, masks=[boundary_label, room_label])
        image = transformed['image']
        
        # Ensure masks are tensors
        boundary_label = transformed['masks'][0]
        room_label = transformed['masks'][1]
        
        if isinstance(boundary_label, np.ndarray):
            boundary_label = torch.from_numpy(boundary_label)
        if isinstance(room_label, np.ndarray):
            room_label = torch.from_numpy(room_label)

        boundary_label = boundary_label.long()
        room_label = room_label.long()
        
        return image, boundary_label, room_label
