from email.headerregistry import DateHeader
import time
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pickle
from concurrent.futures import ThreadPoolExecutor
import threading

class FloorplanDataset(Dataset):
    def __init__(self, data_dir, config, is_train=True, cache_data=True):
        self.data_dir = data_dir
        self.config = config
        self.is_train = is_train
        self.cache_data = cache_data
        
        # Get all image files
        self.image_files = [
            f for f in os.listdir(os.path.join(data_dir, 'images')) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        # Create cache directory
        self.cache_dir = os.path.join(data_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize memory cache without locks (for multiprocessing compatibility)
        self.memory_cache = {}
        # Remove the threading lock as it can't be pickled
        # self.cache_lock = threading.Lock()  # This causes the pickle error
        
        # Optimized data augmentation - separate image and mask transforms
        if is_train:
            # Transforms that affect both image and masks
            self.spatial_transform = A.Compose([
                A.Resize(config.INPUT_SIZE, config.INPUT_SIZE, interpolation=cv2.INTER_LINEAR),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ], additional_targets={'boundary': 'mask', 'room': 'mask'})
            
            # Transforms that only affect the image
            self.image_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        else:
            # Validation transforms
            self.spatial_transform = A.Compose([
                A.Resize(config.INPUT_SIZE, config.INPUT_SIZE, interpolation=cv2.INTER_LINEAR),
            ], additional_targets={'boundary': 'mask', 'room': 'mask'})
            
            self.image_transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        
        # Pre-cache data if enabled
        if cache_data:
            self._cache_dataset()
    
    def _cache_dataset(self):
        """Pre-cache processed data to disk for faster loading"""
        cache_file = os.path.join(self.cache_dir, f'cached_data_{"train" if self.is_train else "val"}.pkl')
        
        if os.path.exists(cache_file):
            print(f"Loading cached dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.cached_data = pickle.load(f)
            return
        
        print("Caching dataset for faster loading...")
        self.cached_data = []
        
        for idx in tqdm(range(len(self.image_files)), desc="Caching data"):
            try:
                data = self._load_raw_data(idx)
                self.cached_data.append(data)
            except Exception as e:
                print(f"Error caching item {idx}: {e}")
                self.cached_data.append(None)
        
        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(self.cached_data, f)
        print(f"Dataset cached to {cache_file}")
    
    def _load_raw_data(self, idx):
        """Load raw data without transformations"""
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, 'images', img_name)
        
        # Use PIL for faster loading
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Load labels with error handling
        label_name = os.path.splitext(img_name)[0] + "_boundary.png"
        boundary_path = os.path.join(self.data_dir, 'boundary_labels', label_name)
        
        room_label_name = os.path.splitext(img_name)[0] + "_room.png"
        room_path = os.path.join(self.data_dir, 'room_labels', room_label_name)
        
        try:
            boundary_label = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
            if boundary_label is None:
                boundary_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        except:
            boundary_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        try:
            room_label = cv2.imread(room_path, cv2.IMREAD_GRAYSCALE)
            if room_label is None:
                room_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        except:
            room_label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        return {
            'image': image,
            'boundary_label': boundary_label,
            'room_label': room_label
        }
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Check memory cache (without locks since each worker has its own instance)
        if idx in self.memory_cache:
            return self.memory_cache[idx]
        
        # Load from disk cache or raw data
        if hasattr(self, 'cached_data') and self.cached_data[idx] is not None:
            data = self.cached_data[idx]
            image = data['image']
            boundary_label = data['boundary_label']
            room_label = data['room_label']
        else:
            data = self._load_raw_data(idx)
            image = data['image']
            boundary_label = data['boundary_label']
            room_label = data['room_label']
        
        # Apply transformations
        try:
            # Apply spatial transforms to image and masks together
            spatial_transformed = self.spatial_transform(
                image=image, 
                boundary=boundary_label, 
                room=room_label
            )
            
            # Apply image-only transforms
            image_transformed = self.image_transform(image=spatial_transformed['image'])
            
            image = image_transformed['image']
            boundary_label = torch.from_numpy(spatial_transformed['boundary']).long()
            room_label = torch.from_numpy(spatial_transformed['room']).long()
            
        except Exception as e:
            print(f"Transform error for idx {idx}: {e}")
            # Fallback to simple resize - ensure we're working with numpy arrays
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            if isinstance(boundary_label, torch.Tensor):
                boundary_label = boundary_label.cpu().numpy()
            if isinstance(room_label, torch.Tensor):
                room_label = room_label.cpu().numpy()
                
            image = cv2.resize(image, (self.config.INPUT_SIZE, self.config.INPUT_SIZE))
            boundary_label = cv2.resize(boundary_label, (self.config.INPUT_SIZE, self.config.INPUT_SIZE))
            room_label = cv2.resize(room_label, (self.config.INPUT_SIZE, self.config.INPUT_SIZE))
            
            # Convert to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            boundary_label = torch.from_numpy(boundary_label).long()
            room_label = torch.from_numpy(room_label).long()
        
        result = (image, boundary_label, room_label)
        
        # Cache in memory for frequently accessed items (no lock needed in multiprocessing)
        if len(self.memory_cache) < 50:  # Reduced cache size for multiprocessing
            self.memory_cache[idx] = result
        
        return result


# Enhanced config for faster training
class OptimizedConfig:
    def __init__(self):
        # Model parameters
        self.NUM_BOUNDARY_CLASSES = 3
        self.NUM_ROOM_CLASSES = 10
        self.INPUT_SIZE = 256  # Reduced from 512 for faster training
        
        # Training parameters - optimized for speed
        self.BATCH_SIZE = 16  # Increase if GPU memory allows
        self.NUM_EPOCHS = 50
        self.LEARNING_RATE = 3e-4  # Slightly higher for faster convergence
        
        # Loss weights
        self.BOUNDARY_WEIGHT = 1.0
        self.ROOM_WEIGHT = 1.0
        
        # Hardware optimization - Windows multiprocessing fix
        # Use fewer workers on Windows to avoid pickle issues
        self.NUM_WORKERS = 0 if os.name == 'nt' else min(4, os.cpu_count())
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Memory optimization
        self.PIN_MEMORY = torch.cuda.is_available()
        self.PERSISTENT_WORKERS = False if os.name == 'nt' else True  # Disable on Windows
        
        # Early stopping parameters
        self.PATIENCE = 10
        self.MIN_DELTA = 0.001


def get_optimal_dataloader_config():
    """Get optimal DataLoader configuration for current platform"""
    if os.name == 'nt':  # Windows
        return {
            'num_workers': 0,  # Use main thread on Windows
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': False,
            'prefetch_factor': None  # Only available with num_workers > 0
        }
    else:  # Linux/Mac
        return {
            'num_workers': min(4, os.cpu_count()),
            'pin_memory': torch.cuda.is_available(),
            'persistent_workers': True,
            'prefetch_factor': 2
        }


def benchmark_data_loading():
    """Benchmark different data loading configurations"""
    config = OptimizedConfig()
    
    print("🚀 Benchmarking data loading configurations...")
    
    # Get platform-specific optimal config
    optimal_config = get_optimal_dataloader_config()
    
    # Test configurations suitable for Windows
    if os.name == 'nt':
        configurations = [
            {'batch_size': 8, 'num_workers': 0, 'pin_memory': False},
            {'batch_size': 16, 'num_workers': 0, 'pin_memory': True},
            {'batch_size': 32, 'num_workers': 0, 'pin_memory': True},
        ]
    else:
        configurations = [
            {'batch_size': 8, 'num_workers': 2, 'pin_memory': False},
            {'batch_size': 16, 'num_workers': 4, 'pin_memory': True},
            {'batch_size': 32, 'num_workers': 8, 'pin_memory': True},
            {'batch_size': 16, 'num_workers': 8, 'pin_memory': True, 'persistent_workers': True},
        ]
    
    for i, cfg in enumerate(configurations):
        print(f"\nTesting configuration {i+1}: {cfg}")
        
        try:
            dataset = FloorplanDataset('data/data/processed/train', config, is_train=True)
            loader = DataLoader(dataset, **cfg)
            
            start_time = time.time()
            for j, batch in enumerate(loader):
                if j >= 10:  # Test 10 batches
                    break
            
            avg_time = (time.time() - start_time) / 10
            print(f"Average time per batch: {avg_time:.3f}s")
            
        except Exception as e:
            print(f"Configuration failed: {e}")


if __name__ == "__main__":
    # Run benchmark first to find optimal settings
    benchmark_data_loading()