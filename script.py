import torch
import os
import numpy as np
from data.datasets import FloorplanDataset
from config import Config
import matplotlib.pyplot as plt
from PIL import Image

def check_raw_data_files():
    """Check the raw data files in your processed directory"""
    train_dir = 'data/data/processed/train'
    val_dir = 'data/data/processed/val'
    
    print("🔍 Checking raw data files...")
    
    for split, data_dir in [('Train', train_dir), ('Val', val_dir)]:
        print(f"\n{split} directory: {data_dir}")
        
        if not os.path.exists(data_dir):
            print(f"❌ Directory doesn't exist: {data_dir}")
            continue
            
        # List subdirectories
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        print(f"   Subdirectories: {subdirs}")
        
        # Check each subdirectory for files
        for subdir in subdirs:
            subdir_path = os.path.join(data_dir, subdir)
            files = os.listdir(subdir_path)
            print(f"   {subdir}: {len(files)} files")
            
            # Show first few files
            if files:
                print(f"     Sample files: {files[:3]}")
                
                # Check first file
                first_file = os.path.join(subdir_path, files[0])
                try:
                    if subdir in ['images', 'boundary', 'room']:
                        img = Image.open(first_file)
                        img_array = np.array(img)
                        print(f"     First file shape: {img_array.shape}, dtype: {img_array.dtype}")
                        print(f"     Value range: [{img_array.min()}, {img_array.max()}]")
                        print(f"     Unique values: {np.unique(img_array)[:10]}...")  # First 10 unique values
                except Exception as e:
                    print(f"     Error reading {first_file}: {e}")

def check_dataset_class():
    """Check what the FloorplanDataset class is actually loading"""
    print("\n🔍 Checking FloorplanDataset class...")
    
    config = Config()
    
    try:
        # Create dataset without cache to see what's happening
        train_dataset = FloorplanDataset('data/data/processed/train', config, is_train=True)
        
        print(f"Dataset length: {len(train_dataset)}")
        
        # Check first 10 samples
        print("\nChecking first 10 samples:")
        for i in range(min(10, len(train_dataset))):
            try:
                image, boundary, room = train_dataset[i]
                
                print(f"Sample {i}:")
                print(f"  Image: {image.shape}, range=[{image.min():.3f}, {image.max():.3f}]")
                print(f"  Boundary: {boundary.shape}, unique={torch.unique(boundary).tolist()}")
                print(f"  Room: {room.shape}, unique={torch.unique(room).tolist()}")
                
                # Check for problems
                if len(torch.unique(room)) == 1:
                    print(f"  ❌ Sample {i}: Room has only one class!")
                if torch.unique(room).max() >= config.NUM_ROOM_CLASSES:
                    print(f"  ❌ Sample {i}: Room class exceeds NUM_ROOM_CLASSES!")
                    
            except Exception as e:
                print(f"  ❌ Error loading sample {i}: {e}")
                
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()

def visualize_sample_data():
    """Visualize a sample to see what the data looks like"""
    print("\n🔍 Visualizing sample data...")
    
    config = Config()
    
    try:
        train_dataset = FloorplanDataset('data/data/processed/train', config, is_train=True)
        
        # Get first sample
        image, boundary, room = train_dataset[0]
        
        # Convert to numpy for visualization
        image_np = image.permute(1, 2, 0).cpu().numpy()
        boundary_np = boundary.cpu().numpy()
        room_np = room.cpu().numpy()
        
        # Normalize image for display
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(boundary_np, cmap='viridis')
        axes[1].set_title(f'Boundary Labels\nUnique: {np.unique(boundary_np)}')
        axes[1].axis('off')
        
        axes[2].imshow(room_np, cmap='tab10')
        axes[2].set_title(f'Room Labels\nUnique: {np.unique(room_np)}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_visualization.png', dpi=150, bbox_inches='tight')
        print("✅ Sample visualization saved as 'sample_visualization.png'")
        
        # Print detailed stats
        print(f"\nDetailed statistics:")
        print(f"  Image stats: mean={image_np.mean():.3f}, std={image_np.std():.3f}")
        print(f"  Boundary distribution: {dict(zip(*np.unique(boundary_np, return_counts=True)))}")
        print(f"  Room distribution: {dict(zip(*np.unique(room_np, return_counts=True)))}")
        
    except Exception as e:
        print(f"❌ Error visualizing data: {e}")
        import traceback
        traceback.print_exc()

def check_cached_data():
    """Check the cached data files"""
    print("\n🔍 Checking cached data...")
    
    cache_files = [
        'data/data/processed/train/cache/cached_data_train.pkl',
        'data/data/processed/val/cache/cached_data_val.pkl'
    ]
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                # Get file size
                file_size = os.path.getsize(cache_file) / (1024*1024)  # MB
                print(f"✅ {cache_file}: {file_size:.1f} MB")
                
                # Try to load and check a sample
                import pickle
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                print(f"   Cached data type: {type(data)}")
                if hasattr(data, '__len__'):
                    print(f"   Cached data length: {len(data)}")
                    
                    # Check first item if it's a list/tuple
                    if len(data) > 0:
                        first_item = data[0]
                        print(f"   First item type: {type(first_item)}")
                        if isinstance(first_item, (list, tuple)) and len(first_item) >= 3:
                            _, boundary, room = first_item[:3]
                            if hasattr(boundary, 'shape') and hasattr(room, 'shape'):
                                print(f"   Sample shapes: Boundary={boundary.shape}, Room={room.shape}")
                                print(f"   Sample unique values: Boundary={np.unique(boundary)[:5]}, Room={np.unique(room)}")
                
            except Exception as e:
                print(f"❌ Error reading {cache_file}: {e}")
        else:
            print(f"❌ Cache file doesn't exist: {cache_file}")

def suggest_fixes():
    """Suggest potential fixes based on the diagnosis"""
    print("\n" + "="*50)
    print("DIAGNOSIS SUMMARY & SUGGESTED FIXES")
    print("="*50)
    
    print("\n🔍 ISSUE IDENTIFIED:")
    print("Your room labels contain only class 0 (background), which means:")
    print("- No room pixels are labeled in your dataset")
    print("- The model cannot learn room segmentation")
    print("- This explains the stuck loss and poor metrics")
    
    print("\n🛠️  POTENTIAL FIXES:")
    print("1. CHECK DATA PREPROCESSING:")
    print("   - Verify your data preprocessing pipeline")
    print("   - Make sure room annotations are being converted to labels correctly")
    print("   - Check if room pixels should be labeled as classes 1-4")
    
    print("\n2. DELETE CACHE AND REGENERATE:")
    print("   - Delete cache files: rm data/data/processed/*/cache/*.pkl")
    print("   - Regenerate dataset to ensure proper label creation")
    
    print("\n3. CHECK ORIGINAL ANNOTATIONS:")
    print("   - Verify your source annotations have room information")
    print("   - Check if you need separate room annotation files")
    
    print("\n4. MODIFY FLOORPLANDATASET CLASS:")
    print("   - Add debug prints to see what's happening during label loading")
    print("   - Verify the room label generation logic")
    
    print("\n5. TEMPORARY WORKAROUND - BOUNDARY ONLY:")
    print("   - Train only on boundary detection first")
    print("   - Set ROOM_WEIGHT = 0.0 in config")
    print("   - Focus on getting boundary segmentation working")

if __name__ == "__main__":
    check_raw_data_files()
    check_dataset_class() 
    check_cached_data()
    visualize_sample_data()
    suggest_fixes()