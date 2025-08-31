import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

class FloorplanProcessor:
    def __init__(self):
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = [
            'data/raw/all_images',
            'data/processed/train/images',
            'data/processed/train/boundary_labels',
            'data/processed/train/room_labels',
            'data/processed/val/images', 
            'data/processed/val/boundary_labels',
            'data/processed/val/room_labels',
            'data/processed/test/images',
            'data/processed/test/boundary_labels', 
            'data/processed/test/room_labels'
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def split_dataset(self, source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split 3000 images into train/val/test sets
        """
        print(f"Splitting dataset from {source_dir}")
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            image_files.extend(Path(source_dir).glob(ext))
            image_files.extend(Path(source_dir).glob(ext.upper()))
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("No images found! Check your source directory path.")
            return
        
        # Convert to list of strings
        image_files = [str(f) for f in image_files]
        
        # Split dataset
        train_files, temp_files = train_test_split(image_files, train_size=train_ratio, random_state=42)
        val_files, test_files = train_test_split(temp_files, train_size=val_ratio/(val_ratio+test_ratio), random_state=42)
        
        print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Copy files to respective directories
        self._copy_files_to_split(train_files, 'train')
        self._copy_files_to_split(val_files, 'val') 
        self._copy_files_to_split(test_files, 'test')
        
        return len(train_files), len(val_files), len(test_files)
    
    def _copy_files_to_split(self, file_list, split_name):
        """Copy files to train/val/test directories"""
        dest_dir = f'data/processed/{split_name}/images'
        
        for i, file_path in enumerate(file_list):
            # Get filename
            filename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, filename)
            
            # Copy image
            shutil.copy2(file_path, dest_path)
            
            if i % 100 == 0:
                print(f"Copied {i+1}/{len(file_list)} files to {split_name}")
    
    def extract_walls_from_bw_image(self, image_path):
        """
        Extract walls from black and white floorplan
        Assumes: Black = background, White = walls/structure
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Threshold to ensure pure black/white
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        
        # Create wall mask (white areas = walls)
        wall_mask = (binary > 127).astype(np.uint8)
        
        # Clean up noise
        kernel = np.ones((3,3), np.uint8)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kernel)
        
        return wall_mask
    
    def detect_doors_windows_from_walls(self, wall_mask):
        """
        Detect potential doors and windows from wall structure
        This is heuristic-based for black/white images
        """
        # Find wall boundaries
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        door_mask = np.zeros_like(wall_mask)
        window_mask = np.zeros_like(wall_mask)
        
        # Look for gaps/openings in walls (potential doors)
        # This is a simple heuristic - you may need to adjust
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Small rectangular regions might be doors/windows
            area = cv2.contourArea(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Heuristics for doors (tall and narrow)
            if 50 < area < 500 and 0.2 < aspect_ratio < 0.8:
                cv2.fillPoly(door_mask, [contour], 1)
            
            # Heuristics for windows (more square)
            elif 30 < area < 300 and 0.5 < aspect_ratio < 2.0:
                cv2.fillPoly(window_mask, [contour], 1)
        
        return door_mask, window_mask
    
    def create_boundary_labels(self, split_name):
        """Create boundary labels for a dataset split"""
        images_dir = f'data/processed/{split_name}/images'
        boundary_dir = f'data/processed/{split_name}/boundary_labels'
        
        image_files = os.listdir(images_dir)
        print(f"Processing {len(image_files)} images for {split_name} boundary labels...")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(images_dir, img_file)
            
            # Extract wall mask
            wall_mask = self.extract_walls_from_bw_image(img_path)
            if wall_mask is None:
                continue
            
            # Detect doors and windows
            door_mask, window_mask = self.detect_doors_windows_from_walls(wall_mask)
            
            # Create combined boundary mask
            # 0=background, 1=wall, 2=door, 3=window
            boundary_mask = np.zeros_like(wall_mask, dtype=np.uint8)
            boundary_mask[wall_mask == 1] = 1  # walls
            boundary_mask[door_mask == 1] = 2   # doors
            boundary_mask[window_mask == 1] = 3 # windows
            
            # Save boundary label
            label_name = img_file.replace('.jpg', '_boundary.png').replace('.jpeg', '_boundary.png').replace('.png', '_boundary.png')
            label_path = os.path.join(boundary_dir, label_name)
            cv2.imwrite(label_path, boundary_mask)
            
            if i % 100 == 0:
                print(f"Processed {i+1}/{len(image_files)} boundary labels for {split_name}")
    
    def create_room_labels_placeholder(self, split_name):
        """
        Create placeholder room labels
        For black/white images, we can't automatically detect room types
        These will need manual annotation later
        """
        images_dir = f'data/processed/{split_name}/images'
        room_dir = f'data/processed/{split_name}/room_labels'
        
        image_files = os.listdir(images_dir)
        print(f"Creating {len(image_files)} placeholder room labels for {split_name}...")
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            # Create empty room mask (all background for now)
            room_mask = np.zeros_like(img, dtype=np.uint8)
            
            # You could add some heuristics here to identify large open areas as rooms
            # For now, keeping as placeholder
            
            # Save room label
            label_name = img_file.replace('.jpg', '_room.png').replace('.jpeg', '_room.png').replace('.png', '_room.png')
            label_path = os.path.join(room_dir, label_name)
            cv2.imwrite(label_path, room_mask)
            
            if i % 100 == 0:
                print(f"Created {i+1}/{len(image_files)} room labels for {split_name}")
    
    def visualize_sample(self, split_name='train', num_samples=3):
        """Visualize some processed samples"""
        import matplotlib.pyplot as plt
        
        images_dir = f'data/processed/{split_name}/images'
        boundary_dir = f'data/processed/{split_name}/boundary_labels'
        
        image_files = os.listdir(images_dir)[:num_samples]
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_file in enumerate(image_files):
            # Load original image
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Load boundary label
            label_name = img_file.replace('.jpg', '_boundary.png').replace('.jpeg', '_boundary.png').replace('.png', '_boundary.png')
            label_path = os.path.join(boundary_dir, label_name)
            boundary_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            # Plot
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Original: {img_file}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(boundary_label, cmap='tab10')
            axes[i, 1].set_title('Boundary Labels (0=bg, 1=wall, 2=door, 3=window)')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Sample visualization saved as 'dataset_samples.png'")

def main():
    processor = FloorplanProcessor()
    
    print("=== Floorplan Dataset Processing ===")
    print("\nStep 1: Place your 3000 images in 'data/raw/all_images/' directory")
    
    source_dir = input("\nEnter path to your 3000 images directory (or press Enter for 'data/raw/all_images'): ").strip()
    if not source_dir:
        source_dir = 'data/raw/all_images'
    
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist!")
        print("Please create it and put your images there.")
        return
    
    # Step 1: Split dataset
    print(f"\nStep 2: Splitting dataset from {source_dir}...")
    train_count, val_count, test_count = processor.split_dataset(source_dir)
    
    if train_count == 0:
        print("No images found to process!")
        return
    
    # Step 2: Create boundary labels
    print("\nStep 3: Creating boundary labels...")
    processor.create_boundary_labels('train')
    processor.create_boundary_labels('val')
    processor.create_boundary_labels('test')
    
    # Step 3: Create placeholder room labels
    print("\nStep 4: Creating placeholder room labels...")
    processor.create_room_labels_placeholder('train')
    processor.create_room_labels_placeholder('val')
    processor.create_room_labels_placeholder('test')
    
    # Step 4: Visualize samples
    print("\nStep 5: Creating sample visualizations...")
    processor.visualize_sample('train', 3)
    
    print("\n=== Processing Complete! ===")
    print(f"Dataset split: Train={train_count}, Val={val_count}, Test={test_count}")
    print("\nNext steps:")
    print("1. Check 'dataset_samples.png' to verify boundary detection quality")
    print("2. For better results, manually refine some boundary labels")
    print("3. Create proper room type annotations (currently all background)")
    print("4. Run: python train.py")

if __name__ == "__main__":
    main()