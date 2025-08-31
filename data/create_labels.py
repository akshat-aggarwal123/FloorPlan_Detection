import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import ndimage

class ArchitecturalFloorplanProcessor:
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
            
    def preprocess_architectural_drawing(self, image_path):
        """
        Preprocess architectural floorplan drawing
        Handles: black lines on white background, text, curved elements
        """
        # Read image in color first to handle any blue marks
        img_color = cv2.imread(image_path)
        if img_color is None:
            return None, None
            
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # Remove blue pen marks (if any) by analyzing color channels
        if len(img_color.shape) == 3:
            # Create mask for blue areas (blue pen marks)
            hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
            blue_lower = np.array([100, 50, 50])
            blue_upper = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
            
            # Remove blue areas by setting them to white
            img_gray[blue_mask > 0] = 255
        
        # Adaptive thresholding to handle varying lighting/scanning conditions
        binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Invert so that lines (walls) are white and background is black
        binary_inverted = cv2.bitwise_not(binary)
        
        # Clean up noise and small artifacts
        kernel_small = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary_inverted, cv2.MORPH_OPEN, kernel_small)
        
        # Connect nearby line segments
        kernel_connect = np.ones((3,3), np.uint8)
        connected = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_connect)
        
        return connected, img_gray
    
    def remove_text_from_floorplan(self, binary_img):
        """
        Remove text elements from the floorplan
        Text usually appears as small connected components
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
        
        # Create mask without text
        no_text_mask = np.zeros_like(binary_img)
        
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Filter out text-like components
            # Text is usually small, has certain aspect ratios
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
            
            # Keep components that are likely walls/structure
            # Remove small isolated components (likely text)
            if area > 50 and (area > 200 or aspect_ratio > 3):  # Keep long lines or large areas
                component_mask = (labels == i).astype(np.uint8)
                no_text_mask = cv2.bitwise_or(no_text_mask, component_mask)
        
        return no_text_mask
    
    def detect_doors_and_openings(self, wall_mask, original_gray):
        """
        Detect doors and openings in the floorplan
        Uses gap analysis and curve detection
        """
        door_mask = np.zeros_like(wall_mask)
        window_mask = np.zeros_like(wall_mask)
        
        # Method 1: Find gaps in walls (doorways)
        # Dilate walls slightly to find where they should connect
        kernel = np.ones((5,5), np.uint8)
        dilated_walls = cv2.dilate(wall_mask, kernel, iterations=1)
        
        # Find the difference (potential door openings)
        potential_openings = dilated_walls - wall_mask
        
        # Find contours of openings
        contours, _ = cv2.findContours(potential_openings, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
            
            # Door criteria: medium size, rectangular
            if 50 < area < 400 and 1.5 < aspect_ratio < 6:
                cv2.fillPoly(door_mask, [contour], 1)
            # Window criteria: smaller, more square
            elif 20 < area < 150 and 1 < aspect_ratio < 3:
                cv2.fillPoly(window_mask, [contour], 1)
        
        # Method 2: Detect curved elements (door swings) using Hough circles
        circles = cv2.HoughCircles(original_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                 param1=50, param2=30, minRadius=10, maxRadius=50)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Small circles likely represent door swings
                if 10 < r < 40:
                    cv2.circle(door_mask, (x, y), r//2, 1, -1)
        
        return door_mask, window_mask
    
    def identify_room_regions(self, wall_mask):
        """
        Identify room regions from wall boundaries
        Uses flood fill and morphological operations
        """
        # Invert walls to get open spaces
        open_space = 1 - wall_mask
        
        # Remove small holes and noise
        kernel = np.ones((5,5), np.uint8)
        open_space_cleaned = cv2.morphologyEx(open_space.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        open_space_cleaned = cv2.morphologyEx(open_space_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components (rooms)
        num_labels, labels = cv2.connectedComponents(open_space_cleaned)
        
        room_mask = np.zeros_like(wall_mask, dtype=np.uint8)
        
        # Process each potential room
        room_id = 1
        for label_id in range(1, num_labels):  # Skip background
            component_mask = (labels == label_id)
            area = np.sum(component_mask)
            
            # Only consider areas large enough to be rooms
            if area > 1000:  # Minimum room area (adjust based on your image resolution)
                room_mask[component_mask] = room_id
                room_id = min(room_id + 1, 4)  # Cap at 4 room types for now
        
        return room_mask
    
    def split_dataset(self, source_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """Split dataset into train/val/test sets"""
        print(f"Splitting dataset from {source_dir}")
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            image_files.extend(Path(source_dir).glob(ext))
            image_files.extend(Path(source_dir).glob(ext.upper()))
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("No images found! Check your source directory path.")
            return None
        
        # Convert to list of strings
        image_files = [str(f) for f in image_files]
        
        # Shuffle and split
        random.shuffle(image_files)
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
            filename = os.path.basename(file_path)
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(file_path, dest_path)
            
            if i % 100 == 0:
                print(f"Copied {i+1}/{len(file_list)} files to {split_name}")
    
    def create_boundary_labels(self, split_name):
        """Create boundary labels for architectural drawings"""
        images_dir = f'data/processed/{split_name}/images'
        boundary_dir = f'data/processed/{split_name}/boundary_labels'
        
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing {len(image_files)} images for {split_name} boundary labels...")
        
        processed_count = 0
        failed_count = 0
        
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(images_dir, img_file)
            
            try:
                # Preprocess the architectural drawing
                wall_mask, original_gray = self.preprocess_architectural_drawing(img_path)
                if wall_mask is None:
                    failed_count += 1
                    continue
                
                # Remove text elements
                wall_mask_clean = self.remove_text_from_floorplan(wall_mask)
                
                # Detect doors and windows
                door_mask, window_mask = self.detect_doors_and_openings(wall_mask_clean, original_gray)
                
                # Create combined boundary mask
                # 0=background, 1=wall, 2=door, 3=window
                boundary_mask = np.zeros_like(wall_mask_clean, dtype=np.uint8)
                boundary_mask[wall_mask_clean > 0] = 1  # walls
                boundary_mask[door_mask > 0] = 2         # doors
                boundary_mask[window_mask > 0] = 3       # windows
                
                # Save boundary label
                label_name = os.path.splitext(img_file)[0] + '_boundary.png'
                label_path = os.path.join(boundary_dir, label_name)
                cv2.imwrite(label_path, boundary_mask)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                failed_count += 1
                continue
            
            if i % 50 == 0:
                print(f"Processed {i+1}/{len(image_files)} boundary labels for {split_name}")
        
        print(f"Successfully processed {processed_count} boundary labels for {split_name}")
        print(f"Failed: {failed_count} images")
        return processed_count
    
    def create_room_labels_from_walls(self, split_name):
        """Create room labels by identifying enclosed spaces"""
        images_dir = f'data/processed/{split_name}/images'
        boundary_dir = f'data/processed/{split_name}/boundary_labels'
        room_dir = f'data/processed/{split_name}/room_labels'
        
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Creating {len(image_files)} room labels for {split_name}...")
        
        processed_count = 0
        
        for i, img_file in enumerate(image_files):
            try:
                # Load the boundary mask we created
                boundary_name = os.path.splitext(img_file)[0] + '_boundary.png'
                boundary_path = os.path.join(boundary_dir, boundary_name)
                
                if not os.path.exists(boundary_path):
                    print(f"Warning: No boundary file for {img_file}")
                    continue
                
                boundary_mask = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
                
                # Extract wall mask (class 1 from boundary mask)
                wall_mask = (boundary_mask == 1).astype(np.uint8)
                
                # Create room segments
                room_mask = self.identify_room_regions(wall_mask)
                
                # Save room label
                label_name = os.path.splitext(img_file)[0] + '_room.png'
                label_path = os.path.join(room_dir, label_name)
                cv2.imwrite(label_path, room_mask)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error creating room labels for {img_file}: {e}")
                continue
            
            if i % 50 == 0:
                print(f"Created {i+1}/{len(image_files)} room labels for {split_name}")
        
        print(f"Successfully created {processed_count} room labels for {split_name}")
        return processed_count
    
    def analyze_sample_image(self, image_path):
        """Analyze a sample image to understand its characteristics"""
        print(f"\nAnalyzing sample image: {image_path}")
        
        img_color = cv2.imread(image_path)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        print(f"Image shape: {img_gray.shape}")
        print(f"Pixel value range: {img_gray.min()} to {img_gray.max()}")
        
        # Analyze pixel distribution
        unique_vals, counts = np.unique(img_gray, return_counts=True)
        print(f"Most common pixel values: {unique_vals[:10]}")
        print(f"Pixel value distribution (top 5): {list(zip(unique_vals[:5], counts[:5]))}")
        
        # Check if it's mostly binary
        binary_pixels = np.sum((img_gray < 50) | (img_gray > 200))
        total_pixels = img_gray.size
        binary_ratio = binary_pixels / total_pixels
        print(f"Binary-like pixels: {binary_ratio:.2%}")
        
        return img_gray
    
    def analyze_labels(self, split_name='train', num_check=5):
        """Analyze the created labels to check quality"""
        boundary_dir = f'data/processed/{split_name}/boundary_labels'
        room_dir = f'data/processed/{split_name}/room_labels'
        
        if not os.path.exists(boundary_dir):
            print(f"No boundary directory found for {split_name}")
            return
        
        boundary_files = [f for f in os.listdir(boundary_dir) if f.endswith('.png')]
        room_files = [f for f in os.listdir(room_dir) if f.endswith('.png')] if os.path.exists(room_dir) else []
        
        print(f"\n=== Label Analysis for {split_name} ===")
        print(f"Boundary label files: {len(boundary_files)}")
        print(f"Room label files: {len(room_files)}")
        
        # Analyze a few files
        for i in range(min(num_check, len(boundary_files))):
            boundary_path = os.path.join(boundary_dir, boundary_files[i])
            
            # Load and analyze boundary labels
            boundary_img = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
            
            if boundary_img is not None:
                boundary_unique, boundary_counts = np.unique(boundary_img, return_counts=True)
                
                print(f"\nFile {i+1} - {boundary_files[i]}:")
                print(f"  Boundary classes: {boundary_unique}")
                
                # Calculate percentages
                total_pixels = boundary_img.size
                for val, count in zip(boundary_unique, boundary_counts):
                    pct = (count / total_pixels) * 100
                    class_name = {0: 'background', 1: 'wall', 2: 'door', 3: 'window'}.get(val, 'unknown')
                    print(f"    Class {val} ({class_name}): {count} pixels ({pct:.1f}%)")
            
            # Analyze room labels if they exist
            if i < len(room_files):
                room_path = os.path.join(room_dir, room_files[i])
                room_img = cv2.imread(room_path, cv2.IMREAD_GRAYSCALE)
                
                if room_img is not None:
                    room_unique, room_counts = np.unique(room_img, return_counts=True)
                    print(f"  Room classes: {room_unique}")
                    
                    for val, count in zip(room_unique, room_counts):
                        pct = (count / total_pixels) * 100
                        class_name = {0: 'background', 1: 'room'}.get(val, f'room_type_{val}')
                        print(f"    Class {val} ({class_name}): {count} pixels ({pct:.1f}%)")
    
    def visualize_sample(self, split_name='train', num_samples=3):
        """Visualize processed samples"""
        images_dir = f'data/processed/{split_name}/images'
        boundary_dir = f'data/processed/{split_name}/boundary_labels'
        room_dir = f'data/processed/{split_name}/room_labels'
        
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:num_samples]
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_file in enumerate(image_files):
            # Load original image
            img_path = os.path.join(images_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Load boundary label
            boundary_name = os.path.splitext(img_file)[0] + '_boundary.png'
            boundary_path = os.path.join(boundary_dir, boundary_name)
            
            # Load room label
            room_name = os.path.splitext(img_file)[0] + '_room.png'
            room_path = os.path.join(room_dir, room_name)
            
            # Plot original
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f'Original: {img_file}')
            axes[i, 0].axis('off')
            
            # Plot boundary labels
            if os.path.exists(boundary_path):
                boundary_label = cv2.imread(boundary_path, cv2.IMREAD_GRAYSCALE)
                axes[i, 1].imshow(boundary_label, cmap='tab10', vmin=0, vmax=3)
                axes[i, 1].set_title('Boundaries: 0=bg, 1=wall, 2=door, 3=window')
                axes[i, 1].axis('off')
            
            # Plot room labels
            if os.path.exists(room_path):
                room_label = cv2.imread(room_path, cv2.IMREAD_GRAYSCALE)
                axes[i, 2].imshow(room_label, cmap='tab10', vmin=0, vmax=4)
                axes[i, 2].set_title('Rooms: 0=bg, 1-4=different rooms')
                axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('architectural_dataset_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Sample visualization saved as 'architectural_dataset_samples.png'")

def main():
    processor = ArchitecturalFloorplanProcessor()
    
    print("=== Architectural Floorplan Dataset Processing ===")
    print("Designed for: black line drawings on white background")
    print("Handles: text removal, curved elements, door/window detection")
    
    source_dir = input("\nEnter path to your 3000 images directory (or press Enter for 'data/raw/all_images'): ").strip()
    if not source_dir:
        source_dir = 'data/raw/all_images'
    
    if not os.path.exists(source_dir):
        print(f"Directory {source_dir} does not exist!")
        print("Please create it and put your floorplan images there.")
        return
    
    # Analyze a sample image first
    sample_files = list(Path(source_dir).glob('*'))[:1]
    if sample_files:
        processor.analyze_sample_image(str(sample_files[0]))
        
        proceed = input("\nDoes the analysis look correct? Proceed with processing? (y/n): ").strip().lower()
        if proceed != 'y':
            print("Processing cancelled. Check your images and try again.")
            return
    
    # Step 1: Split dataset
    print(f"\nStep 1: Splitting dataset from {source_dir}...")
    split_result = processor.split_dataset(source_dir)
    
    if split_result is None:
        print("No images found to process!")
        return
    
    train_count, val_count, test_count = split_result
    
    # Step 2: Create boundary labels
    print("\nStep 2: Creating boundary labels (walls, doors, windows)...")
    boundary_train = processor.create_boundary_labels('train')
    boundary_val = processor.create_boundary_labels('val')
    boundary_test = processor.create_boundary_labels('test')
    
    # Step 3: Create room labels from wall structure
    print("\nStep 3: Creating room labels from enclosed spaces...")
    room_train = processor.create_room_labels_from_walls('train')
    room_val = processor.create_room_labels_from_walls('val')
    room_test = processor.create_room_labels_from_walls('test')
    
    # Step 4: Analyze results
    print("\nStep 4: Analyzing label quality...")
    processor.analyze_labels('train')
    
    # Step 5: Visualize samples
    print("\nStep 5: Creating sample visualizations...")
    processor.visualize_sample('train', 3)
    
    print("\n=== Processing Complete! ===")
    print(f"Dataset split: Train={train_count}, Val={val_count}, Test={test_count}")
    print(f"Boundary labels: Train={boundary_train}, Val={boundary_val}, Test={boundary_test}")
    print(f"Room labels: Train={room_train}, Val={room_val}, Test={room_test}")
    
    print("\n=== Quality Check ===")
    print("1. Check 'architectural_dataset_samples.png' for visual verification")
    print("2. Look at the label analysis above")
    print("3. Wall detection should be good")
    print("4. Door/window detection is heuristic-based")
    print("5. Room detection identifies enclosed spaces only")
    
    print("\n=== Next Steps ===")
    print("1. If labels look good, update your config.py")
    print("2. Consider boundary-only training first (set ROOM_WEIGHT=0.1)")
    print("3. Delete any existing cache files before training")
    print("4. Run: python train.py")
    
    print("\n=== Recommendations ===")
    print("- Start with boundary detection only (more reliable)")
    print("- Room detection will be basic (just identifies spaces)")
    print("- Consider manual annotation for better room classification")

if __name__ == "__main__":
    main()