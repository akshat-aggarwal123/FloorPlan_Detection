import torch
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from models.multi_task_net import DeepFloorplanNet
from utils.preprocessing import FloorplanPreprocessor
from utils.postprocessing import FloorplanPostprocessor
from utils.json_converter import FloorplanJSONConverter
from config import Config

class FloorplanInference:
    def __init__(self, model_path):
        self.config = Config()
        self.device = self.config.DEVICE
        
        # Load model
        self.model = DeepFloorplanNet(
            self.config.NUM_BOUNDARY_CLASSES,
            self.config.NUM_ROOM_CLASSES
        ).to(self.device)
        
        # 🔥 Load checkpoint properly
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)  # fallback if only raw weights saved

        self.model.eval()
        
        # Initialize processors
        self.preprocessor = FloorplanPreprocessor(self.config.INPUT_SIZE)
        self.postprocessor = FloorplanPostprocessor(self.config)
        self.json_converter = FloorplanJSONConverter(self.config)
        
        # Color mappings for visualization
        self.room_colors = {
            'bedroom': '#FFB6C1',      # Light pink
            'bathroom': '#87CEEB',     # Sky blue
            'kitchen': '#98FB98',      # Pale green
            'living_room': '#F0E68C',  # Khaki
            'dining_room': '#DDA0DD',  # Plum
            'closet': '#D2691E',       # Chocolate
            'balcony': '#20B2AA',      # Light sea green
            'corridor': '#F5DEB3',     # Wheat
            'laundry': '#FF6347',      # Tomato
            'office': '#9370DB',       # Medium purple
            'other': '#DCDCDC'         # Gainsboro
        }
        
        self.boundary_colors = {
            'wall': '#000000',         # Black
            'door': '#FF0000',         # Red
            'window': '#0000FF'        # Blue
        }
    
    def predict(self, image_path, output_path, visualize=True, viz_output_path=None):
        """Run inference on a single floorplan image"""
        
        # Preprocess image
        input_tensor = self.preprocessor.preprocess_image(image_path)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            boundary_pred = torch.softmax(outputs['boundary'], dim=1)
            room_pred = torch.softmax(outputs['room'], dim=1)
        
        # Convert to numpy
        boundary_mask = torch.argmax(boundary_pred, dim=1)[0].cpu().numpy()
        room_mask = torch.argmax(room_pred, dim=1)[0].cpu().numpy()
        
        # Print debug info
        print(f"Boundary mask shape: {boundary_mask.shape}, unique values: {np.unique(boundary_mask)}")
        print(f"Room mask shape: {room_mask.shape}, unique values: {np.unique(room_mask)}")
        
        # Post-process predictions
        wall_mask = (boundary_mask == 1).astype(float)
        walls = self.postprocessor.extract_walls_as_lines(wall_mask)
        corners = self.postprocessor.extract_corners_from_walls(walls)
        doors, windows = self.postprocessor.extract_doors_and_windows(boundary_mask)
        rooms = self.postprocessor.extract_rooms(room_mask)
        
        # Debug extracted elements
        print(f"Extracted {len(walls)} walls, {len(corners)} corners, {len(doors)} doors, {len(windows)} windows, {len(rooms)} rooms")
        
        # Convert to simple JSON format for visualization
        simple_json = self._create_simple_json_format(walls, corners, doors, windows, rooms)
        
        # Convert to original complex JSON format
        json_output = self.json_converter.convert_to_json(
            walls, corners, doors, windows, rooms
        )
        
        # Save original JSON output
        with open(output_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"Prediction saved to {output_path}")
        
        # Generate visualization if requested
        if visualize:
            if viz_output_path is None:
                viz_output_path = output_path.replace('.json', '_visualization.png')
            
            self.visualize_predictions(
                image_path, simple_json, boundary_mask, room_mask, viz_output_path
            )
            print(f"Visualization saved to {viz_output_path}")
        
        return json_output
    
    def visualize_predictions(self, original_image_path, json_data, boundary_mask, room_mask, output_path):
        """Create comprehensive visualization of predictions"""
        
        # Load original image
        original_img = Image.open(original_image_path).convert('RGB')
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Floorplan Analysis Results', fontsize=16, fontweight='bold')
        
        # 1. Original Image
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('Original Floorplan', fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. Boundary Predictions
        self._plot_boundary_mask(axes[0, 1], boundary_mask)
        axes[0, 1].set_title('Boundary Detection', fontweight='bold')
        
        # 3. Room Segmentation
        self._plot_room_mask(axes[0, 2], room_mask)
        axes[0, 2].set_title('Room Segmentation', fontweight='bold')
        
        # 4. Structural Elements (Walls, Doors, Windows)
        self._plot_structural_elements(axes[1, 0], json_data, original_img.size)
        axes[1, 0].set_title('Structural Elements', fontweight='bold')
        
        # 5. Room Layout with Labels
        self._plot_room_layout(axes[1, 1], json_data, original_img.size)
        axes[1, 1].set_title('Room Layout', fontweight='bold')
        
        # 6. Complete Overlay
        self._plot_complete_overlay(axes[1, 2], original_img, json_data)
        axes[1, 2].set_title('Complete Analysis', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_boundary_mask(self, ax, boundary_mask):
        """Plot boundary predictions"""
        # Create colored boundary mask
        h, w = boundary_mask.shape
        colored_mask = np.zeros((h, w, 3))
        
        # Background (class 0) - white
        colored_mask[boundary_mask == 0] = [1, 1, 1]
        # Walls (class 1) - black
        colored_mask[boundary_mask == 1] = [0, 0, 0]
        # Doors (class 2) - red
        colored_mask[boundary_mask == 2] = [1, 0, 0]
        # Windows (class 3) - blue
        colored_mask[boundary_mask == 3] = [0, 0, 1]
        
        ax.imshow(colored_mask)
        ax.axis('off')
    
    def _create_simple_json_format(self, walls, corners, doors, windows, rooms):
        """Create simplified JSON format for visualization"""
        simple_json = {
            'walls': [],
            'corners': [],
            'doors': [],
            'windows': [],
            'rooms': []
        }
        
        # Convert walls to simple format
        for wall in walls:
            if hasattr(wall, 'start_point') and hasattr(wall, 'end_point'):
                simple_json['walls'].append({
                    'start_point': wall.start_point,
                    'end_point': wall.end_point
                })
            elif isinstance(wall, dict) and 'start_point' in wall and 'end_point' in wall:
                simple_json['walls'].append(wall)
            else:
                # Handle other wall formats - extract coordinates
                print(f"Unknown wall format: {type(wall)}, {wall}")
        
        # Convert corners to simple format
        for corner in corners:
            if hasattr(corner, 'position'):
                simple_json['corners'].append({
                    'position': corner.position
                })
            elif isinstance(corner, dict) and 'position' in corner:
                simple_json['corners'].append(corner)
            else:
                print(f"Unknown corner format: {type(corner)}, {corner}")
        
        # Convert doors to simple format
        for door in doors:
            if hasattr(door, 'bounding_box'):
                simple_json['doors'].append({
                    'bounding_box': door.bounding_box
                })
            elif isinstance(door, dict) and 'bounding_box' in door:
                simple_json['doors'].append(door)
            else:
                print(f"Unknown door format: {type(door)}, {door}")
        
        # Convert windows to simple format
        for window in windows:
            if hasattr(window, 'bounding_box'):
                simple_json['windows'].append({
                    'bounding_box': window.bounding_box
                })
            elif isinstance(window, dict) and 'bounding_box' in window:
                simple_json['windows'].append(window)
            else:
                print(f"Unknown window format: {type(window)}, {window}")
        
        # Convert rooms to simple format
        for room in rooms:
            if hasattr(room, 'type') and hasattr(room, 'contour'):
                simple_json['rooms'].append({
                    'type': room.type,
                    'contour': room.contour,
                    'bounding_box': getattr(room, 'bounding_box', None)
                })
            elif isinstance(room, dict):
                simple_json['rooms'].append(room)
            else:
                print(f"Unknown room format: {type(room)}, {room}")
        
        return simple_json
        
        # Add legend
        legend_elements = [
            patches.Patch(color='black', label='Walls'),
            patches.Patch(color='red', label='Doors'),
            patches.Patch(color='blue', label='Windows')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    def _plot_room_mask(self, ax, room_mask):
        """Plot room segmentation"""
        # Create colored room mask
        h, w = room_mask.shape
        colored_mask = np.ones((h, w, 3))  # Start with white background
        
        # Get unique room classes
        unique_classes = np.unique(room_mask)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_id in enumerate(unique_classes):
            if class_id > 0:  # Skip background
                colored_mask[room_mask == class_id] = colors[i][:3]
        
        ax.imshow(colored_mask)
        ax.axis('off')
    
    def _plot_structural_elements(self, ax, json_data, img_size):
        """Plot walls, doors, and windows"""
        width, height = img_size
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip y-axis to match image coordinates
        ax.set_aspect('equal')
        
        elements_plotted = False
        
        # Plot walls
        walls = json_data.get('walls', [])
        for wall in walls:
            if 'start_point' in wall and 'end_point' in wall:
                start = wall['start_point']
                end = wall['end_point']
                ax.plot([start[0], end[0]], [start[1], end[1]], 
                       color='black', linewidth=3, label='Wall' if not elements_plotted else "")
                elements_plotted = True
        
        # Plot doors
        doors = json_data.get('doors', [])
        for door in doors:
            if 'bounding_box' in door:
                bbox = door['bounding_box']
                rect = patches.Rectangle(
                    (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                    linewidth=2, edgecolor='red', facecolor='red', alpha=0.7,
                    label='Door' if 'Door' not in [t.get_text() for t in ax.get_legend().get_texts() if ax.get_legend()] else ""
                )
                ax.add_patch(rect)
                elements_plotted = True
        
        # Plot windows
        windows = json_data.get('windows', [])
        for window in windows:
            if 'bounding_box' in window:
                bbox = window['bounding_box']
                rect = patches.Rectangle(
                    (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                    linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.7,
                    label='Window' if 'Window' not in [t.get_text() for t in ax.get_legend().get_texts() if ax.get_legend()] else ""
                )
                ax.add_patch(rect)
                elements_plotted = True
        
        # Plot corners
        corners = json_data.get('corners', [])
        for corner in corners:
            if 'position' in corner:
                ax.plot(corner['position'][0], corner['position'][1], 
                       'go', markersize=6, label='Corner' if 'Corner' not in [t.get_text() for t in ax.get_legend().get_texts() if ax.get_legend()] else "")
                elements_plotted = True
        
        # If no elements were plotted, add a message
        if not elements_plotted:
            ax.text(width/2, height/2, 'No structural elements detected\nCheck postprocessor output', 
                   ha='center', va='center', fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
        
        if elements_plotted:
            ax.legend()
        ax.axis('off')
    
    def _plot_room_layout(self, ax, json_data, img_size):
        """Plot room layout with labels"""
        width, height = img_size
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)  # Flip y-axis
        ax.set_aspect('equal')
        
        rooms = json_data.get('rooms', [])
        rooms_plotted = False
        
        # Plot rooms with different colors
        for room in rooms:
            room_type = room.get('type', 'other')
            color = self.room_colors.get(room_type, self.room_colors['other'])
            
            # Create room polygon if contour is available
            if 'contour' in room and room['contour']:
                contour = np.array(room['contour'])
                if len(contour) > 2:  # Need at least 3 points for polygon
                    polygon = patches.Polygon(contour, closed=True, 
                                            facecolor=color, alpha=0.7, 
                                            edgecolor='black', linewidth=1)
                    ax.add_patch(polygon)
                    
                    # Add room label at centroid
                    centroid_x = np.mean(contour[:, 0])
                    centroid_y = np.mean(contour[:, 1])
                    ax.text(centroid_x, centroid_y, room_type.replace('_', '\n'), 
                           ha='center', va='center', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    rooms_plotted = True
            
            # Fallback: use bounding box if contour not available
            elif 'bounding_box' in room and room['bounding_box']:
                bbox = room['bounding_box']
                rect = patches.Rectangle(
                    (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                    facecolor=color, alpha=0.7, edgecolor='black', linewidth=1
                )
                ax.add_patch(rect)
                
                # Add room label at center
                center_x = bbox['x'] + bbox['width'] / 2
                center_y = bbox['y'] + bbox['height'] / 2
                ax.text(center_x, center_y, room_type.replace('_', '\n'), 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                rooms_plotted = True
        
        # If no rooms were plotted, add a message
        if not rooms_plotted:
            ax.text(width/2, height/2, 'No rooms detected\nRoom segmentation may need adjustment', 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='yellow', alpha=0.8))
        
        ax.axis('off')
    
    def _plot_complete_overlay(self, ax, original_img, json_data):
        """Plot complete overlay on original image"""
        ax.imshow(original_img, alpha=0.7)
        
        img_size = original_img.size
        width, height = img_size
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        
        # Overlay walls
        for wall in json_data.get('walls', []):
            start = wall['start_point']
            end = wall['end_point']
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                   'k-', linewidth=2, alpha=0.8)
        
        # Overlay doors and windows
        for door in json_data.get('doors', []):
            bbox = door['bounding_box']
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
        
        for window in json_data.get('windows', []):
            bbox = window['bounding_box']
            rect = patches.Rectangle(
                (bbox['x'], bbox['y']), bbox['width'], bbox['height'],
                linewidth=2, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(rect)
        
        # Overlay room labels
        for room in json_data.get('rooms', []):
            if 'contour' in room:
                contour = np.array(room['contour'])
                centroid_x = np.mean(contour[:, 0])
                centroid_y = np.mean(contour[:, 1])
            elif 'bounding_box' in room:
                bbox = room['bounding_box']
                centroid_x = bbox['x'] + bbox['width'] / 2
                centroid_y = bbox['y'] + bbox['height'] / 2
            else:
                continue
            
            room_type = room.get('type', 'other')
            ax.text(centroid_x, centroid_y, room_type.replace('_', '\n'), 
                   ha='center', va='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        ax.axis('off')
    
    def save_individual_visualizations(self, image_path, json_data, boundary_mask, room_mask, output_dir):
        """Save individual visualization components separately"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        original_img = Image.open(image_path).convert('RGB')
        
        # Save boundary visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        self._plot_boundary_mask(ax, boundary_mask)
        plt.savefig(os.path.join(output_dir, 'boundary_prediction.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save room segmentation
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        self._plot_room_mask(ax, room_mask)
        plt.savefig(os.path.join(output_dir, 'room_segmentation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save structural elements
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        self._plot_structural_elements(ax, json_data, original_img.size)
        plt.savefig(os.path.join(output_dir, 'structural_elements.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save complete overlay
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        self._plot_complete_overlay(ax, original_img, json_data)
        plt.savefig(os.path.join(output_dir, 'complete_overlay.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Individual visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    inference = FloorplanInference("./best_model.pth")
    
    # Run prediction with visualization
    result = inference.predict(
        "./F1_original_9.png", 
        "output1.json", 
        visualize=True, 
        viz_output_path="floorplan_analysis.png"
    )
    
    # Optional: Save individual visualization components
    # inference.save_individual_visualizations(
    #     "./F1_original_9.png",
    #     result,
    #     boundary_mask,  # You'd need to store this from predict method
    #     room_mask,      # You'd need to store this from predict method
    #     "./individual_viz/"
    # )