import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize, thin

class FloorplanPostprocessor:
    def __init__(self, config):
        self.config = config
        
    def extract_walls_as_lines(self, wall_mask):
        """Extract wall lines from wall mask"""
        # Skeletonize wall mask to get centerlines
        skeleton = skeletonize(wall_mask > 0.5)
        
        # Find line segments using HoughLinesP
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
        lines = cv2.HoughLinesP(skeleton_uint8, 1, np.pi/180, 
                               threshold=20, minLineLength=10, maxLineGap=5)
        
        wall_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                wall_lines.append({
                    'start': {'x': float(x1), 'y': float(y1)},
                    'end': {'x': float(x2), 'y': float(y2)},
                    'thickness': self.config.WALL_THICKNESS
                })
        
        return wall_lines
    
    def extract_corners_from_walls(self, wall_lines):
        """Extract corner points from wall lines"""
        corners = {}
        corner_id = 0
        
        # Collect all endpoints
        points = []
        for line in wall_lines:
            points.extend([
                (line['start']['x'], line['start']['y']),
                (line['end']['x'], line['end']['y'])
            ])
        
        # Cluster nearby points to form corners
        corners_list = []
        tolerance = 10
        
        for point in points:
            # Check if point is close to existing corner
            found_corner = False
            for corner in corners_list:
                dist = np.sqrt((point[0] - corner[0])**2 + (point[1] - corner[1])**2)
                if dist < tolerance:
                    found_corner = True
                    break
            
            if not found_corner:
                corners_list.append(point)
        
        # Convert to corner dictionary format
        for i, corner in enumerate(corners_list):
            corner_id = f"corner_{i}"
            corners[corner_id] = {
                'x': corner[0] / self.config.INPUT_SIZE * 20 - 10,  # Scale to (-10, 10)
                'y': corner[1] / self.config.INPUT_SIZE * 20 - 10,
                'elevation': 2.5
            }
        
        return corners
    
    def extract_doors_and_windows(self, boundary_mask):
        """Extract door and window positions"""
        door_mask = (boundary_mask == 2).astype(np.uint8)
        window_mask = (boundary_mask == 3).astype(np.uint8)
        
        doors = []
        windows = []
        
        # Find door regions
        door_labels = label(door_mask)
        for region in regionprops(door_labels):
            if region.area > 50:  # Filter small regions
                centroid = region.centroid
                doors.append({
                    'position': [
                        float(centroid[1]), 
                        105.0,  # Default door height
                        float(centroid[0])
                    ],
                    'type': 'door',
                    'size': [80, 200, 20]  # Default door dimensions
                })
        
        # Find window regions
        window_labels = label(window_mask)
        for region in regionprops(window_labels):
            if region.area > 30:  # Filter small regions
                centroid = region.centroid
                windows.append({
                    'position': [
                        float(centroid[1]), 
                        135.0,  # Default window height
                        float(centroid[0])
                    ],
                    'type': 'window',
                    'size': [120, 120, 15]  # Default window dimensions
                })
        
        return doors, windows
    
    def extract_rooms(self, room_mask):
        """Extract room regions and types"""
        rooms = {}
        
        for class_id in range(1, len(self.config.ROOM_LABELS)):
            class_mask = (room_mask == class_id).astype(np.uint8)
            room_labels = label(class_mask)
            
            for region in regionprops(room_labels):
                if region.area > self.config.MIN_ROOM_AREA:
                    # Create room identifier from region coordinates
                    bbox = region.bbox
                    room_id = f"room_{class_id}_{bbox[0]}_{bbox[1]}"
                    
                    rooms[room_id] = {
                        'name': self.config.ROOM_LABELS[class_id],
                        'area': float(region.area),
                        'centroid': [float(region.centroid[1]), float(region.centroid[0])]
                    }
        
        return rooms