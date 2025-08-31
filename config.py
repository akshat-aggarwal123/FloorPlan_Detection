import torch
class Config:
    # Dataset
    INPUT_SIZE = 512
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    
    # Model - Updated for your B/W dataset
    BACKBONE = 'vgg16'
    NUM_BOUNDARY_CLASSES = 4  # background, wall, door, window
    NUM_ROOM_CLASSES = 5      # Reduced since B/W limits room detection
    
    # Training
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss weights
    BOUNDARY_WEIGHT = 0.7  # Higher weight since walls are more reliable in B/W
    ROOM_WEIGHT = 0.3
    
    # Post-processing
    MIN_ROOM_AREA = 500    # Larger minimum for B/W images
    WALL_THICKNESS = 15    # Thicker walls typical in B/W plans
    
    # Updated label mappings for B/W dataset
    BOUNDARY_LABELS = {
        0: 'background',
        1: 'wall',
        2: 'door', 
        3: 'window'
    }
    
    ROOM_LABELS = {
        0: 'background',
        1: 'room',        # Generic room since B/W can't distinguish types
        2: 'corridor',    # Hallways/corridors
        3: 'entrance',    # Entry areas
        4: 'outdoor'      # Outdoor/balcony areas
    }