import torch
import os

class Config:
    # Dataset
    INPUT_SIZE = 512
    BATCH_SIZE = 8  # Increased for better stability
    NUM_WORKERS = 0 if os.name == 'nt' else 4  # Windows compatibility
    
    # Model - Updated for your B/W dataset
    BACKBONE = 'vgg16'
    NUM_BOUNDARY_CLASSES = 4  # background, wall, door, window
    NUM_ROOM_CLASSES = 5      # background, room, corridor, entrance, outdoor
    
    # Training - More conservative settings
    LEARNING_RATE = 5e-5      # Reduced learning rate for stability
    NUM_EPOCHS = 100          # More epochs with lower LR
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Loss weights - More balanced
    BOUNDARY_WEIGHT = 0.5     # Equal weight to start
    ROOM_WEIGHT = 0.5         # Equal weight to start
    
    # Training stability
    GRAD_CLIP_NORM = 1.0      # Gradient clipping
    WEIGHT_DECAY = 1e-4       # L2 regularization
    LABEL_SMOOTHING = 0.0     # Start without label smoothing
    
    # Early stopping
    PATIENCE = 15             # Stop if no improvement for 15 epochs
    MIN_DELTA = 1e-4          # Minimum change to consider improvement
    
    # Validation
    VAL_FREQ = 1              # Validate every epoch
    SAVE_FREQ = 5             # Save checkpoint every 5 epochs
    
    # Post-processing
    MIN_ROOM_AREA = 500       # Larger minimum for B/W images
    WALL_THICKNESS = 15       # Thicker walls typical in B/W plans
    
    # Data augmentation (if enabled)
    USE_AUGMENTATION = True
    ROTATION_RANGE = 15       # Small rotations
    BRIGHTNESS_RANGE = 0.1    # Slight brightness changes
    CONTRAST_RANGE = 0.1      # Slight contrast changes
    
    # Memory optimization
    ACCUMULATE_GRAD_BATCHES = 2  # Gradient accumulation for effective batch size 16
    MIXED_PRECISION = True       # Use AMP
    COMPILE_MODEL = False        # Disable on Windows
    
    # Scheduler settings
    SCHEDULER_TYPE = 'cosine'    # 'cosine', 'step', or 'plateau'
    COSINE_ETA_MIN = 1e-6       # Minimum LR for cosine annealing
    STEP_SIZE = 20              # For StepLR
    STEP_GAMMA = 0.5            # For StepLR
    PLATEAU_PATIENCE = 10       # For ReduceLROnPlateau
    PLATEAU_FACTOR = 0.5        # For ReduceLROnPlateau
    
    # Monitoring
    LOG_INTERVAL = 10           # Log every N batches
    METRICS_INTERVAL = 20       # Compute detailed metrics every N batches
    
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
    
    # Model architecture tweaks
    DROPOUT_RATE = 0.1          # Add dropout for regularization
    BATCH_NORM_MOMENTUM = 0.1   # BatchNorm momentum
    
    # Loss function settings
    FOCAL_LOSS_ALPHA = 1.0      # For focal loss (if used)
    FOCAL_LOSS_GAMMA = 2.0      # For focal loss (if used)
    CLASS_WEIGHTS = None        # Will be computed from data if needed
    
    def __init__(self):
        """Initialize config and perform validation"""
        self.validate_config()
        self.print_config()
    
    def validate_config(self):
        """Validate configuration parameters"""
        assert self.BATCH_SIZE > 0, "Batch size must be positive"
        assert self.LEARNING_RATE > 0, "Learning rate must be positive"
        assert self.NUM_EPOCHS > 0, "Number of epochs must be positive"
        assert self.BOUNDARY_WEIGHT + self.ROOM_WEIGHT > 0, "Loss weights must sum to positive value"
        assert len(self.BOUNDARY_LABELS) == self.NUM_BOUNDARY_CLASSES, "Boundary labels mismatch"
        assert len(self.ROOM_LABELS) == self.NUM_ROOM_CLASSES, "Room labels mismatch"
        
        # Warn about potential issues
        if self.BATCH_SIZE < 8:
            print(f"⚠️  Warning: Small batch size ({self.BATCH_SIZE}) may cause training instability")
        
        if self.LEARNING_RATE > 1e-3:
            print(f"⚠️  Warning: High learning rate ({self.LEARNING_RATE}) may cause training instability")
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print("TRAINING CONFIGURATION")
        print("=" * 50)
        print(f"Device: {self.DEVICE}")
        print(f"Input Size: {self.INPUT_SIZE}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Epochs: {self.NUM_EPOCHS}")
        print(f"Backbone: {self.BACKBONE}")
        print(f"Classes: Boundary={self.NUM_BOUNDARY_CLASSES}, Room={self.NUM_ROOM_CLASSES}")
        print(f"Loss Weights: Boundary={self.BOUNDARY_WEIGHT}, Room={self.ROOM_WEIGHT}")
        print(f"Mixed Precision: {self.MIXED_PRECISION}")
        print(f"Gradient Accumulation: {self.ACCUMULATE_GRAD_BATCHES}")
        print("=" * 50)
    
    def get_effective_batch_size(self):
        """Get effective batch size with gradient accumulation"""
        return self.BATCH_SIZE * self.ACCUMULATE_GRAD_BATCHES
    
    def update_for_debugging(self):
        """Update config for debugging mode"""
        self.NUM_EPOCHS = 5
        self.VAL_FREQ = 1
        self.LOG_INTERVAL = 5
        self.MIXED_PRECISION = False  # Disable AMP for debugging
        print("🐛 Config updated for debugging mode")
    
    def update_for_fast_training(self):
        """Update config for faster training (less validation)"""
        self.VAL_FREQ = 2
        self.METRICS_INTERVAL = 50
        self.LOG_INTERVAL = 20
        print("🚀 Config updated for fast training mode")