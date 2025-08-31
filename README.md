# Deep Floor Plan Recognition Implementation

This is a simplified implementation of the Deep Floor Plan Recognition algorithm based on the ICCV 2019 paper "Deep Floor Plan Recognition Using a Multi-Task Network with Room-Boundary-Guided Attention" by Zeng et al.

## Features

- Multi-task network for simultaneous room boundary and room type prediction
- Room-boundary-guided attention mechanism 
- Cross-and-within-task weighted loss function
- Support for hand-drawn floorplans
- JSON output format compatible with 3D model generation
- Visualization tools for predictions

## Project Structure

```
floorplan_recognition/
├── data/                           # Training and test data
├── models/                         # Neural network models
│   ├── multitask_network.py       # Main multi-task network
│   ├── spatial_contextual_module.py # Attention mechanism
│   └── vgg_encoder_decoder.py     # VGG backbone
├── utils/                          # Utility functions
│   ├── data_loader.py             # Dataset classes
│   ├── json_converter.py          # Convert predictions to JSON
│   └── preprocessing.py           # Image preprocessing
├── training/                       # Training scripts
│   ├── train.py                   # Main training script
│   ├── loss_functions.py          # Custom loss functions
│   └── config.py                  # Configuration
├── inference/                      # Inference scripts
├── main.py                        # Main inference script
└── requirements.txt               # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd floorplan_recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

Prepare your dataset in the following structure:

```
data/
├── train/
│   ├── images/           # Original floorplan images
│   ├── wall_masks/       # Room boundary masks (wall, door, window)
│   ├── room_masks/       # Room type masks
│   └── annotations.json  # Optional metadata
├── val/
│   └── ... (same structure)
└── test/
    └── ... (same structure)
```

### Mask Format

- **Wall masks**: Pixel values represent different classes:
  - 0: Background
  - 255: Wall
  - 128: Door  
  - 64: Window

- **Room masks**: Pixel values represent room types:
  - 0: Background
  - 30: Living room
  - 60: Bedroom
  - 90: Bathroom
  - ... (increment by 30 for each room type)

## Training

1. Configure training parameters in `training/config.py`

2. Start training:
```bash
python training/train.py
```

3. Monitor training with Weights & Biases:
```bash
wandb login
```

## Inference

### Single Image

Process a single floorplan image:

```bash
python main.py --image path/to/floorplan.jpg --model checkpoints/best_model.pth --output result.json --visualize
```

### Batch Processing

Process multiple images:

```bash
python main.py --batch path/to/images/ --model checkpoints/best_model.pth --output output_dir/
```

### Python API

```python
from main import FloorPlanInference
from training.config import Config

# Initialize inference engine
config = Config()
inference_engine = FloorPlanInference('checkpoints/best_model.pth', config)

# Process image
json_output = inference_engine.predict('path/to/floorplan.jpg')

# Save result
with open('output.json', 'w') as f:
    json.dump(json_output, f, indent=2)
```

## Output Format

The algorithm outputs JSON in the exact format required for 3D model generation:

```json
{
  "floorplanner": {
    "version": "2.0.1a",
    "corners": {
      "corner_id": {"x": 0.0, "y": 0.0, "elevation": 2.5}
    },
    "walls": [
      {
        "corner1": "corner_id_1",
        "corner2": "corner_id_2", 
        "frontTexture": {...},
        "backTexture": {...},
        "wallType": "STRAIGHT",
        "thickness": 0.1
      }
    ],
    "rooms": {
      "room_corners": {"name": "bedroom"}
    },
    "newFloorTextures": {...},
    "units": "m"
  },
  "items": [],
  "lights": [...],
  "sunlight": [...],
  "hemlight": [...],
  "amblight": [...]
}
```

## Model Architecture

The implementation follows the paper's architecture:

1. **Shared VGG Encoder**: Extracts common features
2. **Dual VGG Decoders**: Separate branches for room boundaries and room types  
3. **Spatial Contextual Module**: Room-boundary-guided attention mechanism
4. **Multi-Task Learning**: Joint optimization with weighted loss

### Key Components

- **Room-Boundary-Guided Attention**: Uses room boundary features to guide room type predictions
- **Direction-Aware Kernels**: Horizontal, vertical, and diagonal convolutions for spatial context
- **Cross-and-Within-Task Weighted Loss**: Balances class imbalance within and across tasks

## Hand-Drawn Support

The implementation includes special preprocessing for hand-drawn floorplans:

- Gaussian blur to smooth irregular lines
- Edge enhancement with Canny detector
- Dual-channel input processing
- Robust augmentation strategies

## Configuration

Key parameters in `training/config.py`:

```python
class Config:
    # Data parameters
    image_size = (512, 512)
    batch_size = 4
    
    # Training parameters  
    num_epochs = 100
    learning_rate = 1e-4
    
    # Output parameters
    pixel_to_meter_scale = 0.05  # 5cm per pixel
    wall_thickness = 0.1         # 10cm walls
    elevation = 2.5              # 2.5m ceiling height
```

## Performance

The algorithm handles:
- Irregular wall thickness
- Non-rectangular rooms
- Curved walls
- Hand-drawn sketches
- Various room types (8+ categories)
- Door and window detection

## Limitations

- Requires pixel-wise annotations for training
- Computational intensive (GPU recommended)
- May struggle with very complex architectural drawings
- Room boundary detection depends on clear wall definitions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request

## Citation

If you use this implementation, please cite the original paper:

```
@inproceedings{zeng2019deep,
  title={Deep Floor Plan Recognition Using a Multi-Task Network with Room-Boundary-Guided Attention},
  author={Zeng, Zhiliang and Li, Xianzhi and Yu, Ying Kin and Fu, Chi-Wing},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={9096--9104},
  year={2019}
}
```

## License

This implementation is provided for research and educational purposes.