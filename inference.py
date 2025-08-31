import torch
import json
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
    
    def predict(self, image_path, output_path):
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
        
        # Post-process predictions
        wall_mask = (boundary_mask == 1).astype(float)
        walls = self.postprocessor.extract_walls_as_lines(wall_mask)
        corners = self.postprocessor.extract_corners_from_walls(walls)
        doors, windows = self.postprocessor.extract_doors_and_windows(boundary_mask)
        rooms = self.postprocessor.extract_rooms(room_mask)
        
        # Convert to JSON format
        json_output = self.json_converter.convert_to_json(
            walls, corners, doors, windows, rooms
        )
        
        # Save JSON output
        with open(output_path, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"Prediction saved to {output_path}")
        return json_output

if __name__ == "__main__":
    # Example usage
    inference = FloorplanInference("./best_model.pth")
    result = inference.predict("./F1_original_2.png", "output.json")
