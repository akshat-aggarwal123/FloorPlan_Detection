import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import VGGEncoder, VGGDecoder
from .spatial_context import SpatialContextModule

class DeepFloorplanNet(nn.Module):
    def __init__(self, num_boundary_classes, num_room_classes):
        super(DeepFloorplanNet, self).__init__()
        
        # Shared encoder
        self.encoder = VGGEncoder()
        
        # Task-specific decoders
        self.boundary_decoder = VGGDecoder(num_boundary_classes)
        self.room_decoder = VGGDecoder(num_room_classes)
        
        # Spatial context module
        self.spatial_context = SpatialContextModule()
        
        # Final room prediction layer
        self.room_final = nn.Conv2d(64, num_room_classes, 1)
        
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.encoder(x)
        
        # Boundary prediction
        boundary_output, boundary_intermediate = self.boundary_decoder(shared_features)
        
        # Room prediction with spatial context
        room_output, room_intermediate = self.room_decoder(shared_features)
        
        # Apply spatial context module
        contextual_room_features = self.spatial_context(boundary_intermediate, room_intermediate)
        
        # Ensure contextual features are 512x512 before final conv
        if contextual_room_features.shape[2] != 512 or contextual_room_features.shape[3] != 512:
            contextual_room_features = F.interpolate(
                contextual_room_features, 
                size=(512, 512), 
                mode="bilinear", 
                align_corners=False
            )
        
        # Final room prediction
        room_output_refined = self.room_final(contextual_room_features)
        
        return {
            'boundary': boundary_output,
            'room': room_output_refined
        }
