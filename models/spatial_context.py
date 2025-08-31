import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialContextModule(nn.Module):
    def __init__(self, boundary_channels=64, room_channels=64):
        super(SpatialContextModule, self).__init__()
        
        # Room-boundary-guided attention
        self.attention_conv = nn.Sequential(
            nn.Conv2d(boundary_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Direction-aware kernels
        self.horizontal_conv = nn.Conv2d(room_channels, room_channels, (1, 3), padding=(0, 1))
        self.vertical_conv = nn.Conv2d(room_channels, room_channels, (3, 1), padding=(1, 0))
        self.diagonal_conv = nn.Conv2d(room_channels, room_channels, 3, padding=1)
        self.anti_diagonal_conv = nn.Conv2d(room_channels, room_channels, 3, padding=1)
        
        self.fusion_conv = nn.Conv2d(room_channels * 4, room_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, boundary_features, room_features):
        # Ensure both features have the same spatial dimensions
        if boundary_features.shape[2:] != room_features.shape[2:]:
            target_size = max(boundary_features.shape[2], room_features.shape[2])
            target_size = (target_size, target_size)
            
            if boundary_features.shape[2] < target_size[0]:
                boundary_features = F.interpolate(
                    boundary_features, size=target_size, mode="bilinear", align_corners=False
                )
            if room_features.shape[2] < target_size[0]:
                room_features = F.interpolate(
                    room_features, size=target_size, mode="bilinear", align_corners=False
                )
        
        # Generate attention weights from boundary features
        attention_weights = self.attention_conv(boundary_features)
        
        # Apply first attention to room features
        attended_features = room_features * attention_weights
        
        # Apply direction-aware convolutions
        h_feat = self.horizontal_conv(attended_features)
        v_feat = self.vertical_conv(attended_features)
        d_feat = self.diagonal_conv(attended_features)
        ad_feat = self.anti_diagonal_conv(attended_features)
        
        # Concatenate directional features
        directional_features = torch.cat([h_feat, v_feat, d_feat, ad_feat], dim=1)
        
        # Fuse directional features
        fused_features = self.relu(self.fusion_conv(directional_features))
        
        # Apply second attention
        contextual_features = fused_features * attention_weights
        
        # Ensure output is always 512x512
        if contextual_features.shape[2] != 512 or contextual_features.shape[3] != 512:
            contextual_features = F.interpolate(
                contextual_features, 
                size=(512, 512), 
                mode="bilinear", 
                align_corners=False
            )
        
        return contextual_features
