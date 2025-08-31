import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

        # Extract feature maps at different scales
        self.pool1 = nn.Sequential(*list(self.features.children())[:5])    # 64 channels
        self.pool2 = nn.Sequential(*list(self.features.children())[5:10])  # 128 channels
        self.pool3 = nn.Sequential(*list(self.features.children())[10:17]) # 256 channels
        self.pool4 = nn.Sequential(*list(self.features.children())[17:24]) # 512 channels
        self.pool5 = nn.Sequential(*list(self.features.children())[24:31]) # 512 channels

    def forward(self, x):
        # print(f"Encoder input shape: {x.shape}")
        x1 = self.pool1(x)  # 64
        x2 = self.pool2(x1) # 128
        x3 = self.pool3(x2) # 256
        x4 = self.pool4(x3) # 512
        x5 = self.pool5(x4) # 512
        # print(f"Encoder features shapes: {[f.shape for f in [x1, x2, x3, x4, x5]]}")
        return [x1, x2, x3, x4, x5]


class VGGDecoder(nn.Module):
    def __init__(self, num_classes):
        super(VGGDecoder, self).__init__()

        self.upconv5 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.conv5 = nn.Conv2d(512 + 512, 512, 3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv4 = nn.Conv2d(256 + 256, 256, 3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = nn.Conv2d(128 + 128, 128, 3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = nn.Conv2d(64 + 64, 64, 3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        # After concat: 32 (upconv1) + 64 (intermediate_up) = 96 channels
        self.conv1 = nn.Conv2d(96, 32, 3, padding=1)  # Fixed: 96 input channels

        self.final_conv = nn.Conv2d(32, num_classes, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        # print(f"Decoder input features shapes: {[f.shape for f in features]}")

        # Decoder path
        x = self.relu(self.upconv5(x5))
        # print(f"After upconv5: {x.shape}")
        x = torch.cat([x, x4], dim=1)
        x = self.relu(self.conv5(x))
        # print(f"After conv5: {x.shape}")

        x = self.relu(self.upconv4(x))
        # print(f"After upconv4: {x.shape}")
        x = torch.cat([x, x3], dim=1)
        x = self.relu(self.conv4(x))
        # print(f"After conv4: {x.shape}")

        x = self.relu(self.upconv3(x))
        # print(f"After upconv3: {x.shape}")
        x = torch.cat([x, x2], dim=1)
        x = self.relu(self.conv3(x))
        # print(f"After conv3: {x.shape}")

        x = self.relu(self.upconv2(x))
        # print(f"After upconv2: {x.shape}")
        x = torch.cat([x, x1], dim=1)
        x = self.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")

        # Save intermediate BEFORE final upsampling (64 channels)
        intermediate = x  # 64 channels

        # Upsample once more
        x = self.relu(self.upconv1(x))  # 32 channels
        # print(f"After upconv1: {x.shape}")

        # Match shapes before concatenation
        intermediate_up = F.interpolate(
            intermediate, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        # print(f"Intermediate_up shape: {intermediate_up.shape}")

        # Concatenate → 32 + 64 = 96 channels
        x = torch.cat([x, intermediate_up], dim=1)
        # print(f"After concatenation: {x.shape}")
        x = self.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")

        # FORCE upsampling to 512x512 - THIS IS THE KEY FIX
        # print(f"Before final upsampling: {x.shape}")
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        # print(f"After final upsampling: {x.shape}")

        output = self.final_conv(x)
        # print(f"Final output shape: {output.shape}")

        return output, intermediate