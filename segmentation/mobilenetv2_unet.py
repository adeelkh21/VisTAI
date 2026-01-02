"""
MobileNetV2-based U-Net for binary tumor segmentation.
Uses MobileNetV2 as encoder and custom decoder with skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class MobileNetV2UNet(nn.Module):
    """
    U-Net style segmentation model with MobileNetV2 encoder.
    Outputs binary segmentation masks (tumor vs background).
    """
    
    def __init__(self, pretrained=True):
        super(MobileNetV2UNet, self).__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = mobilenet_v2(pretrained=pretrained)
        
        # Encoder - Extract features from different stages
        self.encoder1 = nn.Sequential(*list(mobilenet.features[:2]))   # 16 channels
        self.encoder2 = nn.Sequential(*list(mobilenet.features[2:4]))  # 24 channels
        self.encoder3 = nn.Sequential(*list(mobilenet.features[4:7]))  # 32 channels
        self.encoder4 = nn.Sequential(*list(mobilenet.features[7:14])) # 96 channels
        self.encoder5 = nn.Sequential(*list(mobilenet.features[14:]))  # 1280 channels
        
        # Decoder - Upsampling path with skip connections
        self.upconv5 = self._make_upconv(1280, 96)
        self.decoder5 = self._make_decoder_block(96 + 96, 96)
        
        self.upconv4 = self._make_upconv(96, 32)
        self.decoder4 = self._make_decoder_block(32 + 32, 32)
        
        self.upconv3 = self._make_upconv(32, 24)
        self.decoder3 = self._make_decoder_block(24 + 24, 24)
        
        self.upconv2 = self._make_upconv(24, 16)
        self.decoder2 = self._make_decoder_block(16 + 16, 16)
        
        # Final upsampling to original resolution
        self.upconv1 = self._make_upconv(16, 16)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1)  # Binary output
        )
        
    def _make_upconv(self, in_channels, out_channels):
        """Create upsampling block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block with conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder path with skip connections
        e1 = self.encoder1(x)      # 1/2
        e2 = self.encoder2(e1)     # 1/4
        e3 = self.encoder3(e2)     # 1/8
        e4 = self.encoder4(e3)     # 1/16
        e5 = self.encoder5(e4)     # 1/32
        
        # Decoder path with skip connections
        d5 = self.upconv5(e5)      # 1/16
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.decoder5(d5)
        
        d4 = self.upconv4(d5)      # 1/8
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)      # 1/4
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)      # 1/2
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)      # 1/1
        out = self.final_conv(d1)
        
        return out


def create_segmentation_model(pretrained=True):
    """Create and return MobileNetV2 U-Net segmentation model."""
    model = MobileNetV2UNet(pretrained=pretrained)
    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Creating MobileNetV2 U-Net Segmentation Model...")
    model = create_segmentation_model(pretrained=True)
    
    # Count parameters
    params = count_parameters(model)
    print(f"Total trainable parameters: {params:,}")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test with dummy input
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    print(f"\nInput shape: {dummy_input.shape}")
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"\n✓ Model created successfully!")
    print(f"✓ Expected output: [batch_size, 1, 224, 224]")
    print(f"✓ Use sigmoid activation for binary segmentation")
