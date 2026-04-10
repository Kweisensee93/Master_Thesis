# Defines the model architecture for the hybrid approach, combining image features and landmark anchors.

import torch
import torch.nn as nn

class HybridFishNet(nn.Module):
    def __init__(self, output_dim, input_shape=(3, 60, 270)):
        super(HybridFishNet, self).__init__()
        
        # 1. Image Branch
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # -> 30x135
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), # -> 15x67
            nn.Flatten()
        )
        
        # 2. Auto-calculate flattening size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            cnn_out = self.cnn(dummy_input)
            self.cnn_flat_size = cnn_out.shape[1]
            
        # 3. Landmark Branch (Anchors: Points 1 & 13)
        self.anchor_mlp = nn.Sequential(nn.Linear(4, 16), nn.ReLU())

        # 4. Combined Head
        self.combined = nn.Sequential(
            nn.Linear(self.cnn_flat_size + 16, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim) 
        )

    def forward(self, img, anchors):
        img_features = self.cnn(img.permute(0, 3, 1, 2))
        anchor_features = self.anchor_mlp(anchors)
        x = torch.cat((img_features, anchor_features), dim=1)
        return self.combined(x)