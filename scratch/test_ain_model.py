import sys
sys.path.insert(0, '.')
import os
import torch
import torch.nn as nn
from src.reid.backbones.osnet import (
    Conv1x1, Conv1x1Linear, Conv3x3, Conv7x7, LightConv3x3, ChannelGate, OSBlock, OSNet
)

class OSBlockAIN(nn.Module):
    """OSBlock with Adaptive/Instance Normalization (AIN)."""
    def __init__(self, in_channels: int, out_channels: int, IN: bool = False, bottleneck_reduction: int = 4) -> None:
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.IN = nn.InstanceNorm2d(in_channels, affine=False) if IN else None
        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2 = nn.ModuleList([
            LightConv3x3(mid_channels, mid_channels),
            nn.Sequential(
                LightConv3x3(mid_channels, mid_channels),
                LightConv3x3(mid_channels, mid_channels),
            ),
            nn.Sequential(
                LightConv3x3(mid_channels, mid_channels),
                LightConv3x3(mid_channels, mid_channels),
                LightConv3x3(mid_channels, mid_channels),
            ),
            nn.Sequential(
                LightConv3x3(mid_channels, mid_channels),
                LightConv3x3(mid_channels, mid_channels),
                LightConv3x3(mid_channels, mid_channels),
                LightConv3x3(mid_channels, mid_channels),
            ),
        ])
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.IN is not None:
            x = self.IN(x)
        x1 = self.conv1(x)
        x2 = sum(c(x1) for c in self.conv2)
        x2 = self.gate(x2)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(x3 + identity)


class OSNetAIN(nn.Module):
    def __init__(self, blocks=[2, 2, 2], channels=[64, 256, 384, 512], feature_dim=512, IN_blocks=[True, True, False, True, True, False]):
        super().__init__()
        self.conv1 = Conv7x7(3, channels[0], stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Stages
        self.conv2 = nn.Sequential(
            OSBlockAIN(channels[0], channels[1], IN=True),
            OSBlockAIN(channels[1], channels[1], IN=True),
        )
        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2)
        )
        self.conv3 = nn.Sequential(
            OSBlockAIN(channels[1], channels[2], IN=False),
            OSBlockAIN(channels[2], channels[2], IN=True),
        )
        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2)
        )
        self.conv4 = nn.Sequential(
            OSBlockAIN(channels[2], channels[3], IN=True),
            OSBlockAIN(channels[3], channels[3], IN=False),
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        return self.fc(v)


def test_match():
    models_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    p2 = os.path.join(models_dir, "osnet_ain_x1_0_msmt17.pt")
    sd2 = torch.load(p2, map_location="cpu", weights_only=False)
    if "state_dict" in sd2: sd2 = sd2["state_dict"]
    
    model = OSNetAIN()
    model_dict = model.state_dict()
    filtered = {k.replace("module.", ""): v for k, v in sd2.items() if k.replace("module.", "") in model_dict and model_dict[k.replace("module.", "")].shape == v.shape}
    print(f"Matched {len(filtered)} / {len(model_dict)} tensors for OSNet-AIN-x1.0")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    x = torch.randn(2, 3, 256, 128)
    out = model(x)
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    test_match()
