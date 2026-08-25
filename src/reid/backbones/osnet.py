"""Omni-Scale Network (OSNet) for Person Re-Identification.

Reference:
    Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV 2019.
    https://arxiv.org/abs/1905.00953
"""

from __future__ import annotations

import logging
from typing import List, Optional
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

logger = logging.getLogger(__name__)

# Official & HuggingFace verified pretrained model URLs
OSNET_URLS = {
    "osnet_x0_25_msmt17": "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt",
    "osnet_x0_25_market": "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt",
}


class Conv1x1(nn.Module):
    """1x1 convolution + batchnorm + relu."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + batchnorm (linear, no relu)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Conv3x3(nn.Module):
    """3x3 convolution + batchnorm + relu."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 convolution (1x1 conv + 3x3 depthwise conv)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    """Channel-attention aggregation gate."""

    def __init__(self, in_channels: int, reduction_rate: int = 16) -> None:
        super().__init__()
        reduced_channels = max(1, in_channels // reduction_rate)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.global_avgpool(x)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w))
        return x * w


class OSBlock(nn.Module):
    """Omni-scale residual block combining multi-scale stream receptive fields."""

    def __init__(self, in_channels: int, out_channels: int, bottleneck_reduction: int = 4) -> None:
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid_channels)

        # 4 multi-scale receptive field streams
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )

        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)

        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)

        x2 = self.gate(x2a + x2b + x2c + x2d)
        x3 = self.conv3(x2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(x3 + identity)
        return out


class OSNet(nn.Module):
    """OSNet person Re-Identification feature extraction backbone."""

    def __init__(
        self,
        blocks: List[int] = [2, 2, 2],
        channels: List[int] = [64, 256, 384, 512],
        feature_dim: int = 512,
        loss: str = "softmax",
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        # Initial conv layer
        self.conv1 = Conv7x7(3, channels[0], stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stage 1
        self.conv2 = self._make_stage(channels[0], channels[1], blocks[0])
        self.transition1 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),
        )

        # Stage 2
        self.conv3 = self._make_stage(channels[1], channels[2], blocks[1])
        self.transition2 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2),
        )

        # Stage 3
        self.conv4 = self._make_stage(channels[2], channels[3], blocks[2])

        # Final projection layers
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(channels[3], feature_dim)

    def _construct_fc_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = [OSBlock(in_channels, out_channels)]
        for _ in range(1, num_blocks):
            layers.append(OSBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.transition1(self.conv2(x))
        x = self.transition2(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        feat = self.fc(v)
        return feat


class Conv7x7(nn.Module):
    """7x7 convolution + batchnorm + relu."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def _load_pretrained_weights(model: nn.Module, url: str) -> None:
    """Loads state dict from URL and maps parameters into the model."""
    try:
        state_dict = load_state_dict_from_url(url, map_location="cpu", progress=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace("module.", "")
            if k_clean in model_dict and model_dict[k_clean].shape == v.shape:
                filtered_dict[k_clean] = v

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Loaded {len(filtered_dict)}/{len(model_dict)} pretrained OSNet weights from {url}")
    except Exception as e:
        logger.warning(f"Could not load OSNet pretrained weights from URL: {e}")


def osnet_x0_25(pretrained: bool = True, dataset: str = "market1501") -> OSNet:
    """Instantiates lightweight OSNet-x0.25 model (~0.2M params, 512D output)."""
    model = OSNet(
        blocks=[2, 2, 2],
        channels=[16, 64, 96, 128],
        feature_dim=512,
    )
    if pretrained:
        url_key = f"osnet_x0_25_{dataset}" if f"osnet_x0_25_{dataset}" in OSNET_URLS else "osnet_x0_25_market"
        url = OSNET_URLS[url_key]
        _load_pretrained_weights(model, url)
    return model


def osnet_x1_0(pretrained: bool = True, dataset: str = "market1501") -> OSNet:
    """Instantiates standard OSNet-x1.0 model (~2.2M params, 512D output)."""
    model = OSNet(
        blocks=[2, 2, 2],
        channels=[64, 256, 384, 512],
        feature_dim=512,
    )
    if pretrained:
        url_key = f"osnet_x1_0_{dataset}" if f"osnet_x1_0_{dataset}" in OSNET_URLS else "osnet_x1_0_market"
        url = OSNET_URLS[url_key]
        _load_pretrained_weights(model, url)
    return model


def build_osnet(model_name: str = "osnet_x0_25", pretrained: bool = True) -> OSNet:
    """Factory helper to build OSNet variants."""
    if "x0_25" in model_name or "small" in model_name:
        return osnet_x0_25(pretrained=pretrained)
    elif "x1_0" in model_name:
        return osnet_x1_0(pretrained=pretrained)
    else:
        return osnet_x0_25(pretrained=pretrained)
