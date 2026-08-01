"""Custom model modules for GF-InsuYOLO.

These modules are intentionally small and dependency-light. They are registered
by train.py before Ultralytics parses the model YAML.
"""

from __future__ import annotations

import torch
from torch import nn


class FrequencyEnhance(nn.Module):
    """Add a lightweight high-frequency residual to early feature maps.

    The module preserves the input shape and channel count, so it can be inserted
    into an Ultralytics YAML without changing the following layer dimensions.
    """

    def __init__(self, strength: float = 0.12, cutoff_ratio: float = 0.18) -> None:
        super().__init__()
        self.strength = float(strength)
        self.cutoff_ratio = float(cutoff_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.strength <= 0 or x.ndim != 4:
            return x
        height, width = x.shape[-2:]
        if height < 8 or width < 8:
            return x

        freq = torch.fft.rfft2(x.float(), norm="ortho")
        mask = self._high_pass_mask(height, width, x.device, x.dtype)
        high = torch.fft.irfft2(freq * mask, s=(height, width), norm="ortho").to(x.dtype)
        return x + self.strength * high

    def _high_pass_mask(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        yy = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(height, 1)
        xx = torch.linspace(0.0, 1.0, width // 2 + 1, device=device, dtype=dtype).view(1, -1)
        radius = torch.sqrt(yy * yy + xx * xx)
        cutoff = max(0.01, min(self.cutoff_ratio, 0.95))
        return (radius >= cutoff).view(1, 1, height, width // 2 + 1)


class ContextGuidedEnhance(nn.Module):
    """Fuse local, dilated surrounding, and global channel context.

    This is a dependency-light adaptation of the Context Guided block used by
    TLINet. It preserves feature shape so it can refine YOLO detection features
    without changing the feature-pyramid contract.
    """

    def __init__(self, channels: int, dilation: int = 2, reduction: int = 16) -> None:
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("channels must be an even integer >= 2")
        hidden = channels // 2
        squeeze = max(channels // reduction, 4)
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.surround = nn.Conv2d(
            hidden,
            hidden,
            3,
            padding=dilation,
            dilation=dilation,
            groups=hidden,
            bias=False,
        )
        self.fuse = nn.Sequential(nn.BatchNorm2d(channels), nn.SiLU(inplace=True))
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeeze, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze, channels, 1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        context = self.fuse(torch.cat((self.local(reduced), self.surround(reduced)), dim=1))
        context = context * self.channel_gate(context)
        return x + self.residual_scale * context
