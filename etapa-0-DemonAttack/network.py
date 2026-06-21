"""CNN Q-network for RGB stacked frames: input (batch, n_frames, H, W, 3)."""

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Three-layer CNN followed by two fully-connected layers."""

    def __init__(self, n_frames: int, height: int, width: int, n_actions: int):
        super().__init__()
        in_channels = n_frames * 3  # RGB × stacked frames flattened into channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out = self._conv_output_size(n_frames, height, width)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _conv_output_size(self, n_frames: int, height: int, width: int) -> int:
        dummy = torch.zeros(1, n_frames * 3, height, width)
        with torch.no_grad():
            out = self.conv(dummy)
        return int(out.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_frames, H, W, 3) → merge frames and channels
        b = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).contiguous()  # (b, n_frames, 3, H, W)
        x = x.view(b, -1, x.shape[3], x.shape[4])   # (b, n_frames*3, H, W)
        x = self.conv(x)
        x = x.view(b, -1)
        return self.fc(x)
