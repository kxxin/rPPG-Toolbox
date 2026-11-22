"""Frame CNN -> gated bidirectional temporal Mamba -> one sample per frame.

Mamba is the official mamba_ssm.Mamba (Mamba-1); the two directions have
independent parameters. No spatial patches enter the temporal sequence.
"""
import torch
from torch import nn
from mamba_ssm import Mamba


class BiMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.forward_mamba = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.backward_mamba = Mamba(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate = nn.Linear(2 * dim, dim)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.norm(x)
        forward = self.forward_mamba(z)
        # Reverse time only, then restore original frame alignment before fusion.
        backward = self.backward_mamba(z.flip(1)).flip(1)
        gate = torch.sigmoid(self.gate(torch.cat((forward, backward), dim=-1)))
        return x + self.dropout(self.projection(gate * forward + (1 - gate) * backward))


class BiMambaPhys(nn.Module):
    def __init__(self, in_channels=3, dim=64, depth=2, d_state=16,
                 d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        # (1,k,k) convolutions implement a shared frame CNN without downsampling T.
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, (1, 5, 5), stride=(1, 2, 2), padding=(0, 2, 2), bias=False),
            nn.BatchNorm3d(16), nn.SiLU(),
            nn.Conv3d(16, 32, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(32), nn.SiLU(),
            nn.Conv3d(32, dim, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(dim), nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[
            BiMambaBlock(dim, d_state, d_conv, expand, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Conv1d(dim, dim // 2, kernel_size=5, padding=2),
            nn.SiLU(), nn.Conv1d(dim // 2, 1, kernel_size=1),
        )

    def forward(self, video):
        if video.ndim != 5 or video.shape[1] != self.in_channels:
            raise ValueError("BiMambaPhys expects [B,C,T,H,W] with C=%d" % self.in_channels)
        x = self.stem(video).mean(dim=(-2, -1)).transpose(1, 2)  # B,T,D
        x = self.norm(self.blocks(x)).transpose(1, 2)
        return self.head(x).squeeze(1)  # B,T; unrestricted signed waveform
