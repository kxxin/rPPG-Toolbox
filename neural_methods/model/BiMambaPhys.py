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


class ReferenceModel(nn.Module):
    """Runs the unmodified reference DualPhysMamba / RhythmMamba on toolbox NCDHW clips.

    The reference models take [B,T,C,H,W] (NDCHW); this adapter only permutes the input.
    DualPhysMamba keeps its own two-stage forward (stage 1 = rPPG path only); RhythmMamba is
    single-path, so only stage 2 exists.
    """

    def __init__(self, arch, dim, depth, mlp_ratio, drop_path_rate, cross_attn_heads, noise_path=True):
        super().__init__()
        self.arch = arch
        self.single_path = arch == 'bidualphysmamba' and not noise_path
        if arch == 'bidualphysmamba':
            from neural_methods.model.reference.BiDualPhysMamba import BiDualPhysMamba
            self.net = BiDualPhysMamba(depth=depth, embed_dim=dim, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate,
                                       cross_attn_heads=cross_attn_heads, single_path=self.single_path)
        elif arch == 'dualphysmamba':
            from neural_methods.model.reference.DualPhysMamba import DualPhysMamba
            self.net = DualPhysMamba(depth=depth, embed_dim=dim, mlp_ratio=mlp_ratio,
                                     drop_path_rate=drop_path_rate, cross_attn_heads=cross_attn_heads)
        elif arch == 'rhythmformer':
            # the toolbox's own (unmodified) RhythmFormer port; its size is fixed (DIM/DEPTH ignored)
            from neural_methods.model.RhythmFormer import RhythmFormer
            self.net = RhythmFormer()
        elif arch == 'rhythmmamba':
            from neural_methods.model.reference.RhythmMamba import RhythmMamba
            self.net = RhythmMamba(depth=depth, embed_dim=dim, mlp_ratio=mlp_ratio,
                                   drop_path_rate=drop_path_rate)
        else:
            raise ValueError('Unknown ARCH %r (bimamba | dualphysmamba | bidualphysmamba | rhythmmamba | rhythmformer)' % arch)

    def forward(self, video, stage=2):
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError('%s expects toolbox NCDHW clips [B,3,T,H,W]' % self.arch)
        if self.arch == 'rhythmformer' and tuple(video.shape[2:]) != (160, 128, 128):
            raise ValueError('RhythmFormer is built for 160-frame 128x128 clips, got %s' % (tuple(video.shape[2:]),))
        x = video.permute(0, 2, 1, 3, 4).contiguous()  # -> [B,T,C,H,W]
        if self.arch in ('dualphysmamba', 'bidualphysmamba') and not self.single_path:
            return self.net(x, stage=stage)
        if stage != 2:
            raise ValueError('%s is single-path here; use STAGE1_EPOCHS 0' % self.arch)
        return self.net(x)


class BiMambaPhys(nn.Module):
    def __init__(self, in_channels=3, dim=64, depth=2, d_state=16,
                 d_conv=4, expand=2, dropout=0.1, dual_path=False,
                 mlp_ratio=2, drop_path_rate=0.1, cross_attn_heads=4, noise_path=True, arch='bimamba'):
        super().__init__()
        self.in_channels = in_channels
        self.dual_path = dual_path
        self.arch = arch
        if arch != 'bimamba':
            if in_channels != 3:
                raise ValueError('Reference models require RGB input')
            self.reference = ReferenceModel(arch, dim, depth, mlp_ratio, drop_path_rate, cross_attn_heads, noise_path)
            return
        if dual_path:
            if in_channels != 3:
                raise ValueError('Dual-path fusion stem requires RGB input')
            from neural_methods.model.BiMambaPhysDual import DualPathBiMambaPhys
            self.dual = DualPathBiMambaPhys(dim, depth, d_state, d_conv, expand,
                                          dropout, mlp_ratio, drop_path_rate, cross_attn_heads,
                                          noise_path=noise_path)
            return
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

    def forward(self, video, stage=2):
        if self.arch != 'bimamba':
            return self.reference(video, stage=stage)
        if self.dual_path:
            return self.dual(video, stage=stage)
        if stage != 2:
            raise ValueError('Two-stage training requires DUAL_PATH=True')
        if video.ndim != 5 or video.shape[1] != self.in_channels:
            raise ValueError("BiMambaPhys expects [B,C,T,H,W] with C=%d" % self.in_channels)
        x = self.stem(video).mean(dim=(-2, -1)).transpose(1, 2)  # B,T,D
        x = self.norm(self.blocks(x)).transpose(1, 2)
        return self.head(x).squeeze(1)  # B,T; unrestricted signed waveform
