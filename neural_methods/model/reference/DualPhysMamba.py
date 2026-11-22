"""
DualPhysMamba: Efficient end-to-end network based on noise modeling for
remote photoplethysmography measurement.  (Li & Peng, Applied Soft Computing 2026)

Reproduction built on top of the RhythmMamba backbone.

The paper reuses several RhythmMamba components without restating their
internals (Frame/Fusion Stem, self-attention mask, the Multi-temporal
Constraint Mamba block, and the Frequency-Domain Feed-forward / FDF).
Those parts are taken from RhythmMamba here.  DualPhysMamba adds:
  * a Noise Path fed by the *inverse* attention region,
  * a Cross-attention that lets the noise path attend to the noisy rPPG
    features (Q from noise path, K/V from rPPG path),
  * subtraction of the modeled noise from the noisy rPPG feature,
  * a two-stage training schedule (see DualPhysMambaTrainer).

Overall data flow (Fig. 2 of the paper):

    X ---FrameStem---> X_stem ---stem3(3D,stride2 on T)---> f  [B,C,T/2,H/8,W/8]
                                                         |
                        mask = Attn(sigmoid(f)) ---------+--------- inv = Attn(1 - sigmoid(f))
                                |                                        |
       rPPG path:  f*mask -> avgpool -> Mamba x N -> X_rPPG   noise path: f*inv -> avgpool -> Mamba x N -> h
                                |                                        |
                                +-----------> CrossAttention (Q=h, K,V=X_rPPG) -> X_noise
                                |                                        |
   stage1: pred = Predictor(X_rPPG)                                     |
   stage2: pred = Predictor(X_rPPG - X_noise)  <-----------------------+
"""
import math
from functools import partial

import torch
import torch.fft
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
from torch import nn

from mamba_ssm.modules.mamba_simple import Mamba


# ---------------------------------------------------------------------------
#  Components reused from RhythmMamba
# ---------------------------------------------------------------------------
class Fusion_Stem(nn.Module):
    """Frame Stem with difference fusion (RhythmMamba)."""

    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super().__init__()
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),
        )
        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """x: [N, D, C, H, W] -> [N*D, dim, H/8, W/8]"""
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], 1)
        x2 = torch.cat([x[:, :1], x[:, :D - 1]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D - 1:]], 1)
        x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], 1)
        x_diff = self.stem12(
            torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 12, H, W)
        )
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)
        # fusion layer 1
        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        # fusion layer 2
        x_path2 = self.stem22(x_diff)
        x = self.apha * x_path1 + self.belta * x_path2
        return x


class Attention_mask(nn.Module):
    """Normalizes a soft attention map so it sums to (H*W*0.5)."""

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5


class Frequencydomain_FFN(nn.Module):
    """Frequency-Domain Feed-forward (FDF) from RhythmMamba."""

    def __init__(self, dim, mlp_ratio):
        super().__init__()
        self.scale = 0.02
        self.dim = dim * mlp_ratio
        self.r = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.dim, self.dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.dim))
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, dim * mlp_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim * mlp_ratio),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(dim * mlp_ratio, dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x_fre = torch.fft.fft(x, dim=1, norm='ortho')  # FFT along temporal dim
        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r)
            - torch.einsum('bnc,cc->bnc', x_fre.imag, self.i)
            + self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r)
            + torch.einsum('bnc,cc->bnc', x_fre.real, self.i)
            + self.ib
        )
        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)
        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)
        return x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        return self.mamba(self.norm(x))


class Block_mamba(nn.Module):
    """Mamba Block = Multi-temporal Constraint Mamba layer + FDF (RhythmMamba)."""

    def __init__(self, dim, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, D, C = x.size()
        # Multi-temporal Constraint (parallel multi-scale temporal scanning)
        path = 3
        segment = 2 ** (path - 1)
        tt = D // segment
        x_r = x.repeat(segment, 1, 1)
        x_o = x_r.clone()
        for i in range(1, segment):
            x_o[i * B:(i + 1) * B, :D - i * tt, :] = x_r[i * B:(i + 1) * B, i * tt:, :]
        x_o = self.attn(x_o)
        for i in range(1, segment):
            for j in range(i):
                x_o[0:B, tt * i: tt * (i + 1), :] = (
                    x_o[0:B, tt * i: tt * (i + 1), :]
                    + x_o[B * (j + 1):B * (j + 2), tt * (i - j - 1): tt * (i - j), :]
                )
            x_o[0:B, tt * i: tt * (i + 1), :] = x_o[0:B, tt * i: tt * (i + 1), :] / (i + 1)
        x = x + self.drop_path(self.norm1(x_o[0:B]))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
#  DualPhysMamba-specific: Cross-attention (Fig. 3, Eq. 9)
# ---------------------------------------------------------------------------
class CrossAttention(nn.Module):
    """Cross-attention between the noise path and the rPPG path.

    Query  Q comes from the noise-path feature (X_noise_in),
    Key/Value K,V come from the noisy rPPG feature (X_rPPG).
        Att = softmax(Q K^T / sqrt(d)) V
    The output is the refined noise feature X_noise.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)   # from noise path
        self.k_proj = nn.Linear(dim, dim, bias=False)   # from rPPG path
        self.v_proj = nn.Linear(dim, dim, bias=False)   # from rPPG path
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_noise, x_rppg):
        """x_noise, x_rppg: [B, T, C] -> [B, T, C] refined noise feature."""
        B, T, C = x_noise.shape
        q = self.q_proj(x_noise)
        k = self.k_proj(x_rppg)
        v = self.v_proj(x_rppg)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v                                  # [B, heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        # residual on the noise-path query, then norm
        return self.norm(x_noise + out)


# ---------------------------------------------------------------------------
#  Weight-init helpers (RhythmMamba)
# ---------------------------------------------------------------------------
def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# ---------------------------------------------------------------------------
#  DualPhysMamba
# ---------------------------------------------------------------------------
class DualPhysMamba(nn.Module):
    def __init__(self,
                 depth=12,               # 12 Mamba blocks (paper's optimal)
                 embed_dim=96,
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 cross_attn_heads=4,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 **kwargs):
        super().__init__()
        self.embed_dim = embed_dim

        # ---- shared front-end (Frame Stem + stem3) ----
        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()
        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim // 4, embed_dim, kernel_size=(2, 5, 5),
                      stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr

        # ---- rPPG path Mamba blocks ----
        self.blocks_rppg = nn.ModuleList([
            Block_mamba(dim=embed_dim, mlp_ratio=mlp_ratio,
                        drop_path=inter_dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])
        # ---- Noise path Mamba blocks (same setting as rPPG path) ----
        self.blocks_noise = nn.ModuleList([
            Block_mamba(dim=embed_dim, mlp_ratio=mlp_ratio,
                        drop_path=inter_dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])

        # ---- Cross-attention (noise queries rPPG) ----
        self.cross_attn = CrossAttention(embed_dim, num_heads=cross_attn_heads)

        # ---- Predictor (shared) ----
        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1, stride=1, padding=0)

        # init
        self.apply(segm_init_weights)
        self.apply(partial(_init_weights, n_layer=depth,
                           **(initializer_cfg if initializer_cfg is not None else {})))

    # -- shared front-end producing the spatio-temporal feature map + mask --
    def _frontend(self, x):
        B, D, C, H, W = x.shape
        x = self.Fusion_Stem(x)                                  # [B*D, C/4, H/8, W/8]
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        f = self.stem3(x)                                        # [B, C, T/2, H/8, W/8]
        mask = self.attn_mask(torch.sigmoid(f))                  # attention (face)
        inv_mask = self.attn_mask(torch.sigmoid(-f))             # inverse attention (bg)
        return f, mask, inv_mask

    @staticmethod
    def _spatial_pool(feat):
        """[B, C, T, H, W] -> [B, T, C] via spatial average pooling."""
        feat = torch.mean(feat, 4)
        feat = torch.mean(feat, 3)
        return rearrange(feat, 'b c t -> b t c')

    def _predict(self, seq):
        """[B, T, C] -> [B, 2T] rPPG signal."""
        rPPG = seq.permute(0, 2, 1)          # [B, C, T]
        rPPG = self.upsample(rPPG)           # [B, C, 2T]
        rPPG = self.ConvBlockLast(rPPG)      # [B, 1, 2T]
        return rPPG.squeeze(1)

    def forward(self, x, stage=2):
        """
        stage=1 : train only the rPPG path (self-attention learns to focus on
                  the face); prediction uses X_rPPG directly.
        stage=2 : joint training; noise is modeled from the inverse-attention
                  region and subtracted from X_rPPG.
        """
        f, mask, inv_mask = self._frontend(x)

        # ---- rPPG path ----
        x_att = f * mask
        x_rppg = self._spatial_pool(x_att)                # [B, T/2, C]
        for blk in self.blocks_rppg:
            x_rppg = blk(x_rppg)
        X_rPPG = x_rppg

        if stage == 1:
            return self._predict(X_rPPG)

        # ---- Noise path ----
        x_inv = f * inv_mask
        x_noise = self._spatial_pool(x_inv)               # [B, T/2, C]
        for blk in self.blocks_noise:
            x_noise = blk(x_noise)
        # cross-attention: noise attends to noisy rPPG features
        X_noise = self.cross_attn(x_noise, X_rPPG)         # [B, T/2, C]

        # remove noise from the noisy rPPG feature
        clean = X_rPPG - X_noise
        return self._predict(clean)
