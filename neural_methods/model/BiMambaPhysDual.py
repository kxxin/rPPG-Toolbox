"""Dual-path BiMambaPhys, two-stage version (DualPhysMamba paper alignment).

Fusion stem, add/norm placement and FDF: RhythmMamba.py / DualPhysMamba.py.
Q=noise, K/V=rPPG and noise subtraction: DualPhysMamba.py. The temporal mixer
is the BiMambaPhys gated pair of independent stock Mamba-1 modules.

Changes vs. the previous BiMambaPhysDual (bimamba_dual_experiment):
  1. Spatial attention mask (DualPhysMamba Eq. 4, DeepPhys/CAN lineage):
     a 1x1x1 Conv3d maps the stem3 features to ONE logit map per frame,
     Mask = (H*W) * sigmoid(z) / (2 * ||sigmoid(z)||_1).
     The inverse mask is the complement sigmoid(-z) = 1 - sigmoid(z), taken
     BEFORE normalization (non-negative; Nowara et al. ICCV 2021 complement of
     the attention map) and normalized the same way. Both maps are broadcast
     over all channels, so the noise path really sees the spatial region the
     attention ignores. The old version gated each of the C channels with its
     own sigmoid(features) and had no Conv.
  2. Stage argument: stage=1 runs the rPPG path only and predicts from it;
     stage=2 (default; validation/test/teacher) runs the full dual path.
     The schedule lives in BiMambaPhysTrainer (MODEL.BIMAMBAPHYS.STAGE1_EPOCHS).
  4. return_aux also returns 'rppg_blocks' (output of every rPPG-path block) for feature KD.
  3. noise_path=False (MODEL.BIMAMBAPHYS.NOISE_PATH): single-path model. Same stem,
     spatial attention mask and rPPG Bi-Mamba path; no noise blocks, no cross-attention.
     Motivated by the noise-path diagnostic on UBFC-rPPG (subtracted signal ~1-2 % of
     the output, identical HR with it switched off). forward() ignores `stage`.
"""
import torch
from torch import nn
from torch.nn import functional as F
from mamba_ssm import Mamba


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep


class FusionStem(nn.Module):
    """RGB + four adjacent-frame differences, with replicated temporal endpoints."""
    def __init__(self, dim):
        super().__init__()
        def conv(cin, cout, stride):
            return nn.Sequential(nn.Conv2d(cin, cout, 7, stride, 3),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))
        self.stem11 = conv(3, dim // 2, 2)
        self.stem12 = conv(12, dim // 2, 2)
        self.stem21 = conv(dim // 2, dim, 1)
        self.stem22 = conv(dim // 2, dim, 1)

    def forward(self, video):
        x = video.permute(0, 2, 1, 3, 4)  # B,T,C,H,W
        b, t, c, h, w = x.shape
        index = torch.arange(t, device=x.device)
        neighbors = [x.index_select(1, (index + offset).clamp(0, t - 1))
                     for offset in (-2, -1, 0, 1, 2)]
        diff = torch.cat([neighbors[i+1] - neighbors[i] for i in range(4)], dim=2)
        diff = self.stem12(diff.reshape(b*t, 12, h, w))
        rgb = self.stem11(x.reshape(b*t, c, h, w))
        fused = 0.5 * self.stem21(0.5 * rgb + 0.5 * diff) + 0.5 * self.stem22(diff)
        return fused.reshape(b, t, *fused.shape[1:]).permute(0, 2, 1, 3, 4)


def attention_masks(logits):
    """logits: [B,1,T,H,W] single-channel spatial map per frame.

    Eq. 4: Mask = H*W*sigmoid(z) / (2*||sigmoid(z)||_1), i.e. spatial mean 0.5
    per frame. Inverse = same normalization applied to 1 - sigmoid(z).
    """
    if logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError('attention logits must be [B,1,T,H,W]')
    signal = torch.sigmoid(logits)
    inverse = torch.sigmoid(-logits)  # == 1 - sigmoid(logits), numerically stable
    def normalize(mask):
        return 0.5 * mask / mask.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return normalize(signal), normalize(inverse)


class FrequencyDomainFFN(nn.Module):
    """Reference FDF: expand -> temporal FFT -> complex diagonal map -> IFFT."""
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()
        hidden = dim * mlp_ratio
        self.r = nn.Parameter(0.02 * torch.randn(hidden, hidden))
        self.i = nn.Parameter(0.02 * torch.randn(hidden, hidden))
        self.rb = nn.Parameter(0.02 * torch.randn(hidden))
        self.ib = nn.Parameter(0.02 * torch.randn(hidden))
        self.fc1 = nn.Sequential(nn.Conv1d(dim, hidden, 1, bias=False),
                                 nn.BatchNorm1d(hidden), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Conv1d(hidden, dim, 1, bias=False), nn.BatchNorm1d(dim))

    def forward(self, x):
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        # FFT in float32 also supports lengths that are not powers of two.
        freq = torch.fft.fft(x.float(), dim=1, norm='ortho')
        real = F.relu(torch.einsum('bnc,cc->bnc', freq.real, self.r)
                      - torch.einsum('bnc,cc->bnc', freq.imag, self.i) + self.rb)
        imag = F.relu(torch.einsum('bnc,cc->bnc', freq.imag, self.r)
                      + torch.einsum('bnc,cc->bnc', freq.real, self.i) + self.ib)
        # The reference casts complex IFFT to float; make its real-part operation explicit.
        x = torch.fft.ifft(torch.complex(real, imag), dim=1, norm='ortho').real
        return self.fc2(x.transpose(1, 2)).transpose(1, 2)


class BiMambaLayer(nn.Module):
    """BiMambaPhys mixer WITHOUT a residual; the outer block owns add/norm."""
    def __init__(self, dim, d_state, d_conv, expand, dropout):
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
        backward = self.backward_mamba(z.flip(1)).flip(1)
        gate = torch.sigmoid(self.gate(torch.cat((forward, backward), dim=-1)))
        return self.dropout(self.projection(gate * forward + (1 - gate) * backward))


class BiMambaBlock(nn.Module):
    def __init__(self, dim, d_state, d_conv, expand, dropout, mlp_ratio, drop_path):
        super().__init__()
        self.attn = BiMambaLayer(dim, d_state, d_conv, expand, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FrequencyDomainFFN(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.norm1(self.attn(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class CrossAttention(nn.Module):
    """Noise queries rPPG; residual and normalization match the supplied reference."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, noise, rppg):
        b, t, c = noise.shape
        def split(x): return x.reshape(b, t, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj(noise)), split(self.k_proj(rppg)), split(self.v_proj(rppg))
        weights = ((q @ k.transpose(-2, -1)) * self.head_dim ** -0.5).softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, t, c)
        return self.norm(noise + self.out_proj(out))


class DualPathBiMambaPhys(nn.Module):
    def __init__(self, dim=64, depth=12, d_state=16, d_conv=4, expand=2,
                 dropout=0.1, mlp_ratio=2, drop_path_rate=0.1, cross_attn_heads=4, noise_path=True):
        super().__init__()
        if dim < 8 or dim % 8 or cross_attn_heads < 1 or dim % cross_attn_heads:
            raise ValueError('DIM must be a multiple of 8 and divisible by CROSS_ATTN_HEADS')
        if depth < 1 or not isinstance(mlp_ratio, int) or mlp_ratio < 1:
            raise ValueError('DEPTH and integer MLP_RATIO must be positive')
        if not 0 <= drop_path_rate < 1:
            raise ValueError('DROP_PATH_RATE must be in [0,1)')
        self.frame_stem = FusionStem(dim // 4)
        self.stem3 = nn.Sequential(nn.Conv3d(dim // 4, dim, (2, 5, 5), (2, 1, 1), (0, 2, 2)),
                                   nn.BatchNorm3d(dim))
        # Eq. 4 Conv: features -> one spatial attention logit map per frame.
        self.attention_conv = nn.Conv3d(dim, 1, kernel_size=1)
        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        def blocks():
            return nn.Sequential(*[BiMambaBlock(dim, d_state, d_conv, expand, dropout,
                                                mlp_ratio, rate) for rate in rates])
        self.noise_path = bool(noise_path)
        self.blocks_rppg = blocks()
        if self.noise_path:
            self.blocks_noise = blocks()
            self.cross_attn = CrossAttention(dim, cross_attn_heads)
        # Reference pointwise predictor: no zero-padded temporal k=5 output layer.
        self.predictor = nn.Conv1d(dim, 1, 1)
        # Preserve stock Mamba's dt/A/D initialization and no-weight-decay flags.

    def _predict(self, sequence, t):
        return self.predictor(F.interpolate(sequence.transpose(1, 2), scale_factor=2,
                                            mode='nearest')).squeeze(1)[:, :t]

    def forward(self, video, stage=2, return_aux=False):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 2 or min(video.shape[-2:]) < 8:
            raise ValueError('Expected RGB [B,3,T,H,W], T>=2 and H,W>=8')
        if stage not in (1, 2):
            raise ValueError('stage must be 1 (rPPG path only) or 2 (dual path)')
        t = video.shape[2]
        features = self.frame_stem(video)
        if t % 2:
            features = torch.cat((features, features[:, :, -1:]), dim=2)
        features = self.stem3(features)
        logits = self.attention_conv(features)            # B,1,T',H',W'
        mask, inverse = attention_masks(logits)           # broadcast over channels
        x = (features * mask).mean((-2, -1)).transpose(1, 2)
        if return_aux:  # identical to self.blocks_rppg(x), keeping every block output for feature KD
            rppg_blocks = []
            for block in self.blocks_rppg:
                x = block(x); rppg_blocks.append(x)
            rppg = x
        else:
            rppg = self.blocks_rppg(x)
        if stage == 1 or not self.noise_path:
            # Stage 1 or single-path model: noise path/cross-attention not run (no grads).
            prediction = self._predict(rppg, t)
            if return_aux:
                return prediction, dict(attention_logits=logits, mask=mask,
                                        inverse_mask=inverse, rppg=rppg, rppg_blocks=rppg_blocks)
            return prediction
        noise = self.blocks_noise((features * inverse).mean((-2, -1)).transpose(1, 2))
        modeled_noise = self.cross_attn(noise, rppg)
        clean = rppg - modeled_noise
        prediction = self._predict(clean, t)
        if return_aux:
            return prediction, dict(attention_logits=logits, mask=mask, inverse_mask=inverse, rppg=rppg,
                                    rppg_blocks=rppg_blocks, noise=noise, modeled_noise=modeled_noise, clean=clean)
        return prediction
