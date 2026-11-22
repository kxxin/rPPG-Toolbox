"""DualPhysMamba reproduction with BIDIRECTIONAL Mamba (the framework: DualPhysMamba backbone, Mamba -> Bi-Mamba).

Identical to reference/DualPhysMamba.py except for the Mamba scan inside the multi-temporal Block_mamba
of BOTH paths, which becomes bidirectional:

    z = LayerNorm(x_copy)
    f = Mamba_fwd(z)                                   # forward scan, exactly as the reference
    b = backward scan of the VALID part of each copy   # see below
    g = sigmoid(Linear([f, b]));  out = g * f + (1 - g) * b

Multi-temporal constraint and why the backward scan needs care (v2 fix):
  The reference block runs Mamba on 4 copies of the sequence; copy k is shifted left by k*T/4 so its first
  T - k*T/4 steps are x[k*T/4:], and its last k*T/4 steps are STALE leftovers of the unshifted sequence.
  The aggregation only reads the valid prefix of each copy, and a causal (forward) scan never lets the
  stale tail reach that prefix. A plain backward scan over the whole copy would start in the stale tail
  and leak it into every position that is read. v1 of this file did that.
  v2 runs the backward scan over each copy's valid prefix only: flip(valid) + zero padding at the END of
  the flipped sequence (never seen by earlier outputs of a causal scan) -> Mamba_bwd -> un-flip.

Same d_state 48 / d_conv 4 / expand 2 as the reference; independent forward/backward parameters; the
stem, masks, FDF, noise path, cross-attention, subtraction, predictor and init are the reference code.

single_path=True drops the noise path and cross-attention (rPPG path only = stage-1 forward), i.e. a
bidirectional RhythmMamba-style student of the same family.
"""
from functools import partial

import torch
from torch import nn

from mamba_ssm.modules.mamba_simple import Mamba
from neural_methods.model.reference.DualPhysMamba import (Block_mamba, DualPhysMamba, _init_weights,
                                                         segm_init_weights)

PATHS = 3  # as the reference Block_mamba.forward (segment = 2 ** (PATHS - 1) = 4 copies)


class BiMambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.gate = nn.Linear(2 * dim, dim)

    def forward(self, x, valid=None):
        """x: [copies*B, D, C]; valid: list of valid prefix lengths per copy (None = whole sequence)."""
        z = self.norm(x)
        f = self.mamba_fwd(z)
        if valid is None:
            b = self.mamba_bwd(z.flip(1)).flip(1)
        else:
            n, d, _ = z.shape
            batch = n // len(valid)
            zb = torch.zeros_like(z)
            for k, length in enumerate(valid):  # flip only the valid prefix; zeros go after it
                zb[k * batch:(k + 1) * batch, :length] = z[k * batch:(k + 1) * batch, :length].flip(1)
            rb = self.mamba_bwd(zb)
            b = torch.zeros_like(rb)
            for k, length in enumerate(valid):
                b[k * batch:(k + 1) * batch, :length] = rb[k * batch:(k + 1) * batch, :length].flip(1)
        g = torch.sigmoid(self.gate(torch.cat((f, b), dim=-1)))
        return g * f + (1 - g) * b


class BiBlockMamba(Block_mamba):
    """Reference Block_mamba with the bidirectional layer; forward copied from the reference except that
    the layer is told each copy's valid prefix length."""

    def __init__(self, dim, mlp_ratio, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__(dim, mlp_ratio, drop_path=drop_path, norm_layer=norm_layer)
        self.attn = BiMambaLayer(dim)
        self.attn.apply(self._init_weights)

    def forward(self, x):
        B, D, C = x.size()
        segment = 2 ** (PATHS - 1)
        tt = D // segment
        x_r = x.repeat(segment, 1, 1)
        x_o = x_r.clone()
        for i in range(1, segment):
            x_o[i * B:(i + 1) * B, :D - i * tt, :] = x_r[i * B:(i + 1) * B, i * tt:, :]
        x_o = self.attn(x_o, valid=[D - i * tt for i in range(segment)])
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


class BiDualPhysMamba(DualPhysMamba):
    def __init__(self, depth=12, embed_dim=96, mlp_ratio=2, drop_path_rate=0.1, single_path=False, **kwargs):
        super().__init__(depth=depth, embed_dim=embed_dim, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate, **kwargs)
        self.single_path = single_path
        if single_path:
            del self.blocks_noise, self.cross_attn
        inter_dpr = [0.0] + [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # as the reference
        names = ['blocks_rppg'] if single_path else ['blocks_rppg', 'blocks_noise']
        for name in names:
            blocks = nn.ModuleList([BiBlockMamba(embed_dim, mlp_ratio, drop_path=inter_dpr[i], norm_layer=nn.LayerNorm)
                                    for i in range(depth)])
            # the initialisation the reference model applies to all of its modules
            blocks.apply(segm_init_weights)
            blocks.apply(partial(_init_weights, n_layer=depth))
            setattr(self, name, blocks)

    def forward(self, x, stage=2):
        if self.single_path:
            return super().forward(x, stage=1)  # rPPG path only
        return super().forward(x, stage=stage)
