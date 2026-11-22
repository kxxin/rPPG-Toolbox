"""Feature-level KD from the dual-path teacher to the single-path student (DISTILL.FEATURE).

All terms compare the student's aux outputs with the frozen teacher's (forward(..., return_aux=True)).
Teacher and student share the stem, the spatial attention and the feature width (DIM), so no projector
is needed; a width mismatch raises.

  AT_WEIGHT     attention transfer (Zagoruyko & Komodakis 2017) on the spatial attention mask
                [B,1,T',H',W']: per frame, L2-normalise the flattened map, then the L2 distance.
                Range 0..2. Teaches where the skin is.
  HINT_WEIGHT   rPPG-path block hints (FitNets-style, cosine): student block k (of Ns) is matched to
                teacher block round((k+1)*Nt/Ns)-1 (6 <- 12: teacher blocks 2,4,...,12, 1-based);
                1 - cosine over channels per time step, averaged over blocks. Range 0..2.
  FINAL_WEIGHT  the feature that feeds the predictor: teacher 'clean' (rppg - modeled noise) vs the
                student's final rPPG feature; 1 - cosine per time step.
  TSM_WEIGHT    relational: temporal self-similarity matrices (T' x T' cosine similarity of the final
                features) of teacher and student, MSE. Transfers rhythm structure, not exact values.
The noise path is not distilled (the noise-path diagnostic showed it is ~1-2 % of the teacher output,
and the single-path student has none).
"""
import torch
from torch import nn
from torch.nn import functional as F


def block_mapping(student_blocks, teacher_blocks):
    return [round((k + 1) * teacher_blocks / student_blocks) - 1 for k in range(student_blocks)]


def attention_transfer(student_mask, teacher_mask):
    s = F.normalize(student_mask.flatten(3).squeeze(1), dim=-1)  # B,T',H'W'
    t = F.normalize(teacher_mask.flatten(3).squeeze(1), dim=-1)
    return (s - t).norm(dim=-1).mean()


def cosine_distance(student, teacher):
    """[B,T,C] -> mean over B,T of 1 - cos."""
    return (1 - F.cosine_similarity(student, teacher, dim=-1)).mean()


def temporal_self_similarity(x):
    x = F.normalize(x, dim=-1)
    return x @ x.transpose(1, 2)


class FeatureDistillationLoss(nn.Module):
    def __init__(self, at_weight=0.0, hint_weight=0.0, final_weight=0.0, tsm_weight=0.0):
        super().__init__()
        self.weights = dict(at=float(at_weight), hint=float(hint_weight), final=float(final_weight),
                            tsm=float(tsm_weight))
        if min(self.weights.values()) < 0:
            raise ValueError('DISTILL.FEATURE weights must be nonnegative')

    @property
    def active(self):
        return any(w > 0 for w in self.weights.values())

    def forward(self, student_aux, teacher_aux):
        t_final = teacher_aux.get('clean', teacher_aux['rppg']).detach()
        s_final = student_aux['rppg']
        if s_final.shape[-1] != t_final.shape[-1]:
            raise ValueError('Teacher/student feature width differs (%d vs %d): feature KD needs equal DIM'
                             % (t_final.shape[-1], s_final.shape[-1]))
        zero = s_final.new_zeros(())
        parts = dict(at=zero, hint=zero, final=zero, tsm=zero)
        if self.weights['at'] > 0:
            parts['at'] = attention_transfer(student_aux['mask'], teacher_aux['mask'].detach())
        if self.weights['hint'] > 0:
            s_blocks, t_blocks = student_aux['rppg_blocks'], teacher_aux['rppg_blocks']
            mapping = block_mapping(len(s_blocks), len(t_blocks))
            parts['hint'] = torch.stack([cosine_distance(s, t_blocks[j].detach())
                                         for s, j in zip(s_blocks, mapping)]).mean()
        if self.weights['final'] > 0:
            parts['final'] = cosine_distance(s_final, t_final)
        if self.weights['tsm'] > 0:
            parts['tsm'] = F.mse_loss(temporal_self_similarity(s_final), temporal_self_similarity(t_final))
        total = sum(self.weights[k] * v for k, v in parts.items())
        return total, parts


def build_feature_loss(config):
    f = getattr(config.DISTILL, 'FEATURE', None)
    if f is None:
        return FeatureDistillationLoss()
    return FeatureDistillationLoss(f.AT_WEIGHT, f.HINT_WEIGHT, f.FINAL_WEIGHT, f.TSM_WEIGHT)
