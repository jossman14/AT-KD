"""AT-KD loss (Zagoruyko & Komodakis, ICLR 2017) and Vanilla-KD baseline (Hinton 2015)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _attention(feat: torch.Tensor) -> torch.Tensor:
    """Sum-of-squared activations along the channel axis -> single-channel attention map."""
    return feat.pow(2).mean(dim=1, keepdim=True)


def _normalize(attn: torch.Tensor) -> torch.Tensor:
    B = attn.size(0)
    flat = attn.view(B, -1)
    return F.normalize(flat, p=2, dim=1).view_as(attn)


class VanillaKDLoss(nn.Module):
    """L = alpha * CE(student, y) + beta * T^2 * KL(softmax(s/T) || softmax(t/T))."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, temperature: float = 4.0):
        super().__init__()
        self.alpha, self.beta, self.T = alpha, beta, temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, labels):
        ce = self.ce(s_logits, labels)
        kl = F.kl_div(F.log_softmax(s_logits / self.T, 1),
                      F.softmax(t_logits / self.T, 1),
                      reduction="batchmean")
        return self.alpha * ce + self.beta * (self.T ** 2) * kl


class ATKDLoss(nn.Module):
    """L = alpha * CE + beta * T^2 * KL + gamma * MSE(norm(A_s), norm(A_t)).

    A_* are channel-wise sum-of-squared attention maps from a matched feature stage of
    student and teacher; if spatial sizes differ, teacher map is bilinearly resized to
    the student's spatial dims before the MSE.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 temperature: float = 4.0):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.T = alpha, beta, gamma, temperature
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s_logits, t_logits, s_feat, t_feat, labels):
        ce = self.ce(s_logits, labels)
        kl = F.kl_div(F.log_softmax(s_logits / self.T, 1),
                      F.softmax(t_logits / self.T, 1),
                      reduction="batchmean")
        a_s, a_t = _attention(s_feat), _attention(t_feat)
        if a_t.shape[-2:] != a_s.shape[-2:]:
            a_t = F.interpolate(a_t, size=a_s.shape[-2:], mode="bilinear", align_corners=False)
        at = F.mse_loss(_normalize(a_s), _normalize(a_t))
        return self.alpha * ce + self.beta * (self.T ** 2) * kl + self.gamma * at


def _selfcheck():
    torch.manual_seed(0)
    sl, tl = torch.randn(4, 3, requires_grad=True), torch.randn(4, 3)
    y = torch.randint(0, 3, (4,))
    sf = torch.randn(4, 64, 12, 12, requires_grad=True); tf = torch.randn(4, 96, 6, 6)
    assert torch.isfinite(VanillaKDLoss()(sl, tl, y))
    assert torch.isfinite(ATKDLoss()(sl, tl, sf, tf, y))


if __name__ == "__main__":
    _selfcheck(); print("loss.py self-check OK")
