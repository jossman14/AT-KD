"""Teacher and Student wrappers that expose (logits, deep feature map) per forward call.

Uses timm's `forward_features` / `forward_head` split: `forward_features(x)` returns the
deepest pre-pool feature tensor (NCHW or NHWC depending on backbone), `forward_head(feat)`
runs global pool + classifier. The wrapper auto-permutes channels-last NHWC features
(ConvNeXt, SwinV2) back to NCHW so the attention loss can treat them uniformly.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import timm


def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """ConvNeXt / SwinV2 stages emit NHWC; detect and permute back to NCHW."""
    if x.dim() == 4 and x.shape[1] == x.shape[2] and x.shape[3] > x.shape[1]:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


def _ensure_4d(feat: torch.Tensor) -> torch.Tensor:
    """Some backbones return a (B, N, C) token tensor; reshape to (B, C, H, W)."""
    if feat.dim() == 3:
        B, N, C = feat.shape
        H = W = int(N ** 0.5)
        if H * W == N:
            feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
    return feat


class TeacherModel(nn.Module):
    """Frozen-by-default teacher. Default: SwinV2-Tiny (matches AT-KD paper)."""

    def __init__(self, name: str = "swinv2_tiny_window8_256", num_classes: int = 3,
                 pretrained: bool = True, freeze: bool = True):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def forward(self, x):
        feat = self.model.forward_features(x)
        logits = self.model.forward_head(feat)
        feat = _ensure_4d(_to_nchw(feat))
        return logits, feat


class StudentModel(nn.Module):
    """Default: MobileNetV3-Large (matches AT-KD paper)."""

    def __init__(self, name: str = "mobilenetv3_large_100", num_classes: int = 3,
                 pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        feat = self.model.forward_features(x)
        logits = self.model.forward_head(feat)
        feat = _ensure_4d(_to_nchw(feat))
        return logits, feat


def _selfcheck():
    s = StudentModel(num_classes=3, pretrained=False)
    t = TeacherModel(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    sl, sf = s(x); tl, tf = t(x)
    assert sl.shape == (2, 3) and sf.dim() == 4, f"student bad: {sl.shape} feat={sf.shape}"
    assert tl.shape == (2, 3) and tf.dim() == 4, f"teacher bad: {tl.shape} feat={tf.shape}"


if __name__ == "__main__":
    _selfcheck(); print("models.py self-check OK")
