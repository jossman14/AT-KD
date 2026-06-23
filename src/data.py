"""ImageFolder-style dataset and train/eval transforms.

Expected layout:
    <data_dir>/
        class_a/  *.jpg
        class_b/  *.jpg
        ...
Dotfiles (e.g. macOS .DS_*.jpg) are filtered out — PIL cannot open them.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _make_transforms(img_size: int):
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class ImageFolderDataset(Dataset):
    def __init__(self, root: str, transform=None):
        root = Path(root)
        self.classes = sorted(
            d.name for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for c in self.classes:
            for p in sorted((root / c).iterdir()):
                if p.suffix.lower() in EXTS and not p.name.startswith("."):
                    self.samples.append((p, self.class_to_idx[c]))
        self.transform = transform
        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def build_loaders(data_dir: str, train_idx, val_idx, test_idx, img_size: int,
                  batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = _make_transforms(img_size)
    train_set = ImageFolderDataset(data_dir, transform=train_tf)
    eval_set = ImageFolderDataset(data_dir, transform=eval_tf)
    mk = lambda ds, idx, sh: DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=sh,
                                        num_workers=num_workers, pin_memory=True)
    return mk(train_set, train_idx, True), mk(eval_set, val_idx, False), mk(eval_set, test_idx, False)


def labels_of(data_dir: str) -> np.ndarray:
    ds = ImageFolderDataset(data_dir, transform=None)
    return np.array([y for _, y in ds.samples]), ds.classes
