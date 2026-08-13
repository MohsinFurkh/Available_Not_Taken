"""Torch datasets over a packaged variant store.

One dataset object serves any variant, so the train x test matrix is produced by
swapping a string rather than by rebuilding a pipeline -- which is what keeps the
off-diagonal cells honest: nothing else about the data path changes between
(train A-full, test A-full) and (train A-full, test A-only).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import ExperimentConfig
from .shortcut import inject_synthetic_shortcut

#: Suffix requesting a variant WITHOUT the synthetic marker. The positive
#: control needs to measure what it costs to *remove* a shortcut the model is
#: known to use, and that requires a test condition where the marker is absent
#: but everything else is identical. Handled here rather than as a fifth packed
#: array, because the marker is injected at load time and never stored.
NOMARKER_SUFFIX = "__nomarker"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VariantDataset(Dataset):
    """Serves one variant of a packaged dataset.

    Augmentation is applied strictly after splitting and identically across all
    regimes (proposal Section 9); the only thing that differs between regimes is
    which variant array is read.
    """

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: Sequence[int],
        labels: np.ndarray,
        variant: str,
        train: bool,
        image_size: int = 224,
        aux_targets: dict[str, np.ndarray] | None = None,
        annotation_masks: np.ndarray | None = None,
        augment: bool = True,
        annotation_randomising_aug: bool = False,
        annotation_aug_prob: float = 0.5,
        synthetic=None,
        num_classes: int = 0,
        seed: int = 0,
    ) -> None:
        base_variant = variant[: -len(NOMARKER_SUFFIX)] if variant.endswith(NOMARKER_SUFFIX) else variant
        if base_variant not in arrays:
            raise KeyError(f"variant {variant!r} not in package; have {sorted(arrays)}")
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = labels
        self.variant = variant
        self.train = train
        self.image_size = image_size
        self.aux_targets = aux_targets or {}
        self.annotation_masks = annotation_masks
        self.augment = augment and train
        self.annotation_randomising_aug = annotation_randomising_aug and train
        self.annotation_aug_prob = annotation_aug_prob
        self.synthetic = synthetic if (synthetic is not None and synthetic.enabled) else None
        self.num_classes = num_classes
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

    def _augment(self, image: np.ndarray) -> np.ndarray:
        import cv2

        if self._rng.random() < 0.5:
            image = image[:, ::-1]
        angle = float(self._rng.uniform(-15, 15))
        if abs(angle) > 0.5:
            h, w = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        gain = float(self._rng.uniform(0.85, 1.15))
        bias = float(self._rng.uniform(-15, 15))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return image

    def __getitem__(self, position: int) -> dict[str, Any]:
        row = int(self.indices[position])
        variant = self.variant

        if self.annotation_randomising_aug and self._rng.random() < self.annotation_aug_prob:
            # Section C.8: at training time, randomly show the model the
            # annotation-free version of an otherwise annotated example. The
            # cheapest plausible mitigation, and one the proposal insists must
            # be tested before anything more elaborate is proposed.
            variant = "a_removed" if variant == "a_full" else variant

        inject = self.synthetic is not None
        if variant.endswith(NOMARKER_SUFFIX):
            variant = variant[: -len(NOMARKER_SUFFIX)]
            inject = False

        image = np.asarray(self.arrays[variant][row])

        # Injected before augmentation so the marker is subject to the same
        # flips and rotations as everything else -- a marker exempt from
        # augmentation would be easier to learn than any real annotation.
        if inject:
            image = inject_synthetic_shortcut(
                image, int(self.labels[position]), self.num_classes, self.synthetic, row
            )
        if self.augment:
            image = self._augment(image)

        sample: dict[str, Any] = {
            "image": self._to_tensor(image),
            "row": row,
        }
        label = self.labels[position]
        sample["target"] = (
            torch.tensor(float(label), dtype=torch.float32)
            if np.issubdtype(self.labels.dtype, np.floating)
            else torch.tensor(int(label), dtype=torch.long)
        )
        for name, values in self.aux_targets.items():
            value = values[position]
            sample[f"aux_{name}"] = (
                torch.tensor(int(value), dtype=torch.long)
                if np.issubdtype(values.dtype, np.integer)
                else torch.tensor(float(value), dtype=torch.float32)
            )
        if self.annotation_masks is not None:
            sample["annotation_mask"] = torch.from_numpy(
                (np.asarray(self.annotation_masks[row]) > 127).astype(np.float32)
            )
        return sample


def make_loader(dataset: VariantDataset, cfg: ExperimentConfig, shuffle: bool, seed: int = 0) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=cfg.train.num_workers > 0,
    )
