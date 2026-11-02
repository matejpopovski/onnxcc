"""Registry shared by the reporting scripts: where each model lives and what data it sees."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import data as datasets

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CALIBRATION_SAMPLES = 1000


@dataclass(frozen=True)
class Reported:
    name: str
    load: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    normalize: Callable[[np.ndarray], np.ndarray]

    @property
    def path(self) -> Path:
        return REPO / "models" / f"{self.name}.onnx"

    def splits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calibration inputs from the training set, then test inputs and labels."""
        train_x, _, test_x, test_y = self.load()
        chosen = np.random.default_rng(0).permutation(len(train_x))[:CALIBRATION_SAMPLES]
        return (
            self.normalize(train_x[chosen])[:, None],
            self.normalize(test_x)[:, None],
            test_y,
        )


FASHION = Reported("fashion_cnn", datasets.fashion_mnist, datasets.normalize_fashion)
CIFAR = Reported("cifar_mobilenet", datasets.cifar10, datasets.normalize_cifar)
ALL = (FASHION, CIFAR)


def available() -> list[Reported]:
    return [item for item in ALL if item.path.exists()]
