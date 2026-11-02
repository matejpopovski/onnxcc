"""Download and load the datasets. Nothing here is committed: files land in data/ (git-ignored).

Fashion-MNIST: Xiao, Rasul and Vollgraf, 2017, MIT licence.
CIFAR-10: Krizhevsky, 2009, distributed by the University of Toronto without an explicit licence.
"""

from __future__ import annotations

import gzip
import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FASHION_FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def _fetch(url: str, target: Path, md5: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, target)
    digest = hashlib.md5(target.read_bytes(), usedforsecurity=False).hexdigest()
    if digest != md5:
        target.unlink()
        raise RuntimeError(f"{target.name}: checksum mismatch")
    return target


def _idx(path: Path, offset: int) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        return np.frombuffer(handle.read(), np.uint8, offset=offset)


def fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """uint8 images (N, 1, 28, 28) and int64 labels, as (train_x, train_y, test_x, test_y)."""
    paths = {
        name: _fetch(FASHION_URL + name, DATA / "fashion" / name, md5)
        for name, md5 in FASHION_FILES.items()
    }
    train_x = _idx(paths["train-images-idx3-ubyte.gz"], 16).reshape(-1, 1, 28, 28)
    train_y = _idx(paths["train-labels-idx1-ubyte.gz"], 8).astype(np.int64)
    test_x = _idx(paths["t10k-images-idx3-ubyte.gz"], 16).reshape(-1, 1, 28, 28)
    test_y = _idx(paths["t10k-labels-idx1-ubyte.gz"], 8).astype(np.int64)
    return train_x, train_y, test_x, test_y


def cifar10() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """uint8 images (N, 3, 32, 32) and int64 labels. The archive holds trusted-format pickles."""
    archive = _fetch(CIFAR_URL, DATA / "cifar" / "cifar-10-python.tar.gz", CIFAR_MD5)
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            stem = Path(member.name).name
            if stem.startswith(("data_batch", "test_batch")):
                handle = tar.extractfile(member)
                assert handle is not None
                batch = pickle.load(handle, encoding="bytes")
                images = np.asarray(batch[b"data"], np.uint8).reshape(-1, 3, 32, 32)
                arrays[stem] = (images, np.asarray(batch[b"labels"], np.int64))
    names = sorted(key for key in arrays if key.startswith("data_batch"))
    train_x = np.concatenate([arrays[name][0] for name in names])
    train_y = np.concatenate([arrays[name][1] for name in names])
    return train_x, train_y, *arrays["test_batch"]


def normalize_fashion(images: np.ndarray) -> np.ndarray:
    return (images.astype(np.float32) / 255.0 - 0.2860) / 0.3530


def normalize_cifar(images: np.ndarray) -> np.ndarray:
    mean = np.array([0.4914, 0.4822, 0.4465], np.float32).reshape(3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], np.float32).reshape(3, 1, 1)
    return (images.astype(np.float32) / 255.0 - mean) / std
