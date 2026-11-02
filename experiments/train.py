"""Train the two reporting models on CPU and export them to ONNX.

    uv run --group train python -m experiments.train fashion_cnn
    uv run --group train python -m experiments.train cifar_mobilenet

Uses at most four threads, checkpoints every epoch, and resumes if interrupted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from . import data as datasets

REPO = Path(__file__).resolve().parent.parent
THREADS = 4


class FashionCNN(nn.Module):
    """Plain VGG-like stack. Small enough for a Cortex-M4 with a few hundred kB of RAM."""

    def __init__(self) -> None:
        super().__init__()

        def block(channels_in: int, channels_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 48))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(48 * 3 * 3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TrainableDepthwise(nn.Conv2d):
    """3x3 depthwise convolution with a faster training path.

    On CPU builds without oneDNN, autograd through a grouped convolution ran about 20 times slower
    than this equivalent unfold form (measured on torch 2.15, Apple arm64). Inference and export use
    the real operator, so the ONNX graph holds an ordinary Conv with group=channels.
    """

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__(channels, channels, 3, stride, 1, groups=channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super().forward(x)
        step = self.stride[0]
        patches = nn.functional.pad(x, (1, 1, 1, 1)).unfold(2, 3, step).unfold(3, 3, step)
        return (patches * self.weight[:, 0][None, :, None, None]).sum((-1, -2))


class InvertedResidual(nn.Module):
    """MobileNetV2 block: expand, depthwise, project, with a residual when shapes allow."""

    def __init__(self, channels_in: int, channels_out: int, stride: int, expansion: int) -> None:
        super().__init__()
        hidden = channels_in * expansion
        self.residual = stride == 1 and channels_in == channels_out
        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            TrainableDepthwise(hidden, stride),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layers(x)
        return x + output if self.residual else output


class CifarMobileNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        stages = [
            (16, 16, 1, 1),
            (16, 24, 2, 3),
            (24, 24, 1, 3),
            (24, 40, 2, 3),
            (40, 40, 1, 3),
            (40, 80, 2, 3),
            (80, 80, 1, 3),
        ]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(*[InvertedResidual(*stage) for stage in stages])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(80, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.blocks(self.stem(x))))


def augment(
    batch: torch.Tensor, padding: int, flip: bool, generator: torch.Generator
) -> torch.Tensor:
    """Random translation by reflection-free zero padding, and horizontal flips."""
    n, _, height, width = batch.shape
    padded = nn.functional.pad(batch, [padding] * 4)
    tops = torch.randint(0, 2 * padding + 1, (n,), generator=generator)
    lefts = torch.randint(0, 2 * padding + 1, (n,), generator=generator)
    rows = tops[:, None] + torch.arange(height)[None]
    columns = lefts[:, None] + torch.arange(width)[None]
    index = torch.arange(n)[:, None, None, None]
    channels = torch.arange(batch.shape[1])[None, :, None, None]
    crops = padded[index, channels, rows[:, None, :, None], columns[:, None, None, :]]
    if flip:
        mask = torch.rand(n, generator=generator) < 0.5
        crops[mask] = crops[mask].flip(-1)
    return crops


@torch.no_grad()
def accuracy(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    correct = 0
    for start in range(0, len(images), 512):
        prediction = model(images[start : start + 512]).argmax(1)
        correct += int((prediction == labels[start : start + 512]).sum())
    return correct / len(images)


SPECS = {
    "fashion_cnn": dict(
        make=FashionCNN,
        load=datasets.fashion_mnist,
        normalize=datasets.normalize_fashion,
        shape=(1, 1, 28, 28),
        epochs=20,
        lr=0.003,
        padding=2,
        flip=True,
    ),
    "cifar_mobilenet": dict(
        make=CifarMobileNet,
        load=datasets.cifar10,
        normalize=datasets.normalize_cifar,
        shape=(1, 3, 32, 32),
        epochs=16,
        lr=0.004,
        padding=4,
        flip=True,
    ),
}


def train(name: str, seed: int) -> None:
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    spec = SPECS[name]
    train_x, train_y, test_x, test_y = spec["load"]()
    images = torch.from_numpy(spec["normalize"](train_x))
    labels = torch.from_numpy(train_y)
    test_images = torch.from_numpy(spec["normalize"](test_x))
    test_labels = torch.from_numpy(test_y)

    model = spec["make"]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    batch_size = 128
    steps_per_epoch = len(images) // batch_size
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, spec["lr"], total_steps=spec["epochs"] * steps_per_epoch
    )
    generator = torch.Generator().manual_seed(seed)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    checkpoint = REPO / "data" / "checkpoints" / f"{name}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    first_epoch = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        schedule.load_state_dict(state["schedule"])
        generator.set_state(state["generator"])
        history = state["history"]
        first_epoch = len(history)

    for epoch in range(first_epoch, spec["epochs"]):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(images), generator=generator)
        total = 0.0
        for step in range(steps_per_epoch):
            index = order[step * batch_size : (step + 1) * batch_size]
            batch = augment(images[index], spec["padding"], spec["flip"], generator)
            loss = criterion(model(batch), labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
            total += loss.item()
        record = {
            "epoch": epoch,
            "loss": total / steps_per_epoch,
            "test_accuracy": accuracy(model, test_images, test_labels),
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedule": schedule.state_dict(),
                "generator": generator.get_state(),
                "history": history,
            },
            checkpoint,
        )

    model.eval()
    target = REPO / "models" / f"{name}.onnx"
    dummy = torch.zeros(spec["shape"])
    torch.onnx.export(
        model, (dummy,), str(target), input_names=["image"], output_names=["logits"],
        opset_version=18, external_data=False,
    )  # fmt: skip
    summary = {
        "model": name,
        "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()),
        "threads": THREADS,
        "epochs": spec["epochs"],
        "final_test_accuracy": history[-1]["test_accuracy"],
        "train_seconds": sum(item["seconds"] for item in history),
        "torch": torch.__version__,
        "history": history,
    }
    (REPO / "results" / f"train_{name}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{name}: test accuracy {summary['final_test_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(SPECS))
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    train(arguments.model, arguments.seed)
