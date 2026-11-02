"""Random Conv and pool geometry against onnxruntime: edges are where window kernels go wrong."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from onnx import helper

from onnxcc.errors import UnsupportedError

from .support import Case, check_case, fuzz_budget

node = helper.make_node
F = np.float32


@dataclass(frozen=True)
class Geometry:
    height: int
    width: int
    kernel: tuple[int, int]
    strides: tuple[int, int]
    dilations: tuple[int, int]
    pads: tuple[int, int, int, int]

    @property
    def attrs(self) -> dict[str, object]:
        return {
            "kernel_shape": list(self.kernel),
            "strides": list(self.strides),
            "pads": list(self.pads),
        }

    def output_exists(self) -> bool:
        vertical = self.height + self.pads[0] + self.pads[2]
        horizontal = self.width + self.pads[1] + self.pads[3]
        return (
            vertical >= (self.kernel[0] - 1) * self.dilations[0] + 1
            and horizontal >= (self.kernel[1] - 1) * self.dilations[1] + 1
        )


small = st.integers(1, 4)
geometries = st.builds(
    Geometry,
    height=st.integers(1, 9),
    width=st.integers(1, 9),
    kernel=st.tuples(small, small),
    strides=st.tuples(st.integers(1, 3), st.integers(1, 3)),
    dilations=st.tuples(st.integers(1, 3), st.integers(1, 3)),
    pads=st.tuples(*[st.integers(0, 3)] * 4),
)


def check(case: Case) -> None:
    with tempfile.TemporaryDirectory() as directory:
        check_case(case, Path(directory))


@settings(max_examples=fuzz_budget(30))
@given(
    geometry=geometries,
    channels=st.integers(1, 4),
    multiplier=st.integers(1, 3),
    group=st.sampled_from([1, 2]),
    seed=st.integers(0, 2**31),
)
def test_conv_geometry(
    geometry: Geometry, channels: int, multiplier: int, group: int, seed: int
) -> None:
    assume(geometry.output_exists())
    rng = np.random.default_rng(seed)
    c_in = channels * group
    c_out = group * multiplier
    x = rng.uniform(-1, 1, (1, c_in, geometry.height, geometry.width)).astype(F)
    w = rng.uniform(-1, 1, (c_out, channels, *geometry.kernel)).astype(F)
    conv = node(
        "Conv",
        ["x", "w"],
        ["y"],
        group=group,
        dilations=list(geometry.dilations),
        **geometry.attrs,
    )
    check(Case("conv", [conv], {"x": x}, {"w": w}))


def smaller_pads(kernel: tuple[int, int]) -> st.SearchStrategy[tuple[int, int, int, int]]:
    """ONNX requires pooling pads smaller than the kernel."""
    high, wide = st.integers(0, kernel[0] - 1), st.integers(0, kernel[1] - 1)
    return st.tuples(high, wide, high, wide)


@st.composite
def pool_geometries(draw: st.DrawFn) -> Geometry:
    kernel = draw(st.tuples(small, small))
    return Geometry(
        height=draw(st.integers(kernel[0], 9)),
        width=draw(st.integers(kernel[1], 9)),
        kernel=kernel,
        strides=draw(st.tuples(st.integers(1, 3), st.integers(1, 3))),
        dilations=draw(st.tuples(st.integers(1, 2), st.integers(1, 2))),
        pads=draw(smaller_pads(kernel)),
    )


@settings(max_examples=fuzz_budget(30))
@given(
    geometry=pool_geometries(),
    op=st.sampled_from(["MaxPool", "AveragePool", "AveragePoolPad"]),
    ceil=st.booleans(),
    seed=st.integers(0, 2**31),
)
def test_pool_geometry(geometry: Geometry, op: str, ceil: bool, seed: int) -> None:
    assume(geometry.output_exists())
    x = np.random.default_rng(seed).uniform(-1, 1, (1, 2, geometry.height, geometry.width))
    attrs = dict(geometry.attrs)
    if op == "MaxPool":
        # Only MaxPool: runtimes disagree on the divisor of partial ceil-mode average windows.
        attrs["dilations"] = list(geometry.dilations)
        attrs["ceil_mode"] = int(ceil)
    if op == "AveragePoolPad":
        op = "AveragePool"
        attrs["count_include_pad"] = 1
    try:
        check(Case("pool", [node(op, ["x"], ["y"], **attrs)], {"x": x.astype(F)}))
    except UnsupportedError:
        assume(False)


@settings(max_examples=fuzz_budget(30))
@given(
    geometry=geometries,
    depthwise=st.booleans(),
    per_channel=st.booleans(),
    seed=st.integers(0, 2**31),
)
def test_qlinear_conv_geometry(
    geometry: Geometry, depthwise: bool, per_channel: bool, seed: int
) -> None:
    assume(geometry.output_exists())
    rng = np.random.default_rng(seed)
    channels = 3
    group = channels if depthwise else 1
    x = rng.integers(-128, 128, (1, channels, geometry.height, geometry.width)).astype(np.int8)
    w = rng.integers(-63, 64, (channels, channels // group, *geometry.kernel)).astype(np.int8)
    w_scale = rng.uniform(0.005, 0.02, channels) if per_channel else np.array(0.01)
    initializers = {
        "w": w,
        "w_s": np.asarray(w_scale, F),
        "w_z": np.zeros(channels if per_channel else (), np.int8),
        "x_s": np.array(0.05, F),
        "x_z": np.array(int(rng.integers(-20, 20)), np.int8),
        "y_s": np.array(float(rng.uniform(0.2, 2.0)), F),
        "y_z": np.array(int(rng.integers(-20, 20)), np.int8),
    }
    conv = node(
        "QLinearConv",
        ["x", "x_s", "x_z", "w", "w_s", "w_z", "y_s", "y_z"],
        ["y"],
        group=group,
        dilations=list(geometry.dilations),
        **geometry.attrs,
    )
    check(Case("qconv", [conv], {"x": x}, initializers, atol=1, rtol=0.0))
