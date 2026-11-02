"""Golden cases for int8 operators, compared to onnxruntime within one quantization step.

onnxruntime requantizes through float32 while the kernels here use fixed point, so exact equality
is not the contract. Weights stay within +/-63 because onnxruntime's x86 kernels use a 16-bit
pairwise multiply that can saturate on full-range int8 weights, a behaviour of that kernel and not
of the ONNX definition.
"""

from __future__ import annotations

import numpy as np
from onnx import helper

from .cases_float import int64, noise, rng
from .support import Case

node = helper.make_node
I8 = np.int8
F = np.float32


def bytes_of(seed: str, *shape: int) -> np.ndarray:
    return rng(seed).integers(-128, 128, shape).astype(I8)


def weights(seed: str, *shape: int) -> np.ndarray:
    return rng(seed).integers(-63, 64, shape).astype(I8)


def scalar(value: float) -> np.ndarray:
    return np.array(value, F)


def zero(value: int) -> np.ndarray:
    return np.array(value, I8)


def qparams(prefix: str, scale: float, zero_point: int) -> dict[str, np.ndarray]:
    return {f"{prefix}_s": scalar(scale), f"{prefix}_z": zero(zero_point)}


QCASE = dict(atol=1, rtol=0.0)


def qlinear_conv(
    name: str,
    x_shape: tuple[int, ...],
    channels: int,
    group: int = 1,
    y_scale: float = 0.5,
    per_channel: bool = False,
    bias: bool = True,
    **attrs: object,
) -> Case:
    kernel = attrs["kernel_shape"]
    w = weights(f"qw{name}", channels, x_shape[1] // group, *kernel)  # type: ignore[misc]
    init = {
        **qparams("x", 0.05, 7),
        "w": w,
        "w_s": rng(f"qs{name}").uniform(0.005, 0.02, channels).astype(F)
        if per_channel
        else scalar(0.01),
        "w_z": np.zeros(channels, I8) if per_channel else zero(0),
        **qparams("y", y_scale, -5),
    }
    inputs = ["x", "x_s", "x_z", "w", "w_s", "w_z", "y_s", "y_z"]
    if bias:
        init["b"] = rng(f"qb{name}").integers(-2000, 2000, channels).astype(np.int32)
        inputs.append("b")
    return Case(
        f"QLinearConv_{name}",
        [node("QLinearConv", inputs, ["y"], group=group, **attrs)],
        {"x": bytes_of(f"qx{name}", *x_shape)},
        init,
        **QCASE,
    )


def conv_cases() -> list[Case]:
    pads = [1, 1, 1, 1]
    return [
        qlinear_conv("k3", (1, 2, 8, 8), 3, kernel_shape=[3, 3]),
        qlinear_conv("k3_nobias", (1, 2, 8, 8), 3, bias=False, kernel_shape=[3, 3]),
        qlinear_conv("k3_pad", (1, 2, 8, 8), 3, kernel_shape=[3, 3], pads=pads),
        qlinear_conv("k1", (1, 3, 5, 5), 2, kernel_shape=[1, 1]),
        qlinear_conv("k5x3", (1, 1, 9, 9), 2, kernel_shape=[5, 3], pads=[2, 1, 2, 1]),
        qlinear_conv("stride2", (1, 2, 9, 9), 3, kernel_shape=[3, 3], strides=[2, 2], pads=pads),
        qlinear_conv(
            "dilated", (1, 2, 9, 9), 3, kernel_shape=[3, 3], dilations=[2, 2], pads=[2, 2, 2, 2]
        ),
        qlinear_conv(
            "per_channel", (1, 2, 8, 8), 4, per_channel=True, kernel_shape=[3, 3], pads=pads
        ),
        qlinear_conv("depthwise", (1, 4, 7, 7), 4, group=4, kernel_shape=[3, 3], pads=pads),
        qlinear_conv(
            "depthwise_per_channel",
            (1, 4, 7, 7),
            4,
            group=4,
            per_channel=True,
            kernel_shape=[3, 3],
            pads=pads,
            strides=[2, 2],
        ),
        qlinear_conv("batch2", (2, 2, 6, 6), 3, kernel_shape=[3, 3]),
        qlinear_conv("pad_beyond_kernel", (1, 2, 5, 5), 3, kernel_shape=[1, 3], pads=[2, 4, 3, 5]),
        qlinear_conv("gain_above_one", (1, 1, 6, 6), 2, y_scale=0.0001, kernel_shape=[1, 1]),
        qlinear_conv("tiny_gain", (1, 2, 8, 8), 3, y_scale=40.0, kernel_shape=[3, 3]),
    ]


def dense_cases() -> list[Case]:
    a = bytes_of("qda", 3, 4)
    init = {
        **qparams("a", 0.05, 3),
        "b": weights("qdb", 4, 5),
        "b_s": scalar(0.02),
        "b_z": zero(0),
        **qparams("y", 0.3, -2),
    }
    per_column = dict(init)
    per_column["b_s"] = rng("qdc").uniform(0.01, 0.03, 5).astype(F)
    per_column["b_z"] = np.zeros(5, I8)
    bias = rng("qdbias").integers(-500, 500, 5).astype(np.int32)
    gemm_inputs = ["a", "a_s", "a_z", "b", "b_s", "b_z", "c", "y_s", "y_z"]
    no_bias = ["a", "a_s", "a_z", "b", "b_s", "b_z", "", "y_s", "y_z"]
    transposed = dict(init)
    transposed["b"] = weights("qdt", 5, 4)
    return [
        Case(
            "QLinearMatMul",
            [node("QLinearMatMul", ["a", "a_s", "a_z", "b", "b_s", "b_z", "y_s", "y_z"], ["y"])],
            {"a": a},
            init,
            **QCASE,
        ),
        Case(
            "QGemm_bias",
            [node("QGemm", gemm_inputs, ["y"], domain="com.microsoft")],
            {"a": a},
            {**init, "c": bias},
            microsoft=True,
            **QCASE,
        ),
        Case(
            "QGemm_nobias",
            [node("QGemm", no_bias, ["y"], domain="com.microsoft")],
            {"a": a},
            init,
            microsoft=True,
            **QCASE,
        ),
        Case(
            "QGemm_per_column",
            [node("QGemm", gemm_inputs, ["y"], domain="com.microsoft")],
            {"a": a},
            {**per_column, "c": bias},
            microsoft=True,
            **QCASE,
        ),
        Case(
            "QGemm_transB",
            [node("QGemm", gemm_inputs, ["y"], domain="com.microsoft", transB=1)],
            {"a": a},
            {**transposed, "c": bias},
            microsoft=True,
            **QCASE,
        ),
    ]


def binary_cases() -> list[Case]:
    cases = []
    grid = [
        ("same", (2, 3), (2, 3)),
        ("scalar", (2, 3), ()),
        ("outer", (3, 1), (1, 4)),
    ]
    for op in ("QLinearAdd", "QLinearMul"):
        for name, left, right in grid:
            init = {
                **qparams("a", 0.1, 5),
                **qparams("b", 0.07, -11),
                **qparams("y", 0.2 if op == "QLinearAdd" else 1.0, 3),
            }
            inputs = ["a", "a_s", "a_z", "b", "b_s", "b_z", "y_s", "y_z"]
            cases.append(
                Case(
                    f"{op}_{name}",
                    [node(op, inputs, ["y"], domain="com.microsoft")],
                    {"a": bytes_of(f"{op}a{name}", *left), "b": bytes_of(f"{op}b{name}", *right)},
                    init,
                    microsoft=True,
                    **QCASE,
                )
            )
    return cases


def pool_cases() -> list[Case]:
    average = ["x", "x_s", "x_z", "y_s", "y_z"]
    init = {**qparams("x", 0.1, 4), **qparams("y", 0.08, -3)}
    same = {**qparams("x", 0.1, 4), **qparams("y", 0.1, 4)}
    cases = []
    grid = [
        ("k2s2", dict(kernel_shape=[2, 2], strides=[2, 2])),
        ("pads", dict(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])),
        ("count_pad", dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1], count_include_pad=1)),
    ]
    for name, attrs in grid:
        for rescale, params in (("rescale", init), ("same", same)):
            cases.append(
                Case(
                    f"QLinearAveragePool_{name}_{rescale}",
                    [node("QLinearAveragePool", average, ["y"], domain="com.microsoft", **attrs)],
                    {"x": bytes_of(f"qap{name}", 1, 2, 8, 8)},
                    params,
                    microsoft=True,
                    **QCASE,
                )
            )
    cases.append(
        Case(
            "QLinearGlobalAveragePool",
            [node("QLinearGlobalAveragePool", average, ["y"], domain="com.microsoft")],
            {"x": bytes_of("qgap", 1, 3, 5, 6)},
            init,
            microsoft=True,
            **QCASE,
        )
    )
    return cases


def concat_cases() -> list[Case]:
    inputs = ["y_s", "y_z", "a", "a_s", "a_z", "b", "b_s", "b_z"]
    return [
        Case(
            "QLinearConcat_rescale",
            [node("QLinearConcat", inputs, ["y"], axis=1, domain="com.microsoft")],
            {"a": bytes_of("qc1", 2, 3), "b": bytes_of("qc2", 2, 2)},
            {**qparams("a", 0.1, 5), **qparams("b", 0.05, -7), **qparams("y", 0.2, 1)},
            microsoft=True,
            **QCASE,
        ),
        Case(
            "QLinearConcat_shared_params",
            [node("QLinearConcat", inputs, ["y"], axis=0, domain="com.microsoft")],
            {"a": bytes_of("qc3", 2, 3), "b": bytes_of("qc4", 1, 3)},
            {**qparams("a", 0.1, 5), **qparams("b", 0.1, 5), **qparams("y", 0.1, 5)},
            microsoft=True,
            **QCASE,
        ),
    ]


def conversion_cases() -> list[Case]:
    wide = rng("wideq").uniform(-20.0, 20.0, (5, 6)).astype(F)
    ties = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 1e9, -1e9], F)
    return [
        Case(
            "QuantizeLinear",
            [node("QuantizeLinear", ["x", "s", "z"], ["y"])],
            {"x": wide},
            {"s": scalar(0.25), "z": zero(9)},
            atol=0,
        ),
        Case(
            "QuantizeLinear_ties",
            [node("QuantizeLinear", ["x", "s", "z"], ["y"])],
            {"x": ties},
            {"s": scalar(1.0), "z": zero(0)},
            atol=0,
        ),
        Case(
            "DequantizeLinear",
            [node("DequantizeLinear", ["x", "s", "z"], ["y"])],
            {"x": bytes_of("deq", 3, 4)},
            {"s": scalar(0.125), "z": zero(-3)},
        ),
        Case(
            "DequantizeLinear_folded_per_axis",
            [
                node("DequantizeLinear", ["w", "s", "z"], ["c"], axis=0),
                node("Add", ["x", "c"], ["y"]),
            ],
            {"x": noise("fold", 2, 3)},
            {"w": weights("foldw", 2, 3), "s": np.array([0.1, 0.2], F), "z": np.zeros(2, I8)},
            opset=17,
        ),
        Case(
            "QuantizeLinear_folded_per_axis",
            [
                node("QuantizeLinear", ["c", "s", "z"], ["q"], axis=0),
                node("DequantizeLinear", ["q", "t", "u"], ["d"]),
                node("Add", ["x", "d"], ["y"]),
            ],
            {"x": noise("fold2", 2, 3)},
            {
                "c": noise("foldc", 2, 3),
                "s": np.array([0.01, 0.02], F),
                "z": np.zeros(2, I8),
                "t": scalar(0.015),
                "u": zero(0),
            },
        ),
    ]


def passthrough_cases() -> list[Case]:
    x = bytes_of("pt", 1, 2, 6, 6)
    return [
        Case(
            "MaxPool_int8",
            [node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], strides=[2, 2])],
            {"x": x},
            atol=0,
        ),
        Case(
            "MaxPool_int8_pads",
            [node("MaxPool", ["x"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
            {"x": x},
            atol=0,
        ),
        Case(
            "Reshape_int8",
            [node("Reshape", ["x", "s"], ["y"])],
            {"x": x},
            {"s": int64([1, -1])},
            atol=0,
        ),
        Case(
            "Flatten_int8_inner",
            [node("Flatten", ["x"], ["f"]), node("Identity", ["f"], ["y"])],
            {"x": x},
            atol=0,
        ),
        Case(
            "Transpose_int8", [node("Transpose", ["x"], ["y"], perm=[0, 2, 3, 1])], {"x": x}, atol=0
        ),
        Case(
            "Slice_int8",
            [node("Slice", ["x", "s", "e", "a"], ["y"])],
            {"x": x},
            {"s": int64([1]), "e": int64([5]), "a": int64([2])},
            atol=0,
        ),
        Case(
            "Pad_int8",
            [node("Pad", ["x", "p", "v"], ["y"])],
            {"x": bytes_of("ptp", 2, 3)},
            {"p": int64([1, 0, 0, 2]), "v": zero(-7)},
            atol=0,
        ),
        Case(
            "Clip_int8",
            [node("Clip", ["x", "lo", "hi"], ["y"])],
            {"x": x},
            {"lo": zero(-20), "hi": zero(50)},
            atol=0,
        ),
    ]


def quant_cases() -> list[Case]:
    return [
        *conv_cases(),
        *dense_cases(),
        *binary_cases(),
        *pool_cases(),
        *concat_cases(),
        *conversion_cases(),
        *passthrough_cases(),
    ]
