"""Golden cases for float operators. Each is compared against onnxruntime in test_golden.py."""

from __future__ import annotations

import zlib

import numpy as np
from onnx import TensorProto, helper

from .support import Case

node = helper.make_node
F = np.float32


def rng(seed: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(seed.encode()))


def noise(seed: str, *shape: int) -> np.ndarray:
    return rng(seed).uniform(-1.0, 1.0, shape).astype(F)


def conv_cases() -> list[Case]:
    grid = [
        ("k1", dict(kernel_shape=[1, 1]), (1, 3, 6, 6), 4),
        ("k3", dict(kernel_shape=[3, 3]), (1, 3, 8, 8), 4),
        ("k3_pad", dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1]), (1, 2, 7, 9), 5),
        ("k3_asym_pad", dict(kernel_shape=[3, 3], pads=[0, 2, 1, 0]), (1, 2, 7, 9), 3),
        ("k5x3", dict(kernel_shape=[5, 3], pads=[2, 1, 2, 1]), (1, 1, 9, 9), 2),
        ("stride2", dict(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]), (1, 2, 9, 9), 3),
        ("stride_mixed", dict(kernel_shape=[3, 3], strides=[3, 1]), (1, 2, 10, 8), 3),
        (
            "dilated",
            dict(kernel_shape=[3, 3], dilations=[2, 2], pads=[2, 2, 2, 2]),
            (1, 2, 9, 9),
            3,
        ),
        ("dilated_mixed", dict(kernel_shape=[2, 3], dilations=[3, 2]), (1, 2, 9, 9), 3),
        (
            "same_upper",
            dict(kernel_shape=[4, 4], strides=[2, 2], auto_pad="SAME_UPPER"),
            (1, 2, 9, 7),
            3,
        ),
        (
            "same_lower",
            dict(kernel_shape=[4, 4], strides=[2, 2], auto_pad="SAME_LOWER"),
            (1, 2, 9, 7),
            3,
        ),
        ("valid", dict(kernel_shape=[3, 3], auto_pad="VALID"), (1, 2, 9, 7), 3),
        ("batch2", dict(kernel_shape=[3, 3], pads=[1, 1, 1, 1]), (2, 2, 6, 6), 3),
        ("pad_beyond_kernel", dict(kernel_shape=[1, 3], pads=[2, 4, 3, 5]), (1, 2, 5, 5), 3),
    ]
    cases = []
    for name, attrs, x_shape, out_channels in grid:
        k = attrs["kernel_shape"]
        w = noise(f"w{name}", out_channels, x_shape[1], *k)
        b = noise(f"b{name}", out_channels)
        for bias in (True, False):
            inputs = ["x", "w", "b"] if bias else ["x", "w"]
            init = {"w": w, "b": b} if bias else {"w": w}
            cases.append(
                Case(
                    f"Conv_{name}_{'bias' if bias else 'nobias'}",
                    [node("Conv", inputs, ["y"], **attrs)],
                    {"x": noise(f"x{name}", *x_shape)},
                    init,
                )
            )
    cases.append(
        Case(
            "Conv_depthwise",
            [node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], group=4, pads=[1, 1, 1, 1])],
            {"x": noise("dx", 1, 4, 7, 7)},
            {"w": noise("dw", 4, 1, 3, 3)},
        )
    )
    cases.append(
        Case(
            "Conv_depthwise_multiplier2",
            [node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], group=2, strides=[2, 2])],
            {"x": noise("dmx", 1, 2, 9, 9)},
            {"w": noise("dmw", 4, 1, 3, 3), "b": noise("dmb", 4)},
        )
    )
    cases.append(
        Case(
            "Conv_grouped",
            [node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], group=2)],
            {"x": noise("gx", 1, 4, 7, 7)},
            {"w": noise("gw", 6, 2, 3, 3)},
        )
    )
    cases.append(
        Case(
            "Conv_1d",
            [node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[5], strides=[2], pads=[2, 2])],
            {"x": noise("c1x", 1, 3, 20)},
            {"w": noise("c1w", 4, 3, 5), "b": noise("c1b", 4)},
        )
    )
    return cases


def gemm_cases() -> list[Case]:
    a = noise("ga", 3, 4)
    wide = noise("gb", 4, 5)
    c1 = noise("gc", 5)
    return [
        Case("Gemm_plain", [node("Gemm", ["a", "b"], ["y"])], {"a": a}, {"b": wide}),
        Case("Gemm_bias", [node("Gemm", ["a", "b", "c"], ["y"])], {"a": a}, {"b": wide, "c": c1}),
        Case(
            "Gemm_alpha_beta",
            [node("Gemm", ["a", "b", "c"], ["y"], alpha=0.5, beta=2.0)],
            {"a": a},
            {"b": wide, "c": c1},
        ),
        Case(
            "Gemm_transB",
            [node("Gemm", ["a", "b", "c"], ["y"], transB=1)],
            {"a": a},
            {"b": wide.T.copy().T[:, :].T.reshape(5, 4), "c": noise("gc2", 5)},
        ),
        Case(
            "Gemm_transA",
            [node("Gemm", ["a", "b"], ["y"], transA=1)],
            {"a": noise("gta", 4, 3)},
            {"b": noise("gtb", 4, 2)},
        ),
        Case(
            "Gemm_scalar_c",
            [node("Gemm", ["a", "b", "c"], ["y"])],
            {"a": a},
            {"b": wide, "c": np.array(0.25, F)},
        ),
        Case(
            "Gemm_matrix_c",
            [node("Gemm", ["a", "b", "c"], ["y"])],
            {"a": a},
            {"b": wide, "c": noise("gmc", 3, 5)},
        ),
        Case(
            "Gemm_runtime_operands",
            [node("Gemm", ["a", "b", "c"], ["y"], alpha=0.5, beta=0.5, transB=1)],
            {"a": a, "b": noise("grb", 5, 4), "c": noise("grc", 3, 1)},
        ),
    ]


def matmul_cases() -> list[Case]:
    return [
        Case(
            "MatMul_2x2",
            [node("MatMul", ["a", "b"], ["y"])],
            {"a": noise("m1", 3, 4)},
            {"b": noise("m2", 4, 2)},
        ),
        Case(
            "MatMul_runtime",
            [node("MatMul", ["a", "b"], ["y"])],
            {"a": noise("m3", 3, 4), "b": noise("m4", 4, 2)},
        ),
        Case(
            "MatMul_3x2",
            [node("MatMul", ["a", "b"], ["y"])],
            {"a": noise("m5", 2, 3, 4)},
            {"b": noise("m6", 4, 2)},
        ),
        Case(
            "MatMul_1x2",
            [node("MatMul", ["a", "b"], ["y"])],
            {"a": noise("m7", 4)},
            {"b": noise("m8", 4, 3)},
        ),
        Case(
            "MatMul_2x1",
            [node("MatMul", ["a", "b"], ["y"])],
            {"a": noise("m9", 3, 4)},
            {"b": noise("m10", 4)},
        ),
        Case(
            "MatMul_1x1",
            [node("MatMul", ["a", "b"], ["y"])],
            {"a": noise("m11", 4)},
            {"b": noise("m12", 4)},
        ),
    ]


def binary_cases() -> list[Case]:
    shapes = [
        ("same", (2, 3), (2, 3)),
        ("scalar", (2, 3), ()),
        ("scalar_left", (), (2, 3)),
        ("outer", (3, 1), (1, 4)),
        ("rank_gap", (2, 3, 4), (4,)),
        ("mixed", (2, 1, 4), (3, 1)),
        ("deep", (2, 1, 3, 1, 2), (1, 2, 1, 2, 1)),
    ]
    cases = []
    for op in ("Add", "Sub", "Mul", "Div", "Min", "Max"):
        for name, left, right in shapes:
            a = noise(f"{op}{name}a", *left)
            b = noise(f"{op}{name}b", *right)
            if op == "Div":
                b = np.where(np.abs(b) < 0.1, 0.5, b).astype(F)
            cases.append(Case(f"{op}_{name}", [node(op, ["a", "b"], ["y"])], {"a": a, "b": b}))
    cases.append(
        Case(
            "Add_const",
            [node("Add", ["a", "b"], ["y"])],
            {"a": noise("ac", 2, 3)},
            {"b": noise("bc", 3)},
        )
    )
    cases.append(Case("Add_same_input", [node("Add", ["a", "a"], ["y"])], {"a": noise("as", 2, 3)}))
    cases.append(
        Case(
            "Min_variadic",
            [node("Min", ["a", "b", "c"], ["y"])],
            {"a": noise("mv1", 4), "b": noise("mv2", 4), "c": noise("mv3", 4)},
        )
    )
    cases.append(
        Case(
            "Max_variadic",
            [node("Max", ["a", "b", "c"], ["y"])],
            {"a": noise("xv1", 4), "b": noise("xv2", 4), "c": noise("xv3", 1)},
        )
    )
    cases.append(Case("Max_single", [node("Max", ["a"], ["y"])], {"a": noise("xs", 4)}))
    cases.append(
        Case(
            "PRelu_scalar",
            [node("PRelu", ["a", "s"], ["y"])],
            {"a": noise("pr", 2, 5)},
            {"s": np.array(0.25, F)},
        )
    )
    cases.append(
        Case(
            "PRelu_channel",
            [node("PRelu", ["a", "s"], ["y"])],
            {"a": noise("pc", 2, 3, 5)},
            {"s": noise("ps", 3, 1)},
        )
    )
    return cases


def unary_cases() -> list[Case]:
    wide = rng("wide").uniform(-30.0, 30.0, (6, 7)).astype(F)
    x = noise("u", 3, 4)
    small = rng("small").uniform(-2.0, 2.0, (40,)).astype(F)
    cases = [
        Case("Abs", [node("Abs", ["x"], ["y"])], {"x": x}),
        Case("Neg", [node("Neg", ["x"], ["y"])], {"x": x}),
        Case("Relu", [node("Relu", ["x"], ["y"])], {"x": x}),
        Case("LeakyRelu_default", [node("LeakyRelu", ["x"], ["y"])], {"x": x}),
        Case("LeakyRelu_alpha", [node("LeakyRelu", ["x"], ["y"], alpha=0.3)], {"x": x}),
        Case("Sigmoid", [node("Sigmoid", ["x"], ["y"])], {"x": wide}),
        Case("Sigmoid_small", [node("Sigmoid", ["x"], ["y"])], {"x": small}),
        Case("Tanh", [node("Tanh", ["x"], ["y"])], {"x": wide}),
        Case("Tanh_small", [node("Tanh", ["x"], ["y"])], {"x": small}),
        Case("Exp", [node("Exp", ["x"], ["y"])], {"x": wide}),
        Case("Exp_small", [node("Exp", ["x"], ["y"])], {"x": small}),
        Case(
            "Exp_extremes",
            [node("Exp", ["x"], ["y"])],
            {"x": np.array([-200.0, -95.0, -87.0, 0.0, 87.0, 88.7, 200.0], F)},
            rtol=2e-3,
            atol=1e-37,
        ),
        Case("HardSigmoid", [node("HardSigmoid", ["x"], ["y"])], {"x": wide}),
        Case(
            "HardSigmoid_params",
            [node("HardSigmoid", ["x"], ["y"], alpha=0.1, beta=0.2)],
            {"x": wide},
        ),
        Case("HardSwish", [node("HardSwish", ["x"], ["y"])], {"x": wide}),
        Case(
            "Clip_both",
            [node("Clip", ["x", "lo", "hi"], ["y"])],
            {"x": x},
            {"lo": np.array(-0.25, F), "hi": np.array(0.5, F)},
        ),
        Case(
            "Clip_inverted_bounds",
            [node("Clip", ["x", "lo", "hi"], ["y"])],
            {"x": x},
            {"lo": np.array(0.5, F), "hi": np.array(-0.25, F)},
        ),
        Case(
            "Clip_inverted_bounds_fused",
            [
                node("Conv", ["x", "w"], ["c"], kernel_shape=[1, 1]),
                node("Clip", ["c", "lo", "hi"], ["y"]),
            ],
            {"x": noise("cif", 1, 1, 2, 2)},
            {"w": np.ones((1, 1, 1, 1), F), "lo": np.array(0.5, F), "hi": np.array(-0.25, F)},
        ),
        Case("Clip_min", [node("Clip", ["x", "lo"], ["y"])], {"x": x}, {"lo": np.array(-0.25, F)}),
        Case(
            "Clip_max", [node("Clip", ["x", "", "hi"], ["y"])], {"x": x}, {"hi": np.array(0.5, F)}
        ),
    ]
    for axis, shape in ((-1, (2, 3, 4)), (1, (2, 3, 4)), (0, (2, 3, 4)), (1, (2, 7))):
        cases.append(
            Case(
                f"Softmax_axis{axis}_{len(shape)}d",
                [node("Softmax", ["x"], ["y"], axis=axis)],
                {"x": rng(f"sm{axis}").uniform(-5, 5, shape).astype(F)},
            )
        )
    return cases


def pool_cases() -> list[Case]:
    cases = []
    grid = [
        ("k2s2", dict(kernel_shape=[2, 2], strides=[2, 2]), (1, 2, 8, 8)),
        ("k3s1", dict(kernel_shape=[3, 3]), (1, 2, 6, 7)),
        ("pads", dict(kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1]), (1, 2, 9, 9)),
        ("asym", dict(kernel_shape=[3, 2], strides=[2, 1], pads=[0, 1, 2, 0]), (1, 1, 8, 8)),
        ("ceil", dict(kernel_shape=[3, 3], strides=[2, 2], ceil_mode=1), (1, 2, 8, 8)),
        (
            "same_upper",
            dict(kernel_shape=[3, 3], strides=[2, 2], auto_pad="SAME_UPPER"),
            (1, 2, 8, 9),
        ),
        (
            "same_lower",
            dict(kernel_shape=[2, 2], strides=[2, 2], auto_pad="SAME_LOWER"),
            (1, 2, 7, 9),
        ),
        ("batch2", dict(kernel_shape=[2, 2], strides=[1, 2]), (2, 3, 5, 6)),
    ]
    for op in ("MaxPool", "AveragePool"):
        for name, attrs, shape in grid:
            cases.append(
                Case(
                    f"{op}_{name}",
                    [node(op, ["x"], ["y"], **attrs)],
                    {"x": noise(f"{op}{name}", *shape)},
                )
            )
        cases.append(
            Case(
                f"{op}_1d",
                [node(op, ["x"], ["y"], kernel_shape=[3], strides=[2], pads=[1, 1])],
                {"x": noise(f"{op}1d", 1, 2, 11)},
            )
        )
    cases.append(
        Case(
            "MaxPool_dilated",
            [node("MaxPool", ["x"], ["y"], kernel_shape=[2, 2], dilations=[2, 2], strides=[1, 1])],
            {"x": noise("mpd", 1, 2, 7, 7)},
        )
    )
    cases.append(
        Case(
            "AveragePool_count_pad",
            [
                node(
                    "AveragePool",
                    ["x"],
                    ["y"],
                    kernel_shape=[3, 3],
                    strides=[2, 2],
                    pads=[1, 1, 1, 1],
                    count_include_pad=1,
                )
            ],
            {"x": noise("apc", 1, 2, 9, 9)},
        )
    )
    cases.append(
        Case(
            "GlobalAveragePool_2d",
            [node("GlobalAveragePool", ["x"], ["y"])],
            {"x": noise("gap", 2, 3, 5, 6)},
        )
    )
    cases.append(
        Case(
            "GlobalAveragePool_1d",
            [node("GlobalAveragePool", ["x"], ["y"])],
            {"x": noise("gap1", 1, 3, 9)},
        )
    )
    return cases


def batch_norm_cases() -> list[Case]:
    def params(seed: str, channels: int) -> dict[str, np.ndarray]:
        r = rng(seed)
        return {
            "scale": r.uniform(0.5, 1.5, channels).astype(F),
            "bias": r.uniform(-0.5, 0.5, channels).astype(F),
            "mean": r.uniform(-0.5, 0.5, channels).astype(F),
            "var": r.uniform(0.5, 1.5, channels).astype(F),
        }

    stats = ["scale", "bias", "mean", "var"]
    return [
        Case(
            "BatchNormalization_standalone_4d",
            [node("BatchNormalization", ["x", *stats], ["y"], epsilon=1e-3)],
            {"x": noise("bn4", 1, 3, 4, 4)},
            params("bn4p", 3),
        ),
        Case(
            "BatchNormalization_standalone_2d",
            [node("BatchNormalization", ["x", *stats], ["y"])],
            {"x": noise("bn2", 2, 3)},
            params("bn2p", 3),
        ),
        Case(
            "BatchNormalization_fused_conv",
            [
                node("Conv", ["x", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
                node("BatchNormalization", ["c", *stats], ["y"]),
            ],
            {"x": noise("bnc", 1, 2, 6, 6)},
            {"w": noise("bnw", 3, 2, 3, 3), "b": noise("bnb", 3), **params("bncp", 3)},
        ),
        Case(
            "BatchNormalization_fused_conv_nobias",
            [
                node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3]),
                node("BatchNormalization", ["c", *stats], ["n"]),
                node("Relu", ["n"], ["y"]),
            ],
            {"x": noise("bnn", 1, 2, 6, 6)},
            {"w": noise("bnnw", 3, 2, 3, 3), **params("bnnp", 3)},
        ),
        Case(
            "BatchNormalization_fused_gemm",
            [
                node("Gemm", ["x", "w", "b"], ["c"], transB=1),
                node("BatchNormalization", ["c", *stats], ["y"]),
            ],
            {"x": noise("bng", 2, 4)},
            {"w": noise("bngw", 3, 4), "b": noise("bngb", 3), **params("bngp", 3)},
        ),
        Case(
            "BatchNormalization_shared_conv_output",
            [
                node("Conv", ["x", "w"], ["c"], kernel_shape=[1, 1]),
                node("BatchNormalization", ["c", *stats], ["n"]),
                node("Add", ["n", "c"], ["y"]),
            ],
            {"x": noise("bns", 1, 3, 4, 4)},
            {"w": noise("bnsw", 3, 3, 1, 1), **params("bnsp", 3)},
        ),
    ]


def int64(values: list[int]) -> np.ndarray:
    return np.array(values, np.int64)


def shape_cases() -> list[Case]:
    x = noise("shape", 2, 3, 4)
    cases = [
        Case(
            "Reshape_explicit", [node("Reshape", ["x", "s"], ["y"])], {"x": x}, {"s": int64([4, 6])}
        ),
        Case(
            "Reshape_minus_one",
            [node("Reshape", ["x", "s"], ["y"])],
            {"x": x},
            {"s": int64([2, -1])},
        ),
        Case(
            "Reshape_zero",
            [node("Reshape", ["x", "s"], ["y"])],
            {"x": x},
            {"s": int64([0, 0, -1, 1])},
        ),
        Case("Identity", [node("Identity", ["x"], ["y"])], {"x": x}),
        Case("Cast_noop", [node("Cast", ["x"], ["y"], to=TensorProto.FLOAT)], {"x": x}),
        Case("Transpose_scalar", [node("Transpose", ["x"], ["y"])], {"x": np.array(0.5, F)}),
        Case(
            "Identity_inner",
            [node("Abs", ["x"], ["a"]), node("Identity", ["a"], ["b"]), node("Neg", ["b"], ["y"])],
            {"x": x},
        ),
        Case("Dropout", [node("Dropout", ["x"], ["y"])], {"x": x}),
        Case(
            "Dropout_ratio",
            [
                node("Dropout", ["x", "r", "t"], ["y"]),
            ],
            {"x": x},
            {"r": np.array(0.5, F), "t": np.array(False)},
        ),
        Case("Squeeze_all", [node("Squeeze", ["x"], ["y"])], {"x": noise("sq", 1, 3, 1, 4)}),
        Case(
            "Squeeze_axes",
            [node("Squeeze", ["x", "a"], ["y"])],
            {"x": noise("sqa", 1, 3, 1)},
            {"a": int64([2])},
        ),
        Case("Unsqueeze", [node("Unsqueeze", ["x", "a"], ["y"])], {"x": x}, {"a": int64([0, -1])}),
        Case("Transpose_default", [node("Transpose", ["x"], ["y"])], {"x": x}),
        Case("Transpose_perm", [node("Transpose", ["x"], ["y"], perm=[1, 2, 0])], {"x": x}),
        Case(
            "Transpose_nhwc",
            [node("Transpose", ["x"], ["y"], perm=[0, 2, 3, 1])],
            {"x": noise("tn", 1, 2, 3, 4)},
        ),
        Case(
            "Transpose_partial",
            [node("Transpose", ["x"], ["y"], perm=[0, 1, 3, 2])],
            {"x": noise("tp", 2, 2, 3, 4)},
        ),
        Case(
            "Concat_axis0",
            [node("Concat", ["a", "b"], ["y"], axis=0)],
            {"a": noise("c1", 2, 3), "b": noise("c2", 1, 3)},
        ),
        Case(
            "Concat_axis1",
            [node("Concat", ["a", "b", "c"], ["y"], axis=1)],
            {"a": noise("c3", 2, 3), "b": noise("c4", 2, 1), "c": noise("c5", 2, 2)},
        ),
        Case(
            "Concat_negative_axis",
            [node("Concat", ["a", "b"], ["y"], axis=-2)],
            {"a": noise("c6", 2, 3, 4), "b": noise("c7", 2, 1, 4)},
        ),
        Case(
            "Concat_const",
            [node("Concat", ["a", "b"], ["y"], axis=1)],
            {"a": noise("c8", 2, 3)},
            {"b": noise("c9", 2, 2)},
        ),
        Case(
            "Slice_basic",
            [node("Slice", ["x", "s", "e"], ["y"])],
            {"x": x},
            {"s": int64([0, 1]), "e": int64([1, 3])},
        ),
        Case(
            "Slice_axes_steps",
            [node("Slice", ["x", "s", "e", "a", "t"], ["y"])],
            {"x": noise("sl", 4, 5, 6)},
            {"s": int64([1, 0]), "e": int64([6, 5]), "a": int64([2, 1]), "t": int64([2, 2])},
        ),
        Case(
            "Slice_negative_step",
            [node("Slice", ["x", "s", "e", "a", "t"], ["y"])],
            {"x": noise("sln", 4, 5)},
            {"s": int64([-1]), "e": int64([-1000]), "a": int64([1]), "t": int64([-1])},
        ),
        Case(
            "Slice_negative_index",
            [node("Slice", ["x", "s", "e"], ["y"])],
            {"x": noise("sli", 6)},
            {"s": int64([-4]), "e": int64([-1])},
        ),
        Case(
            "Split_equal",
            [node("Split", ["x"], ["a", "b"], axis=1, num_outputs=2)],
            {"x": noise("sp", 2, 6)},
            outputs=["a", "b"],
            opset=18,
        ),
        Case(
            "Split_sizes",
            [node("Split", ["x", "s"], ["a", "b", "c"], axis=-1)],
            {"x": noise("sp2", 2, 6)},
            {"s": int64([1, 3, 2])},
            outputs=["a", "b", "c"],
        ),
        Case(
            "Pad_constant",
            [node("Pad", ["x", "p"], ["y"])],
            {"x": noise("pd", 2, 3)},
            {"p": int64([0, 1, 2, 3])},
        ),
        Case(
            "Pad_value",
            [node("Pad", ["x", "p", "v"], ["y"])],
            {"x": noise("pv", 1, 2, 3, 3)},
            {"p": int64([0, 0, 1, 1, 0, 0, 1, 2]), "v": np.array(0.5, F)},
        ),
        Case(
            "Pad_axes",
            [node("Pad", ["x", "p", "", "a"], ["y"])],
            {"x": noise("pa", 2, 3)},
            {"p": int64([1, 2]), "a": int64([1])},
            opset=18,
        ),
    ]
    for axis, shape in ((0, (1, 3)), (1, (2, 3, 4)), (2, (2, 3, 4)), (-1, (2, 3, 4))):
        cases.append(
            Case(
                f"Flatten_axis{axis}_{len(shape)}d",
                [node("Flatten", ["x"], ["y"], axis=axis)],
                {"x": noise(f"fl{axis}", *shape)},
            )
        )
    return cases


def reduce_cases() -> list[Case]:
    x = noise("reduce", 2, 3, 4)
    four = noise("reduce4", 2, 3, 4, 5)
    return [
        Case("ReduceMean_axes", [node("ReduceMean", ["x"], ["y"], axes=[1])], {"x": x}),
        Case(
            "ReduceMean_keep0",
            [node("ReduceMean", ["x"], ["y"], axes=[0, 2], keepdims=0)],
            {"x": x},
        ),
        Case("ReduceMean_all", [node("ReduceMean", ["x"], ["y"], keepdims=0)], {"x": x}),
        Case("ReduceMean_pool_form", [node("ReduceMean", ["x"], ["y"], axes=[2, 3])], {"x": four}),
        Case(
            "ReduceMean_pool_form_keep0",
            [node("ReduceMean", ["x"], ["y"], axes=[2, 3], keepdims=0)],
            {"x": four},
        ),
        Case(
            "ReduceMean_opset18",
            [node("ReduceMean", ["x", "a"], ["y"], keepdims=1)],
            {"x": x},
            {"a": int64([-1])},
            opset=18,
        ),
        Case(
            "ReduceSum_axes", [node("ReduceSum", ["x", "a"], ["y"])], {"x": x}, {"a": int64([1, 2])}
        ),
        Case(
            "ReduceSum_keep0",
            [node("ReduceSum", ["x", "a"], ["y"], keepdims=0)],
            {"x": x},
            {"a": int64([0])},
        ),
    ]


def compile_time_cases() -> list[Case]:
    """Ops that must fold away. The first case is what PyTorch emits for x.view(x.size(0), -1)."""
    x = noise("ct", 1, 2, 3, 4)
    return [
        Case(
            "Shape_Gather_view",
            [
                node("Shape", ["x"], ["s"]),
                node("Gather", ["s", "i"], ["n"], axis=0),
                node("Unsqueeze", ["n", "axes"], ["n1"]),
                node("Concat", ["n1", "minus"], ["t"], axis=0),
                node("Reshape", ["x", "t"], ["y"]),
            ],
            {"x": x},
            {"i": np.array(0, np.int64), "axes": int64([0]), "minus": int64([-1])},
        ),
        Case(
            "Shape_slice_attrs",
            [
                node("Shape", ["x"], ["s"], start=1, end=3),
                node("Concat", ["s", "m"], ["t"], axis=0),
                node("Reshape", ["x", "t"], ["y"]),
            ],
            {"x": x},
            {"m": int64([-1])},
            opset=19,
        ),
        Case(
            "Size_Cast",
            [
                node("Size", ["x"], ["n"]),
                node("Cast", ["n"], ["f"], to=TensorProto.FLOAT),
                node("Div", ["x", "f"], ["y"]),
            ],
            {"x": x},
        ),
        Case(
            "Constant_float",
            [node("Constant", [], ["c"], value_float=2.5), node("Mul", ["x", "c"], ["y"])],
            {"x": x},
        ),
        Case(
            "Constant_tensor",
            [
                node(
                    "Constant",
                    [],
                    ["c"],
                    value=helper.make_tensor("v", TensorProto.FLOAT, [4], [1, 2, 3, 4]),
                ),
                node("Add", ["x", "c"], ["y"]),
            ],
            {"x": x},
        ),
        Case(
            "ConstantOfShape_add",
            [
                node(
                    "ConstantOfShape",
                    ["s"],
                    ["c"],
                    value=helper.make_tensor("v", TensorProto.FLOAT, [1], [0.5]),
                ),
                node("Add", ["x", "c"], ["y"]),
            ],
            {"x": x},
            {"s": int64([1, 2, 3, 4])},
        ),
        Case(
            "Range_reshape",
            [
                node("Range", ["a", "b", "c"], ["r"]),
                node("Mul", ["r", "two"], ["t"]),
                node("Reshape", ["x", "t"], ["y"]),
            ],
            {"x": noise("rg", 2, 4)},
            {
                "a": np.array(1, np.int64),
                "b": np.array(3, np.int64),
                "c": np.array(1, np.int64),
                "two": int64([2, 2]),
            },
        ),
        Case(
            "Float_chain_folds",
            [
                node("Add", ["p", "q"], ["c"]),
                node("Sub", ["c", "p"], ["d"]),
                node("Neg", ["d"], ["e"]),
                node("Mul", ["x", "e"], ["y"]),
            ],
            {"x": x},
            {"p": np.array(2.0, F), "q": np.array(3.0, F)},
        ),
        Case(
            "Output_is_input",
            [node("Abs", ["x"], ["y"])],
            {"x": x},
            outputs=["x", "y"],
        ),
        Case(
            "Output_is_constant",
            [node("Add", ["c", "c"], ["k"]), node("Abs", ["x"], ["y"])],
            {"x": x},
            {"c": np.array([1.5, 2.5], F)},
            outputs=["k", "y"],
        ),
    ]


def float_cases() -> list[Case]:
    return [
        *conv_cases(),
        *gemm_cases(),
        *matmul_cases(),
        *binary_cases(),
        *unary_cases(),
        *pool_cases(),
        *batch_norm_cases(),
        *shape_cases(),
        *reduce_cases(),
        *compile_time_cases(),
    ]
