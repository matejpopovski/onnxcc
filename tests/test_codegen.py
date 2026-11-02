"""Properties of the emitted C that a downstream firmware build depends on."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from onnx import helper

from onnxcc import codegen
from onnxcc.codegen import KERNELS, c_identifier, comment_text, float_literal
from onnxcc.compiler import compile_model
from onnxcc.errors import CompileError
from onnxcc.memcheck import verify
from onnxcc.toolchain import STRICT_FLAGS, compile_object, freestanding_flags, host_cc

from .cases_float import float_cases, noise
from .cases_quant import quant_cases
from .support import build, make_model, ort_run, sanitizers_work

node = helper.make_node
F = np.float32
FORBIDDEN = re.compile(
    r"\b(malloc|calloc|realloc|free|printf|fprintf|memcpy|memset|sqrtf?|exp|expf|tanhf?)\s*\("
)


def sample_model():
    x = noise("cg", 1, 2, 6, 6)
    w = noise("cgw", 3, 2, 3, 3)
    nodes = [
        node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], name="conv"),
        node("Relu", ["c"], ["r"], name="relu"),
        node("GlobalAveragePool", ["r"], ["g"]),
        node("Flatten", ["g"], ["f"]),
        node("Softmax", ["f"], ["y"], axis=1),
    ]
    return make_model(nodes, {"x": x}, {"w": w}, ["y"]), {"x": x}


def test_output_is_byte_for_byte_deterministic() -> None:
    model, _ = sample_model()
    assert compile_model(model).source == compile_model(model).source


@pytest.mark.parametrize("case", [*float_cases(), *quant_cases()], ids=lambda case: case.id)
def test_generated_code_includes_only_stdint_and_calls_no_libc(case) -> None:
    source = compile_model(case.model(), name="m").source
    assert re.findall(r"#include\s+\S+", source) == ["#include <stdint.h>"]
    assert not FORBIDDEN.search(source)
    assert "double" not in source


def test_object_has_no_undefined_symbols(tmp_path: Path) -> None:
    """With loop idiom recognition off the object file links against nothing at all."""
    model, _ = sample_model()
    source = tmp_path / "m.c"
    source.write_text(compile_model(model, name="m").source)
    obj = tmp_path / "m.o"
    flags = [*STRICT_FLAGS, *freestanding_flags(), "-O2"]
    compile_object(source, obj, flags)
    listing = subprocess.run(["nm", "-u", str(obj)], capture_output=True, text=True, check=True)
    symbols = set(listing.stdout.split()) - {"U", "_GLOBAL_OFFSET_TABLE_"}
    assert symbols == set()


def test_float_literals_round_trip_exactly() -> None:
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2**32, 20000, dtype=np.uint32)
    values = bits.view(np.float32)
    values = values[np.isfinite(values)]
    for value in values:
        text = float_literal(float(value))
        assert text.endswith("f")
        assert np.float32(float.fromhex(text[:-1])) == value
    assert float_literal(-0.0) == "-0x0p+0f"
    assert float_literal(1.0) == "0x1p+0f"


def test_non_finite_weights_are_a_diagnostic() -> None:
    w = np.array([[[[np.inf]]]], F)
    model = make_model(
        [node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 1])],
        {"x": np.zeros((1, 1, 2, 2), F)},
        {"w": w},
        ["y"],
    )
    with pytest.raises(CompileError, match="non-finite"):
        compile_model(model)


def test_subnormal_and_negative_zero_weights_compile_strictly(tmp_path: Path) -> None:
    smallest = np.nextafter(np.float32(0), np.float32(1))
    w = np.array([[[[smallest, -0.0, 1e-40, 1.0]]]], F)
    x = noise("sub", 1, 1, 1, 4)
    model = make_model(
        [node("Conv", ["x", "w"], ["y"], kernel_shape=[1, 4])], {"x": x}, {"w": w}, ["y"]
    )
    output = build(compile_model(model, name="m"), tmp_path, [*STRICT_FLAGS, "-O1"]).run([x[None]])
    np.testing.assert_allclose(output[0][0], ort_run(model, {"x": x})[0], atol=1e-6)


@pytest.mark.parametrize(
    "name",
    ["a*/ int x; /*", "??/", "back\\", "new\nline", 'quo"te', "/* nested", "çelik", ""],
)
def test_hostile_names_stay_inside_comments(name: str, tmp_path: Path) -> None:
    model, feed = sample_model()
    model.graph.node[0].name = name
    model.graph.node[1].name = name + "2"
    model.graph.name = name or "x"
    result = compile_model(model, name="hostile")
    binary = build(result, tmp_path, [*STRICT_FLAGS, "-O1"])
    output = binary.run([feed["x"][None]])[0][0]
    np.testing.assert_allclose(output, ort_run(model, feed)[0], rtol=1e-4)


def test_identifier_and_comment_sanitizers() -> None:
    assert c_identifier("my-model.v2") == "my_model_v2"
    assert c_identifier("9lives") == "m_9lives"
    assert "*/" not in comment_text("x*/y")


def test_unused_kernels_are_not_emitted() -> None:
    model, _ = sample_model()
    source = compile_model(model).source
    assert "ox_conv2d_f32" in source
    for absent in ("ox_dense_f32", "ox_gather", "ox_qconv2d_s8", "ox_tanh", "OX_BINARY"):
        assert absent not in source


def test_kernel_library_compiles_as_one_strict_translation_unit(tmp_path: Path) -> None:
    text = (Path(codegen.__file__).parent / "kernels.c").read_text()
    source = tmp_path / "kernels.c"
    source.write_text(text)
    compile_object(source, tmp_path / "kernels.o", [*STRICT_FLAGS, "-Wno-unused-function"])


def test_every_kernel_section_parses() -> None:
    assert {"prelude", "conv2d_f32", "qconv2d_s8", "quantize_s8"} <= set(KERNELS)
    for kernel in KERNELS.values():
        for item in kernel.requires:
            assert item in KERNELS


def test_entry_point_signature_and_header(tmp_path: Path) -> None:
    model, _ = sample_model()
    result = compile_model(model, name="my-model")
    assert result.prototype == (
        "void my_model_run(const float *restrict in0, float *restrict out0)"
    )
    header = tmp_path / "m.h"
    header.write_text(result.header)
    use = tmp_path / "use.c"
    use.write_text('#include "m.h"\nint main(void) { return MY_MODEL_STATIC_RAM_BYTES < 0; }\n')
    subprocess.run(
        [host_cc(), *STRICT_FLAGS, str(use), "-c", "-o", str(tmp_path / "u.o")], check=True
    )


def test_two_models_link_into_one_program(tmp_path: Path) -> None:
    model, _ = sample_model()
    for stem in ("alpha", "beta"):
        (tmp_path / f"{stem}.c").write_text(compile_model(model, name=stem).source)
    main = tmp_path / "main.c"
    main.write_text("int main(void) { return 0; }\n")
    subprocess.run(
        [
            host_cc(),
            str(tmp_path / "alpha.c"),
            str(tmp_path / "beta.c"),
            str(main),
            "-o",
            str(tmp_path / "both"),
        ],
        check=True,
    )


def test_input_that_no_step_reads_still_compiles(tmp_path: Path) -> None:
    x = noise("u", 2)
    model = make_model([node("Abs", ["x"], ["y"])], {"x": x, "unused": x}, {}, ["y"])
    build(compile_model(model, name="m"), tmp_path, [*STRICT_FLAGS, "-O1"])


def test_in_place_shrinks_the_arena_and_reshape_costs_nothing() -> None:
    x = noise("ip", 1, 4)
    nodes = [
        node("Abs", ["x"], ["a"]),
        node("Sigmoid", ["a"], ["b"]),
        node("Reshape", ["b", "s"], ["c"]),
        node("Tanh", ["c"], ["d"]),
        node("Neg", ["d"], ["y"]),
    ]
    model = make_model(nodes, {"x": x}, {"s": np.array([2, 2], np.int64)}, ["y"])
    assert compile_model(model).memory.arena_bytes == 16
    assert compile_model(model, inplace=False).memory.arena_bytes == 32
    verify(compile_model(model).source)


def test_scalar_outputs_and_rank_zero_shapes(tmp_path: Path) -> None:
    x = noise("sc", 4)
    b = noise("sc2", 4)
    model = make_model([node("MatMul", ["x", "b"], ["y"])], {"x": x}, {"b": b}, ["y"])
    feed = {"x": x}
    output = build(compile_model(model, name="m"), tmp_path, ["-O1"]).run([x[None]])[0][0]
    np.testing.assert_allclose(output, ort_run(model, feed)[0], rtol=1e-5)
    assert output.shape == ()


@pytest.mark.skipif(not sanitizers_work(), reason="no sanitizer runtime for this compiler")
def test_sanitizers_see_an_overrun_of_one_tensor(tmp_path: Path) -> None:
    """Proof the isolated build gives ASan per-tensor resolution, using a planted kernel bug."""
    from onnxcc.errors import CompileError as BuildError
    from onnxcc.toolchain import SANITIZE_FLAGS

    x = noise("bug", 1, 4)
    model = make_model(
        [node("Abs", ["x"], ["a"]), node("Neg", ["a"], ["b"]), node("Abs", ["b"], ["y"])],
        {"x": x},
        {},
        ["y"],
    )
    result = compile_model(model, name="buggy", isolated=True)
    planted = result.source.replace(
        "for (int32_t i = 0; i < n; ++i) {", "for (int32_t i = 0; i <= n; ++i) {"
    )
    assert planted != result.source
    result.source = planted
    binary = build(result, tmp_path, [*SANITIZE_FLAGS, "-O1"])
    with pytest.raises(BuildError):
        binary.run([x[None]])
