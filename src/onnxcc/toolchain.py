"""Host C compiler driver: strict builds, and a stdin/stdout harness around a generated model."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .codegen import C_TYPES, Signature
from .errors import CompileError
from .ir import F32, I8

STRICT_FLAGS = (
    "-std=c99",
    "-Wall",
    "-Wextra",
    "-Wpedantic",
    "-Werror",
    "-Wshadow",
    "-Wconversion",
    "-Wdouble-promotion",
    "-Wstrict-prototypes",
    "-Wvla",
)
SANITIZE_FLAGS = ("-fsanitize=address,undefined", "-fno-sanitize-recover=all", "-g")
NUMPY_TYPES: dict[str, np.dtype[Any]] = {C_TYPES[F32]: F32, C_TYPES[I8]: I8}


def host_cc() -> str:
    return os.environ.get("ONNXCC_CC") or os.environ.get("CC") or "cc"


def freestanding_flags(compiler: str | None = None) -> list[str]:
    """Flags under which an object needs no libc at all, not even memset or memcpy.

    Without them gcc and clang are entitled to turn a fill loop into a call to memset.
    """
    flags = ["-ffreestanding", "-fno-builtin", "-fno-stack-protector"]
    banner = subprocess.run(
        [compiler or host_cc(), "--version"], capture_output=True, text=True, check=True
    ).stdout
    if "Free Software Foundation" in banner:
        flags.append("-fno-tree-loop-distribute-patterns")
    return flags


def run_cc(arguments: Sequence[str]) -> None:
    done = subprocess.run([host_cc(), *arguments], capture_output=True, text=True)
    if done.returncode != 0:
        raise CompileError(f"{host_cc()} failed:\n{done.stdout}{done.stderr}")


def compile_object(source: Path, output: Path, flags: Sequence[str]) -> None:
    run_cc([*flags, "-c", str(source), "-o", str(output)])


def harness_source(
    name: str, prototype: str, inputs: list[Signature], outputs: list[Signature], observe: bool
) -> str:
    lines = ["#include <stdint.h>", "#include <stdio.h>", f"{prototype};"]
    for item in [*inputs, *outputs]:
        lines.append(f"static {item.c_type} {item.name}[{item.elements}];")
    if observe:
        lines += [
            "void onnxcc_observe(int32_t id, const float *data, int32_t n) {",
            "    fwrite(&id, sizeof id, 1, stdout);",
            "    fwrite(&n, sizeof n, 1, stdout);",
            "    fwrite(data, sizeof data[0], (size_t)n, stdout);",
            "}",
        ]
    lines.append("int main(void) {")
    lines.append("    for (;;) {")
    for item in inputs:
        count = f"{item.elements}u"
        lines.append(
            f"        if (fread({item.name}, sizeof {item.name}[0], {count}, stdin) != {count})"
            " return 0;"
        )
    arguments = ", ".join(item.name for item in [*inputs, *outputs])
    lines.append(f"        {name}_run({arguments});")
    if not observe:
        for item in outputs:
            lines.append(
                f"        fwrite({item.name}, sizeof {item.name}[0], {item.elements}u, stdout);"
            )
    if not inputs:
        lines.append("        return 0;")
    lines += ["    }", "}"]
    return "\n".join(lines) + "\n"


@dataclass
class Binary:
    path: Path
    inputs: list[Signature]
    outputs: list[Signature]

    def _write_inputs(self, samples: Sequence[np.ndarray[Any, Any]], handle: Any) -> None:
        """`samples[i]` holds input i as a batch with a leading sample axis."""
        count = len(samples[0]) if samples else 0
        for index in range(count):
            for array, spec in zip(samples, self.inputs, strict=True):
                item = np.asarray(array[index], NUMPY_TYPES[spec.c_type])
                handle.write(item.reshape(spec.shape).tobytes())
        handle.flush()

    def run(self, samples: Sequence[np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
        with tempfile.TemporaryFile() as stream:
            self._write_inputs(samples, stream)
            stream.seek(0)
            done = subprocess.run([self.path], stdin=stream, capture_output=True, timeout=600)
        if done.returncode != 0:
            raise CompileError(
                f"{self.path.name} exited {done.returncode}:\n{done.stderr.decode()[:2000]}"
            )
        return self._split_outputs(done.stdout, len(samples[0]) if samples else 1)

    def _split_outputs(self, raw: bytes, count: int) -> list[np.ndarray[Any, Any]]:
        """The stream holds each sample's outputs in order. Regroup them per output."""
        columns: list[list[np.ndarray[Any, Any]]] = [[] for _ in self.outputs]
        cursor = 0
        for _ in range(count):
            for position, spec in enumerate(self.outputs):
                dtype = NUMPY_TYPES[spec.c_type]
                size = spec.elements * dtype.itemsize
                part = np.frombuffer(raw[cursor : cursor + size], dtype).reshape(spec.shape)
                columns[position].append(part)
                cursor += size
        return [np.stack(column) for column in columns]

    def stream_observations(
        self, samples: Sequence[np.ndarray[Any, Any]]
    ) -> Iterator[tuple[int, np.ndarray[Any, Any]]]:
        with tempfile.TemporaryFile() as stream:
            self._write_inputs(samples, stream)
            stream.seek(0)
            process = subprocess.Popen([self.path], stdin=stream, stdout=subprocess.PIPE)
            assert process.stdout is not None
            with process.stdout as pipe:
                while header := pipe.read(8):
                    identifier, count = np.frombuffer(header, np.int32)
                    payload = pipe.read(int(count) * 4)
                    yield int(identifier), np.frombuffer(payload, np.float32)
            if process.wait() != 0:
                raise CompileError(f"calibration binary exited {process.returncode}")


def link_harness(
    source: str,
    harness: str,
    directory: Path,
    stem: str,
    flags: Sequence[str],
) -> Path:
    model_c = directory / f"{stem}.c"
    harness_c = directory / f"{stem}_main.c"
    model_c.write_text(source)
    harness_c.write_text(harness)
    model_o = directory / f"{stem}.o"
    compile_object(model_c, model_o, flags)
    executable = directory / f"{stem}.bin"
    extra = [flag for flag in flags if flag.startswith("-fsanitize") or flag == "-g"]
    run_cc([*extra, "-O1", str(harness_c), str(model_o), "-o", str(executable)])
    return executable
