"""Diagnostics raised by the compiler. All of them carry a message meant for the end user."""


class CompileError(Exception):
    """The model is valid ONNX but cannot be turned into correct C."""


class UnsupportedError(CompileError):
    """The model uses an operator, attribute, or type outside the supported subset."""
