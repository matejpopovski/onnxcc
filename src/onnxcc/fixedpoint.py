"""Translate real-valued scale ratios into the integer parameters the int8 kernels expect."""

from __future__ import annotations

import math

from .errors import CompileError


def multiplier(real: float) -> tuple[int, int]:
    """Return (mult, shift) with real ~= mult * 2^(-31 - shift) and mult in [2^30, 2^31)."""
    if real < 0.0 or not math.isfinite(real):
        raise CompileError(f"invalid requantization factor {real}")
    if real == 0.0:
        return 0, 0
    fraction, exponent = math.frexp(real)
    mult = round(fraction * (1 << 31))
    if mult == 1 << 31:
        mult >>= 1
        exponent += 1
    shift = -exponent
    if shift > 30:
        return 0, 0
    if shift < -24:
        raise CompileError(f"requantization factor {real} is too large")
    return mult, shift


def add_multipliers(ratio_a: float, ratio_b: float) -> tuple[int, int, int]:
    """Scale both operands of a quantized add to `bits` fractional bits, staying under 2^21.

    The sum of two (8-bit difference * multiplier) terms then fits in int32, so the addition needs
    no 64-bit arithmetic on a 32-bit target.
    """
    largest = max(ratio_a, ratio_b)
    bits = 20 - math.ceil(math.log2(largest)) if largest > 0.0 else 20
    bits = min(bits, 30)
    if bits < 1:
        raise CompileError(f"quantized add input/output scale ratio {largest} is too large")
    return round(ratio_a * (1 << bits)), round(ratio_b * (1 << bits)), bits
