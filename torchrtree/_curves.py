"""
Space-filling curves used to order leaves before packing.

Both curves quantise each axis independently to `bits` bits, so axes with very
different ranges (metres vs. seconds) are weighted equally along the curve.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Curve = Literal["morton", "hilbert"]
CURVES: tuple[str, ...] = ("morton", "hilbert")

# 2^52 - 1 is the largest odd value exactly representable in float64, which the
# quantiser multiplies by. Only the 1D case would otherwise exceed it.
_MAX_BITS = 52


def sfc_bits(ndim: int) -> int:
    """Bits per axis so the interleaved key fits a signed int64."""
    return min(63 // ndim, _MAX_BITS)


def quantize(centers: Tensor, g_min: Tensor, g_range: Tensor, bits: int) -> Tensor:
    """Per-axis normalise to [0, 2^bits - 1] and truncate to int64."""
    max_q = (1 << bits) - 1
    norm = ((centers - g_min) / g_range).to(torch.float64)
    norm.mul_(max_q)  # in place: avoids a second N x ndim float64 temporary
    return norm.to(torch.int64).clamp_(0, max_q)


def interleave_bits(q: Tensor, bits: int, axis0_msb: bool) -> Tensor:
    """
    Interleave the low `bits` bits of each column of an (N, ndim) int64 tensor.
    Bit b of axis d lands at b * ndim + slot(d), where slot(d) is d (axis 0
    least significant, the Morton convention) or ndim - 1 - d (axis 0 most
    significant, the Hilbert transpose convention).
    """
    ndim = q.shape[-1]
    code = torch.zeros(q.shape[0], dtype=torch.int64, device=q.device)
    for b in range(bits):
        for d in range(ndim):
            slot = (ndim - 1 - d) if axis0_msb else d
            code |= ((q[:, d] >> b) & 1) << (b * ndim + slot)
    return code


def morton_codes(q: Tensor, bits: int) -> Tensor:
    return interleave_bits(q, bits, axis0_msb=False)


def hilbert_transpose(q: Tensor, bits: int) -> Tensor:
    """
    Axes -> transposed Hilbert coordinates (Skilling, "Programming the Hilbert
    curve", 2004), vectorised over rows. Input/output are (N, ndim) int64.
    """
    n = q.shape[1]
    X = [q[:, i].clone() for i in range(n)]
    M = 1 << (bits - 1)
    Q = M
    while Q > 1:  # inverse undo
        P = Q - 1
        for i in range(n):
            flag = (X[i] & Q) != 0
            # flag: invert X[0] by P. else: exchange the P bits of X[0] and X[i].
            t = torch.where(flag, torch.full_like(X[0], P), (X[0] ^ X[i]) & P)
            X[0] = X[0] ^ t
            if i > 0:
                X[i] = torch.where(flag, X[i], X[i] ^ t)
        Q >>= 1
    for i in range(1, n):  # gray encode
        X[i] = X[i] ^ X[i - 1]
    t = torch.zeros_like(X[0])
    Q = M
    while Q > 1:
        t = torch.where((X[n - 1] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    return torch.stack([x ^ t for x in X], dim=1)


def hilbert_codes(q: Tensor, bits: int) -> Tensor:
    if q.shape[1] == 1:  # the 1D Hilbert curve is the identity
        return q[:, 0].clone()
    return interleave_bits(hilbert_transpose(q, bits), bits, axis0_msb=True)


def sfc_codes(centers: Tensor, g_min: Tensor, g_range: Tensor, curve: str) -> Tensor:
    """Map (N, ndim) centers to a 1-D sortable int64 key along the chosen curve."""
    if curve not in CURVES:
        raise ValueError(f"curve must be one of {CURVES}, got {curve!r}")
    bits = sfc_bits(centers.shape[-1])
    q = quantize(centers, g_min, g_range, bits)
    return morton_codes(q, bits) if curve == "morton" else hilbert_codes(q, bits)
