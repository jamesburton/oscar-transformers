"""Per-token asymmetric INT2 quantization with group-128 codebooks.

OSCAR (after the rotation) quantizes each token's K (and V) head_dim vector in
groups of ``group_size`` (default 128) consecutive channels. Each group gets a
shared ``(scale, zero_point)`` derived from the group's clipped min/max:

    min'  = clip_lo * min(group)        # clip_lo = -clip_ratio expanded to a
    max'  = clip_hi * max(group)        # symmetric inward shrink; see below
    scale = (max' - min') / (2^bits - 1)
    zero  = round(-min' / scale)        # uint8 zero point in [0, 2^bits - 1]
    code  = clamp(round(x/scale) + zero, 0, 2^bits - 1)

For the reference implementation we store codes as ``uint8`` (one INT2 value
per byte) plus FP16 scales and zero_points per group. A subsequent kernel pass
can pack 4 INT2s per byte to get the full 8x memory reduction. The reference
path saves ~4x vs storing dequantized FP16 (which would save 0x).

"Clip ratio" usage in OSCAR
---------------------------

Upstream's RotationZoo metadata lists ``K_CLIP=0.96`` and ``V_CLIP=0.92`` —
both <1, meaning the quantization grid is intentionally shrunk inward so a few
extreme outliers in the post-rotation distribution land outside the grid and
get clamped. This trades a small bit of distortion on outliers for finer
resolution on the bulk of the distribution, where attention scores spend most
of their probability mass. We apply the ratio symmetrically by scaling the
``(max - min)`` interval by ``clip_ratio`` inward of its midpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class QuantizedBlock:
    """A single (batch, heads, tokens) slab of quantized K or V.

    Tensor shapes (with ``T = num_tokens``, ``H = num_heads``, ``D = head_dim``,
    ``G = D // group_size``):

    - ``codes``: ``(B, H, T, D)`` uint8 in ``[0, 2**bits - 1]``
    - ``scale``: ``(B, H, T, G)`` fp16
    - ``zero``:  ``(B, H, T, G)`` fp16
    """

    codes: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    bits: int
    group_size: int


def quantize_per_token(
    x: torch.Tensor,
    *,
    bits: int = 2,
    group_size: int = 128,
    clip_ratio: float = 1.0,
) -> QuantizedBlock:
    """Quantize ``x`` of shape ``(B, H, T, D)`` per-token, per-group.

    Returns a :class:`QuantizedBlock` whose dequantization is a lossy
    approximation of ``x``. Group size must divide head_dim ``D``.
    """
    if x.dim() != 4:
        raise ValueError(f"quantize_per_token expects (B, H, T, D); got {tuple(x.shape)}")
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8]; got {bits}")
    if not 0.0 < clip_ratio <= 1.0:
        raise ValueError(f"clip_ratio must be in (0, 1]; got {clip_ratio}")

    b, h, t, d = x.shape
    if d % group_size != 0:
        raise ValueError(f"head_dim={d} not divisible by group_size={group_size}")
    g = d // group_size

    grouped = x.reshape(b, h, t, g, group_size)
    mins = grouped.amin(dim=-1)
    maxs = grouped.amax(dim=-1)

    if clip_ratio < 1.0:
        mid = 0.5 * (mins + maxs)
        half = 0.5 * (maxs - mins) * clip_ratio
        mins = mid - half
        maxs = mid + half

    q_max = (1 << bits) - 1
    denom = (maxs - mins).clamp(min=1e-8)
    scale = denom / q_max
    zero = (-mins / scale).round().clamp(0, q_max)

    x_q = (grouped / scale.unsqueeze(-1) + zero.unsqueeze(-1)).round().clamp(0, q_max)
    codes = x_q.to(torch.uint8).reshape(b, h, t, d)

    return QuantizedBlock(
        codes=codes,
        scale=scale.to(torch.float16),
        zero=zero.to(torch.float16),
        bits=bits,
        group_size=group_size,
    )


def dequantize(block: QuantizedBlock, *, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Reverse of :func:`quantize_per_token`. Returns ``(B, H, T, D)`` in
    the requested floating-point dtype.
    """
    b, h, t, d = block.codes.shape
    g = d // block.group_size
    codes = block.codes.to(dtype).reshape(b, h, t, g, block.group_size)
    scale = block.scale.to(dtype).unsqueeze(-1)
    zero = block.zero.to(dtype).unsqueeze(-1)
    x = (codes - zero) * scale
    return x.reshape(b, h, t, d)


def concat_blocks(blocks: Tuple[QuantizedBlock, ...]) -> QuantizedBlock:
    """Concatenate quantized blocks along the token dim. All blocks must share
    bits and group_size.
    """
    if not blocks:
        raise ValueError("concat_blocks needs at least one block")
    bits = blocks[0].bits
    group_size = blocks[0].group_size
    for b in blocks[1:]:
        if b.bits != bits or b.group_size != group_size:
            raise ValueError("blocks must share bits and group_size to concat")
    return QuantizedBlock(
        codes=torch.cat([b.codes for b in blocks], dim=2),
        scale=torch.cat([b.scale for b in blocks], dim=2),
        zero=torch.cat([b.zero for b in blocks], dim=2),
        bits=bits,
        group_size=group_size,
    )
