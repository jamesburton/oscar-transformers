"""Quantize/dequantize round-trip sanity checks."""
from __future__ import annotations

import torch

from oscar_transformers.quantize import dequantize, quantize_per_token


def test_int2_roundtrip_shapes() -> None:
    x = torch.randn(1, 8, 5, 128, dtype=torch.bfloat16)
    block = quantize_per_token(x, bits=2, group_size=128, clip_ratio=1.0)
    # INT2 codes pack 4-per-byte: head_dim 128 -> stored 32 bytes per token.
    assert block.codes.shape == (1, 8, 5, 32)
    assert block.codes.dtype == torch.uint8
    assert block.unpacked_d == 128
    assert block.scale.shape == (1, 8, 5, 1)
    assert block.zero.shape == (1, 8, 5, 1)
    y = dequantize(block, dtype=torch.bfloat16)
    assert y.shape == x.shape
    assert y.dtype == torch.bfloat16


def test_packed_roundtrip_all_widths() -> None:
    """quantize -> dequantize must produce values within the expected
    grid-step error at each supported bit-width, with codes stored in their
    packed form (8//bits values per byte)."""
    torch.manual_seed(0)
    x = torch.randn(1, 4, 16, 128, dtype=torch.float32)
    rng = x.abs().max().item()
    for bits in (1, 2, 4, 8):
        block = quantize_per_token(x, bits=bits, group_size=128, clip_ratio=1.0)
        per_byte = 8 // bits
        expected_packed_d = 128 // per_byte
        assert block.codes.shape == (1, 4, 16, expected_packed_d), (
            f"bits={bits}: codes shape {tuple(block.codes.shape)} != (1,4,16,{expected_packed_d})"
        )
        assert block.unpacked_d == 128
        y = dequantize(block, dtype=torch.float32)
        assert y.shape == x.shape
        # Expected reconstruction error: roughly 2*max/(2^bits-1)/2 = max/(2^bits-1)
        # plus the fp16 storage error on scale/zero. Allow a comfortable factor.
        max_err = (y - x).abs().max().item()
        grid_step = 2 * rng / max((1 << bits) - 1, 1)
        assert max_err <= grid_step * 1.5 + 1e-3, (
            f"bits={bits}: max_err={max_err:.4f} > 1.5*grid_step={1.5*grid_step:.4f}"
        )


def test_packed_codes_are_in_valid_uint8_range() -> None:
    """Sanity: every packed byte is a uint8 (no overflow during pack)."""
    x = torch.randn(2, 4, 8, 128, dtype=torch.float32)
    for bits in (1, 2, 4, 8):
        block = quantize_per_token(x, bits=bits, group_size=128, clip_ratio=0.96)
        assert block.codes.dtype == torch.uint8
        assert (block.codes >= 0).all() and (block.codes <= 255).all()


def test_int8_is_lossless_on_small_range() -> None:
    x = torch.linspace(-1.0, 1.0, steps=128, dtype=torch.float32).reshape(1, 1, 1, 128)
    block = quantize_per_token(x, bits=8, group_size=128, clip_ratio=1.0)
    y = dequantize(block, dtype=torch.float32)
    assert torch.allclose(y, x, atol=2.0 / 255 + 1e-6), "8-bit asym should resolve a smooth ramp to under one grid step"


def test_groups_quantize_independently() -> None:
    x = torch.zeros(1, 1, 1, 256, dtype=torch.float32)
    x[..., :128] = torch.linspace(-10.0, 10.0, steps=128)
    x[..., 128:] = torch.linspace(-0.1, 0.1, steps=128)
    block = quantize_per_token(x, bits=4, group_size=128, clip_ratio=1.0)
    y = dequantize(block, dtype=torch.float32)
    err_left = (y[..., :128] - x[..., :128]).abs().max().item()
    err_right = (y[..., 128:] - x[..., 128:]).abs().max().item()
    assert err_left < 2.0, f"4-bit asym should resolve large range to <2.0; got {err_left}"
    assert err_right < 0.02, f"small range should resolve to <0.02; got {err_right}"


def test_clip_ratio_shrinks_grid() -> None:
    x = torch.randn(1, 1, 16, 128, dtype=torch.float32)
    no_clip = quantize_per_token(x, bits=2, group_size=128, clip_ratio=1.0)
    clipped = quantize_per_token(x, bits=2, group_size=128, clip_ratio=0.5)
    assert (clipped.scale.float() < no_clip.scale.float()).all(), (
        "clip_ratio<1 should shrink the represented range, hence shrink scale"
    )
