"""OSCARCache state-management tests with a stub config."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from oscar_transformers import OSCARCache


def _cfg(layers: int = 2):
    return SimpleNamespace(num_hidden_layers=layers)


def _step(cache: OSCARCache, layer_idx: int, n_tokens: int, *, head_dim: int = 128, heads: int = 8):
    k = torch.randn(1, heads, n_tokens, head_dim, dtype=torch.bfloat16)
    v = torch.randn(1, heads, n_tokens, head_dim, dtype=torch.bfloat16)
    out_k, out_v = cache.update(k, v, layer_idx)
    return out_k, out_v


def test_sink_only_no_quantization() -> None:
    cache = OSCARCache(config=_cfg(1), sink_tokens=64, recent_tokens=256)
    out_k, _ = _step(cache, 0, 32)
    assert cache.get_seq_length(0) == 32
    layer = cache.layers[0]
    assert layer.sink_k is not None and layer.sink_k.shape[2] == 32
    assert layer.middle_k is None
    assert layer.recent_k is None
    assert out_k.shape[2] == 32


def test_fills_sink_then_recent() -> None:
    cache = OSCARCache(config=_cfg(1), sink_tokens=64, recent_tokens=256)
    _step(cache, 0, 100)
    layer = cache.layers[0]
    assert layer.sink_k.shape[2] == 64
    assert layer.recent_k.shape[2] == 36
    assert layer.middle_k is None
    assert cache.get_seq_length(0) == 100


def test_spills_to_middle_when_recent_overflows() -> None:
    cache = OSCARCache(config=_cfg(1), sink_tokens=8, recent_tokens=16)
    _step(cache, 0, 50)
    layer = cache.layers[0]
    assert layer.sink_k.shape[2] == 8
    assert layer.recent_k.shape[2] == 16
    assert layer.middle_k is not None
    assert layer.middle_k.codes.shape[2] == 50 - 8 - 16
    assert cache.get_seq_length(0) == 50


def test_per_layer_state_isolated() -> None:
    cache = OSCARCache(config=_cfg(3))
    _step(cache, 0, 10)
    _step(cache, 2, 70)
    assert cache.get_seq_length(0) == 10
    assert cache.get_seq_length(1) == 0
    assert cache.get_seq_length(2) == 70


def test_assembled_output_token_count_matches_total() -> None:
    cache = OSCARCache(config=_cfg(1), sink_tokens=8, recent_tokens=16)
    for n in (5, 5, 10, 30):
        out_k, out_v = _step(cache, 0, n)
    expected = 5 + 5 + 10 + 30
    assert cache.get_seq_length(0) == expected
    assert out_k.shape[2] == expected
    assert out_v.shape[2] == expected


def test_shadow_disabled_assemble_matches_enabled() -> None:
    """With OSCAR_DISABLE_DEQUANT_SHADOW=1, _assemble should dequantize the
    middle on demand and produce the same shape and (approximately) the same
    values as the shadow-enabled path.
    """
    import os
    import importlib
    import oscar_transformers.cache as cache_mod

    # Build a deterministic input.
    torch.manual_seed(0)
    head_dim, heads = 128, 8

    # 1) Shadow enabled (default).
    os.environ.pop("OSCAR_DISABLE_DEQUANT_SHADOW", None)
    importlib.reload(cache_mod)
    from oscar_transformers.cache import OSCARCache as OSCARCacheEnabled
    cache_a = OSCARCacheEnabled(config=_cfg(1), sink_tokens=8, recent_tokens=16)
    torch.manual_seed(0)
    for n in (5, 5, 10, 30):
        k = torch.randn(1, heads, n, head_dim, dtype=torch.bfloat16)
        v = torch.randn(1, heads, n, head_dim, dtype=torch.bfloat16)
        out_k_a, out_v_a = cache_a.update(k, v, 0)
    assert cache_a.layers[0]._middle_k_dq is not None  # shadow alive

    # 2) Shadow disabled.
    os.environ["OSCAR_DISABLE_DEQUANT_SHADOW"] = "1"
    importlib.reload(cache_mod)
    from oscar_transformers.cache import OSCARCache as OSCARCacheDisabled
    cache_b = OSCARCacheDisabled(config=_cfg(1), sink_tokens=8, recent_tokens=16)
    torch.manual_seed(0)
    for n in (5, 5, 10, 30):
        k = torch.randn(1, heads, n, head_dim, dtype=torch.bfloat16)
        v = torch.randn(1, heads, n, head_dim, dtype=torch.bfloat16)
        out_k_b, out_v_b = cache_b.update(k, v, 0)
    assert cache_b.layers[0]._middle_k_dq is None  # shadow skipped

    # Same shape, same values (math is identical — both dequantize the same codes).
    assert out_k_a.shape == out_k_b.shape
    assert torch.allclose(out_k_a, out_k_b, atol=0, rtol=0)
    assert torch.allclose(out_v_a, out_v_b, atol=0, rtol=0)

    # Restore default for downstream tests.
    os.environ.pop("OSCAR_DISABLE_DEQUANT_SHADOW", None)
    importlib.reload(cache_mod)
