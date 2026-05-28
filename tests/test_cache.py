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
