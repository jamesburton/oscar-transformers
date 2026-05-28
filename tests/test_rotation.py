"""Rotation loader + bake_rotations smoke tests using a tiny stub model."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from oscar_transformers.rotation import (
    LayerRotation,
    RotationSet,
    bake_rotations,
    load_rotation_file,
)


def _orthogonal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q, _ = torch.linalg.qr(a)
    return q.to(torch.float32)


def _save_rotation(path: Path, layers: dict[int, torch.Tensor]) -> None:
    blob = {
        "format_version": 1,
        "objective": "test",
        "source_grouping": "layer",
        "layers": {
            lid: {
                "layer_id": lid,
                "rotation": r,
                "eigenvalues": torch.ones(r.shape[0]),
            }
            for lid, r in layers.items()
        },
    }
    torch.save(blob, path)


def test_load_rotation_file_roundtrip(tmp_path: Path) -> None:
    r0 = _orthogonal(64, 0)
    r1 = _orthogonal(64, 1)
    p = tmp_path / "k_rotation.pt"
    _save_rotation(p, {0: r0, 1: r1})
    rs = load_rotation_file(p)
    assert rs.head_dim == 64
    assert len(rs) == 2
    assert torch.allclose(rs.layers[0].rotation, r0)
    assert torch.allclose(rs.layers[1].rotation, r1)


class _StubAttn(nn.Module):
    def __init__(self, hidden_dim: int, num_q_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_dim, bias=False)


class _StubLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.self_attn = _StubAttn(**kwargs)


class _StubModel(nn.Module):
    def __init__(self, n_layers: int, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([_StubLayer(**kwargs) for _ in range(n_layers)])


class _Wrapper(nn.Module):
    def __init__(self, inner: _StubModel):
        super().__init__()
        self.model = inner


def test_bake_rotations_is_orthogonally_invariant() -> None:
    """End-to-end: a baked model produces the same attention block output as
    the unbaked one (rotations are orthogonal). We do this by computing the
    block ``attn_out = softmax(QK^T)V`` mapped through o_proj for a random
    input, both before and after baking.
    """
    torch.manual_seed(0)
    hidden = 32
    num_q = 4
    num_kv = 2
    head_dim = 16

    model = _Wrapper(_StubModel(n_layers=2, hidden_dim=hidden, num_q_heads=num_q, num_kv_heads=num_kv, head_dim=head_dim))

    x = torch.randn(1, 7, hidden)

    def _forward_block(attn: _StubAttn) -> torch.Tensor:
        q = attn.q_proj(x).view(1, 7, num_q, head_dim).transpose(1, 2)
        k = attn.k_proj(x).view(1, 7, num_kv, head_dim).transpose(1, 2)
        v = attn.v_proj(x).view(1, 7, num_kv, head_dim).transpose(1, 2)
        k_rep = k.repeat_interleave(num_q // num_kv, dim=1)
        v_rep = v.repeat_interleave(num_q // num_kv, dim=1)
        scores = (q @ k_rep.transpose(-1, -2)) / (head_dim ** 0.5)
        out = torch.softmax(scores, dim=-1) @ v_rep
        out = out.transpose(1, 2).reshape(1, 7, num_q * head_dim)
        return attn.o_proj(out)

    baseline = [_forward_block(layer.self_attn).clone() for layer in model.model.layers]

    r_k = {i: _orthogonal(head_dim, 100 + i) for i in range(2)}
    r_v = {i: _orthogonal(head_dim, 200 + i) for i in range(2)}
    k_set = RotationSet(objective="test", layers={i: LayerRotation(i, r_k[i], torch.ones(head_dim)) for i in r_k})
    v_set = RotationSet(objective="test", layers={i: LayerRotation(i, r_v[i], torch.ones(head_dim)) for i in r_v})

    bake_rotations(model, k_rotations=k_set, v_rotations=v_set)

    baked = [_forward_block(layer.self_attn).clone() for layer in model.model.layers]

    for i, (a, b) in enumerate(zip(baseline, baked)):
        assert torch.allclose(a, b, atol=1e-4), (
            f"layer {i}: baked block output diverges from baseline (orthogonal "
            f"rotation should be a no-op end-to-end). max err = "
            f"{(a - b).abs().max().item():.3e}"
        )


def test_bake_refuses_partial_coverage() -> None:
    model = _Wrapper(_StubModel(n_layers=3, hidden_dim=32, num_q_heads=4, num_kv_heads=2, head_dim=16))
    r = _orthogonal(16, 0)
    half = RotationSet(
        objective="test",
        layers={i: LayerRotation(i, r, torch.ones(16)) for i in (0, 1)},
    )
    with pytest.raises(KeyError):
        bake_rotations(model, k_rotations=half, v_rotations=half)
