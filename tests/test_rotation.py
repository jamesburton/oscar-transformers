"""Rotation loader + apply_rotations smoke tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from oscar_transformers.rotation import (
    LayerRotation,
    RotationSet,
    apply_rotations,
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


def test_bake_rotations_is_deprecated() -> None:
    with pytest.raises(NotImplementedError, match="apply_rotations"):
        bake_rotations(None, k_rotations=None, v_rotations=None)


def _rotation_set(num_layers: int, head_dim: int, seed_base: int) -> RotationSet:
    return RotationSet(
        objective="test",
        layers={
            i: LayerRotation(i, _orthogonal(head_dim, seed_base + i), torch.ones(head_dim))
            for i in range(num_layers)
        },
    )


def test_apply_rotations_preserves_qwen3_block_output() -> None:
    """End-to-end: apply_rotations must produce the same attention output as
    an unrotated forward in infinite precision (rotations are orthogonal).

    We test against a real Qwen3Attention block — with q_norm, k_norm, RoPE,
    GQA, and the full attention machinery — at fp32, where bf16 accumulation
    noise is not a confound. This is the test that would have caught the
    initial bake_rotations bug.
    """
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    head_dim = 32
    num_q_heads = 4
    num_kv_heads = 2
    cfg = Qwen3Config(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        num_hidden_layers=1,
        intermediate_size=64,
        max_position_embeddings=64,
        attention_dropout=0.0,
        _attn_implementation="eager",
    )

    torch.manual_seed(0)
    attn = Qwen3Attention(cfg, layer_idx=0).to(torch.float32).eval()

    seq = 7
    hidden = torch.randn(1, seq, cfg.hidden_size, dtype=torch.float32)
    # Build cos/sin for RoPE manually (head_dim cosine sweep).
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(seq).float().unsqueeze(-1)
    freqs = positions * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().unsqueeze(0)
    sin = emb.sin().unsqueeze(0)
    position_embeddings = (cos, sin)

    out_baseline, _ = attn(hidden, position_embeddings, attention_mask=None, past_key_values=None)
    out_baseline = out_baseline.clone()

    class _Wrap(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([
                torch.nn.Module(),
            ])
            self.model.layers[0].self_attn = attn

    model = _Wrap(attn)
    k_set = _rotation_set(num_layers=1, head_dim=head_dim, seed_base=100)
    v_set = _rotation_set(num_layers=1, head_dim=head_dim, seed_base=200)

    apply_rotations(model, k_rotations=k_set, v_rotations=v_set)

    out_rotated, _ = attn(hidden, position_embeddings, attention_mask=None, past_key_values=None)

    max_err = (out_baseline - out_rotated).abs().max().item()
    assert max_err < 1e-4, (
        f"rotated block output diverges from baseline (orthogonal rotation "
        f"should be a no-op end-to-end). max err = {max_err:.3e}"
    )


def test_apply_rotations_refuses_partial_coverage() -> None:
    pytest.importorskip("transformers")
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

    head_dim = 32
    cfg = Qwen3Config(
        hidden_size=4 * head_dim,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        num_hidden_layers=3,
        intermediate_size=64,
        max_position_embeddings=64,
        _attn_implementation="eager",
    )
    attns = [Qwen3Attention(cfg, layer_idx=i).to(torch.float32) for i in range(3)]

    class _Wrap(torch.nn.Module):
        def __init__(self, attns):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in attns])
            for layer, a in zip(self.model.layers, attns):
                layer.self_attn = a

    model = _Wrap(attns)
    half = _rotation_set(num_layers=2, head_dim=head_dim, seed_base=0)
    with pytest.raises(KeyError):
        apply_rotations(model, k_rotations=half, v_rotations=half)
