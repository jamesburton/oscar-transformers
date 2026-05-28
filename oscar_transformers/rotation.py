"""Rotation loading and weight baking.

OSCAR's rotation tensors are per-layer fp32 orthogonal matrices of shape
``(head_dim, head_dim)``. Two separate files exist per model — one for K
(``k_rotation_qqt_r_h_pbr.pt``) and one for V (``v_rotation_sst_r_h_pbr.pt``).

Format (per upstream ``compute_kv_rotation.py``)::

    {
        "format_version": 1,
        "objective": "<method>_<composition>",
        "source_grouping": "layer",
        "layers": {
            layer_id: {
                "layer_id": int,
                "rotation": Tensor (head_dim, head_dim) fp32,
                "eigenvalues": Tensor (head_dim,) fp32,
            },
            ...
        },
    }

``bake_rotations`` mutates an HF transformers model's attention projection
weights in place so that subsequent forward passes produce rotated Q/K/V and
de-rotated attention output. The transformation is mathematically equivalent
to the unrotated model (rotations are orthogonal) but the K/V tensors that
reach the cache are already in the rotated basis, ready to be quantized.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn


@dataclass
class LayerRotation:
    layer_id: int
    rotation: torch.Tensor
    eigenvalues: torch.Tensor


@dataclass
class RotationSet:
    objective: str
    layers: Dict[int, LayerRotation]

    @property
    def head_dim(self) -> int:
        first = next(iter(self.layers.values()))
        return int(first.rotation.shape[0])

    def __len__(self) -> int:
        return len(self.layers)


def load_rotation_file(path: str | Path) -> RotationSet:
    """Load a ``k_rotation_*.pt`` or ``v_rotation_*.pt`` file produced by
    OSCAR's ``compute_kv_rotation.py``.
    """
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if "layers" not in blob:
        raise ValueError(
            f"{path}: missing 'layers' key; got top-level keys {list(blob.keys())}. "
            f"Expected an OSCAR rotation file (format_version=1)."
        )
    layers: Dict[int, LayerRotation] = {}
    for lid, entry in blob["layers"].items():
        rot = entry["rotation"].to(torch.float32)
        if rot.dim() != 2 or rot.shape[0] != rot.shape[1]:
            raise ValueError(
                f"layer {lid}: rotation has shape {tuple(rot.shape)}; "
                f"expected square (head_dim, head_dim)."
            )
        eig = entry.get("eigenvalues")
        eig_t = eig.to(torch.float32) if eig is not None else torch.empty(0)
        layers[int(lid)] = LayerRotation(
            layer_id=int(lid), rotation=rot, eigenvalues=eig_t,
        )
    return RotationSet(objective=str(blob.get("objective", "")), layers=layers)


def _iter_attention_blocks(model: nn.Module):
    """Yield (layer_id, attention_module) for each attention block in the model.

    Works with Qwen2/Qwen3-style HF models whose decoder layers expose
    ``self_attn`` with ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj`` as
    ``nn.Linear`` children.
    """
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise AttributeError(
            f"Could not find decoder layers on {type(model).__name__}; "
            f"expected `model.model.layers` (Qwen2/Qwen3 layout)."
        )
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            raise AttributeError(f"Layer {idx} has no `self_attn`.")
        yield idx, attn


def _rotate_input_proj(proj: nn.Linear, rotation: torch.Tensor, head_dim: int) -> None:
    """Rotate the output side of an input projection (q/k/v) in place.

    ``nn.Linear.weight`` has shape ``(out, in)`` and forward computes
    ``y = x @ W.T``. The output dim is laid out as ``num_heads * head_dim`` per
    head. To rotate each head's slice of the output by R on the right
    (``y_h <- y_h @ R``), we set ``W_h <- R.T @ W_h`` row-wise per head.
    """
    out_features, _ = proj.weight.shape
    if out_features % head_dim != 0:
        raise ValueError(
            f"projection out_features={out_features} not divisible by head_dim={head_dim}."
        )
    num_heads = out_features // head_dim
    rot = rotation.to(proj.weight.device, proj.weight.dtype)
    with torch.no_grad():
        w = proj.weight.data.view(num_heads, head_dim, -1)
        w.copy_(torch.einsum("ij,hjk->hik", rot.T, w))
        if proj.bias is not None:
            b = proj.bias.data.view(num_heads, head_dim)
            b.copy_(torch.einsum("ij,hj->hi", rot.T, b))


def _rotate_output_proj(proj: nn.Linear, rotation: torch.Tensor, head_dim: int) -> None:
    """Rotate the input side of an output projection (o_proj) in place.

    ``o_proj.weight`` has shape ``(hidden_dim, num_q_heads * head_dim)``. To
    undo a per-head rotation ``R`` applied to ``attn_out`` before it reaches
    ``o_proj``, we set ``W_h <- W_h @ R`` column-wise per head (where the
    forward is ``y = x @ W.T``; rotating ``x`` by ``R`` on the right is
    cancelled by composing ``R`` into the right of ``W``).
    """
    out_features, in_features = proj.weight.shape
    if in_features % head_dim != 0:
        raise ValueError(
            f"o_proj in_features={in_features} not divisible by head_dim={head_dim}."
        )
    num_heads = in_features // head_dim
    rot = rotation.to(proj.weight.device, proj.weight.dtype)
    with torch.no_grad():
        w = proj.weight.data.view(out_features, num_heads, head_dim)
        w.copy_(torch.einsum("ohj,jk->ohk", w, rot))


def bake_rotations(
    model: nn.Module,
    *,
    k_rotations: RotationSet,
    v_rotations: RotationSet,
) -> None:
    """Mutate model attention projections in place so subsequent forwards
    produce K/V already in OSCAR's rotated basis.

    Operations per layer ``L``::

        q_proj.weight <- block_diag(R_k.T per head) @ q_proj.weight
        k_proj.weight <- block_diag(R_k.T per head) @ k_proj.weight
        v_proj.weight <- block_diag(R_v.T per head) @ v_proj.weight
        o_proj.weight <- o_proj.weight @ block_diag(R_v per head)

    Rotations are orthogonal, so attention scores ``Q K^T`` and the final
    block output are mathematically unchanged in infinite precision; the K/V
    tensors that flow into the cache are now in a basis chosen offline to
    flatten per-channel outliers for INT2 quantization.

    Idempotency: applying ``bake_rotations`` twice produces a different model.
    Always start from freshly-loaded weights.
    """
    if k_rotations.head_dim != v_rotations.head_dim:
        raise ValueError(
            f"K head_dim={k_rotations.head_dim} != V head_dim={v_rotations.head_dim}"
        )
    head_dim = k_rotations.head_dim

    baked_layers = 0
    for layer_idx, attn in _iter_attention_blocks(model):
        if layer_idx not in k_rotations.layers:
            raise KeyError(f"K rotations missing for layer {layer_idx}")
        if layer_idx not in v_rotations.layers:
            raise KeyError(f"V rotations missing for layer {layer_idx}")
        r_k = k_rotations.layers[layer_idx].rotation
        r_v = v_rotations.layers[layer_idx].rotation

        _rotate_input_proj(attn.q_proj, r_k, head_dim)
        _rotate_input_proj(attn.k_proj, r_k, head_dim)
        _rotate_input_proj(attn.v_proj, r_v, head_dim)
        _rotate_output_proj(attn.o_proj, r_v, head_dim)
        baked_layers += 1

    if baked_layers != len(k_rotations.layers):
        raise RuntimeError(
            f"Model has {baked_layers} attention layers but rotation file has "
            f"{len(k_rotations.layers)}. Refusing to leave the model half-baked."
        )
