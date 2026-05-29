"""Rotation loading and attention-forward patching.

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

Why we don't bake rotations into projection weights on Qwen3
-------------------------------------------------------------

Modern Qwen3 attention applies QK-Norm and rotary positional embeddings
between the q/k/v projections and the attention dot product::

    query_states = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
    key_states   = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(x).view(hidden_shape).transpose(1, 2)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

Neither the per-channel learnable scale γ inside ``q_norm`` / ``k_norm`` nor
the RoPE block-diagonal rotation commute with an arbitrary orthogonal R, so
pre-multiplying R into ``q_proj.weight`` / ``k_proj.weight`` produces wrong
outputs (verified empirically on Qwen3-4B-Instruct-2507: rotation-only output
collapses to repeated '?' tokens). V has no such preprocessing, so V-side
baking is mathematically valid, but for API simplicity we apply *all*
rotations inline.

:func:`apply_rotations` therefore monkey-patches the attention class's
``forward`` to inject the rotation at the right point:

* Rotate Q and K by ``R_k`` **after** q_norm/k_norm and RoPE, **before**
  ``past_key_values.update``.
* Rotate V by ``R_v`` before ``past_key_values.update``.
* Un-rotate the attention output by ``R_v.T`` before ``o_proj`` so the block
  output is unchanged in infinite precision.

The rotation is end-to-end mathematically invariant (R is orthogonal); the
benefit is that the K, V tensors that flow into the cache are in a basis
chosen offline to flatten per-channel outliers for INT2 quantization.
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
    """Yield (layer_id, attention_module) for each attention block in the model."""
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


_PATCHED_CLASSES: set[type] = set()


def _build_patched_forward():
    """Construct the patched forward function. Imported at patch time so
    transformers is not an import-time hard dependency of this module.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import (
        apply_rotary_pos_emb, eager_attention_forward,
    )

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        r_k = self._oscar_k_rot.to(query_states.dtype)
        r_v = self._oscar_v_rot.to(value_states.dtype)
        query_states = torch.einsum("bhtd,de->bhte", query_states, r_k)
        key_states = torch.einsum("bhtd,de->bhte", key_states, r_k)
        value_states = torch.einsum("bhtd,de->bhte", value_states, r_v)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        b, t = attn_output.shape[0], attn_output.shape[1]
        attn_output = attn_output.reshape(b, t, -1, self.head_dim)
        r_v_T = r_v.transpose(-1, -2).to(attn_output.dtype)
        attn_output = torch.einsum("bthd,de->bthe", attn_output, r_v_T)
        attn_output = attn_output.reshape(b, t, -1).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return patched_forward


def apply_rotations(
    model: nn.Module,
    *,
    k_rotations: RotationSet,
    v_rotations: RotationSet,
) -> None:
    """Wire OSCAR K and V rotations into ``model`` by:

    1. Attaching ``_oscar_k_rot`` and ``_oscar_v_rot`` buffers (head_dim x
       head_dim, in the attention module's dtype/device) to every attention
       block.
    2. Replacing the attention class's ``forward`` with a patched version
       (once per class per process) that applies the rotations at the right
       point relative to q_norm, k_norm, RoPE, and the cache.

    The patched ``forward`` produces the same output as the original in
    infinite precision (rotations are orthogonal). The benefit is that the
    K, V tensors that pass through ``past_key_values.update`` are in a basis
    where INT2 quantization is well-conditioned.

    Calling :func:`apply_rotations` a second time on the same model is safe:
    the buffers are overwritten and the class is already patched.
    """
    if k_rotations.head_dim != v_rotations.head_dim:
        raise ValueError(
            f"K head_dim={k_rotations.head_dim} != V head_dim={v_rotations.head_dim}"
        )

    seen_classes: set[type] = set()
    attached_layers = 0
    for layer_idx, attn in _iter_attention_blocks(model):
        if layer_idx not in k_rotations.layers:
            raise KeyError(f"K rotations missing for layer {layer_idx}")
        if layer_idx not in v_rotations.layers:
            raise KeyError(f"V rotations missing for layer {layer_idx}")
        r_k = k_rotations.layers[layer_idx].rotation
        r_v = v_rotations.layers[layer_idx].rotation
        device = next(attn.parameters()).device
        dtype = next(attn.parameters()).dtype
        attn.register_buffer("_oscar_k_rot", r_k.to(device=device, dtype=dtype), persistent=False)
        attn.register_buffer("_oscar_v_rot", r_v.to(device=device, dtype=dtype), persistent=False)
        seen_classes.add(type(attn))
        attached_layers += 1

    if attached_layers != len(k_rotations.layers):
        raise RuntimeError(
            f"Model has {attached_layers} attention layers but rotation file "
            f"has {len(k_rotations.layers)}. Refusing to leave the model "
            f"half-rotated."
        )

    patched_forward = _build_patched_forward()
    for cls in seen_classes:
        if cls in _PATCHED_CLASSES:
            continue
        cls.forward = patched_forward
        _PATCHED_CLASSES.add(cls)


def bake_rotations(*args, **kwargs):
    """Deprecated. Qwen3's QK-Norm + RoPE pipeline means projection-weight
    baking is not mathematically valid; use :func:`apply_rotations` instead.
    """
    raise NotImplementedError(
        "bake_rotations was deprecated after empirical validation: Qwen3's "
        "q_norm/k_norm (per-channel γ) and RoPE both fail to commute with an "
        "arbitrary orthogonal rotation, so pre-multiplying R into the "
        "projection weights produces gibberish at inference time. Use "
        "apply_rotations(model, k_rotations=..., v_rotations=...) — it "
        "registers the rotations as buffers on each attention module and "
        "monkey-patches the attention forward to apply them after "
        "q_norm/k_norm+RoPE."
    )
