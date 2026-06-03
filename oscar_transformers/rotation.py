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
        attn_output = attn_output.reshape(b, t, -1).contiguous()

        # ``o_proj.weight`` has the R_v.T per-head un-rotation baked in by
        # :func:`apply_rotations`, so no runtime un-rotation einsum here.
        # Math: ``W_o' = W_o @ block_diag(R_v.T, ..., R_v.T per head)`` gives
        # the same output as the prior reshape/einsum/reshape sequence.
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return patched_forward


def _looks_like_delta_mem_attention(attn: nn.Module) -> bool:
    """Duck-type check for delta-Mem's DeltaMemAttention wrapper.

    delta-Mem wraps each ``Qwen3Attention`` (or ``SmolLM3Attention``) inside a
    ``DeltaMemAttention`` that re-implements attention internally to thread
    the delta-state through, instead of delegating to ``self.base.forward``.
    That means our class-level patch on ``Qwen3Attention.forward`` is bypassed
    entirely on the delta arm. We detect the wrapper by structure rather than
    importing it, to keep oscar-transformers free of a delta-Mem dependency.
    """
    return (
        hasattr(attn, "base")
        and hasattr(attn.base, "q_proj")
        and hasattr(attn, "_apply_standard_rotary")
        and hasattr(attn, "_normalize_value_states")
    )


def _bake_r_v_T_into_o_proj(o_proj: nn.Module, r_v: torch.Tensor) -> None:
    """Absorb the R_v.T per-head un-rotation into ``o_proj.weight`` in place.

    Math: at runtime the rotated attention output ``attn_rot`` (B, T, H, D)
    needs ``R_v.T`` applied per head before the linear ``W_o``. That is:

        attn_unrot[..., h, :] = attn_rot[..., h, :] @ R_v.T
        out = W_o @ attn_unrot.flatten()

    Equivalent baked form: ``W_o' = W_o @ block_diag(R_v.T, ..., R_v.T)`` with
    one ``R_v.T`` block per head. Then ``out = W_o' @ attn_rot.flatten()``
    skips the un-rotation entirely.

    Delta-mem compatibility: in ``DeltaMemAttention.forward``
    (``delta-Mem/deltamem/core/delta_impl.py:2280-2294``), ``delta_o`` is
    added *after* ``base.o_proj(attn_output)`` and so already lives in the
    un-rotated basis — no further adjustment needed.

    Idempotent: the original weight is saved as
    ``o_proj._oscar_original_weight`` on first call. Subsequent calls
    re-bake from the saved original, so changing rotations is safe.
    """
    if not hasattr(o_proj, "weight"):
        raise TypeError(
            f"_bake_r_v_T_into_o_proj expects a linear-like module with .weight; "
            f"got {type(o_proj).__name__}"
        )
    head_dim = int(r_v.shape[0])
    # Save original on first bake; restore from it on subsequent re-bakes.
    if not hasattr(o_proj, "_oscar_original_weight"):
        o_proj._oscar_original_weight = o_proj.weight.detach().clone()  # type: ignore[attr-defined]
    w = o_proj._oscar_original_weight  # type: ignore[attr-defined]
    out_features, in_features = w.shape
    if in_features % head_dim != 0:
        raise ValueError(
            f"o_proj in_features={in_features} not divisible by head_dim={head_dim}"
        )
    n_heads = in_features // head_dim
    # Cast rotation to match weight dtype; einsum stays in that dtype.
    r_v_t = r_v.to(device=w.device, dtype=w.dtype)
    w_per_head = w.view(out_features, n_heads, head_dim)
    # The runtime un-rotation that we are absorbing is
    # ``O_orig[h, e] = sum_d O_rot[h, d] * R_v[e, d]``  (== O_rot @ R_v.T).
    # Therefore the baked weight must satisfy
    # ``(W_baked @ O_rot_flat) == (W @ O_orig_flat)`` per head, which works out to
    # ``W_baked[o, h, d_new] = sum_e_old W[o, h, e_old] * R_v[e_old, d_new]``
    # i.e. ``W @ R_v`` per head (NOT ``W @ R_v.T``).
    # einsum "ohd,de->ohe" gives ``output[o, h, e] = sum_d w[o, h, d] * r_v[d, e]``,
    # which is the correct ``W @ R_v`` per head.
    w_baked = torch.einsum("ohd,de->ohe", w_per_head, r_v_t).reshape(out_features, in_features)
    with torch.no_grad():
        o_proj.weight.data.copy_(w_baked)


def _patch_delta_mem_instance(attn: nn.Module, r_k: torch.Tensor, r_v: torch.Tensor) -> None:
    """Per-instance rotation injection for a single DeltaMemAttention block.

    DeltaMemAttention.forward (delta-Mem/deltamem/core/delta_impl.py) does:

        query_states, key_states, value_states = self._apply_delta_qkv(...)
        query_states = self._normalize_query_states(query_states).transpose(1, 2)
        key_states   = self._normalize_key_states(key_states).transpose(1, 2)
        value_states = self._normalize_value_states(value_states).transpose(1, 2)
        query_states, key_states = self._apply_standard_rotary(...)
        # *** rotation injected HERE — before past_key_values.update ***
        past_key_values.update(key_states, value_states, ...)
        attn_output = attention_interface(self.base, query_states, key_states, value_states, ...)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        base_o_output = self.base.o_proj(attn_output)   # un-rotation BAKED into o_proj.weight
        attn_output = base_o_output + delta_o
        return attn_output, attn_weights

    Two instance-level patches plus one weight bake:

    1. ``_apply_standard_rotary`` → also rotate Q and K by R_k after RoPE.
    2. ``_normalize_value_states`` → also rotate V by R_v after the norm.
       Note V is in shape (B, T, H, D) here (still pre-transpose); we rotate
       along the last (head_dim) axis, which is shape-agnostic.
    3. ``self.base.o_proj.weight`` is rewritten by
       :func:`_bake_r_v_T_into_o_proj` to absorb the R_v.T un-rotation. No
       wrapper module needed; the original o_proj forward runs unchanged
       but on the rotated inputs.

    The instance-level method patches keep the change local to one block
    and avoid any class-level side effects. The weight bake is global to
    the o_proj module but reversible via the saved
    ``_oscar_original_weight``.
    """
    head_dim = attn.head_dim
    device = next(attn.parameters()).device
    dtype = next(attn.parameters()).dtype
    r_k_t = r_k.to(device=device, dtype=dtype)
    r_v_t = r_v.to(device=device, dtype=dtype)

    attn._oscar_k_rot = r_k_t  # type: ignore[attr-defined]
    attn._oscar_v_rot = r_v_t  # type: ignore[attr-defined]
    # The o_proj bake always reads ``r_v`` (full precision input here, will
    # be cast to weight dtype inside the helper) and snapshots/restores the
    # original weight, so re-baking with a new rotation is safe.
    _bake_r_v_T_into_o_proj(attn.base.o_proj, r_v)

    if getattr(attn, "_oscar_delta_patched", False):
        # Method patches are class-method shadows on the instance; only need
        # to wire once. Buffer values are updated above and read at call
        # time by the closures below.
        return

    orig_apply_rotary = attn._apply_standard_rotary

    def patched_apply_standard_rotary(query_states, key_states, cos, sin):
        q, k = orig_apply_rotary(query_states, key_states, cos, sin)
        rk = attn._oscar_k_rot.to(q.dtype)
        q = torch.einsum("bhtd,de->bhte", q, rk)
        k = torch.einsum("bhtd,de->bhte", k, rk)
        return q, k

    attn._apply_standard_rotary = patched_apply_standard_rotary  # type: ignore[assignment]

    orig_normalize_v = attn._normalize_value_states

    def patched_normalize_value_states(states):
        out = orig_normalize_v(states)
        # ``out`` is (B, T, H, D) at this point — _normalize_value_states is
        # called before .transpose(1, 2) in DeltaMemAttention.forward. We
        # rotate the trailing head_dim axis which is shape-invariant.
        rv = attn._oscar_v_rot.to(out.dtype)
        return torch.einsum("...d,de->...e", out, rv)

    attn._normalize_value_states = patched_normalize_value_states  # type: ignore[assignment]
    attn._oscar_delta_patched = True  # type: ignore[attr-defined]


def apply_rotations(
    model: nn.Module,
    *,
    k_rotations: RotationSet,
    v_rotations: RotationSet,
) -> None:
    """Wire OSCAR K and V rotations into ``model``.

    Two code paths depending on what wraps each layer's ``self_attn``:

    1. **Plain ``Qwen3Attention``** (the typical case, also the base arm of
       delta-mem-tests): attach ``_oscar_k_rot`` / ``_oscar_v_rot`` buffers
       to each attention module, replace the class's ``forward`` with a
       patched version that injects the rotation between
       q_norm/k_norm + RoPE and ``past_key_values.update``, and bake the
       R_v.T un-rotation directly into ``o_proj.weight`` so the runtime
       does not pay an extra einsum.

    2. **``DeltaMemAttention`` wrapper** (delta-Mem's delta arm): the wrapper
       re-implements attention internally and does not call
       ``self.base.forward``, so the class-level patch from (1) is bypassed.
       For each instance we monkey-patch two methods at instance level
       (``_apply_standard_rotary``, ``_normalize_value_states``) and bake
       R_v.T into ``self.base.o_proj.weight``; see
       :func:`_patch_delta_mem_instance`.

    Calling :func:`apply_rotations` repeatedly on the same model is safe:
    buffers are refreshed and patches are guarded by sentinel flags.
    """
    if k_rotations.head_dim != v_rotations.head_dim:
        raise ValueError(
            f"K head_dim={k_rotations.head_dim} != V head_dim={v_rotations.head_dim}"
        )

    seen_qwen3_classes: set[type] = set()
    attached_layers = 0
    for layer_idx, attn in _iter_attention_blocks(model):
        if layer_idx not in k_rotations.layers:
            raise KeyError(f"K rotations missing for layer {layer_idx}")
        if layer_idx not in v_rotations.layers:
            raise KeyError(f"V rotations missing for layer {layer_idx}")
        r_k = k_rotations.layers[layer_idx].rotation
        r_v = v_rotations.layers[layer_idx].rotation

        if _looks_like_delta_mem_attention(attn):
            _patch_delta_mem_instance(attn, r_k, r_v)
        else:
            device = next(attn.parameters()).device
            dtype = next(attn.parameters()).dtype
            attn.register_buffer(
                "_oscar_k_rot", r_k.to(device=device, dtype=dtype), persistent=False,
            )
            attn.register_buffer(
                "_oscar_v_rot", r_v.to(device=device, dtype=dtype), persistent=False,
            )
            # Absorb the R_v.T un-rotation into o_proj.weight in place so
            # the patched forward can skip the reshape/einsum/reshape
            # un-rotation step. Idempotent and reversible (helper saves
            # ``_oscar_original_weight`` on first call).
            _bake_r_v_T_into_o_proj(attn.o_proj, r_v)
            seen_qwen3_classes.add(type(attn))
        attached_layers += 1

    if attached_layers != len(k_rotations.layers):
        raise RuntimeError(
            f"Model has {attached_layers} attention layers but rotation file "
            f"has {len(k_rotations.layers)}. Refusing to leave the model "
            f"half-rotated."
        )

    if seen_qwen3_classes:
        patched_forward = _build_patched_forward()
        for cls in seen_qwen3_classes:
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
